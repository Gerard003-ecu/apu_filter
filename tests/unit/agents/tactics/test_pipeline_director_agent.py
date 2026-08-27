# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para pipeline_director_agent.py."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.agents.strategy.pipeline_director_agent import (
    _MACHINE_EPSILON, _BASE_SPECTRAL_TOLERANCE,
    PipelineDirectorAgentError, AdjacencyMatrixFormatError, CausalLoopVetoError,
    NilpotenceIndexVetoError, StratumMappingError, SelfLoopVetoError,
    FiltrationViolationVeto, AdjacencySupportVetoError, MayerVietorisInputError,
    HomologicalFusionVeto, EulerPoincareMismatchError,
    NilpotenceAuditData, PosetFiltrationData, MayerVietorisAuditData, CausalGovernanceState,
    Phase1_SpectralNilpotenceCertifier, PipelineDirectorAgent
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_machine_epsilon(self):
        assert _MACHINE_EPSILON > 0.0

    def test_base_spectral_tolerance(self):
        assert _BASE_SPECTRAL_TOLERANCE > 0.0

class TestExceptionHierarchy:
    """Verify exception inheritance chain."""
    def test_adjacency_matrix_format_error(self):
        assert issubclass(AdjacencyMatrixFormatError, PipelineDirectorAgentError)

    def test_causal_loop_veto_error(self):
        assert issubclass(CausalLoopVetoError, PipelineDirectorAgentError)
        
    def test_homological_fusion_veto(self):
        assert issubclass(HomologicalFusionVeto, PipelineDirectorAgentError)

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_nilpotence_audit_data_frozen(self):
        obj = NilpotenceAuditData(
            dimension=2,
            spectral_radius=0.0,
            tolerance=1e-10,
            adjacency_inf_norm=1.0,
            frobenius_norm=1.0,
            nonzero_entries=1,
            directed_density=0.5,
            is_strictly_nilpotent=True
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.dimension = 3
            
    def test_poset_filtration_data_frozen(self):
        obj = PosetFiltrationData(
            edge_count=1,
            audited_edge_count=1,
            ignored_edge_count=0,
            min_slack=0,
            max_slack=1,
            is_monotonic_filtration=True
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.edge_count = 2

class TestPhase1_SpectralNilpotenceCertifier:
    """Granular tests for Phase 1 methods."""
    def setup_method(self):
        self.certifier = Phase1_SpectralNilpotenceCertifier()
        
    def test_validate_and_condition_adjacency_valid(self):
        mat = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        res = self.certifier._validate_and_condition_adjacency(mat)
        np.testing.assert_allclose(res, mat)
        
    def test_validate_and_condition_adjacency_invalid_shape(self):
        mat = np.array([0.0, 1.0], dtype=np.float64)
        with pytest.raises(AdjacencyMatrixFormatError):
            self.certifier._validate_and_condition_adjacency(mat)
            
    def test_schur_decompose_nilpotent(self):
        mat = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        T, Q, kappa, blocks = self.certifier._schur_decompose(mat)
        np.testing.assert_allclose(np.diag(T), [0.0, 0.0])

class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def setup_method(self):
        with patch.object(PipelineDirectorAgent, '__abstractmethods__', frozenset()):
            self.agent = PipelineDirectorAgent()
