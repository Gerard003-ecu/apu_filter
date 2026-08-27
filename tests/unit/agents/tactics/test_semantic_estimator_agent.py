# -*- coding: utf-8 -*-
"""Batería de pruebas unitarias para semantic_estimator_agent.py."""
from __future__ import annotations
import dataclasses
import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.agents.tactics.semantic_estimator_agent import (
    _MACHINE_EPSILON, _TAU_MIN_SIMILARITY,
    SemanticEstimatorAgentError, TopologicalMappingError, VectorDegeneracyError,
    DimensionalIncompatibilityError, ThermodynamicFrictionAnomaly, FunctorialityError,
    ProjectorIntegrityError,
    TopologicalNeighborhoodData, TensorFrictionData, RankNullityProjectionData,
    Phase1TopologicalBridge, Phase2FrictionBridge, SemanticEstimatorAuditState,
    Phase1_TopologicalNeighborhoodCertifier, Phase2_TensorFrictionAuditor,
    Phase3_RankNullityProjector, SemanticEstimatorAgent
)

class TestConstants:
    """Verify physical constants and tolerances exist and are positive."""
    def test_machine_epsilon(self):
        assert _MACHINE_EPSILON > 0.0

    def test_tau_min_similarity(self):
        assert _TAU_MIN_SIMILARITY > 0.0

class TestExceptionHierarchy:
    """Verify exception inheritance chain."""
    def test_topological_mapping_error(self):
        assert issubclass(TopologicalMappingError, SemanticEstimatorAgentError)

    def test_vector_degeneracy_error(self):
        assert issubclass(VectorDegeneracyError, SemanticEstimatorAgentError)

class TestDataclasses:
    """Verify frozen DTOs, field types, immutability."""
    def test_topological_neighborhood_data_frozen(self):
        obj = TopologicalNeighborhoodData(
            cosine_similarity=1.0,
            angle_radians=0.0,
            angle_degrees=0.0,
            euclidean_distance=0.0,
            query_norm=1.0,
            retrieved_norm=1.0,
            dimensionality=2,
            is_homotopically_valid=True
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.cosine_similarity = 0.5
            
    def test_tensor_friction_data_frozen(self):
        obj = TensorFrictionData(
            condition_number=1.0,
            spectral_min=1.0,
            spectral_max=1.0,
            spectral_mean=1.0,
            spectral_std=0.0,
            total_cost_norm=1.0,
            total_cost_vector=np.array([1.0], dtype=np.float64),
            is_positive_definite=True
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            obj.condition_number = 2.0

class TestPhase1_TopologicalNeighborhoodCertifier:
    """Granular tests for Phase 1 methods."""
    def setup_method(self):
        self.certifier = Phase1_TopologicalNeighborhoodCertifier()
        
    def test_coerce_finite_scalar_valid(self):
        val = self.certifier._coerce_finite_scalar("test", 1.0)
        assert abs(val - 1.0) < 1e-12
        
    def test_coerce_finite_scalar_invalid(self):
        with pytest.raises(SemanticEstimatorAgentError):
            self.certifier._coerce_finite_scalar("test", True)
            
    def test_coerce_finite_vector_valid(self):
        vec = np.array([1.0, 2.0], dtype=np.float64)
        res = self.certifier._coerce_finite_vector("test", vec)
        np.testing.assert_allclose(res, vec)

    def test_coerce_finite_vector_invalid_shape(self):
        with pytest.raises(SemanticEstimatorAgentError):
            self.certifier._coerce_finite_vector("test", np.array([[1.0]], dtype=np.float64))

class TestMainAgentOrEngine:
    """Integration tests for the main class."""
    def setup_method(self):
        with patch.object(SemanticEstimatorAgent, '__abstractmethods__', frozenset()):
            self.agent = SemanticEstimatorAgent()
