# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Semantic Estimator Agent (Certificador Espectral)        ║
║ Ruta: tests/unit/agents/tactics/test_semantic_estimator_agent.py             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.agents.tactics.semantic_estimator_agent import (
    SemanticEstimatorAgent,
    Phase1_TopologicalNeighborhoodCertifier,
    Phase2_TensorFrictionAuditor,
    Phase3_RankNullityProjector,
    TopologicalNeighborhoodData,
    TensorFrictionData,
    RankNullityProjectionData,
    SemanticEstimatorAuditState,
    VectorDegeneracyError,
    DimensionalIncompatibilityError,
    ThermodynamicFrictionAnomaly,
)


class TestPhase1TopologicalNeighborhoodCertifier:
    """Evaluación de isometría esférica de Hilbert y Procrustes."""

    def setup_method(self):
        self.certifier = Phase1_TopologicalNeighborhoodCertifier()

    def test_procrustes_alignment_and_isometry(self):
        """Verifica alineamiento Procrustes y distancia cosenoidal en el espacio de Hilbert."""
        q = np.array([1.0, 0.0], dtype=np.float64)
        v = np.array([1.0, 0.0], dtype=np.float64)

        data = self.certifier._certify_topological_neighborhood(
            query_vector=q,
            retrieved_vector=v,
        )
        assert data.is_homotopically_valid is True
        assert data.cosine_similarity >= 0.85

    def test_zero_vector_raises_degeneracy(self):
        """Detona VectorDegeneracyError ante vector nulo."""
        q = np.array([0.0, 0.0], dtype=np.float64)
        v = np.array([1.0, 0.0], dtype=np.float64)

        with pytest.raises(VectorDegeneracyError):
            self.certifier._certify_topological_neighborhood(
                query_vector=q,
                retrieved_vector=v,
            )


class TestPhase2TensorFrictionAuditor:
    """Evaluación de tensor de fricción y métrica de Mahalanobis."""

    def setup_method(self):
        self.auditor = Phase2_TensorFrictionAuditor()

    def test_friction_tensor_auditing(self):
        """Audita la positividad espectral del operador con inyección de Rango 1."""
        q = np.array([1.0, 0.0], dtype=np.float64)
        v = np.array([1.0, 0.0], dtype=np.float64)
        c = np.array([2.0, 3.0], dtype=np.float64)
        F = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
        T = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)

        p1_bridge = self.auditor._phase1_certify_and_bridge_to_phase2(
            query_vector=q,
            retrieved_vector=v,
            cost_vector_c=c,
            friction_operator_F=F,
            injection_matrix_T=T,
        )
        p2_bridge = self.auditor._phase2_audit_and_bridge_to_phase3(p1_bridge)
        assert p2_bridge.friction_audit.is_positive_definite is True
        assert p2_bridge.friction_audit.spectral_max > 0.0


class TestSemanticEstimatorAgentPipeline:
    """Integración completa del endofuntor semántico."""

    def setup_method(self):
        self.agent = SemanticEstimatorAgent()

    def test_full_governance_execution(self):
        """Ejecuta el pipeline completo de estimación semántica con matriz T de rango 1."""
        q = np.array([1.0, 0.0], dtype=np.float64)
        v = np.array([1.0, 0.0], dtype=np.float64)
        c = np.array([1.0, 2.0], dtype=np.float64)
        F = np.eye(2, dtype=np.float64)
        T = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)

        state = self.agent(
            query_vector=q,
            retrieved_vector=v,
            cost_vector_c=c,
            friction_operator_F=F,
            injection_matrix_T=T,
        )
        assert isinstance(state, SemanticEstimatorAuditState)
        assert state.is_epistemologically_valid is True
        assert state.neighborhood_audit.is_homotopically_valid is True
        assert state.friction_audit.is_positive_definite is True
