# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Batería de Pruebas: Pipeline Director Agent (Custodio de Causalidad)         ║
║ Ruta: tests/unit/agents/tactics/test_pipeline_director_agent.py            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pytest
import numpy as np

from app.agents.strategy.pipeline_director_agent import (
    PipelineDirectorAgent,
    Phase1_SpectralNilpotenceCertifier,
    Phase2_PosetFiltrationAuditor,
    Phase3_MayerVietorisInterceptor,
    NilpotenceAuditData,
    PosetFiltrationData,
    MayerVietorisAuditData,
    CausalGovernanceState,
    CausalLoopVetoError,
    HomologicalFusionVeto,
    AdjacencyMatrixFormatError,
)


class TestPhase1SpectralNilpotenceCertifier:
    """Evaluación de nilpotencia espectral y aciclicidad (DAG)."""

    def setup_method(self):
        self.certifier = Phase1_SpectralNilpotenceCertifier()

    def test_strictly_nilpotent_dag_matrix(self):
        """Matriz triangular superior con diagonal cero (DAG nilpotente)."""
        A = np.array([
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float64)

        audit = self.certifier._certify_spectral_nilpotence(A)
        assert audit.is_strictly_nilpotent is True
        assert audit.spectral_radius < 1e-8
        assert audit.dimension == 3

    def test_cyclic_matrix_raises_veto(self):
        """Detona CausalLoopVetoError ante ciclo parásito cerrado."""
        A = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float64)

        with pytest.raises(CausalLoopVetoError):
            self.certifier._certify_spectral_nilpotence(A)


class TestPhase2PosetFiltrationAuditor:
    """Evaluación de filtración de subespacios en la jerarquía DIKW."""

    def setup_method(self):
        self.auditor = Phase2_PosetFiltrationAuditor()

    def test_monotonic_filtration_success(self):
        """Verifica aristas que respetan la ordenación de estratos (stratum_v >= stratum_u)."""
        node_strata = {
            "physics_node": 0,
            "tactics_node": 1,
            "strategy_node": 2,
            "wisdom_node": 3,
        }
        edges = [
            ("physics_node", "tactics_node"),
            ("tactics_node", "strategy_node"),
            ("strategy_node", "wisdom_node"),
        ]

        nilp_data = NilpotenceAuditData(
            dimension=4,
            spectral_radius=0.0,
            tolerance=1e-10,
            adjacency_inf_norm=1.0,
            frobenius_norm=1.0,
            nonzero_entries=3,
            directed_density=0.25,
            is_strictly_nilpotent=True,
        )

        filtration = self.auditor._audit_poset_filtration_from_nilpotence(
            nilpotence_audit=nilp_data,
            edges=edges,
            node_strata=node_strata,
        )
        assert filtration.is_monotonic_filtration is True
        assert filtration.edge_count == 3


class TestPhase3MayerVietorisInterceptor:
    """Evaluación de la secuencia homológica de Mayer-Vietoris para fusiones."""

    def setup_method(self):
        self.interceptor = Phase3_MayerVietorisInterceptor()

    def test_mayer_vietoris_exact_fusion(self):
        """Verifica la nulidad del residuo homológico de primer orden."""
        mayer_data = self.interceptor._intercept_mayer_vietoris_sequence(
            betti_1_A=0,
            betti_1_B=0,
            betti_1_intersection=0,
            betti_1_union=0,
        )
        assert mayer_data.is_fusion_homologous is True
        assert mayer_data.delta_betti_1 == 0

    def test_homological_cycle_injection_raises(self):
        """Detona HomologicalFusionVeto si la fusión inyecta ciclos espurios."""
        with pytest.raises(HomologicalFusionVeto):
            self.interceptor._intercept_mayer_vietoris_sequence(
                betti_1_A=0,
                betti_1_B=0,
                betti_1_intersection=0,
                betti_1_union=2,
            )


class TestPipelineDirectorAgentFullPipeline:
    """Integración completa del orquestador de causalidad."""

    def setup_method(self):
        self.agent = PipelineDirectorAgent()

    def test_execute_causal_governance_success(self):
        """Ejecuta la composición de gobernanza causal."""
        A = np.array([
            [0.0, 1.0],
            [0.0, 0.0]
        ], dtype=np.float64)
        edges = [("A", "B")]
        node_strata = {"A": 0, "B": 1}

        state = self.agent.execute_causal_governance(
            adjacency_matrix=A,
            edges=edges,
            node_strata=node_strata,
            betti_1_A=0,
            betti_1_B=0,
            betti_1_intersection=0,
            betti_1_union=0,
        )
        assert isinstance(state, CausalGovernanceState)
        assert state.is_causally_valid is True
        assert state.nilpotence_audit.is_strictly_nilpotent is True
