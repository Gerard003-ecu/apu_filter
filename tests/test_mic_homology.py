
import pytest
import networkx as nx
from unittest.mock import MagicMock, patch
from app.adapters.mic_vectors import vector_audit_homological_fusion, VectorResultStatus
from app.schemas import Stratum

class TestVectorAuditHomologicalFusion:
    """
    Pruebas unitarias para el vector 'audit_fusion_homology'.
    """

    def test_fusion_success_no_ghost_cycles(self):
        """
        Verifica que la fusión sea exitosa cuando no se generan ciclos fantasma.
        """
        # Grafo A: A -> B
        graph_a = nx.DiGraph()
        graph_a.add_edge("A", "B")

        # Grafo B: B -> C
        graph_b = nx.DiGraph()
        graph_b.add_edge("B", "C")

        payload = {"graph_a": graph_a, "graph_b": graph_b}

        # Mockear BusinessTopologicalAnalyzer para evitar complejidad real y aislar el vector
        with patch("agent.business_topology.BusinessTopologicalAnalyzer") as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            mock_instance.audit_integration_homology.return_value = {
                "delta_beta_1": 0,
                "status": "CLEAN_MERGE"
            }

            result = vector_audit_homological_fusion(payload)

            assert result["success"] is True
            assert result["stratum"] == Stratum.TACTICS
            assert result["payload"]["merged_graph_valid"] is True
            assert result["metrics"]["topological_coherence"] == 1.0

    def test_fusion_failure_ghost_cycles(self):
        """
        Verifica que se rechace la fusión cuando aparecen ciclos fantasma.
        """
        # Grafo A: A -> B
        graph_a = nx.DiGraph()
        graph_a.add_edge("A", "B")

        # Grafo B: B -> A (Introduciendo ciclo al fusionar)
        graph_b = nx.DiGraph()
        graph_b.add_edge("B", "A")

        payload = {"graph_a": graph_a, "graph_b": graph_b}

        with patch("agent.business_topology.BusinessTopologicalAnalyzer") as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            mock_instance.audit_integration_homology.return_value = {
                "delta_beta_1": 1,
                "status": "INTEGRATION_CONFLICT"
            }

            result = vector_audit_homological_fusion(payload)

            assert result["success"] is False
            assert result["status"] == VectorResultStatus.TOPOLOGY_ERROR.value
            assert "Anomalía de Mayer-Vietoris" in result["error"]
            assert "ciclos fantasma" in result["error"]

    def test_missing_graphs(self):
        """Rechazo inmediato si faltan grafos."""
        payload = {"graph_a": nx.DiGraph()} # Falta graph_b

        result = vector_audit_homological_fusion(payload)

        assert result["success"] is False
        assert result["status"] == VectorResultStatus.LOGIC_ERROR.value
        assert "Faltan grafos" in result["error"]
