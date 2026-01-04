import networkx as nx
import pytest

from agent.business_topology import (
    BusinessTopologicalAnalyzer,
    ConstructionRiskReport,
)


class TestExecutiveReport:
    @pytest.fixture
    def analyzer(self):
        return BusinessTopologicalAnalyzer()

    def test_healthy_dag(self, analyzer):
        """Test a healthy Directed Acyclic Graph (DAG) with Pyramidal Stability."""
        # Create a graph with a solid base (More INSUMO than APU)
        G = nx.DiGraph()

        # To get a high stability score, we need a high ratio of Insumos/APUs
        # With 1 APU and 500 Insumos, stability is ~2.69 and score ~96.9

        G.add_node("APU1", type="APU")
        for i in range(200):
            insumo_id = f"Insumo{i}"
            G.add_node(insumo_id, type="INSUMO")
            G.add_edge("APU1", insumo_id)

        report = analyzer.generate_executive_report(G)

        assert isinstance(report, ConstructionRiskReport)

        assert report.integrity_score > 90.0
        assert len(report.circular_risks) == 0
        assert len(report.waste_alerts) == 0
        assert report.complexity_level in ["Baja", "Media", "Alta"]

        audit_lines = analyzer.get_audit_report(G)
        assert any("AUDITORÍA ESTRUCTURAL DEL PRESUPUESTO" in line for line in audit_lines)

    def test_circular_reference(self, analyzer):
        """Test detection of circular references (errors) with new Narrative."""
        G = nx.DiGraph()
        G.add_edge("APU1", "APU2")
        G.add_edge("APU2", "APU1")  # Cycle

        nx.set_node_attributes(G, {"APU1": "APU", "APU2": "APU"}, "type")

        report = analyzer.generate_executive_report(G)

        assert report.integrity_score <= 50.0
        assert len(report.circular_risks) > 0
        assert any("CRÍTICO" in r or "SOCAVONES LÓGICOS" in r for r in report.circular_risks)

        audit_lines = analyzer.get_audit_report(G)
        assert any("Referencias circulares detectadas" in line for line in audit_lines)
        assert any("❌" in line for line in audit_lines)

    def test_isolated_nodes(self, analyzer):
        """Test detection of isolated nodes (waste)"""
        G = nx.DiGraph()
        G.add_node("InsumoFantasma", type="INSUMO")

        report = analyzer.generate_executive_report(G)

        assert report.integrity_score < 100.0
        assert len(report.waste_alerts) > 0
        assert any("nodo(s) aislado(s) detectado(s)" in alert for alert in report.waste_alerts)

        audit_lines = analyzer.get_audit_report(G)
        assert any("Recursos Fantasma" in line for line in audit_lines)

    def test_orphan_insumos(self, analyzer):
        """Test detection of orphan insumos (defined but not used in APUs)"""
        G = nx.DiGraph()
        G.add_edge("Insumo1", "Trash")
        nx.set_node_attributes(G, {"Insumo1": "INSUMO", "Trash": "OTHER"}, "type")

        report = analyzer.generate_executive_report(G)

        # Should trigger orphan alert
        assert len(report.waste_alerts) > 0
        assert any("insumo(s) huérfano(s)" in alert for alert in report.waste_alerts)

    def test_integration_backward_compatibility(self, analyzer):
        """Test that analyze_structural_integrity method output can still be processed by get_audit_report"""
        G = nx.DiGraph()
        G.add_edge("A", "B")

        result = analyzer.analyze_structural_integrity(G)

        audit_lines = analyzer.get_audit_report(result)

        assert any("AUDITORÍA ESTRUCTURAL DEL PRESUPUESTO" in line for line in audit_lines)
        assert any("Ciclos de Costo" in line for line in audit_lines)
