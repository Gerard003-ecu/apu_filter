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
        """Test a healthy Directed Acyclic Graph (DAG)"""
        G = nx.DiGraph()
        G.add_edge("APU1", "Insumo1")
        G.add_edge("APU1", "Insumo2")
        G.add_edge("APU2", "Insumo2")
        G.add_edge("APU2", "Insumo3")

        # Add types
        nx.set_node_attributes(
            G,
            {
                "APU1": "APU",
                "APU2": "APU",
                "Insumo1": "INSUMO",
                "Insumo2": "INSUMO",
                "Insumo3": "INSUMO",
            },
            "type",
        )

        report = analyzer.generate_executive_report(G)

        assert isinstance(report, ConstructionRiskReport)
        assert report.integrity_score == 100.0
        assert len(report.circular_risks) == 0
        assert len(report.waste_alerts) == 0
        assert report.complexity_level in ["Baja", "Media", "Alta"]

        audit_lines = analyzer.get_audit_report(G)
        assert any("AUDITORÍA ESTRUCTURAL DEL PRESUPUESTO" in line for line in audit_lines)
        assert any(
            "Estructura de Costos Saludable y Auditable" in line for line in audit_lines
        )

    def test_circular_reference(self, analyzer):
        """Test detection of circular references (errors)"""
        G = nx.DiGraph()
        G.add_edge("APU1", "APU2")
        G.add_edge("APU2", "APU1")  # Cycle

        nx.set_node_attributes(G, {"APU1": "APU", "APU2": "APU"}, "type")

        report = analyzer.generate_executive_report(G)

        assert report.integrity_score <= 50.0
        assert len(report.circular_risks) > 0
        assert "CRÍTICO" in report.circular_risks[0]

        audit_lines = analyzer.get_audit_report(G)
        assert any("Ciclos de Costo (Errores)" in line for line in audit_lines)
        assert any("❌" in line for line in audit_lines)

    def test_isolated_nodes(self, analyzer):
        """Test detection of isolated nodes (waste)"""
        G = nx.DiGraph()
        G.add_node("InsumoFantasma", type="INSUMO")

        report = analyzer.generate_executive_report(G)

        assert report.integrity_score < 100.0
        assert len(report.waste_alerts) > 0
        assert "Insumos no utilizados" in report.waste_alerts[0]

        audit_lines = analyzer.get_audit_report(G)
        assert any("POSIBLE DESPERDICIO" in line for line in audit_lines)

    def test_orphan_insumos(self, analyzer):
        """Test detection of orphan insumos (defined but not used in APUs)"""
        # Node with type INSUMO, 0 in-degree, but maybe has out-degree?
        # Or just 0 in-degree and not isolated (connected to something else?)
        # Actually logic is: type=INSUMO, in_degree=0.
        # If it is isolated, it is also captured as isolated.
        # Let's create an insumo that points to something else (weird but possible in graph)
        # so it's not strictly isolated, but has no incoming edges (so not used by any APU).

        G = nx.DiGraph()
        G.add_edge("Insumo1", "Trash")
        nx.set_node_attributes(G, {"Insumo1": "INSUMO", "Trash": "OTHER"}, "type")

        report = analyzer.generate_executive_report(G)

        # Should trigger orphan alert
        assert len(report.waste_alerts) > 0
        assert any(
            "Recursos sin asignación" in alert for alert in report.waste_alerts
        )

    def test_integration_backward_compatibility(self, analyzer):
        """Test that old analyze method output can still be processed by get_audit_report"""
        G = nx.DiGraph()
        G.add_edge("A", "B")

        # analyze() returns dict with metrics/anomalies
        old_result = analyzer.analyze(G)

        audit_lines = analyzer.get_audit_report(old_result)

        assert any("AUDITORÍA ESTRUCTURAL DEL PRESUPUESTO" in line for line in audit_lines)
        assert any("Ciclos de Costo" in line for line in audit_lines)
