import networkx as nx
import pytest

from agent.business_topology import (
    BusinessTopologicalAnalyzer,
    ConstructionRiskReport,
)


class TestExecutiveReport:
    """Suite de pruebas para la generación de reportes ejecutivos de riesgo."""

    @pytest.fixture
    def analyzer(self):
        """Fixture que proporciona una instancia de BusinessTopologicalAnalyzer."""
        return BusinessTopologicalAnalyzer()

    def test_healthy_dag(self, analyzer):
        """
        Prueba un Grafo Acíclico Dirigido (DAG) saludable con Estabilidad Piramidal.

        Verifica que:
        1. Una estructura con base sólida (más insumos que APUs) genere un score alto.
        2. No se detecten riesgos circulares.
        3. No se detecten alertas de desperdicio.
        4. El nivel de complejidad sea categorizado correctamente.
        """
        # Crear un grafo con una base sólida (Más INSUMO que APU)
        G = nx.DiGraph()

        # Para obtener un score de estabilidad alto, necesitamos una alta proporción Insumos/APUs
        # Con 1 APU y 200 Insumos, la estabilidad es alta.

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

        # Verificar que el reporte de auditoría se genere correctamente
        result = analyzer.analyze_structural_integrity(G)
        audit_lines = analyzer.get_audit_report(result)
        assert any("AUDITORIA ESTRUCTURAL" in line for line in audit_lines)

    def test_circular_reference(self, analyzer):
        """
        Prueba la detección de referencias circulares (errores lógicos).

        Verifica que:
        1. Se penalice severamente el score de integridad.
        2. Se detecten los riesgos circulares.
        3. El mensaje de error contenga el texto esperado 'ciclo(s)'.
        """
        G = nx.DiGraph()
        G.add_edge("APU1", "APU2")
        G.add_edge("APU2", "APU1")  # Ciclo

        nx.set_node_attributes(G, {"APU1": "APU", "APU2": "APU"}, "type")

        report = analyzer.generate_executive_report(G)

        assert report.integrity_score <= 50.0
        assert len(report.circular_risks) > 0
        # Expectativas de mensaje actualizadas para V3
        assert any("ciclo(s)" in r for r in report.circular_risks)

        # Verificar reporte de auditoría compatible
        result = analyzer.analyze_structural_integrity(G)
        audit_lines = analyzer.get_audit_report(result)
        assert any("ALERTA" in line for line in audit_lines)

    def test_isolated_nodes(self, analyzer):
        """
        Prueba la detección de nodos aislados (desperdicio).

        Verifica que:
        1. Se penalice el score de integridad.
        2. Se generen alertas de desperdicio.
        3. El mensaje contenga 'nodo(s) aislado(s)'.
        """
        G = nx.DiGraph()
        G.add_node("InsumoFantasma", type="INSUMO")

        report = analyzer.generate_executive_report(G)

        assert report.integrity_score < 100.0
        assert len(report.waste_alerts) > 0
        assert any("nodo(s) aislado(s)" in alert for alert in report.waste_alerts)

        result = analyzer.analyze_structural_integrity(G)
        audit_lines = analyzer.get_audit_report(result)
        assert any("ADVERTENCIA" in line for line in audit_lines)

    def test_orphan_insumos(self, analyzer):
        """
        Prueba la detección de insumos huérfanos (definidos pero no usados).
        """
        G = nx.DiGraph()
        # Insumo Huérfano: ¿Debe estar totalmente aislado o solo sin aristas entrantes?
        # _classify_anomalous_nodes: si in_degree == 0 -> orphan_insumos (si tipo INSUMO)

        # Escenario 1: Insumo Totalmente Aislado
        G.add_node("InsumoOrphan", type="INSUMO")

        report = analyzer.generate_executive_report(G)

        # Debe disparar alertas
        assert len(report.waste_alerts) > 0
        # El formato del mensaje en generate_executive_report es "nodo(s) aislado(s)"
        assert any("nodo(s) aislado(s)" in alert for alert in report.waste_alerts)

    def test_integration_backward_compatibility(self, analyzer):
        """
        Prueba que la salida del método analyze_structural_integrity pueda ser procesada
        por get_audit_report (compatibilidad hacia atrás).
        """
        G = nx.DiGraph()
        G.add_edge("A", "B")

        result = analyzer.analyze_structural_integrity(G)

        audit_lines = analyzer.get_audit_report(result)

        assert any("AUDITORIA ESTRUCTURAL" in line for line in audit_lines)
        assert any("Ciclos de Costo" in line for line in audit_lines)
