import pytest
import networkx as nx
from agent.business_topology import TopologicalMetrics
from app.semantic_translator import SemanticTranslator

class TestSemanticTranslator:
    def setup_method(self):
        self.translator = SemanticTranslator()

    def test_translate_topology_with_cycles(self):
        metrics = TopologicalMetrics(beta_0=1, beta_1=2, euler_characteristic=-1)
        narrative = self.translator.translate_topology(metrics, stability=5.0)

        assert "Bloqueos Logísticos Detectados" in narrative
        assert "2 dependencias circulares" in narrative

    def test_translate_topology_clean(self):
        # stability > 20.0 triggers "Sólida"
        metrics = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)
        narrative = self.translator.translate_topology(metrics, stability=25.0)

        assert "Flujo Logístico Optimizado" in narrative
        assert "Cohesión del Proyecto" in narrative
        assert "Robustez de Cadena de Suministro (Sólida)" in narrative

    def test_translate_financial_success(self):
        metrics = {
            "wacc": 0.12,
            "var": 50000.0,
            "contingency": {"recommended": 60000.0},
            "performance": {"recommendation": "ACEPTAR", "profitability_index": 1.25}
        }

        narrative = self.translator.translate_financial(metrics)

        # Checking with Markdown formatting
        assert "**Costo de Oportunidad del Capital (WACC)**: 12.00%" in narrative
        assert "Exposición al Riesgo Financiero" in narrative
        assert "Veredicto de Viabilidad" in narrative
        assert "FINANCIERAMENTE VIABLE" in narrative

    def test_compose_strategic_narrative(self):
        topo_metrics = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)
        fin_metrics = {
            "wacc": 0.10,
            "var": 1000.0,
            "contingency": {"recommended": 1500.0},
            "performance": {"recommendation": "ACEPTAR"}
        }

        full_report = self.translator.compose_strategic_narrative(topo_metrics, fin_metrics, stability=12.0)

        assert "INFORME DE INTELIGENCIA ESTRATÉGICA" in full_report
        assert "Salud Estructural y Operativa" in full_report
        assert "Análisis de Viabilidad Económica" in full_report
        assert "Inteligencia de Mercado" in full_report
        assert "LUZ VERDE" in full_report

    def test_revisar_status(self):
        topo_metrics = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)
        fin_metrics = {
            "wacc": 0.10,
            "var": 1000.0,
            "contingency": {"recommended": 1500.0},
            "performance": {"recommendation": "REVISAR"}
        }

        full_report = self.translator.compose_strategic_narrative(topo_metrics, fin_metrics, stability=12.0)

        assert "EVALUACIÓN INCOMPLETA" in full_report
        assert "LUZ VERDE" not in full_report
