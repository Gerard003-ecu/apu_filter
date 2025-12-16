
import pytest
import logging
from app.business_agent import SemanticTranslator, TopologicalMetrics

# Configure logging to see output if needed
logging.basicConfig(level=logging.INFO)

class TestSemanticTranslator:
    def setup_method(self):
        self.translator = SemanticTranslator()

    def test_translate_topology_redundancies(self):
        # Case: Betti 1 > 0 (Cycles/Redundancies)
        metrics = TopologicalMetrics(beta_0=1, beta_1=3, euler_characteristic=-2)
        text = self.translator.translate_topology(metrics)
        assert "Alerta de Flujo" in text
        assert "3 redundancias" in text
        assert "procesos aislados" not in text

    def test_translate_topology_islands(self):
        # Case: Betti 0 > 1 (Disconnected components)
        metrics = TopologicalMetrics(beta_0=4, beta_1=0, euler_characteristic=4)
        text = self.translator.translate_topology(metrics)
        assert "Islas de Información" in text
        assert "4 partes del presupuesto aisladas" in text
        assert "Eficiencia de Flujo" in text # Beta 1 is 0

    def test_translate_topology_optimal(self):
        # Case: Connected and Acyclic
        metrics = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)
        text = self.translator.translate_topology(metrics)
        assert "Eficiencia de Flujo" in text
        assert "Integridad" in text

    def test_translate_financial_high_risk(self):
        metrics = {
            "var": 50000.0,
            "contingency": {
                "recommended": 15000.0,
                "percentage_rate": 0.08
            },
            "performance": {"recommendation": "ACEPTAR"}
        }
        text = self.translator.translate_financial(metrics)
        assert "Advertencia" in text
        assert "aumentar el fondo de contingencia en un 8.0%" in text
        assert "$15,000.00" in text

    def test_translate_financial_low_risk(self):
        metrics = {
            "var": 1000.0,
            "contingency": {
                "recommended": 0.0
            },
            "performance": {"recommendation": "ACEPTAR"}
        }
        text = self.translator.translate_financial(metrics)
        assert "Solidez Financiera" in text

    def test_compose_narrative(self):
        topo_metrics = TopologicalMetrics(beta_0=1, beta_1=2, euler_characteristic=-1)
        fin_metrics = {
            "var": 20000.0,
            "contingency": {"recommended": 5000.0},
             "performance": {"recommendation": "ACEPTAR"}
        }

        narrative = self.translator.compose_narrative(topo_metrics, fin_metrics)

        assert "AUDITORÍA ESTRATÉGICA" in narrative
        assert "1. Estructura Operativa" in narrative
        assert "2. Análisis Financiero" in narrative
        assert "3. Visión de Mercado" in narrative
        assert "CONCLUSIÓN" in narrative
        assert "desafíos estructurales" in narrative # Due to beta_1 > 0

    def test_market_context(self):
        text = self.translator._get_market_context()
        assert "Contexto de Mercado" in text
        assert len(text) > 20

if __name__ == "__main__":
    # Manually run if executed as script
    t = TestSemanticTranslator()
    t.setup_method()
    t.test_translate_topology_redundancies()
    print("Tests passed manually.")
