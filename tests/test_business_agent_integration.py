
import pytest
import pandas as pd
import networkx as nx
from unittest.mock import MagicMock
from app.business_agent import BusinessAgent
from agent.business_topology import TopologicalMetrics

class TestBusinessAgentIntegration:
    def setup_method(self):
        # Setup config
        self.config = {
            "financial_config": {
                "risk_free_rate": 0.05
            }
        }
        self.agent = BusinessAgent(self.config)

    def test_evaluate_project_flow(self):
        # Create dummy dataframes
        df_presupuesto = pd.DataFrame({
            "CODIGO_APU": ["APU-1"],
            "DESCRIPCION_APU": ["Test APU"],
            "CANTIDAD_PRESUPUESTO": [10.0]
        })

        df_merged = pd.DataFrame({
            "CODIGO_APU": ["APU-1"],
            "DESCRIPCION_INSUMO": ["Insumo 1"],
            "TIPO_INSUMO": ["MATERIAL"],
            "COSTO_INSUMO_EN_APU": [100.0],
            "CANTIDAD_APU": [1.0]
        })

        context = {
            "df_presupuesto": df_presupuesto,
            "df_merged": df_merged,
            "initial_investment": 1000.0,
            "cash_flows": [200.0, 300.0, 400.0]
        }

        # Execute
        report = self.agent.evaluate_project(context)

        # Verification
        assert report is not None
        assert report.strategic_narrative is not None
        assert "AUDITORÍA ESTRATÉGICA" in report.strategic_narrative
        # Verify narrative contains topology and finance parts
        assert "Estructura Operativa" in report.strategic_narrative
        assert "Análisis Financiero" in report.strategic_narrative

        # Verify details were populated
        assert "metrics" in report.details
        assert "financial_metrics_input" in report.details
        assert "strategic_narrative" in report.details

if __name__ == "__main__":
    t = TestBusinessAgentIntegration()
    t.setup_method()
    t.test_evaluate_project_flow()
    print("Integration test passed.")
