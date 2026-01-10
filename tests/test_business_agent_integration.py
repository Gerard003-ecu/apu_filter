"""
Pruebas de integración para el Agente de Negocios (BusinessAgent).

Este módulo verifica el flujo completo de evaluación de proyectos, asegurando
que la integración entre los componentes de topología, financieros y de
traducción semántica funcione correctamente dentro del agente.
"""

import pandas as pd
import pytest

from app.business_agent import BusinessAgent


class TestBusinessAgentIntegration:
    """Suite de pruebas de integración para BusinessAgent."""

    def setup_method(self):
        """Configuración inicial para cada prueba."""
        # Configuración simulada
        self.config = {"financial_config": {"risk_free_rate": 0.05}}
        self.agent = BusinessAgent(self.config)

    def test_evaluate_project_flow(self):
        """
        Prueba el flujo completo de `evaluate_project`.

        Verifica que:
        1. El agente acepte el contexto con DataFrames y parámetros financieros.
        2. Se genere un reporte de riesgo (`ConstructionRiskReport`).
        3. La narrativa estratégica contenga las secciones clave esperadas.
        4. Los detalles del reporte incluyan métricas financieras y topológicas.
        """
        # Crear DataFrames simulados
        df_presupuesto = pd.DataFrame(
            {
                "CODIGO_APU": ["APU-1"],
                "DESCRIPCION_APU": ["Test APU"],
                "CANTIDAD_PRESUPUESTO": [10.0],
            }
        )

        df_merged = pd.DataFrame(
            {
                "CODIGO_APU": ["APU-1"],
                "DESCRIPCION_INSUMO": ["Insumo 1"],
                "TIPO_INSUMO": ["MATERIAL"],
                "COSTO_INSUMO_EN_APU": [100.0],
                "CANTIDAD_APU": [1.0],
            }
        )

        context = {
            "df_presupuesto": df_presupuesto,
            "df_merged": df_merged,
            "initial_investment": 1000.0,
            "cash_flows": [200.0, 300.0, 400.0],
        }

        # Ejecutar la evaluación
        report = self.agent.evaluate_project(context)

        # Verificaciones
        assert report is not None
        assert report.strategic_narrative is not None
        assert "INFORME DE INGENIERÍA ESTRATÉGICA" in report.strategic_narrative

        # Verificar que la narrativa contenga las partes de topología y finanzas
        assert "Auditoría de Integridad Estructural" in report.strategic_narrative
        assert "Análisis de Cargas Financieras" in report.strategic_narrative
        assert "Geotecnia de Mercado" in report.strategic_narrative
        assert "Dictamen del Ingeniero Jefe" in report.strategic_narrative

        # Verificar que los detalles se hayan poblado correctamente
        assert "metrics" in report.details
        assert "financial_metrics_input" in report.details
        assert "strategic_narrative" in report.details


if __name__ == "__main__":
    t = TestBusinessAgentIntegration()
    t.setup_method()
    t.test_evaluate_project_flow()
    print("Integration test passed.")
