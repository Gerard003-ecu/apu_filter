# -*- coding: utf-8 -*-
"""
Agente de Inteligencia de Negocio.

Este agente se encarga de evaluar la viabilidad y riesgos de un proyecto
desde una perspectiva de negocio, combinando análisis estructural
(topología del presupuesto) y financiero (costos, riesgos).
"""

import logging
from typing import Dict, Any, Optional

from agent.business_topology import (
    BudgetGraphBuilder,
    BusinessTopologicalAnalyzer,
    ConstructionRiskReport,
)
from app.financial_engine import FinancialEngine, FinancialConfig
from app.telemetry import TelemetryContext

logger = logging.getLogger(__name__)


class BusinessAgent:
    """
    Orquesta la inteligencia de negocio para evaluar proyectos de construcción.
    """

    def __init__(self, config: Dict[str, Any], telemetry: Optional[TelemetryContext] = None):
        """
        Inicializa el agente de negocio.

        Args:
            config (Dict[str, Any]): Configuración global de la aplicación.
            telemetry (Optional[TelemetryContext]): Contexto para telemetría.
        """
        self.config = config
        self.telemetry = telemetry or TelemetryContext()
        self.graph_builder = BudgetGraphBuilder()
        self.topological_analyzer = BusinessTopologicalAnalyzer(self.telemetry)

        # Inicializar motor financiero con configuración por defecto o específica
        financial_config_data = self.config.get("financial_config", {})
        financial_config = FinancialConfig(**financial_config_data)
        self.financial_engine = FinancialEngine(financial_config)

    def evaluate_project(self, context: Dict[str, Any]) -> Optional[ConstructionRiskReport]:
        """
        Ejecuta una evaluación completa del proyecto.

        Args:
            context (Dict[str, Any]): El contexto del pipeline con los dataframes.

        Returns:
            Optional[ConstructionRiskReport]: Un reporte de riesgos si la evaluación es exitosa.
        """
        logger.info("🤖 Iniciando evaluación de negocio del proyecto...")

        df_presupuesto = context.get("df_presupuesto")
        df_apus_detail = context.get("df_merged")

        if df_presupuesto is None or df_apus_detail is None:
            logger.warning("DataFrames requeridos no disponibles para BusinessAgent.")
            return None

        try:
            # 1. Construir el grafo de negocio
            logger.info("🏗️  Paso 1: Construyendo topología del presupuesto...")
            graph = self.graph_builder.build(df_presupuesto, df_apus_detail)

            # 2. Analizar la topología para obtener riesgos estructurales
            logger.info("🧠  Paso 2: Analizando integridad estructural...")
            # El reporte base se genera con la información topológica
            report = self.topological_analyzer.generate_executive_report(graph)

            # 3. (Opcional) Enriquecer con análisis financiero si hay datos
            # Esta sección se puede expandir para tomar datos del contexto
            # Por ahora, se mantiene simple
            logger.info("💰  Paso 3: Realizando análisis financiero (simulado)...")

            # Aquí se podrían extraer métricas financieras del contexto
            # Por ejemplo: initial_investment, expected_cash_flows, etc.
            # financial_metrics = self.financial_engine.analyze_project(...)
            # report.financial_risk_level = financial_metrics['...']

            logger.info("✅ Evaluación de negocio completada.")
            return report

        except Exception as e:
            logger.error(f"❌ Error durante la evaluación del BusinessAgent: {e}", exc_info=True)
            self.telemetry.record_error("business_agent", str(e))
            return None
