# -*- coding: utf-8 -*-
"""
Agente de Inteligencia de Negocio.

Este agente se encarga de evaluar la viabilidad y riesgos de un proyecto
desde una perspectiva de negocio, combinando análisis estructural
(topología del presupuesto) y financiero (costos, riesgos).
"""

import logging
from typing import Any, Dict, Optional

from agent.business_topology import (
    BudgetGraphBuilder,
    BusinessTopologicalAnalyzer,
    ConstructionRiskReport,
)
from app.financial_engine import FinancialConfig, FinancialEngine
from app.telemetry import TelemetryContext
from app.semantic_translator import SemanticTranslator

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
        self.translator = SemanticTranslator()

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

            # 2. Obtener métricas topológicas puras para el traductor
            topo_metrics = self.topological_analyzer.calculate_betti_numbers(graph)
            pyramid_stability = self.topological_analyzer.calculate_pyramid_stability(graph)

            # 3. Análisis Financiero (Simulado/Real)
            logger.info("💰  Paso 2: Realizando análisis financiero...")

            # Intentamos obtener datos del contexto para el análisis financiero
            # En un entorno real, estos vendrían de input del usuario o inferencias
            initial_investment = context.get("initial_investment", 1000000.0)  # Placeholder

            # Simulamos flujos para la demo si no existen
            cash_flows = context.get(
                "cash_flows", [initial_investment * 0.3 for _ in range(5)]
            )

            financial_metrics = self.financial_engine.analyze_project(
                initial_investment=initial_investment,
                expected_cash_flows=cash_flows,
                cost_std_dev=initial_investment * 0.15,  # Asumimos 15% desviación estándar
                project_volatility=0.20,
            )

            # 4. Generar Reporte Base
            logger.info("🧠  Paso 3: Integrando inteligencia y generando narrativa...")
            report = self.topological_analyzer.generate_executive_report(
                graph, financial_metrics
            )

            # 5. Enriquecer con Narrativa Estratégica (Capa Semántica)
            synergy = report.details.get("synergy_risk")
            strategic_narrative = self.translator.compose_strategic_narrative(
                topo_metrics, financial_metrics, stability=pyramid_stability, synergy_risk=synergy
            )

            # Actualizamos el reporte
            # ConstructionRiskReport no es frozen por defecto, así que podemos asignar
            if report:
                report.strategic_narrative = strategic_narrative

                # También lo agregamos a details por si acaso la serialización lo requiere
                report.details["strategic_narrative"] = strategic_narrative
                # Add financial metrics to details to satisfy tests and frontend needs
                report.details["financial_metrics_input"] = financial_metrics

            logger.info("✅ Evaluación de negocio completada con éxito.")
            return report

        except Exception as e:
            logger.error(
                f"❌ Error durante la evaluación del BusinessAgent: {e}", exc_info=True
            )
            self.telemetry.record_error("business_agent", str(e))
            return None
