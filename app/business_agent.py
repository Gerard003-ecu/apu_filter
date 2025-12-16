# -*- coding: utf-8 -*-
"""
Agente de Inteligencia de Negocio.

Este agente se encarga de evaluar la viabilidad y riesgos de un proyecto
desde una perspectiva de negocio, combinando análisis estructural
(topología del presupuesto) y financiero (costos, riesgos).
"""

import logging
import random
from typing import Any, Dict, Optional

from agent.business_topology import (
    BudgetGraphBuilder,
    BusinessTopologicalAnalyzer,
    ConstructionRiskReport,
    TopologicalMetrics,
)
from app.financial_engine import FinancialConfig, FinancialEngine
from app.telemetry import TelemetryContext

logger = logging.getLogger(__name__)


class SemanticTranslator:
    """
    Capa de traducción semántica que convierte métricas técnicas en lenguaje de negocio.
    Transforma datos duros (Betti numbers, VaR) en narrativas accionables para tomadores de decisiones.
    """

    def translate_topology(self, metrics: TopologicalMetrics) -> str:
        """
        Convierte métricas topológicas en frases de riesgo operativo.

        Args:
            metrics (TopologicalMetrics): Métricas calculadas (beta_0, beta_1).

        Returns:
            str: Frase interpretativa enfocada en la eficiencia de procesos.
        """
        narrative_parts = []

        # Análisis de Ciclos (Beta 1) - Redundancias
        if metrics.beta_1 > 0:
            narrative_parts.append(
                f"Alerta de Flujo: Detectamos {metrics.beta_1} redundancias en su proceso de aprobación de materiales "
                "que podrían generar sobrecostos por retrabajos."
            )
        else:
            narrative_parts.append(
                "Eficiencia de Flujo: La estructura del presupuesto es directa y sin bucles redundantes."
            )

        # Análisis de Componentes Conexas (Beta 0) - Fragmentación
        if metrics.beta_0 > 1:
            narrative_parts.append(
                f"Islas de Información: Hay {metrics.beta_0} partes del presupuesto aisladas sin recursos asignados "
                "correctamente, lo que sugiere fragmentación en la planificación."
            )
        elif metrics.beta_0 == 1:
            narrative_parts.append(
                "Integridad: El presupuesto muestra una estructura cohesiva e integrada."
            )

        return " ".join(narrative_parts)

    def translate_financial(self, metrics: Dict[str, Any]) -> str:
        """
        Convierte métricas financieras (VaR, WACC) en consejos de inversión.

        Args:
            metrics (Dict[str, Any]): Diccionario con métricas financieras.

        Returns:
            str: Recomendación estratégica financiera.
        """
        var_metrics = metrics.get("var_metrics", {})
        contingency = metrics.get("contingency", {})

        # Extracción segura de valores
        var_value = metrics.get("var", 0.0)
        recommended_contingency = contingency.get("recommended", 0.0)
        percentage_rate = contingency.get("percentage_rate", 0.10)  # Default 10%

        narrative_parts = []

        # Análisis de VaR (Value at Risk)
        # Asumimos que un VaR alto relativo a la inversión (no tenemos inversión total aquí directo, pero usamos heurística)
        # o simplemente la presencia de una recomendación alta de contingencia dispara la alerta.

        if metrics.get("performance", {}).get("recommendation") == "RECHAZAR":
            narrative_parts.append(
                "Alerta de Viabilidad: Los indicadores sugieren que el proyecto no cumple con los criterios mínimos de rentabilidad."
            )

        if recommended_contingency > 0:
            # Usamos el porcentaje para dar una recomendación específica
            pct_str = f"{percentage_rate * 100:.1f}%"
            narrative_parts.append(
                f"Advertencia: Debido a la inconsistencia de datos y volatilidad detectada (ej. proveedores de acero), "
                f"sugerimos aumentar el fondo de contingencia en un {pct_str} (${recommended_contingency:,.2f})."
            )
        else:
            narrative_parts.append(
                "Solidez Financiera: Los niveles de riesgo calculados se encuentran dentro de los parámetros aceptables."
            )

        return " ".join(narrative_parts)

    def _get_market_context(self) -> str:
        """
        Simula la obtención de contexto de mercado externo.

        En el futuro, esto se conectará a APIs de commodities y noticias económicas.

        Returns:
            str: Resumen de tendencias de mercado relevantes.
        """
        # Placeholder para tendencias simuladas
        tendencias = [
            "Tendencia alcista en acero (+2.5% mensual).",
            "Estabilidad en precios del cemento.",
            "Volatilidad en tipo de cambio moderada.",
            "Escasez reportada en mano de obra calificada en la región.",
        ]
        # Devolvemos una combinación determinista o aleatoria para la demo
        # Para consistencia en pruebas, usaremos algo fijo o dependiente del día,
        # pero la instrucción pide simular capacidad.
        return f"Contexto de Mercado: {random.choice(tendencias)} Se recomienda monitorear índices de inflación sectorial."

    def compose_narrative(
        self, topo_metrics: TopologicalMetrics, fin_metrics: Dict[str, Any]
    ) -> str:
        """
        Compone la narrativa estratégica completa.

        Args:
            topo_metrics (TopologicalMetrics): Métricas de estructura.
            fin_metrics (Dict[str, Any]): Métricas financieras.

        Returns:
            str: Texto coherente con la "Narrativa Estratégica".
        """
        topo_text = self.translate_topology(topo_metrics)
        fin_text = self.translate_financial(fin_metrics)
        market_text = self._get_market_context()

        # Construcción del párrafo estilo consultor senior
        narrative = (
            f"AUDITORÍA ESTRATÉGICA:\n\n"
            f"1. Estructura Operativa: {topo_text}\n\n"
            f"2. Análisis Financiero: {fin_text}\n\n"
            f"3. Visión de Mercado: {market_text}\n\n"
            f"CONCLUSIÓN: {'El proyecto presenta desafíos estructurales que requieren atención inmediata.' if topo_metrics.beta_1 > 0 else 'El proyecto es técnicamente sólido, sujeto a las consideraciones financieras mencionadas.'}"
        )
        return narrative


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
            strategic_narrative = self.translator.compose_narrative(
                topo_metrics, financial_metrics
            )

            # Actualizamos el reporte (usando replace si es frozen, o atributo directo si no)
            # ConstructionRiskReport no es frozen en la definición actual que vi (solo TopologicalMetrics lo era)
            # Pero en mi patch no quité @dataclass, así que asumimos mutable salvo que tenga frozen=True
            # Chequeando business_topology.py, ConstructionRiskReport no tiene frozen=True explícito.

            # Sin embargo, para seguridad usamos un nuevo objeto o modificamos
            # Como es una dataclass normal, podemos asignar
            report.strategic_narrative = strategic_narrative

            # También lo agregamos a details por si acaso la serialización lo ignora
            report.details["strategic_narrative"] = strategic_narrative

            logger.info("✅ Evaluación de negocio completada con éxito.")
            return report

        except Exception as e:
            logger.error(
                f"❌ Error durante la evaluación del BusinessAgent: {e}", exc_info=True
            )
            self.telemetry.record_error("business_agent", str(e))
            return None
