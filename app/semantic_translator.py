# -*- coding: utf-8 -*-
"""
Módulo de Traducción Semántica.

Este módulo actúa como un puente lingüístico entre las métricas técnicas
(topología, finanzas) y el lenguaje de negocio estratégico. Transforma
datos duros en narrativas accionables para gerentes de proyectos.
"""

import logging
import random
from typing import Any, Dict

from agent.business_topology import TopologicalMetrics

logger = logging.getLogger(__name__)


class SemanticTranslator:
    """
    Traductor semántico que convierte métricas técnicas en narrativa estratégica.
    """

    def translate_topology(self, metrics: TopologicalMetrics, stability: float = 0.0) -> str:
        """
        Traduce métricas topológicas (Betti numbers, estabilidad) a lenguaje de negocio.

        Args:
            metrics (TopologicalMetrics): Métricas de Betti (β0, β1).
            stability (float): Métrica de estabilidad piramidal (Ψ).

        Returns:
            str: Narrativa sobre la salud estructural del proyecto.
        """
        narrative_parts = []

        # 1. Interpretación de Beta 1 (Ciclos) -> Bloqueos Logísticos
        if metrics.beta_1 > 0:
            narrative_parts.append(
                f"🚨 **Bloqueos Logísticos Detectados**: Se han identificado {metrics.beta_1} dependencias circulares "
                "en la estructura de costos. Esto representa riesgos críticos de sobrecostos por retrabajos administrativos."
            )
        else:
            narrative_parts.append(
                "✅ **Flujo Logístico Optimizado**: No se detectan dependencias circulares. La estructura de costos es directa y auditable."
            )

        # 2. Interpretación de Beta 0 (Componentes) -> Coherencia del Alcance
        if metrics.beta_0 > 1:
            narrative_parts.append(
                f"⚠️ **Fragmentación de Recursos**: El presupuesto muestra {metrics.beta_0} islas de información desconectadas. "
                "Esto sugiere que hay insumos o APUs sin una trazabilidad clara hacia el proyecto central."
            )
        else:
            narrative_parts.append(
                "🔗 **Cohesión del Proyecto**: La totalidad del alcance está conectada en una estructura unificada."
            )

        # 3. Interpretación de Estabilidad (Ψ) -> Robustez de la Cadena de Suministro
        # Umbrales heurísticos: < 1.0 (Riesgoso/Invertida), > 10.0 (Sólida), muy alto (Dispersa)
        if stability < 1.0:
            narrative_parts.append(
                f"📉 **Robustez de Cadena de Suministro (Crítica)**: El índice de estabilidad es bajo ({stability:.2f}). "
                "La base de insumos es insuficiente para soportar la complejidad de los APUs definidos (Pirámide Invertida)."
            )
        elif stability > 20.0:
            narrative_parts.append(
                f"🛡️ **Robustez de Cadena de Suministro (Sólida)**: El índice de estabilidad es alto ({stability:.2f}), "
                "indicando una base de recursos diversificada y resiliente ante interrupciones."
            )
        else:
            narrative_parts.append(
                f"⚖️ **Robustez de Cadena de Suministro (Equilibrada)**: El índice de estabilidad ({stability:.2f}) "
                "muestra una relación saludable entre insumos elementales y actividades compuestas."
            )

        return "\n".join(narrative_parts)

    def translate_financial(self, metrics: Dict[str, Any]) -> str:
        """
        Traduce métricas financieras (VaR, WACC, ROI) a lenguaje de inversión estratégica.

        Args:
            metrics (Dict[str, Any]): Diccionario de métricas del FinancialEngine.

        Returns:
            str: Narrativa sobre la viabilidad económica y riesgos financieros.
        """
        narrative_parts = []

        # Extraer métricas clave
        wacc = metrics.get("wacc", 0.0)
        var_value = metrics.get("var", 0.0)
        contingency = metrics.get("contingency", {})
        performance = metrics.get("performance", {})
        recommendation = performance.get("recommendation", "REVISAR")
        profitability_index = performance.get("profitability_index", 0.0)

        # 1. WACC -> Costo de Oportunidad
        narrative_parts.append(
            f"💰 **Costo de Oportunidad del Capital (WACC)**: {wacc:.2%}. "
            "Este es el rendimiento mínimo que el proyecto debe generar para satisfacer a los inversores y acreedores."
        )

        # 2. VaR -> Exposición al Riesgo
        recommended_cont = contingency.get("recommended", 0.0)
        narrative_parts.append(
            f"📊 **Exposición al Riesgo Financiero**: Se estima una contingencia sugerida de ${recommended_cont:,.2f} "
            f"(basada en VaR y volatilidad de mercado) para blindar el margen del proyecto."
        )

        # 3. Recomendación Accionable
        if recommendation == "ACEPTAR":
            narrative_parts.append(
                f"🚀 **Veredicto de Viabilidad**: El proyecto es FINANCIERAMENTE VIABLE (Índice de Rentabilidad: {profitability_index:.2f}). "
                "Se recomienda proceder, manteniendo vigilancia sobre la contingencia sugerida."
            )
        elif recommendation == "RECHAZAR":
            narrative_parts.append(
                f"🛑 **Veredicto de Viabilidad**: El proyecto presenta RIESGOS CRÍTICOS (Índice de Rentabilidad: {profitability_index:.2f}). "
                "Se recomienda reestructurar los costos o buscar eficiencias operativas antes de aprobar."
            )
        else:
            narrative_parts.append(
                "🔍 **Veredicto de Viabilidad**: Se requiere una revisión manual profunda debido a inconsistencias en los flujos o inversión inicial."
            )

        return "\n".join(narrative_parts)

    def _get_market_context(self) -> str:
        """
        Simula la obtención de inteligencia de mercado externa.
        """
        # En el futuro, esto conectará con APIs reales.
        tendencias = [
            "📈 Inflación en materiales de acero (+2.5% m/m). Se sugiere stockeo anticipado.",
            "📉 Tipo de cambio favorable para importaciones. Oportunidad de negociar con proveedores extranjeros.",
            "⚠️ Escasez de mano de obra calificada en la región. Considerar ajustar rendimientos en APUs.",
            "⚖️ Estabilidad en precios del cemento y agregados.",
            "🌪️ Alta volatilidad energética proyectada para el próximo trimestre."
        ]
        selected_trend = random.choice(tendencias)
        return f"🌍 **Contexto de Mercado**: {selected_trend}"

    def compose_strategic_narrative(
        self, topo_metrics: TopologicalMetrics, fin_metrics: Dict[str, Any], stability: float = 0.0
    ) -> str:
        """
        Compone el reporte ejecutivo final combinando todas las dimensiones.

        Args:
            topo_metrics (TopologicalMetrics): Métricas estructurales.
            fin_metrics (Dict[str, Any]): Métricas financieras.
            stability (float): Estabilidad piramidal.

        Returns:
            str: Texto consolidado listo para el reporte ejecutivo.
        """
        topo_narrative = self.translate_topology(topo_metrics, stability)
        fin_narrative = self.translate_financial(fin_metrics)
        market_narrative = self._get_market_context()

        full_narrative = (
            "## 🏗️ INFORME DE INTELIGENCIA ESTRATÉGICA\n\n"
            "### 1. Salud Estructural y Operativa\n"
            f"{topo_narrative}\n\n"
            "### 2. Análisis de Viabilidad Económica\n"
            f"{fin_narrative}\n\n"
            "### 3. Inteligencia de Mercado\n"
            f"{market_narrative}\n\n"
            "### 💡 Recomendación Estratégica\n"
            f"{self._generate_final_advice(topo_metrics, fin_metrics)}"
        )
        return full_narrative

    def _generate_final_advice(self, topo_metrics: TopologicalMetrics, fin_metrics: Dict[str, Any]) -> str:
        """Genera una frase de cierre contundente."""
        beta_1 = topo_metrics.beta_1
        recommendation = fin_metrics.get("performance", {}).get("recommendation", "REVISAR")

        if beta_1 > 0 and recommendation == "RECHAZAR":
            return "❌ **ACCIÓN INMEDIATA REQUERIDA**: El proyecto es inviable técnica y financieramente. Detener procesos de contratación y auditar dependencias circulares."
        elif beta_1 > 0:
            return "⚠️ **PROCEDER CON CAUTELA**: La viabilidad financiera es positiva, pero los errores lógicos en el presupuesto (ciclos) deben corregirse antes de la ejecución para evitar litigios."
        elif recommendation == "RECHAZAR":
            return "📉 **REVISIÓN FINANCIERA**: La estructura técnica es sólida, pero los números no cierran. Revisar alcance o buscar fuentes de financiamiento más baratas."
        elif recommendation == "ACEPTAR":
            return "✅ **LUZ VERDE**: El proyecto demuestra coherencia técnica y solidez financiera. Proceder a la siguiente fase de planificación."
        else:
             return "🔍 **EVALUACIÓN INCOMPLETA**: No hay suficiente certeza financiera para dar luz verde (Estado: REVISAR). Auditar entradas de inversión y flujos."
