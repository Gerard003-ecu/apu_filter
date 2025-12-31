# -*- coding: utf-8 -*-
"""
Agente de Inteligencia de Negocio.

Este agente se encarga de evaluar la viabilidad y riesgos de un proyecto
desde una perspectiva de negocio, combinando análisis estructural
(topología del presupuesto) y financiero (costos, riesgos).
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agent.business_topology import (
    BudgetGraphBuilder,
    BusinessTopologicalAnalyzer,
    ConstructionRiskReport,
)
from app.financial_engine import FinancialConfig, FinancialEngine
from app.telemetry import TelemetryContext
from app.semantic_translator import SemanticTranslator
from app.constants import ColumnNames

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinancialParameters:
    """
    Parámetros financieros para el análisis del proyecto.

    Encapsula los valores de entrada para el motor financiero,
    garantizando inmutabilidad y validación en construcción.
    """

    initial_investment: float
    cash_flows: Tuple[float, ...]
    cost_std_dev: float
    project_volatility: float

    def __post_init__(self):
        if self.initial_investment <= 0:
            raise ValueError("La inversión inicial debe ser positiva")
        if self.cost_std_dev < 0:
            raise ValueError("La desviación estándar no puede ser negativa")
        if not (0 <= self.project_volatility <= 1):
            raise ValueError("La volatilidad debe estar en el rango [0, 1]")


@dataclass
class TopologicalMetricsBundle:
    """
    Conjunto cohesivo de métricas topológicas del presupuesto.

    Agrupa los invariantes topológicos (números de Betti, estabilidad)
    para facilitar su transporte entre componentes del pipeline.
    """

    betti_numbers: Dict[str, Any]
    pyramid_stability: float
    graph: Any  # Tipo del grafo según la implementación

    @property
    def structural_coherence(self) -> float:
        """
        Calcula un índice de coherencia estructural basado en β₀ y β₁.

        β₀ (componentes conexas): Valores altos indican fragmentación.
        β₁ (ciclos independientes): Valores altos indican dependencias circulares.

        Returns:
            float: Índice normalizado [0, 1] donde 1 es máxima coherencia.
        """
        beta_0 = self.betti_numbers.get("beta_0", 1)
        beta_1 = self.betti_numbers.get("beta_1", 0)

        # Penalización por fragmentación (idealmente β₀ = 1)
        fragmentation_penalty = 1.0 / max(beta_0, 1)

        # Penalización por ciclos (decaimiento exponencial)
        import math

        cycle_penalty = math.exp(-0.5 * beta_1)

        return fragmentation_penalty * cycle_penalty * self.pyramid_stability


class RiskChallenger:
    """
    Debate adversarial para auditar la coherencia entre las métricas
    financieras y topológicas del reporte.

    Actúa como un 'Fiscal' que busca contradicciones en el veredicto.
    """

    def challenge_verdict(self, report: ConstructionRiskReport) -> ConstructionRiskReport:
        """
        Analiza la coherencia entre las métricas financieras y topológicas.

        Regla Adversarial:
        Si financial_risk == "BAJO" PERO pyramid_stability < 1.0 (Pirámide Invertida),
        el Challenger debe cambiar el veredicto a "FALSO POSITIVO FINANCIERO"
        y degradar el score de integridad.

        Args:
            report: El reporte preliminar generado por el agente.

        Returns:
            ConstructionRiskReport: El reporte auditado (y posiblemente modificado).
        """
        logger.info("⚖️  Risk Challenger: Auditando coherencia del reporte...")

        # Extraer métricas clave para el debate
        financial_risk = report.financial_risk_level

        # Obtener estabilidad piramidal de los detalles
        # Se asume que está en details['topological_invariants']['pyramid_stability']
        # o directamente en details['pyramid_stability'] según la implementación previa
        details = report.details or {}
        stability = details.get("pyramid_stability")

        # Intentar obtener de la estructura anidada si no está en el primer nivel
        if stability is None and "topological_invariants" in details:
            stability = details["topological_invariants"].get("pyramid_stability")

        # Si no se encuentra, usar un valor seguro que no dispare la alerta (o loguear advertencia)
        if stability is None:
            logger.warning("Risk Challenger: No se encontró métrica de estabilidad piramidal.")
            return report

        # Regla Adversarial: Pirámide Invertida con Riesgo Financiero Bajo
        # "BAJO" debe coincidir con los niveles definidos en el sistema (FinancialRiskLevel)
        # Asumimos que "LOW" o "BAJO" son los valores para riesgo bajo.
        is_financial_safe = str(financial_risk).upper() in ["LOW", "BAJO", "MODERATE", "MODERADO"]
        is_inverted_pyramid = stability < 1.0

        if is_financial_safe and is_inverted_pyramid:
            logger.warning("🚨 Risk Challenger: CONTRADICCIÓN DETECTADA (Pirámide Invertida + Finanzas Sanas)")

            # Degradar veredicto
            new_financial_risk = "RIESGO ESTRUCTURAL OCULTO"

            # Penalizar integridad (ej. reducir un 20%)
            original_integrity = report.integrity_score
            new_integrity = max(0.0, original_integrity * 0.8)

            # Actualizar narrativa estratégica
            new_narrative = (
                f"⚠️ VETO DEL CHALLENGER: {report.strategic_narrative}\n\n"
                f"[FISCALÍA DE RIESGOS]: Se ha detectado una contradicción crítica. "
                f"Aunque los indicadores financieros sugieren solidez ({financial_risk}), "
                f"la estructura topológica es una 'Pirámide Invertida' (Estabilidad {stability:.2f} < 1.0). "
                f"Esto indica que el proyecto es financieramente atractivo pero estructuralmente inviable. "
                f"Se reclasifica como FALSO POSITIVO FINANCIERO."
            )

            # Modificar detalles para reflejar el challenge
            new_details = details.copy()
            new_details["challenger_verdict"] = "VETO_STRUCTURAL_CONTRADICTION"
            new_details["original_financial_risk"] = financial_risk
            new_details["original_integrity_score"] = original_integrity

            # Retornar reporte modificado
            # Usamos replace si es dataclass frozen, o constructor si no
            # ConstructionRiskReport es dataclass, asumimos que no es frozen o usamos constructor
            return ConstructionRiskReport(
                integrity_score=new_integrity,
                waste_alerts=report.waste_alerts,
                circular_risks=report.circular_risks,
                complexity_level=report.complexity_level,
                financial_risk_level=new_financial_risk, # Sobrescribimos el nivel de riesgo
                details=new_details,
                strategic_narrative=new_narrative
            )

        logger.info("✅ Risk Challenger: Coherencia verificada.")
        return report


class BusinessAgent:
    """
    Orquesta la inteligencia de negocio para evaluar proyectos de construcción.

    Combina análisis topológico (estructura del presupuesto como complejo simplicial)
    con análisis financiero (VPN, TIR, simulación de Monte Carlo).
    """

    # Configuración por defecto para parámetros financieros
    DEFAULT_FINANCIAL_PARAMS = {
        "initial_investment": 1_000_000.0,
        "cash_flow_ratio": 0.30,
        "cash_flow_periods": 5,
        "cost_std_dev_ratio": 0.15,
        "project_volatility": 0.20,
    }

    def __init__(self, config: Dict[str, Any], telemetry: Optional[TelemetryContext] = None):
        """
        Inicializa el agente de negocio.

        Args:
            config: Configuración global de la aplicación.
            telemetry: Contexto para telemetría y observabilidad.

        Raises:
            ValueError: Si la configuración financiera es inválida.
        """
        self._validate_config(config)
        self.config = config
        self.telemetry = telemetry or TelemetryContext()

        # Componentes del pipeline (inicialización eager para fail-fast)
        self.graph_builder = BudgetGraphBuilder()
        self.topological_analyzer = BusinessTopologicalAnalyzer(self.telemetry)
        self.translator = SemanticTranslator()
        self.financial_engine = self._create_financial_engine()

        # Inicializar el Challenger
        self.risk_challenger = RiskChallenger()

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Valida la estructura y tipos de la configuración.

        Args:
            config: Diccionario de configuración a validar.

        Raises:
            ValueError: Si la configuración no cumple los requisitos.
        """
        if not isinstance(config, dict):
            raise ValueError("La configuración debe ser un diccionario")

        financial_cfg = config.get("financial_config", {})

        numeric_fields = ["risk_free_rate", "discount_rate", "market_return"]
        for field_name in numeric_fields:
            if field_name in financial_cfg:
                value = financial_cfg[field_name]
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(
                        f"'{field_name}' debe ser un número no negativo, recibido: {value}"
                    )

    def _create_financial_engine(self) -> FinancialEngine:
        """
        Construye el motor financiero con la configuración provista.

        Returns:
            FinancialEngine configurado.
        """
        financial_config_data = self.config.get("financial_config", {})
        financial_config = FinancialConfig(**financial_config_data)
        return FinancialEngine(financial_config)

    def _validate_dataframes(
        self, df_presupuesto: Optional[pd.DataFrame], df_apus_detail: Optional[pd.DataFrame]
    ) -> Tuple[bool, str]:
        """
        Valida que los DataFrames requeridos existan y tengan estructura válida.

        Args:
            df_presupuesto: DataFrame del presupuesto general.
            df_apus_detail: DataFrame con detalle de APUs mergeado.

        Returns:
            Tupla (es_válido, mensaje_de_error).
        """
        if df_presupuesto is None:
            return False, "DataFrame 'df_presupuesto' no disponible"

        if df_apus_detail is None:
            return False, "DataFrame 'df_merged' no disponible"

        if df_presupuesto.empty:
            return False, "DataFrame 'df_presupuesto' está vacío"

        if df_apus_detail.empty:
            return False, "DataFrame 'df_merged' está vacío"

        # Validar columnas mínimas requeridas para construir el grafo
        required_budget_cols = {
            ColumnNames.CODIGO_APU,
            ColumnNames.DESCRIPCION_APU,
        }

        present_cols = set(df_presupuesto.columns)
        missing_cols = required_budget_cols - present_cols

        if missing_cols:
            # Fallback para compatibilidad
            legacy_mapping = {
                "item": ColumnNames.CODIGO_APU,
                "descripcion": ColumnNames.DESCRIPCION_APU,
            }

            still_missing = set()
            for col in missing_cols:
                legacy_name = None
                for leg, new in legacy_mapping.items():
                    if new == col:
                        legacy_name = leg
                        break

                if legacy_name and legacy_name in present_cols:
                    continue
                still_missing.add(col)

            if still_missing:
                return False, f"Columnas faltantes en presupuesto: {still_missing}"

        return True, ""

    def _extract_financial_parameters(self, context: Dict[str, Any]) -> FinancialParameters:
        """
        Extrae y valida los parámetros financieros del contexto.

        Aplica valores por defecto configurables cuando los parámetros
        no están presentes en el contexto.

        Args:
            context: Contexto del pipeline con datos del proyecto.

        Returns:
            FinancialParameters validados y listos para el análisis.
        """
        defaults = self.config.get("default_financial_params", self.DEFAULT_FINANCIAL_PARAMS)

        initial_investment = context.get(
            "initial_investment", defaults["initial_investment"]
        )

        # Generar flujos de caja si no se proporcionan
        if "cash_flows" in context:
            cash_flows = tuple(context["cash_flows"])
        else:
            cash_flow_ratio = defaults["cash_flow_ratio"]
            periods = defaults["cash_flow_periods"]
            cash_flows = tuple(initial_investment * cash_flow_ratio for _ in range(periods))

        # Calcular desviación estándar de costos
        cost_std_dev = context.get(
            "cost_std_dev", initial_investment * defaults["cost_std_dev_ratio"]
        )

        project_volatility = context.get(
            "project_volatility", defaults["project_volatility"]
        )

        return FinancialParameters(
            initial_investment=initial_investment,
            cash_flows=cash_flows,
            cost_std_dev=cost_std_dev,
            project_volatility=project_volatility,
        )

    def _build_topological_model(
        self, df_presupuesto: pd.DataFrame, df_apus_detail: pd.DataFrame
    ) -> TopologicalMetricsBundle:
        """
        Construye el modelo topológico del presupuesto.

        El presupuesto se modela como un complejo simplicial donde:
        - Vértices: Partidas individuales del presupuesto
        - Aristas: Relaciones de composición/dependencia entre partidas
        - Triángulos: Clusters de partidas con dependencias mutuas

        Los números de Betti resultantes caracterizan la estructura:
        - β₀: Número de componentes conexas (fragmentación del presupuesto)
        - β₁: Número de ciclos independientes (dependencias circulares)

        Args:
            df_presupuesto: DataFrame del presupuesto.
            df_apus_detail: DataFrame con detalle de APUs.

        Returns:
            TopologicalMetricsBundle con todas las métricas estructurales.

        Raises:
            RuntimeError: Si la construcción del grafo falla.
        """
        logger.info("🏗️  Construyendo topología del presupuesto...")

        try:
            graph = self.graph_builder.build(df_presupuesto, df_apus_detail)
        except Exception as e:
            raise RuntimeError(f"Error construyendo grafo topológico: {e}") from e

        betti_numbers = asdict(self.topological_analyzer.calculate_betti_numbers(graph))
        pyramid_stability = self.topological_analyzer.calculate_pyramid_stability(graph)

        logger.debug(
            f"Métricas topológicas: β₀={betti_numbers.get('beta_0')}, "
            f"β₁={betti_numbers.get('beta_1')}, "
            f"estabilidad={pyramid_stability:.3f}"
        )

        return TopologicalMetricsBundle(
            betti_numbers=betti_numbers,
            pyramid_stability=pyramid_stability,
            graph=graph,
        )

    def _perform_financial_analysis(self, params: FinancialParameters) -> Dict[str, Any]:
        """
        Ejecuta el análisis financiero del proyecto.

        Args:
            params: Parámetros financieros validados.

        Returns:
            Diccionario con métricas financieras (VPN, TIR, VaR, etc.).

        Raises:
            RuntimeError: Si el análisis financiero falla.
        """
        logger.info("💰 Realizando análisis financiero...")

        try:
            financial_metrics = self.financial_engine.analyze_project(
                initial_investment=params.initial_investment,
                expected_cash_flows=list(params.cash_flows),
                cost_std_dev=params.cost_std_dev,
                project_volatility=params.project_volatility,
            )
        except Exception as e:
            raise RuntimeError(f"Error en análisis financiero: {e}") from e

        logger.debug(f"Métricas financieras calculadas: {list(financial_metrics.keys())}")

        return financial_metrics

    def _compose_enriched_report(
        self,
        topological_bundle: TopologicalMetricsBundle,
        financial_metrics: Dict[str, Any],
        thermal_metrics: Dict[str, Any],
        entropy: float = 0.5,
        exergy: float = 0.6,
    ) -> ConstructionRiskReport:
        """
        Genera el reporte ejecutivo integrando análisis topológico, financiero y TERMODINÁMICO.

        La narrativa estratégica se construye considerando:
        1. Coherencia estructural del presupuesto (invariantes topológicos)
        2. Viabilidad financiera (VPN, TIR, período de recuperación)
        3. Riesgo sistémico (sinergia entre riesgos estructurales y financieros)
        4. Estado Termodinámico (Fiebre del Proyecto, Exergía, Entropía)

        Args:
            topological_bundle: Métricas topológicas del presupuesto.
            financial_metrics: Métricas del análisis financiero.
            thermal_metrics: Métricas de flujo térmico (temperatura del sistema).
            entropy: Entropía del sistema (desde FluxCondenser).
            exergy: Exergía del presupuesto (desde MatterGenerator).

        Returns:
            ConstructionRiskReport completo con narrativa estratégica.
        """
        logger.info("🧠 Integrando inteligencia (Topología + Finanzas + Termodinámica)...")

        # Generar reporte base desde el analizador topológico
        base_report = self.topological_analyzer.generate_executive_report(
            topological_bundle.graph, financial_metrics
        )

        if base_report is None:
            raise RuntimeError("El analizador topológico retornó un reporte nulo")

        # Extraer riesgo de sinergia para la narrativa
        synergy_risk = base_report.details.get("synergy_risk")

        # 1. Obtener Narrativa Estructural y Financiera
        strategic_narrative_base = self.translator.compose_strategic_narrative(
            topological_metrics=topological_bundle.betti_numbers,
            financial_metrics=financial_metrics,
            stability=topological_bundle.pyramid_stability,
            synergy_risk=synergy_risk,
        )

        # 2. Generar Narrativa Termodinámica
        thermo_narrative = self.translator.translate_thermodynamics(
            entropy=entropy,
            exergy=exergy,
            temperature=thermal_metrics.get("system_temperature", 0.0)
        )

        # 3. Fusionar Narrativas
        # Insertar la termodinámica antes del veredicto final si es posible, o al final
        full_narrative = f"{strategic_narrative_base}\n\n### 4. Análisis Termodinámico (Calor y Eficiencia)\n{thermo_narrative}"

        # Enriquecer el reporte con datos adicionales
        enriched_details = {
            **base_report.details,
            "strategic_narrative": full_narrative,
            "financial_metrics_input": financial_metrics,
            "thermal_metrics": thermal_metrics,
            "thermodynamics": {
                "entropy": entropy,
                "exergy": exergy,
                "temperature": thermal_metrics.get("system_temperature", 0.0)
            },
            "structural_coherence": topological_bundle.structural_coherence,
            "topological_invariants": {
                "betti_numbers": topological_bundle.betti_numbers,
                "pyramid_stability": topological_bundle.pyramid_stability,
            },
        }

        # Construir nuevo reporte inmutable con datos enriquecidos
        report = ConstructionRiskReport(
            integrity_score=base_report.integrity_score,
            waste_alerts=base_report.waste_alerts,
            circular_risks=base_report.circular_risks,
            complexity_level=base_report.complexity_level,
            financial_risk_level=base_report.financial_risk_level,
            details=enriched_details,
            strategic_narrative=full_narrative,
        )

        # Aplicar Risk Challenger para auditar el reporte
        audited_report = self.risk_challenger.challenge_verdict(report)

        return audited_report

    def evaluate_project(self, context: Dict[str, Any]) -> Optional[ConstructionRiskReport]:
        """
        Ejecuta una evaluación completa del proyecto.

        El pipeline de evaluación sigue tres fases:

        1. **Análisis Topológico**: Construye un complejo simplicial del presupuesto
           y calcula sus invariantes (números de Betti, estabilidad piramidal).

        2. **Análisis Financiero**: Evalúa VPN, TIR, VaR y realiza simulación
           de Monte Carlo para caracterizar el riesgo.

        3. **Síntesis Estratégica**: Integra ambas perspectivas en una narrativa
           ejecutiva que identifica riesgos y oportunidades.

        4. **Auditoría Adversarial**: El Risk Challenger revisa la coherencia.

        Args:
            context: El contexto del pipeline conteniendo:
                - df_presupuesto: DataFrame del presupuesto general
                - df_merged: DataFrame con detalle de APUs
                - initial_investment (opcional): Inversión inicial
                - cash_flows (opcional): Lista de flujos de caja esperados
                - project_volatility (opcional): Volatilidad del proyecto [0,1]

        Returns:
            ConstructionRiskReport con el análisis completo, o None si falla.
        """
        logger.info("🤖 Iniciando evaluación de negocio del proyecto...")

        # Fase 0: Validación de entrada
        # Preferir df_final si existe
        df_presupuesto = context.get("df_final")
        if df_presupuesto is None:
            df_presupuesto = context.get("df_presupuesto")

        df_apus_detail = context.get("df_merged")

        is_valid, error_msg = self._validate_dataframes(df_presupuesto, df_apus_detail)
        if not is_valid:
            logger.warning(f"Validación fallida: {error_msg}")
            self.telemetry.record_error("business_agent.validation", error_msg)
            return None

        # Fase 1: Análisis Topológico
        try:
            topological_bundle = self._build_topological_model(
                df_presupuesto, df_apus_detail
            )
            # Guardar el grafo en el contexto para otros pasos
            context["graph"] = topological_bundle.graph

        except RuntimeError as e:
            logger.error(f"❌ Fase topológica fallida: {e}", exc_info=True)
            self.telemetry.record_error("business_agent.topology", str(e))
            return None

        # Fase 2: Análisis Financiero
        try:
            financial_params = self._extract_financial_parameters(context)
            financial_metrics = self._perform_financial_analysis(financial_params)
        except (ValueError, RuntimeError) as e:
            logger.error(f"❌ Fase financiera fallida: {e}", exc_info=True)
            self.telemetry.record_error("business_agent.financial", str(e))
            return None

        # Fase 2.5: Análisis Termodinámico (Nuevo)
        try:
            # 1. Flujo Térmico (Topology)
            thermal_metrics = self.topological_analyzer.analyze_thermal_flow(topological_bundle.graph)

            # 2. Entropía (FluxCondenser - Simulado o del contexto si existe)
            # Idealmente vendría de FluxCondenser.get_metrics(), pero aquí extraemos del contexto
            # o usamos un valor por defecto si no se ha ejecutado el condensador aún.
            entropy = context.get("system_entropy", 0.5)

            # 3. Exergía (MatterGenerator - Simulado o del contexto)
            exergy = context.get("budget_exergy", 0.6)

        except Exception as e:
             logger.warning(f"⚠️ Fallo parcial en termodinámica: {e}")
             thermal_metrics = {"system_temperature": 0.0}
             entropy = 0.5
             exergy = 0.5

        # Fase 3 y 4: Síntesis y Auditoría Adversarial
        try:
            report = self._compose_enriched_report(
                topological_bundle,
                financial_metrics,
                thermal_metrics,
                entropy,
                exergy
            )
        except RuntimeError as e:
            logger.error(f"❌ Fase de síntesis fallida: {e}", exc_info=True)
            self.telemetry.record_error("business_agent.synthesis", str(e))
            return None

        logger.info("✅ Evaluación de negocio completada con éxito.")
        return report
