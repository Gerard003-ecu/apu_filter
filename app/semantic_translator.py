# -*- coding: utf-8 -*-
"""
Módulo de Traducción Semántica.

Este módulo actúa como un puente lingüístico entre las métricas técnicas
(topología, finanzas) y el lenguaje de negocio estratégico. Transforma
datos duros en narrativas accionables para gerentes de proyectos,
adoptando un enfoque de Ingeniería Civil/Estructural.
"""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from agent.business_topology import TopologicalMetrics

logger = logging.getLogger(__name__)


class FinancialVerdict(Enum):
    """Enumeración de veredictos financieros para tipado seguro."""
    ACCEPT = "ACEPTAR"
    REJECT = "RECHAZAR"
    REVIEW = "REVISAR"


@dataclass(frozen=True)
class StabilityThresholds:
    """
    Umbrales para interpretación del índice de estabilidad piramidal (Ψ).

    Fundamentación topológica:
    - Ψ < critical: Pirámide Invertida (Cimentación insuficiente).
    - Ψ ≥ solid: Estructura Antisísmica (Base robusta).
    """
    critical: float = 1.0
    solid: float = 10.0


@dataclass(frozen=True)
class TopologicalThresholds:
    """
    Umbrales para interpretación de números de Betti.

    Fundamentación:
    - β₀: Componentes conexos (fragmentación si > 1).
    - β₁: Ciclos independientes (socavones lógicos si > 0).
    """
    connected_components_optimal: int = 1
    cycles_optimal: int = 0


@dataclass(frozen=True)
class WACCThresholds:
    """
    Umbrales para evaluación del Costo Promedio Ponderado de Capital.
    """
    low: float = 0.05
    high: float = 0.15


@dataclass(frozen=True)
class CycleSeverityThresholds:
    """
    Umbrales para gradación de severidad en dependencias circulares (β₁).
    """
    moderate: int = 3
    critical: int = 5


class SemanticTranslator:
    """
    Traductor semántico que convierte métricas técnicas en narrativa de ingeniería estructural.

    Interpreta el presupuesto como una estructura física donde:
    - Insumos = Cimentación de Recursos (Nivel 3)
    - APUs = Cuerpo Táctico (Nivel 2)
    - Capítulos = Pilares Estructurales (Nivel 1)
    - Proyecto = Ápice / Objetivo Final (Nivel 0)
    """

    def __init__(
        self,
        stability_thresholds: Optional[StabilityThresholds] = None,
        topo_thresholds: Optional[TopologicalThresholds] = None,
        market_provider: Optional[Callable[[], str]] = None,
        random_seed: Optional[int] = None,
        wacc_thresholds: Optional[WACCThresholds] = None,
        cycle_severity: Optional[CycleSeverityThresholds] = None
    ) -> None:
        """Inicializa el traductor con configuración opcional."""
        self._validate_init_arguments(
            stability_thresholds, topo_thresholds, wacc_thresholds, cycle_severity
        )

        self.stability_thresholds = stability_thresholds or StabilityThresholds()
        self.topo_thresholds = topo_thresholds or TopologicalThresholds()
        self._wacc_thresholds = wacc_thresholds or WACCThresholds()
        self._cycle_severity = cycle_severity or CycleSeverityThresholds()
        self._market_provider = market_provider

        if random_seed is not None:
            self._rng = random.Random(random_seed)
        else:
            self._rng = random.Random()

        logger.debug(
            "SemanticTranslator inicializado con Lógica Piramidal | "
            f"Ψ_critical={self.stability_thresholds.critical:.2f}"
        )

    def _validate_init_arguments(
        self,
        stability_thresholds: Optional[StabilityThresholds],
        topo_thresholds: Optional[TopologicalThresholds],
        wacc_thresholds: Optional[WACCThresholds],
        cycle_severity: Optional[CycleSeverityThresholds]
    ) -> None:
        """Valida tipos de argumentos de inicialización."""
        type_checks = [
            (stability_thresholds, StabilityThresholds, "stability_thresholds"),
            (topo_thresholds, TopologicalThresholds, "topo_thresholds"),
            (wacc_thresholds, WACCThresholds, "wacc_thresholds"),
            (cycle_severity, CycleSeverityThresholds, "cycle_severity"),
        ]

        for value, expected_type, name in type_checks:
            if value is not None and not isinstance(value, expected_type):
                raise TypeError(
                    f"{name} debe ser {expected_type.__name__}, "
                    f"recibido: {type(value).__name__}"
                )

    def translate_topology(
        self,
        metrics: TopologicalMetrics,
        stability: float = 0.0
    ) -> str:
        """
        Traduce métricas topológicas a una Auditoría de Ingeniería Civil.

        Args:
            metrics: Métricas de Betti (β₀, β₁).
            stability: Índice de estabilidad piramidal (Ψ).

        Returns:
            Narrativa de auditoría estructural.
        """
        self._validate_topological_metrics(metrics, stability)

        narrative_parts: List[str] = []

        # 1. β₁: Genus Estructural / Socavones
        narrative_parts.append(self._translate_cycles(metrics.beta_1))

        # 2. β₀: Coherencia de Obra (Unidad Estructural)
        narrative_parts.append(self._translate_connectivity(metrics.beta_0))

        # 3. Ψ: Solidez de Cimentación (Física del Negocio)
        narrative_parts.append(self._translate_stability(stability))

        return "\n".join(narrative_parts)

    def _validate_topological_metrics(
        self,
        metrics: TopologicalMetrics,
        stability: float
    ) -> None:
        """Valida la coherencia matemática de las métricas."""
        if not isinstance(metrics, TopologicalMetrics):
            raise TypeError(f"Se esperaba TopologicalMetrics, recibido {type(metrics).__name__}")
        if not isinstance(stability, (int, float)):
            raise TypeError("Estabilidad debe ser numérica")

        if metrics.beta_0 < 0 or metrics.beta_1 < 0:
            raise ValueError("Los números de Betti deben ser no-negativos.")
        if stability < 0:
            raise ValueError("La estabilidad Ψ debe ser no-negativa.")

    def _translate_cycles(self, beta_1: int) -> str:
        """
        Traduce β₁ como 'Genus Estructural' o 'Socavones Lógicos'.
        """
        if beta_1 <= self.topo_thresholds.cycles_optimal:
            return (
                "✅ **Integridad Estructural (Genus 0)**: No se detectan socavones lógicos "
                "(β₁ = 0). La Trazabilidad de Carga de Costos fluye verticalmente desde la "
                "Cimentación hasta el Ápice sin recirculaciones."
            )

        genus_label = "Genus Elevado" if beta_1 > 1 else "Genus 1"
        severity = self._classify_cycle_severity(beta_1)

        if severity == "moderate":
            return (
                f"🔶 **Falla Estructural Local ({genus_label})**: Se detectaron {beta_1} "
                "socavones lógicos en la estructura de costos. Estos 'agujeros' impiden "
                "la correcta Trazabilidad de Carga de Costos y deben ser rellenados (corregidos) para "
                "evitar asentamientos diferenciales en el presupuesto."
            )
        else:
            return (
                f"🚨 **Estructura Geológicamente Inestable ({genus_label})**: Se detectó un "
                f"Genus Estructural de {beta_1}, lo que indica una estructura tipo 'esponja' en lugar "
                "de sólida. Existen múltiples bucles de retroalimentación de costos que "
                "impiden la Trazabilidad de Carga de Costos y hacen colapsar cualquier valoración estática."
            )

    def _classify_cycle_severity(self, beta_1: int) -> str:
        if beta_1 >= self._cycle_severity.critical:
            return "critical"
        if beta_1 >= self._cycle_severity.moderate:
            return "severe"
        return "moderate"

    def _translate_connectivity(self, beta_0: int) -> str:
        """
        Traduce β₀ como 'Unidad de Obra' o 'Fragmentación Edilicia'.
        """
        optimal = self.topo_thresholds.connected_components_optimal

        if beta_0 == 0:
            return "⚠️ **Terreno Vacío**: No hay estructura proyectada (β₀ = 0)."

        if beta_0 == optimal:
            return (
                "🔗 **Unidad de Obra Monolítica**: El proyecto funciona como un solo "
                "edificio interconectado (β₀ = 1). Todas las cargas tácticas (APUs) "
                "se transfieren correctamente hacia un único Ápice Estratégico."
            )

        return (
            f"⚠️ **Edificios Desconectados (Fragmentación)**: El proyecto no es una "
            f"estructura única, sino un archipiélago de {beta_0} sub-estructuras aisladas. "
            "No existe un Ápice unificado que centralice la carga financiera."
        )

    def _translate_stability(self, stability: float) -> str:
        """
        Traduce Ψ como 'Solidez de Cimentación'.
        """
        thresholds = self.stability_thresholds

        if stability < thresholds.critical:
            # Lógica Pirámide Invertida
            return (
                f"📉 **COLAPSO POR BASE ESTRECHA (Pirámide Invertida)**: "
                f"Ψ = {stability:.2f}. La Cimentación Logística (Insumos) es demasiado "
                "angosta para soportar el Peso Táctico (APUs) que tiene encima. "
                "El centro de gravedad está muy alto; riesgo inminente de vuelco financiero."
            )

        if stability >= thresholds.solid:
            # Estructura Resiliente
            return (
                f"🛡️ **ESTRUCTURA ANTISÍSMICA (Resiliente)**: "
                f"Ψ = {stability:.2f}. La Cimentación de Recursos es amplia y redundante. "
                "El proyecto tiene un bajo centro de gravedad, capaz de absorber "
                "vibraciones del mercado (volatilidad) sin sufrir daños estructurales."
            )

        # Rango intermedio
        return (
            f"⚖️ **Estructura Isostática (Estable)**: "
            f"Ψ = {stability:.2f}. El equilibrio entre la carga de actividades y "
            "el soporte de insumos es adecuado, aunque no posee redundancia sísmica."
        )

    def translate_financial(self, metrics: Dict[str, Any]) -> str:
        """Traduce métricas financieras (sin cambios mayores, solo integración)."""
        validated = self._validate_financial_metrics(metrics)
        narrative_parts: List[str] = []
        narrative_parts.append(self._translate_wacc(validated["wacc"]))
        narrative_parts.append(self._translate_risk_exposure(validated["contingency_recommended"]))
        narrative_parts.append(self._translate_verdict(validated["recommendation"], validated["profitability_index"]))
        return "\n".join(narrative_parts)

    def _validate_financial_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Valida y normaliza métricas financieras (Parse, Don't Validate)."""
        if not isinstance(metrics, dict):
            raise TypeError(f"Se esperaba dict de métricas, recibido: {type(metrics).__name__}")

        return {
            "wacc": self._extract_numeric(metrics, "wacc", default=0.0),
            "contingency_recommended": self._extract_nested_numeric(metrics, ["contingency", "recommended"], default=0.0),
            "recommendation": self._extract_verdict(metrics),
            "profitability_index": self._extract_nested_numeric(metrics, ["performance", "profitability_index"], default=0.0)
        }

    def _extract_numeric(self, data: Dict[str, Any], key: str, default: float = 0.0) -> float:
        value = data.get(key)
        if value is None: return default
        if not isinstance(value, (int, float)): return default
        return float(value)

    def _extract_nested_numeric(self, data: Dict[str, Any], path: List[str], default: float = 0.0) -> float:
        current = data
        for key in path:
            if not isinstance(current, dict): return default
            current = current.get(key)
            if current is None: return default
        if not isinstance(current, (int, float)): return default
        return float(current)

    def _extract_verdict(self, metrics: Dict[str, Any]) -> FinancialVerdict:
        if not isinstance(metrics, dict):
            return FinancialVerdict.REVIEW

        performance = metrics.get("performance", {})
        if not isinstance(performance, dict): return FinancialVerdict.REVIEW
        rec = performance.get("recommendation", "REVISAR")
        try:
            return FinancialVerdict(rec)
        except ValueError:
            return FinancialVerdict.REVIEW

    def _translate_wacc(self, wacc: float) -> str:
        return f"💰 **Costo de Oportunidad**: WACC = {wacc:.2%}."

    def _translate_risk_exposure(self, contingency: float) -> str:
        return f"📊 **Blindaje Financiero**: Contingencia sugerida de ${contingency:,.2f}."

    def _translate_verdict(self, rec: FinancialVerdict, pi: float) -> str:
        if rec == FinancialVerdict.ACCEPT:
            return f"🚀 **Veredicto**: VIABLE (IR={pi:.2f}). Estructura financiable."
        if rec == FinancialVerdict.REJECT:
            return f"🛑 **Veredicto**: RIESGO CRÍTICO (IR={pi:.2f}). No procedente."
        return "🔍 **Veredicto**: REVISIÓN REQUERIDA."

    def _get_market_context(self) -> str:
        """Obtiene inteligencia de mercado externa o simulada."""
        if self._market_provider:
            try:
                return f"🌍 **Suelo de Mercado**: {self._market_provider()}"
            except Exception:
                return "🌍 **Suelo de Mercado**: No disponible."

        tendencias = [
            "Terreno Inflacionario: Acero al alza (+2.5%). Reforzar estimaciones.",
            "Suelo Estable: Precios de cemento sin variación significativa.",
            "Vientos de Cambio: Volatilidad cambiaria favorable para importaciones.",
            "Falla Geológica Laboral: Escasez de mano de obra calificada."
        ]
        return f"🌍 **Suelo de Mercado**: {self._rng.choice(tendencias)}"

    def compose_strategic_narrative(
        self,
        topo_metrics: TopologicalMetrics,
        fin_metrics: Dict[str, Any],
        stability: float = 0.0
    ) -> str:
        """
        Compone el reporte ejecutivo con metáforas de ingeniería estructural.
        """
        sections = []
        is_analysis_valid = True
        errors = []

        # Header
        sections.append(self._generate_report_header())

        # 1. Estructura
        sections.append("### 1. Auditoría de Integridad Estructural")
        try:
            sections.append(self.translate_topology(topo_metrics, stability))
        except Exception as e:
            error_msg = f"Error analizando estructura: {e}"
            sections.append(f"❌ {error_msg}")
            errors.append(error_msg)
            is_analysis_valid = False
        sections.append("")

        # 2. Finanzas
        sections.append("### 2. Análisis de Cargas Financieras")
        try:
            sections.append(self.translate_financial(fin_metrics))
        except Exception as e:
            error_msg = f"Error analizando finanzas: {e}"
            sections.append(f"❌ {error_msg}")
            errors.append(error_msg)
            is_analysis_valid = False
        sections.append("")

        # 3. Mercado
        sections.append("### 3. Geotecnia de Mercado")
        sections.append(self._get_market_context())
        sections.append("")

        # 4. Recomendación
        sections.append("### 💡 Dictamen del Ingeniero Jefe")
        sections.append(self._generate_final_advice(topo_metrics, fin_metrics, stability, is_analysis_valid))

        return "\n".join(sections)

    def _generate_report_header(self) -> str:
        return (
            "## 🏗️ INFORME DE INGENIERÍA ESTRATÉGICA\n"
            f"*Análisis de Coherencia Fractal | "
            f"Estabilidad Crítica: Ψ < {self.stability_thresholds.critical}*"
        )

    def _generate_final_advice(
        self,
        topo_metrics: TopologicalMetrics,
        fin_metrics: Dict[str, Any],
        stability: float,
        is_valid_analysis: bool = True
    ) -> str:
        """Genera el dictamen final basado en la solidez de la pirámide."""

        if not is_valid_analysis:
            return (
                "⚠️ ANÁLISIS ESTRUCTURAL INTERRUMPIDO: Se detectaron inconsistencias matemáticas "
                "o falta de datos críticos que impiden certificar la solidez del proyecto. "
                "Revise los errores en las secciones técnicas."
            )

        # Factores de decisión
        has_holes = topo_metrics.beta_1 > 0
        is_inverted_pyramid = stability < self.stability_thresholds.critical
        financial_verdict = self._extract_verdict(fin_metrics)

        # 1. Caso Pirámide Invertida (Prioridad Alta)
        if is_inverted_pyramid:
            if financial_verdict == FinancialVerdict.ACCEPT:
                return (
                    f"⚠️ **PRECAUCIÓN LOGÍSTICA (Estructura Inestable)**: Aunque los números "
                    f"financieros cuadran, el proyecto es una **Pirámide Invertida** (Ψ={stability:.2f}). "
                    "Se sostiene sobre una base de recursos demasiado estrecha. "
                    "RECOMENDACIÓN: Ampliar la base de proveedores antes de construir, o el riesgo de "
                    "desabastecimiento derrumbará la rentabilidad."
                )
            else:
                return (
                    f"❌ **PROYECTO INVIABLE (Riesgo de Colapso)**: Combinación letal de "
                    "inestabilidad estructural (Pirámide Invertida) e inviabilidad financiera. "
                    "No proceder bajo ninguna circunstancia sin rediseño total."
                )

        # 2. Caso Genus Elevado (Agujeros)
        if has_holes:
            return (
                f"🛑 **DETENER PARA REPARACIONES**: Se detectaron {topo_metrics.beta_1} socavones "
                "lógicos (ciclos). No se puede verter dinero en una estructura con agujeros. "
                "Sanear la topología antes de aprobar presupuesto."
            )

        # 3. Caso Ideal
        if financial_verdict == FinancialVerdict.ACCEPT:
            return (
                "✅ **CERTIFICADO DE SOLIDEZ**: Estructura piramidal estable, sin socavones "
                "lógicos y financieramente viable. Proceder a fase de ejecución."
            )

        # 4. Fallback
        return "🔍 **REVISIÓN TÉCNICA REQUERIDA**: La estructura es sólida pero los números no convencen."
