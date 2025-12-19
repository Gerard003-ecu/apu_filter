# -*- coding: utf-8 -*-
"""
Módulo de Traducción Semántica.

Este módulo actúa como un puente lingüístico entre las métricas técnicas
(topología, finanzas) y el lenguaje de negocio estratégico. Transforma
datos duros en narrativas accionables para gerentes de proyectos.
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
    - Ψ < critical: Pirámide invertida (más APUs compuestos que insumos base)
    - critical ≤ Ψ < solid: Estructura equilibrada
    - Ψ ≥ solid: Base diversificada y resiliente
    """
    critical: float = 1.0
    solid: float = 10.0


@dataclass(frozen=True)
class TopologicalThresholds:
    """
    Umbrales para interpretación de números de Betti.

    Fundamentación:
    - β₀: Componentes conexos (fragmentación si > 1)
    - β₁: Ciclos independientes (dependencias circulares si > 0)
    """
    connected_components_optimal: int = 1
    cycles_optimal: int = 0


@dataclass(frozen=True)
class WACCThresholds:
    """
    Umbrales para evaluación del Costo Promedio Ponderado de Capital.

    Fundamentación financiera:
    - WACC < low: Costo de capital competitivo (acceso favorable a financiamiento)
    - low ≤ WACC ≤ high: Rango típico del sector construcción
    - WACC > high: Costo elevado que erosiona márgenes
    """
    low: float = 0.05
    high: float = 0.15


@dataclass(frozen=True)
class CycleSeverityThresholds:
    """
    Umbrales para gradación de severidad en dependencias circulares (β₁).

    Fundamentación topológica:
    - β₁ ∈ [1, moderate): Ciclos manejables con reestructuración local
    - β₁ ∈ [moderate, critical): Requiere intervención arquitectónica
    - β₁ ≥ critical: Estructura fundamentalmente defectuosa
    """
    moderate: int = 3
    critical: int = 5


class SemanticTranslator:
    """
    Traductor semántico que convierte métricas técnicas en narrativa estratégica.

    Attributes:
        stability_thresholds: Configuración de umbrales de estabilidad.
        topo_thresholds: Configuración de umbrales topológicos.
        market_provider: Función inyectable para obtener contexto de mercado.
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
        """
        Inicializa el traductor con configuración opcional.

        Args:
            stability_thresholds: Umbrales personalizados de estabilidad piramidal.
            topo_thresholds: Umbrales personalizados para números de Betti.
            market_provider: Función que provee contexto de mercado (inyección de dependencias).
            random_seed: Semilla para reproducibilidad en selección de tendencias.
            wacc_thresholds: Umbrales para evaluación del costo de capital.
            cycle_severity: Umbrales para gradación de severidad en ciclos.

        Raises:
            TypeError: Si los tipos de configuración son inválidos.
        """
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
            "SemanticTranslator inicializado | "
            f"Ψ_critical={self.stability_thresholds.critical:.2f}, "
            f"Ψ_solid={self.stability_thresholds.solid:.2f}, "
            f"WACC_range=[{self._wacc_thresholds.low:.2%}, {self._wacc_thresholds.high:.2%}]"
        )

    def _validate_init_arguments(
        self,
        stability_thresholds: Optional[StabilityThresholds],
        topo_thresholds: Optional[TopologicalThresholds],
        wacc_thresholds: Optional[WACCThresholds],
        cycle_severity: Optional[CycleSeverityThresholds]
    ) -> None:
        """
        Valida tipos de argumentos de inicialización.

        Raises:
            TypeError: Si algún argumento tiene tipo incorrecto.
        """
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
        Traduce métricas topológicas (números de Betti, estabilidad) a lenguaje de negocio.

        La traducción se fundamenta en la teoría de homología:
        - β₀ (componentes conexos) → Fragmentación de recursos
        - β₁ (ciclos/agujeros) → Dependencias circulares
        - Ψ (estabilidad piramidal) → Robustez de cadena de suministro

        Args:
            metrics: Métricas de Betti (β₀, β₁).
            stability: Métrica de estabilidad piramidal (Ψ ≥ 0).

        Returns:
            Narrativa sobre la salud estructural del proyecto.

        Raises:
            ValueError: Si las métricas son topológicamente inválidas.
        """
        self._validate_topological_metrics(metrics, stability)

        narrative_parts: List[str] = []

        # 1. β₁ (Primer número de Betti: ciclos) → Bloqueos Logísticos
        narrative_parts.append(self._translate_cycles(metrics.beta_1))

        # 2. β₀ (Número de Betti cero: componentes conexos) → Coherencia del Alcance
        narrative_parts.append(self._translate_connectivity(metrics.beta_0))

        # 3. Ψ (Estabilidad piramidal) → Robustez de la Cadena de Suministro
        narrative_parts.append(self._translate_stability(stability))

        return "\n".join(narrative_parts)

    def _validate_topological_metrics(
        self,
        metrics: TopologicalMetrics,
        stability: float
    ) -> None:
        """
        Valida la coherencia matemática de las métricas topológicas.

        Invariantes topológicos verificados:
        - βₖ ∈ ℤ≥₀ (números de Betti son enteros no-negativos)
        - Ψ ∈ ℝ≥₀ (estabilidad es real no-negativa)
        - metrics es instancia válida de TopologicalMetrics

        Args:
            metrics: Métricas de Betti a validar.
            stability: Índice de estabilidad a validar.

        Raises:
            TypeError: Si los tipos son incorrectos.
            ValueError: Si alguna métrica viola invariantes topológicos.
        """
        if not isinstance(metrics, TopologicalMetrics):
            raise TypeError(
                f"Se esperaba TopologicalMetrics, se recibió {type(metrics).__name__}"
            )

        if not isinstance(stability, (int, float)):
            raise TypeError(
                f"Estabilidad debe ser numérica, recibido: {type(stability).__name__}"
            )

        # Validar que los números de Betti sean enteros
        for name, value in [("β₀", metrics.beta_0), ("β₁", metrics.beta_1)]:
            if not isinstance(value, int):
                raise TypeError(
                    f"{name} debe ser entero (recibido: {type(value).__name__}). "
                    "Los números de Betti son invariantes topológicos en ℤ≥₀."
                )
            if value < 0:
                raise ValueError(
                    f"{name} debe ser no-negativo (recibido: {value}). "
                    f"Por definición: {name} = dim(ker(∂ₖ)) - dim(im(∂ₖ₊₁)) ≥ 0."
                )

        if stability < 0:
            raise ValueError(
                f"Estabilidad Ψ debe ser no-negativa (recibido: {stability:.4f}). "
                "Ψ = |insumos_base| / |APUs_compuestos| está definida en ℝ≥₀."
            )

    def _translate_cycles(self, beta_1: int) -> str:
        """
        Traduce β₁ (primer número de Betti) a narrativa de bloqueos logísticos.

        Fundamentación topológica:
        β₁ = dim(H₁(X)) cuenta los ciclos 1-dimensionales independientes
        del espacio simplicial. En el grafo de dependencias:
        - Cada ciclo representa una dependencia circular A → B → ... → A
        - Ciclos múltiples pueden compartir vértices (complejos de Venn)

        Gradación de severidad basada en experiencia empírica de proyectos:
        - [1, moderate): Impacto localizado, corrección mediante refactorización
        - [moderate, critical): Impacto sistémico, requiere rediseño de estructura
        - [critical, ∞): Estructura fundamentalmente malformada

        Args:
            beta_1: Primer número de Betti (ciclos independientes).

        Returns:
            Narrativa contextualizada sobre dependencias circulares.
        """
        if beta_1 <= self.topo_thresholds.cycles_optimal:
            return (
                "✅ **Flujo Logístico Optimizado**: No se detectan dependencias "
                "circulares (β₁ = 0). El grafo de dependencias es un DAG válido, "
                "garantizando trazabilidad unidireccional de costos."
            )

        severity = self._classify_cycle_severity(beta_1)
        plural_s = "s" if beta_1 > 1 else ""
        plural_es = "es" if beta_1 > 1 else ""

        severity_descriptions = {
            "moderate": (
                f"🔶 **Bloqueos Logísticos Moderados**: Se identificaron "
                f"{beta_1} ciclo{plural_s} de dependencia en la estructura de costos. "
                "Estos ciclos pueden resolverse con refactorización local de APUs. "
                "Riesgo: sobrecostos por recálculo iterativo de precios unitarios."
            ),
            "severe": (
                f"🔴 **Bloqueos Logísticos Severos**: Se detectaron {beta_1} "
                f"dependencia{plural_s} circular{plural_es} interdependientes. "
                "La complejidad topológica requiere rediseño estructural del presupuesto. "
                "Riesgo crítico: imposibilidad de establecer línea base de costos."
            ),
            "critical": (
                f"🚨 **Estructura Topológicamente Inviable**: {beta_1} ciclos "
                "independientes detectados. El espacio de costos tiene genus alto "
                f"(g ≈ {beta_1}), indicando una estructura irreconciliable. "
                "Acción: reconstruir el presupuesto desde taxonomía base."
            )
        }

        return severity_descriptions[severity]

    def _classify_cycle_severity(self, beta_1: int) -> str:
        """
        Clasifica la severidad de los ciclos basándose en umbrales configurados.

        Args:
            beta_1: Número de ciclos independientes.

        Returns:
            Nivel de severidad: "moderate", "severe", o "critical".
        """
        if beta_1 >= self._cycle_severity.critical:
            return "critical"
        if beta_1 >= self._cycle_severity.moderate:
            return "severe"
        return "moderate"

    def _translate_connectivity(self, beta_0: int) -> str:
        """
        Traduce β₀ (número de Betti cero) a narrativa de coherencia del alcance.

        Fundamentación topológica:
        β₀ = dim(H₀(X)) cuenta las componentes conexas del espacio.
        - β₀ = 1: Espacio conexo (proyecto cohesivo)
        - β₀ > 1: Espacio desconectado (fragmentación)
        - β₀ = 0: Espacio vacío (∅) - caso degenerado

        En teoría de categorías, β₀ corresponde al número de objetos
        iniciales en la categoría de componentes conexas.

        Args:
            beta_0: Número de componentes conexas.

        Returns:
            Narrativa sobre la coherencia estructural del proyecto.
        """
        optimal = self.topo_thresholds.connected_components_optimal

        if beta_0 == 0:
            logger.warning(
                "β₀ = 0 detectado: espacio topológico vacío (∅). "
                "Verificar entrada de datos."
            )
            return (
                "⚠️ **Espacio Topológico Vacío**: β₀ = 0 indica ausencia total "
                "de componentes. Matemáticamente, H₀(∅) = 0. "
                "Verificar que el presupuesto contenga al menos un elemento."
            )

        if beta_0 == optimal:
            return (
                "🔗 **Cohesión Estructural Óptima**: El proyecto forma un espacio "
                f"conexo (β₀ = {optimal}). Todos los elementos del presupuesto "
                "tienen trazabilidad hacia un objetivo común, garantizando "
                "consistencia en la propagación de costos."
            )

        fragmentation_ratio = beta_0 / optimal
        severity, action = self._classify_fragmentation(fragmentation_ratio)

        return (
            f"⚠️ **Fragmentación de Recursos ({severity})**: "
            f"El presupuesto presenta {beta_0} componentes conexas disjuntas "
            f"(fragmentación {fragmentation_ratio:.1f}x respecto al óptimo). "
            f"Cada 'isla' representa un subproyecto sin vínculos de costo compartido. "
            f"Acción sugerida: {action}"
        )

    def _classify_fragmentation(self, ratio: float) -> tuple:
        """
        Clasifica el nivel de fragmentación y sugiere acción correctiva.

        Args:
            ratio: Proporción de fragmentación respecto al óptimo.

        Returns:
            Tupla (severidad, acción_sugerida).
        """
        if ratio <= 2:
            return ("Leve", "verificar si la separación es intencional (fases de proyecto)")
        if ratio <= 4:
            return ("Moderada", "consolidar APUs huérfanos o crear enlaces de trazabilidad")
        return ("Severa", "auditar estructura completa y reunificar bajo taxonomía común")

    def _translate_stability(self, stability: float) -> str:
        """
        Traduce el índice de estabilidad piramidal (Ψ) a narrativa de robustez.

        Fundamentación matemática:
        Ψ = |I| / |A| donde:
        - |I| = cardinalidad del conjunto de insumos elementales (hojas)
        - |A| = cardinalidad del conjunto de APUs compuestos (nodos internos)

        Interpretación geométrica:
        - Ψ < 1: Pirámide invertida (base estrecha, cúspide ancha)
        - Ψ = 1: Pirámide degenerada (cuadrado)
        - Ψ > 1: Pirámide estable (base ancha, cúspide estrecha)
        - Ψ → ∞: Estructura plana (solo insumos, sin composición)

        Args:
            stability: Índice de estabilidad piramidal (Ψ ≥ 0).

        Returns:
            Narrativa sobre la robustez de la cadena de suministro.
        """
        thresholds = self.stability_thresholds

        if stability < thresholds.critical:
            deficit_ratio = thresholds.critical / max(stability, 0.001)
            return (
                f"📉 **Cadena de Suministro Crítica (Pirámide Invertida)**: "
                f"Ψ = {stability:.2f} < {thresholds.critical:.1f}. "
                f"Se requieren {deficit_ratio:.1f}x más insumos base para estabilizar. "
                "Riesgo: alta concentración en pocos proveedores. Un fallo de suministro "
                "cascadea hacia múltiples APUs dependientes."
            )

        if stability >= thresholds.solid:
            resilience_factor = stability / thresholds.solid
            return (
                f"🛡️ **Cadena de Suministro Resiliente**: "
                f"Ψ = {stability:.2f} (factor de resiliencia: {resilience_factor:.1f}x). "
                "La base de insumos está altamente diversificada. "
                "El proyecto puede absorber interrupciones parciales de suministro "
                "sin impacto crítico en la ejecución."
            )

        # Rango equilibrado: [critical, solid)
        position_in_range = (stability - thresholds.critical) / (thresholds.solid - thresholds.critical)
        return (
            f"⚖️ **Cadena de Suministro Equilibrada**: "
            f"Ψ = {stability:.2f} (percentil {position_in_range:.0%} del rango saludable). "
            "La estructura piramidal es estable. Se recomienda mantener vigilancia "
            "sobre concentración de proveedores clave."
        )

    def translate_financial(self, metrics: Dict[str, Any]) -> str:
        """
        Traduce métricas financieras (VaR, WACC, ROI) a lenguaje de inversión estratégica.

        Args:
            metrics: Diccionario de métricas del FinancialEngine.
                Estructura esperada:
                {
                    "wacc": float,
                    "var": float,
                    "contingency": {"recommended": float, ...},
                    "performance": {"recommendation": str, "profitability_index": float, ...}
                }

        Returns:
            Narrativa sobre la viabilidad económica y riesgos financieros.

        Raises:
            ValueError: Si la estructura de métricas es inválida.
        """
        validated = self._validate_financial_metrics(metrics)

        narrative_parts: List[str] = []

        # 1. WACC → Costo de Oportunidad del Capital
        narrative_parts.append(self._translate_wacc(validated["wacc"]))

        # 2. VaR y Contingencia → Exposición al Riesgo
        narrative_parts.append(
            self._translate_risk_exposure(validated["contingency_recommended"])
        )

        # 3. Recomendación → Veredicto de Viabilidad
        narrative_parts.append(
            self._translate_verdict(
                validated["recommendation"],
                validated["profitability_index"]
            )
        )

        return "\n".join(narrative_parts)

    def _validate_financial_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida y normaliza métricas financieras con extracción defensiva.

        Implementa el patrón "Parse, Don't Validate" para garantizar
        que el diccionario de salida siempre tenga estructura consistente.

        Args:
            metrics: Diccionario crudo de métricas financieras.

        Returns:
            Diccionario normalizado con claves garantizadas:
            - wacc: float
            - contingency_recommended: float
            - recommendation: FinancialVerdict
            - profitability_index: float

        Raises:
            TypeError: Si metrics no es un diccionario.
            ValueError: Si métricas críticas tienen tipos inválidos.
        """
        if not isinstance(metrics, dict):
            raise TypeError(
                f"Se esperaba dict de métricas, recibido: {type(metrics).__name__}"
            )

        return {
            "wacc": self._extract_numeric(metrics, "wacc", default=0.0),
            "contingency_recommended": self._extract_nested_numeric(
                metrics, ["contingency", "recommended"], default=0.0
            ),
            "recommendation": self._extract_verdict(metrics),
            "profitability_index": self._extract_nested_numeric(
                metrics, ["performance", "profitability_index"], default=0.0
            )
        }

    def _extract_numeric(
        self,
        data: Dict[str, Any],
        key: str,
        default: float = 0.0
    ) -> float:
        """
        Extrae un valor numérico de un diccionario con validación de tipo.

        Args:
            data: Diccionario fuente.
            key: Clave a extraer.
            default: Valor por defecto si la clave no existe.

        Returns:
            Valor numérico convertido a float.

        Raises:
            ValueError: Si el valor existe pero no es numérico.
        """
        value = data.get(key)

        if value is None:
            logger.debug(f"Clave '{key}' no encontrada, usando default={default}")
            return default

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"'{key}' debe ser numérico, recibido: {type(value).__name__} ({value!r})"
            )

        return float(value)

    def _extract_nested_numeric(
        self,
        data: Dict[str, Any],
        path: List[str],
        default: float = 0.0
    ) -> float:
        """
        Extrae un valor numérico de una ruta anidada en el diccionario.

        Args:
            data: Diccionario fuente.
            path: Lista de claves que forman la ruta (ej: ["contingency", "recommended"]).
            default: Valor por defecto si la ruta no existe.

        Returns:
            Valor numérico encontrado o default.
        """
        current = data

        for i, key in enumerate(path):
            if not isinstance(current, dict):
                logger.debug(f"Ruta {path[:i]} no es dict, usando default={default}")
                return default
            current = current.get(key)
            if current is None:
                logger.debug(f"Clave '{key}' no encontrada en ruta {path}, usando default={default}")
                return default

        if not isinstance(current, (int, float)):
            logger.warning(
                f"Valor en ruta {path} no es numérico ({type(current).__name__}), "
                f"usando default={default}"
            )
            return default

        return float(current)

    def _extract_verdict(self, metrics: Dict[str, Any]) -> FinancialVerdict:
        """
        Extrae y valida el veredicto financiero de las métricas.

        Args:
            metrics: Diccionario de métricas financieras.

        Returns:
            FinancialVerdict validado (REVIEW si no se puede determinar).
        """
        performance = metrics.get("performance", {})
        if not isinstance(performance, dict):
            return FinancialVerdict.REVIEW

        recommendation_raw = performance.get("recommendation", "REVISAR")

        try:
            return FinancialVerdict(recommendation_raw)
        except ValueError:
            logger.warning(
                f"Veredicto '{recommendation_raw}' no reconocido en enum FinancialVerdict, "
                "defaulting a REVIEW"
            )
            return FinancialVerdict.REVIEW

    def _translate_wacc(self, wacc: float) -> str:
        """
        Traduce WACC a narrativa de costo de oportunidad del capital.

        El WACC (Weighted Average Cost of Capital) representa la tasa mínima
        de retorno que el proyecto debe generar para satisfacer a todos
        los proveedores de capital (equity + deuda).

        Args:
            wacc: Costo promedio ponderado de capital (como decimal, ej: 0.12 = 12%).

        Returns:
            Narrativa contextualizada sobre el costo de capital.
        """
        assessment = self._assess_wacc_level(wacc)

        base_narrative = (
            f"💰 **Costo de Oportunidad del Capital**: WACC = {wacc:.2%}{assessment}. "
        )

        if wacc > self._wacc_thresholds.high:
            return base_narrative + (
                "El alto costo de capital erosiona márgenes. Considerar: "
                "(1) renegociar tasas de deuda, (2) optimizar estructura de capital, "
                "(3) buscar inversionistas estratégicos con menor costo de equity."
            )

        if wacc < self._wacc_thresholds.low:
            return base_narrative + (
                "El acceso favorable a financiamiento permite mayor flexibilidad "
                "en la selección de proyectos y absorción de contingencias."
            )

        return base_narrative + (
            "Este es el rendimiento mínimo que el proyecto debe superar "
            "para generar valor económico agregado (EVA > 0)."
        )

    def _assess_wacc_level(self, wacc: float) -> str:
        """
        Evalúa el nivel del WACC respecto a los umbrales configurados.

        Args:
            wacc: Costo promedio ponderado de capital.

        Returns:
            Calificación textual del nivel de WACC.
        """
        if wacc > self._wacc_thresholds.high:
            excess = wacc - self._wacc_thresholds.high
            return f" (elevado +{excess:.1%} sobre umbral)"
        if wacc < self._wacc_thresholds.low:
            advantage = self._wacc_thresholds.low - wacc
            return f" (competitivo -{advantage:.1%} bajo umbral)"
        return " (dentro del rango típico del sector)"

    def _translate_risk_exposure(self, contingency_recommended: float) -> str:
        """Traduce contingencia/VaR a narrativa de exposición al riesgo."""
        if contingency_recommended <= 0:
            return (
                "📊 **Exposición al Riesgo Financiero**: No se ha calculado "
                "contingencia. Revisar parámetros de VaR y volatilidad."
            )

        return (
            f"📊 **Exposición al Riesgo Financiero**: Se estima una contingencia "
            f"sugerida de ${contingency_recommended:,.2f} (basada en VaR y "
            "volatilidad de mercado) para blindar el margen del proyecto."
        )

    def _translate_verdict(
        self,
        recommendation: FinancialVerdict,
        profitability_index: float
    ) -> str:
        """Traduce la recomendación financiera a veredicto ejecutivo."""
        verdicts = {
            FinancialVerdict.ACCEPT: (
                f"🚀 **Veredicto de Viabilidad**: El proyecto es FINANCIERAMENTE "
                f"VIABLE (Índice de Rentabilidad: {profitability_index:.2f}). "
                "Se recomienda proceder, manteniendo vigilancia sobre la "
                "contingencia sugerida."
            ),
            FinancialVerdict.REJECT: (
                f"🛑 **Veredicto de Viabilidad**: El proyecto presenta RIESGOS "
                f"CRÍTICOS (Índice de Rentabilidad: {profitability_index:.2f}). "
                "Se recomienda reestructurar los costos o buscar eficiencias "
                "operativas antes de aprobar."
            ),
            FinancialVerdict.REVIEW: (
                "🔍 **Veredicto de Viabilidad**: Se requiere una revisión manual "
                "profunda debido a inconsistencias en los flujos o inversión inicial."
            )
        }
        return verdicts.get(recommendation, verdicts[FinancialVerdict.REVIEW])

    def _get_market_context(self) -> str:
        """
        Obtiene inteligencia de mercado externa.

        Si se inyectó un proveedor personalizado, lo utiliza.
        De lo contrario, simula con tendencias predefinidas.

        Returns:
            Narrativa de contexto de mercado.
        """
        if self._market_provider is not None:
            try:
                context = self._market_provider()
                return f"🌍 **Contexto de Mercado**: {context}"
            except Exception as e:
                logger.error(f"Error obteniendo contexto de mercado: {e}")
                return "🌍 **Contexto de Mercado**: No disponible temporalmente."

        tendencias = [
            "📈 Inflación en materiales de acero (+2.5% m/m). "
            "Se sugiere stockeo anticipado.",

            "📉 Tipo de cambio favorable para importaciones. "
            "Oportunidad de negociar con proveedores extranjeros.",

            "⚠️ Escasez de mano de obra calificada en la región. "
            "Considerar ajustar rendimientos en APUs.",

            "⚖️ Estabilidad en precios del cemento y agregados. "
            "Momento oportuno para contratos a largo plazo.",

            "🌪️ Alta volatilidad energética proyectada para el próximo trimestre. "
            "Evaluar cláusulas de ajuste en contratos."
        ]

        selected_trend = self._rng.choice(tendencias)
        return f"🌍 **Contexto de Mercado**: {selected_trend}"

    def compose_strategic_narrative(
        self,
        topo_metrics: TopologicalMetrics,
        fin_metrics: Dict[str, Any],
        stability: float = 0.0
    ) -> str:
        """
        Compone el reporte ejecutivo consolidando todas las dimensiones analíticas.

        Orquesta la traducción de métricas topológicas, financieras y de mercado
        en un documento unificado con estructura jerárquica para toma de decisiones.

        Estructura del reporte:
        1. Salud Estructural (Topología + Estabilidad)
        2. Viabilidad Económica (WACC, VaR, ROI)
        3. Inteligencia de Mercado (Contexto externo)
        4. Recomendación Estratégica (Síntesis ejecutiva)

        Args:
            topo_metrics: Métricas de números de Betti (β₀, β₁).
            fin_metrics: Diccionario de métricas financieras.
            stability: Índice de estabilidad piramidal (Ψ ≥ 0).

        Returns:
            Documento Markdown estructurado listo para presentación ejecutiva.

        Note:
            El método captura errores por sección para maximizar la información
            disponible incluso con datos parcialmente inválidos.
        """
        sections = []
        errors: List[str] = []

        # Header con metadata
        sections.append(self._generate_report_header())

        # Sección 1: Análisis Estructural
        topo_narrative, topo_error = self._safe_translate_topology(topo_metrics, stability)
        sections.append("### 1. Salud Estructural y Operativa")
        sections.append(topo_narrative)
        sections.append("")
        if topo_error:
            errors.append(topo_error)

        # Sección 2: Análisis Financiero
        fin_narrative, fin_error = self._safe_translate_financial(fin_metrics)
        sections.append("### 2. Análisis de Viabilidad Económica")
        sections.append(fin_narrative)
        sections.append("")
        if fin_error:
            errors.append(fin_error)

        # Sección 3: Contexto de Mercado
        sections.append("### 3. Inteligencia de Mercado")
        sections.append(self._get_market_context())
        sections.append("")

        # Sección 4: Recomendación Final
        sections.append("### 💡 Recomendación Estratégica")
        sections.append(
            self._generate_final_advice_with_fallback(
                topo_metrics, fin_metrics, stability, errors
            )
        )

        return "\n".join(sections)

    def _generate_report_header(self) -> str:
        """Genera el encabezado del reporte con metadatos."""
        return (
            "## 🏗️ INFORME DE INTELIGENCIA ESTRATÉGICA\n"
            f"*Generado por SemanticTranslator | "
            f"Umbrales: Ψ_crit={self.stability_thresholds.critical}, "
            f"Ψ_solid={self.stability_thresholds.solid}*\n"
        )

    def _safe_translate_topology(
        self,
        metrics: TopologicalMetrics,
        stability: float
    ) -> tuple:
        """
        Ejecuta traducción topológica con manejo de errores.

        Returns:
            Tupla (narrativa, error_opcional).
        """
        try:
            return self.translate_topology(metrics, stability), None
        except (ValueError, TypeError) as e:
            logger.error(f"Error en traducción topológica: {e}")
            return (
                "❌ No se pudo generar el análisis estructural.",
                f"Análisis estructural: {e}"
            )

    def _safe_translate_financial(self, metrics: Dict[str, Any]) -> tuple:
        """
        Ejecuta traducción financiera con manejo de errores.

        Returns:
            Tupla (narrativa, error_opcional).
        """
        try:
            return self.translate_financial(metrics), None
        except (ValueError, TypeError) as e:
            logger.error(f"Error en traducción financiera: {e}")
            return (
                "❌ No se pudo generar el análisis financiero.",
                f"Análisis financiero: {e}"
            )

    def _generate_final_advice_with_fallback(
        self,
        topo_metrics: TopologicalMetrics,
        fin_metrics: Dict[str, Any],
        stability: float,
        errors: List[str]
    ) -> str:
        """
        Genera consejo final con fallback si hay errores previos.

        Args:
            topo_metrics: Métricas topológicas.
            fin_metrics: Métricas financieras.
            stability: Índice de estabilidad.
            errors: Lista de errores acumulados.

        Returns:
            Recomendación estratégica o mensaje de análisis incompleto.
        """
        if errors:
            error_summary = "; ".join(errors)
            return (
                f"⚠️ **ANÁLISIS INCOMPLETO**: No es posible emitir una recomendación "
                f"confiable debido a errores en el procesamiento.\n\n"
                f"**Errores detectados**: {error_summary}\n\n"
                "Acción requerida: corregir los datos de entrada y regenerar el informe."
            )

        return self._generate_final_advice(topo_metrics, fin_metrics, stability)

    def _generate_final_advice(
        self,
        topo_metrics: TopologicalMetrics,
        fin_metrics: Dict[str, Any],
        stability: float = 0.0
    ) -> str:
        """
        Genera recomendación estratégica basada en matriz de decisión tridimensional.

        Dimensiones de la matriz:
        1. Topológica (β₁): Presencia de ciclos en grafo de dependencias
        2. Financiera: Veredicto del análisis económico
        3. Estructural (Ψ): Estabilidad piramidal de la cadena de suministro

        Lógica de degradación:
        Si Ψ < Ψ_critical (Pirámide Invertida), cualquier recomendación positiva
        se degrada a "PRECAUCIÓN LOGÍSTICA" para prevenir fragilidad oculta.

        Args:
            topo_metrics: Métricas topológicas validadas.
            fin_metrics: Diccionario de métricas financieras.
            stability: Índice de estabilidad piramidal (Ψ).

        Returns:
            Recomendación estratégica accionable con justificación.
        """
        analysis = self._analyze_decision_factors(topo_metrics, fin_metrics, stability)

        # Caso especial: Degradación por inestabilidad estructural
        if analysis["is_structurally_unstable"] and analysis["is_financially_viable"]:
            return self._generate_stability_warning(stability)

        return self._lookup_decision_matrix(analysis)

    def _analyze_decision_factors(
        self,
        topo_metrics: TopologicalMetrics,
        fin_metrics: Dict[str, Any],
        stability: float
    ) -> Dict[str, Any]:
        """
        Analiza los factores de decisión y los normaliza para la matriz.

        Returns:
            Diccionario con factores de decisión normalizados.
        """
        beta_1 = topo_metrics.beta_1
        has_cycles = beta_1 > self.topo_thresholds.cycles_optimal
        is_unstable = stability < self.stability_thresholds.critical

        recommendation = self._extract_verdict(fin_metrics)
        is_viable = recommendation == FinancialVerdict.ACCEPT

        return {
            "has_cycles": has_cycles,
            "cycle_count": beta_1,
            "is_structurally_unstable": is_unstable,
            "stability_value": stability,
            "recommendation": recommendation,
            "is_financially_viable": is_viable
        }

    def _generate_stability_warning(self, stability: float) -> str:
        """
        Genera advertencia específica para proyectos con Pirámide Invertida.

        Args:
            stability: Índice de estabilidad actual.

        Returns:
            Mensaje de precaución logística.
        """
        deficit = self.stability_thresholds.critical - stability
        return (
            f"⚠️ **PRECAUCIÓN LOGÍSTICA**: El proyecto es financieramente rentable, "
            f"pero su estructura de 'Pirámide Invertida' (Ψ = {stability:.2f}, "
            f"déficit = {deficit:.2f}) lo hace extremadamente frágil. "
            "Un único fallo de proveedor puede colapsar múltiples APUs. "
            "**Acción requerida**: diversificar base de insumos hasta Ψ ≥ "
            f"{self.stability_thresholds.critical:.1f} antes de proceder."
        )

    def _lookup_decision_matrix(self, analysis: Dict[str, Any]) -> str:
        """
        Consulta la matriz de decisión y retorna la recomendación apropiada.

        La matriz implementa una tabla de verdad bidimensional:
        (has_cycles × recommendation) → advice

        Args:
            analysis: Factores de decisión analizados.

        Returns:
            Recomendación estratégica de la matriz.
        """
        has_cycles = analysis["has_cycles"]
        cycle_count = analysis["cycle_count"]
        recommendation = analysis["recommendation"]

        decision_matrix = {
            (True, FinancialVerdict.REJECT): (
                f"❌ **PROYECTO INVIABLE**: Confluencia de riesgos críticos. "
                f"{cycle_count} ciclo(s) topológico(s) detectado(s) + rechazo financiero. "
                "Acciones: (1) Congelar contrataciones, (2) Auditar estructura de costos, "
                "(3) Reevaluar alcance del proyecto desde cero."
            ),
            (True, FinancialVerdict.ACCEPT): (
                f"⚠️ **PROCEDER CON CORRECCIONES**: Viabilidad financiera confirmada, "
                f"pero {cycle_count} dependencia(s) circular(es) deben resolverse. "
                "Riesgo legal: los ciclos pueden generar disputas contractuales sobre "
                "responsabilidades de costo. Corregir antes de fase de ejecución."
            ),
            (True, FinancialVerdict.REVIEW): (
                f"⚠️ **AUDITORÍA PRIORITARIA**: {cycle_count} ciclo(s) estructural(es) "
                "detectados con evaluación financiera inconclusa. "
                "Secuencia recomendada: (1) Eliminar ciclos, (2) Recalcular métricas, "
                "(3) Reevaluar viabilidad económica."
            ),
            (False, FinancialVerdict.REJECT): (
                "📉 **OPTIMIZACIÓN REQUERIDA**: Estructura técnica sólida (β₁ = 0, DAG válido), "
                "pero indicadores financieros negativos. "
                "Palancas de mejora: reducir alcance, renegociar contratos, "
                "o buscar financiamiento con menor WACC."
            ),
            (False, FinancialVerdict.ACCEPT): (
                "✅ **LUZ VERDE TOTAL**: El proyecto demuestra excelencia en ambas dimensiones. "
                "Coherencia topológica (β₁ = 0) + solidez financiera verificada. "
                "Proceder a fase de planificación detallada con confianza fundamentada."
            ),
            (False, FinancialVerdict.REVIEW): (
                "🔍 **CLARIFICACIÓN PENDIENTE**: Estructura técnica impecable, "
                "pero insuficiente certeza financiera. "
                "Verificar: (1) Inversión inicial correctamente capturada, "
                "(2) Proyección de flujos completa, (3) Tasa de descuento apropiada."
            )
        }

        return decision_matrix.get(
            (has_cycles, recommendation),
            decision_matrix[(False, FinancialVerdict.REVIEW)]
        )
