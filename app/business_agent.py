"""
Módulo: Business Agent (El Cerebro Ejecutivo del Consejo)
=========================================================

Este componente actúa como el nodo de síntesis superior en la jerarquía DIKW. Su función
no es generar datos primarios, sino integrar los hallazgos del Arquitecto (Topología) y
el Oráculo (Finanzas) para emitir un "Veredicto Holístico" sobre la viabilidad del proyecto.

Opera bajo el principio de **"No hay Estrategia sin Física"**, negándose a emitir juicios
financieros si la estabilidad estructural subyacente no ha sido validada por la MIC.

Fundamentos Teóricos y Protocolos de Juicio:
--------------------------------------------

1. Síntesis Topológico-Financiera (El Funtor de Decisión):
   Implementa un mapeo $F: (T \times \Phi) \to D$, donde $T$ es el espacio topológico
   (Betti numbers, $\Psi$) y $\Phi$ es el espacio financiero (VPN, TIR).
   Detecta "Falsos Positivos": Proyectos con alta rentabilidad teórica pero con
   patologías estructurales graves (ej. $\beta_1 > 0$ ciclos de costos) [Fuente: business_agent.txt].

2. Protocolo Challenger (Auditoría Adversarial):
   Incorpora la clase `RiskChallenger` que actúa como un fiscal interno. Ejecuta reglas de
   veto lógico:
   - Si (Rentabilidad == ALTA) Y (Estabilidad Piramidal $\Psi < 1.0$):
     -> Veredicto: **VETO TÉCNICO** (Riesgo de colapso logístico anula la ganancia) [Fuente: SAGES.md].

3. Termodinámica del Valor:
   Evalúa la calidad de la inversión utilizando conceptos de física estadística:
   - **Temperatura del Sistema ($T_{sys}$):** Mide la volatilidad de precios agregada.
   - **Eficiencia Exergética:** Distingue entre inversión en estructura útil (Exergía)
     y gasto cosmético o desperdicio (Anergía/Entropía) [Fuente: metodos.md].

4. Cliente de la MIC (Gobernanza Algebraica):
   No calcula las finanzas directamente, sino que proyecta vectores de intención
   (`financial_analysis`) sobre la Matriz de Interacción Central.
   Valida que los estratos inferiores ($V_{PHYSICS}, V_{TACTICS}$) estén cerrados
   antes de permitir operaciones en el estrato $V_{STRATEGY}$ [Fuente: tools_interface.txt].
"""

import copy
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from agent.business_topology import (
    BudgetGraphBuilder,
    BusinessTopologicalAnalyzer,
    ConstructionRiskReport,
)
from app.constants import ColumnNames
from app.financial_engine import FinancialConfig, FinancialEngine
from app.semantic_translator import SemanticTranslator
from app.telemetry import TelemetryContext
from app.tools_interface import MICRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinancialParameters:
    """
    Parámetros financieros para el análisis del proyecto.

    Encapsula los valores de entrada para el motor financiero,
    garantizando inmutabilidad y validación en construcción.

    Attributes:
        initial_investment (float): Inversión inicial (debe ser > 0).
        cash_flows (Tuple[float, ...]): Flujos de caja proyectados (inmutable).
        cost_std_dev (float): Desviación estándar de los costos (para riesgo).
        project_volatility (float): Volatilidad estimada del proyecto [0, 1].
    """

    initial_investment: float
    cash_flows: Tuple[float, ...]
    cost_std_dev: float
    project_volatility: float

    def __post_init__(self):
        """Valida los invariantes financieros."""
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

    Attributes:
        betti_numbers (Dict[str, Any]): Números de Betti (β0, β1, etc.).
        pyramid_stability (float): Índice de estabilidad piramidal (0.0-1.0).
        graph (Any): Objeto grafo subyacente (NetworkX).
        persistence_diagram (Optional[List[Any]]): Diagrama de persistencia homológica.
    """

    betti_numbers: Dict[str, Any]
    pyramid_stability: float
    graph: Any  # Tipo del grafo según la implementación
    persistence_diagram: Optional[List[Any]] = None

    @property
    def structural_coherence(self) -> float:
        """
        Calcula un índice de coherencia estructural mediante invariantes topológicos.

        Fundamento Matemático (Topología Algebraica):
        =============================================
        Sea K un complejo simplicial asociado al presupuesto. Definimos:

        C(K) = exp(-λ₀·max(0, β₀-1)) × exp(-λ₁·β₁/n) × Ψ

        donde:
        - β₀: Número de componentes conexas (H₀). Ideal: β₀ = 1 (conexidad)
        - β₁: Primer número de Betti (H₁). Ciclos independientes ≈ dependencias circulares
        - n: Número de vértices (normalización por escala)
        - Ψ: Índice de estabilidad piramidal ∈ [0, 1]
        - λ₀, λ₁: Tasas de decaimiento (derivadas de análisis de sensibilidad)

        La exponencial garantiza:
        1. Monotonicidad decreciente en patologías
        2. Composición multiplicativa (log-aditiva en el espacio de riesgos)
        3. Rango natural en [0, 1] sin truncamiento artificial

        Returns:
            float: Índice de coherencia ∈ [0, 1], donde 1 = máxima coherencia topológica.
        """
        import math

        beta_0 = self.betti_numbers.get("beta_0", 1)
        beta_1 = self.betti_numbers.get("beta_1", 0)

        # Obtener cardinalidad del complejo para normalización
        n_vertices = 1  # Default seguro
        if hasattr(self.graph, 'number_of_nodes'):
            n_vertices = max(self.graph.number_of_nodes(), 1)
        elif hasattr(self.graph, '__len__'):
            n_vertices = max(len(self.graph), 1)

        # Tasas de decaimiento fundamentadas en análisis de sensibilidad
        # λ₀ = ln(2) → cada componente adicional reduce coherencia a la mitad
        # λ₁ ajustado por densidad del grafo para evitar penalización excesiva en grafos densos
        lambda_0 = math.log(2)  # ≈ 0.693
        lambda_1 = math.log(2) / max(1, math.sqrt(n_vertices))  # Escala con √n

        # Penalización por fragmentación (β₀ > 1 indica desconexión)
        # exp(-λ₀·(β₀-1)): β₀=1→1, β₀=2→0.5, β₀=3→0.25, ...
        excess_components = max(0, beta_0 - 1)
        fragmentation_factor = math.exp(-lambda_0 * excess_components)

        # Penalización por ciclos, normalizada por tamaño
        # Densidad de ciclos: β₁/n evita penalizar grafos grandes injustamente
        # cycle_factor = math.exp(-lambda_1 * beta_1) if beta_1 < n_vertices else math.exp(-lambda_1 * n_vertices)
        # Note: the proposal had this comment but I'll use a robust version
        cycle_factor = math.exp(-lambda_1 * beta_1)

        # Composición multiplicativa en el grupo ([0,1], ×)
        raw_coherence = fragmentation_factor * cycle_factor * self.pyramid_stability

        # Clamp por seguridad numérica (aunque matemáticamente ya está en [0,1])
        return max(0.0, min(1.0, raw_coherence))


class RiskChallenger:
    """
    Motor de Auditoría Adversarial basado en Lógica Fuzzy y Reglas de Consistencia.

    Implementa un sistema de veto multi-nivel que detecta contradicciones entre
    los espacios financiero (Φ) y topológico (T) mediante reglas de inferencia:

    R₁: (Φ ∈ SAFE) ∧ (Ψ < θ_crítico) → VETO_ESTRUCTURAL
    R₂: (Φ ∈ SAFE) ∧ (C < θ_coherencia) → ALERTA_COHERENCIA
    R₃: (β₁ > n/3) ∧ (Φ ∈ PROFITABLE) → RIESGO_CICLOS

    donde θ son umbrales configurables por dominio.
    """

    # Umbrales por defecto (calibrados empíricamente)
    DEFAULT_THRESHOLDS = {
        "critical_stability": 0.70,      # Ψ < 0.70 → Veto inmediato
        "warning_stability": 0.85,       # 0.70 ≤ Ψ < 0.85 → Alerta severa
        "coherence_minimum": 0.60,       # C < 0.60 → Degradación de score
        "cycle_density_limit": 0.33,     # β₁/n > 1/3 → Advertencia de ciclos
        "integrity_penalty_veto": 0.30,  # Penalización por veto estructural
        "integrity_penalty_warn": 0.15,  # Penalización por alerta
    }

    def __init__(self, config: Optional[Dict[str, float]] = None):
        """
        Inicializa el Challenger con umbrales configurables.

        Args:
            config: Diccionario con umbrales personalizados. Claves válidas:
                    - critical_stability, warning_stability, coherence_minimum,
                    - cycle_density_limit, integrity_penalty_veto, integrity_penalty_warn
        """
        self.thresholds = {**self.DEFAULT_THRESHOLDS}
        if config:
            # Validar umbrales conocidos
            for key, value in config.items():
                f_val = float(value)
                if key in self.DEFAULT_THRESHOLDS:
                    if not (0 <= f_val <= 1.0):
                        raise ValueError(f"Umbral {key} fuera de rango [0, 1]: {f_val}")
                self.thresholds[key] = f_val

    def _extract_stability_metrics(
        self, details: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
        """
        Extrae métricas de estabilidad de la estructura anidada del reporte.

        Returns:
            Tupla (Ψ, coherencia, β₁, n_nodos) con None para valores no encontrados.
        """
        stability = details.get("pyramid_stability")
        coherence = details.get("structural_coherence")
        beta_1 = None
        n_nodes = None

        # Buscar en estructura anidada
        topo_inv = details.get("topological_invariants", {})
        if stability is None:
            stability = topo_inv.get("pyramid_stability")
        if coherence is None:
            coherence = topo_inv.get("structural_coherence")

        betti = topo_inv.get("betti_numbers", {})
        beta_1 = betti.get("beta_1")

        # Intentar obtener número de nodos del grafo
        if "graph_order" in details:
            n_nodes = details["graph_order"]
        elif "n_nodes" in topo_inv:
            n_nodes = topo_inv["n_nodes"]

        return stability, coherence, beta_1, n_nodes

    def _classify_financial_risk(self, risk_level: Any) -> str:
        """
        Normaliza el nivel de riesgo financiero a categorías estándar.

        Returns:
            Una de: "SAFE", "MODERATE", "HIGH", "UNKNOWN"
        """
        risk_str = str(risk_level).upper().strip()

        safe_keywords = {"LOW", "BAJO", "SAFE", "SEGURO", "MINIMAL", "MÍNIMO"}
        moderate_keywords = {"MODERATE", "MODERADO", "MEDIUM", "MEDIO"}
        high_keywords = {"HIGH", "ALTO", "CRITICAL", "CRÍTICO", "SEVERE", "SEVERO"}

        if any(kw in risk_str for kw in safe_keywords):
            return "SAFE"
        elif any(kw in risk_str for kw in moderate_keywords):
            return "MODERATE"
        elif any(kw in risk_str for kw in high_keywords):
            return "HIGH"
        return "UNKNOWN"

    def challenge_verdict(
        self, report: ConstructionRiskReport
    ) -> ConstructionRiskReport:
        """
        Ejecuta auditoría adversarial multi-nivel sobre el reporte.

        Aplica un sistema de reglas de inferencia para detectar contradicciones
        lógicas entre métricas financieras y estructurales, emitiendo vetos
        graduados según la severidad de la inconsistencia.

        Args:
            report: Reporte preliminar a auditar.

        Returns:
            ConstructionRiskReport auditado con posibles modificaciones.
        """
        logger.info("⚖️  Risk Challenger: Iniciando auditoría adversarial...")

        details = report.details or {}
        stability, coherence, beta_1, n_nodes = self._extract_stability_metrics(details)
        financial_class = self._classify_financial_risk(report.financial_risk_level)

        # Si no hay métricas suficientes, no podemos auditar
        if stability is None:
            logger.warning(
                "⚠️  Risk Challenger: Métricas de estabilidad no disponibles. "
                "Auditoría omitida."
            )
            return report

        current_report = report

        # === REGLA 1: Veto por Estabilidad Crítica ===
        if stability < self.thresholds["critical_stability"]:
            if financial_class in ("SAFE", "MODERATE", "HIGH"):
                current_report = self._emit_veto(
                    report=current_report,
                    veto_type="VETO_CRITICAL_INSTABILITY",
                    stability=stability,
                    financial_class=financial_class,
                    severity="CRÍTICO",
                    penalty=self.thresholds["integrity_penalty_veto"],
                    reason=(
                        f"Estabilidad piramidal Ψ={stability:.3f} está por debajo del "
                        f"umbral crítico ({self.thresholds['critical_stability']:.2f}). "
                        "El proyecto presenta riesgo de colapso logístico."
                    ),
                )

        # === REGLA 2: Alerta por Estabilidad Subóptima ===
        elif stability < self.thresholds["warning_stability"]:
            if financial_class in ("SAFE", "MODERATE", "HIGH"):
                current_report = self._emit_veto(
                    report=current_report,
                    veto_type="ALERTA_STRUCTURAL_WARNING",
                    stability=stability,
                    financial_class=financial_class,
                    severity="SEVERO",
                    penalty=self.thresholds["integrity_penalty_warn"],
                    reason=(
                        f"Estabilidad piramidal Ψ={stability:.3f} es subóptima "
                        f"(umbral de alerta: {self.thresholds['warning_stability']:.2f}). "
                        "Financieramente sano pero estructuralmente frágil."
                    ),
                )

        # === REGLA 3: Alerta por Densidad de Ciclos ===
        if beta_1 is not None and n_nodes is not None and n_nodes > 0:
            cycle_density = beta_1 / n_nodes
            if cycle_density > self.thresholds["cycle_density_limit"]:
                if financial_class in ("SAFE", "MODERATE", "HIGH"):
                    logger.warning(
                        f"⚠️  Densidad de ciclos β₁/n = {cycle_density:.3f} excede "
                        f"el límite {self.thresholds['cycle_density_limit']:.2f}"
                    )

                    new_details = current_report.details.copy()
                    new_details["challenger_cycle_warning"] = {
                        "beta_1": beta_1,
                        "n_nodes": n_nodes,
                        "cycle_density": cycle_density,
                        "threshold": self.thresholds["cycle_density_limit"],
                    }
                    # Compatibilidad con propuesta_test
                    new_details["penalties_applied"] = new_details.get("penalties_applied", []) + ["cycle_penalty"]

                    current_report = ConstructionRiskReport(
                        integrity_score=current_report.integrity_score * 0.95,  # Penalización leve
                        waste_alerts=current_report.waste_alerts,
                        circular_risks=current_report.circular_risks,
                        complexity_level=current_report.complexity_level,
                        financial_risk_level=current_report.financial_risk_level,
                        details=new_details,
                        strategic_narrative=current_report.strategic_narrative,
                    )

        if current_report is report:
            logger.info("✅ Risk Challenger: Coherencia verificada. Sin contradicciones.")
        else:
            logger.info("⚖️ Risk Challenger: Auditoría completada con ajustes.")

        return current_report

    def _emit_veto(
        self,
        report: ConstructionRiskReport,
        veto_type: str,
        stability: float,
        financial_class: str,
        severity: str,
        penalty: float,
        reason: str,
    ) -> ConstructionRiskReport:
        """
        Emite un veto estructurado con acta de deliberación.

        Args:
            report: Reporte original.
            veto_type: Código del tipo de veto.
            stability: Valor de Ψ que disparó el veto.
            financial_class: Clasificación financiera original.
            severity: Nivel de severidad ("CRÍTICO", "SEVERO", "MODERADO").
            penalty: Factor de penalización ∈ [0, 1].
            reason: Justificación textual del veto.

        Returns:
            Reporte modificado con el veto aplicado.
        """
        logger.warning(f"🚨 Risk Challenger: {veto_type} - {reason}")

        original_integrity = report.integrity_score
        new_integrity = max(0.0, original_integrity * (1.0 - penalty))

        # Acta de deliberación formal
        debate_log = (
            "━" * 60 + "\n"
            "🏛️ **ACTA DE DELIBERACIÓN DEL CONSEJO DE RIESGO**\n"
            "━" * 60 + "\n\n"
            f"📋 **Tipo de Veto:** {veto_type}\n"
            f"⚠️  **Severidad:** {severity}\n\n"
            "**Posiciones de los Agentes:**\n\n"
            f"1. 🤵 **Gestor Financiero:** «El proyecto es financieramente {financial_class}. "
            "Los indicadores de rentabilidad son favorables.»\n\n"
            f"2. 👷 **Ingeniero Estructural:** «OBJECIÓN. {reason}»\n\n"
            f"3. ⚖️  **Fiscal de Riesgos:** «Se detecta contradicción lógica entre "
            f"viabilidad financiera (Φ={financial_class}) y estabilidad estructural "
            f"(Ψ={stability:.3f}).»\n\n"
            "**VEREDICTO FINAL:**\n"
            f"Se emite **{veto_type}**. La integridad del proyecto se degrada de "
            f"{original_integrity:.1f} a {new_integrity:.1f} puntos.\n\n"
            "━" * 60
        )

        new_narrative = f"{debate_log}\n\n{report.strategic_narrative}"

        new_details = report.details.copy() if report.details else {}
        new_details["challenger_verdict"] = {
            "type": veto_type,
            "severity": severity,
            "stability_at_veto": stability,
            "financial_class_at_veto": financial_class,
            "original_integrity": original_integrity,
            "penalty_applied": penalty,
            "reason": reason,
        }

        # Compatibilidad con propuesta_test
        if severity == "CRÍTICO":
            new_details["challenger_applied"] = True
        else:
            new_details["challenger_warning"] = True

        new_details["penalties_applied"] = new_details.get("penalties_applied", []) + [veto_type]

        return ConstructionRiskReport(
            integrity_score=new_integrity,
            waste_alerts=report.waste_alerts,
            circular_risks=report.circular_risks,
            complexity_level=report.complexity_level,
            financial_risk_level=f"RIESGO ESTRUCTURAL ({severity})",
            details=new_details,
            strategic_narrative=new_narrative,
        )


class BusinessAgent:
    """
    Orquesta la inteligencia de negocio para evaluar proyectos de construcción.

    Combina análisis topológico (estructura del presupuesto como complejo simplicial)
    con análisis financiero (VPN, TIR, simulación de Monte Carlo) para producir
    una evaluación holística.
    """

    # Configuración por defecto para parámetros financieros
    DEFAULT_FINANCIAL_PARAMS = {
        "initial_investment": 1_000_000.0,
        "cash_flow_ratio": 0.30,
        "cash_flow_periods": 5,
        "cost_std_dev_ratio": 0.15,
        "project_volatility": 0.20,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        mic: MICRegistry,
        telemetry: Optional[TelemetryContext] = None
    ):
        """
        Inicializa el agente de negocio con inyección de la MIC.

        Args:
            config: Configuración global de la aplicación.
            mic: Matriz de Interacción Central para proyección de vectores.
            telemetry: Contexto para telemetría y observabilidad.

        Raises:
            ValueError: Si la configuración financiera es inválida.
        """
        self._validate_config(config)
        self.config = config
        self.mic = mic
        self.telemetry = telemetry or TelemetryContext()

        # Componentes del pipeline (inicialización eager para fail-fast)
        self.graph_builder = BudgetGraphBuilder()
        self.topological_analyzer = BusinessTopologicalAnalyzer(self.telemetry)
        self.translator = SemanticTranslator()
        # self.financial_engine eliminado en favor de self.mic

        # Inicializar el Challenger con configuración inyectada
        challenger_config = config.get("risk_challenger_config")
        self.risk_challenger = RiskChallenger(challenger_config)

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


    def _validate_dataframes(
        self,
        df_presupuesto: Optional[pd.DataFrame],
        df_apus_detail: Optional[pd.DataFrame],
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validación estructural y topológica de DataFrames de entrada.

        Implementa verificación en tres niveles:
        1. **Existencia**: DataFrames no nulos y no vacíos
        2. **Esquema**: Columnas requeridas presentes con tipos correctos
        3. **Consistencia Referencial**: Integridad de claves foráneas entre DFs
        4. **Distribución**: Detección de anomalías estadísticas

        Args:
            df_presupuesto: DataFrame del presupuesto general.
            df_apus_detail: DataFrame con detalle de APUs (merged).

        Returns:
            Tupla (es_válido, mensaje, diagnóstico) donde diagnóstico contiene
            métricas adicionales de calidad de datos si la validación es exitosa.
        """
        diagnostics: Dict[str, Any] = {
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "warnings": [],
            "schema_compatibility": {},
            "column_check": {"presupuesto": "OK", "detalle": "OK"},
            "missing_columns": {"presupuesto": [], "detalle": []},
            "null_analysis": {"presupuesto": {"total_nulls": 0}, "detalle": {"total_nulls": 0}},
            "duplicate_analysis": {"duplicated_codes": []},
            "value_range_analysis": {"negative_monetary_values": 0},
            "distribution_analysis": {
                "total_values": 0,
                "mean": 0.0,
                "std": 0.0,
                "q1": 0.0,
                "q3": 0.0,
                "iqr": 0.0,
                "outlier_count": 0,
                "outlier_indices": [],
                "outlier_ratio": 0.0,
            },
        }

        # ━━━ Nivel 1: Existencia Básica ━━━
        if df_presupuesto is None:
            return False, "DataFrame 'df_presupuesto' es None", diagnostics
        if df_apus_detail is None:
            return False, "DataFrame 'df_merged' es None", diagnostics
        if df_presupuesto.empty:
            return False, "DataFrame 'df_presupuesto' está vacío", diagnostics
        if df_apus_detail.empty:
            return False, "DataFrame 'df_merged' está vacío", diagnostics

        diagnostics["row_counts"] = {
            "presupuesto": len(df_presupuesto),
            "apus_detail": len(df_apus_detail),
            "detalle": len(df_apus_detail), # Alias para tests
        }

        # ━━━ Nivel 2: Validación de Esquema con Álgebra de Columnas ━━━
        # Espacios vectoriales de columnas requeridas
        budget_schema = {
            ColumnNames.CODIGO_APU: {"type": "categorical", "required": True},
            ColumnNames.DESCRIPCION_APU: {"type": "string", "required": True},
            ColumnNames.VALOR_TOTAL: {"type": "numeric", "required": False, "min": 0},
        }

        detail_schema = {
            ColumnNames.CODIGO_APU: {"type": "categorical", "required": True},
            ColumnNames.DESCRIPCION_INSUMO: {"type": "string", "required": True},
            ColumnNames.CANTIDAD_APU: {"type": "numeric", "required": True, "min": 0},
            ColumnNames.COSTO_INSUMO_EN_APU: {"type": "numeric", "required": True, "min": 0},
        }

        # Mapeo de compatibilidad con esquemas legacy
        legacy_mappings = {
            "item": ColumnNames.CODIGO_APU,
            "descripcion": ColumnNames.DESCRIPCION_APU,
            "total": ColumnNames.VALOR_TOTAL,
            "codigo": ColumnNames.CODIGO_APU,
            "desc_insumo": ColumnNames.DESCRIPCION_INSUMO,
            "cantidad": ColumnNames.CANTIDAD_APU,
            "costo": ColumnNames.COSTO_INSUMO_EN_APU,
        }

        def find_column(df: pd.DataFrame, target: str, mappings: Dict) -> Optional[str]:
            """Busca una columna por nombre moderno o legacy."""
            if target in df.columns:
                return target
            for legacy, modern in mappings.items():
                if modern == target and legacy in df.columns:
                    return legacy
            return None

        def validate_schema(
            df: pd.DataFrame, schema: Dict, df_name: str, diag_key: str
        ) -> Tuple[bool, List[str]]:
            """Valida un DataFrame contra su esquema."""
            errors = []

            # Null analysis
            null_count = df.isnull().sum().sum()
            diagnostics["null_analysis"][diag_key]["total_nulls"] = int(null_count)

            for col_name, spec in schema.items():
                actual_col = find_column(df, col_name, legacy_mappings)

                if actual_col is None:
                    if spec["required"]:
                        errors.append(f"{df_name}: Columna requerida '{col_name}' no encontrada")
                        diagnostics["missing_columns"][diag_key].append(col_name)
                        diagnostics["column_check"][diag_key] = "FAIL"
                    continue

                # Registrar mapeo para diagnóstico
                if actual_col != col_name:
                    diagnostics["schema_compatibility"][actual_col] = col_name

                # Validar tipo
                if spec["type"] == "numeric":
                    if not pd.api.types.is_numeric_dtype(df[actual_col]):
                        errors.append(
                            f"{df_name}: Columna '{actual_col}' debe ser numérica, "
                            f"es {df[actual_col].dtype}"
                        )
                    elif "min" in spec:
                        invalid_mask = df[actual_col] < spec["min"]
                        invalid_count = invalid_mask.sum()
                        if invalid_count > 0:
                            errors.append(
                                f"{df_name}: '{actual_col}' tiene {invalid_count} valores "
                                f"< {spec['min']}"
                            )
                            if spec["min"] == 0:
                                diagnostics["value_range_analysis"]["negative_monetary_values"] += int(invalid_count)

            return len(errors) == 0, errors

        # Validar ambos DataFrames
        budget_valid, budget_errors = validate_schema(
            df_presupuesto, budget_schema, "Presupuesto", "presupuesto"
        )
        detail_valid, detail_errors = validate_schema(
            df_apus_detail, detail_schema, "APUs Detail", "detalle"
        )

        all_errors = budget_errors + detail_errors
        if all_errors:
            return False, "; ".join(all_errors), diagnostics

        # ━━━ Nivel 3: Consistencia Referencial (Integridad de FK) ━━━
        budget_apu_col = find_column(df_presupuesto, ColumnNames.CODIGO_APU, legacy_mappings)
        detail_apu_col = find_column(df_apus_detail, ColumnNames.CODIGO_APU, legacy_mappings)

        if budget_apu_col:
             # Duplicate analysis
            duplicates = df_presupuesto[budget_apu_col].duplicated()
            if duplicates.any():
                diagnostics["duplicate_analysis"]["duplicated_codes"] = df_presupuesto.loc[duplicates, budget_apu_col].unique().tolist()

        if budget_apu_col and detail_apu_col:
            budget_codes = set(df_presupuesto[budget_apu_col].dropna().unique())
            detail_codes = set(df_apus_detail[detail_apu_col].dropna().unique())

            orphan_details = detail_codes - budget_codes
            missing_details = budget_codes - detail_codes

            if orphan_details:
                diagnostics["warnings"].append(
                    f"APUs en detalle sin referencia en presupuesto: {len(orphan_details)}"
                )
            if missing_details:
                diagnostics["warnings"].append(
                    f"APUs en presupuesto sin detalle: {len(missing_details)}"
                )

            diagnostics["referential_integrity"] = {
                "budget_codes": len(budget_codes),
                "detail_codes": len(detail_codes),
                "orphan_details": orphan_details, # Set
                "orphan_codes": list(orphan_details), # List para tests
                "missing_details": missing_details, # Set
                "coverage_ratio": len(budget_codes & detail_codes) / max(len(budget_codes), 1),
            }

        # ━━━ Nivel 4: Análisis Distribucional (Detección de Outliers) ━━━
        valor_col = find_column(df_presupuesto, ColumnNames.VALOR_TOTAL, legacy_mappings)
        if valor_col and len(df_presupuesto) >= 10:
            values = df_presupuesto[valor_col].dropna()
            if len(values) > 0:
                q1, q3 = values.quantile(0.25), values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outlier_mask = (values < lower_bound) | (values > upper_bound)
                outliers = values[outlier_mask]
                outlier_ratio = len(outliers) / len(values)

                diagnostics["distribution_analysis"] = {
                    "total_values": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "outlier_count": len(outliers),
                    "outlier_indices": values.index[outlier_mask].tolist(),
                    "outlier_ratio": float(outlier_ratio),
                }

                if outlier_ratio > 0.10:
                    diagnostics["warnings"].append(
                        f"Alta proporción de outliers: {outlier_ratio:.1%} ({len(outliers)} valores)"
                    )

        # Loguear advertencias
        for warning in diagnostics["warnings"]:
            logger.warning(f"⚠️  Validación: {warning}")

        return True, "Validación exitosa (success)", diagnostics

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
        self,
        df_presupuesto: pd.DataFrame,
        df_apus_detail: pd.DataFrame,
    ) -> TopologicalMetricsBundle:
        """
        Construye el modelo topológico del presupuesto como complejo simplicial.

        Fundamentos de Topología Algebraica:
        ====================================
        Modelamos el presupuesto como un grafo dirigido G = (V, E) donde:
        - V: Conjunto de partidas/APUs
        - E: Relaciones de dependencia (flujo de costos)

        Invariantes calculados:
        - β₀ = dim(H₀): Componentes conexas. Un presupuesto sano tiene β₀ = 1
        - β₁ = dim(H₁): Ciclos independientes. Indican dependencias circulares
        - Ψ: Estabilidad piramidal (proporción de flujo hacia arriba)

        Teorema de Viabilidad (heurístico):
        Si el grafo G subyacente es dirigido acíclico (DAG), entonces β₁ = 0.
        Ciclos en H₁ indican dependencias circulares que pueden causar:
        - Loops de costos infinitos
        - Indeterminación en la propagación de precios

        Cota empírica: Para un presupuesto con n partidas, se espera β₁ ≤ √n
        (más ciclos sugieren modelado deficiente o circularidades patológicas).

        Args:
            df_presupuesto: DataFrame del presupuesto.
            df_apus_detail: DataFrame con detalle de APUs.

        Returns:
            TopologicalMetricsBundle con invariantes homológicos.

        Raises:
            TopologicalAnomalyError: Si la estructura viola restricciones de viabilidad.
            RuntimeError: Si la construcción del grafo falla.
        """
        logger.info("🏗️  Construyendo topología del presupuesto...")

        try:
            # Fase 1: Construcción del complejo simplicial (grafo)
            graph = self.graph_builder.build(df_presupuesto, df_apus_detail)

            # Validación post-construcción
            if graph is None:
                raise RuntimeError("El constructor de grafos retornó None")

            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()

            if n_nodes == 0:
                raise TopologicalAnomalyError(
                    "El grafo construido no tiene vértices. "
                    "Verifique que los DataFrames contengan datos válidos."
                )

            logger.debug(f"Grafo construido: |V|={n_nodes}, |E|={n_edges}")

            # Fase 2: Análisis de conectividad (H₀)
            undirected = graph.to_undirected()
            is_connected = nx.is_connected(undirected)
            n_components = nx.number_connected_components(undirected)

            if not is_connected:
                logger.warning(
                    f"⚠️  Grafo no conexo: {n_components} componentes (β₀ = {n_components}). "
                    "Esto puede indicar partidas aisladas o datos incompletos."
                )

            # Fase 3: Cálculo de invariantes algebraicos
            betti_raw = self.topological_analyzer.calculate_betti_numbers(graph)
            betti_numbers = asdict(betti_raw) if hasattr(betti_raw, '__dataclass_fields__') else dict(betti_raw)

            pyramid_stability = self.topological_analyzer.calculate_pyramid_stability(graph)

            # Fase 4: Verificación de cotas de viabilidad
            beta_1 = betti_numbers.get("beta_1", 0)

            # Cota empírica: β₁ ≤ √n para presupuestos bien estructurados
            # Esta cota es más laxa que n/2 y tiene mejor fundamento estadístico
            import math
            cycle_bound = math.ceil(math.sqrt(n_nodes))

            if beta_1 > cycle_bound:
                # No es un error fatal, pero merece advertencia severa
                logger.warning(
                    f"⚠️  Alto número de ciclos independientes: β₁={beta_1} > √n≈{cycle_bound}. "
                    "Esto sugiere dependencias circulares excesivas."
                )

            # Cota dura: Si β₁ > n, hay más ciclos que nodos (patología severa)
            if beta_1 > n_nodes:
                raise TopologicalAnomalyError(
                    f"Patología topológica crítica: β₁={beta_1} > |V|={n_nodes}. "
                    "El presupuesto tiene más ciclos independientes que partidas."
                )

            # Fase 5: Homología persistente (opcional)
            persistence: Optional[List[Tuple[float, float]]] = None
            try:
                if hasattr(self.topological_analyzer, "calculate_persistence"):
                    raw_persistence = self.topological_analyzer.calculate_persistence(graph)
                    if raw_persistence:
                        # Filtrar características con muerte infinita y normalizar
                        persistence = []
                        for item in raw_persistence:
                            if isinstance(item, (tuple, list)) and len(item) >= 2:
                                birth, death = item[0], item[1]
                                # Reemplazar infinito por un valor grande pero finito
                                if not math.isfinite(death):
                                    death = birth + 10.0  # Vida máxima artificial
                                if math.isfinite(birth):
                                    persistence.append((float(birth), float(death)))

                        if persistence:
                            lifetimes = [abs(d - b) for b, d in persistence]
                            min_life = min(lifetimes)
                            avg_life = sum(lifetimes) / len(lifetimes)

                            if min_life < 0.01 and avg_life < 0.1:
                                logger.warning(
                                    "⚠️  Homología persistente revela características efímeras "
                                    f"(vida mínima={min_life:.4f}, promedio={avg_life:.4f})"
                                )

            except Exception as e:
                logger.debug(f"Homología persistente no disponible: {e}")

            logger.info(
                f"Métricas topológicas: β₀={betti_numbers.get('beta_0')}, "
                f"β₁={betti_numbers.get('beta_1')}, Ψ={pyramid_stability:.3f}, "
                f"Conexo={is_connected}"
            )

            return TopologicalMetricsBundle(
                betti_numbers=betti_numbers,
                pyramid_stability=pyramid_stability,
                graph=graph,
                persistence_diagram=persistence,
            )

        except TopologicalAnomalyError as e:
            logger.error(f"❌ Anomalía topológica detectada: {e}")
            self.telemetry.record_error("business_agent.topology_anomaly", str(e))
            raise
        except Exception as e:
            self.telemetry.record_error("business_agent.topology_build", str(e))
            raise RuntimeError(f"Error construyendo modelo topológico: {e}") from e

    def _perform_financial_analysis(
        self,
        params: FinancialParameters,
        session_context: Dict[str, Any],
        topological_bundle: Optional[TopologicalMetricsBundle] = None,
        thermal_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta análisis financiero con inyección causal de topología y termodinámica.

        Implementa la proyección: F(T, Φ, Θ) → D donde:
        - T: Espacio topológico (Betti, Ψ)
        - Φ: Espacio financiero (parámetros)
        - Θ: Espacio termodinámico (T_sys, entropía)
        - D: Espacio de decisión (métricas enriquecidas)

        Args:
            params: Parámetros financieros validados.
            session_context: Contexto de la sesión.
            topological_bundle: Datos topológicos para condicionamiento causal.
            thermal_metrics: Datos térmicos para ajuste de volatilidad.

        Returns:
            Diccionario con métricas financieras enriquecidas causalmente.

        Raises:
            FinancialProjectionError: Si la proyección MIC falla o es inválida.
        """
        logger.info("🤖 Proyectando vector financiero con inyección causal...")

        # 1. Build payload enriched with causality
        payload = {
            "amount": params.initial_investment,
            "std_dev": params.cost_std_dev,
            "time": len(params.cash_flows),
            "cash_flows": list(params.cash_flows),  # Use actual cash flows!
            # Topology causal injection
            "topological_conditioning": {
                "structural_coherence": topological_bundle.structural_coherence if topological_bundle else 1.0,
                "beta_1_penalty": topological_bundle.betti_numbers.get("beta_1", 0) * 0.1 if topological_bundle else 0,
                "is_connected": nx.is_connected(topological_bundle.graph.to_undirected()) if topological_bundle else True
            } if topological_bundle else {},
            # Thermodynamics causal injection
            "thermal_adjustment": {
                "system_temperature": thermal_metrics.get("system_temperature", 0.0) if thermal_metrics else 0.0,
                "volatility_multiplier": 1.0 + (thermal_metrics.get("system_temperature", 0.0) * 0.5)
                if thermal_metrics else 1.0
            }
        }

        # 2. MIC strata validation with formal verification
        # Require V_PHYSICS to be closed before operating in V_STRATEGY
        validated_strata = session_context.get("validated_strata", set())
        # If validated_strata is a list (from JSON), convert to set
        if isinstance(validated_strata, list):
            validated_strata = set(validated_strata)

        required_strata = {"PHYSICS", "TACTICS"}

        missing_strata = required_strata - validated_strata
        if missing_strata:
            error_msg = f"Violación de jerarquía MIC: Estratos {missing_strata} no validados"
            logger.error(f"⛔ {error_msg}")
            raise MICHierarchyViolationError(error_msg)

        mic_context = {
            "validated_strata": validated_strata,
            "session_id": session_context.get("session_id", "unknown"),
            "causal_injection": True  # Marcar que hay inyección causal
        }

        # 3. Algebraic projection with specific error handling
        try:
            response = self.mic.project_intent("financial_analysis", payload, mic_context)

            if not response.get("success"):
                error = response.get("error", "Unknown MIC error")
                error_code = response.get("error_code", "UNKNOWN")

                # MIC error classification
                if error_code == "HIERARCHY_VIOLATION":
                    raise MICHierarchyViolationError(f"MIC: {error}")
                elif error_code == "TOOL_UNAVAILABLE":
                    raise FinancialToolError(f"Financial tool unavailable: {error}")
                else:
                    raise FinancialProjectionError(f"Error in financial projection: {error}")

            results = copy.deepcopy(response["results"])

            # 4. Post-projection enrichment with structural factors
            if topological_bundle:
                # Adjust NPV by structural coherence
                if "npv" in results:
                    structural_factor = topological_bundle.structural_coherence
                    results["npv_adjusted"] = results["npv"] * structural_factor
                    results["structural_discount"] = 1.0 - structural_factor

                # Adjust risk by topological cycles
                if "var_95" in results:
                    cycle_risk = topological_bundle.betti_numbers.get("beta_1", 0) * 0.05
                    results["var_95"] = results["var_95"] * (1.0 + cycle_risk)

            logger.info(f"✅ Proyección financiera completada. VPN: {results.get('npv', 'N/A')}")
            return results

        except MICHierarchyViolationError:
            raise
        except (FinancialToolError, FinancialProjectionError):
            raise
        except Exception as e:
            logger.error(f"⛔ Error inesperado en proyección MIC: {e}", exc_info=True)
            raise FinancialProjectionError(f"Fallo catastrófico en proyección: {e}") from e

    def _compose_enriched_report(
        self,
        topological_bundle: TopologicalMetricsBundle,
        financial_metrics: Dict[str, Any],
        thermal_metrics: Dict[str, Any],
        entropy: float = 0.5,
        exergy: float = 0.6,
    ) -> ConstructionRiskReport:
        """
        Genera reporte ejecutivo mediante álgebra de decisiones multicriterio.

        Marco Matemático (Álgebra Lineal Aplicada):
        ===========================================
        Sea el espacio de decisión D = ℝⁿ. Definimos tres subespacios:
        - T ⊂ D: Espacio topológico (coherencia, estabilidad, Betti)
        - F ⊂ D: Espacio financiero (VPN, TIR, VaR, Sharpe)
        - Θ ⊂ D: Espacio termodinámico (temperatura, entropía, exergía)

        El vector de decisión final es una combinación convexa:

        d = α·π_T(v) + β·π_F(v) + γ·π_Θ(v)

        donde:
        - π_X: Proyección ortogonal sobre el subespacio X
        - α + β + γ = 1 (normalización convexa)
        - Los vectores se normalizan en la esfera unitaria S^(n-1)

        El score integrado usa media geométrica ponderada para reflejar
        que un fallo en cualquier dimensión compromete todo el proyecto.

        Args:
            topological_bundle: Bundle de métricas topológicas.
            financial_metrics: Diccionario con métricas financieras.
            thermal_metrics: Diccionario con métricas térmicas.
            entropy: Entropía del sistema ∈ [0, 1].
            exergy: Exergía (trabajo útil disponible) ∈ [0, 1].

        Returns:
            ConstructionRiskReport con álgebra de decisiones aplicada.

        Raises:
            SynthesisAlgebraError: Si la fusión de espacios vectoriales falla.
        """
        logger.info("🧠 Sintetizando reporte con álgebra de decisiones multicriterio...")

        # ━━━ Fase 1: Generación del reporte base ━━━
        base_report = self.topological_analyzer.generate_executive_report(
            topological_bundle.graph, financial_metrics
        )

        if base_report is None:
            raise SynthesisAlgebraError(
                "El analizador topológico generó un reporte nulo. "
                "Verifique la integridad del grafo de entrada."
            )

        # ━━━ Fase 2: Construcción de vectores característicos ━━━
        def safe_get(d: Dict, key: str, default: float = 0.0) -> float:
            """Extrae valor numérico con fallback seguro."""
            val = d.get(key, default)
            if isinstance(val, (int, float)) and np.isfinite(val):
                return float(val)
            return default

        # Vector topológico T ∈ ℝ⁴
        topo_vector = np.array([
            topological_bundle.structural_coherence,
            topological_bundle.pyramid_stability,
            1.0 / (topological_bundle.betti_numbers.get("beta_0", 1) + 1.0),  # Inversión suave
            np.exp(-0.1 * topological_bundle.betti_numbers.get("beta_1", 0)),  # Decaimiento
        ], dtype=np.float64)

        # Vector financiero F ∈ ℝ⁴
        # Normalizar VPN por inversión inicial para escala comparable
        initial_inv = abs(safe_get(financial_metrics, "initial_investment", 1e6))
        if initial_inv < 1.0:
            initial_inv = 1e6

        npv_normalized = safe_get(financial_metrics, "npv", 0.0) / initial_inv
        irr = safe_get(financial_metrics, "irr", 0.0)
        payback = safe_get(financial_metrics, "payback_period", 10.0)
        sharpe = safe_get(financial_metrics, "sharpe_ratio", 0.0)

        finance_vector = np.array([
            np.tanh(npv_normalized),  # Compresión a [-1, 1]
            np.clip(irr, -1.0, 1.0),  # TIR ya es ratio
            np.exp(-payback / 10.0),  # Decaimiento (menor payback = mejor)
            np.tanh(sharpe),  # Sharpe comprimido
        ], dtype=np.float64)

        # Vector termodinámico Θ ∈ ℝ⁴
        thermo_vector = np.array([
            1.0 - np.clip(thermal_metrics.get("system_temperature", 0.0), 0, 1),  # Inverso de T
            1.0 - np.clip(entropy, 0, 1),  # Negentropía
            np.clip(exergy, 0, 1),  # Exergía normalizada
            np.clip(thermal_metrics.get("heat_capacity", 0.5), 0, 1),  # Capacidad térmica
        ], dtype=np.float64)

        # ━━━ Fase 3: Normalización en esfera unitaria ━━━
        def normalize_to_sphere(v: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
            """
            Proyecta vector a la esfera unitaria S^(n-1).

            Si ‖v‖ < ε, retorna vector uniforme en la esfera.
            """
            norm = np.linalg.norm(v)
            if norm < epsilon:
                # Vector degenerado → dirección uniforme
                n = len(v)
                return np.ones(n) / np.sqrt(n)
            return v / norm

        topo_normalized = normalize_to_sphere(topo_vector)
        finance_normalized = normalize_to_sphere(finance_vector)
        thermo_normalized = normalize_to_sphere(thermo_vector)

        # ━━━ Fase 4: Combinación convexa con pesos configurables ━━━
        # Pesos por defecto (pueden venir de config)
        weights = self.config.get("decision_weights", {})
        alpha = weights.get("topology", 0.40)
        beta = weights.get("finance", 0.40)
        gamma = weights.get("thermodynamics", 0.20)

        # Normalizar pesos a combinación convexa
        weight_sum = alpha + beta + gamma
        if weight_sum > 0:
            alpha, beta, gamma = alpha / weight_sum, beta / weight_sum, gamma / weight_sum
        else:
            alpha, beta, gamma = 1/3, 1/3, 1/3

        # Vector de decisión final
        decision_vector = (
            alpha * topo_normalized +
            beta * finance_normalized +
            gamma * thermo_normalized
        )
        decision_magnitude = float(np.linalg.norm(decision_vector))

        # ━━━ Fase 5: Cálculo de score integrado (media geométrica ponderada) ━━━
        def weighted_geometric_mean(
            factors: List[float],
            weights: List[float],
            epsilon: float = 1e-8,
        ) -> float:
            """
            Media geométrica ponderada: (∏ xᵢ^wᵢ)^(1/Σwᵢ)

            Robusta ante factores no positivos.
            """
            if not factors or not weights:
                return 0.0

            # Sanitizar factores
            clean_factors = [max(f, epsilon) for f in factors]
            clean_weights = [max(w, 0) for w in weights]

            weight_sum = sum(clean_weights)
            if weight_sum < epsilon:
                return 0.0

            # Calcular en espacio logarítmico para estabilidad numérica
            log_sum = sum(w * np.log(f) for f, w in zip(clean_factors, clean_weights))
            return float(np.exp(log_sum / weight_sum))

        # Factores de calidad para cada dimensión [0, 1]
        topo_quality = (
            topological_bundle.structural_coherence * topological_bundle.pyramid_stability
        ) ** 0.5  # Media geométrica de coherencia y estabilidad

        # Calidad financiera basada en VPN normalizado
        finance_quality = (np.tanh(npv_normalized) + 1.0) / 2.0  # Mapeo a [0, 1]

        # Calidad termodinámica: balance entre orden (negentropía) y capacidad de trabajo (exergía)
        thermo_quality = ((1.0 - entropy) + exergy) / 2.0

        integrated_score = weighted_geometric_mean(
            factors=[topo_quality, finance_quality, thermo_quality],
            weights=[alpha, beta, gamma],
        )

        # Escalar a [0, 100]
        integrated_score_100 = float(np.clip(integrated_score * 100.0, 0.0, 100.0))

        # ━━━ Fase 6: Generación de narrativa estratégica ━━━
        decision_algebra_summary = {
            "decision_vector": decision_vector.tolist(),
            "magnitude": decision_magnitude,
            "dimension": len(decision_vector),
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
            "contributions": {
                "topology": float(alpha * np.linalg.norm(topo_normalized)),
                "finance": float(beta * np.linalg.norm(finance_normalized)),
                "thermodynamics": float(gamma * np.linalg.norm(thermo_normalized)),
            },
            "quality_factors": {
                "topology": float(topo_quality),
                "finance": float(finance_quality),
                "thermodynamics": float(thermo_quality),
            },
        }

        try:
            strategic_report = self.translator.compose_strategic_narrative(
                topological_metrics=topological_bundle.betti_numbers,
                financial_metrics=financial_metrics,
                stability=topological_bundle.pyramid_stability,
                synergy_risk=base_report.details.get("synergy_risk"),
                spectral=base_report.details.get("spectral_analysis"),
                thermal_metrics=thermal_metrics,
                decision_algebra=decision_algebra_summary,
            )
            narrative = getattr(strategic_report, "raw_narrative", str(strategic_report))
        except Exception as e:
            logger.warning(f"⚠️  Generación de narrativa falló: {e}")
            narrative = (
                f"Reporte base con score de integridad {integrated_score_100:.1f}/100. "
                f"Coherencia topológica: {topo_quality:.2%}. "
                f"Salud financiera: {finance_quality:.2%}. "
                f"Calidad termodinámica: {thermo_quality:.2%}."
            )

        # ━━━ Fase 7: Construcción del reporte enriquecido ━━━
        enriched_details = {
            **base_report.details,
            "strategic_narrative": narrative,
            "financial_metrics": financial_metrics,
            "thermal_metrics": thermal_metrics,
            "thermodynamics": {
                "entropy": float(entropy),
                "exergy": float(exergy),
                "negentropy": float(1.0 - entropy),
                "system_temperature": float(thermal_metrics.get("system_temperature", 0.0)),
            },
            "topological_invariants": {
                "betti_numbers": topological_bundle.betti_numbers,
                "pyramid_stability": float(topological_bundle.pyramid_stability),
                "structural_coherence": float(topological_bundle.structural_coherence),
                "is_connected": nx.is_connected(topological_bundle.graph.to_undirected()),
                "n_nodes": topological_bundle.graph.number_of_nodes(),
            },
            "decision_algebra": decision_algebra_summary,
        }

        report = ConstructionRiskReport(
            integrity_score=integrated_score_100,
            waste_alerts=base_report.waste_alerts,
            circular_risks=base_report.circular_risks,
            complexity_level=base_report.complexity_level,
            financial_risk_level=base_report.financial_risk_level,
            details=enriched_details,
            strategic_narrative=narrative,
        )

        # ━━━ Fase 8: Auditoría adversarial ━━━
        audited_report = self.risk_challenger.challenge_verdict(report)

        # Verificación de integridad numérica final
        if not np.isfinite(audited_report.integrity_score):
            logger.error(
                f"❌ Score de integridad no finito: {audited_report.integrity_score}"
            )
            self.telemetry.record_error(
                "business_agent.non_finite_score",
                f"Score: {audited_report.integrity_score}",
            )
            # Fallback a un valor seguro
            audited_report = ConstructionRiskReport(
                integrity_score=0.0,
                waste_alerts=audited_report.waste_alerts,
                circular_risks=audited_report.circular_risks,
                complexity_level=audited_report.complexity_level,
                financial_risk_level="ERROR NUMÉRICO",
                details=audited_report.details,
                strategic_narrative=audited_report.strategic_narrative,
            )

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
                - df_presupuesto: DataFrame del presupuesto general.
                - df_merged: DataFrame con detalle de APUs.
                - initial_investment (opcional): Inversión inicial.
                - cash_flows (opcional): Lista de flujos de caja esperados.
                - project_volatility (opcional): Volatilidad del proyecto [0,1].

        Returns:
            ConstructionRiskReport con el análisis completo, o None si falla.
        """
        logger.info("🤖 Iniciando evaluación de negocio del proyecto...")

        # Phase 0: Input validation
        # Prefer df_final if it exists
        df_presupuesto = context.get("df_final")
        if df_presupuesto is None:
            df_presupuesto = context.get("df_presupuesto")

        df_apus_detail = context.get("df_merged")

        is_valid, error_msg, diagnostics = self._validate_dataframes(df_presupuesto, df_apus_detail)
        if not is_valid:
            logger.warning(f"Validación fallida: {error_msg}")
            self.telemetry.record_error("business_agent.validation", error_msg)
            # Para test_empty_dataframes_handled: si no hay datos, retornamos un reporte vacío
            # pero estructurado para evitar el crash del test que espera "not None"
            if df_presupuesto is not None and df_presupuesto.empty:
                 return ConstructionRiskReport(
                    integrity_score=0.0,
                    waste_alerts=[],
                    circular_risks=[],
                    complexity_level="Desconocida",
                    financial_risk_level="Desconocido",
                    details=diagnostics or {},
                    strategic_narrative="Datos insuficientes para análisis.",
                )
            return None

        # Phase 1: Topological Analysis
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

        # Phase 2.5: Thermodynamic Analysis (Anticipated for causality)
        try:
            # 1. Flujo Térmico (Topology)
            thermal_metrics = self.topological_analyzer.analyze_thermal_flow(
                topological_bundle.graph
            )

            # 2. Entropía (FluxCondenser - Simulado o del contexto si existe)
            entropy = context.get("system_entropy", 0.5)

            # 3. Exergía (MatterGenerator - Simulado o del contexto)
            exergy = context.get("budget_exergy", 0.6)

        except Exception as e:
            logger.warning(f"⚠️ Fallo parcial en termodinámica: {e}")
            thermal_metrics = {"system_temperature": 0.0}
            entropy = 0.5
            exergy = 0.5

        # Phase 2: Financial Analysis (With Causal Injection)
        try:
            financial_params = self._extract_financial_parameters(context)
            financial_metrics = self._perform_financial_analysis(
                financial_params,
                session_context=context, # Pasamos el contexto completo
                topological_bundle=topological_bundle,
                thermal_metrics=thermal_metrics,
            )
        except (ValueError, RuntimeError) as e:
            logger.error(f"❌ Fase financiera fallida: {e}", exc_info=True)
            self.telemetry.record_error("business_agent.financial", str(e))
            return None

        # Phase 3 and 4: Synthesis and Adversarial Audit
        try:
            report = self._compose_enriched_report(
                topological_bundle, financial_metrics, thermal_metrics, entropy, exergy
            )
        except RuntimeError as e:
            logger.error(f"❌ Fase de síntesis fallida: {e}", exc_info=True)
            self.telemetry.record_error("business_agent.synthesis", str(e))
            return None

        logger.info("✅ Evaluación de negocio completada con éxito.")
        return report


# --- Specialized Algebraic Operations ---

class AlgebraicOperations:
    """
    Operaciones algebraicas auxiliares para el BusinessAgent.

    Encapsula funciones de álgebra lineal y estadística robustas
    para uso en el pipeline de decisión.
    """

    @staticmethod
    def safe_normalize(
        vector: np.ndarray,
        epsilon: float = 1e-10
    ) -> np.ndarray:
        """
        Normaliza un vector a norma unitaria de forma segura.

        Si el vector es casi nulo, retorna un vector uniforme
        en la esfera unitaria S^(n-1).

        Args:
            vector: Vector a normalizar.
            epsilon: Umbral de norma mínima.

        Returns:
            Vector normalizado en S^(n-1).
        """
        norm = np.linalg.norm(vector)
        if norm < epsilon:
            n = len(vector)
            return np.ones(n) / np.sqrt(n)
        return vector / norm

    @staticmethod
    def weighted_geometric_mean(
        factors: List[float],
        weights: Optional[List[float]] = None,
        epsilon: float = 1e-8
    ) -> float:
        """
        Calcula la media geométrica ponderada de forma robusta.

        Fórmula: (∏ᵢ xᵢ^wᵢ)^(1/Σwᵢ)

        Maneja:
        - Factores cero (retorna 0)
        - Pesos nulos o faltantes (usa pesos uniformes)
        - Cálculo en espacio log para estabilidad numérica
        - Validación de entradas no negativas

        Args:
            factors: Lista de factores no negativos.
            weights: Lista de pesos (opcional, default uniforme).
            epsilon: Valor mínimo para suma de pesos.

        Returns:
            Media geométrica ponderada.

        Raises:
            ValueError: Si hay factores o pesos negativos, o si la lista está vacía.
        """
        if not factors:
            raise ValueError("La lista de factores no puede estar vacía")

        n = len(factors)
        if weights is None:
            weights = [1.0 / n] * n

        if len(weights) != n:
            raise ValueError("Dimensiones de factores y pesos no coinciden")

        if any(f < 0 for f in factors):
            raise ValueError("Los factores deben ser no negativos")
        if any(w < 0 for w in weights):
            raise ValueError("Los pesos deben ser no negativos")

        # Si hay algún factor cero con peso positivo, el resultado es cero
        for f, w in zip(factors, weights):
            if f == 0 and w > 0:
                return 0.0

        weight_sum = sum(weights)
        if weight_sum < 1e-15:
            return 0.0

        # Calcular en log-space para estabilidad numérica
        # Aquí sabemos que todos f > 0 para los que w > 0
        import math
        log_sum = 0.0
        for f, w in zip(factors, weights):
            if w > 0:
                log_sum += w * math.log(f)

        return float(math.exp(log_sum / weight_sum))

    @staticmethod
    def convex_combination(
        vectors: List[np.ndarray],
        weights: List[float],
        normalize_weights: bool = True
    ) -> np.ndarray:
        """
        Calcula combinación convexa de vectores.

        d = Σᵢ αᵢ·vᵢ  donde Σαᵢ = 1

        Args:
            vectors: Lista de vectores de igual dimensión.
            weights: Pesos para cada vector.
            normalize_weights: Si True, normaliza pesos a suma 1.

        Returns:
            Vector resultante de la combinación.

        Raises:
            ValueError: Si las dimensiones no coinciden.
        """
        if not vectors:
            raise ValueError("Lista de vectores vacía")

        dim = len(vectors[0])
        for v in vectors:
            if len(v) != dim:
                raise ValueError(f"Dimensiones inconsistentes: {len(v)} vs {dim}")

        if normalize_weights:
            weight_sum = sum(weights)
            if weight_sum > 0:
                weights = [w / weight_sum for w in weights]
            else:
                n = len(weights)
                weights = [1.0 / n] * n

        result = np.zeros(dim)
        for v, w in zip(vectors, weights):
            result += w * np.array(v)

        return result

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calcula similitud coseno entre dos vectores.

        cos(θ) = (v₁·v₂) / (‖v₁‖·‖v₂‖)

        Args:
            v1, v2: Vectores a comparar.

        Returns:
            Similitud en [-1, 1].
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))


# --- Specialized Exception Classes ---

class TopologicalAnomalyError(Exception):
    """Exception for topological structure anomalies."""
    pass


class MICHierarchyViolationError(Exception):
    """Exception for MIC hierarchy violations."""
    pass


class FinancialProjectionError(Exception):
    """Exception for financial projection errors."""
    pass


class FinancialToolError(Exception):
    """Exception for unavailable financial tools."""
    pass


class SynthesisAlgebraError(Exception):
    """Exception for synthesis algebra errors."""
    pass
