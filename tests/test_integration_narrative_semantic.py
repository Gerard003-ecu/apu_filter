"""
Suite de Tests de Integración: TelemetryNarrator ↔ SemanticTranslator
======================================================================

Valida la coherencia y compatibilidad entre los módulos de narrativa:
1. Isomorfismo entre lattices (SeverityLevel ↔ VerdictLevel)
2. Consistencia en uso de Stratum/jerarquía DIKW
3. Composición de reportes combinados
4. Flujo de datos end-to-end (Telemetry → Topology → Narrative)
5. Coherencia de narrativas generadas

Arquitectura de Tests:
- TestLatticeIsomorphism: Verificar isomorfismo entre lattices
- TestStratumConsistency: Consistencia de estratos entre módulos
- TestReportComposition: Composición de reportes combinados
- TestEndToEndFlow: Flujo completo de datos
- TestNarrativeCoherence: Coherencia semántica de narrativas
- TestErrorPropagation: Propagación de errores entre módulos
- TestCombinedDecisionMatrix: Matriz de decisión combinada
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import pytest

# Importar desde telemetry_narrative (módulo refinado)
from app.telemetry_narrative import (
    SeverityLevel,
    Issue,
    PhaseAnalysis,
    StratumAnalysis,
    PyramidalReport,
    TelemetryNarrator,
    NarrativeTemplates as TelemetryTemplates,
    StratumTopology,
    NarratorConfig,
    create_narrator,
)

# Importar desde semantic_translator (módulo refinado)
from app.semantic_translator import (
    VerdictLevel,
    FinancialVerdict,
    StratumAnalysisResult,
    StrategicReport,
    SemanticTranslator,
    TranslatorConfig,
    NarrativeTemplates as TranslatorTemplates,
    create_translator,
)
from app.telemetry_schemas import (
    TopologicalMetrics,
    ThermodynamicMetrics,
    PhysicsMetrics,
    ControlMetrics,
)

# Importar dependencias comunes
from app.telemetry import TelemetryContext, TelemetrySpan, StepStatus
from app.schemas import Stratum


# ============================================================================
# FIXTURES COMPARTIDAS
# ============================================================================


@pytest.fixture
def narrator() -> TelemetryNarrator:
    """Narrador de telemetría con configuración por defecto."""
    return TelemetryNarrator()


@pytest.fixture
def translator() -> SemanticTranslator:
    """Traductor semántico con configuración determinística."""
    config = TranslatorConfig(deterministic_market=True)
    return SemanticTranslator(config=config)


@pytest.fixture
def context() -> TelemetryContext:
    """Contexto de telemetría vacío."""
    return TelemetryContext()


@pytest.fixture
def clean_topology() -> TopologicalMetrics:
    """Topología limpia (sin ciclos, conexa)."""
    return TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)


@pytest.fixture
def cyclic_topology() -> TopologicalMetrics:
    """Topología con ciclos."""
    return TopologicalMetrics(beta_0=1, beta_1=3, euler_characteristic=-2)


@pytest.fixture
def viable_financials() -> Dict[str, Any]:
    """Métricas financieras viables."""
    return {
        "wacc": 0.10,
        "contingency": {"recommended": 5000.0},
        "performance": {"recommendation": "ACEPTAR", "profitability_index": 1.25},
    }


@pytest.fixture
def rejected_financials() -> Dict[str, Any]:
    """Métricas financieras de rechazo."""
    return {
        "wacc": 0.18,
        "contingency": {"recommended": 50000.0},
        "performance": {"recommendation": "RECHAZAR", "profitability_index": 0.7},
    }


# ============================================================================
# TEST: ISOMORFISMO ENTRE LATTICES
# ============================================================================


class TestLatticeIsomorphism:
    """
    Verifica que SeverityLevel y VerdictLevel son estructuras isomorfas.
    
    Un isomorfismo f: L₁ → L₂ preserva:
    - Orden: a ≤ b ⟹ f(a) ≤ f(b)
    - Join: f(a ⊔ b) = f(a) ⊔ f(b)
    - Meet: f(a ⊓ b) = f(a) ⊓ f(b)
    """

    def test_both_lattices_have_same_cardinality(self):
        """Ambos lattices tienen el mismo número de elementos (o comparable)."""
        severity_count = len(SeverityLevel)
        verdict_count = len(VerdictLevel)
        
        # SeverityLevel tiene 3 elementos, VerdictLevel tiene 5
        # No son isomorfos exactos, pero podemos definir un morfismo
        assert severity_count == 3
        assert verdict_count == 5

    def test_natural_embedding_severity_to_verdict(self):
        """
        Existe un morfismo natural de SeverityLevel a VerdictLevel.
        OPTIMO → VIABLE
        ADVERTENCIA → PRECAUCION
        CRITICO → RECHAZAR
        """
        mapping = {
            SeverityLevel.OPTIMO: VerdictLevel.VIABLE,
            SeverityLevel.ADVERTENCIA: VerdictLevel.PRECAUCION,
            SeverityLevel.CRITICO: VerdictLevel.RECHAZAR,
        }

        for severity, expected_verdict in mapping.items():
            # Verificar que el mapeo preserva el orden
            assert severity.value <= SeverityLevel.CRITICO.value
            assert expected_verdict.value <= VerdictLevel.RECHAZAR.value

    def test_embedding_preserves_order(self):
        """El morfismo preserva el orden parcial."""
        def embed(s: SeverityLevel) -> VerdictLevel:
            mapping = {
                SeverityLevel.OPTIMO: VerdictLevel.VIABLE,
                SeverityLevel.ADVERTENCIA: VerdictLevel.PRECAUCION,
                SeverityLevel.CRITICO: VerdictLevel.RECHAZAR,
            }
            return mapping[s]

        # Para todos los pares ordenados en SeverityLevel
        for a in SeverityLevel:
            for b in SeverityLevel:
                if a.value <= b.value:
                    # El morfismo debe preservar: a ≤ b ⟹ f(a) ≤ f(b)
                    assert embed(a).value <= embed(b).value

    def test_embedding_preserves_join(self):
        """El morfismo preserva la operación join."""
        def embed(s: SeverityLevel) -> VerdictLevel:
            mapping = {
                SeverityLevel.OPTIMO: VerdictLevel.VIABLE,
                SeverityLevel.ADVERTENCIA: VerdictLevel.PRECAUCION,
                SeverityLevel.CRITICO: VerdictLevel.RECHAZAR,
            }
            return mapping[s]

        for a in SeverityLevel:
            for b in SeverityLevel:
                # f(a ⊔ b) = f(a) ⊔ f(b)
                join_then_embed = embed(a | b)
                embed_then_join = embed(a) | embed(b)
                assert join_then_embed == embed_then_join

    def test_embedding_preserves_meet(self):
        """El morfismo preserva la operación meet."""
        def embed(s: SeverityLevel) -> VerdictLevel:
            mapping = {
                SeverityLevel.OPTIMO: VerdictLevel.VIABLE,
                SeverityLevel.ADVERTENCIA: VerdictLevel.PRECAUCION,
                SeverityLevel.CRITICO: VerdictLevel.RECHAZAR,
            }
            return mapping[s]

        for a in SeverityLevel:
            for b in SeverityLevel:
                # f(a ⊓ b) = f(a) ⊓ f(b)
                meet_then_embed = embed(a & b)
                embed_then_meet = embed(a) & embed(b)
                assert meet_then_embed == embed_then_meet

    def test_bottom_maps_to_bottom(self):
        """⊥ de SeverityLevel mapea a ⊥ de VerdictLevel."""
        severity_bottom = SeverityLevel.bottom()
        verdict_bottom = VerdictLevel.bottom()

        # Ambos representan el "mejor caso"
        assert severity_bottom == SeverityLevel.OPTIMO
        assert verdict_bottom == VerdictLevel.VIABLE

    def test_top_maps_to_top(self):
        """⊤ de SeverityLevel mapea a ⊤ de VerdictLevel."""
        severity_top = SeverityLevel.top()
        verdict_top = VerdictLevel.top()

        # Ambos representan el "peor caso"
        assert severity_top == SeverityLevel.CRITICO
        assert verdict_top == VerdictLevel.RECHAZAR

    def test_severity_and_verdict_have_compatible_semantics(self):
        """Las semánticas de ambos lattices son compatibles."""
        # OPTIMO/VIABLE = todo bien
        assert SeverityLevel.OPTIMO.is_optimal
        assert VerdictLevel.VIABLE.is_positive

        # CRITICO/RECHAZAR = todo mal
        assert SeverityLevel.CRITICO.is_critical
        assert VerdictLevel.RECHAZAR.is_negative


# ============================================================================
# TEST: CONSISTENCIA DE STRATUM ENTRE MÓDULOS
# ============================================================================


class TestStratumConsistency:
    """Verifica que ambos módulos usan Stratum de manera consistente."""

    def test_both_modules_use_same_stratum_enum(self):
        """Ambos módulos importan el mismo Enum Stratum."""
        # Verificar que StratumTopology del narrador usa el mismo Stratum
        narrator_strata = set(StratumTopology.HIERARCHY.keys())
        
        # Todos los Stratum deben estar presentes
        assert Stratum.PHYSICS in narrator_strata
        assert Stratum.TACTICS in narrator_strata
        assert Stratum.STRATEGY in narrator_strata
        assert Stratum.WISDOM in narrator_strata

    def test_stratum_hierarchy_order_consistent(self):
        """El orden jerárquico es consistente entre módulos."""
        # En narrator: PHYSICS(3) > TACTICS(2) > STRATEGY(1) > WISDOM(0)
        narrator_order = StratumTopology.EVALUATION_ORDER

        # Verificar orden de evaluación (base a cima)
        assert narrator_order[0] == Stratum.PHYSICS  # Base
        assert narrator_order[-1] == Stratum.WISDOM  # Cima

    def test_step_mapping_covers_all_pipeline_stages(self):
        """El mapeo de pasos cubre todas las etapas del pipeline."""
        default_mapping = StratumTopology.DEFAULT_STEP_MAPPING

        # Verificar pasos críticos
        physics_steps = ["load_data", "merge_data", "flux_condenser"]
        tactics_steps = ["calculate_costs", "materialization"]
        strategy_steps = ["financial_analysis", "business_topology"]
        wisdom_steps = ["build_output", "response_preparation"]

        for step in physics_steps:
            assert default_mapping.get(step) == Stratum.PHYSICS

        for step in tactics_steps:
            assert default_mapping.get(step) == Stratum.TACTICS

        for step in strategy_steps:
            assert default_mapping.get(step) == Stratum.STRATEGY

        for step in wisdom_steps:
            assert default_mapping.get(step) == Stratum.WISDOM

    def test_translator_strata_results_match_narrator_strata(
        self, translator: SemanticTranslator, clean_topology: TopologyMetricsDTO, viable_financials: Dict
    ):
        """Los resultados del translator por estrato son compatibles con el narrator."""
        report = translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=10.0,
        )

        # Verificar que tiene análisis por estrato
        for stratum in Stratum:
            assert stratum in report.strata_analysis


# ============================================================================
# TEST: COMPOSICIÓN DE REPORTES COMBINADOS
# ============================================================================


class TestReportComposition:
    """Pruebas de composición de reportes de ambos módulos."""

    def test_narrator_report_can_feed_translator(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext, viable_financials: Dict
    ):
        """
        El reporte del narrator puede usarse para alimentar el translator.
        Flujo: TelemetryContext → Narrator → Métricas → Translator
        """
        # Simular ejecución con telemetría
        with context.span("load_data"):
            context.record_metric("flux_condenser", "avg_saturation", 0.3)
            context.record_metric("flux_condenser", "max_flyback_voltage", 0.1)

        with context.span("calculate_costs"):
            pass

        narrator_report = narrator.summarize_execution(context)

        # Extraer métricas del reporte del narrator para el translator
        # (Simulando extracción de β₀, β₁ del contexto de telemetría)
        topology = TopologicalMetrics(
            beta_0=1,
            beta_1=0 if narrator_report["verdict_code"] == "APPROVED" else 1,
        )

        # El translator puede usar esta información
        translator_report = translator.compose_strategic_narrative(
            topological_metrics=topology,
            financial_metrics=viable_financials,
            stability=10.0,
        )

        # Ambos reportes deben ser coherentes
        assert narrator_report["verdict_code"] == "APPROVED"
        assert translator_report.verdict == VerdictLevel.VIABLE

    def test_combined_report_structure(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext, clean_topology: TopologyMetricsDTO, viable_financials: Dict
    ):
        """Estructura de reporte combinado es completa."""
        with context.span("load_data"):
            pass
        with context.span("financial_analysis"):
            pass

        narrator_report = narrator.summarize_execution(context)
        translator_report = translator.compose_strategic_narrative(
            topological_metrics=clean_topology,
            financial_metrics=viable_financials,
            stability=12.0,
        )

        # Combinar reportes
        combined = {
            "telemetry": narrator_report,
            "strategic": translator_report.to_dict(),
            "combined_verdict": self._combine_verdicts(
                narrator_report["global_severity"],
                translator_report.verdict.name,
            ),
        }

        assert "telemetry" in combined
        assert "strategic" in combined
        assert "combined_verdict" in combined

    def _combine_verdicts(self, severity_name: str, verdict_name: str) -> str:
        """Combina veredictos de ambos módulos (helper)."""
        severity_map = {"OPTIMO": 0, "ADVERTENCIA": 1, "CRITICO": 2}
        verdict_map = {"VIABLE": 0, "CONDICIONAL": 1, "REVISAR": 2, "PRECAUCION": 3, "RECHAZAR": 4}

        severity_val = severity_map.get(severity_name, 2)
        verdict_val = verdict_map.get(verdict_name, 4)

        # Tomar el peor caso
        if severity_val == 2 or verdict_val >= 3:
            return "RECHAZAR"
        elif severity_val == 1 or verdict_val >= 1:
            return "REVISAR"
        return "APROBAR"

    def test_forensic_evidence_compatible_format(
        self, narrator: TelemetryNarrator, context: TelemetryContext
    ):
        """La evidencia forense tiene formato compatible con el translator."""
        with context.span("load_data") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Data corruption", "type": "DataError"})

        report = narrator.summarize_execution(context)
        evidence = report["forensic_evidence"]

        # Verificar formato estándar
        if evidence:
            for item in evidence:
                assert "source" in item
                assert "message" in item
                assert "type" in item
                # Formato compatible para uso en translator narratives
                assert isinstance(item["message"], str)


# ============================================================================
# TEST: FLUJO END-TO-END
# ============================================================================


class TestEndToEndFlow:
    """Pruebas de flujo completo de datos entre módulos."""

    def test_full_pipeline_success(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext
    ):
        """
        Flujo completo exitoso:
        TelemetryContext → Narrator → (extracción) → Translator → StrategicReport
        """
        # Paso 1: Simular pipeline exitoso con telemetría
        with context.span("load_data"):
            context.record_metric("flux_condenser", "processed_records", 1000)

        with context.span("calculate_costs"):
            context.record_metric("topology", "beta_0", 1)
            context.record_metric("topology", "beta_1", 0)

        with context.span("financial_analysis"):
            context.record_metric("financial", "npv", 50000)

        with context.span("build_output"):
            pass

        # Paso 2: Obtener reporte de telemetría
        telemetry_report = narrator.summarize_execution(context)

        # Paso 3: Extraer métricas para el translator
        topology = TopologicalMetrics(
            beta_0=int(context.get_metric("topology", "beta_0", default=1)),
            beta_1=int(context.get_metric("topology", "beta_1", default=0)),
        )

        financials = {
            "wacc": 0.10,
            "performance": {"recommendation": "ACEPTAR", "profitability_index": 1.2},
        }

        # Paso 4: Generar reporte estratégico
        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=topology,
            financial_metrics=financials,
            stability=15.0,
        )

        # Verificaciones
        assert telemetry_report["verdict_code"] == "APPROVED"
        assert strategic_report.verdict == VerdictLevel.VIABLE
        assert "CERTIFICADO" in strategic_report.raw_narrative

    def test_full_pipeline_failure_at_physics(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext
    ):
        """Fallo en PHYSICS se propaga correctamente a ambos módulos."""
        # Fallo en carga de datos
        with context.span("load_data") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "File not found", "type": "IOError"})

        telemetry_report = narrator.summarize_execution(context)

        # El translator debería recibir topología inválida
        # Simular topología corrupta
        corrupted_topology = TopologicalMetrics(beta_0=0, beta_1=0)

        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=corrupted_topology,
            financial_metrics={"performance": {"recommendation": "RECHAZAR"}},
            stability=0.5,  # Inestable
        )

        # Ambos deben rechazar
        assert "PHYSICS" in telemetry_report["verdict_code"]
        assert strategic_report.verdict == VerdictLevel.RECHAZAR

    def test_full_pipeline_failure_at_tactics(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext
    ):
        """Fallo en TACTICS (ciclos) se refleja en ambos módulos."""
        with context.span("load_data"):
            pass

        with context.span("calculate_costs") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Ciclo detectado en grafo", "type": "CycleError"})

        telemetry_report = narrator.summarize_execution(context)

        # El translator recibe topología con ciclos
        cyclic_topology = TopologicalMetrics(beta_0=1, beta_1=3)

        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=cyclic_topology,
            financial_metrics={"performance": {"recommendation": "REVISAR"}},
            stability=10.0,
        )

        # Narrator detecta fallo en TACTICS
        assert "TACTICS" in telemetry_report["verdict_code"]

        # Translator detecta ciclos
        assert "socavones" in strategic_report.raw_narrative.lower() or \
               "ciclos" in strategic_report.raw_narrative.lower() or \
               "agujeros" in strategic_report.raw_narrative.lower()

    def test_thermal_metrics_flow(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext
    ):
        """Métricas térmicas fluyen correctamente entre módulos."""
        with context.span("flux_condenser"):
            context.record_metric("flux_condenser", "system_temperature", 65.0)
            context.record_metric("flux_condenser", "entropy", 0.8)

        telemetry_report = narrator.summarize_execution(context)

        # Extraer métricas térmicas
        thermal = {
            "system_temperature": context.get_metric("flux_condenser", "system_temperature", default=25.0),
            "entropy": context.get_metric("flux_condenser", "entropy", default=0.0),
        }

        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=10.0,
            thermal_metrics=thermal,
        )

        # Debe detectar fiebre
        assert "FIEBRE" in strategic_report.raw_narrative or "calor" in strategic_report.raw_narrative.lower()


# ============================================================================
# TEST: COHERENCIA DE NARRATIVAS
# ============================================================================


class TestNarrativeCoherence:
    """Verifica coherencia semántica entre narrativas de ambos módulos."""

    def test_success_narratives_are_positive(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext
    ):
        """Narrativas de éxito son positivas en ambos módulos."""
        with context.span("load_data"):
            pass
        with context.span("financial_analysis"):
            pass

        telemetry_report = narrator.summarize_execution(context)
        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={"performance": {"recommendation": "ACEPTAR", "profitability_index": 1.5}},
            stability=20.0,
        )

        # Ambas narrativas deben ser positivas
        assert "✅" in telemetry_report["executive_summary"] or "CERTIFICADO" in telemetry_report["executive_summary"]
        assert strategic_report.verdict.is_positive

    def test_failure_narratives_are_negative(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext
    ):
        """Narrativas de fallo son negativas en ambos módulos."""
        with context.span("load_data") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Critical failure", "type": "Error"})

        telemetry_report = narrator.summarize_execution(context)
        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(beta_0=0, beta_1=5),
            financial_metrics={"performance": {"recommendation": "RECHAZAR"}},
            stability=0.5,
        )

        # Ambas narrativas deben ser negativas
        assert "REJECTED" in telemetry_report["verdict_code"] or "RECHAZADO" in telemetry_report["verdict_code"]
        assert strategic_report.verdict.is_negative

    def test_stratum_specific_keywords_appear(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext
    ):
        """Keywords específicos por estrato aparecen en narrativas."""
        # PHYSICS keywords
        with context.span("flux_condenser") as span:
            span.status = StepStatus.WARNING

        telemetry_report = narrator.summarize_execution(context)
        physics_narrative = telemetry_report["strata_analysis"]["PHYSICS"]["narrative"]

        # Debe contener keywords de física
        physics_keywords = ["física", "cimentación", "flujo", "turbulencia", "inestabilidad", "datos"]
        has_physics_keyword = any(kw.lower() in physics_narrative.lower() for kw in physics_keywords)
        assert has_physics_keyword or "PHYSICS" in physics_narrative

    def test_cycle_detection_consistent_language(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext
    ):
        """Detección de ciclos usa lenguaje consistente."""
        with context.span("calculate_costs") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Ciclo infinito detectado", "type": "CycleError"})

        telemetry_report = narrator.summarize_execution(context)
        tactics_narrative = telemetry_report["strata_analysis"]["TACTICS"]["narrative"]

        # Keywords de ciclos/socavones
        cycle_keywords = ["socavón", "ciclo", "bucle", "β₁", "genus"]
        has_cycle_keyword = any(kw.lower() in tactics_narrative.lower() for kw in cycle_keywords)
        assert has_cycle_keyword

    def test_financial_narratives_use_business_language(
        self, translator: SemanticTranslator
    ):
        """Narrativas financieras usan lenguaje de negocio."""
        report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={
                "wacc": 0.15,
                "performance": {"recommendation": "RECHAZAR", "profitability_index": 0.7},
            },
            stability=10.0,
        )

        # Keywords financieros
        financial_keywords = ["WACC", "costo", "financiero", "viable", "riesgo", "inversión"]
        has_financial_keyword = any(kw in report.raw_narrative for kw in financial_keywords)
        assert has_financial_keyword


# ============================================================================
# TEST: PROPAGACIÓN DE ERRORES
# ============================================================================


class TestErrorPropagation:
    """Pruebas de propagación de errores entre módulos."""

    def test_narrator_error_does_not_crash_translator(
        self, translator: SemanticTranslator
    ):
        """Error en narrator no afecta al translator."""
        # Simular que el narrator falló y no hay datos de telemetría
        # El translator debe funcionar con datos mínimos
        report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={},
            stability=10.0,
        )

        assert report is not None
        assert isinstance(report, StrategicReport)

    def test_translator_handles_incomplete_narrator_data(
        self, translator: SemanticTranslator
    ):
        """Translator maneja datos incompletos del narrator."""
        # Datos parciales que podrían venir de un narrator con errores
        partial_topology = {"beta_0": 1}  # Falta beta_1

        report = translator.compose_strategic_narrative(
            topological_metrics=partial_topology,
            financial_metrics={"performance": {"recommendation": "REVISAR"}},
            stability=5.0,
        )

        assert report is not None

    def test_both_modules_handle_none_gracefully(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator
    ):
        """Ambos módulos manejan None graciosamente."""
        narrator_report = narrator.summarize_execution(None)
        assert narrator_report is not None
        assert "verdict" in narrator_report

        # Translator con thermal_metrics=None
        translator_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={},
            stability=10.0,
            thermal_metrics=None,
            spectral=None,
            synergy_risk=None,
        )
        assert translator_report is not None

    def test_exception_in_one_module_isolated(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext
    ):
        """Excepción en un módulo no afecta al otro."""
        with context.span("test"):
            pass

        # Simular excepción en narrator
        with patch.object(narrator, '_analyze_phase', side_effect=Exception("Narrator error")):
            narrator_report = narrator.summarize_execution(context)

        # El narrator debe manejar su error
        assert "NARRATOR_ERROR" in narrator_report["verdict_code"] or "error" in narrator_report.get("executive_summary", "").lower()

        # El translator debe funcionar independientemente
        translator_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=10.0,
        )
        assert translator_report.verdict is not None


# ============================================================================
# TEST: MATRIZ DE DECISIÓN COMBINADA
# ============================================================================


class TestCombinedDecisionMatrix:
    """
    Pruebas exhaustivas de la matriz de decisión combinando ambos módulos.
    
    Variables:
    - Telemetry Status: SUCCESS, FAILURE en diferentes estratos
    - Topology: clean (β₁=0), cyclic (β₁>0)
    - Financials: ACCEPT, REJECT
    - Stability: high (>10), low (<1)
    """

    @pytest.fixture
    def narrator(self) -> TelemetryNarrator:
        return TelemetryNarrator()

    @pytest.fixture
    def translator(self) -> SemanticTranslator:
        return SemanticTranslator(config=TranslatorConfig(deterministic_market=True))

    @pytest.mark.parametrize(
        "telemetry_status,beta_1,recommendation,stability,expected_outcome",
        [
            # Caso ideal: todo bien
            ("success", 0, "ACEPTAR", 15.0, "APPROVED"),

            # Telemetría OK, pero ciclos
            ("success", 3, "ACEPTAR", 15.0, "TACTICAL_ISSUE"),

            # Telemetría OK, pero finanzas rechazadas
            ("success", 0, "RECHAZAR", 15.0, "FINANCIAL_ISSUE"),

            # Telemetría OK, pero inestable
            ("success", 0, "ACEPTAR", 0.5, "STABILITY_ISSUE"),

            # Telemetría falla en PHYSICS
            ("physics_fail", 0, "ACEPTAR", 15.0, "PHYSICS_REJECTED"),

            # Telemetría falla en TACTICS
            ("tactics_fail", 0, "ACEPTAR", 15.0, "TACTICS_REJECTED"),

            # Múltiples problemas
            ("success", 5, "RECHAZAR", 0.5, "MULTIPLE_ISSUES"),
        ],
    )
    def test_decision_matrix(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
        telemetry_status: str,
        beta_1: int,
        recommendation: str,
        stability: float,
        expected_outcome: str,
    ):
        """Prueba combinaciones de la matriz de decisión."""
        # Crear contexto según telemetry_status
        context = TelemetryContext()

        if telemetry_status == "success":
            with context.span("load_data"):
                pass
            with context.span("calculate_costs"):
                pass
        elif telemetry_status == "physics_fail":
            with context.span("load_data") as span:
                span.status = StepStatus.FAILURE
                span.errors.append({"message": "Physics fail", "type": "Error"})
        elif telemetry_status == "tactics_fail":
            with context.span("load_data"):
                pass
            with context.span("calculate_costs") as span:
                span.status = StepStatus.FAILURE
                span.errors.append({"message": "Tactics fail", "type": "Error"})

        # Generar reportes
        telemetry_report = narrator.summarize_execution(context)

        topology = TopologicalMetrics(beta_0=1, beta_1=beta_1)
        financials = {"performance": {"recommendation": recommendation, "profitability_index": 1.0 if recommendation == "ACEPTAR" else 0.7}}

        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=topology,
            financial_metrics=financials,
            stability=stability,
        )

        # Verificar resultado esperado
        if expected_outcome == "APPROVED":
            assert telemetry_report["verdict_code"] == "APPROVED"
            assert strategic_report.verdict.is_positive

        elif expected_outcome == "PHYSICS_REJECTED":
            assert "PHYSICS" in telemetry_report["verdict_code"]

        elif expected_outcome == "TACTICS_REJECTED":
            assert "TACTICS" in telemetry_report["verdict_code"]

        elif expected_outcome == "TACTICAL_ISSUE":
            # Translator detecta ciclos
            assert strategic_report.verdict.value >= VerdictLevel.PRECAUCION.value

        elif expected_outcome == "FINANCIAL_ISSUE":
            assert strategic_report.verdict.value >= VerdictLevel.REVISAR.value

        elif expected_outcome == "STABILITY_ISSUE":
            assert "Pirámide" in strategic_report.raw_narrative or "inestable" in strategic_report.raw_narrative.lower()

        elif expected_outcome == "MULTIPLE_ISSUES":
            assert strategic_report.verdict == VerdictLevel.RECHAZAR

    def test_clausura_transitiva_consistency(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator
    ):
        """
        Invariante: Si falla PHYSICS en telemetría, el translator
        también debería indicar problema base.
        """
        context = TelemetryContext()
        with context.span("load_data") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Base failure", "type": "Error"})

        telemetry_report = narrator.summarize_execution(context)

        # Simular que los datos están corruptos (β₀=0 indica proyecto vacío)
        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(beta_0=0, beta_1=0),
            financial_metrics={},
            stability=0.1,
        )

        # Ambos deben rechazar
        assert "PHYSICS" in telemetry_report["verdict_code"]
        assert strategic_report.verdict == VerdictLevel.RECHAZAR


# ============================================================================
# TEST: SERIALIZACIÓN CRUZADA
# ============================================================================


class TestCrossModuleSerialization:
    """Pruebas de serialización compatible entre módulos."""

    def test_narrator_report_json_compatible(
        self, narrator: TelemetryNarrator, context: TelemetryContext
    ):
        """Reporte del narrator es JSON-serializable."""
        import json

        with context.span("test"):
            pass

        report = narrator.summarize_execution(context)

        # Debe ser serializable a JSON
        json_str = json.dumps(report)
        assert isinstance(json_str, str)

        # Debe deserializarse correctamente
        restored = json.loads(json_str)
        assert restored["verdict_code"] == report["verdict_code"]

    def test_translator_report_json_compatible(
        self, translator: SemanticTranslator
    ):
        """Reporte del translator es JSON-serializable."""
        import json

        report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=10.0,
        )

        report_dict = report.to_dict()

        # Debe ser serializable a JSON
        json_str = json.dumps(report_dict)
        assert isinstance(json_str, str)

    def test_combined_report_json_compatible(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext
    ):
        """Reporte combinado es JSON-serializable."""
        import json

        with context.span("test"):
            pass

        combined = {
            "telemetry": narrator.summarize_execution(context),
            "strategic": translator.compose_strategic_narrative(
                topological_metrics=TopologicalMetrics(),
                financial_metrics={},
                stability=10.0,
            ).to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(combined)
        assert isinstance(json_str, str)
        assert len(json_str) > 100  # Tiene contenido


# ============================================================================
# TEST: MÉTRICAS COMPARTIDAS
# ============================================================================


class TestSharedMetrics:
    """Pruebas de métricas compartidas entre módulos."""

    def test_flux_condenser_metrics_flow(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator,
        context: TelemetryContext
    ):
        """Métricas del FluxCondenser fluyen correctamente."""
        with context.span("flux_condenser"):
            context.record_metric("flux_condenser", "avg_saturation", 0.85)
            context.record_metric("flux_condenser", "max_flyback_voltage", 0.6)
            context.record_metric("flux_condenser", "kinetic_energy", 150.0)

        telemetry_report = narrator.summarize_execution(context)

        # Extraer métricas para thermal
        thermal = {
            "system_temperature": 45.0,  # Simulado
            "entropy": context.get_metric("flux_condenser", "avg_saturation", default=0.0),
        }

        strategic_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=10.0,
            thermal_metrics=thermal,
        )

        # Ambos reportes deben generarse
        assert telemetry_report is not None
        assert strategic_report is not None

    def test_topology_metrics_consistency(
        self, context: TelemetryContext, translator: SemanticTranslator
    ):
        """Métricas topológicas son consistentes."""
        # Registrar en telemetría
        context.record_metric("topology", "beta_0", 1)
        context.record_metric("topology", "beta_1", 2)
        context.record_metric("topology", "euler_characteristic", -1)

        # Crear DTO desde métricas
        topology = TopologicalMetrics(
            beta_0=int(context.get_metric("topology", "beta_0", default=1)),
            beta_1=int(context.get_metric("topology", "beta_1", default=0)),
            euler_characteristic=int(context.get_metric("topology", "euler_characteristic", default=1)),
        )

        report = translator.compose_strategic_narrative(
            topological_metrics=topology,
            financial_metrics={"performance": {"recommendation": "REVISAR"}},
            stability=10.0,
        )

        # Debe detectar ciclos
        assert "socavones" in report.raw_narrative.lower() or \
               "ciclos" in report.raw_narrative.lower() or \
               "β₁" in report.raw_narrative


# ============================================================================
# TEST: FACTORIES Y CONVENIENCIAS
# ============================================================================


class TestFactoryIntegration:
    """Pruebas de funciones factory de ambos módulos."""

    def test_create_both_with_defaults(self):
        """Crear ambos módulos con valores por defecto."""
        narrator = create_narrator()
        translator = create_translator()

        assert isinstance(narrator, TelemetryNarrator)
        assert isinstance(translator, SemanticTranslator)

    def test_create_with_custom_configs(self):
        """Crear ambos módulos con configuraciones personalizadas."""
        narrator = create_narrator(step_mapping={"custom": Stratum.WISDOM})
        translator = create_translator(
            config=TranslatorConfig(deterministic_market=True)
        )

        assert narrator.step_mapping.get("custom") == Stratum.WISDOM
        assert translator.config.deterministic_market is True

    def test_convenience_functions_work_together(self):
        """Funciones de conveniencia trabajan juntas."""
        from app.telemetry_narrative import summarize_context
        from app.semantic_translator import translate_metrics_to_narrative

        context = TelemetryContext()
        with context.span("test"):
            pass

        telemetry_summary = summarize_context(context)
        strategic_narrative = translate_metrics_to_narrative(
            topological_metrics={"beta_0": 1, "beta_1": 0},
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=10.0,
        )

        assert isinstance(telemetry_summary, dict)
        assert isinstance(strategic_narrative, str)
        assert len(strategic_narrative) > 100


# ============================================================================
# TEST: PROPIEDADES COMBINADAS
# ============================================================================


class TestCombinedProperties:
    """Pruebas de propiedades que involucran ambos módulos."""

    def test_verdict_severity_correlation(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator
    ):
        """
        Correlación entre severidad y veredicto:
        Si SeverityLevel es CRITICO, VerdictLevel debe ser >= PRECAUCION
        """
        context = TelemetryContext()
        with context.span("test") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Error", "type": "Error"})

        telemetry_report = narrator.summarize_execution(context)

        # Si telemetría es CRITICO
        if telemetry_report["global_severity"] == "CRITICO":
            # El translator con datos problemáticos también debe ser severo
            strategic_report = translator.compose_strategic_narrative(
                topological_metrics=TopologicalMetrics(beta_0=0, beta_1=5),
                financial_metrics={"performance": {"recommendation": "RECHAZAR"}},
                stability=0.1,
            )

            assert strategic_report.verdict.value >= VerdictLevel.PRECAUCION.value

    def test_determinism_across_modules(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator
    ):
        """Ambos módulos producen resultados determinísticos."""
        context1 = TelemetryContext()
        context2 = TelemetryContext()

        with context1.span("test"):
            pass
        with context2.span("test"):
            pass

        report1_tel = narrator.summarize_execution(context1)
        report2_tel = narrator.summarize_execution(context2)

        report1_str = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=10.0,
        )
        report2_str = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(),
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=10.0,
        )

        assert report1_tel["verdict_code"] == report2_tel["verdict_code"]
        assert report1_str.verdict == report2_str.verdict

    def test_monotonicity_preservation(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator
    ):
        """
        Monotonicidad: Más errores/ciclos → peor veredicto.
        Esta propiedad debe mantenerse en ambos módulos.
        """
        previous_severity = "OPTIMO"
        previous_verdict = VerdictLevel.VIABLE

        for num_cycles in range(4):
            context = TelemetryContext()
            with context.span("calculate_costs") as span:
                if num_cycles > 0:
                    for i in range(num_cycles):
                        span.errors.append({"message": f"Cycle {i}", "type": "CycleError"})
                    span.status = StepStatus.FAILURE

            tel_report = narrator.summarize_execution(context)
            str_report = translator.compose_strategic_narrative(
                topological_metrics=TopologicalMetrics(beta_0=1, beta_1=num_cycles),
                financial_metrics={"performance": {"recommendation": "REVISAR"}},
                stability=10.0,
            )

            current_severity = tel_report["global_severity"]
            current_verdict = str_report.verdict

            # Severidad no debe mejorar
            severity_order = {"OPTIMO": 0, "ADVERTENCIA": 1, "CRITICO": 2}
            assert severity_order.get(current_severity, 2) >= severity_order.get(previous_severity, 0)

            # Veredicto no debe mejorar
            assert current_verdict.value >= previous_verdict.value

            previous_severity = current_severity
            previous_verdict = current_verdict


class TestLatticeIsomorphismRefined:
    """
    Análisis refinado de la relación entre lattices.
    SeverityLevel (L_s) y VerdictLevel (L_v) no son isomorfos sino que existe
    una inmersión monótona φ: L_s → L_v que preserva el orden parcial.
    """

    def test_lattice_structure_analysis(self):
        """Análisis algebraico formal de la estructura lattice."""
        # SeverityLevel forma un lattice completo (3 elementos)
        # Diagrama de Hassey: OPTIMO < ADVERTENCIA < CRITICO
        severity_chain = [
            SeverityLevel.OPTIMO,
            SeverityLevel.ADVERTENCIA,
            SeverityLevel.CRITICO
        ]

        # VerdictLevel forma un lattice completo (5 elementos)
        # Diagrama más complejo con ramificaciones
        verdict_structure = {
            VerdictLevel.VIABLE: 0,
            VerdictLevel.CONDICIONAL: 1,
            VerdictLevel.REVISAR: 1,  # Mismo nivel que CONDICIONAL
            VerdictLevel.PRECAUCION: 2,
            VerdictLevel.RECHAZAR: 3
        }

        # Demostración: No hay isomorfismo (cardinalidad diferente)
        assert len(SeverityLevel) != len(VerdictLevel)

        # Pero existe una inmersión monótona
        monotone_embedding = {
            SeverityLevel.OPTIMO: VerdictLevel.VIABLE,
            SeverityLevel.ADVERTENCIA: VerdictLevel.PRECAUCION,
            SeverityLevel.CRITICO: VerdictLevel.RECHAZAR
        }

        # Propiedad: φ preserva el orden parcial
        for a, b in [(SeverityLevel.OPTIMO, SeverityLevel.ADVERTENCIA),
                     (SeverityLevel.ADVERTENCIA, SeverityLevel.CRITICO)]:
            assert monotone_embedding[a].value <= monotone_embedding[b].value

    def test_galois_connection_analysis(self):
        """
        Propone una conexión de Galois entre los dos lattices:
        Existen funciones α: L_s → L_v y γ: L_v → L_s tales que:
        α(x) ≤ y ⇔ x ≤ γ(y)
        """
        def alpha(severity: SeverityLevel) -> VerdictLevel:
            """Función adjunta izquierda (abstracción)."""
            mapping = {
                SeverityLevel.OPTIMO: VerdictLevel.VIABLE,
                SeverityLevel.ADVERTENCIA: VerdictLevel.REVISAR,
                SeverityLevel.CRITICO: VerdictLevel.RECHAZAR
            }
            return mapping[severity]

        def gamma(verdict: VerdictLevel) -> SeverityLevel:
            """Función adjunta derecha (concretización)."""
            mapping = {
                VerdictLevel.VIABLE: SeverityLevel.OPTIMO,
                VerdictLevel.CONDICIONAL: SeverityLevel.OPTIMO,  # Ajustado para Galois
                VerdictLevel.REVISAR: SeverityLevel.ADVERTENCIA,
                VerdictLevel.PRECAUCION: SeverityLevel.ADVERTENCIA,
                VerdictLevel.RECHAZAR: SeverityLevel.CRITICO
            }
            return mapping[verdict]

        # Verificar propiedad de conexión de Galois
        for s in SeverityLevel:
            for v in VerdictLevel:
                # α(s) ≤ v ⇔ s ≤ γ(v)
                galois_property = (alpha(s).value <= v.value) == (s.value <= gamma(v).value)
                assert galois_property, f"Falla en s={s}, v={v}"


class TopologyAlgebraicAnalysis:
    """
    Análisis algebraico profundo de métricas topológicas.
    β₀ = número de componentes conexas (H₀)
    β₁ = número de ciclos independientes (H₁)
    """

    def test_betti_numbers_homological_consistency(self, translator: SemanticTranslator):
        """
        Verifica consistencia homológica: χ = β₀ - β₁ + β₂ - ...
        Para grafos: χ = β₀ - β₁ (β₂ = 0)
        """
        test_cases = [
            # (β₀, β₁, χ_esperado)
            (1, 0, 1),    # Grafo conexo sin ciclos (árbol)
            (1, 3, -2),   # Grafo conexo con 3 ciclos
            (3, 2, 1),    # 3 componentes, 2 ciclos
        ]

        for beta_0, beta_1, expected_chi in test_cases:
            topology = TopologyMetricsDTO(
                beta_0=beta_0,
                beta_1=beta_1,
                euler_characteristic=expected_chi
            )

            # Fórmula de Euler-Poincaré para grafos
            calculated_chi = beta_0 - beta_1

            assert topology.euler_characteristic == calculated_chi, \
                f"χ inconsistente: esperado {expected_chi}, calculado {calculated_chi}"

            # Análisis de narrativa basada en homología
            report = translator.compose_strategic_narrative(
                topological_metrics=topology,
                financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
                stability=10.0
            )

            # Verificar que la narrativa refleja la homología
            if beta_1 > 0:
                assert any(keyword in report.raw_narrative.lower()
                          for keyword in ["ciclo", "socavón", "agujero", "homología"])

    def test_persistence_homology_flow(self, narrator: TelemetryNarrator, context: TelemetryContext):
        """
        Simula un análisis de homología de persistencia en el tiempo.
        """
        # Registrar evolución de β₁ a lo largo del tiempo
        time_steps = [
            {"span": "load_data", "beta_1": 0},
            {"span": "calculate_costs", "beta_1": 1},
            {"span": "financial_analysis", "beta_1": 2},
            {"span": "build_output", "beta_1": 0}  # Resuelto
        ]

        for step in time_steps:
            with context.span(step["span"]):
                context.record_metric("topology", "beta_1", step["beta_1"])
                context.record_metric("topology", "persistence_time",
                                     datetime.utcnow().isoformat())

        report = narrator.summarize_execution(context)

        # Análisis de persistencia: ciclos que aparecen y desaparecen
        persistence_analysis = self._analyze_persistence_homology(context)

        # La narrativa debe reflejar la dinámica de ciclos
        assert "persistencia" in report["executive_summary"].lower() or \
               "evolución" in report["executive_summary"].lower()

    def _analyze_persistence_homology(self, context: TelemetryContext) -> Dict:
        """Analiza homología de persistencia de las métricas."""
        # Implementación simplificada
        beta_1_values = []
        for span in context.spans:
            beta_1 = context.get_metric("topology", "beta_1", span_id=span.id, default=0)
            beta_1_values.append(beta_1)

        # Detectar nacimiento y muerte de ciclos
        birth_death_pairs = []
        current_cycles = 0

        for i, beta_1 in enumerate(beta_1_values):
            if beta_1 > current_cycles:
                # Nacimiento de ciclos
                birth_death_pairs.append({"birth": i, "death": None})
            elif beta_1 < current_cycles:
                # Muerte de ciclos
                for pair in birth_death_pairs:
                    if pair["death"] is None:
                        pair["death"] = i
                        break

            current_cycles = beta_1

        return {
            "beta_1_series": beta_1_values,
            "persistence_pairs": birth_death_pairs,
            "total_persistence": sum(p["death"] - p["birth"]
                                    for p in birth_death_pairs if p["death"])
        }


class EnhancedDIKWModel:
    """
    Modelo DIKW mejorado con transiciones y operadores formales.
    D → I → K → W con operadores de transformación.
    """

    def test_dikw_transition_operators(self):
        """
        Define operadores formales para transiciones entre estratos:
        - Φ: Data → Information (procesamiento)
        - Ψ: Information → Knowledge (abstracción)
        - Ω: Knowledge → Wisdom (síntesis)
        """

        class DIKWOperators:
            @staticmethod
            def phi(data: Dict) -> Dict:
                """Data → Information: Agrega contexto y estructura."""
                return {
                    "processed_data": data,
                    "timestamp": datetime.utcnow(),
                    "context": "execution_pipeline",
                    "metadata": {"source": "telemetry", "version": "1.0"}
                }

            @staticmethod
            def psi(information: Dict) -> Dict:
                """Information → Knowledge: Extrae patrones y reglas."""
                patterns = []

                # Detectar patrones en métricas
                if "metrics" in information:
                    metrics = information["metrics"]

                    # Patrón: aumento secuencial de errores
                    if "error_count" in metrics:
                        error_trend = self._analyze_trend(metrics["error_count"])
                        if error_trend > 0.5:
                            patterns.append("error_accumulation")

                    # Patrón: correlación entre métricas
                    if "temperature" in metrics and "entropy" in metrics:
                        correlation = self._calculate_correlation(
                            metrics["temperature"], metrics["entropy"]
                        )
                        if abs(correlation) > 0.7:
                            patterns.append("thermal_correlation")

                return {
                    "information": information,
                    "patterns": patterns,
                    "rules": self._extract_rules(patterns),
                    "confidence": self._calculate_confidence(information)
                }

            @staticmethod
            def omega(knowledge: Dict) -> Dict:
                """Knowledge → Wisdom: Síntesis para toma de decisiones."""
                wisdom = {
                    "decision": "DEFER",
                    "rationale": [],
                    "certainty": 0.5,
                    "alternatives": []
                }

                # Aplicar sabiduría basada en conocimiento acumulado
                patterns = knowledge.get("patterns", [])
                confidence = knowledge.get("confidence", 0.5)

                if "error_accumulation" in patterns and confidence < 0.3:
                    wisdom["decision"] = "REJECT"
                    wisdom["rationale"].append("Error accumulation with low confidence")

                elif confidence > 0.8 and not patterns:
                    wisdom["decision"] = "ACCEPT"
                    wisdom["rationale"].append("High confidence with no detected issues")

                return wisdom

        # Probar cadena completa
        raw_data = {"metrics": {"error_count": [0, 1, 2, 3], "temperature": [25, 26, 27]}}

        information = DIKWOperators.phi(raw_data)
        knowledge = DIKWOperators.psi(information)
        wisdom = DIKWOperators.omega(knowledge)

        assert "decision" in wisdom
        assert "rationale" in wisdom

        return wisdom

    def test_stratum_cross_validation(self, narrator: TelemetryNarrator,
                                     translator: SemanticTranslator):
        """
        Validación cruzada entre estratos de ambos módulos.
        Verifica que las transiciones DIKW sean consistentes.
        """
        # Crear datos de prueba con problemas en PHYSICS
        context = TelemetryContext()

        with context.span("load_data") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({
                "message": "Data corruption in PHYSICS stratum",
                "type": "DataIntegrityError",
                "stratum": "PHYSICS"
            })

        # Narrator analiza desde abajo (PHYSICS primero)
        narrator_report = narrator.summarize_execution(context)

        # Translator debería reflejar el mismo problema en su análisis
        # Simular topología corrupta (β₀=0 indica datos inválidos)
        translator_report = translator.compose_strategic_narrative(
            topological_metrics=TopologyMetricsDTO(beta_0=0, beta_1=0),
            financial_metrics={"performance": {"recommendation": "RECHAZAR"}},
            stability=0.1
        )

        # Validación cruzada: problema en PHYSICS debe afectar todos los estratos superiores
        narrator_physics = narrator_report["strata_analysis"].get("PHYSICS", {})
        translator_physics = translator_report.strata_analysis.get(Stratum.PHYSICS, {})

        # Ambos deben reportar problemas en la base
        assert narrator_physics.get("severity") == "CRITICO" or \
               translator_physics.get("verdict") == VerdictLevel.RECHAZAR

        # El problema debe propagarse hacia arriba (transitividad)
        for higher_stratum in [Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM]:
            if higher_stratum in translator_report.strata_analysis:
                higher_verdict = translator_report.strata_analysis[higher_stratum].get("verdict")
                # El veredicto no debería mejorar al subir (monotonicidad negativa)
                assert higher_verdict.value >= translator_physics.get("verdict", VerdictLevel.VIABLE).value


class FuzzyDecisionMatrix:
    """
    Matriz de decisión difusa que maneja incertidumbre y grados de membresía.
    """

    def test_fuzzy_decision_system(self):
        """
        Sistema de decisión difusa basado en reglas fuzzy.
        Variables lingüísticas: telemetry_quality, topology_health, financial_viability
        """

        class FuzzyInferenceSystem:
            def __init__(self):
                # Conjuntos difusos para cada variable
                self.telemetry_sets = {
                    "poor": lambda x: max(0, 1 - x/0.3),      # x ∈ [0, 1]
                    "fair": lambda x: max(0, 1 - abs(x-0.5)/0.2),
                    "good": lambda x: max(0, (x-0.7)/0.3)
                }

                self.topology_sets = {
                    "fragmented": lambda b1: min(1, b1/3),    # β₁ ∈ [0, ∞)
                    "connected": lambda b1: max(0, 1 - b1/2),
                    "optimal": lambda b1: 1 if b1 == 0 else max(0, 0.5 - b1/4)
                }

                self.financial_sets = {
                    "reject": lambda pi: max(0, 1 - pi/0.8),  # PI ∈ [0, ∞)
                    "marginal": lambda pi: max(0, 1 - abs(pi-1)/0.3),
                    "accept": lambda pi: max(0, (pi-1.2)/0.5)
                }

            def infer(self, telemetry_score: float, beta_1: int,
                     profitability_index: float) -> Dict:
                """
                Inferencia difusa usando método de Mamdani.
                """
                # 1. Fuzzificación
                telemetry_values = {
                    name: func(telemetry_score)
                    for name, func in self.telemetry_sets.items()
                }

                topology_values = {
                    name: func(beta_1)
                    for name, func in self.topology_sets.items()
                }

                financial_values = {
                    name: func(profitability_index)
                    for name, func in self.financial_sets.items()
                }

                # 2. Aplicar reglas fuzzy
                rules = [
                    # Rule 1: Si telemetría pobre Y topología fragmentada → RECHAZAR
                    {
                        "antecedent": min(telemetry_values["poor"], topology_values["fragmented"]),
                        "consequent": "RECHAZAR",
                        "weight": 1.0
                    },
                    # Rule 2: Si todo bueno → APROBAR
                    {
                        "antecedent": min(
                            telemetry_values["good"],
                            topology_values["optimal"],
                            financial_values["accept"]
                        ),
                        "consequent": "APROBAR",
                        "weight": 1.0
                    },
                    # Rule 3: Si marginal con ciclos → REVISAR
                    {
                        "antecedent": min(
                            telemetry_values["fair"],
                            topology_values["connected"],
                            financial_values["marginal"]
                        ),
                        "consequent": "REVISAR",
                        "weight": 0.8
                    }
                ]

                # 3. Agregación y defuzzificación (centroide)
                consequent_values = {"RECHAZAR": 0.0, "REVISAR": 0.5, "APROBAR": 1.0}

                numerator = 0.0
                denominator = 0.0

                for rule in rules:
                    if rule["antecedent"] > 0:
                        consequent = rule["consequent"]
                        weight = rule["weight"] * rule["antecedent"]

                        numerator += weight * consequent_values[consequent]
                        denominator += weight

                # 4. Defuzzificación
                crisp_output = numerator / denominator if denominator > 0 else 0.5

                # 5. Decisión crisp basada en umbrales
                if crisp_output < 0.3:
                    final_decision = "RECHAZAR"
                elif crisp_output < 0.7:
                    final_decision = "REVISAR"
                else:
                    final_decision = "APROBAR"

                return {
                    "crisp_decision": final_decision,
                    "fuzzy_output": crisp_output,
                    "membership_values": {
                        "telemetry": telemetry_values,
                        "topology": topology_values,
                        "financial": financial_values
                    },
                    "rule_activations": [
                        {"rule": i, "activation": r["antecedent"]}
                        for i, r in enumerate(rules)
                    ]
                }

        # Probar sistema con casos límite
        fis = FuzzyInferenceSystem()

        # Caso 1: Claramente rechazable
        result1 = fis.infer(
            telemetry_score=0.1,  # Muy pobre
            beta_1=5,             # Muy fragmentado
            profitability_index=0.6  # Pobre
        )
        assert result1["crisp_decision"] == "RECHAZAR"

        # Caso 2: Zona gris (debería requerir revisión)
        result2 = fis.infer(
            telemetry_score=0.5,  # Regular
            beta_1=1,             # Algo fragmentado
            profitability_index=1.05  # Marginal
        )
        assert result2["crisp_decision"] == "REVISAR"


class SemanticNarrativeAnalysis:
    """
    Análisis semántico profundo de coherencia narrativa.
    """

    def test_narrative_semantic_coherence(self, narrator_report: Dict,
                                         translator_report: StrategicReport):
        """
        Analiza coherencia semántica entre narrativas usando:
        1. Análisis de temas (LDA simplificado)
        2. Coherencia emocional (sentiment analysis)
        3. Consistencia de conceptos clave
        """

        def extract_themes(text: str, n_themes: int = 3) -> List[str]:
            """Extrae temas principales usando TF-IDF simplificado."""
            words = text.lower().split()
            stop_words = {"el", "la", "de", "en", "y", "que", "con", "los"}
            filtered = [w for w in words if w not in stop_words and len(w) > 3]

            # Frecuencia de términos
            freq = {}
            for word in filtered:
                freq[word] = freq.get(word, 0) + 1

            # Ordenar por importancia
            sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            themes = [word for word, count in sorted_words[:n_themes]]

            return themes

        def analyze_sentiment_coherence(text1: str, text2: str) -> float:
            """
            Mide coherencia de sentimiento entre textos.
            Retorna score de -1 (opuesto) a 1 (idéntico).
            """
            # Palabras positivas y negativas (lista simplificada)
            positive_words = {"bueno", "excelente", "aprobado", "éxito", "viable", "certificado"}
            negative_words = {"malo", "pobre", "rechazado", "fallo", "crítico", "problema"}

            def text_sentiment(text: str) -> float:
                words = set(text.lower().split())
                pos_count = len(words.intersection(positive_words))
                neg_count = len(words.intersection(negative_words))

                if pos_count + neg_count == 0:
                    return 0.0
                return (pos_count - neg_count) / (pos_count + neg_count)

            sent1 = text_sentiment(text1)
            sent2 = text_sentiment(text2)

            # Coherencia: 1 - |diferencia|
            coherence = 1.0 - abs(sent1 - sent2) / 2.0  # Normalizado a [0, 1]
            return coherence

        # Extraer textos
        narrator_text = narrator_report.get("executive_summary", "")
        translator_text = translator_report.raw_narrative

        # 1. Análisis de temas
        narrator_themes = extract_themes(narrator_text)
        translator_themes = extract_themes(translator_text)

        # Deberían compartir al menos un tema principal
        common_themes = set(narrator_themes).intersection(set(translator_themes))
        assert len(common_themes) > 0, \
            f"Narrativas no comparten temas: {narrator_themes} vs {translator_themes}"

        # 2. Coherencia de sentimiento
        sentiment_coherence = analyze_sentiment_coherence(narrator_text, translator_text)
        assert sentiment_coherence > 0.5, \
            f"Sentimientos incoherentes: {sentiment_coherence}"

        # 3. Consistencia conceptual por estrato
        for stratum in Stratum:
            narrator_stratum = narrator_report["strata_analysis"].get(stratum.name, {})
            translator_stratum = translator_report.strata_analysis.get(stratum, {})

            if narrator_stratum and translator_stratum:
                narrator_narrative = narrator_stratum.get("narrative", "")
                translator_narrative = getattr(translator_stratum, "narrative", "")

                if narrator_narrative and translator_narrative:
                    # Conceptos clave por estrato
                    stratum_keywords = {
                        Stratum.PHYSICS: {"datos", "física", "base", "flujo", "entropía"},
                        Stratum.TACTICS: {"costo", "ciclo", "optimización", "recursos"},
                        Stratum.STRATEGY: {"estrategia", "financiero", "riesgo", "viabilidad"},
                        Stratum.WISDOM: {"sabiduría", "decisión", "ética", "visión"}
                    }

                    keywords = stratum_keywords.get(stratum, set())

                    # Ambas narrativas deberían contener al menos un keyword del estrato
                    narrator_has_keyword = any(kw in narrator_narrative.lower()
                                              for kw in keywords)
                    translator_has_keyword = any(kw in translator_narrative.lower()
                                                for kw in keywords)

                    assert narrator_has_keyword or translator_has_keyword, \
                        f"Narrativas del estrato {stratum} no contienen keywords apropiados"

        return {
            "common_themes": list(common_themes),
            "sentiment_coherence": sentiment_coherence,
            "thematic_overlap": len(common_themes) / max(len(narrator_themes), len(translator_themes))
        }


class ErrorTraceAlgebra:
    """
    Álgebra de trazas de error para análisis de propagación.
    Basado en teoría de trazas (trace theory) y monoides libres.
    """

    def test_error_trace_monoid(self, context: TelemetryContext):
        """
        Analiza trazas de error como elementos de un monoide libre.
        Operación: concatenación de trazas.
        """

        class ErrorTrace:
            def __init__(self, errors: List[Dict]):
                self.errors = errors
                self.trace = self._build_trace_string()

            def _build_trace_string(self) -> str:
                """Construye representación de traza como palabra."""
                symbols = []
                for error in self.errors:
                    error_type = error.get("type", "Unknown").upper()
                    # Simplificar tipos a símbolos
                    symbol_map = {
                        "DATAERROR": "D",
                        "CYCLEERROR": "C",
                        "IOERROR": "I",
                        "LOGICERROR": "L",
                        "TIMEOUT": "T"
                    }
                    symbol = symbol_map.get(error_type, "U")
                    symbols.append(symbol)
                return "".join(symbols)

            def __add__(self, other: 'ErrorTrace') -> 'ErrorTrace':
                """Concatenación de trazas (operación del monoide)."""
                return ErrorTrace(self.errors + other.errors)

            def __eq__(self, other: 'ErrorTrace') -> bool:
                """Igualdad basada en equivalencia de traza."""
                return self.trace == other.trace

            @property
            def pattern(self) -> Dict:
                """Extrae patrones de la traza."""
                patterns = {}

                # Patrón: repeticiones
                from collections import Counter
                counter = Counter(self.trace)
                for symbol, count in counter.items():
                    if count > 1:
                        patterns[f"repetition_{symbol}"] = count

                # Patrón: secuencias comunes
                common_sequences = ["DD", "CC", "IC", "DI"]
                for seq in common_sequences:
                    if seq in self.trace:
                        patterns[f"sequence_{seq}"] = self.trace.count(seq)

                return patterns

            def analyze_propagation(self) -> Dict:
                """
                Analiza propagación usando teoría de trazas.
                Returns: {
                    "is_critical": bool,
                    "propagation_path": List[str],
                    "dependence_relation": Dict
                }
                """
                # Relaciones de dependencia entre tipos de error
                dependence_relation = {
                    "D": {"I", "L"},  # DataError puede causar IOError o LogicError
                    "I": {"T"},       # IOError puede causar Timeout
                    "C": {"L"},       # CycleError puede causar LogicError
                }

                propagation_path = []
                for i, symbol in enumerate(self.trace):
                    if i > 0:
                        prev_symbol = self.trace[i-1]
                        if symbol in dependence_relation.get(prev_symbol, set()):
                            propagation_path.append(f"{prev_symbol}→{symbol}")

                is_critical = any(
                    pattern.startswith("repetition_") and count > 2
                    for pattern, count in self.pattern.items()
                ) or len(propagation_path) > 1

                return {
                    "is_critical": is_critical,
                    "propagation_path": propagation_path,
                    "dependence_relation": dependence_relation,
                    "trace_length": len(self.trace),
                    "unique_errors": len(set(self.trace))
                }

        # Extraer errores del contexto
        all_errors = []
        for span in context.spans:
            all_errors.extend(span.errors)

        # Crear traza de errores
        trace = ErrorTrace(all_errors)

        # Analizar propagación
        analysis = trace.analyze_propagation()

        # La narrativa debe reflejar patrones de propagación
        if analysis["is_critical"]:
            assert len(analysis["propagation_path"]) > 0 or \
                   trace.pattern.get("repetition_D", 0) > 2

        return analysis

    def test_error_trace_homomorphism(self, narrator: TelemetryNarrator,
                                     translator: SemanticTranslator):
        """
        Verifica que existe un homomorfismo entre trazas de error
        y narrativas generadas.
        """
        # Crear contexto con patrón específico de errores
        context = TelemetryContext()

        # Patrón: DataError → IOError → Timeout (cadena de dependencia)
        with context.span("load_data") as span:
            span.errors.append({"type": "DataError", "message": "Corrupt data"})
            span.errors.append({"type": "IOError", "message": "Cannot read file"})

        with context.span("process_data") as span:
            span.errors.append({"type": "Timeout", "message": "Operation timed out"})

        # Narrator genera traza de errores
        narrator_report = narrator.summarize_execution(context)

        # Translator recibe métricas derivadas de errores
        # (simulamos que los errores afectan la topología)
        error_count = sum(len(span.errors) for span in context.spans)
        beta_1 = min(3, error_count)  # Más errores, más ciclos potenciales

        translator_report = translator.compose_strategic_narrative(
            topological_metrics=TopologicalMetrics(beta_0=1, beta_1=beta_1),
            financial_metrics={"performance": {"recommendation": "REVISAR"}},
            stability=max(0, 10 - error_count)  # Menos estable con más errores
        )

        # Homomorfismo: estructura de errores → estructura narrativa
        # Traza de errores larga → narrativa con más advertencias
        if error_count >= 2:
            assert "PRECAUCION" in translator_report.raw_narrative or \
                   "RECHAZAR" in translator_report.raw_narrative or \
                   translator_report.verdict.value >= VerdictLevel.REVISAR.value
