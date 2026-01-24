"""
Suite de Integración Piramidal: Validación de Clausura Transitiva DIKW
======================================================================

Objetivo: Garantizar que la incoherencia en un nivel inferior (N+1) 
vete o degrade automáticamente la validez del nivel superior (N).

Jerarquía de Prueba (Pirámide DIKW):
------------------------------------
    Level 0: WISDOM    - Veredicto Final (SemanticTranslator)
    Level 1: STRATEGY  - Viabilidad Financiera (FinancialEngine)
    Level 2: TACTICS   - Coherencia Topológica (BusinessTopology)
    Level 3: PHYSICS   - Estabilidad de Flujo (FluxCondenser)

Regla de Clausura Transitiva:
-----------------------------
Si el nivel N+1 falla, entonces el nivel N está comprometido.
Matemáticamente: ∀n: Fail(n+1) ⟹ Compromised(n)

Estructura de Tests:
- TestPhysicsVeto: Fallo en PHYSICS invalida todo
- TestTacticsVeto: Fallo en TACTICS invalida STRATEGY y WISDOM
- TestStrategyVeto: Fallo en STRATEGY invalida WISDOM
- TestCombinedFailures: Múltiples fallos simultáneos
- TestTransitiveClosureProperties: Propiedades algebraicas
- TestCoherenceMatrix: Matriz exhaustiva de combinaciones
- TestRiskChallengerIntegration: Auditoría adversarial (si disponible)
- TestNarrativeCoherence: Coherencia de narrativas generadas
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type
from unittest.mock import Mock, MagicMock, patch

import pytest

# ============================================================================
# IMPORTACIONES CORE (Siempre disponibles)
# ============================================================================

from app.schemas import Stratum
from app.telemetry import TelemetryContext, TelemetrySpan, StepStatus, TelemetryHealth
from app.telemetry_narrative import (
    TelemetryNarrator,
    SeverityLevel,
    PyramidalReport,
    StratumAnalysis,
    Issue,
    NarratorConfig,
    StratumTopology,
)
from app.semantic_translator import (
    SemanticTranslator,
    VerdictLevel,
    TopologyMetricsDTO,
    ThermalMetricsDTO,
    SpectralMetricsDTO,
    SynergyRiskDTO,
    StrategicReport,
    TranslatorConfig,
    FinancialVerdict,
)


# ============================================================================
# IMPORTACIONES OPCIONALES (Pueden no existir)
# ============================================================================

try:
    from app.business_agent import RiskChallenger, ConstructionRiskReport
    HAS_RISK_CHALLENGER = True
except ImportError:
    HAS_RISK_CHALLENGER = False
    RiskChallenger = None
    ConstructionRiskReport = None


# ============================================================================
# FIXTURES Y HELPERS
# ============================================================================


@pytest.fixture
def narrator() -> TelemetryNarrator:
    """Narrador piramidal configurado."""
    return TelemetryNarrator()


@pytest.fixture
def translator() -> SemanticTranslator:
    """Traductor semántico con configuración determinística."""
    config = TranslatorConfig(deterministic_market=True)
    return SemanticTranslator(config=config)


@pytest.fixture
def fresh_context() -> TelemetryContext:
    """Contexto de telemetría fresco."""
    return TelemetryContext()


@pytest.fixture
def context_with_physics_failure() -> TelemetryContext:
    """Contexto con fallo en PHYSICS."""
    context = TelemetryContext()
    
    with context.span("flux_condenser", stratum=Stratum.PHYSICS) as span:
        span.status = StepStatus.FAILURE
        context.record_metric("flux_condenser", "max_flyback_voltage", 50.0)
        context.record_metric("flux_condenser", "avg_saturation", 0.95)
        span.errors.append({
            "message": "Voltaje Flyback Crítico: Ruptura de inercia de datos",
            "type": "PhysicalInstability",
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    return context


@pytest.fixture
def context_with_tactics_failure() -> TelemetryContext:
    """Contexto con PHYSICS OK pero fallo en TACTICS."""
    context = TelemetryContext()
    
    # PHYSICS OK
    with context.span("flux_condenser", stratum=Stratum.PHYSICS) as span:
        context.record_metric("flux_condenser", "max_flyback_voltage", 0.1)
        context.record_metric("flux_condenser", "avg_saturation", 0.3)
    
    # TACTICS FAIL
    with context.span("calculate_costs", stratum=Stratum.TACTICS) as span:
        span.status = StepStatus.FAILURE
        context.record_metric("topology", "beta_1", 5)  # Ciclos
        span.errors.append({
            "message": "Ciclos de dependencia detectados (β₁=5)",
            "type": "TopologyError",
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    return context


@pytest.fixture
def context_with_strategy_failure() -> TelemetryContext:
    """Contexto con PHYSICS y TACTICS OK, pero fallo en STRATEGY."""
    context = TelemetryContext()
    
    # PHYSICS OK
    with context.span("flux_condenser", stratum=Stratum.PHYSICS):
        context.record_metric("flux_condenser", "max_flyback_voltage", 0.1)
    
    # TACTICS OK
    with context.span("calculate_costs", stratum=Stratum.TACTICS):
        context.record_metric("topology", "beta_0", 1)
        context.record_metric("topology", "beta_1", 0)
    
    # STRATEGY FAIL
    with context.span("financial_analysis", stratum=Stratum.STRATEGY) as span:
        span.status = StepStatus.FAILURE
        context.record_metric("financial", "npv", -50000)  # NPV negativo
        span.errors.append({
            "message": "VaR excede umbral crítico",
            "type": "FinancialRiskError",
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    return context


@pytest.fixture
def context_all_success() -> TelemetryContext:
    """Contexto con todos los estratos exitosos."""
    context = TelemetryContext()
    
    with context.span("flux_condenser", stratum=Stratum.PHYSICS):
        context.record_metric("flux_condenser", "max_flyback_voltage", 0.05)
        context.record_metric("flux_condenser", "avg_saturation", 0.3)
    
    with context.span("calculate_costs", stratum=Stratum.TACTICS):
        context.record_metric("topology", "beta_0", 1)
        context.record_metric("topology", "beta_1", 0)
    
    with context.span("financial_analysis", stratum=Stratum.STRATEGY):
        context.record_metric("financial", "npv", 100000)
        context.record_metric("financial", "roi", 0.25)
    
    with context.span("build_output", stratum=Stratum.WISDOM):
        pass
    
    return context


@pytest.fixture
def clean_topology() -> TopologyMetricsDTO:
    """Topología limpia (sin ciclos, conexa)."""
    return TopologyMetricsDTO(beta_0=1, beta_1=0, euler_characteristic=1)


@pytest.fixture
def cyclic_topology() -> TopologyMetricsDTO:
    """Topología con ciclos (Socavones Lógicos)."""
    return TopologyMetricsDTO(beta_0=1, beta_1=5, euler_characteristic=-4, euler_efficiency=0.2)


@pytest.fixture
def fragmented_topology() -> TopologyMetricsDTO:
    """Topología fragmentada."""
    return TopologyMetricsDTO(beta_0=5, beta_1=0, euler_characteristic=5)


@pytest.fixture
def viable_financials() -> Dict[str, Any]:
    """Métricas financieras viables."""
    return {
        "wacc": 0.08,
        "contingency": {"recommended": 5000.0},
        "performance": {
            "recommendation": "ACEPTAR",
            "profitability_index": 1.5,
        },
    }


@pytest.fixture
def rejected_financials() -> Dict[str, Any]:
    """Métricas financieras de rechazo."""
    return {
        "wacc": 0.20,
        "contingency": {"recommended": 100000.0},
        "performance": {
            "recommendation": "RECHAZAR",
            "profitability_index": 0.6,
        },
    }


def assert_contains_any(text: str, keywords: List[str], msg: str = "") -> None:
    """Helper: Verifica que el texto contiene al menos una keyword."""
    found = any(kw.lower() in text.lower() for kw in keywords)
    if not found:
        raise AssertionError(
            f"{msg}\nExpected any of {keywords} in:\n{text[:500]}..."
        )


def get_stratum_from_verdict_code(verdict_code: str) -> Optional[str]:
    """Extrae el nombre del estrato del código de veredicto."""
    for stratum in Stratum:
        if stratum.name in verdict_code:
            return stratum.name
    return None


# ============================================================================
# TEST: VETO FÍSICO (PHYSICS invalida todo)
# ============================================================================


class TestPhysicsVeto:
    """
    Pruebas de que un fallo en PHYSICS invalida todos los niveles superiores.
    
    Regla: Fail(PHYSICS) ⟹ Compromised(TACTICS) ∧ Compromised(STRATEGY) ∧ Compromised(WISDOM)
    """

    def test_physics_failure_rejects_verdict(
        self,
        narrator: TelemetryNarrator,
        context_with_physics_failure: TelemetryContext,
    ):
        """Fallo en PHYSICS produce veredicto de rechazo técnico."""
        report = narrator.summarize_execution(context_with_physics_failure)
        
        # Debe ser rechazo relacionado con PHYSICS
        assert "PHYSICS" in report["verdict_code"] or "REJECTED" in report["verdict_code"]
        assert report["global_severity"] == "CRITICO"

    def test_physics_failure_ignores_good_financials(
        self,
        narrator: TelemetryNarrator,
        fresh_context: TelemetryContext,
    ):
        """Fallo en PHYSICS ignora métricas financieras positivas."""
        # PHYSICS falla
        with fresh_context.span("flux_condenser", stratum=Stratum.PHYSICS) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Critical instability", "type": "Error"})
        
        # STRATEGY reporta ganancias (alucinación)
        with fresh_context.span("financial_analysis", stratum=Stratum.STRATEGY):
            fresh_context.record_metric("financial", "roi", 0.50)  # 50% ROI
            fresh_context.record_metric("financial", "npv", 1000000)
        
        report = narrator.summarize_execution(fresh_context)
        
        # Veredicto debe ser rechazo por PHYSICS
        assert "PHYSICS" in report["verdict_code"]
        
        # La narrativa NO debe celebrar el ROI
        summary = report["executive_summary"]
        assert_contains_any(
            summary,
            ["FÍSICA", "PHYSICS", "ABORTADO", "INESTABILIDAD", "BASE"],
            "La narrativa debe mencionar el problema físico"
        )

    def test_physics_failure_root_cause_is_physics(
        self,
        narrator: TelemetryNarrator,
        context_with_physics_failure: TelemetryContext,
    ):
        """El root_cause_stratum es PHYSICS cuando PHYSICS falla."""
        report = narrator.summarize_execution(context_with_physics_failure)
        
        # Verificar root cause (puede estar en diferentes lugares)
        root_cause = report.get("root_cause_stratum")
        if root_cause:
            if hasattr(root_cause, 'name'):
                assert root_cause.name == "PHYSICS"
            else:
                assert root_cause == "PHYSICS"

    def test_physics_failure_forensic_evidence_points_to_physics(
        self,
        narrator: TelemetryNarrator,
        context_with_physics_failure: TelemetryContext,
    ):
        """La evidencia forense apunta a PHYSICS."""
        report = narrator.summarize_execution(context_with_physics_failure)
        
        evidence = report.get("forensic_evidence", [])
        if evidence:
            # Al menos un item de evidencia debe mencionar PHYSICS
            physics_evidence = [
                e for e in evidence
                if e.get("stratum") == "PHYSICS" or 
                   e.get("context", {}).get("stratum") == "PHYSICS" or
                   "flux" in e.get("source", "").lower() or
                   "physics" in str(e).lower()
            ]
            assert len(physics_evidence) > 0

    def test_physics_failure_propagates_warnings_to_upper_strata(
        self,
        fresh_context: TelemetryContext,
    ):
        """Fallo crítico en PHYSICS propaga warnings a estratos superiores."""
        fresh_context.start_step("physics_crash", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="physics_crash",
            error_message="Critical data corruption",
            severity="CRITICAL",
            stratum=Stratum.PHYSICS
        )
        
        # Verificar propagación
        assert len(fresh_context._strata_health[Stratum.TACTICS].warnings) > 0
        assert len(fresh_context._strata_health[Stratum.STRATEGY].warnings) > 0
        assert len(fresh_context._strata_health[Stratum.WISDOM].warnings) > 0


# ============================================================================
# TEST: VETO TÁCTICO (TACTICS invalida STRATEGY y WISDOM)
# ============================================================================


class TestTacticsVeto:
    """
    Pruebas de que un fallo en TACTICS invalida STRATEGY y WISDOM.
    
    Regla: Fail(TACTICS) ⟹ Compromised(STRATEGY) ∧ Compromised(WISDOM)
           pero NOT Compromised(PHYSICS)
    """

    def test_tactics_failure_rejects_verdict(
        self,
        narrator: TelemetryNarrator,
        context_with_tactics_failure: TelemetryContext,
    ):
        """Fallo en TACTICS produce veredicto de veto estructural."""
        report = narrator.summarize_execution(context_with_tactics_failure)
        
        assert "TACTICS" in report["verdict_code"]
        assert report["strata_analysis"]["PHYSICS"]["severity"] == "OPTIMO"
        assert report["strata_analysis"]["TACTICS"]["severity"] == "CRITICO"

    def test_tactics_cycles_invalidate_financials(
        self,
        translator: SemanticTranslator,
        cyclic_topology: TopologyMetricsDTO,
        viable_financials: Dict[str, Any],
    ):
        """Ciclos topológicos invalidan análisis financiero positivo."""
        report = translator.compose_strategic_narrative(
            topological_metrics=cyclic_topology,
            financial_metrics=viable_financials,
            stability=15.0,  # Estabilidad buena
        )
        
        # Veredicto debe ser precaución o rechazo
        assert report.verdict.value >= VerdictLevel.PRECAUCION.value
        
        # Narrativa debe mencionar ciclos/socavones
        assert_contains_any(
            report.raw_narrative,
            ["Socavón", "Ciclo", "β₁", "Genus", "agujero", "bucle"],
            "La narrativa debe mencionar los ciclos"
        )

    def test_tactics_failure_does_not_affect_physics(
        self,
        narrator: TelemetryNarrator,
        context_with_tactics_failure: TelemetryContext,
    ):
        """Fallo en TACTICS no afecta el estado de PHYSICS."""
        report = narrator.summarize_execution(context_with_tactics_failure)
        
        # PHYSICS debe estar OK
        assert report["strata_analysis"]["PHYSICS"]["severity"] == "OPTIMO"
        assert report["strata_analysis"]["PHYSICS"]["is_compromised"] is False

    def test_tactics_failure_propagates_to_strategy_and_wisdom(
        self,
        fresh_context: TelemetryContext,
    ):
        """Fallo crítico en TACTICS propaga a STRATEGY y WISDOM, no a PHYSICS."""
        fresh_context.start_step("topology_fail", stratum=Stratum.TACTICS)
        fresh_context.record_error(
            step_name="topology_fail",
            error_message="Infinite cycle detected",
            severity="CRITICAL",
            stratum=Stratum.TACTICS
        )
        
        # STRATEGY y WISDOM deben tener warnings
        assert len(fresh_context._strata_health[Stratum.STRATEGY].warnings) > 0
        assert len(fresh_context._strata_health[Stratum.WISDOM].warnings) > 0
        
        # PHYSICS NO debe tener warnings
        assert len(fresh_context._strata_health[Stratum.PHYSICS].warnings) == 0


# ============================================================================
# TEST: VETO ESTRATÉGICO (STRATEGY invalida WISDOM)
# ============================================================================


class TestStrategyVeto:
    """
    Pruebas de que un fallo en STRATEGY invalida WISDOM.
    
    Regla: Fail(STRATEGY) ⟹ Compromised(WISDOM)
           pero NOT Compromised(TACTICS) ∧ NOT Compromised(PHYSICS)
    """

    def test_strategy_failure_rejects_verdict(
        self,
        narrator: TelemetryNarrator,
        context_with_strategy_failure: TelemetryContext,
    ):
        """Fallo en STRATEGY produce veredicto de riesgo financiero."""
        report = narrator.summarize_execution(context_with_strategy_failure)
        
        assert "STRATEGY" in report["verdict_code"]
        assert report["strata_analysis"]["PHYSICS"]["severity"] == "OPTIMO"
        assert report["strata_analysis"]["TACTICS"]["severity"] == "OPTIMO"
        assert report["strata_analysis"]["STRATEGY"]["severity"] == "CRITICO"

    def test_strategy_failure_mentions_financial_risk(
        self,
        narrator: TelemetryNarrator,
        context_with_strategy_failure: TelemetryContext,
    ):
        """La narrativa menciona el riesgo financiero."""
        report = narrator.summarize_execution(context_with_strategy_failure)
        
        assert_contains_any(
            report["executive_summary"],
            ["FINANCIERO", "ORÁCULO", "RIESGO", "VaR", "NPV", "mercado"],
            "La narrativa debe mencionar el problema financiero"
        )

    def test_strategy_failure_does_not_affect_lower_strata(
        self,
        fresh_context: TelemetryContext,
    ):
        """Fallo en STRATEGY no afecta PHYSICS ni TACTICS."""
        fresh_context.start_step("financial_fail", stratum=Stratum.STRATEGY)
        fresh_context.record_error(
            step_name="financial_fail",
            error_message="NPV negative",
            severity="CRITICAL",
            stratum=Stratum.STRATEGY
        )
        
        # PHYSICS y TACTICS NO deben tener warnings
        assert len(fresh_context._strata_health[Stratum.PHYSICS].warnings) == 0
        assert len(fresh_context._strata_health[Stratum.TACTICS].warnings) == 0
        
        # Solo WISDOM debe tener warnings
        assert len(fresh_context._strata_health[Stratum.WISDOM].warnings) > 0


# ============================================================================
# TEST: FALLOS COMBINADOS
# ============================================================================


class TestCombinedFailures:
    """
    Pruebas de múltiples fallos simultáneos.
    
    Regla: El fallo en el nivel más bajo determina el veredicto principal.
    """

    def test_physics_and_strategy_failure_reports_physics(
        self,
        narrator: TelemetryNarrator,
        fresh_context: TelemetryContext,
    ):
        """Si fallan PHYSICS y STRATEGY, se reporta PHYSICS (más bajo)."""
        # PHYSICS falla
        with fresh_context.span("flux_condenser", stratum=Stratum.PHYSICS) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Physics fail", "type": "Error"})
        
        # STRATEGY también falla
        with fresh_context.span("financial_analysis", stratum=Stratum.STRATEGY) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Strategy fail", "type": "Error"})
        
        report = narrator.summarize_execution(fresh_context)
        
        # El veredicto debe ser por PHYSICS (causa raíz)
        assert "PHYSICS" in report["verdict_code"]

    def test_all_strata_fail_reports_physics(
        self,
        narrator: TelemetryNarrator,
        fresh_context: TelemetryContext,
    ):
        """Si todos los estratos fallan, se reporta PHYSICS."""
        for stratum, step_name in [
            (Stratum.PHYSICS, "physics_fail"),
            (Stratum.TACTICS, "tactics_fail"),
            (Stratum.STRATEGY, "strategy_fail"),
            (Stratum.WISDOM, "wisdom_fail"),
        ]:
            with fresh_context.span(step_name, stratum=stratum) as span:
                span.status = StepStatus.FAILURE
                span.errors.append({"message": f"{stratum.name} fail", "type": "Error"})
        
        report = narrator.summarize_execution(fresh_context)
        
        # El veredicto debe ser por PHYSICS
        assert "PHYSICS" in report["verdict_code"]

    def test_forensic_evidence_only_from_root_cause(
        self,
        narrator: TelemetryNarrator,
        fresh_context: TelemetryContext,
    ):
        """La evidencia forense solo viene del estrato de causa raíz."""
        # PHYSICS falla
        with fresh_context.span("physics_step", stratum=Stratum.PHYSICS) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Physics error message", "type": "Error"})
        
        # STRATEGY también falla
        with fresh_context.span("strategy_step", stratum=Stratum.STRATEGY) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Strategy error message", "type": "Error"})
        
        report = narrator.summarize_execution(fresh_context)
        
        evidence = report.get("forensic_evidence", [])
        evidence_messages = [e.get("message", "") for e in evidence]
        
        # Solo debe aparecer evidencia de PHYSICS
        assert any("Physics" in msg for msg in evidence_messages)
        # NO debe aparecer evidencia de STRATEGY
        assert not any("Strategy" in msg for msg in evidence_messages)


# ============================================================================
# TEST: PROPIEDADES DE CLAUSURA TRANSITIVA
# ============================================================================


class TestTransitiveClosureProperties:
    """
    Pruebas de las propiedades algebraicas de la clausura transitiva.
    """

    def test_transitivity_physics_to_wisdom(
        self,
        fresh_context: TelemetryContext,
    ):
        """Transitividad: Fail(PHYSICS) ⟹ Warning(WISDOM)."""
        fresh_context.record_error(
            step_name="physics",
            error_message="Critical",
            severity="CRITICAL",
            stratum=Stratum.PHYSICS
        )
        
        # La advertencia debe llegar hasta WISDOM
        wisdom_warnings = fresh_context._strata_health[Stratum.WISDOM].warnings
        assert len(wisdom_warnings) > 0

    def test_non_transitivity_upward(
        self,
        fresh_context: TelemetryContext,
    ):
        """No hay propagación hacia abajo: Fail(STRATEGY) ⟹ NOT Warning(PHYSICS)."""
        fresh_context.record_error(
            step_name="strategy",
            error_message="Critical",
            severity="CRITICAL",
            stratum=Stratum.STRATEGY
        )
        
        # PHYSICS NO debe tener warnings
        physics_warnings = fresh_context._strata_health[Stratum.PHYSICS].warnings
        assert len(physics_warnings) == 0

    @pytest.mark.parametrize(
        "failing_stratum,affected_strata,unaffected_strata",
        [
            (Stratum.PHYSICS, [Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM], []),
            (Stratum.TACTICS, [Stratum.STRATEGY, Stratum.WISDOM], [Stratum.PHYSICS]),
            (Stratum.STRATEGY, [Stratum.WISDOM], [Stratum.PHYSICS, Stratum.TACTICS]),
            (Stratum.WISDOM, [], [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY]),
        ],
    )
    def test_propagation_pattern(
        self,
        failing_stratum: Stratum,
        affected_strata: List[Stratum],
        unaffected_strata: List[Stratum],
    ):
        """Verifica el patrón de propagación para cada estrato."""
        context = TelemetryContext()
        
        context.record_error(
            step_name="test",
            error_message="Critical error",
            severity="CRITICAL",
            stratum=failing_stratum
        )
        
        # Estratos afectados deben tener warnings
        for stratum in affected_strata:
            assert len(context._strata_health[stratum].warnings) > 0, \
                f"{stratum.name} should have warnings from {failing_stratum.name}"
        
        # Estratos no afectados NO deben tener warnings
        for stratum in unaffected_strata:
            assert len(context._strata_health[stratum].warnings) == 0, \
                f"{stratum.name} should NOT have warnings from {failing_stratum.name}"

    def test_propagation_is_monotonic(self):
        """La propagación es monótona: más fallos no reducen warnings."""
        context = TelemetryContext()
        
        # Primer fallo
        context.record_error("f1", "Error 1", severity="CRITICAL", stratum=Stratum.PHYSICS)
        warnings_after_1 = len(context._strata_health[Stratum.TACTICS].warnings)
        
        # Segundo fallo
        context.record_error("f2", "Error 2", severity="CRITICAL", stratum=Stratum.PHYSICS)
        warnings_after_2 = len(context._strata_health[Stratum.TACTICS].warnings)
        
        # Tercer fallo
        context.record_error("f3", "Error 3", severity="CRITICAL", stratum=Stratum.PHYSICS)
        warnings_after_3 = len(context._strata_health[Stratum.TACTICS].warnings)
        
        assert warnings_after_1 <= warnings_after_2 <= warnings_after_3


# ============================================================================
# TEST: MATRIZ DE COHERENCIA EXHAUSTIVA
# ============================================================================


class TestCoherenceMatrix:
    """
    Matriz exhaustiva de combinaciones de estados por estrato.
    """

    @pytest.mark.parametrize(
        "physics_ok,tactics_ok,strategy_ok,expected_verdict_contains",
        [
            # Caso ideal
            (True, True, True, "APPROVED"),
            
            # Un solo fallo
            (False, True, True, "PHYSICS"),
            (True, False, True, "TACTICS"),
            (True, True, False, "STRATEGY"),
            
            # Dos fallos - siempre el más bajo
            (False, False, True, "PHYSICS"),
            (False, True, False, "PHYSICS"),
            (True, False, False, "TACTICS"),
            
            # Todos fallan
            (False, False, False, "PHYSICS"),
        ],
    )
    def test_verdict_matrix(
        self,
        narrator: TelemetryNarrator,
        physics_ok: bool,
        tactics_ok: bool,
        strategy_ok: bool,
        expected_verdict_contains: str,
    ):
        """Matriz de veredictos según combinación de estados."""
        context = TelemetryContext()
        
        # PHYSICS
        with context.span("physics", stratum=Stratum.PHYSICS) as span:
            if not physics_ok:
                span.status = StepStatus.FAILURE
                span.errors.append({"message": "Physics fail", "type": "Error"})
        
        # TACTICS
        with context.span("tactics", stratum=Stratum.TACTICS) as span:
            if not tactics_ok:
                span.status = StepStatus.FAILURE
                span.errors.append({"message": "Tactics fail", "type": "Error"})
        
        # STRATEGY
        with context.span("strategy", stratum=Stratum.STRATEGY) as span:
            if not strategy_ok:
                span.status = StepStatus.FAILURE
                span.errors.append({"message": "Strategy fail", "type": "Error"})
        
        report = narrator.summarize_execution(context)
        
        assert expected_verdict_contains in report["verdict_code"], \
            f"Expected '{expected_verdict_contains}' in verdict, got '{report['verdict_code']}'"

    @pytest.mark.parametrize(
        "beta_1,stability,recommendation,expected_verdict_min",
        [
            # Caso ideal
            (0, 15.0, "ACEPTAR", VerdictLevel.VIABLE),
            
            # Solo ciclos
            (3, 15.0, "ACEPTAR", VerdictLevel.PRECAUCION),
            (5, 15.0, "ACEPTAR", VerdictLevel.RECHAZAR),
            
            # Solo inestabilidad
            (0, 0.5, "ACEPTAR", VerdictLevel.PRECAUCION),
            
            # Solo rechazo financiero
            (0, 15.0, "RECHAZAR", VerdictLevel.RECHAZAR),
            
            # Combinaciones
            (3, 0.5, "ACEPTAR", VerdictLevel.RECHAZAR),
            (0, 0.5, "RECHAZAR", VerdictLevel.RECHAZAR),
        ],
    )
    def test_translator_verdict_matrix(
        self,
        translator: SemanticTranslator,
        beta_1: int,
        stability: float,
        recommendation: str,
        expected_verdict_min: VerdictLevel,
    ):
        """Matriz de veredictos del translator."""
        topology = TopologyMetricsDTO(beta_0=1, beta_1=beta_1)
        financials = {"performance": {"recommendation": recommendation, "profitability_index": 1.0}}
        
        report = translator.compose_strategic_narrative(
            topological_metrics=topology,
            financial_metrics=financials,
            stability=stability,
        )
        
        assert report.verdict.value >= expected_verdict_min.value, \
            f"Expected verdict >= {expected_verdict_min.name}, got {report.verdict.name}"


# ============================================================================
# TEST: INTEGRACIÓN CON RISK CHALLENGER (OPCIONAL)
# ============================================================================


@pytest.mark.skipif(not HAS_RISK_CHALLENGER, reason="RiskChallenger not available")
class TestRiskChallengerIntegration:
    """
    Pruebas de integración con el RiskChallenger.
    
    El Challenger detecta contradicciones entre métricas y veredictos.
    """

    def test_challenger_detects_inverted_pyramid(self):
        """El Challenger detecta la 'Pirámide Invertida'."""
        challenger = RiskChallenger()
        
        # Reporte "optimista" pero estructuralmente frágil
        naive_report = ConstructionRiskReport(
            integrity_score=95.0,
            financial_risk_level="BAJO",
            details={
                "topological_invariants": {
                    "pyramid_stability": 0.45  # Ψ < 1.0 (CRÍTICO)
                }
            },
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Baja"
        )
        
        # El Challenger audita
        audited_report = challenger.challenge_verdict(naive_report)
        
        # El riesgo debe haber sido degradado
        assert audited_report.financial_risk_level != "BAJO"
        assert audited_report.integrity_score < 95.0

    def test_challenger_adds_deliberation_notes(self):
        """El Challenger agrega notas de deliberación."""
        challenger = RiskChallenger()
        
        naive_report = ConstructionRiskReport(
            integrity_score=90.0,
            financial_risk_level="MEDIO",
            details={"topological_invariants": {"pyramid_stability": 0.8}},
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Media"
        )
        
        audited_report = challenger.challenge_verdict(naive_report)
        
        # Debe haber evidencia del debate
        narrative = getattr(audited_report, 'strategic_narrative', '')
        if narrative:
            assert_contains_any(
                narrative,
                ["CONTRADICCIÓN", "ACTA", "DELIBERACIÓN", "CHALLENGER", "DEBATE"],
                "Debe haber notas de deliberación"
            )


# ============================================================================
# TEST: COHERENCIA DE NARRATIVAS
# ============================================================================


class TestNarrativeCoherence:
    """
    Pruebas de coherencia entre narrativas de diferentes módulos.
    """

    def test_narrator_and_translator_agree_on_physics_failure(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
        context_with_physics_failure: TelemetryContext,
    ):
        """Narrator y Translator acuerdan sobre fallo en PHYSICS."""
        # Narrator
        narrator_report = narrator.summarize_execution(context_with_physics_failure)
        
        # Translator (simulando datos corruptos)
        translator_report = translator.compose_strategic_narrative(
            topological_metrics=TopologyMetricsDTO(beta_0=0, beta_1=0),  # Vacío
            financial_metrics={},
            stability=0.1,  # Muy inestable
        )
        
        # Ambos deben rechazar
        assert "PHYSICS" in narrator_report["verdict_code"] or "REJECTED" in narrator_report["verdict_code"]
        assert translator_report.verdict.value >= VerdictLevel.PRECAUCION.value

    def test_narrator_and_translator_agree_on_success(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
        context_all_success: TelemetryContext,
    ):
        """Narrator y Translator acuerdan sobre éxito."""
        narrator_report = narrator.summarize_execution(context_all_success)
        
        translator_report = translator.compose_strategic_narrative(
            topological_metrics=TopologyMetricsDTO(beta_0=1, beta_1=0),
            financial_metrics={
                "performance": {"recommendation": "ACEPTAR", "profitability_index": 1.5}
            },
            stability=15.0,
        )
        
        assert narrator_report["verdict_code"] == "APPROVED"
        assert translator_report.verdict == VerdictLevel.VIABLE

    def test_narratives_use_consistent_language(
        self,
        narrator: TelemetryNarrator,
        translator: SemanticTranslator,
        context_with_tactics_failure: TelemetryContext,
    ):
        """Las narrativas usan lenguaje consistente para el mismo problema."""
        # Narrator
        narrator_report = narrator.summarize_execution(context_with_tactics_failure)
        tactics_narrative = narrator_report["strata_analysis"]["TACTICS"]["narrative"]
        
        # Translator con ciclos
        translator_report = translator.compose_strategic_narrative(
            topological_metrics=TopologyMetricsDTO(beta_0=1, beta_1=5),
            financial_metrics={"performance": {"recommendation": "REVISAR"}},
            stability=10.0,
        )
        
        # Ambos deben mencionar problemas topológicos
        topo_keywords = ["ciclo", "socavón", "β₁", "genus", "topología", "estructura"]
        
        assert_contains_any(
            tactics_narrative,
            topo_keywords,
            "Narrator debe mencionar problema topológico"
        )
        
        assert_contains_any(
            translator_report.raw_narrative,
            topo_keywords,
            "Translator debe mencionar problema topológico"
        )


# ============================================================================
# TEST: CASOS ESPECIALES
# ============================================================================


class TestSpecialCases:
    """Pruebas de casos especiales y edge cases."""

    def test_empty_context_is_approved(
        self,
        narrator: TelemetryNarrator,
        fresh_context: TelemetryContext,
    ):
        """Contexto vacío genera veredicto neutro/aprobado."""
        report = narrator.summarize_execution(fresh_context)
        
        # Sin spans, debe ser EMPTY o APPROVED
        assert report["verdict_code"] in ["EMPTY", "APPROVED"]

    def test_warning_only_does_not_reject(
        self,
        narrator: TelemetryNarrator,
        fresh_context: TelemetryContext,
    ):
        """Solo warnings (sin errores) no produce rechazo."""
        with fresh_context.span("physics", stratum=Stratum.PHYSICS) as span:
            span.status = StepStatus.WARNING
        
        report = narrator.summarize_execution(fresh_context)
        
        # No debe ser rechazo duro
        assert "REJECTED" not in report["verdict_code"] or \
               report["global_severity"] != "CRITICO"

    def test_synergy_risk_always_rejects(
        self,
        translator: SemanticTranslator,
    ):
        """Sinergia de riesgo (Efecto Dominó) siempre produce rechazo."""
        synergy = {
            "synergy_detected": True,
            "intersecting_cycles_count": 3,
            "intersecting_cycles": [["A", "B", "C"]],
        }
        
        report = translator.compose_strategic_narrative(
            topological_metrics=TopologyMetricsDTO(beta_0=1, beta_1=0),  # Limpio
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=20.0,  # Muy estable
            synergy_risk=synergy,
        )
        
        assert report.verdict == VerdictLevel.RECHAZAR
        assert_contains_any(
            report.raw_narrative,
            ["Dominó", "EMERGENCIA", "Sinergia", "contagio"],
            "Debe mencionar el efecto dominó"
        )

    def test_thermal_fever_degrades_verdict(
        self,
        translator: SemanticTranslator,
    ):
        """Fiebre térmica (alta temperatura) degrada el veredicto."""
        thermal = {"system_temperature": 75.0, "entropy": 0.8}
        
        report = translator.compose_strategic_narrative(
            topological_metrics=TopologyMetricsDTO(),
            financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
            stability=10.0,
            thermal_metrics=thermal,
        )
        
        # Debe mencionar la fiebre
        assert_contains_any(
            report.raw_narrative,
            ["FIEBRE", "calor", "temperatura", "térmic"],
            "Debe mencionar problemas térmicos"
        )
        
        # Veredicto debe estar degradado
        assert report.verdict.value >= VerdictLevel.CONDICIONAL.value


# ============================================================================
# TEST: DETERMINISMO
# ============================================================================


class TestDeterminism:
    """Pruebas de determinismo en los reportes."""

    def test_narrator_is_deterministic(
        self,
        narrator: TelemetryNarrator,
    ):
        """El narrator produce el mismo resultado para el mismo input."""
        def create_context():
            ctx = TelemetryContext()
            with ctx.span("physics", stratum=Stratum.PHYSICS):
                ctx.record_metric("flux_condenser", "saturation", 0.5)
            with ctx.span("tactics", stratum=Stratum.TACTICS):
                ctx.record_metric("topology", "beta_1", 0)
            return ctx
        
        report1 = narrator.summarize_execution(create_context())
        report2 = narrator.summarize_execution(create_context())
        
        assert report1["verdict_code"] == report2["verdict_code"]
        assert report1["global_severity"] == report2["global_severity"]

    def test_translator_is_deterministic(
        self,
        translator: SemanticTranslator,
    ):
        """El translator produce el mismo resultado para el mismo input."""
        topology = TopologyMetricsDTO(beta_0=1, beta_1=2)
        financials = {"performance": {"recommendation": "REVISAR"}}
        
        report1 = translator.compose_strategic_narrative(
            topological_metrics=topology,
            financial_metrics=financials,
            stability=5.0,
        )
        report2 = translator.compose_strategic_narrative(
            topological_metrics=topology,
            financial_metrics=financials,
            stability=5.0,
        )
        
        assert report1.verdict == report2.verdict
        assert report1.raw_narrative == report2.raw_narrative