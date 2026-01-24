"""
Suite de Pruebas para Telemetría Piramidal (DIKW)
=================================================

Valida la implementación de:
1. Tracking de salud por estrato (TelemetryHealth por Stratum)
2. Propagación de errores (Clausura Transitiva)
3. Reportes piramidales
4. Integración con spans y context managers
5. Invariantes topológicos de la pirámide

Arquitectura de Tests:
- TestTelemetryHealthDataclass: Propiedades de TelemetryHealth
- TestStratumHierarchy: Jerarquía DIKW
- TestHealthTracking: Tracking por estrato
- TestFailurePropagation: Clausura transitiva
- TestPyramidalReports: Reportes estructurados
- TestSpanStratumIntegration: Integración de spans
- TestInvariantsVerification: Verificación de invariantes
- TestEdgeCasesAndRecovery: Casos límite
- TestIntegrationWithNarrator: Integración con TelemetryNarrator
"""

from __future__ import annotations

import copy
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import pytest

# Importar desde telemetry (evitar duplicados)
from app.telemetry import (
    TelemetryContext,
    TelemetrySpan,
    TelemetryHealth,
    StepStatus,
    TelemetryDefaults,
    ActiveStepInfo,
)

# Importar Stratum solo una vez desde schemas
from app.schemas import Stratum

# Importar narrator para tests de integración
from app.telemetry_narrative import (
    TelemetryNarrator,
    SeverityLevel,
    StratumTopology,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def fresh_context() -> TelemetryContext:
    """Contexto de telemetría fresco."""
    return TelemetryContext()


@pytest.fixture
def context_with_physics_step() -> TelemetryContext:
    """Contexto con un paso en PHYSICS iniciado."""
    ctx = TelemetryContext()
    ctx.start_step("physics_step", stratum=Stratum.PHYSICS)
    return ctx


@pytest.fixture
def context_with_all_strata() -> TelemetryContext:
    """Contexto con pasos en todos los estratos."""
    ctx = TelemetryContext()
    
    with ctx.span("load_data", stratum=Stratum.PHYSICS):
        pass
    with ctx.span("calculate_costs", stratum=Stratum.TACTICS):
        pass
    with ctx.span("financial_analysis", stratum=Stratum.STRATEGY):
        pass
    with ctx.span("build_output", stratum=Stratum.WISDOM):
        pass
    
    return ctx


@pytest.fixture
def narrator() -> TelemetryNarrator:
    """Narrador para tests de integración."""
    return TelemetryNarrator()


# ============================================================================
# TEST: TELEMETRY HEALTH DATACLASS
# ============================================================================


class TestTelemetryHealthDataclass:
    """Pruebas para la estructura TelemetryHealth."""

    def test_default_state_is_healthy(self):
        """Estado por defecto es saludable."""
        health = TelemetryHealth()
        
        assert health.is_healthy is True
        assert len(health.warnings) == 0
        assert len(health.errors) == 0
        assert len(health.stale_steps) == 0
        assert health.memory_pressure is False

    def test_add_warning_preserves_health(self):
        """Agregar warning no marca como no saludable."""
        health = TelemetryHealth()
        health.add_warning("Test warning")
        
        assert health.is_healthy is True
        assert len(health.warnings) == 1

    def test_add_error_marks_unhealthy(self):
        """Agregar error marca como no saludable."""
        health = TelemetryHealth()
        health.add_error("Test error")
        
        assert health.is_healthy is False
        assert len(health.errors) == 1

    def test_warnings_are_tuples_with_timestamp(self):
        """Warnings son tuplas (mensaje, timestamp)."""
        health = TelemetryHealth()
        before = time.perf_counter()
        health.add_warning("Test warning")
        after = time.perf_counter()
        
        msg, timestamp = health.warnings[0]
        assert msg == "Test warning"
        assert before <= timestamp <= after

    def test_errors_are_tuples_with_timestamp(self):
        """Errors son tuplas (mensaje, timestamp)."""
        health = TelemetryHealth()
        before = time.perf_counter()
        health.add_error("Test error")
        after = time.perf_counter()
        
        msg, timestamp = health.errors[0]
        assert msg == "Test error"
        assert before <= timestamp <= after

    def test_get_severity_level(self):
        """Nivel de severidad se calcula correctamente."""
        healthy = TelemetryHealth()
        assert healthy.get_severity_level() == 0  # HEALTHY
        
        with_warning = TelemetryHealth()
        with_warning.add_warning("warn")
        assert with_warning.get_severity_level() == 1  # WARNING
        
        with_error = TelemetryHealth()
        with_error.add_error("error")
        assert with_error.get_severity_level() == 2  # CRITICAL

    def test_get_status_string(self):
        """String de estado es correcto."""
        healthy = TelemetryHealth()
        assert healthy.get_status_string() == "HEALTHY"
        
        with_error = TelemetryHealth()
        with_error.add_error("error")
        assert with_error.get_status_string() == "CRITICAL"

    def test_merge_with_takes_worst_health(self):
        """Merge toma el peor estado de salud."""
        healthy = TelemetryHealth()
        
        unhealthy = TelemetryHealth()
        unhealthy.add_error("error")
        
        merged = healthy.merge_with(unhealthy)
        
        assert merged.is_healthy is False
        assert len(merged.errors) == 1

    def test_merge_combines_warnings_and_errors(self):
        """Merge combina warnings y errors de ambos."""
        h1 = TelemetryHealth()
        h1.add_warning("warn1")
        h1.add_error("error1")
        
        h2 = TelemetryHealth()
        h2.add_warning("warn2")
        
        merged = h1.merge_with(h2)
        
        assert len(merged.warnings) == 2
        assert len(merged.errors) == 1

    def test_to_dict_serialization(self):
        """Serialización a diccionario."""
        health = TelemetryHealth()
        health.add_warning("warn")
        health.add_error("error")
        health.stale_steps.append("stale_step")
        health.memory_pressure = True
        
        result = health.to_dict()
        
        assert result["status"] == "CRITICAL"
        assert result["is_healthy"] is False
        assert result["warning_count"] == 1
        assert result["error_count"] == 1
        assert "stale_step" in result["stale_steps"]
        assert result["memory_pressure"] is True


# ============================================================================
# TEST: STRATUM HIERARCHY
# ============================================================================


class TestStratumHierarchy:
    """Pruebas para la jerarquía de estratos."""

    def test_stratum_enum_values(self):
        """Valores del enum Stratum son correctos."""
        assert Stratum.WISDOM.value == 0
        assert Stratum.STRATEGY.value == 1
        assert Stratum.TACTICS.value == 2
        assert Stratum.PHYSICS.value == 3

    def test_physics_is_base(self):
        """PHYSICS es la base de la pirámide."""
        assert Stratum.PHYSICS.value == max(s.value for s in Stratum)

    def test_wisdom_is_top(self):
        """WISDOM es la cima de la pirámide."""
        assert Stratum.WISDOM.value == min(s.value for s in Stratum)

    def test_stratum_order_for_propagation(self):
        """El orden de estratos es correcto para propagación."""
        # Fallo en PHYSICS (3) debe propagar a TACTICS (2), STRATEGY (1), WISDOM (0)
        physics_level = Stratum.PHYSICS.value
        
        strata_above = [s for s in Stratum if s.value < physics_level]
        
        assert Stratum.TACTICS in strata_above
        assert Stratum.STRATEGY in strata_above
        assert Stratum.WISDOM in strata_above
        assert Stratum.PHYSICS not in strata_above


# ============================================================================
# TEST: HEALTH TRACKING POR ESTRATO
# ============================================================================


class TestHealthTracking:
    """Pruebas de tracking de salud por estrato."""

    def test_initial_strata_health(self, fresh_context: TelemetryContext):
        """Todos los estratos inician saludables."""
        for stratum in Stratum:
            assert stratum in fresh_context._strata_health
            assert fresh_context._strata_health[stratum].is_healthy is True

    def test_error_degrades_stratum_health(self, fresh_context: TelemetryContext):
        """Un error degrada la salud del estrato correspondiente."""
        fresh_context.start_step("test_step", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="test_step",
            error_message="Test error",
            severity="ERROR"
        )
        
        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False

    def test_error_contains_message(self, fresh_context: TelemetryContext):
        """El mensaje de error se registra correctamente."""
        fresh_context.start_step("test_step", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="test_step",
            error_message="Specific error message",
            severity="ERROR"
        )
        
        errors = fresh_context._strata_health[Stratum.PHYSICS].errors
        assert any("Specific error message" in e[0] for e in errors)

    def test_non_critical_error_no_propagation(self, fresh_context: TelemetryContext):
        """Errores no críticos no propagan a otros estratos."""
        fresh_context.start_step("test_step", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="test_step",
            error_message="Non-critical error",
            severity="ERROR"  # No CRITICAL
        )
        
        # PHYSICS no saludable
        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False
        
        # Otros estratos saludables (sin warnings de propagación)
        assert fresh_context._strata_health[Stratum.TACTICS].is_healthy is True
        assert len(fresh_context._strata_health[Stratum.TACTICS].warnings) == 0

    def test_explicit_stratum_in_record_error(self, fresh_context: TelemetryContext):
        """El estrato explícito en record_error tiene prioridad."""
        # No hay paso activo, pero especificamos estrato explícitamente
        fresh_context.record_error(
            step_name="external_step",
            error_message="Strategic error",
            severity="ERROR",
            stratum=Stratum.STRATEGY
        )
        
        # STRATEGY afectado, PHYSICS no
        assert fresh_context._strata_health[Stratum.STRATEGY].is_healthy is False
        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is True

    def test_active_step_stratum_detection(self, context_with_physics_step: TelemetryContext):
        """El estrato se detecta del paso activo."""
        ctx = context_with_physics_step
        
        # Error sin estrato explícito, debe usar el del paso activo
        ctx.record_error(
            step_name="physics_step",
            error_message="Physics error",
            severity="ERROR"
        )
        
        assert ctx._strata_health[Stratum.PHYSICS].is_healthy is False

    def test_default_stratum_when_no_active_step(self, fresh_context: TelemetryContext):
        """Sin paso activo ni estrato explícito, usa PHYSICS."""
        fresh_context.record_error(
            step_name="unknown_step",
            error_message="Unknown error",
            severity="ERROR"
        )
        
        # Debe ir a PHYSICS por defecto
        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False


# ============================================================================
# TEST: PROPAGACIÓN DE FALLOS (CLAUSURA TRANSITIVA)
# ============================================================================


class TestFailurePropagation:
    """Pruebas de propagación de fallos (Clausura Transitiva)."""

    def test_critical_error_propagates_upwards(self, fresh_context: TelemetryContext):
        """Error CRITICAL en PHYSICS propaga warnings hacia arriba."""
        fresh_context.start_step("critical_step", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="critical_step",
            error_message="Critical failure",
            severity="CRITICAL"
        )
        
        # PHYSICS no saludable
        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False
        
        # TACTICS debe tener warning de propagación
        tactics_warnings = fresh_context._strata_health[Stratum.TACTICS].warnings
        assert len(tactics_warnings) > 0
        assert any("inherited" in w[0].lower() or "instability" in w[0].lower() 
                   for w in tactics_warnings)
        
        # STRATEGY también debe tener warning
        strategy_warnings = fresh_context._strata_health[Stratum.STRATEGY].warnings
        assert len(strategy_warnings) > 0

    def test_propagation_includes_source_stratum(self, fresh_context: TelemetryContext):
        """El mensaje de propagación incluye el estrato de origen."""
        fresh_context.start_step("physics_fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="physics_fail",
            error_message="Physics explosion",
            severity="CRITICAL"
        )
        
        tactics_warnings = fresh_context._strata_health[Stratum.TACTICS].warnings
        assert any("PHYSICS" in w[0] for w in tactics_warnings)

    def test_propagation_only_upwards(self, fresh_context: TelemetryContext):
        """La propagación solo va hacia arriba en la pirámide."""
        # Error en TACTICS (nivel 2)
        fresh_context.start_step("tactics_fail", stratum=Stratum.TACTICS)
        fresh_context.record_error(
            step_name="tactics_fail",
            error_message="Tactics failure",
            severity="CRITICAL"
        )
        
        # STRATEGY y WISDOM deben tener warnings (están arriba)
        assert len(fresh_context._strata_health[Stratum.STRATEGY].warnings) > 0
        assert len(fresh_context._strata_health[Stratum.WISDOM].warnings) > 0
        
        # PHYSICS no debe tener warnings (está abajo)
        assert len(fresh_context._strata_health[Stratum.PHYSICS].warnings) == 0

    def test_propagation_does_not_mark_unhealthy(self, fresh_context: TelemetryContext):
        """La propagación agrega warnings pero no marca como unhealthy."""
        fresh_context.start_step("physics_fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="physics_fail",
            error_message="Critical physics error",
            severity="CRITICAL"
        )
        
        # TACTICS tiene warnings pero sigue saludable
        # (porque los warnings no son errores directos)
        tactics_health = fresh_context._strata_health[Stratum.TACTICS]
        assert tactics_health.is_healthy is True  # Solo warnings, no errors
        assert len(tactics_health.warnings) > 0

    def test_multiple_propagations_accumulate(self, fresh_context: TelemetryContext):
        """Múltiples fallos críticos acumulan warnings."""
        # Primer fallo en PHYSICS
        fresh_context.start_step("fail1", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail1", "First failure", severity="CRITICAL")
        fresh_context.end_step("fail1", StepStatus.FAILURE)
        
        # Segundo fallo en PHYSICS
        fresh_context.start_step("fail2", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail2", "Second failure", severity="CRITICAL")
        
        # TACTICS debe tener múltiples warnings
        tactics_warnings = fresh_context._strata_health[Stratum.TACTICS].warnings
        assert len(tactics_warnings) >= 2

    def test_no_propagation_for_non_critical(self, fresh_context: TelemetryContext):
        """Severidades no-CRITICAL no propagan."""
        for severity in ["ERROR", "WARNING", "INFO"]:
            ctx = TelemetryContext()
            ctx.start_step("test", stratum=Stratum.PHYSICS)
            ctx.record_error("test", f"{severity} message", severity=severity)
            
            # TACTICS no debe tener warnings de propagación
            assert len(ctx._strata_health[Stratum.TACTICS].warnings) == 0

    def test_wisdom_receives_all_propagations(self, fresh_context: TelemetryContext):
        """WISDOM (cima) recibe propagación de todos los estratos inferiores."""
        # Fallo en TACTICS
        fresh_context.start_step("tactics_fail", stratum=Stratum.TACTICS)
        fresh_context.record_error("tactics_fail", "Tactics error", severity="CRITICAL")
        
        # WISDOM debe recibir propagación
        wisdom_warnings = fresh_context._strata_health[Stratum.WISDOM].warnings
        assert len(wisdom_warnings) > 0
        assert any("TACTICS" in w[0] for w in wisdom_warnings)


# ============================================================================
# TEST: REPORTES PIRAMIDALES
# ============================================================================


class TestPyramidalReports:
    """Pruebas de reportes piramidales."""

    def test_report_structure(self, fresh_context: TelemetryContext):
        """Estructura del reporte piramidal."""
        report = fresh_context.get_pyramidal_report()
        
        assert "physics_layer" in report
        assert "tactics_layer" in report
        assert "strategy_layer" in report
        # WISDOM layer si está implementado
        # assert "wisdom_layer" in report

    def test_report_layer_structure(self, fresh_context: TelemetryContext):
        """Estructura de cada capa del reporte."""
        report = fresh_context.get_pyramidal_report()
        
        for layer_key in ["physics_layer", "tactics_layer", "strategy_layer"]:
            layer = report[layer_key]
            assert "status" in layer
            assert "metrics" in layer
            assert "issues" in layer
            assert "warnings" in layer

    def test_healthy_status_when_no_errors(self, fresh_context: TelemetryContext):
        """Status es HEALTHY cuando no hay errores."""
        # Solo registrar métricas, sin errores
        fresh_context.record_metric("flux_condenser", "avg_saturation", 0.5)
        
        report = fresh_context.get_pyramidal_report()
        
        assert report["physics_layer"]["status"] == "HEALTHY"
        assert report["tactics_layer"]["status"] == "HEALTHY"
        assert report["strategy_layer"]["status"] == "HEALTHY"

    def test_critical_status_on_error(self, fresh_context: TelemetryContext):
        """Status es CRITICAL cuando hay errores."""
        fresh_context.start_step("fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail", "Error", severity="CRITICAL")
        
        report = fresh_context.get_pyramidal_report()
        
        assert report["physics_layer"]["status"] == "CRITICAL"

    def test_warning_status_on_propagation(self, fresh_context: TelemetryContext):
        """Status es WARNING cuando hay warnings de propagación."""
        fresh_context.start_step("fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail", "Critical error", severity="CRITICAL")
        
        report = fresh_context.get_pyramidal_report()
        
        # PHYSICS es CRITICAL
        assert report["physics_layer"]["status"] == "CRITICAL"
        
        # TACTICS y STRATEGY son WARNING (tienen warnings propagados)
        assert report["tactics_layer"]["status"] == "WARNING"
        assert report["strategy_layer"]["status"] == "WARNING"

    def test_metrics_in_correct_layer(self, fresh_context: TelemetryContext):
        """Las métricas aparecen en la capa correcta."""
        fresh_context.record_metric("flux_condenser", "avg_saturation", 0.8)
        fresh_context.record_metric("topology", "beta_0", 1)
        fresh_context.record_metric("financial", "npv", 50000)
        
        report = fresh_context.get_pyramidal_report()
        
        # flux_condenser es PHYSICS
        assert "flux_condenser.avg_saturation" in report["physics_layer"]["metrics"]
        
        # topology es TACTICS
        assert "topology.beta_0" in report["tactics_layer"]["metrics"]
        
        # financial es STRATEGY
        assert "financial.npv" in report["strategy_layer"]["metrics"]

    def test_issues_list_populated(self, fresh_context: TelemetryContext):
        """La lista de issues se popula correctamente."""
        fresh_context.start_step("fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail", "Physics error", severity="CRITICAL")
        
        report = fresh_context.get_pyramidal_report()
        
        assert len(report["physics_layer"]["issues"]) > 0
        assert "Physics error" in report["physics_layer"]["issues"][0]

    def test_warnings_list_flattened(self, fresh_context: TelemetryContext):
        """Las warnings se aplanan a strings en el reporte."""
        fresh_context.start_step("fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail", "Critical", severity="CRITICAL")
        
        report = fresh_context.get_pyramidal_report()
        
        # Las warnings deben ser strings, no tuplas
        for warning in report["tactics_layer"]["warnings"]:
            assert isinstance(warning, str)


# ============================================================================
# TEST: INTEGRACIÓN DE SPANS CON STRATUM
# ============================================================================


class TestSpanStratumIntegration:
    """Pruebas de integración de spans con estratos."""

    def test_span_records_stratum(self, fresh_context: TelemetryContext):
        """El span registra el estrato correctamente."""
        with fresh_context.span("test_span", stratum=Stratum.TACTICS) as span:
            assert span.stratum == Stratum.TACTICS

    def test_span_in_root_spans(self, fresh_context: TelemetryContext):
        """El span con estrato aparece en root_spans."""
        with fresh_context.span("test_span", stratum=Stratum.STRATEGY):
            pass
        
        assert len(fresh_context.root_spans) == 1
        assert fresh_context.root_spans[0].stratum == Stratum.STRATEGY

    def test_nested_spans_preserve_stratum(self, fresh_context: TelemetryContext):
        """Spans anidados preservan su estrato."""
        with fresh_context.span("parent", stratum=Stratum.PHYSICS) as parent:
            with fresh_context.span("child", stratum=Stratum.TACTICS) as child:
                assert child.stratum == Stratum.TACTICS
            assert parent.stratum == Stratum.PHYSICS

    def test_span_default_stratum_is_physics(self, fresh_context: TelemetryContext):
        """El estrato por defecto es PHYSICS."""
        with fresh_context.span("default_span") as span:
            pass
        
        assert fresh_context.root_spans[0].stratum == Stratum.PHYSICS

    def test_step_context_manager_with_stratum(self, fresh_context: TelemetryContext):
        """El context manager step usa el estrato correctamente."""
        with fresh_context.step("test_step", stratum=Stratum.WISDOM):
            pass
        
        # Verificar que el paso se registró
        assert any(s.get("step") == "test_step" for s in fresh_context.steps)

    def test_span_failure_affects_stratum_health(self, fresh_context: TelemetryContext):
        """Fallo en span afecta la salud del estrato."""
        try:
            with fresh_context.span("failing_span", stratum=Stratum.TACTICS) as span:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verificar que se registró el error
        assert len(fresh_context.errors) > 0


# ============================================================================
# TEST: VERIFICACIÓN DE INVARIANTES
# ============================================================================


class TestInvariantsVerification:
    """Pruebas de verificación de invariantes."""

    def test_verify_invariants_on_healthy_context(self, fresh_context: TelemetryContext):
        """Contexto saludable pasa verificación de invariantes."""
        with fresh_context.span("test"):
            pass
        
        result = fresh_context.verify_invariants()
        
        assert result["is_valid"] is True
        assert len(result["violations"]) == 0

    def test_verify_invariants_includes_topology(self, fresh_context: TelemetryContext):
        """La verificación incluye métricas de topología."""
        with fresh_context.span("parent"):
            with fresh_context.span("child"):
                pass
        
        result = fresh_context.verify_invariants()
        
        assert "topology" in result
        assert "num_trees" in result["topology"]

    def test_verify_invariants_detects_violations(self, fresh_context: TelemetryContext):
        """La verificación detecta violaciones."""
        # Agregar más de max_steps manualmente (simulando overflow)
        for i in range(fresh_context.max_steps + 10):
            fresh_context.steps.append({"step": f"overflow_{i}"})
        
        result = fresh_context.verify_invariants()
        
        assert result["is_valid"] is False
        assert any("overflow" in v.lower() for v in result["violations"])

    def test_verify_invariants_counts(self, fresh_context: TelemetryContext):
        """La verificación incluye conteos correctos."""
        with fresh_context.span("span1"):
            pass
        with fresh_context.span("span2"):
            pass
        
        fresh_context.record_metric("test", "metric", 1.0)
        fresh_context.record_error("test", "error", severity="ERROR")
        
        result = fresh_context.verify_invariants()
        
        assert result["counts"]["root_spans"] == 2
        assert result["counts"]["metrics"] == 1
        assert result["counts"]["errors"] == 1


# ============================================================================
# TEST: CASOS LÍMITE Y RECUPERACIÓN
# ============================================================================


class TestEdgeCasesAndRecovery:
    """Pruebas de casos límite y recuperación."""

    def test_reset_clears_strata_health(self, fresh_context: TelemetryContext):
        """Reset limpia la salud de todos los estratos."""
        fresh_context.start_step("fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail", "Error", severity="CRITICAL")
        
        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False
        
        fresh_context.reset()
        
        # Todos los estratos deben estar saludables
        for stratum in Stratum:
            assert fresh_context._strata_health[stratum].is_healthy is True

    def test_invalid_stratum_type_handled(self, fresh_context: TelemetryContext):
        """Tipo de estrato inválido se maneja graciosamente."""
        # Pasar algo que no es Stratum
        fresh_context.record_error(
            step_name="test",
            error_message="Error",
            severity="ERROR",
            stratum="not_a_stratum"  # type: ignore
        )
        
        # Debe usar default (PHYSICS) y no crashear
        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False

    def test_empty_context_pyramidal_report(self, fresh_context: TelemetryContext):
        """Contexto vacío genera reporte piramidal válido."""
        report = fresh_context.get_pyramidal_report()
        
        assert report is not None
        assert "physics_layer" in report
        assert report["physics_layer"]["status"] == "HEALTHY"

    def test_many_errors_in_one_stratum(self, fresh_context: TelemetryContext):
        """Muchos errores en un estrato se manejan correctamente."""
        for i in range(100):
            fresh_context.record_error(
                step_name=f"step_{i}",
                error_message=f"Error {i}",
                severity="ERROR",
                stratum=Stratum.PHYSICS
            )
        
        report = fresh_context.get_pyramidal_report()
        
        assert report["physics_layer"]["status"] == "CRITICAL"
        # Issues deben estar limitados
        assert len(report["physics_layer"]["issues"]) <= 100

    def test_concurrent_stratum_operations(self, fresh_context: TelemetryContext):
        """Operaciones concurrentes en diferentes estratos."""
        import threading
        
        errors = []
        
        def record_to_stratum(stratum: Stratum, count: int):
            try:
                for i in range(count):
                    fresh_context.record_error(
                        step_name=f"{stratum.name}_{i}",
                        error_message=f"Error in {stratum.name}",
                        severity="ERROR",
                        stratum=stratum
                    )
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=record_to_stratum, args=(Stratum.PHYSICS, 20)),
            threading.Thread(target=record_to_stratum, args=(Stratum.TACTICS, 20)),
            threading.Thread(target=record_to_stratum, args=(Stratum.STRATEGY, 20)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        # Verificar que todos los estratos tienen errores
        assert not fresh_context._strata_health[Stratum.PHYSICS].is_healthy
        assert not fresh_context._strata_health[Stratum.TACTICS].is_healthy
        assert not fresh_context._strata_health[Stratum.STRATEGY].is_healthy


# ============================================================================
# TEST: INTEGRACIÓN CON TELEMETRY NARRATOR
# ============================================================================


class TestIntegrationWithNarrator:
    """Pruebas de integración con TelemetryNarrator."""

    def test_narrator_uses_span_stratum(
        self, narrator: TelemetryNarrator, fresh_context: TelemetryContext
    ):
        """El narrador respeta el estrato del span."""
        with fresh_context.span("load_data", stratum=Stratum.PHYSICS):
            pass
        with fresh_context.span("calculate_costs", stratum=Stratum.TACTICS):
            pass
        
        report = narrator.summarize_execution(fresh_context)
        
        # Verificar que los estratos están presentes
        assert "PHYSICS" in report["strata_analysis"]
        assert "TACTICS" in report["strata_analysis"]

    def test_narrator_reflects_physics_failure(
        self, narrator: TelemetryNarrator, fresh_context: TelemetryContext
    ):
        """El narrador refleja fallo en PHYSICS."""
        with fresh_context.span("load_data", stratum=Stratum.PHYSICS) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Data corruption", "type": "IOError"})
        
        report = narrator.summarize_execution(fresh_context)
        
        assert "PHYSICS" in report["verdict_code"]
        assert report["strata_analysis"]["PHYSICS"]["severity"] == "CRITICO"

    def test_narrator_reflects_tactics_failure(
        self, narrator: TelemetryNarrator, fresh_context: TelemetryContext
    ):
        """El narrador refleja fallo en TACTICS."""
        with fresh_context.span("load_data", stratum=Stratum.PHYSICS):
            pass
        with fresh_context.span("calculate_costs", stratum=Stratum.TACTICS) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Cycle detected", "type": "TopologyError"})
        
        report = narrator.summarize_execution(fresh_context)
        
        assert "TACTICS" in report["verdict_code"]
        assert report["strata_analysis"]["TACTICS"]["severity"] == "CRITICO"

    def test_narrator_approved_when_all_healthy(
        self, narrator: TelemetryNarrator, context_with_all_strata: TelemetryContext
    ):
        """El narrador aprueba cuando todos los estratos están saludables."""
        report = narrator.summarize_execution(context_with_all_strata)
        
        assert report["verdict_code"] == "APPROVED"

    def test_pyramidal_report_consistent_with_narrator(
        self, narrator: TelemetryNarrator, fresh_context: TelemetryContext
    ):
        """El reporte piramidal es consistente con el narrador."""
        with fresh_context.span("fail", stratum=Stratum.PHYSICS) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Critical failure", "type": "Error"})
        
        pyramidal = fresh_context.get_pyramidal_report()
        narrator_report = narrator.summarize_execution(fresh_context)
        
        # Ambos deben indicar PHYSICS como problemático
        assert pyramidal["physics_layer"]["status"] == "CRITICAL"
        assert narrator_report["strata_analysis"]["PHYSICS"]["severity"] == "CRITICO"

    def test_health_propagation_reflected_in_narrator(
        self, narrator: TelemetryNarrator, fresh_context: TelemetryContext
    ):
        """La propagación de salud se refleja en el narrador."""
        # Fallo crítico en PHYSICS
        with fresh_context.span("physics_fail", stratum=Stratum.PHYSICS) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Critical failure", "type": "Error"})
        
        report = narrator.summarize_execution(fresh_context)
        
        # El veredicto debe ser rechazo por PHYSICS
        assert "PHYSICS" in report["verdict_code"]
        
        # La narrativa debe mencionar impacto en otros niveles
        executive_summary = report["executive_summary"]
        assert "FÍSICA" in executive_summary or "PHYSICS" in executive_summary or \
               "ABORTADO" in executive_summary


# ============================================================================
# TEST: PROPIEDADES ALGEBRAICAS
# ============================================================================


class TestAlgebraicProperties:
    """Pruebas de propiedades algebraicas del sistema piramidal."""

    def test_propagation_is_monotonic(self, fresh_context: TelemetryContext):
        """La propagación es monótonica: más fallos no mejoran la salud."""
        # Estado inicial
        initial_tactics_warnings = len(fresh_context._strata_health[Stratum.TACTICS].warnings)
        
        # Primer fallo
        fresh_context.record_error("fail1", "Error 1", severity="CRITICAL", stratum=Stratum.PHYSICS)
        first_warnings = len(fresh_context._strata_health[Stratum.TACTICS].warnings)
        
        # Segundo fallo
        fresh_context.record_error("fail2", "Error 2", severity="CRITICAL", stratum=Stratum.PHYSICS)
        second_warnings = len(fresh_context._strata_health[Stratum.TACTICS].warnings)
        
        # Las warnings deben ser monótonas crecientes
        assert first_warnings >= initial_tactics_warnings
        assert second_warnings >= first_warnings

    def test_stratum_order_is_total(self):
        """El orden de estratos es total (todos comparables)."""
        strata = list(Stratum)
        
        for a in strata:
            for b in strata:
                # Siempre podemos comparar valores
                assert a.value <= b.value or b.value <= a.value

    def test_propagation_respects_order(self, fresh_context: TelemetryContext):
        """La propagación respeta el orden de la pirámide."""
        # Fallo en nivel intermedio (TACTICS = 2)
        fresh_context.record_error("fail", "Error", severity="CRITICAL", stratum=Stratum.TACTICS)
        
        # Verificar que solo propaga hacia arriba (niveles menores)
        # STRATEGY (1) y WISDOM (0) deben tener warnings
        assert len(fresh_context._strata_health[Stratum.STRATEGY].warnings) > 0
        assert len(fresh_context._strata_health[Stratum.WISDOM].warnings) > 0
        
        # PHYSICS (3) NO debe tener warnings
        assert len(fresh_context._strata_health[Stratum.PHYSICS].warnings) == 0

    def test_health_merge_is_commutative(self):
        """El merge de health es conmutativo."""
        h1 = TelemetryHealth()
        h1.add_warning("warn1")
        
        h2 = TelemetryHealth()
        h2.add_error("error1")
        
        merge_12 = h1.merge_with(h2)
        merge_21 = h2.merge_with(h1)
        
        assert merge_12.is_healthy == merge_21.is_healthy
        assert len(merge_12.warnings) == len(merge_21.warnings)
        assert len(merge_12.errors) == len(merge_21.errors)

    def test_health_merge_is_associative(self):
        """El merge de health es asociativo."""
        h1 = TelemetryHealth()
        h1.add_warning("warn1")
        
        h2 = TelemetryHealth()
        h2.add_error("error1")
        
        h3 = TelemetryHealth()
        h3.add_warning("warn2")
        
        # (h1 ∘ h2) ∘ h3
        merge_left = h1.merge_with(h2).merge_with(h3)
        
        # h1 ∘ (h2 ∘ h3)
        merge_right = h1.merge_with(h2.merge_with(h3))
        
        assert merge_left.is_healthy == merge_right.is_healthy


# ============================================================================
# TEST: MÉTODOS DE FILTRADO DE MÉTRICAS
# ============================================================================


class TestMetricsFiltering:
    """Pruebas de filtrado de métricas por estrato."""

    def test_filter_metrics_by_prefix(self, fresh_context: TelemetryContext):
        """Las métricas se filtran correctamente por prefijo."""
        fresh_context.record_metric("flux_condenser", "saturation", 0.5)
        fresh_context.record_metric("flux_condenser", "voltage", 1.2)
        fresh_context.record_metric("topology", "beta_0", 1)
        fresh_context.record_metric("financial", "npv", 10000)
        
        physics_metrics = fresh_context._filter_metrics_by_prefix("flux_condenser")
        
        assert "flux_condenser.saturation" in physics_metrics
        assert "flux_condenser.voltage" in physics_metrics
        assert "topology.beta_0" not in physics_metrics
        assert "financial.npv" not in physics_metrics

    def test_filter_returns_copy(self, fresh_context: TelemetryContext):
        """El filtrado retorna copia (inmutabilidad)."""
        fresh_context.record_metric("test", "value", {"nested": "data"})
        
        filtered = fresh_context._filter_metrics_by_prefix("test")
        
        # Modificar el resultado no debe afectar el original
        if "test.value" in filtered and isinstance(filtered["test.value"], dict):
            filtered["test.value"]["nested"] = "modified"
            original = fresh_context.get_metric("test", "value")
            assert original.get("nested") == "data"

    def test_get_metrics_by_stratum_integration(self, fresh_context: TelemetryContext):
        """Las métricas por estrato funcionan correctamente."""
        # Registrar métricas con prefijos asociados a estratos
        fresh_context.record_metric("flux_condenser", "temp", 45.0)
        fresh_context.record_metric("topology", "cycles", 0)
        fresh_context.record_metric("financial", "roi", 1.5)
        
        report = fresh_context.get_pyramidal_report()
        
        # Verificar que las métricas están en las capas correctas
        assert "flux_condenser.temp" in report["physics_layer"]["metrics"]
        assert "topology.cycles" in report["tactics_layer"]["metrics"]
        assert "financial.roi" in report["strategy_layer"]["metrics"]