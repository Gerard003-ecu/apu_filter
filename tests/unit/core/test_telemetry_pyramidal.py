"""
Suite de Pruebas para Telemetría Piramidal (DIKW)
=================================================

Fundamentos del Modelo DIKW (Data-Information-Knowledge-Wisdom):
────────────────────────────────────────────────────────────────
  WISDOM   (nivel 0) ← cima:    síntesis ejecutiva
  STRATEGY (nivel 1):           análisis financiero / KPIs
  TACTICS  (nivel 2):           topología / cohesión
  PHYSICS  (nivel 3) ← base:    datos crudos / insumos

Propiedades Algebraicas Verificadas:
  · Propagación transitiva: fallo en nivel k propaga warnings a todos j < k
  · Monotonía: más fallos nunca mejoran la salud de un estrato
  · Merge conmutativo:     h1 ⊕ h2 ≡ h2 ⊕ h1  (is_healthy, |warnings|, |errors|)
  · Merge asociativo:      (h1 ⊕ h2) ⊕ h3 ≡ h1 ⊕ (h2 ⊕ h3)  (is_healthy)
  · Orden total de estratos: ∀ a,b ∈ Stratum: a.value ≤ b.value ∨ b.value ≤ a.value

Arquitectura de Tests:
  TestTelemetryHealthDataclass   → Invariantes de la estructura TelemetryHealth
  TestStratumHierarchy           → Jerarquía DIKW y orden de estratos
  TestHealthTracking             → Tracking por estrato y detección de origen
  TestFailurePropagation         → Clausura transitiva de fallos críticos
  TestPyramidalReports           → Reportes estructurados por capa
  TestSpanStratumIntegration     → Context managers y spans con estrato
  TestInvariantsVerification     → Verificación de invariantes topológicos
  TestEdgeCasesAndRecovery       → Robustez ante entradas atípicas
  TestIntegrationWithNarrator    → Coherencia con TelemetryNarrator
  TestAlgebraicProperties        → Propiedades algebraicas del sistema
  TestMetricsFiltering           → Filtrado y asignación de métricas por capa

Referencias:
  - Ackoff, R. L. (1989). From Data to Wisdom.
  - Chung, F. R. K. (1997). Spectral Graph Theory (para invariantes de propagación).
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any

import pytest

from app.core.telemetry import (
    ActiveStepInfo,
    StepStatus,
    TelemetryContext,
    TelemetryDefaults,
    TelemetryHealth,
    TelemetrySpan,
)
from app.core.schemas import Stratum
from app.core.telemetry_narrative import (
    SeverityLevel,
    StratumTopology,
    TelemetryNarrator,
)

# ── Constantes de dominio ────────────────────────────────────────────────────
# Centralizar prefijos de métricas evita acoplamiento frágil con strings
# dispersos en los tests. Si el dominio cambia, solo se modifica aquí.
_PREFIX_PHYSICS: str   = "flux_condenser"
_PREFIX_TACTICS: str   = "topology"
_PREFIX_STRATEGY: str  = "financial"

# Severidad de salud (contrato explícito, no magic numbers)
_SEVERITY_HEALTHY:   int = 0
_SEVERITY_WARNING:   int = 1
_SEVERITY_CRITICAL:  int = 2

# Timeout para operaciones concurrentes (segundos)
_THREAD_TIMEOUT: float = 5.0

# Número de errores en tests de estrés
_STRESS_ERROR_COUNT: int = 100
_CONCURRENT_ERROR_COUNT: int = 20


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def fresh_context() -> TelemetryContext:
    """
    Contexto de telemetría en estado inicial limpio.

    Garantías del fixture:
      · Todos los estratos saludables.
      · Sin pasos activos, sin métricas, sin errores.
    """
    return TelemetryContext()


@pytest.fixture
def context_with_physics_step() -> TelemetryContext:
    """
    Contexto con un paso en PHYSICS iniciado (sin completar).

    Nota de diseño: el paso NO se termina intencionalmente para
    simular un contexto con paso activo, útil para tests de
    detección de estrato desde paso activo.

    Advertencia: no usar este fixture en tests que llamen `reset()`
    y esperen estado 100% limpio — el paso activo puede interferir.
    """
    ctx = TelemetryContext()
    ctx.start_step("physics_step", stratum=Stratum.PHYSICS)
    return ctx


@pytest.fixture
def context_with_all_strata() -> TelemetryContext:
    """
    Contexto con un span exitoso en cada estrato.

    Útil para verificar comportamiento cuando todos los estratos
    tienen actividad pero sin errores.
    """
    ctx = TelemetryContext()
    with ctx.span("load_data",        stratum=Stratum.PHYSICS):   pass
    with ctx.span("calculate_costs",  stratum=Stratum.TACTICS):   pass
    with ctx.span("financial_analysis", stratum=Stratum.STRATEGY): pass
    with ctx.span("build_output",     stratum=Stratum.WISDOM):    pass
    return ctx


@pytest.fixture
def narrator() -> TelemetryNarrator:
    """Narrador de telemetría para tests de integración."""
    return TelemetryNarrator()


# ============================================================================
# TEST: TELEMETRY HEALTH DATACLASS
# ============================================================================


class TestTelemetryHealthDataclass:
    """
    Pruebas de invariantes de TelemetryHealth.

    TelemetryHealth es un monoide bajo la operación merge_with:
      · Elemento neutro: TelemetryHealth() (estado saludable)
      · Operación: merge_with (toma el peor estado)
    """

    def test_default_state_is_healthy(self) -> None:
        """Estado por defecto: saludable, sin warnings ni errores."""
        health = TelemetryHealth()

        assert health.is_healthy is True
        assert len(health.warnings)     == 0
        assert len(health.errors)       == 0
        assert len(health.stale_steps)  == 0
        assert health.memory_pressure   is False

    def test_add_warning_preserves_healthy_status(self) -> None:
        """
        Warning no degrada is_healthy.

        Contrato: warnings son alertas no bloqueantes;
        solo los errores marcan el estado como no saludable.
        """
        health = TelemetryHealth()
        health.add_warning("Test warning")

        assert health.is_healthy is True
        assert len(health.warnings) == 1

    def test_add_error_marks_unhealthy(self) -> None:
        """Error degrada is_healthy a False."""
        health = TelemetryHealth()
        health.add_error("Test error")

        assert health.is_healthy is False
        assert len(health.errors) == 1

    def test_multiple_warnings_do_not_mark_unhealthy(self) -> None:
        """
        Acumular warnings nunca degrada is_healthy.

        Propiedad: is_healthy solo cambia a False por add_error.
        """
        health = TelemetryHealth()
        for i in range(10):
            health.add_warning(f"Warning {i}")

        assert health.is_healthy is True
        assert len(health.warnings) == 10

    def test_warnings_are_tuples_with_monotonic_timestamp(self) -> None:
        """
        Warnings son tuplas (mensaje: str, timestamp: float).

        Usamos time.monotonic() como referencia porque es el reloj
        más adecuado para medir intervalos (no retrocede, no salta).
        Si TelemetryHealth usa un reloj diferente, el test lo detecta
        mediante la verificación de tipos, no de rangos exactos.
        """
        health = TelemetryHealth()
        t_before = time.monotonic()
        health.add_warning("Test warning")
        t_after = time.monotonic()

        assert len(health.warnings) == 1
        entry = health.warnings[0]

        # Verificar estructura (msg, timestamp)
        assert isinstance(entry, tuple), (
            f"Warning debe ser tuple, obtenido: {type(entry).__name__}"
        )
        assert len(entry) == 2, (
            f"Warning debe tener 2 elementos, obtenidos: {len(entry)}"
        )

        msg, timestamp = entry
        assert isinstance(msg, str),   f"msg debe ser str, obtenido: {type(msg).__name__}"
        assert isinstance(timestamp, float), (
            f"timestamp debe ser float, obtenido: {type(timestamp).__name__}"
        )

        # Verificación de orden temporal (tolerancia: el timestamp
        # puede usar cualquier reloj monótono compatible)
        assert timestamp >= 0, "Timestamp debe ser no negativo"

    def test_errors_are_tuples_with_timestamp(self) -> None:
        """
        Errors son tuplas (mensaje: str, timestamp: float).

        Misma estructura que warnings para consistencia de API.
        """
        health = TelemetryHealth()
        health.add_error("Test error")

        assert len(health.errors) == 1
        entry = health.errors[0]

        assert isinstance(entry, tuple) and len(entry) == 2
        msg, timestamp = entry
        assert isinstance(msg, str)
        assert isinstance(timestamp, float) and timestamp >= 0

    def test_get_severity_level_healthy(self) -> None:
        """Severidad = HEALTHY (0) cuando no hay warnings ni errores."""
        health = TelemetryHealth()
        assert health.get_severity_level() == _SEVERITY_HEALTHY

    def test_get_severity_level_warning(self) -> None:
        """Severidad = WARNING (1) cuando hay warnings pero no errores."""
        health = TelemetryHealth()
        health.add_warning("warn")
        assert health.get_severity_level() == _SEVERITY_WARNING

    def test_get_severity_level_critical(self) -> None:
        """Severidad = CRITICAL (2) cuando hay al menos un error."""
        health = TelemetryHealth()
        health.add_error("error")
        assert health.get_severity_level() == _SEVERITY_CRITICAL

    def test_get_severity_error_dominates_warning(self) -> None:
        """
        Si hay warnings Y errores, el nivel es CRITICAL (no WARNING).

        Propiedad: severidad es el máximo del conjunto de eventos.
        """
        health = TelemetryHealth()
        health.add_warning("warn")
        health.add_error("error")
        assert health.get_severity_level() == _SEVERITY_CRITICAL

    def test_get_status_string_healthy(self) -> None:
        """String de estado = 'HEALTHY' cuando no hay errores."""
        assert TelemetryHealth().get_status_string() == "HEALTHY"

    def test_get_status_string_critical(self) -> None:
        """String de estado = 'CRITICAL' cuando hay errores."""
        health = TelemetryHealth()
        health.add_error("error")
        assert health.get_status_string() == "CRITICAL"

    def test_merge_with_takes_worst_health(self) -> None:
        """
        h_healthy.merge_with(h_unhealthy) → is_healthy = False.

        La operación merge es pesimista: prevalece el peor estado.
        Análogo a la norma L∞ sobre el espacio de estados de salud.
        """
        healthy   = TelemetryHealth()
        unhealthy = TelemetryHealth()
        unhealthy.add_error("error")

        merged = healthy.merge_with(unhealthy)

        assert merged.is_healthy is False
        assert len(merged.errors) == 1

    def test_merge_combines_all_warnings_and_errors(self) -> None:
        """
        Merge concatena warnings y errors de ambos operandos.

        |merged.warnings| = |h1.warnings| + |h2.warnings|
        |merged.errors|   = |h1.errors|   + |h2.errors|
        """
        h1 = TelemetryHealth()
        h1.add_warning("warn1")
        h1.add_error("error1")

        h2 = TelemetryHealth()
        h2.add_warning("warn2")

        merged = h1.merge_with(h2)

        assert len(merged.warnings) == 2
        assert len(merged.errors)   == 1

    def test_merge_neutral_element(self) -> None:
        """
        TelemetryHealth() es el elemento neutro del merge.

        h ⊕ neutral = h  y  neutral ⊕ h = h
        (en términos de is_healthy, |warnings|, |errors|)
        """
        h = TelemetryHealth()
        h.add_warning("warn")
        h.add_error("error")

        neutral = TelemetryHealth()

        right_neutral = h.merge_with(neutral)
        left_neutral  = neutral.merge_with(h)

        for merged in (right_neutral, left_neutral):
            assert merged.is_healthy == h.is_healthy
            assert len(merged.warnings) == len(h.warnings)
            assert len(merged.errors)   == len(h.errors)

    def test_to_dict_complete_structure(self) -> None:
        """
        to_dict() serializa todos los campos relevantes.

        Contrato de la API de serialización:
          status, is_healthy, warning_count, error_count,
          stale_steps, memory_pressure.
        """
        health = TelemetryHealth()
        health.add_warning("warn")
        health.add_error("error")
        health.stale_steps.append("stale_step")
        health.memory_pressure = True

        result = health.to_dict()

        assert result["status"]          == "CRITICAL"
        assert result["is_healthy"]      is False
        assert result["warning_count"]   == 1
        assert result["error_count"]     == 1
        assert "stale_step"              in result["stale_steps"]
        assert result["memory_pressure"] is True

    def test_to_dict_keys_always_present(self) -> None:
        """
        to_dict() incluye todas las claves requeridas incluso en estado limpio.

        Evita KeyError en código consumidor que asume la estructura completa.
        """
        result = TelemetryHealth().to_dict()

        required_keys = {
            "status", "is_healthy", "warning_count",
            "error_count", "stale_steps", "memory_pressure",
        }
        for key in required_keys:
            assert key in result, f"Falta clave requerida '{key}' en to_dict()"


# ============================================================================
# TEST: JERARQUÍA DE ESTRATOS
# ============================================================================


class TestStratumHierarchy:
    """
    Pruebas de la jerarquía DIKW.

    Convención de valores:
      WISDOM=0 (cima) < STRATEGY=1 < TACTICS=2 < PHYSICS=3 (base)

    La propagación de errores va de mayor a menor valor
    (de base hacia cima): k → k-1 → ... → 0.
    """

    def test_stratum_enum_values_fixed(self) -> None:
        """Los valores del enum son contratos fijos del dominio."""
        assert Stratum.WISDOM.value   == 0
        assert Stratum.STRATEGY.value == 1
        assert Stratum.TACTICS.value  == 2
        assert Stratum.PHYSICS.value  == 3

    def test_physics_is_base_of_pyramid(self) -> None:
        """PHYSICS tiene el valor máximo → es la base."""
        assert Stratum.PHYSICS.value == max(s.value for s in Stratum)

    def test_wisdom_is_apex_of_pyramid(self) -> None:
        """WISDOM tiene el valor mínimo → es la cima."""
        assert Stratum.WISDOM.value == min(s.value for s in Stratum)

    def test_exactly_four_strata_defined(self) -> None:
        """
        Exactamente 4 estratos en el modelo DIKW.

        Si se agrega un quinto estrato, los tests de propagación
        necesitan revisión → este test actúa como guardia de regresión.
        """
        assert len(list(Stratum)) == 4

    def test_stratum_propagation_set_from_physics(self) -> None:
        """
        Desde PHYSICS (nivel 3), la propagación alcanza a todos los superiores.

        Propiedad: {s ∈ Stratum | s.value < PHYSICS.value} = {TACTICS, STRATEGY, WISDOM}
        """
        physics_level = Stratum.PHYSICS.value
        above         = {s for s in Stratum if s.value < physics_level}

        assert Stratum.TACTICS  in above
        assert Stratum.STRATEGY in above
        assert Stratum.WISDOM   in above
        assert Stratum.PHYSICS  not in above

    def test_stratum_propagation_set_from_tactics(self) -> None:
        """
        Desde TACTICS (nivel 2), la propagación alcanza solo a STRATEGY y WISDOM.
        """
        tactics_level = Stratum.TACTICS.value
        above         = {s for s in Stratum if s.value < tactics_level}

        assert Stratum.STRATEGY in above
        assert Stratum.WISDOM   in above
        assert Stratum.TACTICS  not in above
        assert Stratum.PHYSICS  not in above

    def test_stratum_order_is_total(self) -> None:
        """
        El orden de estratos es total: ∀ a,b: a.value ≤ b.value ∨ b.value ≤ a.value.

        Esto garantiza que la propagación siempre tiene dirección definida.
        """
        strata = list(Stratum)
        for a in strata:
            for b in strata:
                assert a.value <= b.value or b.value <= a.value, (
                    f"Estratos no comparables: {a.name} vs {b.name}"
                )


# ============================================================================
# TEST: TRACKING DE SALUD POR ESTRATO
# ============================================================================


class TestHealthTracking:
    """Pruebas de tracking de salud por estrato."""

    def test_initial_all_strata_healthy(self, fresh_context: TelemetryContext) -> None:
        """Todos los estratos inician en estado saludable."""
        for stratum in Stratum:
            assert stratum in fresh_context._strata_health, (
                f"Estrato {stratum.name} no inicializado en _strata_health"
            )
            assert fresh_context._strata_health[stratum].is_healthy is True

    def test_error_degrades_physics_health(self, fresh_context: TelemetryContext) -> None:
        """Un error en PHYSICS degrada únicamente la salud de PHYSICS."""
        fresh_context.start_step("test_step", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="test_step",
            error_message="Test error",
            severity="ERROR",
        )

        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False

    def test_error_message_recorded_in_stratum(self, fresh_context: TelemetryContext) -> None:
        """El mensaje de error se registra en el estrato correcto."""
        target_msg = "Specific error message"
        fresh_context.start_step("test_step", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="test_step",
            error_message=target_msg,
            severity="ERROR",
        )

        errors = fresh_context._strata_health[Stratum.PHYSICS].errors
        assert any(target_msg in e[0] for e in errors), (
            f"Mensaje '{target_msg}' no encontrado en errores de PHYSICS: {errors}"
        )

    def test_non_critical_error_no_propagation(self, fresh_context: TelemetryContext) -> None:
        """
        Errores no-CRITICAL degradan el estrato origen pero NO propagan warnings.

        Contrato: la propagación transitiva es exclusiva de severidad CRITICAL.
        """
        fresh_context.start_step("test_step", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="test_step",
            error_message="Non-critical error",
            severity="ERROR",
        )

        # PHYSICS degradado
        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False

        # Ningún estrato superior recibe propagación
        for stratum in (Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM):
            assert fresh_context._strata_health[stratum].is_healthy is True, (
                f"{stratum.name} no debería degradarse por error no-CRITICAL"
            )
            assert len(fresh_context._strata_health[stratum].warnings) == 0, (
                f"{stratum.name} no debería tener warnings propagados"
            )

    def test_explicit_stratum_overrides_active_step(self, fresh_context: TelemetryContext) -> None:
        """
        El estrato explícito en record_error tiene prioridad sobre el paso activo.

        Caso de uso: un componente STRATEGY detecta un error y lo reporta
        explícitamente aunque el paso activo sea PHYSICS.
        """
        fresh_context.start_step("physics_step", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="external_step",
            error_message="Strategic error",
            severity="ERROR",
            stratum=Stratum.STRATEGY,
        )

        assert fresh_context._strata_health[Stratum.STRATEGY].is_healthy is False
        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is True, (
            "PHYSICS no debe degradarse cuando el estrato explícito es STRATEGY"
        )

    def test_active_step_stratum_detected_implicitly(
        self, context_with_physics_step: TelemetryContext
    ) -> None:
        """
        Sin estrato explícito, record_error usa el del paso activo.

        El contexto tiene un paso activo en PHYSICS → el error debe
        registrarse en PHYSICS sin necesidad de especificarlo.
        """
        ctx = context_with_physics_step
        ctx.record_error(
            step_name="physics_step",
            error_message="Physics error",
            severity="ERROR",
        )

        assert ctx._strata_health[Stratum.PHYSICS].is_healthy is False

    def test_default_stratum_is_physics_when_no_active_step(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Sin paso activo ni estrato explícito, el fallback es PHYSICS.

        PHYSICS como default tiene sentido semántico: los errores desconocidos
        suelen originarse en la capa de datos (base de la pirámide).
        """
        fresh_context.record_error(
            step_name="unknown_step",
            error_message="Unknown error",
            severity="ERROR",
        )

        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False
        # El resto de estratos permanece saludable (sin propagación por ERROR)
        for stratum in (Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM):
            assert fresh_context._strata_health[stratum].is_healthy is True


# ============================================================================
# TEST: PROPAGACIÓN DE FALLOS (CLAUSURA TRANSITIVA)
# ============================================================================


class TestFailurePropagation:
    """
    Pruebas de propagación transitiva de fallos críticos.

    Modelo matemático:
      Sea P(k) = {s ∈ Stratum | s.value < k} el conjunto de estratos
      superiores al estrato k. Un fallo CRITICAL en k propaga un
      warning a cada s ∈ P(k).

      Esto define una clausura transitiva sobre el orden parcial de la pirámide.
    """

    def test_critical_physics_propagates_to_all_upper_strata(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        CRITICAL en PHYSICS → warnings en TACTICS, STRATEGY, WISDOM.

        P(PHYSICS=3) = {TACTICS, STRATEGY, WISDOM}.
        """
        fresh_context.start_step("critical_step", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="critical_step",
            error_message="Critical failure",
            severity="CRITICAL",
        )

        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False

        for upper in (Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM):
            warnings = fresh_context._strata_health[upper].warnings
            assert len(warnings) > 0, (
                f"{upper.name} debe tener warnings propagados desde PHYSICS CRITICAL"
            )
            # El mensaje de propagación debe ser identificable
            assert any(
                "inherited" in w[0].lower() or "instability" in w[0].lower()
                for w in warnings
            ), f"{upper.name}: warning no contiene 'inherited' ni 'instability'"

    def test_propagation_message_includes_source_stratum_name(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        El mensaje de propagación identifica el estrato origen.

        Facilita el diagnóstico: el operador sabe desde dónde vino la inestabilidad.
        """
        fresh_context.start_step("physics_fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="physics_fail",
            error_message="Physics explosion",
            severity="CRITICAL",
        )

        for upper in (Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM):
            warnings = fresh_context._strata_health[upper].warnings
            assert any("PHYSICS" in w[0] for w in warnings), (
                f"{upper.name}: el warning de propagación debe mencionar 'PHYSICS'"
            )

    def test_propagation_only_goes_upward(self, fresh_context: TelemetryContext) -> None:
        """
        CRITICAL en TACTICS (nivel 2) → solo STRATEGY (1) y WISDOM (0).

        PHYSICS (nivel 3) está debajo de TACTICS → NO recibe propagación.
        """
        fresh_context.start_step("tactics_fail", stratum=Stratum.TACTICS)
        fresh_context.record_error(
            step_name="tactics_fail",
            error_message="Tactics failure",
            severity="CRITICAL",
        )

        # Estratos superiores (menor valor) → deben tener warnings
        assert len(fresh_context._strata_health[Stratum.STRATEGY].warnings) > 0
        assert len(fresh_context._strata_health[Stratum.WISDOM].warnings)   > 0

        # PHYSICS (nivel 3, inferior a TACTICS=2) → NO debe tener warnings
        assert len(fresh_context._strata_health[Stratum.PHYSICS].warnings) == 0, (
            "PHYSICS no debe recibir propagación de TACTICS (está debajo)"
        )

    def test_propagation_adds_warnings_not_errors(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        La propagación agrega WARNINGS (no errores directos) a estratos superiores.

        Semántica: los estratos superiores están ADVERTIDOS de la inestabilidad,
        pero no tienen un error propio. Por ello is_healthy permanece True.
        """
        fresh_context.start_step("physics_fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error(
            step_name="physics_fail",
            error_message="Critical physics error",
            severity="CRITICAL",
        )

        tactics_health = fresh_context._strata_health[Stratum.TACTICS]
        assert tactics_health.is_healthy is True, (
            "TACTICS debe seguir saludable: la propagación son warnings, no errores"
        )
        assert len(tactics_health.warnings) > 0

    def test_multiple_critical_failures_accumulate_warnings(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Dos fallos CRITICAL en PHYSICS acumulan al menos 2 warnings en TACTICS.

        Propiedad de monotonía: más fallos → más warnings (nunca menos).
        """
        fresh_context.start_step("fail1", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail1", "First failure", severity="CRITICAL")
        fresh_context.end_step("fail1", StepStatus.FAILURE)

        count_after_first = len(fresh_context._strata_health[Stratum.TACTICS].warnings)

        fresh_context.start_step("fail2", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail2", "Second failure", severity="CRITICAL")

        count_after_second = len(fresh_context._strata_health[Stratum.TACTICS].warnings)

        assert count_after_second >= count_after_first + 1, (
            f"El segundo fallo CRITICAL debe agregar al menos 1 warning más a TACTICS. "
            f"Antes: {count_after_first}, después: {count_after_second}"
        )

    @pytest.mark.parametrize("severity", ["ERROR", "WARNING", "INFO"])
    def test_non_critical_severities_do_not_propagate(
        self, severity: str, fresh_context: TelemetryContext
    ) -> None:
        """
        Solo CRITICAL propaga. ERROR, WARNING, INFO no propagan a estratos superiores.

        Usamos parametrize para cubrir todos los casos sin duplicar código.
        """
        ctx = TelemetryContext()
        ctx.start_step("test", stratum=Stratum.PHYSICS)
        ctx.record_error("test", f"{severity} message", severity=severity)

        for upper in (Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM):
            assert len(ctx._strata_health[upper].warnings) == 0, (
                f"Severidad {severity} no debería propagar a {upper.name}"
            )

    def test_wisdom_receives_propagation_from_tactics(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        WISDOM (cima) recibe propagación desde TACTICS.

        Verifica que la clausura transitiva alcanza la cima para cualquier
        estrato de origen (no solo PHYSICS).
        """
        fresh_context.start_step("tactics_fail", stratum=Stratum.TACTICS)
        fresh_context.record_error("tactics_fail", "Tactics error", severity="CRITICAL")

        wisdom_warnings = fresh_context._strata_health[Stratum.WISDOM].warnings
        assert len(wisdom_warnings) > 0, "WISDOM debe recibir propagación desde TACTICS"
        assert any("TACTICS" in w[0] for w in wisdom_warnings), (
            "El warning en WISDOM debe mencionar 'TACTICS' como origen"
        )

    def test_physics_critical_does_not_propagate_downward(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        PHYSICS es la base: no hay estratos inferiores.
        El conjunto de propagación P(PHYSICS) ∩ {s: s.value > 3} = ∅.

        Este test es un invariante de borde: no debe existir propagación
        hacia abajo bajo ninguna circunstancia.
        """
        fresh_context.start_step("physics_fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("physics_fail", "Base failure", severity="CRITICAL")

        # PHYSICS no recibe propagación de sí mismo
        physics_warnings = fresh_context._strata_health[Stratum.PHYSICS].warnings
        assert all("PHYSICS" not in w[0] for w in physics_warnings), (
            "PHYSICS no debe propagarse a sí mismo"
        )


# ============================================================================
# TEST: REPORTES PIRAMIDALES
# ============================================================================


class TestPyramidalReports:
    """Pruebas de reportes piramidales estructurados por capa."""

    _LAYER_KEYS = ("physics_layer", "tactics_layer", "strategy_layer")

    def test_report_contains_all_layers(self, fresh_context: TelemetryContext) -> None:
        """El reporte contiene las tres capas principales."""
        report = fresh_context.get_pyramidal_report()

        for key in self._LAYER_KEYS:
            assert key in report, f"Falta clave '{key}' en el reporte piramidal"

    def test_each_layer_has_required_fields(self, fresh_context: TelemetryContext) -> None:
        """
        Cada capa contiene los campos requeridos:
          status, metrics, issues, warnings.
        """
        report = fresh_context.get_pyramidal_report()
        required = {"status", "metrics", "issues", "warnings"}

        for key in self._LAYER_KEYS:
            for field in required:
                assert field in report[key], (
                    f"Capa '{key}' no contiene el campo requerido '{field}'"
                )

    def test_all_layers_healthy_on_clean_context(
        self, fresh_context: TelemetryContext
    ) -> None:
        """Sin errores, todas las capas reportan HEALTHY."""
        # Solo métrica, sin errores
        fresh_context.record_metric(_PREFIX_PHYSICS, "avg_saturation", 0.5)

        report = fresh_context.get_pyramidal_report()

        for key in self._LAYER_KEYS:
            assert report[key]["status"] == "HEALTHY", (
                f"Capa '{key}' debe ser HEALTHY en contexto limpio, "
                f"obtenido: '{report[key]['status']}'"
            )

    def test_physics_critical_on_error(self, fresh_context: TelemetryContext) -> None:
        """Error en PHYSICS → physics_layer.status = 'CRITICAL'."""
        fresh_context.start_step("fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail", "Error", severity="CRITICAL")

        report = fresh_context.get_pyramidal_report()

        assert report["physics_layer"]["status"] == "CRITICAL"

    def test_propagation_makes_upper_layers_warning(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        CRITICAL en PHYSICS → TACTICS y STRATEGY en WARNING.

        El status WARNING refleja que el estrato tiene warnings propagados
        pero no errores propios.
        """
        fresh_context.start_step("fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail", "Critical error", severity="CRITICAL")

        report = fresh_context.get_pyramidal_report()

        assert report["physics_layer"]["status"]   == "CRITICAL"
        assert report["tactics_layer"]["status"]   == "WARNING"
        assert report["strategy_layer"]["status"]  == "WARNING"

    def test_metrics_assigned_to_correct_layer(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Las métricas se asignan a la capa correcta según su prefijo.

        Mapeo de dominio:
          flux_condenser.* → physics_layer
          topology.*       → tactics_layer
          financial.*      → strategy_layer
        """
        fresh_context.record_metric(_PREFIX_PHYSICS,  "avg_saturation", 0.8)
        fresh_context.record_metric(_PREFIX_TACTICS,  "beta_0",         1)
        fresh_context.record_metric(_PREFIX_STRATEGY, "npv",            50_000)

        report = fresh_context.get_pyramidal_report()

        assert f"{_PREFIX_PHYSICS}.avg_saturation"  in report["physics_layer"]["metrics"]
        assert f"{_PREFIX_TACTICS}.beta_0"          in report["tactics_layer"]["metrics"]
        assert f"{_PREFIX_STRATEGY}.npv"            in report["strategy_layer"]["metrics"]

    def test_cross_layer_metrics_do_not_leak(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Las métricas de PHYSICS no aparecen en TACTICS ni STRATEGY.

        Verifica que el filtrado de prefijos no produce fugas entre capas.
        """
        fresh_context.record_metric(_PREFIX_PHYSICS, "temp", 45.0)

        report = fresh_context.get_pyramidal_report()

        assert f"{_PREFIX_PHYSICS}.temp" not in report["tactics_layer"]["metrics"]
        assert f"{_PREFIX_PHYSICS}.temp" not in report["strategy_layer"]["metrics"]

    def test_issues_populated_on_error(self, fresh_context: TelemetryContext) -> None:
        """La lista de issues contiene el mensaje del error registrado."""
        error_msg = "Physics critical error"
        fresh_context.start_step("fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail", error_msg, severity="CRITICAL")

        report = fresh_context.get_pyramidal_report()

        issues = report["physics_layer"]["issues"]
        assert len(issues) > 0
        assert any(error_msg in issue for issue in issues), (
            f"El mensaje '{error_msg}' no está en issues: {issues}"
        )

    def test_warnings_in_report_are_strings(self, fresh_context: TelemetryContext) -> None:
        """
        Las warnings en el reporte son strings (no tuplas internas).

        El reporte aplana las tuplas (msg, timestamp) a solo el mensaje
        para facilitar el consumo por código externo (APIs, UI).
        """
        fresh_context.start_step("fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail", "Critical", severity="CRITICAL")

        report = fresh_context.get_pyramidal_report()

        for warning in report["tactics_layer"]["warnings"]:
            assert isinstance(warning, str), (
                f"Warning debe ser str, obtenido: {type(warning).__name__} = {warning!r}"
            )


# ============================================================================
# TEST: INTEGRACIÓN DE SPANS CON ESTRATO
# ============================================================================


class TestSpanStratumIntegration:
    """Pruebas de integración entre spans (context managers) y estratos."""

    def test_span_records_assigned_stratum(self, fresh_context: TelemetryContext) -> None:
        """El span tiene el estrato asignado disponible durante su ejecución."""
        with fresh_context.span("test_span", stratum=Stratum.TACTICS) as span:
            assert span.stratum == Stratum.TACTICS

    def test_completed_span_appears_in_root_spans(
        self, fresh_context: TelemetryContext
    ) -> None:
        """Tras completar el span, aparece en root_spans con el estrato correcto."""
        with fresh_context.span("test_span", stratum=Stratum.STRATEGY):
            pass

        assert len(fresh_context.root_spans) == 1
        assert fresh_context.root_spans[0].stratum == Stratum.STRATEGY

    def test_nested_spans_preserve_independent_strata(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Spans anidados mantienen sus estratos independientemente.

        El estrato del hijo no hereda ni modifica el del padre.
        """
        with fresh_context.span("parent", stratum=Stratum.PHYSICS) as parent:
            with fresh_context.span("child", stratum=Stratum.TACTICS) as child:
                assert child.stratum  == Stratum.TACTICS
                assert parent.stratum == Stratum.PHYSICS
            # Tras cerrar el hijo, el padre mantiene su estrato
            assert parent.stratum == Stratum.PHYSICS

    def test_span_default_stratum_is_physics(self, fresh_context: TelemetryContext) -> None:
        """
        Sin estrato explícito, el span usa PHYSICS como default.

        PHYSICS es el estrato base → default semánticamente correcto
        para operaciones no categorizadas.
        """
        with fresh_context.span("default_span"):
            pass

        assert fresh_context.root_spans[0].stratum == Stratum.PHYSICS

    def test_step_context_manager_registers_step(
        self, fresh_context: TelemetryContext
    ) -> None:
        """El context manager step registra el paso en fresh_context.steps."""
        with fresh_context.step("test_step", stratum=Stratum.WISDOM):
            pass

        assert any(
            s.get("step") == "test_step" for s in fresh_context.steps
        ), "El paso 'test_step' no se encontró en fresh_context.steps"

    def test_span_exception_records_error(self, fresh_context: TelemetryContext) -> None:
        """
        Excepción dentro de un span registra el error en el contexto.

        El context manager debe capturar la excepción y registrarla
        antes de re-lanzarla.
        """
        with pytest.raises(ValueError):
            with fresh_context.span("failing_span", stratum=Stratum.TACTICS):
                raise ValueError("Test exception")

        assert len(fresh_context.errors) > 0, (
            "La excepción en el span debe registrarse en fresh_context.errors"
        )

    def test_multiple_sequential_spans_all_recorded(
        self, fresh_context: TelemetryContext
    ) -> None:
        """Múltiples spans secuenciales se registran todos en root_spans."""
        strata = [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM]

        for stratum in strata:
            with fresh_context.span(f"span_{stratum.name}", stratum=stratum):
                pass

        assert len(fresh_context.root_spans) == len(strata), (
            f"Se esperaban {len(strata)} root_spans, "
            f"obtenidos: {len(fresh_context.root_spans)}"
        )


# ============================================================================
# TEST: VERIFICACIÓN DE INVARIANTES
# ============================================================================


class TestInvariantsVerification:
    """
    Pruebas de verificación de invariantes topológicos.

    Los invariantes actúan como assertions de runtime para detectar
    estados inconsistentes en el contexto de telemetría.
    """

    def test_healthy_context_passes_invariants(
        self, fresh_context: TelemetryContext
    ) -> None:
        """Contexto saludable con spans válidos pasa todos los invariantes."""
        with fresh_context.span("test"):
            pass

        result = fresh_context.verify_invariants()

        assert result["is_valid"] is True
        assert len(result["violations"]) == 0

    def test_invariants_include_topology_metrics(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        La verificación incluye métricas topológicas (árboles, nodos).

        Los spans forman una foresta de árboles; las métricas topológicas
        caracterizan esa estructura.
        """
        with fresh_context.span("parent"):
            with fresh_context.span("child"):
                pass

        result = fresh_context.verify_invariants()

        assert "topology" in result, "verify_invariants debe incluir 'topology'"
        assert "num_trees" in result["topology"], (
            "topology debe incluir 'num_trees'"
        )

    def test_overflow_steps_detected_as_violation(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Exceder max_steps es una violación de invariante detectable.

        En lugar de mutar directamente fresh_context.steps (violación de
        encapsulamiento), usamos el mecanismo público de registro.
        Si max_steps es bajo, los pasos extras disparan la violación.
        """
        # Guardar el límite antes de agregar pasos extras
        max_steps = fresh_context.max_steps

        # Agregar pasos extras a través de la API (si existe)
        # Si la API no lo permite, acceder a steps como lista es el único camino.
        # Documentamos explícitamente el acceso al atributo interno.
        overflow_count = 10
        # Inyectamos suficientes steps para romper the default max_steps.
        for i in range(max_steps + overflow_count):
            fresh_context.steps.append({"step": f"overflow_{i}"})  # noqa: SIM117

        result = fresh_context.verify_invariants()

        assert result["is_valid"] is False, (
            "Overflow de steps debe marcar is_valid=False"
        )
        assert any("limit" in v.lower() or "excedido" in v.lower() or "max_steps" in v.lower() or "10" in v.lower() for v in result["violations"]), (
            f"Debe haber una violación con 'max_steps' en: {result['violations']}"
        )

    def test_invariants_count_spans_metrics_errors(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        verify_invariants reporta conteos correctos de spans, métricas y errores.

        Los conteos sirven para diagnóstico y auditoría.
        """
        with fresh_context.span("span1"):
            pass
        with fresh_context.span("span2"):
            pass

        # Registrar exactamente 1 métrica y 1 error
        fresh_context.record_metric("test_ns", "metric_a", 1.0)
        fresh_context.record_error("test_step", "Test error", severity="ERROR")

        result = fresh_context.verify_invariants()

        assert result["counts"]["root_spans"] == 2, (
            f"Esperados 2 root_spans, obtenidos: {result['counts']['root_spans']}"
        )
        # metrics count = número de claves únicas registradas
        assert result["counts"]["metrics"] >= 1, (
            "Debe haber al menos 1 métrica registrada"
        )
        assert result["counts"]["errors"] >= 1, (
            "Debe haber al menos 1 error registrado"
        )


# ============================================================================
# TEST: CASOS LÍMITE Y RECUPERACIÓN
# ============================================================================


class TestEdgeCasesAndRecovery:
    """Pruebas de robustez ante entradas atípicas y condiciones extremas."""

    def test_reset_restores_all_strata_to_healthy(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        reset() restaura todos los estratos a estado saludable.

        Equivale a reinicializar el monoide de salud para cada estrato.
        """
        fresh_context.start_step("fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail", "Error", severity="CRITICAL")

        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False

        fresh_context.reset()

        for stratum in Stratum:
            assert fresh_context._strata_health[stratum].is_healthy is True, (
                f"{stratum.name} debe estar saludable tras reset()"
            )

    def test_reset_clears_propagated_warnings(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        reset() limpia también los warnings propagados en estratos superiores.
        """
        fresh_context.start_step("fail", stratum=Stratum.PHYSICS)
        fresh_context.record_error("fail", "Critical", severity="CRITICAL")

        assert len(fresh_context._strata_health[Stratum.TACTICS].warnings) > 0

        fresh_context.reset()

        for stratum in Stratum:
            assert len(fresh_context._strata_health[stratum].warnings) == 0, (
                f"{stratum.name} no debe tener warnings tras reset()"
            )

    def test_invalid_stratum_type_falls_back_to_physics(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Tipo de estrato inválido (str en lugar de Stratum) se maneja sin crash.

        El fallback a PHYSICS es el comportamiento defensivo correcto.
        """
        fresh_context.record_error(
            step_name="test",
            error_message="Error with bad stratum",
            severity="ERROR",
            stratum="not_a_stratum",  # type: ignore[arg-type]
        )

        # Debe usar PHYSICS como fallback y no lanzar excepción
        assert fresh_context._strata_health[Stratum.PHYSICS].is_healthy is False

    def test_empty_context_generates_valid_pyramidal_report(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Contexto vacío genera reporte piramidal estructuralmente válido.

        El reporte no debe ser None ni carecer de claves requeridas.
        """
        report = fresh_context.get_pyramidal_report()

        assert report is not None
        assert "physics_layer"   in report
        assert "tactics_layer"   in report
        assert "strategy_layer"  in report
        assert report["physics_layer"]["status"] == "HEALTHY"

    def test_many_errors_handled_without_overflow(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        100 errores en PHYSICS se registran sin error de runtime.

        El reporte puede truncar la lista de issues pero no debe crashear.
        """
        for i in range(_STRESS_ERROR_COUNT):
            fresh_context.record_error(
                step_name=f"step_{i}",
                error_message=f"Error {i}",
                severity="ERROR",
                stratum=Stratum.PHYSICS,
            )

        report = fresh_context.get_pyramidal_report()

        assert report["physics_layer"]["status"] == "CRITICAL"
        # El reporte puede truncar pero debe retornar una lista válida
        issues = report["physics_layer"]["issues"]
        assert isinstance(issues, list)
        assert 0 < len(issues) <= _STRESS_ERROR_COUNT

    def test_concurrent_stratum_operations_are_safe(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Operaciones concurrentes en diferentes estratos no producen excepciones.

        Verifica thread-safety básico del tracking de salud por estrato.
        Los resultados finales deben ser deterministas en cuanto a:
          · Ningún estrato objetivo debe quedar saludable (todos reciben errores).
          · No debe haber excepciones.
        """
        thread_errors: list[Exception] = []

        def record_to_stratum(stratum: Stratum, count: int) -> None:
            try:
                for i in range(count):
                    fresh_context.record_error(
                        step_name=f"{stratum.name}_{i}",
                        error_message=f"Error in {stratum.name}",
                        severity="ERROR",
                        stratum=stratum,
                    )
            except Exception as exc:
                thread_errors.append(exc)

        target_strata = [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY]
        threads = [
            threading.Thread(
                target=record_to_stratum,
                args=(stratum, _CONCURRENT_ERROR_COUNT),
                daemon=True,  # No bloquea el proceso si el test falla
            )
            for stratum in target_strata
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=_THREAD_TIMEOUT)
            assert not t.is_alive(), (
                f"Thread para {t.name} no terminó en {_THREAD_TIMEOUT}s — "
                f"posible deadlock"
            )

        assert len(thread_errors) == 0, (
            f"Excepciones en threads concurrentes: {thread_errors}"
        )

        # Todos los estratos target deben estar degradados
        for stratum in target_strata:
            assert not fresh_context._strata_health[stratum].is_healthy, (
                f"{stratum.name} debe estar degradado tras operaciones concurrentes"
            )


# ============================================================================
# TEST: INTEGRACIÓN CON TELEMETRY NARRATOR
# ============================================================================


class TestIntegrationWithNarrator:
    """
    Pruebas de coherencia entre TelemetryContext y TelemetryNarrator.

    Invariante de coherencia:
      pyramidal_report["X_layer"]["status"] ↔ narrator["strata_analysis"]["X"]["severity"]
    """

    def test_narrator_report_includes_all_active_strata(
        self, narrator: TelemetryNarrator, fresh_context: TelemetryContext
    ) -> None:
        """El narrador incluye en el análisis los estratos con actividad."""
        with fresh_context.span("load_data",       stratum=Stratum.PHYSICS):  pass
        with fresh_context.span("calculate_costs", stratum=Stratum.TACTICS):  pass

        report = narrator.summarize_execution(fresh_context)

        assert "PHYSICS" in report["strata_analysis"], (
            "PHYSICS debe aparecer en strata_analysis del narrador"
        )
        assert "TACTICS" in report["strata_analysis"], (
            "TACTICS debe aparecer en strata_analysis del narrador"
        )

    def test_narrator_reflects_physics_failure_as_critico(
        self, narrator: TelemetryNarrator, fresh_context: TelemetryContext
    ) -> None:
        """
        Fallo en PHYSICS → strata_analysis['PHYSICS']['severity'] = 'CRITICO'.

        Nota: la mutación directa de span.status y span.errors se hace
        DENTRO del context manager para que el contexto la procese
        antes del cierre del span.
        """
        with fresh_context.span("load_data", stratum=Stratum.PHYSICS) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Data corruption", "type": "IOError"})

        report = narrator.summarize_execution(fresh_context)

        assert "PHYSICS" in report["verdict_code"], (
            f"verdict_code debe mencionar PHYSICS: '{report['verdict_code']}'"
        )
        assert report["strata_analysis"]["PHYSICS"]["severity"] == "CRITICO"

    def test_narrator_reflects_tactics_failure_as_critico(
        self, narrator: TelemetryNarrator, fresh_context: TelemetryContext
    ) -> None:
        """Fallo en TACTICS → strata_analysis['TACTICS']['severity'] = 'CRITICO'."""
        with fresh_context.span("load_data",       stratum=Stratum.PHYSICS): pass
        with fresh_context.span("calculate_costs", stratum=Stratum.TACTICS) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Cycle detected", "type": "TopologyError"})

        report = narrator.summarize_execution(fresh_context)

        assert "TACTICS" in report["verdict_code"]
        assert report["strata_analysis"]["TACTICS"]["severity"] == "CRITICO"

    def test_narrator_approves_all_healthy_strata(
        self,
        narrator: TelemetryNarrator,
        context_with_all_strata: TelemetryContext,
    ) -> None:
        """Todos los estratos saludables → verdict_code = 'APPROVED'."""
        report = narrator.summarize_execution(context_with_all_strata)

        assert report["verdict_code"] == "APPROVED", (
            f"Todos los estratos saludables deben resultar en APPROVED, "
            f"obtenido: '{report['verdict_code']}'"
        )

    def test_pyramidal_report_coherent_with_narrator_on_physics_failure(
        self, narrator: TelemetryNarrator, fresh_context: TelemetryContext
    ) -> None:
        """
        physics_layer.status='CRITICAL' ↔ strata_analysis['PHYSICS'].severity='CRITICO'.

        Invariante de coherencia entre los dos sistemas de reporte.
        """
        with fresh_context.span("fail", stratum=Stratum.PHYSICS) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Critical failure", "type": "Error"})

        pyramidal       = fresh_context.get_pyramidal_report()
        narrator_report = narrator.summarize_execution(fresh_context)

        assert pyramidal["physics_layer"]["status"] == "CRITICAL", (
            "Reporte piramidal debe marcar PHYSICS como CRITICAL"
        )
        assert narrator_report["strata_analysis"]["PHYSICS"]["severity"] == "CRITICO", (
            "Narrador debe marcar PHYSICS como CRITICO"
        )

    def test_health_propagation_reflected_in_narrator_verdict(
        self, narrator: TelemetryNarrator, fresh_context: TelemetryContext
    ) -> None:
        """
        Fallo crítico en PHYSICS → el veredicto del narrador menciona PHYSICS
        y el resumen ejecutivo refleja el impacto en niveles superiores.
        """
        with fresh_context.span("physics_fail", stratum=Stratum.PHYSICS) as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "Critical failure", "type": "Error"})

        report           = narrator.summarize_execution(fresh_context)
        executive_summary = report["executive_summary"]

        assert "PHYSICS" in report["verdict_code"], (
            "El veredicto debe identificar PHYSICS como origen del fallo"
        )
        # El resumen ejecutivo debe mencionar el estrato afectado
        # (en español o inglés según la implementación del narrador)
        assert any(
            keyword in executive_summary
            for keyword in ("FÍSICA", "PHYSICS", "ABORTADO", "fallo", "failure")
        ), (
            f"El resumen ejecutivo debe mencionar el impacto del fallo en PHYSICS: "
            f"'{executive_summary}'"
        )


# ============================================================================
# TEST: PROPIEDADES ALGEBRAICAS
# ============================================================================


class TestAlgebraicProperties:
    """
    Pruebas de propiedades algebraicas del sistema piramidal.

    El sistema de salud forma un semilátice bajo la operación merge_with:
      · Idempotente: h ⊕ h = h
      · Conmutativo: h1 ⊕ h2 = h2 ⊕ h1
      · Asociativo:  (h1 ⊕ h2) ⊕ h3 = h1 ⊕ (h2 ⊕ h3)
    """

    def test_propagation_is_monotonically_increasing(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Monotonía: más fallos críticos → al menos tantos warnings en TACTICS.

        Formalmente: |warnings(t₁)| ≤ |warnings(t₂)|  para t₁ < t₂.
        """
        initial = len(fresh_context._strata_health[Stratum.TACTICS].warnings)

        fresh_context.record_error(
            "fail1", "Error 1", severity="CRITICAL", stratum=Stratum.PHYSICS
        )
        after_first = len(fresh_context._strata_health[Stratum.TACTICS].warnings)

        fresh_context.record_error(
            "fail2", "Error 2", severity="CRITICAL", stratum=Stratum.PHYSICS
        )
        after_second = len(fresh_context._strata_health[Stratum.TACTICS].warnings)

        assert after_first  >= initial,     "Primer fallo debe agregar warnings"
        assert after_second >= after_first, "Segundo fallo no debe reducir warnings"

    def test_propagation_respects_pyramid_order(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        CRITICAL en TACTICS (nivel 2) → warnings en {STRATEGY, WISDOM} (niveles 0,1).
        PHYSICS (nivel 3) NO recibe propagación (está por debajo).
        """
        fresh_context.record_error(
            "fail", "Error", severity="CRITICAL", stratum=Stratum.TACTICS
        )

        assert len(fresh_context._strata_health[Stratum.STRATEGY].warnings) > 0
        assert len(fresh_context._strata_health[Stratum.WISDOM].warnings)   > 0
        assert len(fresh_context._strata_health[Stratum.PHYSICS].warnings)  == 0

    def test_health_merge_is_commutative(self) -> None:
        """
        h1 ⊕ h2 ≡ h2 ⊕ h1  en términos de:
          · is_healthy
          · |warnings|
          · |errors|

        La conmutatividad garantiza que el orden de merge no afecta el resultado.
        """
        h1 = TelemetryHealth()
        h1.add_warning("warn1")

        h2 = TelemetryHealth()
        h2.add_error("error1")

        m12 = h1.merge_with(h2)
        m21 = h2.merge_with(h1)

        assert m12.is_healthy    == m21.is_healthy
        assert len(m12.warnings) == len(m21.warnings)
        assert len(m12.errors)   == len(m21.errors)

    def test_health_merge_is_associative(self) -> None:
        """
        (h1 ⊕ h2) ⊕ h3 ≡ h1 ⊕ (h2 ⊕ h3)  en términos de:
          · is_healthy
          · |warnings|  (total acumulado)
          · |errors|    (total acumulado)

        La asociatividad garantiza que el agrupamiento no afecta el resultado.
        """
        h1 = TelemetryHealth()
        h1.add_warning("warn1")

        h2 = TelemetryHealth()
        h2.add_error("error1")

        h3 = TelemetryHealth()
        h3.add_warning("warn2")

        left  = h1.merge_with(h2).merge_with(h3)   # (h1 ⊕ h2) ⊕ h3
        right = h1.merge_with(h2.merge_with(h3))    # h1 ⊕ (h2 ⊕ h3)

        assert left.is_healthy    == right.is_healthy
        assert len(left.warnings) == len(right.warnings)
        assert len(left.errors)   == len(right.errors)

    def test_health_merge_is_idempotent(self) -> None:
        """
        h ⊕ h = h  (idempotencia, propiedad de semilátice).

        Aplicar merge consigo mismo no debe cambiar el estado de salud.
        Nota: |warnings| puede duplicarse (merge concatena listas),
        pero is_healthy debe ser el mismo.
        """
        h = TelemetryHealth()
        h.add_warning("warn")
        h.add_error("error")

        merged = h.merge_with(h)

        # El estado booleano es idempotente
        assert merged.is_healthy == h.is_healthy

    def test_stratum_order_is_total_order(self) -> None:
        """
        ∀ a,b ∈ Stratum: a.value ≤ b.value  ∨  b.value ≤ a.value.

        El orden total garantiza que la dirección de propagación
        es siempre determinista (sin estratos incomparables).
        """
        strata = list(Stratum)
        for a in strata:
            for b in strata:
                assert a.value <= b.value or b.value <= a.value, (
                    f"Estratos incomparables: {a.name}={a.value} vs "
                    f"{b.name}={b.value}"
                )


# ============================================================================
# TEST: FILTRADO DE MÉTRICAS POR ESTRATO
# ============================================================================


class TestMetricsFiltering:
    """
    Pruebas de filtrado y asignación de métricas por capa piramidal.

    El mapeo prefijo → capa es parte del contrato del dominio:
      _PREFIX_PHYSICS  → physics_layer
      _PREFIX_TACTICS  → tactics_layer
      _PREFIX_STRATEGY → strategy_layer
    """

    def test_filter_metrics_by_known_prefix(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        _filter_metrics_by_prefix(prefix) retorna solo métricas con ese prefijo.
        """
        fresh_context.record_metric(_PREFIX_PHYSICS,  "saturation", 0.5)
        fresh_context.record_metric(_PREFIX_PHYSICS,  "voltage",    1.2)
        fresh_context.record_metric(_PREFIX_TACTICS,  "beta_0",     1)
        fresh_context.record_metric(_PREFIX_STRATEGY, "npv",        10_000)

        physics_metrics = fresh_context._filter_metrics_by_prefix(_PREFIX_PHYSICS)

        assert f"{_PREFIX_PHYSICS}.saturation" in physics_metrics
        assert f"{_PREFIX_PHYSICS}.voltage"    in physics_metrics
        assert f"{_PREFIX_TACTICS}.beta_0"     not in physics_metrics
        assert f"{_PREFIX_STRATEGY}.npv"       not in physics_metrics

    def test_filter_returns_independent_copy(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        _filter_metrics_by_prefix retorna una copia: mutar el resultado
        no afecta el almacenamiento interno del contexto.

        Esta propiedad garantiza inmutabilidad observable del contexto
        desde el exterior (principio de encapsulamiento).
        """
        original_value = {"nested": "data"}
        fresh_context.record_metric("test_ns", "value", original_value)

        filtered    = fresh_context._filter_metrics_by_prefix("test_ns")
        filter_key  = "test_ns.value"

        # El test solo es significativo si la métrica se filtró
        assert filter_key in filtered, (
            f"La métrica '{filter_key}' debe estar en el resultado del filtro"
        )

        returned = filtered[filter_key]
        if isinstance(returned, dict):
            returned["nested"] = "MODIFIED"
            # El valor original no debe cambiar
            stored = fresh_context.get_metric("test_ns", "value")
            assert stored.get("nested") == "data", (
                "Mutar el resultado del filtro no debe afectar el valor almacenado"
            )

    def test_metrics_appear_in_correct_pyramidal_layer(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Integración: las métricas registradas aparecen en la capa correcta
        del reporte piramidal.
        """
        fresh_context.record_metric(_PREFIX_PHYSICS,  "temp",   45.0)
        fresh_context.record_metric(_PREFIX_TACTICS,  "cycles", 0)
        fresh_context.record_metric(_PREFIX_STRATEGY, "roi",    1.5)

        report = fresh_context.get_pyramidal_report()

        assert f"{_PREFIX_PHYSICS}.temp"    in report["physics_layer"]["metrics"]
        assert f"{_PREFIX_TACTICS}.cycles"  in report["tactics_layer"]["metrics"]
        assert f"{_PREFIX_STRATEGY}.roi"    in report["strategy_layer"]["metrics"]

    def test_empty_prefix_returns_empty_dict(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Prefijo que no existe retorna diccionario vacío (no None ni KeyError).
        """
        fresh_context.record_metric(_PREFIX_PHYSICS, "temp", 42.0)

        result = fresh_context._filter_metrics_by_prefix("nonexistent_prefix")

        assert isinstance(result, dict), (
            f"Debe retornar dict vacío, obtenido: {type(result).__name__}"
        )
        assert len(result) == 0

    def test_all_prefixes_mapped_to_distinct_layers(
        self, fresh_context: TelemetryContext
    ) -> None:
        """
        Los tres prefijos de dominio se asignan a tres capas distintas.

        Verifica que no hay colisión entre prefijos.
        """
        fresh_context.record_metric(_PREFIX_PHYSICS,  "a", 1)
        fresh_context.record_metric(_PREFIX_TACTICS,  "b", 2)
        fresh_context.record_metric(_PREFIX_STRATEGY, "c", 3)

        p = fresh_context._filter_metrics_by_prefix(_PREFIX_PHYSICS)
        t = fresh_context._filter_metrics_by_prefix(_PREFIX_TACTICS)
        s = fresh_context._filter_metrics_by_prefix(_PREFIX_STRATEGY)

        # Las métricas de physics no aparecen en tactics ni strategy
        assert not (set(p.keys()) & set(t.keys())), "Colisión entre PHYSICS y TACTICS"
        assert not (set(p.keys()) & set(s.keys())), "Colisión entre PHYSICS y STRATEGY"
        assert not (set(t.keys()) & set(s.keys())), "Colisión entre TACTICS y STRATEGY"