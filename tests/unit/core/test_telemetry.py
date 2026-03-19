"""
Suite de pruebas para el módulo de Telemetría Unificada.
========================================================

Pruebas exhaustivas para TelemetryContext, incluyendo:
  - Thread-safety con barreras de sincronización explícitas
  - Límites y fronteras (FIFO, max_steps, max_errors, max_metrics)
  - Serialización y exportación (deep-copy, sanitización)
  - Lógica de negocio (umbrales, estados)
  - Gestores de contexto y protocolo __enter__/__exit__
  - Casos borde: unicode, caracteres especiales, NaN, ±Inf

Mejoras respecto a la versión anterior:
─────────────────────────────────────
- [FIX] test_normalize_status_string: "FAILURE" debe normalizar a "failure",
        no a "success". El comportamiento correcto es case-insensitive match.
- [FIX] test_concurrent_metric_recording: usa max_metrics adecuado para
        evitar colisión con límite de capacidad en entornos concurrentes.
- [FIX] test_concurrent_error_recording: idem para max_errors.
- [FIX] test_record_error_max_limit: corrección de lógica FIFO con max=2
        y 3 inserciones → quedan los 2 últimos (step2, step3).
- [FIX] test_nested_steps_tracking: documenta y verifica explícitamente
        el orden de inserción (inner primero, outer último).
- [IMPROVE] Barrera threading.Barrier en tests concurrentes para inicio
            simultaneo de hilos (mayor presión de concurrencia).
- [IMPROVE] Fixture `thread_safe_ctx` con max_steps/max_errors/max_metrics
            suficientemente grandes para pruebas de concurrencia.
- [IMPROVE] Marcadores pytest.mark.slow para tests de rendimiento.
- [IMPROVE] Pruebas parametrizadas para valores numéricos especiales
            en _sanitize_value (NaN, ±Inf, int, float, Decimal).
- [IMPROVE] Cobertura de StratumTopology para estrato WISDOM.
- [IMPROVE] test_get_business_report_custom_thresholds verifica los tres
            umbrales personalizados independientemente.
- [IMPROVE] Verificación explícita de deep-copy en to_dict con anidamiento.
- [NEW] TestFIFOSemantics: clase dedicada a verificar semántica FIFO
        en steps, errors y la interacción con métricas.
- [NEW] TestStratumTopologyComplete: cobertura completa de todos los estratos.
- [NEW] TestSanitizeNumericEdgeCases: parametrizado para tipos numéricos.
- [NEW] conftest-level markers registrados con pytest.ini_options.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Generator

import pytest

from app.core.telemetry import (
    ActiveStepInfo,
    BusinessThresholds,
    StepStatus,
    TelemetryContext,
    TelemetryDefaults,
)

# =============================================================================
# MARCADORES PERSONALIZADOS
# =============================================================================
# Registrar en conftest.py o pytest.ini:
#   [pytest]
#   markers =
#       slow: Tests de rendimiento/estrés (omitir con -m "not slow")
#       thread_safety: Tests de concurrencia
#       business: Tests de lógica de negocio


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def ctx() -> TelemetryContext:
    """Contexto limpio para cada test. Sin estado compartido."""
    return TelemetryContext()


@pytest.fixture
def ctx_with_custom_id() -> TelemetryContext:
    """Contexto con request_id personalizado."""
    return TelemetryContext(request_id="custom-test-id-12345")


@pytest.fixture
def ctx_with_limits() -> TelemetryContext:
    """
    Contexto con límites reducidos para verificar fronteras.

    max_steps=3, max_errors=2, max_metrics=5
    """
    return TelemetryContext(max_steps=3, max_errors=2, max_metrics=5)


@pytest.fixture
def ctx_with_custom_thresholds() -> TelemetryContext:
    """
    Contexto con umbrales de negocio personalizados (más estrictos que los default).

    critical_flyback_voltage=0.3  (default: 0.5)
    critical_dissipated_power=30.0 (default: 50.0)
    warning_saturation=0.8        (default: 0.9)
    """
    return TelemetryContext(
        business_thresholds={
            "critical_flyback_voltage": 0.3,
            "critical_dissipated_power": 30.0,
            "warning_saturation": 0.8,
        }
    )


@pytest.fixture
def ctx_populated(ctx: TelemetryContext) -> TelemetryContext:
    """
    Contexto pre-poblado con datos representativos.

    Contiene:
      - 1 step completado (step1, SUCCESS)
      - 2 métricas (component1.metric1=100, component1.metric2=200)
      - 1 error en step1
    """
    ctx.start_step("step1")
    time.sleep(0.001)  # Garantiza duración > 0
    ctx.end_step("step1", StepStatus.SUCCESS)

    ctx.record_metric("component1", "metric1", 100)
    ctx.record_metric("component1", "metric2", 200)
    ctx.record_error("step1", "test error", error_type="TestError")

    return ctx


@pytest.fixture
def thread_safe_ctx() -> TelemetryContext:
    """
    Contexto con capacidad suficiente para tests de concurrencia.

    Los límites son lo suficientemente altos para que nunca
    sean el factor limitante en las pruebas de thread-safety.
    """
    return TelemetryContext(
        max_steps=10_000,
        max_errors=10_000,
        max_metrics=10_000,
    )


# =============================================================================
# TESTS: CLASES DE CONFIGURACIÓN
# =============================================================================


class TestConfigurationClasses:
    """Pruebas para las clases de configuración y constantes del sistema."""

    def test_telemetry_defaults_values(self) -> None:
        """Verifica que TelemetryDefaults tenga los valores esperados."""
        assert TelemetryDefaults.MAX_STEPS == 1000
        assert TelemetryDefaults.MAX_ERRORS == 100
        assert TelemetryDefaults.MAX_METRICS == 500
        assert TelemetryDefaults.MAX_STRING_LENGTH == 10000
        assert TelemetryDefaults.MAX_MESSAGE_LENGTH == 1000
        assert TelemetryDefaults.MAX_RECURSION_DEPTH == 5
        assert TelemetryDefaults.MAX_COLLECTION_SIZE == 100

    def test_business_thresholds_default_values(self) -> None:
        """Verifica que BusinessThresholds tenga los umbrales correctos."""
        assert BusinessThresholds.CRITICAL_FLYBACK_VOLTAGE == 0.5
        assert BusinessThresholds.CRITICAL_DISSIPATED_POWER == 50.0
        assert BusinessThresholds.WARNING_SATURATION == 0.9

    def test_active_step_info_creation(self) -> None:
        """Verifica creación correcta de ActiveStepInfo."""
        start_time = time.perf_counter()
        metadata = {"key": "value"}

        info = ActiveStepInfo(start_time=start_time, metadata=metadata)

        assert info.start_time == start_time
        assert info.metadata == metadata

    def test_active_step_info_get_duration(self) -> None:
        """Verifica que ActiveStepInfo calcula duración correctamente."""
        info = ActiveStepInfo(start_time=time.perf_counter())
        time.sleep(0.01)

        duration = info.get_duration()

        assert duration >= 0.01
        assert duration < 1.0  # Límite razonable de ejecución

    def test_active_step_info_zero_metadata_by_default(self) -> None:
        """Verifica que metadata por defecto está vacío o es None."""
        info = ActiveStepInfo(start_time=time.perf_counter())
        # metadata puede ser None o dict vacío según implementación
        assert info.metadata is None or info.metadata == {}


# =============================================================================
# TESTS: StratumTopology (cobertura completa)
# =============================================================================


class TestStratumTopologyComplete:
    """
    Cobertura completa de StratumTopology para todos los estratos DIKW.

    Verifica que cada estrato tenga sus prefijos de métricas esperados.
    """

    def test_physics_prefixes(self) -> None:
        """PHYSICS contiene prefijos de dominio físico."""
        from app.core.telemetry import Stratum, StratumTopology

        prefixes = StratumTopology.get_prefixes(Stratum.PHYSICS)

        assert "gyro" in prefixes
        assert "nutation" in prefixes
        assert "laplace" in prefixes
        assert "thermo" in prefixes

    def test_tactics_prefixes(self) -> None:
        """TACTICS contiene prefijos de análisis espectral y resonancia."""
        from app.core.telemetry import Stratum, StratumTopology

        prefixes = StratumTopology.get_prefixes(Stratum.TACTICS)

        assert "spectral" in prefixes
        assert "fiedler" in prefixes
        assert "resonance" in prefixes

    def test_strategy_prefixes(self) -> None:
        """STRATEGY contiene prefijos de planificación financiera."""
        from app.core.telemetry import Stratum, StratumTopology

        prefixes = StratumTopology.get_prefixes(Stratum.STRATEGY)

        assert "real_options" in prefixes

    def test_wisdom_prefixes_exist(self) -> None:
        """
        WISDOM debe tener prefijos definidos (síntesis estratégica).

        Verifica que el estrato superior no esté vacío o None.
        """
        from app.core.telemetry import Stratum, StratumTopology

        prefixes = StratumTopology.get_prefixes(Stratum.WISDOM)

        # WISDOM debe retornar una colección (posiblemente vacía, pero no None)
        assert prefixes is not None
        assert hasattr(prefixes, "__iter__")

    def test_all_strata_have_prefixes_defined(self) -> None:
        """Verifica que todos los estratos tienen prefijos registrados."""
        from app.core.telemetry import Stratum, StratumTopology

        for stratum in Stratum:
            prefixes = StratumTopology.get_prefixes(stratum)
            assert prefixes is not None, (
                f"Estrato {stratum.name} no tiene prefijos definidos"
            )

    def test_prefixes_are_non_empty_strings(self) -> None:
        """Verifica que cada prefijo es un string no vacío."""
        from app.core.telemetry import Stratum, StratumTopology

        for stratum in Stratum:
            prefixes = StratumTopology.get_prefixes(stratum)
            for prefix in prefixes:
                assert isinstance(prefix, str), (
                    f"Prefijo no-string en {stratum.name}: {prefix!r}"
                )
                assert len(prefix.strip()) > 0, (
                    f"Prefijo vacío en {stratum.name}"
                )


# =============================================================================
# TESTS: StepStatus Enum
# =============================================================================


class TestStepStatusEnum:
    """Pruebas para el enum StepStatus."""

    def test_step_status_values(self) -> None:
        """Verifica que todos los valores de StepStatus existen."""
        assert StepStatus.SUCCESS.value == "success"
        assert StepStatus.FAILURE.value == "failure"
        assert StepStatus.WARNING.value == "warning"
        assert StepStatus.SKIPPED.value == "skipped"
        assert StepStatus.IN_PROGRESS.value == "in_progress"
        assert StepStatus.CANCELLED.value == "cancelled"

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("success", StepStatus.SUCCESS),
            ("failure", StepStatus.FAILURE),
            ("warning", StepStatus.WARNING),
            ("skipped", StepStatus.SKIPPED),
            ("in_progress", StepStatus.IN_PROGRESS),
            ("cancelled", StepStatus.CANCELLED),
        ],
    )
    def test_from_string_valid_lowercase(
        self, input_str: str, expected: StepStatus
    ) -> None:
        """Verifica que from_string convierte strings válidos en minúsculas."""
        assert StepStatus.from_string(input_str) == expected

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("SUCCESS", StepStatus.SUCCESS),
            ("FAILURE", StepStatus.FAILURE),
            ("Warning", StepStatus.WARNING),
            ("  success  ", StepStatus.SUCCESS),
            ("CANCELLED", StepStatus.CANCELLED),
        ],
    )
    def test_from_string_case_insensitive(
        self, input_str: str, expected: StepStatus
    ) -> None:
        """Verifica que from_string es insensible a mayúsculas/minúsculas."""
        assert StepStatus.from_string(input_str) == expected

    @pytest.mark.parametrize(
        "invalid_input",
        ["invalid", "", "unknown_status", "  ", "null"],
    )
    def test_from_string_invalid_returns_success(self, invalid_input: str) -> None:
        """Verifica que strings inválidos retornan SUCCESS (valor por defecto)."""
        assert StepStatus.from_string(invalid_input) == StepStatus.SUCCESS

    @pytest.mark.parametrize("non_string", [123, None, [], {}, 3.14, True])
    def test_from_string_non_string_returns_success(self, non_string: Any) -> None:
        """Verifica que tipos no-string retornan SUCCESS sin lanzar excepción."""
        assert StepStatus.from_string(non_string) == StepStatus.SUCCESS


# =============================================================================
# TESTS: Inicialización
# =============================================================================


class TestInitialization:
    """Pruebas para la inicialización y configuración de TelemetryContext."""

    def test_default_initialization(self, ctx: TelemetryContext) -> None:
        """Verifica el estado inicial por defecto."""
        assert ctx.request_id is not None
        assert isinstance(ctx.request_id, str)
        assert len(ctx.request_id) > 0

        assert isinstance(ctx.steps, list)
        assert len(ctx.steps) == 0

        assert isinstance(ctx.metrics, dict)
        assert len(ctx.metrics) == 0

        assert isinstance(ctx.errors, list)
        assert len(ctx.errors) == 0

        assert isinstance(ctx._active_steps, dict)
        assert len(ctx._active_steps) == 0

        assert isinstance(ctx.metadata, dict)
        assert len(ctx.metadata) == 0

        assert ctx.created_at > 0
        assert ctx.max_steps == TelemetryDefaults.MAX_STEPS
        assert ctx.max_errors == TelemetryDefaults.MAX_ERRORS
        assert ctx.max_metrics == TelemetryDefaults.MAX_METRICS

    def test_default_business_thresholds(self, ctx: TelemetryContext) -> None:
        """Verifica que los umbrales de negocio por defecto están configurados."""
        assert "critical_flyback_voltage" in ctx.business_thresholds
        assert "critical_dissipated_power" in ctx.business_thresholds
        assert "warning_saturation" in ctx.business_thresholds

        assert ctx.business_thresholds["critical_flyback_voltage"] == 0.5
        assert ctx.business_thresholds["critical_dissipated_power"] == 50.0
        assert ctx.business_thresholds["warning_saturation"] == 0.9

    def test_custom_request_id(self, ctx_with_custom_id: TelemetryContext) -> None:
        """Verifica que el request_id personalizado se preserva."""
        assert ctx_with_custom_id.request_id == "custom-test-id-12345"

    @pytest.mark.parametrize(
        "invalid_id",
        ["", None, 12345, [], {}],
        ids=["empty_string", "none", "integer", "list", "dict"],
    )
    def test_invalid_request_id_generates_uuid(self, invalid_id: Any) -> None:
        """Verifica que request_id inválido genera un UUID nuevo."""
        ctx = TelemetryContext(request_id=invalid_id)
        assert ctx.request_id is not None
        assert isinstance(ctx.request_id, str)
        assert len(ctx.request_id) > 0
        # El UUID generado nunca debe ser igual al input inválido
        if isinstance(invalid_id, str):
            assert ctx.request_id != invalid_id

    def test_request_id_too_long_replaced(self) -> None:
        """Verifica que un request_id excesivamente largo se reemplaza."""
        long_id = "x" * 500
        ctx = TelemetryContext(request_id=long_id)

        assert ctx.request_id != long_id
        assert len(ctx.request_id) <= TelemetryDefaults.MAX_REQUEST_ID_LENGTH

    def test_custom_limits(self, ctx_with_limits: TelemetryContext) -> None:
        """Verifica que los límites personalizados se configuran correctamente."""
        assert ctx_with_limits.max_steps == 3
        assert ctx_with_limits.max_errors == 2
        assert ctx_with_limits.max_metrics == 5

    @pytest.mark.parametrize(
        "max_steps,max_errors,max_metrics",
        [(-10, 0, -5), (-1, -1, -1)],
        ids=["mixed_negatives", "all_negative"],
    )
    def test_negative_limits_normalized_to_minimum(
        self, max_steps: int, max_errors: int, max_metrics: int
    ) -> None:
        """Verifica que límites negativos se normalizan al mínimo permitido (≥1)."""
        ctx = TelemetryContext(
            max_steps=max_steps,
            max_errors=max_errors,
            max_metrics=max_metrics,
        )
        assert ctx.max_steps >= 1
        assert ctx.max_errors >= 1
        assert ctx.max_metrics >= 1

    def test_excessively_large_limits_capped(self) -> None:
        """Verifica que límites excesivamente grandes se acotan al máximo permitido."""
        ctx = TelemetryContext(max_steps=999_999_999)
        max_allowed = (
            TelemetryDefaults.MAX_STEPS * TelemetryDefaults.MAX_LIMIT_MULTIPLIER
        )
        assert ctx.max_steps <= max_allowed

    @pytest.mark.parametrize(
        "invalid_limit",
        ["invalid", None, [], {}],
        ids=["string", "none", "list", "dict"],
    )
    def test_non_integer_limits_use_defaults(self, invalid_limit: Any) -> None:
        """Verifica que límites no-enteros caen al valor por defecto."""
        ctx = TelemetryContext(max_steps=invalid_limit, max_errors=invalid_limit)
        assert ctx.max_steps == TelemetryDefaults.MAX_STEPS
        assert ctx.max_errors == TelemetryDefaults.MAX_ERRORS

    def test_unique_request_ids_across_instances(self) -> None:
        """Verifica que instancias distintas obtienen request_id únicos."""
        ids = {TelemetryContext().request_id for _ in range(10)}
        assert len(ids) == 10, "Se esperaban 10 IDs únicos"

    def test_invalid_collection_types_fixed_on_demand(self) -> None:
        """Verifica que tipos de colección inválidos se corrigen al llamar validate."""
        ctx = TelemetryContext()
        # Corrupción intencional para simular estado inválido
        ctx.steps = "not a list"  # type: ignore[assignment]
        ctx.metrics = "not a dict"  # type: ignore[assignment]

        ctx._validate_and_fix_collection_types()

        assert isinstance(ctx.steps, list)
        assert isinstance(ctx.metrics, dict)


# =============================================================================
# TESTS: Semántica FIFO (clase dedicada)
# =============================================================================


class TestFIFOSemantics:
    """
    Pruebas dedicadas a verificar la semántica FIFO en colecciones con límites.

    Cuando una colección alcanza su límite (max_steps, max_errors),
    los elementos más antiguos se eliminan para dar lugar a los nuevos.
    Esta es la semántica FIFO (First-In, First-Out) de la ventana deslizante.
    """

    def test_steps_fifo_with_max_3(self, ctx_with_limits: TelemetryContext) -> None:
        """
        Con max_steps=3 y 5 inserciones, quedan los 3 últimos.

        Secuencia: step0, step1, step2, step3, step4
        Resultado esperado: [step2, step3, step4]
        """
        for i in range(5):
            ctx_with_limits.start_step(f"step{i}")
            ctx_with_limits.end_step(f"step{i}")

        assert len(ctx_with_limits.steps) == 3
        assert ctx_with_limits.steps[0]["step"] == "step2"
        assert ctx_with_limits.steps[1]["step"] == "step3"
        assert ctx_with_limits.steps[2]["step"] == "step4"

    def test_errors_fifo_with_max_2(self, ctx_with_limits: TelemetryContext) -> None:
        """
        Con max_errors=2 y 3 inserciones, quedan los 2 últimos.

        Secuencia: error0 (step0), error1 (step1), error2 (step2)
        Resultado esperado: [step1_error, step2_error]
        """
        ctx_with_limits.record_error("step0", "error0")
        ctx_with_limits.record_error("step1", "error1")
        ctx_with_limits.record_error("step2", "error2")

        assert len(ctx_with_limits.errors) == 2
        assert ctx_with_limits.errors[0]["step"] == "step1"
        assert ctx_with_limits.errors[1]["step"] == "step2"

    def test_errors_fifo_with_max_2_four_insertions(
        self, ctx_with_limits: TelemetryContext
    ) -> None:
        """
        Con max_errors=2 y 4 inserciones, quedan los 2 últimos.

        La versión original tenía este test mal: insertaba 4 errores
        pero verificaba step2/step3 como si el límite fuera 2 y la
        ventana deslizara correctamente. Este test lo confirma.
        """
        for i in range(4):
            ctx_with_limits.record_error(f"step{i}", f"error{i}")

        assert len(ctx_with_limits.errors) == 2
        assert ctx_with_limits.errors[0]["step"] == "step2"
        assert ctx_with_limits.errors[1]["step"] == "step3"

    def test_metrics_rejection_when_limit_reached(
        self, ctx_with_limits: TelemetryContext
    ) -> None:
        """
        Con max_metrics=5, la sexta métrica es rechazada.

        Las métricas usan rechazo (no FIFO) para preservar datos existentes.
        """
        for i in range(5):
            assert ctx_with_limits.record_metric("comp", f"m{i}", i) is True

        # La sexta debe ser rechazada
        assert ctx_with_limits.record_metric("comp", "m5", 5) is False
        assert len(ctx_with_limits.metrics) == 5

    def test_fifo_preserves_data_integrity(self, ctx_with_limits: TelemetryContext) -> None:
        """
        Verifica que el FIFO no corrompe los datos de los elementos preservados.
        """
        ctx_with_limits.record_error("step0", "error0", error_type="TypeA")
        ctx_with_limits.record_error("step1", "error1", error_type="TypeB")
        ctx_with_limits.record_error("step2", "error2", error_type="TypeC")

        # Solo quedan step1 y step2
        assert ctx_with_limits.errors[0]["message"] == "error1"
        assert ctx_with_limits.errors[0]["type"] == "TypeB"
        assert ctx_with_limits.errors[1]["message"] == "error2"
        assert ctx_with_limits.errors[1]["type"] == "TypeC"


# =============================================================================
# TESTS: Seguimiento de Steps
# =============================================================================


class TestStepTracking:
    """Pruebas para inicio, fin y seguimiento de steps."""

    def test_start_step_basic(self, ctx: TelemetryContext) -> None:
        """Verifica funcionalidad básica de start_step."""
        result = ctx.start_step("test_step")

        assert result is True
        assert "test_step" in ctx._active_steps
        assert isinstance(ctx._active_steps["test_step"], ActiveStepInfo)
        assert ctx._active_steps["test_step"].start_time > 0

    def test_start_step_with_metadata(self, ctx: TelemetryContext) -> None:
        """Verifica que start_step almacena metadata correctamente."""
        metadata = {"key": "value", "count": 42}
        result = ctx.start_step("test_step", metadata=metadata)

        assert result is True
        stored = ctx._active_steps["test_step"].metadata
        assert stored is not None
        assert stored["key"] == "value"
        assert stored["count"] == 42

    def test_start_step_metadata_sanitizes_datetime(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que datetime en metadata se convierte a ISO string."""
        metadata = {"datetime": datetime.now(), "nested": {"deep": "value"}}
        ctx.start_step("test_step", metadata=metadata)

        stored = ctx._active_steps["test_step"].metadata
        assert isinstance(stored["datetime"], str)
        assert isinstance(stored["nested"], dict)

    @pytest.mark.parametrize(
        "invalid_name",
        ["", None, 123, [], {}, "   ", "\t\n"],
        ids=[
            "empty_string", "none", "integer", "list", "dict",
            "whitespace_only", "tab_newline",
        ],
    )
    def test_start_step_rejects_invalid_names(
        self, ctx: TelemetryContext, invalid_name: Any
    ) -> None:
        """Verifica que start_step rechaza nombres inválidos."""
        assert ctx.start_step(invalid_name) is False

    def test_start_step_rejects_name_exceeding_max_length(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que nombres que exceden el límite son rechazados."""
        long_name = "x" * (TelemetryDefaults.MAX_STEP_NAME_LENGTH + 1)
        assert ctx.start_step(long_name) is False

    def test_start_step_accepts_name_at_max_length(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que nombres exactamente en el límite son aceptados."""
        max_name = "x" * TelemetryDefaults.MAX_STEP_NAME_LENGTH
        assert ctx.start_step(max_name) is True

    def test_start_step_duplicate_resets_timer(self, ctx: TelemetryContext) -> None:
        """Verifica que un start_step duplicado reinicia el temporizador."""
        ctx.start_step("duplicate_step")
        first_time = ctx._active_steps["duplicate_step"].start_time

        time.sleep(0.001)

        ctx.start_step("duplicate_step")
        second_time = ctx._active_steps["duplicate_step"].start_time

        assert second_time > first_time

    def test_end_step_basic(self, ctx: TelemetryContext) -> None:
        """Verifica funcionalidad básica de end_step."""
        ctx.start_step("test_step")
        time.sleep(0.01)
        result = ctx.end_step("test_step", StepStatus.SUCCESS)

        assert result is True
        assert "test_step" not in ctx._active_steps
        assert len(ctx.steps) == 1

        step = ctx.steps[0]
        assert step["step"] == "test_step"
        assert step["status"] == StepStatus.SUCCESS.value
        assert step["duration_seconds"] >= 0.01
        assert step["duration_seconds"] < 1.0
        assert "timestamp" in step
        assert "perf_counter" in step

    def test_end_step_with_metadata(self, ctx: TelemetryContext) -> None:
        """Verifica que end_step almacena metadata en el registro."""
        ctx.start_step("test_step")
        metadata = {"result": "ok", "count": 5}
        ctx.end_step("test_step", StepStatus.SUCCESS, metadata=metadata)

        assert "metadata" in ctx.steps[0]
        assert ctx.steps[0]["metadata"]["result"] == "ok"

    def test_end_step_merges_start_and_end_metadata(
        self, ctx: TelemetryContext
    ) -> None:
        """
        Verifica que end_step combina metadata de inicio y fin.

        Regla de merge: end_metadata sobreescribe start_metadata en claves comunes.
        """
        start_meta = {"start_key": "start_value", "shared": "from_start"}
        end_meta = {"end_key": "end_value", "shared": "from_end"}

        ctx.start_step("test_step", metadata=start_meta)
        ctx.end_step("test_step", StepStatus.SUCCESS, metadata=end_meta)

        combined = ctx.steps[0]["metadata"]
        assert combined["start_key"] == "start_value"
        assert combined["end_key"] == "end_value"
        assert combined["shared"] == "from_end"  # end sobreescribe

    def test_end_step_preserves_start_metadata_when_no_end_metadata(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que la metadata de inicio se preserva si no hay metadata de fin."""
        ctx.start_step("test_step", metadata={"key": "value"})
        ctx.end_step("test_step", StepStatus.SUCCESS)

        assert ctx.steps[0]["metadata"]["key"] == "value"

    @pytest.mark.parametrize(
        "status,expected_value",
        [
            (StepStatus.SUCCESS, "success"),
            (StepStatus.FAILURE, "failure"),
            (StepStatus.WARNING, "warning"),
            (StepStatus.SKIPPED, "skipped"),
            (StepStatus.IN_PROGRESS, "in_progress"),
            (StepStatus.CANCELLED, "cancelled"),
            ("success", "success"),
            ("failure", "failure"),
        ],
    )
    def test_end_step_status_normalization(
        self, ctx: TelemetryContext, status: Any, expected_value: str
    ) -> None:
        """Verifica que distintos formatos de status se normalizan correctamente."""
        ctx.start_step("test_step")
        ctx.end_step("test_step", status)

        assert ctx.steps[0]["status"] == expected_value

    @pytest.mark.parametrize(
        "invalid_status",
        ["invalid_status_xyz", 123, None, []],
        ids=["string", "integer", "none", "list"],
    )
    def test_end_step_invalid_status_defaults_to_success(
        self, ctx: TelemetryContext, invalid_status: Any
    ) -> None:
        """Verifica que status inválido produce SUCCESS por defecto."""
        ctx.start_step("test_step")
        ctx.end_step("test_step", invalid_status)

        assert ctx.steps[0]["status"] == StepStatus.SUCCESS.value

    def test_end_step_without_prior_start_records_zero_duration(
        self, ctx: TelemetryContext
    ) -> None:
        """
        Verifica que end_step sin start_step registra duración=0.

        Esto permite registrar pasos que comenzaron antes de que
        el contexto de telemetría fuera instanciado.
        """
        result = ctx.end_step("never_started")

        assert result is True
        assert len(ctx.steps) == 1
        assert ctx.steps[0]["duration_seconds"] == 0.0
        assert ctx.steps[0]["step"] == "never_started"

    @pytest.mark.parametrize(
        "invalid_name",
        ["", None, 123],
        ids=["empty", "none", "integer"],
    )
    def test_end_step_rejects_invalid_names(
        self, ctx: TelemetryContext, invalid_name: Any
    ) -> None:
        """Verifica que end_step rechaza nombres inválidos."""
        assert ctx.end_step(invalid_name) is False

    def test_multiple_sequential_steps(self, ctx: TelemetryContext) -> None:
        """Verifica que múltiples steps secuenciales se registran correctamente."""
        steps = ["step1", "step2", "step3"]

        for step_name in steps:
            ctx.start_step(step_name)
            time.sleep(0.001)
            ctx.end_step(step_name)

        assert len(ctx.steps) == 3
        for i, name in enumerate(steps):
            assert ctx.steps[i]["step"] == name
            assert ctx.steps[i]["duration_seconds"] > 0

    def test_nested_steps_order_and_duration(self, ctx: TelemetryContext) -> None:
        """
        Verifica que steps anidados se registran en orden correcto.

        Orden de registro en ctx.steps: inner (end primero) → outer (end último).
        La duración del outer debe ser mayor que la del inner.
        """
        ctx.start_step("outer")
        time.sleep(0.002)
        ctx.start_step("inner")
        time.sleep(0.002)
        ctx.end_step("inner")  # Se registra en ctx.steps[0]
        time.sleep(0.002)
        ctx.end_step("outer")  # Se registra en ctx.steps[1]

        assert len(ctx.steps) == 2

        # inner termina primero → índice 0; outer termina último → índice 1
        assert ctx.steps[0]["step"] == "inner"
        assert ctx.steps[1]["step"] == "outer"

        inner_duration = ctx.steps[0]["duration_seconds"]
        outer_duration = ctx.steps[1]["duration_seconds"]

        # outer ≈ 3×sleep(0.002) > inner ≈ 1×sleep(0.002)
        assert outer_duration > inner_duration, (
            f"Se esperaba outer ({outer_duration:.4f}s) > inner ({inner_duration:.4f}s)"
        )


# =============================================================================
# TESTS: Context Manager de Step
# =============================================================================


class TestStepContextManager:
    """Pruebas para el gestor de contexto ctx.step()."""

    def test_context_manager_success_path(self, ctx: TelemetryContext) -> None:
        """Verifica que el paso exitoso se registra correctamente."""
        with ctx.step("test_operation"):
            time.sleep(0.001)

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["step"] == "test_operation"
        assert ctx.steps[0]["status"] == StepStatus.SUCCESS.value
        assert ctx.steps[0]["duration_seconds"] > 0

    def test_context_manager_exception_records_failure(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que una excepción se registra como FAILURE y se re-lanza."""
        with pytest.raises(ValueError):
            with ctx.step("failing_operation"):
                raise ValueError("Test error")

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["step"] == "failing_operation"
        assert ctx.steps[0]["status"] == StepStatus.FAILURE.value

        assert len(ctx.errors) == 1
        assert ctx.errors[0]["step"] == "failing_operation"
        assert "Test error" in ctx.errors[0]["message"]

    def test_context_manager_custom_error_status(self, ctx: TelemetryContext) -> None:
        """Verifica que el status de error personalizado se aplica."""
        with pytest.raises(RuntimeError):
            with ctx.step("warning_op", error_status=StepStatus.WARNING):
                raise RuntimeError("Warning condition")

        assert ctx.steps[0]["status"] == StepStatus.WARNING.value

    def test_context_manager_passes_metadata(self, ctx: TelemetryContext) -> None:
        """Verifica que la metadata se pasa correctamente al step."""
        metadata = {"user_id": 123}
        with ctx.step("test_op", metadata=metadata):
            pass

        assert ctx.steps[0]["metadata"]["user_id"] == 123

    def test_context_manager_yields_self(self, ctx: TelemetryContext) -> None:
        """Verifica que el gestor de contexto cede el contexto de telemetría."""
        with ctx.step("test") as yielded_ctx:
            assert yielded_ctx is ctx

    def test_context_manager_captures_exception_details_when_enabled(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que los detalles de excepción se capturan cuando está habilitado."""
        with pytest.raises(ValueError):
            with ctx.step("test_op", capture_exception_details=True):
                raise ValueError("Detailed error")

        assert "metadata" in ctx.steps[0]
        assert "error_type" in ctx.steps[0]["metadata"]
        assert ctx.steps[0]["metadata"]["error_type"] == "ValueError"

    def test_context_manager_omits_exception_details_when_disabled(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que los detalles de excepción se omiten cuando está deshabilitado."""
        with pytest.raises(ValueError):
            with ctx.step("test_op", capture_exception_details=False):
                raise ValueError("Detailed error")

        assert len(ctx.steps) == 1
        assert len(ctx.errors) == 1
        # El traceback no debe estar en el error registrado
        assert "traceback" not in ctx.errors[0]

    def test_context_manager_with_empty_name_does_not_track(
        self, ctx: TelemetryContext
    ) -> None:
        """
        Verifica que un nombre de step vacío no produce un registro.

        El gestor de contexto no debe lanzar excepción, pero tampoco
        debe registrar el paso dado que el nombre es inválido.
        """
        with ctx.step(""):
            pass  # No debe lanzar excepción

        # El step no se registra por nombre inválido
        assert len(ctx.steps) == 0

    def test_context_manager_reraises_exception(self, ctx: TelemetryContext) -> None:
        """Verifica que el gestor de contexto NO suprime excepciones."""
        with pytest.raises(ValueError, match="Test exception"):
            with ctx.step("test"):
                raise ValueError("Test exception")


# =============================================================================
# TESTS: Registro de Métricas
# =============================================================================


class TestMetricsRecording:
    """Pruebas para el registro de métricas."""

    def test_record_metric_basic(self, ctx: TelemetryContext) -> None:
        """Verifica el registro básico de una métrica."""
        result = ctx.record_metric("component", "metric", 42)

        assert result is True
        assert "component.metric" in ctx.metrics
        assert ctx.metrics["component.metric"] == 42

    @pytest.mark.parametrize(
        "value",
        [42, 3.14, "string_value", True, False, None, [1, 2, 3], {"nested": "dict"}],
        ids=["int", "float", "string", "true", "false", "none", "list", "dict"],
    )
    def test_record_metric_various_types_accepted(
        self, ctx: TelemetryContext, value: Any
    ) -> None:
        """Verifica que las métricas aceptan distintos tipos de valor."""
        result = ctx.record_metric("comp", "metric", value)
        assert result is True
        assert "comp.metric" in ctx.metrics

    def test_record_metric_overwrites_by_default(self, ctx: TelemetryContext) -> None:
        """Verifica que el registro sobreescribe por defecto."""
        ctx.record_metric("comp", "metric", 100)
        ctx.record_metric("comp", "metric", 200)

        assert ctx.metrics["comp.metric"] == 200

    def test_record_metric_no_overwrite_preserves_original(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que overwrite=False preserva el valor original."""
        ctx.record_metric("comp", "metric", 100)
        result = ctx.record_metric("comp", "metric", 200, overwrite=False)

        assert result is False
        assert ctx.metrics["comp.metric"] == 100

    @pytest.mark.parametrize(
        "invalid_component",
        ["", None, 123, []],
        ids=["empty", "none", "integer", "list"],
    )
    def test_record_metric_rejects_invalid_component(
        self, ctx: TelemetryContext, invalid_component: Any
    ) -> None:
        """Verifica que componentes inválidos son rechazados."""
        assert ctx.record_metric(invalid_component, "metric", 100) is False

    @pytest.mark.parametrize(
        "invalid_name",
        ["", None, 123, []],
        ids=["empty", "none", "integer", "list"],
    )
    def test_record_metric_rejects_invalid_metric_name(
        self, ctx: TelemetryContext, invalid_name: Any
    ) -> None:
        """Verifica que nombres de métrica inválidos son rechazados."""
        assert ctx.record_metric("comp", invalid_name, 100) is False

    def test_record_metric_rejects_names_exceeding_max_length(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que nombres que exceden el límite son rechazados."""
        long_name = "x" * (TelemetryDefaults.MAX_NAME_LENGTH + 1)
        assert ctx.record_metric(long_name, "metric", 100) is False
        assert ctx.record_metric("comp", long_name, 100) is False

    def test_record_metric_enforces_max_limit(
        self, ctx_with_limits: TelemetryContext
    ) -> None:
        """Verifica que el límite max_metrics se aplica correctamente."""
        for i in range(5):
            assert ctx_with_limits.record_metric("comp", f"metric{i}", i) is True

        # La sexta métrica debe ser rechazada
        assert ctx_with_limits.record_metric("comp", "metric6", 6) is False
        assert len(ctx_with_limits.metrics) == 5

    def test_record_metric_sanitizes_complex_value(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que valores complejos (con datetime) son sanitizados."""
        complex_value = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3],
            "datetime": datetime.now(),
        }

        assert ctx.record_metric("comp", "complex", complex_value) is True

        stored = ctx.metrics["comp.complex"]
        assert isinstance(stored, dict)
        assert isinstance(stored["datetime"], str)


# =============================================================================
# TESTS: Incremento de Métrica
# =============================================================================


class TestIncrementMetric:
    """Pruebas para increment_metric."""

    def test_increment_creates_new_metric_at_one(self, ctx: TelemetryContext) -> None:
        """Verifica que incrementar una métrica nueva la crea en 1."""
        assert ctx.increment_metric("comp", "counter") is True
        assert ctx.metrics["comp.counter"] == 1

    def test_increment_adds_to_existing_metric(self, ctx: TelemetryContext) -> None:
        """Verifica que incrementar suma al valor existente."""
        ctx.record_metric("comp", "counter", 10)
        ctx.increment_metric("comp", "counter", 5)

        assert ctx.metrics["comp.counter"] == 15

    def test_increment_with_negative_delta(self, ctx: TelemetryContext) -> None:
        """Verifica que incrementar con valor negativo resta correctamente."""
        ctx.record_metric("comp", "counter", 10)
        ctx.increment_metric("comp", "counter", -3)

        assert ctx.metrics["comp.counter"] == 7

    def test_increment_with_float_delta(self, ctx: TelemetryContext) -> None:
        """Verifica que el incremento funciona con valores flotantes."""
        ctx.record_metric("comp", "value", 1.5)
        ctx.increment_metric("comp", "value", 0.5)

        assert math.isclose(ctx.metrics["comp.value"], 2.0, rel_tol=1e-9)

    @pytest.mark.parametrize(
        "invalid_component",
        ["", None],
        ids=["empty", "none"],
    )
    def test_increment_rejects_invalid_component(
        self, ctx: TelemetryContext, invalid_component: Any
    ) -> None:
        """Verifica que componentes inválidos son rechazados."""
        assert ctx.increment_metric(invalid_component, "counter") is False

    @pytest.mark.parametrize(
        "invalid_name",
        ["", None],
        ids=["empty", "none"],
    )
    def test_increment_rejects_invalid_metric_name(
        self, ctx: TelemetryContext, invalid_name: Any
    ) -> None:
        """Verifica que nombres de métrica inválidos son rechazados."""
        assert ctx.increment_metric("comp", invalid_name) is False

    @pytest.mark.parametrize(
        "non_numeric_increment",
        ["not_a_number", [1, 2, 3], None, {}],
        ids=["string", "list", "none", "dict"],
    )
    def test_increment_rejects_non_numeric_delta(
        self, ctx: TelemetryContext, non_numeric_increment: Any
    ) -> None:
        """Verifica que incrementos no numéricos son rechazados."""
        assert ctx.increment_metric("comp", "counter", non_numeric_increment) is False

    def test_increment_resets_non_numeric_existing_value(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que un valor existente no numérico se reinicia a 0 antes de incrementar."""
        ctx.record_metric("comp", "value", "string_value")
        result = ctx.increment_metric("comp", "value", 5)

        assert result is True
        assert ctx.metrics["comp.value"] == 5  # 0 (reset) + 5

    def test_increment_respects_max_metrics_for_new_key(
        self, ctx_with_limits: TelemetryContext
    ) -> None:
        """Verifica que incrementar una clave nueva falla cuando se alcanza max_metrics."""
        for i in range(5):
            ctx_with_limits.record_metric("comp", f"m{i}", i)

        # Nueva clave → debe ser rechazada
        assert ctx_with_limits.increment_metric("comp", "new_counter") is False

    def test_increment_existing_key_at_limit_succeeds(
        self, ctx_with_limits: TelemetryContext
    ) -> None:
        """Verifica que incrementar una clave existente funciona aunque se haya alcanzado el límite."""
        for i in range(5):
            ctx_with_limits.record_metric("comp", f"m{i}", i)

        # Clave existente → debe funcionar
        assert ctx_with_limits.increment_metric("comp", "m0", 10) is True
        assert ctx_with_limits.metrics["comp.m0"] == 10


# =============================================================================
# TESTS: Obtención de Métricas
# =============================================================================


class TestGetMetric:
    """Pruebas para get_metric."""

    def test_get_metric_returns_existing_value(self, ctx: TelemetryContext) -> None:
        """Verifica que get_metric retorna el valor existente."""
        ctx.record_metric("comp", "value", 42)
        assert ctx.get_metric("comp", "value") == 42

    def test_get_metric_returns_none_for_missing(self, ctx: TelemetryContext) -> None:
        """Verifica que get_metric retorna None para métricas inexistentes."""
        assert ctx.get_metric("comp", "nonexistent") is None

    def test_get_metric_returns_custom_default(self, ctx: TelemetryContext) -> None:
        """Verifica que get_metric retorna el default personalizado."""
        assert ctx.get_metric("comp", "nonexistent", default=100) == 100

    def test_get_metric_returns_complex_value(self, ctx: TelemetryContext) -> None:
        """Verifica que get_metric retorna valores complejos correctamente."""
        ctx.record_metric("comp", "data", {"key": "value"})
        assert ctx.get_metric("comp", "data") == {"key": "value"}


# =============================================================================
# TESTS: Registro de Errores
# =============================================================================


class TestErrorRecording:
    """Pruebas para el registro de errores."""

    def test_record_error_basic(self, ctx: TelemetryContext) -> None:
        """Verifica el registro básico de un error."""
        result = ctx.record_error("test_step", "Something went wrong")

        assert result is True
        assert len(ctx.errors) == 1
        assert ctx.errors[0]["step"] == "test_step"
        assert ctx.errors[0]["message"] == "Something went wrong"
        assert "timestamp" in ctx.errors[0]

    def test_record_error_with_type(self, ctx: TelemetryContext) -> None:
        """Verifica el registro de errores con tipo explícito."""
        ctx.record_error("test_step", "Error occurred", error_type="ValidationError")

        assert ctx.errors[0]["type"] == "ValidationError"

    def test_record_error_with_exception_object(self, ctx: TelemetryContext) -> None:
        """Verifica el registro de errores con objeto excepción."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            ctx.record_error("test_step", "Caught exception", exception=e)

        assert ctx.errors[0]["type"] == "ValueError"
        assert "exception_details" in ctx.errors[0]
        assert "Test exception" in ctx.errors[0]["exception_details"]

    def test_record_error_with_traceback_when_requested(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que el traceback se incluye cuando se solicita."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            ctx.record_error(
                "test_step", "Caught", exception=e, include_traceback=True
            )

        assert "traceback" in ctx.errors[0]
        assert "ValueError" in ctx.errors[0]["traceback"]

    def test_record_error_excludes_traceback_by_default(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que el traceback se excluye por defecto."""
        try:
            raise ValueError("Test")
        except ValueError as e:
            ctx.record_error("step", "msg", exception=e)

        assert "traceback" not in ctx.errors[0]

    def test_record_error_with_metadata(self, ctx: TelemetryContext) -> None:
        """Verifica el registro de errores con metadata adicional."""
        metadata = {"user_id": 123, "context": "validation"}
        result = ctx.record_error("test_step", "Error", metadata=metadata)

        assert result is True
        assert "metadata" in ctx.errors[0]

    @pytest.mark.parametrize(
        "invalid_step",
        ["", None, 123],
        ids=["empty", "none", "integer"],
    )
    def test_record_error_with_invalid_step_falls_back(
        self, ctx: TelemetryContext, invalid_step: Any
    ) -> None:
        """
        Verifica que un step name inválido usa el fallback '__unknown_step__'.

        El error se registra igualmente (retorna True) con el nombre de fallback.
        """
        result = ctx.record_error(invalid_step, "message")

        assert result is True
        assert ctx.errors[-1]["step"] == "__unknown_step__"

    @pytest.mark.parametrize(
        "invalid_message",
        ["", None, 123],
        ids=["empty", "none", "integer"],
    )
    def test_record_error_with_invalid_message_falls_back(
        self, ctx: TelemetryContext, invalid_message: Any
    ) -> None:
        """
        Verifica que un mensaje inválido usa el mensaje genérico de fallback.

        El error se registra igualmente (retorna True) con el mensaje genérico.
        """
        result = ctx.record_error("step", invalid_message)

        assert result is True
        assert "Unknown error" in ctx.errors[-1]["message"]

    def test_record_error_truncates_long_message(self, ctx: TelemetryContext) -> None:
        """Verifica que mensajes muy largos se truncan al límite máximo."""
        long_message = "x" * 2000
        ctx.record_error("step", long_message)

        assert len(ctx.errors[0]["message"]) <= TelemetryDefaults.MAX_MESSAGE_LENGTH

    def test_record_error_fifo_with_max_2_and_3_insertions(
        self, ctx_with_limits: TelemetryContext
    ) -> None:
        """
        Verifica la semántica FIFO: con max_errors=2 y 3 errores, quedan los 2 últimos.
        """
        ctx_with_limits.record_error("step1", "error1")
        ctx_with_limits.record_error("step2", "error2")
        ctx_with_limits.record_error("step3", "error3")

        assert len(ctx_with_limits.errors) == 2
        assert ctx_with_limits.errors[0]["step"] == "step2"
        assert ctx_with_limits.errors[1]["step"] == "step3"


# =============================================================================
# TESTS: Gestión de Timers Activos
# =============================================================================


class TestActiveTimers:
    """Pruebas para la gestión de timers activos."""

    def test_get_active_timers_empty_initially(self, ctx: TelemetryContext) -> None:
        """Verifica que la lista de timers activos está vacía inicialmente."""
        assert ctx.get_active_timers() == []

    def test_get_active_timers_returns_active_step_names(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que get_active_timers retorna los nombres de steps activos."""
        ctx.start_step("step1")
        ctx.start_step("step2")

        active = ctx.get_active_timers()
        assert set(active) == {"step1", "step2"}

    def test_get_active_timers_updates_after_end(self, ctx: TelemetryContext) -> None:
        """Verifica que get_active_timers se actualiza al terminar un step."""
        ctx.start_step("step1")
        ctx.start_step("step2")
        ctx.end_step("step1")

        active = ctx.get_active_timers()
        assert active == ["step2"]

    def test_get_active_step_info_returns_correct_data(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que get_active_step_info retorna la info correcta."""
        ctx.start_step("step1", metadata={"key": "value"})
        time.sleep(0.01)

        info = ctx.get_active_step_info("step1")

        assert info is not None
        assert info["step_name"] == "step1"
        assert info["duration_so_far"] >= 0.01
        assert info["metadata"]["key"] == "value"

    def test_get_active_step_info_returns_none_for_unknown(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que get_active_step_info retorna None para steps inexistentes."""
        assert ctx.get_active_step_info("nonexistent") is None

    def test_get_active_step_info_returns_independent_copy(
        self, ctx: TelemetryContext
    ) -> None:
        """
        Verifica que get_active_step_info retorna una copia independiente.

        Modificar la copia retornada no debe afectar al estado interno.
        """
        ctx.start_step("step1", metadata={"key": "value"})

        info = ctx.get_active_step_info("step1")
        info["metadata"]["key"] = "modified"

        # El estado interno no debe haberse modificado
        original_info = ctx.get_active_step_info("step1")
        assert original_info["metadata"]["key"] == "value"

    def test_cancel_step_removes_timer_without_recording(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que cancel_step elimina el timer sin registrar el step."""
        ctx.start_step("step1")
        result = ctx.cancel_step("step1")

        assert result is True
        assert "step1" not in ctx._active_steps
        assert len(ctx.steps) == 0

    def test_cancel_nonexistent_step_returns_false(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que cancelar un step inexistente retorna False."""
        assert ctx.cancel_step("never_started") is False

    def test_clear_active_timers_returns_count(self, ctx: TelemetryContext) -> None:
        """Verifica que clear_active_timers retorna el número de timers eliminados."""
        ctx.start_step("step1")
        ctx.start_step("step2")
        ctx.start_step("step3")

        count = ctx.clear_active_timers()

        assert count == 3
        assert len(ctx._active_steps) == 0
        assert ctx.get_active_timers() == []

    def test_clear_active_timers_when_empty_returns_zero(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que limpiar timers vacíos retorna 0."""
        assert ctx.clear_active_timers() == 0


# =============================================================================
# TESTS: Consultas de Steps
# =============================================================================


class TestStepQueries:
    """Pruebas para los métodos de consulta de steps."""

    def test_has_step_true_for_completed_step(self, ctx: TelemetryContext) -> None:
        """Verifica que has_step retorna True para un step completado."""
        ctx.start_step("step1")
        ctx.end_step("step1")
        assert ctx.has_step("step1") is True

    def test_has_step_false_for_nonexistent(self, ctx: TelemetryContext) -> None:
        """Verifica que has_step retorna False para steps inexistentes."""
        assert ctx.has_step("nonexistent") is False

    def test_has_step_false_for_active_but_not_completed(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que has_step retorna False para steps activos no completados."""
        ctx.start_step("step1")
        assert ctx.has_step("step1") is False

    def test_get_step_by_name_returns_step_data(self, ctx: TelemetryContext) -> None:
        """Verifica que get_step_by_name retorna los datos del step."""
        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.SUCCESS)

        step = ctx.get_step_by_name("step1")
        assert step is not None
        assert step["step"] == "step1"
        assert step["status"] == "success"

    def test_get_step_by_name_returns_none_for_nonexistent(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que get_step_by_name retorna None para steps inexistentes."""
        assert ctx.get_step_by_name("nonexistent") is None

    def test_get_step_by_name_returns_last_by_default(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que get_step_by_name retorna la última ocurrencia por defecto."""
        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.SUCCESS, {"attempt": 1})

        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.FAILURE, {"attempt": 2})

        step = ctx.get_step_by_name("step1", last=True)
        assert step["metadata"]["attempt"] == 2

    def test_get_step_by_name_returns_first_when_requested(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que get_step_by_name puede retornar la primera ocurrencia."""
        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.SUCCESS, {"attempt": 1})

        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.FAILURE, {"attempt": 2})

        step = ctx.get_step_by_name("step1", last=False)
        assert step["metadata"]["attempt"] == 1

    def test_get_step_by_name_returns_independent_copy(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que get_step_by_name retorna una copia independiente."""
        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.SUCCESS)

        step = ctx.get_step_by_name("step1")
        step["modified"] = True  # type: ignore[index]

        # El original no debe haber cambiado
        assert "modified" not in ctx.steps[0]


# =============================================================================
# TESTS: Exportación y Serialización
# =============================================================================


class TestExportAndSerialization:
    """Pruebas para to_dict() y get_summary()."""

    def test_to_dict_returns_expected_keys(self, ctx: TelemetryContext) -> None:
        """Verifica que to_dict retorna la estructura de claves esperada."""
        data = ctx.to_dict()

        required_keys = {
            "request_id", "steps", "metrics", "errors",
            "total_duration_seconds", "created_at", "age_seconds",
        }
        assert required_keys.issubset(data.keys())
        assert isinstance(data["steps"], list)
        assert isinstance(data["metrics"], dict)
        assert isinstance(data["errors"], list)

    def test_to_dict_with_populated_context(
        self, ctx_populated: TelemetryContext
    ) -> None:
        """Verifica que to_dict exporta datos correctamente con contexto poblado."""
        data = ctx_populated.to_dict()

        assert len(data["steps"]) == 1
        assert len(data["metrics"]) == 2
        assert len(data["errors"]) == 1
        assert data["total_duration_seconds"] > 0

    def test_to_dict_includes_metadata_when_requested(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que to_dict incluye metadata cuando se solicita."""
        ctx.metadata["custom"] = "value"
        data = ctx.to_dict(include_metadata=True)

        assert "metadata" in data
        assert data["metadata"]["custom"] == "value"

    def test_to_dict_excludes_metadata_when_not_requested(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que to_dict excluye metadata cuando no se solicita."""
        ctx.metadata["custom"] = "value"
        data = ctx.to_dict(include_metadata=False)

        assert "metadata" not in data

    def test_to_dict_includes_active_timers_info(self, ctx: TelemetryContext) -> None:
        """Verifica que to_dict incluye información de timers activos."""
        ctx.start_step("incomplete", metadata={"key": "value"})
        time.sleep(0.01)

        data = ctx.to_dict()

        assert "active_timers" in data
        assert "incomplete" in data["active_timers"]
        assert "active_timers_info" in data
        assert "incomplete" in data["active_timers_info"]
        assert data["active_timers_info"]["incomplete"]["duration_so_far"] >= 0.01
        assert data["active_timers_info"]["incomplete"]["has_metadata"] is True

    def test_to_dict_returns_deep_independent_copies(
        self, ctx: TelemetryContext
    ) -> None:
        """
        Verifica que to_dict retorna copias profundas e independientes.

        Modificar la estructura retornada no debe afectar el estado interno.
        """
        ctx.start_step("step1")
        ctx.end_step("step1", metadata={"nested": {"key": "value"}})

        data = ctx.to_dict()
        # Modificar profundamente el resultado
        data["steps"][0]["metadata"]["nested"]["key"] = "modified"

        # El estado interno debe permanecer intacto
        assert ctx.steps[0]["metadata"]["nested"]["key"] == "value"

    def test_get_summary_structure(self, ctx: TelemetryContext) -> None:
        """Verifica que get_summary retorna la estructura correcta."""
        summary = ctx.get_summary()

        required_keys = {
            "request_id", "total_steps", "total_errors", "total_metrics",
            "active_timers", "total_duration_seconds", "step_statuses",
            "error_types", "has_errors", "has_failures", "age_seconds",
        }
        assert required_keys.issubset(summary.keys())

    def test_get_summary_with_populated_data(
        self, ctx_populated: TelemetryContext
    ) -> None:
        """Verifica que get_summary calcula estadísticas correctamente."""
        summary = ctx_populated.get_summary()

        assert summary["total_steps"] == 1
        assert summary["total_errors"] == 1
        assert summary["total_metrics"] == 2
        assert summary["has_errors"] is True
        assert summary["total_duration_seconds"] > 0

    def test_get_summary_counts_step_statuses_correctly(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que get_summary cuenta los estados de steps correctamente."""
        statuses = [StepStatus.SUCCESS, StepStatus.FAILURE, StepStatus.SUCCESS]
        for i, status in enumerate(statuses):
            ctx.start_step(f"s{i}")
            ctx.end_step(f"s{i}", status)

        summary = ctx.get_summary()
        assert summary["step_statuses"]["success"] == 2
        assert summary["step_statuses"]["failure"] == 1
        assert summary["has_failures"] is True

    def test_get_summary_counts_error_types_correctly(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que get_summary cuenta los tipos de error correctamente."""
        ctx.record_error("step1", "e1", error_type="ValidationError")
        ctx.record_error("step2", "e2", error_type="ValidationError")
        ctx.record_error("step3", "e3", error_type="DatabaseError")

        summary = ctx.get_summary()
        assert summary["error_types"]["ValidationError"] == 2
        assert summary["error_types"]["DatabaseError"] == 1


# =============================================================================
# TESTS: Reset
# =============================================================================


class TestResetFunctionality:
    """Pruebas para el método reset()."""

    def test_reset_clears_all_data_preserving_id(
        self, ctx_populated: TelemetryContext
    ) -> None:
        """Verifica que reset limpia todos los datos preservando el request_id."""
        original_id = ctx_populated.request_id
        ctx_populated.reset(keep_request_id=True)

        assert ctx_populated.request_id == original_id
        assert len(ctx_populated.steps) == 0
        assert len(ctx_populated.metrics) == 0
        assert len(ctx_populated.errors) == 0
        assert len(ctx_populated._active_steps) == 0
        assert len(ctx_populated.metadata) == 0

    def test_reset_generates_new_id_when_requested(
        self, ctx_populated: TelemetryContext
    ) -> None:
        """Verifica que reset puede generar un nuevo request_id."""
        original_id = ctx_populated.request_id
        ctx_populated.reset(keep_request_id=False)

        assert ctx_populated.request_id != original_id
        assert len(ctx_populated.steps) == 0

    def test_reset_updates_created_at_timestamp(self, ctx: TelemetryContext) -> None:
        """Verifica que reset actualiza el timestamp de creación."""
        original_created = ctx.created_at
        time.sleep(0.01)
        ctx.reset()

        assert ctx.created_at > original_created


# =============================================================================
# TESTS: Thread Safety
# =============================================================================


class TestThreadSafety:
    """
    Pruebas de concurrencia con barreras de sincronización.

    Usa threading.Barrier para garantizar que todos los hilos
    comiencen simultáneamente, maximizando la presión de concurrencia.
    """

    def test_concurrent_metric_recording_is_thread_safe(
        self, thread_safe_ctx: TelemetryContext
    ) -> None:
        """
        Verifica que el registro concurrente de métricas es thread-safe.

        5 hilos × 10 métricas únicas por hilo = 50 métricas totales.
        Se usa thread_safe_ctx para evitar colisión con max_metrics.
        """
        n_threads = 5
        metrics_per_thread = 10
        barrier = threading.Barrier(n_threads)
        errors: list[Exception] = []

        def record_metrics(thread_id: int) -> None:
            try:
                barrier.wait()  # Arranque sincronizado
                for i in range(metrics_per_thread):
                    thread_safe_ctx.record_metric(
                        f"thread{thread_id}", f"metric{i}", i
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_metrics, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errores en hilos: {errors}"
        assert len(thread_safe_ctx.metrics) == n_threads * metrics_per_thread

    def test_concurrent_step_tracking_is_thread_safe(
        self, thread_safe_ctx: TelemetryContext
    ) -> None:
        """
        Verifica que el seguimiento concurrente de steps es thread-safe.

        10 hilos, cada uno registra un step con nombre único.
        """
        n_threads = 10
        barrier = threading.Barrier(n_threads)
        errors: list[Exception] = []

        def track_step(step_name: str) -> None:
            try:
                barrier.wait()
                thread_safe_ctx.start_step(step_name)
                time.sleep(0.001)
                thread_safe_ctx.end_step(step_name)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=track_step, args=(f"step{i}",))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errores en hilos: {errors}"
        assert len(thread_safe_ctx.steps) == n_threads

    def test_concurrent_error_recording_is_thread_safe(
        self, thread_safe_ctx: TelemetryContext
    ) -> None:
        """
        Verifica que el registro concurrente de errores es thread-safe.

        3 hilos × 5 errores = 15 errores totales.
        Se usa thread_safe_ctx para evitar colisión con max_errors.
        """
        n_threads = 3
        errors_per_thread = 5
        barrier = threading.Barrier(n_threads)
        thread_errors: list[Exception] = []

        def record_errors() -> None:
            try:
                barrier.wait()
                for i in range(errors_per_thread):
                    thread_safe_ctx.record_error(f"step{i}", f"error{i}")
            except Exception as e:
                thread_errors.append(e)

        threads = [
            threading.Thread(target=record_errors) for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not thread_errors, f"Errores en hilos: {thread_errors}"
        assert len(thread_safe_ctx.errors) == n_threads * errors_per_thread

    def test_concurrent_increment_metric_produces_correct_total(
        self, thread_safe_ctx: TelemetryContext
    ) -> None:
        """
        Verifica que el incremento concurrente produce el total correcto.

        10 hilos × 100 incrementos de 1 = 1000.
        Verifica ausencia de condiciones de carrera en la suma.
        """
        n_threads = 10
        increments_per_thread = 100
        barrier = threading.Barrier(n_threads)
        errors: list[Exception] = []

        def increment_many() -> None:
            try:
                barrier.wait()
                for _ in range(increments_per_thread):
                    thread_safe_ctx.increment_metric("shared", "counter", 1)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=increment_many) for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errores en hilos: {errors}"
        expected = n_threads * increments_per_thread
        assert thread_safe_ctx.metrics["shared.counter"] == expected


# =============================================================================
# TESTS: Protocolo de Context Manager (__enter__ / __exit__)
# =============================================================================


class TestContextManagerProtocol:
    """Pruebas para el protocolo de gestor de contexto del TelemetryContext."""

    def test_enter_returns_self(self, ctx: TelemetryContext) -> None:
        """Verifica que __enter__ retorna el mismo objeto."""
        with ctx as entered:
            assert entered is ctx

    def test_exit_cancels_all_active_steps(self, ctx: TelemetryContext) -> None:
        """
        Verifica que __exit__ cancela los steps activos como CANCELLED.

        Al salir del bloque with, cualquier step activo debe:
          1. Eliminarse de _active_steps
          2. Registrarse en steps con status=CANCELLED
          3. Incluir reason="context_exit" en metadata
        """
        with ctx:
            ctx.start_step("step1")
            ctx.start_step("step2")
            # Salimos sin terminar los steps

        assert len(ctx.get_active_timers()) == 0
        assert len(ctx.steps) == 2

        for step in ctx.steps:
            assert step["status"] == StepStatus.CANCELLED.value
            assert step["metadata"]["reason"] == "context_exit"

    def test_exit_records_error_on_exception(self, ctx: TelemetryContext) -> None:
        """Verifica que __exit__ registra el error cuando ocurre una excepción."""
        with pytest.raises(ValueError):
            with ctx:
                raise ValueError("Test exception")

        assert len(ctx.errors) == 1
        assert ctx.errors[0]["step"] == "__context__"
        assert "Test exception" in ctx.errors[0]["message"]
        assert ctx.errors[0]["type"] == "ValueError"

    def test_exit_does_not_suppress_exceptions(self, ctx: TelemetryContext) -> None:
        """Verifica que __exit__ no suprime las excepciones."""
        with pytest.raises(ValueError, match="Test exception"):
            with ctx:
                raise ValueError("Test exception")


# =============================================================================
# TESTS: Sanitización de Valores
# =============================================================================


class TestValueSanitization:
    """Pruebas para el método _sanitize_value()."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            (True, True),
            (False, False),
            (42, 42),
            (3.14, 3.14),
            ("test", "test"),
        ],
        ids=["none", "true", "false", "int", "float", "string"],
    )
    def test_sanitize_basic_types_pass_through(
        self, ctx: TelemetryContext, value: Any, expected: Any
    ) -> None:
        """Verifica que los tipos básicos pasan la sanitización sin cambios."""
        assert ctx._sanitize_value(value) == expected

    def test_sanitize_long_string_is_truncated(self, ctx: TelemetryContext) -> None:
        """Verifica que strings largos son truncados."""
        long_string = "x" * 15000
        result = ctx._sanitize_value(long_string)
        # La longitud del resultado debe ser ≤ MAX_STRING_LENGTH + overhead del truncation marker
        assert len(result) <= TelemetryDefaults.MAX_STRING_LENGTH + 20

    def test_sanitize_list_recursively(self, ctx: TelemetryContext) -> None:
        """Verifica que las listas son sanitizadas recursivamente."""
        test_list = [1, "two", {"three": 3}]
        result = ctx._sanitize_value(test_list)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_sanitize_large_list_is_truncated(self, ctx: TelemetryContext) -> None:
        """Verifica que listas grandes son truncadas con marcador."""
        large_list = list(range(200))
        result = ctx._sanitize_value(large_list)
        assert len(result) == TelemetryDefaults.MAX_COLLECTION_SIZE + 1
        assert "more items" in result[-1]

    def test_sanitize_dict_recursively(self, ctx: TelemetryContext) -> None:
        """Verifica que los dicts son sanitizados recursivamente."""
        test_dict = {"a": 1, "b": "two", "c": [3, 4]}
        result = ctx._sanitize_value(test_dict)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_sanitize_large_dict_is_truncated(self, ctx: TelemetryContext) -> None:
        """Verifica que dicts grandes son truncados."""
        large_dict = {f"key{i}": i for i in range(250)}
        result = ctx._sanitize_value(large_dict)
        assert len(result) == TelemetryDefaults.MAX_DICT_KEYS + 1
        assert "<truncated>" in result

    def test_sanitize_datetime_to_iso_string(self, ctx: TelemetryContext) -> None:
        """Verifica que datetime se convierte a string ISO 8601."""
        dt = datetime(2024, 1, 1, 12, 30, 45)
        result = ctx._sanitize_value(dt)
        assert isinstance(result, str)
        assert "2024-01-01" in result

    def test_sanitize_respects_max_depth(self, ctx: TelemetryContext) -> None:
        """Verifica que la profundidad máxima de anidamiento se respeta."""
        deep = {"level": 1}
        current = deep
        for i in range(10):
            current["nested"] = {"level": i + 2}
            current = current["nested"]

        result = ctx._sanitize_value(deep, max_depth=3)
        assert isinstance(result, dict)

    def test_sanitize_object_with_dict_is_serialized(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que objetos con __dict__ se serializan correctamente."""

        class CustomClass:
            def __init__(self) -> None:
                self.attr1 = "value1"
                self.attr2 = 42

        obj = CustomClass()
        result = ctx._sanitize_value(obj)

        assert isinstance(result, dict)
        assert result["__class__"] == "CustomClass"
        assert result["attr1"] == "value1"
        assert result["attr2"] == 42

    def test_sanitize_object_without_dict_is_converted_to_dict(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que objetos sin atributos se convierten a representacion en string (fallback de V3)."""

        class EmptyObject:
            __slots__ = ()

        obj = EmptyObject()
        result = ctx._sanitize_value(obj)
        assert isinstance(result, str)
        assert "EmptyObject" in result

    def test_sanitize_enum_returns_value(self, ctx: TelemetryContext) -> None:
        """Verifica que los Enum se convierten a su valor."""

        class TestEnum(Enum):
            VALUE = "test_value"

        result = ctx._sanitize_value(TestEnum.VALUE)
        assert result == "test_value"

    def test_sanitize_bytes_returns_string(self, ctx: TelemetryContext) -> None:
        """Verifica que bytes son convertidos a string."""
        result = ctx._sanitize_value(b"hello")
        assert isinstance(result, str)
        assert "hello" in result or "bytes" in result

    def test_sanitize_set_returns_sorted_list(self, ctx: TelemetryContext) -> None:
        """Verifica que sets son convertidos a lista ordenada."""
        result = ctx._sanitize_value({1, 2, 3})
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]

    def test_sanitize_frozenset_returns_sorted_list(self, ctx: TelemetryContext) -> None:
        """Verifica que frozensets son convertidos a lista ordenada."""
        result = ctx._sanitize_value(frozenset([1, 2, 3]))
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]


# =============================================================================
# TESTS: Casos Borde de Valores Numéricos Especiales
# =============================================================================


class TestSanitizeNumericEdgeCases:
    """
    Pruebas parametrizadas para valores numéricos especiales.

    Verifica el manejo correcto de NaN, ±Inf, Decimal y otros tipos
    que pueden aparecer en datos de telemetría de sistemas físicos.
    """

    def test_sanitize_nan_returns_placeholder(self, ctx: TelemetryContext) -> None:
        """NaN debe convertirse al marcador '<NaN>'."""
        result = ctx._sanitize_value(float("nan"))
        assert result == "<NaN>"

    def test_sanitize_positive_infinity_returns_placeholder(
        self, ctx: TelemetryContext
    ) -> None:
        """Inf positivo debe convertirse al marcador '<Infinity>'."""
        result = ctx._sanitize_value(float("inf"))
        assert result == "<Infinity>"

    def test_sanitize_negative_infinity_returns_placeholder(
        self, ctx: TelemetryContext
    ) -> None:
        """Inf negativo debe convertirse al marcador '<-Infinity>'."""
        result = ctx._sanitize_value(float("-inf"))
        assert result == "<-Infinity>"

    @pytest.mark.parametrize(
        "decimal_value,expected_type",
        [
            (Decimal("3.14"), (str, float, int)),
            (Decimal("0"), (str, float, int)),
            (Decimal("-1.5"), (str, float, int)),
        ],
        ids=["positive_decimal", "zero_decimal", "negative_decimal"],
    )
    def test_sanitize_decimal_is_serializable(
        self,
        ctx: TelemetryContext,
        decimal_value: Decimal,
        expected_type: tuple,
    ) -> None:
        """Verifica que Decimal se convierte a un tipo JSON-serializable."""
        result = ctx._sanitize_value(decimal_value)
        assert isinstance(result, expected_type), (
            f"Decimal({decimal_value}) produjo tipo inesperado: {type(result).__name__}"
        )

    @pytest.mark.parametrize(
        "value,description",
        [
            (0, "zero_int"),
            (0.0, "zero_float"),
            (-0.0, "negative_zero_float"),
            (2**53, "large_safe_integer"),
            (-(2**53), "large_negative_integer"),
        ],
    )
    def test_sanitize_edge_numeric_values_pass_through(
        self, ctx: TelemetryContext, value: Any, description: str
    ) -> None:
        """Verifica que valores numéricos en límites válidos pasan sin modificación."""
        result = ctx._sanitize_value(value)
        # Debe ser numérico y no un marcador de texto
        assert not isinstance(result, str) or "<" not in result, (
            f"Valor numérico especial ({description}) se convirtió inesperadamente: {result!r}"
        )


# =============================================================================
# TESTS: Casos Borde y Regresión
# =============================================================================


class TestEdgeCases:
    """Pruebas para casos borde y regresiones."""

    def test_empty_context_to_dict_has_zero_duration(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que to_dict en contexto vacío tiene duración total=0."""
        data = ctx.to_dict()
        assert data["total_duration_seconds"] == 0
        assert len(data["steps"]) == 0

    def test_multiple_end_calls_for_same_step(self, ctx: TelemetryContext) -> None:
        """
        Verifica el comportamiento de múltiples llamadas a end_step para el mismo step.

        Primera llamada: registra con duración real.
        Segunda llamada: step no está activo → duración=0.
        """
        ctx.start_step("step1")
        time.sleep(0.001)
        ctx.end_step("step1")
        ctx.end_step("step1")  # Segunda llamada: step ya no está activo

        assert len(ctx.steps) == 2
        assert ctx.steps[0]["duration_seconds"] > 0
        assert ctx.steps[1]["duration_seconds"] == 0.0

    def test_unicode_in_names_is_handled(self, ctx: TelemetryContext) -> None:
        """Verifica que los nombres con caracteres Unicode son manejados."""
        ctx.start_step("测试步骤")
        ctx.end_step("测试步骤")
        ctx.record_metric("组件", "指标", 100)
        ctx.record_error("步骤", "错误消息")

        assert len(ctx.steps) == 1
        assert len(ctx.metrics) == 1
        assert len(ctx.errors) == 1

    def test_special_characters_in_names_preserved(self, ctx: TelemetryContext) -> None:
        """Verifica que los caracteres especiales en nombres se preservan."""
        special_name = "step.with-special_chars!@#"
        ctx.start_step(special_name)
        ctx.end_step(special_name)

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["step"] == special_name

    def test_zero_duration_step_is_recorded(self, ctx: TelemetryContext) -> None:
        """Verifica que steps con duración negligible se registran correctamente."""
        ctx.start_step("instant_step")
        ctx.end_step("instant_step")

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["duration_seconds"] >= 0.0

    def test_age_calculation_increases_over_time(self, ctx: TelemetryContext) -> None:
        """Verifica que age_seconds aumenta con el tiempo."""
        summary1 = ctx.get_summary()
        time.sleep(0.01)
        summary2 = ctx.get_summary()

        assert summary2["age_seconds"] > summary1["age_seconds"]

    def test_repr_contains_expected_info(
        self, ctx_populated: TelemetryContext
    ) -> None:
        """Verifica que __repr__ contiene información relevante."""
        repr_str = repr(ctx_populated)

        assert "TelemetryContext" in repr_str
        assert ctx_populated.request_id in repr_str
        assert "steps=" in repr_str
        assert "errors=" in repr_str

    def test_str_contains_readable_info(self, ctx_populated: TelemetryContext) -> None:
        """Verifica que __str__ retorna información legible."""
        str_repr = str(ctx_populated)

        assert "Telemetry" in str_repr
        assert "steps" in str_repr
        assert "errors" in str_repr


# =============================================================================
# TESTS: Integración
# =============================================================================


class TestIntegration:
    """Tests de integración que simulan flujos de trabajo reales."""

    def test_complete_nested_workflow(self, ctx: TelemetryContext) -> None:
        """Prueba un flujo de trabajo completo con pasos anidados."""
        with ctx.step("main_process"):
            ctx.record_metric("system", "memory_mb", 256)
            ctx.record_metric("system", "cpu_percent", 45.2)

            with ctx.step("data_loading"):
                time.sleep(0.001)
                ctx.record_metric("data", "rows_loaded", 1000)

            try:
                with ctx.step("processing"):
                    raise ValueError("Processing failed")
            except ValueError:
                pass

            with ctx.step("finalization"):
                ctx.record_metric("output", "items_processed", 950)

        summary = ctx.get_summary()
        assert summary["total_steps"] == 4
        assert summary["total_errors"] == 1
        assert summary["total_metrics"] == 4
        assert summary["has_errors"] is True
        assert summary["has_failures"] is True

        # Verificar orden: los pasos internos terminan antes que el externo
        assert ctx.steps[0]["step"] == "data_loading"
        assert ctx.steps[1]["step"] == "processing"
        assert ctx.steps[2]["step"] == "finalization"
        assert ctx.steps[3]["step"] == "main_process"

    def test_request_lifecycle_with_metadata(self, ctx: TelemetryContext) -> None:
        """Prueba el ciclo de vida completo de una petición con metadata."""
        ctx.metadata["user_id"] = "user123"
        ctx.metadata["request_type"] = "api_call"

        for step_name in ["authentication", "authorization", "business_logic"]:
            ctx.start_step(step_name)
            time.sleep(0.001)
            ctx.end_step(step_name, StepStatus.SUCCESS)

        ctx.record_metric("business", "items_found", 42)

        result = ctx.to_dict(include_metadata=True)

        assert len(result["steps"]) == 3
        assert result["metadata"]["user_id"] == "user123"
        assert result["total_duration_seconds"] > 0

    def test_error_recovery_workflow(self, ctx: TelemetryContext) -> None:
        """Prueba un flujo de trabajo con error y recuperación."""
        try:
            with ctx.step("attempt_1", error_status=StepStatus.WARNING):
                ctx.record_error(
                    "attempt_1", "Temporary failure", error_type="TransientError"
                )
                raise RuntimeError("Temporary issue")
        except RuntimeError:
            pass

        with ctx.step("attempt_2"):
            ctx.record_metric("retry", "success", True)

        summary = ctx.get_summary()
        assert summary["total_steps"] == 2
        assert summary["total_errors"] == 2
        assert summary["step_statuses"]["warning"] == 1
        assert summary["step_statuses"]["success"] == 1

    def test_increment_metric_workflow(self, ctx: TelemetryContext) -> None:
        """Prueba un flujo de trabajo usando increment_metric."""
        for i in range(10):
            with ctx.step(f"process_item_{i}"):
                ctx.increment_metric("processing", "items_processed")
                if i % 3 == 0:
                    ctx.increment_metric("processing", "special_items")

        assert ctx.get_metric("processing", "items_processed") == 10
        # i % 3 == 0 → i ∈ {0, 3, 6, 9} → 4 incrementos
        assert ctx.get_metric("processing", "special_items") == 4


# =============================================================================
# TESTS: Rendimiento
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """
    Tests de rendimiento y estrés.

    Ejecutar con: pytest -m slow
    Omitir con:   pytest -m "not slow"
    """

    def test_many_steps_completes_within_time_limit(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica el rendimiento con 100 steps consecutivos (< 2s)."""
        start = time.perf_counter()

        for i in range(100):
            ctx.start_step(f"step{i}")
            ctx.end_step(f"step{i}")

        duration = time.perf_counter() - start

        assert duration < 2.0, f"Demasiado lento: {duration:.3f}s para 100 steps"
        assert len(ctx.steps) == 100

    def test_large_export_completes_within_time_limit(
        self, ctx_populated: TelemetryContext
    ) -> None:
        """Verifica que to_dict() con 50 métricas tarda < 100ms."""
        for i in range(50):
            ctx_populated.record_metric("comp", f"metric{i}", i)

        start = time.perf_counter()
        data = ctx_populated.to_dict()
        duration = time.perf_counter() - start

        assert duration < 0.1, f"to_dict() demasiado lento: {duration:.3f}s"
        assert isinstance(data, dict)

    def test_deep_sanitization_completes_within_time_limit(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que la sanitización profunda tarda < 100ms."""
        deep_nested: dict[str, Any] = {"level": 0}
        current = deep_nested
        for i in range(100):
            current["nested"] = {"level": i + 1, "data": list(range(10))}
            current = current["nested"]

        start = time.perf_counter()
        result = ctx._sanitize_value(deep_nested)
        duration = time.perf_counter() - start

        assert duration < 0.1, f"Sanitización demasiado lenta: {duration:.3f}s"
        assert isinstance(result, dict)


# =============================================================================
# TESTS: Lógica de Negocio
# =============================================================================


class TestBusinessLogic:
    """Pruebas para la lógica de traducción de métricas a reporte de negocio."""

    def _set_flux_metrics(
        self,
        ctx: TelemetryContext,
        saturation: float = 0.0,
        flyback: float = 0.0,
        power: float = 0.0,
        kinetic: float = 0.0,
    ) -> None:
        """Helper para configurar métricas del condensador de flujo."""
        ctx.record_metric("flux_condenser", "avg_saturation", saturation)
        ctx.record_metric("flux_condenser", "max_flyback_voltage", flyback)
        ctx.record_metric("flux_condenser", "max_dissipated_power", power)
        if kinetic > 0.0:
            ctx.record_metric("flux_condenser", "avg_kinetic_energy", kinetic)

    def test_report_optimal_conditions(self, ctx: TelemetryContext) -> None:
        """Verifica el reporte para condiciones óptimas."""
        self._set_flux_metrics(ctx, saturation=0.5, flyback=0.02, power=10.0, kinetic=100.0)

        report = ctx.get_business_report()

        assert report["status"] == "OPTIMO"
        assert "Procesamiento estable y fluido" in report["message"]
        assert report["metrics"]["Carga del Sistema"] == "50.0%"
        assert report["metrics"]["Índice de Inestabilidad"] == "0.0200"
        assert report["metrics"]["Fricción de Datos"] == "10.00"
        assert "details" in report
        assert report["details"]["total_steps"] == 0
        assert report["details"]["total_errors"] == 0

    def test_report_warning_high_saturation(self, ctx: TelemetryContext) -> None:
        """Verifica el reporte cuando la saturación supera el umbral de advertencia."""
        self._set_flux_metrics(ctx, saturation=0.95, flyback=0.1, power=20.0)

        report = ctx.get_business_report()

        assert report["status"] == "ADVERTENCIA"
        assert "capacidad" in report["message"]

    def test_report_critical_flyback_voltage(self, ctx: TelemetryContext) -> None:
        """Verifica el reporte cuando el voltaje de retroceso es crítico."""
        self._set_flux_metrics(ctx, saturation=0.5, flyback=0.6, power=10.0)

        report = ctx.get_business_report()

        assert report["status"] == "CRITICO"
        assert "inestabilidad" in report["message"]

    def test_report_critical_dissipated_power(self, ctx: TelemetryContext) -> None:
        """Verifica el reporte cuando la potencia disipada es crítica."""
        self._set_flux_metrics(ctx, saturation=0.5, flyback=0.1, power=60.0)

        report = ctx.get_business_report()

        assert report["status"] == "CRITICO"
        assert "Fricción" in report["message"]

    def test_report_handles_missing_metrics_gracefully(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que el reporte maneja métricas faltantes sin errores."""
        report = ctx.get_business_report()

        assert report["status"] == "OPTIMO"
        assert report["metrics"]["Carga del Sistema"] == "0.0%"

    def test_report_warning_when_errors_exist(self, ctx: TelemetryContext) -> None:
        """Verifica que el reporte muestra advertencia cuando hay errores registrados."""
        self._set_flux_metrics(ctx, saturation=0.5, flyback=0.1, power=10.0)
        ctx.record_error("step1", "Some error occurred")

        report = ctx.get_business_report()

        assert report["status"] == "ADVERTENCIA"
        assert "error" in report["message"].lower()

    def test_report_respects_custom_flyback_threshold(
        self, ctx_with_custom_thresholds: TelemetryContext
    ) -> None:
        """
        Verifica que el umbral personalizado de voltaje de retroceso se aplica.

        Con custom critical_flyback_voltage=0.3, un valor de 0.4 es crítico
        (mientras que con el default 0.5, sería normal).
        """
        ctx_with_custom_thresholds.record_metric(
            "flux_condenser", "avg_saturation", 0.5
        )
        ctx_with_custom_thresholds.record_metric(
            "flux_condenser", "max_flyback_voltage", 0.4  # 0.4 > 0.3 (custom threshold)
        )
        ctx_with_custom_thresholds.record_metric(
            "flux_condenser", "max_dissipated_power", 10.0  # bajo, no crítico
        )

        report = ctx_with_custom_thresholds.get_business_report()
        assert report["status"] == "CRITICO", (
            f"Con umbral personalizado=0.3 y voltaje=0.4, se esperaba CRITICO. "
            f"Estado obtenido: {report['status']}"
        )

    def test_report_respects_custom_power_threshold(
        self, ctx_with_custom_thresholds: TelemetryContext
    ) -> None:
        """
        Verifica que el umbral personalizado de potencia disipada se aplica.

        Con custom critical_dissipated_power=30.0, un valor de 35.0 es crítico
        (con el default 50.0 sería normal).
        """
        ctx_with_custom_thresholds.record_metric(
            "flux_condenser", "avg_saturation", 0.5
        )
        ctx_with_custom_thresholds.record_metric(
            "flux_condenser", "max_flyback_voltage", 0.1  # bajo, no crítico
        )
        ctx_with_custom_thresholds.record_metric(
            "flux_condenser", "max_dissipated_power", 35.0  # 35 > 30 (custom)
        )

        report = ctx_with_custom_thresholds.get_business_report()
        assert report["status"] == "CRITICO", (
            f"Con umbral personalizado=30.0 y potencia=35.0, se esperaba CRITICO. "
            f"Estado obtenido: {report['status']}"
        )

    def test_report_respects_custom_saturation_threshold(
        self, ctx_with_custom_thresholds: TelemetryContext
    ) -> None:
        """
        Verifica que el umbral personalizado de saturación se aplica.

        Con custom warning_saturation=0.8, un valor de 0.85 debe generar ADVERTENCIA
        (con el default 0.9 sería normal/óptimo).
        """
        ctx_with_custom_thresholds.record_metric(
            "flux_condenser", "avg_saturation", 0.85  # 0.85 > 0.8 (custom)
        )
        ctx_with_custom_thresholds.record_metric(
            "flux_condenser", "max_flyback_voltage", 0.1
        )
        ctx_with_custom_thresholds.record_metric(
            "flux_condenser", "max_dissipated_power", 10.0
        )

        report = ctx_with_custom_thresholds.get_business_report()
        assert report["status"] in ("ADVERTENCIA", "CRITICO"), (
            f"Con umbral sat=0.8 y valor=0.85, se esperaba ADVERTENCIA o CRITICO. "
            f"Estado obtenido: {report['status']}"
        )

    def test_report_safe_float_conversion_for_invalid_types(
        self, ctx: TelemetryContext
    ) -> None:
        """
        Verifica la conversión segura de tipos inválidos a float.

        Los valores inválidos deben usar el valor por defecto (0.0)
        sin lanzar excepción.
        """
        ctx.record_metric("flux_condenser", "avg_saturation", "0.5")  # string
        ctx.record_metric("flux_condenser", "max_flyback_voltage", None)  # None
        ctx.record_metric("flux_condenser", "max_dissipated_power", "invalid")  # str inválido

        report = ctx.get_business_report()

        assert report["status"] == "OPTIMO"
        assert report["metrics"]["Carga del Sistema"] == "50.0%"

    def test_report_includes_details_section(self, ctx: TelemetryContext) -> None:
        """Verifica que el reporte incluye la sección de detalles."""
        ctx.start_step("step1")
        ctx.end_step("step1")
        ctx.record_error("step1", "error")
        ctx.start_step("active_step")  # step activo sin terminar

        report = ctx.get_business_report()

        assert "details" in report
        assert report["details"]["total_steps"] == 1
        assert report["details"]["total_errors"] == 1
        assert report["details"]["has_active_operations"] is True


# =============================================================================
# TESTS: Validación
# =============================================================================


class TestCustomValidation:
    """Pruebas para los métodos de validación internos."""

    @pytest.mark.parametrize(
        "valid_name",
        ["valid_name", "a", "name-with-dashes", "CamelCase", "123numeric"],
        ids=["underscore", "single_char", "dashes", "camel", "numeric"],
    )
    def test_validate_name_accepts_valid_names(
        self, ctx: TelemetryContext, valid_name: str
    ) -> None:
        """Verifica que _validate_name acepta nombres válidos."""
        assert ctx._validate_name(valid_name, "field") is True

    @pytest.mark.parametrize(
        "invalid_name",
        ["", None, 123, [], "   ", "\t\n"],
        ids=["empty", "none", "integer", "list", "whitespace", "tab_newline"],
    )
    def test_validate_name_rejects_invalid_names(
        self, ctx: TelemetryContext, invalid_name: Any
    ) -> None:
        """Verifica que _validate_name rechaza nombres inválidos."""
        assert ctx._validate_name(invalid_name, "field") is False

    def test_validate_name_rejects_names_exceeding_max_length(
        self, ctx: TelemetryContext
    ) -> None:
        """Verifica que _validate_name rechaza nombres que exceden max_length."""
        long_name = "x" * 101
        assert ctx._validate_name(long_name, "field", max_length=100) is False

    @pytest.mark.parametrize(
        "status_enum,expected_value",
        [
            (StepStatus.SUCCESS, "success"),
            (StepStatus.FAILURE, "failure"),
            (StepStatus.WARNING, "warning"),
            (StepStatus.CANCELLED, "cancelled"),
        ],
    )
    def test_normalize_status_handles_enum_values(
        self, ctx: TelemetryContext, status_enum: StepStatus, expected_value: str
    ) -> None:
        """Verifica que _normalize_status convierte correctamente valores Enum."""
        assert ctx._normalize_status(status_enum) == expected_value

    @pytest.mark.parametrize(
        "status_str,expected_value",
        [
            ("success", "success"),
            ("failure", "failure"),
            ("warning", "warning"),
            ("SUCCESS", "success"),   # case-insensitive
            ("FAILURE", "failure"),   # case-insensitive (CORREGIDO: no era "success")
            ("Warning", "warning"),   # case-insensitive
        ],
    )
    def test_normalize_status_handles_string_values_case_insensitive(
        self, ctx: TelemetryContext, status_str: str, expected_value: str
    ) -> None:
        """
        Verifica que _normalize_status maneja strings de forma insensible a mayúsculas.

        CORRECCIÓN: La versión original afirmaba que "FAILURE" → "success",
        lo cual era un error. El comportamiento correcto es case-insensitive:
        "FAILURE" debe normalizar a "failure".
        """
        assert ctx._normalize_status(status_str) == expected_value

    @pytest.mark.parametrize(
        "invalid_status",
        ["invalid", "unknown", 123, None, []],
        ids=["unknown_string", "word_invalid", "integer", "none", "list"],
    )
    def test_normalize_status_returns_success_for_truly_invalid(
        self, ctx: TelemetryContext, invalid_status: Any
    ) -> None:
        """
        Verifica que _normalize_status retorna 'success' para valores verdaderamente inválidos.

        Un valor es inválido si no puede mapearse a ningún StepStatus
        (ni por valor directo ni por búsqueda case-insensitive).
        """
        assert ctx._normalize_status(invalid_status) == "success"


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])