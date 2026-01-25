"""
Suite de pruebas para el módulo de Telemetría Unificada.

Pruebas exhaustivas para TelemetryContext, incluyendo thread-safety,
límites, serialización y lógica de negocio.
"""

import threading
import time
from datetime import datetime
from enum import Enum
from typing import Any

import pytest

from app.telemetry import (
    ActiveStepInfo,
    BusinessThresholds,
    StepStatus,
    TelemetryContext,
    TelemetryDefaults,
)

# ========== Fixtures ==========


@pytest.fixture
def ctx() -> TelemetryContext:
    """Provides a fresh TelemetryContext instance for each test."""
    return TelemetryContext()


@pytest.fixture
def ctx_with_custom_id() -> TelemetryContext:
    """Provides a TelemetryContext with a custom request_id."""
    return TelemetryContext(request_id="custom-test-id-12345")


@pytest.fixture
def ctx_with_limits() -> TelemetryContext:
    """Provides a TelemetryContext with custom limits for testing boundaries."""
    return TelemetryContext(max_steps=3, max_errors=2, max_metrics=5)


@pytest.fixture
def ctx_with_custom_thresholds() -> TelemetryContext:
    """Provides a TelemetryContext with custom business thresholds."""
    return TelemetryContext(
        business_thresholds={
            "critical_flyback_voltage": 0.3,
            "critical_dissipated_power": 30.0,
            "warning_saturation": 0.8,
        }
    )


@pytest.fixture
def ctx_populated(ctx: TelemetryContext) -> TelemetryContext:
    """Provides a TelemetryContext with some pre-populated data."""
    ctx.start_step("step1")
    time.sleep(0.001)
    ctx.end_step("step1", StepStatus.SUCCESS)

    ctx.record_metric("component1", "metric1", 100)
    ctx.record_metric("component1", "metric2", 200)
    ctx.record_error("step1", "test error", error_type="TestError")

    return ctx


# ========== Configuration Classes Tests ==========


class TestConfigurationClasses:
    """Tests for configuration classes."""

    def test_telemetry_defaults_values(self):
        """Verify TelemetryDefaults has expected default values."""
        assert TelemetryDefaults.MAX_STEPS == 1000
        assert TelemetryDefaults.MAX_ERRORS == 100
        assert TelemetryDefaults.MAX_METRICS == 500
        assert TelemetryDefaults.MAX_STRING_LENGTH == 10000
        assert TelemetryDefaults.MAX_MESSAGE_LENGTH == 1000
        assert TelemetryDefaults.MAX_RECURSION_DEPTH == 5
        assert TelemetryDefaults.MAX_COLLECTION_SIZE == 100

    def test_business_thresholds_values(self):
        """Verify BusinessThresholds has expected default values."""
        assert BusinessThresholds.CRITICAL_FLYBACK_VOLTAGE == 0.5
        assert BusinessThresholds.CRITICAL_DISSIPATED_POWER == 50.0
        assert BusinessThresholds.WARNING_SATURATION == 0.9

    def test_metric_prefixes_evolution(self):
        """Verify metric prefixes include new physics and tactics domains."""
        from app.telemetry import Stratum, StratumTopology

        # PHYSICS
        physics_prefixes = StratumTopology.get_prefixes(Stratum.PHYSICS)
        assert "gyro" in physics_prefixes
        assert "nutation" in physics_prefixes
        assert "laplace" in physics_prefixes
        assert "thermo" in physics_prefixes

        # TACTICS
        tactics_prefixes = StratumTopology.get_prefixes(Stratum.TACTICS)
        assert "spectral" in tactics_prefixes
        assert "fiedler" in tactics_prefixes
        assert "resonance" in tactics_prefixes

        # STRATEGY
        strategy_prefixes = StratumTopology.get_prefixes(Stratum.STRATEGY)
        assert "real_options" in strategy_prefixes

    def test_active_step_info_creation(self):
        """Verify ActiveStepInfo can be created correctly."""
        start_time = time.perf_counter()
        metadata = {"key": "value"}

        info = ActiveStepInfo(start_time=start_time, metadata=metadata)

        assert info.start_time == start_time
        assert info.metadata == metadata

    def test_active_step_info_get_duration(self):
        """Verify ActiveStepInfo calculates duration correctly."""
        info = ActiveStepInfo(start_time=time.perf_counter())
        time.sleep(0.01)

        duration = info.get_duration()
        assert duration >= 0.01
        assert duration < 1.0


# ========== StepStatus Enum Tests ==========


class TestStepStatusEnum:
    """Tests for StepStatus enum."""

    def test_step_status_values(self):
        """Verify all StepStatus values exist."""
        assert StepStatus.SUCCESS.value == "success"
        assert StepStatus.FAILURE.value == "failure"
        assert StepStatus.WARNING.value == "warning"
        assert StepStatus.SKIPPED.value == "skipped"
        assert StepStatus.IN_PROGRESS.value == "in_progress"
        assert StepStatus.CANCELLED.value == "cancelled"

    def test_step_status_from_string_valid(self):
        """Verify from_string converts valid strings correctly."""
        assert StepStatus.from_string("success") == StepStatus.SUCCESS
        assert StepStatus.from_string("failure") == StepStatus.FAILURE
        assert StepStatus.from_string("warning") == StepStatus.WARNING
        assert StepStatus.from_string("skipped") == StepStatus.SKIPPED
        assert StepStatus.from_string("in_progress") == StepStatus.IN_PROGRESS
        assert StepStatus.from_string("cancelled") == StepStatus.CANCELLED

    def test_step_status_from_string_case_insensitive(self):
        """Verify from_string handles case variations."""
        assert StepStatus.from_string("SUCCESS") == StepStatus.SUCCESS
        assert StepStatus.from_string("Success") == StepStatus.SUCCESS
        assert StepStatus.from_string("  success  ") == StepStatus.SUCCESS

    def test_step_status_from_string_invalid(self):
        """Verify from_string returns SUCCESS for invalid values."""
        assert StepStatus.from_string("invalid") == StepStatus.SUCCESS
        assert StepStatus.from_string("") == StepStatus.SUCCESS
        assert StepStatus.from_string("unknown_status") == StepStatus.SUCCESS

    def test_step_status_from_string_non_string(self):
        """Verify from_string handles non-string input."""
        assert StepStatus.from_string(123) == StepStatus.SUCCESS
        assert StepStatus.from_string(None) == StepStatus.SUCCESS
        assert StepStatus.from_string([]) == StepStatus.SUCCESS


# ========== Initialization Tests ==========


class TestInitialization:
    """Tests for TelemetryContext initialization and setup."""

    def test_default_initialization(self, ctx: TelemetryContext):
        """Verify default initial state of TelemetryContext."""
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

    def test_default_business_thresholds(self, ctx: TelemetryContext):
        """Verify default business thresholds are set."""
        assert "critical_flyback_voltage" in ctx.business_thresholds
        assert "critical_dissipated_power" in ctx.business_thresholds
        assert "warning_saturation" in ctx.business_thresholds

        assert ctx.business_thresholds["critical_flyback_voltage"] == 0.5
        assert ctx.business_thresholds["critical_dissipated_power"] == 50.0
        assert ctx.business_thresholds["warning_saturation"] == 0.9

    def test_custom_request_id(self, ctx_with_custom_id: TelemetryContext):
        """Verify custom request_id is preserved."""
        assert ctx_with_custom_id.request_id == "custom-test-id-12345"

    def test_invalid_request_id_empty_string(self):
        """Verify empty string request_id triggers generation of new UUID."""
        ctx = TelemetryContext(request_id="")
        assert ctx.request_id != ""
        assert len(ctx.request_id) > 0

    def test_invalid_request_id_none(self):
        """Verify None request_id triggers generation of new UUID."""
        ctx = TelemetryContext(request_id=None)
        assert ctx.request_id is not None
        assert len(ctx.request_id) > 0

    def test_invalid_request_id_non_string(self):
        """Verify non-string request_id triggers generation of new UUID."""
        ctx = TelemetryContext(request_id=12345)
        assert isinstance(ctx.request_id, str)
        assert len(ctx.request_id) > 0

    def test_request_id_too_long(self):
        """Verify overly long request_id triggers generation of new UUID."""
        long_id = "x" * 500
        ctx = TelemetryContext(request_id=long_id)
        assert ctx.request_id != long_id
        assert len(ctx.request_id) <= TelemetryDefaults.MAX_REQUEST_ID_LENGTH

    def test_custom_limits(self, ctx_with_limits: TelemetryContext):
        """Verify custom limits are set correctly."""
        assert ctx_with_limits.max_steps == 3
        assert ctx_with_limits.max_errors == 2
        assert ctx_with_limits.max_metrics == 5

    def test_negative_limits_normalized(self):
        """Verify negative limits are normalized to positive values."""
        ctx = TelemetryContext(max_steps=-10, max_errors=0, max_metrics=-5)
        assert ctx.max_steps >= 1
        assert ctx.max_errors >= 1
        assert ctx.max_metrics >= 1

    def test_excessively_large_limits_capped(self):
        """Verify excessively large limits are capped."""
        ctx = TelemetryContext(max_steps=999999999)
        max_allowed = TelemetryDefaults.MAX_STEPS * TelemetryDefaults.MAX_LIMIT_MULTIPLIER
        assert ctx.max_steps <= max_allowed

    def test_non_integer_limits_handled(self):
        """Verify non-integer limits are handled gracefully."""
        ctx = TelemetryContext(max_steps="invalid", max_errors=None)
        assert ctx.max_steps == TelemetryDefaults.MAX_STEPS
        assert ctx.max_errors == TelemetryDefaults.MAX_ERRORS

    def test_unique_request_ids(self):
        """Verify each instance gets unique request_id by default."""
        ctx1 = TelemetryContext()
        ctx2 = TelemetryContext()
        assert ctx1.request_id != ctx2.request_id

    def test_invalid_collection_types_fixed(self):
        """Verify invalid collection types are fixed during initialization."""
        ctx = TelemetryContext()
        # Manually corrupt the collections
        ctx.steps = "not a list"
        ctx.metrics = "not a dict"
        ctx._validate_and_fix_collection_types()

        assert isinstance(ctx.steps, list)
        assert isinstance(ctx.metrics, dict)


# ========== Step Tracking Tests ==========


class TestStepTracking:
    """Tests for step start, end, and tracking logic."""

    def test_start_step_basic(self, ctx: TelemetryContext):
        """Verify basic step start functionality."""
        step_name = "test_step"
        result = ctx.start_step(step_name)

        assert result is True
        assert step_name in ctx._active_steps
        assert isinstance(ctx._active_steps[step_name], ActiveStepInfo)
        assert ctx._active_steps[step_name].start_time > 0

    def test_start_step_with_metadata(self, ctx: TelemetryContext):
        """Verify step start with metadata stores metadata."""
        metadata = {"key": "value", "count": 42}
        result = ctx.start_step("test_step", metadata=metadata)

        assert result is True
        assert "test_step" in ctx._active_steps
        assert ctx._active_steps["test_step"].metadata is not None
        assert ctx._active_steps["test_step"].metadata["key"] == "value"

    def test_start_step_metadata_sanitized(self, ctx: TelemetryContext):
        """Verify step metadata is sanitized on start."""
        metadata = {"datetime": datetime.now(), "nested": {"deep": "value"}}
        ctx.start_step("test_step", metadata=metadata)

        stored_metadata = ctx._active_steps["test_step"].metadata
        assert isinstance(stored_metadata["datetime"], str)  # ISO format
        assert isinstance(stored_metadata["nested"], dict)

    def test_start_step_invalid_name_empty(self, ctx: TelemetryContext):
        """Verify start_step rejects empty string."""
        assert ctx.start_step("") is False

    def test_start_step_invalid_name_none(self, ctx: TelemetryContext):
        """Verify start_step rejects None."""
        assert ctx.start_step(None) is False

    def test_start_step_invalid_name_non_string(self, ctx: TelemetryContext):
        """Verify start_step rejects non-string types."""
        assert ctx.start_step(123) is False
        assert ctx.start_step([]) is False
        assert ctx.start_step({}) is False

    def test_start_step_invalid_name_whitespace_only(self, ctx: TelemetryContext):
        """Verify start_step rejects whitespace-only names."""
        assert ctx.start_step("   ") is False
        assert ctx.start_step("\t\n") is False

    def test_start_step_too_long_name(self, ctx: TelemetryContext):
        """Verify start_step rejects names exceeding max length."""
        long_name = "x" * (TelemetryDefaults.MAX_STEP_NAME_LENGTH + 1)
        assert ctx.start_step(long_name) is False

    def test_start_step_max_length_name_accepted(self, ctx: TelemetryContext):
        """Verify start_step accepts names at max length."""
        max_name = "x" * TelemetryDefaults.MAX_STEP_NAME_LENGTH
        assert ctx.start_step(max_name) is True

    def test_start_step_duplicate_resets_timer(self, ctx: TelemetryContext):
        """Verify duplicate start_step resets timer."""
        step_name = "duplicate_step"

        ctx.start_step(step_name)
        first_time = ctx._active_steps[step_name].start_time

        time.sleep(0.001)

        ctx.start_step(step_name)
        second_time = ctx._active_steps[step_name].start_time

        assert second_time != first_time
        assert second_time > first_time

    def test_end_step_basic(self, ctx: TelemetryContext):
        """Verify basic step end functionality."""
        step_name = "test_step"

        ctx.start_step(step_name)
        time.sleep(0.01)
        result = ctx.end_step(step_name, StepStatus.SUCCESS)

        assert result is True
        assert step_name not in ctx._active_steps
        assert len(ctx.steps) == 1

        step_data = ctx.steps[0]
        assert step_data["step"] == step_name
        assert step_data["status"] == StepStatus.SUCCESS.value
        assert step_data["duration_seconds"] > 0
        assert step_data["duration_seconds"] < 1.0
        assert "timestamp" in step_data
        assert "perf_counter" in step_data

    def test_end_step_with_metadata(self, ctx: TelemetryContext):
        """Verify end_step with metadata."""
        ctx.start_step("test_step")
        metadata = {"result": "ok", "count": 5}
        ctx.end_step("test_step", StepStatus.SUCCESS, metadata=metadata)

        assert len(ctx.steps) == 1
        assert "metadata" in ctx.steps[0]
        assert ctx.steps[0]["metadata"]["result"] == "ok"

    def test_end_step_combines_start_and_end_metadata(self, ctx: TelemetryContext):
        """Verify end_step combines metadata from start and end."""
        start_metadata = {"start_key": "start_value", "shared": "from_start"}
        end_metadata = {"end_key": "end_value", "shared": "from_end"}

        ctx.start_step("test_step", metadata=start_metadata)
        ctx.end_step("test_step", StepStatus.SUCCESS, metadata=end_metadata)

        combined = ctx.steps[0]["metadata"]
        assert combined["start_key"] == "start_value"
        assert combined["end_key"] == "end_value"
        assert combined["shared"] == "from_end"  # End overwrites

    def test_end_step_preserves_start_metadata_when_no_end_metadata(
        self, ctx: TelemetryContext
    ):
        """Verify start metadata is preserved when no end metadata provided."""
        start_metadata = {"key": "value"}
        ctx.start_step("test_step", metadata=start_metadata)
        ctx.end_step("test_step", StepStatus.SUCCESS)

        assert ctx.steps[0]["metadata"]["key"] == "value"

    @pytest.mark.parametrize(
        "status,expected",
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
        self, ctx: TelemetryContext, status: Any, expected: str
    ):
        """Verify different status formats are normalized correctly."""
        ctx.start_step("test_step")
        ctx.end_step("test_step", status)

        assert ctx.steps[0]["status"] == expected

    def test_end_step_invalid_status_defaults_to_success(self, ctx: TelemetryContext):
        """Verify invalid status defaults to success."""
        ctx.start_step("test_step")
        ctx.end_step("test_step", "invalid_status_xyz")

        assert ctx.steps[0]["status"] == StepStatus.SUCCESS.value

    def test_end_step_non_string_status_defaults_to_success(self, ctx: TelemetryContext):
        """Verify non-string status defaults to success."""
        ctx.start_step("test_step")
        ctx.end_step("test_step", 123)

        assert ctx.steps[0]["status"] == StepStatus.SUCCESS.value

    def test_end_step_without_start(self, ctx: TelemetryContext):
        """Verify end_step without start_step sets duration to 0."""
        result = ctx.end_step("never_started")

        assert result is True
        assert len(ctx.steps) == 1
        assert ctx.steps[0]["duration_seconds"] == 0.0
        assert ctx.steps[0]["step"] == "never_started"

    def test_end_step_invalid_name(self, ctx: TelemetryContext):
        """Verify end_step rejects invalid names."""
        assert ctx.end_step("") is False
        assert ctx.end_step(None) is False
        assert ctx.end_step(123) is False

    def test_multiple_steps_sequence(self, ctx: TelemetryContext):
        """Verify multiple sequential steps are tracked correctly."""
        steps = ["step1", "step2", "step3"]

        for step in steps:
            ctx.start_step(step)
            time.sleep(0.001)
            ctx.end_step(step)

        assert len(ctx.steps) == 3
        for i, step in enumerate(steps):
            assert ctx.steps[i]["step"] == step
            assert ctx.steps[i]["duration_seconds"] > 0

    def test_nested_steps_tracking(self, ctx: TelemetryContext):
        """Verify nested/overlapping steps are tracked independently."""
        ctx.start_step("outer")
        time.sleep(0.001)
        ctx.start_step("inner")
        time.sleep(0.001)
        ctx.end_step("inner")
        time.sleep(0.001)
        ctx.end_step("outer")

        assert len(ctx.steps) == 2
        outer_duration = ctx.steps[1]["duration_seconds"]
        inner_duration = ctx.steps[0]["duration_seconds"]
        assert outer_duration > inner_duration


# ========== Context Manager Tests ==========


class TestStepContextManager:
    """Tests for the step() context manager."""

    def test_context_manager_success(self, ctx: TelemetryContext):
        """Verify context manager tracks successful step."""
        with ctx.step("test_operation"):
            time.sleep(0.001)

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["step"] == "test_operation"
        assert ctx.steps[0]["status"] == StepStatus.SUCCESS.value
        assert ctx.steps[0]["duration_seconds"] > 0

    def test_context_manager_with_exception(self, ctx: TelemetryContext):
        """Verify context manager handles exceptions and records error."""
        with pytest.raises(ValueError):
            with ctx.step("failing_operation"):
                raise ValueError("Test error")

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["step"] == "failing_operation"
        assert ctx.steps[0]["status"] == StepStatus.FAILURE.value

        assert len(ctx.errors) == 1
        assert ctx.errors[0]["step"] == "failing_operation"
        assert "Test error" in ctx.errors[0]["message"]

    def test_context_manager_custom_error_status(self, ctx: TelemetryContext):
        """Verify context manager can use custom error status."""
        with pytest.raises(RuntimeError):
            with ctx.step("warning_op", error_status=StepStatus.WARNING):
                raise RuntimeError("Warning condition")

        assert ctx.steps[0]["status"] == StepStatus.WARNING.value

    def test_context_manager_with_metadata(self, ctx: TelemetryContext):
        """Verify context manager passes metadata correctly."""
        metadata = {"user_id": 123}
        with ctx.step("test_op", metadata=metadata):
            pass

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["metadata"]["user_id"] == 123

    def test_context_manager_yields_context(self, ctx: TelemetryContext):
        """Verify context manager yields the context itself."""
        with ctx.step("test") as yielded_ctx:
            assert yielded_ctx is ctx

    def test_context_manager_capture_exception_details_true(self, ctx: TelemetryContext):
        """Verify exception details are captured when enabled."""
        with pytest.raises(ValueError):
            with ctx.step("test_op", capture_exception_details=True):
                raise ValueError("Detailed error")

        # Check error metadata in step
        assert "metadata" in ctx.steps[0]
        assert "error_type" in ctx.steps[0]["metadata"]
        assert ctx.steps[0]["metadata"]["error_type"] == "ValueError"

    def test_context_manager_capture_exception_details_false(self, ctx: TelemetryContext):
        """Verify exception details are not captured when disabled."""
        with pytest.raises(ValueError):
            with ctx.step("test_op", capture_exception_details=False):
                raise ValueError("Detailed error")

        # Should still record the step but without exception metadata
        assert len(ctx.steps) == 1
        # Error IS recorded (auto_record_error=True default), but traceback should be missing
        assert len(ctx.errors) == 1
        assert "traceback" not in ctx.errors[0]

    def test_context_manager_invalid_step_name_continues(self, ctx: TelemetryContext):
        """Verify context manager continues even with invalid step name."""
        # Should not raise, but step won't be tracked
        with ctx.step(""):
            pass

        # Step was not recorded due to invalid name
        assert len(ctx.steps) == 0


# ========== Metrics Recording Tests ==========


class TestMetricsRecording:
    """Tests for metric recording functionality."""

    def test_record_metric_basic(self, ctx: TelemetryContext):
        """Verify basic metric recording."""
        result = ctx.record_metric("component", "metric", 42)

        assert result is True
        assert "component.metric" in ctx.metrics
        assert ctx.metrics["component.metric"] == 42

    @pytest.mark.parametrize(
        "value",
        [
            42,
            3.14,
            "string_value",
            True,
            False,
            None,
            [1, 2, 3],
            {"nested": "dict"},
        ],
    )
    def test_record_metric_various_types(self, ctx: TelemetryContext, value: Any):
        """Verify metrics can store various sanitized types."""
        result = ctx.record_metric("comp", "metric", value)
        assert result is True
        assert "comp.metric" in ctx.metrics

    def test_record_metric_overwrite_default(self, ctx: TelemetryContext):
        """Verify metrics overwrite by default."""
        ctx.record_metric("comp", "metric", 100)
        ctx.record_metric("comp", "metric", 200)

        assert ctx.metrics["comp.metric"] == 200

    def test_record_metric_no_overwrite(self, ctx: TelemetryContext):
        """Verify overwrite=False prevents overwriting."""
        ctx.record_metric("comp", "metric", 100)
        result = ctx.record_metric("comp", "metric", 200, overwrite=False)

        assert result is False
        assert ctx.metrics["comp.metric"] == 100

    def test_record_metric_invalid_component(self, ctx: TelemetryContext):
        """Verify invalid component name is rejected."""
        assert ctx.record_metric("", "metric", 100) is False
        assert ctx.record_metric(None, "metric", 100) is False
        assert ctx.record_metric(123, "metric", 100) is False

    def test_record_metric_invalid_metric_name(self, ctx: TelemetryContext):
        """Verify invalid metric name is rejected."""
        assert ctx.record_metric("comp", "", 100) is False
        assert ctx.record_metric("comp", None, 100) is False
        assert ctx.record_metric("comp", 123, 100) is False

    def test_record_metric_too_long_names(self, ctx: TelemetryContext):
        """Verify overly long names are rejected."""
        long_name = "x" * (TelemetryDefaults.MAX_NAME_LENGTH + 1)
        assert ctx.record_metric(long_name, "metric", 100) is False
        assert ctx.record_metric("comp", long_name, 100) is False

    def test_record_metric_max_limit(self, ctx_with_limits: TelemetryContext):
        """Verify max_metrics limit is enforced."""
        for i in range(5):
            result = ctx_with_limits.record_metric("comp", f"metric{i}", i)
            assert result is True

        result = ctx_with_limits.record_metric("comp", "metric6", 6)
        assert result is False
        assert len(ctx_with_limits.metrics) == 5

    def test_record_metric_complex_value_sanitization(self, ctx: TelemetryContext):
        """Verify complex values are sanitized properly."""
        complex_value = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3],
            "datetime": datetime.now(),
        }

        result = ctx.record_metric("comp", "complex", complex_value)
        assert result is True

        stored = ctx.metrics["comp.complex"]
        assert isinstance(stored, dict)
        assert isinstance(stored["datetime"], str)


# ========== Increment Metric Tests ==========


class TestIncrementMetric:
    """Tests for increment_metric functionality."""

    def test_increment_metric_new(self, ctx: TelemetryContext):
        """Verify increment_metric creates new metric."""
        result = ctx.increment_metric("comp", "counter")

        assert result is True
        assert ctx.metrics["comp.counter"] == 1

    def test_increment_metric_existing(self, ctx: TelemetryContext):
        """Verify increment_metric adds to existing metric."""
        ctx.record_metric("comp", "counter", 10)
        ctx.increment_metric("comp", "counter", 5)

        assert ctx.metrics["comp.counter"] == 15

    def test_increment_metric_negative(self, ctx: TelemetryContext):
        """Verify increment_metric works with negative values."""
        ctx.record_metric("comp", "counter", 10)
        ctx.increment_metric("comp", "counter", -3)

        assert ctx.metrics["comp.counter"] == 7

    def test_increment_metric_float(self, ctx: TelemetryContext):
        """Verify increment_metric works with float values."""
        ctx.record_metric("comp", "value", 1.5)
        ctx.increment_metric("comp", "value", 0.5)

        assert ctx.metrics["comp.value"] == 2.0

    def test_increment_metric_invalid_component(self, ctx: TelemetryContext):
        """Verify increment_metric rejects invalid component."""
        assert ctx.increment_metric("", "counter") is False
        assert ctx.increment_metric(None, "counter") is False

    def test_increment_metric_invalid_metric_name(self, ctx: TelemetryContext):
        """Verify increment_metric rejects invalid metric name."""
        assert ctx.increment_metric("comp", "") is False
        assert ctx.increment_metric("comp", None) is False

    def test_increment_metric_non_numeric_increment(self, ctx: TelemetryContext):
        """Verify increment_metric rejects non-numeric increment."""
        assert ctx.increment_metric("comp", "counter", "not_a_number") is False
        assert ctx.increment_metric("comp", "counter", [1, 2, 3]) is False

    def test_increment_metric_resets_non_numeric_existing(self, ctx: TelemetryContext):
        """Verify increment_metric resets non-numeric existing value."""
        ctx.record_metric("comp", "value", "string_value")
        result = ctx.increment_metric("comp", "value", 5)

        assert result is True
        assert ctx.metrics["comp.value"] == 5  # Reset to 0 then added 5

    def test_increment_metric_respects_max_limit(self, ctx_with_limits: TelemetryContext):
        """Verify increment_metric respects max_metrics limit."""
        # Fill up metrics
        for i in range(5):
            ctx_with_limits.record_metric("comp", f"m{i}", i)

        # Try to increment a new metric
        result = ctx_with_limits.increment_metric("comp", "new_counter")
        assert result is False

    def test_increment_metric_existing_at_limit(self, ctx_with_limits: TelemetryContext):
        """Verify increment_metric works on existing metric at limit."""
        for i in range(5):
            ctx_with_limits.record_metric("comp", f"m{i}", i)

        # Increment existing metric should work
        result = ctx_with_limits.increment_metric("comp", "m0", 10)
        assert result is True
        assert ctx_with_limits.metrics["comp.m0"] == 10


# ========== Get Metric Tests ==========


class TestGetMetric:
    """Tests for get_metric functionality."""

    def test_get_metric_existing(self, ctx: TelemetryContext):
        """Verify get_metric returns existing metric."""
        ctx.record_metric("comp", "value", 42)
        result = ctx.get_metric("comp", "value")

        assert result == 42

    def test_get_metric_non_existing_default(self, ctx: TelemetryContext):
        """Verify get_metric returns default for non-existing metric."""
        result = ctx.get_metric("comp", "nonexistent")
        assert result is None

    def test_get_metric_custom_default(self, ctx: TelemetryContext):
        """Verify get_metric returns custom default."""
        result = ctx.get_metric("comp", "nonexistent", default=100)
        assert result == 100

    def test_get_metric_complex_value(self, ctx: TelemetryContext):
        """Verify get_metric returns complex values."""
        ctx.record_metric("comp", "data", {"key": "value"})
        result = ctx.get_metric("comp", "data")

        assert result == {"key": "value"}


# ========== Error Recording Tests ==========


class TestErrorRecording:
    """Tests for error recording functionality."""

    def test_record_error_basic(self, ctx: TelemetryContext):
        """Verify basic error recording."""
        result = ctx.record_error("test_step", "Something went wrong")

        assert result is True
        assert len(ctx.errors) == 1
        assert ctx.errors[0]["step"] == "test_step"
        assert ctx.errors[0]["message"] == "Something went wrong"
        assert "timestamp" in ctx.errors[0]

    def test_record_error_with_type(self, ctx: TelemetryContext):
        """Verify error recording with error type."""
        result = ctx.record_error(
            "test_step", "Error occurred", error_type="ValidationError"
        )

        assert result is True
        assert ctx.errors[0]["type"] == "ValidationError"

    def test_record_error_with_exception(self, ctx: TelemetryContext):
        """Verify error recording with exception object."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            result = ctx.record_error("test_step", "Caught exception", exception=e)

        assert result is True
        assert ctx.errors[0]["type"] == "ValueError"
        assert "exception_details" in ctx.errors[0]
        assert "Test exception" in ctx.errors[0]["exception_details"]

    def test_record_error_with_traceback(self, ctx: TelemetryContext):
        """Verify error recording includes traceback when requested."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            result = ctx.record_error(
                "test_step", "Caught exception", exception=e, include_traceback=True
            )

        assert result is True
        assert "traceback" in ctx.errors[0]
        assert "ValueError" in ctx.errors[0]["traceback"]

    def test_record_error_without_traceback(self, ctx: TelemetryContext):
        """Verify error recording excludes traceback by default."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            ctx.record_error("test_step", "Caught exception", exception=e)

        assert "traceback" not in ctx.errors[0]

    def test_record_error_with_metadata(self, ctx: TelemetryContext):
        """Verify error recording with metadata."""
        metadata = {"user_id": 123, "context": "validation"}
        result = ctx.record_error("test_step", "Error", metadata=metadata)

        assert result is True
        assert "metadata" in ctx.errors[0]

    def test_record_error_invalid_step_name(self, ctx: TelemetryContext):
        """Verify invalid step name falls back to default."""
        # Now returns True because it falls back to __unknown_step__
        assert ctx.record_error("", "message") is True
        assert ctx.record_error(None, "message") is True
        assert ctx.record_error(123, "message") is True

        # Verify fallback occurred
        assert ctx.errors[-1]["step"] == "__unknown_step__"

    def test_record_error_invalid_message(self, ctx: TelemetryContext):
        """Verify invalid error message falls back to generic message."""
        # Now returns True because it falls back to generic message
        assert ctx.record_error("step", "") is True
        assert ctx.record_error("step", None) is True
        assert ctx.record_error("step", 123) is True

        # Verify fallback occurred
        # The default generic message is "Unknown error" for non-string inputs
        assert "Unknown error" in ctx.errors[-1]["message"]

    def test_record_error_long_message_truncation(self, ctx: TelemetryContext):
        """Verify very long error messages are truncated."""
        long_message = "x" * 2000
        ctx.record_error("step", long_message)

        stored_message = ctx.errors[0]["message"]
        assert len(stored_message) <= TelemetryDefaults.MAX_MESSAGE_LENGTH

    def test_record_error_max_limit(self, ctx_with_limits: TelemetryContext):
        """Verify max_errors limit is enforced with FIFO."""
        ctx_with_limits.record_error("step1", "error1")
        ctx_with_limits.record_error("step2", "error2")
        ctx_with_limits.record_error("step3", "error3")

        assert len(ctx_with_limits.errors) == 2
        assert ctx_with_limits.errors[0]["step"] == "step2"
        assert ctx_with_limits.errors[1]["step"] == "step3"


# ========== Active Timers Tests ==========


class TestActiveTimers:
    """Tests for active timer management."""

    def test_get_active_timers_empty(self, ctx: TelemetryContext):
        """Verify get_active_timers returns empty list initially."""
        assert ctx.get_active_timers() == []

    def test_get_active_timers_with_active(self, ctx: TelemetryContext):
        """Verify get_active_timers returns active step names."""
        ctx.start_step("step1")
        ctx.start_step("step2")

        active = ctx.get_active_timers()
        assert len(active) == 2
        assert "step1" in active
        assert "step2" in active

    def test_get_active_timers_after_end(self, ctx: TelemetryContext):
        """Verify get_active_timers updates after ending steps."""
        ctx.start_step("step1")
        ctx.start_step("step2")
        ctx.end_step("step1")

        active = ctx.get_active_timers()
        assert len(active) == 1
        assert "step2" in active

    def test_get_active_step_info_existing(self, ctx: TelemetryContext):
        """Verify get_active_step_info returns info for active step."""
        metadata = {"key": "value"}
        ctx.start_step("step1", metadata=metadata)
        time.sleep(0.01)

        info = ctx.get_active_step_info("step1")

        assert info is not None
        assert info["step_name"] == "step1"
        assert info["duration_so_far"] >= 0.01
        assert info["metadata"]["key"] == "value"

    def test_get_active_step_info_non_existing(self, ctx: TelemetryContext):
        """Verify get_active_step_info returns None for non-active step."""
        info = ctx.get_active_step_info("nonexistent")
        assert info is None

    def test_get_active_step_info_returns_copy(self, ctx: TelemetryContext):
        """Verify get_active_step_info returns a copy of metadata."""
        metadata = {"key": "value"}
        ctx.start_step("step1", metadata=metadata)

        info = ctx.get_active_step_info("step1")
        info["metadata"]["key"] = "modified"

        # Original should be unchanged
        original_info = ctx.get_active_step_info("step1")
        assert original_info["metadata"]["key"] == "value"

    def test_cancel_step(self, ctx: TelemetryContext):
        """Verify cancel_step removes timer without recording."""
        ctx.start_step("step1")
        assert "step1" in ctx._active_steps

        result = ctx.cancel_step("step1")
        assert result is True
        assert "step1" not in ctx._active_steps
        assert len(ctx.steps) == 0

    def test_cancel_step_non_existent(self, ctx: TelemetryContext):
        """Verify canceling non-existent step returns False."""
        result = ctx.cancel_step("never_started")
        assert result is False

    def test_clear_active_timers(self, ctx: TelemetryContext):
        """Verify clear_active_timers removes all timers."""
        ctx.start_step("step1")
        ctx.start_step("step2")
        ctx.start_step("step3")

        count = ctx.clear_active_timers()
        assert count == 3
        assert len(ctx._active_steps) == 0
        assert ctx.get_active_timers() == []

    def test_clear_active_timers_when_empty(self, ctx: TelemetryContext):
        """Verify clearing empty timers returns 0."""
        count = ctx.clear_active_timers()
        assert count == 0


# ========== Step Query Tests ==========


class TestStepQueries:
    """Tests for step query methods."""

    def test_has_step_existing(self, ctx: TelemetryContext):
        """Verify has_step returns True for completed step."""
        ctx.start_step("step1")
        ctx.end_step("step1")

        assert ctx.has_step("step1") is True

    def test_has_step_non_existing(self, ctx: TelemetryContext):
        """Verify has_step returns False for non-existing step."""
        assert ctx.has_step("nonexistent") is False

    def test_has_step_active_not_completed(self, ctx: TelemetryContext):
        """Verify has_step returns False for active but not completed step."""
        ctx.start_step("step1")

        assert ctx.has_step("step1") is False

    def test_get_step_by_name_existing(self, ctx: TelemetryContext):
        """Verify get_step_by_name returns step data."""
        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.SUCCESS)

        step = ctx.get_step_by_name("step1")

        assert step is not None
        assert step["step"] == "step1"
        assert step["status"] == "success"

    def test_get_step_by_name_non_existing(self, ctx: TelemetryContext):
        """Verify get_step_by_name returns None for non-existing step."""
        step = ctx.get_step_by_name("nonexistent")
        assert step is None

    def test_get_step_by_name_multiple_returns_last(self, ctx: TelemetryContext):
        """Verify get_step_by_name returns last occurrence by default."""
        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.SUCCESS, {"attempt": 1})

        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.FAILURE, {"attempt": 2})

        step = ctx.get_step_by_name("step1", last=True)

        assert step["metadata"]["attempt"] == 2

    def test_get_step_by_name_multiple_returns_first(self, ctx: TelemetryContext):
        """Verify get_step_by_name can return first occurrence."""
        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.SUCCESS, {"attempt": 1})

        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.FAILURE, {"attempt": 2})

        step = ctx.get_step_by_name("step1", last=False)

        assert step["metadata"]["attempt"] == 1

    def test_get_step_by_name_returns_copy(self, ctx: TelemetryContext):
        """Verify get_step_by_name returns a copy."""
        ctx.start_step("step1")
        ctx.end_step("step1", StepStatus.SUCCESS)

        step = ctx.get_step_by_name("step1")
        step["modified"] = True

        # Original should be unchanged
        assert "modified" not in ctx.steps[0]


# ========== Limits and Boundaries Tests ==========


class TestLimitsAndBoundaries:
    """Tests for max limits enforcement."""

    def test_max_steps_limit_fifo(self, ctx_with_limits: TelemetryContext):
        """Verify max_steps enforces FIFO removal."""
        for i in range(5):
            ctx_with_limits.start_step(f"step{i}")
            ctx_with_limits.end_step(f"step{i}")

        assert len(ctx_with_limits.steps) == 3
        assert ctx_with_limits.steps[0]["step"] == "step2"
        assert ctx_with_limits.steps[1]["step"] == "step3"
        assert ctx_with_limits.steps[2]["step"] == "step4"

    def test_max_errors_limit_fifo(self, ctx_with_limits: TelemetryContext):
        """Verify max_errors enforces FIFO removal."""
        for i in range(4):
            ctx_with_limits.record_error(f"step{i}", f"error{i}")

        assert len(ctx_with_limits.errors) == 2
        assert ctx_with_limits.errors[0]["step"] == "step2"
        assert ctx_with_limits.errors[1]["step"] == "step3"

    def test_max_metrics_limit_rejection(self, ctx_with_limits: TelemetryContext):
        """Verify max_metrics rejects new metrics when limit reached."""
        for i in range(5):
            assert ctx_with_limits.record_metric("comp", f"m{i}", i) is True

        assert ctx_with_limits.record_metric("comp", "m5", 5) is False
        assert len(ctx_with_limits.metrics) == 5


# ========== Export and Serialization Tests ==========


class TestExportAndSerialization:
    """Tests for to_dict() and get_summary() methods."""

    def test_to_dict_basic(self, ctx: TelemetryContext):
        """Verify to_dict exports basic structure."""
        data = ctx.to_dict()

        assert "request_id" in data
        assert "steps" in data
        assert "metrics" in data
        assert "errors" in data
        assert "total_duration_seconds" in data
        assert "created_at" in data
        assert "age_seconds" in data
        assert isinstance(data["steps"], list)
        assert isinstance(data["metrics"], dict)
        assert isinstance(data["errors"], list)

    def test_to_dict_with_data(self, ctx_populated: TelemetryContext):
        """Verify to_dict exports populated data correctly."""
        data = ctx_populated.to_dict()

        assert len(data["steps"]) == 1
        assert len(data["metrics"]) == 2
        assert len(data["errors"]) == 1
        assert data["total_duration_seconds"] > 0

    def test_to_dict_includes_metadata(self, ctx: TelemetryContext):
        """Verify to_dict includes metadata when requested."""
        ctx.metadata["custom"] = "value"
        data = ctx.to_dict(include_metadata=True)

        assert "metadata" in data
        assert data["metadata"]["custom"] == "value"

    def test_to_dict_excludes_metadata(self, ctx: TelemetryContext):
        """Verify to_dict excludes metadata when requested."""
        ctx.metadata["custom"] = "value"
        data = ctx.to_dict(include_metadata=False)

        assert "metadata" not in data

    def test_to_dict_includes_active_timers_info(self, ctx: TelemetryContext):
        """Verify to_dict includes active timers info."""
        ctx.start_step("incomplete", metadata={"key": "value"})
        time.sleep(0.01)

        data = ctx.to_dict()

        assert "active_timers" in data
        assert "incomplete" in data["active_timers"]
        assert "active_timers_info" in data
        assert "incomplete" in data["active_timers_info"]
        assert data["active_timers_info"]["incomplete"]["duration_so_far"] >= 0.01
        assert data["active_timers_info"]["incomplete"]["has_metadata"] is True

    def test_to_dict_creates_deep_copies(self, ctx: TelemetryContext):
        """Verify to_dict returns deep copies, not references."""
        ctx.start_step("step1")
        ctx.end_step("step1", metadata={"nested": {"key": "value"}})

        data = ctx.to_dict()

        # Modify the returned data deeply
        data["steps"][0]["metadata"]["nested"]["key"] = "modified"

        # Original should be unchanged
        assert ctx.steps[0]["metadata"]["nested"]["key"] == "value"

    def test_get_summary_basic(self, ctx: TelemetryContext):
        """Verify get_summary returns correct structure."""
        summary = ctx.get_summary()

        assert "request_id" in summary
        assert "total_steps" in summary
        assert "total_errors" in summary
        assert "total_metrics" in summary
        assert "active_timers" in summary
        assert "total_duration_seconds" in summary
        assert "step_statuses" in summary
        assert "error_types" in summary
        assert "has_errors" in summary
        assert "has_failures" in summary
        assert "age_seconds" in summary

    def test_get_summary_with_data(self, ctx_populated: TelemetryContext):
        """Verify get_summary calculates statistics correctly."""
        summary = ctx_populated.get_summary()

        assert summary["total_steps"] == 1
        assert summary["total_errors"] == 1
        assert summary["total_metrics"] == 2
        assert summary["has_errors"] is True
        assert summary["total_duration_seconds"] > 0

    def test_get_summary_step_statuses(self, ctx: TelemetryContext):
        """Verify get_summary counts step statuses correctly."""
        ctx.start_step("s1")
        ctx.end_step("s1", StepStatus.SUCCESS)
        ctx.start_step("s2")
        ctx.end_step("s2", StepStatus.FAILURE)
        ctx.start_step("s3")
        ctx.end_step("s3", StepStatus.SUCCESS)

        summary = ctx.get_summary()
        assert summary["step_statuses"]["success"] == 2
        assert summary["step_statuses"]["failure"] == 1
        assert summary["has_failures"] is True

    def test_get_summary_error_types(self, ctx: TelemetryContext):
        """Verify get_summary counts error types correctly."""
        ctx.record_error("step1", "error1", error_type="ValidationError")
        ctx.record_error("step2", "error2", error_type="ValidationError")
        ctx.record_error("step3", "error3", error_type="DatabaseError")

        summary = ctx.get_summary()
        assert summary["error_types"]["ValidationError"] == 2
        assert summary["error_types"]["DatabaseError"] == 1


# ========== Reset Functionality Tests ==========


class TestResetFunctionality:
    """Tests for reset() method."""

    def test_reset_clears_all_data(self, ctx_populated: TelemetryContext):
        """Verify reset clears all tracking data."""
        original_id = ctx_populated.request_id
        ctx_populated.reset(keep_request_id=True)

        assert ctx_populated.request_id == original_id
        assert len(ctx_populated.steps) == 0
        assert len(ctx_populated.metrics) == 0
        assert len(ctx_populated.errors) == 0
        assert len(ctx_populated._active_steps) == 0
        assert len(ctx_populated.metadata) == 0

    def test_reset_generates_new_id(self, ctx_populated: TelemetryContext):
        """Verify reset can generate new request_id."""
        original_id = ctx_populated.request_id
        ctx_populated.reset(keep_request_id=False)

        assert ctx_populated.request_id != original_id
        assert len(ctx_populated.steps) == 0

    def test_reset_updates_created_at(self, ctx: TelemetryContext):
        """Verify reset updates created_at timestamp."""
        original_created = ctx.created_at
        time.sleep(0.01)
        ctx.reset()

        assert ctx.created_at > original_created


# ========== Thread Safety Tests ==========


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_metric_recording(self, ctx: TelemetryContext):
        """Verify concurrent metric recording is thread-safe."""

        def record_metrics(thread_id: int, count: int) -> None:
            for i in range(count):
                ctx.record_metric(f"thread{thread_id}", f"metric{i}", i)

        threads = []
        for t in range(5):
            thread = threading.Thread(target=record_metrics, args=(t, 10))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(ctx.metrics) == 50

    def test_concurrent_step_tracking(self, ctx: TelemetryContext):
        """Verify concurrent step tracking is thread-safe."""

        def track_step(step_name: str) -> None:
            ctx.start_step(step_name)
            time.sleep(0.001)
            ctx.end_step(step_name)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=track_step, args=(f"step{i}",))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(ctx.steps) == 10

    def test_concurrent_error_recording(self, ctx: TelemetryContext):
        """Verify concurrent error recording is thread-safe."""

        def record_errors(count: int) -> None:
            for i in range(count):
                ctx.record_error(f"step{i}", f"error{i}")

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=record_errors, args=(5,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(ctx.errors) == 15

    def test_concurrent_increment_metric(self, ctx: TelemetryContext):
        """Verify concurrent increment_metric is thread-safe."""

        def increment_many(count: int) -> None:
            for _ in range(count):
                ctx.increment_metric("shared", "counter", 1)

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=increment_many, args=(100,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert ctx.metrics["shared.counter"] == 1000


# ========== Context Manager Protocol Tests ==========


class TestContextManagerProtocol:
    """Tests for context manager protocol (__enter__/__exit__)."""

    def test_context_manager_enter(self, ctx: TelemetryContext):
        """Verify __enter__ returns self."""
        with ctx as entered_ctx:
            assert entered_ctx is ctx

    def test_context_manager_exit_ends_active_steps_as_cancelled(
        self, ctx: TelemetryContext
    ):
        """Verify __exit__ ends active steps as CANCELLED."""
        with ctx:
            ctx.start_step("step1")
            ctx.start_step("step2")
            # Don't end them

        # Steps should be ended as CANCELLED
        assert len(ctx.get_active_timers()) == 0
        assert len(ctx.steps) == 2

        for step in ctx.steps:
            assert step["status"] == StepStatus.CANCELLED.value
            assert step["metadata"]["reason"] == "context_exit"

    def test_context_manager_records_error_on_exception(self, ctx: TelemetryContext):
        """Verify __exit__ records error when exception occurs."""
        with pytest.raises(ValueError):
            with ctx:
                raise ValueError("Test exception")

        assert len(ctx.errors) == 1
        assert ctx.errors[0]["step"] == "__context__"
        assert "Test exception" in ctx.errors[0]["message"]
        assert ctx.errors[0]["type"] == "ValueError"

    def test_context_manager_doesnt_suppress_exceptions(self, ctx: TelemetryContext):
        """Verify __exit__ doesn't suppress exceptions."""
        with pytest.raises(ValueError):
            with ctx:
                raise ValueError("Test exception")


# ========== Value Sanitization Tests ==========


class TestValueSanitization:
    """Tests for _sanitize_value() method."""

    def test_sanitize_basic_types(self, ctx: TelemetryContext):
        """Verify basic types pass through sanitization."""
        assert ctx._sanitize_value(None) is None
        assert ctx._sanitize_value(True) is True
        assert ctx._sanitize_value(False) is False
        assert ctx._sanitize_value(42) == 42
        assert ctx._sanitize_value(3.14) == 3.14
        assert ctx._sanitize_value("test") == "test"

    def test_sanitize_long_string(self, ctx: TelemetryContext):
        """Verify long strings are truncated."""
        long_string = "x" * 15000
        result = ctx._sanitize_value(long_string)
        assert len(result) <= TelemetryDefaults.MAX_STRING_LENGTH + 14

    def test_sanitize_list(self, ctx: TelemetryContext):
        """Verify lists are sanitized recursively."""
        test_list = [1, "two", {"three": 3}]
        result = ctx._sanitize_value(test_list)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_sanitize_large_list(self, ctx: TelemetryContext):
        """Verify large lists are truncated."""
        large_list = list(range(200))
        result = ctx._sanitize_value(large_list)
        # Should be 100 items + truncation message
        assert len(result) == TelemetryDefaults.MAX_COLLECTION_SIZE + 1
        assert "more items" in result[-1]

    def test_sanitize_dict(self, ctx: TelemetryContext):
        """Verify dicts are sanitized recursively."""
        test_dict = {"a": 1, "b": "two", "c": [3, 4]}
        result = ctx._sanitize_value(test_dict)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_sanitize_large_dict(self, ctx: TelemetryContext):
        """Verify large dicts are truncated."""
        # Create dict larger than MAX_DICT_KEYS (200)
        large_dict = {f"key{i}": i for i in range(250)}
        result = ctx._sanitize_value(large_dict)
        # Should have 200 items + truncation marker
        assert len(result) == TelemetryDefaults.MAX_DICT_KEYS + 1
        assert "<truncated>" in result

    def test_sanitize_datetime(self, ctx: TelemetryContext):
        """Verify datetime objects are converted to ISO format."""
        dt = datetime(2024, 1, 1, 12, 30, 45)
        result = ctx._sanitize_value(dt)
        assert isinstance(result, str)
        assert "2024-01-01" in result

    def test_sanitize_max_depth(self, ctx: TelemetryContext):
        """Verify deep nesting is limited by max_depth."""
        deep = {"level": 1}
        current = deep
        for i in range(10):
            current["nested"] = {"level": i + 2}
            current = current["nested"]

        result = ctx._sanitize_value(deep, max_depth=3)
        assert isinstance(result, dict)

    def test_sanitize_unserializable_object(self, ctx: TelemetryContext):
        """Verify objects with __dict__ are converted to dict representation."""

        class CustomObject:
            pass

        obj = CustomObject()
        result = ctx._sanitize_value(obj)
        # Robust implementation converts objects with __dict__ to dict
        assert isinstance(result, dict)
        assert result["__class__"] == "CustomObject"

    def test_sanitize_enum(self, ctx: TelemetryContext):
        """Verify Enum values are converted to their value."""

        class TestEnum(Enum):
            VALUE = "test_value"

        result = ctx._sanitize_value(TestEnum.VALUE)
        assert result == "test_value"

    def test_sanitize_bytes(self, ctx: TelemetryContext):
        """Verify bytes are handled."""
        result = ctx._sanitize_value(b"hello")
        assert isinstance(result, str)
        assert "hello" in result or "bytes" in result

    def test_sanitize_set(self, ctx: TelemetryContext):
        """Verify sets are converted to lists."""
        result = ctx._sanitize_value({1, 2, 3})
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]

    def test_sanitize_frozenset(self, ctx: TelemetryContext):
        """Verify frozensets are converted to lists."""
        result = ctx._sanitize_value(frozenset([1, 2, 3]))
        assert isinstance(result, list)
        assert sorted(result) == [1, 2, 3]

    def test_sanitize_nan(self, ctx: TelemetryContext):
        """Verify NaN is handled."""
        result = ctx._sanitize_value(float("nan"))
        assert result == "<NaN>"

    def test_sanitize_infinity(self, ctx: TelemetryContext):
        """Verify infinity values are handled."""
        assert ctx._sanitize_value(float("inf")) == "<Infinity>"
        assert ctx._sanitize_value(float("-inf")) == "<-Infinity>"

    def test_sanitize_object_with_dict(self, ctx: TelemetryContext):
        """Verify objects with __dict__ are serialized."""

        class CustomClass:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = 42

        obj = CustomClass()
        result = ctx._sanitize_value(obj)

        assert isinstance(result, dict)
        assert result["__class__"] == "CustomClass"
        assert result["attr1"] == "value1"
        assert result["attr2"] == 42


# ========== Edge Cases and Regression Tests ==========


class TestEdgeCases:
    """Tests for edge cases and potential issues."""

    def test_empty_context_to_dict(self, ctx: TelemetryContext):
        """Verify to_dict works with empty context."""
        data = ctx.to_dict()
        assert data["total_duration_seconds"] == 0
        assert len(data["steps"]) == 0

    def test_multiple_end_calls(self, ctx: TelemetryContext):
        """Verify multiple end_step calls for same step."""
        ctx.start_step("step1")
        ctx.end_step("step1")
        ctx.end_step("step1")

        assert len(ctx.steps) == 2
        assert ctx.steps[1]["duration_seconds"] == 0

    def test_unicode_in_names(self, ctx: TelemetryContext):
        """Verify unicode characters in names are handled."""
        ctx.start_step("测试步骤")
        ctx.end_step("测试步骤")
        ctx.record_metric("组件", "指标", 100)
        ctx.record_error("步骤", "错误消息")

        assert len(ctx.steps) == 1
        assert len(ctx.metrics) == 1
        assert len(ctx.errors) == 1

    def test_special_characters_in_names(self, ctx: TelemetryContext):
        """Verify special characters in names are handled."""
        special_name = "step.with-special_chars!@#"
        ctx.start_step(special_name)
        ctx.end_step(special_name)

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["step"] == special_name

    def test_zero_duration_step(self, ctx: TelemetryContext):
        """Verify steps with negligible duration are recorded."""
        ctx.start_step("instant_step")
        ctx.end_step("instant_step")

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["duration_seconds"] >= 0

    def test_age_calculation(self, ctx: TelemetryContext):
        """Verify age_seconds increases over time."""
        summary1 = ctx.get_summary()
        time.sleep(0.01)
        summary2 = ctx.get_summary()

        assert summary2["age_seconds"] > summary1["age_seconds"]

    def test_repr_method(self, ctx_populated: TelemetryContext):
        """Verify __repr__ returns valid representation."""
        repr_str = repr(ctx_populated)

        assert "TelemetryContext" in repr_str
        assert ctx_populated.request_id in repr_str
        assert "steps=" in repr_str
        assert "errors=" in repr_str

    def test_str_method(self, ctx_populated: TelemetryContext):
        """Verify __str__ returns readable string."""
        str_repr = str(ctx_populated)

        assert "Telemetry" in str_repr
        assert "steps" in str_repr
        assert "errors" in str_repr


# ========== Integration Tests ==========


class TestIntegration:
    """Integration tests simulating real-world usage."""

    def test_complete_workflow(self, ctx: TelemetryContext):
        """Test a complete workflow with multiple operations."""
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

        assert ctx.steps[0]["step"] == "data_loading"
        assert ctx.steps[1]["step"] == "processing"
        assert ctx.steps[2]["step"] == "finalization"
        assert ctx.steps[3]["step"] == "main_process"

    def test_request_lifecycle(self, ctx: TelemetryContext):
        """Test complete request lifecycle."""
        ctx.metadata["user_id"] = "user123"
        ctx.metadata["request_type"] = "api_call"

        ctx.start_step("authentication")
        time.sleep(0.001)
        ctx.end_step("authentication", StepStatus.SUCCESS)

        ctx.start_step("authorization")
        time.sleep(0.001)
        ctx.end_step("authorization", StepStatus.SUCCESS)

        ctx.start_step("business_logic")
        ctx.record_metric("business", "items_found", 42)
        time.sleep(0.001)
        ctx.end_step("business_logic", StepStatus.SUCCESS)

        result = ctx.to_dict(include_metadata=True)

        assert len(result["steps"]) == 3
        assert result["metadata"]["user_id"] == "user123"
        assert result["total_duration_seconds"] > 0

    def test_error_recovery_workflow(self, ctx: TelemetryContext):
        """Test workflow with error and recovery."""
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

    def test_increment_metric_workflow(self, ctx: TelemetryContext):
        """Test workflow using increment_metric."""
        for i in range(10):
            with ctx.step(f"process_item_{i}"):
                ctx.increment_metric("processing", "items_processed")
                if i % 3 == 0:
                    ctx.increment_metric("processing", "special_items")

        assert ctx.get_metric("processing", "items_processed") == 10
        assert ctx.get_metric("processing", "special_items") == 4


# ========== Performance Tests ==========


class TestPerformance:
    """Performance and stress tests."""

    def test_many_steps_performance(self, ctx: TelemetryContext):
        """Verify performance with many steps."""
        start = time.perf_counter()

        for i in range(100):
            ctx.start_step(f"step{i}")
            ctx.end_step(f"step{i}")

        duration = time.perf_counter() - start

        assert duration < 2.0
        assert len(ctx.steps) == 100

    def test_large_export(self, ctx_populated: TelemetryContext):
        """Verify to_dict() performs well with populated context."""
        for i in range(50):
            ctx_populated.record_metric("comp", f"metric{i}", i)

        start = time.perf_counter()
        data = ctx_populated.to_dict()
        duration = time.perf_counter() - start

        assert duration < 0.1
        assert isinstance(data, dict)

    def test_deep_sanitization_performance(self, ctx: TelemetryContext):
        """Verify deep sanitization performs well."""
        deep_nested = {"level": 0}
        current = deep_nested
        for i in range(100):
            current["nested"] = {"level": i + 1, "data": list(range(10))}
            current = current["nested"]

        start = time.perf_counter()
        result = ctx._sanitize_value(deep_nested)
        duration = time.perf_counter() - start

        assert duration < 0.1
        assert isinstance(result, dict)


# ========== Business Logic Tests ==========


class TestBusinessLogic:
    """Tests for business logic translation."""

    def test_get_business_report_optimal(self, ctx: TelemetryContext):
        """Verify report for optimal conditions."""
        ctx.record_metric("flux_condenser", "avg_saturation", 0.5)
        ctx.record_metric("flux_condenser", "max_flyback_voltage", 0.02)
        ctx.record_metric("flux_condenser", "max_dissipated_power", 10.0)
        ctx.record_metric("flux_condenser", "avg_kinetic_energy", 100.0)

        report = ctx.get_business_report()

        assert report["status"] == "OPTIMO"
        # Adjusted expectation to match new implementation (no period at end)
        assert "Procesamiento estable y fluido" in report["message"]
        assert report["metrics"]["Carga del Sistema"] == "50.0%"
        assert report["metrics"]["Índice de Inestabilidad"] == "0.0200"
        assert report["metrics"]["Fricción de Datos"] == "10.00"
        assert "details" in report
        assert report["details"]["total_steps"] == 0
        assert report["details"]["total_errors"] == 0

    def test_get_business_report_warning_saturation(self, ctx: TelemetryContext):
        """Verify report for high saturation warning."""
        ctx.record_metric("flux_condenser", "avg_saturation", 0.95)
        ctx.record_metric("flux_condenser", "max_flyback_voltage", 0.1)
        ctx.record_metric("flux_condenser", "max_dissipated_power", 20.0)

        report = ctx.get_business_report()

        assert report["status"] == "ADVERTENCIA"
        # Matches 'Sistema operando a 95% de capacidad'
        assert "capacidad" in report["message"]

    def test_get_business_report_critical_voltage(self, ctx: TelemetryContext):
        """Verify report for critical flyback voltage."""
        ctx.record_metric("flux_condenser", "avg_saturation", 0.5)
        ctx.record_metric("flux_condenser", "max_flyback_voltage", 0.6)
        ctx.record_metric("flux_condenser", "max_dissipated_power", 10.0)

        report = ctx.get_business_report()

        assert report["status"] == "CRITICO"
        # Matches 'Alta inestabilidad detectada'
        assert "inestabilidad" in report["message"]

    def test_get_business_report_critical_power(self, ctx: TelemetryContext):
        """Verify report for critical dissipated power."""
        ctx.record_metric("flux_condenser", "avg_saturation", 0.5)
        ctx.record_metric("flux_condenser", "max_flyback_voltage", 0.1)
        ctx.record_metric("flux_condenser", "max_dissipated_power", 60.0)

        report = ctx.get_business_report()

        assert report["status"] == "CRITICO"
        # Matches 'Fricción de datos excesiva'
        assert "Fricción" in report["message"]

    def test_get_business_report_missing_metrics(self, ctx: TelemetryContext):
        """Verify report handles missing metrics gracefully."""
        report = ctx.get_business_report()

        assert report["status"] == "OPTIMO"
        assert report["metrics"]["Carga del Sistema"] == "0.0%"

    def test_get_business_report_warning_with_errors(self, ctx: TelemetryContext):
        """Verify report shows warning when errors exist."""
        ctx.record_metric("flux_condenser", "avg_saturation", 0.5)
        ctx.record_metric("flux_condenser", "max_flyback_voltage", 0.1)
        ctx.record_metric("flux_condenser", "max_dissipated_power", 10.0)

        ctx.record_error("step1", "Some error occurred")

        report = ctx.get_business_report()

        assert report["status"] == "ADVERTENCIA"
        assert "error" in report["message"].lower()

    def test_get_business_report_custom_thresholds(
        self, ctx_with_custom_thresholds: TelemetryContext
    ):
        """Verify report respects custom thresholds."""
        ctx = ctx_with_custom_thresholds

        # These values would be OK with default thresholds but critical with custom
        ctx.record_metric("flux_condenser", "max_flyback_voltage", 0.4)
        ctx.record_metric("flux_condenser", "max_dissipated_power", 35.0)

        report = ctx.get_business_report()

        assert report["status"] == "CRITICO"

    def test_get_business_report_safe_float_conversion(self, ctx: TelemetryContext):
        """Verify safe float conversion handles various types."""
        ctx.record_metric("flux_condenser", "avg_saturation", "0.5")  # String
        ctx.record_metric("flux_condenser", "max_flyback_voltage", None)  # None
        ctx.record_metric("flux_condenser", "max_dissipated_power", "invalid")

        report = ctx.get_business_report()

        # Should not raise and should use defaults for invalid values
        assert report["status"] == "OPTIMO"
        assert report["metrics"]["Carga del Sistema"] == "50.0%"

    def test_get_business_report_includes_details(self, ctx: TelemetryContext):
        """Verify report includes details section."""
        ctx.start_step("step1")
        ctx.end_step("step1")
        ctx.record_error("step1", "error")
        ctx.start_step("active_step")

        report = ctx.get_business_report()

        assert "details" in report
        assert report["details"]["total_steps"] == 1
        assert report["details"]["total_errors"] == 1
        assert report["details"]["has_active_operations"] is True


# ========== Custom Validation Tests ==========


class TestCustomValidation:
    """Tests for validation helper methods."""

    def test_validate_name_valid(self, ctx: TelemetryContext):
        """Verify _validate_name accepts valid names."""
        assert ctx._validate_name("valid_name", "field") is True
        assert ctx._validate_name("a", "field") is True
        assert ctx._validate_name("name-with-dashes", "field") is True

    def test_validate_name_empty(self, ctx: TelemetryContext):
        """Verify _validate_name rejects empty names."""
        assert ctx._validate_name("", "field") is False

    def test_validate_name_none(self, ctx: TelemetryContext):
        """Verify _validate_name rejects None."""
        assert ctx._validate_name(None, "field") is False

    def test_validate_name_non_string(self, ctx: TelemetryContext):
        """Verify _validate_name rejects non-strings."""
        assert ctx._validate_name(123, "field") is False
        assert ctx._validate_name([], "field") is False

    def test_validate_name_whitespace_only(self, ctx: TelemetryContext):
        """Verify _validate_name rejects whitespace-only."""
        assert ctx._validate_name("   ", "field") is False
        assert ctx._validate_name("\t\n", "field") is False

    def test_validate_name_too_long(self, ctx: TelemetryContext):
        """Verify _validate_name rejects names exceeding max length."""
        long_name = "x" * 101
        assert ctx._validate_name(long_name, "field", max_length=100) is False

    def test_normalize_status_enum(self, ctx: TelemetryContext):
        """Verify _normalize_status handles enum values."""
        assert ctx._normalize_status(StepStatus.SUCCESS) == "success"
        assert ctx._normalize_status(StepStatus.FAILURE) == "failure"

    def test_normalize_status_string(self, ctx: TelemetryContext):
        """Verify _normalize_status handles string values."""
        assert ctx._normalize_status("success") == "success"
        assert ctx._normalize_status("FAILURE") == "success"  # Invalid case

    def test_normalize_status_invalid(self, ctx: TelemetryContext):
        """Verify _normalize_status returns success for invalid values."""
        assert ctx._normalize_status("invalid") == "success"
        assert ctx._normalize_status(123) == "success"
        assert ctx._normalize_status(None) == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
