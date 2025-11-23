import threading
import time
from datetime import datetime

import pytest

from app.telemetry import StepStatus, TelemetryContext

# ========== Fixtures ==========


@pytest.fixture
def ctx():
    """Provides a fresh TelemetryContext instance for each test."""
    return TelemetryContext()


@pytest.fixture
def ctx_with_custom_id():
    """Provides a TelemetryContext with a custom request_id."""
    return TelemetryContext(request_id="custom-test-id-12345")


@pytest.fixture
def ctx_with_limits():
    """Provides a TelemetryContext with custom limits for testing boundaries."""
    return TelemetryContext(max_steps=3, max_errors=2, max_metrics=5)


@pytest.fixture
def ctx_populated(ctx):
    """Provides a TelemetryContext with some pre-populated data."""
    ctx.start_step("step1")
    time.sleep(0.001)
    ctx.end_step("step1", StepStatus.SUCCESS)

    ctx.record_metric("component1", "metric1", 100)
    ctx.record_metric("component1", "metric2", 200)
    ctx.record_error("step1", "test error", error_type="TestError")

    return ctx


# ========== Initialization Tests ==========


class TestInitialization:
    """Tests for TelemetryContext initialization and setup."""

    def test_default_initialization(self, ctx):
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

        assert isinstance(ctx._timers, dict)
        assert len(ctx._timers) == 0

        assert isinstance(ctx.metadata, dict)
        assert len(ctx.metadata) == 0

        assert ctx.created_at > 0
        assert ctx.max_steps == 1000
        assert ctx.max_errors == 100
        assert ctx.max_metrics == 500

    def test_custom_request_id(self, ctx_with_custom_id):
        """Verify custom request_id is preserved."""
        assert ctx_with_custom_id.request_id == "custom-test-id-12345"

    def test_invalid_request_id_generates_new(self):
        """Verify invalid request_id triggers generation of new UUID."""
        # Empty string
        ctx = TelemetryContext(request_id="")
        assert ctx.request_id != ""
        assert len(ctx.request_id) > 0

        # None
        ctx = TelemetryContext(request_id=None)
        assert ctx.request_id is not None
        assert len(ctx.request_id) > 0

    def test_custom_limits(self, ctx_with_limits):
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

    def test_unique_request_ids(self):
        """Verify each instance gets unique request_id by default."""
        ctx1 = TelemetryContext()
        ctx2 = TelemetryContext()
        assert ctx1.request_id != ctx2.request_id


# ========== Step Tracking Tests ==========


class TestStepTracking:
    """Tests for step start, end, and tracking logic."""

    def test_start_step_basic(self, ctx):
        """Verify basic step start functionality."""
        step_name = "test_step"
        result = ctx.start_step(step_name)

        assert result is True
        assert step_name in ctx._timers
        assert ctx._timers[step_name] > 0

    def test_start_step_with_metadata(self, ctx):
        """Verify step start with metadata."""
        metadata = {"key": "value", "count": 42}
        result = ctx.start_step("test_step", metadata=metadata)

        assert result is True
        assert "test_step" in ctx._timers

    def test_start_step_invalid_name(self, ctx):
        """Verify start_step rejects invalid names."""
        # Empty string
        assert ctx.start_step("") is False

        # None
        assert ctx.start_step(None) is False

        # Non-string
        assert ctx.start_step(123) is False
        assert ctx.start_step([]) is False

    def test_start_step_too_long_name(self, ctx):
        """Verify start_step rejects names exceeding max length."""
        long_name = "x" * 300  # Exceeds 255 char limit
        assert ctx.start_step(long_name) is False

    def test_start_step_duplicate_warning(self, ctx):
        """Verify duplicate start_step generates warning but resets timer."""
        step_name = "duplicate_step"

        ctx.start_step(step_name)
        first_time = ctx._timers[step_name]

        time.sleep(0.001)

        # Start again - should warn but reset
        ctx.start_step(step_name)
        second_time = ctx._timers[step_name]

        assert second_time != first_time
        assert second_time > first_time

    def test_end_step_basic(self, ctx):
        """Verify basic step end functionality."""
        step_name = "test_step"

        ctx.start_step(step_name)
        time.sleep(0.01)  # Ensure measurable duration
        result = ctx.end_step(step_name, StepStatus.SUCCESS)

        assert result is True
        assert step_name not in ctx._timers
        assert len(ctx.steps) == 1

        step_data = ctx.steps[0]
        assert step_data["step"] == step_name
        assert step_data["status"] == StepStatus.SUCCESS.value
        assert step_data["duration_seconds"] > 0
        assert step_data["duration_seconds"] < 1.0  # Reasonable upper bound
        assert "timestamp" in step_data
        assert "perf_counter" in step_data

    def test_end_step_with_metadata(self, ctx):
        """Verify end_step with metadata."""
        ctx.start_step("test_step")
        metadata = {"result": "ok", "count": 5}
        ctx.end_step("test_step", StepStatus.SUCCESS, metadata=metadata)

        assert len(ctx.steps) == 1
        assert "metadata" in ctx.steps[0]
        assert ctx.steps[0]["metadata"]["result"] == "ok"

    @pytest.mark.parametrize(
        "status,expected",
        [
            (StepStatus.SUCCESS, "success"),
            (StepStatus.FAILURE, "failure"),
            (StepStatus.WARNING, "warning"),
            (StepStatus.SKIPPED, "skipped"),
            ("success", "success"),
            ("failure", "failure"),
        ],
    )
    def test_end_step_status_normalization(self, ctx, status, expected):
        """Verify different status formats are normalized correctly."""
        ctx.start_step("test_step")
        ctx.end_step("test_step", status)

        assert ctx.steps[0]["status"] == expected

    def test_end_step_invalid_status_defaults(self, ctx):
        """Verify invalid status defaults to success."""
        ctx.start_step("test_step")
        ctx.end_step("test_step", "invalid_status_xyz")

        assert ctx.steps[0]["status"] == StepStatus.SUCCESS.value

    def test_end_step_without_start(self, ctx):
        """Verify end_step without start_step sets duration to 0 and warns."""
        result = ctx.end_step("never_started")

        assert result is True
        assert len(ctx.steps) == 1
        assert ctx.steps[0]["duration_seconds"] == 0.0
        assert ctx.steps[0]["step"] == "never_started"

    def test_end_step_invalid_name(self, ctx):
        """Verify end_step rejects invalid names."""
        assert ctx.end_step("") is False
        assert ctx.end_step(None) is False
        assert ctx.end_step(123) is False

    def test_multiple_steps_sequence(self, ctx):
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

    def test_nested_steps_tracking(self, ctx):
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

    def test_context_manager_success(self, ctx):
        """Verify context manager tracks successful step."""
        with ctx.step("test_operation"):
            time.sleep(0.001)
            # Simulate work
            pass

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["step"] == "test_operation"
        assert ctx.steps[0]["status"] == StepStatus.SUCCESS.value
        assert ctx.steps[0]["duration_seconds"] > 0

    def test_context_manager_with_exception(self, ctx):
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

    def test_context_manager_custom_error_status(self, ctx):
        """Verify context manager can use custom error status."""
        with pytest.raises(RuntimeError):
            with ctx.step("warning_op", error_status=StepStatus.WARNING):
                raise RuntimeError("Warning condition")

        assert ctx.steps[0]["status"] == StepStatus.WARNING.value

    def test_context_manager_with_metadata(self, ctx):
        """Verify context manager passes metadata correctly."""
        metadata = {"user_id": 123}
        with ctx.step("test_op", metadata=metadata):
            pass

        assert len(ctx.steps) == 1

    def test_context_manager_yields_context(self, ctx):
        """Verify context manager yields the context itself."""
        with ctx.step("test") as yielded_ctx:
            assert yielded_ctx is ctx


# ========== Metrics Recording Tests ==========


class TestMetricsRecording:
    """Tests for metric recording functionality."""

    def test_record_metric_basic(self, ctx):
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
    def test_record_metric_various_types(self, ctx, value):
        """Verify metrics can store various sanitized types."""
        result = ctx.record_metric("comp", "metric", value)
        assert result is True
        assert "comp.metric" in ctx.metrics

    def test_record_metric_overwrite_default(self, ctx):
        """Verify metrics overwrite by default."""
        ctx.record_metric("comp", "metric", 100)
        ctx.record_metric("comp", "metric", 200)

        assert ctx.metrics["comp.metric"] == 200

    def test_record_metric_no_overwrite(self, ctx):
        """Verify overwrite=False prevents overwriting."""
        ctx.record_metric("comp", "metric", 100)
        result = ctx.record_metric("comp", "metric", 200, overwrite=False)

        assert result is False
        assert ctx.metrics["comp.metric"] == 100

    def test_record_metric_invalid_component(self, ctx):
        """Verify invalid component name is rejected."""
        assert ctx.record_metric("", "metric", 100) is False
        assert ctx.record_metric(None, "metric", 100) is False
        assert ctx.record_metric(123, "metric", 100) is False

    def test_record_metric_invalid_metric_name(self, ctx):
        """Verify invalid metric name is rejected."""
        assert ctx.record_metric("comp", "", 100) is False
        assert ctx.record_metric("comp", None, 100) is False
        assert ctx.record_metric("comp", 123, 100) is False

    def test_record_metric_too_long_names(self, ctx):
        """Verify overly long names are rejected."""
        long_name = "x" * 200
        assert ctx.record_metric(long_name, "metric", 100) is False
        assert ctx.record_metric("comp", long_name, 100) is False

    def test_record_metric_max_limit(self, ctx_with_limits):
        """Verify max_metrics limit is enforced."""
        # ctx_with_limits has max_metrics=5
        for i in range(5):
            result = ctx_with_limits.record_metric("comp", f"metric{i}", i)
            assert result is True

        # 6th metric should fail
        result = ctx_with_limits.record_metric("comp", "metric6", 6)
        assert result is False
        assert len(ctx_with_limits.metrics) == 5

    def test_record_metric_complex_value_sanitization(self, ctx):
        """Verify complex values are sanitized properly."""
        complex_value = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3],
            "datetime": datetime.now(),
        }

        result = ctx.record_metric("comp", "complex", complex_value)
        assert result is True

        # Should be sanitized and stored
        stored = ctx.metrics["comp.complex"]
        assert isinstance(stored, dict)


# ========== Error Recording Tests ==========


class TestErrorRecording:
    """Tests for error recording functionality."""

    def test_record_error_basic(self, ctx):
        """Verify basic error recording."""
        result = ctx.record_error("test_step", "Something went wrong")

        assert result is True
        assert len(ctx.errors) == 1
        assert ctx.errors[0]["step"] == "test_step"
        assert ctx.errors[0]["message"] == "Something went wrong"
        assert "timestamp" in ctx.errors[0]

    def test_record_error_with_type(self, ctx):
        """Verify error recording with error type."""
        result = ctx.record_error(
            "test_step", "Error occurred", error_type="ValidationError"
        )

        assert result is True
        assert ctx.errors[0]["type"] == "ValidationError"

    def test_record_error_with_exception(self, ctx):
        """Verify error recording with exception object."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            result = ctx.record_error("test_step", "Caught exception", exception=e)

        assert result is True
        assert ctx.errors[0]["type"] == "ValueError"
        assert "exception_details" in ctx.errors[0]
        assert "Test exception" in ctx.errors[0]["exception_details"]

    def test_record_error_with_metadata(self, ctx):
        """Verify error recording with metadata."""
        metadata = {"user_id": 123, "context": "validation"}
        result = ctx.record_error("test_step", "Error", metadata=metadata)

        assert result is True
        assert "metadata" in ctx.errors[0]

    def test_record_error_invalid_step_name(self, ctx):
        """Verify invalid step name is rejected."""
        assert ctx.record_error("", "message") is False
        assert ctx.record_error(None, "message") is False
        assert ctx.record_error(123, "message") is False

    def test_record_error_invalid_message(self, ctx):
        """Verify invalid error message is rejected."""
        assert ctx.record_error("step", "") is False
        assert ctx.record_error("step", None) is False
        assert ctx.record_error("step", 123) is False

    def test_record_error_long_message_truncation(self, ctx):
        """Verify very long error messages are truncated."""
        long_message = "x" * 2000
        ctx.record_error("step", long_message)

        stored_message = ctx.errors[0]["message"]
        assert len(stored_message) <= 1000

    def test_record_error_max_limit(self, ctx_with_limits):
        """Verify max_errors limit is enforced with FIFO."""
        # ctx_with_limits has max_errors=2
        ctx_with_limits.record_error("step1", "error1")
        ctx_with_limits.record_error("step2", "error2")
        ctx_with_limits.record_error("step3", "error3")  # Should remove first

        assert len(ctx_with_limits.errors) == 2
        assert ctx_with_limits.errors[0]["step"] == "step2"
        assert ctx_with_limits.errors[1]["step"] == "step3"


# ========== Active Timers Tests ==========


class TestActiveTimers:
    """Tests for active timer management."""

    def test_get_active_timers_empty(self, ctx):
        """Verify get_active_timers returns empty list initially."""
        assert ctx.get_active_timers() == []

    def test_get_active_timers_with_active(self, ctx):
        """Verify get_active_timers returns active step names."""
        ctx.start_step("step1")
        ctx.start_step("step2")

        active = ctx.get_active_timers()
        assert len(active) == 2
        assert "step1" in active
        assert "step2" in active

    def test_get_active_timers_after_end(self, ctx):
        """Verify get_active_timers updates after ending steps."""
        ctx.start_step("step1")
        ctx.start_step("step2")
        ctx.end_step("step1")

        active = ctx.get_active_timers()
        assert len(active) == 1
        assert "step2" in active

    def test_cancel_step(self, ctx):
        """Verify cancel_step removes timer without recording."""
        ctx.start_step("step1")
        assert "step1" in ctx._timers

        result = ctx.cancel_step("step1")
        assert result is True
        assert "step1" not in ctx._timers
        assert len(ctx.steps) == 0  # Not recorded

    def test_cancel_step_non_existent(self, ctx):
        """Verify canceling non-existent step returns False."""
        result = ctx.cancel_step("never_started")
        assert result is False

    def test_clear_active_timers(self, ctx):
        """Verify clear_active_timers removes all timers."""
        ctx.start_step("step1")
        ctx.start_step("step2")
        ctx.start_step("step3")

        count = ctx.clear_active_timers()
        assert count == 3
        assert len(ctx._timers) == 0
        assert ctx.get_active_timers() == []

    def test_clear_active_timers_when_empty(self, ctx):
        """Verify clearing empty timers returns 0."""
        count = ctx.clear_active_timers()
        assert count == 0


# ========== Limits and Boundaries Tests ==========


class TestLimitsAndBoundaries:
    """Tests for max limits enforcement."""

    def test_max_steps_limit_fifo(self, ctx_with_limits):
        """Verify max_steps enforces FIFO removal."""
        # ctx_with_limits has max_steps=3
        for i in range(5):
            ctx_with_limits.start_step(f"step{i}")
            ctx_with_limits.end_step(f"step{i}")

        assert len(ctx_with_limits.steps) == 3
        # Should keep last 3: step2, step3, step4
        assert ctx_with_limits.steps[0]["step"] == "step2"
        assert ctx_with_limits.steps[1]["step"] == "step3"
        assert ctx_with_limits.steps[2]["step"] == "step4"

    def test_max_errors_limit_fifo(self, ctx_with_limits):
        """Verify max_errors enforces FIFO removal."""
        # ctx_with_limits has max_errors=2
        for i in range(4):
            ctx_with_limits.record_error(f"step{i}", f"error{i}")

        assert len(ctx_with_limits.errors) == 2
        # Should keep last 2
        assert ctx_with_limits.errors[0]["step"] == "step2"
        assert ctx_with_limits.errors[1]["step"] == "step3"

    def test_max_metrics_limit_rejection(self, ctx_with_limits):
        """Verify max_metrics rejects new metrics when limit reached."""
        # ctx_with_limits has max_metrics=5
        for i in range(5):
            assert ctx_with_limits.record_metric("comp", f"m{i}", i) is True

        # 6th should fail
        assert ctx_with_limits.record_metric("comp", "m5", 5) is False
        assert len(ctx_with_limits.metrics) == 5


# ========== Export and Serialization Tests ==========


class TestExportAndSerialization:
    """Tests for to_dict() and get_summary() methods."""

    def test_to_dict_basic(self, ctx):
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

    def test_to_dict_with_data(self, ctx_populated):
        """Verify to_dict exports populated data correctly."""
        data = ctx_populated.to_dict()

        assert len(data["steps"]) == 1
        assert len(data["metrics"]) == 2
        assert len(data["errors"]) == 1
        assert data["total_duration_seconds"] > 0

    def test_to_dict_includes_metadata(self, ctx):
        """Verify to_dict includes metadata when requested."""
        ctx.metadata["custom"] = "value"
        data = ctx.to_dict(include_metadata=True)

        assert "metadata" in data
        assert data["metadata"]["custom"] == "value"

    def test_to_dict_excludes_metadata(self, ctx):
        """Verify to_dict excludes metadata when requested."""
        ctx.metadata["custom"] = "value"
        data = ctx.to_dict(include_metadata=False)

        assert "metadata" not in data

    def test_to_dict_warns_active_timers(self, ctx):
        """Verify to_dict warns about active timers."""
        ctx.start_step("incomplete")
        data = ctx.to_dict()

        assert "active_timers" in data
        assert "incomplete" in data["active_timers"]

    def test_to_dict_creates_copies(self, ctx):
        """Verify to_dict returns copies, not references."""
        ctx.start_step("step1")
        ctx.end_step("step1")

        data = ctx.to_dict()

        # Modify the returned data
        data["steps"][0]["modified"] = True
        data["metrics"]["new"] = "value"

        # Original should be unchanged
        assert "modified" not in ctx.steps[0]
        assert "new" not in ctx.metrics

    def test_get_summary_basic(self, ctx):
        """Verify get_summary returns correct structure."""
        summary = ctx.get_summary()

        assert "request_id" in summary
        assert "total_steps" in summary
        assert "total_errors" in summary
        assert "total_metrics" in summary
        assert "active_timers" in summary
        assert "total_duration_seconds" in summary
        assert "step_statuses" in summary
        assert "has_errors" in summary
        assert "age_seconds" in summary

    def test_get_summary_with_data(self, ctx_populated):
        """Verify get_summary calculates statistics correctly."""
        summary = ctx_populated.get_summary()

        assert summary["total_steps"] == 1
        assert summary["total_errors"] == 1
        assert summary["total_metrics"] == 2
        assert summary["has_errors"] is True
        assert summary["total_duration_seconds"] > 0

    def test_get_summary_step_statuses(self, ctx):
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


# ========== Reset Functionality Tests ==========


class TestResetFunctionality:
    """Tests for reset() method."""

    def test_reset_clears_all_data(self, ctx_populated):
        """Verify reset clears all tracking data."""
        original_id = ctx_populated.request_id
        ctx_populated.reset(keep_request_id=True)

        assert ctx_populated.request_id == original_id
        assert len(ctx_populated.steps) == 0
        assert len(ctx_populated.metrics) == 0
        assert len(ctx_populated.errors) == 0
        assert len(ctx_populated._timers) == 0
        assert len(ctx_populated.metadata) == 0

    def test_reset_generates_new_id(self, ctx_populated):
        """Verify reset can generate new request_id."""
        original_id = ctx_populated.request_id
        ctx_populated.reset(keep_request_id=False)

        assert ctx_populated.request_id != original_id
        assert len(ctx_populated.steps) == 0

    def test_reset_updates_created_at(self, ctx):
        """Verify reset updates created_at timestamp."""
        original_created = ctx.created_at
        time.sleep(0.01)
        ctx.reset()

        assert ctx.created_at > original_created


# ========== Thread Safety Tests ==========


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_metric_recording(self, ctx):
        """Verify concurrent metric recording is thread-safe."""

        def record_metrics(thread_id, count):
            for i in range(count):
                ctx.record_metric(f"thread{thread_id}", f"metric{i}", i)

        threads = []
        for t in range(5):
            thread = threading.Thread(target=record_metrics, args=(t, 10))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have 5 threads * 10 metrics = 50 metrics
        assert len(ctx.metrics) == 50

    def test_concurrent_step_tracking(self, ctx):
        """Verify concurrent step tracking is thread-safe."""

        def track_step(step_name):
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

    def test_concurrent_error_recording(self, ctx):
        """Verify concurrent error recording is thread-safe."""

        def record_errors(count):
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


# ========== Context Manager Protocol Tests ==========


class TestContextManagerProtocol:
    """Tests for context manager protocol (__enter__/__exit__)."""

    def test_context_manager_enter(self, ctx):
        """Verify __enter__ returns self."""
        with ctx as entered_ctx:
            assert entered_ctx is ctx

    def test_context_manager_exit_clears_timers(self, ctx):
        """Verify __exit__ clears active timers."""
        with ctx:
            ctx.start_step("step1")
            ctx.start_step("step2")
            # Don't end them

        # Should be cleared on exit
        assert len(ctx.get_active_timers()) == 0

    def test_context_manager_doesnt_suppress_exceptions(self, ctx):
        """Verify __exit__ doesn't suppress exceptions."""
        with pytest.raises(ValueError):
            with ctx:
                raise ValueError("Test exception")


# ========== Value Sanitization Tests ==========


class TestValueSanitization:
    """Tests for _sanitize_value() method."""

    def test_sanitize_basic_types(self, ctx):
        """Verify basic types pass through sanitization."""
        assert ctx._sanitize_value(None) is None
        assert ctx._sanitize_value(True) is True
        assert ctx._sanitize_value(42) == 42
        assert ctx._sanitize_value(3.14) == 3.14
        assert ctx._sanitize_value("test") == "test"

    def test_sanitize_long_string(self, ctx):
        """Verify long strings are truncated."""
        long_string = "x" * 15000
        result = ctx._sanitize_value(long_string)
        assert len(result) <= 10014  # 10000 + "...<truncated>" (14 chars)

    def test_sanitize_list(self, ctx):
        """Verify lists are sanitized recursively."""
        test_list = [1, "two", {"three": 3}]
        result = ctx._sanitize_value(test_list)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_sanitize_large_list(self, ctx):
        """Verify large lists are truncated to 100 items."""
        large_list = list(range(200))
        result = ctx._sanitize_value(large_list)
        assert len(result) == 100

    def test_sanitize_dict(self, ctx):
        """Verify dicts are sanitized recursively."""
        test_dict = {"a": 1, "b": "two", "c": [3, 4]}
        result = ctx._sanitize_value(test_dict)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_sanitize_large_dict(self, ctx):
        """Verify large dicts are truncated to 100 items."""
        large_dict = {f"key{i}": i for i in range(200)}
        result = ctx._sanitize_value(large_dict)
        assert len(result) == 100

    def test_sanitize_datetime(self, ctx):
        """Verify datetime objects are converted to ISO format."""
        dt = datetime(2024, 1, 1, 12, 30, 45)
        result = ctx._sanitize_value(dt)
        assert isinstance(result, str)
        assert "2024-01-01" in result

    def test_sanitize_max_depth(self, ctx):
        """Verify deep nesting is limited by max_depth."""
        # Create deeply nested structure
        deep = {"level": 1}
        current = deep
        for i in range(10):
            current["nested"] = {"level": i + 2}
            current = current["nested"]

        result = ctx._sanitize_value(deep, max_depth=3)
        # Should have truncation marker at depth limit
        assert isinstance(result, dict)

    def test_sanitize_unserializable_object(self, ctx):
        """Verify unserializable objects are converted to string representation."""

        class CustomObject:
            pass

        obj = CustomObject()
        result = ctx._sanitize_value(obj)
        assert isinstance(result, str)
        assert "unserializable" in result.lower() or "CustomObject" in result


# ========== Edge Cases and Regression Tests ==========


class TestEdgeCases:
    """Tests for edge cases and potential issues."""

    def test_empty_context_to_dict(self, ctx):
        """Verify to_dict works with empty context."""
        data = ctx.to_dict()
        assert data["total_duration_seconds"] == 0
        assert len(data["steps"]) == 0

    def test_multiple_end_calls(self, ctx):
        """Verify multiple end_step calls for same step."""
        ctx.start_step("step1")
        ctx.end_step("step1")
        ctx.end_step("step1")  # Second end without start

        # Should have 2 entries (second with 0 duration)
        assert len(ctx.steps) == 2
        assert ctx.steps[1]["duration_seconds"] == 0

    def test_unicode_in_names(self, ctx):
        """Verify unicode characters in names are handled."""
        ctx.start_step("测试步骤")
        ctx.end_step("测试步骤")
        ctx.record_metric("组件", "指标", 100)
        ctx.record_error("步骤", "错误消息")

        assert len(ctx.steps) == 1
        assert len(ctx.metrics) == 1
        assert len(ctx.errors) == 1

    def test_special_characters_in_names(self, ctx):
        """Verify special characters in names are handled."""
        special_name = "step.with-special_chars!@#"
        ctx.start_step(special_name)
        ctx.end_step(special_name)

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["step"] == special_name

    def test_zero_duration_step(self, ctx):
        """Verify steps with negligible duration are recorded."""
        ctx.start_step("instant_step")
        ctx.end_step("instant_step")  # No sleep

        assert len(ctx.steps) == 1
        assert ctx.steps[0]["duration_seconds"] >= 0

    def test_age_calculation(self, ctx):
        """Verify age_seconds increases over time."""
        summary1 = ctx.get_summary()
        time.sleep(0.01)
        summary2 = ctx.get_summary()

        assert summary2["age_seconds"] > summary1["age_seconds"]


# ========== Integration Tests ==========


class TestIntegration:
    """Integration tests simulating real-world usage."""

    def test_complete_workflow(self, ctx):
        """Test a complete workflow with multiple operations."""
        # Start main process
        with ctx.step("main_process"):
            # Record some metrics
            ctx.record_metric("system", "memory_mb", 256)
            ctx.record_metric("system", "cpu_percent", 45.2)

            # Nested operations
            with ctx.step("data_loading"):
                time.sleep(0.001)
                ctx.record_metric("data", "rows_loaded", 1000)

            # Error handling
            try:
                with ctx.step("processing"):
                    raise ValueError("Processing failed")
            except ValueError:
                pass  # Already recorded by context manager

            # More work
            with ctx.step("finalization"):
                ctx.record_metric("output", "items_processed", 950)

        # Verify results
        summary = ctx.get_summary()
        assert summary["total_steps"] == 4  # main, loading, processing, finalization
        assert summary["total_errors"] == 1
        assert summary["total_metrics"] == 4
        assert summary["has_errors"] is True

        # Verify step order
        assert ctx.steps[0]["step"] == "data_loading"
        assert ctx.steps[1]["step"] == "processing"
        assert ctx.steps[2]["step"] == "finalization"
        assert ctx.steps[3]["step"] == "main_process"

    def test_request_lifecycle(self, ctx):
        """Test complete request lifecycle."""
        # Initialize
        ctx.metadata["user_id"] = "user123"
        ctx.metadata["request_type"] = "api_call"

        # Execute
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

        # Export
        result = ctx.to_dict(include_metadata=True)

        assert len(result["steps"]) == 3
        assert result["metadata"]["user_id"] == "user123"
        assert result["total_duration_seconds"] > 0

    def test_error_recovery_workflow(self, ctx):
        """Test workflow with error and recovery."""
        try:
            with ctx.step("attempt_1", error_status=StepStatus.WARNING):
                ctx.record_error(
                    "attempt_1", "Temporary failure", error_type="TransientError"
                )
                raise RuntimeError("Temporary issue")
        except RuntimeError:
            pass  # Expected exception

        # Retry succeeds
        with ctx.step("attempt_2"):
            ctx.record_metric("retry", "success", True)

        summary = ctx.get_summary()
        assert summary["total_steps"] == 2
        assert summary["total_errors"] == 2  # One explicit, one from context manager
        assert summary["step_statuses"]["warning"] == 1
        assert summary["step_statuses"]["success"] == 1


# ========== Performance Tests ==========


class TestPerformance:
    """Performance and stress tests."""

    def test_many_steps_performance(self, ctx):
        """Verify performance with many steps."""
        import time as time_module

        start = time_module.perf_counter()

        for i in range(100):
            ctx.start_step(f"step{i}")
            ctx.end_step(f"step{i}")

        duration = time_module.perf_counter() - start

        # Should complete reasonably fast (< 1 second for 100 steps)
        assert duration < 1.0
        assert len(ctx.steps) == 100

    def test_large_export(self, ctx_populated):
        """Verify to_dict() performs well with populated context."""
        # Add more data
        for i in range(50):
            ctx_populated.record_metric("comp", f"metric{i}", i)

        import time as time_module

        start = time_module.perf_counter()
        data = ctx_populated.to_dict()
        duration = time_module.perf_counter() - start

        # Should export quickly
        assert duration < 0.1
        assert isinstance(data, dict)


# ========== Business Logic Tests ==========


class TestBusinessLogic:
    """Tests for business logic translation."""

    def test_get_business_report_optimal(self, ctx):
        """Verify report for optimal conditions."""
        # Inject "good" metrics
        ctx.record_metric("flux_condenser", "avg_saturation", 0.5)
        ctx.record_metric("flux_condenser", "max_flyback_voltage", 0.02)
        ctx.record_metric("flux_condenser", "max_dissipated_power", 10.0)
        ctx.record_metric("flux_condenser", "avg_kinetic_energy", 100.0)

        report = ctx.get_business_report()

        assert report["status"] == "OPTIMO"
        assert report["message"] == "Procesamiento estable y fluido."
        assert report["metrics"]["Carga del Sistema"] == "50.0%"
        assert report["metrics"]["Índice de Inestabilidad"] == "0.0200"
        assert report["metrics"]["Fricción de Datos"] == "10.00"

    def test_get_business_report_warning_saturation(self, ctx):
        """Verify report for high saturation warning."""
        ctx.record_metric("flux_condenser", "avg_saturation", 0.95)
        ctx.record_metric("flux_condenser", "max_flyback_voltage", 0.1)
        ctx.record_metric("flux_condenser", "max_dissipated_power", 20.0)

        report = ctx.get_business_report()

        assert report["status"] == "ADVERTENCIA"
        assert report["message"] == "Sistema operando a máxima capacidad."

    def test_get_business_report_critical_voltage(self, ctx):
        """Verify report for critical flyback voltage."""
        ctx.record_metric("flux_condenser", "avg_saturation", 0.5)
        ctx.record_metric("flux_condenser", "max_flyback_voltage", 0.6)  # > 0.5
        ctx.record_metric("flux_condenser", "max_dissipated_power", 10.0)

        report = ctx.get_business_report()

        assert report["status"] == "CRITICO"
        assert report["message"] == "Archivo inestable o con baja calidad de datos."

    def test_get_business_report_critical_power(self, ctx):
        """Verify report for critical dissipated power."""
        ctx.record_metric("flux_condenser", "avg_saturation", 0.5)
        ctx.record_metric("flux_condenser", "max_flyback_voltage", 0.1)
        ctx.record_metric("flux_condenser", "max_dissipated_power", 60.0)  # > 50.0

        report = ctx.get_business_report()

        assert report["status"] == "CRITICO"
        assert report["message"] == "Archivo inestable o con baja calidad de datos."

    def test_get_business_report_missing_metrics(self, ctx):
        """Verify report handles missing metrics gracefully."""
        # No metrics recorded
        report = ctx.get_business_report()

        assert report["status"] == "OPTIMO"  # Default
        assert report["metrics"]["Carga del Sistema"] == "0.0%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
