import time
import pytest
from app.telemetry import TelemetryContext

class TestTelemetryContext:
    def test_initialization(self):
        """Verify initial state of TelemetryContext."""
        ctx = TelemetryContext()
        assert ctx.request_id is not None
        assert isinstance(ctx.steps, list)
        assert len(ctx.steps) == 0
        assert isinstance(ctx.metrics, dict)
        assert len(ctx.metrics) == 0
        assert isinstance(ctx.errors, list)
        assert len(ctx.errors) == 0

    def test_step_tracking(self):
        """Verify step start and end logic."""
        ctx = TelemetryContext()
        step_name = "test_step"

        ctx.start_step(step_name)
        assert step_name in ctx._timers

        time.sleep(0.01) # Ensure measurable duration
        ctx.end_step(step_name, "success")

        assert step_name not in ctx._timers
        assert len(ctx.steps) == 1
        step_data = ctx.steps[0]
        assert step_data["step"] == step_name
        assert step_data["status"] == "success"
        assert step_data["duration_seconds"] > 0

    def test_metrics_recording(self):
        """Verify metric recording."""
        ctx = TelemetryContext()
        ctx.record_metric("component", "metric", 42)
        assert ctx.metrics["component.metric"] == 42

    def test_error_recording(self):
        """Verify error recording."""
        ctx = TelemetryContext()
        ctx.record_error("step_failed", "Something went wrong")
        assert len(ctx.errors) == 1
        assert ctx.errors[0]["step"] == "step_failed"
        assert ctx.errors[0]["message"] == "Something went wrong"

    def test_to_dict(self):
        """Verify export to dictionary."""
        ctx = TelemetryContext()
        ctx.start_step("step1")
        ctx.end_step("step1")
        ctx.record_metric("comp", "val", 1)

        data = ctx.to_dict()
        assert data["request_id"] == ctx.request_id
        assert len(data["steps"]) == 1
        assert len(data["metrics"]) == 1
        assert "total_duration" in data
