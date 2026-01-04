import time

import pytest

from app.telemetry import TelemetryContext
from app.telemetry_narrative import TelemetryNarrator


def test_telemetry_span_hierarchy():
    ctx = TelemetryContext()

    with ctx.span("Root Phase") as root:
        time.sleep(0.01)
        with ctx.span("Child Operation") as child:
            time.sleep(0.01)
            child.metrics["processed"] = 100

    assert len(ctx.root_spans) == 1
    assert ctx.root_spans[0].name == "Root Phase"
    assert len(ctx.root_spans[0].children) == 1
    assert ctx.root_spans[0].children[0].name == "Child Operation"
    assert ctx.root_spans[0].children[0].metrics["processed"] == 100
    assert ctx.root_spans[0].duration > 0
    assert ctx.root_spans[0].children[0].duration > 0


def test_telemetry_narrative_generation():
    ctx = TelemetryContext()

    # Simulate a process with some errors
    try:
        with ctx.span("Data Loading") as load_span:
            with ctx.span("Read CSV"):
                pass
            with ctx.span("Validate Schema"):
                raise ValueError("Invalid Column: 'Price'")
    except ValueError:
        pass  # Expected

    narrator = TelemetryNarrator()
    report = narrator.summarize_execution(ctx)

    assert report["verdict"] == "CRITICO"
    assert len(report["phases"]) == 1
    assert report["phases"][0]["name"] == "Data Loading"
    assert report["phases"][0]["status"] == "CRITICO"

    evidence = report["forensic_evidence"]
    assert len(evidence) > 0
    # The error propagates up, so we check if any evidence matches the source
    assert any(e["source"] == "Validate Schema" for e in evidence)
    assert any("Invalid Column" in e["message"] for e in evidence)


def test_telemetry_legacy_fallback():
    ctx = TelemetryContext()
    ctx.start_step("Legacy Step 1")
    ctx.end_step("Legacy Step 1")
    ctx.record_error("Legacy Step 2", "Something went wrong")

    narrator = TelemetryNarrator()
    report = narrator.summarize_execution(ctx)

    assert report["verdict"] == "CRITICO"
    assert "modo compatibilidad" in report["narrative"]
    assert len(report["forensic_evidence"]) == 1


def test_mixed_mode():
    """Test using both spans and legacy steps together."""
    ctx = TelemetryContext()

    with ctx.span("Hybrid Phase"):
        ctx.start_step("Legacy Inside Span")
        ctx.record_metric("legacy", "count", 50)
        ctx.end_step("Legacy Inside Span")

    assert len(ctx.root_spans) == 1
    # Check that legacy metrics were recorded globally
    assert ctx.metrics["legacy.count"] == 50
    # And also in the span
    assert ctx.root_spans[0].metrics["legacy.count"] == 50


if __name__ == "__main__":
    pytest.main([__file__])
