import pytest
from app.telemetry import TelemetryContext, Stratum, TelemetryHealth
from app.schemas import Stratum

class TestPyramidalTelemetry:
    def test_stratum_health_tracking(self):
        ctx = TelemetryContext()

        # Start a step in PHYSICS stratum
        ctx.start_step("read_io", stratum=Stratum.PHYSICS)

        # Record a non-critical error
        ctx.record_error(
            step_name="read_io",
            error_message="Minor IO glitch",
            severity="ERROR"
        )

        # Verify PHYSICS health is degraded
        assert not ctx._strata_health[Stratum.PHYSICS].is_healthy
        # Errors are tuples (msg, timestamp)
        assert any("Minor IO glitch" in e[0] for e in ctx._strata_health[Stratum.PHYSICS].errors)

        # Verify TACTICS is still healthy (no propagation for non-CRITICAL)
        assert ctx._strata_health[Stratum.TACTICS].is_healthy

    def test_failure_propagation(self):
        ctx = TelemetryContext()

        # Start a step in PHYSICS stratum
        ctx.start_step("db_crash", stratum=Stratum.PHYSICS)

        # Record a CRITICAL error
        ctx.record_error(
            step_name="db_crash",
            error_message="Database exploded",
            severity="CRITICAL"
        )

        # Verify PHYSICS is unhealthy
        assert not ctx._strata_health[Stratum.PHYSICS].is_healthy

        # Verify propagation to TACTICS (Level 2)
        assert len(ctx._strata_health[Stratum.TACTICS].warnings) > 0
        # Warnings are tuples (msg, timestamp), and message is in English in V2
        assert any("Instability inherited from PHYSICS" in w[0] for w in ctx._strata_health[Stratum.TACTICS].warnings)

        # Verify propagation to STRATEGY (Level 1)
        assert len(ctx._strata_health[Stratum.STRATEGY].warnings) > 0
        assert any("Instability inherited from PHYSICS" in w[0] for w in ctx._strata_health[Stratum.STRATEGY].warnings)

    def test_pyramidal_report_structure(self):
        ctx = TelemetryContext()

        # Simulate some activity
        ctx.record_metric("flux_condenser", "avg_saturation", 0.8)
        ctx.record_metric("topology", "beta_0", 1)

        report = ctx.get_pyramidal_report()

        assert "physics_layer" in report
        assert "tactics_layer" in report
        assert "strategy_layer" in report

        assert report["physics_layer"]["status"] == "HEALTHY"
        assert "flux_condenser.avg_saturation" in report["physics_layer"]["metrics"]

        # Introduce a failure
        ctx.start_step("critical_fail", stratum=Stratum.PHYSICS)
        ctx.record_error("critical_fail", "BOOM", severity="CRITICAL")

        report_fail = ctx.get_pyramidal_report()
        assert report_fail["physics_layer"]["status"] == "CRITICAL"

        # Check Strategy Layer: Should be WARNING because of propagation
        # Strategy health: is_healthy=True (no errors), but has warnings.
        # Logic: if health.warnings: return "WARNING"
        assert report_fail["strategy_layer"]["status"] == "WARNING"
        assert len(report_fail["strategy_layer"]["warnings"]) > 0
        # Report flattens tuples to strings, check for English message
        assert "Instability inherited" in report_fail["strategy_layer"]["warnings"][0]

    def test_context_manager_stratum(self):
        ctx = TelemetryContext()

        with ctx.span("test_span", stratum=Stratum.TACTICS) as span:
            assert span.stratum == Stratum.TACTICS

        assert ctx.root_spans[-1].stratum == Stratum.TACTICS

    def test_explicit_record_error_stratum(self):
        ctx = TelemetryContext()

        # Record error with explicit stratum, overriding any active step or default
        ctx.record_error(
            step_name="outside_step",
            error_message="Strategic Blunder",
            severity="ERROR",
            stratum=Stratum.STRATEGY
        )

        assert not ctx._strata_health[Stratum.STRATEGY].is_healthy
        assert any("Strategic Blunder" in e[0] for e in ctx._strata_health[Stratum.STRATEGY].errors)

        # Ensure PHYSICS is unaffected
        assert ctx._strata_health[Stratum.PHYSICS].is_healthy
