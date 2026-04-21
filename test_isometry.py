from app.adapters.tools_interface import ExecutionCommand, ProjectionContext, MICMetrics, MICConfiguration, TTLCache
from app.core.schemas import Stratum

def test_isometry():
    cache = TTLCache(max_size=10, ttl_seconds=10)
    metrics = MICMetrics()
    config = MICConfiguration()
    cmd = ExecutionCommand(cache, metrics, config)

    ctx = ProjectionContext(
        service_name="test",
        payload={},
        context={"dissipated_power": -1.0, "_phase_correction": 0.5, "exergy_level": 10.0},
        use_cache=False
    )
    ctx.target_stratum = Stratum.WISDOM
    ctx.handler = lambda: {}

    try:
        cmd.execute(ctx)
        print("Failed to raise error!")
    except Exception as e:
        print(f"Raised correctly: {type(e).__name__}: {e}")

test_isometry()
