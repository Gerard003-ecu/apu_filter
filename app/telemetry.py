import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class TelemetryContext:
    """
    Acts as the 'Passport' for a request, carrying its identity,
    execution history (steps), metrics, and errors.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    _timers: Dict[str, float] = field(default_factory=dict)

    def start_step(self, step_name: str):
        """Marks the beginning of a processing step."""
        self._timers[step_name] = time.time()
        logger.info(f"[{self.request_id}] Starting step: {step_name}")

    def end_step(self, step_name: str, status: str = "success"):
        """Marks the end of a processing step and calculates duration."""
        start_time = self._timers.pop(step_name, None)
        duration = 0.0
        if start_time:
            duration = time.time() - start_time

        self.steps.append({
            "step": step_name,
            "status": status,
            "duration_seconds": round(duration, 4),
            "timestamp": time.time()
        })
        logger.info(f"[{self.request_id}] Finished step: {step_name} ({status}) in {duration:.4f}s")

    def record_metric(self, component: str, metric_name: str, value: Any):
        """Records a specific metric for a component."""
        key = f"{component}.{metric_name}"
        self.metrics[key] = value
        logger.debug(f"[{self.request_id}] Metric {key} = {value}")

    def record_error(self, step_name: str, error_message: str):
        """Records an error that occurred during a step."""
        self.errors.append({
            "step": step_name,
            "message": error_message,
            "timestamp": time.time()
        })
        logger.error(f"[{self.request_id}] Error in {step_name}: {error_message}")

    def to_dict(self) -> Dict[str, Any]:
        """Exports the entire telemetry history."""
        return {
            "request_id": self.request_id,
            "steps": self.steps,
            "metrics": self.metrics,
            "errors": self.errors,
            "total_duration": sum(s["duration_seconds"] for s in self.steps)
        }
