import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from contextlib import contextmanager
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Enumeration for step statuses to ensure consistency."""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class TelemetryContext:
    """
    Acts as the 'Passport' for a request, carrying its identity,
    execution history (steps), metrics, and errors.
    
    Thread-safe implementation with validation and defensive programming.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    _timers: Dict[str, float] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    
    # Limits to prevent memory issues
    max_steps: int = field(default=1000)
    max_errors: int = field(default=100)
    max_metrics: int = field(default=500)
    
    # Additional context
    created_at: float = field(default_factory=time.perf_counter)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate initial state and sanitize inputs."""
        if not self.request_id or not isinstance(self.request_id, str):
            self.request_id = str(uuid.uuid4())
            logger.warning(f"Invalid request_id provided, generated new: {self.request_id}")
        
        # Ensure limits are positive
        self.max_steps = max(1, self.max_steps)
        self.max_errors = max(1, self.max_errors)
        self.max_metrics = max(1, self.max_metrics)

    def start_step(self, step_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Marks the beginning of a processing step.
        
        Args:
            step_name: Name of the step (must be non-empty string)
            metadata: Optional metadata to attach to the step
            
        Returns:
            bool: True if step started successfully, False otherwise
        """
        if not self._validate_step_name(step_name):
            return False
            
        with self._lock:
            # Warn if timer already exists (possible duplicate start)
            if step_name in self._timers:
                logger.warning(
                    f"[{self.request_id}] Step '{step_name}' already started at "
                    f"{self._timers[step_name]:.4f}. Timer will be reset."
                )
            
            # Use perf_counter for monotonic, high-resolution timing
            self._timers[step_name] = time.perf_counter()
            logger.info(f"[{self.request_id}] Starting step: {step_name}")
            
            if metadata:
                logger.debug(f"[{self.request_id}] Step metadata: {metadata}")
                
        return True

    def end_step(
        self, 
        step_name: str, 
        status: Union[StepStatus, str] = StepStatus.SUCCESS,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Marks the end of a processing step and calculates duration.
        
        Args:
            step_name: Name of the step to end
            status: Status of the step (StepStatus enum or string)
            metadata: Optional metadata to attach to the step
            
        Returns:
            bool: True if step ended successfully, False otherwise
        """
        if not self._validate_step_name(step_name):
            return False
        
        # Normalize status to StepStatus
        status_value = self._normalize_status(status)
        
        with self._lock:
            start_time = self._timers.pop(step_name, None)
            
            if start_time is None:
                logger.warning(
                    f"[{self.request_id}] Ending step '{step_name}' that was never started. "
                    f"Duration will be 0."
                )
                duration = 0.0
            else:
                duration = time.perf_counter() - start_time
            
            # Enforce max_steps limit (FIFO)
            if len(self.steps) >= self.max_steps:
                removed = self.steps.pop(0)
                logger.warning(
                    f"[{self.request_id}] Max steps ({self.max_steps}) reached. "
                    f"Removed oldest: {removed.get('step', 'unknown')}"
                )
            
            step_data = {
                "step": step_name,
                "status": status_value,
                "duration_seconds": round(duration, 6),
                "timestamp": datetime.utcnow().isoformat(),
                "perf_counter": time.perf_counter()
            }
            
            if metadata:
                step_data["metadata"] = self._sanitize_value(metadata)
            
            self.steps.append(step_data)
            
            logger.info(
                f"[{self.request_id}] Finished step: {step_name} ({status_value}) "
                f"in {duration:.6f}s"
            )
            
        return True

    @contextmanager
    def step(
        self, 
        step_name: str, 
        metadata: Optional[Dict[str, Any]] = None,
        error_status: StepStatus = StepStatus.FAILURE
    ):
        """
        Context manager for automatic step tracking with exception handling.
        
        Usage:
            with telemetry.step("processing"):
                # do work here
                pass
        
        Args:
            step_name: Name of the step
            metadata: Optional metadata to attach
            error_status: Status to use if an exception occurs (default: FAILURE)
        """
        self.start_step(step_name, metadata)
        try:
            yield self
            self.end_step(step_name, StepStatus.SUCCESS, metadata)
        except Exception as e:
            self.end_step(step_name, error_status, metadata)
            self.record_error(
                step_name=step_name,
                error_message=str(e),
                error_type=type(e).__name__,
                exception=e
            )
            raise  # Re-raise to not suppress the exception

    def record_metric(
        self, 
        component: str, 
        metric_name: str, 
        value: Any,
        overwrite: bool = True
    ) -> bool:
        """
        Records a specific metric for a component.
        
        Args:
            component: Component name (non-empty string)
            metric_name: Metric name (non-empty string)
            value: Metric value (will be sanitized for serialization)
            overwrite: Whether to overwrite existing metric (default: True)
            
        Returns:
            bool: True if metric recorded successfully, False otherwise
        """
        if not self._validate_name(component, "component"):
            return False
        
        if not self._validate_name(metric_name, "metric_name"):
            return False
        
        key = f"{component}.{metric_name}"
        
        with self._lock:
            # Check if metric exists and overwrite is disabled
            if not overwrite and key in self.metrics:
                logger.warning(
                    f"[{self.request_id}] Metric '{key}' already exists and overwrite=False"
                )
                return False
            
            # Enforce max_metrics limit
            if len(self.metrics) >= self.max_metrics and key not in self.metrics:
                logger.error(
                    f"[{self.request_id}] Max metrics ({self.max_metrics}) reached. "
                    f"Cannot record '{key}'"
                )
                return False
            
            sanitized_value = self._sanitize_value(value)
            self.metrics[key] = sanitized_value
            logger.debug(f"[{self.request_id}] Metric {key} = {sanitized_value}")
            
        return True

    def record_error(
        self, 
        step_name: str, 
        error_message: str,
        error_type: Optional[str] = None,
        exception: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Records an error that occurred during a step.
        
        Args:
            step_name: Name of the step where error occurred
            error_message: Error message (non-empty string)
            error_type: Optional error type/category
            exception: Optional exception object for additional context
            metadata: Optional additional metadata
            
        Returns:
            bool: True if error recorded successfully, False otherwise
        """
        if not self._validate_name(step_name, "step_name"):
            return False
        
        if not error_message or not isinstance(error_message, str):
            logger.error(
                f"[{self.request_id}] Invalid error_message: must be non-empty string"
            )
            return False
        
        with self._lock:
            # Enforce max_errors limit (FIFO)
            if len(self.errors) >= self.max_errors:
                removed = self.errors.pop(0)
                logger.warning(
                    f"[{self.request_id}] Max errors ({self.max_errors}) reached. "
                    f"Removed oldest from step: {removed.get('step', 'unknown')}"
                )
            
            error_data = {
                "step": step_name,
                "message": error_message[:1000],  # Limit message length
                "timestamp": datetime.utcnow().isoformat(),
                "perf_counter": time.perf_counter()
            }
            
            # Add error type
            if error_type:
                error_data["type"] = error_type
            elif exception:
                error_data["type"] = type(exception).__name__
            
            # Add exception details if available
            if exception:
                error_data["exception_details"] = str(exception)[:500]
            
            if metadata:
                error_data["metadata"] = self._sanitize_value(metadata)
            
            self.errors.append(error_data)
            logger.error(f"[{self.request_id}] Error in {step_name}: {error_message}")
            
        return True

    def get_active_timers(self) -> List[str]:
        """
        Returns list of steps that have been started but not ended.
        Useful for detecting incomplete steps.
        
        Returns:
            List of step names with active timers
        """
        with self._lock:
            return list(self._timers.keys())

    def cancel_step(self, step_name: str) -> bool:
        """
        Cancels an active step without recording it in the step history.
        
        Args:
            step_name: Name of the step to cancel
            
        Returns:
            bool: True if step was cancelled, False if step wasn't active
        """
        with self._lock:
            if step_name in self._timers:
                del self._timers[step_name]
                logger.info(f"[{self.request_id}] Cancelled step: {step_name}")
                return True
            else:
                logger.debug(
                    f"[{self.request_id}] Cannot cancel '{step_name}': not active"
                )
                return False

    def clear_active_timers(self) -> int:
        """
        Clears all active timers without recording them.
        Useful for cleanup in error scenarios.
        
        Returns:
            Number of timers cleared
        """
        with self._lock:
            count = len(self._timers)
            if count > 0:
                timer_names = list(self._timers.keys())
                logger.warning(
                    f"[{self.request_id}] Clearing {count} active timer(s): {timer_names}"
                )
                self._timers.clear()
            return count

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns a concise summary of the telemetry context.
        
        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            total_duration = sum(s.get("duration_seconds", 0) for s in self.steps)
            
            # Count steps by status
            step_statuses = {}
            for step in self.steps:
                status = step.get("status", "unknown")
                step_statuses[status] = step_statuses.get(status, 0) + 1
            
            return {
                "request_id": self.request_id,
                "total_steps": len(self.steps),
                "total_errors": len(self.errors),
                "total_metrics": len(self.metrics),
                "active_timers": len(self._timers),
                "total_duration_seconds": round(total_duration, 6),
                "step_statuses": step_statuses,
                "has_errors": len(self.errors) > 0,
                "age_seconds": round(time.perf_counter() - self.created_at, 6)
            }

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Exports the entire telemetry history as a dictionary.
        
        Args:
            include_metadata: Whether to include metadata fields
            
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        with self._lock:
            total_duration = sum(s.get("duration_seconds", 0) for s in self.steps)
            
            result = {
                "request_id": self.request_id,
                "steps": [dict(s) for s in self.steps],  # Create copies
                "metrics": dict(self.metrics),  # Create copy
                "errors": [dict(e) for e in self.errors],  # Create copies
                "total_duration_seconds": round(total_duration, 6),
                "created_at": self.created_at,
                "age_seconds": round(time.perf_counter() - self.created_at, 6)
            }
            
            if include_metadata and self.metadata:
                result["metadata"] = dict(self.metadata)
            
            # Warn about active timers
            active_timers = list(self._timers.keys())
            if active_timers:
                result["active_timers"] = active_timers
                logger.warning(
                    f"[{self.request_id}] to_dict() called with {len(active_timers)} "
                    f"active timer(s): {active_timers}. These steps are incomplete."
                )
            
            return result

    def reset(self, keep_request_id: bool = True) -> None:
        """
        Resets the telemetry context to initial state.
        
        Args:
            keep_request_id: Whether to keep the current request_id (default: True)
        """
        with self._lock:
            if not keep_request_id:
                self.request_id = str(uuid.uuid4())
            
            self.steps.clear()
            self.metrics.clear()
            self.errors.clear()
            self._timers.clear()
            self.metadata.clear()
            self.created_at = time.perf_counter()
            
            logger.info(f"[{self.request_id}] Telemetry context reset")

    def get_business_report(self) -> Dict[str, Any]:
        """
        Generates a business-friendly report based on technical metrics.
        Translates physical engineering metrics into business health indicators.

        Returns:
            Dictionary containing status, message, and translated metrics.
        """
        with self._lock:
            # 1. Extract raw metrics (safely)
            saturation = float(self.metrics.get("flux_condenser.avg_saturation", 0.0))
            flyback_voltage = float(self.metrics.get("flux_condenser.max_flyback_voltage", 0.0))
            dissipated_power = float(self.metrics.get("flux_condenser.max_dissipated_power", 0.0))
            kinetic_energy = float(self.metrics.get("flux_condenser.avg_kinetic_energy", 0.0))

            # 2. Translate to Business Terms
            business_metrics = {
                "Carga del Sistema": f"{saturation * 100:.1f}%",
                "Índice de Inestabilidad": f"{flyback_voltage:.4f}",
                "Fricción de Datos": f"{dissipated_power:.2f}",
                "Velocidad de Procesamiento": f"{kinetic_energy:.2f}"
            }

            # 3. Determine Status (Traffic Light Logic)
            status = "OPTIMO"
            message = "Procesamiento estable y fluido."

            # CRITICAL check (Highest priority)
            if flyback_voltage > 0.5 or dissipated_power > 50.0:
                status = "CRITICO"
                message = "Archivo inestable o con baja calidad de datos."

            # WARNING check
            elif saturation > 0.9:
                status = "ADVERTENCIA"
                message = "Sistema operando a máxima capacidad."

            return {
                "status": status,
                "message": message,
                "metrics": business_metrics
            }

    # ========== Validation and Helper Methods ==========

    def _validate_step_name(self, step_name: str) -> bool:
        """Validates step name format and length."""
        return self._validate_name(step_name, "step_name", max_length=255)

    def _validate_name(
        self, 
        name: str, 
        field_name: str, 
        max_length: int = 100
    ) -> bool:
        """
        Generic validation for string names.
        
        Args:
            name: The value to validate
            field_name: Name of the field (for logging)
            max_length: Maximum allowed length
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not name or not isinstance(name, str):
            logger.error(
                f"[{self.request_id}] Invalid {field_name}: must be non-empty string"
            )
            return False
        
        if len(name) > max_length:
            logger.error(
                f"[{self.request_id}] {field_name} too long: "
                f"{len(name)} chars (max {max_length})"
            )
            return False
        
        return True

    def _normalize_status(self, status: Union[StepStatus, str]) -> str:
        """
        Normalizes status to a string value.
        
        Args:
            status: StepStatus enum or string
            
        Returns:
            Normalized status string
        """
        if isinstance(status, StepStatus):
            return status.value
        
        if isinstance(status, str):
            # Try to match to a valid StepStatus
            try:
                return StepStatus(status).value
            except ValueError:
                logger.warning(
                    f"[{self.request_id}] Invalid status '{status}', "
                    f"using '{StepStatus.SUCCESS.value}'"
                )
                return StepStatus.SUCCESS.value
        
        # Fallback
        logger.warning(
            f"[{self.request_id}] Unexpected status type: {type(status)}, "
            f"using '{StepStatus.SUCCESS.value}'"
        )
        return StepStatus.SUCCESS.value

    def _sanitize_value(
        self, 
        value: Any, 
        max_depth: int = 5, 
        current_depth: int = 0
    ) -> Any:
        """
        Sanitizes a value to ensure JSON serializability.
        
        Handles nested structures with depth limiting to prevent
        infinite recursion and memory issues.
        
        Args:
            value: Value to sanitize
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
            
        Returns:
            Sanitized, JSON-serializable value
        """
        # Prevent excessive recursion
        if current_depth > max_depth:
            return "<max_depth_exceeded>"
        
        # None and basic JSON types
        if value is None or isinstance(value, (bool, int, float, str)):
            # Limit string length
            if isinstance(value, str) and len(value) > 10000:
                return value[:10000] + "...<truncated>"
            return value
        
        # Handle lists and tuples
        if isinstance(value, (list, tuple)):
            # Limit list size to prevent memory issues
            limited_list = list(value)[:100]
            return [
                self._sanitize_value(v, max_depth, current_depth + 1) 
                for v in limited_list
            ]
        
        # Handle dictionaries
        if isinstance(value, dict):
            # Limit dict size
            limited_items = list(value.items())[:100]
            return {
                str(k): self._sanitize_value(v, max_depth, current_depth + 1)
                for k, v in limited_items
            }
        
        # Handle datetime objects
        if hasattr(value, 'isoformat'):
            try:
                return value.isoformat()
            except Exception as e:
                logger.debug(f"Failed to serialize datetime-like object: {e}")
        
        # Fallback: convert to string
        try:
            str_value = str(value)
            if len(str_value) > 1000:
                return str_value[:1000] + "...<truncated>"
            return str_value
        except Exception as e:
            logger.debug(f"Failed to serialize value: {e}")
            return f"<unserializable:{type(value).__name__}>"

    # ========== Context Manager Support ==========

    def __enter__(self):
        """Support for 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Cleanup on context exit.
        Warns about and clears any active timers.
        """
        active = self.get_active_timers()
        if active:
            logger.warning(
                f"[{self.request_id}] Exiting context with {len(active)} "
                f"active timer(s): {active}. Clearing them."
            )
            self.clear_active_timers()
        
        # Don't suppress exceptions
        return False