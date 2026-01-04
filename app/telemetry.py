"""
Módulo de Telemetría Unificada ("Pasaporte").

Implementa el contexto de telemetría que viaja con cada solicitud para
registrar métricas, errores y pasos de ejecución de manera centralizada
y thread-safe. Soporta una estructura jerárquica de spans (Pirámide de Observabilidad).
"""

import copy
import logging
import math
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ========== Constantes de Configuración ==========


class TelemetryDefaults:
    """Constantes de configuración por defecto para telemetría."""

    MAX_STEPS: int = 1000
    MAX_ERRORS: int = 100
    MAX_METRICS: int = 500
    MAX_ACTIVE_STEPS: int = 50

    MAX_STRING_LENGTH: int = 10000
    MAX_MESSAGE_LENGTH: int = 1000
    MAX_EXCEPTION_DETAIL_LENGTH: int = 500
    MAX_NAME_LENGTH: int = 100
    MAX_STEP_NAME_LENGTH: int = 255
    MAX_REQUEST_ID_LENGTH: int = 256
    MAX_TRACEBACK_LENGTH: int = 5000

    MAX_RECURSION_DEPTH: int = 5
    MAX_COLLECTION_SIZE: int = 100
    MAX_DICT_KEYS: int = 200

    MAX_LIMIT_MULTIPLIER: int = 10

    MAX_STEP_DURATION_WARNING: float = 300.0  # 5 minutos
    STALE_STEP_THRESHOLD: float = 3600.0  # 1 hora

    MAX_METRIC_VALUE: float = 1e15
    MIN_METRIC_VALUE: float = -1e15


class BusinessThresholds:
    """Umbrales configurables para el informe de negocio."""

    CRITICAL_FLYBACK_VOLTAGE: float = 0.5
    CRITICAL_DISSIPATED_POWER: float = 50.0
    WARNING_SATURATION: float = 0.9
    CRITICAL_ERROR_COUNT: int = 10
    WARNING_STEP_FAILURE_RATIO: float = 0.3


@dataclass
class TelemetryHealth:
    """Estado de salud del contexto de telemetría."""

    is_healthy: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stale_steps: List[str] = field(default_factory=list)
    memory_pressure: bool = False

    def add_warning(self, msg: str) -> None:
        """Agrega una advertencia."""
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        """Agrega un error y marca como no saludable."""
        self.errors.append(msg)
        self.is_healthy = False


class StepStatus(Enum):
    """Enumeración para los estados de los pasos de ejecución."""

    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    SKIPPED = "skipped"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, value: str) -> "StepStatus":
        """Convierte una cadena a StepStatus de forma segura."""
        if not isinstance(value, str):
            return cls.SUCCESS

        value_stripped = value.strip()
        for status in cls:
            if status.value == value_stripped:
                return status

        return cls.SUCCESS


@dataclass
class ActiveStepInfo:
    """Información de un paso activo (en progreso)."""

    start_time: float
    metadata: Optional[Dict[str, Any]] = None

    def get_duration(self) -> float:
        """Calcula la duración actual del paso."""
        return time.perf_counter() - self.start_time


@dataclass
class TelemetrySpan:
    """Representa un nodo en la jerarquía de ejecución (Pirámide de Observabilidad)."""

    name: str
    level: int
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    children: List["TelemetrySpan"] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.IN_PROGRESS
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.perf_counter() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el span a diccionario."""
        return {
            "name": self.name,
            "level": self.level,
            "duration": round(self.duration, 6),
            "status": self.status.value,
            "children": [child.to_dict() for child in self.children],
            "metrics": self.metrics,
            "metadata": self.metadata,
            "errors": self.errors,
            "timestamp": datetime.utcnow().isoformat(),
        }


@dataclass
class TelemetryContext:
    """
    Actúa como el 'Pasaporte' de una solicitud, transportando su identidad,
    historial de ejecución (pasos), métricas y errores.

    Implementación thread-safe con validación y programación defensiva.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    root_spans: List[TelemetrySpan] = field(default_factory=list)
    _scope_stack: List[TelemetrySpan] = field(default_factory=list)

    _active_steps: Dict[str, ActiveStepInfo] = field(default_factory=dict)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    max_steps: int = field(default=TelemetryDefaults.MAX_STEPS)
    max_errors: int = field(default=TelemetryDefaults.MAX_ERRORS)
    max_metrics: int = field(default=TelemetryDefaults.MAX_METRICS)

    created_at: float = field(default_factory=time.perf_counter)
    metadata: Dict[str, Any] = field(default_factory=dict)

    business_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "critical_flyback_voltage": BusinessThresholds.CRITICAL_FLYBACK_VOLTAGE,
            "critical_dissipated_power": BusinessThresholds.CRITICAL_DISSIPATED_POWER,
            "warning_saturation": BusinessThresholds.WARNING_SATURATION,
        }
    )

    def __post_init__(self) -> None:
        """Valida el estado inicial y sanitiza las entradas."""
        if not hasattr(self, "_lock") or self._lock is None:
            object.__setattr__(self, "_lock", threading.RLock())

        self._validate_and_fix_request_id()
        self._validate_and_fix_limits()
        self._validate_and_fix_collection_types()
        self._validate_and_fix_business_thresholds()

        if not hasattr(self, "_active_steps") or self._active_steps is None:
            object.__setattr__(self, "_active_steps", {})

        if not isinstance(self.created_at, (int, float)) or self.created_at <= 0:
            object.__setattr__(self, "created_at", time.perf_counter())
            logger.warning(f"[{self.request_id}] Invalid created_at, reset to current time")

        logger.debug(
            f"[{self.request_id}] TelemetryContext initialized: "
            f"max_steps={self.max_steps}, max_errors={self.max_errors}, "
            f"max_metrics={self.max_metrics}"
        )

    def _validate_and_fix_request_id(self) -> None:
        """Valida y corrige el request_id si es necesario."""
        original_id = self.request_id

        if not self._is_valid_request_id(self.request_id):
            new_id = str(uuid.uuid4())
            logger.warning(
                f"Invalid request_id provided (type={type(original_id).__name__}, "
                f"value={repr(original_id)[:50]}), generated new: {new_id}"
            )
            object.__setattr__(self, "request_id", new_id)
        else:
            sanitized = self._sanitize_request_id(self.request_id)
            if sanitized != self.request_id:
                logger.debug(f"Request ID sanitized: '{self.request_id}' -> '{sanitized}'")
                object.__setattr__(self, "request_id", sanitized)

    def _is_valid_request_id(self, request_id: Any) -> bool:
        """Verifica si el request_id es válido."""
        if not isinstance(request_id, str):
            return False

        if not request_id or not request_id.strip():
            return False

        if len(request_id) > TelemetryDefaults.MAX_REQUEST_ID_LENGTH:
            return False

        return True

    def _sanitize_request_id(self, request_id: str) -> str:
        """Sanitiza el request_id eliminando caracteres problemáticos."""
        sanitized = "".join(c for c in request_id if c.isprintable() and c not in "\r\n\t")
        return sanitized.strip()[: TelemetryDefaults.MAX_REQUEST_ID_LENGTH]

    def _validate_and_fix_limits(self) -> None:
        """Valida y corrige los límites de almacenamiento."""
        max_multiplier = TelemetryDefaults.MAX_LIMIT_MULTIPLIER

        limit_configs = [
            (
                "max_steps",
                1,
                TelemetryDefaults.MAX_STEPS * max_multiplier,
                TelemetryDefaults.MAX_STEPS,
            ),
            (
                "max_errors",
                1,
                TelemetryDefaults.MAX_ERRORS * max_multiplier,
                TelemetryDefaults.MAX_ERRORS,
            ),
            (
                "max_metrics",
                1,
                TelemetryDefaults.MAX_METRICS * max_multiplier,
                TelemetryDefaults.MAX_METRICS,
            ),
        ]

        for attr_name, min_val, max_val, default in limit_configs:
            current_value = getattr(self, attr_name, default)
            corrected_value = self._clamp_limit(
                current_value, min_val, max_val, attr_name, default
            )
            if corrected_value != current_value:
                object.__setattr__(self, attr_name, corrected_value)

    def _clamp_limit(
        self,
        value: Any,
        min_val: int,
        max_val: int,
        name: str,
        default: int,
    ) -> int:
        """Restringe un valor a un rango válido con logging."""
        request_id = getattr(self, "request_id", "UNKNOWN")

        if not isinstance(value, (int, float)):
            logger.warning(
                f"[{request_id}] {name} must be numeric, "
                f"got {type(value).__name__}. Using default: {default}"
            )
            return default

        try:
            int_value = int(value)
        except (ValueError, OverflowError):
            logger.warning(
                f"[{request_id}] {name}={value} cannot be converted to int. "
                f"Using default: {default}"
            )
            return default

        if int_value < min_val:
            logger.warning(f"[{request_id}] {name}={int_value} below minimum {min_val}")
            return min_val

        if int_value > max_val:
            logger.warning(f"[{request_id}] {name}={int_value} above maximum {max_val}")
            return max_val

        return int_value

    def _validate_and_fix_collection_types(self) -> None:
        """Valida que las colecciones sean del tipo correcto."""
        request_id = getattr(self, "request_id", "UNKNOWN")

        collections_config = [
            ("steps", list, []),
            ("metrics", dict, {}),
            ("errors", list, []),
            ("metadata", dict, {}),
            ("_active_steps", dict, {}),
            ("root_spans", list, []),
            ("_scope_stack", list, []),
        ]

        for attr_name, expected_type, default_value in collections_config:
            current_value = getattr(self, attr_name, None)

            if current_value is None:
                object.__setattr__(self, attr_name, default_value.copy())
                logger.debug(f"[{request_id}] {attr_name} was None, initialized to default")
                continue

            if not isinstance(current_value, expected_type):
                converted = self._try_convert_collection(
                    current_value, expected_type, attr_name
                )
                object.__setattr__(self, attr_name, converted)

    def _try_convert_collection(self, value: Any, target_type: type, attr_name: str) -> Any:
        """Intenta convertir un valor al tipo de colección objetivo."""
        request_id = getattr(self, "request_id", "UNKNOWN")

        try:
            if target_type == list:
                if isinstance(value, (tuple, set, frozenset)):
                    result = list(value)
                    logger.info(
                        f"[{request_id}] Converted {attr_name} from "
                        f"{type(value).__name__} to list"
                    )
                    return result
            elif target_type == dict:
                if hasattr(value, "__dict__"):
                    result = dict(value.__dict__)
                    logger.info(
                        f"[{request_id}] Converted {attr_name} from "
                        f"{type(value).__name__} to dict"
                    )
                    return result
        except Exception as e:
            logger.debug(f"[{request_id}] Conversion failed for {attr_name}: {e}")

        logger.warning(
            f"[{request_id}] {attr_name} must be {target_type.__name__}, "
            f"got {type(value).__name__}. Resetting to default."
        )
        return [] if target_type == list else {}

    def _validate_and_fix_business_thresholds(self) -> None:
        """Valida y corrige los umbrales de negocio."""
        request_id = getattr(self, "request_id", "UNKNOWN")

        if not isinstance(self.business_thresholds, dict):
            logger.warning(f"[{request_id}] business_thresholds must be dict, resetting")
            object.__setattr__(
                self,
                "business_thresholds",
                {
                    "critical_flyback_voltage": BusinessThresholds.CRITICAL_FLYBACK_VOLTAGE,
                    "critical_dissipated_power": BusinessThresholds.CRITICAL_DISSIPATED_POWER,
                    "warning_saturation": BusinessThresholds.WARNING_SATURATION,
                },
            )
            return

        threshold_defaults = {
            "critical_flyback_voltage": BusinessThresholds.CRITICAL_FLYBACK_VOLTAGE,
            "critical_dissipated_power": BusinessThresholds.CRITICAL_DISSIPATED_POWER,
            "warning_saturation": BusinessThresholds.WARNING_SATURATION,
        }

        for key, default in threshold_defaults.items():
            value = self.business_thresholds.get(key)
            if not isinstance(value, (int, float)) or value < 0:
                self.business_thresholds[key] = default
                logger.debug(f"[{request_id}] Fixed threshold {key}: {value} -> {default}")

    @contextmanager
    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Crea un nuevo span jerárquico."""
        level = len(self._scope_stack)
        new_span = TelemetrySpan(
            name=name,
            level=level,
            metadata=self._sanitize_value(metadata) if metadata else {},
        )

        with self._lock:
            if self._scope_stack:
                parent = self._scope_stack[-1]
                parent.children.append(new_span)
            else:
                self.root_spans.append(new_span)

            self._scope_stack.append(new_span)

        logger.debug(f"[{self.request_id}] SPAN START: {'  ' * level}{name}")

        try:
            yield new_span
            new_span.status = StepStatus.SUCCESS
        except Exception as e:
            new_span.status = StepStatus.FAILURE
            error_data = self._build_error_data(
                step_name=name,
                error_message=str(e),
                error_type=type(e).__name__,
                exception=e,
                metadata=None,
                include_traceback=True,
                severity="ERROR",
            )
            new_span.errors.append(error_data)
            self.errors.append(error_data)
            raise
        finally:
            new_span.end_time = time.perf_counter()
            with self._lock:
                if self._scope_stack and self._scope_stack[-1] == new_span:
                    self._scope_stack.pop()

            logger.debug(
                f"[{self.request_id}] SPAN END: {'  ' * level}{name} ({new_span.duration:.4f}s)"
            )

    def start_step(self, step_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Marca el inicio de un paso de procesamiento."""
        if not self._validate_step_name(step_name):
            return False

        sanitized_metadata = None
        if metadata is not None:
            if not isinstance(metadata, dict):
                logger.warning(
                    f"[{self.request_id}] metadata for step '{step_name}' is not dict "
                    f"(type={type(metadata).__name__}), ignoring"
                )
            else:
                try:
                    sanitized_metadata = self._sanitize_value(metadata)
                except Exception as e:
                    logger.warning(
                        f"[{self.request_id}] Failed to sanitize metadata for '{step_name}': {e}"
                    )

        with self._lock:
            if len(self._active_steps) >= TelemetryDefaults.MAX_ACTIVE_STEPS:
                self._cleanup_stale_steps()

                if len(self._active_steps) >= TelemetryDefaults.MAX_ACTIVE_STEPS:
                    logger.error(
                        f"[{self.request_id}] Max active steps "
                        f"({TelemetryDefaults.MAX_ACTIVE_STEPS}) reached. "
                        f"Cannot start '{step_name}'."
                    )
                    return False

            if step_name in self._active_steps:
                existing = self._active_steps[step_name]
                duration = existing.get_duration()

                if duration > TelemetryDefaults.MAX_STEP_DURATION_WARNING:
                    logger.warning(
                        f"[{self.request_id}] Step '{step_name}' already active for "
                        f"{duration:.1f}s (possible stuck step). Resetting timer."
                    )
                else:
                    logger.warning(
                        f"[{self.request_id}] Step '{step_name}' already started "
                        f"{duration:.4f}s ago. Timer will be reset."
                    )

            self._active_steps[step_name] = ActiveStepInfo(
                start_time=time.perf_counter(),
                metadata=sanitized_metadata,
            )

            logger.info(f"[{self.request_id}] Starting step: {step_name}")

        return True

    def _cleanup_stale_steps(self) -> int:
        """Limpia pasos que han estado activos demasiado tiempo."""
        if not self._active_steps:
            return 0

        current_time = time.perf_counter()
        stale_threshold = TelemetryDefaults.STALE_STEP_THRESHOLD
        stale_steps = []

        for step_name, info in self._active_steps.items():
            duration = current_time - info.start_time
            if duration > stale_threshold:
                stale_steps.append((step_name, duration))

        for step_name, duration in stale_steps:
            logger.warning(
                f"[{self.request_id}] Cleaning up stale step '{step_name}' "
                f"(active for {duration:.1f}s)"
            )
            self._force_end_step(
                step_name,
                StepStatus.FAILURE,
                {"reason": "stale_cleanup", "duration_before_cleanup": duration},
            )

        return len(stale_steps)

    def _force_end_step(
        self,
        step_name: str,
        status: StepStatus,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fuerza el fin de un paso sin validaciones adicionales."""
        end_time = time.perf_counter()
        step_info = self._active_steps.pop(step_name, None)

        duration = (end_time - step_info.start_time) if step_info else 0.0

        step_data = {
            "step": step_name,
            "status": status.value,
            "duration_seconds": round(duration, 6),
            "timestamp": datetime.utcnow().isoformat(),
            "perf_counter": end_time,
            "forced": True,
        }

        if metadata:
            step_data["metadata"] = metadata

        if len(self.steps) >= self.max_steps:
            self.steps.pop(0)

        self.steps.append(step_data)

    def end_step(
        self,
        step_name: str,
        status: Union[StepStatus, str] = StepStatus.SUCCESS,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Marca el final de un paso de procesamiento."""
        if not self._validate_step_name(step_name):
            return False

        status_value = self._normalize_status(status)
        end_time = time.perf_counter()

        with self._lock:
            step_info = self._active_steps.pop(step_name, None)

            if step_info is None:
                logger.warning(
                    f"[{self.request_id}] Ending step '{step_name}' that was never started. "
                    "Recording with duration=0."
                )
                duration = 0.0
                combined_metadata = (
                    self._sanitize_value(metadata) if metadata else None
                ) or {}
                combined_metadata["warning"] = "step_never_started"
            else:
                duration = end_time - step_info.start_time
                combined_metadata = self._merge_metadata(step_info.metadata, metadata)

                if duration > TelemetryDefaults.MAX_STEP_DURATION_WARNING:
                    logger.warning(
                        f"[{self.request_id}] Step '{step_name}' took {duration:.1f}s "
                        f"(exceeds warning threshold)"
                    )
                elif duration < 0:
                    logger.error(
                        f"[{self.request_id}] Step '{step_name}' has negative duration."
                    )
                    duration = 0.0

            self._enforce_limit_fifo(
                self.steps,
                self.max_steps,
                "steps",
                lambda s: s.get("step", "unknown"),
            )

            step_data = {
                "step": step_name,
                "status": status_value,
                "duration_seconds": round(duration, 6),
                "timestamp": datetime.utcnow().isoformat(),
                "perf_counter": end_time,
            }

            if combined_metadata:
                step_data["metadata"] = combined_metadata

            self.steps.append(step_data)

            log_func = (
                logger.info if status_value == StepStatus.SUCCESS.value else logger.warning
            )
            log_func(
                f"[{self.request_id}] Finished step: {step_name} ({status_value}) "
                f"in {duration:.6f}s"
            )

        return True

    def _merge_metadata(
        self,
        start_metadata: Optional[Dict[str, Any]],
        end_metadata: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Combina los metadatos de inicio y fin de un paso."""
        if not start_metadata and not end_metadata:
            return None

        if not start_metadata:
            sanitized = self._sanitize_value(end_metadata)
            return sanitized if isinstance(sanitized, dict) else None

        if not end_metadata:
            return start_metadata

        try:
            combined = dict(start_metadata)
            sanitized_end = self._sanitize_value(end_metadata)

            if isinstance(sanitized_end, dict):
                if len(combined) + len(sanitized_end) > TelemetryDefaults.MAX_DICT_KEYS:
                    logger.warning(
                        f"[{self.request_id}] Combined metadata exceeds key limit, truncating"
                    )
                    remaining_slots = TelemetryDefaults.MAX_DICT_KEYS - len(sanitized_end)
                    if remaining_slots > 0:
                        combined = dict(list(combined.items())[:remaining_slots])
                    else:
                        combined = {}

                combined.update(sanitized_end)

            return combined

        except Exception as e:
            logger.warning(
                f"[{self.request_id}] Error merging metadata: {e}. Using end_metadata only."
            )
            return self._sanitize_value(end_metadata) if end_metadata else start_metadata

    def _enforce_limit_fifo(
        self,
        collection: List[Any],
        max_size: int,
        collection_name: str,
        identifier_func: Callable[[Any], str],
    ) -> int:
        """Aplica límite FIFO a una colección."""
        if not isinstance(collection, list):
            return 0

        if max_size <= 0:
            return 0

        removed_count = 0
        removed_ids = []

        while len(collection) >= max_size:
            if not collection:
                break

            try:
                removed = collection.pop(0)
                removed_id = (
                    identifier_func(removed) if callable(identifier_func) else "unknown"
                )
                removed_ids.append(removed_id)
                removed_count += 1
            except Exception as e:
                logger.error(f"[{self.request_id}] Error removing item: {e}")
                break

        if removed_count > 0:
            if removed_count <= 3:
                for rid in removed_ids:
                    logger.warning(
                        f"[{self.request_id}] Max {collection_name} ({max_size}) reached. "
                        f"Removed: {rid}"
                    )
            else:
                logger.warning(
                    f"[{self.request_id}] Max {collection_name} ({max_size}) reached. "
                    f"Removed {removed_count} oldest items."
                )

        return removed_count

    @contextmanager
    def step(
        self,
        step_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        error_status: StepStatus = StepStatus.FAILURE,
        capture_exception_details: bool = True,
        auto_record_error: bool = True,
        suppress_start_failure: bool = True,
    ):
        """Gestor de contexto para el seguimiento automático de pasos."""
        if not isinstance(step_name, str) or not step_name.strip():
            logger.error(f"[{self.request_id}] Invalid step_name for context manager")
            if not suppress_start_failure:
                raise ValueError("step_name must be a non-empty string")
            yield self
            return

        if not isinstance(error_status, StepStatus):
            if isinstance(error_status, str):
                error_status = StepStatus.from_string(error_status)
            else:
                error_status = StepStatus.FAILURE

        started = False
        try:
            started = self.start_step(step_name, metadata)
        except Exception as e:
            logger.error(f"[{self.request_id}] Exception in start_step('{step_name}'): {e}")
            if not suppress_start_failure:
                raise

        if not started and not suppress_start_failure:
            raise RuntimeError(f"Failed to start step: {step_name}")
        elif not started:
            logger.warning(
                f"[{self.request_id}] Step '{step_name}' failed to start, "
                "proceeding without telemetry for this step."
            )

        exception_occurred = False
        captured_exception: Optional[BaseException] = None

        try:
            yield self
        except BaseException as e:
            exception_occurred = True
            captured_exception = e
            raise
        finally:
            if started:
                final_status = error_status if exception_occurred else StepStatus.SUCCESS

                error_metadata = None
                if exception_occurred and capture_exception_details and captured_exception:
                    error_metadata = self._build_exception_metadata(captured_exception)

                try:
                    self.end_step(step_name, final_status, error_metadata)
                except Exception as end_error:
                    logger.error(
                        f"[{self.request_id}] Exception in end_step('{step_name}'): {end_error}"
                    )

                if (
                    exception_occurred
                    and auto_record_error
                    and capture_exception_details
                    and isinstance(captured_exception, Exception)
                ):
                    try:
                        self.record_error(
                            step_name=step_name,
                            error_message=str(captured_exception),
                            error_type=type(captured_exception).__name__,
                            exception=captured_exception,
                            include_traceback=capture_exception_details,
                        )
                    except Exception as record_error:
                        logger.error(
                            f"[{self.request_id}] Failed to record error: {record_error}"
                        )

    def _build_exception_metadata(self, exc: BaseException) -> Dict[str, Any]:
        """Construye metadata de excepción de forma segura."""
        try:
            return {
                "error_type": type(exc).__name__,
                "error_message": str(exc)[: TelemetryDefaults.MAX_EXCEPTION_DETAIL_LENGTH],
                "is_base_exception": not isinstance(exc, Exception),
            }
        except Exception:
            return {"error_type": "unknown", "error_message": "failed to capture"}

    def record_metric(
        self,
        component: str,
        metric_name: str,
        value: Any,
        overwrite: bool = True,
        validate_numeric: bool = False,
    ) -> bool:
        """Registra una métrica específica para un componente."""
        if not self._validate_name(component, "component"):
            return False

        if not self._validate_name(metric_name, "metric_name"):
            return False

        key = f"{component}.{metric_name}"

        if validate_numeric:
            if not isinstance(value, (int, float)):
                return False

            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return False

        current_span = None
        with self._lock:
            if self._scope_stack:
                current_span = self._scope_stack[-1]

        with self._lock:
            is_new_metric = key not in self.metrics

            if not overwrite and not is_new_metric:
                return False

            if is_new_metric and len(self.metrics) >= self.max_metrics:
                logger.error(
                    f"[{self.request_id}] Max metrics ({self.max_metrics}) reached."
                )
                return False

            try:
                sanitized_value = self._sanitize_value(value)
            except Exception:
                return False

            if isinstance(sanitized_value, (int, float)):
                if sanitized_value > TelemetryDefaults.MAX_METRIC_VALUE:
                    sanitized_value = TelemetryDefaults.MAX_METRIC_VALUE
                elif sanitized_value < TelemetryDefaults.MIN_METRIC_VALUE:
                    sanitized_value = TelemetryDefaults.MIN_METRIC_VALUE

            self.metrics[key] = sanitized_value

            if current_span:
                current_span.metrics[key] = sanitized_value

        return True

    def increment_metric(
        self,
        component: str,
        metric_name: str,
        increment: Union[int, float] = 1,
        create_if_missing: bool = True,
    ) -> bool:
        """Incrementa una métrica numérica."""
        if not self._validate_name(component, "component"):
            return False

        if not self._validate_name(metric_name, "metric_name"):
            return False

        if not isinstance(increment, (int, float)):
            return False

        if isinstance(increment, float) and (math.isnan(increment) or math.isinf(increment)):
            return False

        key = f"{component}.{metric_name}"

        with self._lock:
            current_value = self.metrics.get(key)
            is_new_metric = current_value is None

            if is_new_metric:
                if not create_if_missing:
                    return False

                if len(self.metrics) >= self.max_metrics:
                    return False

                current_value = 0

            if not isinstance(current_value, (int, float)):
                current_value = 0

            try:
                new_value = current_value + increment

                if isinstance(new_value, float):
                    if math.isinf(new_value):
                        new_value = (
                            TelemetryDefaults.MAX_METRIC_VALUE
                            if increment > 0
                            else TelemetryDefaults.MIN_METRIC_VALUE
                        )
                    elif math.isnan(new_value):
                        return False

            except OverflowError:
                new_value = TelemetryDefaults.MAX_METRIC_VALUE

            new_value = max(
                TelemetryDefaults.MIN_METRIC_VALUE,
                min(TelemetryDefaults.MAX_METRIC_VALUE, new_value),
            )

            self.metrics[key] = new_value

            if self._scope_stack:
                self._scope_stack[-1].metrics[key] = new_value

        return True

    def get_metric(
        self,
        component: str,
        metric_name: str,
        default: Any = None,
        expected_type: Optional[type] = None,
    ) -> Any:
        """Obtiene el valor de una métrica."""
        key = f"{component}.{metric_name}"

        with self._lock:
            value = self.metrics.get(key)

            if value is None:
                return default

            if expected_type is not None and not isinstance(value, expected_type):
                return default

            if isinstance(value, (dict, list)):
                return copy.deepcopy(value)

            return value

    def record_error(
        self,
        step_name: str,
        error_message: str,
        error_type: Optional[str] = None,
        exception: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_traceback: bool = False,
        severity: str = "ERROR",
    ) -> bool:
        """Registra un error ocurrido durante un paso."""
        if not self._validate_name(step_name, "step_name"):
            step_name = "__unknown_step__"

        if not self._validate_error_message(error_message):
            error_message = "Unknown error"

        valid_severities = {"ERROR", "WARNING", "CRITICAL", "INFO"}
        if severity not in valid_severities:
            severity = "ERROR"

        with self._lock:
            self._enforce_limit_fifo(
                self.errors,
                self.max_errors,
                "errors",
                lambda e: f"{e.get('step', 'unknown')}:{e.get('type', 'unknown')}",
            )

            try:
                error_data = self._build_error_data(
                    step_name=step_name,
                    error_message=error_message,
                    error_type=error_type,
                    exception=exception,
                    metadata=metadata,
                    include_traceback=include_traceback,
                    severity=severity,
                )
            except Exception as build_error:
                error_data = {
                    "step": step_name,
                    "message": str(error_message)[:500],
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": severity,
                    "build_error": str(build_error)[:100],
                }

            self.errors.append(error_data)

            if self._scope_stack:
                self._scope_stack[-1].errors.append(error_data)

            log_func = {
                "CRITICAL": logger.critical,
                "ERROR": logger.error,
                "WARNING": logger.warning,
                "INFO": logger.info,
            }.get(severity, logger.error)

            log_func(
                f"[{self.request_id}] {severity} in {step_name}: "
                f"{error_message[:200]}{'...' if len(error_message) > 200 else ''}"
            )

        return True

    def _validate_error_message(self, error_message: Any) -> bool:
        """Valida que el mensaje de error sea válido."""
        if error_message is None:
            return False

        if not isinstance(error_message, str):
            return False

        if not error_message.strip():
            return False

        return True

    def _build_error_data(
        self,
        step_name: str,
        error_message: str,
        error_type: Optional[str],
        exception: Optional[Exception],
        metadata: Optional[Dict[str, Any]],
        include_traceback: bool,
        severity: str = "ERROR",
    ) -> Dict[str, Any]:
        """Construye el diccionario de datos del error."""
        error_data = {
            "step": step_name,
            "message": error_message[: TelemetryDefaults.MAX_MESSAGE_LENGTH],
            "timestamp": datetime.utcnow().isoformat(),
            "perf_counter": time.perf_counter(),
            "severity": severity,
        }

        if error_type:
            error_data["type"] = str(error_type)[: TelemetryDefaults.MAX_NAME_LENGTH]
        elif exception:
            error_data["type"] = type(exception).__name__

        if exception:
            try:
                exc_str = str(exception)
                error_data["exception_details"] = exc_str[
                    : TelemetryDefaults.MAX_EXCEPTION_DETAIL_LENGTH
                ]
            except Exception as e:
                error_data["exception_details"] = f"<failed to stringify: {e}>"

            if include_traceback:
                error_data["traceback"] = self._capture_traceback_safe(exception)

            if hasattr(exception, "__cause__") and exception.__cause__:
                error_data["cause"] = str(exception.__cause__)[:200]
            if hasattr(exception, "__context__") and exception.__context__:
                error_data["context"] = str(exception.__context__)[:200]

        if metadata:
            try:
                sanitized_metadata = self._sanitize_value(metadata)
                if isinstance(sanitized_metadata, dict):
                    error_data["metadata"] = sanitized_metadata
            except Exception as e:
                error_data["metadata_error"] = str(e)[:100]

        return error_data

    def _capture_traceback_safe(self, exception: Exception) -> str:
        """Captura el traceback de forma segura."""
        try:
            tb_lines = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
            tb_str = "".join(tb_lines)
            return tb_str[: TelemetryDefaults.MAX_TRACEBACK_LENGTH]
        except Exception as tb_error:
            return f"<traceback capture failed: {tb_error}>"

    def get_active_timers(self) -> List[str]:
        """Retorna la lista de pasos que han iniciado pero no finalizado."""
        with self._lock:
            return list(self._active_steps.keys())

    def get_active_step_info(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un paso activo."""
        with self._lock:
            step_info = self._active_steps.get(step_name)
            if step_info is None:
                return None

            return {
                "step_name": step_name,
                "duration_so_far": step_info.get_duration(),
                "metadata": (
                    copy.deepcopy(step_info.metadata) if step_info.metadata else None
                ),
            }

    def has_step(self, step_name: str) -> bool:
        """Verifica si un paso ya fue completado."""
        with self._lock:
            return any(s.get("step") == step_name for s in self.steps)

    def get_step_by_name(
        self, step_name: str, last: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Obtiene información de un paso completado por nombre."""
        with self._lock:
            matching_steps = [s for s in self.steps if s.get("step") == step_name]

            if not matching_steps:
                return None

            step = matching_steps[-1] if last else matching_steps[0]
            return copy.deepcopy(step)

    def cancel_step(self, step_name: str) -> bool:
        """Cancela un paso activo sin registrarlo en el historial de pasos."""
        with self._lock:
            if step_name in self._active_steps:
                step_info = self._active_steps.pop(step_name)
                duration = step_info.get_duration()
                logger.info(
                    f"[{self.request_id}] Cancelled step: {step_name} "
                    f"(was active for {duration:.6f}s)"
                )
                return True
            else:
                return False

    def clear_active_timers(self) -> int:
        """Limpia todos los temporizadores activos sin registrarlos."""
        with self._lock:
            count = len(self._active_steps)
            if count > 0:
                self._active_steps.clear()
            return count

    def _validate_step_name(self, step_name: str) -> bool:
        """Valida el formato y longitud del nombre del paso."""
        return self._validate_name(
            step_name,
            "step_name",
            max_length=TelemetryDefaults.MAX_STEP_NAME_LENGTH,
        )

    def _validate_name(
        self,
        name: Any,
        field_name: str,
        max_length: int = TelemetryDefaults.MAX_NAME_LENGTH,
    ) -> bool:
        """Validación genérica para nombres de cadena."""
        if not name or not isinstance(name, str):
            logger.error(
                f"[{self.request_id}] Invalid {field_name}: must be non-empty string"
            )
            return False

        if not name.strip():
            logger.error(
                f"[{self.request_id}] Invalid {field_name}: cannot be only whitespace"
            )
            return False

        if len(name) > max_length:
            logger.error(f"[{self.request_id}] {field_name} too long")
            return False

        return True

    def _normalize_status(self, status: Union[StepStatus, str]) -> str:
        """Normaliza el estado a un valor de cadena."""
        if isinstance(status, StepStatus):
            return status.value

        if isinstance(status, str):
            normalized = StepStatus.from_string(status)
            return normalized.value

        return StepStatus.SUCCESS.value

    def _sanitize_value(
        self,
        value: Any,
        max_depth: int = TelemetryDefaults.MAX_RECURSION_DEPTH,
        current_depth: int = 0,
        _seen: Optional[set] = None,
    ) -> Any:
        """Sanitiza un valor para asegurar que sea serializable a JSON."""
        if _seen is None:
            _seen = set()

        if current_depth > max_depth:
            return f"<max_depth_exceeded:{current_depth}>"

        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return self._sanitize_numeric(value)

        if isinstance(value, str):
            return self._sanitize_string(value)

        if isinstance(value, (dict, list, set)):
            value_id = id(value)
            if value_id in _seen:
                return "<circular_reference>"
            _seen = _seen | {value_id}

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, bytes):
            return self._sanitize_bytes(value, max_depth, current_depth, _seen)

        if isinstance(value, (set, frozenset)):
            items = list(value)[: TelemetryDefaults.MAX_COLLECTION_SIZE]
            result = [
                self._sanitize_value(v, max_depth, current_depth + 1, _seen) for v in items
            ]
            if len(value) > TelemetryDefaults.MAX_COLLECTION_SIZE:
                result.append(
                    f"<... {len(value) - TelemetryDefaults.MAX_COLLECTION_SIZE} more>"
                )
            return result

        if isinstance(value, (list, tuple)):
            return self._sanitize_sequence(value, max_depth, current_depth, _seen)

        if isinstance(value, dict):
            return self._sanitize_dict(value, max_depth, current_depth, _seen)

        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                return f"<datetime:{type(value).__name__}>"

        if hasattr(value, "__dict__") and isinstance(value.__dict__, dict):
            try:
                obj_dict = {"__class__": type(value).__name__}
                obj_dict.update(value.__dict__)
                return self._sanitize_value(obj_dict, max_depth, current_depth + 1, _seen)
            except Exception:
                pass

        if hasattr(value, "_asdict"):
            try:
                return self._sanitize_value(
                    value._asdict(), max_depth, current_depth + 1, _seen
                )
            except Exception:
                pass

        return self._sanitize_to_string(value)

    def _sanitize_numeric(self, value: Union[int, float]) -> Union[int, float, str]:
        """Sanitiza un valor numérico."""
        if isinstance(value, float):
            if math.isnan(value):
                return "<NaN>"
            if math.isinf(value):
                return "<Infinity>" if value > 0 else "<-Infinity>"
            if abs(value) > 0 and abs(value) < 1e-10:
                return 0.0
        return value

    def _sanitize_string(self, value: str) -> str:
        """Sanitiza una cadena de texto."""
        max_len = TelemetryDefaults.MAX_STRING_LENGTH
        cleaned = "".join(c for c in value if c.isprintable() or c in "\n\t\r")

        if len(cleaned) > max_len:
            return cleaned[:max_len] + "...<truncated>"
        return cleaned

    def _sanitize_bytes(
        self, value: bytes, max_depth: int, current_depth: int, _seen: set
    ) -> Union[str, Dict[str, Any]]:
        """Sanitiza un valor de bytes."""
        try:
            decoded = value.decode("utf-8", errors="replace")
            return self._sanitize_value(decoded, max_depth, current_depth, _seen)
        except Exception:
            return {
                "type": "bytes",
                "length": len(value),
                "preview": value[:50].hex() if len(value) > 0 else "",
            }

    def _sanitize_sequence(
        self,
        value: Union[list, tuple],
        max_depth: int,
        current_depth: int,
        _seen: set,
    ) -> List[Any]:
        """Sanitiza una secuencia (lista o tupla)."""
        max_size = TelemetryDefaults.MAX_COLLECTION_SIZE
        limited_list = list(value)[:max_size]

        result = []
        for item in limited_list:
            try:
                sanitized = self._sanitize_value(item, max_depth, current_depth + 1, _seen)
                result.append(sanitized)
            except Exception as e:
                result.append(f"<sanitization_error:{e}>")

        if len(value) > max_size:
            result.append(f"<... {len(value) - max_size} more items>")

        return result

    def _sanitize_dict(
        self, value: dict, max_depth: int, current_depth: int, _seen: set
    ) -> Dict[str, Any]:
        """Sanitiza un diccionario."""
        max_size = TelemetryDefaults.MAX_DICT_KEYS
        result = {}
        items = list(value.items())[:max_size]

        for k, v in items:
            try:
                str_key = str(k)[: TelemetryDefaults.MAX_NAME_LENGTH]
                sanitized_value = self._sanitize_value(
                    v, max_depth, current_depth + 1, _seen
                )
                result[str_key] = sanitized_value
            except Exception as e:
                result[f"<key_error:{k}>"] = f"<sanitization_error:{e}>"

        if len(value) > max_size:
            result["<truncated>"] = f"{len(value) - max_size} more keys"

        return result

    def _sanitize_to_string(self, value: Any) -> str:
        """Convierte un valor a string de forma segura."""
        try:
            str_value = str(value)
            max_len = TelemetryDefaults.MAX_MESSAGE_LENGTH
            if len(str_value) > max_len:
                return str_value[:max_len] + "...<truncated>"
            return str_value
        except Exception as e:
            return f"<unserializable:{type(value).__name__}:{e}>"

    def get_summary(self) -> Dict[str, Any]:
        """Retorna un resumen conciso del contexto de telemetría."""
        with self._lock:
            try:
                total_duration = 0.0
                for step in self.steps:
                    duration = step.get("duration_seconds")
                    if isinstance(duration, (int, float)) and not math.isnan(duration):
                        total_duration += duration

                step_statuses: Dict[str, int] = {}
                for step in self.steps:
                    status = step.get("status", "unknown")
                    if isinstance(status, str):
                        step_statuses[status] = step_statuses.get(status, 0) + 1

                error_types: Dict[str, int] = {}
                for error in self.errors:
                    error_type = error.get("type", "unknown")
                    if isinstance(error_type, str):
                        error_types[error_type] = error_types.get(error_type, 0) + 1

                current_time = time.perf_counter()
                age = current_time - self.created_at if self.created_at > 0 else 0.0

                stale_count = sum(
                    1
                    for info in self._active_steps.values()
                    if info.get_duration() > TelemetryDefaults.STALE_STEP_THRESHOLD
                )

                return {
                    "request_id": self.request_id,
                    "total_steps": len(self.steps),
                    "total_errors": len(self.errors),
                    "total_metrics": len(self.metrics),
                    "total_spans": len(self.root_spans),
                    "active_timers": len(self._active_steps),
                    "stale_timers": stale_count,
                    "total_duration_seconds": round(total_duration, 6),
                    "step_statuses": step_statuses,
                    "error_types": error_types,
                    "has_errors": len(self.errors) > 0,
                    "has_failures": step_statuses.get(StepStatus.FAILURE.value, 0) > 0,
                    "success_rate": self._calculate_success_rate(step_statuses),
                    "age_seconds": round(age, 6),
                    "limits": {
                        "steps": f"{len(self.steps)}/{self.max_steps}",
                        "errors": f"{len(self.errors)}/{self.max_errors}",
                        "metrics": f"{len(self.metrics)}/{self.max_metrics}",
                    },
                }

            except Exception as e:
                logger.error(f"[{self.request_id}] Error generating summary: {e}")
                return {
                    "request_id": self.request_id,
                    "error": f"Failed to generate summary: {e}",
                }

    def _calculate_success_rate(self, step_statuses: Dict[str, int]) -> float:
        """Calcula la tasa de éxito de los pasos."""
        total = sum(step_statuses.values())
        if total == 0:
            return 1.0

        success_count = step_statuses.get(StepStatus.SUCCESS.value, 0)
        return round(success_count / total, 4)

    def to_dict(
        self,
        include_metadata: bool = True,
        include_active_timers: bool = True,
        deep_copy: bool = True,
    ) -> Dict[str, Any]:
        """Exporta todo el historial de telemetría como un diccionario."""
        with self._lock:
            try:
                total_duration = sum(
                    s.get("duration_seconds", 0)
                    for s in self.steps
                    if isinstance(s.get("duration_seconds"), (int, float))
                )

                if deep_copy:
                    steps_data = copy.deepcopy(self.steps)
                    metrics_data = copy.deepcopy(self.metrics)
                    errors_data = copy.deepcopy(self.errors)
                else:
                    steps_data = list(self.steps)
                    metrics_data = dict(self.metrics)
                    errors_data = list(self.errors)

                result = {
                    "request_id": self.request_id,
                    "steps": steps_data,
                    "metrics": metrics_data,
                    "errors": errors_data,
                    "spans": [s.to_dict() for s in self.root_spans],
                    "total_duration_seconds": round(total_duration, 6),
                    "created_at": self.created_at,
                    "age_seconds": round(time.perf_counter() - self.created_at, 6),
                    "summary": self.get_summary(),
                }

                if include_metadata and self.metadata:
                    if deep_copy:
                        result["metadata"] = copy.deepcopy(self.metadata)
                    else:
                        result["metadata"] = dict(self.metadata)

                active_timers = list(self._active_steps.keys())
                if active_timers and include_active_timers:
                    result["active_timers"] = active_timers
                    result["active_timers_info"] = {
                        name: {
                            "duration_so_far": round(info.get_duration(), 6),
                            "has_metadata": info.metadata is not None,
                        }
                        for name, info in self._active_steps.items()
                    }

                return result

            except Exception as e:
                logger.error(f"[{self.request_id}] Error in to_dict(): {e}")
                return {
                    "request_id": self.request_id,
                    "error": f"Failed to export: {e}",
                    "partial_data": {
                        "steps_count": len(self.steps),
                        "errors_count": len(self.errors),
                        "metrics_count": len(self.metrics),
                    },
                }

    def reset(self, keep_request_id: bool = True) -> None:
        """Restablece el contexto de telemetría."""
        with self._lock:
            if not keep_request_id:
                self.request_id = str(uuid.uuid4())

            self.steps.clear()
            self.metrics.clear()
            self.errors.clear()
            self._active_steps.clear()
            self.root_spans.clear()
            self._scope_stack.clear()
            self.metadata.clear()
            self.created_at = time.perf_counter()

    def get_business_report(self) -> Dict[str, Any]:
        """Genera un informe amigable para el negocio."""
        with self._lock:
            try:
                raw_metrics = self._extract_business_raw_metrics()
                business_metrics = self._translate_to_business_metrics(raw_metrics)
                status, message = self._determine_business_status(raw_metrics)
                step_stats = self._calculate_step_statistics()
                financial_health = self._determine_financial_health()

                return {
                    "status": status,
                    "message": message,
                    "metrics": business_metrics,
                    "raw_metrics": raw_metrics,
                    "details": {
                        "total_steps": len(self.steps),
                        "successful_steps": step_stats["success"],
                        "failed_steps": step_stats["failure"],
                        "total_errors": len(self.errors),
                        "has_active_operations": len(self._active_steps) > 0,
                        "active_operation_names": list(self._active_steps.keys())[:5],
                        "total_duration": step_stats["total_duration"],
                        "success_rate": step_stats["success_rate"],
                    },
                    "financial_health": financial_health,
                    "health": self._assess_health(),
                    "timestamp": datetime.utcnow().isoformat(),
                }

            except Exception as e:
                logger.error(f"[{self.request_id}] Error generating business report: {e}")
                return {
                    "status": "ERROR",
                    "message": f"Failed to generate report: {e}",
                    "metrics": {},
                    "details": {"error": str(e)},
                }

    def _extract_business_raw_metrics(self) -> Dict[str, float]:
        """Extrae y convierte métricas crudas de forma segura."""
        metric_keys = [
            ("saturation", "flux_condenser.avg_saturation", 0.0),
            ("flyback_voltage", "flux_condenser.max_flyback_voltage", 0.0),
            ("dissipated_power", "flux_condenser.max_dissipated_power", 0.0),
            ("kinetic_energy", "flux_condenser.avg_kinetic_energy", 0.0),
            ("processed_records", "flux_condenser.processed_records", 0),
            ("total_records", "flux_condenser.total_records", 0),
            ("processing_time", "flux_condenser.processing_time", 0.0),
        ]

        result = {}
        for name, key, default in metric_keys:
            value = self.metrics.get(key, default)
            result[name] = self._safe_float(value, default)

        return result

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Convierte un valor a float de forma segura."""
        if value is None:
            return default

        if isinstance(value, bool):
            return 1.0 if value else 0.0

        if isinstance(value, (int, float)):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return default
            return float(value)

        if isinstance(value, str):
            try:
                parsed = float(value)
                if math.isnan(parsed) or math.isinf(parsed):
                    return default
                return parsed
            except (ValueError, TypeError):
                return default

        return default

    def _translate_to_business_metrics(
        self, raw_metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """Traduce métricas técnicas a términos de negocio."""
        return {
            "Carga del Sistema": f"{raw_metrics['saturation'] * 100:.1f}%",
            "Índice de Inestabilidad": f"{raw_metrics['flyback_voltage']:.4f}",
            "Fricción de Datos": f"{raw_metrics['dissipated_power']:.2f}",
            "Velocidad de Procesamiento": f"{raw_metrics['kinetic_energy']:.2f}",
            "Registros Procesados": f"{int(raw_metrics['processed_records']):,}",
            "Tiempo de Proceso": f"{raw_metrics['processing_time']:.2f}s",
        }

    def _determine_business_status(self, raw_metrics: Dict[str, float]) -> Tuple[str, str]:
        """Determina el estado de negocio basado en métricas y umbrales."""
        thresholds = self.business_thresholds

        critical_flyback = thresholds.get(
            "critical_flyback_voltage",
            BusinessThresholds.CRITICAL_FLYBACK_VOLTAGE,
        )
        critical_power = thresholds.get(
            "critical_dissipated_power",
            BusinessThresholds.CRITICAL_DISSIPATED_POWER,
        )
        warning_saturation = thresholds.get(
            "warning_saturation",
            BusinessThresholds.WARNING_SATURATION,
        )

        if raw_metrics["flyback_voltage"] > critical_flyback:
            return (
                "CRITICO",
                f"Alta inestabilidad detectada (V={raw_metrics['flyback_voltage']:.2f})",
            )

        if raw_metrics["dissipated_power"] > critical_power:
            return (
                "CRITICO",
                f"Fricción de datos excesiva (P={raw_metrics['dissipated_power']:.1f})",
            )

        if len(self.errors) >= BusinessThresholds.CRITICAL_ERROR_COUNT:
            return "CRITICO", f"Demasiados errores registrados ({len(self.errors)})"

        if raw_metrics["saturation"] > warning_saturation:
            return (
                "ADVERTENCIA",
                f"Sistema operando a {raw_metrics['saturation'] * 100:.0f}% de capacidad",
            )

        step_stats = self._calculate_step_statistics()
        if step_stats["failure_rate"] > BusinessThresholds.WARNING_STEP_FAILURE_RATIO:
            return (
                "ADVERTENCIA",
                f"Alta tasa de fallos: {step_stats['failure_rate'] * 100:.1f}%",
            )

        if self.errors:
            return (
                "ADVERTENCIA",
                f"Se registraron {len(self.errors)} error(es) durante el procesamiento",
            )

        return "OPTIMO", "Procesamiento estable y fluido"

    def _determine_financial_health(self) -> Dict[str, Any]:
        """Determina la salud financiera basada en métricas financieras."""
        financial_metrics = {
            "roi": self.get_metric("financial", "roi"),
            "volatility": self.get_metric("financial", "volatility"),
            "npv": self.get_metric("financial", "npv"),
        }

        present_metrics = {k: v for k, v in financial_metrics.items() if v is not None}

        if not present_metrics:
            return {
                "status": "NO_DISPONIBLE",
                "message": "No hay métricas financieras registradas.",
                "metrics": {},
            }

        status = "OPTIMO"
        message = "Proyecto viable."

        if "roi" in present_metrics and present_metrics["roi"] < 0:
            status = "CRITICO"
            message = "Destrucción de Valor proyectada."
        elif "volatility" in present_metrics and present_metrics["volatility"] > 0.20:
            status = "ADVERTENCIA"
            message = "Alta volatilidad de mercado."

        return {"status": status, "message": message, "metrics": present_metrics}

    def _calculate_step_statistics(self) -> Dict[str, Any]:
        """Calcula estadísticas de pasos de forma segura."""
        total = len(self.steps)

        if total == 0:
            return {
                "success": 0,
                "failure": 0,
                "warning": 0,
                "total_duration": 0.0,
                "success_rate": 1.0,
                "failure_rate": 0.0,
            }

        success = sum(1 for s in self.steps if s.get("status") == StepStatus.SUCCESS.value)
        failure = sum(1 for s in self.steps if s.get("status") == StepStatus.FAILURE.value)
        warning = sum(1 for s in self.steps if s.get("status") == StepStatus.WARNING.value)

        total_duration = sum(
            s.get("duration_seconds", 0)
            for s in self.steps
            if isinstance(s.get("duration_seconds"), (int, float))
        )

        return {
            "success": success,
            "failure": failure,
            "warning": warning,
            "total_duration": round(total_duration, 6),
            "success_rate": success / total if total > 0 else 1.0,
            "failure_rate": failure / total if total > 0 else 0.0,
        }

    def _assess_health(self) -> Dict[str, Any]:
        """Evalúa la salud del contexto de telemetría."""
        health = TelemetryHealth()

        for name, info in self._active_steps.items():
            duration = info.get_duration()
            if duration > TelemetryDefaults.STALE_STEP_THRESHOLD:
                health.stale_steps.append(name)
                health.add_warning(f"Step '{name}' stuck for {duration:.0f}s")

        if len(self.steps) > self.max_steps * 0.9:
            health.memory_pressure = True
            health.add_warning("Steps approaching limit")

        if len(self.errors) > self.max_errors * 0.9:
            health.memory_pressure = True
            health.add_warning("Errors approaching limit")

        if len(self.errors) > BusinessThresholds.CRITICAL_ERROR_COUNT:
            health.add_error(f"High error count: {len(self.errors)}")

        return {
            "is_healthy": health.is_healthy,
            "warnings": health.warnings,
            "errors": health.errors,
            "stale_steps": health.stale_steps,
            "memory_pressure": health.memory_pressure,
        }

    def __enter__(self) -> "TelemetryContext":
        """Soporte para la declaración 'with'."""
        logger.debug(f"[{self.request_id}] Entering telemetry context")
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Limpieza al salir del contexto."""
        try:
            if exc_val is not None:
                try:
                    self.record_error(
                        step_name="__context__",
                        error_message=str(exc_val)[:500],
                        error_type=exc_type.__name__ if exc_type else "UnknownException",
                        exception=exc_val if isinstance(exc_val, Exception) else None,
                        include_traceback=True,
                        severity="CRITICAL",
                    )
                except Exception as record_error:
                    logger.error(
                        f"[{self.request_id}] Failed to record context exception: {record_error}"
                    )

            active = self.get_active_timers()
            if active:
                logger.warning(
                    f"[{self.request_id}] Exiting context with {len(active)} "
                    f"active timer(s): {active[:5]}{'...' if len(active) > 5 else ''}. "
                    "Marking as cancelled."
                )

                for step_name in active:
                    try:
                        self.end_step(
                            step_name,
                            StepStatus.CANCELLED,
                            {
                                "reason": "context_exit",
                                "had_exception": exc_val is not None,
                            },
                        )
                    except Exception as end_error:
                        logger.error(
                            f"[{self.request_id}] Failed to end step '{step_name}': {end_error}"
                        )

            logger.debug(
                f"[{self.request_id}] Exiting telemetry context "
                f"(exception={exc_type.__name__ if exc_type else 'None'})"
            )

        except Exception as cleanup_error:
            logger.error(
                f"[{self.request_id}] Error during context cleanup: {cleanup_error}"
            )

        return False

    def __repr__(self) -> str:
        """Representación concisa del contexto."""
        with self._lock:
            return (
                f"TelemetryContext("
                f"request_id='{self.request_id}', "
                f"steps={len(self.steps)}, "
                f"errors={len(self.errors)}, "
                f"metrics={len(self.metrics)}, "
                f"active={len(self._active_steps)})"
            )

    def __str__(self) -> str:
        """Representación en cadena del resumen."""
        summary = self.get_summary()
        return (
            f"Telemetry[{summary['request_id'][:8]}...]: "
            f"{summary['total_steps']} steps, "
            f"{summary['total_errors']} errors, "
            f"{summary['total_duration_seconds']:.3f}s"
        )
