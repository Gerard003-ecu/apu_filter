"""
Módulo de Telemetría Unificada ("Pasaporte").

Implementa el contexto de telemetría que viaja con cada solicitud para
registrar métricas, errores y pasos de ejecución de manera centralizada
y thread-safe.
"""

import copy
import logging
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
    MAX_STRING_LENGTH: int = 10000
    MAX_MESSAGE_LENGTH: int = 1000
    MAX_EXCEPTION_DETAIL_LENGTH: int = 500
    MAX_NAME_LENGTH: int = 100
    MAX_STEP_NAME_LENGTH: int = 255
    MAX_RECURSION_DEPTH: int = 5
    MAX_COLLECTION_SIZE: int = 100
    MAX_REQUEST_ID_LENGTH: int = 256
    MAX_LIMIT_MULTIPLIER: int = 10


class BusinessThresholds:
    """Umbrales configurables para el informe de negocio."""

    CRITICAL_FLYBACK_VOLTAGE: float = 0.5
    CRITICAL_DISSIPATED_POWER: float = 50.0
    WARNING_SATURATION: float = 0.9


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
        """
        Convierte una cadena a StepStatus de forma segura.

        Args:
            value: Cadena a convertir.

        Returns:
            StepStatus correspondiente o SUCCESS como fallback.
        """
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
class TelemetryContext:
    """
    Actúa como el 'Pasaporte' de una solicitud, transportando su identidad,
    historial de ejecución (pasos), métricas y errores.

    Implementación thread-safe con validación y programación defensiva.

    Attributes:
        request_id: Identificador único de la solicitud.
        steps: Lista de pasos ejecutados.
        metrics: Diccionario de métricas recolectadas.
        errors: Lista de errores registrados.
        created_at: Timestamp de creación (perf_counter).
        metadata: Metadatos adicionales del contexto.
        max_steps: Límite máximo de pasos a almacenar.
        max_errors: Límite máximo de errores a almacenar.
        max_metrics: Límite máximo de métricas a almacenar.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    _active_steps: Dict[str, ActiveStepInfo] = field(default_factory=dict)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    # Límites para prevenir problemas de memoria
    max_steps: int = field(default=TelemetryDefaults.MAX_STEPS)
    max_errors: int = field(default=TelemetryDefaults.MAX_ERRORS)
    max_metrics: int = field(default=TelemetryDefaults.MAX_METRICS)

    # Contexto adicional
    created_at: float = field(default_factory=time.perf_counter)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Umbrales de negocio (configurables por instancia)
    business_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "critical_flyback_voltage": BusinessThresholds.CRITICAL_FLYBACK_VOLTAGE,
            "critical_dissipated_power": BusinessThresholds.CRITICAL_DISSIPATED_POWER,
            "warning_saturation": BusinessThresholds.WARNING_SATURATION,
        }
    )

    def __post_init__(self) -> None:
        """Valida el estado inicial y sanitiza las entradas."""
        self._validate_and_fix_request_id()
        self._validate_and_fix_limits()
        self._validate_and_fix_collection_types()

    def _validate_and_fix_request_id(self) -> None:
        """Valida y corrige el request_id si es necesario."""
        if not self._is_valid_request_id(self.request_id):
            new_id = str(uuid.uuid4())
            logger.warning(
                f"Invalid request_id provided (type={type(self.request_id).__name__}), "
                f"generated new: {new_id}"
            )
            self.request_id = new_id

    def _is_valid_request_id(self, request_id: Any) -> bool:
        """Verifica si el request_id es válido."""
        return (
            isinstance(request_id, str)
            and 0 < len(request_id) <= TelemetryDefaults.MAX_REQUEST_ID_LENGTH
        )

    def _validate_and_fix_limits(self) -> None:
        """Valida y corrige los límites de almacenamiento."""
        max_multiplier = TelemetryDefaults.MAX_LIMIT_MULTIPLIER

        self.max_steps = self._clamp_limit(
            self.max_steps,
            1,
            TelemetryDefaults.MAX_STEPS * max_multiplier,
            "max_steps",
            TelemetryDefaults.MAX_STEPS,
        )
        self.max_errors = self._clamp_limit(
            self.max_errors,
            1,
            TelemetryDefaults.MAX_ERRORS * max_multiplier,
            "max_errors",
            TelemetryDefaults.MAX_ERRORS,
        )
        self.max_metrics = self._clamp_limit(
            self.max_metrics,
            1,
            TelemetryDefaults.MAX_METRICS * max_multiplier,
            "max_metrics",
            TelemetryDefaults.MAX_METRICS,
        )

    def _clamp_limit(
        self,
        value: Any,
        min_val: int,
        max_val: int,
        name: str,
        default: int,
    ) -> int:
        """
        Restringe un valor a un rango válido con logging.

        Args:
            value: Valor a validar.
            min_val: Valor mínimo permitido.
            max_val: Valor máximo permitido.
            name: Nombre del parámetro (para logging).
            default: Valor por defecto si el tipo es inválido.

        Returns:
            Valor restringido al rango válido.
        """
        if not isinstance(value, int):
            logger.warning(
                f"[{self.request_id}] {name} must be int, "
                f"got {type(value).__name__}. Using default: {default}"
            )
            return default

        if value < min_val:
            logger.warning(
                f"[{self.request_id}] {name}={value} below minimum, using {min_val}"
            )
            return min_val

        if value > max_val:
            logger.warning(
                f"[{self.request_id}] {name}={value} above maximum, using {max_val}"
            )
            return max_val

        return value

    def _validate_and_fix_collection_types(self) -> None:
        """Valida que las colecciones sean del tipo correcto."""
        collections_config = [
            ("steps", list, []),
            ("metrics", dict, {}),
            ("errors", list, []),
            ("metadata", dict, {}),
            (
                "business_thresholds",
                dict,
                {
                    "critical_flyback_voltage": BusinessThresholds.CRITICAL_FLYBACK_VOLTAGE,
                    "critical_dissipated_power": BusinessThresholds.CRITICAL_DISSIPATED_POWER,
                    "warning_saturation": BusinessThresholds.WARNING_SATURATION,
                },
            ),
        ]

        for attr_name, expected_type, default_value in collections_config:
            current_value = getattr(self, attr_name, None)
            if not isinstance(current_value, expected_type):
                logger.warning(
                    f"[{self.request_id}] {attr_name} must be {expected_type.__name__}, "
                    f"got {type(current_value).__name__}. Resetting to default."
                )
                setattr(self, attr_name, default_value)

    # ========== Gestión de Pasos ==========

    def start_step(self, step_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Marca el inicio de un paso de procesamiento.

        Args:
            step_name: Nombre del paso (debe ser una cadena no vacía).
            metadata: Metadatos opcionales para adjuntar al paso.

        Returns:
            bool: True si el paso se inició correctamente, False en caso contrario.
        """
        if not self._validate_step_name(step_name):
            return False

        sanitized_metadata = self._sanitize_value(metadata) if metadata else None

        with self._lock:
            if step_name in self._active_steps:
                existing = self._active_steps[step_name]
                logger.warning(
                    f"[{self.request_id}] Step '{step_name}' already started "
                    f"{existing.get_duration():.4f}s ago. Timer will be reset."
                )

            self._active_steps[step_name] = ActiveStepInfo(
                start_time=time.perf_counter(),
                metadata=sanitized_metadata,
            )
            logger.info(f"[{self.request_id}] Starting step: {step_name}")

            if sanitized_metadata:
                logger.debug(f"[{self.request_id}] Step metadata: {sanitized_metadata}")

        return True

    def end_step(
        self,
        step_name: str,
        status: Union[StepStatus, str] = StepStatus.SUCCESS,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Marca el final de un paso de procesamiento y calcula su duración.

        Combina los metadatos del inicio con los del final si ambos existen.

        Args:
            step_name: Nombre del paso a finalizar.
            status: Estado del paso (enum StepStatus o cadena).
            metadata: Metadatos opcionales para adjuntar al paso.

        Returns:
            bool: True si el paso finalizó correctamente, False en caso contrario.
        """
        if not self._validate_step_name(step_name):
            return False

        status_value = self._normalize_status(status)
        end_time = time.perf_counter()

        with self._lock:
            step_info = self._active_steps.pop(step_name, None)

            if step_info is None:
                logger.warning(
                    f"[{self.request_id}] Ending step '{step_name}' that was never started. "
                    "Duration will be 0."
                )
                duration = 0.0
                combined_metadata = None
            else:
                duration = end_time - step_info.start_time
                # Combinar metadatos de inicio y fin
                combined_metadata = self._merge_metadata(step_info.metadata, metadata)

            # Aplicar límite max_steps (FIFO)
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

            logger.info(
                f"[{self.request_id}] Finished step: {step_name} ({status_value}) "
                f"in {duration:.6f}s"
            )

        return True

    def _merge_metadata(
        self,
        start_metadata: Optional[Dict[str, Any]],
        end_metadata: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Combina los metadatos de inicio y fin de un paso.

        Args:
            start_metadata: Metadatos del inicio del paso.
            end_metadata: Metadatos del fin del paso.

        Returns:
            Diccionario combinado o None si ambos son None.
        """
        if not start_metadata and not end_metadata:
            return None

        if not start_metadata:
            return self._sanitize_value(end_metadata)

        if not end_metadata:
            return start_metadata  # Ya está sanitizado

        # Combinar con end_metadata sobrescribiendo start_metadata
        combined = dict(start_metadata)
        sanitized_end = self._sanitize_value(end_metadata)
        if isinstance(sanitized_end, dict):
            combined.update(sanitized_end)

        return combined

    def _enforce_limit_fifo(
        self,
        collection: List[Any],
        max_size: int,
        collection_name: str,
        identifier_func: Callable[[Any], str],
    ) -> int:
        """
        Aplica límite FIFO a una colección, eliminando elementos antiguos.

        Args:
            collection: Lista a limitar.
            max_size: Tamaño máximo permitido.
            collection_name: Nombre para logging.
            identifier_func: Función para obtener identificador de elementos.

        Returns:
            Número de elementos eliminados.
        """
        removed_count = 0
        while len(collection) >= max_size:
            removed = collection.pop(0)
            removed_id = identifier_func(removed)
            logger.warning(
                f"[{self.request_id}] Max {collection_name} ({max_size}) reached. "
                f"Removed oldest: {removed_id}"
            )
            removed_count += 1

        return removed_count

    @contextmanager
    def step(
        self,
        step_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        error_status: StepStatus = StepStatus.FAILURE,
        capture_exception_details: bool = True,
    ):
        """
        Gestor de contexto para el seguimiento automático de pasos con manejo de excepciones.

        Uso:
            with telemetry.step("processing"):
                # realizar trabajo aquí
                pass

        Args:
            step_name: Nombre del paso.
            metadata: Metadatos opcionales a adjuntar al inicio.
            error_status: Estado a usar si ocurre una excepción (default: FAILURE).
            capture_exception_details: Si capturar detalles de excepción (default: True).

        Yields:
            Self para encadenamiento o acceso a métodos adicionales.
        """
        started = self.start_step(step_name, metadata)

        if not started:
            logger.warning(
                f"[{self.request_id}] Step '{step_name}' failed to start, "
                "proceeding without telemetry for this step."
            )

        exception_occurred = False
        captured_exception: Optional[Exception] = None

        try:
            yield self
        except Exception as e:
            exception_occurred = True
            captured_exception = e
            raise
        finally:
            if started:
                final_status = error_status if exception_occurred else StepStatus.SUCCESS
                error_metadata = None

                if exception_occurred and capture_exception_details and captured_exception:
                    error_metadata = {
                        "error_type": type(captured_exception).__name__,
                        "error_message": str(captured_exception)[:500],
                    }

                self.end_step(step_name, final_status, error_metadata)

                if exception_occurred and capture_exception_details and captured_exception:
                    self.record_error(
                        step_name=step_name,
                        error_message=str(captured_exception),
                        error_type=type(captured_exception).__name__,
                        exception=captured_exception,
                        include_traceback=capture_exception_details,
                    )

    # ========== Gestión de Métricas ==========

    def record_metric(
        self,
        component: str,
        metric_name: str,
        value: Any,
        overwrite: bool = True,
    ) -> bool:
        """
        Registra una métrica específica para un componente.

        Args:
            component: Nombre del componente (cadena no vacía).
            metric_name: Nombre de la métrica (cadena no vacía).
            value: Valor de la métrica (se sanitizará para serialización).
            overwrite: Si se debe sobrescribir una métrica existente (default: True).

        Returns:
            bool: True si la métrica se registró correctamente, False en caso contrario.
        """
        if not self._validate_name(component, "component"):
            return False

        if not self._validate_name(metric_name, "metric_name"):
            return False

        key = f"{component}.{metric_name}"

        with self._lock:
            is_new_metric = key not in self.metrics

            if not overwrite and not is_new_metric:
                logger.warning(
                    f"[{self.request_id}] Metric '{key}' already exists and overwrite=False"
                )
                return False

            if is_new_metric and len(self.metrics) >= self.max_metrics:
                logger.error(
                    f"[{self.request_id}] Max metrics ({self.max_metrics}) reached. "
                    f"Cannot record new metric '{key}'"
                )
                return False

            sanitized_value = self._sanitize_value(value)
            self.metrics[key] = sanitized_value
            logger.debug(f"[{self.request_id}] Metric {key} = {sanitized_value}")

        return True

    def increment_metric(
        self,
        component: str,
        metric_name: str,
        increment: Union[int, float] = 1,
    ) -> bool:
        """
        Incrementa una métrica numérica existente o la inicializa.

        Args:
            component: Nombre del componente.
            metric_name: Nombre de la métrica.
            increment: Valor a incrementar (default: 1).

        Returns:
            bool: True si se incrementó correctamente, False en caso contrario.
        """
        if not self._validate_name(component, "component"):
            return False

        if not self._validate_name(metric_name, "metric_name"):
            return False

        if not isinstance(increment, (int, float)):
            logger.error(
                f"[{self.request_id}] increment must be numeric, "
                f"got {type(increment).__name__}"
            )
            return False

        key = f"{component}.{metric_name}"

        with self._lock:
            current_value = self.metrics.get(key, 0)

            if not isinstance(current_value, (int, float)):
                logger.warning(
                    f"[{self.request_id}] Metric '{key}' is not numeric "
                    f"(type={type(current_value).__name__}). Resetting to 0."
                )
                current_value = 0

            is_new_metric = key not in self.metrics
            if is_new_metric and len(self.metrics) >= self.max_metrics:
                logger.error(
                    f"[{self.request_id}] Max metrics ({self.max_metrics}) reached. "
                    f"Cannot create metric '{key}'"
                )
                return False

            new_value = current_value + increment
            self.metrics[key] = new_value
            logger.debug(
                f"[{self.request_id}] Metric {key}: {current_value} + {increment} = {new_value}"
            )

        return True

    def get_metric(self, component: str, metric_name: str, default: Any = None) -> Any:
        """
        Obtiene el valor de una métrica.

        Args:
            component: Nombre del componente.
            metric_name: Nombre de la métrica.
            default: Valor por defecto si no existe.

        Returns:
            Valor de la métrica o el valor por defecto.
        """
        key = f"{component}.{metric_name}"
        with self._lock:
            return self.metrics.get(key, default)

    # ========== Gestión de Errores ==========

    def record_error(
        self,
        step_name: str,
        error_message: str,
        error_type: Optional[str] = None,
        exception: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_traceback: bool = False,
    ) -> bool:
        """
        Registra un error ocurrido durante un paso.

        Args:
            step_name: Nombre del paso donde ocurrió el error.
            error_message: Mensaje de error (cadena no vacía).
            error_type: Tipo/categoría opcional del error.
            exception: Objeto de excepción opcional para contexto adicional.
            metadata: Metadatos adicionales opcionales.
            include_traceback: Si incluir el traceback completo (default: False).

        Returns:
            bool: True si el error se registró correctamente, False en caso contrario.
        """
        if not self._validate_name(step_name, "step_name"):
            return False

        if not self._validate_error_message(error_message):
            return False

        with self._lock:
            self._enforce_limit_fifo(
                self.errors,
                self.max_errors,
                "errors",
                lambda e: e.get("step", "unknown"),
            )

            error_data = self._build_error_data(
                step_name=step_name,
                error_message=error_message,
                error_type=error_type,
                exception=exception,
                metadata=metadata,
                include_traceback=include_traceback,
            )

            self.errors.append(error_data)
            logger.error(f"[{self.request_id}] Error in {step_name}: {error_message}")

        return True

    def _validate_error_message(self, error_message: Any) -> bool:
        """Valida que el mensaje de error sea válido."""
        if not error_message or not isinstance(error_message, str):
            logger.error(
                f"[{self.request_id}] Invalid error_message: must be non-empty string"
            )
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
    ) -> Dict[str, Any]:
        """Construye el diccionario de datos del error."""
        error_data = {
            "step": step_name,
            "message": error_message[: TelemetryDefaults.MAX_MESSAGE_LENGTH],
            "timestamp": datetime.utcnow().isoformat(),
            "perf_counter": time.perf_counter(),
        }

        # Determinar tipo de error
        if error_type:
            error_data["type"] = str(error_type)[: TelemetryDefaults.MAX_NAME_LENGTH]
        elif exception:
            error_data["type"] = type(exception).__name__

        # Agregar detalles de la excepción
        if exception:
            error_data["exception_details"] = str(exception)[
                : TelemetryDefaults.MAX_EXCEPTION_DETAIL_LENGTH
            ]

            if include_traceback:
                try:
                    tb = traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    )
                    error_data["traceback"] = "".join(tb)[:2000]
                except Exception as tb_error:
                    logger.debug(f"Failed to capture traceback: {tb_error}")

        if metadata:
            error_data["metadata"] = self._sanitize_value(metadata)

        return error_data

    # ========== Consultas y Estado ==========

    def get_active_timers(self) -> List[str]:
        """
        Retorna la lista de pasos que han iniciado pero no finalizado.

        Returns:
            Lista de nombres de pasos con temporizadores activos.
        """
        with self._lock:
            return list(self._active_steps.keys())

    def get_active_step_info(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un paso activo.

        Args:
            step_name: Nombre del paso.

        Returns:
            Diccionario con información del paso o None si no está activo.
        """
        with self._lock:
            step_info = self._active_steps.get(step_name)
            if step_info is None:
                return None

            return {
                "step_name": step_name,
                "duration_so_far": step_info.get_duration(),
                "metadata": copy.deepcopy(step_info.metadata)
                if step_info.metadata
                else None,
            }

    def has_step(self, step_name: str) -> bool:
        """
        Verifica si un paso ya fue completado.

        Args:
            step_name: Nombre del paso a buscar.

        Returns:
            True si el paso existe en el historial.
        """
        with self._lock:
            return any(s.get("step") == step_name for s in self.steps)

    def get_step_by_name(
        self, step_name: str, last: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un paso completado por nombre.

        Args:
            step_name: Nombre del paso.
            last: Si obtener la última ocurrencia (default: True).

        Returns:
            Copia del diccionario del paso o None si no existe.
        """
        with self._lock:
            matching_steps = [s for s in self.steps if s.get("step") == step_name]

            if not matching_steps:
                return None

            step = matching_steps[-1] if last else matching_steps[0]
            return copy.deepcopy(step)

    def cancel_step(self, step_name: str) -> bool:
        """
        Cancela un paso activo sin registrarlo en el historial de pasos.

        Args:
            step_name: Nombre del paso a cancelar.

        Returns:
            bool: True si el paso fue cancelado, False si no estaba activo.
        """
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
                logger.debug(f"[{self.request_id}] Cannot cancel '{step_name}': not active")
                return False

    def clear_active_timers(self) -> int:
        """
        Limpia todos los temporizadores activos sin registrarlos.

        Returns:
            Número de temporizadores limpiados.
        """
        with self._lock:
            count = len(self._active_steps)
            if count > 0:
                timer_names = list(self._active_steps.keys())
                logger.warning(
                    f"[{self.request_id}] Clearing {count} active timer(s): {timer_names}"
                )
                self._active_steps.clear()
            return count

    # ========== Resúmenes y Exportación ==========

    def get_summary(self) -> Dict[str, Any]:
        """
        Retorna un resumen conciso del contexto de telemetría.

        Returns:
            Diccionario con estadísticas resumidas.
        """
        with self._lock:
            total_duration = sum(s.get("duration_seconds", 0) for s in self.steps)

            step_statuses: Dict[str, int] = {}
            for step in self.steps:
                status = step.get("status", "unknown")
                step_statuses[status] = step_statuses.get(status, 0) + 1

            error_types: Dict[str, int] = {}
            for error in self.errors:
                error_type = error.get("type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1

            return {
                "request_id": self.request_id,
                "total_steps": len(self.steps),
                "total_errors": len(self.errors),
                "total_metrics": len(self.metrics),
                "active_timers": len(self._active_steps),
                "total_duration_seconds": round(total_duration, 6),
                "step_statuses": step_statuses,
                "error_types": error_types,
                "has_errors": len(self.errors) > 0,
                "has_failures": step_statuses.get(StepStatus.FAILURE.value, 0) > 0,
                "age_seconds": round(time.perf_counter() - self.created_at, 6),
            }

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Exporta todo el historial de telemetría como un diccionario.

        Args:
            include_metadata: Si se deben incluir los campos de metadatos.

        Returns:
            Representación en diccionario adecuada para serialización JSON.
        """
        with self._lock:
            total_duration = sum(s.get("duration_seconds", 0) for s in self.steps)

            result = {
                "request_id": self.request_id,
                "steps": copy.deepcopy(self.steps),
                "metrics": copy.deepcopy(self.metrics),
                "errors": copy.deepcopy(self.errors),
                "total_duration_seconds": round(total_duration, 6),
                "created_at": self.created_at,
                "age_seconds": round(time.perf_counter() - self.created_at, 6),
            }

            if include_metadata and self.metadata:
                result["metadata"] = copy.deepcopy(self.metadata)

            active_timers = list(self._active_steps.keys())
            if active_timers:
                result["active_timers"] = active_timers
                result["active_timers_info"] = {
                    name: {
                        "duration_so_far": info.get_duration(),
                        "has_metadata": info.metadata is not None,
                    }
                    for name, info in self._active_steps.items()
                }
                logger.warning(
                    f"[{self.request_id}] to_dict() called with {len(active_timers)} "
                    f"active timer(s): {active_timers}. These steps are incomplete."
                )

            return result

    def reset(self, keep_request_id: bool = True) -> None:
        """
        Restablece el contexto de telemetría a su estado inicial.

        Args:
            keep_request_id: Si se debe mantener el request_id actual (default: True).
        """
        with self._lock:
            old_request_id = self.request_id

            if not keep_request_id:
                self.request_id = str(uuid.uuid4())

            self.steps.clear()
            self.metrics.clear()
            self.errors.clear()
            self._active_steps.clear()
            self.metadata.clear()
            self.created_at = time.perf_counter()

            logger.info(
                f"[{self.request_id}] Telemetry context reset "
                f"(previous request_id: {old_request_id if not keep_request_id else 'kept'})"
            )

    # ========== Informe de Negocio ==========

    def get_business_report(self) -> Dict[str, Any]:
        """
        Genera un informe amigable para el negocio basado en métricas técnicas.

        Returns:
            Diccionario con status, message, metrics y details.
        """
        with self._lock:
            raw_metrics = self._extract_business_raw_metrics()
            business_metrics = self._translate_to_business_metrics(raw_metrics)
            status, message = self._determine_business_status(raw_metrics)

            return {
                "status": status,
                "message": message,
                "metrics": business_metrics,
                "details": {
                    "total_steps": len(self.steps),
                    "total_errors": len(self.errors),
                    "has_active_operations": len(self._active_steps) > 0,
                },
            }

    def _extract_business_raw_metrics(self) -> Dict[str, float]:
        """Extrae y convierte métricas crudas de forma segura."""
        metric_keys = [
            ("saturation", "flux_condenser.avg_saturation", 0.0),
            ("flyback_voltage", "flux_condenser.max_flyback_voltage", 0.0),
            ("dissipated_power", "flux_condenser.max_dissipated_power", 0.0),
            ("kinetic_energy", "flux_condenser.avg_kinetic_energy", 0.0),
        ]

        result = {}
        for name, key, default in metric_keys:
            result[name] = self._safe_float(self.metrics.get(key, default), default)

        return result

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Convierte un valor a float de forma segura."""
        if value is None:
            return default

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.debug(
                    f"[{self.request_id}] Cannot convert '{value}' to float, "
                    f"using default: {default}"
                )
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

        # Verificación CRÍTICA (Prioridad más alta)
        if (
            raw_metrics["flyback_voltage"] > critical_flyback
            or raw_metrics["dissipated_power"] > critical_power
        ):
            return "CRITICO", "Archivo inestable o con baja calidad de datos."

        # Verificación de ADVERTENCIA
        if raw_metrics["saturation"] > warning_saturation:
            return "ADVERTENCIA", "Sistema operando a máxima capacidad."

        # Verificación de errores registrados
        if self.errors:
            return (
                "ADVERTENCIA",
                f"Se registraron {len(self.errors)} error(es) durante el procesamiento.",
            )

        return "OPTIMO", "Procesamiento estable y fluido."

    # ========== Validación y Métodos Auxiliares ==========

    def _validate_step_name(self, step_name: str) -> bool:
        """Valida el formato y longitud del nombre del paso."""
        return self._validate_name(
            step_name, "step_name", max_length=TelemetryDefaults.MAX_STEP_NAME_LENGTH
        )

    def _validate_name(
        self, name: Any, field_name: str, max_length: int = TelemetryDefaults.MAX_NAME_LENGTH
    ) -> bool:
        """
        Validación genérica para nombres de cadena.

        Args:
            name: El valor a validar.
            field_name: Nombre del campo (para logging).
            max_length: Longitud máxima permitida.

        Returns:
            bool: True si es válido, False en caso contrario.
        """
        if not name or not isinstance(name, str):
            logger.error(
                f"[{self.request_id}] Invalid {field_name}: must be non-empty string, "
                f"got {type(name).__name__}"
            )
            return False

        name_stripped = name.strip()
        if not name_stripped:
            logger.error(
                f"[{self.request_id}] Invalid {field_name}: cannot be only whitespace"
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
        Normaliza el estado a un valor de cadena.

        Args:
            status: Enum StepStatus o cadena.

        Returns:
            Cadena de estado normalizada.
        """
        if isinstance(status, StepStatus):
            return status.value

        if isinstance(status, str):
            normalized = StepStatus.from_string(status)
            if normalized.value != status.lower().strip():
                logger.warning(
                    f"[{self.request_id}] Invalid status '{status}', "
                    f"using '{normalized.value}'"
                )
            return normalized.value

        logger.warning(
            f"[{self.request_id}] Unexpected status type: {type(status).__name__}, "
            f"using '{StepStatus.SUCCESS.value}'"
        )
        return StepStatus.SUCCESS.value

    def _sanitize_value(
        self,
        value: Any,
        max_depth: int = TelemetryDefaults.MAX_RECURSION_DEPTH,
        current_depth: int = 0,
    ) -> Any:
        """
        Sanitiza un valor para asegurar que sea serializable a JSON.

        Args:
            value: Valor a sanitizar.
            max_depth: Profundidad máxima de recursión.
            current_depth: Profundidad actual de recursión.

        Returns:
            Valor sanitizado y serializable a JSON.
        """
        if current_depth > max_depth:
            return "<max_depth_exceeded>"

        # None y tipos JSON básicos
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            # Manejar valores especiales de float
            if isinstance(value, float):
                if value != value:  # NaN check
                    return "<NaN>"
                if value == float("inf"):
                    return "<Infinity>"
                if value == float("-inf"):
                    return "<-Infinity>"
            return value

        if isinstance(value, str):
            max_len = TelemetryDefaults.MAX_STRING_LENGTH
            if len(value) > max_len:
                return value[:max_len] + "...<truncated>"
            return value

        # Manejar Enums
        if isinstance(value, Enum):
            return value.value

        # Manejar bytes
        if isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8", errors="replace")
                return self._sanitize_value(decoded, max_depth, current_depth)
            except Exception:
                return f"<bytes:{len(value)}>"

        # Manejar sets
        if isinstance(value, (set, frozenset)):
            return self._sanitize_value(list(value), max_depth, current_depth)

        # Manejar listas y tuplas
        if isinstance(value, (list, tuple)):
            max_size = TelemetryDefaults.MAX_COLLECTION_SIZE
            limited_list = list(value)[:max_size]
            result = [
                self._sanitize_value(v, max_depth, current_depth + 1) for v in limited_list
            ]
            if len(value) > max_size:
                result.append(f"<... {len(value) - max_size} more items>")
            return result

        # Manejar diccionarios
        if isinstance(value, dict):
            max_size = TelemetryDefaults.MAX_COLLECTION_SIZE
            items = list(value.items())[:max_size]
            result = {
                str(k): self._sanitize_value(v, max_depth, current_depth + 1)
                for k, v in items
            }
            if len(value) > max_size:
                result["<truncated>"] = f"{len(value) - max_size} more keys"
            return result

        # Manejar objetos datetime
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception as e:
                logger.debug(f"Failed to serialize datetime-like object: {e}")

        # Manejar objetos con __dict__
        if hasattr(value, "__dict__") and value.__dict__:
            try:
                return self._sanitize_value(
                    {"__class__": type(value).__name__, **value.__dict__},
                    max_depth,
                    current_depth + 1,
                )
            except Exception:
                pass

        # Fallback: convertir a string
        try:
            str_value = str(value)
            max_len = TelemetryDefaults.MAX_MESSAGE_LENGTH
            if len(str_value) > max_len:
                return str_value[:max_len] + "...<truncated>"
            return str_value
        except Exception as e:
            logger.debug(f"Failed to serialize value: {e}")
            return f"<unserializable:{type(value).__name__}>"

    # ========== Soporte para Context Manager ==========

    def __enter__(self) -> "TelemetryContext":
        """Soporte para la declaración 'with'."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """
        Limpieza al salir del contexto.

        Registra error si hay excepción y limpia temporizadores activos.
        """
        # Registrar excepción si ocurrió
        if exc_val is not None:
            self.record_error(
                step_name="__context__",
                error_message=str(exc_val),
                error_type=type(exc_val).__name__ if exc_type else None,
                exception=exc_val if isinstance(exc_val, Exception) else None,
                include_traceback=True,
            )

        # Limpiar temporizadores activos
        active = self.get_active_timers()
        if active:
            logger.warning(
                f"[{self.request_id}] Exiting context with {len(active)} "
                f"active timer(s): {active}. Marking as incomplete."
            )
            # Finalizar pasos activos con estado CANCELLED en lugar de solo limpiar
            for step_name in active:
                self.end_step(step_name, StepStatus.CANCELLED, {"reason": "context_exit"})

        # No suprimir excepciones
        return False

    # ========== Representación ==========

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
