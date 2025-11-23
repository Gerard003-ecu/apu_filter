"""
Módulo de Telemetría Unificada ("Pasaporte").

Implementa el contexto de telemetría que viaja con cada solicitud para
registrar métricas, errores y pasos de ejecución de manera centralizada
y thread-safe.
"""

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Enumeración para los estados de los pasos de ejecución."""

    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    SKIPPED = "skipped"


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
        created_at: Timestamp de creación.
        metadata: Metadatos adicionales.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    _timers: Dict[str, float] = field(default_factory=dict)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    # Límites para prevenir problemas de memoria
    max_steps: int = field(default=1000)
    max_errors: int = field(default=100)
    max_metrics: int = field(default=500)

    # Contexto adicional
    created_at: float = field(default_factory=time.perf_counter)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Valida el estado inicial y sanitiza las entradas."""
        if not self.request_id or not isinstance(self.request_id, str):
            self.request_id = str(uuid.uuid4())
            logger.warning(f"Invalid request_id provided, generated new: {self.request_id}")

        # Asegurar que los límites sean positivos
        self.max_steps = max(1, self.max_steps)
        self.max_errors = max(1, self.max_errors)
        self.max_metrics = max(1, self.max_metrics)

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

        with self._lock:
            # Advertir si el temporizador ya existe (posible inicio duplicado)
            if step_name in self._timers:
                logger.warning(
                    f"[{self.request_id}] Step '{step_name}' already started at "
                    f"{self._timers[step_name]:.4f}. Timer will be reset."
                )

            # Usar perf_counter para cronometraje monotónico de alta resolución
            self._timers[step_name] = time.perf_counter()
            logger.info(f"[{self.request_id}] Starting step: {step_name}")

            if metadata:
                logger.debug(f"[{self.request_id}] Step metadata: {metadata}")

        return True

    def end_step(
        self,
        step_name: str,
        status: Union[StepStatus, str] = StepStatus.SUCCESS,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Marca el final de un paso de procesamiento y calcula su duración.

        Args:
            step_name: Nombre del paso a finalizar.
            status: Estado del paso (enum StepStatus o cadena).
            metadata: Metadatos opcionales para adjuntar al paso.

        Returns:
            bool: True si el paso finalizó correctamente, False en caso contrario.
        """
        if not self._validate_step_name(step_name):
            return False

        # Normalizar estado a StepStatus
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

            # Aplicar límite max_steps (FIFO)
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
                "perf_counter": time.perf_counter(),
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
        error_status: StepStatus = StepStatus.FAILURE,
    ):
        """
        Gestor de contexto para el seguimiento automático de pasos con manejo de excepciones.

        Uso:
            with telemetry.step("processing"):
                # realizar trabajo aquí
                pass

        Args:
            step_name: Nombre del paso.
            metadata: Metadatos opcionales a adjuntar.
            error_status: Estado a usar si ocurre una excepción (default: FAILURE).
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
                exception=e,
            )
            raise  # Re-lanzar para no suprimir la excepción

    def record_metric(
        self, component: str, metric_name: str, value: Any, overwrite: bool = True
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
            # Verificar si la métrica existe y la sobrescritura está deshabilitada
            if not overwrite and key in self.metrics:
                logger.warning(
                    f"[{self.request_id}] Metric '{key}' already exists and overwrite=False"
                )
                return False

            # Aplicar límite max_metrics
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Registra un error ocurrido durante un paso.

        Args:
            step_name: Nombre del paso donde ocurrió el error.
            error_message: Mensaje de error (cadena no vacía).
            error_type: Tipo/categoría opcional del error.
            exception: Objeto de excepción opcional para contexto adicional.
            metadata: Metadatos adicionales opcionales.

        Returns:
            bool: True si el error se registró correctamente, False en caso contrario.
        """
        if not self._validate_name(step_name, "step_name"):
            return False

        if not error_message or not isinstance(error_message, str):
            logger.error(
                f"[{self.request_id}] Invalid error_message: must be non-empty string"
            )
            return False

        with self._lock:
            # Aplicar límite max_errors (FIFO)
            if len(self.errors) >= self.max_errors:
                removed = self.errors.pop(0)
                logger.warning(
                    f"[{self.request_id}] Max errors ({self.max_errors}) reached. "
                    f"Removed oldest from step: {removed.get('step', 'unknown')}"
                )

            error_data = {
                "step": step_name,
                "message": error_message[:1000],  # Limitar longitud del mensaje
                "timestamp": datetime.utcnow().isoformat(),
                "perf_counter": time.perf_counter(),
            }

            # Agregar tipo de error
            if error_type:
                error_data["type"] = error_type
            elif exception:
                error_data["type"] = type(exception).__name__

            # Agregar detalles de la excepción si están disponibles
            if exception:
                error_data["exception_details"] = str(exception)[:500]

            if metadata:
                error_data["metadata"] = self._sanitize_value(metadata)

            self.errors.append(error_data)
            logger.error(f"[{self.request_id}] Error in {step_name}: {error_message}")

        return True

    def get_active_timers(self) -> List[str]:
        """
        Retorna la lista de pasos que han iniciado pero no finalizado.
        Útil para detectar pasos incompletos.

        Returns:
            Lista de nombres de pasos con temporizadores activos.
        """
        with self._lock:
            return list(self._timers.keys())

    def cancel_step(self, step_name: str) -> bool:
        """
        Cancela un paso activo sin registrarlo en el historial de pasos.

        Args:
            step_name: Nombre del paso a cancelar.

        Returns:
            bool: True si el paso fue cancelado, False si no estaba activo.
        """
        with self._lock:
            if step_name in self._timers:
                del self._timers[step_name]
                logger.info(f"[{self.request_id}] Cancelled step: {step_name}")
                return True
            else:
                logger.debug(f"[{self.request_id}] Cannot cancel '{step_name}': not active")
                return False

    def clear_active_timers(self) -> int:
        """
        Limpia todos los temporizadores activos sin registrarlos.
        Útil para limpieza en escenarios de error.

        Returns:
            Número de temporizadores limpiados.
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
        Retorna un resumen conciso del contexto de telemetría.

        Returns:
            Diccionario con estadísticas resumidas.
        """
        with self._lock:
            total_duration = sum(s.get("duration_seconds", 0) for s in self.steps)

            # Contar pasos por estado
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
                "steps": [dict(s) for s in self.steps],  # Crear copias
                "metrics": dict(self.metrics),  # Crear copia
                "errors": [dict(e) for e in self.errors],  # Crear copias
                "total_duration_seconds": round(total_duration, 6),
                "created_at": self.created_at,
                "age_seconds": round(time.perf_counter() - self.created_at, 6),
            }

            if include_metadata and self.metadata:
                result["metadata"] = dict(self.metadata)

            # Advertir sobre temporizadores activos
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
        Restablece el contexto de telemetría a su estado inicial.

        Args:
            keep_request_id: Si se debe mantener el request_id actual (default: True).
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
        Genera un informe amigable para el negocio basado en métricas técnicas.
        Traduce métricas de ingeniería física a indicadores de salud del negocio.

        Returns:
            Diccionario con estado, mensaje y métricas traducidas.
        """
        with self._lock:
            # 1. Extraer métricas crudas (de forma segura)
            saturation = float(self.metrics.get("flux_condenser.avg_saturation", 0.0))
            flyback_voltage = float(
                self.metrics.get("flux_condenser.max_flyback_voltage", 0.0)
            )
            dissipated_power = float(
                self.metrics.get("flux_condenser.max_dissipated_power", 0.0)
            )
            kinetic_energy = float(
                self.metrics.get("flux_condenser.avg_kinetic_energy", 0.0)
            )

            # 2. Traducir a Términos de Negocio
            business_metrics = {
                "Carga del Sistema": f"{saturation * 100:.1f}%",
                "Índice de Inestabilidad": f"{flyback_voltage:.4f}",
                "Fricción de Datos": f"{dissipated_power:.2f}",
                "Velocidad de Procesamiento": f"{kinetic_energy:.2f}",
            }

            # 3. Determinar Estado (Lógica de Semáforo)
            status = "OPTIMO"
            message = "Procesamiento estable y fluido."

            # Verificación CRÍTICA (Prioridad más alta)
            if flyback_voltage > 0.5 or dissipated_power > 50.0:
                status = "CRITICO"
                message = "Archivo inestable o con baja calidad de datos."

            # Verificación de ADVERTENCIA
            elif saturation > 0.9:
                status = "ADVERTENCIA"
                message = "Sistema operando a máxima capacidad."

            return {"status": status, "message": message, "metrics": business_metrics}

    # ========== Validación y Métodos Auxiliares ==========

    def _validate_step_name(self, step_name: str) -> bool:
        """Valida el formato y longitud del nombre del paso."""
        return self._validate_name(step_name, "step_name", max_length=255)

    def _validate_name(self, name: str, field_name: str, max_length: int = 100) -> bool:
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
        Normaliza el estado a un valor de cadena.

        Args:
            status: Enum StepStatus o cadena.

        Returns:
            Cadena de estado normalizada.
        """
        if isinstance(status, StepStatus):
            return status.value

        if isinstance(status, str):
            # Intentar coincidir con un StepStatus válido
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

    def _sanitize_value(self, value: Any, max_depth: int = 5, current_depth: int = 0) -> Any:
        """
        Sanitiza un valor para asegurar que sea serializable a JSON.

        Maneja estructuras anidadas con límite de profundidad para prevenir
        recursión infinita y problemas de memoria.

        Args:
            value: Valor a sanitizar.
            max_depth: Profundidad máxima de recursión.
            current_depth: Profundidad actual de recursión.

        Returns:
            Valor sanitizado y serializable a JSON.
        """
        # Prevenir recursión excesiva
        if current_depth > max_depth:
            return "<max_depth_exceeded>"

        # None y tipos JSON básicos
        if value is None or isinstance(value, (bool, int, float, str)):
            # Limitar longitud de cadenas
            if isinstance(value, str) and len(value) > 10000:
                return value[:10000] + "...<truncated>"
            return value

        # Manejar listas y tuplas
        if isinstance(value, (list, tuple)):
            # Limitar tamaño de lista para prevenir problemas de memoria
            limited_list = list(value)[:100]
            return [
                self._sanitize_value(v, max_depth, current_depth + 1) for v in limited_list
            ]

        # Manejar diccionarios
        if isinstance(value, dict):
            # Limitar tamaño de diccionario
            limited_items = list(value.items())[:100]
            return {
                str(k): self._sanitize_value(v, max_depth, current_depth + 1)
                for k, v in limited_items
            }

        # Manejar objetos datetime
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception as e:
                logger.debug(f"Failed to serialize datetime-like object: {e}")

        # Fallback: convertir a string
        try:
            str_value = str(value)
            if len(str_value) > 1000:
                return str_value[:1000] + "...<truncated>"
            return str_value
        except Exception as e:
            logger.debug(f"Failed to serialize value: {e}")
            return f"<unserializable:{type(value).__name__}>"

    # ========== Soporte para Context Manager ==========

    def __enter__(self):
        """Soporte para la declaración 'with'."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Limpieza al salir del contexto.
        Advierte sobre y limpia cualquier temporizador activo.
        """
        active = self.get_active_timers()
        if active:
            logger.warning(
                f"[{self.request_id}] Exiting context with {len(active)} "
                f"active timer(s): {active}. Clearing them."
            )
            self.clear_active_timers()

        # No suprimir excepciones
        return False
