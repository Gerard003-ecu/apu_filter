"""
Este módulo implementa el "Vector de Estado" del sistema. Actúa como un pasaporte
thread-safe que viaja con cada solicitud, acumulando métricas, trazas y contexto
a través de todas las capas de la arquitectura (Pirámide de Observabilidad).

Capacidades y Mecanismos:
-------------------------
1. Pasaporte de Ejecución (TelemetryContext):
   Contenedor unificado que transporta la identidad del request, métricas de
   rendimiento y el stack de ejecución, garantizando observabilidad end-to-end
   sin acoplamiento global.

2. Integración de Física de Datos (RLC):
   Soporta la recolección de métricas del `FluxPhysicsEngine` (Energía Cinética,
   Voltaje Flyback, Potencia Disipada), permitiendo al Agente diagnosticar
   "Fricción" y "Presión" en el flujo de datos.

3. Jerarquía de Spans (Observabilidad Fractal):
   Implementa un sistema de `TelemetrySpan` anidados que permite un análisis
   granular (micro) y sistémico (macro) del rendimiento, facilitando la
   detección de cuellos de botella.

4. Protocolos de Seguridad y Límites:
   Incluye mecanismos de autoprotección (Circuit Breakers lógicos) que limitan
   la cantidad de trazas y errores almacenados para prevenir fugas de memoria
   o ataques de denegación de servicio por telemetría.
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator

from app.schemas import Stratum

logger = logging.getLogger(__name__)


# ========== Constantes de Configuración ==========

class StratumTopology:
    """
    Define la topología de la pirámide DIKW.

    Orden: PHYSICS (base) → TACTICS → STRATEGY → WISDOM (cima)

    Interpretación Algebraica:
    - Filtración ascendente: F₀ ⊂ F₁ ⊂ F₂ ⊂ F₃
    - Morfismo de inclusión induce mapas en cohomología
    - Errores propagan hacia arriba (functor covariante)
    """

    # Nivel 0 = base de la pirámide (más concreto/observable)
    HIERARCHY: Dict[Stratum, int] = {
        Stratum.PHYSICS: 0,
        Stratum.TACTICS: 1,
        Stratum.STRATEGY: 2,
        Stratum.WISDOM: 3,
    }

    # Prefijos de métricas por estrato
    METRIC_PREFIXES: Dict[Stratum, Tuple[str, ...]] = {
        Stratum.PHYSICS: ("flux_condenser", "rlc", "energy", "kinetic"),
        Stratum.TACTICS: ("topology", "betti", "parsing", "homology"),
        Stratum.STRATEGY: ("financial", "npv", "wacc", "roi", "risk"),
        Stratum.WISDOM: ("semantic", "narrative", "insight", "recommendation"),
    }

    @classmethod
    def get_level(cls, stratum: Stratum) -> int:
        """Obtiene el nivel jerárquico de un estrato."""
        return cls.HIERARCHY.get(stratum, 0)

    @classmethod
    def get_higher_strata(cls, stratum: Stratum) -> List[Stratum]:
        """Retorna estratos superiores al dado (para propagación de errores)."""
        current_level = cls.get_level(stratum)
        return [s for s, level in cls.HIERARCHY.items() if level > current_level]

    @classmethod
    def get_prefixes(cls, stratum: Stratum) -> Tuple[str, ...]:
        """Obtiene los prefijos de métricas para un estrato."""
        return cls.METRIC_PREFIXES.get(stratum, ())


class TelemetryDefaults:
    """Constantes de configuración por defecto para telemetría."""

    # Límites de almacenamiento
    MAX_STEPS: int = 1000
    MAX_ERRORS: int = 100
    MAX_METRICS: int = 500
    MAX_EVENTS: int = 1000
    MAX_ACTIVE_STEPS: int = 50
    MAX_SPANS_PER_TREE: int = 500

    # Límites de longitud de strings
    MAX_STRING_LENGTH: int = 10000
    MAX_MESSAGE_LENGTH: int = 1000
    MAX_EXCEPTION_DETAIL_LENGTH: int = 500
    MAX_NAME_LENGTH: int = 100
    MAX_STEP_NAME_LENGTH: int = 255
    MAX_REQUEST_ID_LENGTH: int = 256
    MAX_TRACEBACK_LENGTH: int = 5000

    # Límites de recursión y colecciones
    MAX_RECURSION_DEPTH: int = 5
    MAX_COLLECTION_SIZE: int = 100
    MAX_DICT_KEYS: int = 200
    MAX_LIMIT_MULTIPLIER: int = 10

    # Umbrales temporales
    MAX_STEP_DURATION_WARNING: float = 300.0  # 5 minutos
    STALE_STEP_THRESHOLD: float = 3600.0  # 1 hora
    SPAN_TIMEOUT_WARNING: float = 60.0  # 1 minuto

    # Límites numéricos
    MAX_METRIC_VALUE: float = 1e15
    MIN_METRIC_VALUE: float = -1e15
    EPSILON: float = 1e-10  # Para comparaciones de punto flotante


class BusinessThresholds:
    """Umbrales configurables para el informe de negocio."""

    CRITICAL_FLYBACK_VOLTAGE: float = 0.5
    CRITICAL_DISSIPATED_POWER: float = 50.0
    WARNING_SATURATION: float = 0.9
    CRITICAL_ERROR_COUNT: int = 10
    WARNING_STEP_FAILURE_RATIO: float = 0.3


@dataclass
class TelemetryHealth:
    """
    Estado de salud del contexto de telemetría.

    Implementa un semi-lattice donde:
    - HEALTHY ⊓ WARNING = WARNING
    - WARNING ⊓ CRITICAL = CRITICAL
    - El operador ⊓ (meet) preserva el peor estado
    """

    is_healthy: bool = True
    warnings: List[Tuple[str, float]] = field(default_factory=list)  # (msg, timestamp)
    errors: List[Tuple[str, float]] = field(default_factory=list)
    stale_steps: List[str] = field(default_factory=list)
    memory_pressure: bool = False
    _created_at: float = field(default_factory=time.perf_counter)

    def add_warning(self, msg: str) -> None:
        """Agrega una advertencia con timestamp."""
        self.warnings.append((msg, time.perf_counter()))

    def add_error(self, msg: str) -> None:
        """Agrega un error y marca como no saludable."""
        self.errors.append((msg, time.perf_counter()))
        self.is_healthy = False

    def get_severity_level(self) -> int:
        """
        Retorna nivel de severidad numérico.
        0 = HEALTHY, 1 = WARNING, 2 = CRITICAL
        """
        if not self.is_healthy or len(self.errors) > 0:
            return 2
        if self.warnings or self.memory_pressure or self.stale_steps:
            return 1
        return 0

    def get_status_string(self) -> str:
        """Retorna estado como string."""
        level = self.get_severity_level()
        return {0: "HEALTHY", 1: "WARNING", 2: "CRITICAL"}[level]

    def merge_with(self, other: "TelemetryHealth") -> "TelemetryHealth":
        """
        Fusiona con otro estado de salud (operación meet del lattice).
        Útil para agregar salud de sub-componentes.
        """
        merged = TelemetryHealth(
            is_healthy=self.is_healthy and other.is_healthy,
            warnings=self.warnings + other.warnings,
            errors=self.errors + other.errors,
            stale_steps=self.stale_steps + other.stale_steps,
            memory_pressure=self.memory_pressure or other.memory_pressure,
        )
        return merged

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "status": self.get_status_string(),
            "is_healthy": self.is_healthy,
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
            "warnings": [msg for msg, _ in self.warnings[-10:]],  # Últimas 10
            "errors": [msg for msg, _ in self.errors[-10:]],
            "stale_steps": self.stale_steps[:5],
            "memory_pressure": self.memory_pressure,
            "age_seconds": round(time.perf_counter() - self._created_at, 3),
        }


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
    stratum: Stratum = Stratum.PHYSICS

    def get_duration(self) -> float:
        """Calcula la duración actual del paso."""
        return time.perf_counter() - self.start_time


@dataclass
class TelemetrySpan:
    """
    Representa un nodo en la jerarquía de ejecución (Pirámide de Observabilidad).

    Invariantes del árbol:
    - Cada span tiene exactamente un padre (excepto roots)
    - No hay ciclos
    - level == profundidad desde la raíz
    """

    name: str
    level: int
    stratum: Stratum = field(default=Stratum.PHYSICS)
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    children: List["TelemetrySpan"] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.IN_PROGRESS
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Nuevos campos para tracking
    _node_count: int = field(default=1, repr=False)
    _max_child_depth: int = field(default=0, repr=False)

    @property
    def duration(self) -> float:
        """Calcula la duración del span."""
        end = self.end_time if self.end_time else time.perf_counter()
        return max(0.0, end - self.start_time)

    @property
    def is_complete(self) -> bool:
        """Indica si el span ha finalizado."""
        return self.end_time is not None

    @property
    def depth(self) -> int:
        """Profundidad máxima del subárbol."""
        return self._max_child_depth

    @property
    def subtree_size(self) -> int:
        """Número total de nodos en el subárbol (incluyendo este)."""
        return self._node_count

    def add_child(self, child: "TelemetrySpan") -> bool:
        """
        Agrega un hijo actualizando métricas del árbol.
        Retorna False si excede límites.
        """
        if len(self.children) >= TelemetryDefaults.MAX_COLLECTION_SIZE:
            return False

        # Validar nivel del hijo
        if child.level != self.level + 1:
            child.level = self.level + 1

        self.children.append(child)
        self._update_tree_metrics()
        return True

    def _update_tree_metrics(self) -> None:
        """Recalcula métricas del árbol de forma eficiente."""
        if not self.children:
            self._node_count = 1
            self._max_child_depth = 0
        else:
            self._node_count = 1 + sum(c._node_count for c in self.children)
            self._max_child_depth = 1 + max(c._max_child_depth for c in self.children)

    def finalize(self, status: Optional[StepStatus] = None) -> None:
        """Finaliza el span con el estado dado."""
        self.end_time = time.perf_counter()
        if status is not None:
            self.status = status
        elif self.status == StepStatus.IN_PROGRESS:
            self.status = StepStatus.SUCCESS

    def get_failed_children(self) -> List["TelemetrySpan"]:
        """Retorna lista de hijos con status de fallo."""
        failed = []
        for child in self.children:
            if child.status == StepStatus.FAILURE:
                failed.append(child)
            failed.extend(child.get_failed_children())
        return failed

    def to_dict(self, include_children: bool = True, max_depth: int = 10) -> Dict[str, Any]:
        """Serializa el span a diccionario con control de profundidad."""
        result = {
            "name": self.name,
            "level": self.level,
            "stratum": self.stratum.name,
            "duration_seconds": round(self.duration, 6),
            "status": self.status.value,
            "is_complete": self.is_complete,
            "subtree_size": self._node_count,
            "max_depth": self._max_child_depth,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
            "error_count": len(self.errors),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if self.errors:
            result["errors"] = self.errors[:5]  # Limitar errores serializados

        if include_children and max_depth > 0:
            result["children"] = [
                c.to_dict(include_children=True, max_depth=max_depth - 1)
                for c in self.children
            ]
        elif self.children:
            result["children_count"] = len(self.children)

        return result


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
    events: List[Dict[str, Any]] = field(default_factory=list)

    root_spans: List[TelemetrySpan] = field(default_factory=list)
    _scope_stack: List[TelemetrySpan] = field(default_factory=list)

    # Nuevo: Rastreo de salud por estrato
    _strata_health: Dict[Stratum, TelemetryHealth] = field(default_factory=dict)

    _active_steps: Dict[str, ActiveStepInfo] = field(default_factory=dict)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    max_steps: int = field(default=TelemetryDefaults.MAX_STEPS)
    max_errors: int = field(default=TelemetryDefaults.MAX_ERRORS)
    max_metrics: int = field(default=TelemetryDefaults.MAX_METRICS)
    max_events: int = field(default=TelemetryDefaults.MAX_EVENTS)

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

        # Inicializar salud por estrato
        for s in Stratum:
            self._strata_health[s] = TelemetryHealth()

        if not hasattr(self, "_active_steps") or self._active_steps is None:
            object.__setattr__(self, "_active_steps", {})

        if not isinstance(self.created_at, (int, float)) or self.created_at <= 0:
            object.__setattr__(self, "created_at", time.perf_counter())
            logger.warning(f"[{self.request_id}] Invalid created_at, reset to current time")

        logger.debug(
            f"[{self.request_id}] TelemetryContext initialized: "
            f"max_steps={self.max_steps}, max_errors={self.max_errors}, "
            f"max_metrics={self.max_metrics}, max_events={self.max_events}"
        )

    # =========================================================================
    # CORE: Spans & Steps
    # =========================================================================

    def start_step(
        self,
        step_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        stratum: Stratum = Stratum.PHYSICS,
    ) -> bool:
        """
        Marca el inicio de un paso de procesamiento.

        Retorna True si el paso se inició correctamente.
        Maneja pasos duplicados y límites de forma segura.
        """
        # Validación temprana fuera del lock
        if not self._validate_step_name(step_name):
            return False

        sanitized_metadata = self._prepare_metadata(metadata, step_name)
        current_time = time.perf_counter()

        with self._lock:
            # Verificar límite de pasos activos
            if len(self._active_steps) >= TelemetryDefaults.MAX_ACTIVE_STEPS:
                cleaned = self._cleanup_stale_steps()
                if cleaned == 0 and len(self._active_steps) >= TelemetryDefaults.MAX_ACTIVE_STEPS:
                    logger.error(
                        f"[{self.request_id}] Cannot start '{step_name}': "
                        f"max active steps ({TelemetryDefaults.MAX_ACTIVE_STEPS}) reached"
                    )
                    self._record_dropped_step(step_name, "max_active_limit")
                    return False

            # Manejar paso duplicado
            existing = self._active_steps.get(step_name)
            if existing:
                existing_duration = current_time - existing.start_time
                self._handle_duplicate_step(step_name, existing, existing_duration)

            # Registrar nuevo paso activo
            self._active_steps[step_name] = ActiveStepInfo(
                start_time=current_time,
                metadata=sanitized_metadata,
                stratum=stratum,
            )

            # Actualizar salud del estrato
            self._mark_stratum_active(stratum)

        logger.info(f"[{self.request_id}] ▶ Step started: {step_name} [{stratum.name}]")
        return True

    def end_step(
        self,
        step_name: str,
        status: Union[StepStatus, str] = StepStatus.SUCCESS,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Marca el final de un paso de procesamiento.

        Maneja graciosamente pasos nunca iniciados y combina metadata.
        """
        if not self._validate_step_name(step_name):
            return False

        normalized_status = self._normalize_status(status)
        end_time = time.perf_counter()

        with self._lock:
            step_info = self._active_steps.pop(step_name, None)
            step_data = self._build_step_record(
                step_name, step_info, normalized_status, end_time, metadata
            )

            # Aplicar límite FIFO
            self._enforce_limit_fifo(
                self.steps,
                self.max_steps,
                "steps",
                lambda s: s.get("step", "unknown"),
            )

            self.steps.append(step_data)

            # Actualizar salud del estrato si hubo fallo
            if normalized_status == StepStatus.FAILURE.value:
                stratum = step_info.stratum if step_info else Stratum.PHYSICS
                self._record_stratum_failure(stratum, step_name)

        # Logging fuera del lock
        duration = step_data.get("duration_seconds", 0)
        log_level = logging.INFO if normalized_status == StepStatus.SUCCESS.value else logging.WARNING
        logger.log(
            log_level,
            f"[{self.request_id}] ■ Step finished: {step_name} ({normalized_status}) "
            f"in {duration:.6f}s"
        )

        return True

    @contextmanager
    def step(
        self,
        step_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        error_status: StepStatus = StepStatus.FAILURE,
        capture_exception_details: bool = True,
        auto_record_error: bool = True,
        suppress_start_failure: bool = True,
        stratum: Stratum = Stratum.PHYSICS,
        create_span: bool = False,
    ):
        """
        Gestor de contexto para el seguimiento automático de pasos.

        Garantías:
        - El paso siempre se finaliza (incluso con excepciones)
        - Los errores se registran automáticamente si auto_record_error=True
        - Thread-safe para uso concurrente
        """
        # Validación temprana
        if not isinstance(step_name, str) or not step_name.strip():
            logger.error(f"[{self.request_id}] Invalid step_name for context manager")
            if not suppress_start_failure:
                raise ValueError("step_name must be a non-empty string")
            yield self
            return

        # Normalizar error_status
        if not isinstance(error_status, StepStatus):
            error_status = (
                StepStatus.from_string(error_status)
                if isinstance(error_status, str)
                else StepStatus.FAILURE
            )

        # Intentar iniciar el paso
        started = self._safe_start_step(step_name, metadata, stratum, suppress_start_failure)

        if not started:
            if not suppress_start_failure:
                raise RuntimeError(f"Failed to start step: {step_name}")
            logger.warning(
                f"[{self.request_id}] Step '{step_name}' failed to start, "
                "proceeding without telemetry"
            )
            yield self
            return

        # Crear span opcional
        span_context = None
        if create_span:
            span_context = self.span(step_name, metadata, stratum)
            span_context.__enter__()

        exception_occurred = False
        captured_exception: Optional[BaseException] = None

        try:
            yield self
        except BaseException as e:
            exception_occurred = True
            captured_exception = e
            raise
        finally:
            # Finalizar span si fue creado
            if span_context is not None:
                try:
                    span_context.__exit__(
                        type(captured_exception) if captured_exception else None,
                        captured_exception,
                        captured_exception.__traceback__ if captured_exception else None,
                    )
                except Exception as span_error:
                    logger.error(f"[{self.request_id}] Error closing span: {span_error}")

            # Siempre finalizar el paso
            self._finalize_step(
                step_name=step_name,
                exception_occurred=exception_occurred,
                captured_exception=captured_exception,
                error_status=error_status,
                capture_exception_details=capture_exception_details,
                auto_record_error=auto_record_error,
            )

    @contextmanager
    def span(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        stratum: Stratum = Stratum.PHYSICS,
        timeout_warning: Optional[float] = None,
    ):
        """
        Crea un nuevo span jerárquico con tracking de integridad.

        Los spans forman un bosque (forest) donde:
        - Cada span tiene a lo sumo un padre
        - El nivel indica profundidad desde la raíz
        - χ(forest) = número de árboles (roots)
        """
        timeout_warning = timeout_warning or TelemetryDefaults.SPAN_TIMEOUT_WARNING

        with self._lock:
            level = len(self._scope_stack)

            # Verificar profundidad máxima
            if level >= TelemetryDefaults.MAX_RECURSION_DEPTH:
                logger.warning(
                    f"[{self.request_id}] Max span depth ({level}) reached for '{name}'. "
                    "Span will be created but not nested."
                )
                level = TelemetryDefaults.MAX_RECURSION_DEPTH

            new_span = TelemetrySpan(
                name=name,
                level=level,
                stratum=stratum,
                metadata=self._sanitize_value(metadata) if metadata else {},
            )

            # Enlazar con padre si existe
            if self._scope_stack:
                parent = self._scope_stack[-1]
                if not parent.add_child(new_span):
                    logger.warning(
                        f"[{self.request_id}] Parent span has too many children. "
                        f"'{name}' attached as root instead."
                    )
                    self.root_spans.append(new_span)
            else:
                self.root_spans.append(new_span)

            self._scope_stack.append(new_span)
            stack_depth_at_entry = len(self._scope_stack)

        logger.debug(f"[{self.request_id}] SPAN ▶ {'  ' * level}{name} [{stratum.name}]")
        span_start = time.perf_counter()

        try:
            yield new_span
            if new_span.status == StepStatus.IN_PROGRESS:
                new_span.status = StepStatus.SUCCESS
        except Exception as e:
            new_span.status = StepStatus.FAILURE
            self._record_span_error(new_span, e)
            raise
        finally:
            new_span.finalize()

            # Sincronizar salud del estrato con el estado del span
            if new_span.status == StepStatus.FAILURE:
                with self._lock:
                    self._record_stratum_failure(stratum, name)

            duration = new_span.duration

            # Advertencia de timeout
            if duration > timeout_warning:
                logger.warning(
                    f"[{self.request_id}] Span '{name}' took {duration:.2f}s "
                    f"(threshold: {timeout_warning}s)"
                )

            # Limpiar stack con validación de integridad
            with self._lock:
                self._safe_pop_span(new_span, stack_depth_at_entry)

            status_symbol = "✓" if new_span.status == StepStatus.SUCCESS else "✗"
            logger.debug(
                f"[{self.request_id}] SPAN ■ {'  ' * level}{name} {status_symbol} ({duration:.4f}s)"
            )

    # =========================================================================
    # CORE: Metrics & Events
    # =========================================================================

    def record_metric(
        self,
        component: str,
        metric_name: str,
        value: Any,
        overwrite: bool = True,
        validate_numeric: bool = False,
        stratum: Optional[Stratum] = None,
    ) -> bool:
        """
        Registra una métrica específica para un componente.
        """
        if not self._validate_metric_key(component, metric_name):
            return False

        key = f"{component}.{metric_name}"

        # Validación numérica temprana
        if validate_numeric:
            if not self._is_valid_numeric(value):
                logger.warning(
                    f"[{self.request_id}] Metric '{key}' requires numeric value, "
                    f"got {type(value).__name__}"
                )
                return False

        with self._lock:
            # Verificar si es nueva métrica
            is_new = key not in self.metrics

            if not overwrite and not is_new:
                return False

            if is_new and len(self.metrics) >= self.max_metrics:
                logger.error(f"[{self.request_id}] Max metrics ({self.max_metrics}) reached")
                return False

            # Sanitizar y limitar valor
            try:
                sanitized_value = self._sanitize_metric_value(value)
            except Exception as e:
                logger.warning(f"[{self.request_id}] Failed to sanitize metric '{key}': {e}")
                return False

            self.metrics[key] = sanitized_value

            # Asociar al span activo si existe
            if self._scope_stack:
                self._scope_stack[-1].metrics[key] = sanitized_value

        return True

    def increment_metric(
        self,
        component: str,
        metric_name: str,
        increment: Union[int, float] = 1,
        create_if_missing: bool = True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> bool:
        """
        Incrementa una métrica numérica de forma atómica.
        """
        if not self._validate_metric_key(component, metric_name):
            return False

        if not self._is_valid_numeric(increment):
            logger.warning(f"[{self.request_id}] Invalid increment value: {increment}")
            return False

        key = f"{component}.{metric_name}"

        with self._lock:
            current = self.metrics.get(key)

            # Obtener valor base
            if current is None:
                if not create_if_missing:
                    return False
                if len(self.metrics) >= self.max_metrics:
                    return False
                base_value = 0.0
            elif not isinstance(current, (int, float)):
                logger.warning(f"[{self.request_id}] Metric '{key}' is not numeric, resetting to 0")
                base_value = 0.0
            else:
                base_value = float(current)

            # Calcular nuevo valor
            try:
                new_value = base_value + increment
            except OverflowError:
                new_value = (
                    TelemetryDefaults.MAX_METRIC_VALUE if increment > 0
                    else TelemetryDefaults.MIN_METRIC_VALUE
                )

            # Aplicar límites opcionales
            if min_value is not None:
                new_value = max(min_value, new_value)
            if max_value is not None:
                new_value = min(max_value, new_value)

            # Aplicar límites globales
            new_value = self._clamp_numeric(new_value)

            # Verificar NaN (puede ocurrir con operaciones límite)
            if isinstance(new_value, float) and math.isnan(new_value):
                logger.error(f"[{self.request_id}] Metric '{key}' resulted in NaN")
                return False

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
        """Obtiene el valor de una métrica de forma thread-safe."""
        key = f"{component}.{metric_name}"

        with self._lock:
            value = self.metrics.get(key)

            if value is None:
                return default

            if expected_type is not None and not isinstance(value, expected_type):
                return default

            # Copia profunda para valores mutables
            if isinstance(value, (dict, list)):
                return copy.deepcopy(value)

            return value

    def record_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> bool:
        """Registra un evento puntual en la línea de tiempo."""
        if not self._validate_name(name, "event_name"):
            return False

        with self._lock:
            # Aplicar límite FIFO
            self._enforce_limit_fifo(
                self.events, self.max_events, "events", lambda e: e.get("name", "unknown")
            )

            event_data = {
                "name": name,
                "payload": self._sanitize_value(payload) if payload else {},
                "timestamp": datetime.utcnow().isoformat(),
                "perf_counter": time.perf_counter(),
            }

            self.events.append(event_data)
            logger.info(f"[{self.request_id}] EVENT: {name}")
            return True

    # =========================================================================
    # CORE: Errors & Propagation
    # =========================================================================

    def record_error(
        self,
        step_name: str,
        error_message: str,
        error_type: Optional[str] = None,
        exception: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_traceback: bool = False,
        severity: str = "ERROR",
        stratum: Optional[Stratum] = None,
        propagate: bool = True,
    ) -> bool:
        """
        Registra un error ocurrido durante un paso.

        Los errores CRITICAL propagan automáticamente hacia estratos superiores
        siguiendo la topología de la pirámide DIKW.
        """
        # Validación de inputs
        step_name = step_name if self._validate_name(step_name, "step_name") else "__unknown_step__"
        error_message = error_message if self._validate_error_message(error_message) else "Unknown error"
        severity = severity if severity in {"ERROR", "WARNING", "CRITICAL", "INFO"} else "ERROR"

        # Validar stratum
        if stratum is not None and not isinstance(stratum, Stratum):
            logger.warning(f"[{self.request_id}] Invalid stratum type, auto-detecting")
            stratum = None

        with self._lock:
            # Aplicar límite FIFO
            self._enforce_limit_fifo(
                self.errors,
                self.max_errors,
                "errors",
                lambda e: f"{e.get('step', 'unknown')}:{e.get('type', 'unknown')}",
            )

            # Construir datos del error
            error_data = self._build_error_data(
                step_name=step_name,
                error_message=error_message,
                error_type=error_type,
                exception=exception,
                metadata=metadata,
                include_traceback=include_traceback,
                severity=severity,
            )

            self.errors.append(error_data)

            # Asociar al span activo
            if self._scope_stack:
                self._scope_stack[-1].errors.append(error_data)

            # Determinar estrato del error
            current_stratum = self._resolve_error_stratum(stratum, step_name)
            error_data["stratum"] = current_stratum.name

            # Actualizar salud del estrato
            self._update_stratum_health(current_stratum, severity, error_message)

            # Propagación hacia arriba para errores CRITICAL
            if severity == "CRITICAL" and propagate:
                self._propagate_failure_upwards(current_stratum, error_message)

        # Logging fuera del lock
        log_func = {
            "CRITICAL": logger.critical,
            "ERROR": logger.error,
            "WARNING": logger.warning,
            "INFO": logger.info,
        }.get(severity, logger.error)

        truncated_msg = error_message[:200] + ("..." if len(error_message) > 200 else "")
        log_func(f"[{self.request_id}] {severity} in {step_name}: {truncated_msg}")

        return True

    def _resolve_error_stratum(self, explicit: Optional[Stratum], step_name: str) -> Stratum:
        """Resuelve el estrato de un error con prioridad: explícito > activo > default."""
        if explicit is not None:
            return explicit

        active_info = self._active_steps.get(step_name)
        if active_info:
            return active_info.stratum

        return Stratum.PHYSICS

    def _update_stratum_health(self, stratum: Stratum, severity: str, message: str) -> None:
        """Actualiza el estado de salud del estrato según la severidad."""
        if stratum not in self._strata_health:
            self._strata_health[stratum] = TelemetryHealth()

        health = self._strata_health[stratum]

        if severity in ("CRITICAL", "ERROR"):
            health.add_error(message)
        elif severity == "WARNING":
            health.add_warning(message)

    def _propagate_failure_upwards(self, failed_stratum: Stratum, original_message: str) -> None:
        """
        Implementa el Colapso Piramidal.

        Topológicamente: Si X ⊂ Y en la filtración de estratos,
        un fallo en X induce inestabilidad en Y.
        """
        failed_level = StratumTopology.get_level(failed_stratum)
        affected_strata = StratumTopology.get_higher_strata(failed_stratum)

        propagation_message = (
            f"Instability inherited from {failed_stratum.name} (level {failed_level}): "
            f"{original_message[:100]}"
        )

        for target_stratum in affected_strata:
            if target_stratum not in self._strata_health:
                self._strata_health[target_stratum] = TelemetryHealth()

            self._strata_health[target_stratum].add_warning(propagation_message)

            logger.debug(
                f"[{self.request_id}] Propagated failure from {failed_stratum.name} "
                f"to {target_stratum.name}"
            )

    # =========================================================================
    # REPORTS & ANALYTICS
    # =========================================================================

    def _filter_metrics_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """Filtra métricas por prefijo (thread-safe)."""
        with self._lock:
            result = {}
            for key, value in self.metrics.items():
                if key.startswith(prefix):
                    result[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
            return result

    def get_pyramidal_report(self) -> Dict[str, Any]:
        """
        Genera un reporte organizado por la jerarquía DIKW.

        Thread-safe con copias profundas de todos los datos.
        Incluye métricas de salud y topología del sistema.
        """
        with self._lock:
            report = {}

            for stratum in Stratum:
                layer_name = f"{stratum.name.lower()}_layer"
                health = self._strata_health.get(stratum, TelemetryHealth())

                report[layer_name] = {
                    "status": health.get_status_string(),
                    "metrics": self._get_stratum_metrics_unsafe(stratum),
                    "issues": [msg for msg, _ in health.errors],
                    "warnings": [msg for msg, _ in health.warnings],
                    "step_count": self._count_steps_by_stratum(stratum),
                }

            # Agregar métricas de topología del bosque de spans
            report["span_topology"] = self._calculate_span_topology_unsafe()

            # Agregar resumen de propagación de errores
            report["error_propagation"] = self._get_error_propagation_summary()

            return report

    def get_business_report(self) -> Dict[str, Any]:
        """
        Genera un informe amigable para el negocio.
        """
        with self._lock:
            try:
                raw_metrics = self._extract_business_raw_metrics()
                translated = self._translate_to_business_metrics(raw_metrics)
                status, message = self._determine_business_status(raw_metrics)
                step_stats = self._calculate_step_statistics()

                report = {
                    "status": status,
                    "message": message,
                    "metrics": translated,
                    "raw_metrics": raw_metrics,
                    "details": {
                        "total_steps": len(self.steps),
                        "successful_steps": step_stats["success"],
                        "failed_steps": step_stats["failure"],
                        "total_errors": len(self.errors),
                        "has_active_operations": len(self._active_steps) > 0,
                        "total_duration": step_stats["total_duration"],
                        "success_rate": step_stats["success_rate"],
                    },
                    "pyramidal_health": self._get_pyramidal_health_summary(),
                    "recommendations": self._generate_recommendations(raw_metrics, step_stats),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return report

            except Exception as e:
                logger.error(f"[{self.request_id}] Error generating business report: {e}")
                return {
                    "status": "ERROR",
                    "message": f"Failed to generate report: {e}",
                    "timestamp": datetime.utcnow().isoformat(),
                }

    def verify_invariants(self) -> Dict[str, Any]:
        """
        Verifica invariantes algebraicos del sistema de telemetría.
        """
        violations = []
        warnings = []

        with self._lock:
            # Invariante 1: Bosque de spans válido
            topology = self._calculate_span_topology_unsafe()
            if not topology.get("is_valid_forest", False):
                violations.append(
                    f"Span graph is not a valid forest: "
                    f"χ={topology['euler_characteristic']} ≠ β₀={topology['num_trees']}"
                )

            # Invariante 2: Consistencia del stack de scopes
            in_progress_count = sum(
                1 for span in self._iter_all_spans_unsafe()
                if span.status == StepStatus.IN_PROGRESS
            )
            if len(self._scope_stack) != in_progress_count:
                violations.append(
                    f"Scope stack inconsistent: {len(self._scope_stack)} in stack vs "
                    f"{in_progress_count} spans in-progress"
                )

            # Invariante 3: Límites de almacenamiento
            if len(self.steps) > self.max_steps:
                violations.append(f"Steps overflow: {len(self.steps)} > {self.max_steps}")
            if len(self.errors) > self.max_errors:
                violations.append(f"Errors overflow: {len(self.errors)} > {self.max_errors}")
            if len(self.metrics) > self.max_metrics:
                violations.append(f"Metrics overflow: {len(self.metrics)} > {self.max_metrics}")

            # Invariante 4: Duraciones no negativas
            negative_durations = [
                step.get("step") for step in self.steps
                if isinstance(step.get("duration_seconds"), (int, float))
                and step.get("duration_seconds", 0) < 0
            ]
            if negative_durations:
                violations.append(f"Negative durations in steps: {negative_durations[:3]}")

            # Invariante 5: Jerarquía de niveles en spans
            for span in self._iter_all_spans_unsafe():
                for child in span.children:
                    if child.level != span.level + 1:
                        warnings.append(
                            f"Span level mismatch: {span.name}(L{span.level}) -> "
                            f"{child.name}(L{child.level})"
                        )

            # Invariante 6: Coherencia de estratos en pasos activos
            for step_name, info in self._active_steps.items():
                if not isinstance(info.stratum, Stratum):
                    violations.append(f"Invalid stratum type in active step: {step_name}")

            # Advertencia: Pasos activos de larga duración
            current_time = time.perf_counter()
            for step_name, info in self._active_steps.items():
                duration = current_time - info.start_time
                if duration > TelemetryDefaults.STALE_STEP_THRESHOLD:
                    warnings.append(f"Stale step detected: {step_name} ({duration:.0f}s)")

        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "topology": topology,
            "checked_at": datetime.utcnow().isoformat(),
            "counts": {
                "steps": len(self.steps),
                "errors": len(self.errors),
                "metrics": len(self.metrics),
                "active_steps": len(self._active_steps),
                "root_spans": len(self.root_spans),
                "scope_depth": len(self._scope_stack),
            }
        }

    def reset(
        self,
        keep_request_id: bool = True,
        keep_metadata: bool = False,
        verify_before_reset: bool = False,
    ) -> Dict[str, Any]:
        """Restablece el contexto de telemetría."""
        with self._lock:
            # Capturar estado anterior
            pre_reset_summary = {
                "steps_cleared": len(self.steps),
                "errors_cleared": len(self.errors),
                "metrics_cleared": len(self.metrics),
                "events_cleared": len(self.events),
                "active_steps_cleared": len(self._active_steps),
                "spans_cleared": len(self.root_spans),
            }

            # Verificar invariantes si se solicita
            if verify_before_reset:
                invariants = self.verify_invariants()
                pre_reset_summary["invariants"] = invariants

            # Advertir sobre pasos activos
            if self._active_steps:
                active_names = list(self._active_steps.keys())[:5]
                logger.warning(
                    f"[{self.request_id}] Resetting with {len(self._active_steps)} "
                    f"active steps: {active_names}"
                )

            # Preservar según configuración
            old_request_id = self.request_id
            old_metadata = copy.deepcopy(self.metadata) if keep_metadata else None

            # Limpiar colecciones
            self.steps.clear()
            self.metrics.clear()
            self.errors.clear()
            self.events.clear()
            self._active_steps.clear()
            self.root_spans.clear()
            self._scope_stack.clear()
            self.metadata.clear()

            # Reinicializar salud por estrato
            self._strata_health.clear()
            for s in Stratum:
                self._strata_health[s] = TelemetryHealth()

            # Restaurar según configuración
            if not keep_request_id:
                self.request_id = str(uuid.uuid4())
                pre_reset_summary["new_request_id"] = self.request_id

            if keep_metadata and old_metadata:
                self.metadata.update(old_metadata)

            self.created_at = time.perf_counter()

            logger.info(
                f"[{old_request_id}] Context reset. "
                f"Cleared: {pre_reset_summary['steps_cleared']} steps, "
                f"{pre_reset_summary['errors_cleared']} errors"
            )

            return pre_reset_summary

    # =========================================================================
    # UTILS & HELPERS
    # =========================================================================

    def _prepare_metadata(
        self, metadata: Optional[Dict[str, Any]], context: str
    ) -> Optional[Dict[str, Any]]:
        """Prepara y valida metadata de forma segura."""
        if metadata is None:
            return None

        if not isinstance(metadata, dict):
            logger.warning(
                f"[{self.request_id}] metadata for '{context}' is not dict "
                f"(type={type(metadata).__name__}), ignoring"
            )
            return None

        try:
            return self._sanitize_value(metadata)
        except Exception as e:
            logger.warning(f"[{self.request_id}] Failed to sanitize metadata for '{context}': {e}")
            return None

    def _handle_duplicate_step(
        self, step_name: str, existing: ActiveStepInfo, duration: float
    ) -> None:
        """Maneja un paso que se intenta iniciar cuando ya está activo."""
        if duration > TelemetryDefaults.MAX_STEP_DURATION_WARNING:
            logger.warning(
                f"[{self.request_id}] Step '{step_name}' was active for {duration:.1f}s "
                "(likely stuck). Force-ending previous instance."
            )
            # Registrar el paso anterior como fallido
            self._force_end_step(
                step_name,
                StepStatus.FAILURE,
                {
                    "reason": "superseded_by_restart",
                    "duration_before_restart": round(duration, 6),
                },
            )
        else:
            logger.debug(
                f"[{self.request_id}] Step '{step_name}' restarted after {duration:.4f}s"
            )

    def _record_dropped_step(self, step_name: str, reason: str) -> None:
        """Registra un paso que fue descartado (no pudo iniciarse)."""
        self.record_event(
            "step_dropped",
            {"step_name": step_name, "reason": reason}
        )

    def _mark_stratum_active(self, stratum: Stratum) -> None:
        """Marca un estrato como activo (tiene operaciones en curso)."""
        pass

    def _build_step_record(
        self,
        step_name: str,
        step_info: Optional[ActiveStepInfo],
        status: str,
        end_time: float,
        end_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Construye el registro de un paso completado."""
        if step_info is None:
            duration = 0.0
            combined_metadata = self._prepare_metadata(end_metadata, step_name) or {}
            combined_metadata["warning"] = "step_never_started"
            stratum = Stratum.PHYSICS
        else:
            duration = max(0.0, end_time - step_info.start_time)
            combined_metadata = self._merge_metadata(step_info.metadata, end_metadata)
            stratum = step_info.stratum

            if duration > TelemetryDefaults.MAX_STEP_DURATION_WARNING:
                logger.warning(
                    f"[{self.request_id}] Step '{step_name}' took {duration:.1f}s "
                    f"(exceeds {TelemetryDefaults.MAX_STEP_DURATION_WARNING}s threshold)"
                )

        record = {
            "step": step_name,
            "status": status,
            "stratum": stratum.name,
            "duration_seconds": round(duration, 6),
            "timestamp": datetime.utcnow().isoformat(),
            "perf_counter": end_time,
        }

        if combined_metadata:
            record["metadata"] = combined_metadata

        return record

    def _record_stratum_failure(self, stratum: Stratum, step_name: str) -> None:
        """Registra un fallo en el estrato correspondiente."""
        if stratum in self._strata_health:
            self._strata_health[stratum].add_error(f"Step failed: {step_name}")

    def _safe_start_step(
        self,
        step_name: str,
        metadata: Optional[Dict[str, Any]],
        stratum: Stratum,
        suppress_failure: bool,
    ) -> bool:
        """Inicia un paso de forma segura, capturando excepciones."""
        try:
            return self.start_step(step_name, metadata, stratum=stratum)
        except Exception as e:
            logger.error(f"[{self.request_id}] Exception in start_step('{step_name}'): {e}")
            if not suppress_failure:
                raise
            return False

    def _finalize_step(
        self,
        step_name: str,
        exception_occurred: bool,
        captured_exception: Optional[BaseException],
        error_status: StepStatus,
        capture_exception_details: bool,
        auto_record_error: bool,
    ) -> None:
        """Finaliza un paso de forma segura."""
        final_status = error_status if exception_occurred else StepStatus.SUCCESS

        error_metadata = None
        if exception_occurred and capture_exception_details and captured_exception:
            error_metadata = self._build_exception_metadata(captured_exception)

        try:
            self.end_step(step_name, final_status, error_metadata)
        except Exception as end_error:
            logger.error(f"[{self.request_id}] Exception in end_step('{step_name}'): {end_error}")

        if (
            exception_occurred
            and auto_record_error
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
                logger.error(f"[{self.request_id}] Failed to record error: {record_error}")

    def _record_span_error(self, span: TelemetrySpan, exception: Exception) -> None:
        """Registra un error en el span actual."""
        error_data = self._build_error_data(
            step_name=span.name,
            error_message=str(exception),
            error_type=type(exception).__name__,
            exception=exception,
            metadata=None,
            include_traceback=True,
            severity="ERROR",
        )
        span.errors.append(error_data)

        with self._lock:
            self._enforce_limit_fifo(
                self.errors, self.max_errors, "errors",
                lambda e: f"{e.get('step', 'unknown')}:{e.get('type', 'unknown')}"
            )
            self.errors.append(error_data)

    def _safe_pop_span(self, span: TelemetrySpan, expected_depth: int) -> None:
        """Remueve un span del stack con validación de integridad."""
        if not self._scope_stack:
            logger.error(f"[{self.request_id}] Span stack empty when trying to pop '{span.name}'")
            return

        if self._scope_stack[-1] is span:
            self._scope_stack.pop()
            return

        current_depth = len(self._scope_stack)
        logger.warning(
            f"[{self.request_id}] Span stack inconsistency for '{span.name}': "
            f"expected depth {expected_depth}, current {current_depth}. Recovering..."
        )

        for i in range(len(self._scope_stack) - 1, -1, -1):
            if self._scope_stack[i] is span:
                self._scope_stack.pop(i)
                logger.info(f"[{self.request_id}] Span '{span.name}' removed from position {i}")
                return

        logger.error(f"[{self.request_id}] Span '{span.name}' not found in stack during recovery")

    def _validate_metric_key(self, component: str, metric_name: str) -> bool:
        """Valida los componentes de la clave de métrica."""
        if not self._validate_name(component, "component"):
            return False
        if not self._validate_name(metric_name, "metric_name"):
            return False
        return True

    def _is_valid_numeric(self, value: Any) -> bool:
        """Verifica si un valor es numérico válido (no NaN, no Inf)."""
        if not isinstance(value, (int, float)):
            return False
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return False
        return True

    def _sanitize_metric_value(self, value: Any) -> Any:
        """Sanitiza un valor de métrica, aplicando límites numéricos."""
        sanitized = self._sanitize_value(value)

        if isinstance(sanitized, (int, float)):
            return self._clamp_numeric(sanitized)

        return sanitized

    def _clamp_numeric(self, value: Union[int, float]) -> Union[int, float]:
        """Limita un valor numérico al rango permitido."""
        if value > TelemetryDefaults.MAX_METRIC_VALUE:
            return TelemetryDefaults.MAX_METRIC_VALUE
        if value < TelemetryDefaults.MIN_METRIC_VALUE:
            return TelemetryDefaults.MIN_METRIC_VALUE
        return value

    def _get_stratum_metrics_unsafe(self, stratum: Stratum) -> Dict[str, Any]:
        """Obtiene métricas de un estrato (versión sin lock)."""
        prefixes = StratumTopology.get_prefixes(stratum)
        result = {}

        for key, value in self.metrics.items():
            if any(key.startswith(prefix) for prefix in prefixes):
                result[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

        return result

    def _count_steps_by_stratum(self, stratum: Stratum) -> int:
        """Cuenta pasos completados en un estrato específico."""
        return sum(
            1 for step in self.steps
            if step.get("stratum") == stratum.name
        )

    def _calculate_span_topology_unsafe(self) -> Dict[str, Any]:
        """
        Calcula invariantes topológicos del bosque de spans.
        """
        total_nodes = 0
        total_edges = 0
        max_depth = 0

        for root in self.root_spans:
            total_nodes += root.subtree_size
            total_edges += root.subtree_size - 1
            max_depth = max(max_depth, root.depth + 1)

        num_trees = len(self.root_spans)
        euler_char = total_nodes - total_edges

        return {
            "num_trees": num_trees,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "max_depth": max_depth,
            "euler_characteristic": euler_char,
            "is_valid_forest": euler_char == num_trees,
            "avg_tree_size": round(total_nodes / max(num_trees, 1), 2),
        }

    def _get_error_propagation_summary(self) -> Dict[str, Any]:
        """Resume la propagación de errores entre estratos."""
        summary = {}

        for stratum in Stratum:
            health = self._strata_health.get(stratum, TelemetryHealth())

            inherited_count = sum(
                1 for msg, _ in health.warnings
                if "inherited" in msg.lower() or "instability" in msg.lower()
            )

            summary[stratum.name] = {
                "direct_errors": len(health.errors),
                "inherited_warnings": inherited_count,
                "total_issues": len(health.errors) + len(health.warnings),
            }

        return summary

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

    def _get_pyramidal_health_summary(self) -> Dict[str, str]:
        """Obtiene resumen de salud por estrato."""
        return {
            stratum.name: self._strata_health.get(stratum, TelemetryHealth()).get_status_string()
            for stratum in Stratum
        }

    def _generate_recommendations(
        self,
        raw_metrics: Dict[str, float],
        step_stats: Dict[str, Any]
    ) -> List[str]:
        """Genera recomendaciones basadas en el estado actual."""
        recommendations = []

        # Basadas en saturación
        if raw_metrics.get("saturation", 0) > 0.8:
            recommendations.append(
                "Alta saturación del sistema. Considere escalar recursos o reducir carga."
            )

        # Basadas en voltaje flyback
        if raw_metrics.get("flyback_voltage", 0) > 0.3:
            recommendations.append(
                "Inestabilidad detectada en el flujo de datos. Revise fuentes de entrada."
            )

        # Basadas en tasa de fallos
        if step_stats.get("failure_rate", 0) > 0.2:
            recommendations.append(
                f"Tasa de fallos elevada ({step_stats['failure_rate']:.1%}). "
                "Revise logs de errores para identificar patrones."
            )

        # Basadas en salud de estratos
        for stratum in Stratum:
            health = self._strata_health.get(stratum, TelemetryHealth())
            if not health.is_healthy:
                recommendations.append(
                    f"Estrato {stratum.name} en estado crítico. "
                    f"Errores: {len(health.errors)}"
                )

        if not recommendations:
            recommendations.append("Sistema operando dentro de parámetros normales.")

        return recommendations[:5]

    def _iter_all_spans_unsafe(self) -> Iterator[TelemetrySpan]:
        """
        Iterador DFS sobre todos los spans (sin lock, caller debe tener lock).
        """
        def traverse(span: TelemetrySpan) -> Iterator[TelemetrySpan]:
            yield span
            for child in span.children:
                yield from traverse(child)

        for root in self.root_spans:
            yield from traverse(root)

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
            (
                "max_events",
                1,
                TelemetryDefaults.MAX_EVENTS * max_multiplier,
                TelemetryDefaults.MAX_EVENTS,
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
        except (ValueError, TypeError, OverflowError):
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
            ("events", list, []),
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
                {
                    "reason": "stale_cleanup",
                    "duration_before_cleanup": duration,
                },
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

    def _build_exception_metadata(self, exc: BaseException) -> Dict[str, Any]:
        """Construye metadata de excepción de forma segura."""
        try:
            return {
                "error_type": type(exc).__name__,
                "error_message": str(exc)[: TelemetryDefaults.MAX_EXCEPTION_DETAIL_LENGTH],
                "is_base_exception": not isinstance(exc, Exception),
            }
        except Exception:
            return {
                "error_type": "unknown",
                "error_message": "failed to capture",
            }

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

    def _validate_error_message(self, error_message: Any) -> bool:
        """Valida que el mensaje de error sea válido."""
        if error_message is None:
            return False

        if not isinstance(error_message, str):
            return False

        if not error_message.strip():
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
                    "total_events": len(self.events),
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
                    events_data = copy.deepcopy(self.events)
                else:
                    steps_data = list(self.steps)
                    metrics_data = dict(self.metrics)
                    errors_data = list(self.errors)
                    events_data = list(self.events)

                result = {
                    "request_id": self.request_id,
                    "steps": steps_data,
                    "metrics": metrics_data,
                    "errors": errors_data,
                    "events": events_data,
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
                f"events={len(self.events)}, "
                f"active={len(self._active_steps)})"
            )

    def __str__(self) -> str:
        """Representación en cadena del resumen."""
        summary = self.get_summary()
        return (
            f"Telemetry[{summary['request_id'][:8]}...]: "
            f"{summary['total_steps']} steps, "
            f"{summary['total_errors']} errors, "
            f"{summary['total_events']} events, "
            f"{summary['total_duration_seconds']:.3f}s"
        )
