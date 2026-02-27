"""
Módulo: Autonomous Agent (El Coordinador del Ciclo OODA)
========================================================

Este componente actúa como la "Conciencia Operativa" del sistema APU Filter.
A diferencia de los componentes pasivos, el Agente es una entidad activa que
gobierna el ciclo de vida del procesamiento mediante un bucle de control
continuo OODA (Observar, Orientar, Decidir, Actuar).

Su objetivo primario es mantener la **Homeostasis del Sistema**: asegurar que
el flujo de datos sea laminar y que la infraestructura no colapse bajo estrés
térmico o topológico.

Arquitectura Cognitiva (El Ciclo OODA):
---------------------------------------

1. OBSERVE (Observar - La Percepción Física):
   Morfismo O: Infraestructura → Telemetría
   Recolecta telemetría cruda del FluxPhysicsEngine en tiempo real.

2. ORIENT (Orientar - El Mapa Topológico):
   Morfismo R: Telemetría × Homología → Estado
   Utiliza Homología Persistente para distinguir ruido de características estructurales.

3. DECIDE (Decidir - La Matriz de Juicio):
   Morfismo D: Estado × Contexto → Acción
   Evalúa la situación mediante una cadena de evaluadores ordenados por criticidad.

4. ACT (Actuar - La Ejecución de Vectores):
   Morfismo A: Acción × Diagnóstico → Efectos
   Proyecta intenciones sobre la Matriz de Interacción Central (MIC).

Invariantes del Sistema:
------------------------
- Conectividad: β₀ = 1 (sistema conectado)
- Estabilidad: health_score ≥ 0.4 (umbral crítico)
- Latencia: cycle_time < check_interval
- Disponibilidad: consecutive_failures < max_failures
"""

from __future__ import annotations

import logging
import math
import os
import signal
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum, auto
from functools import cached_property
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.schemas import Stratum
from agent.topological_analyzer import (
    HealthLevel,
    MetricState,
    PersistenceAnalysisResult,
    PersistenceHomology,
    SystemTopology,
    TopologicalHealth,
)


# ============================================================================
# CONSTANTES GLOBALES
# ============================================================================

# Tolerancia numérica
EPSILON: Final[float] = 1e-9

# Valores por defecto de configuración
DEFAULT_CORE_URL: Final[str] = "http://localhost:5002"
DEFAULT_CHECK_INTERVAL: Final[int] = 10
DEFAULT_REQUEST_TIMEOUT: Final[int] = 10
DEFAULT_MAX_FAILURES: Final[int] = 5
DEFAULT_DEBOUNCE_SECONDS: Final[int] = 60
DEFAULT_PERSISTENCE_WINDOW: Final[int] = 20
DEFAULT_TOPOLOGY_HISTORY: Final[int] = 100

# Nodos de la topología esperada
NODE_AGENT: Final[str] = "Agent"
NODE_CORE: Final[str] = "Core"
NODE_REDIS: Final[str] = "Redis"
NODE_FILESYSTEM: Final[str] = "Filesystem"


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================


def setup_logging(name: str = "AutonomousAgent") -> logging.Logger:
    """
    Configura y retorna el logger del agente.
    
    Args:
        name: Nombre del logger.
        
    Returns:
        Logger configurado.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger(name)


logger = setup_logging()


# ============================================================================
# EXCEPCIONES DEL DOMINIO
# ============================================================================


class AgentError(Exception):
    """Excepción base del agente autónomo."""
    pass


class ConfigurationError(AgentError):
    """Error de configuración del agente."""
    pass


class ObservationError(AgentError):
    """Error durante la fase de observación."""
    
    def __init__(self, message: str, error_type: str, request_id: Optional[str] = None):
        self.error_type = error_type
        self.request_id = request_id
        super().__init__(message)


class TopologyError(AgentError):
    """Error relacionado con la topología del sistema."""
    pass


class TelemetryValidationError(AgentError):
    """Error en validación de datos de telemetría."""
    pass


class ConnectionHealthError(AgentError):
    """Error de salud de conexión."""
    pass


# ============================================================================
# ENUMERACIONES
# ============================================================================


class SystemStatus(Enum):
    """
    Estados posibles del sistema monitoreado.
    
    Forma un orden parcial por severidad:
    NOMINAL < UNKNOWN < INESTABLE < SATURADO < CRITICO < DISCONNECTED
    """

    NOMINAL = auto()
    UNKNOWN = auto()
    INESTABLE = auto()
    SATURADO = auto()
    CRITICO = auto()
    DISCONNECTED = auto()

    @property
    def severity(self) -> int:
        """Severidad numérica del estado (mayor = peor)."""
        severity_map = {
            SystemStatus.NOMINAL: 0,
            SystemStatus.UNKNOWN: 1,
            SystemStatus.INESTABLE: 2,
            SystemStatus.SATURADO: 3,
            SystemStatus.CRITICO: 4,
            SystemStatus.DISCONNECTED: 5,
        }
        return severity_map[self]

    @property
    def is_healthy(self) -> bool:
        """Indica si el sistema está sano."""
        return self == SystemStatus.NOMINAL

    @property
    def is_critical(self) -> bool:
        """Indica si requiere atención inmediata."""
        return self in (SystemStatus.CRITICO, SystemStatus.DISCONNECTED)

    @property
    def emoji(self) -> str:
        """Representación visual del estado."""
        return {
            SystemStatus.NOMINAL: "✅",
            SystemStatus.UNKNOWN: "❓",
            SystemStatus.INESTABLE: "⚠️",
            SystemStatus.SATURADO: "🔶",
            SystemStatus.CRITICO: "🚨",
            SystemStatus.DISCONNECTED: "💀",
        }[self]

    def __lt__(self, other: SystemStatus) -> bool:
        """Comparación por severidad."""
        return self.severity < other.severity

    def __le__(self, other: SystemStatus) -> bool:
        return self.severity <= other.severity

    @classmethod
    def worst(cls, *statuses: SystemStatus) -> SystemStatus:
        """Retorna el peor estado (máxima severidad)."""
        if not statuses:
            return cls.NOMINAL
        return max(statuses, key=lambda s: s.severity)


class AgentDecision(Enum):
    """
    Decisiones que el agente puede tomar.
    
    Cada decisión mapea a una acción específica sobre el sistema.
    """

    HEARTBEAT = auto()          # Sistema nominal, solo señal de vida
    EJECUTAR_LIMPIEZA = auto()  # Iniciar proceso de limpieza/estabilización
    AJUSTAR_VELOCIDAD = auto()  # Aplicar backpressure
    ALERTA_CRITICA = auto()     # Emitir alerta crítica
    WAIT = auto()               # Esperar más información
    RECONNECT = auto()          # Intentar reconexión

    @property
    def requires_immediate_action(self) -> bool:
        """Indica si la decisión no debe ser debounceada."""
        return self in (AgentDecision.ALERTA_CRITICA, AgentDecision.RECONNECT)

    @property
    def emoji(self) -> str:
        """Representación visual de la decisión."""
        return {
            AgentDecision.HEARTBEAT: "💓",
            AgentDecision.EJECUTAR_LIMPIEZA: "🧹",
            AgentDecision.AJUSTAR_VELOCIDAD: "🔽",
            AgentDecision.ALERTA_CRITICA: "🚨",
            AgentDecision.WAIT: "⏳",
            AgentDecision.RECONNECT: "🔄",
        }[self]


class ObservationErrorType(Enum):
    """Tipos de errores de observación."""
    
    TIMEOUT = "TIMEOUT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    INVALID_JSON = "INVALID_JSON"
    INVALID_TELEMETRY = "INVALID_TELEMETRY"
    REQUEST_ERROR = "REQUEST_ERROR"
    UNKNOWN = "UNKNOWN"


# ============================================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================================


@dataclass(frozen=True)
class ThresholdConfig:
    """
    Configuración inmutable de umbrales para análisis de telemetría.
    
    Invariantes:
    - 0 ≤ warning < critical ≤ 1.0 para cada par de umbrales
    """

    flyback_voltage_warning: float = 0.5
    flyback_voltage_critical: float = 0.8
    saturation_warning: float = 0.9
    saturation_critical: float = 0.95

    def __post_init__(self) -> None:
        """Valida coherencia de umbrales tras inicialización."""
        self._validate_pair(
            "flyback_voltage",
            self.flyback_voltage_warning,
            self.flyback_voltage_critical,
        )
        self._validate_pair(
            "saturation",
            self.saturation_warning,
            self.saturation_critical,
        )

    @staticmethod
    def _validate_pair(name: str, warning: float, critical: float) -> None:
        """Valida que un par de umbrales sea coherente."""
        if not (0.0 <= warning < critical <= 1.0):
            raise ConfigurationError(
                f"{name} thresholds inválidos: "
                f"debe cumplir 0 ≤ warning({warning}) < critical({critical}) ≤ 1.0"
            )

    def classify_voltage(self, value: float) -> str:
        """Clasifica un valor de voltaje."""
        if value >= self.flyback_voltage_critical:
            return "critical"
        if value >= self.flyback_voltage_warning:
            return "warning"
        return "nominal"

    def classify_saturation(self, value: float) -> str:
        """Clasifica un valor de saturación."""
        if value >= self.saturation_critical:
            return "critical"
        if value >= self.saturation_warning:
            return "warning"
        return "nominal"


@dataclass(frozen=True)
class ConnectionConfig:
    """Configuración de conexión al Core."""
    
    base_url: str = DEFAULT_CORE_URL
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    max_retries: int = 3
    backoff_factor: float = 0.5
    pool_connections: int = 10
    pool_maxsize: int = 10
    retry_status_codes: Tuple[int, ...] = (500, 502, 503, 504)
    
    @property
    def telemetry_endpoint(self) -> str:
        """URL del endpoint de telemetría."""
        return f"{self.base_url}/api/telemetry/status"
    
    def tools_endpoint(self, vector: str) -> str:
        """URL del endpoint de herramientas."""
        return f"{self.base_url}/api/tools/{vector}"


@dataclass(frozen=True)
class TopologyConfig:
    """Configuración de análisis topológico."""
    
    max_history: int = DEFAULT_TOPOLOGY_HISTORY
    persistence_window: int = DEFAULT_PERSISTENCE_WINDOW
    health_critical_threshold: float = 0.4
    health_warning_threshold: float = 0.7
    
    # Topología esperada (conexiones iniciales)
    expected_edges: Tuple[Tuple[str, str], ...] = (
        (NODE_AGENT, NODE_CORE),
        (NODE_CORE, NODE_REDIS),
        (NODE_CORE, NODE_FILESYSTEM),
    )


@dataclass(frozen=True)
class TimingConfig:
    """Configuración de tiempos del agente."""
    
    check_interval: int = DEFAULT_CHECK_INTERVAL
    debounce_window_seconds: int = DEFAULT_DEBOUNCE_SECONDS
    max_consecutive_failures: int = DEFAULT_MAX_FAILURES
    startup_backoff_initial: float = 5.0
    startup_backoff_max: float = 60.0
    startup_backoff_multiplier: float = 1.5
    startup_max_attempts: int = 20


@dataclass(frozen=True)
class AgentConfig:
    """
    Configuración consolidada del agente autónomo.
    
    Agrupa todas las configuraciones en una estructura inmutable.
    """
    
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    
    @classmethod
    def from_environment(cls) -> AgentConfig:
        """
        Construye configuración desde variables de entorno.
        
        Variables soportadas:
        - CORE_API_URL: URL del Core
        - CHECK_INTERVAL: Intervalo entre ciclos
        - REQUEST_TIMEOUT: Timeout de requests
        - PERSISTENCE_WINDOW_SIZE: Tamaño de ventana de persistencia
        """
        core_url = os.getenv("CORE_API_URL", DEFAULT_CORE_URL)
        core_url = cls._validate_and_normalize_url(core_url)
        
        check_interval = cls._parse_env_int("CHECK_INTERVAL", DEFAULT_CHECK_INTERVAL)
        request_timeout = cls._parse_env_int("REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)
        persistence_window = cls._parse_env_int("PERSISTENCE_WINDOW_SIZE", DEFAULT_PERSISTENCE_WINDOW)
        
        return cls(
            connection=ConnectionConfig(
                base_url=core_url,
                request_timeout=request_timeout,
            ),
            topology=TopologyConfig(
                persistence_window=persistence_window,
            ),
            timing=TimingConfig(
                check_interval=check_interval,
            ),
        )
    
    @staticmethod
    def _validate_and_normalize_url(url: str) -> str:
        """Valida y normaliza una URL."""
        if not url or not url.strip():
            raise ConfigurationError("CORE_API_URL no puede estar vacía")
        
        url = url.strip()
        
        # Agregar esquema si falta
        if not url.lower().startswith(("http://", "https://")):
            url = f"http://{url}"
        
        # Validar estructura
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                raise ConfigurationError(f"URL sin host válido: {url}")
            if parsed.scheme not in ("http", "https"):
                raise ConfigurationError(f"Esquema no soportado: {parsed.scheme}")
        except Exception as e:
            raise ConfigurationError(f"URL inválida '{url}': {e}")
        
        return url.rstrip("/")
    
    @staticmethod
    def _parse_env_int(name: str, default: int) -> int:
        """Parsea un entero desde variable de entorno."""
        env_value = os.getenv(name)
        if env_value is None:
            return default
        
        try:
            value = int(env_value)
            if value <= 0:
                raise ValueError()
            return value
        except (TypeError, ValueError):
            logger.warning(
                f"Valor inválido para {name}='{env_value}', usando default={default}"
            )
            return default


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================


@dataclass
class TelemetryData:
    """
    Datos de telemetría estructurados y validados.
    
    Los valores se normalizan al rango [0, 1] durante la construcción.
    
    Attributes:
        flyback_voltage: Voltaje de flyback normalizado [0, 1]
        saturation: Nivel de saturación normalizado [0, 1]
        integrity_score: Score de integridad [0, 1]
        timestamp: Momento de la captura
        raw_data: Datos originales sin procesar
    """

    flyback_voltage: float
    saturation: float
    integrity_score: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    _clamped: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Clampea valores al rango válido [0, 1]."""
        # Detectar si se clampearon valores
        original_v = self.flyback_voltage
        original_s = self.saturation
        original_i = self.integrity_score
        
        self.flyback_voltage = self._clamp(self.flyback_voltage)
        self.saturation = self._clamp(self.saturation)
        self.integrity_score = self._clamp(self.integrity_score)
        
        # Marcar si hubo clamping
        object.__setattr__(
            self,
            "_clamped",
            (
                original_v != self.flyback_voltage
                or original_s != self.saturation
                or original_i != self.integrity_score
            ),
        )
    
    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clampea un valor al rango [min_val, max_val]."""
        if math.isnan(value) or math.isinf(value):
            return min_val
        return max(min_val, min(max_val, value))

    @classmethod
    def from_dict(cls, data: Any) -> Optional[TelemetryData]:
        """
        Factory method para crear TelemetryData desde un diccionario.
        
        Implementa extracción con múltiples paths de fallback.
        
        Args:
            data: Diccionario con datos de telemetría.
            
        Returns:
            TelemetryData si la extracción es exitosa, None en caso contrario.
        """
        if not isinstance(data, dict):
            logger.warning(
                f"[TELEMETRY] Tipo inválido: {type(data).__name__}, esperado dict"
            )
            return None

        # Paths de búsqueda ordenados por prioridad
        voltage_paths = (
            "flux_condenser.max_flyback_voltage",
            "flyback_voltage",
            "voltage",
            "v",
        )
        saturation_paths = (
            "flux_condenser.avg_saturation",
            "saturation",
            "sat",
            "s",
        )

        flyback = cls._extract_metric(data, voltage_paths)
        saturation = cls._extract_metric(data, saturation_paths)
        integrity = cls._extract_numeric(data, "integrity_score", 1.0)

        # Log si no hay datos
        if flyback is None and saturation is None:
            logger.debug("[TELEMETRY] Sin métricas válidas, usando defaults (0.0, 0.0)")
        
        # Advertencias para valores fuera de rango
        for name, val in [("flyback_voltage", flyback), ("saturation", saturation)]:
            if val is not None and not (0.0 <= val <= 1.0):
                logger.warning(f"[TELEMETRY] {name}={val:.4f} ∉ [0,1]")

        return cls(
            flyback_voltage=flyback if flyback is not None else 0.0,
            saturation=saturation if saturation is not None else 0.0,
            integrity_score=integrity,
            raw_data=data,
        )

    @classmethod
    def _extract_metric(
        cls,
        data: Dict[str, Any],
        paths: Tuple[str, ...],
    ) -> Optional[float]:
        """Extrae una métrica buscando en múltiples paths."""
        # Buscar en namespace 'metrics' si existe
        search_spaces = [data]
        if isinstance(data.get("metrics"), dict):
            search_spaces.insert(0, data["metrics"])
        
        for space in search_spaces:
            for path in paths:
                value = space.get(path)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
        return None

    @classmethod
    def _extract_numeric(
        cls,
        data: Dict[str, Any],
        key: str,
        default: float,
    ) -> float:
        """Extrae un valor numérico con default."""
        value = data.get(key)
        if value is None:
            return default
        try:
            result = float(value)
            if math.isnan(result) or math.isinf(result):
                return default
            return result
        except (TypeError, ValueError):
            return default

    @property
    def is_idle(self) -> bool:
        """Indica si la telemetría representa un estado idle."""
        return self.flyback_voltage == 0.0 and self.saturation == 0.0

    @property
    def was_clamped(self) -> bool:
        """Indica si algún valor fue clampeado durante la construcción."""
        return self._clamped

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "flyback_voltage": round(self.flyback_voltage, 4),
            "saturation": round(self.saturation, 4),
            "integrity_score": round(self.integrity_score, 4),
            "timestamp": self.timestamp.isoformat(),
            "is_idle": self.is_idle,
        }


@dataclass
class AgentMetrics:
    """
    Métricas internas del agente para observabilidad.
    
    Permite monitorear el comportamiento y salud del propio agente.
    """

    cycles_executed: int = 0
    successful_observations: int = 0
    failed_observations: int = 0
    last_successful_observation: Optional[datetime] = None
    consecutive_failures: int = 0
    decisions_count: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    last_cycle_duration_ms: float = 0.0

    def record_success(self) -> None:
        """Registra una observación exitosa."""
        self.successful_observations += 1
        self.last_successful_observation = datetime.now()
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Registra una observación fallida."""
        self.failed_observations += 1
        self.consecutive_failures += 1

    def record_decision(self, decision: AgentDecision) -> None:
        """Registra una decisión tomada."""
        key = decision.name
        self.decisions_count[key] = self.decisions_count.get(key, 0) + 1

    def increment_cycle(self) -> None:
        """Incrementa el contador de ciclos."""
        self.cycles_executed += 1

    def record_cycle_duration(self, duration_seconds: float) -> None:
        """Registra la duración del último ciclo."""
        self.last_cycle_duration_ms = duration_seconds * 1000.0

    @property
    def success_rate(self) -> float:
        """Calcula la tasa de éxito de observaciones."""
        total = self.successful_observations + self.failed_observations
        return self.successful_observations / total if total > 0 else 0.0

    @property
    def uptime_seconds(self) -> float:
        """Retorna el tiempo de ejecución en segundos."""
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def observation_rate(self) -> float:
        """Tasa de observaciones por minuto."""
        uptime_minutes = self.uptime_seconds / 60.0
        if uptime_minutes < EPSILON:
            return 0.0
        return self.successful_observations / uptime_minutes

    def to_dict(self) -> Dict[str, Any]:
        """Serializa las métricas a diccionario."""
        return {
            "cycles_executed": self.cycles_executed,
            "successful_observations": self.successful_observations,
            "failed_observations": self.failed_observations,
            "success_rate": round(self.success_rate, 4),
            "consecutive_failures": self.consecutive_failures,
            "last_successful_observation": (
                self.last_successful_observation.isoformat()
                if self.last_successful_observation
                else None
            ),
            "decisions_count": dict(self.decisions_count),
            "uptime_seconds": round(self.uptime_seconds, 2),
            "last_cycle_duration_ms": round(self.last_cycle_duration_ms, 2),
            "observation_rate_per_min": round(self.observation_rate, 2),
        }


@dataclass
class TopologicalDiagnosis:
    """
    Diagnóstico topológico estructurado para el ciclo OODA.
    
    Encapsula toda la información topológica relevante para la toma de decisiones.
    """

    health: TopologicalHealth
    voltage_persistence: PersistenceAnalysisResult
    saturation_persistence: PersistenceAnalysisResult
    summary: str
    recommended_status: SystemStatus
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_structurally_healthy(self) -> bool:
        """Sistema estructuralmente sano (conectado, sin fragmentación)."""
        return self.health.betti.is_connected

    @property
    def has_retry_loops(self) -> bool:
        """Hay patrones de reintentos detectados."""
        return len(self.health.request_loops) > 0

    @property
    def health_score(self) -> float:
        """Score de salud topológica."""
        return self.health.health_score

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario completo."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "recommended_status": self.recommended_status.name,
            "is_structurally_healthy": self.is_structurally_healthy,
            "has_retry_loops": self.has_retry_loops,
            "health": {
                "score": round(self.health.health_score, 4),
                "level": self.health.level.name,
                "betti": {
                    "b0": self.health.betti.b0,
                    "b1": self.health.betti.b1,
                    "is_connected": self.health.betti.is_connected,
                },
                "disconnected_nodes": list(self.health.disconnected_nodes),
                "retry_loops_count": len(self.health.request_loops),
            },
            "persistence": {
                "voltage_state": self.voltage_persistence.state.name,
                "saturation_state": self.saturation_persistence.state.name,
            },
        }

    def to_log_dict(self) -> Dict[str, Any]:
        """Serializa para logging estructurado (versión compacta)."""
        return {
            "betti_b0": self.health.betti.b0,
            "health_score": round(self.health.health_score, 3),
            "health_level": self.health.level.name,
            "voltage_state": self.voltage_persistence.state.name,
            "saturation_state": self.saturation_persistence.state.name,
            "disconnected_nodes": list(self.health.disconnected_nodes),
            "retry_loops": len(self.health.request_loops),
            "recommended_status": self.recommended_status.name,
        }


@dataclass
class ObservationResult:
    """
    Resultado de una operación de observación.
    
    Encapsula tanto éxitos como fallos de manera uniforme.
    """
    
    success: bool
    telemetry: Optional[TelemetryData]
    error_type: Optional[ObservationErrorType]
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def success_result(
        cls,
        telemetry: TelemetryData,
        request_id: str,
    ) -> ObservationResult:
        """Crea resultado de éxito."""
        return cls(
            success=True,
            telemetry=telemetry,
            error_type=None,
            request_id=request_id,
        )
    
    @classmethod
    def failure_result(
        cls,
        error_type: ObservationErrorType,
        request_id: str,
    ) -> ObservationResult:
        """Crea resultado de fallo."""
        return cls(
            success=False,
            telemetry=None,
            error_type=error_type,
            request_id=request_id,
        )


# ============================================================================
# EVALUADORES DE ESTADO (CHAIN OF RESPONSIBILITY)
# ============================================================================


@runtime_checkable
class StateEvaluator(Protocol):
    """
    Protocolo para evaluadores de estado del sistema.
    
    Cada evaluador implementa un morfismo parcial E: (T, H, P) → S ∪ {⊥}
    donde ⊥ indica que el evaluador no aplica.
    """
    
    @property
    def name(self) -> str:
        """Nombre del evaluador."""
        ...
    
    @property
    def priority(self) -> int:
        """Prioridad (menor = se evalúa primero)."""
        ...
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """
        Evalúa el estado del sistema.
        
        Returns:
            Tupla (status, summary) si aplica, None si no aplica.
        """
        ...


@dataclass(frozen=True)
class FragmentationEvaluator:
    """Evalúa fragmentación topológica (β₀ > 1)."""
    
    name: str = "FragmentationEvaluator"
    priority: int = 10
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        if not topo_health.betti.is_connected:
            nodes = ", ".join(topo_health.disconnected_nodes) or "∅"
            return (
                SystemStatus.DISCONNECTED,
                f"Fragmentación Topológica: β₀={topo_health.betti.b0}. Nodos: [{nodes}]",
            )
        return None


@dataclass(frozen=True)
class NoTelemetryEvaluator:
    """Evalúa ausencia de telemetría."""
    
    name: str = "NoTelemetryEvaluator"
    priority: int = 20
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        if telemetry is None:
            if metrics.consecutive_failures > 0:
                return (
                    SystemStatus.UNKNOWN,
                    f"Sin telemetría ({metrics.consecutive_failures} fallos consecutivos)",
                )
            return (SystemStatus.UNKNOWN, "Esperando telemetría inicial")
        return None


@dataclass(frozen=True)
class CriticalVoltageEvaluator:
    """Evalúa voltaje crítico instantáneo (safety net)."""
    
    name: str = "CriticalVoltageEvaluator"
    priority: int = 30
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        if telemetry and telemetry.flyback_voltage >= config.flyback_voltage_critical:
            return (
                SystemStatus.CRITICO,
                f"Voltaje crítico: {telemetry.flyback_voltage:.3f} ≥ {config.flyback_voltage_critical}",
            )
        return None


@dataclass(frozen=True)
class CriticalSaturationEvaluator:
    """Evalúa saturación crítica instantánea (safety net)."""
    
    name: str = "CriticalSaturationEvaluator"
    priority: int = 31
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        if telemetry and telemetry.saturation >= config.saturation_critical:
            return (
                SystemStatus.CRITICO,
                f"Saturación crítica: {telemetry.saturation:.3f} ≥ {config.saturation_critical}",
            )
        return None


@dataclass(frozen=True)
class TopologyHealthCriticalEvaluator:
    """Evalúa salud topológica crítica."""
    
    name: str = "TopologyHealthCriticalEvaluator"
    priority: int = 40
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        if topo_health.level == HealthLevel.CRITICAL:
            return (
                SystemStatus.CRITICO,
                f"Salud topológica crítica: score={topo_health.health_score:.2f}",
            )
        return None


@dataclass(frozen=True)
class PersistentSaturationCriticalEvaluator:
    """Evalúa saturación persistente crítica."""
    
    name: str = "PersistentSaturationCriticalEvaluator"
    priority: int = 50
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        if saturation_analysis.state == MetricState.CRITICAL:
            duration = saturation_analysis.metadata.get("active_duration", "?")
            return (
                SystemStatus.SATURADO,
                f"Saturación persistente: {duration} muestras",
            )
        return None


@dataclass(frozen=True)
class SaturationFeatureEvaluator:
    """Evalúa saturación con característica estructural."""
    
    name: str = "SaturationFeatureEvaluator"
    priority: int = 51
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        if saturation_analysis.state == MetricState.FEATURE:
            return (
                SystemStatus.SATURADO,
                f"Patrón estructural saturación: {saturation_analysis.feature_count} feature(s), "
                f"π={saturation_analysis.total_persistence:.1f}",
            )
        return None


@dataclass(frozen=True)
class PersistentVoltageCriticalEvaluator:
    """Evalúa voltaje persistente crítico."""
    
    name: str = "PersistentVoltageCriticalEvaluator"
    priority: int = 60
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        if voltage_analysis.state == MetricState.CRITICAL:
            duration = voltage_analysis.metadata.get("active_duration", "?")
            return (
                SystemStatus.INESTABLE,
                f"Inestabilidad voltaje: {duration} muestras",
            )
        return None


@dataclass(frozen=True)
class VoltageFeatureEvaluator:
    """Evalúa voltaje con característica estructural."""
    
    name: str = "VoltageFeatureEvaluator"
    priority: int = 61
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        if voltage_analysis.state == MetricState.FEATURE:
            return (
                SystemStatus.INESTABLE,
                f"Patrón estructural voltaje: λ_max={voltage_analysis.max_lifespan:.1f}",
            )
        return None


@dataclass(frozen=True)
class RetryLoopEvaluator:
    """Evalúa patrones de reintentos excesivos."""
    
    name: str = "RetryLoopEvaluator"
    priority: int = 70
    min_loop_count: int = 5
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        if not topo_health.request_loops:
            return None
        
        first_loop = topo_health.request_loops[0]
        if (
            first_loop.count >= self.min_loop_count
            and first_loop.request_id.startswith("FAIL_")
        ):
            return (
                SystemStatus.INESTABLE,
                f"Patrón reintentos: '{first_loop.request_id}' ×{first_loop.count}",
            )
        return None


@dataclass(frozen=True)
class UnhealthyTopologyEvaluator:
    """Evalúa salud topológica degradada."""
    
    name: str = "UnhealthyTopologyEvaluator"
    priority: int = 80
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        if topo_health.level == HealthLevel.UNHEALTHY:
            return (
                SystemStatus.INESTABLE,
                f"Salud degradada: score={topo_health.health_score:.2f}",
            )
        return None


class EvaluatorChain:
    """
    Cadena de evaluadores ordenada por prioridad.
    
    Implementa el patrón Chain of Responsibility para evaluación de estado.
    """
    
    def __init__(self, evaluators: Optional[List[StateEvaluator]] = None):
        """
        Inicializa la cadena.
        
        Args:
            evaluators: Lista de evaluadores. Si es None, usa los defaults.
        """
        if evaluators is None:
            evaluators = self._create_default_evaluators()
        
        # Ordenar por prioridad
        self._evaluators = sorted(evaluators, key=lambda e: e.priority)
    
    @staticmethod
    def _create_default_evaluators() -> List[StateEvaluator]:
        """Crea la lista de evaluadores por defecto."""
        return [
            FragmentationEvaluator(),
            NoTelemetryEvaluator(),
            CriticalVoltageEvaluator(),
            CriticalSaturationEvaluator(),
            TopologyHealthCriticalEvaluator(),
            PersistentSaturationCriticalEvaluator(),
            SaturationFeatureEvaluator(),
            PersistentVoltageCriticalEvaluator(),
            VoltageFeatureEvaluator(),
            RetryLoopEvaluator(),
            UnhealthyTopologyEvaluator(),
        ]
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Tuple[SystemStatus, str]:
        """
        Evalúa el estado del sistema recorriendo la cadena.
        
        Returns:
            Tupla (status, summary) del primer evaluador que aplica,
            o (NOMINAL, summary) si ninguno aplica.
        """
        for evaluator in self._evaluators:
            try:
                result = evaluator.evaluate(
                    telemetry,
                    topo_health,
                    voltage_analysis,
                    saturation_analysis,
                    config,
                    metrics,
                )
                if result is not None:
                    status, summary = result
                    log_level = (
                        logging.CRITICAL
                        if status == SystemStatus.CRITICO
                        else logging.WARNING
                    )
                    logger.log(log_level, f"[EVAL:{evaluator.name}] {summary}")
                    return result
            except Exception as e:
                logger.warning(f"[EVAL] Error en {evaluator.name}: {e}")
                continue
        
        # Estado nominal
        return (
            SystemStatus.NOMINAL,
            f"Sistema nominal: β₀={topo_health.betti.b0}, h={topo_health.health_score:.2f}",
        )


# ============================================================================
# AGENTE AUTÓNOMO PRINCIPAL
# ============================================================================


class AutonomousAgent:
    """
    Agente autónomo que opera bajo un ciclo OODA (Observe, Orient, Decide, Act).
    
    Monitorea la salud del Core y toma decisiones basadas en métricas de telemetría.

    Características:
        - Ciclo OODA continuo con manejo robusto de errores
        - Configuración flexible via AgentConfig o variables de entorno
        - Reintentos automáticos con backoff exponencial
        - Graceful shutdown ante señales del sistema
        - Métricas internas para observabilidad
        - Debounce de decisiones para evitar spam
        - Cadena de evaluadores extensible
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        evaluator_chain: Optional[EvaluatorChain] = None,
    ) -> None:
        """
        Inicializa el agente autónomo.
        
        Args:
            config: Configuración del agente. Si es None, se crea desde env vars.
            evaluator_chain: Cadena de evaluadores. Si es None, usa los defaults.
            
        Raises:
            ConfigurationError: Si la configuración es inválida.
        """
        # Configuración
        self.config = config or AgentConfig.from_environment()
        
        # Alias para acceso rápido
        self._thresholds = self.config.thresholds
        self._conn_config = self.config.connection
        self._topo_config = self.config.topology
        self._timing_config = self.config.timing
        
        # Estado interno
        self._running: bool = False
        self._last_decision: Optional[AgentDecision] = None
        self._last_decision_time: Optional[datetime] = None
        self._last_status: Optional[SystemStatus] = None
        self._last_diagnosis: Optional[TopologicalDiagnosis] = None
        self._last_telemetry: Optional[TelemetryData] = None
        self._metrics = AgentMetrics()
        
        # Componentes de análisis
        self.topology = SystemTopology(max_history=self._topo_config.max_history)
        self.persistence = PersistenceHomology(
            window_size=self._topo_config.persistence_window
        )
        self._evaluator_chain = evaluator_chain or EvaluatorChain()
        
        # Inicializar topología esperada
        self._initialize_expected_topology()
        
        # Sesión HTTP
        self._session = self._create_robust_session()
        
        # Signal handlers
        self._original_handlers: Dict[int, Any] = {}
        self._setup_signal_handlers()
        
        logger.info(
            f"AutonomousAgent inicializado | "
            f"Core: {self._conn_config.base_url} | "
            f"Intervalo: {self._timing_config.check_interval}s | "
            f"Timeout: {self._conn_config.request_timeout}s"
        )

    # =========================================================================
    # INICIALIZACIÓN
    # =========================================================================

    def _initialize_expected_topology(self) -> None:
        """Establece la topología inicial esperada del sistema."""
        edges_added, warnings = self.topology.update_connectivity(
            list(self._topo_config.expected_edges),
            validate_nodes=True,
            auto_add_nodes=True,
        )
        
        for warn in warnings:
            logger.warning(f"[TOPO-INIT] {warn}")
        
        logger.debug(
            f"[TOPO-INIT] Topología inicial: {edges_added} conexiones activas"
        )

    def _create_robust_session(self) -> requests.Session:
        """Crea sesión HTTP con política de reintentos."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self._conn_config.max_retries,
            backoff_factor=self._conn_config.backoff_factor,
            status_forcelist=list(self._conn_config.retry_status_codes),
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self._conn_config.pool_connections,
            pool_maxsize=self._conn_config.pool_maxsize,
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            "User-Agent": "APU-Agent-Internal",
            "Content-Type": "application/json",
        })
        
        return session

    def _setup_signal_handlers(self) -> None:
        """Configura manejadores para shutdown graceful."""
        signals_to_handle = [signal.SIGINT, signal.SIGTERM]
        
        for sig in signals_to_handle:
            self._original_handlers[sig] = signal.signal(sig, self._handle_shutdown)

    def _restore_signal_handlers(self) -> None:
        """Restaura los manejadores de señales originales."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Manejador de señales de terminación."""
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = str(signum)
        
        logger.info(f"Señal {sig_name} recibida. Iniciando shutdown...")
        self._running = False

    # =========================================================================
    # OODA LOOP - OBSERVE
    # =========================================================================

    def observe(self) -> Optional[TelemetryData]:
        """
        OBSERVE - Primera fase del ciclo OODA.
        
        Implementa el morfismo O: Infraestructura → Telemetría
        con manejo de errores que preserva coherencia topológica.
        
        Returns:
            TelemetryData si la observación es exitosa, None en caso contrario.
        """
        request_id = f"obs_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        result = self._execute_observation(request_id)
        self._process_observation_result(result)
        
        return result.telemetry

    def _execute_observation(self, request_id: str) -> ObservationResult:
        """Ejecuta la observación HTTP."""
        try:
            response = self._session.get(
                self._conn_config.telemetry_endpoint,
                timeout=self._conn_config.request_timeout,
            )
            
            if not response.ok:
                return ObservationResult.failure_result(
                    ObservationErrorType.HTTP_ERROR,
                    request_id,
                )
            
            try:
                raw_data = response.json()
            except ValueError:
                return ObservationResult.failure_result(
                    ObservationErrorType.INVALID_JSON,
                    request_id,
                )
            
            telemetry = TelemetryData.from_dict(raw_data)
            if telemetry is None:
                return ObservationResult.failure_result(
                    ObservationErrorType.INVALID_TELEMETRY,
                    request_id,
                )
            
            return ObservationResult.success_result(telemetry, request_id)
            
        except requests.exceptions.Timeout:
            return ObservationResult.failure_result(
                ObservationErrorType.TIMEOUT,
                request_id,
            )
        except requests.exceptions.ConnectionError:
            return ObservationResult.failure_result(
                ObservationErrorType.CONNECTION_ERROR,
                request_id,
            )
        except requests.exceptions.RequestException:
            return ObservationResult.failure_result(
                ObservationErrorType.REQUEST_ERROR,
                request_id,
            )
        except Exception as e:
            logger.error(f"[OBSERVE] Error inesperado: {e}")
            return ObservationResult.failure_result(
                ObservationErrorType.UNKNOWN,
                request_id,
            )

    def _process_observation_result(self, result: ObservationResult) -> None:
        """Procesa el resultado de la observación actualizando estado interno."""
        if result.success and result.telemetry:
            self._metrics.record_success()
            self.topology.record_request(result.request_id)
            self._last_telemetry = result.telemetry
            
            # Actualizar topología desde telemetría
            self._update_topology_from_telemetry(result.telemetry)
            self.topology.clear_request_history()
            
            logger.debug(
                f"[OBSERVE] ✓ v={result.telemetry.flyback_voltage:.3f}, "
                f"s={result.telemetry.saturation:.3f}"
            )
        else:
            self._metrics.record_failure()
            error_id = f"FAIL_{result.error_type.value if result.error_type else 'UNKNOWN'}"
            self.topology.record_request(error_id)
            
            if self._metrics.consecutive_failures >= self._timing_config.max_consecutive_failures:
                logger.warning(
                    f"[TOPO] Degradación: {self._metrics.consecutive_failures} fallos"
                )
                self.topology.remove_edge(NODE_AGENT, NODE_CORE)
            
            if result.error_type:
                logger.warning(f"[OBSERVE] {result.error_type.value}")

    def _update_topology_from_telemetry(self, telemetry: TelemetryData) -> None:
        """Actualiza la topología basándose en la telemetría."""
        raw = telemetry.raw_data
        active_connections = [(NODE_AGENT, NODE_CORE)]
        
        if raw.get("redis_connected", True):
            active_connections.append((NODE_CORE, NODE_REDIS))
        if raw.get("filesystem_accessible", True):
            active_connections.append((NODE_CORE, NODE_FILESYSTEM))
        
        self.topology.update_connectivity(
            active_connections,
            validate_nodes=True,
            auto_add_nodes=False,
        )

    # =========================================================================
    # OODA LOOP - ORIENT
    # =========================================================================

    def orient(self, telemetry: Optional[TelemetryData]) -> SystemStatus:
        """
        ORIENT - Segunda fase del ciclo OODA.
        
        Implementa el morfismo R: Telemetría × Homología → Estado
        
        Args:
            telemetry: Datos de telemetría (puede ser None).
            
        Returns:
            Estado del sistema evaluado.
        """
        # Calcular invariantes topológicos
        topo_health = self.topology.get_topological_health(calculate_b1=False)
        
        # Analizar persistencia de métricas
        voltage_analysis = self._analyze_metric_persistence(
            "flyback_voltage",
            telemetry.flyback_voltage if telemetry else None,
            self._thresholds.flyback_voltage_warning,
        )
        saturation_analysis = self._analyze_metric_persistence(
            "saturation",
            telemetry.saturation if telemetry else None,
            self._thresholds.saturation_warning,
        )
        
        # Evaluar estado mediante la cadena
        status, summary = self._evaluator_chain.evaluate(
            telemetry,
            topo_health,
            voltage_analysis,
            saturation_analysis,
            self._thresholds,
            self._metrics,
        )
        
        # Construir y almacenar diagnóstico
        self._last_diagnosis = TopologicalDiagnosis(
            health=topo_health,
            voltage_persistence=voltage_analysis,
            saturation_persistence=saturation_analysis,
            summary=summary,
            recommended_status=status,
        )
        
        # Log para estados no nominales
        if status != SystemStatus.NOMINAL:
            logger.info(f"[ORIENT] {self._last_diagnosis.to_log_dict()}")
        
        # Log de ruido filtrado
        for name, analysis in [("voltaje", voltage_analysis), ("saturación", saturation_analysis)]:
            if analysis.state == MetricState.NOISE:
                logger.debug(
                    f"[PERSIST] Ruido {name} filtrado: {analysis.noise_count} excursiones"
                )
        
        return status

    def _analyze_metric_persistence(
        self,
        metric_name: str,
        current_value: Optional[float],
        threshold: float,
    ) -> PersistenceAnalysisResult:
        """Analiza la persistencia de una métrica."""
        if current_value is not None:
            self.persistence.add_reading(metric_name, current_value)
        
        return self.persistence.analyze_persistence(
            metric_name,
            threshold=threshold,
            noise_ratio=0.2,
            critical_ratio=0.5,
        )

    # =========================================================================
    # OODA LOOP - DECIDE
    # =========================================================================

    def decide(self, status: SystemStatus) -> AgentDecision:
        """
        DECIDE - Tercera fase del ciclo OODA.
        
        Implementa el morfismo D: Estado × Contexto → Acción
        
        Args:
            status: Estado actual del sistema.
            
        Returns:
            Decisión a ejecutar.
        """
        # Matriz de decisión
        decision_matrix: Dict[SystemStatus, AgentDecision] = {
            SystemStatus.NOMINAL: AgentDecision.HEARTBEAT,
            SystemStatus.UNKNOWN: AgentDecision.WAIT,
            SystemStatus.INESTABLE: AgentDecision.EJECUTAR_LIMPIEZA,
            SystemStatus.SATURADO: AgentDecision.AJUSTAR_VELOCIDAD,
            SystemStatus.CRITICO: AgentDecision.ALERTA_CRITICA,
            SystemStatus.DISCONNECTED: AgentDecision.RECONNECT,
        }
        
        decision = decision_matrix.get(status, AgentDecision.WAIT)
        
        # Refinamiento contextual
        if self._last_diagnosis and decision == AgentDecision.HEARTBEAT:
            error_loops = [
                loop for loop in self._last_diagnosis.health.request_loops
                if loop.request_id.startswith("FAIL_")
            ]
            if error_loops:
                total_errors = sum(loop.count for loop in error_loops)
                logger.debug(
                    f"[DECIDE] Nominal con {len(error_loops)} patrones de error (Σ={total_errors})"
                )
        
        self._metrics.record_decision(decision)
        self._last_status = status
        
        return decision

    # =========================================================================
    # OODA LOOP - ACT
    # =========================================================================

    def act(self, decision: AgentDecision) -> bool:
        """
        ACT - Cuarta fase del ciclo OODA.
        
        Implementa el morfismo A: Decisión × Diagnóstico → Efectos
        
        Args:
            decision: Decisión a ejecutar.
            
        Returns:
            True si la acción se ejecutó, False si fue suprimida.
        """
        if self._should_debounce(decision):
            logger.debug(f"[ACT] Suprimido por debounce: {decision.name}")
            return False
        
        diagnosis_msg = self._build_diagnosis_message()
        
        # Ejecutar acción
        action_handlers: Dict[AgentDecision, Callable[[], None]] = {
            AgentDecision.HEARTBEAT: self._emit_heartbeat,
            AgentDecision.EJECUTAR_LIMPIEZA: lambda: self._execute_cleanup(diagnosis_msg),
            AgentDecision.AJUSTAR_VELOCIDAD: lambda: self._apply_backpressure(diagnosis_msg),
            AgentDecision.ALERTA_CRITICA: lambda: self._raise_critical_alert(diagnosis_msg),
            AgentDecision.RECONNECT: lambda: self._attempt_reconnection(diagnosis_msg),
            AgentDecision.WAIT: lambda: logger.info("[BRAIN] ⏳ Esperando telemetría..."),
        }
        
        handler = action_handlers.get(decision, action_handlers[AgentDecision.WAIT])
        handler()
        
        # Actualizar estado temporal
        self._last_decision = decision
        self._last_decision_time = datetime.now()
        
        return True

    def _should_debounce(self, decision: AgentDecision) -> bool:
        """Determina si una acción debe ser suprimida por debounce."""
        # Decisiones críticas nunca se suprimen
        if decision.requires_immediate_action:
            return False
        
        if self._last_decision is None or self._last_decision_time is None:
            return False
        
        if decision != self._last_decision:
            return False
        
        # Verificar ventana de tiempo
        elapsed = datetime.now() - self._last_decision_time
        return elapsed < timedelta(seconds=self._timing_config.debounce_window_seconds)

    def _emit_heartbeat(self) -> None:
        """Emite señal de sistema nominal."""
        health_score = (
            self._last_diagnosis.health.health_score
            if self._last_diagnosis
            else 1.0
        )
        indicator = "✅" if health_score >= 0.9 else "🟢" if health_score >= 0.7 else "🟡"
        logger.info(f"[BRAIN] {indicator} NOMINAL - h={health_score:.2f}")

    def _execute_cleanup(self, diagnosis_msg: str) -> None:
        """Proyecta vector de limpieza al estrato físico."""
        logger.warning(f"[BRAIN] ⚠️ INESTABILIDAD - {diagnosis_msg}")
        
        success = self._project_intent(
            vector="clean",
            stratum="PHYSICS",
            payload={
                "mode": "EMERGENCY",
                "reason": diagnosis_msg,
                "scope": "flux_condenser",
            },
        )
        
        event = "instability_resolved" if success else "instability_correction_failed"
        self._notify_external_system(event, {"method": "clean"})

    def _apply_backpressure(self, diagnosis_msg: str) -> None:
        """Aplica backpressure reduciendo tasa de entrada."""
        logger.warning(f"[BRAIN] ⚠️ SATURACIÓN - {diagnosis_msg}")
        
        success = self._project_intent(
            vector="configure",
            stratum="PHYSICS",
            payload={
                "target": "flux_condenser",
                "parameter": "input_rate",
                "action": "decrease",
                "factor": 0.5,
            },
        )
        
        event = "saturation_mitigated" if success else "saturation_correction_failed"
        self._notify_external_system(event, {"method": "throttle"})

    def _raise_critical_alert(self, diagnosis_msg: str) -> None:
        """Emite alerta crítica."""
        logger.critical(f"[BRAIN] 🚨 CRÍTICO - {diagnosis_msg}")
        logger.critical("[BRAIN] → Intervención inmediata requerida")
        
        context: Dict[str, Any] = {"diagnosis": diagnosis_msg}
        if self._last_diagnosis:
            context.update({
                "health_score": self._last_diagnosis.health.health_score,
                "betti_b0": self._last_diagnosis.health.betti.b0,
                "is_connected": self._last_diagnosis.is_structurally_healthy,
            })
        
        self._notify_external_system("critical_alert", context)

    def _attempt_reconnection(self, diagnosis_msg: str) -> None:
        """Intenta reconexión reinicializando topología."""
        logger.warning(f"[BRAIN] 🔄 DESCONEXIÓN - {diagnosis_msg}")
        logger.warning("[BRAIN] → Reinicializando topología...")
        self._initialize_expected_topology()

    def _project_intent(
        self,
        vector: str,
        stratum: str,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Proyecta intención sobre la MIC.
        
        Args:
            vector: Nombre del vector (herramienta).
            stratum: Nivel de gobernanza.
            payload: Datos específicos.
            
        Returns:
            True si la proyección fue exitosa.
        """
        intent = {
            "vector": vector,
            "stratum": stratum,
            "payload": payload,
            "context": {
                "agent_id": "apu_agent_sidecar",
                "timestamp": datetime.now().isoformat(),
                "force_physics_override": True,
                "topology_health": (
                    self._last_diagnosis.health.health_score
                    if self._last_diagnosis
                    else None
                ),
            },
        }
        
        url = self._conn_config.tools_endpoint(vector)
        logger.info(f"[INTENT] Proyectando '{vector}' → estrato '{stratum}'")
        
        try:
            response = self._session.post(
                url,
                json=intent,
                timeout=self._conn_config.request_timeout,
            )
            
            if response.ok:
                logger.info(f"[INTENT] ✅ {vector} ejecutado exitosamente")
                return True
            
            logger.error(
                f"[INTENT] ❌ HTTP {response.status_code}: {response.text[:100]}"
            )
            return False
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[INTENT] Error de proyección: {type(e).__name__}")
            return False

    def _build_diagnosis_message(self) -> str:
        """Construye mensaje de diagnóstico."""
        if not self._last_diagnosis:
            return "Sin diagnóstico"
        
        diag = self._last_diagnosis
        components = [diag.summary]
        
        betti = diag.health.betti
        if not betti.is_ideal:
            components.append(f"β₀={betti.b0}")
        
        if diag.health.health_score < 0.9:
            components.append(f"h={diag.health.health_score:.2f}")
        
        return " | ".join(components)

    def _notify_external_system(
        self,
        event_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Hook para notificaciones externas."""
        log_data: Dict[str, Any] = {"event": event_type}
        if context:
            log_data.update(context)
        
        logger.debug(f"[NOTIFY] {log_data}")

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def health_check(self) -> bool:
        """
        Verifica conectividad con el Core y estado topológico.
        
        Returns:
            True si el Core es accesible y la topología es válida.
        """
        logger.info(f"Ejecutando health check: {self._conn_config.telemetry_endpoint}")
        
        try:
            response = self._session.get(
                self._conn_config.telemetry_endpoint,
                timeout=self._conn_config.request_timeout,
            )
            
            if response.ok:
                self._initialize_expected_topology()
                topo_health = self.topology.get_topological_health(calculate_b1=False)
                
                logger.info(
                    f"✅ Health check exitoso - "
                    f"Core accesible, topología: {topo_health.level.name} "
                    f"(score={topo_health.health_score:.2f})"
                )
                return True
            else:
                logger.warning(f"⚠️ Health check: HTTP {response.status_code}")
                return True
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Health check fallido: {e}")
            
            self.topology.remove_edge(NODE_AGENT, NODE_CORE)
            topo_health = self.topology.get_topological_health(calculate_b1=False)
            
            logger.error(
                f"Topología degradada: β₀={topo_health.betti.b0}, "
                f"health={topo_health.health_score:.2f}"
            )
            return False

    def get_stratum_health(self, stratum: Stratum) -> Dict[str, Any]:
        """
        Retorna la salud filtrada por estrato.
        
        Args:
            stratum: Nivel jerárquico a consultar.
            
        Returns:
            Dict con métricas específicas del nivel.
        """
        if stratum == Stratum.PHYSICS:
            return self._get_physics_stratum_health()
        elif stratum == Stratum.TACTICS:
            return self._get_tactics_stratum_health()
        elif stratum == Stratum.STRATEGY:
            return self._get_strategy_stratum_health()
        elif stratum == Stratum.WISDOM:
            return self._get_wisdom_stratum_health()
        else:
            return {"error": "Invalid Stratum"}

    def _get_physics_stratum_health(self) -> Dict[str, Any]:
        """Obtiene salud del estrato PHYSICS."""
        telemetry = self._last_telemetry
        
        if telemetry is None:
            return {
                "stratum": "PHYSICS",
                "voltage": None,
                "saturation": None,
                "status": "UNKNOWN",
                "integrity": 0.0,
                "timestamp": None,
            }
        
        # Clasificar estado
        v_class = self._thresholds.classify_voltage(telemetry.flyback_voltage)
        s_class = self._thresholds.classify_saturation(telemetry.saturation)
        
        if v_class == "critical" or s_class == "critical":
            status = "CRITICO"
        elif v_class == "warning" or s_class == "warning":
            status = "WARNING"
        else:
            status = "NOMINAL"
        
        return {
            "stratum": "PHYSICS",
            "voltage": telemetry.flyback_voltage,
            "saturation": telemetry.saturation,
            "status": status,
            "integrity": telemetry.integrity_score,
            "timestamp": telemetry.timestamp.isoformat(),
        }

    def _get_tactics_stratum_health(self) -> Dict[str, Any]:
        """Obtiene salud del estrato TACTICS."""
        health = self.topology.get_topological_health(calculate_b1=True)
        betti = health.betti
        
        return {
            "stratum": "TACTICS",
            "betti_0": betti.b0,
            "betti_1": betti.b1,
            "is_connected": betti.is_connected,
            "health_score": round(health.health_score, 3),
            "euler": betti.euler_characteristic,
        }

    def _get_strategy_stratum_health(self) -> Dict[str, Any]:
        """Obtiene salud del estrato STRATEGY."""
        confidence = 0.0
        if self._last_diagnosis:
            confidence = self._last_diagnosis.health.health_score
        
        status_age = 0.0
        if self._last_decision_time:
            status_age = (datetime.now() - self._last_decision_time).total_seconds()
        
        risk_detected = self._last_status in (
            SystemStatus.SATURADO,
            SystemStatus.CRITICO,
        )
        
        return {
            "stratum": "STRATEGY",
            "risk_detected": risk_detected,
            "last_decision": self._last_decision.name if self._last_decision else None,
            "confidence": confidence,
            "status_age": status_age,
        }

    def _get_wisdom_stratum_health(self) -> Dict[str, Any]:
        """Obtiene salud del estrato WISDOM."""
        rationale = "Sin diagnóstico previo."
        if self._last_diagnosis:
            rationale = self._last_diagnosis.summary
        
        return {
            "stratum": "WISDOM",
            "verdict": self._last_status.name if self._last_status else "UNKNOWN",
            "certainty": 1.0 if self._last_diagnosis else 0.0,
            "rationale": rationale,
            "cycles_executed": self._metrics.cycles_executed,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas completas del agente."""
        metrics = self._metrics.to_dict()
        metrics.update({
            "core_api_url": self._conn_config.base_url,
            "check_interval": self._timing_config.check_interval,
            "is_running": self._running,
            "last_status": self._last_status.name if self._last_status else None,
        })
        
        # Topología
        topo_health = self.topology.get_topological_health(calculate_b1=False)
        metrics["topology"] = {
            "betti": {
                "b0": topo_health.betti.b0,
                "b1": topo_health.betti.b1,
            },
            "connectivity": {
                "is_connected": topo_health.betti.is_connected,
                "is_ideal": topo_health.betti.is_ideal,
                "euler_char": topo_health.betti.euler_characteristic,
            },
            "health": {
                "score": round(topo_health.health_score, 3),
                "level": topo_health.level.name,
            },
            "issues": {
                "disconnected_nodes": list(topo_health.disconnected_nodes),
                "missing_edges": [list(e) for e in topo_health.missing_edges],
                "retry_loops": len(topo_health.request_loops),
            },
        }
        
        # Persistencia
        persistence_data = {}
        for metric_name in ("flyback_voltage", "saturation"):
            stats = self.persistence.get_statistics(metric_name)
            if stats:
                persistence_data[metric_name] = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in stats.items()
                    if k in ("count", "min", "max", "mean", "std")
                }
        
        if persistence_data:
            metrics["persistence"] = persistence_data
        
        # Diagnóstico
        if self._last_diagnosis:
            metrics["last_diagnosis"] = {
                "summary": self._last_diagnosis.summary,
                "status": self._last_diagnosis.recommended_status.name,
                "metric_states": {
                    "voltage": self._last_diagnosis.voltage_persistence.state.name,
                    "saturation": self._last_diagnosis.saturation_persistence.state.name,
                },
            }
        
        return metrics

    def get_topological_summary(self) -> Dict[str, Any]:
        """Retorna resumen topológico para dashboards."""
        health = self.topology.get_topological_health()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "betti": {
                "values": {"b0": health.betti.b0, "b1": health.betti.b1},
                "interpretation": (
                    "Sistema conectado"
                    if health.betti.is_connected
                    else f"Sistema fragmentado en {health.betti.b0} componentes"
                ),
            },
            "health": {
                "score": round(health.health_score, 3),
                "level": health.level.name,
                "is_healthy": health.is_healthy,
            },
            "issues": {
                "disconnected": list(health.disconnected_nodes),
                "missing": [f"{u}↔{v}" for u, v in health.missing_edges],
                "diagnostics": health.diagnostics,
            },
            "patterns": [
                {"id": loop.request_id, "frequency": loop.count}
                for loop in health.request_loops[:5]
            ],
        }

    # =========================================================================
    # RUN LOOP
    # =========================================================================

    def _wait_for_startup(self) -> bool:
        """
        Espera a que el Core esté disponible (Cold Start).
        
        Implementa backoff exponencial con límite máximo.
        
        Returns:
            True si se conectó exitosamente, False si se excedió el límite.
        """
        logger.info("Iniciando protocolo de espera de arranque (Cold Start)...")
        
        backoff = self._timing_config.startup_backoff_initial
        attempts = 0
        max_attempts = self._timing_config.startup_max_attempts
        
        while self._running and attempts < max_attempts:
            attempts += 1
            
            try:
                response = requests.get(
                    self._conn_config.telemetry_endpoint,
                    timeout=self._conn_config.request_timeout,
                )
                
                if response.ok:
                    logger.info("✅ Core detectado y operativo (200 OK).")
                    return True
                else:
                    logger.info(
                        f"Esperando Core... (HTTP {response.status_code}) "
                        f"[{attempts}/{max_attempts}]"
                    )
                    
            except requests.exceptions.ConnectionError:
                logger.info(
                    f"Esperando Core (Cold Start)... [Conexión rechazada] "
                    f"[{attempts}/{max_attempts}]"
                )
            except requests.exceptions.RequestException as e:
                logger.info(
                    f"Esperando Core... [{type(e).__name__}] "
                    f"[{attempts}/{max_attempts}]"
                )
            
            if self._running:
                time.sleep(backoff)
                # Backoff exponencial
                backoff = min(
                    backoff * self._timing_config.startup_backoff_multiplier,
                    self._timing_config.startup_backoff_max,
                )
        
        if not self._running:
            logger.info("Startup cancelado por señal de shutdown")
            return False
        
        logger.error(f"Timeout de startup después de {attempts} intentos")
        return False

    def run(self, skip_health_check: bool = False) -> None:
        """
        Bucle principal del agente - Ejecuta el ciclo OODA continuamente.
        
        Args:
            skip_health_check: Si True, omite verificación inicial.
        """
        self._running = True
        
        if not skip_health_check:
            if not self._wait_for_startup():
                logger.error("No se pudo conectar al Core")
                return
            
            if not self.health_check():
                logger.warning("Iniciando agente con advertencias de salud...")
        
        logger.info("🚀 Iniciando OODA Loop...")
        
        try:
            while self._running:
                cycle_start = time.monotonic()
                self._metrics.increment_cycle()
                
                try:
                    # CICLO OODA
                    telemetry = self.observe()
                    status = self.orient(telemetry)
                    decision = self.decide(status)
                    self.act(decision)
                    
                except Exception as e:
                    logger.error(
                        f"Error en ciclo OODA #{self._metrics.cycles_executed}: {e}",
                        exc_info=True,
                    )
                
                # Registrar duración del ciclo
                cycle_duration = time.monotonic() - cycle_start
                self._metrics.record_cycle_duration(cycle_duration)
                
                # Sleep adaptativo
                sleep_time = max(
                    0.0,
                    self._timing_config.check_interval - cycle_duration,
                )
                
                if sleep_time > 0 and self._running:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            logger.critical(f"Error fatal en bucle principal: {e}", exc_info=True)
            raise
            
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Detiene el agente de forma controlada."""
        logger.info("Solicitando detención del agente...")
        self._running = False

    def _shutdown(self) -> None:
        """Limpieza al terminar el agente."""
        logger.info("Iniciando shutdown del AutonomousAgent...")
        
        final_metrics = self.get_metrics()
        logger.info(f"Métricas finales: {final_metrics}")
        
        if self._session:
            try:
                self._session.close()
            except Exception as e:
                logger.warning(f"Error cerrando sesión HTTP: {e}")
        
        self._restore_signal_handlers()
        
        logger.info("👋 AutonomousAgent detenido correctamente")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_agent(
    config: Optional[AgentConfig] = None,
    evaluator_chain: Optional[EvaluatorChain] = None,
) -> AutonomousAgent:
    """
    Factory function para crear un agente configurado.
    
    Args:
        config: Configuración del agente.
        evaluator_chain: Cadena de evaluadores personalizada.
        
    Returns:
        Agente configurado.
    """
    return AutonomousAgent(config=config, evaluator_chain=evaluator_chain)


def create_minimal_agent(core_url: str) -> AutonomousAgent:
    """
    Crea un agente con configuración mínima.
    
    Args:
        core_url: URL del Core API.
        
    Returns:
        Agente configurado.
    """
    config = AgentConfig(
        connection=ConnectionConfig(
            base_url=AgentConfig._validate_and_normalize_url(core_url),
        ),
    )
    return AutonomousAgent(config=config)


# ============================================================================
# ENTRY POINT
# ============================================================================


def main() -> int:
    """
    Punto de entrada principal.
    
    Returns:
        Código de salida (0=éxito, 1=error).
    """
    try:
        agent = AutonomousAgent()
        agent.run()
        return 0
        
    except ConfigurationError as e:
        logger.error(f"Error de configuración: {e}")
        return 1
        
    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario")
        return 0
        
    except Exception as e:
        logger.critical(f"Error no manejado: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())