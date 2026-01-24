"""
Este componente implementa un agente autónomo que gobierna el ciclo de vida del 
procesamiento de datos mediante el ciclo OODA (Observar, Orientar, Decidir, Actuar).
Su objetivo es mantener la estabilidad operativa ("Homeostasis") ajustando dinámicamente
el comportamiento del sistema ante la presión de datos o fallos de infraestructura.

Ciclo Cognitivo (OODA Loop):
----------------------------
1. Observe (Observar): 
   Recolecta telemetría cruda del `FluxPhysicsEngine` (voltaje, saturación) y 
   el estado de conectividad de los microservicios.

2. Orient (Orientar):
   Utiliza el `TopologicalAnalyzer` para mapear el "Terreno Operativo". Aplica 
   Homología Persistente para distinguir entre "Ruido Transitorio" (picos ignorables) 
   y "Características Estructurales" (fallos reales o saturación sistémica).

3. Decide (Decidir):
   Evalúa la situación frente a una Matriz de Decisiones. Determina si el sistema 
   debe continuar (HEARTBEAT), frenar (RECOMENDAR_REDUCIR_VELOCIDAD) o reiniciar 
   conexiones (RECONNECT), priorizando la supervivencia del pipeline.

4. Act (Actuar):
   Ejecuta vectores de transformación sobre la infraestructura a través de la API, 
   cerrando el bucle de control y registrando el impacto para el siguiente ciclo.
"""

import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, Callable
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
# LOGGING CONFIGURATION
# ============================================================================


def setup_logging() -> logging.Logger:
    """Configura y retorna el logger del agente."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger("AutonomousAgent")


logger = setup_logging()

# ============================================================================
# ENUMS - Estados y Decisiones tipados
# ============================================================================


class SystemStatus(Enum):
    """Estados posibles del sistema monitoreado."""

    NOMINAL = auto()
    INESTABLE = auto()
    SATURADO = auto()
    CRITICO = auto()
    UNKNOWN = auto()
    DISCONNECTED = auto()


class AgentDecision(Enum):
    """Decisiones que el agente puede tomar."""

    HEARTBEAT = auto()
    EJECUTAR_LIMPIEZA = auto()
    AJUSTAR_VELOCIDAD = auto()
    ALERTA_CRITICA = auto()
    WAIT = auto()
    RECONNECT = auto()


# ============================================================================
# DATA CLASSES - Estructuras de datos tipadas
# ============================================================================


@dataclass(frozen=True)
class ThresholdConfig:
    """
    Configuración inmutable de umbrales para análisis de telemetría.

    Attributes:
        flyback_voltage_warning: Umbral de advertencia para voltaje (default: 0.5)
        flyback_voltage_critical: Umbral crítico para voltaje (default: 0.8)
        saturation_warning: Umbral de advertencia para saturación (default: 0.9)
        saturation_critical: Umbral crítico para saturación (default: 0.95)
    """

    flyback_voltage_warning: float = 0.5
    flyback_voltage_critical: float = 0.8
    saturation_warning: float = 0.9
    saturation_critical: float = 0.95

    def __post_init__(self) -> None:
        """Valida coherencia de umbrales tras inicialización."""
        self._validate_threshold_pair(
            "flyback_voltage", self.flyback_voltage_warning, self.flyback_voltage_critical
        )
        self._validate_threshold_pair(
            "saturation", self.saturation_warning, self.saturation_critical
        )

    @staticmethod
    def _validate_threshold_pair(name: str, warning: float, critical: float) -> None:
        """Valida que un par de umbrales sea coherente."""
        if not (0 <= warning < critical <= 1.0):
            raise ValueError(
                f"{name} thresholds inválidos: "
                f"debe cumplir 0 <= warning({warning}) < critical({critical}) <= 1.0"
            )


@dataclass
class TelemetryData:
    """
    Datos de telemetría estructurados y validados.

    Attributes:
        flyback_voltage: Voltaje de flyback normalizado [0, 1]
        saturation: Nivel de saturación normalizado [0, 1]
        timestamp: Momento de la captura
        raw_data: Datos originales sin procesar
    """

    flyback_voltage: float
    saturation: float
    timestamp: datetime = field(default_factory=datetime.now)
    integrity_score: float = 1.0
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Clampea valores al rango válido [0, 1]."""
        self.flyback_voltage = max(0.0, min(1.0, self.flyback_voltage))
        self.saturation = max(0.0, min(1.0, self.saturation))
        self.integrity_score = max(0.0, min(1.0, self.integrity_score))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["TelemetryData"]:
        """
        Factory method con extracción funcional y proyección al espacio de métricas.

        Implementa un functor desde el espacio de datos crudos hacia el espacio
        normalizado [0,1]², preservando la estructura mediante defaults seguros.
        """
        if not isinstance(data, dict):
            logger.warning(f"[TELEMETRY] Morfismo inválido: {type(data).__name__} ∉ Dict")
            return None

        # Definir el espacio de búsqueda como lista de proyecciones ordenadas por prioridad
        metric_paths: Dict[str, Tuple[str, ...]] = {
            "flyback": ("flux_condenser.max_flyback_voltage", "flyback_voltage", "voltage"),
            "saturation": ("flux_condenser.avg_saturation", "saturation", "sat"),
        }

        def extract_metric(source: Dict[str, Any], paths: Tuple[str, ...]) -> Optional[float]:
            """Proyección con fallback a través de caminos alternativos."""
            metrics_ns = source.get("metrics", source)
            search_space = metrics_ns if isinstance(metrics_ns, dict) else source

            for path in paths:
                if (value := search_space.get(path)) is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
            return None

        flyback = extract_metric(data, metric_paths["flyback"])
        saturation = extract_metric(data, metric_paths["saturation"])

        # Proyección al punto base (0,0) del espacio si no hay datos
        is_idle = flyback is None or saturation is None
        if is_idle:
            logger.debug("[TELEMETRY] Proyectando al origen: estado IDLE (0.0, 0.0)")

        flyback = flyback if flyback is not None else 0.0
        saturation = saturation if saturation is not None else 0.0

        # Extraer integridad si existe
        integrity = float(data.get("integrity_score", 1.0))

        # Advertencias para valores fuera del compacto [0,1]
        for name, val in [("flyback_voltage", flyback), ("saturation", saturation)]:
            if not (0.0 <= val <= 1.0):
                logger.warning(f"[TELEMETRY] {name}={val:.4f} ∉ [0,1]")

        return cls(
            flyback_voltage=flyback,
            saturation=saturation,
            integrity_score=integrity,
            raw_data=data
        )


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

    @property
    def success_rate(self) -> float:
        """Calcula la tasa de éxito de observaciones."""
        total = self.successful_observations + self.failed_observations
        return self.successful_observations / total if total > 0 else 0.0

    @property
    def uptime_seconds(self) -> float:
        """Retorna el tiempo de ejecución en segundos."""
        return (datetime.now() - self.start_time).total_seconds()

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
            "decisions_count": self.decisions_count.copy(),
            "uptime_seconds": round(self.uptime_seconds, 2),
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

    @property
    def is_structurally_healthy(self) -> bool:
        """Sistema estructuralmente sano (conectado, sin fragmentación)."""
        return self.health.betti.is_connected

    @property
    def has_retry_loops(self) -> bool:
        """Hay patrones de reintentos detectados."""
        return len(self.health.request_loops) > 0

    def to_log_dict(self) -> Dict[str, Any]:
        """Serializa para logging estructurado."""
        return {
            "betti": {"b0": self.health.betti.b0},
            "health_score": round(self.health.health_score, 3),
            "health_level": self.health.level.name,
            "voltage_state": self.voltage_persistence.state.name,
            "saturation_state": self.saturation_persistence.state.name,
            "disconnected_nodes": list(self.health.disconnected_nodes),
            "retry_loops": len(self.health.request_loops),
            "recommended_status": self.recommended_status.name,
        }


# ============================================================================
# AUTONOMOUS AGENT - Implementación Principal
# ============================================================================


class AutonomousAgent:
    """
    Agente autónomo que opera bajo un ciclo OODA (Observe, Orient, Decide, Act).

    Monitorea la salud del Core y toma decisiones basadas en métricas de telemetría.

    Características:
        - Ciclo OODA continuo con manejo robusto de errores
        - Configuración flexible via variables de entorno
        - Reintentos automáticos con backoff exponencial
        - Graceful shutdown ante señales del sistema
        - Métricas internas para observabilidad
        - Debounce de decisiones para evitar spam

    Environment Variables:
        CORE_API_URL: URL del API del Core (default: http://localhost:5002)
        CHECK_INTERVAL: Intervalo entre ciclos en segundos (default: 10)
        REQUEST_TIMEOUT: Timeout de requests en segundos (default: 5)
        LOG_LEVEL: Nivel de logging (default: INFO)
    """

    # Configuración por defecto
    DEFAULT_CORE_URL: str = "http://localhost:5002"
    DEFAULT_CHECK_INTERVAL: int = 10
    DEFAULT_REQUEST_TIMEOUT: int = 10
    MAX_CONSECUTIVE_FAILURES: int = 5
    DEBOUNCE_WINDOW_SECONDS: int = 60

    # Nuevas constantes para análisis topológico
    TOPOLOGY_HEALTH_CRITICAL_THRESHOLD: float = 0.4
    TOPOLOGY_HEALTH_WARNING_THRESHOLD: float = 0.7
    PERSISTENCE_WINDOW_SIZE: int = 20

    def __init__(
        self,
        core_api_url: Optional[str] = None,
        check_interval: Optional[int] = None,
        request_timeout: Optional[int] = None,
        thresholds: Optional[ThresholdConfig] = None,
        persistence_window: Optional[int] = None,
    ) -> None:
        """
        Inicializa el agente autónomo.

        Args:
            core_api_url: URL del API del Core
            check_interval: Intervalo entre ciclos (segundos)
            request_timeout: Timeout de requests (segundos)
            thresholds: Configuración de umbrales de análisis
            persistence_window: Tamaño de ventana para homología persistente

        Raises:
            ValueError: Si la configuración es inválida
        """
        # Configuración de conexión
        self.core_api_url = self._validate_and_normalize_url(
            core_api_url or os.getenv("CORE_API_URL", self.DEFAULT_CORE_URL)
        )
        logger.debug(f"DEBUG: Connecting to Core API at: {self.core_api_url}")
        self.telemetry_endpoint = f"{self.core_api_url}/api/telemetry/status"

        # Configuración de tiempos
        self.check_interval = self._parse_positive_int(
            check_interval,
            os.getenv("CHECK_INTERVAL"),
            self.DEFAULT_CHECK_INTERVAL,
            "check_interval",
        )
        self.request_timeout = self._parse_positive_int(
            request_timeout,
            os.getenv("REQUEST_TIMEOUT"),
            self.DEFAULT_REQUEST_TIMEOUT,
            "request_timeout",
        )

        # Configuración de umbrales
        self.thresholds = thresholds or ThresholdConfig()

        # Estado interno del agente
        self._running: bool = False
        self._last_decision: Optional[AgentDecision] = None
        self._last_decision_time: Optional[datetime] = None
        self._last_status: Optional[SystemStatus] = None
        self._last_diagnosis: Optional[TopologicalDiagnosis] = None
        self._metrics = AgentMetrics()

        # Configuración de ventana de persistencia
        window_size = persistence_window or self._parse_positive_int(
            None,
            os.getenv("PERSISTENCE_WINDOW_SIZE"),
            self.PERSISTENCE_WINDOW_SIZE,
            "persistence_window",
        )

        # Componentes de análisis topológico
        self.topology = SystemTopology(
            max_history=100,  # Historial amplio para detección de loops
        )
        self.persistence = PersistenceHomology(window_size=window_size)

        # Establecer topología inicial esperada
        self._initialize_expected_topology()

        # Sesión HTTP con reintentos
        self._session = self._create_robust_session()

        # Manejadores de señales
        self._original_handlers: Dict[signal.Signals, Any] = {}
        self._setup_signal_handlers()

        logger.info(
            f"AutonomousAgent inicializado | "
            f"Core: {self.core_api_url} | "
            f"Intervalo: {self.check_interval}s | "
            f"Timeout: {self.request_timeout}s | "
            f"Ventana Persistencia: {window_size}"
        )

    def _initialize_expected_topology(self) -> None:
        """
        Establece la topología inicial esperada del sistema.

        La topología esperada representa el estado ideal del sistema:
        Agent -> Core -> {Redis, Filesystem}
        """
        initial_connections = [
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem"),
        ]

        edges_added, warnings = self.topology.update_connectivity(
            initial_connections, validate_nodes=True, auto_add_nodes=True
        )

        if warnings:
            for warn in warnings:
                logger.warning(f"[TOPO-INIT] {warn}")

        logger.debug(
            f"[TOPO-INIT] Topología inicial establecida: {edges_added} conexiones activas"
        )

    @staticmethod
    def _validate_and_normalize_url(url: str) -> str:
        """
        Valida y normaliza la URL del API.

        Args:
            url: URL a validar

        Returns:
            URL normalizada

        Raises:
            ValueError: Si la URL es inválida
        """
        if not url or not url.strip():
            raise ValueError("CORE_API_URL no puede estar vacía")

        url = url.strip()

        # Agregar esquema si falta
        if not url.lower().startswith(("http://", "https://")):
            url = f"http://{url}"

        # Validar estructura
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                raise ValueError(f"URL sin host válido: {url}")
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Esquema no soportado: {parsed.scheme}")
        except Exception as e:
            raise ValueError(f"URL inválida '{url}': {e}")

        return url.rstrip("/")

    @staticmethod
    def _parse_positive_int(
        explicit: Optional[int], env_value: Optional[str], default: int, name: str
    ) -> int:
        """
        Parsea un entero positivo desde múltiples fuentes.

        Prioridad: explicit > env_value > default
        """
        if explicit is not None:
            if not isinstance(explicit, int) or explicit <= 0:
                raise ValueError(f"{name} debe ser un entero positivo")
            return explicit

        if env_value is not None:
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

    def _create_robust_session(self) -> requests.Session:
        """
        Crea una sesión HTTP con política de reintentos y backoff.

        Returns:
            Sesión configurada con retry logic
        """
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,  # 0.5s, 1s, 2s
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy, pool_connections=10, pool_maxsize=10
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Configurar cabeceras predeterminadas (Pasaporte Interno)
        session.headers.update(
            {"User-Agent": "APU-Agent-Internal", "Content-Type": "application/json"}
        )

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
        """
        Manejador de señales de terminación.

        Args:
            signum: Número de señal recibida
            frame: Stack frame (no usado)
        """
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = str(signum)

        logger.info(f"Señal {sig_name} recibida. Iniciando shutdown graceful...")
        self._running = False

    def _analyze_metric_persistence(
        self, metric_name: str, current_value: Optional[float], threshold: float
    ) -> PersistenceAnalysisResult:
        """
        Analiza la persistencia de una métrica alimentando nuevos datos.

        Args:
            metric_name: Nombre de la métrica
            current_value: Valor actual (None si no hay datos)
            threshold: Umbral para análisis de excursiones

        Returns:
            Resultado del análisis de persistencia
        """
        # Alimentar nuevo dato si existe
        if current_value is not None:
            self.persistence.add_reading(metric_name, current_value)

        # Obtener análisis
        return self.persistence.analyze_persistence(
            metric_name, threshold=threshold, noise_ratio=0.2, critical_ratio=0.5
        )

    # =========================================================================
    # OODA LOOP - Métodos principales (REFINADOS)
    # =========================================================================

    def observe(self) -> Optional[TelemetryData]:
        """
        OBSERVE - Primera fase del ciclo OODA.

        Implementa la observación como un morfismo O: Infraestructura → Telemetría
        con manejo de errores que preserva la coherencia topológica del sistema.
        """
        request_id = f"obs_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        # Definir el espacio de errores como mapeo a handlers uniformes
        error_handlers: Dict[type, Tuple[str, str]] = {
            requests.exceptions.Timeout: ("TIMEOUT", f"después de {self.request_timeout}s"),
            requests.exceptions.ConnectionError: ("CONNECTION_ERROR", "conexión rechazada"),
            requests.exceptions.RequestException: ("REQUEST_ERROR", "error de request"),
        }

        try:
            response = self._session.get(
                self.telemetry_endpoint, timeout=self.request_timeout
            )

            if not response.ok:
                self._handle_observation_failure(request_id, f"HTTP_{response.status_code}")
                logger.warning(f"[OBSERVE] HTTP {response.status_code}")
                return None

            try:
                raw_data = response.json()
            except ValueError as e:
                self._handle_observation_failure(request_id, "INVALID_JSON")
                logger.warning(f"[OBSERVE] JSON inválido: {e}")
                return None

            if (telemetry := TelemetryData.from_dict(raw_data)) is None:
                self._handle_observation_failure(request_id, "INVALID_TELEMETRY")
                return None

            self._handle_observation_success(request_id, telemetry)
            return telemetry

        except tuple(error_handlers.keys()) as e:
            error_type, msg = error_handlers.get(type(e), ("UNKNOWN", str(e)))
            logger.warning(f"[OBSERVE] {error_type}: {msg}")
            self._handle_observation_failure(request_id, error_type)
            return None


    def _handle_observation_result(
        self, request_id: str, telemetry: Optional[TelemetryData], failure_type: Optional[str]
    ) -> None:
        """
        Unifica el manejo de resultados de observación preservando invariantes.

        Actúa como un functor que mapea el resultado al espacio de métricas
        y actualiza la topología de manera coherente.
        """
        if telemetry is not None:
            # Morfismo de éxito: actualizar espacio de estados
            self._metrics.record_success()
            self.topology.record_request(request_id)

            # Inferir conectividad desde telemetría
            raw = telemetry.raw_data
            active_connections = [("Agent", "Core")]

            # Extensión del grafo según estado reportado
            if raw.get("redis_connected", True):
                active_connections.append(("Core", "Redis"))
            if raw.get("filesystem_accessible", True):
                active_connections.append(("Core", "Filesystem"))

            self.topology.update_connectivity(
                active_connections, validate_nodes=True, auto_add_nodes=False
            )
            self.topology.clear_request_history()

            logger.debug(
                f"[OBSERVE] ✓ v={telemetry.flyback_voltage:.3f}, s={telemetry.saturation:.3f}"
            )
        else:
            # Morfismo de fallo: degradar espacio topológico
            self._metrics.record_failure()
            self.topology.record_request(f"FAIL_{failure_type}")

            if self._metrics.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                logger.warning(f"[TOPO] Degradación: {self._metrics.consecutive_failures} fallos")
                self.topology.remove_edge("Agent", "Core")


    def _handle_observation_success(self, request_id: str, telemetry: TelemetryData) -> None:
        """Wrapper para compatibilidad - delega al método unificado."""
        self._handle_observation_result(request_id, telemetry, None)


    def _handle_observation_failure(self, request_id: str, failure_type: str) -> None:
        """Wrapper para compatibilidad - delega al método unificado."""
        self._handle_observation_result(request_id, None, failure_type)

    def orient(self, telemetry: Optional[TelemetryData]) -> SystemStatus:
        """
        ORIENT - Segunda fase del ciclo OODA (Motor Topológico).

        Implementa el morfismo de orientación O: T × H → S donde:
        - T = espacio de telemetría
        - H = espacio de homología persistente
        - S = espacio de estados del sistema

        La composición preserva la estructura algebraica del diagnóstico.
        """
        # Calcular invariantes topológicos (β₀ suficiente para conectividad)
        topo_health = self.topology.get_topological_health(calculate_b1=False)

        # Proyectar métricas al espacio de persistencia
        analyses = {
            "voltage": self._analyze_metric_persistence(
                "flyback_voltage",
                telemetry.flyback_voltage if telemetry else None,
                self.thresholds.flyback_voltage_warning,
            ),
            "saturation": self._analyze_metric_persistence(
                "saturation",
                telemetry.saturation if telemetry else None,
                self.thresholds.saturation_warning,
            ),
        }

        # Evaluar estado mediante composición de diagnósticos
        status, summary = self._evaluate_system_state(
            telemetry, topo_health, analyses["voltage"], analyses["saturation"]
        )

        # Construir y almacenar diagnóstico como elemento del fibrado
        self._last_diagnosis = TopologicalDiagnosis(
            health=topo_health,
            voltage_persistence=analyses["voltage"],
            saturation_persistence=analyses["saturation"],
            summary=summary,
            recommended_status=status,
        )

        if status != SystemStatus.NOMINAL:
            logger.info(f"[ORIENT] {self._last_diagnosis.to_log_dict()}")

        return status


    def _evaluate_system_state(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
    ) -> Tuple[SystemStatus, str]:
        """
        Evalúa el estado del sistema mediante una cadena de evaluadores ordenados.

        Implementa un retículo de decisiones donde cada evaluador es un morfismo
        parcial que puede retornar un estado o delegar al siguiente en la jerarquía.

        La jerarquía forma un orden parcial (poset) por criticidad:
        FRAGMENTACIÓN > CRÍTICO > SATURADO > INESTABLE > UNKNOWN > NOMINAL
        """
        # Definir evaluadores como tuplas (condición, estado, generador_de_resumen)
        # Cada evaluador es un morfismo parcial E: (T, H, P) → S ∪ {⊥}

        evaluators: list[tuple[Callable[[], bool], SystemStatus, Callable[[], str]]] = [
            # 1. Fragmentación topológica (β₀ > 1)
            (
                lambda: not topo_health.betti.is_connected,
                SystemStatus.DISCONNECTED,
                lambda: (
                    f"Fragmentación Topológica: β₀={topo_health.betti.b0}. "
                    f"Nodos: [{', '.join(topo_health.disconnected_nodes) or '∅'}]"
                ),
            ),
            # 2. Sin telemetría
            (
                lambda: telemetry is None,
                SystemStatus.UNKNOWN,
                lambda: (
                    f"Sin telemetría ({self._metrics.consecutive_failures} fallos)"
                    if self._metrics.consecutive_failures > 0
                    else "Esperando telemetría"
                ),
            ),
            # 3. Voltaje crítico instantáneo (safety net)
            (
                lambda: telemetry is not None and telemetry.flyback_voltage >= self.thresholds.flyback_voltage_critical,
                SystemStatus.CRITICO,
                lambda: f"Voltaje crítico: {telemetry.flyback_voltage:.3f} >= {self.thresholds.flyback_voltage_critical}",
            ),
            # 4. Saturación crítica instantánea (safety net)
            (
                lambda: telemetry is not None and telemetry.saturation >= self.thresholds.saturation_critical,
                SystemStatus.CRITICO,
                lambda: f"Saturación crítica: {telemetry.saturation:.3f} >= {self.thresholds.saturation_critical}",
            ),
            # 5. Salud topológica crítica
            (
                lambda: topo_health.level == HealthLevel.CRITICAL,
                SystemStatus.CRITICO,
                lambda: f"Salud topológica crítica: score={topo_health.health_score:.2f}",
            ),
            # 6. Saturación persistente CRITICAL
            (
                lambda: saturation_analysis.state == MetricState.CRITICAL,
                SystemStatus.SATURADO,
                lambda: f"Saturación persistente: {saturation_analysis.metadata.get('active_duration', '?')} muestras",
            ),
            # 7. Saturación con característica estructural (FEATURE)
            (
                lambda: saturation_analysis.state == MetricState.FEATURE,
                SystemStatus.SATURADO,
                lambda: (
                    f"Patrón estructural saturación: {saturation_analysis.feature_count} feature(s), "
                    f"π={saturation_analysis.total_persistence:.1f}"
                ),
            ),
            # 8. Voltaje persistente CRITICAL
            (
                lambda: voltage_analysis.state == MetricState.CRITICAL,
                SystemStatus.INESTABLE,
                lambda: f"Inestabilidad voltaje: {voltage_analysis.metadata.get('active_duration', '?')} muestras",
            ),
            # 9. Voltaje con característica estructural
            (
                lambda: voltage_analysis.state == MetricState.FEATURE,
                SystemStatus.INESTABLE,
                lambda: f"Patrón estructural voltaje: λ_max={voltage_analysis.max_lifespan:.1f}",
            ),
            # 10. Loops de reintentos significativos
            (
                lambda: (
                    len(topo_health.request_loops) > 0
                    and topo_health.request_loops[0].count >= 5
                    and topo_health.request_loops[0].request_id.startswith("FAIL_")
                ),
                SystemStatus.INESTABLE,
                lambda: f"Patrón reintentos: '{topo_health.request_loops[0].request_id}' ×{topo_health.request_loops[0].count}",
            ),
            # 11. Salud degradada
            (
                lambda: topo_health.level == HealthLevel.UNHEALTHY,
                SystemStatus.INESTABLE,
                lambda: f"Salud degradada: score={topo_health.health_score:.2f}",
            ),
        ]

        # Recorrer la cadena de evaluadores (primer match gana)
        for condition, status, summary_fn in evaluators:
            try:
                if condition():
                    summary = summary_fn()
                    log_level = logging.CRITICAL if status == SystemStatus.CRITICO else logging.WARNING
                    logger.log(log_level, f"[EVAL] {summary}")
                    return status, summary
            except Exception:
                continue  # Evaluador falló, continuar con siguiente

        # Log de ruido filtrado (inmunidad a falsos positivos)
        for name, analysis in [("voltaje", voltage_analysis), ("saturación", saturation_analysis)]:
            if analysis.state == MetricState.NOISE:
                logger.debug(f"[PERSIST] Ruido {name} filtrado: {analysis.noise_count} excursiones")

        # Estado nominal (punto fijo del sistema)
        return (
            SystemStatus.NOMINAL,
            f"Sistema nominal: β₀={topo_health.betti.b0}, h={topo_health.health_score:.2f}",
        )

    def decide(self, status: SystemStatus) -> AgentDecision:
        """
        DECIDE - Tercera fase del ciclo OODA.

        Implementa el morfismo de decisión D: S × C → A donde:
        - S = espacio de estados
        - C = contexto topológico (diagnóstico previo)
        - A = espacio de acciones

        La decisión es una función que preserva la estructura del problema.
        """
        # Matriz de decisión base (morfismo S → A)
        decision_matrix: Dict[SystemStatus, AgentDecision] = {
            SystemStatus.NOMINAL: AgentDecision.HEARTBEAT,
            SystemStatus.INESTABLE: AgentDecision.EJECUTAR_LIMPIEZA,
            SystemStatus.SATURADO: AgentDecision.AJUSTAR_VELOCIDAD,
            SystemStatus.CRITICO: AgentDecision.ALERTA_CRITICA,
            SystemStatus.DISCONNECTED: AgentDecision.RECONNECT,
            SystemStatus.UNKNOWN: AgentDecision.WAIT,
        }

        decision = decision_matrix.get(status, AgentDecision.WAIT)

        # Refinamiento contextual: modular decisión según diagnóstico topológico
        if self._last_diagnosis and decision == AgentDecision.HEARTBEAT:
            # Analizar campo vectorial de errores en el historial
            error_loops = [
                loop for loop in self._last_diagnosis.health.request_loops
                if loop.request_id.startswith("FAIL_")
            ]

            if error_loops:
                total_errors = sum(loop.count for loop in error_loops)
                logger.debug(
                    f"[DECIDE] Nominal con {len(error_loops)} patrones de error "
                    f"(Σ={total_errors} eventos)"
                )
                # Potencial escalamiento futuro: considerar WAIT si errores recientes

        self._metrics.record_decision(decision)
        self._last_status = status

        return decision

    def act(self, decision: AgentDecision) -> bool:
        """
        ACT - Cuarta fase del ciclo OODA.

        Implementa el morfismo de acción A: D × Σ → Ω donde:
        - D = espacio de decisiones
        - Σ = estado del diagnóstico
        - Ω = espacio de efectos (side effects sobre infraestructura)

        Incluye debounce como operador de suavizado temporal.
        """
        if self._should_debounce(decision):
            logger.debug(f"[ACT] Suprimido por debounce: {decision.name}")
            return False

        diagnosis_msg = self._build_diagnosis_message()

        # Tabla de handlers como morfismos parciales
        action_handlers: Dict[AgentDecision, Callable[[], None]] = {
            AgentDecision.HEARTBEAT: lambda: self._emit_heartbeat(),
            AgentDecision.EJECUTAR_LIMPIEZA: lambda: self._execute_cleanup(diagnosis_msg),
            AgentDecision.AJUSTAR_VELOCIDAD: lambda: self._apply_backpressure(diagnosis_msg),
            AgentDecision.ALERTA_CRITICA: lambda: self._raise_critical_alert(diagnosis_msg),
            AgentDecision.RECONNECT: lambda: self._attempt_reconnection(diagnosis_msg),
            AgentDecision.WAIT: lambda: logger.info("[BRAIN] ⏳ Esperando telemetría..."),
        }

        handler = action_handlers.get(decision, action_handlers[AgentDecision.WAIT])
        handler()

        # Actualizar estado temporal para debounce
        self._last_decision = decision
        self._last_decision_time = datetime.now()

        return True

    def _emit_heartbeat(self) -> None:
        """Emite señal de sistema nominal con indicador de salud."""
        health_score = self._last_diagnosis.health.health_score if self._last_diagnosis else 1.0
        indicator = "✅" if health_score >= 0.9 else "🟢" if health_score >= 0.7 else "🟡"
        logger.info(f"[BRAIN] {indicator} NOMINAL - h={health_score:.2f}")


    def _execute_cleanup(self, diagnosis_msg: str) -> None:
        """Proyecta vector de limpieza al estrato físico."""
        logger.warning(f"[BRAIN] ⚠️ INESTABILIDAD - {diagnosis_msg}")

        success = self._project_intent(
            vector="clean",
            stratum="PHYSICS",
            payload={"mode": "EMERGENCY", "reason": diagnosis_msg, "scope": "flux_condenser"},
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
                "factor": 0.5,  # Factor de reducción (homotecia)
            },
        )

        event = "saturation_mitigated" if success else "saturation_correction_failed"
        self._notify_external_system(event, {"method": "throttle"})


    def _raise_critical_alert(self, diagnosis_msg: str) -> None:
        """Emite alerta crítica con contexto topológico completo."""
        logger.critical(f"[BRAIN] 🚨 CRÍTICO - {diagnosis_msg}")
        logger.critical("[BRAIN] → Intervención inmediata requerida")

        context = {"diagnosis": diagnosis_msg}
        if self._last_diagnosis:
            context.update({
                "health_score": self._last_diagnosis.health.health_score,
                "betti_b0": self._last_diagnosis.health.betti.b0,
                "is_connected": self._last_diagnosis.is_structurally_healthy,
            })

        self._notify_external_system("critical_alert", context)


    def _attempt_reconnection(self, diagnosis_msg: str) -> None:
        """Intenta reconexión reinicializando topología esperada."""
        logger.warning(f"[BRAIN] 🔄 DESCONEXIÓN - {diagnosis_msg}")
        logger.warning("[BRAIN] → Reinicializando topología...")
        self._initialize_expected_topology()

    def _should_debounce(self, decision: AgentDecision) -> bool:
        """
        Determina si una acción debe ser suprimida por debounce.

        Las alertas críticas y reconexiones nunca se suprimen.

        Args:
            decision: Decisión a evaluar

        Returns:
            True si debe suprimirse, False en caso contrario
        """
        # Decisiones que nunca se suprimen
        always_execute = {AgentDecision.ALERTA_CRITICA, AgentDecision.RECONNECT}
        if decision in always_execute:
            return False

        # Sin decisión previa = no suprimir
        if self._last_decision is None or self._last_decision_time is None:
            return False

        # Decisión diferente = no suprimir
        if decision != self._last_decision:
            return False

        # Verificar ventana de tiempo
        elapsed = datetime.now() - self._last_decision_time
        return elapsed < timedelta(seconds=self.DEBOUNCE_WINDOW_SECONDS)

    def _project_intent(self, vector: str, stratum: str, payload: Dict[str, Any]) -> bool:
        """
        Proyecta intención sobre la MIC como morfismo I: V × S × P → {⊤, ⊥}.

        Args:
            vector: Nombre del vector (herramienta) - elemento del espacio de acciones
            stratum: Nivel de gobernanza - fibra del haz de control
            payload: Datos específicos - sección local del haz

        Returns:
            True si la proyección fue exitosa (imagen en ⊤)
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
                    if self._last_diagnosis else None
                ),
            },
        }

        url = f"{self.core_api_url}/api/tools/{vector}"
        logger.info(f"[INTENT] Proyectando '{vector}' → estrato '{stratum}'")

        try:
            response = self._session.post(url, json=intent, timeout=self.request_timeout)

            if response.ok:
                logger.info(f"[INTENT] ✅ {vector} ejecutado exitosamente")
                return True

            logger.error(f"[INTENT] ❌ HTTP {response.status_code}: {response.text[:100]}")
            return False

        except requests.exceptions.RequestException as e:
            logger.error(f"[INTENT] Error de proyección: {type(e).__name__}")
            return False

    def _build_diagnosis_message(self) -> str:
        """
        Construye mensaje de diagnóstico como proyección del fibrado topológico.

        Serializa los invariantes relevantes del diagnóstico actual.
        """
        if not self._last_diagnosis:
            return "Sin diagnóstico"

        diag = self._last_diagnosis
        components = [diag.summary]

        # Añadir invariantes topológicos si son informativos
        betti = diag.health.betti
        if not betti.is_ideal:
            components.append(f"β₀={betti.b0}")

        if diag.health.health_score < 0.9:
            components.append(f"h={diag.health.health_score:.2f}")

        return " | ".join(components)

    def _notify_external_system(
        self, event_type: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Hook para notificaciones externas (webhooks, métricas, etc).

        Args:
            event_type: Tipo de evento a notificar
            context: Contexto adicional del evento
        """
        # Registrar para métricas/observabilidad
        log_data = {"event": event_type}
        if context:
            log_data.update(context)

        logger.debug(f"[NOTIFY] {log_data}")

        # Placeholder para integración futura con:
        # - Webhooks
        # - Sistemas de alertas (PagerDuty, OpsGenie)
        # - Métricas (Prometheus, DataDog)

    # =========================================================================
    # LIFECYCLE METHODS - Control del ciclo de vida
    # =========================================================================

    def health_check(self) -> bool:
        """
        Verifica conectividad con el Core y estado topológico inicial.

        Returns:
            True si el Core es accesible y la topología es válida
        """
        logger.info(f"Ejecutando health check: {self.telemetry_endpoint}")

        try:
            response = self._session.get(
                self.telemetry_endpoint, timeout=self.request_timeout
            )

            if response.ok:
                # Actualizar topología con conexión confirmada
                self._initialize_expected_topology()

                # Verificar salud topológica (modo sistema)
                topo_health = self.topology.get_topological_health(calculate_b1=False)

                logger.info(
                    f"✅ Health check exitoso - "
                    f"Core accesible, topología: {topo_health.level.name} "
                    f"(score={topo_health.health_score:.2f})"
                )
                return True
            else:
                logger.warning(
                    f"⚠️ Health check con advertencia: HTTP {response.status_code}"
                )
                return True  # Permitir continuar con warning

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Health check fallido: {e}")

            # Degradar topología
            self.topology.remove_edge("Agent", "Core")
            topo_health = self.topology.get_topological_health(calculate_b1=False)

            logger.error(
                f"Topología degradada: β₀={topo_health.betti.b0}, "
                f"health={topo_health.health_score:.2f}"
            )
            return False

    def get_stratum_health(self, stratum: Stratum) -> Dict[str, Any]:
        """
        Retorna la salud filtrada por estrato (Pirámide de Observabilidad).

        Args:
            stratum: Nivel jerárquico a consultar.

        Returns:
            Dict con métricas específicas del nivel.
        """
        # PHYSICS: Métricas de FluxCondenser
        if stratum == Stratum.PHYSICS:
            # Intentar obtener última observación (o usar una ligera)
            # Para evitar overhead, usamos estado interno si es reciente, o una nueva observación
            # si es explícitamente solicitada. Aquí asumimos que queremos el estado actual real.
            obs = self.observe()
            status = "UNKNOWN"
            if obs:
                # Determinar estado basado en umbrales simples para coherencia con el test
                if (obs.flyback_voltage >= self.thresholds.flyback_voltage_critical or
                    obs.saturation >= self.thresholds.saturation_critical):
                    status = "CRITICO"  # O CRITICAL, pero el enum es CRITICO. Usamos string para el dict.
                elif (obs.flyback_voltage >= self.thresholds.flyback_voltage_warning or
                      obs.saturation >= self.thresholds.saturation_warning):
                    status = "WARNING"
                else:
                    status = "NOMINAL"

            return {
                "stratum": "PHYSICS",
                "voltage": obs.flyback_voltage if obs else None,
                "saturation": obs.saturation if obs else None,
                "status": status,
                "integrity": obs.integrity_score if obs else 0.0,
                "timestamp": obs.timestamp.isoformat() if obs else None,
            }

        # TACTICS: Métricas Topológicas
        elif stratum == Stratum.TACTICS:
            health = self.topology.get_topological_health(calculate_b1=True)
            # Safe access to nested attributes
            betti = getattr(health, "betti", None)
            b0 = getattr(betti, "b0", 0) if betti else 0
            b1 = getattr(betti, "b1", 0) if betti else 0
            is_connected = getattr(betti, "is_connected", False) if betti else False
            euler = getattr(betti, "euler_characteristic", 0) if betti else 0

            return {
                "stratum": "TACTICS",
                "betti_0": b0,
                "betti_1": b1,  # Ciclos
                "is_connected": is_connected,
                "health_score": round(health.health_score, 3),
                "euler": euler,
            }

        # STRATEGY: Estado Financiero (Si existe diagnóstico previo)
        elif stratum == Stratum.STRATEGY:
            # Basamos en si el diagnóstico actual reporta problemas financieros o sistémicos
            confidence = 0.0
            if self._last_decision and hasattr(self._last_decision, "confidence"):
                 confidence = self._last_decision.confidence
            elif self._last_diagnosis:
                 # Inferir confianza de la salud topológica
                 confidence = self._last_diagnosis.health.health_score

            status_age = 0.0
            if self._last_decision_time:
                status_age = (datetime.now() - self._last_decision_time).total_seconds()

            return {
                "stratum": "STRATEGY",
                "risk_detected": self._last_status in [SystemStatus.SATURADO, SystemStatus.CRITICO],
                "last_decision": self._last_decision.name if self._last_decision else None,
                "confidence": confidence,
                "status_age": status_age
            }

        # WISDOM: Veredicto Global
        elif stratum == Stratum.WISDOM:
            rationale = "Sin diagnóstico previo."
            if self._last_diagnosis:
                # Safe access to summary
                rationale = getattr(self._last_diagnosis, "summary", "Diagnóstico sin resumen")

            return {
                "stratum": "WISDOM",
                "verdict": self._last_status.name if self._last_status else "UNKNOWN",
                "certainty": 1.0 if self._last_diagnosis else 0.0,
                "rationale": rationale,
                "cycles_executed": self._metrics.cycles_executed
            }

        return {"error": "Invalid Stratum"}

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas completas como proyección al espacio de observabilidad.

        Construye un diccionario estructurado que representa el estado completo
        del agente en el espacio de métricas M = B × T × P × D donde:
        - B = métricas base del agente
        - T = invariantes topológicos
        - P = estadísticas de persistencia
        - D = último diagnóstico
        """
        # Base: métricas del agente
        metrics = self._metrics.to_dict()
        metrics.update({
            "core_api_url": self.core_api_url,
            "check_interval": self.check_interval,
            "is_running": self._running,
            "last_status": self._last_status.name if self._last_status else None,
        })

        # Topología: invariantes de la estructura
        topo_health = self.topology.get_topological_health(calculate_b1=False)
        metrics["topology"] = {
            "betti": {"b0": topo_health.betti.b0, "b1": topo_health.betti.b1},
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

        # Persistencia: estadísticas de series temporales
        persistence_data = {}
        for metric_name in ("flyback_voltage", "saturation"):
            if (stats := self.persistence.get_statistics(metric_name)):
                persistence_data[metric_name] = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in stats.items()
                    if k in ("count", "min", "max", "mean", "std")
                }

        if persistence_data:
            metrics["persistence"] = persistence_data

        # Diagnóstico: último análisis
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
        """
        Retorna resumen topológico para dashboards.

        Proyecta el estado del sistema al espacio de visualización
        preservando la interpretación semántica de los invariantes.
        """
        health = self.topology.get_topological_health()

        return {
            "timestamp": datetime.now().isoformat(),
            "betti": {
                "values": {"b0": health.betti.b0, "b1": health.betti.b1},
                "interpretation": (
                    "Sistema conectado" if health.betti.is_connected
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

    def _wait_for_startup(self) -> None:
        """
        Implementa el 'Modo de Espera de Arranque' para manejar el Cold Start del Core.
        Tolera 'Connection refused' y espera pacientemente.
        """
        logger.info("Iniciando protocolo de espera de arranque (Cold Start)...")

        backoff = 5

        while self._running:
            try:
                # Usamos una sesión fresca para el handshake inicial para evitar envenenamiento del pool
                response = requests.get(
                    self.telemetry_endpoint, timeout=self.request_timeout
                )

                if response.ok:
                    logger.info("✅ Core detectado y operativo (200 OK).")
                    return
                else:
                    logger.info(
                        f"Esperando a que el Core inicie... (HTTP {response.status_code})"
                    )

            except requests.exceptions.ConnectionError:
                # Manejo específico para Cold Start (Connection refused)
                logger.info(
                    "Esperando a que el Core inicie (Cold Start)... [Conexión rechazada]"
                )
            except requests.exceptions.RequestException as e:
                logger.info(f"Esperando disponibilidad del Core... [{type(e).__name__}]")

            # Backoff de 5 segundos como solicitado
            time.sleep(backoff)

    def run(self, skip_health_check: bool = False) -> None:
        """
        Bucle principal del agente - Ejecuta el ciclo OODA continuamente.

        Args:
            skip_health_check: Si True, omite verificación inicial
        """
        # Habilitar flag de ejecución para permitir shutdown durante espera
        self._running = True

        # Health check inicial con tolerancia a Cold Start
        if not skip_health_check:
            self._wait_for_startup()
            # Una vez conectado, ejecutamos el health check estándar para inicializar topología
            if not self.health_check():
                logger.warning("Iniciando agente con advertencias de salud...")

        logger.info("🚀 Iniciando OODA Loop...")

        try:
            while self._running:
                cycle_start = time.monotonic()
                self._metrics.increment_cycle()

                try:
                    # ═══════════════════════════════════════
                    # CICLO OODA
                    # ═══════════════════════════════════════

                    # 1. OBSERVE
                    telemetry = self.observe()

                    # 2. ORIENT
                    status = self.orient(telemetry)

                    # 3. DECIDE
                    decision = self.decide(status)

                    # 4. ACT
                    self.act(decision)

                except Exception as e:
                    logger.error(
                        f"Error en ciclo OODA #{self._metrics.cycles_executed}: {e}",
                        exc_info=True,
                    )

                # Sleep adaptativo (considera duración del ciclo)
                cycle_duration = time.monotonic() - cycle_start
                sleep_time = max(0.0, self.check_interval - cycle_duration)

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

        # Log de métricas finales
        final_metrics = self.get_metrics()
        logger.info(f"Métricas finales: {final_metrics}")

        # Cerrar sesión HTTP
        if self._session:
            try:
                self._session.close()
            except Exception as e:
                logger.warning(f"Error cerrando sesión HTTP: {e}")

        # Restaurar signal handlers
        self._restore_signal_handlers()

        logger.info("👋 AutonomousAgent detenido correctamente")


# ============================================================================
# ENTRY POINT
# ============================================================================


def main() -> int:
    """
    Punto de entrada principal.

    Returns:
        Código de salida (0=éxito, 1=error)
    """
    try:
        agent = AutonomousAgent()
        agent.run()
        return 0

    except ValueError as e:
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
