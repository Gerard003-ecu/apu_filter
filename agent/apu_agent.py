#!/usr/bin/env python3
"""
Autonomous Agent - OODA Loop Implementation
============================================
Agente autónomo que monitorea la salud del Core mediante el ciclo OODA.

Mejoras implementadas:
- Type hints completos
- Enums para estados y decisiones
- Configuración flexible via env vars
- Retry logic con backoff exponencial
- Graceful shutdown
- Debounce de acciones repetitivas
- Validación robusta de telemetría
- Métricas internas del agente
- Health check inicial
- Integración de Motor Topológico (Persistence Homology + Betti Numbers)
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
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agent.topological_analyzer import (
    SystemTopology,
    PersistenceHomology,
    MetricState,
    HealthLevel,
    BettiNumbers,
    TopologicalHealth,
    PersistenceAnalysisResult,
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
        handlers=[logging.StreamHandler(sys.stdout)]
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
    RECOMENDAR_LIMPIEZA = auto()
    RECOMENDAR_REDUCIR_VELOCIDAD = auto()
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
            "flyback_voltage",
            self.flyback_voltage_warning,
            self.flyback_voltage_critical
        )
        self._validate_threshold_pair(
            "saturation",
            self.saturation_warning,
            self.saturation_critical
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
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Clampea valores al rango válido [0, 1]."""
        self.flyback_voltage = max(0.0, min(1.0, self.flyback_voltage))
        self.saturation = max(0.0, min(1.0, self.saturation))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["TelemetryData"]:
        """
        Factory method: crea instancia desde diccionario con validación.
        
        Args:
            data: Diccionario con datos de telemetría
            
        Returns:
            TelemetryData si los datos son válidos, None en caso contrario
        """
        if not isinstance(data, dict):
            logger.warning(f"[TELEMETRY] Tipo inválido: esperado dict, recibido {type(data).__name__}")
            return None

        flyback = data.get("flyback_voltage")
        saturation = data.get("saturation")

        # Validar campos requeridos
        missing_fields = []
        if flyback is None:
            missing_fields.append("flyback_voltage")
        if saturation is None:
            missing_fields.append("saturation")
            
        if missing_fields:
            logger.warning(f"[TELEMETRY] Campos faltantes: {missing_fields}")
            return None

        # Intentar conversión a float
        try:
            flyback_float = float(flyback)
            saturation_float = float(saturation)
        except (TypeError, ValueError) as e:
            logger.warning(f"[TELEMETRY] Error de conversión numérica: {e}")
            return None

        # Log de advertencia si valores fuera de rango (pero continuar)
        if not (0 <= flyback_float <= 1.0):
            logger.warning(f"[TELEMETRY] flyback_voltage={flyback_float} fuera de [0,1]")
        if not (0 <= saturation_float <= 1.0):
            logger.warning(f"[TELEMETRY] saturation={saturation_float} fuera de [0,1]")

        return cls(
            flyback_voltage=flyback_float,
            saturation=saturation_float,
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
                if self.last_successful_observation else None
            ),
            "decisions_count": self.decisions_count.copy(),
            "uptime_seconds": round(self.uptime_seconds, 2)
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
            "betti": {"b0": self.health.betti.b0, "b1": self.health.betti.b1},
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
    DEFAULT_REQUEST_TIMEOUT: int = 5
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
        self.telemetry_endpoint = f"{self.core_api_url}/api/telemetry/status"

        # Configuración de tiempos
        self.check_interval = self._parse_positive_int(
            check_interval,
            os.getenv("CHECK_INTERVAL"),
            self.DEFAULT_CHECK_INTERVAL,
            "check_interval"
        )
        self.request_timeout = self._parse_positive_int(
            request_timeout,
            os.getenv("REQUEST_TIMEOUT"),
            self.DEFAULT_REQUEST_TIMEOUT,
            "request_timeout"
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
            "persistence_window"
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
            initial_connections,
            validate_nodes=True,
            auto_add_nodes=True
        )

        if warnings:
            for warn in warnings:
                logger.warning(f"[TOPO-INIT] {warn}")

        logger.debug(
            f"[TOPO-INIT] Topología inicial establecida: "
            f"{edges_added} conexiones activas"
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
        explicit: Optional[int],
        env_value: Optional[str],
        default: int,
        name: str
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
            raise_on_status=False
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)

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

    # =========================================================================
    # OODA LOOP - Métodos principales
    # =========================================================================

    def observe(self) -> Optional[TelemetryData]:
        """
        OBSERVE - Primera fase del ciclo OODA.
        
        Realiza una solicitud GET al endpoint de telemetría del Core.
        Registra el request para detección de patrones de reintentos.
        Actualiza la topología basándose en el resultado.
        
        Returns:
            TelemetryData si exitoso, None si hay error
        """
        # Generar ID único para tracking de este request
        request_id = f"obs_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        try:
            response = self._session.get(
                self.telemetry_endpoint,
                timeout=self.request_timeout
            )

            # Verificar código de respuesta
            if not response.ok:
                logger.warning(
                    f"[OBSERVE] HTTP {response.status_code}: "
                    f"{response.text[:100] if response.text else 'Sin cuerpo'}"
                )
                self._handle_observation_failure(request_id, f"HTTP_{response.status_code}")
                return None

            # Parsear JSON
            try:
                raw_data = response.json()
            except ValueError as e:
                logger.warning(f"[OBSERVE] Respuesta JSON inválida: {e}")
                self._handle_observation_failure(request_id, "INVALID_JSON")
                return None

            # Estructurar y validar datos
            telemetry = TelemetryData.from_dict(raw_data)

            if telemetry:
                self._handle_observation_success(request_id, telemetry)
            else:
                self._handle_observation_failure(request_id, "INVALID_TELEMETRY")
                return None

            return telemetry

        except requests.exceptions.Timeout:
            logger.warning(f"[OBSERVE] Timeout después de {self.request_timeout}s")
            self._handle_observation_failure(request_id, "TIMEOUT")
            return None

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"[OBSERVE] Error de conexión: {type(e).__name__}")
            self._handle_observation_failure(request_id, "CONNECTION_ERROR")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"[OBSERVE] Error inesperado: {e}")
            self._handle_observation_failure(request_id, "REQUEST_ERROR")
            return None

    def _handle_observation_success(
        self,
        request_id: str,
        telemetry: TelemetryData
    ) -> None:
        """
        Maneja una observación exitosa actualizando métricas y topología.

        Args:
            request_id: ID del request para tracking
            telemetry: Datos de telemetría recibidos
        """
        self._metrics.record_success()

        # Registrar request exitoso (no genera loops porque IDs son únicos)
        self.topology.record_request(request_id)

        # Actualizar topología: conexión Agent-Core confirmada
        # Inferimos estado de subsistemas desde la respuesta del Core
        active_connections = [("Agent", "Core")]

        # Si la telemetría incluye estado de subsistemas, usarlo
        raw = telemetry.raw_data
        if raw.get("redis_connected", True):
            active_connections.append(("Core", "Redis"))
        if raw.get("filesystem_accessible", True):
            active_connections.append(("Core", "Filesystem"))

        self.topology.update_connectivity(
            active_connections,
            validate_nodes=True,
            auto_add_nodes=False
        )

        logger.debug(
            f"[OBSERVE] ✓ Datos recibidos: "
            f"voltage={telemetry.flyback_voltage:.3f}, "
            f"saturation={telemetry.saturation:.3f}"
        )

    def _handle_observation_failure(
        self,
        request_id: str,
        failure_type: str
    ) -> None:
        """
        Maneja una observación fallida actualizando métricas y topología.

        Args:
            request_id: ID del request para tracking
            failure_type: Tipo de fallo para diagnóstico
        """
        self._metrics.record_failure()

        # Registrar request fallido con tipo de error para detección de patrones
        # Usamos el tipo de fallo como "request_id" para detectar fallos repetitivos
        self.topology.record_request(f"FAIL_{failure_type}")

        # Si hay fallos consecutivos significativos, degradar topología
        if self._metrics.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                f"[TOPO] Degradando topología: "
                f"{self._metrics.consecutive_failures} fallos consecutivos"
            )
            # Remover conexión Agent-Core para reflejar desconexión
            self.topology.remove_edge("Agent", "Core")

    def orient(self, telemetry: Optional[TelemetryData]) -> SystemStatus:
        """
        ORIENT - Segunda fase del ciclo OODA (Motor Topológico).
        
        Analiza el estado del sistema usando:
        1. Invariantes topológicos (números de Betti, salud topológica)
        2. Homología persistente (patrones estructurales vs ruido)
        3. Detección de loops de reintentos
        
        Args:
            telemetry: Datos de telemetría (puede ser None si falló observe)
            
        Returns:
            Estado del sistema determinado por análisis topológico
        """
        # Obtener salud topológica completa
        topo_health = self.topology.get_topological_health()

        # Preparar análisis de persistencia (pueden ser vacíos si no hay datos)
        voltage_analysis = self._analyze_metric_persistence(
            "flyback_voltage",
            telemetry.flyback_voltage if telemetry else None,
            self.thresholds.flyback_voltage_warning
        )

        saturation_analysis = self._analyze_metric_persistence(
            "saturation",
            telemetry.saturation if telemetry else None,
            self.thresholds.saturation_warning
        )

        # Determinar estado y construir diagnóstico
        status, summary = self._evaluate_system_state(
            telemetry=telemetry,
            topo_health=topo_health,
            voltage_analysis=voltage_analysis,
            saturation_analysis=saturation_analysis
        )

        # Almacenar diagnóstico completo
        self._last_diagnosis = TopologicalDiagnosis(
            health=topo_health,
            voltage_persistence=voltage_analysis,
            saturation_persistence=saturation_analysis,
            summary=summary,
            recommended_status=status
        )

        # Log estructurado del diagnóstico
        if status != SystemStatus.NOMINAL:
            logger.info(
                f"[ORIENT] Diagnóstico: {self._last_diagnosis.to_log_dict()}"
            )

        return status

    def _analyze_metric_persistence(
        self,
        metric_name: str,
        current_value: Optional[float],
        threshold: float
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
            metric_name,
            threshold=threshold,
            noise_ratio=0.2,
            critical_ratio=0.5
        )

    def _evaluate_system_state(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult
    ) -> Tuple[SystemStatus, str]:
        """
        Evalúa el estado del sistema integrando todas las fuentes de análisis.

        Jerarquía de evaluación (prioridad descendente):
        1. Fragmentación topológica (b0 > 1)
        2. Umbrales críticos instantáneos (safety net)
        3. Salud topológica crítica (health_score < 0.4)
        4. Patrones de persistencia (CRITICAL/FEATURE)
        5. Loops de reintentos
        6. Estado nominal

        Args:
            telemetry: Datos de telemetría actuales
            topo_health: Salud topológica del sistema
            voltage_analysis: Análisis de persistencia de voltaje
            saturation_analysis: Análisis de persistencia de saturación

        Returns:
            Tupla (SystemStatus, resumen_diagnóstico)
        """
        betti = topo_health.betti

        # =====================================================================
        # 1. FRAGMENTACIÓN TOPOLÓGICA (Máxima prioridad)
        # =====================================================================
        if not betti.is_connected:
            disconnected = ", ".join(topo_health.disconnected_nodes) or "desconocidos"
            summary = (
                f"Fragmentación Topológica: β₀={betti.b0} componentes. "
                f"Nodos desconectados: [{disconnected}]"
            )
            logger.warning(f"[TOPO] ✂️ {summary}")
            return SystemStatus.DISCONNECTED, summary

        # =====================================================================
        # 2. VERIFICAR DATOS DE TELEMETRÍA
        # =====================================================================
        if telemetry is None:
            # Sin datos pero conectado = estado desconocido
            if self._metrics.consecutive_failures > 0:
                summary = f"Sin telemetría ({self._metrics.consecutive_failures} fallos)"
            else:
                summary = "Esperando datos de telemetría"
            return SystemStatus.UNKNOWN, summary

        # =====================================================================
        # 3. UMBRALES CRÍTICOS INSTANTÁNEOS (Safety Net)
        # =====================================================================
        if telemetry.flyback_voltage > self.thresholds.flyback_voltage_critical:
            summary = (
                f"Voltaje Crítico Instantáneo: "
                f"{telemetry.flyback_voltage:.3f} > "
                f"{self.thresholds.flyback_voltage_critical}"
            )
            logger.critical(f"[SAFETY] 🚨 {summary}")
            return SystemStatus.CRITICO, summary

        if telemetry.saturation > self.thresholds.saturation_critical:
            summary = (
                f"Saturación Crítica Instantánea: "
                f"{telemetry.saturation:.3f} > "
                f"{self.thresholds.saturation_critical}"
            )
            logger.critical(f"[SAFETY] 🚨 {summary}")
            return SystemStatus.CRITICO, summary

        # =====================================================================
        # 4. SALUD TOPOLÓGICA GENERAL
        # =====================================================================
        if topo_health.level == HealthLevel.CRITICAL:
            summary = (
                f"Salud Topológica Crítica: score={topo_health.health_score:.2f}. "
                f"Diagnósticos: {topo_health.diagnostics}"
            )
            logger.warning(f"[TOPO] 🔴 {summary}")
            return SystemStatus.CRITICO, summary

        # =====================================================================
        # 5. ANÁLISIS DE PERSISTENCIA
        # =====================================================================

        # Saturación persistente (más grave que inestabilidad)
        if saturation_analysis.state == MetricState.CRITICAL:
            duration = saturation_analysis.metadata.get('active_duration', '?')
            summary = (
                f"Saturación Persistente Crítica: "
                f"excursión activa por {duration} muestras"
            )
            logger.warning(f"[PERSIST] 🟠 {summary}")
            return SystemStatus.SATURADO, summary

        if saturation_analysis.state == MetricState.FEATURE:
            summary = (
                f"Saturación con Patrón Estructural: "
                f"{saturation_analysis.feature_count} característica(s), "
                f"persistencia total={saturation_analysis.total_persistence:.1f}"
            )
            logger.warning(f"[PERSIST] 🟡 {summary}")
            return SystemStatus.SATURADO, summary

        # Inestabilidad de voltaje
        if voltage_analysis.state == MetricState.CRITICAL:
            duration = voltage_analysis.metadata.get('active_duration', '?')
            summary = (
                f"Inestabilidad de Voltaje Crítica: "
                f"excursión activa por {duration} muestras"
            )
            logger.warning(f"[PERSIST] 🟠 {summary}")
            return SystemStatus.INESTABLE, summary

        if voltage_analysis.state == MetricState.FEATURE:
            summary = (
                f"Inestabilidad de Voltaje Estructural: "
                f"{voltage_analysis.feature_count} característica(s), "
                f"max_lifespan={voltage_analysis.max_lifespan:.1f}"
            )
            logger.warning(f"[PERSIST] 🟡 {summary}")
            return SystemStatus.INESTABLE, summary

        # =====================================================================
        # 6. DETECCIÓN DE LOOPS DE REINTENTOS
        # =====================================================================
        if topo_health.request_loops:
            worst_loop = topo_health.request_loops[0]  # Ya ordenados por frecuencia
            if worst_loop.count >= 5:
                summary = (
                    f"Patrón de Reintentos Detectado: "
                    f"'{worst_loop.request_id}' apareció {worst_loop.count} veces"
                )
                logger.warning(f"[TOPO] 🔄 {summary}")
                # Los loops de error indican inestabilidad
                if worst_loop.request_id.startswith("FAIL_"):
                    return SystemStatus.INESTABLE, summary

        # =====================================================================
        # 7. SALUD TOPOLÓGICA DEGRADADA (pero no crítica)
        # =====================================================================
        if topo_health.level == HealthLevel.UNHEALTHY:
            summary = (
                f"Salud Topológica Degradada: score={topo_health.health_score:.2f}"
            )
            logger.info(f"[TOPO] 🟡 {summary}")
            # No es crítico, pero vale la pena monitorear
            # Continuamos a evaluar como NOMINAL por ahora

        # =====================================================================
        # 8. RUIDO - Ignorar (inmunidad a falsos positivos)
        # =====================================================================
        if voltage_analysis.state == MetricState.NOISE:
            logger.debug(
                f"[PERSIST] Ruido en voltaje ignorado: "
                f"{voltage_analysis.noise_count} excursiones cortas"
            )

        if saturation_analysis.state == MetricState.NOISE:
            logger.debug(
                f"[PERSIST] Ruido en saturación ignorado: "
                f"{saturation_analysis.noise_count} excursiones cortas"
            )

        # =====================================================================
        # 9. ESTADO NOMINAL
        # =====================================================================
        summary = (
            f"Sistema Nominal: β₀={betti.b0}, β₁={betti.b1}, "
            f"health={topo_health.health_score:.2f}"
        )
        return SystemStatus.NOMINAL, summary

    def decide(self, status: SystemStatus) -> AgentDecision:
        """
        DECIDE - Tercera fase del ciclo OODA.
        
        Mapea el estado del sistema a una decisión de acción.
        Considera el contexto topológico para decisiones más informadas.
        
        Args:
            status: Estado actual del sistema
            
        Returns:
            Decisión a ejecutar
        """
        # Mapeo base de estados a decisiones
        decision_mapping: Dict[SystemStatus, AgentDecision] = {
            SystemStatus.NOMINAL: AgentDecision.HEARTBEAT,
            SystemStatus.INESTABLE: AgentDecision.RECOMENDAR_LIMPIEZA,
            SystemStatus.SATURADO: AgentDecision.RECOMENDAR_REDUCIR_VELOCIDAD,
            SystemStatus.CRITICO: AgentDecision.ALERTA_CRITICA,
            SystemStatus.DISCONNECTED: AgentDecision.RECONNECT,
            SystemStatus.UNKNOWN: AgentDecision.WAIT,
        }

        decision = decision_mapping.get(status, AgentDecision.WAIT)

        # Ajustar decisión basándose en contexto topológico
        if self._last_diagnosis and decision == AgentDecision.HEARTBEAT:
            # Aunque nominal, si hay loops de error, considerar advertencia
            if self._last_diagnosis.has_retry_loops:
                loops = self._last_diagnosis.health.request_loops
                error_loops = [l for l in loops if l.request_id.startswith("FAIL_")]
                if error_loops:
                    logger.debug(
                        f"[DECIDE] Sistema nominal pero con {len(error_loops)} "
                        f"patrones de error en historial"
                    )

        self._metrics.record_decision(decision)
        self._last_status = status

        return decision

    def act(self, decision: AgentDecision) -> bool:
        """
        ACT - Cuarta fase del ciclo OODA.
        
        Ejecuta la acción decidida con información topológica enriquecida.
        Implementa debounce para evitar spam de acciones repetitivas.
        
        Args:
            decision: Decisión a ejecutar
            
        Returns:
            True si la acción fue ejecutada, False si fue suprimida
        """
        # Verificar debounce
        if self._should_debounce(decision):
            logger.debug(f"[ACT] Acción suprimida por debounce: {decision.name}")
            return False

        # Construir mensaje de diagnóstico enriquecido
        diagnosis_msg = self._build_diagnosis_message()

        # Ejecutar acción según decisión
        action_handlers = {
            AgentDecision.HEARTBEAT: self._act_heartbeat,
            AgentDecision.RECOMENDAR_LIMPIEZA: self._act_recomendar_limpieza,
            AgentDecision.RECOMENDAR_REDUCIR_VELOCIDAD: self._act_recomendar_reducir_velocidad,
            AgentDecision.ALERTA_CRITICA: self._act_alerta_critica,
            AgentDecision.RECONNECT: self._act_reconnect,
            AgentDecision.WAIT: self._act_wait,
        }

        handler = action_handlers.get(decision, self._act_wait)
        handler(diagnosis_msg)

        # Actualizar estado para debounce
        self._last_decision = decision
        self._last_decision_time = datetime.now()

        return True

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

    def _build_diagnosis_message(self) -> str:
        """
        Construye mensaje de diagnóstico estructurado desde el análisis topológico.

        Returns:
            Mensaje formateado para logging
        """
        if not self._last_diagnosis:
            return ""

        diag = self._last_diagnosis
        parts = [diag.summary]

        # Agregar contexto de Betti si es relevante
        betti = diag.health.betti
        if not betti.is_ideal:
            parts.append(f"[β₀={betti.b0}, β₁={betti.b1}]")

        # Agregar health score si no es óptimo
        if diag.health.health_score < 0.9:
            parts.append(f"[health={diag.health.health_score:.2f}]")

        return " | ".join(parts)

    def _act_heartbeat(self, diagnosis_msg: str) -> None:
        """Acción: Sistema nominal."""
        health_indicator = "✅"
        if self._last_diagnosis and self._last_diagnosis.health.health_score < 1.0:
            health_indicator = "✅" if self._last_diagnosis.health.health_score >= 0.9 else "🟢"

        logger.info(
            f"[BRAIN] {health_indicator} Sistema NOMINAL - Operación estable"
        )

    def _act_recomendar_limpieza(self, diagnosis_msg: str) -> None:
        """Acción: Recomendar limpieza por inestabilidad."""
        logger.warning(
            f"[BRAIN] ⚠️ INESTABILIDAD DETECTADA - {diagnosis_msg}"
        )
        logger.warning(
            "[BRAIN] → Recomendación: Revisar y limpiar datos CSV"
        )
        self._notify_external_system("instability_detected", {
            "diagnosis": diagnosis_msg,
            "voltage_state": self._last_diagnosis.voltage_persistence.state.name
            if self._last_diagnosis else None
        })

    def _act_recomendar_reducir_velocidad(self, diagnosis_msg: str) -> None:
        """Acción: Recomendar reducir velocidad por saturación."""
        logger.warning(
            f"[BRAIN] ⚠️ SATURACIÓN DETECTADA - {diagnosis_msg}"
        )
        logger.warning(
            "[BRAIN] → Recomendación: Reducir velocidad de carga"
        )
        self._notify_external_system("saturation_detected", {
            "diagnosis": diagnosis_msg,
            "saturation_state": self._last_diagnosis.saturation_persistence.state.name
            if self._last_diagnosis else None
        })

    def _act_alerta_critica(self, diagnosis_msg: str) -> None:
        """Acción: Alerta crítica."""
        logger.critical(
            f"[BRAIN] 🚨 ALERTA CRÍTICA - {diagnosis_msg}"
        )
        logger.critical(
            "[BRAIN] → Intervención inmediata requerida"
        )
        self._notify_external_system("critical_alert", {
            "diagnosis": diagnosis_msg,
            "health_score": self._last_diagnosis.health.health_score
            if self._last_diagnosis else None,
            "betti": self._last_diagnosis.health.betti.b0
            if self._last_diagnosis else None
        })

    def _act_reconnect(self, diagnosis_msg: str) -> None:
        """Acción: Intentar reconexión."""
        logger.warning(
            f"[BRAIN] 🔄 Conexión perdida - {diagnosis_msg}"
        )
        logger.warning(
            f"[BRAIN] → Reintentando conexión con Core..."
        )
        # Restaurar topología esperada para próximo intento
        self._initialize_expected_topology()

    def _act_wait(self, diagnosis_msg: str) -> None:
        """Acción: Esperar datos."""
        logger.info("[BRAIN] ⏳ Esperando datos de telemetría...")

    def _notify_external_system(
        self,
        event_type: str,
        context: Optional[Dict[str, Any]] = None
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
                self.telemetry_endpoint,
                timeout=self.request_timeout
            )

            if response.ok:
                # Actualizar topología con conexión confirmada
                self._initialize_expected_topology()

                # Verificar salud topológica
                topo_health = self.topology.get_topological_health()

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
            topo_health = self.topology.get_topological_health()

            logger.error(
                f"Topología degradada: β₀={topo_health.betti.b0}, "
                f"health={topo_health.health_score:.2f}"
            )
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna las métricas completas del agente incluyendo estado topológico.
        
        Returns:
            Diccionario con métricas del agente y análisis topológico
        """
        metrics = self._metrics.to_dict()

        # Información básica del agente
        metrics.update({
            "core_api_url": self.core_api_url,
            "check_interval": self.check_interval,
            "is_running": self._running,
            "last_status": self._last_status.name if self._last_status else None,
        })

        # Métricas topológicas
        topo_health = self.topology.get_topological_health()
        metrics["topology"] = {
            "betti_b0": topo_health.betti.b0,
            "betti_b1": topo_health.betti.b1,
            "is_connected": topo_health.betti.is_connected,
            "is_ideal": topo_health.betti.is_ideal,
            "health_score": round(topo_health.health_score, 3),
            "health_level": topo_health.level.name,
            "disconnected_nodes": list(topo_health.disconnected_nodes),
            "missing_edges": [list(e) for e in topo_health.missing_edges],
            "request_loops_count": len(topo_health.request_loops),
            "euler_characteristic": topo_health.betti.euler_characteristic,
        }

        # Métricas de persistencia
        persistence_metrics = {}
        for metric_name in ["flyback_voltage", "saturation"]:
            stats = self.persistence.get_statistics(metric_name)
            if stats:
                persistence_metrics[metric_name] = {
                    "samples": stats["count"],
                    "min": round(stats["min"], 4),
                    "max": round(stats["max"], 4),
                    "mean": round(stats["mean"], 4),
                    "std": round(stats["std"], 4),
                }

        if persistence_metrics:
            metrics["persistence"] = persistence_metrics

        # Último diagnóstico
        if self._last_diagnosis:
            metrics["last_diagnosis"] = {
                "summary": self._last_diagnosis.summary,
                "recommended_status": self._last_diagnosis.recommended_status.name,
                "voltage_state": self._last_diagnosis.voltage_persistence.state.name,
                "saturation_state": self._last_diagnosis.saturation_persistence.state.name,
            }

        return metrics

    def get_topological_summary(self) -> Dict[str, Any]:
        """
        Retorna un resumen del estado topológico actual.

        Útil para dashboards y monitoreo en tiempo real.

        Returns:
            Resumen topológico estructurado
        """
        health = self.topology.get_topological_health()

        return {
            "timestamp": datetime.now().isoformat(),
            "betti_numbers": {
                "b0": health.betti.b0,
                "b1": health.betti.b1,
                "interpretation": {
                    "b0": "conectado" if health.betti.is_connected else f"{health.betti.b0} fragmentos",
                    "b1": "acíclico" if health.betti.is_acyclic else f"{health.betti.b1} ciclos",
                }
            },
            "health": {
                "score": health.health_score,
                "level": health.level.name,
                "is_healthy": health.is_healthy,
            },
            "issues": {
                "disconnected_nodes": list(health.disconnected_nodes),
                "missing_connections": [f"{u}-{v}" for u, v in health.missing_edges],
                "diagnostics": health.diagnostics,
            },
            "retry_patterns": [
                {
                    "request_id": loop.request_id,
                    "count": loop.count,
                }
                for loop in health.request_loops[:5]  # Top 5
            ],
        }

    def run(self, skip_health_check: bool = False) -> None:
        """
        Bucle principal del agente - Ejecuta el ciclo OODA continuamente.
        
        Args:
            skip_health_check: Si True, omite verificación inicial
        """
        # Health check inicial
        if not skip_health_check:
            if not self.health_check():
                logger.warning(
                    "Iniciando agente a pesar de health check fallido..."
                )

        self._running = True
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
                        exc_info=True
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
