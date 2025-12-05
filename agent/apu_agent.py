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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agent.topological_analyzer import SystemTopology, PersistenceHomology, MetricState


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

    def __init__(
        self,
        core_api_url: Optional[str] = None,
        check_interval: Optional[int] = None,
        request_timeout: Optional[int] = None,
        thresholds: Optional[ThresholdConfig] = None
    ) -> None:
        """
        Inicializa el agente autónomo.
        
        Args:
            core_api_url: URL del API del Core
            check_interval: Intervalo entre ciclos (segundos)
            request_timeout: Timeout de requests (segundos)
            thresholds: Configuración de umbrales de análisis
            
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
        self._last_diagnosis: Optional[str] = None  # Almacena diagnóstico topológico
        self._metrics = AgentMetrics()

        # Componentes de análisis topológico
        self.topology = SystemTopology()
        self.persistence = PersistenceHomology(window_size=20)

        # Inicializar topología esperada (Agent -> Core -> Subsistemas)
        # Asumimos estado inicial conectado para evitar alertas prematuras al arranque
        self.topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem")
        ])

        # Sesión HTTP con reintentos
        self._session = self._create_robust_session()

        # Manejadores de señales
        self._original_handlers: Dict[signal.Signals, Any] = {}
        self._setup_signal_handlers()

        logger.info(
            f"AutonomousAgent inicializado | "
            f"Core: {self.core_api_url} | "
            f"Intervalo: {self.check_interval}s | "
            f"Timeout: {self.request_timeout}s"
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
        
        Realiza una solicitud GET al endpoint de telemetría del Core
        y retorna datos estructurados y validados.
        Además, alimenta la topología con el estado de conectividad.
        
        Returns:
            TelemetryData si exitoso, None si hay error
        """
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
                self._metrics.record_failure()
                # Registrar posible desconexión si falla
                # Nota: la desconexión total se decide en Orient basado en persistencia de fallos
                return None

            # Parsear JSON
            try:
                raw_data = response.json()
            except ValueError as e:
                logger.warning(f"[OBSERVE] Respuesta JSON inválida: {e}")
                self._metrics.record_failure()
                return None

            # Estructurar y validar datos
            telemetry = TelemetryData.from_dict(raw_data)

            if telemetry:
                self._metrics.record_success()

                # Topología: Registrar conexión exitosa Agent -> Core
                # Asumimos que si el Core responde, sus subsistemas están accesibles
                # (idealmente esto vendría en la telemetría, pero por ahora lo inferimos para evitar b0 > 1)
                self.topology.update_connectivity([
                    ("Agent", "Core"),
                    ("Core", "Redis"),
                    ("Core", "Filesystem")
                ])

                logger.debug(
                    f"[OBSERVE] Datos recibidos: "
                    f"voltage={telemetry.flyback_voltage:.3f}, "
                    f"saturation={telemetry.saturation:.3f}"
                )
            else:
                self._metrics.record_failure()

            return telemetry

        except requests.exceptions.Timeout:
            logger.warning(f"[OBSERVE] Timeout después de {self.request_timeout}s")
            self._metrics.record_failure()
            return None

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"[OBSERVE] Error de conexión: {type(e).__name__}")
            self._metrics.record_failure()
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"[OBSERVE] Error inesperado: {e}")
            self._metrics.record_failure()
            return None

    def orient(self, telemetry: Optional[TelemetryData]) -> SystemStatus:
        """
        ORIENT - Segunda fase del ciclo OODA (Lógica Topológica).
        
        Analiza los datos usando Persistence Homology y Betti Numbers.
        Sustituye umbrales simples por análisis de estabilidad y estructura.
        
        Args:
            telemetry: Datos de telemetría (puede ser None)
            
        Returns:
            Estado del sistema determinado
        """
        self._last_diagnosis = None

        # 1. Alimentar la Topología y verificar integridad
        if telemetry is None:
            # Si hay fallo, y persiste, cortamos la arista
            if self._metrics.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                 self.topology.remove_edge("Agent", "Core")

        # Obtener invariantes topológicos
        betti = self.topology.calculate_betti_numbers()
        b0, b1 = betti.b0, betti.b1

        # Diagnóstico Topológico 1: Conectividad (b0)
        if b0 > 1:
            self._last_diagnosis = f"Fragmentación Topológica (b0={b0})"
            logger.warning(f"[TOPO] ✂️ {self._last_diagnosis} - Desconexión detectada")
            return SystemStatus.DISCONNECTED

        if telemetry is None:
            return SystemStatus.UNKNOWN

        # 2. Ingesta para Homología Persistente
        self.persistence.add_reading("flyback_voltage", telemetry.flyback_voltage)
        self.persistence.add_reading("saturation", telemetry.saturation)

        # 3. Análisis de Persistencia
        # "Flyback Voltage" -> Buscamos inestabilidad (excursiones persistentes)
        voltage_analysis = self.persistence.analyze_persistence(
            "flyback_voltage",
            threshold=self.thresholds.flyback_voltage_warning,
            critical_ratio=0.5 # 50% de la ventana activa = Crítico
        )

        # "Saturation" -> Buscamos saturación (excursiones persistentes)
        saturation_analysis = self.persistence.analyze_persistence(
            "saturation",
            threshold=self.thresholds.saturation_warning
        )

        # 4. Determinación de Estado (Lógica del Ingeniero Senior)

        # Salvaguarda: Umbrales Críticos Instantáneos (Safety Net)
        # Asegura respuesta inmediata ante condiciones peligrosas, independiente de la persistencia
        if telemetry.flyback_voltage > self.thresholds.flyback_voltage_critical:
             self._last_diagnosis = f"Voltaje Crítico Instantáneo ({telemetry.flyback_voltage:.2f})"
             return SystemStatus.CRITICO

        if telemetry.saturation > self.thresholds.saturation_critical:
             self._last_diagnosis = f"Saturación Crítica Instantánea ({telemetry.saturation:.2f})"
             return SystemStatus.CRITICO

        # Caso: Saturación Persistente (Warning sostenido)
        if saturation_analysis.state == MetricState.CRITICAL:
            lifespan = saturation_analysis.metadata.get('active_duration', 'unknown')
            self._last_diagnosis = f"Saturación Persistente (Vida: {lifespan})"
            return SystemStatus.SATURADO

        # Caso: Inestabilidad de Voltaje (Critical o Feature)
        # "Si Persistencia de Flyback es CRITICAL o FEATURE -> SystemStatus.INESTABLE"
        if voltage_analysis.state in (MetricState.CRITICAL, MetricState.FEATURE):
            lifespan = voltage_analysis.max_lifespan
            if voltage_analysis.state == MetricState.CRITICAL:
                lifespan = voltage_analysis.metadata.get('active_duration', 'unknown')

            self._last_diagnosis = f"Inestabilidad de Voltaje Persistente (Estado: {voltage_analysis.state.name}, Vida: {lifespan})"
            logger.warning(f"[TOPO] ⚠️ {self._last_diagnosis}")
            return SystemStatus.INESTABLE

        # Caso: Ruido (Noise) -> Ignorar
        if voltage_analysis.state == MetricState.NOISE:
            logger.debug("[TOPO] Ruido transitorio detectado en voltaje - Ignorando (Inmune a falsos positivos)")
            return SystemStatus.NOMINAL

        if saturation_analysis.state == MetricState.NOISE:
             logger.debug("[TOPO] Ruido transitorio detectado en saturación - Ignorando")
             return SystemStatus.NOMINAL

        # Estado Base
        return SystemStatus.NOMINAL

    def decide(self, status: SystemStatus) -> AgentDecision:
        """
        DECIDE - Tercera fase del ciclo OODA.
        
        Determina la acción a tomar basada en el estado del sistema.
        
        Args:
            status: Estado actual del sistema
            
        Returns:
            Decisión a ejecutar
        """
        decision_mapping: Dict[SystemStatus, AgentDecision] = {
            SystemStatus.NOMINAL: AgentDecision.HEARTBEAT,
            SystemStatus.INESTABLE: AgentDecision.RECOMENDAR_LIMPIEZA,
            SystemStatus.SATURADO: AgentDecision.RECOMENDAR_REDUCIR_VELOCIDAD,
            SystemStatus.CRITICO: AgentDecision.ALERTA_CRITICA,
            SystemStatus.DISCONNECTED: AgentDecision.RECONNECT,
            SystemStatus.UNKNOWN: AgentDecision.WAIT,
        }

        decision = decision_mapping.get(status, AgentDecision.WAIT)
        self._metrics.record_decision(decision)
        self._last_status = status

        return decision

    def act(self, decision: AgentDecision) -> bool:
        """
        ACT - Cuarta fase del ciclo OODA.
        
        Ejecuta la acción decidida. Implementa debounce para evitar
        spam de acciones repetitivas.
        
        Args:
            decision: Decisión a ejecutar
            
        Returns:
            True si la acción fue ejecutada, False si fue suprimida
        """
        # Verificar debounce
        if self._should_debounce(decision):
            logger.debug(f"[ACT] Acción suprimida por debounce: {decision.name}")
            return False

        # Incluir diagnóstico en logs si existe
        diagnosis_msg = f" - {self._last_diagnosis}" if self._last_diagnosis else ""

        # Mapa de acciones
        # Usamos lambdas o wrappers para inyectar el contexto de logging si es necesario

        if decision == AgentDecision.HEARTBEAT:
             logger.info("[BRAIN] ✅ Sistema NOMINAL - Operación estable")

        elif decision == AgentDecision.RECOMENDAR_LIMPIEZA:
             logger.warning(
                f"[BRAIN] ⚠️ INESTABILIDAD PERSISTENTE{diagnosis_msg} - "
                "Se recomienda revisión y limpieza de CSV"
            )
             self._notify_external_system("instability_detected")

        elif decision == AgentDecision.RECOMENDAR_REDUCIR_VELOCIDAD:
             logger.warning(
                f"[BRAIN] ⚠️ SATURACIÓN DETECTADA{diagnosis_msg} - "
                "Se recomienda reducir la velocidad de carga"
            )
             self._notify_external_system("saturation_detected")

        elif decision == AgentDecision.ALERTA_CRITICA:
             logger.critical(
                f"[BRAIN] 🚨 ALERTA CRÍTICA{diagnosis_msg} - "
                "Sistema en estado crítico. Intervención inmediata requerida."
            )
             self._notify_external_system("critical_alert")

        elif decision == AgentDecision.RECONNECT:
             logger.warning(
                f"[BRAIN] 🔄 Conexión perdida con Core{diagnosis_msg}. "
                f"Reintentando..."
            )

        elif decision == AgentDecision.WAIT:
             logger.info("[BRAIN] ⏳ Esperando datos de telemetría...")

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

    # =========================================================================
    # ACTION HANDLERS - (Legacy methods removed in favor of inline logic in act)
    # =========================================================================

    def _notify_external_system(self, event_type: str) -> None:
        """
        Hook para notificaciones externas (webhooks, etc).
        
        Args:
            event_type: Tipo de evento a notificar
        """
        # Placeholder para integración futura
        logger.debug(f"[NOTIFY] Evento: {event_type}")

    # =========================================================================
    # LIFECYCLE METHODS - Control del ciclo de vida
    # =========================================================================

    def health_check(self) -> bool:
        """
        Verifica conectividad con el Core antes de iniciar.
        
        Returns:
            True si el Core es accesible, False en caso contrario
        """
        logger.info(f"Ejecutando health check: {self.telemetry_endpoint}")

        try:
            response = self._session.get(
                self.telemetry_endpoint,
                timeout=self.request_timeout
            )

            if response.ok:
                logger.info("✅ Health check exitoso - Core accesible")
                return True
            else:
                logger.warning(
                    f"⚠️ Health check con advertencia: HTTP {response.status_code}"
                )
                return True  # Permitir continuar con warning

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Health check fallido: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna las métricas actuales del agente.
        
        Returns:
            Diccionario con todas las métricas
        """
        metrics = self._metrics.to_dict()
        metrics.update({
            "core_api_url": self.core_api_url,
            "check_interval": self.check_interval,
            "is_running": self._running,
            "last_status": self._last_status.name if self._last_status else None,
        })
        return metrics

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
