"""
=========================================================================================
Módulo: Autonomous Agent - Controlador OODA con Fibrado Gauge
Ubicación: app/tactics/apu_agent.py
Versión: 2.0.0 (Refactorización Rigurosa)
=========================================================================================

FUNDAMENTOS MATEMÁTICOS
=======================

Este módulo implementa un Controlador de Lazo Cerrado sobre una Variedad Diferenciable
M (espacio de estados del sistema) mediante el ciclo OODA:

    OODA: M → M,  φ ↦ Act(Decide(Orient(Observe(φ))))

donde cada fase es un morfismo en la categoría de espacios de estados.

TEORÍA DE CONTROL Y ESTABILIDAD
-------------------------------

1. **Función de Lyapunov Global**:
   Sea V: M → ℝ₊ una función de energía tal que:
   
   V(φ) = Σᵢ wᵢ·Vᵢ(φ)
   
   donde cada Vᵢ es una función de Lyapunov local (positiva definida) que penaliza
   desviaciones de la homeostasis en el subsistema i.

2. **Condición de Estabilidad Asintótica** (Teorema de Lyapunov):
   El sistema converge al equilibrio φ* si:
   
   V̇(φ) = ⟨∇V(φ), f(φ)⟩ < 0  ∀φ ≠ φ*
   
   donde f(φ) es el campo vectorial de control.

3. **Ecuación de Poisson en el Fibrado Gauge**:
   El agente resuelve:
   
   ΔΦ = -ρ
   
   donde:
   - Δ := B₁ᵀB₁ es el Laplaciano combinatorio (0-formas)
   - B₁ es la matriz de incidencia orientada (operador coborde)
   - ρ := ∇V es la densidad de carga (gradiente de penalización)
   - Φ es el potencial gauge (selector de agentes)

4. **Invarianza de Gauge y Neutralidad de Carga**:
   Para garantizar existencia de solución a la ecuación de Poisson:
   
   ρ ∈ im(B₁ᵀ) ⟺ ρ ⊥ ker(Δ)
   
   Como ker(Δ) = {vectores constantes}, imponemos:
   
   Σᵢ ρᵢ = 0  (neutralidad de carga total)
   
   La renormalización ρ̃ := ρ - ρ̄ proyecta sobre im(B₁ᵀ).

5. **Certificación de Estabilidad**:
   La potencia disipada debe ser no negativa:
   
   P_diss := ⟨Φ, ∇V⟩ ≥ 0
   
   Esto garantiza que el control reduce la energía V.

TOPOLOGÍA ALGEBRAICA
--------------------

1. **Números de Betti**:
   - β₀: Número de componentes conexas (fragmentación)
   - β₁: Número de ciclos independientes (redundancia)
   
   Invariantes topológicos que caracterizan la salud estructural.

2. **Característica de Euler**:
   χ = β₀ - β₁  (para complejos simpliciales 1-dimensionales)
   
   Invariante topológico global.

3. **Homología Persistente**:
   Captura características topológicas que persisten en múltiples escalas
   temporales, filtrando ruido de señales genuinas.

ÁLGEBRA DE BOOLE Y LÓGICA DE DECISIONES
---------------------------------------

El espacio de decisiones forma un retículo (lattice) bajo el orden parcial
de severidad, con operaciones:

- ⊔ (join): max por severidad (peor caso)
- ⊓ (meet): min por severidad (mejor caso)

Esto garantiza composicionalidad de decisiones.

=========================================================================================
"""

from __future__ import annotations

import logging
from app.adapters.tools_interface import get_global_mic
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

import numpy as np
import requests
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.core.immune_system.gauge_field_router import GaugeFieldRouter
from app.core.mic_algebra import CategoricalState
from app.core.schemas import Stratum
from app.tactics.topological_analyzer import (
    HealthLevel,
    MetricState,
    PersistenceAnalysisResult,
    PersistenceHomology,
    SystemTopology,
    TopologicalHealth,
)


# ============================================================================
# CONSTANTES FÍSICAS Y MATEMÁTICAS
# ============================================================================

class PhysicalConstants:
    """
    Constantes fundamentales del sistema con justificación física.
    
    Siguiendo el Sistema Internacional, todas las constantes tienen
    unidades explícitas y valores derivados de primeros principios.
    """
    
    # Tolerancia numérica (épsilon de máquina efectivo)
    # Justificación: IEEE 754 double precision → 2.22e-16
    # Usamos 1e-9 para robustez ante errores de redondeo acumulados
    EPSILON: Final[float] = 1e-9
    
    # Dimensión del espacio de fase (número de nodos en topología nominal)
    # Justificación: {Agent, Core, Redis, Filesystem} → dim(M) = 4
    PHASE_SPACE_DIM: Final[int] = 4
    
    # Umbral de Lyapunov para estabilidad asintótica
    # V̇ < -LYAPUNOV_THRESHOLD garantiza decaimiento exponencial
    LYAPUNOV_STABILITY_THRESHOLD: Final[float] = -1e-6


class NetworkTopology:
    """
    Constantes de topología de red del sistema.
    
    Define el 1-esqueleto del complejo simplicial esperado.
    """
    
    # Nodos fundamentales (0-símplices)
    NODE_AGENT: Final[str] = "Agent"
    NODE_CORE: Final[str] = "Core"
    NODE_REDIS: Final[str] = "Redis"
    NODE_FILESYSTEM: Final[str] = "Filesystem"
    
    # Conjunto completo de nodos
    ALL_NODES: Final[FrozenSet[str]] = frozenset({
        NODE_AGENT,
        NODE_CORE,
        NODE_REDIS,
        NODE_FILESYSTEM,
    })
    
    # Aristas esperadas (1-símplices orientadas)
    # Representan el flujo de información en estado nominal
    EXPECTED_EDGES: Final[Tuple[Tuple[str, str], ...]] = (
        (NODE_AGENT, NODE_CORE),
        (NODE_CORE, NODE_REDIS),
        (NODE_CORE, NODE_FILESYSTEM),
    )
    
    @classmethod
    def validate_topology(cls, nodes: Set[str], edges: Set[Tuple[str, str]]) -> bool:
        """
        Valida que una topología dada sea un subconjunto válido.
        
        Args:
            nodes: Conjunto de nodos observados
            edges: Conjunto de aristas observadas
            
        Returns:
            True si la topología es estructuralmente válida
            
        Invariantes:
            - nodes ⊆ ALL_NODES
            - ∀(u,v) ∈ edges: {u,v} ⊆ nodes
        """
        if not nodes.issubset(cls.ALL_NODES):
            return False
        
        for u, v in edges:
            if u not in nodes or v not in nodes:
                return False
        
        return True


class DefaultConfig:
    """
    Valores por defecto de configuración con justificación empírica.
    """
    
    # URL del Core API
    CORE_URL: Final[str] = "http://localhost:5002"
    
    # Intervalo del ciclo OODA (segundos)
    # Justificación: Balance entre latencia de reacción y carga del sistema
    # Frecuencia de Nyquist: f_sample ≥ 2·f_signal
    # Con dinámicas esperadas en escala de 1-10s → 10s es adecuado
    CHECK_INTERVAL: Final[int] = 10
    
    # Timeout de requests HTTP (segundos)
    # Justificación: Percentil 99 de latencia de red local + margen
    REQUEST_TIMEOUT: Final[int] = 10
    
    # Fallos consecutivos antes de degradación topológica
    # Justificación: Evitar false positives por transitorios
    MAX_CONSECUTIVE_FAILURES: Final[int] = 5
    
    # Ventana de debounce para evitar spam de acciones (segundos)
    # Justificación: Tiempo mínimo para que una acción tenga efecto observable
    DEBOUNCE_WINDOW: Final[int] = 60
    
    # Tamaño de ventana para homología persistente
    # Justificación: Capturar patrones con vida útil de 2-3 minutos
    PERSISTENCE_WINDOW: Final[int] = 20
    
    # Historia topológica máxima (número de snapshots)
    # Justificación: Límite de memoria (~10 KB por snapshot × 100 = 1 MB)
    TOPOLOGY_HISTORY: Final[int] = 100


# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

class LogConfig:
    """Configuración centralizada de logging con niveles semánticos."""
    
    # Formato estructurado para parsing automatizado
    LOG_FORMAT: Final[str] = (
        "%(asctime)s - %(levelname)-8s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
    )
    
    # Mapeo de niveles de log
    LEVEL_MAP: Final[Dict[str, int]] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    @classmethod
    def setup_logger(cls, name: str) -> logging.Logger:
        """
        Configura un logger con formato estandarizado.
        
        Args:
            name: Nombre del logger (usualmente __name__)
            
        Returns:
            Logger configurado
            
        Postcondiciones:
            - Logger tiene handler de consola
            - Formato sigue LOG_FORMAT
            - Nivel leído de LOG_LEVEL env var (default: INFO)
        """
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = cls.LEVEL_MAP.get(log_level_str, logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format=cls.LOG_FORMAT,
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        
        return logger


# Logger global del módulo
logger = LogConfig.setup_logger(__name__)


# ============================================================================
# JERARQUÍA DE EXCEPCIONES DEL DOMINIO
# ============================================================================

class AgentError(Exception):
    """
    Clase base para todas las excepciones del agente.
    
    Invariante: Todas las excepciones específicas del dominio heredan de esta.
    """
    
    def __init__(self, message: str, **context: Any):
        """
        Inicializa excepción con mensaje y contexto adicional.
        
        Args:
            message: Descripción legible del error
            **context: Metadata adicional (request_id, timestamp, etc.)
        """
        super().__init__(message)
        self.message = message
        self.context = context
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa la excepción para logging estructurado."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            **self.context,
        }


class ConfigurationError(AgentError):
    """
    Error en la configuración del agente.
    
    Ejemplos:
        - URL inválida
        - Parámetros fuera de rango
        - Archivos de configuración corruptos
    """
    pass


class ObservationError(AgentError):
    """
    Error durante la fase OBSERVE del ciclo OODA.
    
    Captura fallos en la adquisición de telemetría.
    """
    
    def __init__(
        self,
        message: str,
        error_type: ObservationErrorType,
        request_id: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(message, error_type=error_type.value, request_id=request_id, **context)
        self.error_type = error_type
        self.request_id = request_id


class TopologyError(AgentError):
    """
    Error relacionado con invariantes topológicos.
    
    Ejemplos:
        - Violación de β₀ = 1 (fragmentación)
        - Detección de ciclos patológicos (β₁ > umbral)
    """
    pass


class TelemetryValidationError(AgentError):
    """
    Error en validación de esquema de telemetría.
    
    Se lanza cuando los datos recibidos no cumplen el contrato esperado.
    """
    pass


class StabilityError(AgentError):
    """
    Error en la certificación de estabilidad del controlador.
    
    Se lanza cuando P_diss = ⟨Φ, ∇V⟩ < 0, indicando frustración cohomológica.
    """
    pass


class GaugeRouterError(AgentError):
    """Error en la resolución de la ecuación de Poisson gauge."""
    pass


# ============================================================================
# ENUMERACIONES CON ORDEN PARCIAL
# ============================================================================

class ObservationErrorType(Enum):
    """
    Tipos de errores de observación clasificados semánticamente.
    
    Forman una partición del espacio de errores posibles.
    """
    
    TIMEOUT = "TIMEOUT"                      # Timeout de red
    CONNECTION_ERROR = "CONNECTION_ERROR"    # Conexión rechazada/caída
    HTTP_ERROR = "HTTP_ERROR"                # Status code 4xx/5xx
    INVALID_JSON = "INVALID_JSON"            # JSON malformado
    INVALID_TELEMETRY = "INVALID_TELEMETRY"  # Schema inválido
    REQUEST_ERROR = "REQUEST_ERROR"          # Error genérico de requests
    UNKNOWN = "UNKNOWN"                      # Catch-all
    
    @property
    def is_transient(self) -> bool:
        """
        Indica si el error es potencialmente transitorio.
        
        Returns:
            True si un retry podría tener éxito
        """
        return self in {
            ObservationErrorType.TIMEOUT,
            ObservationErrorType.CONNECTION_ERROR,
            ObservationErrorType.HTTP_ERROR,
        }
    
    @property
    def requires_reconnection(self) -> bool:
        """Indica si se requiere reconexión completa."""
        return self == ObservationErrorType.CONNECTION_ERROR


# ============================================================================
# ENUMERACIONES CON ESTRUCTURA ALGEBRAICA
# ============================================================================

class SystemStatus(Enum):
    """
    Estados del sistema formando un Orden Parcial Estricto (Poset).
    
    ESTRUCTURA ALGEBRAICA
    ---------------------
    
    El conjunto S = {NOMINAL, UNKNOWN, INESTABLE, SATURADO, CRITICO, DISCONNECTED}
    con la relación de orden ≤ definida por severidad forma un poset (S, ≤):
    
    Diagrama de Hasse:
    
                    DISCONNECTED
                         |
                      CRITICO
                      /     \
                SATURADO   INESTABLE
                      \     /
                      UNKNOWN
                         |
                      NOMINAL
    
    PROPIEDADES
    -----------
    
    1. **Reflexividad**: ∀s ∈ S: s ≤ s
    2. **Antisimetría**: s₁ ≤ s₂ ∧ s₂ ≤ s₁ ⟹ s₁ = s₂
    3. **Transitividad**: s₁ ≤ s₂ ∧ s₂ ≤ s₃ ⟹ s₁ ≤ s₃
    
    OPERACIONES RETICULARES
    -----------------------
    
    - **Join (⊔)**: worst(s₁, s₂) = max{s₁, s₂} (supremo)
    - **Meet (⊓)**: best(s₁, s₂) = min{s₁, s₂} (ínfimo)
    
    Esto convierte (S, ≤) en un retículo (lattice) completo.
    """
    
    NOMINAL = auto()       # Sistema en homeostasis termodinámica
    UNKNOWN = auto()       # Estado epistémico: información insuficiente
    INESTABLE = auto()     # Fluctuaciones detectadas, λ_max cercano a 0
    SATURADO = auto()      # Aproximación a capacidad límite
    CRITICO = auto()       # Violación de invariantes de seguridad
    DISCONNECTED = auto()  # Fragmentación topológica (β₀ > 1)
    
    @property
    def severity(self) -> int:
        """
        Mapeo al espacio ordenado de enteros (ℤ, ≤).
        
        Returns:
            Nivel de severidad ∈ [0, 5]
            
        Postcondición:
            self ≤ other ⟺ self.severity ≤ other.severity
        """
        severity_map: Dict[SystemStatus, int] = {
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
        """
        Predicado de salud: sistema en región de atracción del equilibrio.
        
        Returns:
            True ⟺ self = NOMINAL
        """
        return self == SystemStatus.NOMINAL
    
    @property
    def is_critical(self) -> bool:
        """
        Predicado de criticidad: requiere intervención inmediata.
        
        Returns:
            True ⟺ self ∈ {CRITICO, DISCONNECTED}
        """
        return self in (SystemStatus.CRITICO, SystemStatus.DISCONNECTED)
    
    @property
    def requires_stabilization(self) -> bool:
        """
        Predicado: sistema requiere esfuerzo de control activo.
        
        Returns:
            True ⟺ self ∉ {NOMINAL, UNKNOWN}
        """
        return self not in (SystemStatus.NOMINAL, SystemStatus.UNKNOWN)
    
    @property
    def emoji(self) -> str:
        """
        Representación visual para interfaces humanas.
        
        Returns:
            Símbolo Unicode representativo del estado
        """
        emoji_map: Dict[SystemStatus, str] = {
            SystemStatus.NOMINAL: "✅",
            SystemStatus.UNKNOWN: "❓",
            SystemStatus.INESTABLE: "⚠️",
            SystemStatus.SATURADO: "🔶",
            SystemStatus.CRITICO: "🚨",
            SystemStatus.DISCONNECTED: "💀",
        }
        return emoji_map[self]
    
    def __lt__(self, other: SystemStatus) -> bool:
        """
        Implementación del orden parcial estricto.
        
        Args:
            other: Estado a comparar
            
        Returns:
            True ⟺ self.severity < other.severity
        """
        if not isinstance(other, SystemStatus):
            return NotImplemented
        return self.severity < other.severity
    
    def __le__(self, other: SystemStatus) -> bool:
        """Orden parcial no estricto (≤)."""
        if not isinstance(other, SystemStatus):
            return NotImplemented
        return self.severity <= other.severity
    
    def __gt__(self, other: SystemStatus) -> bool:
        """Orden parcial inverso estricto (>)."""
        if not isinstance(other, SystemStatus):
            return NotImplemented
        return self.severity > other.severity
    
    def __ge__(self, other: SystemStatus) -> bool:
        """Orden parcial inverso no estricto (≥)."""
        if not isinstance(other, SystemStatus):
            return NotImplemented
        return self.severity >= other.severity
    
    @classmethod
    def worst(cls, *statuses: SystemStatus) -> SystemStatus:
        """
        Operación JOIN del retículo: calcula el supremo.
        
        Args:
            *statuses: Secuencia de estados (puede estar vacía)
            
        Returns:
            max{statuses} según el orden ≤
            
        Postcondición:
            ∀s ∈ statuses: s ≤ result
            
        Casos especiales:
            - worst() = NOMINAL (elemento mínimo/bottom)
        """
        if not statuses:
            return cls.NOMINAL
        return max(statuses, key=lambda s: s.severity)
    
    @classmethod
    def best(cls, *statuses: SystemStatus) -> SystemStatus:
        """
        Operación MEET del retículo: calcula el ínfimo.
        
        Args:
            *statuses: Secuencia de estados (puede estar vacía)
            
        Returns:
            min{statuses} según el orden ≤
            
        Postcondición:
            ∀s ∈ statuses: result ≤ s
            
        Casos especiales:
            - best() = DISCONNECTED (elemento máximo/top)
        """
        if not statuses:
            return cls.DISCONNECTED
        return min(statuses, key=lambda s: s.severity)


class AgentDecision(Enum):
    """
    Decisiones del agente formando un Espacio de Acciones Discreto.
    
    ESTRUCTURA ALGEBRAICA
    ---------------------
    
    El conjunto D = {HEARTBEAT, EJECUTAR_LIMPIEZA, ...} representa el
    espacio finito de acciones disponibles. No tiene estructura de orden
    natural, pero se particiona en subconjuntos:
    
    - D_passive = {HEARTBEAT, WAIT}
    - D_active = {EJECUTAR_LIMPIEZA, AJUSTAR_VELOCIDAD, RECONNECT}
    - D_critical = {ALERTA_CRITICA}
    
    MAPEO ESTADO → DECISIÓN
    -----------------------
    
    Existe un morfismo (parcial) δ: SystemStatus → AgentDecision tal que:
    
    δ(NOMINAL) = HEARTBEAT
    δ(UNKNOWN) = WAIT
    δ(INESTABLE) = EJECUTAR_LIMPIEZA
    δ(SATURADO) = AJUSTAR_VELOCIDAD
    δ(CRITICO) = ALERTA_CRITICA
    δ(DISCONNECTED) = RECONNECT
    """
    
    HEARTBEAT = auto()           # Señal de vida (sistema nominal)
    EJECUTAR_LIMPIEZA = auto()   # Iniciar limpieza/estabilización
    AJUSTAR_VELOCIDAD = auto()   # Aplicar backpressure (throttling)
    ALERTA_CRITICA = auto()      # Emitir alerta de emergencia
    WAIT = auto()                # Esperar más información
    RECONNECT = auto()           # Reintentar conexión
    
    @property
    def requires_immediate_action(self) -> bool:
        """
        Predicado: decisión no admite debouncing temporal.
        
        Returns:
            True ⟺ self ∈ D_critical ∪ {RECONNECT}
            
        Justificación:
            Alertas críticas y reconexiones no deben retrasarse por políticas
            de rate-limiting, ya que corresponden a violaciones de invariantes.
        """
        return self in (AgentDecision.ALERTA_CRITICA, AgentDecision.RECONNECT)
    
    @property
    def is_active_control(self) -> bool:
        """
        Predicado: decisión implica inyección de control activo.
        
        Returns:
            True ⟺ self ∈ D_active
        """
        return self in (
            AgentDecision.EJECUTAR_LIMPIEZA,
            AgentDecision.AJUSTAR_VELOCIDAD,
            AgentDecision.RECONNECT,
        )
    
    @property
    def is_passive(self) -> bool:
        """
        Predicado: decisión no altera el estado del sistema.
        
        Returns:
            True ⟺ self ∈ D_passive
        """
        return self in (AgentDecision.HEARTBEAT, AgentDecision.WAIT)
    
    @property
    def emoji(self) -> str:
        """Representación visual para interfaces humanas."""
        emoji_map: Dict[AgentDecision, str] = {
            AgentDecision.HEARTBEAT: "💓",
            AgentDecision.EJECUTAR_LIMPIEZA: "🧹",
            AgentDecision.AJUSTAR_VELOCIDAD: "🔽",
            AgentDecision.ALERTA_CRITICA: "🚨",
            AgentDecision.WAIT: "⏳",
            AgentDecision.RECONNECT: "🔄",
        }
        return emoji_map[self]
    
    @property
    def expected_duration_seconds(self) -> Optional[float]:
        """
        Duración esperada de la acción (tiempo hasta efecto observable).
        
        Returns:
            Duración en segundos, None si no aplica
            
        Usada para ajustar ventanas de debounce.
        """
        duration_map: Dict[AgentDecision, Optional[float]] = {
            AgentDecision.HEARTBEAT: None,
            AgentDecision.EJECUTAR_LIMPIEZA: 30.0,
            AgentDecision.AJUSTAR_VELOCIDAD: 60.0,
            AgentDecision.ALERTA_CRITICA: None,
            AgentDecision.WAIT: None,
            AgentDecision.RECONNECT: 10.0,
        }
        return duration_map[self]


# ============================================================================
# ESTRUCTURAS DE DATOS INMUTABLES CON INVARIANTES
# ============================================================================

@dataclass(frozen=True)
class ThresholdConfig:
    """
    Configuración de umbrales para clasificación de métricas.
    
    INVARIANTES ALGEBRAICOS
    -----------------------
    
    Para cada par (warning, critical):
    
    1. **Orden estricto**: 0 ≤ warning < critical ≤ 1
    2. **Separación mínima**: critical - warning ≥ ε (histeresis)
    
    JUSTIFICACIÓN FÍSICA
    -------------------
    
    Los umbrales definen una partición del espacio [0,1] en regiones:
    
    - [0, warning): NOMINAL (región segura)
    - [warning, critical): WARNING (zona de precaución)
    - [critical, 1]: CRITICAL (zona de peligro)
    
    La histeresis previene oscilaciones (chattering) en la frontera.
    """
    
    # Umbrales de voltaje de flyback (normalizado)
    flyback_voltage_warning: float = 0.5
    flyback_voltage_critical: float = 0.8
    
    # Umbrales de saturación (normalizado)
    saturation_warning: float = 0.9
    saturation_critical: float = 0.95
    
    # Histeresis mínima (prevenir chattering)
    min_hysteresis: float = 0.05
    
    def __post_init__(self) -> None:
        """
        Valida invariantes tras construcción.
        
        Raises:
            ConfigurationError: Si se viola algún invariante
        """
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
    
    def _validate_pair(self, name: str, warning: float, critical: float) -> None:
        """
        Valida un par de umbrales.
        
        Args:
            name: Nombre de la métrica (para error messages)
            warning: Umbral de advertencia
            critical: Umbral crítico
            
        Raises:
            ConfigurationError: Si se viola la condición de orden
            
        Invariantes verificados:
            1. 0 ≤ warning < critical ≤ 1
            2. critical - warning ≥ min_hysteresis
        """
        # Validar rango
        if not (0.0 <= warning < critical <= 1.0):
            raise ConfigurationError(
                f"Umbrales {name} inválidos: debe cumplir 0 ≤ warning({warning:.3f}) "
                f"< critical({critical:.3f}) ≤ 1.0"
            )
        
        # Validar histeresis
        hysteresis = critical - warning
        if hysteresis < self.min_hysteresis:
            raise ConfigurationError(
                f"Histeresis {name} insuficiente: {hysteresis:.3f} < {self.min_hysteresis:.3f}. "
                f"Riesgo de chattering en frontera."
            )
    
    def classify_voltage(self, value: float) -> str:
        """
        Clasifica un valor de voltaje en su región correspondiente.
        
        Args:
            value: Voltaje normalizado ∈ [0, 1]
            
        Returns:
            Etiqueta de clasificación: "nominal" | "warning" | "critical"
            
        Postcondición:
            resultado ∈ {"nominal", "warning", "critical"}
        """
        if value >= self.flyback_voltage_critical:
            return "critical"
        if value >= self.flyback_voltage_warning:
            return "warning"
        return "nominal"
    
    def classify_saturation(self, value: float) -> str:
        """
        Clasifica un valor de saturación en su región correspondiente.
        
        Args:
            value: Saturación normalizada ∈ [0, 1]
            
        Returns:
            Etiqueta de clasificación: "nominal" | "warning" | "critical"
        """
        if value >= self.saturation_critical:
            return "critical"
        if value >= self.saturation_warning:
            return "warning"
        return "nominal"
    
    def get_voltage_margin(self, value: float) -> float:
        """
        Calcula margen de seguridad respecto al umbral crítico.
        
        Args:
            value: Voltaje normalizado
            
        Returns:
            Margen = (critical - value) / critical ∈ [-∞, 1]
            
        Interpretación:
            - margin > 0: zona segura
            - margin = 0: en el umbral
            - margin < 0: violación del umbral
        """
        if self.flyback_voltage_critical == 0:
            return math.inf if value == 0 else -math.inf
        return (self.flyback_voltage_critical - value) / self.flyback_voltage_critical
    
    def get_saturation_margin(self, value: float) -> float:
        """Calcula margen de seguridad de saturación."""
        if self.saturation_critical == 0:
            return math.inf if value == 0 else -math.inf
        return (self.saturation_critical - value) / self.saturation_critical


@dataclass(frozen=True)
class ConnectionConfig:
    """
    Configuración de conexión HTTP con políticas de resiliencia.
    
    TEORÍA DE COLAS Y POOLING
    --------------------------
    
    El pool de conexiones implementa un modelo M/M/c con:
    - pool_connections: número de workers (c)
    - pool_maxsize: tamaño del buffer
    
    Para prevenir deadlocks: pool_maxsize ≥ pool_connections
    """
    
    # URL base del Core API
    base_url: str = DefaultConfig.CORE_URL
    
    # Timeout de requests individuales (segundos)
    request_timeout: int = DefaultConfig.REQUEST_TIMEOUT
    
    # Política de reintentos exponenciales
    max_retries: int = 3
    backoff_factor: float = 0.5  # T_n = backoff_factor * 2^n
    
    # Pool de conexiones
    pool_connections: int = 10
    pool_maxsize: int = 10
    
    # Status codes que disparan retry
    retry_status_codes: Tuple[int, ...] = (500, 502, 503, 504)
    
    def __post_init__(self) -> None:
        """
        Valida configuración de conexión.
        
        Raises:
            ConfigurationError: Si hay inconsistencias
        """
        # Validar timeout positivo
        if self.request_timeout <= 0:
            raise ConfigurationError(
                f"request_timeout debe ser positivo, recibido: {self.request_timeout}"
            )
        
        # Validar pool
        if self.pool_maxsize < self.pool_connections:
            raise ConfigurationError(
                f"pool_maxsize ({self.pool_maxsize}) < pool_connections ({self.pool_connections}). "
                f"Riesgo de deadlock."
            )
        
        # Validar backoff
        if self.backoff_factor < 0:
            raise ConfigurationError(
                f"backoff_factor debe ser no negativo: {self.backoff_factor}"
            )
    
    @property
    def telemetry_endpoint(self) -> str:
        """
        URL completa del endpoint de telemetría.
        
        Returns:
            URL absoluta
        """
        return f"{self.base_url}/api/telemetry/status"
    
    def tools_endpoint(self, vector: str) -> str:
        """
        URL del endpoint de herramientas para un vector específico.
        
        Args:
            vector: Nombre del vector (e.g., "clean", "configure")
            
        Returns:
            URL absoluta del endpoint
        """
        return f"{self.base_url}/api/tools/{vector}"
    
    def calculate_retry_delay(self, attempt: int) -> float:
        """
        Calcula delay para un intento de retry (exponential backoff).
        
        Args:
            attempt: Número de intento (0-indexed)
            
        Returns:
            Delay en segundos = backoff_factor * 2^attempt
            
        Ejemplo:
            backoff_factor=0.5 → delays: [0.5, 1.0, 2.0, 4.0, ...]
        """
        return self.backoff_factor * (2 ** attempt)


@dataclass(frozen=True)
class TopologyConfig:
    """
    Configuración de análisis topológico y homología persistente.
    
    PARÁMETROS DE FILTRACIÓN
    ------------------------
    
    - max_history: Tamaño del buffer circular de snapshots topológicos
    - persistence_window: Ancho de la ventana deslizante para análisis temporal
    
    UMBRALES DE SALUD
    -----------------
    
    Definen el mapeo TopologicalHealth.score → HealthLevel:
    
    - [critical_threshold, warning_threshold): UNHEALTHY
    - [warning_threshold, 1.0]: HEALTHY
    - [0, critical_threshold): CRITICAL
    """
    
    # Historia de snapshots topológicos
    max_history: int = DefaultConfig.TOPOLOGY_HISTORY
    
    # Ventana de persistencia temporal
    persistence_window: int = DefaultConfig.PERSISTENCE_WINDOW
    
    # Umbrales de salud topológica
    health_critical_threshold: float = 0.4
    health_warning_threshold: float = 0.7
    
    # Topología esperada (1-esqueleto nominal)
    expected_edges: Tuple[Tuple[str, str], ...] = NetworkTopology.EXPECTED_EDGES
    
    def __post_init__(self) -> None:
        """Valida configuración topológica."""
        # Validar umbrales
        if not (0.0 <= self.health_critical_threshold < self.health_warning_threshold <= 1.0):
            raise ConfigurationError(
                f"Umbrales de salud inválidos: debe cumplir 0 ≤ critical({self.health_critical_threshold}) "
                f"< warning({self.health_warning_threshold}) ≤ 1"
            )
        
        # Validar ventanas
        if self.persistence_window <= 0:
            raise ConfigurationError(
                f"persistence_window debe ser positiva: {self.persistence_window}"
            )
        
        if self.max_history < self.persistence_window:
            raise ConfigurationError(
                f"max_history ({self.max_history}) debe ser ≥ persistence_window ({self.persistence_window})"
            )
    
    def classify_health_score(self, score: float) -> HealthLevel:
        """
        Mapea un score continuo a un nivel discreto de salud.
        
        Args:
            score: Score de salud ∈ [0, 1]
            
        Returns:
            Nivel de salud: CRITICAL | UNHEALTHY | HEALTHY
        """
        if score < self.health_critical_threshold:
            return HealthLevel.CRITICAL
        if score < self.health_warning_threshold:
            return HealthLevel.UNHEALTHY
        return HealthLevel.HEALTHY


@dataclass(frozen=True)
class TimingConfig:
    """
    Configuración de parámetros temporales del ciclo OODA.
    
    TEOREMA DE NYQUIST-SHANNON
    --------------------------
    
    Para capturar dinámicas con frecuencia característica f_signal:
    
    check_interval ≤ 1 / (2 * f_signal)
    
    Con señales esperadas en escala de 1-10 segundos → check_interval ~ 10s
    
    POLÍTICA DE BACKOFF EXPONENCIAL
    -------------------------------
    
    Para cold start, el delay entre intentos sigue:
    
    T_n = min(startup_backoff_initial * multiplier^n, startup_backoff_max)
    """
    
    # Intervalo del ciclo OODA (segundos)
    check_interval: int = DefaultConfig.CHECK_INTERVAL
    
    # Ventana de debounce para supresión de acciones repetidas (segundos)
    debounce_window_seconds: int = DefaultConfig.DEBOUNCE_WINDOW
    
    # Fallos consecutivos antes de degradación
    max_consecutive_failures: int = DefaultConfig.MAX_CONSECUTIVE_FAILURES
    
    # Parámetros de cold start (backoff exponencial)
    startup_backoff_initial: float = 5.0
    startup_backoff_max: float = 60.0
    startup_backoff_multiplier: float = 1.5
    startup_max_attempts: int = 20
    
    def __post_init__(self) -> None:
        """Valida configuración temporal."""
        # Validar intervalos positivos
        for attr in ["check_interval", "debounce_window_seconds", "max_consecutive_failures"]:
            value = getattr(self, attr)
            if value <= 0:
                raise ConfigurationError(f"{attr} debe ser positivo: {value}")
        
        # Validar backoff
        if self.startup_backoff_initial <= 0:
            raise ConfigurationError(
                f"startup_backoff_initial debe ser positivo: {self.startup_backoff_initial}"
            )
        
        if self.startup_backoff_max < self.startup_backoff_initial:
            raise ConfigurationError(
                f"startup_backoff_max ({self.startup_backoff_max}) < initial ({self.startup_backoff_initial})"
            )
        
        if self.startup_backoff_multiplier <= 1.0:
            raise ConfigurationError(
                f"startup_backoff_multiplier debe ser > 1.0: {self.startup_backoff_multiplier}"
            )
    
    def calculate_startup_delay(self, attempt: int) -> float:
        """
        Calcula delay para cold start con backoff exponencial saturado.
        
        Args:
            attempt: Número de intento (0-indexed)
            
        Returns:
            Delay en segundos
            
        Postcondición:
            startup_backoff_initial ≤ resultado ≤ startup_backoff_max
        """
        unbounded_delay = self.startup_backoff_initial * (self.startup_backoff_multiplier ** attempt)
        return min(unbounded_delay, self.startup_backoff_max)
    
    def should_debounce(
        self,
        last_decision_time: Optional[datetime],
        current_time: Optional[datetime] = None,
    ) -> bool:
        """
        Determina si una acción debe ser suprimida por debounce.
        
        Args:
            last_decision_time: Timestamp de la última decisión del mismo tipo
            current_time: Timestamp actual (default: now)
            
        Returns:
            True si debe suprimirse (dentro de ventana de debounce)
        """
        if last_decision_time is None:
            return False
        
        if current_time is None:
            current_time = datetime.now()
        
        elapsed = (current_time - last_decision_time).total_seconds()
        return elapsed < self.debounce_window_seconds


@dataclass(frozen=True)
class AgentConfig:
    """
    Configuración consolidada del agente autónomo.
    
    PRINCIPIO DE COMPOSICIÓN
    ------------------------
    
    Esta clase agrega todas las configuraciones mediante composición
    (has-a), no herencia, siguiendo el principio de Liskov.
    
    INMUTABILIDAD
    -------------
    
    frozen=True garantiza inmutabilidad profunda, previniendo:
    - Race conditions en acceso concurrente
    - Mutaciones accidentales
    - Violaciones de invariantes post-construcción
    """
    
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    
    @classmethod
    def from_environment(cls) -> AgentConfig:
        """
        Factory method: construye configuración desde variables de entorno.
        
        Variables soportadas:
            - CORE_API_URL: URL del Core (default: http://localhost:5002)
            - CHECK_INTERVAL: Intervalo del ciclo OODA en segundos (default: 10)
            - REQUEST_TIMEOUT: Timeout de HTTP requests en segundos (default: 10)
            - PERSISTENCE_WINDOW_SIZE: Tamaño de ventana de persistencia (default: 20)
            - LOG_LEVEL: Nivel de logging (default: INFO)
            
        Returns:
            Configuración construida desde env vars con fallback a defaults
            
        Raises:
            ConfigurationError: Si hay valores inválidos en env vars
        """
        # Parsear URL del Core
        core_url = os.getenv("CORE_API_URL", DefaultConfig.CORE_URL)
        core_url = cls._validate_and_normalize_url(core_url)
        
        # Parsear parámetros temporales
        check_interval = cls._parse_env_int("CHECK_INTERVAL", DefaultConfig.CHECK_INTERVAL)
        request_timeout = cls._parse_env_int("REQUEST_TIMEOUT", DefaultConfig.REQUEST_TIMEOUT)
        persistence_window = cls._parse_env_int(
            "PERSISTENCE_WINDOW_SIZE",
            DefaultConfig.PERSISTENCE_WINDOW,
        )
        
        # Construir configuración compuesta
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
        """
        Valida y normaliza una URL HTTP(S).
        
        Args:
            url: URL a validar (puede omitir esquema)
            
        Returns:
            URL normalizada con esquema explícito
            
        Raises:
            ConfigurationError: Si la URL es inválida
            
        Transformaciones aplicadas:
            1. Agregar http:// si falta esquema
            2. Remover trailing slash
            3. Validar netloc no vacío
        """
        if not url or not url.strip():
            raise ConfigurationError("CORE_API_URL no puede estar vacía")
        
        url = url.strip()
        
        # Agregar esquema si falta
        if not url.lower().startswith(("http://", "https://")):
            url = f"http://{url}"
        
        # Validar estructura con urllib
        try:
            parsed = urlparse(url)
            
            if not parsed.netloc:
                raise ConfigurationError(f"URL sin host válido: {url}")
            
            if parsed.scheme not in ("http", "https"):
                raise ConfigurationError(
                    f"Esquema no soportado '{parsed.scheme}', se espera http/https"
                )
                
        except Exception as e:
            raise ConfigurationError(f"URL inválida '{url}': {e}")
        
        # Normalizar: remover trailing slash
        return url.rstrip("/")
    
    @staticmethod
    def _parse_env_int(name: str, default: int) -> int:
        """
        Parsea un entero desde variable de entorno con validación.
        
        Args:
            name: Nombre de la variable de entorno
            default: Valor por defecto si la var no existe
            
        Returns:
            Valor parseado (debe ser > 0)
            
        Comportamiento:
            - Si la var no existe → retorna default
            - Si la var existe pero es inválida → warning + retorna default
            - Si la var existe y es válida → retorna valor
        """
        env_value = os.getenv(name)
        if env_value is None:
            return default
        
        try:
            value = int(env_value)
            if value <= 0:
                raise ValueError("Debe ser positivo")
            return value
        except (TypeError, ValueError) as e:
            logger.warning(
                f"Valor inválido para {name}='{env_value}' ({e}), "
                f"usando default={default}"
            )
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa la configuración a diccionario (para logging/debugging).
        
        Returns:
            Dict con estructura anidada reflejando la composición
        """
        return {
            "thresholds": {
                "flyback_voltage": {
                    "warning": self.thresholds.flyback_voltage_warning,
                    "critical": self.thresholds.flyback_voltage_critical,
                },
                "saturation": {
                    "warning": self.thresholds.saturation_warning,
                    "critical": self.thresholds.saturation_critical,
                },
            },
            "connection": {
                "base_url": self.connection.base_url,
                "request_timeout": self.connection.request_timeout,
                "max_retries": self.connection.max_retries,
            },
            "topology": {
                "max_history": self.topology.max_history,
                "persistence_window": self.topology.persistence_window,
                "expected_edges": [list(edge) for edge in self.topology.expected_edges],
            },
            "timing": {
                "check_interval": self.timing.check_interval,
                "debounce_window": self.timing.debounce_window_seconds,
                "max_consecutive_failures": self.timing.max_consecutive_failures,
            },
        }


# ============================================================================
# ESTRUCTURAS DE DATOS DE TELEMETRÍA
# ============================================================================

@dataclass
class TelemetryData:
    """
    Datos de telemetría estructurados con normalización y validación.
    
    ESPACIO DE FASE
    ---------------
    
    Los datos de telemetría representan un punto en el espacio de fase:
    
    Ψ: M → ℝ³,  φ ↦ (V, S, I)
    
    donde:
    - V: Voltaje de flyback normalizado ∈ [0, 1]
    - S: Saturación normalizada ∈ [0, 1]
    - I: Integrity score ∈ [0, 1]
    
    NORMALIZACIÓN
    -------------
    
    Todos los valores se clampean al hipercubo unitario [0,1]³ para:
    1. Prevenir desbordamientos numéricos
    2. Garantizar precondiciones de clasificadores
    3. Facilitar comparaciones métricas
    
    INVARIANTES
    -----------
    
    Post-construcción:
    - 0 ≤ flyback_voltage ≤ 1
    - 0 ≤ saturation ≤ 1
    - 0 ≤ integrity_score ≤ 1
    """
    
    flyback_voltage: float
    saturation: float
    integrity_score: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    # Flag interno: indica si hubo clamping
    _clamped: bool = field(default=False, repr=False, compare=False)
    
    def __post_init__(self) -> None:
        """
        Post-procesamiento: normaliza valores al hipercubo [0,1]³.
        
        Side effects:
            - Clampea valores fuera de rango
            - Marca _clamped=True si hubo ajustes
        """
        # Guardar valores originales
        original_v = self.flyback_voltage
        original_s = self.saturation
        original_i = self.integrity_score
        
        # Clampear al rango válido
        object.__setattr__(self, "flyback_voltage", self._clamp(self.flyback_voltage))
        object.__setattr__(self, "saturation", self._clamp(self.saturation))
        object.__setattr__(self, "integrity_score", self._clamp(self.integrity_score))
        
        # Detectar si hubo clamping
        clamped = (
            original_v != self.flyback_voltage
            or original_s != self.saturation
            or original_i != self.integrity_score
        )
        object.__setattr__(self, "_clamped", clamped)
        
        # Advertir si hubo clamping significativo
        if clamped:
            changes = []
            if original_v != self.flyback_voltage:
                changes.append(f"voltage: {original_v:.4f}→{self.flyback_voltage:.4f}")
            if original_s != self.saturation:
                changes.append(f"saturation: {original_s:.4f}→{self.saturation:.4f}")
            if original_i != self.integrity_score:
                changes.append(f"integrity: {original_i:.4f}→{self.integrity_score:.4f}")
            
            logger.debug(f"[TELEMETRY:CLAMP] {', '.join(changes)}")
    
    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Clampea un valor al intervalo [min_val, max_val].
        
        Args:
            value: Valor a clampear
            min_val: Límite inferior
            max_val: Límite superior
            
        Returns:
            max(min_val, min(max_val, value))
            
        Casos especiales:
            - NaN → min_val
            - ±∞ → min_val o max_val según signo
        """
        if math.isnan(value):
            return min_val
        if math.isinf(value):
            return max_val if value > 0 else min_val
        return max(min_val, min(max_val, value))
    
    def __eq__(self, other: Any) -> bool:
        """
        Igualdad semántica: compara valores, ignora timestamp.
        
        Args:
            other: Objeto a comparar
            
        Returns:
            True si los valores numéricos coinciden (≈ con tolerancia)
        """
        if not isinstance(other, TelemetryData):
            return NotImplemented
        
        return (
            math.isclose(self.flyback_voltage, other.flyback_voltage, abs_tol=PhysicalConstants.EPSILON)
            and math.isclose(self.saturation, other.saturation, abs_tol=PhysicalConstants.EPSILON)
            and math.isclose(self.integrity_score, other.integrity_score, abs_tol=PhysicalConstants.EPSILON)
            and self.raw_data == other.raw_data
        )
    
    @classmethod
    def from_dict(cls, data: Any) -> Optional[TelemetryData]:
        """
        Factory method: construye TelemetryData desde dict con paths flexibles.
        
        Args:
            data: Diccionario con telemetría (puede estar anidado)
            
        Returns:
            TelemetryData si se extraen métricas, None si el dict es inválido
            
        Estrategia de extracción:
            1. Buscar en paths prioritarios (flux_condenser.*)
            2. Fallback a keys de nivel superior
            3. Defaults conservadores si no se encuentra nada
        """
        if not isinstance(data, dict):
            logger.warning(
                f"[TELEMETRY:PARSE] Tipo inválido: esperado dict, recibido {type(data).__name__}"
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
        
        # Extraer métricas con múltiples paths de fallback
        flyback = cls._extract_metric(data, voltage_paths)
        saturation = cls._extract_metric(data, saturation_paths)
        integrity = cls._extract_numeric(data, "integrity_score", 1.0)
        
        # Log si no hay datos útiles
        if flyback is None and saturation is None:
            logger.debug(
                "[TELEMETRY:PARSE] Sin métricas válidas en dict, usando defaults (0.0, 0.0)"
            )
        
        # Advertir para valores fuera de rango (antes de clamping)
        for name, val in [("flyback_voltage", flyback), ("saturation", saturation)]:
            if val is not None and not (0.0 <= val <= 1.0):
                logger.warning(
                    f"[TELEMETRY:PARSE] {name}={val:.4f} fuera de rango [0,1], "
                    f"será clampeado"
                )
        
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
        """
        Extrae una métrica buscando en múltiples paths con navegación de dict anidados.
        
        Args:
            data: Diccionario fuente
            paths: Secuencia de paths a intentar (e.g., "a.b.c")
            
        Returns:
            Primer valor numérico encontrado, None si ninguno existe
            
        Ejemplo:
            data = {"metrics": {"voltage": 0.5}}
            _extract_metric(data, ("metrics.voltage", "v")) → 0.5
        """
        # Buscar en namespace 'metrics' si existe
        search_spaces = [data]
        if isinstance(data.get("metrics"), dict):
            search_spaces.insert(0, data["metrics"])
        
        for space in search_spaces:
            for path in paths:
                # Soportar navegación anidada con '.'
                value = cls._navigate_dict_path(space, path)
                
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
        
        return None
    
    @staticmethod
    def _navigate_dict_path(data: Dict[str, Any], path: str) -> Any:
        """
        Navega un dict anidado usando notación de puntos.
        
        Args:
            data: Diccionario a navegar
            path: Ruta con puntos (e.g., "a.b.c")
            
        Returns:
            Valor encontrado, None si el path no existe
            
        Ejemplo:
            data = {"a": {"b": {"c": 42}}}
            _navigate_dict_path(data, "a.b.c") → 42
        """
        keys = path.split(".")
        current = data
        
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if current is None:
                return None
        
        return current
    
    @classmethod
    def _extract_numeric(
        cls,
        data: Dict[str, Any],
        key: str,
        default: float,
    ) -> float:
        """
        Extrae un valor numérico con default y sanitización.
        
        Args:
            data: Diccionario fuente
            key: Clave a buscar
            default: Valor por defecto si no existe o es inválido
            
        Returns:
            Valor numérico extraído o default
        """
        value = data.get(key)
        if value is None:
            return default
        
        try:
            result = float(value)
            # Rechazar NaN/Inf
            if math.isnan(result) or math.isinf(result):
                return default
            return result
        except (TypeError, ValueError):
            return default
    
    @property
    def is_idle(self) -> bool:
        """
        Predicado: telemetría representa estado idle (sin actividad).
        
        Returns:
            True ⟺ (V, S) = (0, 0)
        """
        return (
            abs(self.flyback_voltage) < PhysicalConstants.EPSILON
            and abs(self.saturation) < PhysicalConstants.EPSILON
        )
    
    @property
    def was_clamped(self) -> bool:
        """
        Predicado: algún valor fue ajustado durante construcción.
        
        Returns:
            True si hubo clamping
        """
        return self._clamped
    
    @property
    def norm_l2(self) -> float:
        """
        Calcula la norma L2 en el espacio de fase (distancia euclidiana al origen).
        
        Returns:
            ‖Ψ‖₂ = √(V² + S² + I²)
        """
        return math.sqrt(
            self.flyback_voltage ** 2
            + self.saturation ** 2
            + self.integrity_score ** 2
        )
    
    @property
    def norm_linf(self) -> float:
        """
        Calcula la norma L∞ (máxima componente).
        
        Returns:
            ‖Ψ‖∞ = max(|V|, |S|, |I|)
        """
        return max(
            abs(self.flyback_voltage),
            abs(self.saturation),
            abs(self.integrity_score),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa a diccionario (para JSON/logging).
        
        Returns:
            Dict con valores redondeados para legibilidad
        """
        return {
            "flyback_voltage": round(self.flyback_voltage, 4),
            "saturation": round(self.saturation, 4),
            "integrity_score": round(self.integrity_score, 4),
            "timestamp": self.timestamp.isoformat(),
            "is_idle": self.is_idle,
            "was_clamped": self.was_clamped,
            "norm_l2": round(self.norm_l2, 4),
        }
    
    def __repr__(self) -> str:
        """Representación compacta para debugging."""
        return (
            f"TelemetryData(V={self.flyback_voltage:.3f}, S={self.saturation:.3f}, "
            f"I={self.integrity_score:.3f}{'*' if self._clamped else ''})"
        )


# ============================================================================
# PROTOCOLOS DE EVALUACIÓN DE ESTADO
# ============================================================================

@runtime_checkable
class StateEvaluator(Protocol):
    """
    Protocolo para evaluadores de estado basados en Teoría del Potencial.
    
    FUNDAMENTACIÓN MATEMÁTICA
    -------------------------
    
    Cada evaluador implementa un Funtor de Gradiente que mapea el estado
    multidimensional del sistema a un campo vectorial de penalización:
    
        ∇Vᵢ: M → T*M
    
    donde:
    - M es la variedad de estados (espacio de fase)
    - T*M es el fibrado cotangente (espacio de 1-formas)
    - Vᵢ: M → ℝ₊ es una función de Lyapunov local
    
    COMPOSICIÓN LINEAL
    ------------------
    
    El gradiente total se construye por superposición:
    
        ∇V_total = Σᵢ wᵢ · ∇Vᵢ
    
    donde wᵢ son pesos implícitos (actualmente unitarios).
    
    INVARIANZA DE GAUGE
    -------------------
    
    Para garantizar que ∇V_total ∈ im(B₁ᵀ) (condición de neutralidad de carga),
    el Sintetizador Hamiltoniano realiza una proyección ortogonal:
    
        ∇V_renorm = ∇V_total - ⟨∇V_total, 𝟙⟩ · 𝟙
    
    donde 𝟙 = (1/√N, ..., 1/√N) es el vector constante normalizado.
    
    CONTRATO DEL PROTOCOLO
    ----------------------
    
    Cada evaluador debe implementar:
    
    1. **compute_gradient**: Retorna ∇Vᵢ ∈ ℝᴺ
       - Precondición: num_nodes ≥ 1
       - Postcondición: ‖∇Vᵢ‖ < ∞, ∇Vᵢ ≥ 0 (penalizaciones no negativas)
    
    2. **evaluate**: Retorna (SystemStatus, str) o None (método legacy)
       - Usado para compatibilidad con lógica decisional anterior
       - Será deprecado en favor de evaluación puramente hamiltoniana
    """
    
    @property
    def name(self) -> str:
        """
        Identificador único del evaluador.
        
        Returns:
            Nombre canónico (e.g., "FragmentationEvaluator")
        """
        ...
    
    @property
    def priority(self) -> int:
        """
        Prioridad de evaluación (menor = más urgente).
        
        Returns:
            Entero que define orden de evaluación en cadena legacy
            
        Nota:
            En el enfoque Hamiltoniano puro, este campo es obsoleto
            ya que todos los gradientes se suman independientemente.
        """
        ...
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """
        Calcula el gradiente de penalización ∇Vᵢ ∈ ℝᴺ.
        
        Args:
            telemetry: Estado actual de telemetría (puede ser None)
            topo_health: Salud topológica del sistema
            voltage_analysis: Análisis de persistencia de voltaje
            saturation_analysis: Análisis de persistencia de saturación
            config: Configuración de umbrales
            metrics: Métricas internas del agente
            num_nodes: Dimensión del espacio de fase (N)
            
        Returns:
            Vector gradiente ∇Vᵢ ∈ ℝᴺ con:
            - ‖∇Vᵢ‖₂ < ∞ (norma finita)
            - ∀j: (∇Vᵢ)ⱼ ≥ 0 (componentes no negativas)
            
        Invariantes:
            - len(resultado) == num_nodes
            - np.all(np.isfinite(resultado))
            - np.all(resultado >= 0)
            
        Semántica de las componentes:
            (∇Vᵢ)ⱼ = magnitud de penalización sobre el nodo j
            
        Ejemplo:
            Si el nodo "Core" (índice 1) presenta voltaje crítico:
            ∇V_voltage = [0, 0.8, 0, 0]
        """
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
        Evalúa el estado del sistema (método legacy para compatibilidad).
        
        Args:
            (mismos que compute_gradient, excepto num_nodes)
            
        Returns:
            - (SystemStatus, mensaje) si el evaluador detecta una condición
            - None si no aplica
            
        Nota de deprecación:
            Este método será removido en versiones futuras. La lógica
            decisional debe derivarse únicamente del campo ∇V_total.
        """
        ...


# ============================================================================
# CLASE BASE PARA EVALUADORES
# ============================================================================

@dataclass(frozen=True)
class BaseEvaluator:
    """
    Clase base abstracta para evaluadores con mapeo canónico de nodos.
    
    MAPEO CANÓNICO DEL ESPACIO DE FASE
    -----------------------------------
    
    Establece un isomorfismo entre nombres de nodos y índices:
    
        Φ: {Agent, Core, Redis, Filesystem} → {0, 1, 2, 3}
    
    Este mapeo es fundamental para la construcción de gradientes
    y debe ser consistente con la matriz de incidencia B₁ del
    GaugeFieldRouter.
    
    INVARIANTE GLOBAL
    -----------------
    
    Para todo evaluador E:
        E.NODE_INDEX_MAP == BaseEvaluator.NODE_INDEX_MAP
    
    Esto garantiza coherencia entre evaluadores independientes.
    """
    
    # Mapeo canónico nodo → índice
    NODE_INDEX_MAP: Final[Mapping[str, int]] = field(
        default_factory=lambda: {
            NetworkTopology.NODE_AGENT: 0,
            NetworkTopology.NODE_CORE: 1,
            NetworkTopology.NODE_REDIS: 2,
            NetworkTopology.NODE_FILESYSTEM: 3,
        }
    )
    
    def _get_node_idx(self, node_name: str) -> Optional[int]:
        """
        Mapea nombre de nodo a índice del espacio de gradientes.
        
        Args:
            node_name: Nombre canónico del nodo
            
        Returns:
            Índice j ∈ {0, 1, 2, 3} si el nodo existe, None en caso contrario
            
        Ejemplo:
            _get_node_idx("Core") → 1
            _get_node_idx("Unknown") → None
        """
        return self.NODE_INDEX_MAP.get(node_name)
    
    def _get_node_name(self, index: int) -> Optional[str]:
        """
        Mapeo inverso: índice → nombre de nodo.
        
        Args:
            index: Índice en el espacio de fase
            
        Returns:
            Nombre del nodo si index ∈ {0,1,2,3}, None en caso contrario
        """
        inverse_map = {v: k for k, v in self.NODE_INDEX_MAP.items()}
        return inverse_map.get(index)
    
    def _validate_gradient(self, gradient: np.ndarray, num_nodes: int) -> None:
        """
        Valida que un gradiente cumpla los invariantes del protocolo.
        
        Args:
            gradient: Vector a validar
            num_nodes: Dimensión esperada
            
        Raises:
            ValueError: Si el gradiente es inválido
            
        Invariantes verificados:
            1. len(gradient) == num_nodes
            2. np.all(np.isfinite(gradient))
            3. np.all(gradient >= 0)
        """
        if len(gradient) != num_nodes:
            raise ValueError(
                f"Dimensión inválida: esperado {num_nodes}, recibido {len(gradient)}"
            )
        
        if not np.all(np.isfinite(gradient)):
            raise ValueError(
                f"Gradiente contiene NaN/Inf: {gradient}"
            )
        
        if not np.all(gradient >= -PhysicalConstants.EPSILON):
            raise ValueError(
                f"Gradiente contiene valores negativos: {gradient}"
            )


# ============================================================================
# EVALUADORES CONCRETOS
# ============================================================================

@dataclass(frozen=True)
class FragmentationEvaluator(BaseEvaluator):
    """
    Evaluador de fragmentación topológica (violación de conectividad).
    
    FUNCIÓN DE LYAPUNOV
    -------------------
    
    V_frag(β₀) = (β₀ - 1)² / (N - 1)
    
    donde:
    - β₀ = número de componentes conexas (número de Betti 0)
    - N = número total de nodos
    
    GRADIENTE
    ---------
    
    ∇V_frag asigna penalización uniforme a nodos desconectados:
    
        (∇V_frag)ⱼ = {
            (β₀ - 1) / (N - 1)  si j ∈ nodos_desconectados
            0                    en otro caso
        }
    
    INTERPRETACIÓN FÍSICA
    ---------------------
    
    La penalización crece linealmente con el número de componentes
    adicionales, incentivando la reconexión de islas topológicas.
    
    INVARIANTE DE SALUD
    -------------------
    
    Sistema sano ⟺ β₀ = 1 (grafo conexo)
    """
    
    name: str = "FragmentationEvaluator"
    priority: int = 10
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """
        Calcula gradiente de penalización por fragmentación.
        
        Postcondición:
            Si β₀ > 1: ‖∇V_frag‖₀ = |nodos_desconectados| (norma L0)
            Si β₀ = 1: ∇V_frag = 0
        """
        grad = np.zeros(num_nodes, dtype=np.float64)
        
        # Solo penalizar si hay fragmentación
        if not topo_health.betti.is_connected:
            # Severidad normalizada por tamaño del grafo
            num_components = topo_health.betti.b0
            max_components = max(1, num_nodes - 1)  # Evitar división por cero
            severity = (num_components - 1) / max_components
            
            # Asignar penalización a nodos desconectados
            for node in topo_health.disconnected_nodes:
                idx = self._get_node_idx(node)
                if idx is not None and idx < num_nodes:
                    grad[idx] = severity
            
            logger.debug(
                f"[FRAG:GRAD] β₀={num_components}, severidad={severity:.3f}, "
                f"nodos_afectados={list(topo_health.disconnected_nodes)}"
            )
        
        self._validate_gradient(grad, num_nodes)
        return grad
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """Evaluación legacy: fragmentación → DISCONNECTED."""
        if not topo_health.betti.is_connected:
            nodes = ", ".join(sorted(topo_health.disconnected_nodes)) or "∅"
            return (
                SystemStatus.DISCONNECTED,
                f"Fragmentación Topológica: β₀={topo_health.betti.b0}, "
                f"nodos_desconectados=[{nodes}]",
            )
        return None


@dataclass(frozen=True)
class NoTelemetryEvaluator(BaseEvaluator):
    """
    Evaluador de ausencia de telemetría (incertidumbre epistémica).
    
    FUNCIÓN DE LYAPUNOV
    -------------------
    
    V_nodata(n_fallos) = min(1, α · n_fallos)
    
    donde α = 0.2 es el factor de crecimiento.
    
    GRADIENTE
    ---------
    
    En ausencia de telemetría, se asigna penalización uniforme:
    
        ∇V_nodata = V_nodata(n) / N · 𝟙
    
    donde 𝟙 es el vector de unos.
    
    INTERPRETACIÓN
    --------------
    
    La penalización crece con fallos consecutivos (exponente de Lyapunov
    del proceso de adquisición), saturando en 1.0 tras 5 fallos.
    """
    
    name: str = "NoTelemetryEvaluator"
    priority: int = 20
    severity_growth_rate: float = 0.2
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """
        Calcula gradiente uniforme por ausencia de datos.
        
        Postcondición:
            Si telemetry is None:
                ∀j: (∇V)ⱼ = severity / num_nodes
            Si telemetry is not None:
                ∇V = 0
        """
        grad = np.zeros(num_nodes, dtype=np.float64)
        
        if telemetry is None:
            # Severidad crece con fallos consecutivos, saturando en 1.0
            severity = min(1.0, metrics.consecutive_failures * self.severity_growth_rate)
            
            # Distribuir uniformemente
            grad.fill(severity / num_nodes)
            
            logger.debug(
                f"[NODATA:GRAD] fallos={metrics.consecutive_failures}, "
                f"severidad={severity:.3f}, grad_per_node={severity/num_nodes:.4f}"
            )
        
        self._validate_gradient(grad, num_nodes)
        return grad
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """Evaluación legacy: sin telemetría → UNKNOWN."""
        if telemetry is None:
            if metrics.consecutive_failures > 0:
                return (
                    SystemStatus.UNKNOWN,
                    f"Sin telemetría ({metrics.consecutive_failures} fallos consecutivos)",
                )
            return (SystemStatus.UNKNOWN, "Esperando telemetría inicial")
        return None


@dataclass(frozen=True)
class CriticalVoltageEvaluator(BaseEvaluator):
    """
    Evaluador de voltaje crítico instantáneo (safety net).
    
    FUNCIÓN DE LYAPUNOV
    -------------------
    
    V_voltage(v) = max(0, v - v_critical)
    
    GRADIENTE
    ---------
    
    Penalización localizada en el nodo Core:
    
        (∇V_voltage)ⱼ = {
            v  si j = Core ∧ v > v_critical
            0  en otro caso
        }
    
    JUSTIFICACIÓN
    -------------
    
    El voltaje crítico es una alarma de hardware que requiere
    respuesta inmediata (fast-fail). La penalización se asigna
    completamente al Core ya que es la fuente del flujo.
    """
    
    name: str = "CriticalVoltageEvaluator"
    priority: int = 30
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """
        Calcula gradiente por voltaje crítico.
        
        Postcondición:
            ‖∇V‖₀ ∈ {0, 1}  (penalización sparse)
        """
        grad = np.zeros(num_nodes, dtype=np.float64)
        
        if telemetry and telemetry.flyback_voltage > config.flyback_voltage_critical:
            idx = self._get_node_idx(NetworkTopology.NODE_CORE)
            if idx is not None and idx < num_nodes:
                grad[idx] = telemetry.flyback_voltage
                
                logger.warning(
                    f"[VOLTAGE:CRITICAL] V={telemetry.flyback_voltage:.3f} > "
                    f"V_crit={config.flyback_voltage_critical:.3f}, "
                    f"penalización={grad[idx]:.3f}"
                )
        
        self._validate_gradient(grad, num_nodes)
        return grad
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """Evaluación legacy: voltaje crítico → CRITICO."""
        if telemetry and telemetry.flyback_voltage > config.flyback_voltage_critical:
            return (
                SystemStatus.CRITICO,
                f"Voltaje crítico: {telemetry.flyback_voltage:.3f} > "
                f"{config.flyback_voltage_critical:.3f}",
            )
        return None


@dataclass(frozen=True)
class CriticalSaturationEvaluator(BaseEvaluator):
    """
    Evaluador de saturación crítica instantánea (safety net).
    
    FUNCIÓN DE LYAPUNOV
    -------------------
    
    V_saturation(s) = max(0, s - s_critical)
    
    GRADIENTE
    ---------
    
    Similar a voltaje, localizado en Core:
    
        (∇V_sat)ⱼ = {
            s  si j = Core ∧ s > s_critical
            0  en otro caso
        }
    """
    
    name: str = "CriticalSaturationEvaluator"
    priority: int = 31
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """Calcula gradiente por saturación crítica."""
        grad = np.zeros(num_nodes, dtype=np.float64)
        
        if telemetry and telemetry.saturation > config.saturation_critical:
            idx = self._get_node_idx(NetworkTopology.NODE_CORE)
            if idx is not None and idx < num_nodes:
                grad[idx] = telemetry.saturation
                
                logger.warning(
                    f"[SAT:CRITICAL] S={telemetry.saturation:.3f} > "
                    f"S_crit={config.saturation_critical:.3f}"
                )
        
        self._validate_gradient(grad, num_nodes)
        return grad
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """Evaluación legacy: saturación crítica → CRITICO."""
        if telemetry and telemetry.saturation > config.saturation_critical:
            return (
                SystemStatus.CRITICO,
                f"Saturación crítica: {telemetry.saturation:.3f} > "
                f"{config.saturation_critical:.3f}",
            )
        return None


@dataclass(frozen=True)
class TopologyHealthCriticalEvaluator(BaseEvaluator):
    """
    Evaluador de salud topológica crítica global.
    
    FUNCIÓN DE LYAPUNOV
    -------------------
    
    V_topo(h) = {
        1 - h  si health_level = CRITICAL
        0      en otro caso
    }
    
    donde h = health_score ∈ [0, 1].
    
    GRADIENTE
    ---------
    
    Penalización uniforme sobre todos los nodos:
    
        ∇V_topo = (1 - h) / N · 𝟙
    
    JUSTIFICACIÓN
    -------------
    
    La salud topológica crítica indica degradación sistémica que
    no se localiza en un solo nodo, por lo que la penalización
    se distribuye uniformemente.
    """
    
    name: str = "TopologyHealthCriticalEvaluator"
    priority: int = 40
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """Calcula gradiente por salud topológica crítica."""
        grad = np.zeros(num_nodes, dtype=np.float64)
        
        if topo_health.level == HealthLevel.CRITICAL:
            # Severidad = degradación de salud
            severity = 1.0 - topo_health.health_score
            grad.fill(severity / num_nodes)
            
            logger.warning(
                f"[TOPO:CRITICAL] health_score={topo_health.health_score:.2f}, "
                f"severidad={severity:.3f}"
            )
        
        self._validate_gradient(grad, num_nodes)
        return grad
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """Evaluación legacy: salud crítica → CRITICO."""
        if topo_health.level == HealthLevel.CRITICAL:
            return (
                SystemStatus.CRITICO,
                f"Salud topológica crítica: score={topo_health.health_score:.2f}",
            )
        return None


@dataclass(frozen=True)
class PersistentSaturationCriticalEvaluator(BaseEvaluator):
    """
    Evaluador de saturación persistente (característica topológica temporal).
    
    FUNCIÓN DE LYAPUNOV
    -------------------
    
    V_persist_sat(τ) = min(1, τ / τ_max)
    
    donde:
    - τ = duración de la característica persistente
    - τ_max = 10 muestras (tiempo de referencia)
    
    GRADIENTE
    ---------
    
    Penalización localizada en Core, proporcional a la persistencia:
    
        (∇V)ⱼ = {
            min(1, τ/10)  si j = Core ∧ estado = CRITICAL
            0             en otro caso
        }
    
    JUSTIFICACIÓN HOMOLÓGICA
    ------------------------
    
    La persistencia captura características que sobreviven múltiples
    escalas de filtración temporal, distinguiendo señal genuina de ruido.
    """
    
    name: str = "PersistentSaturationCriticalEvaluator"
    priority: int = 50
    reference_duration: float = 10.0
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """Calcula gradiente por saturación persistente."""
        grad = np.zeros(num_nodes, dtype=np.float64)
        
        if saturation_analysis.state == MetricState.CRITICAL:
            # Extraer duración de persistencia
            duration = float(saturation_analysis.metadata.get("active_duration", 1.0))
            
            # Severidad normalizada por duración de referencia
            severity = min(1.0, duration / self.reference_duration)
            
            idx = self._get_node_idx(NetworkTopology.NODE_CORE)
            if idx is not None and idx < num_nodes:
                grad[idx] = severity
                
                logger.warning(
                    f"[SAT:PERSIST] duración={duration}, severidad={severity:.3f}"
                )
        
        self._validate_gradient(grad, num_nodes)
        return grad
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """Evaluación legacy: saturación persistente → SATURADO."""
        if saturation_analysis.state == MetricState.CRITICAL:
            duration = saturation_analysis.metadata.get("active_duration", "?")
            return (
                SystemStatus.SATURADO,
                f"Saturación persistente: duración={duration} muestras",
            )
        return None


@dataclass(frozen=True)
class SaturationFeatureEvaluator(BaseEvaluator):
    """
    Evaluador de características estructurales de saturación (β₁ temporal).
    
    FUNCIÓN DE LYAPUNOV
    -------------------
    
    V_feature_sat(π) = min(1, π / π_max)
    
    donde:
    - π = persistencia total de características
    - π_max = 50 (referencia)
    
    GRADIENTE
    ---------
    
    Penalización hacia Redis (destino del flujo saturado):
    
        (∇V)ⱼ = {
            min(1, π/50)  si j = Redis ∧ estado = FEATURE
            0             en otro caso
        }
    """
    
    name: str = "SaturationFeatureEvaluator"
    priority: int = 51
    reference_persistence: float = 50.0
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """Calcula gradiente por patrón estructural de saturación."""
        grad = np.zeros(num_nodes, dtype=np.float64)
        
        if saturation_analysis.state == MetricState.FEATURE:
            # Severidad basada en persistencia total
            persistence = saturation_analysis.total_persistence
            severity = min(1.0, persistence / self.reference_persistence)
            
            idx = self._get_node_idx(NetworkTopology.NODE_REDIS)
            if idx is not None and idx < num_nodes:
                grad[idx] = severity
                
                logger.info(
                    f"[SAT:FEATURE] características={saturation_analysis.feature_count}, "
                    f"π={persistence:.1f}, severidad={severity:.3f}"
                )
        
        self._validate_gradient(grad, num_nodes)
        return grad
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """Evaluación legacy: patrón estructural → SATURADO."""
        if saturation_analysis.state == MetricState.FEATURE:
            return (
                SystemStatus.SATURADO,
                f"Patrón estructural saturación: {saturation_analysis.feature_count} feature(s), "
                f"π={saturation_analysis.total_persistence:.1f}",
            )
        return None


@dataclass(frozen=True)
class PersistentVoltageCriticalEvaluator(BaseEvaluator):
    """Evaluador de voltaje persistente crítico (análogo a saturación)."""
    
    name: str = "PersistentVoltageCriticalEvaluator"
    priority: int = 60
    reference_duration: float = 10.0
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """Calcula gradiente por inestabilidad de voltaje persistente."""
        grad = np.zeros(num_nodes, dtype=np.float64)
        
        if voltage_analysis.state == MetricState.CRITICAL:
            duration = float(voltage_analysis.metadata.get("active_duration", 1.0))
            severity = min(1.0, duration / self.reference_duration)
            
            idx = self._get_node_idx(NetworkTopology.NODE_CORE)
            if idx is not None and idx < num_nodes:
                grad[idx] = severity
                
                logger.warning(
                    f"[VOLTAGE:PERSIST] duración={duration}, severidad={severity:.3f}"
                )
        
        self._validate_gradient(grad, num_nodes)
        return grad
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """Evaluación legacy: voltaje persistente → INESTABLE."""
        if voltage_analysis.state == MetricState.CRITICAL:
            duration = voltage_analysis.metadata.get("active_duration", "?")
            return (
                SystemStatus.INESTABLE,
                f"Inestabilidad voltaje: duración={duration} muestras",
            )
        return None


@dataclass(frozen=True)
class VoltageFeatureEvaluator(BaseEvaluator):
    """Evaluador de características estructurales de voltaje."""
    
    name: str = "VoltageFeatureEvaluator"
    priority: int = 61
    reference_lifespan: float = 20.0
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """Calcula gradiente basado en vida máxima de características de voltaje."""
        grad = np.zeros(num_nodes, dtype=np.float64)
        
        if voltage_analysis.state == MetricState.FEATURE:
            lifespan = voltage_analysis.max_lifespan
            severity = min(1.0, lifespan / self.reference_lifespan)
            
            idx = self._get_node_idx(NetworkTopology.NODE_CORE)
            if idx is not None and idx < num_nodes:
                grad[idx] = severity
                
                logger.info(
                    f"[VOLTAGE:FEATURE] λ_max={lifespan:.1f}, severidad={severity:.3f}"
                )
        
        self._validate_gradient(grad, num_nodes)
        return grad
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """Evaluación legacy: patrón de voltaje → INESTABLE."""
        if voltage_analysis.state == MetricState.FEATURE:
            return (
                SystemStatus.INESTABLE,
                f"Patrón estructural voltaje: λ_max={voltage_analysis.max_lifespan:.1f}",
            )
        return None


@dataclass(frozen=True)
class RetryLoopEvaluator(BaseEvaluator):
    """
    Evaluador de patrones de reintentos excesivos (β₁ topológico).
    
    FUNCIÓN DE LYAPUNOV
    -------------------
    
    V_loop(n_loops) = min(1, n_loops / n_max)
    
    donde n_loops = número de reintentos detectados.
    
    GRADIENTE
    ---------
    
    Penalización distribuida entre Agent y Core (extremos del loop):
    
        (∇V)ⱼ = {
            severity * 0.5  si j ∈ {Agent, Core}
            0               en otro caso
        }
    
    JUSTIFICACIÓN TOPOLÓGICA
    ------------------------
    
    Los loops en el grafo de requests indican ciclos espurios (β₁ > 0 temporal),
    señalando deadlocks o condiciones de carrera.
    """
    
    name: str = "RetryLoopEvaluator"
    priority: int = 70
    min_loop_count: int = 5
    reference_loop_count: float = 20.0
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """Calcula gradiente por bucles de reintentos."""
        grad = np.zeros(num_nodes, dtype=np.float64)
        
        if topo_health.request_loops:
            first_loop = topo_health.request_loops[0]
            
            # Solo penalizar loops significativos de error
            if (
                first_loop.count >= self.min_loop_count
                and first_loop.request_id.startswith("FAIL_")
            ):
                severity = min(1.0, first_loop.count / self.reference_loop_count)
                
                # Distribuir entre Agent y Core
                idx_agent = self._get_node_idx(NetworkTopology.NODE_AGENT)
                idx_core = self._get_node_idx(NetworkTopology.NODE_CORE)
                
                if idx_agent is not None and idx_agent < num_nodes:
                    grad[idx_agent] = severity * 0.5
                if idx_core is not None and idx_core < num_nodes:
                    grad[idx_core] = severity * 0.5
                
                logger.warning(
                    f"[LOOP:DETECT] request_id='{first_loop.request_id}', "
                    f"count={first_loop.count}, severidad={severity:.3f}"
                )
        
        self._validate_gradient(grad, num_nodes)
        return grad
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """Evaluación legacy: bucles de reintentos → INESTABLE."""
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
class UnhealthyTopologyEvaluator(BaseEvaluator):
    """
    Evaluador de salud topológica degradada (no crítica).
    
    Análogo a TopologyHealthCriticalEvaluator pero para nivel UNHEALTHY.
    """
    
    name: str = "UnhealthyTopologyEvaluator"
    priority: int = 80
    
    def compute_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int,
    ) -> np.ndarray:
        """Calcula gradiente por salud topológica degradada."""
        grad = np.zeros(num_nodes, dtype=np.float64)
        
        if topo_health.level == HealthLevel.UNHEALTHY:
            severity = 1.0 - topo_health.health_score
            grad.fill(severity / num_nodes)
            
            logger.info(
                f"[TOPO:UNHEALTHY] health_score={topo_health.health_score:.2f}, "
                f"severidad={severity:.3f}"
            )
        
        self._validate_gradient(grad, num_nodes)
        return grad
    
    def evaluate(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
    ) -> Optional[Tuple[SystemStatus, str]]:
        """Evaluación legacy: salud degradada → INESTABLE."""
        if topo_health.level == HealthLevel.UNHEALTHY:
            return (
                SystemStatus.INESTABLE,
                f"Salud degradada: score={topo_health.health_score:.2f}",
            )
        return None


# ============================================================================
# SINTETIZADOR HAMILTONIANO
# ============================================================================

class HamiltonianSynthesizer:
    """
    Sintetizador Hamiltoniano: Integración de Campos Vectoriales de Penalización.
    
    FUNDAMENTACIÓN TEÓRICA
    ----------------------
    
    Este componente implementa la construcción de la Función de Lyapunov Global
    mediante superposición lineal de Lyapunovs locales:
    
        V_total(φ) = Σᵢ wᵢ · Vᵢ(φ)
    
    El gradiente total se obtiene por linealidad:
    
        ∇V_total = Σᵢ wᵢ · ∇Vᵢ
    
    RENORMALIZACIÓN DE MASA TÉRMICA (Proyección Gauge)
    --------------------------------------------------
    
    Para garantizar que ∇V_total ∈ im(B₁ᵀ) (condición de neutralidad de carga),
    aplicamos la proyección ortogonal sobre ker(L)^⊥:
    
        ∇V_renorm = ∇V_total - ⟨∇V_total, 𝟙⟩ · 𝟙
    
    donde:
    - L = B₁ᵀB₁ es el Laplaciano combinatorio
    - ker(L) = span{𝟙} (vectores constantes)
    - 𝟙 = (1/√N, ..., 1/√N) es la base ortonormal de ker(L)
    
    En términos discretos:
    
        ∇V_renorm = ∇V_total - mean(∇V_total) · 𝟙
    
    TEOREMA (Existencia de Solución a Poisson)
    ------------------------------------------
    
    Sea Δ el Laplaciano combinatorio y ρ una densidad de carga.
    Entonces ∃φ tal que Δφ = ρ ⟺ ρ ⊥ ker(Δ).
    
    Demostración: Teorema de Fredholm para operadores autoadjuntos.
    
    La renormalización garantiza esta condición de compatibilidad.
    
    INVARIANTES
    -----------
    
    Post-síntesis:
    1. ‖∇V_renorm‖₂ < ∞
    2. Σᵢ (∇V_renorm)ᵢ = 0 (neutralidad de carga)
    3. ∇V_renorm ∈ im(B₁ᵀ)
    """
    
    def __init__(self, evaluators: Optional[List[StateEvaluator]] = None):
        """
        Inicializa el sintetizador con una lista de evaluadores.
        
        Args:
            evaluators: Lista de evaluadores (default: evaluadores estándar)
            
        Postcondición:
            len(self._evaluators) > 0
        """
        if evaluators is None:
            evaluators = self._create_default_evaluators()
        
        if not evaluators:
            raise ValueError("Se requiere al menos un evaluador")
        
        self._evaluators = evaluators
        logger.info(
            f"[HAMILTONIAN:INIT] {len(self._evaluators)} evaluadores registrados: "
            f"{[e.name for e in self._evaluators]}"
        )
    
    @staticmethod
    def _create_default_evaluators() -> List[StateEvaluator]:
        """
        Factory: crea la lista estándar de evaluadores.
        
        Returns:
            Lista ordenada por prioridad (legacy, no usado en síntesis)
        """
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
    
    def synthesize_gradient(
        self,
        telemetry: Optional[TelemetryData],
        topo_health: TopologicalHealth,
        voltage_analysis: PersistenceAnalysisResult,
        saturation_analysis: PersistenceAnalysisResult,
        config: ThresholdConfig,
        metrics: AgentMetrics,
        num_nodes: int = PhysicalConstants.PHASE_SPACE_DIM,
    ) -> np.ndarray:
        """
        Computa el gradiente total renormalizado ∇V_total ∈ ℝᴺ.
        
        ALGORITMO
        ---------
        
        1. Inicializar ∇V_total = 0
        2. Para cada evaluador i:
               ∇V_total += compute_gradient(...)
        3. Renormalizar: ∇V_total -= mean(∇V_total)
        4. Retornar ∇V_total
        
        Args:
            telemetry: Estado de telemetría actual
            topo_health: Salud topológica
            voltage_analysis: Análisis de persistencia de voltaje
            saturation_analysis: Análisis de persistencia de saturación
            config: Configuración de umbrales
            metrics: Métricas del agente
            num_nodes: Dimensión del espacio de fase (default: 4)
            
        Returns:
            Vector gradiente renormalizado ∇V_total ∈ ℝᴺ
            
        Postcondiciones:
            - len(resultado) == num_nodes
            - np.abs(np.sum(resultado)) < ε (neutralidad de carga)
            - np.all(np.isfinite(resultado))
            
        Raises:
            ValueError: Si algún evaluador produce gradiente inválido
        """
        grad_total = np.zeros(num_nodes, dtype=np.float64)
        
        # FASE 1: Suma de gradientes locales
        for evaluator in self._evaluators:
            try:
                grad_i = evaluator.compute_gradient(
                    telemetry,
                    topo_health,
                    voltage_analysis,
                    saturation_analysis,
                    config,
                    metrics,
                    num_nodes,
                )
                
                # Validar dimensión
                if len(grad_i) != num_nodes:
                    logger.error(
                        f"[HAMILTONIAN:ERROR] {evaluator.name} retornó gradiente "
                        f"de dimensión {len(grad_i)}, esperado {num_nodes}. Ignorando."
                    )
                    continue
                
                # Log contribuciones no nulas
                if np.any(grad_i > PhysicalConstants.EPSILON):
                    logger.debug(
                        f"[HAMILTONIAN:CONTRIB] {evaluator.name}: ‖∇V‖₂={np.linalg.norm(grad_i):.4e}, "
                        f"‖∇V‖∞={np.max(np.abs(grad_i)):.4e}"
                    )
                
                grad_total += grad_i
                
            except Exception as e:
                logger.warning(
                    f"[HAMILTONIAN:ERROR] Error en {evaluator.name}: {e}",
                    exc_info=True,
                )
                continue
        
        # FASE 2: Renormalización de Masa Térmica (Proyección Gauge)
        if num_nodes > 0:
            mean_force = np.mean(grad_total)
            grad_renormalized = grad_total - mean_force
            
            # Verificar neutralidad de carga
            residual_charge = np.sum(grad_renormalized)
            if abs(residual_charge) > PhysicalConstants.EPSILON * num_nodes:
                logger.warning(
                    f"[HAMILTONIAN:RENORM] Carga residual alta: {residual_charge:.4e} "
                    f"(esperado < {PhysicalConstants.EPSILON * num_nodes:.4e})"
                )
            
            logger.debug(
                f"[HAMILTONIAN:RENORM] Media removida: {mean_force:.4e}, "
                f"‖∇V_renorm‖₂={np.linalg.norm(grad_renormalized):.4e}, "
                f"carga_residual={residual_charge:.4e}"
            )
            
            return grad_renormalized
        
        return grad_total
    
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
        Evaluación legacy: mapeo gradiente → decisión (retrocompatibilidad).
        
        DEPRECACIÓN
        -----------
        
        Este método será removido en versiones futuras. La decisión debe
        derivarse del potencial gauge φ resuelto por GaugeFieldRouter.
        
        Args:
            (mismos que synthesize_gradient, excepto num_nodes)
            
        Returns:
            Tupla (peor_status, mensaje_agregado)
            
        Algoritmo:
            1. Evaluar todos los evaluadores (método evaluate)
            2. Retornar el peor status (max por severidad)
        """
        worst_status = SystemStatus.NOMINAL
        worst_summary = (
            f"Sistema nominal: β₀={topo_health.betti.b0}, "
            f"h={topo_health.health_score:.2f}"
        )
        
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
                    if status > worst_status:
                        worst_status = status
                        worst_summary = summary
                        
            except Exception as e:
                logger.warning(
                    f"[HAMILTONIAN:EVAL] Error en {evaluator.name}: {e}",
                    exc_info=True,
                )
                continue
        
        return worst_status, worst_summary
    
    @property
    def evaluators(self) -> List[StateEvaluator]:
        """Retorna lista de evaluadores registrados (read-only)."""
        return list(self._evaluators)
    
    def add_evaluator(self, evaluator: StateEvaluator) -> None:
        """
        Agrega un evaluador dinámicamente.
        
        Args:
            evaluator: Evaluador a agregar
            
        Raises:
            TypeError: Si evaluator no cumple el protocolo StateEvaluator
        """
        if not isinstance(evaluator, StateEvaluator):
            raise TypeError(
                f"El evaluador debe cumplir el protocolo StateEvaluator, "
                f"recibido: {type(evaluator).__name__}"
            )
        
        self._evaluators.append(evaluator)
        logger.info(f"[HAMILTONIAN:ADD] Evaluador '{evaluator.name}' agregado")
    
    def remove_evaluator(self, name: str) -> bool:
        """
        Remueve un evaluador por nombre.
        
        Args:
            name: Nombre del evaluador a remover
            
        Returns:
            True si se removió, False si no se encontró
        """
        for i, evaluator in enumerate(self._evaluators):
            if evaluator.name == name:
                del self._evaluators[i]
                logger.info(f"[HAMILTONIAN:REMOVE] Evaluador '{name}' removido")
                return True
        
        logger.warning(f"[HAMILTONIAN:REMOVE] Evaluador '{name}' no encontrado")
        return False


# ============================================================================
# ESTRUCTURAS DE OBSERVABILIDAD Y MÉTRICAS INTERNAS
# ============================================================================

@dataclass
class AgentMetrics:
    """
    Métricas internas del agente para observabilidad y análisis de desempeño.
    
    FUNDAMENTACIÓN TEÓRICA
    ----------------------
    
    Esta estructura implementa un Contador de Eventos Monótonico que
    captura la dinámica temporal del ciclo OODA:
    
        M: ℕ → MetricSpace
    
    donde cada ciclo k mapea a un punto en el espacio de métricas.
    
    INVARIANTES
    -----------
    
    1. **Monotonicidad**: Los contadores solo incrementan
       - cycles_executed_{k+1} ≥ cycles_executed_k
       - successful_observations_{k+1} ≥ successful_observations_k
    
    2. **Conservación**: Total de observaciones
       - successful_observations + failed_observations = observaciones_totales
    
    3. **Causalidad temporal**:
       - last_successful_observation ≤ now()
       - start_time ≤ last_successful_observation
    
    MÉTRICAS DERIVADAS
    ------------------
    
    - **Success Rate**: σ = successful / total ∈ [0, 1]
    - **Observation Rate**: λ = successful / uptime_minutes (eventos/min)
    - **Failure Streak**: consecutive_failures (ventana deslizante)
    
    APLICACIONES
    ------------
    
    1. **Monitoreo de salud**: Detección de degradación (σ < σ_threshold)
    2. **SLOs**: Verificación de acuerdos de nivel de servicio
    3. **Debugging**: Trazabilidad de eventos anómalos
    4. **Capacity planning**: Estimación de carga (λ)
    """
    
    # Contadores monotónicos
    cycles_executed: int = 0
    successful_observations: int = 0
    failed_observations: int = 0
    
    # Estado temporal
    last_successful_observation: Optional[datetime] = None
    consecutive_failures: int = 0
    
    # Histograma de decisiones (key: decision.name, value: count)
    decisions_count: Dict[str, int] = field(default_factory=dict)
    
    # Marcas temporales
    start_time: datetime = field(default_factory=datetime.now)
    last_cycle_duration_ms: float = 0.0
    
    def record_success(self) -> None:
        """
        Registra una observación exitosa.
        
        Efectos:
            - Incrementa successful_observations
            - Actualiza last_successful_observation a now()
            - Resetea consecutive_failures a 0
            
        Postcondiciones:
            - consecutive_failures == 0
            - last_successful_observation == now()
        """
        self.successful_observations += 1
        self.last_successful_observation = datetime.now()
        self.consecutive_failures = 0
        
        logger.debug(
            f"[METRICS:SUCCESS] Total exitosas: {self.successful_observations}, "
            f"tasa: {self.success_rate:.2%}"
        )
    
    def record_failure(self) -> None:
        """
        Registra una observación fallida.
        
        Efectos:
            - Incrementa failed_observations
            - Incrementa consecutive_failures
            
        Postcondiciones:
            - consecutive_failures > 0
        """
        self.failed_observations += 1
        self.consecutive_failures += 1
        
        logger.debug(
            f"[METRICS:FAILURE] Total fallidas: {self.failed_observations}, "
            f"consecutivas: {self.consecutive_failures}, "
            f"tasa: {self.success_rate:.2%}"
        )
    
    def record_decision(self, decision: AgentDecision) -> None:
        """
        Registra una decisión tomada en el histograma.
        
        Args:
            decision: Decisión ejecutada
            
        Efectos:
            - Incrementa decisions_count[decision.name]
        """
        key = decision.name
        self.decisions_count[key] = self.decisions_count.get(key, 0) + 1
        
        logger.debug(
            f"[METRICS:DECISION] {decision.name} registrada "
            f"(total: {self.decisions_count[key]})"
        )
    
    def increment_cycle(self) -> None:
        """
        Incrementa el contador de ciclos OODA ejecutados.
        
        Efectos:
            - Incrementa cycles_executed
        """
        self.cycles_executed += 1
    
    def record_cycle_duration(self, duration_seconds: float) -> None:
        """
        Registra la duración del último ciclo OODA.
        
        Args:
            duration_seconds: Duración en segundos
            
        Efectos:
            - Actualiza last_cycle_duration_ms
        """
        self.last_cycle_duration_ms = duration_seconds * 1000.0
    
    @property
    def total_observations(self) -> int:
        """
        Calcula el total de observaciones (exitosas + fallidas).
        
        Returns:
            Suma de successful_observations y failed_observations
            
        Invariante:
            total_observations == successful_observations + failed_observations
        """
        return self.successful_observations + self.failed_observations
    
    @property
    def success_rate(self) -> float:
        """
        Calcula la tasa de éxito de observaciones.
        
        Returns:
            σ = successful / total ∈ [0, 1]
            
        Casos especiales:
            - Si total = 0 → retorna 0.0 (no hay datos)
        """
        total = self.total_observations
        if total == 0:
            return 0.0
        return self.successful_observations / total
    
    @property
    def failure_rate(self) -> float:
        """
        Calcula la tasa de fallo de observaciones.
        
        Returns:
            1 - σ ∈ [0, 1]
        """
        return 1.0 - self.success_rate
    
    @property
    def uptime_seconds(self) -> float:
        """
        Calcula el tiempo de ejecución del agente.
        
        Returns:
            Δt = now() - start_time (en segundos)
        """
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def uptime_minutes(self) -> float:
        """Retorna uptime en minutos."""
        return self.uptime_seconds / 60.0
    
    @property
    def uptime_hours(self) -> float:
        """Retorna uptime en horas."""
        return self.uptime_seconds / 3600.0
    
    @property
    def observation_rate(self) -> float:
        """
        Calcula la tasa de observaciones exitosas por minuto.
        
        Returns:
            λ = successful_observations / uptime_minutes (eventos/min)
            
        Casos especiales:
            - Si uptime < 1s → retorna 0.0 (evitar división inestable)
        """
        uptime_min = self.uptime_minutes
        if uptime_min < (1.0 / 60.0):  # Menos de 1 segundo
            return 0.0
        return self.successful_observations / uptime_min
    
    @property
    def cycle_rate(self) -> float:
        """
        Calcula la tasa de ciclos OODA por minuto.
        
        Returns:
            ω = cycles_executed / uptime_minutes (ciclos/min)
        """
        uptime_min = self.uptime_minutes
        if uptime_min < (1.0 / 60.0):
            return 0.0
        return self.cycles_executed / uptime_min
    
    @property
    def mean_cycle_duration_ms(self) -> float:
        """
        Estima la duración media de ciclo basada en uptime.
        
        Returns:
            Duración media en milisegundos
            
        Nota:
            Usa uptime / cycles_executed, no last_cycle_duration_ms
            (que solo refleja el último ciclo)
        """
        if self.cycles_executed == 0:
            return 0.0
        return (self.uptime_seconds / self.cycles_executed) * 1000.0
    
    @property
    def is_healthy(self) -> bool:
        """
        Predicado de salud básico basado en tasa de éxito.
        
        Returns:
            True si success_rate ≥ 0.9 (90%)
            
        Umbrales:
            - Healthy: σ ≥ 0.9
            - Degraded: 0.5 ≤ σ < 0.9
            - Unhealthy: σ < 0.5
        """
        return self.success_rate >= 0.9
    
    @property
    def health_status(self) -> str:
        """
        Retorna estado de salud categórico.
        
        Returns:
            "healthy" | "degraded" | "unhealthy"
        """
        rate = self.success_rate
        if rate >= 0.9:
            return "healthy"
        if rate >= 0.5:
            return "degraded"
        return "unhealthy"
    
    def get_decision_distribution(self) -> Dict[str, float]:
        """
        Calcula la distribución de probabilidad de decisiones.
        
        Returns:
            Dict {decision_name: probability} donde Σ prob = 1.0
            
        Ejemplo:
            {"HEARTBEAT": 0.7, "EJECUTAR_LIMPIEZA": 0.2, "WAIT": 0.1}
        """
        total_decisions = sum(self.decisions_count.values())
        if total_decisions == 0:
            return {}
        
        return {
            decision: count / total_decisions
            for decision, count in self.decisions_count.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa las métricas a diccionario para JSON/logging.
        
        Returns:
            Dict con estructura jerárquica de métricas
        """
        return {
            # Contadores brutos
            "counters": {
                "cycles_executed": self.cycles_executed,
                "successful_observations": self.successful_observations,
                "failed_observations": self.failed_observations,
                "total_observations": self.total_observations,
                "consecutive_failures": self.consecutive_failures,
            },
            # Tasas y proporciones
            "rates": {
                "success_rate": round(self.success_rate, 4),
                "failure_rate": round(self.failure_rate, 4),
                "observation_rate_per_min": round(self.observation_rate, 2),
                "cycle_rate_per_min": round(self.cycle_rate, 2),
            },
            # Tiempos
            "timing": {
                "uptime_seconds": round(self.uptime_seconds, 2),
                "uptime_hours": round(self.uptime_hours, 4),
                "last_cycle_duration_ms": round(self.last_cycle_duration_ms, 2),
                "mean_cycle_duration_ms": round(self.mean_cycle_duration_ms, 2),
                "last_successful_observation": (
                    self.last_successful_observation.isoformat()
                    if self.last_successful_observation
                    else None
                ),
            },
            # Decisiones
            "decisions": {
                "histogram": dict(self.decisions_count),
                "distribution": {
                    k: round(v, 4)
                    for k, v in self.get_decision_distribution().items()
                },
            },
            # Salud
            "health": {
                "status": self.health_status,
                "is_healthy": self.is_healthy,
            },
        }
    
    def __repr__(self) -> str:
        """Representación compacta para debugging."""
        return (
            f"AgentMetrics(cycles={self.cycles_executed}, "
            f"success_rate={self.success_rate:.2%}, "
            f"consecutive_failures={self.consecutive_failures})"
        )


# ============================================================================
# RESULTADO DE OBSERVACIÓN
# ============================================================================

@dataclass
class ObservationResult:
    """
    Resultado de una operación de observación (éxito o fallo).
    
    TIPO ALGEBRAICO (SUM TYPE)
    --------------------------
    
    Esta estructura implementa un tipo suma (Either) que encapsula:
    
        ObservationResult = Success(TelemetryData) | Failure(ErrorType)
    
    INVARIANTES
    -----------
    
    1. **Exclusividad mutua**:
       - success = True  ⟹ telemetry ≠ None ∧ error_type = None
       - success = False ⟹ telemetry = None ∧ error_type ≠ None
    
    2. **Trazabilidad**: Siempre existe request_id único
    
    PATTERN MATCHING
    ----------------
    
    Uso recomendado:
    
        result = observe()
        if result.success:
            process_telemetry(result.telemetry)
        else:
            handle_error(result.error_type)
    """
    
    success: bool
    telemetry: Optional[TelemetryData]
    error_type: Optional[ObservationErrorType]
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """
        Valida invariantes de exclusividad mutua.
        
        Raises:
            ValueError: Si se viola la invariante success/failure
        """
        if self.success:
            if self.telemetry is None:
                raise ValueError(
                    "Success=True requiere telemetry no nula"
                )
            if self.error_type is not None:
                raise ValueError(
                    "Success=True no admite error_type"
                )
        else:
            if self.telemetry is not None:
                raise ValueError(
                    "Success=False requiere telemetry nula"
                )
            if self.error_type is None:
                raise ValueError(
                    "Success=False requiere error_type"
                )
    
    @classmethod
    def success_result(
        cls,
        telemetry: TelemetryData,
        request_id: str,
    ) -> ObservationResult:
        """
        Factory: crea resultado de éxito.
        
        Args:
            telemetry: Datos de telemetría obtenidos
            request_id: Identificador de la petición
            
        Returns:
            ObservationResult con success=True
            
        Postcondición:
            resultado.success == True
        """
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
        """
        Factory: crea resultado de fallo.
        
        Args:
            error_type: Tipo de error ocurrido
            request_id: Identificador de la petición
            
        Returns:
            ObservationResult con success=False
            
        Postcondición:
            resultado.success == False
        """
        return cls(
            success=False,
            telemetry=None,
            error_type=error_type,
            request_id=request_id,
        )
    
    @property
    def is_transient_error(self) -> bool:
        """
        Indica si el error es potencialmente transitorio.
        
        Returns:
            True si un retry podría tener éxito
        """
        if self.success or self.error_type is None:
            return False
        return self.error_type.is_transient
    
    @property
    def requires_reconnection(self) -> bool:
        """Indica si se requiere reconexión completa."""
        if self.success or self.error_type is None:
            return False
        return self.error_type.requires_reconnection
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        result: Dict[str, Any] = {
            "success": self.success,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.success and self.telemetry:
            result["telemetry"] = self.telemetry.to_dict()
        else:
            result["error"] = {
                "type": self.error_type.value if self.error_type else "UNKNOWN",
                "is_transient": self.is_transient_error,
                "requires_reconnection": self.requires_reconnection,
            }
        
        return result
    
    def __repr__(self) -> str:
        """Representación compacta."""
        if self.success:
            return f"ObservationResult(✓, id={self.request_id[:8]})"
        return f"ObservationResult(✗ {self.error_type.value}, id={self.request_id[:8]})"


# ============================================================================
# DIAGNÓSTICO TOPOLÓGICO
# ============================================================================

@dataclass
class TopologicalDiagnosis:
    """
    Diagnóstico topológico consolidado para el ciclo OODA.
    
    ESTRUCTURA MATEMÁTICA
    ---------------------
    
    Este objeto encapsula el estado completo en el espacio de observables:
    
        Ω: M → DiagnosisSpace
    
    donde:
    - M: Variedad de estados del sistema
    - DiagnosisSpace: Espacio producto de invariantes topológicos
    
    COMPONENTES
    -----------
    
    1. **Salud topológica**: TopologicalHealth (β₀, β₁, χ, score)
    2. **Persistencia de métricas**: PersistenceAnalysisResult × 2
    3. **Recomendación**: SystemStatus
    4. **Resumen narrativo**: str (para humanos)
    
    INVARIANTES
    -----------
    
    - recommended_status es coherente con health.level:
        health.level = CRITICAL ⟹ recommended_status ∈ {CRITICO, DISCONNECTED}
    
    APLICACIONES
    ------------
    
    1. **Fase ORIENT**: Construcción del diagnóstico
    2. **Fase DECIDE**: Mapeo diagnóstico → decisión
    3. **Fase ACT**: Contextualización de la acción
    4. **Logging**: Trazabilidad de eventos
    """
    
    health: TopologicalHealth
    voltage_persistence: PersistenceAnalysisResult
    saturation_persistence: PersistenceAnalysisResult
    summary: str
    recommended_status: SystemStatus
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_structurally_healthy(self) -> bool:
        """
        Predicado: sistema estructuralmente sano (conectado, sin fragmentación).
        
        Returns:
            True ⟺ β₀ = 1 (grafo conexo)
        """
        return self.health.betti.is_connected
    
    @property
    def has_retry_loops(self) -> bool:
        """
        Predicado: patrones de reintentos detectados.
        
        Returns:
            True ⟺ |request_loops| > 0
        """
        return len(self.health.request_loops) > 0
    
    @property
    def health_score(self) -> float:
        """
        Score de salud topológica normalizado.
        
        Returns:
            h ∈ [0, 1] donde 1 = salud perfecta
        """
        return self.health.health_score
    
    @property
    def has_persistent_issues(self) -> bool:
        """
        Predicado: existen características persistentes críticas.
        
        Returns:
            True si voltage o saturation tienen estado CRITICAL o FEATURE
        """
        return (
            self.voltage_persistence.state in (MetricState.CRITICAL, MetricState.FEATURE)
            or self.saturation_persistence.state in (MetricState.CRITICAL, MetricState.FEATURE)
        )
    
    @property
    def disconnected_node_count(self) -> int:
        """Número de nodos desconectados."""
        return len(self.health.disconnected_nodes)
    
    @property
    def missing_edge_count(self) -> int:
        """Número de aristas faltantes respecto a topología esperada."""
        return len(self.health.missing_edges)
    
    @property
    def severity_level(self) -> int:
        """Nivel de severidad numérico del status recomendado."""
        return self.recommended_status.severity
    
    def get_critical_issues(self) -> List[str]:
        """
        Extrae lista de issues críticos detectados.
        
        Returns:
            Lista de cadenas describiendo cada issue
        """
        issues = []
        
        # Fragmentación
        if not self.is_structurally_healthy:
            issues.append(
                f"Fragmentación: β₀={self.health.betti.b0}, "
                f"nodos_desconectados={self.disconnected_node_count}"
            )
        
        # Salud crítica
        if self.health.level == HealthLevel.CRITICAL:
            issues.append(f"Salud crítica: score={self.health_score:.2f}")
        
        # Persistencia de voltaje
        if self.voltage_persistence.state == MetricState.CRITICAL:
            duration = self.voltage_persistence.metadata.get("active_duration", "?")
            issues.append(f"Voltaje persistente: duración={duration}")
        
        # Persistencia de saturación
        if self.saturation_persistence.state == MetricState.CRITICAL:
            duration = self.saturation_persistence.metadata.get("active_duration", "?")
            issues.append(f"Saturación persistente: duración={duration}")
        
        # Bucles de reintentos
        if self.has_retry_loops:
            loop_count = len(self.health.request_loops)
            issues.append(f"Bucles de reintentos: {loop_count}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa a diccionario completo (para APIs/dashboards).
        
        Returns:
            Dict con estructura jerárquica
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "recommended_status": self.recommended_status.name,
            "severity_level": self.severity_level,
            "predicates": {
                "is_structurally_healthy": self.is_structurally_healthy,
                "has_retry_loops": self.has_retry_loops,
                "has_persistent_issues": self.has_persistent_issues,
            },
            "health": {
                "score": round(self.health.health_score, 4),
                "level": self.health.level.name,
                "betti_numbers": {
                    "b0": self.health.betti.b0,
                    "b1": self.health.betti.b1,
                    "is_connected": self.health.betti.is_connected,
                    "is_ideal": self.health.betti.is_ideal,
                    "euler_characteristic": self.health.betti.euler_characteristic,
                },
                "issues": {
                    "disconnected_nodes": list(self.health.disconnected_nodes),
                    "disconnected_node_count": self.disconnected_node_count,
                    "missing_edges": [list(e) for e in self.health.missing_edges],
                    "missing_edge_count": self.missing_edge_count,
                    "retry_loops": len(self.health.request_loops),
                },
                "diagnostics": self.health.diagnostics,
            },
            "persistence": {
                "voltage": {
                    "state": self.voltage_persistence.state.name,
                    "feature_count": self.voltage_persistence.feature_count,
                    "total_persistence": round(self.voltage_persistence.total_persistence, 2),
                    "max_lifespan": round(self.voltage_persistence.max_lifespan, 2),
                },
                "saturation": {
                    "state": self.saturation_persistence.state.name,
                    "feature_count": self.saturation_persistence.feature_count,
                    "total_persistence": round(self.saturation_persistence.total_persistence, 2),
                    "max_lifespan": round(self.saturation_persistence.max_lifespan, 2),
                },
            },
            "critical_issues": self.get_critical_issues(),
        }
    
    def to_log_dict(self) -> Dict[str, Any]:
        """
        Serializa para logging estructurado (versión compacta).
        
        Returns:
            Dict con solo información esencial para logs
        """
        return {
            "betti_b0": self.health.betti.b0,
            "betti_b1": self.health.betti.b1,
            "health_score": round(self.health.health_score, 3),
            "health_level": self.health.level.name,
            "voltage_state": self.voltage_persistence.state.name,
            "saturation_state": self.saturation_persistence.state.name,
            "disconnected_nodes": list(self.health.disconnected_nodes),
            "retry_loops": len(self.health.request_loops),
            "recommended_status": self.recommended_status.name,
            "critical_issues_count": len(self.get_critical_issues()),
        }
    
    def to_summary_string(self) -> str:
        """
        Genera resumen legible en una línea.
        
        Returns:
            Cadena compacta con información clave
            
        Ejemplo:
            "β₀=1, h=0.95, V=NOISE, S=FEATURE, status=NOMINAL"
        """
        return (
            f"β₀={self.health.betti.b0}, h={self.health_score:.2f}, "
            f"V={self.voltage_persistence.state.name}, "
            f"S={self.saturation_persistence.state.name}, "
            f"status={self.recommended_status.name}"
        )
    
    def __repr__(self) -> str:
        """Representación compacta para debugging."""
        return f"TopologicalDiagnosis({self.to_summary_string()})"


# ============================================================================
# SNAPSHOT DE ESTADO DEL AGENTE
# ============================================================================

@dataclass
class AgentSnapshot:
    """
    Snapshot inmutable del estado completo del agente en un instante.
    
    FUNDAMENTACIÓN
    --------------
    
    Implementa un Funtor de Estado que captura el punto actual
    en la trayectoria del sistema:
    
        S: Time → StateSpace
    
    donde StateSpace es el producto cartesiano:
    
        StateSpace = TelemetryData × TopologicalDiagnosis × AgentMetrics × SystemStatus
    
    APLICACIONES
    ------------
    
    1. **Debugging**: Reproducción de estados pasados
    2. **Time-series analysis**: Detección de tendencias
    3. **Post-mortem**: Análisis de incidentes
    4. **Testing**: Snapshots sintéticos para unit tests
    """
    
    timestamp: datetime
    cycle_number: int
    telemetry: Optional[TelemetryData]
    diagnosis: Optional[TopologicalDiagnosis]
    metrics: AgentMetrics
    last_decision: Optional[AgentDecision]
    current_status: SystemStatus
    
    @classmethod
    def capture(
        cls,
        cycle_number: int,
        telemetry: Optional[TelemetryData],
        diagnosis: Optional[TopologicalDiagnosis],
        metrics: AgentMetrics,
        last_decision: Optional[AgentDecision],
        current_status: SystemStatus,
    ) -> AgentSnapshot:
        """
        Factory: captura snapshot del estado actual.
        
        Args:
            cycle_number: Número de ciclo OODA
            telemetry: Telemetría actual (puede ser None)
            diagnosis: Diagnóstico actual (puede ser None)
            metrics: Métricas actuales (copia profunda)
            last_decision: Última decisión tomada
            current_status: Status actual del sistema
            
        Returns:
            Snapshot inmutable
        """
        # Hacer copia profunda de métricas para inmutabilidad
        import copy
        metrics_copy = copy.deepcopy(metrics)
        
        return cls(
            timestamp=datetime.now(),
            cycle_number=cycle_number,
            telemetry=telemetry,
            diagnosis=diagnosis,
            metrics=metrics_copy,
            last_decision=last_decision,
            current_status=current_status,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario completo."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cycle_number": self.cycle_number,
            "current_status": self.current_status.name,
            "last_decision": self.last_decision.name if self.last_decision else None,
            "telemetry": self.telemetry.to_dict() if self.telemetry else None,
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "metrics": self.metrics.to_dict(),
        }
    
    def to_compact_dict(self) -> Dict[str, Any]:
        """Serializa versión compacta (solo esenciales)."""
        compact: Dict[str, Any] = {
            "timestamp": self.timestamp.isoformat(),
            "cycle": self.cycle_number,
            "status": self.current_status.name,
        }
        
        if self.telemetry:
            compact["telemetry"] = {
                "voltage": round(self.telemetry.flyback_voltage, 3),
                "saturation": round(self.telemetry.saturation, 3),
            }
        
        if self.diagnosis:
            compact["diagnosis"] = self.diagnosis.to_summary_string()
        
        compact["success_rate"] = round(self.metrics.success_rate, 3)
        
        return compact
    
    def __repr__(self) -> str:
        """Representación compacta."""
        return (
            f"AgentSnapshot(cycle={self.cycle_number}, "
            f"status={self.current_status.name}, "
            f"timestamp={self.timestamp.isoformat()})"
        )


# ============================================================================
# HISTORIAL CIRCULAR DE SNAPSHOTS
# ============================================================================

class SnapshotHistory:
    """
    Buffer circular de snapshots para análisis temporal.
    
    ESTRUCTURA DE DATOS
    -------------------
    
    Implementa un Ring Buffer de tamaño fijo con índice rotatorio:
    
        H: ℤ/Nℤ → AgentSnapshot
    
    donde N es el tamaño máximo del buffer.
    
    COMPLEJIDAD
    -----------
    
    - add_snapshot: O(1)
    - get_recent: O(k) donde k ≤ N
    - get_all: O(N)
    - Espacio: O(N)
    
    INVARIANTES
    -----------
    
    - len(snapshots) ≤ max_size
    - Los snapshots están ordenados cronológicamente (más reciente = último)
    """
    
    def __init__(self, max_size: int = 100):
        """
        Inicializa el historial.
        
        Args:
            max_size: Tamaño máximo del buffer (default: 100)
            
        Raises:
            ValueError: Si max_size ≤ 0
        """
        if max_size <= 0:
            raise ValueError(f"max_size debe ser positivo: {max_size}")
        
        self._max_size = max_size
        self._snapshots: List[AgentSnapshot] = []
    
    def add_snapshot(self, snapshot: AgentSnapshot) -> None:
        """
        Agrega un snapshot al historial.
        
        Args:
            snapshot: Snapshot a agregar
            
        Efectos:
            - Agrega snapshot al final
            - Si len > max_size: elimina el más antiguo (índice 0)
        """
        self._snapshots.append(snapshot)
        
        # Mantener tamaño máximo (eliminar más antiguo)
        if len(self._snapshots) > self._max_size:
            self._snapshots.pop(0)
    
    def get_recent(self, count: int = 10) -> List[AgentSnapshot]:
        """
        Retorna los últimos N snapshots.
        
        Args:
            count: Número de snapshots a retornar
            
        Returns:
            Lista de snapshots (más reciente = último)
        """
        return self._snapshots[-count:]
    
    def get_all(self) -> List[AgentSnapshot]:
        """Retorna todos los snapshots (copia de la lista)."""
        return list(self._snapshots)
    
    def get_by_cycle(self, cycle_number: int) -> Optional[AgentSnapshot]:
        """
        Busca snapshot por número de ciclo.
        
        Args:
            cycle_number: Número de ciclo a buscar
            
        Returns:
            Snapshot si existe, None en caso contrario
        """
        for snapshot in reversed(self._snapshots):
            if snapshot.cycle_number == cycle_number:
                return snapshot
        return None
    
    def get_status_changes(self) -> List[Tuple[datetime, SystemStatus, SystemStatus]]:
        """
        Detecta transiciones de estado.
        
        Returns:
            Lista de tuplas (timestamp, status_anterior, status_nuevo)
        """
        if len(self._snapshots) < 2:
            return []
        
        changes = []
        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1]
            curr = self._snapshots[i]
            
            if prev.current_status != curr.current_status:
                changes.append((curr.timestamp, prev.current_status, curr.current_status))
        
        return changes
    
    @property
    def size(self) -> int:
        """Número de snapshots almacenados."""
        return len(self._snapshots)
    
    @property
    def max_size(self) -> int:
        """Tamaño máximo del buffer."""
        return self._max_size
    
    @property
    def is_full(self) -> bool:
        """Indica si el buffer está lleno."""
        return len(self._snapshots) >= self._max_size
    
    @property
    def oldest_timestamp(self) -> Optional[datetime]:
        """Timestamp del snapshot más antiguo."""
        if not self._snapshots:
            return None
        return self._snapshots[0].timestamp
    
    @property
    def newest_timestamp(self) -> Optional[datetime]:
        """Timestamp del snapshot más reciente."""
        if not self._snapshots:
            return None
        return self._snapshots[-1].timestamp
    
    def clear(self) -> None:
        """Limpia todo el historial."""
        self._snapshots.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa el historial a diccionario."""
        return {
            "max_size": self._max_size,
            "current_size": self.size,
            "is_full": self.is_full,
            "oldest_timestamp": (
                self.oldest_timestamp.isoformat()
                if self.oldest_timestamp
                else None
            ),
            "newest_timestamp": (
                self.newest_timestamp.isoformat()
                if self.newest_timestamp
                else None
            ),
            "snapshots": [s.to_compact_dict() for s in self._snapshots],
        }
    
    def __len__(self) -> int:
        """Permite len(history)."""
        return len(self._snapshots)
    
    def __getitem__(self, index: int) -> AgentSnapshot:
        """Permite indexación history[i]."""
        return self._snapshots[index]
    
    def __iter__(self) -> Iterator[AgentSnapshot]:
        """Permite iteración for snapshot in history."""
        return iter(self._snapshots)
    
    def __repr__(self) -> str:
        """Representación compacta."""
        return (
            f"SnapshotHistory(size={self.size}/{self._max_size}, "
            f"oldest={self.oldest_timestamp.isoformat() if self.oldest_timestamp else 'N/A'})"
        )


# ============================================================================
# AGENTE AUTÓNOMO - CONTROLADOR OODA
# ============================================================================

class AutonomousAgent:
    """
    Agente Autónomo que opera bajo el ciclo OODA (Observe, Orient, Decide, Act).
    
    FUNDAMENTACIÓN TEÓRICA
    ----------------------
    
    Este componente implementa un Controlador de Lazo Cerrado sobre una
    Variedad Diferenciable M (espacio de estados del sistema):
    
        OODA: M → M,  φ ↦ Act(Decide(Orient(Observe(φ))))
    
    donde cada fase es un morfismo en la categoría de espacios de estados:
    
    1. **Observe**: O: Infrastructure → Telemetry
       - Adquisición de datos del sistema mediante HTTP
       - Manejo robusto de errores con clasificación semántica
    
    2. **Orient**: R: Telemetry × Topology → Diagnosis
       - Análisis topológico (números de Betti, persistencia homológica)
       - Construcción del diagnóstico consolidado
    
    3. **Decide**: D: Diagnosis → Decision
       - Mapeo estado → acción mediante evaluadores en cadena
       - Resolución del peor caso (supremo del retículo de estados)
    
    4. **Act**: A: Decision × Gradient → Morphism
       - Resolución Hamiltoniana de ΔΦ = -ρ via GaugeFieldRouter
       - Inyección de control sobre la MIC (Matriz de Interacción Central)
    
    TEORÍA DE CONTROL
    -----------------
    
    El agente implementa un controlador con las siguientes propiedades:
    
    1. **Estabilidad Asintótica**: ∃V (Lyapunov) tal que V̇ < 0
    2. **Robustez**: Tolerancia a fallos transitorios (retry exponencial)
    3. **Homeostasis**: Restauración automática al equilibrio φ*
    4. **Observabilidad**: Métricas completas para monitoreo
    
    INVARIANTES DEL SISTEMA
    -----------------------
    
    1. **Conectividad Global**: β₀ = 1 (sin fragmentación)
    2. **Estabilidad Dinámica**: λ_max < 0 (convergencia exponencial)
    3. **Cota de Saturación**: S < S_critical
    4. **Continuidad Temporal**: Δt < τ_check (frecuencia de Nyquist)
    
    LIFECYCLE
    ---------
    
    Init → WaitForStartup → HealthCheck → Run(OODA Loop) → Shutdown
    
    - **Init**: Configuración, inicialización de topología, sesión HTTP
    - **WaitForStartup**: Backoff exponencial hasta que Core responda
    - **HealthCheck**: Verificación de conectividad y topología
    - **Run**: Ciclo OODA continuo con manejo de señales
    - **Shutdown**: Limpieza de recursos, cierre de sesión HTTP
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        synthesizer: Optional[HamiltonianSynthesizer] = None,
    ) -> None:
        """
        Inicializa el agente autónomo.
        
        Args:
            config: Configuración del agente (default: desde env vars)
            synthesizer: Sintetizador Hamiltoniano (default: estándar)
            
        Raises:
            ConfigurationError: Si la configuración es inválida
            
        Postcondiciones:
            - self._running == False
            - self.topology inicializada con topología esperada
            - self._session configurada con retry policy
            - Signal handlers instalados para SIGINT/SIGTERM
        """
        # Configuración
        self.config = config or AgentConfig.from_environment()
        
        # Alias para acceso rápido (reduce indirección)
        self._thresholds = self.config.thresholds
        self._conn_config = self.config.connection
        self._topo_config = self.config.topology
        self._timing_config = self.config.timing
        
        # Estado interno del ciclo OODA
        self._running: bool = False
        self._last_decision: Optional[AgentDecision] = None
        self._last_decision_time: Optional[datetime] = None
        self._last_status: Optional[SystemStatus] = None
        self._last_diagnosis: Optional[TopologicalDiagnosis] = None
        self._last_telemetry: Optional[TelemetryData] = None
        
        # Métricas internas
        self._metrics = AgentMetrics()
        
        # Historial de snapshots
        self._snapshot_history = SnapshotHistory(max_size=self._topo_config.max_history)
        
        # Componentes de análisis topológico
        self.topology = SystemTopology(max_history=self._topo_config.max_history)
        self.persistence = PersistenceHomology(
            window_size=self._topo_config.persistence_window
        )
        
        # Sintetizador Hamiltoniano
        self._synthesizer = synthesizer or HamiltonianSynthesizer()
        
        # GaugeFieldRouter (se inicializa después de topología)
        self._gauge_router: Optional[GaugeFieldRouter] = None
        
        # Sesión HTTP con política de reintentos
        self._session = self._create_robust_session()
        
        # Signal handlers para shutdown graceful
        self._original_handlers: Dict[int, Any] = {}
        self._setup_signal_handlers()
        
        # Inicializar topología esperada y sincronizar router gauge
        self._initialize_expected_topology()
        
        logger.info(
            f"[AGENT:INIT] AutonomousAgent inicializado | "
            f"Core: {self._conn_config.base_url} | "
            f"CHECK_INTERVAL: {self._timing_config.check_interval}s | "
            f"REQUEST_TIMEOUT: {self._conn_config.request_timeout}s | "
            f"Evaluadores: {len(self._synthesizer.evaluators)}"
        )
    
    # =========================================================================
    # INICIALIZACIÓN Y CONFIGURACIÓN
    # =========================================================================
    
    def _initialize_expected_topology(self) -> None:
        """
        Establece la topología inicial esperada y sincroniza el GaugeFieldRouter.
        
        TOPOLOGÍA NOMINAL
        -----------------
        
        1-esqueleto del complejo simplicial:
        
            Agent ──→ Core ──→ Redis
                       │
                       └──→ Filesystem
        
        Números de Betti esperados: β₀ = 1, β₁ = 0 (árbol)
        
        Efectos:
            - Agrega nodos y aristas a self.topology
            - Inicializa self._gauge_router con laplaciano L₀ y coborde B₁
            - Registra agentes virtuales en la MIC
        """
        edges_added, warnings = self.topology.update_connectivity(
            list(self._topo_config.expected_edges),
            validate_nodes=True,
            auto_add_nodes=True,
        )
        
        for warn in warnings:
            logger.warning(f"[TOPO:INIT] {warn}")
        
        logger.debug(
            f"[TOPO:INIT] Topología nominal establecida: {edges_added} aristas, "
            f"β₀={self.topology.get_topological_health().betti.b0}"
        )
        
        # Sincronizar router gauge con la topología
        self._sync_gauge_router()
    
    def _sync_gauge_router(self) -> None:
        """
        Sincroniza el GaugeFieldRouter con el estado del complejo simplicial.
        
        CONSTRUCCIÓN DEL OPERADOR GAUGE
        --------------------------------
        
        1. Extraer nodos y aristas del SystemTopology
        2. Construir matriz de incidencia orientada B₁ ∈ ℝ^(m×n)
           donde m = |aristas|, n = |nodos|
        3. Calcular laplaciano L₀ = B₁ᵀB₁ ∈ ℝ^(n×n)
        4. Definir cargas agénticas sobre cada arista
        5. Registrar morfismos virtuales en la MIC
        
        Postcondiciones:
            - self._gauge_router ≠ None
            - ker(L₀) = span{𝟙} verificado
            - Agentes registrados en MIC
        
        Raises:
            GaugeRouterError: Si la construcción falla
        """
        try:
            nodes = sorted(list(self.topology.nodes))
            edges = sorted(list(self.topology.edges))
            n = len(nodes)
            m = len(edges)
            
            if n == 0 or m == 0:
                logger.warning(
                    f"[GAUGE:SYNC] Topología degenerada (n={n}, m={m}), "
                    f"postponiendo sincronización"
                )
                return
            
            # Mapeo nodo → índice
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # Construcción del Operador Coborde B₁ (Matriz de Incidencia Orientada)
            # B₁: C₀ → C₁ donde C₀ = ℝⁿ (0-cadenas), C₁ = ℝᵐ (1-cadenas)
            # Para arista e = (u, v): B₁[e, u] = -1, B₁[e, v] = +1
            b1 = lil_matrix((m, n), dtype=np.float64)
            for i, (u, v) in enumerate(edges):
                b1[i, node_to_idx[u]] = -1.0
                b1[i, node_to_idx[v]] = 1.0
            
            b1_csr = b1.tocsr()
            
            # Laplaciano Combinatorio L₀ = B₁ᵀB₁
            # Es simétrico, semidefinido positivo, con ker(L₀) = span{𝟙}
            l0_csr = (b1_csr.T @ b1_csr).tocsr()
            
            # Generación de la Base de Fock de Cargas Agénticas
            # Asignamos un "agente virtual" por cada arista, con carga localizada
            agent_charges: Dict[str, np.ndarray] = {}
            
            # Mapeo arista → agentes permitidos (policy)
            edge_roles = {
                (NetworkTopology.NODE_AGENT, NetworkTopology.NODE_CORE): ["clean", "reconnect"],
                (NetworkTopology.NODE_CORE, NetworkTopology.NODE_REDIS): ["configure"],
                (NetworkTopology.NODE_CORE, NetworkTopology.NODE_FILESYSTEM): ["configure"],
            }
            
            for i, edge in enumerate(edges):
                # Normalizar arista (orden alfabético para lookup)
                role_key = tuple(sorted(edge))
                agents_for_edge = edge_roles.get(role_key, ["clean"])
                
                for agent_prefix in agents_for_edge:
                    agent_id = f"{agent_prefix}_on_edge_{i}"
                    
                    # Carga localizada: delta de Kronecker en la arista i
                    charge = np.zeros(m, dtype=np.float64)
                    charge[i] = 1.0
                    
                    agent_charges[agent_id] = charge
            
            # Registrar morfismos virtuales en la MIC
            from app.adapters.tools_interface import get_global_mic
            mic = get_global_mic()
            
            for agent_id in agent_charges:
                if not mic.is_registered(agent_id):
                    # Extraer vector base (e.g., "clean" de "clean_on_edge_0")
                    base_vector = agent_id.split("_on_edge_")[0]
                    
                    # Handler virtual: proyecta al vector base
                    def virtual_handler(bv=base_vector, **kwargs):
                        return mic.project_intent(service_name=bv, **kwargs)
                    
                    # Determinar estrato
                    stratum = mic.get_stratum(base_vector) or Stratum.PHYSICS
                    
                    # Registrar
                    mic.register_vector(agent_id, stratum, virtual_handler)
            
            # Inicializar GaugeFieldRouter
            self._gauge_router = GaugeFieldRouter(
                mic_registry=mic,
                laplacian=l0_csr,
                incidence_matrix=b1_csr,
                agent_charges=agent_charges,
                verify_cohomology=True,  # Verificar ker(L₀) = span{𝟙}
            )
            
            logger.info(
                f"[GAUGE:SYNC] Router sincronizado exitosamente: "
                f"N={n} nodos, M={m} aristas, {len(agent_charges)} agentes virtuales"
            )
            
        except Exception as e:
            logger.error(
                f"[GAUGE:SYNC] Fallo crítico en sincronización: {e}",
                exc_info=True,
            )
            raise GaugeRouterError(f"Sincronización de gauge router falló: {e}")
    
    def _create_robust_session(self) -> requests.Session:
        """
        Crea sesión HTTP con política de reintentos exponenciales y pooling.
        
        POLÍTICA DE REINTENTOS
        ----------------------
        
        Usa urllib3.Retry con:
        - Backoff exponencial: T_n = backoff_factor * 2^n
        - Status codes que disparan retry: 500, 502, 503, 504
        - Métodos permitidos: GET, POST
        - Total de reintentos: max_retries
        
        POOL DE CONEXIONES
        ------------------
        
        HTTPAdapter con:
        - pool_connections: Número de workers concurrentes
        - pool_maxsize: Tamaño del buffer de conexiones
        
        Returns:
            Sesión configurada con retry policy y pool
            
        Postcondiciones:
            - session.adapters contiene HTTPAdapter para http:// y https://
            - Headers predeterminados configurados (User-Agent, Content-Type)
        """
        session = requests.Session()
        
        # Estrategia de reintentos
        retry_strategy = Retry(
            total=self._conn_config.max_retries,
            backoff_factor=self._conn_config.backoff_factor,
            status_forcelist=list(self._conn_config.retry_status_codes),
            allowed_methods=["GET", "POST"],
            raise_on_status=False,  # No lanzar excepción en status codes
        )
        
        # Adapter con pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self._conn_config.pool_connections,
            pool_maxsize=self._conn_config.pool_maxsize,
        )
        
        # Montar para http y https
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers predeterminados
        session.headers.update({
            "User-Agent": "APU-Agent-Internal/2.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        
        logger.debug(
            f"[AGENT:SESSION] Sesión HTTP creada: max_retries={self._conn_config.max_retries}, "
            f"pool_size={self._conn_config.pool_maxsize}"
        )
        
        return session
    
    def _setup_signal_handlers(self) -> None:
        """
        Configura manejadores de señales SIGINT y SIGTERM para shutdown graceful.
        
        Efectos:
            - Instala self._handle_shutdown para SIGINT (Ctrl+C)
            - Instala self._handle_shutdown para SIGTERM (kill)
            - Guarda handlers originales en self._original_handlers
        
        Postcondiciones:
            - signal.getsignal(SIGINT) == self._handle_shutdown
            - signal.getsignal(SIGTERM) == self._handle_shutdown
        """
        signals_to_handle = [signal.SIGINT, signal.SIGTERM]
        
        for sig in signals_to_handle:
            original_handler = signal.signal(sig, self._handle_shutdown)
            self._original_handlers[sig] = original_handler
        
        logger.debug("[AGENT:SIGNALS] Signal handlers configurados (SIGINT, SIGTERM)")
    
    def _restore_signal_handlers(self) -> None:
        """
        Restaura los manejadores de señales originales.
        
        Efectos:
            - Reinstala handlers guardados en self._original_handlers
        """
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        
        logger.debug("[AGENT:SIGNALS] Signal handlers restaurados")
    
    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """
        Manejador de señales de terminación.
        
        Args:
            signum: Número de señal recibida
            frame: Frame de ejecución (no usado)
            
        Efectos:
            - Setea self._running = False
            - Logea señal recibida
        """
        try:
            sig_name = signal.Signals(signum).name
        except (ValueError, AttributeError):
            sig_name = str(signum)
        
        logger.info(f"[AGENT:SIGNAL] Señal {sig_name} recibida. Iniciando shutdown graceful...")
        self._running = False
    
    # =========================================================================
    # FASE OBSERVE: ADQUISICIÓN DE TELEMETRÍA
    # =========================================================================
    
    def observe(self) -> Optional[TelemetryData]:
        """
        OBSERVE - Primera fase del ciclo OODA.
        
        Implementa el morfismo O: Infrastructure → Telemetry con manejo robusto
        de errores y actualización topológica.
        
        ALGORITMO
        ---------
        
        1. Generar request_id único
        2. Ejecutar petición HTTP a /api/telemetry/status
        3. Clasificar resultado (éxito/fallo con tipo de error)
        4. Actualizar métricas (consecutive_failures, success_rate)
        5. Actualizar topología según conectividad
        6. Retornar TelemetryData o None
        
        Returns:
            TelemetryData si la observación fue exitosa, None en caso contrario
            
        Efectos secundarios:
            - Actualiza self._metrics (record_success/record_failure)
            - Actualiza self.topology (record_request, update_connectivity)
            - Actualiza self._last_telemetry
            - Logea evento de observación
        """
        # Generar ID único para trazabilidad
        request_id = f"obs_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        # Ejecutar observación
        result = self._execute_observation(request_id)
        
        # Procesar resultado
        self._process_observation_result(result)
        
        return result.telemetry
    
    def _execute_observation(self, request_id: str) -> ObservationResult:
        """
        Realiza la petición HTTP al endpoint de telemetría.
        
        Args:
            request_id: Identificador único de la observación
            
        Returns:
            ObservationResult con éxito o fallo clasificado
            
        Manejo de errores:
            - Timeout → ObservationErrorType.TIMEOUT
            - ConnectionError → ObservationErrorType.CONNECTION_ERROR
            - HTTP error (4xx/5xx) → ObservationErrorType.HTTP_ERROR
            - JSON inválido → ObservationErrorType.INVALID_JSON
            - Telemetría inválida → ObservationErrorType.INVALID_TELEMETRY
            - Otros → ObservationErrorType.UNKNOWN
        """
        try:
            response = self._session.get(
                self._conn_config.telemetry_endpoint,
                timeout=self._conn_config.request_timeout,
            )
            
            # Verificar status code
            if not response.ok:
                logger.warning(
                    f"[OBSERVE:HTTP] Error HTTP {response.status_code} en {request_id}"
                )
                return ObservationResult.failure_result(
                    ObservationErrorType.HTTP_ERROR,
                    request_id,
                )
            
            # Parsear JSON
            try:
                raw_data = response.json()
            except ValueError as e:
                logger.warning(f"[OBSERVE:JSON] JSON inválido en {request_id}: {e}")
                return ObservationResult.failure_result(
                    ObservationErrorType.INVALID_JSON,
                    request_id,
                )
            
            # Construir TelemetryData
            telemetry = TelemetryData.from_dict(raw_data)
            if telemetry is None:
                logger.warning(
                    f"[OBSERVE:SCHEMA] Schema de telemetría inválido en {request_id}"
                )
                return ObservationResult.failure_result(
                    ObservationErrorType.INVALID_TELEMETRY,
                    request_id,
                )
            
            # Éxito
            return ObservationResult.success_result(telemetry, request_id)
            
        except requests.exceptions.Timeout:
            logger.warning(f"[OBSERVE:TIMEOUT] Timeout en {request_id}")
            return ObservationResult.failure_result(
                ObservationErrorType.TIMEOUT,
                request_id,
            )
        except requests.exceptions.ConnectionError:
            logger.warning(f"[OBSERVE:CONN] Error de conexión en {request_id}")
            return ObservationResult.failure_result(
                ObservationErrorType.CONNECTION_ERROR,
                request_id,
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"[OBSERVE:REQ] Error de requests en {request_id}: {e}")
            return ObservationResult.failure_result(
                ObservationErrorType.REQUEST_ERROR,
                request_id,
            )
        except Exception as e:
            logger.error(
                f"[OBSERVE:UNKNOWN] Error inesperado en {request_id}: {e}",
                exc_info=True,
            )
            return ObservationResult.failure_result(
                ObservationErrorType.UNKNOWN,
                request_id,
            )
    
    def _process_observation_result(self, result: ObservationResult) -> None:
        """
        Procesa el resultado de observación actualizando estado y topología.
        
        Args:
            result: Resultado de la observación
            
        Efectos:
            - Si éxito:
                * Actualiza métricas (record_success)
                * Registra request en topología
                * Actualiza conectividad basada en telemetría
                * Limpia historial de requests
                * Actualiza self._last_telemetry
            - Si fallo:
                * Actualiza métricas (record_failure)
                * Registra request con ID de error en topología
                * Si consecutive_failures > threshold: degrada topología
        """
        if result.success and result.telemetry:
            # Éxito
            self._metrics.record_success()
            self.topology.record_request(result.request_id)
            self._last_telemetry = result.telemetry
            
            # Actualizar conectividad basada en flags de telemetría
            self._update_topology_from_telemetry(result.telemetry)
            
            # Limpiar historial de requests para evitar crecimiento ilimitado
            self.topology.clear_request_history()
            
            # Log detallado si hubo clamping
            if result.telemetry.was_clamped:
                logger.debug(
                    f"[OBSERVE:CLAMP] Valores clampeados en {result.request_id}"
                )
            
            logger.debug(
                f"[OBSERVE:SUCCESS] {result.request_id}: "
                f"V={result.telemetry.flyback_voltage:.3f}, "
                f"S={result.telemetry.saturation:.3f}, "
                f"I={result.telemetry.integrity_score:.3f}"
            )
        else:
            # Fallo
            self._metrics.record_failure()
            
            # Registrar request con ID de error para detección de patrones
            error_id = f"FAIL_{result.error_type.value if result.error_type else 'UNKNOWN'}"
            self.topology.record_request(error_id)
            
            # Degradar topología si hay muchos fallos consecutivos
            if self._metrics.consecutive_failures >= self._timing_config.max_consecutive_failures:
                logger.warning(
                    f"[OBSERVE:DEGRADE] {self._metrics.consecutive_failures} fallos consecutivos, "
                    f"degradando topología Agent↔Core"
                )
                self.topology.remove_edge(
                    NetworkTopology.NODE_AGENT,
                    NetworkTopology.NODE_CORE,
                )
            
            if result.error_type:
                logger.warning(
                    f"[OBSERVE:FAILURE] {result.request_id}: {result.error_type.value}"
                )
    
    def _update_topology_from_telemetry(self, telemetry: TelemetryData) -> None:
        """
        Actualiza la topología del sistema basándose en flags de conectividad.
        
        Args:
            telemetry: Datos de telemetría con metadata de conectividad
            
        Efectos:
            - Agrega/remueve aristas según flags booleanos en raw_data
            - Actualiza estado del grafo en self.topology
        
        Flags esperados en telemetry.raw_data:
            - redis_connected: bool (default: True)
            - filesystem_accessible: bool (default: True)
        """
        raw = telemetry.raw_data
        
        # Aristas siempre activas si hay telemetría exitosa
        active_connections = [(NetworkTopology.NODE_AGENT, NetworkTopology.NODE_CORE)]
        
        # Agregar aristas condicionales
        if raw.get("redis_connected", True):
            active_connections.append(
                (NetworkTopology.NODE_CORE, NetworkTopology.NODE_REDIS)
            )
        
        if raw.get("filesystem_accessible", True):
            active_connections.append(
                (NetworkTopology.NODE_CORE, NetworkTopology.NODE_FILESYSTEM)
            )
        
        # Actualizar topología
        edges_added, warnings = self.topology.update_connectivity(
            active_connections,
            validate_nodes=True,
            auto_add_nodes=False,  # No agregar nodos nuevos dinámicamente
        )
        
        if warnings:
            for warn in warnings:
                logger.warning(f"[TOPO:UPDATE] {warn}")
    
    # =========================================================================
    # FASE ORIENT: ANÁLISIS TOPOLÓGICO Y DIAGNÓSTICO
    # =========================================================================
    
    def orient(self, telemetry: Optional[TelemetryData]) -> SystemStatus:
        """
        ORIENT - Segunda fase del ciclo OODA.
        
        Implementa el morfismo R: Telemetry × Topology → Diagnosis mediante:
        
        1. Cálculo de invariantes topológicos (β₀, β₁, χ)
        2. Análisis de persistencia homológica de métricas temporales
        3. Evaluación de estado mediante el HamiltonianSynthesizer
        4. Construcción del diagnóstico consolidado
        
        Args:
            telemetry: Datos de telemetría actuales (puede ser None)
            
        Returns:
            SystemStatus evaluado (peor caso entre todos los evaluadores)
            
        Efectos secundarios:
            - Actualiza self.persistence con nuevas muestras
            - Calcula salud topológica actual
            - Construye y almacena self._last_diagnosis
            - Logea diagnóstico si status ≠ NOMINAL
        
        Postcondiciones:
            - self._last_diagnosis ≠ None
            - self._last_diagnosis.recommended_status == resultado
        """
        # Persistir muestras temporales ANTES de análisis (efecto temporal)
        if telemetry:
            self.persistence.add_sample("flyback_voltage", telemetry.flyback_voltage)
            self.persistence.add_sample("saturation", telemetry.saturation)
        
        # Calcular invariantes topológicos
        topo_health = self.topology.get_topological_health(calculate_b1=True)
        
        # Analizar persistencia de métricas
        voltage_analysis = self.persistence.analyze_persistence(
            "flyback_voltage",
            threshold=self._thresholds.flyback_voltage_warning,
            noise_ratio=0.2,
            critical_ratio=0.5,
        )
        
        saturation_analysis = self.persistence.analyze_persistence(
            "saturation",
            threshold=self._thresholds.saturation_warning,
            noise_ratio=0.2,
            critical_ratio=0.5,
        )
        
        # Evaluar estado mediante sintetizador (método legacy)
        status, summary = self._synthesizer.evaluate(
            telemetry,
            topo_health,
            voltage_analysis,
            saturation_analysis,
            self._thresholds,
            self._metrics,
        )
        
        # Construir diagnóstico consolidado
        self._last_diagnosis = TopologicalDiagnosis(
            health=topo_health,
            voltage_persistence=voltage_analysis,
            saturation_persistence=saturation_analysis,
            summary=summary,
            recommended_status=status,
        )
        
        # Log si no es nominal
        if status != SystemStatus.NOMINAL:
            logger.info(
                f"[ORIENT] {status.emoji} {status.name} | "
                f"{self._last_diagnosis.to_log_dict()}"
            )
        
        # Log de ruido filtrado (debug)
        for name, analysis in [
            ("voltaje", voltage_analysis),
            ("saturación", saturation_analysis),
        ]:
            if analysis.state == MetricState.NOISE:
                logger.debug(
                    f"[ORIENT:NOISE] Ruido {name} filtrado: "
                    f"{analysis.noise_count} excursiones transitorias"
                )
        
        return status
    
    # =========================================================================
    # FASE DECIDE: MAPEO ESTADO → DECISIÓN
    # =========================================================================
    
    def decide(self, status: SystemStatus) -> AgentDecision:
        """
        DECIDE - Tercera fase del ciclo OODA.
        
        Implementa el morfismo D: SystemStatus → AgentDecision mediante
        un mapeo determinista (decision matrix).
        
        DECISION MATRIX
        ---------------
        
        NOMINAL      → HEARTBEAT
        UNKNOWN      → WAIT
        INESTABLE    → EJECUTAR_LIMPIEZA
        SATURADO     → AJUSTAR_VELOCIDAD
        CRITICO      → ALERTA_CRITICA
        DISCONNECTED → RECONNECT
        
        Args:
            status: Estado actual del sistema
            
        Returns:
            Decisión a ejecutar
            
        Efectos secundarios:
            - Registra decisión en self._metrics.decisions_count
            - Actualiza self._last_status
            - Logea patrones de error si status=NOMINAL pero hay loops
        
        Postcondiciones:
            - resultado.name en self._metrics.decisions_count
            - self._last_status == status
        """
        # Decision matrix (mapeo determinista)
        decision_matrix: Dict[SystemStatus, AgentDecision] = {
            SystemStatus.NOMINAL: AgentDecision.HEARTBEAT,
            SystemStatus.UNKNOWN: AgentDecision.WAIT,
            SystemStatus.INESTABLE: AgentDecision.EJECUTAR_LIMPIEZA,
            SystemStatus.SATURADO: AgentDecision.AJUSTAR_VELOCIDAD,
            SystemStatus.CRITICO: AgentDecision.ALERTA_CRITICA,
            SystemStatus.DISCONNECTED: AgentDecision.RECONNECT,
        }
        
        decision = decision_matrix.get(status, AgentDecision.WAIT)
        
        # Refinamiento contextual: log de patrones de error en estado nominal
        if self._last_diagnosis and decision == AgentDecision.HEARTBEAT:
            error_loops = [
                loop for loop in self._last_diagnosis.health.request_loops
                if loop.request_id.startswith("FAIL_")
            ]
            if error_loops:
                total_errors = sum(loop.count for loop in error_loops)
                logger.debug(
                    f"[DECIDE] {decision.emoji} {decision.name} (nominal) pero "
                    f"con {len(error_loops)} patrones de error (Σ={total_errors})"
                )
        
        # Registrar métricas
        self._metrics.record_decision(decision)
        self._last_status = status
        
        logger.debug(f"[DECIDE] {status.emoji} {status.name} → {decision.emoji} {decision.name}")
        
        return decision
    
    # =========================================================================
    # FASE ACT: INYECCIÓN DE CONTROL
    # =========================================================================
    
    def act(self, decision: AgentDecision) -> bool:
        """
        ACT - Cuarta fase del ciclo OODA (Resolución Hamiltoniana).
        
        Implementa el morfismo A: Decision × Gradient → Morphism mediante:
        
        1. Verificación de debounce (suprimir acciones repetidas)
        2. Síntesis del gradiente total ∇V_total
        3. Resolución de Poisson: ΔΦ = -∇V_total via GaugeFieldRouter
        4. Certificación de estabilidad: P_diss = ⟨Φ, ∇V⟩ ≥ 0
        5. Fallback a lógica procedural si gauge router no está disponible
        
        ECUACIÓN DE POISSON
        -------------------
        
        Resuelve: L₀·Φ = -∇V_total
        
        donde:
        - L₀ = B₁ᵀB₁ es el Laplaciano combinatorio
        - ∇V_total = Σᵢ ∇Vᵢ es el gradiente de penalización
        - Φ es el potencial gauge (selector de agentes)
        
        CERTIFICACIÓN DE ESTABILIDAD
        -----------------------------
        
        Verifica: P_diss = ⟨Φ, ∇V_total⟩ ≥ 0
        
        Si P_diss < 0: frustración cohomológica (violación de 2ª ley)
        
        Args:
            decision: Decisión a ejecutar
            
        Returns:
            True si se ejecutó alguna acción, False si fue suprimida
            
        Efectos secundarios:
            - Actualiza self._last_decision y self._last_decision_time
            - Proyecta intención sobre la MIC (si se ejecuta acción)
            - Notifica a sistemas externos
            - Logea acción ejecutada
        
        Raises:
            StabilityError: Si P_diss < 0 (frustración cohomológica)
        """
        # Verificar debounce
        if self._should_debounce(decision):
            logger.debug(
                f"[ACT:DEBOUNCE] Decisión {decision.name} suprimida por debounce "
                f"(última ejecución hace {(datetime.now() - self._last_decision_time).total_seconds():.1f}s)"
            )
            return False
        
        # Síntesis del gradiente total
        if self._last_diagnosis is not None:
            num_nodes = len(self.topology.nodes)
            grad_total = self._synthesizer.synthesize_gradient(
                self._last_telemetry,
                self._last_diagnosis.health,
                self._last_diagnosis.voltage_persistence,
                self._last_diagnosis.saturation_persistence,
                self._thresholds,
                self._metrics,
                num_nodes=num_nodes,
            )
        else:
            grad_total = np.array([])
        
        # RESOLUCIÓN HAMILTONIANA vía GaugeFieldRouter
        if (
            self._gauge_router is not None
            and grad_total.size > 0
            and np.any(grad_total > PhysicalConstants.EPSILON)
        ):
            try:
                logger.info(
                    f"[ACT:HAMILTONIAN] Resolviendo Poisson para ∇V_total "
                    f"(‖∇V‖₂={np.linalg.norm(grad_total):.4e})"
                )
                
                # Construir estado inicial
                initial_state = CategoricalState(
                    payload={"action": decision.name, "decision": decision.emoji},
                    context={
                        "diagnosis": self._last_diagnosis.summary,
                        "cycle": self._metrics.cycles_executed,
                    },
                )
                
                # Resolver Poisson y obtener morfismo de corrección
                result_state = self._gauge_router.route_gradient(initial_state, grad_total)
                
                # Extraer potencial gauge Φ
                selected_agent = result_state.context.get("gauge_selected_agent")
                phi = np.array(result_state.context.get("gauge_potential", []))
                
                # CERTIFICACIÓN DE ESTABILIDAD
                if phi.size > 0 and phi.size == grad_total.size:
                    # Disipación entrópica: P_diss = ⟨Φ, ∇V_total⟩
                    p_diss = float(np.dot(phi, grad_total))
                    
                    if p_diss < -PhysicalConstants.EPSILON:
                        # Frustración cohomológica: el potencial aumenta la energía
                        logger.critical(
                            f"[ACT:STABILITY] ⚠️ FRUSTRACIÓN COHOMOLÓGICA DETECTADA! "
                            f"P_diss={p_diss:.4e} < 0 (violación de 2ª ley)"
                        )
                        raise StabilityError(
                            f"Frustración cohomológica: P_diss={p_diss:.4e} < 0. "
                            f"El sistema no puede alcanzar estabilidad.",
                            p_diss=p_diss,
                            gradient_norm=np.linalg.norm(grad_total),
                        )
                    
                    logger.debug(
                        f"[ACT:STABILITY] ✓ Certificación exitosa: "
                        f"P_diss={p_diss:.4e} ≥ 0 (disipación entrópica)"
                    )
                
                logger.info(
                    f"[ACT:GAUGE] {decision.emoji} Morfismo de corrección aplicado: "
                    f"'{selected_agent}'"
                )
                
                # Actualizar estado
                self._last_decision = decision
                self._last_decision_time = datetime.now()
                
                return True
                
            except StabilityError:
                # Re-lanzar errores de estabilidad
                raise
            except Exception as e:
                logger.error(
                    f"[ACT:GAUGE] Error en resolución de Poisson: {e}",
                    exc_info=True,
                )
                # Continuar con fallback a lógica procedural
        
        # FALLBACK: Lógica procedural legacy
        diagnosis_msg = self._build_diagnosis_message()
        
        action_handlers: Dict[AgentDecision, Callable[[], None]] = {
            AgentDecision.HEARTBEAT: self._emit_heartbeat,
            AgentDecision.EJECUTAR_LIMPIEZA: lambda: self._execute_cleanup(diagnosis_msg),
            AgentDecision.AJUSTAR_VELOCIDAD: lambda: self._apply_backpressure(diagnosis_msg),
            AgentDecision.ALERTA_CRITICA: lambda: self._raise_critical_alert(diagnosis_msg),
            AgentDecision.RECONNECT: lambda: self._attempt_reconnection(diagnosis_msg),
            AgentDecision.WAIT: lambda: logger.info(
                f"[ACT] {AgentDecision.WAIT.emoji} Esperando más información..."
            ),
        }
        
        handler = action_handlers.get(decision, action_handlers[AgentDecision.WAIT])
        handler()
        
        # Actualizar estado
        self._last_decision = decision
        self._last_decision_time = datetime.now()
        
        return True
    
    def _should_debounce(self, decision: AgentDecision) -> bool:
        """
        Determina si una decisión debe ser suprimida por debounce temporal.
        
        Args:
            decision: Decisión a evaluar
            
        Returns:
            True si la decisión debe suprimirse
            
        Lógica:
            - Decisiones críticas (requires_immediate_action): nunca debounce
            - Primera decisión: nunca debounce
            - Decisión diferente a la anterior: nunca debounce
            - Decisión igual y dentro de ventana: sí debounce
        """
        # Decisiones críticas nunca se suprimen
        if decision.requires_immediate_action:
            return False
        
        # Primera decisión
        if self._last_decision is None or self._last_decision_time is None:
            return False
        
        # Decisión diferente
        if decision != self._last_decision:
            return False
        
        # Verificar ventana temporal
        elapsed = datetime.now() - self._last_decision_time
        return elapsed < timedelta(seconds=self._timing_config.debounce_window_seconds)
    
    # -------------------------------------------------------------------------
    # Handlers de acciones específicas (lógica procedural legacy)
    # -------------------------------------------------------------------------
    
    def _emit_heartbeat(self) -> None:
        """Emite señal de sistema nominal con indicador de salud."""
        health_score = (
            self._last_diagnosis.health.health_score
            if self._last_diagnosis
            else 1.0
        )
        
        # Indicador visual basado en score
        if health_score >= 0.9:
            indicator = "✅"
        elif health_score >= 0.7:
            indicator = "🟢"
        else:
            indicator = "🟡"
        
        logger.info(
            f"[ACT] {indicator} HEARTBEAT | h={health_score:.2f} | "
            f"Ciclo #{self._metrics.cycles_executed}"
        )
    
    def _execute_cleanup(self, diagnosis_msg: str) -> None:
        """
        Proyecta vector de limpieza al estrato físico.
        
        Args:
            diagnosis_msg: Mensaje con diagnóstico detallado
        """
        logger.warning(f"[ACT] 🧹 EJECUTAR_LIMPIEZA | {diagnosis_msg}")
        
        success = self._project_intent(
            vector="clean",
            stratum="PHYSICS",
            payload={
                "mode": "EMERGENCY",
                "reason": diagnosis_msg,
                "scope": "flux_condenser",
                "diagnosis": (
                    self._last_diagnosis.to_dict()
                    if self._last_diagnosis
                    else {}
                ),
            },
        )
        
        event = "instability_resolved" if success else "instability_correction_failed"
        self._notify_external_system(event, {"method": "clean", "success": success})
    
    def _apply_backpressure(self, diagnosis_msg: str) -> None:
        """
        Aplica backpressure reduciendo tasa de entrada.
        
        Args:
            diagnosis_msg: Mensaje con diagnóstico detallado
        """
        logger.warning(f"[ACT] 🔽 AJUSTAR_VELOCIDAD | {diagnosis_msg}")
        
        success = self._project_intent(
            vector="configure",
            stratum="PHYSICS",
            payload={
                "target": "flux_condenser",
                "parameter": "input_rate",
                "action": "decrease",
                "factor": 0.5,  # Reducir al 50%
            },
        )
        
        event = "saturation_mitigated" if success else "saturation_correction_failed"
        self._notify_external_system(event, {"method": "throttle", "success": success})
    
    def _raise_critical_alert(self, diagnosis_msg: str) -> None:
        """
        Emite alerta crítica con contexto topológico.
        
        Args:
            diagnosis_msg: Mensaje con diagnóstico detallado
        """
        logger.critical(f"[ACT] 🚨 ALERTA_CRITICA | {diagnosis_msg}")
        logger.critical("[ACT] → Intervención inmediata requerida")
        
        context: Dict[str, Any] = {
            "diagnosis": diagnosis_msg,
            "timestamp": datetime.now().isoformat(),
        }
        
        if self._last_diagnosis:
            context.update({
                "health_score": self._last_diagnosis.health.health_score,
                "betti_b0": self._last_diagnosis.health.betti.b0,
                "is_connected": self._last_diagnosis.is_structurally_healthy,
                "critical_issues": self._last_diagnosis.get_critical_issues(),
            })
        
        self._notify_external_system("critical_alert", context)
    
    def _attempt_reconnection(self, diagnosis_msg: str) -> None:
        """
        Intenta reconexión reinicializando topología.
        
        Args:
            diagnosis_msg: Mensaje con diagnóstico detallado
        """
        logger.warning(f"[ACT] 🔄 RECONNECT | {diagnosis_msg}")
        logger.warning("[ACT] → Reinicializando topología esperada...")
        
        try:
            self._initialize_expected_topology()
            logger.info("[ACT] ✓ Topología reinicializada exitosamente")
        except Exception as e:
            logger.error(f"[ACT] ✗ Fallo al reinicializar topología: {e}")
    
    def _project_intent(
        self,
        vector: str,
        stratum: str,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Proyecta intención sobre la MIC (Matriz de Interacción Central).
        
        Args:
            vector: Nombre del vector (herramienta)
            stratum: Nivel de gobernanza (PHYSICS, TACTICS, etc.)
            payload: Datos específicos de la acción
            
        Returns:
            True si la proyección fue exitosa (HTTP 2xx)
            
        Efectos:
            - Envía POST a /api/tools/{vector}
            - Logea resultado
        """
        intent = {
            "vector": vector,
            "stratum": stratum,
            "payload": payload,
            "context": {
                "agent_id": "apu_agent_autonomous",
                "timestamp": datetime.now().isoformat(),
                "force_physics_override": True,
                "topology_health": (
                    self._last_diagnosis.health.health_score
                    if self._last_diagnosis
                    else None
                ),
                "cycle": self._metrics.cycles_executed,
            },
        }
        
        url = self._conn_config.tools_endpoint(vector)
        logger.info(f"[ACT:INTENT] Proyectando '{vector}' → estrato '{stratum}'")
        
        try:
            response = self._session.post(
                url,
                json=intent,
                timeout=self._conn_config.request_timeout,
            )
            
            if response.ok:
                logger.info(f"[ACT:INTENT] ✓ '{vector}' ejecutado exitosamente")
                return True
            
            logger.error(
                f"[ACT:INTENT] ✗ HTTP {response.status_code}: {response.text[:200]}"
            )
            return False
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[ACT:INTENT] ✗ Error de proyección: {type(e).__name__}: {e}")
            return False
    
    def _build_diagnosis_message(self) -> str:
        """
        Construye mensaje de diagnóstico legible para humanos.
        
        Returns:
            Cadena con la información más relevante
        """
        if not self._last_diagnosis:
            return "Sin diagnóstico disponible"
        
        diag = self._last_diagnosis
        components = [diag.summary]
        
        # Agregar números de Betti si no son ideales
        betti = diag.health.betti
        if not betti.is_ideal:
            components.append(f"β₀={betti.b0}, β₁={betti.b1}")
        
        # Agregar score de salud si está degradado
        if diag.health.health_score < 0.9:
            components.append(f"h={diag.health.health_score:.2f}")
        
        return " | ".join(components)
    
    def _notify_external_system(
        self,
        event_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Hook para notificaciones externas (monitoreo, alertas, webhooks).
        
        Args:
            event_type: Tipo de evento (e.g., "critical_alert")
            context: Datos adicionales del evento
            
        Nota:
            Actualmente solo logea. En producción, aquí se enviarían
            eventos a sistemas de monitoreo (Prometheus, Datadog, etc.)
        """
        log_data: Dict[str, Any] = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
        }
        if context:
            log_data.update(context)
        
        logger.debug(f"[ACT:NOTIFY] {log_data}")
    
    # =========================================================================
    # LIFECYCLE: STARTUP, RUN, SHUTDOWN
    # =========================================================================
    
    def _wait_for_startup(self) -> bool:
        """
        Espera a que el Core esté disponible (Cold Start con backoff exponencial).
        
        ALGORITMO
        ---------
        
        1. Intentar conexión al endpoint de telemetría
        2. Si falla: esperar T_n = initial * multiplier^n (saturado en max)
        3. Repetir hasta éxito o max_attempts
        
        Returns:
            True si el Core respondió exitosamente, False si timeout
            
        Postcondiciones:
            - Si True: Core responde con HTTP 2xx
            - Si False: Se agotaron los intentos
        """
        logger.info("[LIFECYCLE:STARTUP] Iniciando protocolo de Cold Start...")
        
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
                    logger.info(
                        f"[LIFECYCLE:STARTUP] ✅ Core detectado y operativo "
                        f"(HTTP {response.status_code}) tras {attempts} intento(s)"
                    )
                    return True
                else:
                    logger.info(
                        f"[LIFECYCLE:STARTUP] Esperando Core... "
                        f"(HTTP {response.status_code}) [{attempts}/{max_attempts}]"
                    )
                    
            except requests.exceptions.ConnectionError:
                logger.info(
                    f"[LIFECYCLE:STARTUP] Esperando Core (Cold Start)... "
                    f"[Conexión rechazada] [{attempts}/{max_attempts}]"
                )
            except requests.exceptions.RequestException as e:
                logger.info(
                    f"[LIFECYCLE:STARTUP] Esperando Core... "
                    f"[{type(e).__name__}] [{attempts}/{max_attempts}]"
                )
            
            # Sleep con backoff exponencial
            if self._running:
                logger.debug(f"[LIFECYCLE:STARTUP] Esperando {backoff:.1f}s antes del próximo intento...")
                time.sleep(backoff)
                
                # Incrementar backoff con saturación
                backoff = min(
                    backoff * self._timing_config.startup_backoff_multiplier,
                    self._timing_config.startup_backoff_max,
                )
        
        # Verificar si fue cancelado por señal
        if not self._running:
            logger.info("[LIFECYCLE:STARTUP] Cold Start cancelado por señal de shutdown")
            return False
        
        # Timeout
        logger.error(
            f"[LIFECYCLE:STARTUP] ❌ Timeout de Cold Start después de "
            f"{attempts} intentos ({attempts * self._timing_config.startup_backoff_initial:.0f}s)"
        )
        return False
    
    def health_check(self) -> bool:
        """
        Verifica conectividad con el Core y estado topológico.
        
        Returns:
            True si el Core es accesible y la topología es válida
            
        Efectos:
            - Inicializa topología esperada si la conexión es exitosa
            - Degrada topología si la conexión falla
            - Logea resultado
        """
        logger.info(
            f"[LIFECYCLE:HEALTH] Ejecutando health check: "
            f"{self._conn_config.telemetry_endpoint}"
        )
        
        try:
            response = self._session.get(
                self._conn_config.telemetry_endpoint,
                timeout=self._conn_config.request_timeout,
            )
            
            if response.ok:
                # Reinicializar topología tras conexión exitosa
                self._initialize_expected_topology()
                
                topo_health = self.topology.get_topological_health(calculate_b1=False)
                
                logger.info(
                    f"[LIFECYCLE:HEALTH] ✅ Health check exitoso | "
                    f"Core accesible, topología: {topo_health.level.name} "
                    f"(score={topo_health.health_score:.2f}, β₀={topo_health.betti.b0})"
                )
                return True
            else:
                logger.warning(
                    f"[LIFECYCLE:HEALTH] ⚠️ Health check: HTTP {response.status_code}, "
                    f"pero continuando (Core parcialmente disponible)"
                )
                return True  # Permitir inicio con advertencias
                
        except requests.exceptions.RequestException as e:
            logger.error(
                f"[LIFECYCLE:HEALTH] ❌ Health check fallido: {type(e).__name__}: {e}"
            )
            
            # Degradar topología
            self.topology.remove_edge(
                NetworkTopology.NODE_AGENT,
                NetworkTopology.NODE_CORE,
            )
            
            topo_health = self.topology.get_topological_health(calculate_b1=False)
            logger.error(
                f"[LIFECYCLE:HEALTH] Topología degradada: β₀={topo_health.betti.b0}, "
                f"health={topo_health.health_score:.2f}"
            )
            
            return False
    
    def run(self, skip_health_check: bool = False) -> None:
        """
        Bucle principal del agente - Ejecuta el ciclo OODA continuamente.
        
        ALGORITMO
        ---------
        
        1. Esperar a que el Core esté disponible (Cold Start)
        2. Ejecutar health check inicial (opcional)
        3. Loop infinito:
            a. Incrementar contador de ciclos
            b. OBSERVE: Adquirir telemetría
            c. ORIENT: Analizar topología y diagnosticar
            d. DECIDE: Mapear estado → decisión
            e. ACT: Ejecutar acción (con resolución Hamiltoniana)
            f. Capturar snapshot del estado
            g. Sleep adaptativo (compensando duración del ciclo)
        4. Cleanup al salir (shutdown)
        
        Args:
            skip_health_check: Si True, omite verificación inicial (para testing)
            
        Efectos:
            - Setea self._running = True
            - Ejecuta ciclo OODA hasta señal de terminación
            - Captura snapshots en self._snapshot_history
            - Llama self._shutdown() al finalizar
        
        Raises:
            Exception: Errores fatales se propagan tras logging
        """
        self._running = True
        
        # FASE 1: Cold Start
        if not skip_health_check:
            if not self._wait_for_startup():
                logger.error("[LIFECYCLE:RUN] No se pudo conectar al Core. Abortando.")
                return
            
            # Health check inicial
            if not self.health_check():
                logger.warning(
                    "[LIFECYCLE:RUN] Health check inicial falló, "
                    "iniciando agente con advertencias..."
                )
        
        # FASE 2: Ciclo OODA
        logger.info("🚀 [LIFECYCLE:RUN] Iniciando OODA Loop...")
        
        try:
            while self._running:
                cycle_start = time.monotonic()
                self._metrics.increment_cycle()
                
                try:
                    # ═══════════════════════════════════════════════════════
                    # CICLO OODA
                    # ═══════════════════════════════════════════════════════
                    
                    # OBSERVE
                    telemetry = self.observe()
                    
                    # ORIENT
                    status = self.orient(telemetry)
                    
                    # DECIDE
                    decision = self.decide(status)
                    
                    # ACT
                    self.act(decision)
                    
                    # ═══════════════════════════════════════════════════════
                    
                    # Capturar snapshot del estado
                    snapshot = AgentSnapshot.capture(
                        cycle_number=self._metrics.cycles_executed,
                        telemetry=telemetry,
                        diagnosis=self._last_diagnosis,
                        metrics=self._metrics,
                        last_decision=decision,
                        current_status=status,
                    )
                    self._snapshot_history.add_snapshot(snapshot)
                    
                except Exception as e:
                    logger.error(
                        f"[LIFECYCLE:RUN] ⚠️ Error en ciclo OODA #{self._metrics.cycles_executed}: {e}",
                        exc_info=True,
                    )
                    # Continuar con el siguiente ciclo (fault tolerance)
                
                # Registrar duración del ciclo
                cycle_duration = time.monotonic() - cycle_start
                self._metrics.record_cycle_duration(cycle_duration)
                
                # Sleep adaptativo (compensar duración del ciclo)
                sleep_time = max(
                    0.0,
                    self._timing_config.check_interval - cycle_duration,
                )
                
                if sleep_time > 0 and self._running:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("[LIFECYCLE:RUN] KeyboardInterrupt recibido")
        except Exception as e:
            logger.critical(
                f"[LIFECYCLE:RUN] ❌ Error fatal en bucle principal: {e}",
                exc_info=True,
            )
            raise
        finally:
            # FASE 3: Shutdown
            self._shutdown()
    
    def stop(self) -> None:
        """
        Detiene el agente de forma controlada (setter para self._running).
        
        Efectos:
            - Setea self._running = False
            - El bucle principal terminará tras el ciclo actual
        """
        logger.info("[LIFECYCLE:STOP] Solicitando detención del agente...")
        self._running = False
    
    def _shutdown(self) -> None:
        """
        Limpieza final del agente: cierre de sesión, restauración de señales.
        
        Efectos:
            - Logea métricas finales
            - Cierra sesión HTTP
            - Restaura signal handlers originales
        """
        logger.info("[LIFECYCLE:SHUTDOWN] Iniciando shutdown del AutonomousAgent...")
        
        # Logear métricas finales
        final_metrics = self.get_metrics()
        logger.info(
            f"[LIFECYCLE:SHUTDOWN] Métricas finales: "
            f"cycles={final_metrics['counters']['cycles_executed']}, "
            f"success_rate={final_metrics['rates']['success_rate']:.2%}, "
            f"uptime={final_metrics['timing']['uptime_hours']:.2f}h"
        )
        
        # Cerrar sesión HTTP
        if self._session:
            try:
                self._session.close()
                logger.debug("[LIFECYCLE:SHUTDOWN] Sesión HTTP cerrada")
            except Exception as e:
                logger.warning(f"[LIFECYCLE:SHUTDOWN] Error cerrando sesión HTTP: {e}")
        
        # Restaurar signal handlers
        self._restore_signal_handlers()
        
        logger.info("👋 [LIFECYCLE:SHUTDOWN] AutonomousAgent detenido correctamente")
    
    # =========================================================================
    # API PÚBLICA: OBSERVABILIDAD Y MÉTRICAS
    # =========================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas completas del agente para dashboards/APIs.
        
        Returns:
            Dict con estructura jerárquica conteniendo:
            - Contadores (cycles, observations, decisions)
            - Tasas (success_rate, observation_rate)
            - Topología (betti, health, issues)
            - Persistencia (voltage, saturation)
            - Diagnóstico (último estado, resumen)
        """
        metrics = self._metrics.to_dict()
        
        # Metadata de configuración
        metrics.update({
            "config": {
                "core_api_url": self._conn_config.base_url,
                "check_interval": self._timing_config.check_interval,
                "debounce_window": self._timing_config.debounce_window_seconds,
            },
            "status": {
                "is_running": self._running,
                "last_status": self._last_status.name if self._last_status else None,
            },
        })
        
        # Topología
        topo_health = self.topology.get_topological_health(calculate_b1=True)
        metrics["topology"] = {
            "betti": {
                "b0": topo_health.betti.b0,
                "b1": topo_health.betti.b1,
                "euler_characteristic": topo_health.betti.euler_characteristic,
            },
            "connectivity": {
                "is_connected": topo_health.betti.is_connected,
                "is_ideal": topo_health.betti.is_ideal,
            },
            "health": {
                "score": round(topo_health.health_score, 3),
                "level": topo_health.level.name,
                "is_healthy": topo_health.is_healthy,
            },
            "issues": {
                "disconnected_nodes": list(topo_health.disconnected_nodes),
                "missing_edges": [list(e) for e in topo_health.missing_edges],
                "retry_loops_count": len(topo_health.request_loops),
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
                "recommended_status": self._last_diagnosis.recommended_status.name,
                "severity_level": self._last_diagnosis.severity_level,
                "is_structurally_healthy": self._last_diagnosis.is_structurally_healthy,
                "has_persistent_issues": self._last_diagnosis.has_persistent_issues,
                "metric_states": {
                    "voltage": self._last_diagnosis.voltage_persistence.state.name,
                    "saturation": self._last_diagnosis.saturation_persistence.state.name,
                },
                "critical_issues_count": len(self._last_diagnosis.get_critical_issues()),
            }
        
        # Historial de snapshots
        metrics["snapshot_history"] = {
            "size": self._snapshot_history.size,
            "max_size": self._snapshot_history.max_size,
            "is_full": self._snapshot_history.is_full,
        }
        
        return metrics
    
    def get_topological_summary(self) -> Dict[str, Any]:
        """
        Retorna resumen topológico para dashboards.
        
        Returns:
            Dict con interpretación de salud topológica
        """
        health = self.topology.get_topological_health(calculate_b1=True)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "betti": {
                "values": {"b0": health.betti.b0, "b1": health.betti.b1},
                "interpretation": (
                    "Sistema conectado (β₀=1)"
                    if health.betti.is_connected
                    else f"Sistema fragmentado en {health.betti.b0} componentes"
                ),
                "euler_characteristic": health.betti.euler_characteristic,
            },
            "health": {
                "score": round(health.health_score, 3),
                "level": health.level.name,
                "is_healthy": health.is_healthy,
                "status": "SANO" if health.is_healthy else "DEGRADADO",
            },
            "issues": {
                "disconnected": list(health.disconnected_nodes),
                "missing": [f"{u}↔{v}" for u, v in health.missing_edges],
                "diagnostics": health.diagnostics,
            },
            "patterns": [
                {"id": loop.request_id, "frequency": loop.count}
                for loop in health.request_loops[:5]  # Top 5
            ],
        }
    
    def get_stratum_health(self, stratum: Stratum) -> Dict[str, Any]:
        """
        Retorna la salud filtrada por estrato jerárquico.
        
        Args:
            stratum: Nivel jerárquico a consultar
            
        Returns:
            Dict con métricas específicas del estrato
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
            return {"error": f"Estrato inválido: {stratum}"}
    
    def _get_physics_stratum_health(self) -> Dict[str, Any]:
        """Obtiene salud del estrato PHYSICS: voltaje, saturación."""
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
            "voltage": {
                "value": telemetry.flyback_voltage,
                "classification": v_class,
                "margin": self._thresholds.get_voltage_margin(telemetry.flyback_voltage),
            },
            "saturation": {
                "value": telemetry.saturation,
                "classification": s_class,
                "margin": self._thresholds.get_saturation_margin(telemetry.saturation),
            },
            "status": status,
            "integrity": telemetry.integrity_score,
            "timestamp": telemetry.timestamp.isoformat(),
        }
    
    def _get_tactics_stratum_health(self) -> Dict[str, Any]:
        """Obtiene salud del estrato TACTICS: números de Betti, Euler."""
        health = self.topology.get_topological_health(calculate_b1=True)
        betti = health.betti
        
        return {
            "stratum": "TACTICS",
            "betti_numbers": {
                "b0": betti.b0,
                "b1": betti.b1,
                "is_connected": betti.is_connected,
                "is_ideal": betti.is_ideal,
            },
            "euler_characteristic": betti.euler_characteristic,
            "health": {
                "score": round(health.health_score, 3),
                "level": health.level.name,
                "is_healthy": health.is_healthy,
            },
            "issues": {
                "disconnected_nodes_count": len(health.disconnected_nodes),
                "missing_edges_count": len(health.missing_edges),
            },
        }
    
    def _get_strategy_stratum_health(self) -> Dict[str, Any]:
        """Obtiene salud del estrato STRATEGY: riesgo, confianza."""
        confidence = 0.0
        if self._last_diagnosis:
            confidence = self._last_diagnosis.health.health_score
        
        status_age = 0.0
        if self._last_decision_time:
            status_age = (datetime.now() - self._last_decision_time).total_seconds()
        
        risk_detected = self._last_status in (
            SystemStatus.SATURADO,
            SystemStatus.CRITICO,
            SystemStatus.DISCONNECTED,
        ) if self._last_status else False
        
        return {
            "stratum": "STRATEGY",
            "risk_detected": risk_detected,
            "last_decision": self._last_decision.name if self._last_decision else None,
            "confidence": round(confidence, 3),
            "status_age_seconds": round(status_age, 2),
            "requires_intervention": risk_detected,
        }
    
    def _get_wisdom_stratum_health(self) -> Dict[str, Any]:
        """Obtiene salud del estrato WISDOM: veredicto, racional."""
        rationale = "Sin diagnóstico previo."
        if self._last_diagnosis:
            rationale = self._last_diagnosis.summary
        
        certainty = 1.0 if self._last_diagnosis else 0.0
        
        return {
            "stratum": "WISDOM",
            "verdict": self._last_status.name if self._last_status else "UNKNOWN",
            "certainty": certainty,
            "rationale": rationale,
            "cycles_executed": self._metrics.cycles_executed,
            "uptime_hours": round(self._metrics.uptime_hours, 2),
            "health_trend": self._metrics.health_status,
        }
    
    def get_snapshot_history(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Retorna los últimos N snapshots en formato compacto.
        
        Args:
            count: Número de snapshots a retornar
            
        Returns:
            Lista de dicts con snapshots compactos
        """
        snapshots = self._snapshot_history.get_recent(count)
        return [s.to_compact_dict() for s in snapshots]


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_agent(
    config: Optional[AgentConfig] = None,
    synthesizer: Optional[HamiltonianSynthesizer] = None,
) -> AutonomousAgent:
    """
    Factory function: crea un agente configurado.
    
    Args:
        config: Configuración del agente (default: desde env vars)
        synthesizer: Sintetizador Hamiltoniano (default: estándar)
        
    Returns:
        Agente configurado y listo para run()
    """
    return AutonomousAgent(config=config, synthesizer=synthesizer)


def create_minimal_agent(core_url: str) -> AutonomousAgent:
    """
    Factory function: crea un agente con configuración mínima.
    
    Args:
        core_url: URL del Core API (será validada y normalizada)
        
    Returns:
        Agente configurado con defaults
        
    Raises:
        ConfigurationError: Si core_url es inválida
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
    Punto de entrada principal del módulo.
    
    Returns:
        Código de salida: 0=éxito, 1=error
        
    Uso:
        python -m app.core.apu_agent
    """
    try:
        # Crear agente desde env vars
        agent = AutonomousAgent()
        
        # Ejecutar ciclo OODA (bloqueante)
        agent.run()
        
        return 0
        
    except ConfigurationError as e:
        logger.error(f"❌ Error de configuración: {e}")
        return 1
        
    except KeyboardInterrupt:
        logger.info("⌨️  Interrumpido por el usuario (Ctrl+C)")
        return 0
        
    except Exception as e:
        logger.critical(f"❌ Error no manejado: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
