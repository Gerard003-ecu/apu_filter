"""
Tests Refinados para AutonomousAgent (OODA Loop con Topología Algebraica)
=========================================================================

Suite que valida la coherencia del ciclo OODA mediante:
- Invariantes topológicos (números de Betti, característica de Euler)
- Morfismos entre espacios de estados (Observe → Orient → Decide → Act)
- Homología persistente para distinguir señal de ruido
- Propiedades algebraicas de los builders y fixtures

Estructura Matemática:
- O: Infraestructura → Telemetría (Observación como morfismo)
- R: Telemetría × Topología → Estado (Orientación como producto)
- D: Estado × Contexto → Acción (Decisión como función)
- A: Acción × Diagnóstico → Efectos (Actuación como morfismo)
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from agent.apu_agent import (
    AgentDecision,
    AgentMetrics,
    AutonomousAgent,
    SystemStatus,
    TelemetryData,
    ThresholdConfig,
    TopologicalDiagnosis,
)
from agent.topological_analyzer import (
    BettiNumbers,
    HealthLevel,
    MetricState,
    PersistenceAnalysisResult,
    RequestLoopInfo,
    TopologicalHealth,
)


# =============================================================================
# CONSTANTES PARA DETERMINISMO Y COHERENCIA MATEMÁTICA
# =============================================================================

FIXED_TIMESTAMP = "2024-01-15T10:30:00.000000"
FIXED_DATETIME = datetime(2024, 1, 15, 10, 30, 0)

# Umbrales por defecto del agente (espejados para tests)
DEFAULT_VOLTAGE_WARNING = 0.5
DEFAULT_VOLTAGE_CRITICAL = 0.8
DEFAULT_SATURATION_WARNING = 0.9
DEFAULT_SATURATION_CRITICAL = 0.95

# Constantes topológicas
CONNECTED_BETTI = BettiNumbers(b0=1, b1=0)  # Sistema conectado sin ciclos
FRAGMENTED_BETTI = BettiNumbers(b0=2, b1=0)  # Sistema fragmentado


# =============================================================================
# BUILDERS: Patrones de Creación con Validación Algebraica
# =============================================================================


@dataclass(frozen=True)
class HealthScoreBounds:
    """Bounds inmutables para scores de salud por nivel."""
    low: float
    high: float
    
    def contains(self, score: float) -> bool:
        return self.low <= score <= self.high
    
    def midpoint(self) -> float:
        return (self.low + self.high) / 2


class TopologyBuilder:
    """
    Builder para TopologicalHealth con validación de invariantes algebraicos.

    Invariantes Garantizados:
    - χ (Euler) = b0 - b1 para complejos simpliciales
    - b0 ≥ 1 (al menos un componente conexo)
    - b1 ≥ 0 (ciclos no negativos)
    - Fragmentación implica b0 > 1
    - health_score ∈ [0, 1] y coherente con level
    
    La construcción forma un functor desde la categoría de configuraciones
    hacia la categoría de estados topológicos válidos.
    """

    # Bounds como mapeo HealthLevel → HealthScoreBounds
    LEVEL_BOUNDS: Dict[HealthLevel, HealthScoreBounds] = {
        HealthLevel.HEALTHY: HealthScoreBounds(0.8, 1.0),
        HealthLevel.DEGRADED: HealthScoreBounds(0.5, 0.8),
        HealthLevel.UNHEALTHY: HealthScoreBounds(0.4, 0.7),
        HealthLevel.CRITICAL: HealthScoreBounds(0.0, 0.4),
    }

    def __init__(self) -> None:
        self._b0: int = 1
        self._b1: int = 0
        self._disconnected: FrozenSet[str] = frozenset()
        self._missing_edges: FrozenSet[Tuple[str, str]] = frozenset()
        self._request_loops: Tuple[RequestLoopInfo, ...] = ()
        self._health_score: Optional[float] = None
        self._level: Optional[HealthLevel] = None
        self._diagnostics: Dict[str, Any] = {}

    # =========================================================================
    # Configuración de Números de Betti
    # =========================================================================

    def with_betti(self, b0: int, b1: int = 0) -> "TopologyBuilder":
        """
        Configura números de Betti con validación matemática.

        Args:
            b0: Componentes conectados (β₀ ≥ 1)
            b1: Ciclos independientes (β₁ ≥ 0)

        Raises:
            ValueError: Si los valores violan invariantes topológicos
        """
        if b0 < 1:
            raise ValueError(f"Invariante violado: β₀ ≥ 1, recibido β₀={b0}")
        if b1 < 0:
            raise ValueError(f"Invariante violado: β₁ ≥ 0, recibido β₁={b1}")
        
        self._b0 = b0
        self._b1 = b1
        return self

    def connected(self) -> "TopologyBuilder":
        """Atajo: sistema conectado sin ciclos (β₀=1, β₁=0)."""
        return self.with_betti(1, 0)

    def fragmented(self, components: int = 2) -> "TopologyBuilder":
        """Atajo: sistema fragmentado en n componentes."""
        if components < 2:
            raise ValueError("Fragmentación requiere β₀ ≥ 2")
        return self.with_betti(components, 0)

    def with_cycles(self, count: int) -> "TopologyBuilder":
        """Añade ciclos independientes al sistema."""
        if count < 0:
            raise ValueError(f"Ciclos no pueden ser negativos: {count}")
        self._b1 = count
        return self

    # =========================================================================
    # Configuración de Fragmentación y Conectividad
    # =========================================================================

    def with_disconnected_nodes(self, *nodes: str) -> "TopologyBuilder":
        """
        Marca nodos como desconectados.

        Automáticamente ajusta β₀ para mantener coherencia:
        Si hay n nodos desconectados, β₀ ≥ n + 1.
        """
        self._disconnected = frozenset(nodes)
        
        # Ajuste automático de β₀ para coherencia
        min_b0 = len(self._disconnected) + 1
        if self._b0 < min_b0:
            self._b0 = min_b0
            
        return self

    def with_missing_edges(self, *edges: Tuple[str, str]) -> "TopologyBuilder":
        """Marca edges como faltantes en la topología esperada."""
        self._missing_edges = frozenset(edges)
        return self

    # =========================================================================
    # Configuración de Patrones de Reintentos
    # =========================================================================

    def with_request_loops(self, *loops: RequestLoopInfo) -> "TopologyBuilder":
        """Agrega patrones de bucle detectados en el grafo de requests."""
        self._request_loops = tuple(loops)
        return self

    def with_error_pattern(
        self, 
        error_type: str, 
        count: int, 
        first_seen: int = 0
    ) -> "TopologyBuilder":
        """
        Atajo: crea un patrón de error con formato estándar.
        
        El request_id seguirá el formato "FAIL_{error_type}" usado
        por el agente para clasificar errores.
        """
        loop = RequestLoopInfo(
            request_id=f"FAIL_{error_type.upper()}",
            count=count,
            first_seen=first_seen,
            last_seen=first_seen + count - 1,
        )
        self._request_loops = (*self._request_loops, loop)
        return self

    # =========================================================================
    # Configuración de Salud
    # =========================================================================

    def with_health_score(self, score: float) -> "TopologyBuilder":
        """
        Fija health_score explícitamente.
        
        Se validará coherencia con level en build().
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"health_score ∈ [0, 1], recibido: {score}")
        self._health_score = score
        return self

    def with_level(self, level: HealthLevel) -> "TopologyBuilder":
        """Fija nivel de salud explícitamente."""
        self._level = level
        return self

    def healthy(self) -> "TopologyBuilder":
        """Atajo: estado saludable."""
        return self.with_level(HealthLevel.HEALTHY)

    def degraded(self) -> "TopologyBuilder":
        """Atajo: estado degradado."""
        return self.with_level(HealthLevel.DEGRADED)

    def unhealthy(self) -> "TopologyBuilder":
        """Atajo: estado no saludable."""
        return self.with_level(HealthLevel.UNHEALTHY)

    def critical(self) -> "TopologyBuilder":
        """Atajo: estado crítico."""
        return self.with_level(HealthLevel.CRITICAL)

    def with_diagnostics(self, **kwargs: Any) -> "TopologyBuilder":
        """Añade información de diagnóstico."""
        self._diagnostics.update(kwargs)
        return self

    # =========================================================================
    # Cálculos Internos
    # =========================================================================

    def _calculate_health_score(self) -> float:
        """
        Calcula health_score basado en invariantes topológicos.

        Fórmula derivada de propiedades homológicas:
        score = 1.0 - Σ(penalizaciones)
        
        Penalizaciones:
        - Fragmentación: 0.25 × (β₀ - 1)
        - Ciclos de error: 0.15 × β₁
        - Loops de reintentos: 0.05 × |loops|
        - Nodos desconectados: 0.1 × |disconnected|
        """
        penalties = [
            0.25 * (self._b0 - 1),           # Fragmentación
            0.15 * self._b1,                  # Ciclos
            0.05 * len(self._request_loops),  # Loops de reintentos
            0.10 * len(self._disconnected),   # Nodos desconectados
        ]
        
        score = max(0.0, 1.0 - sum(penalties))
        return round(score, 3)

    def _infer_level(self, score: float) -> HealthLevel:
        """Infiere nivel desde score usando bounds definidos."""
        for level, bounds in self.LEVEL_BOUNDS.items():
            if bounds.contains(score):
                return level
        return HealthLevel.CRITICAL

    def _ensure_coherence(self) -> None:
        """
        Asegura coherencia entre score y level.
        
        Si hay discrepancia, ajusta score al punto medio del nivel.
        """
        if self._level is None or self._health_score is None:
            return
            
        bounds = self.LEVEL_BOUNDS.get(self._level)
        if bounds and not bounds.contains(self._health_score):
            self._health_score = bounds.midpoint()

    # =========================================================================
    # Construcción
    # =========================================================================

    def build(self) -> TopologicalHealth:
        """
        Construye TopologicalHealth validando coherencia interna.

        Garantiza:
        - Números de Betti válidos
        - health_score consistente con topología
        - level consistente con health_score
        """
        # Calcular valores faltantes
        if self._health_score is None:
            self._health_score = self._calculate_health_score()

        if self._level is None:
            self._level = self._infer_level(self._health_score)

        # Asegurar coherencia
        self._ensure_coherence()

        return TopologicalHealth(
            betti=BettiNumbers(b0=self._b0, b1=self._b1),
            disconnected_nodes=self._disconnected,
            missing_edges=self._missing_edges,
            request_loops=self._request_loops,
            health_score=self._health_score,
            level=self._level,
            diagnostics=self._diagnostics,
        )

    # =========================================================================
    # Factory Methods para Casos Comunes
    # =========================================================================

    @classmethod
    def nominal(cls) -> "TopologyBuilder":
        """Factory: estado nominal (conectado, saludable)."""
        return cls().connected().healthy()

    @classmethod
    def disconnected_from_core(cls) -> "TopologyBuilder":
        """Factory: agente desconectado del Core."""
        return (
            cls()
            .fragmented(2)
            .with_disconnected_nodes("Core")
            .with_missing_edges(("Agent", "Core"))
            .critical()
        )

    @classmethod
    def with_backend_issues(cls) -> "TopologyBuilder":
        """Factory: problemas de backend (Redis/Filesystem)."""
        return (
            cls()
            .fragmented(3)
            .with_disconnected_nodes("Redis", "Filesystem")
            .unhealthy()
        )


class PersistenceBuilder:
    """
    Builder para PersistenceAnalysisResult con semántica topológica.

    Estados del diagrama de persistencia:
    - STABLE: Sin excursiones significativas (punto fijo)
    - NOISE: Fluctuaciones efímeras (homología trivial)
    - FEATURE: Característica persistente (señal real)
    - CRITICAL: Característica de alta persistencia y severidad

    La persistencia de una característica (death - birth) indica
    su significancia topológica: mayor persistencia = más real.
    """

    def __init__(self) -> None:
        self._state: MetricState = MetricState.STABLE
        self._intervals: Tuple[Tuple[float, float], ...] = ()
        self._feature_count: int = 0
        self._noise_count: int = 0
        self._active_count: int = 0
        self._max_lifespan: float = 0.0
        self._total_persistence: float = 0.0
        self._metadata: Dict[str, Any] = {}

    # =========================================================================
    # Estados Principales
    # =========================================================================

    def stable(self) -> "PersistenceBuilder":
        """
        Estado estable: sin características significativas.
        
        Representa un sistema en equilibrio donde las métricas
        permanecen dentro de umbrales normales.
        """
        self._state = MetricState.STABLE
        self._feature_count = 0
        self._noise_count = 0
        self._max_lifespan = 0.0
        self._total_persistence = 0.0
        return self

    def with_features(
        self, 
        count: int, 
        max_lifespan: float = 10.0,
        avg_persistence: Optional[float] = None
    ) -> "PersistenceBuilder":
        """
        Características topológicas persistentes detectadas.

        Indica patrones que sobreviven múltiples escalas de filtración,
        representando señales reales en los datos.

        Args:
            count: Número de características detectadas
            max_lifespan: Vida máxima de una característica
            avg_persistence: Persistencia promedio (default: max_lifespan/2)
        """
        self._state = MetricState.FEATURE
        self._feature_count = count
        self._max_lifespan = max_lifespan
        
        avg = avg_persistence if avg_persistence is not None else max_lifespan / 2
        self._total_persistence = count * avg
        
        return self

    def with_noise(
        self, 
        count: int, 
        max_lifespan: float = 2.0
    ) -> "PersistenceBuilder":
        """
        Ruido topológico: características efímeras.

        Corta vida en el diagrama de persistencia indica
        fluctuaciones que deben ignorarse (inmunidad al ruido).

        Args:
            count: Número de excursiones ruidosas
            max_lifespan: Vida máxima (típicamente < threshold_ratio)
        """
        self._state = MetricState.NOISE
        self._noise_count = count
        self._feature_count = 0
        self._max_lifespan = max_lifespan
        self._total_persistence = count * max_lifespan
        return self

    def critical(
        self, 
        active_duration: int = 50,
        lifespan: float = 100.0
    ) -> "PersistenceBuilder":
        """
        Estado crítico con alta persistencia.

        Representa una característica que ha persistido
        más allá del umbral crítico.

        Args:
            active_duration: Duración activa en muestras
            lifespan: Vida total de la característica
        """
        self._state = MetricState.CRITICAL
        self._feature_count = 1
        self._active_count = 1
        self._max_lifespan = lifespan
        self._total_persistence = lifespan
        self._metadata["active_duration"] = active_duration
        return self

    # =========================================================================
    # Configuración Adicional
    # =========================================================================

    def with_intervals(
        self, 
        *intervals: Tuple[float, float]
    ) -> "PersistenceBuilder":
        """
        Añade intervalos de persistencia explícitos.

        Cada intervalo (birth, death) representa una característica
        en el diagrama de persistencia.
        """
        self._intervals = intervals
        return self

    def with_metadata(self, **kwargs: Any) -> "PersistenceBuilder":
        """Añade metadatos al resultado."""
        self._metadata.update(kwargs)
        return self

    def with_active_excursion(self, duration: int) -> "PersistenceBuilder":
        """Marca una excursión actualmente activa."""
        self._active_count = 1
        self._metadata["active_duration"] = duration
        return self

    # =========================================================================
    # Construcción
    # =========================================================================

    def build(self) -> PersistenceAnalysisResult:
        """Construye el resultado del análisis de persistencia."""
        return PersistenceAnalysisResult(
            state=self._state,
            intervals=self._intervals,
            feature_count=self._feature_count,
            noise_count=self._noise_count,
            active_count=self._active_count,
            max_lifespan=self._max_lifespan,
            total_persistence=self._total_persistence,
            metadata=self._metadata,
        )

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def nominal(cls) -> "PersistenceBuilder":
        """Factory: métricas estables."""
        return cls().stable()

    @classmethod
    def voltage_spike(cls, persistent: bool = True) -> "PersistenceBuilder":
        """Factory: pico de voltaje."""
        if persistent:
            return cls().with_features(count=1, max_lifespan=15.0)
        return cls().with_noise(count=1, max_lifespan=1.5)

    @classmethod
    def saturation_warning(cls) -> "PersistenceBuilder":
        """Factory: saturación en zona de advertencia."""
        return cls().with_features(count=2, max_lifespan=8.0)


class TelemetryBuilder:
    """
    Builder para TelemetryData con validación de rango.

    Asegura que los valores permanezcan en el compacto [0, 1]²
    del espacio de métricas.
    """

    def __init__(self) -> None:
        self._voltage: float = 0.3
        self._saturation: float = 0.3
        self._timestamp: datetime = FIXED_DATETIME
        self._raw_data: Dict[str, Any] = {}

    def with_voltage(self, voltage: float) -> "TelemetryBuilder":
        """Configura voltaje (se clampea a [0, 1])."""
        self._voltage = max(0.0, min(1.0, voltage))
        return self

    def with_saturation(self, saturation: float) -> "TelemetryBuilder":
        """Configura saturación (se clampea a [0, 1])."""
        self._saturation = max(0.0, min(1.0, saturation))
        return self

    def with_timestamp(self, ts: datetime) -> "TelemetryBuilder":
        """Configura timestamp."""
        self._timestamp = ts
        return self

    def with_raw_data(self, **kwargs: Any) -> "TelemetryBuilder":
        """Añade datos crudos."""
        self._raw_data.update(kwargs)
        return self

    def build(self) -> TelemetryData:
        """Construye TelemetryData."""
        return TelemetryData(
            flyback_voltage=self._voltage,
            saturation=self._saturation,
            timestamp=self._timestamp,
            raw_data=self._raw_data,
        )

    # Factory Methods
    @classmethod
    def nominal(cls) -> "TelemetryBuilder":
        """Factory: telemetría nominal."""
        return cls().with_voltage(0.3).with_saturation(0.3)

    @classmethod
    def voltage_warning(cls) -> "TelemetryBuilder":
        """Factory: voltaje en zona de advertencia."""
        return cls().with_voltage(0.6).with_saturation(0.3)

    @classmethod
    def voltage_critical(cls) -> "TelemetryBuilder":
        """Factory: voltaje crítico."""
        return cls().with_voltage(0.9).with_saturation(0.3)

    @classmethod
    def saturation_critical(cls) -> "TelemetryBuilder":
        """Factory: saturación crítica."""
        return cls().with_voltage(0.3).with_saturation(0.98)

    @classmethod
    def all_critical(cls) -> "TelemetryBuilder":
        """Factory: todas las métricas críticas."""
        return cls().with_voltage(0.9).with_saturation(0.98)


class DiagnosisBuilder:
    """Builder para TopologicalDiagnosis."""

    def __init__(self) -> None:
        self._health: Optional[TopologicalHealth] = None
        self._voltage_persistence: Optional[PersistenceAnalysisResult] = None
        self._saturation_persistence: Optional[PersistenceAnalysisResult] = None
        self._summary: str = "Sin diagnóstico"
        self._status: SystemStatus = SystemStatus.NOMINAL

    def with_health(self, health: TopologicalHealth) -> "DiagnosisBuilder":
        self._health = health
        return self

    def with_voltage(
        self, 
        persistence: PersistenceAnalysisResult
    ) -> "DiagnosisBuilder":
        self._voltage_persistence = persistence
        return self

    def with_saturation(
        self, 
        persistence: PersistenceAnalysisResult
    ) -> "DiagnosisBuilder":
        self._saturation_persistence = persistence
        return self

    def with_summary(self, summary: str) -> "DiagnosisBuilder":
        self._summary = summary
        return self

    def with_status(self, status: SystemStatus) -> "DiagnosisBuilder":
        self._status = status
        return self

    def build(self) -> TopologicalDiagnosis:
        """Construye diagnóstico con defaults para campos faltantes."""
        return TopologicalDiagnosis(
            health=self._health or TopologyBuilder.nominal().build(),
            voltage_persistence=(
                self._voltage_persistence or PersistenceBuilder.nominal().build()
            ),
            saturation_persistence=(
                self._saturation_persistence or PersistenceBuilder.nominal().build()
            ),
            summary=self._summary,
            recommended_status=self._status,
        )


# =============================================================================
# RESPONSE FACTORY: Creación de respuestas HTTP mock
# =============================================================================


class ResponseFactory:
    """
    Factory para crear respuestas HTTP mock con estructura consistente.
    
    Encapsula la creación de mocks de requests.Response con
    configuración determinista.
    """

    @staticmethod
    def success(data: Dict[str, Any], status_code: int = 200) -> Mock:
        """Crea respuesta exitosa."""
        mock = Mock()
        mock.ok = True
        mock.status_code = status_code
        mock.json.return_value = data
        mock.text = str(data)
        mock.headers = {"Content-Type": "application/json"}
        return mock

    @staticmethod
    def error(
        status_code: int, 
        message: str = "Error"
    ) -> Mock:
        """Crea respuesta de error."""
        mock = Mock()
        mock.ok = False
        mock.status_code = status_code
        mock.text = message
        mock.json.side_effect = ValueError("No JSON in error response")
        mock.headers = {"Content-Type": "text/plain"}
        return mock

    @staticmethod
    def nominal_telemetry() -> Mock:
        """Crea respuesta con telemetría nominal."""
        return ResponseFactory.success({
            "flyback_voltage": 0.3,
            "saturation": 0.5,
            "timestamp": FIXED_TIMESTAMP,
            "redis_connected": True,
            "filesystem_accessible": True,
        })

    @staticmethod
    def partial_connectivity(
        redis: bool = True, 
        filesystem: bool = True
    ) -> Mock:
        """Crea respuesta con conectividad parcial."""
        return ResponseFactory.success({
            "flyback_voltage": 0.3,
            "saturation": 0.5,
            "timestamp": FIXED_TIMESTAMP,
            "redis_connected": redis,
            "filesystem_accessible": filesystem,
        })


# =============================================================================
# FIXTURES BASE
# =============================================================================


class TestFixtureBase:
    """
    Clase base con fixtures robustas y correctamente desacopladas.
    
    Proporciona:
    - Limpieza de entorno
    - Mocks inyectables
    - Agente configurado para testing
    """

    @pytest.fixture
    def clean_env(self, monkeypatch):
        """Limpia variables de entorno de forma exhaustiva."""
        env_vars = [
            "CORE_API_URL",
            "CHECK_INTERVAL",
            "REQUEST_TIMEOUT",
            "LOG_LEVEL",
            "PERSISTENCE_WINDOW_SIZE",
        ]
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)
        yield

    @pytest.fixture
    def topology_mock(self) -> MagicMock:
        """
        Mock de SystemTopology con comportamiento por defecto saludable.
        
        Configura todos los métodos necesarios para el ciclo OODA.
        """
        with patch("agent.apu_agent.SystemTopology") as MockClass:
            instance = MagicMock()
            MockClass.return_value = instance

            # Estado por defecto: topología nominal
            instance.get_topological_health.return_value = (
                TopologyBuilder.nominal().build()
            )
            instance.update_connectivity.return_value = (3, [])
            instance.record_request.return_value = None
            instance.remove_edge.return_value = None
            instance.add_edge.return_value = None
            instance.clear_request_history.return_value = None

            yield instance

    @pytest.fixture
    def persistence_mock(self) -> MagicMock:
        """
        Mock de PersistenceHomology con comportamiento estable.
        """
        with patch("agent.apu_agent.PersistenceHomology") as MockClass:
            instance = MagicMock()
            MockClass.return_value = instance

            # Estado por defecto: métricas estables
            instance.analyze_persistence.return_value = (
                PersistenceBuilder.nominal().build()
            )
            instance.add_reading.return_value = None
            instance.get_statistics.return_value = {
                "count": 10,
                "mean": 0.3,
                "std": 0.05,
                "min": 0.2,
                "max": 0.4,
            }

            yield instance

    @pytest.fixture
    def session_mock(self) -> MagicMock:
        """Mock de sesión HTTP."""
        session = MagicMock()
        session.get.return_value = ResponseFactory.nominal_telemetry()
        session.post.return_value = ResponseFactory.success({"status": "ok"})
        return session

    @pytest.fixture
    def agent(
        self, 
        clean_env, 
        topology_mock, 
        persistence_mock, 
        session_mock
    ) -> AutonomousAgent:
        """
        Agente con todos los mocks inyectados.
        
        Garantiza aislamiento completo de dependencias externas.
        """
        with patch.object(AutonomousAgent, "_setup_signal_handlers"):
            agent = AutonomousAgent(
                core_api_url="http://test-core:5000",
                check_interval=1,
                request_timeout=1,
            )
            
            # Inyección explícita de mocks
            agent.topology = topology_mock
            agent.persistence = persistence_mock
            agent._session = session_mock
            
            yield agent
            
            # Cleanup
            if hasattr(agent, "_session") and agent._session:
                try:
                    agent._session.close()
                except Exception:
                    pass


# =============================================================================
# TESTS: FASE OBSERVE
# =============================================================================


class TestObserve(TestFixtureBase):
    """
    Tests para la fase OBSERVE del ciclo OODA.

    Valida el morfismo O: Infraestructura → Telemetría:
    - Generación de IDs únicos para trazabilidad
    - Registro en grafo de dependencias
    - Actualización de conectividad según respuesta
    - Degradación progresiva por fallos
    - Recuperación automática tras éxito
    """

    def test_observe_generates_unique_request_ids(self, agent):
        """Cada observación genera ID único para tracking topológico."""
        agent._session.get.return_value = ResponseFactory.nominal_telemetry()

        agent.observe()
        first_id = agent.topology.record_request.call_args[0][0]

        agent.observe()
        second_id = agent.topology.record_request.call_args[0][0]

        assert first_id.startswith("obs_"), "ID debe tener prefijo 'obs_'"
        assert second_id.startswith("obs_")
        assert first_id != second_id, "IDs deben ser únicos"

    def test_observe_success_returns_telemetry_data(self, agent):
        """Observación exitosa retorna TelemetryData válida."""
        agent._session.get.return_value = ResponseFactory.nominal_telemetry()

        result = agent.observe()

        assert result is not None
        assert isinstance(result, TelemetryData)
        assert 0 <= result.flyback_voltage <= 1
        assert 0 <= result.saturation <= 1

    def test_observe_updates_connectivity_on_success(self, agent):
        """Éxito actualiza grafo con edges activos."""
        agent._session.get.return_value = ResponseFactory.nominal_telemetry()

        agent.observe()

        agent.topology.update_connectivity.assert_called()
        edges = agent.topology.update_connectivity.call_args[0][0]
        
        assert ("Agent", "Core") in edges
        assert ("Core", "Redis") in edges
        assert ("Core", "Filesystem") in edges

    def test_observe_reflects_partial_connectivity(self, agent):
        """Conectividad parcial se refleja en grafo."""
        agent._session.get.return_value = ResponseFactory.partial_connectivity(
            redis=False, filesystem=True
        )

        agent.observe()

        edges = agent.topology.update_connectivity.call_args[0][0]
        
        assert ("Agent", "Core") in edges
        assert ("Core", "Redis") not in edges
        assert ("Core", "Filesystem") in edges

    def test_observe_clears_history_on_success(self, agent):
        """Éxito limpia historial de requests (reset de loops)."""
        agent._session.get.return_value = ResponseFactory.nominal_telemetry()

        agent.observe()

        agent.topology.clear_request_history.assert_called()

    def test_observe_records_typed_failure(self, agent):
        """Fallos se registran con tipo específico."""
        failure_cases = [
            (requests.exceptions.Timeout(), "TIMEOUT"),
            (requests.exceptions.ConnectionError(), "CONNECTION"),
        ]

        for exception, expected_type in failure_cases:
            agent._session.get.side_effect = exception
            agent.topology.record_request.reset_mock()

            agent.observe()

            request_id = agent.topology.record_request.call_args[0][0]
            assert "FAIL" in request_id.upper()
            assert expected_type in request_id.upper()

    def test_observe_increments_failure_counter(self, agent):
        """Fallos incrementan contador de métricas."""
        agent._session.get.side_effect = requests.exceptions.Timeout()
        initial_failures = agent._metrics.failed_observations

        agent.observe()

        assert agent._metrics.failed_observations == initial_failures + 1
        assert agent._metrics.consecutive_failures >= 1

    def test_observe_degrades_topology_on_consecutive_failures(self, agent):
        """Fallos consecutivos degradan topología."""
        agent._session.get.side_effect = requests.exceptions.Timeout()
        max_failures = agent.MAX_CONSECUTIVE_FAILURES

        for _ in range(max_failures):
            agent.observe()

        agent.topology.remove_edge.assert_called_with("Agent", "Core")

    def test_observe_resets_consecutive_failures_on_success(self, agent):
        """Éxito resetea contador de fallos consecutivos."""
        # Simular fallos previos
        agent._session.get.side_effect = requests.exceptions.Timeout()
        agent.observe()
        agent.observe()
        
        assert agent._metrics.consecutive_failures >= 2

        # Éxito
        agent._session.get.side_effect = None
        agent._session.get.return_value = ResponseFactory.nominal_telemetry()
        agent.observe()

        assert agent._metrics.consecutive_failures == 0

    def test_observe_handles_invalid_json(self, agent):
        """JSON inválido retorna None y registra fallo."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        agent._session.get.return_value = mock_response

        result = agent.observe()

        assert result is None
        request_id = agent.topology.record_request.call_args[0][0]
        assert "INVALID_JSON" in request_id.upper()

    def test_observe_handles_http_error(self, agent):
        """Error HTTP retorna None y registra código."""
        agent._session.get.return_value = ResponseFactory.error(503, "Service Unavailable")

        result = agent.observe()

        assert result is None
        request_id = agent.topology.record_request.call_args[0][0]
        assert "503" in request_id


# =============================================================================
# TESTS: FASE ORIENT
# =============================================================================


class TestOrient(TestFixtureBase):
    """
    Tests para la fase ORIENT del ciclo OODA.

    Valida el morfismo R: Telemetría × Topología → Estado
    siguiendo la jerarquía de prioridades:

    P1: Fragmentación (β₀ > 1) → DISCONNECTED
    P2: Safety Net (crítico instantáneo) → CRITICO
    P3: Salud Topológica Crítica → CRITICO
    P4: Persistencia Crítica → SATURADO/INESTABLE
    P5: Patrones de Reintentos → INESTABLE
    P6: Salud Degradada → INESTABLE
    P7: Ruido → Filtrar (inmunidad)
    P8: Nominal → NOMINAL
    """

    # =========================================================================
    # P1: Fragmentación Topológica
    # =========================================================================

    def test_p1_fragmentation_overrides_all(self, agent):
        """P1: β₀ > 1 → DISCONNECTED (máxima prioridad)."""
        fragmented = (
            TopologyBuilder()
            .fragmented(2)
            .with_disconnected_nodes("Redis")
            .critical()
            .build()
        )
        agent.topology.get_topological_health.return_value = fragmented

        # Incluso telemetría perfecta no previene DISCONNECTED
        telemetry = TelemetryBuilder.nominal().build()
        status = agent.orient(telemetry)

        assert status == SystemStatus.DISCONNECTED
        assert agent._last_diagnosis is not None
        assert "Fragmentación" in agent._last_diagnosis.summary

    def test_p1_multiple_fragments_detected(self, agent):
        """Múltiples fragmentos correctamente reportados."""
        fragmented = (
            TopologyBuilder()
            .fragmented(3)
            .with_disconnected_nodes("Redis", "Filesystem")
            .build()
        )
        agent.topology.get_topological_health.return_value = fragmented

        status = agent.orient(TelemetryBuilder.nominal().build())

        assert status == SystemStatus.DISCONNECTED
        assert "β₀=3" in agent._last_diagnosis.summary

    # =========================================================================
    # P2: Safety Net (Umbrales Críticos Instantáneos)
    # =========================================================================

    def test_p2_voltage_critical_triggers_safety_net(self, agent):
        """P2: Voltaje > crítico → CRITICO inmediato."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        telemetry = TelemetryBuilder.voltage_critical().build()
        status = agent.orient(telemetry)

        assert status == SystemStatus.CRITICO
        assert "Voltaje" in agent._last_diagnosis.summary or \
               "crítico" in agent._last_diagnosis.summary.lower()

    def test_p2_saturation_critical_triggers_safety_net(self, agent):
        """P2: Saturación > crítico → CRITICO inmediato."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        telemetry = TelemetryBuilder.saturation_critical().build()
        status = agent.orient(telemetry)

        assert status == SystemStatus.CRITICO

    def test_p2_safety_net_precedence_over_topology_health(self, agent):
        """Safety net actúa incluso con topología saludable."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().healthy().build()
        )

        telemetry = TelemetryBuilder.all_critical().build()
        status = agent.orient(telemetry)

        assert status == SystemStatus.CRITICO

    # =========================================================================
    # P3: Salud Topológica Crítica
    # =========================================================================

    def test_p3_topology_critical_health_detected(self, agent):
        """P3: Salud topológica crítica → CRITICO."""
        critical_health = (
            TopologyBuilder()
            .connected()  # No fragmentado
            .with_health_score(0.2)
            .critical()
            .build()
        )
        agent.topology.get_topological_health.return_value = critical_health

        telemetry = TelemetryBuilder.nominal().build()
        status = agent.orient(telemetry)

        assert status == SystemStatus.CRITICO

    # =========================================================================
    # P4: Persistencia de Métricas
    # =========================================================================

    def test_p4_saturation_persistence_critical(self, agent):
        """P4: Saturación persistente crítica → SATURADO."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        def persistence_handler(metric: str, **kwargs):
            if metric == "saturation":
                return PersistenceBuilder().critical(active_duration=30).build()
            return PersistenceBuilder.nominal().build()

        agent.persistence.analyze_persistence.side_effect = persistence_handler

        telemetry = TelemetryBuilder.nominal().build()
        status = agent.orient(telemetry)

        assert status == SystemStatus.SATURADO

    def test_p4_saturation_feature_detected(self, agent):
        """P4: Característica de saturación → SATURADO."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        def persistence_handler(metric: str, **kwargs):
            if metric == "saturation":
                return PersistenceBuilder().with_features(2, max_lifespan=12).build()
            return PersistenceBuilder.nominal().build()

        agent.persistence.analyze_persistence.side_effect = persistence_handler

        status = agent.orient(TelemetryBuilder.nominal().build())

        assert status == SystemStatus.SATURADO

    def test_p4_voltage_persistence_critical(self, agent):
        """P4: Voltaje persistente crítico → INESTABLE."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        def persistence_handler(metric: str, **kwargs):
            if metric == "flyback_voltage":
                return PersistenceBuilder().critical(active_duration=25).build()
            return PersistenceBuilder.nominal().build()

        agent.persistence.analyze_persistence.side_effect = persistence_handler

        status = agent.orient(TelemetryBuilder.nominal().build())

        assert status == SystemStatus.INESTABLE

    def test_p4_voltage_feature_detected(self, agent):
        """P4: Característica de voltaje → INESTABLE."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        def persistence_handler(metric: str, **kwargs):
            if metric == "flyback_voltage":
                return PersistenceBuilder().with_features(1, max_lifespan=15).build()
            return PersistenceBuilder.nominal().build()

        agent.persistence.analyze_persistence.side_effect = persistence_handler

        status = agent.orient(TelemetryBuilder.nominal().build())

        assert status == SystemStatus.INESTABLE

    # =========================================================================
    # P5: Patrones de Reintentos
    # =========================================================================

    def test_p5_error_loops_trigger_instability(self, agent):
        """P5: Patrones de error repetitivos → INESTABLE."""
        health_with_loops = (
            TopologyBuilder()
            .nominal()
            .with_error_pattern("TIMEOUT", count=6)
            .degraded()
            .build()
        )
        agent.topology.get_topological_health.return_value = health_with_loops

        status = agent.orient(TelemetryBuilder.nominal().build())

        assert status == SystemStatus.INESTABLE
        assert "Patrón" in agent._last_diagnosis.summary or \
               "Reintentos" in agent._last_diagnosis.summary

    def test_p5_non_error_loops_less_severe(self, agent):
        """Loops que no son errores no disparan INESTABLE por sí solos."""
        # Loops de requests normales (no FAIL_*)
        normal_loop = RequestLoopInfo(
            request_id="obs_abc123",
            count=10,
            first_seen=0,
            last_seen=9,
        )
        health = (
            TopologyBuilder()
            .nominal()
            .with_request_loops(normal_loop)
            .healthy()
            .build()
        )
        agent.topology.get_topological_health.return_value = health

        status = agent.orient(TelemetryBuilder.nominal().build())

        # No debería ser INESTABLE porque no son errores
        assert status == SystemStatus.NOMINAL

    # =========================================================================
    # P6: Salud Degradada
    # =========================================================================

    def test_p6_unhealthy_topology_triggers_instability(self, agent):
        """P6: Topología UNHEALTHY → INESTABLE."""
        unhealthy = TopologyBuilder().connected().unhealthy().build()
        agent.topology.get_topological_health.return_value = unhealthy

        status = agent.orient(TelemetryBuilder.nominal().build())

        assert status == SystemStatus.INESTABLE

    # =========================================================================
    # P7: Inmunidad al Ruido
    # =========================================================================

    def test_p7_noise_immunity_voltage(self, agent):
        """P7: Ruido en voltaje se filtra → NOMINAL."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        def persistence_handler(metric: str, **kwargs):
            if metric == "flyback_voltage":
                return PersistenceBuilder().with_noise(count=5, max_lifespan=1.0).build()
            return PersistenceBuilder.nominal().build()

        agent.persistence.analyze_persistence.side_effect = persistence_handler

        # Voltaje elevado pero clasificado como ruido
        telemetry = TelemetryBuilder.voltage_warning().build()
        status = agent.orient(telemetry)

        assert status == SystemStatus.NOMINAL

    def test_p7_noise_immunity_saturation(self, agent):
        """P7: Ruido en saturación se filtra → NOMINAL."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        agent.persistence.analyze_persistence.return_value = (
            PersistenceBuilder().with_noise(count=3, max_lifespan=0.5).build()
        )

        status = agent.orient(TelemetryBuilder.nominal().build())

        assert status == SystemStatus.NOMINAL

    # =========================================================================
    # P8: Estado Nominal
    # =========================================================================

    def test_p8_nominal_state(self, agent):
        """P8: Sin anomalías → NOMINAL."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )
        agent.persistence.analyze_persistence.return_value = (
            PersistenceBuilder.nominal().build()
        )

        telemetry = TelemetryBuilder.nominal().build()
        status = agent.orient(telemetry)

        assert status == SystemStatus.NOMINAL
        assert agent._last_diagnosis is not None

    def test_orient_without_telemetry_returns_unknown(self, agent):
        """Sin telemetría pero conectado → UNKNOWN."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        status = agent.orient(None)

        assert status == SystemStatus.UNKNOWN

    def test_orient_stores_diagnosis(self, agent):
        """Orient siempre almacena diagnóstico."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        agent.orient(TelemetryBuilder.nominal().build())

        assert agent._last_diagnosis is not None
        assert isinstance(agent._last_diagnosis, TopologicalDiagnosis)
        assert agent._last_diagnosis.health is not None


# =============================================================================
# TESTS: FASE DECIDE
# =============================================================================


class TestDecide(TestFixtureBase):
    """
    Tests para la fase DECIDE del ciclo OODA.

    Valida el morfismo D: Estado × Contexto → Acción
    mediante la matriz de decisión:

    DISCONNECTED → RECONNECT
    CRITICO      → ALERTA_CRITICA
    SATURADO     → AJUSTAR_VELOCIDAD
    INESTABLE    → EJECUTAR_LIMPIEZA
    UNKNOWN      → WAIT
    NOMINAL      → HEARTBEAT
    """

    @pytest.mark.parametrize(
        "status,expected_decision",
        [
            (SystemStatus.DISCONNECTED, AgentDecision.RECONNECT),
            (SystemStatus.CRITICO, AgentDecision.ALERTA_CRITICA),
            (SystemStatus.SATURADO, AgentDecision.AJUSTAR_VELOCIDAD),
            (SystemStatus.INESTABLE, AgentDecision.EJECUTAR_LIMPIEZA),
            (SystemStatus.UNKNOWN, AgentDecision.WAIT),
            (SystemStatus.NOMINAL, AgentDecision.HEARTBEAT),
        ],
    )
    def test_decision_matrix(
        self, 
        agent, 
        status: SystemStatus, 
        expected_decision: AgentDecision
    ):
        """Valida matriz de decisión completa."""
        decision = agent.decide(status)
        assert decision == expected_decision

    def test_decide_is_pure_function(self, agent):
        """DECIDE es función pura: mismo input → mismo output."""
        status = SystemStatus.INESTABLE

        decisions = [agent.decide(status) for _ in range(3)]

        assert all(d == decisions[0] for d in decisions)

    def test_decide_records_metrics(self, agent):
        """DECIDE registra decisión en métricas."""
        initial_count = agent._metrics.decisions_count.get("HEARTBEAT", 0)

        agent.decide(SystemStatus.NOMINAL)

        assert agent._metrics.decisions_count.get("HEARTBEAT", 0) == initial_count + 1

    def test_decide_updates_last_status(self, agent):
        """DECIDE actualiza último status."""
        agent.decide(SystemStatus.SATURADO)

        assert agent._last_status == SystemStatus.SATURADO

    def test_decide_with_error_loops_in_nominal(self, agent):
        """
        NOMINAL con loops de error en historial no cambia decisión.
        
        El contexto topológico puede loguear advertencias pero
        no modifica la decisión base.
        """
        # Configurar diagnóstico con loops de error
        loops = (
            RequestLoopInfo("FAIL_TIMEOUT", 3, 0, 2),
        )
        health = TopologyBuilder.nominal().with_request_loops(*loops).build()
        agent._last_diagnosis = (
            DiagnosisBuilder()
            .with_health(health)
            .with_status(SystemStatus.NOMINAL)
            .build()
        )

        decision = agent.decide(SystemStatus.NOMINAL)

        assert decision == AgentDecision.HEARTBEAT


# =============================================================================
# TESTS: FASE ACT
# =============================================================================


class TestAct(TestFixtureBase):
    """
    Tests para la fase ACT del ciclo OODA.

    Valida el morfismo A: Decisión × Diagnóstico → Efectos:
    - Ejecución de handlers específicos
    - Sistema de debounce
    - Proyección de intenciones
    - Logging estructurado
    """

    # =========================================================================
    # Ejecución de Handlers
    # =========================================================================

    def test_act_heartbeat_executes(self, agent):
        """HEARTBEAT ejecuta sin error."""
        result = agent.act(AgentDecision.HEARTBEAT)
        assert result is True

    def test_act_returns_boolean(self, agent):
        """ACT siempre retorna booleano."""
        for decision in AgentDecision:
            # Reset debounce para cada decisión
            agent._last_decision = None
            agent._last_decision_time = None
            
            result = agent.act(decision)
            assert isinstance(result, bool)

    def test_act_ejecutar_limpieza_projects_intent(self, agent):
        """EJECUTAR_LIMPIEZA proyecta vector 'clean'."""
        agent._session.post.return_value = ResponseFactory.success({"ok": True})

        agent.act(AgentDecision.EJECUTAR_LIMPIEZA)

        agent._session.post.assert_called_once()
        args, kwargs = agent._session.post.call_args
        
        assert "clean" in args[0]
        assert kwargs["json"]["vector"] == "clean"
        assert kwargs["json"]["stratum"] == "PHYSICS"

    def test_act_ajustar_velocidad_projects_backpressure(self, agent):
        """AJUSTAR_VELOCIDAD proyecta configuración de backpressure."""
        agent._session.post.return_value = ResponseFactory.success({"ok": True})
        # Reset debounce
        agent._last_decision = None

        agent.act(AgentDecision.AJUSTAR_VELOCIDAD)

        agent._session.post.assert_called_once()
        json_body = agent._session.post.call_args.kwargs["json"]
        
        assert json_body["vector"] == "configure"
        assert json_body["payload"]["action"] == "decrease"

    def test_act_reconnect_reinitializes_topology(self, agent):
        """RECONNECT reinicializa topología esperada."""
        agent.act(AgentDecision.RECONNECT)

        # Debe llamar a update_connectivity para restaurar edges
        agent.topology.update_connectivity.assert_called()

    # =========================================================================
    # Sistema de Debounce
    # =========================================================================

    def test_act_debounce_suppresses_repeated_actions(self, agent):
        """Decisiones repetidas inmediatas se suprimen."""
        agent._session.post.return_value = ResponseFactory.success({})

        result1 = agent.act(AgentDecision.EJECUTAR_LIMPIEZA)
        result2 = agent.act(AgentDecision.EJECUTAR_LIMPIEZA)

        assert result1 is True
        assert result2 is False

    def test_act_critical_bypasses_debounce(self, agent):
        """ALERTA_CRITICA nunca se suprime."""
        results = [agent.act(AgentDecision.ALERTA_CRITICA) for _ in range(5)]

        assert all(results), "Todas las alertas críticas deben ejecutarse"

    def test_act_reconnect_bypasses_debounce(self, agent):
        """RECONNECT nunca se suprime."""
        results = [agent.act(AgentDecision.RECONNECT) for _ in range(3)]

        assert all(results)

    def test_act_different_decisions_not_suppressed(self, agent):
        """Decisiones diferentes no se suprimen entre sí."""
        agent._session.post.return_value = ResponseFactory.success({})

        result1 = agent.act(AgentDecision.EJECUTAR_LIMPIEZA)
        result2 = agent.act(AgentDecision.AJUSTAR_VELOCIDAD)

        assert result1 is True
        assert result2 is True

    def test_act_updates_decision_timestamp(self, agent):
        """ACT actualiza timestamp de última decisión."""
        before = agent._last_decision_time

        agent.act(AgentDecision.HEARTBEAT)

        assert agent._last_decision_time is not None
        if before is not None:
            assert agent._last_decision_time >= before

    # =========================================================================
    # Logging y Diagnóstico
    # =========================================================================

    def test_act_logs_diagnosis_summary(self, agent, caplog):
        """El log incluye resumen del diagnóstico."""
        agent._last_diagnosis = (
            DiagnosisBuilder()
            .with_summary("Voltaje elevado persistente")
            .with_status(SystemStatus.INESTABLE)
            .build()
        )
        agent._session.post.return_value = ResponseFactory.success({})

        with caplog.at_level(logging.WARNING):
            agent.act(AgentDecision.EJECUTAR_LIMPIEZA)

        assert "Voltaje elevado" in caplog.text or "INESTABILIDAD" in caplog.text

    def test_act_includes_health_score_in_context(self, agent):
        """Proyección de intent incluye health_score."""
        agent._last_diagnosis = (
            DiagnosisBuilder()
            .with_health(TopologyBuilder().with_health_score(0.75).build())
            .build()
        )
        agent._session.post.return_value = ResponseFactory.success({})

        agent.act(AgentDecision.EJECUTAR_LIMPIEZA)

        json_body = agent._session.post.call_args.kwargs["json"]
        assert "topology_health" in json_body.get("context", {})


# =============================================================================
# TESTS: INVARIANTES TOPOLÓGICOS
# =============================================================================


class TestTopologicalInvariants:
    """
    Tests que validan invariantes matemáticos de topología algebraica.

    Garantizan coherencia con propiedades fundamentales de
    homología simplicial y teoría de grafos.
    """

    # =========================================================================
    # Números de Betti
    # =========================================================================

    def test_betti_0_minimum_is_one(self):
        """Invariante: β₀ ≥ 1 (al menos un componente)."""
        health = TopologyBuilder().build()
        assert health.betti.b0 >= 1

    def test_betti_1_non_negative(self):
        """Invariante: β₁ ≥ 0 (ciclos no negativos)."""
        health = TopologyBuilder().with_betti(1, 0).build()
        assert health.betti.b1 >= 0

    def test_builder_rejects_invalid_betti_b0(self):
        """Builder rechaza β₀ < 1."""
        with pytest.raises(ValueError, match="β₀ ≥ 1"):
            TopologyBuilder().with_betti(b0=0, b1=0)

    def test_builder_rejects_invalid_betti_b1(self):
        """Builder rechaza β₁ < 0."""
        with pytest.raises(ValueError, match="β₁ ≥ 0"):
            TopologyBuilder().with_betti(b0=1, b1=-1)

    # =========================================================================
    # Característica de Euler
    # =========================================================================

    @pytest.mark.parametrize(
        "b0,b1,expected_euler",
        [
            (1, 0, 1),   # Árbol conectado
            (2, 0, 2),   # Dos componentes sin ciclos
            (1, 1, 0),   # Un ciclo
            (1, 2, -1),  # Dos ciclos independientes
            (3, 1, 2),   # Tres componentes, un ciclo
        ],
    )
    def test_euler_characteristic_formula(
        self, 
        b0: int, 
        b1: int, 
        expected_euler: int
    ):
        """Verifica χ = β₀ - β₁."""
        health = TopologyBuilder().with_betti(b0, b1).build()
        euler = health.betti.b0 - health.betti.b1
        assert euler == expected_euler

    # =========================================================================
    # Fragmentación
    # =========================================================================

    def test_fragmentation_implies_multiple_components(self):
        """Nodos desconectados implican β₀ > 1."""
        fragmented = (
            TopologyBuilder()
            .with_disconnected_nodes("Redis", "Filesystem")
            .build()
        )

        assert fragmented.betti.b0 > 1
        assert len(fragmented.disconnected_nodes) > 0

    def test_health_inversely_proportional_to_fragmentation(self):
        """Mayor fragmentación → menor health_score."""
        connected = TopologyBuilder().with_betti(1, 0).build()
        fragmented = TopologyBuilder().with_betti(3, 0).build()

        assert fragmented.health_score < connected.health_score

    def test_fragmentation_auto_adjusts_betti(self):
        """Builder ajusta β₀ automáticamente con nodos desconectados."""
        # 2 nodos desconectados → β₀ debe ser al menos 3
        health = (
            TopologyBuilder()
            .with_betti(1, 0)  # Inicialmente conectado
            .with_disconnected_nodes("A", "B")  # Añade fragmentación
            .build()
        )

        assert health.betti.b0 >= 3

    # =========================================================================
    # Coherencia Score ↔ Level
    # =========================================================================

    def test_health_score_level_coherence(self):
        """Health score es coherente con level."""
        for level, bounds in TopologyBuilder.LEVEL_BOUNDS.items():
            health = TopologyBuilder().with_level(level).build()
            
            assert bounds.contains(health.health_score), \
                f"Score {health.health_score} fuera de bounds para {level}"

    def test_explicit_score_adjusted_to_level(self):
        """Score explícito se ajusta si contradice level."""
        # Score alto pero level crítico → se ajustará
        health = (
            TopologyBuilder()
            .with_health_score(0.9)
            .with_level(HealthLevel.CRITICAL)
            .build()
        )

        critical_bounds = TopologyBuilder.LEVEL_BOUNDS[HealthLevel.CRITICAL]
        assert critical_bounds.contains(health.health_score)


# =============================================================================
# TESTS: PERSISTENCIA Y RUIDO
# =============================================================================


class TestPersistenceAnalysis:
    """
    Tests para análisis de homología persistente.

    Valida la distinción entre señal (características persistentes)
    y ruido (fluctuaciones efímeras).
    """

    def test_stable_state_no_features(self):
        """Estado STABLE no tiene características."""
        result = PersistenceBuilder().stable().build()

        assert result.state == MetricState.STABLE
        assert result.feature_count == 0
        assert result.noise_count == 0

    def test_feature_state_has_persistent_characteristics(self):
        """Estado FEATURE tiene características con alta persistencia."""
        result = PersistenceBuilder().with_features(3, max_lifespan=20).build()

        assert result.state == MetricState.FEATURE
        assert result.feature_count == 3
        assert result.max_lifespan == 20
        assert result.total_persistence > 0

    def test_noise_state_has_ephemeral_characteristics(self):
        """Estado NOISE tiene características con baja persistencia."""
        result = PersistenceBuilder().with_noise(5, max_lifespan=1.5).build()

        assert result.state == MetricState.NOISE
        assert result.noise_count == 5
        assert result.max_lifespan < 2  # Corta vida

    def test_critical_state_high_persistence(self):
        """Estado CRITICAL tiene alta persistencia y duración."""
        result = PersistenceBuilder().critical(active_duration=50).build()

        assert result.state == MetricState.CRITICAL
        assert result.metadata.get("active_duration") == 50


# =============================================================================
# TESTS: TELEMETRY DATA
# =============================================================================


class TestTelemetryData:
    """Tests para TelemetryData y su factory method."""

    def test_from_dict_with_valid_data(self):
        """from_dict parsea datos válidos correctamente."""
        data = {
            "flyback_voltage": 0.5,
            "saturation": 0.7,
        }

        result = TelemetryData.from_dict(data)

        assert result is not None
        assert result.flyback_voltage == 0.5
        assert result.saturation == 0.7

    def test_from_dict_with_nested_metrics(self):
        """from_dict extrae de estructura anidada."""
        data = {
            "metrics": {
                "flux_condenser.max_flyback_voltage": 0.4,
                "flux_condenser.avg_saturation": 0.6,
            }
        }

        result = TelemetryData.from_dict(data)

        assert result is not None
        assert result.flyback_voltage == 0.4
        assert result.saturation == 0.6

    def test_from_dict_defaults_to_idle_on_missing(self):
        """Datos faltantes defaultean a estado IDLE (0.0)."""
        data = {"other_field": "value"}

        result = TelemetryData.from_dict(data)

        assert result is not None
        assert result.flyback_voltage == 0.0
        assert result.saturation == 0.0

    def test_from_dict_returns_none_for_non_dict(self):
        """from_dict retorna None para input no-dict."""
        assert TelemetryData.from_dict([1, 2, 3]) is None
        assert TelemetryData.from_dict("string") is None
        assert TelemetryData.from_dict(123) is None

    def test_values_clamped_to_unit_interval(self):
        """Valores se clampean a [0, 1]."""
        telemetry = TelemetryData(flyback_voltage=1.5, saturation=-0.3)

        assert telemetry.flyback_voltage == 1.0
        assert telemetry.saturation == 0.0

    def test_builder_creates_valid_telemetry(self):
        """TelemetryBuilder crea datos válidos."""
        telemetry = TelemetryBuilder.nominal().build()

        assert 0 <= telemetry.flyback_voltage <= 1
        assert 0 <= telemetry.saturation <= 1


# =============================================================================
# TESTS: THRESHOLD CONFIG
# =============================================================================


class TestThresholdConfig:
    """Tests para ThresholdConfig."""

    def test_default_thresholds_valid(self):
        """Umbrales por defecto son válidos."""
        config = ThresholdConfig()

        assert 0 < config.flyback_voltage_warning < config.flyback_voltage_critical <= 1
        assert 0 < config.saturation_warning < config.saturation_critical <= 1

    def test_custom_thresholds_valid(self):
        """Umbrales personalizados válidos aceptados."""
        config = ThresholdConfig(
            flyback_voltage_warning=0.4,
            flyback_voltage_critical=0.7,
            saturation_warning=0.8,
            saturation_critical=0.9,
        )

        assert config.flyback_voltage_warning == 0.4
        assert config.saturation_critical == 0.9

    def test_invalid_threshold_order_rejected(self):
        """Umbrales donde warning >= critical rechazados."""
        with pytest.raises(ValueError):
            ThresholdConfig(
                flyback_voltage_warning=0.8,
                flyback_voltage_critical=0.5,  # warning > critical
            )

    def test_out_of_range_threshold_rejected(self):
        """Umbrales fuera de [0, 1] rechazados."""
        with pytest.raises(ValueError):
            ThresholdConfig(flyback_voltage_critical=1.5)


# =============================================================================
# TESTS: AGENT METRICS
# =============================================================================


class TestAgentMetrics:
    """Tests para AgentMetrics."""

    def test_initial_state(self):
        """Estado inicial correcto."""
        metrics = AgentMetrics()

        assert metrics.cycles_executed == 0
        assert metrics.successful_observations == 0
        assert metrics.failed_observations == 0
        assert metrics.consecutive_failures == 0

    def test_record_success_updates_counters(self):
        """record_success actualiza contadores correctamente."""
        metrics = AgentMetrics()
        metrics.consecutive_failures = 3

        metrics.record_success()

        assert metrics.successful_observations == 1
        assert metrics.consecutive_failures == 0
        assert metrics.last_successful_observation is not None

    def test_record_failure_updates_counters(self):
        """record_failure actualiza contadores correctamente."""
        metrics = AgentMetrics()

        metrics.record_failure()
        metrics.record_failure()

        assert metrics.failed_observations == 2
        assert metrics.consecutive_failures == 2

    def test_success_rate_calculation(self):
        """success_rate se calcula correctamente."""
        metrics = AgentMetrics()
        
        for _ in range(7):
            metrics.record_success()
        for _ in range(3):
            metrics.record_failure()

        assert metrics.success_rate == 0.7

    def test_success_rate_zero_when_no_observations(self):
        """success_rate es 0 sin observaciones."""
        metrics = AgentMetrics()
        assert metrics.success_rate == 0.0

    def test_to_dict_serialization(self):
        """to_dict serializa correctamente."""
        metrics = AgentMetrics()
        metrics.record_success()
        metrics.record_decision(AgentDecision.HEARTBEAT)

        result = metrics.to_dict()

        assert "successful_observations" in result
        assert "decisions_count" in result
        assert result["decisions_count"].get("HEARTBEAT", 0) >= 1


# =============================================================================
# TESTS: CICLO OODA INTEGRADO
# =============================================================================


class TestOODAIntegration(TestFixtureBase):
    """
    Tests de integración del ciclo OODA completo.

    Valida que las cuatro fases funcionan coherentemente
    como un morfismo compuesto: O → R → D → A.
    """

    def test_full_cycle_nominal(self, agent):
        """Ciclo completo con sistema nominal."""
        agent._session.get.return_value = ResponseFactory.nominal_telemetry()
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        # O: Observe
        telemetry = agent.observe()
        assert telemetry is not None

        # R: Orient
        status = agent.orient(telemetry)
        assert status == SystemStatus.NOMINAL

        # D: Decide
        decision = agent.decide(status)
        assert decision == AgentDecision.HEARTBEAT

        # A: Act
        result = agent.act(decision)
        assert result is True

    def test_full_cycle_critical(self, agent):
        """Ciclo completo con sistema crítico."""
        agent._session.get.return_value = ResponseFactory.success({
            "flyback_voltage": 0.95,
            "saturation": 0.99,
        })
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        telemetry = agent.observe()
        status = agent.orient(telemetry)
        decision = agent.decide(status)
        result = agent.act(decision)

        assert status == SystemStatus.CRITICO
        assert decision == AgentDecision.ALERTA_CRITICA
        assert result is True

    def test_full_cycle_disconnected(self, agent):
        """Ciclo completo con sistema desconectado."""
        agent._session.get.side_effect = requests.exceptions.ConnectionError()
        
        # Después de varios fallos, topología se fragmenta
        for _ in range(agent.MAX_CONSECUTIVE_FAILURES):
            agent.observe()

        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.disconnected_from_core().build()
        )

        status = agent.orient(None)
        decision = agent.decide(status)
        result = agent.act(decision)

        assert status == SystemStatus.DISCONNECTED
        assert decision == AgentDecision.RECONNECT
        assert result is True

    def test_cycle_preserves_diagnosis_chain(self, agent):
        """El diagnóstico se propaga entre fases."""
        agent._session.get.return_value = ResponseFactory.nominal_telemetry()
        
        health = TopologyBuilder.nominal().with_health_score(0.85).build()
        agent.topology.get_topological_health.return_value = health

        telemetry = agent.observe()
        agent.orient(telemetry)
        
        # Diagnóstico almacenado
        assert agent._last_diagnosis is not None
        assert agent._last_diagnosis.health.health_score == 0.85

        agent.decide(SystemStatus.NOMINAL)
        agent.act(AgentDecision.HEARTBEAT)

        # Diagnóstico persiste tras todas las fases
        assert agent._last_diagnosis is not None


# =============================================================================
# TESTS: EDGE CASES Y ROBUSTEZ
# =============================================================================


class TestEdgeCases(TestFixtureBase):
    """Tests para casos límite y robustez."""

    def test_orient_handles_none_diagnosis_gracefully(self, agent):
        """Orient funciona sin diagnóstico previo."""
        agent._last_diagnosis = None
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        # No debe lanzar excepción
        status = agent.orient(TelemetryBuilder.nominal().build())
        
        assert status is not None

    def test_act_handles_none_diagnosis(self, agent):
        """Act funciona sin diagnóstico (mensaje vacío)."""
        agent._last_diagnosis = None

        # No debe lanzar excepción
        result = agent.act(AgentDecision.HEARTBEAT)
        
        assert result is True

    def test_agent_handles_all_status_types(self, agent):
        """Agente maneja todos los tipos de SystemStatus."""
        for status in SystemStatus:
            decision = agent.decide(status)
            assert decision in AgentDecision

    def test_agent_handles_all_decision_types(self, agent):
        """Agente puede ejecutar todas las decisiones."""
        agent._session.post.return_value = ResponseFactory.success({})
        
        for decision in AgentDecision:
            agent._last_decision = None  # Reset debounce
            result = agent.act(decision)
            assert isinstance(result, bool)

    def test_telemetry_at_exact_thresholds(self, agent):
        """Telemetría exactamente en umbrales."""
        agent.topology.get_topological_health.return_value = (
            TopologyBuilder.nominal().build()
        )

        # Exactamente en warning (no critical)
        telemetry = TelemetryData(
            flyback_voltage=DEFAULT_VOLTAGE_CRITICAL,  # En el umbral
            saturation=0.3,
        )
        
        status = agent.orient(telemetry)
        
        # En el umbral crítico (no mayor) debería ser crítico según >
        assert status == SystemStatus.CRITICO

    def test_metrics_survive_multiple_cycles(self, agent):
        """Métricas se acumulan correctamente entre ciclos."""
        agent._session.get.return_value = ResponseFactory.nominal_telemetry()

        for _ in range(5):
            agent._metrics.increment_cycle()
            agent.observe()

        assert agent._metrics.cycles_executed == 5
        assert agent._metrics.successful_observations == 5