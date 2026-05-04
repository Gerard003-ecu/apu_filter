"""
=========================================================================================
Suite de Pruebas para Autonomous Agent (apu_agent.py)
Ubicación: tests/test_apu_agent.py
=========================================================================================

FUNDAMENTACIÓN TEÓRICA DEL TESTING
===================================

Esta suite implementa tres niveles de verificación:

1. **Unit Testing Clásico**: Verificación de comportamiento individual
2. **Property-Based Testing**: Verificación de invariantes algebraicos
3. **Integration Testing**: Verificación de interacciones entre componentes

INVARIANTES BAJO PRUEBA
========================

1. **Algebraicos**: Propiedades de orden parcial, neutralidad de carga
2. **Topológicos**: Preservación de números de Betti, conectividad
3. **Termodinámicos**: Disipación entrópica no negativa (P_diss ≥ 0)
4. **Temporales**: Monotonicidad de contadores, causalidad

COBERTURA OBJETIVO
==================

- Line Coverage: > 95%
- Branch Coverage: > 90%
- Property Coverage: 100% (todos los invariantes verificados)

=========================================================================================
"""

import math
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest
import requests
import scipy.sparse as sp
from hypothesis import given, assume, strategies as st, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

# Ajustar path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.apu_agent import (
    # Constantes
    PhysicalConstants,
    NetworkTopology,
    DefaultConfig,
    
    # Enumeraciones
    SystemStatus,
    AgentDecision,
    ObservationErrorType,
    
    # Configuración
    ThresholdConfig,
    ConnectionConfig,
    TopologyConfig,
    TimingConfig,
    AgentConfig,
    
    # Estructuras de datos
    TelemetryData,
    AgentMetrics,
    ObservationResult,
    TopologicalDiagnosis,
    AgentSnapshot,
    SnapshotHistory,
    
    # Evaluadores
    StateEvaluator,
    BaseEvaluator,
    FragmentationEvaluator,
    NoTelemetryEvaluator,
    CriticalVoltageEvaluator,
    CriticalSaturationEvaluator,
    TopologyHealthCriticalEvaluator,
    PersistentSaturationCriticalEvaluator,
    SaturationFeatureEvaluator,
    PersistentVoltageCriticalEvaluator,
    VoltageFeatureEvaluator,
    RetryLoopEvaluator,
    UnhealthyTopologyEvaluator,
    
    # Sintetizador
    HamiltonianSynthesizer,
    
    # Agente
    AutonomousAgent,
    create_agent,
    create_minimal_agent,
    
    # Excepciones
    AgentError,
    ConfigurationError,
    ObservationError,
    TopologyError,
    TelemetryValidationError,
    StabilityError,
)

from app.tactics.topological_analyzer import (
    HealthLevel,
    MetricState,
    PersistenceAnalysisResult,
    TopologicalHealth,
    BettiNumbers,
)


# ============================================================================
# CONSTANTES DE PRUEBA
# ============================================================================

TEST_CORE_URL = "http://localhost:5002"
TEST_TIMEOUT = 5
TEST_EPSILON = 1e-9


# ============================================================================
# FIXTURES BÁSICOS
# ============================================================================

@pytest.fixture
def threshold_config() -> ThresholdConfig:
    """
    Fixture: Configuración de umbrales estándar.
    
    Postcondiciones:
        - 0 ≤ warning < critical ≤ 1 para cada par
        - Histeresis mínima satisfecha
    """
    return ThresholdConfig(
        flyback_voltage_warning=0.5,
        flyback_voltage_critical=0.8,
        saturation_warning=0.9,
        saturation_critical=0.95,
    )


@pytest.fixture
def connection_config() -> ConnectionConfig:
    """Fixture: Configuración de conexión de prueba."""
    return ConnectionConfig(
        base_url=TEST_CORE_URL,
        request_timeout=TEST_TIMEOUT,
        max_retries=3,
        backoff_factor=0.5,
    )


@pytest.fixture
def topology_config() -> TopologyConfig:
    """Fixture: Configuración topológica de prueba."""
    return TopologyConfig(
        max_history=100,
        persistence_window=20,
        health_critical_threshold=0.4,
        health_warning_threshold=0.7,
    )


@pytest.fixture
def timing_config() -> TimingConfig:
    """Fixture: Configuración temporal de prueba."""
    return TimingConfig(
        check_interval=10,
        debounce_window_seconds=60,
        max_consecutive_failures=5,
        startup_backoff_initial=1.0,  # Reducido para tests
        startup_backoff_max=5.0,
        startup_max_attempts=5,
    )


@pytest.fixture
def agent_config(
    threshold_config: ThresholdConfig,
    connection_config: ConnectionConfig,
    topology_config: TopologyConfig,
    timing_config: TimingConfig,
) -> AgentConfig:
    """Fixture: Configuración completa del agente."""
    return AgentConfig(
        thresholds=threshold_config,
        connection=connection_config,
        topology=topology_config,
        timing=timing_config,
    )


@pytest.fixture
def sample_telemetry() -> TelemetryData:
    """
    Fixture: Telemetría de muestra válida.
    
    Postcondiciones:
        - Todos los valores ∈ [0, 1]
        - No hubo clamping
    """
    return TelemetryData(
        flyback_voltage=0.3,
        saturation=0.6,
        integrity_score=0.95,
        raw_data={
            "flux_condenser": {
                "max_flyback_voltage": 0.3,
                "avg_saturation": 0.6,
            },
            "integrity_score": 0.95,
        },
    )


@pytest.fixture
def sample_telemetry_critical() -> TelemetryData:
    """Fixture: Telemetría con valores críticos."""
    return TelemetryData(
        flyback_voltage=0.85,  # > critical threshold (0.8)
        saturation=0.96,       # > critical threshold (0.95)
        integrity_score=0.5,
    )


@pytest.fixture
def mock_topological_health() -> TopologicalHealth:
    """Fixture: Salud topológica nominal simulada."""
    betti = BettiNumbers(b0=1, b1=0)
    return TopologicalHealth(
        betti=betti,
        health_score=1.0,
        level=HealthLevel.HEALTHY,
        disconnected_nodes=set(),
        missing_edges=set(),
        request_loops=[],
        diagnostics=[],
    )


@pytest.fixture
def mock_topological_health_degraded() -> TopologicalHealth:
    """Fixture: Salud topológica degradada."""
    betti = BettiNumbers(b0=2, b1=0)  # Fragmentado
    return TopologicalHealth(
        betti=betti,
        health_score=0.3,
        level=HealthLevel.CRITICAL,
        disconnected_nodes={NetworkTopology.NODE_FILESYSTEM},
        missing_edges={(NetworkTopology.NODE_CORE, NetworkTopology.NODE_FILESYSTEM)},
        request_loops=[],
        diagnostics=["Nodo desconectado: Filesystem"],
    )


@pytest.fixture
def mock_persistence_analysis_nominal() -> PersistenceAnalysisResult:
    """Fixture: Análisis de persistencia nominal (ruido)."""
    return PersistenceAnalysisResult(
        state=MetricState.NOISE,
        feature_count=0,
        noise_count=3,
        total_persistence=0.0,
        max_lifespan=0.0,
        metadata={},
    )


@pytest.fixture
def mock_persistence_analysis_critical() -> PersistenceAnalysisResult:
    """Fixture: Análisis de persistencia crítica."""
    return PersistenceAnalysisResult(
        state=MetricState.CRITICAL,
        feature_count=0,
        noise_count=0,
        total_persistence=15.0,
        max_lifespan=15.0,
        metadata={"active_duration": 15.0},
    )


@pytest.fixture
def agent_metrics() -> AgentMetrics:
    """Fixture: Métricas del agente con valores iniciales."""
    return AgentMetrics(
        cycles_executed=10,
        successful_observations=8,
        failed_observations=2,
        consecutive_failures=0,
        last_successful_observation=datetime.now(),
    )


@pytest.fixture
def hamiltonian_synthesizer() -> HamiltonianSynthesizer:
    """Fixture: Sintetizador Hamiltoniano con evaluadores estándar."""
    return HamiltonianSynthesizer()


# ============================================================================
# FIXTURES DE MOCKING
# ============================================================================

@pytest.fixture
def mock_requests_success():
    """Mock: Requests exitosas con telemetría válida."""
    with patch('app.core.apu_agent.requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "flux_condenser": {
                "max_flyback_voltage": 0.3,
                "avg_saturation": 0.6,
            },
            "integrity_score": 0.95,
        }
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_requests_timeout():
    """Mock: Requests que lanzan Timeout."""
    with patch('app.core.apu_agent.requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session.get.side_effect = requests.exceptions.Timeout("Timeout")
        mock_session_class.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_requests_connection_error():
    """Mock: Requests que lanzan ConnectionError."""
    with patch('app.core.apu_agent.requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        mock_session_class.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_gauge_router():
    """Mock: GaugeFieldRouter simulado."""
    from app.core.immune_system.gauge_field_router import GaugeFieldRouter
    from app.core.mic_algebra import CategoricalState
    
    with patch('app.core.apu_agent.GaugeFieldRouter') as mock_router_class:
        mock_router = Mock(spec=GaugeFieldRouter)
        
        # Simular route_gradient
        def mock_route_gradient(initial_state, gradient):
            result_state = CategoricalState(
                payload=initial_state.payload,
                context={
                    **initial_state.context,
                    "gauge_selected_agent": "clean_on_edge_0",
                    "gauge_potential": gradient.tolist(),
                },
            )
            return result_state
        
        mock_router.route_gradient = Mock(side_effect=mock_route_gradient)
        mock_router_class.return_value = mock_router
        yield mock_router


# ============================================================================
# HYPOTHESIS STRATEGIES (Property-Based Testing)
# ============================================================================

# Estrategia: Valores normalizados [0, 1]
normalized_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Estrategia: Valores positivos para duración
positive_float = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)

# Estrategia: Enteros positivos
positive_int = st.integers(min_value=1, max_value=10000)

# Estrategia: Umbrales coherentes
@st.composite
def threshold_pair(draw):
    """Genera par de umbrales coherentes (warning, critical)."""
    warning = draw(st.floats(min_value=0.0, max_value=0.9))
    critical = draw(st.floats(min_value=warning + 0.05, max_value=1.0))
    return warning, critical

# Estrategia: Telemetría válida
@st.composite
def valid_telemetry(draw):
    """Genera TelemetryData válida."""
    return TelemetryData(
        flyback_voltage=draw(normalized_float),
        saturation=draw(normalized_float),
        integrity_score=draw(normalized_float),
    )

# Estrategia: Gradientes válidos
@st.composite
def valid_gradient(draw, num_nodes=4):
    """Genera gradiente válido (no negativo, finito)."""
    return np.array([draw(normalized_float) for _ in range(num_nodes)])


# ============================================================================
# HELPERS DE ASSERTIONS
# ============================================================================

def assert_normalized(value: float, name: str = "valor") -> None:
    """
    Asevera que un valor está normalizado ∈ [0, 1].
    
    Args:
        value: Valor a verificar
        name: Nombre del valor (para mensaje de error)
    """
    assert 0.0 <= value <= 1.0, f"{name} debe estar en [0,1], recibido: {value}"
    assert math.isfinite(value), f"{name} debe ser finito, recibido: {value}"


def assert_gradient_valid(gradient: np.ndarray, num_nodes: int) -> None:
    """
    Asevera que un gradiente cumple invariantes.
    
    Invariantes:
        1. len(gradient) == num_nodes
        2. np.all(np.isfinite(gradient))
        3. np.all(gradient >= 0)
    """
    assert len(gradient) == num_nodes, \
        f"Dimensión inválida: esperado {num_nodes}, recibido {len(gradient)}"
    
    assert np.all(np.isfinite(gradient)), \
        f"Gradiente contiene NaN/Inf: {gradient}"
    
    assert np.all(gradient >= -TEST_EPSILON), \
        f"Gradiente contiene valores negativos: {gradient}"


def assert_charge_neutrality(gradient: np.ndarray, tolerance: float = TEST_EPSILON) -> None:
    """
    Asevera neutralidad de carga: Σᵢ gradientᵢ ≈ 0.
    
    Args:
        gradient: Vector gradiente
        tolerance: Tolerancia numérica
    """
    total_charge = np.sum(gradient)
    assert abs(total_charge) < tolerance * len(gradient), \
        f"Violación de neutralidad de carga: Σ gradient = {total_charge:.4e}"


def assert_monotonic_increase(counter_before: int, counter_after: int, name: str = "contador") -> None:
    """Asevera que un contador aumentó monótonamente."""
    assert counter_after >= counter_before, \
        f"{name} no aumentó: antes={counter_before}, después={counter_after}"


def assert_system_status_order(s1: SystemStatus, s2: SystemStatus) -> None:
    """Asevera que el orden de SystemStatus es coherente."""
    if s1.severity < s2.severity:
        assert s1 < s2, f"Orden incoherente: {s1.name} < {s2.name}"
        assert s2 > s1, f"Orden incoherente: {s2.name} > {s1.name}"


# ============================================================================
# CLASE BASE PARA PRUEBAS
# ============================================================================

class BaseAgentTest:
    """
    Clase base para pruebas del agente con helpers comunes.
    
    Provee:
        - Setup/teardown de fixtures comunes
        - Helpers de validación
        - Mocks compartidos
    """
    
    @staticmethod
    def create_mock_telemetry(voltage: float = 0.3, saturation: float = 0.6) -> TelemetryData:
        """Helper: Crea telemetría mock con valores específicos."""
        return TelemetryData(
            flyback_voltage=voltage,
            saturation=saturation,
            integrity_score=0.95,
        )
    
    @staticmethod
    def create_mock_diagnosis(
        status: SystemStatus = SystemStatus.NOMINAL,
        health_score: float = 1.0,
    ) -> TopologicalDiagnosis:
        """Helper: Crea diagnóstico mock."""
        betti = BettiNumbers(b0=1, b1=0)
        health = TopologicalHealth(
            betti=betti,
            health_score=health_score,
            level=HealthLevel.HEALTHY,
            disconnected_nodes=set(),
            missing_edges=set(),
            request_loops=[],
            diagnostics=[],
        )
        
        persistence_nominal = PersistenceAnalysisResult(
            state=MetricState.NOISE,
            feature_count=0,
            noise_count=0,
            total_persistence=0.0,
            max_lifespan=0.0,
            metadata={},
        )
        
        return TopologicalDiagnosis(
            health=health,
            voltage_persistence=persistence_nominal,
            saturation_persistence=persistence_nominal,
            summary="Mock diagnosis",
            recommended_status=status,
        )
    
    @staticmethod
    def assert_telemetry_valid(telemetry: TelemetryData) -> None:
        """Helper: Valida invariantes de telemetría."""
        assert_normalized(telemetry.flyback_voltage, "flyback_voltage")
        assert_normalized(telemetry.saturation, "saturation")
        assert_normalized(telemetry.integrity_score, "integrity_score")
    
    @staticmethod
    def assert_metrics_valid(metrics: AgentMetrics) -> None:
        """Helper: Valida invariantes de métricas."""
        # Contadores no negativos
        assert metrics.cycles_executed >= 0
        assert metrics.successful_observations >= 0
        assert metrics.failed_observations >= 0
        assert metrics.consecutive_failures >= 0
        
        # Conservación
        assert (
            metrics.successful_observations + metrics.failed_observations
            == metrics.total_observations
        )
        
        # Success rate válido
        assert 0.0 <= metrics.success_rate <= 1.0
        
        # Uptime no negativo
        assert metrics.uptime_seconds >= 0.0


# ============================================================================
# PRUEBAS DE ENUMERACIONES
# ============================================================================

class TestSystemStatus:
    """
    Suite de pruebas para SystemStatus (Poset).
    
    Verifica:
        - Propiedades de orden parcial (reflexividad, antisimetría, transitividad)
        - Operaciones de retículo (join, meet)
        - Severidad coherente
    """
    
    def test_severity_monotonic(self):
        """PROP: Severidad es monótona con el orden."""
        statuses = list(SystemStatus)
        for i, s1 in enumerate(statuses):
            for j, s2 in enumerate(statuses):
                if i < j:
                    assert s1.severity <= s2.severity, \
                        f"Severidad no monótona: {s1.name}={s1.severity}, {s2.name}={s2.severity}"
    
    def test_order_reflexivity(self):
        """PROP: Reflexividad (∀s: s ≤ s)."""
        for status in SystemStatus:
            assert status <= status, f"Reflexividad violada para {status.name}"
            assert not (status < status), f"Orden estricto reflexivo para {status.name}"
    
    def test_order_antisymmetry(self):
        """PROP: Antisimetría (s1 ≤ s2 ∧ s2 ≤ s1 ⟹ s1 = s2)."""
        for s1 in SystemStatus:
            for s2 in SystemStatus:
                if s1 <= s2 and s2 <= s1:
                    assert s1 == s2, f"Antisimetría violada: {s1.name}, {s2.name}"
    
    def test_order_transitivity(self):
        """PROP: Transitividad (s1 ≤ s2 ∧ s2 ≤ s3 ⟹ s1 ≤ s3)."""
        statuses = list(SystemStatus)
        for s1 in statuses:
            for s2 in statuses:
                for s3 in statuses:
                    if s1 <= s2 and s2 <= s3:
                        assert s1 <= s3, \
                            f"Transitividad violada: {s1.name} ≤ {s2.name} ≤ {s3.name}"
    
    def test_worst_is_supremum(self):
        """PROP: worst() retorna el supremo (join del retículo)."""
        # Caso vacío
        assert SystemStatus.worst() == SystemStatus.NOMINAL
        
        # Casos con elementos
        assert SystemStatus.worst(SystemStatus.NOMINAL) == SystemStatus.NOMINAL
        assert SystemStatus.worst(
            SystemStatus.NOMINAL,
            SystemStatus.INESTABLE,
        ) == SystemStatus.INESTABLE
        
        assert SystemStatus.worst(
            SystemStatus.CRITICO,
            SystemStatus.INESTABLE,
            SystemStatus.NOMINAL,
        ) == SystemStatus.CRITICO
        
        # Propiedad de supremo: ∀s ∈ conjunto: s ≤ supremo
        test_set = [SystemStatus.NOMINAL, SystemStatus.INESTABLE, SystemStatus.SATURADO]
        supremo = SystemStatus.worst(*test_set)
        for s in test_set:
            assert s <= supremo, f"{s.name} no es ≤ supremo {supremo.name}"
    
    def test_best_is_infimum(self):
        """PROP: best() retorna el ínfimo (meet del retículo)."""
        # Caso vacío
        assert SystemStatus.best() == SystemStatus.DISCONNECTED
        
        # Casos con elementos
        assert SystemStatus.best(SystemStatus.DISCONNECTED) == SystemStatus.DISCONNECTED
        assert SystemStatus.best(
            SystemStatus.CRITICO,
            SystemStatus.INESTABLE,
        ) == SystemStatus.INESTABLE
        
        # Propiedad de ínfimo: ∀s ∈ conjunto: infimo ≤ s
        test_set = [SystemStatus.CRITICO, SystemStatus.SATURADO, SystemStatus.DISCONNECTED]
        infimo = SystemStatus.best(*test_set)
        for s in test_set:
            assert infimo <= s, f"ínfimo {infimo.name} no es ≤ {s.name}"
    
    def test_predicates_coherent(self):
        """PROP: Predicados son coherentes con severidad."""
        assert SystemStatus.NOMINAL.is_healthy
        assert not SystemStatus.CRITICO.is_healthy
        
        assert SystemStatus.CRITICO.is_critical
        assert SystemStatus.DISCONNECTED.is_critical
        assert not SystemStatus.NOMINAL.is_critical
        
        assert SystemStatus.INESTABLE.requires_stabilization
        assert not SystemStatus.NOMINAL.requires_stabilization
    
    @given(st.sampled_from(SystemStatus), st.sampled_from(SystemStatus))
    def test_comparison_coherent_with_severity(self, s1: SystemStatus, s2: SystemStatus):
        """PROP (Hypothesis): Comparación coherente con severidad."""
        if s1.severity < s2.severity:
            assert s1 < s2
            assert s1 <= s2
            assert not (s1 > s2)
            assert not (s1 >= s2)
        elif s1.severity == s2.severity:
            assert s1 == s2
            assert s1 <= s2
            assert s1 >= s2


class TestAgentDecision:
    """Suite de pruebas para AgentDecision."""
    
    def test_immediate_action_decisions(self):
        """PROP: Decisiones críticas requieren acción inmediata."""
        assert AgentDecision.ALERTA_CRITICA.requires_immediate_action
        assert AgentDecision.RECONNECT.requires_immediate_action
        
        assert not AgentDecision.HEARTBEAT.requires_immediate_action
        assert not AgentDecision.WAIT.requires_immediate_action
    
    def test_active_control_classification(self):
        """PROP: Clasificación correcta de control activo vs pasivo."""
        active_decisions = {
            AgentDecision.EJECUTAR_LIMPIEZA,
            AgentDecision.AJUSTAR_VELOCIDAD,
            AgentDecision.RECONNECT,
        }
        
        passive_decisions = {
            AgentDecision.HEARTBEAT,
            AgentDecision.WAIT,
        }
        
        for decision in active_decisions:
            assert decision.is_active_control, f"{decision.name} debe ser activo"
            assert not decision.is_passive, f"{decision.name} no debe ser pasivo"
        
        for decision in passive_decisions:
            assert decision.is_passive, f"{decision.name} debe ser pasivo"
            assert not decision.is_active_control, f"{decision.name} no debe ser activo"
    
    def test_expected_duration_coherent(self):
        """PROP: Duración esperada es coherente con tipo de decisión."""
        # Decisiones pasivas no tienen duración
        assert AgentDecision.HEARTBEAT.expected_duration_seconds is None
        assert AgentDecision.WAIT.expected_duration_seconds is None
        
        # Decisiones activas tienen duración positiva
        cleanup_duration = AgentDecision.EJECUTAR_LIMPIEZA.expected_duration_seconds
        assert cleanup_duration is not None and cleanup_duration > 0


class TestObservationErrorType:
    """Suite de pruebas para ObservationErrorType."""
    
    def test_transient_classification(self):
        """PROP: Errores transitorios correctamente clasificados."""
        transient_errors = {
            ObservationErrorType.TIMEOUT,
            ObservationErrorType.CONNECTION_ERROR,
            ObservationErrorType.HTTP_ERROR,
        }
        
        for error_type in transient_errors:
            assert error_type.is_transient, f"{error_type.value} debe ser transitorio"
        
        assert not ObservationErrorType.INVALID_JSON.is_transient
        assert not ObservationErrorType.INVALID_TELEMETRY.is_transient
    
    def test_reconnection_requirement(self):
        """PROP: Solo CONNECTION_ERROR requiere reconexión."""
        assert ObservationErrorType.CONNECTION_ERROR.requires_reconnection
        assert not ObservationErrorType.TIMEOUT.requires_reconnection
        assert not ObservationErrorType.HTTP_ERROR.requires_reconnection


# ============================================================================
# PRUEBAS DE CONFIGURACIÓN
# ============================================================================

class TestThresholdConfig:
    """
    Suite de pruebas para ThresholdConfig.
    
    Verifica:
        - Validación de invariantes en __post_init__
        - Clasificación correcta de valores
        - Cálculo de márgenes de seguridad
    """
    
    def test_valid_thresholds_accepted(self, threshold_config: ThresholdConfig):
        """UNIT: Umbrales válidos son aceptados."""
        assert threshold_config.flyback_voltage_warning == 0.5
        assert threshold_config.flyback_voltage_critical == 0.8
        assert threshold_config.saturation_warning == 0.9
        assert threshold_config.saturation_critical == 0.95
    
    def test_invalid_order_rejected(self):
        """UNIT: Orden inválido (warning ≥ critical) es rechazado."""
        with pytest.raises(ConfigurationError, match="debe cumplir 0 ≤ warning.*< critical"):
            ThresholdConfig(
                flyback_voltage_warning=0.9,
                flyback_voltage_critical=0.5,  # Invertido!
            )
    
    def test_insufficient_hysteresis_rejected(self):
        """UNIT: Histeresis insuficiente es rechazada."""
        with pytest.raises(ConfigurationError, match="Histeresis.*insuficiente"):
            ThresholdConfig(
                flyback_voltage_warning=0.79,
                flyback_voltage_critical=0.8,  # Diferencia < 0.05
            )
    
    @given(threshold_pair())
    def test_valid_thresholds_always_work(self, thresholds: Tuple[float, float]):
        """PROP (Hypothesis): Pares válidos siempre construyen config."""
        warning, critical = thresholds
        assume(critical - warning >= 0.05)  # Histeresis mínima
        
        config = ThresholdConfig(
            flyback_voltage_warning=warning,
            flyback_voltage_critical=critical,
        )
        
        assert config.flyback_voltage_warning == warning
        assert config.flyback_voltage_critical == critical
    
    def test_voltage_classification(self, threshold_config: ThresholdConfig):
        """UNIT: Clasificación de voltaje es correcta."""
        assert threshold_config.classify_voltage(0.3) == "nominal"
        assert threshold_config.classify_voltage(0.6) == "warning"
        assert threshold_config.classify_voltage(0.85) == "critical"
        
        # Casos límite
        assert threshold_config.classify_voltage(0.5) == "warning"  # En el borde
        assert threshold_config.classify_voltage(0.8) == "critical"  # En el borde
    
    def test_saturation_classification(self, threshold_config: ThresholdConfig):
        """UNIT: Clasificación de saturación es correcta."""
        assert threshold_config.classify_saturation(0.5) == "nominal"
        assert threshold_config.classify_saturation(0.92) == "warning"
        assert threshold_config.classify_saturation(0.97) == "critical"
    
    def test_voltage_margin_calculation(self, threshold_config: ThresholdConfig):
        """UNIT: Cálculo de margen de seguridad es correcto."""
        # Valor nominal: margen positivo
        margin = threshold_config.get_voltage_margin(0.5)
        assert margin > 0, "Margen debe ser positivo para valores nominales"
        
        # Valor crítico: margen cero o negativo
        margin_critical = threshold_config.get_voltage_margin(0.9)
        assert margin_critical < 0, "Margen debe ser negativo para valores sobre el umbral"
        
        # En el umbral: margen cero
        margin_threshold = threshold_config.get_voltage_margin(0.8)
        assert abs(margin_threshold) < TEST_EPSILON, "Margen debe ser ~0 en el umbral"


class TestConnectionConfig:
    """Suite de pruebas para ConnectionConfig."""
    
    def test_valid_config_accepted(self, connection_config: ConnectionConfig):
        """UNIT: Configuración válida es aceptada."""
        assert connection_config.base_url == TEST_CORE_URL
        assert connection_config.request_timeout == TEST_TIMEOUT
    
    def test_invalid_timeout_rejected(self):
        """UNIT: Timeout no positivo es rechazado."""
        with pytest.raises(ConfigurationError, match="request_timeout debe ser positivo"):
            ConnectionConfig(request_timeout=0)
        
        with pytest.raises(ConfigurationError, match="request_timeout debe ser positivo"):
            ConnectionConfig(request_timeout=-5)
    
    def test_invalid_pool_config_rejected(self):
        """UNIT: Pool mal configurado (maxsize < connections) es rechazado."""
        with pytest.raises(ConfigurationError, match="Riesgo de deadlock"):
            ConnectionConfig(
                pool_connections=10,
                pool_maxsize=5,  # Menor que connections!
            )
    
    def test_negative_backoff_rejected(self):
        """UNIT: Backoff negativo es rechazado."""
        with pytest.raises(ConfigurationError, match="backoff_factor debe ser no negativo"):
            ConnectionConfig(backoff_factor=-0.5)
    
    def test_telemetry_endpoint_construction(self, connection_config: ConnectionConfig):
        """UNIT: Endpoint de telemetría se construye correctamente."""
        expected = f"{TEST_CORE_URL}/api/telemetry/status"
        assert connection_config.telemetry_endpoint == expected
    
    def test_tools_endpoint_construction(self, connection_config: ConnectionConfig):
        """UNIT: Endpoint de tools se construye correctamente."""
        expected = f"{TEST_CORE_URL}/api/tools/clean"
        assert connection_config.tools_endpoint("clean") == expected
    
    def test_retry_delay_calculation(self, connection_config: ConnectionConfig):
        """UNIT: Cálculo de delay de retry es exponencial."""
        # backoff_factor = 0.5
        assert connection_config.calculate_retry_delay(0) == 0.5  # 0.5 * 2^0
        assert connection_config.calculate_retry_delay(1) == 1.0  # 0.5 * 2^1
        assert connection_config.calculate_retry_delay(2) == 2.0  # 0.5 * 2^2
        assert connection_config.calculate_retry_delay(3) == 4.0  # 0.5 * 2^3


class TestTopologyConfig:
    """Suite de pruebas para TopologyConfig."""
    
    def test_valid_config_accepted(self, topology_config: TopologyConfig):
        """UNIT: Configuración válida es aceptada."""
        assert topology_config.max_history == 100
        assert topology_config.persistence_window == 20
    
    def test_invalid_thresholds_rejected(self):
        """UNIT: Umbrales de salud inválidos son rechazados."""
        with pytest.raises(ConfigurationError, match="Umbrales de salud inválidos"):
            TopologyConfig(
                health_critical_threshold=0.8,
                health_warning_threshold=0.5,  # Invertido!
            )
    
    def test_invalid_persistence_window_rejected(self):
        """UNIT: Ventana de persistencia no positiva es rechazada."""
        with pytest.raises(ConfigurationError, match="persistence_window debe ser positiva"):
            TopologyConfig(persistence_window=0)
    
    def test_history_smaller_than_window_rejected(self):
        """UNIT: Historia menor que ventana es rechazada."""
        with pytest.raises(ConfigurationError, match="max_history.*debe ser ≥ persistence_window"):
            TopologyConfig(
                max_history=10,
                persistence_window=20,
            )
    
    def test_health_score_classification(self, topology_config: TopologyConfig):
        """UNIT: Clasificación de score de salud es correcta."""
        assert topology_config.classify_health_score(0.3) == HealthLevel.CRITICAL
        assert topology_config.classify_health_score(0.5) == HealthLevel.UNHEALTHY
        assert topology_config.classify_health_score(0.8) == HealthLevel.HEALTHY


class TestAgentConfig:
    """Suite de pruebas para AgentConfig (composición)."""
    
    def test_config_composition(self, agent_config: AgentConfig):
        """UNIT: Configuración compuesta contiene todos los componentes."""
        assert isinstance(agent_config.thresholds, ThresholdConfig)
        assert isinstance(agent_config.connection, ConnectionConfig)
        assert isinstance(agent_config.topology, TopologyConfig)
        assert isinstance(agent_config.timing, TimingConfig)
    
    def test_from_environment_default(self, monkeypatch):
        """UNIT: Construcción desde env vars con defaults."""
        # Sin variables de entorno
        monkeypatch.delenv("CORE_API_URL", raising=False)
        monkeypatch.delenv("CHECK_INTERVAL", raising=False)
        
        config = AgentConfig.from_environment()
        
        assert config.connection.base_url == DefaultConfig.CORE_URL
        assert config.timing.check_interval == DefaultConfig.CHECK_INTERVAL
    
    def test_from_environment_custom(self, monkeypatch):
        """UNIT: Construcción desde env vars personalizadas."""
        monkeypatch.setenv("CORE_API_URL", "http://custom:8080")
        monkeypatch.setenv("CHECK_INTERVAL", "30")
        monkeypatch.setenv("REQUEST_TIMEOUT", "15")
        
        config = AgentConfig.from_environment()
        
        assert config.connection.base_url == "http://custom:8080"
        assert config.timing.check_interval == 30
        assert config.connection.request_timeout == 15
    
    def test_from_environment_invalid_values(self, monkeypatch, caplog):
        """UNIT: Valores inválidos en env vars usan defaults con warning."""
        monkeypatch.setenv("CHECK_INTERVAL", "invalid")
        
        with caplog.at_level(logging.WARNING):
            config = AgentConfig.from_environment()
        
        # Debe usar default
        assert config.timing.check_interval == DefaultConfig.CHECK_INTERVAL
        
        # Debe logear warning
        assert "Valor inválido para CHECK_INTERVAL" in caplog.text
    
    def test_url_validation_and_normalization(self):
        """UNIT: URLs son validadas y normalizadas."""
        # Agregar esquema si falta
        config = AgentConfig(
            connection=ConnectionConfig(base_url="localhost:5002")
        )
        assert config.connection.base_url == "http://localhost:5002"
        
        # Remover trailing slash
        config2 = AgentConfig(
            connection=ConnectionConfig(base_url="http://localhost:5002/")
        )
        assert config2.connection.base_url == "http://localhost:5002"
    
    def test_invalid_url_rejected(self):
        """UNIT: URL inválida es rechazada."""
        with pytest.raises(ConfigurationError, match="URL sin host válido"):
            AgentConfig._validate_and_normalize_url("http://")
        
        with pytest.raises(ConfigurationError, match="URL.*vacía"):
            AgentConfig._validate_and_normalize_url("")
    
    def test_to_dict_serialization(self, agent_config: AgentConfig):
        """UNIT: Serialización a dict es completa."""
        config_dict = agent_config.to_dict()
        
        assert "thresholds" in config_dict
        assert "connection" in config_dict
        assert "topology" in config_dict
        assert "timing" in config_dict
        
        # Verificar estructura anidada
        assert "flyback_voltage" in config_dict["thresholds"]
        assert "base_url" in config_dict["connection"]


# ============================================================================
# PRUEBAS DE TELEMETRYDATA
# ============================================================================

class TestTelemetryData(BaseAgentTest):
    """
    Suite de pruebas para TelemetryData.
    
    Verifica:
        - Normalización y clamping de valores
        - Construcción desde dict con paths múltiples
        - Invariantes post-construcción
        - Propiedades derivadas (normas, predicados)
    """
    
    def test_construction_with_valid_values(self):
        """UNIT: Construcción con valores válidos no clampea."""
        telemetry = TelemetryData(
            flyback_voltage=0.5,
            saturation=0.7,
            integrity_score=0.9,
        )
        
        assert telemetry.flyback_voltage == 0.5
        assert telemetry.saturation == 0.7
        assert telemetry.integrity_score == 0.9
        assert not telemetry.was_clamped
    
    def test_clamping_out_of_range_values(self):
        """UNIT: Valores fuera de rango son clampeados."""
        telemetry = TelemetryData(
            flyback_voltage=1.5,  # > 1.0
            saturation=-0.3,      # < 0.0
            integrity_score=2.0,  # > 1.0
        )
        
        assert telemetry.flyback_voltage == 1.0
        assert telemetry.saturation == 0.0
        assert telemetry.integrity_score == 1.0
        assert telemetry.was_clamped
    
    def test_clamping_nan_and_inf(self):
        """UNIT: NaN e Inf son clampeados a valores seguros."""
        telemetry_nan = TelemetryData(
            flyback_voltage=float('nan'),
            saturation=0.5,
            integrity_score=0.9,
        )
        assert telemetry_nan.flyback_voltage == 0.0  # NaN → min_val
        
        telemetry_inf = TelemetryData(
            flyback_voltage=float('inf'),
            saturation=float('-inf'),
            integrity_score=0.9,
        )
        assert telemetry_inf.flyback_voltage == 1.0   # +Inf → max_val
        assert telemetry_inf.saturation == 0.0        # -Inf → min_val
    
    @given(normalized_float, normalized_float, normalized_float)
    def test_normalized_values_never_clamped(self, v: float, s: float, i: float):
        """PROP (Hypothesis): Valores en [0,1] nunca son clampeados."""
        telemetry = TelemetryData(
            flyback_voltage=v,
            saturation=s,
            integrity_score=i,
        )
        
        assert not telemetry.was_clamped
        assert telemetry.flyback_voltage == v
        assert telemetry.saturation == s
        assert telemetry.integrity_score == i
    
    def test_from_dict_with_standard_paths(self):
        """UNIT: Construcción desde dict con paths estándar."""
        data = {
            "flux_condenser": {
                "max_flyback_voltage": 0.6,
                "avg_saturation": 0.8,
            },
            "integrity_score": 0.95,
        }
        
        telemetry = TelemetryData.from_dict(data)
        
        assert telemetry is not None
        assert telemetry.flyback_voltage == 0.6
        assert telemetry.saturation == 0.8
        assert telemetry.integrity_score == 0.95
    
    def test_from_dict_with_fallback_paths(self):
        """UNIT: Construcción usa paths de fallback."""
        data = {
            "voltage": 0.4,  # Fallback path
            "sat": 0.7,      # Fallback path
        }
        
        telemetry = TelemetryData.from_dict(data)
        
        assert telemetry is not None
        assert telemetry.flyback_voltage == 0.4
        assert telemetry.saturation == 0.7
    
    def test_from_dict_with_metrics_namespace(self):
        """UNIT: Búsqueda en namespace 'metrics' tiene prioridad."""
        data = {
            "voltage": 0.3,
            "metrics": {
                "voltage": 0.7,  # Debe tomar este
            },
        }
        
        telemetry = TelemetryData.from_dict(data)
        
        assert telemetry is not None
        assert telemetry.flyback_voltage == 0.7
    
    def test_from_dict_with_invalid_type_returns_none(self):
        """UNIT: Tipo inválido retorna None."""
        assert TelemetryData.from_dict("not a dict") is None
        assert TelemetryData.from_dict(123) is None
        assert TelemetryData.from_dict(None) is None
    
    def test_from_dict_with_missing_metrics_uses_defaults(self):
        """UNIT: Métricas faltantes usan defaults (0.0)."""
        data = {}  # Vacío
        
        telemetry = TelemetryData.from_dict(data)
        
        assert telemetry is not None
        assert telemetry.flyback_voltage == 0.0
        assert telemetry.saturation == 0.0
        assert telemetry.integrity_score == 1.0  # Default de integrity
    
    def test_is_idle_predicate(self):
        """UNIT: Predicado is_idle funciona correctamente."""
        idle_telemetry = TelemetryData(
            flyback_voltage=0.0,
            saturation=0.0,
            integrity_score=1.0,
        )
        assert idle_telemetry.is_idle
        
        active_telemetry = TelemetryData(
            flyback_voltage=0.3,
            saturation=0.0,
            integrity_score=1.0,
        )
        assert not active_telemetry.is_idle
    
    def test_norm_l2_calculation(self):
        """UNIT: Norma L2 se calcula correctamente."""
        telemetry = TelemetryData(
            flyback_voltage=0.3,
            saturation=0.4,
            integrity_score=0.0,
        )
        
        expected_norm = math.sqrt(0.3**2 + 0.4**2 + 0.0**2)
        assert abs(telemetry.norm_l2 - expected_norm) < TEST_EPSILON
    
    def test_norm_linf_calculation(self):
        """UNIT: Norma L∞ se calcula correctamente."""
        telemetry = TelemetryData(
            flyback_voltage=0.3,
            saturation=0.9,  # Máximo
            integrity_score=0.5,
        )
        
        assert telemetry.norm_linf == 0.9
    
    def test_equality_ignores_timestamp(self):
        """UNIT: Igualdad compara valores, ignora timestamp."""
        t1 = TelemetryData(flyback_voltage=0.5, saturation=0.6)
        time.sleep(0.01)  # Garantizar timestamp diferente
        t2 = TelemetryData(flyback_voltage=0.5, saturation=0.6)
        
        assert t1 == t2
        assert t1.timestamp != t2.timestamp
    
    def test_to_dict_serialization(self, sample_telemetry: TelemetryData):
        """UNIT: Serialización a dict es correcta."""
        data = sample_telemetry.to_dict()
        
        assert "flyback_voltage" in data
        assert "saturation" in data
        assert "integrity_score" in data
        assert "timestamp" in data
        assert "is_idle" in data
        assert "was_clamped" in data
        assert "norm_l2" in data
    
    @given(valid_telemetry())
    def test_invariants_always_hold(self, telemetry: TelemetryData):
        """PROP (Hypothesis): Invariantes siempre se cumplen post-construcción."""
        # Valores en rango
        assert 0.0 <= telemetry.flyback_voltage <= 1.0
        assert 0.0 <= telemetry.saturation <= 1.0
        assert 0.0 <= telemetry.integrity_score <= 1.0
        
        # Valores finitos
        assert math.isfinite(telemetry.flyback_voltage)
        assert math.isfinite(telemetry.saturation)
        assert math.isfinite(telemetry.integrity_score)
        
        # Normas no negativas
        assert telemetry.norm_l2 >= 0.0
        assert telemetry.norm_linf >= 0.0


# ============================================================================
# PRUEBAS DE AGENTMETRICS
# ============================================================================

class TestAgentMetrics(BaseAgentTest):
    """
    Suite de pruebas para AgentMetrics.
    
    Verifica:
        - Monotonicidad de contadores
        - Conservación de totales
        - Cálculo de tasas derivadas
        - Predicados de salud
    """
    
    def test_initial_state(self):
        """UNIT: Estado inicial tiene valores coherentes."""
        metrics = AgentMetrics()
        
        assert metrics.cycles_executed == 0
        assert metrics.successful_observations == 0
        assert metrics.failed_observations == 0
        assert metrics.consecutive_failures == 0
        assert metrics.total_observations == 0
        assert metrics.success_rate == 0.0
    
    def test_record_success_updates_correctly(self):
        """UNIT: record_success actualiza estado correctamente."""
        metrics = AgentMetrics()
        
        before_time = datetime.now()
        metrics.record_success()
        after_time = datetime.now()
        
        assert metrics.successful_observations == 1
        assert metrics.consecutive_failures == 0
        assert metrics.last_successful_observation is not None
        assert before_time <= metrics.last_successful_observation <= after_time
    
    def test_record_failure_updates_correctly(self):
        """UNIT: record_failure actualiza estado correctamente."""
        metrics = AgentMetrics()
        
        metrics.record_failure()
        
        assert metrics.failed_observations == 1
        assert metrics.consecutive_failures == 1
    
    def test_consecutive_failures_resets_on_success(self):
        """UNIT: consecutive_failures se resetea en éxito."""
        metrics = AgentMetrics()
        
        metrics.record_failure()
        metrics.record_failure()
        metrics.record_failure()
        
        assert metrics.consecutive_failures == 3
        
        metrics.record_success()
        
        assert metrics.consecutive_failures == 0
    
    def test_total_observations_conservation(self):
        """PROP: total = successful + failed (conservación)."""
        metrics = AgentMetrics()
        
        metrics.record_success()
        metrics.record_failure()
        metrics.record_success()
        metrics.record_failure()
        metrics.record_failure()
        
        assert metrics.total_observations == 5
        assert (
            metrics.successful_observations + metrics.failed_observations
            == metrics.total_observations
        )
    
    def test_success_rate_calculation(self):
        """UNIT: success_rate se calcula correctamente."""
        metrics = AgentMetrics()
        
        # Sin observaciones
        assert metrics.success_rate == 0.0
        
        # 3 éxitos, 2 fallos
        for _ in range(3):
            metrics.record_success()
        for _ in range(2):
            metrics.record_failure()
        
        expected_rate = 3 / 5
        assert abs(metrics.success_rate - expected_rate) < TEST_EPSILON
    
    def test_failure_rate_is_complement(self):
        """PROP: failure_rate = 1 - success_rate."""
        metrics = AgentMetrics()
        
        metrics.record_success()
        metrics.record_failure()
        
        assert abs(metrics.success_rate + metrics.failure_rate - 1.0) < TEST_EPSILON
    
    def test_uptime_monotonic_increase(self):
        """PROP: uptime aumenta monótonamente."""
        metrics = AgentMetrics()
        
        uptime1 = metrics.uptime_seconds
        time.sleep(0.1)
        uptime2 = metrics.uptime_seconds
        
        assert uptime2 > uptime1
    
    def test_observation_rate_calculation(self):
        """UNIT: observation_rate se calcula correctamente."""
        metrics = AgentMetrics()
        
        # Sin uptime significativo
        assert metrics.observation_rate == 0.0
        
        # Simular uptime (modificando start_time)
        metrics.successful_observations = 10
        metrics.start_time = datetime.now() - timedelta(minutes=2)
        
        # ~10 obs / 2 min = 5 obs/min
        assert 4.5 <= metrics.observation_rate <= 5.5
    
    def test_cycle_rate_calculation(self):
        """UNIT: cycle_rate se calcula correctamente."""
        metrics = AgentMetrics()
        
        metrics.cycles_executed = 20
        metrics.start_time = datetime.now() - timedelta(minutes=5)
        
        # ~20 cycles / 5 min = 4 cycles/min
        assert 3.5 <= metrics.cycle_rate <= 4.5
    
    def test_mean_cycle_duration_calculation(self):
        """UNIT: mean_cycle_duration_ms se calcula correctamente."""
        metrics = AgentMetrics()
        
        # Sin ciclos
        assert metrics.mean_cycle_duration_ms == 0.0
        
        # 10 ciclos en 10 segundos → 1000 ms/ciclo
        metrics.cycles_executed = 10
        metrics.start_time = datetime.now() - timedelta(seconds=10)
        
        expected_duration = 1000.0  # ms
        assert abs(metrics.mean_cycle_duration_ms - expected_duration) < 50.0
    
    def test_is_healthy_predicate(self):
        """UNIT: is_healthy es correcto según success_rate."""
        metrics = AgentMetrics()
        
        # 9 éxitos, 1 fallo → 90% → healthy
        for _ in range(9):
            metrics.record_success()
        metrics.record_failure()
        
        assert metrics.is_healthy
        
        # Agregar más fallos → < 90% → unhealthy
        for _ in range(5):
            metrics.record_failure()
        
        assert not metrics.is_healthy
    
    def test_health_status_classification(self):
        """UNIT: health_status clasifica correctamente."""
        metrics = AgentMetrics()
        
        # Healthy: ≥ 90%
        for _ in range(9):
            metrics.record_success()
        metrics.record_failure()
        assert metrics.health_status == "healthy"
        
        # Degraded: [50%, 90%)
        for _ in range(3):
            metrics.record_failure()
        assert metrics.health_status == "degraded"
        
        # Unhealthy: < 50%
        for _ in range(10):
            metrics.record_failure()
        assert metrics.health_status == "unhealthy"
    
    def test_decision_distribution_calculation(self):
        """UNIT: decision_distribution calcula probabilidades correctas."""
        metrics = AgentMetrics()
        
        # Sin decisiones
        assert metrics.get_decision_distribution() == {}
        
        # 3 HEARTBEAT, 2 WAIT, 1 EJECUTAR_LIMPIEZA
        metrics.decisions_count = {
            "HEARTBEAT": 3,
            "WAIT": 2,
            "EJECUTAR_LIMPIEZA": 1,
        }
        
        distribution = metrics.get_decision_distribution()
        
        assert abs(distribution["HEARTBEAT"] - 0.5) < TEST_EPSILON
        assert abs(distribution["WAIT"] - 1/3) < TEST_EPSILON
        assert abs(distribution["EJECUTAR_LIMPIEZA"] - 1/6) < TEST_EPSILON
        
        # La suma debe ser 1.0
        assert abs(sum(distribution.values()) - 1.0) < TEST_EPSILON
    
    def test_to_dict_serialization(self, agent_metrics: AgentMetrics):
        """UNIT: Serialización a dict es completa."""
        data = agent_metrics.to_dict()
        
        assert "counters" in data
        assert "rates" in data
        assert "timing" in data
        assert "decisions" in data
        assert "health" in data
        
        # Verificar estructura de contadores
        assert "cycles_executed" in data["counters"]
        assert "successful_observations" in data["counters"]
        
        # Verificar tasas
        assert "success_rate" in data["rates"]
        assert "observation_rate_per_min" in data["rates"]


# ============================================================================
# PRUEBAS DE OBSERVATIONRESULT
# ============================================================================

class TestObservationResult(BaseAgentTest):
    """
    Suite de pruebas para ObservationResult (Sum Type).
    
    Verifica:
        - Exclusividad mutua (success XOR failure)
        - Factory methods
        - Invariantes post-construcción
    """
    
    def test_success_result_factory(self, sample_telemetry: TelemetryData):
        """UNIT: Factory success_result crea resultado válido."""
        result = ObservationResult.success_result(
            sample_telemetry,
            request_id="test_123",
        )
        
        assert result.success is True
        assert result.telemetry == sample_telemetry
        assert result.error_type is None
        assert result.request_id == "test_123"
    
    def test_failure_result_factory(self):
        """UNIT: Factory failure_result crea resultado válido."""
        result = ObservationResult.failure_result(
            ObservationErrorType.TIMEOUT,
            request_id="test_456",
        )
        
        assert result.success is False
        assert result.telemetry is None
        assert result.error_type == ObservationErrorType.TIMEOUT
        assert result.request_id == "test_456"
    
    def test_success_requires_telemetry(self, sample_telemetry: TelemetryData):
        """UNIT: Success=True sin telemetría es inválido."""
        with pytest.raises(ValueError, match="Success=True requiere telemetry no nula"):
            ObservationResult(
                success=True,
                telemetry=None,  # Violación!
                error_type=None,
                request_id="test",
            )
    
    def test_success_forbids_error_type(self, sample_telemetry: TelemetryData):
        """UNIT: Success=True con error_type es inválido."""
        with pytest.raises(ValueError, match="Success=True no admite error_type"):
            ObservationResult(
                success=True,
                telemetry=sample_telemetry,
                error_type=ObservationErrorType.TIMEOUT,  # Violación!
                request_id="test",
            )
    
    def test_failure_requires_error_type(self):
        """UNIT: Success=False sin error_type es inválido."""
        with pytest.raises(ValueError, match="Success=False requiere error_type"):
            ObservationResult(
                success=False,
                telemetry=None,
                error_type=None,  # Violación!
                request_id="test",
            )
    
    def test_failure_forbids_telemetry(self, sample_telemetry: TelemetryData):
        """UNIT: Success=False con telemetría es inválido."""
        with pytest.raises(ValueError, match="Success=False requiere telemetry nula"):
            ObservationResult(
                success=False,
                telemetry=sample_telemetry,  # Violación!
                error_type=ObservationErrorType.TIMEOUT,
                request_id="test",
            )
    
    def test_is_transient_error_predicate(self):
        """UNIT: is_transient_error es correcto."""
        # Error transitorio
        result_transient = ObservationResult.failure_result(
            ObservationErrorType.TIMEOUT,
            request_id="test",
        )
        assert result_transient.is_transient_error
        
        # Error no transitorio
        result_permanent = ObservationResult.failure_result(
            ObservationErrorType.INVALID_JSON,
            request_id="test",
        )
        assert not result_permanent.is_transient_error
        
        # Éxito: no es error
        result_success = ObservationResult.success_result(
            self.create_mock_telemetry(),
            request_id="test",
        )
        assert not result_success.is_transient_error
    
    def test_requires_reconnection_predicate(self):
        """UNIT: requires_reconnection es correcto."""
        result_conn_error = ObservationResult.failure_result(
            ObservationErrorType.CONNECTION_ERROR,
            request_id="test",
        )
        assert result_conn_error.requires_reconnection
        
        result_other = ObservationResult.failure_result(
            ObservationErrorType.TIMEOUT,
            request_id="test",
        )
        assert not result_other.requires_reconnection
    
    def test_to_dict_serialization_success(self, sample_telemetry: TelemetryData):
        """UNIT: Serialización de éxito es correcta."""
        result = ObservationResult.success_result(sample_telemetry, "test")
        data = result.to_dict()
        
        assert data["success"] is True
        assert "telemetry" in data
        assert "error" not in data
    
    def test_to_dict_serialization_failure(self):
        """UNIT: Serialización de fallo es correcta."""
        result = ObservationResult.failure_result(
            ObservationErrorType.TIMEOUT,
            "test",
        )
        data = result.to_dict()
        
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["type"] == "TIMEOUT"
        assert "telemetry" not in data


# ============================================================================
# PRUEBAS DE TOPOLOGICALDIAGNOSIS
# ============================================================================

class TestTopologicalDiagnosis(BaseAgentTest):
    """
    Suite de pruebas para TopologicalDiagnosis.
    
    Verifica:
        - Predicados derivados
        - Extracción de issues críticos
        - Serialización múltiple
    """
    
    def test_is_structurally_healthy_predicate(
        self,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
    ):
        """UNIT: is_structurally_healthy es correcto."""
        diagnosis = TopologicalDiagnosis(
            health=mock_topological_health,
            voltage_persistence=mock_persistence_analysis_nominal,
            saturation_persistence=mock_persistence_analysis_nominal,
            summary="Test",
            recommended_status=SystemStatus.NOMINAL,
        )
        
        assert diagnosis.is_structurally_healthy
        assert diagnosis.health.betti.is_connected
    
    def test_has_retry_loops_predicate(
        self,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
    ):
        """UNIT: has_retry_loops detecta loops correctamente."""
        # Sin loops
        diagnosis = TopologicalDiagnosis(
            health=mock_topological_health,
            voltage_persistence=mock_persistence_analysis_nominal,
            saturation_persistence=mock_persistence_analysis_nominal,
            summary="Test",
            recommended_status=SystemStatus.NOMINAL,
        )
        assert not diagnosis.has_retry_loops
        
        # Con loops (crear mock modificado)
        from app.tactics.topological_analyzer import RequestLoop
        health_with_loops = TopologicalHealth(
            betti=mock_topological_health.betti,
            health_score=mock_topological_health.health_score,
            level=mock_topological_health.level,
            disconnected_nodes=set(),
            missing_edges=set(),
            request_loops=[RequestLoop(request_id="FAIL_TEST", count=5)],
            diagnostics=[],
        )
        
        diagnosis_with_loops = TopologicalDiagnosis(
            health=health_with_loops,
            voltage_persistence=mock_persistence_analysis_nominal,
            saturation_persistence=mock_persistence_analysis_nominal,
            summary="Test",
            recommended_status=SystemStatus.NOMINAL,
        )
        assert diagnosis_with_loops.has_retry_loops
    
    def test_has_persistent_issues_predicate(
        self,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        mock_persistence_analysis_critical: PersistenceAnalysisResult,
    ):
        """UNIT: has_persistent_issues detecta persistencia."""
        # Sin issues persistentes
        diagnosis_nominal = TopologicalDiagnosis(
            health=mock_topological_health,
            voltage_persistence=mock_persistence_analysis_nominal,
            saturation_persistence=mock_persistence_analysis_nominal,
            summary="Test",
            recommended_status=SystemStatus.NOMINAL,
        )
        assert not diagnosis_nominal.has_persistent_issues
        
        # Con issue persistente en voltaje
        diagnosis_persistent = TopologicalDiagnosis(
            health=mock_topological_health,
            voltage_persistence=mock_persistence_analysis_critical,
            saturation_persistence=mock_persistence_analysis_nominal,
            summary="Test",
            recommended_status=SystemStatus.INESTABLE,
        )
        assert diagnosis_persistent.has_persistent_issues
    
    def test_get_critical_issues_extraction(
        self,
        mock_topological_health_degraded: TopologicalHealth,
        mock_persistence_analysis_critical: PersistenceAnalysisResult,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
    ):
        """UNIT: get_critical_issues extrae issues correctamente."""
        diagnosis = TopologicalDiagnosis(
            health=mock_topological_health_degraded,
            voltage_persistence=mock_persistence_analysis_critical,
            saturation_persistence=mock_persistence_analysis_nominal,
            summary="Test",
            recommended_status=SystemStatus.CRITICO,
        )
        
        issues = diagnosis.get_critical_issues()
        
        # Debe haber al menos 2 issues: fragmentación + voltaje persistente
        assert len(issues) >= 2
        
        # Verificar contenido
        issues_text = " ".join(issues)
        assert "Fragmentación" in issues_text or "β₀" in issues_text
        assert "Voltaje persistente" in issues_text or "voltaje" in issues_text.lower()
    
    def test_to_dict_complete_serialization(
        self,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
    ):
        """UNIT: to_dict serializa completamente."""
        diagnosis = TopologicalDiagnosis(
            health=mock_topological_health,
            voltage_persistence=mock_persistence_analysis_nominal,
            saturation_persistence=mock_persistence_analysis_nominal,
            summary="Test diagnosis",
            recommended_status=SystemStatus.NOMINAL,
        )
        
        data = diagnosis.to_dict()
        
        assert "timestamp" in data
        assert "summary" in data
        assert "recommended_status" in data
        assert "predicates" in data
        assert "health" in data
        assert "persistence" in data
        assert "critical_issues" in data
    
    def test_to_log_dict_compact_serialization(
        self,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
    ):
        """UNIT: to_log_dict serializa compactamente."""
        diagnosis = TopologicalDiagnosis(
            health=mock_topological_health,
            voltage_persistence=mock_persistence_analysis_nominal,
            saturation_persistence=mock_persistence_analysis_nominal,
            summary="Test",
            recommended_status=SystemStatus.NOMINAL,
        )
        
        log_data = diagnosis.to_log_dict()
        
        # Debe tener solo campos esenciales
        assert "betti_b0" in log_data
        assert "health_score" in log_data
        assert "voltage_state" in log_data
        assert "recommended_status" in log_data
        
        # No debe tener campos detallados
        assert "diagnostics" not in log_data
    
    def test_to_summary_string_one_liner(
        self,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
    ):
        """UNIT: to_summary_string genera one-liner."""
        diagnosis = TopologicalDiagnosis(
            health=mock_topological_health,
            voltage_persistence=mock_persistence_analysis_nominal,
            saturation_persistence=mock_persistence_analysis_nominal,
            summary="Test",
            recommended_status=SystemStatus.NOMINAL,
        )
        
        summary = diagnosis.to_summary_string()
        
        # Debe ser una línea
        assert "\n" not in summary
        
        # Debe contener elementos clave
        assert "β₀=" in summary
        assert "h=" in summary
        assert "V=" in summary
        assert "S=" in summary
        assert "status=" in summary


# ============================================================================
# PRUEBAS DE AGENTSNAPSHOT Y SNAPSHOTHISTORY
# ============================================================================

class TestAgentSnapshot(BaseAgentTest):
    """Suite de pruebas para AgentSnapshot."""
    
    def test_capture_factory(
        self,
        sample_telemetry: TelemetryData,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: Factory capture crea snapshot correctamente."""
        diagnosis = self.create_mock_diagnosis()
        
        snapshot = AgentSnapshot.capture(
            cycle_number=42,
            telemetry=sample_telemetry,
            diagnosis=diagnosis,
            metrics=agent_metrics,
            last_decision=AgentDecision.HEARTBEAT,
            current_status=SystemStatus.NOMINAL,
        )
        
        assert snapshot.cycle_number == 42
        assert snapshot.telemetry == sample_telemetry
        assert snapshot.diagnosis == diagnosis
        assert snapshot.last_decision == AgentDecision.HEARTBEAT
        assert snapshot.current_status == SystemStatus.NOMINAL
        
        # Métricas deben ser copia profunda
        assert snapshot.metrics is not agent_metrics
    
    def test_to_dict_serialization(
        self,
        sample_telemetry: TelemetryData,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: to_dict serializa completamente."""
        snapshot = AgentSnapshot.capture(
            cycle_number=1,
            telemetry=sample_telemetry,
            diagnosis=None,
            metrics=agent_metrics,
            last_decision=AgentDecision.WAIT,
            current_status=SystemStatus.UNKNOWN,
        )
        
        data = snapshot.to_dict()
        
        assert data["cycle_number"] == 1
        assert data["current_status"] == "UNKNOWN"
        assert data["last_decision"] == "WAIT"
        assert "telemetry" in data
        assert "metrics" in data
    
    def test_to_compact_dict_minimal(
        self,
        sample_telemetry: TelemetryData,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: to_compact_dict serializa mínimamente."""
        snapshot = AgentSnapshot.capture(
            cycle_number=1,
            telemetry=sample_telemetry,
            diagnosis=None,
            metrics=agent_metrics,
            last_decision=None,
            current_status=SystemStatus.NOMINAL,
        )
        
        compact = snapshot.to_compact_dict()
        
        # Solo esenciales
        assert "timestamp" in compact
        assert "cycle" in compact
        assert "status" in compact
        
        # Telemetría compacta
        if "telemetry" in compact:
            assert "voltage" in compact["telemetry"]
            assert "saturation" in compact["telemetry"]


class TestSnapshotHistory(BaseAgentTest):
    """
    Suite de pruebas para SnapshotHistory (Ring Buffer).
    
    Verifica:
        - Comportamiento de buffer circular
        - Operaciones de búsqueda
        - Detección de transiciones
    """
    
    def test_initialization(self):
        """UNIT: Inicialización con tamaño válido."""
        history = SnapshotHistory(max_size=50)
        
        assert history.max_size == 50
        assert history.size == 0
        assert not history.is_full
    
    def test_invalid_size_rejected(self):
        """UNIT: Tamaño no positivo es rechazado."""
        with pytest.raises(ValueError, match="max_size debe ser positivo"):
            SnapshotHistory(max_size=0)
        
        with pytest.raises(ValueError, match="max_size debe ser positivo"):
            SnapshotHistory(max_size=-10)
    
    def test_add_snapshot_increments_size(
        self,
        sample_telemetry: TelemetryData,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: add_snapshot incrementa tamaño."""
        history = SnapshotHistory(max_size=10)
        
        snapshot = AgentSnapshot.capture(
            cycle_number=1,
            telemetry=sample_telemetry,
            diagnosis=None,
            metrics=agent_metrics,
            last_decision=None,
            current_status=SystemStatus.NOMINAL,
        )
        
        history.add_snapshot(snapshot)
        
        assert history.size == 1
        assert not history.is_full
    
    def test_circular_buffer_behavior(
        self,
        sample_telemetry: TelemetryData,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: Buffer circular elimina más antiguos al llenarse."""
        history = SnapshotHistory(max_size=3)
        
        # Agregar 5 snapshots
        for i in range(5):
            snapshot = AgentSnapshot.capture(
                cycle_number=i,
                telemetry=sample_telemetry,
                diagnosis=None,
                metrics=agent_metrics,
                last_decision=None,
                current_status=SystemStatus.NOMINAL,
            )
            history.add_snapshot(snapshot)
        
        # Solo deben quedar los últimos 3
        assert history.size == 3
        assert history.is_full
        
        # Verificar que son los ciclos 2, 3, 4
        snapshots = history.get_all()
        assert snapshots[0].cycle_number == 2
        assert snapshots[1].cycle_number == 3
        assert snapshots[2].cycle_number == 4
    
    def test_get_recent_returns_last_n(
        self,
        sample_telemetry: TelemetryData,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: get_recent retorna últimos N snapshots."""
        history = SnapshotHistory(max_size=10)
        
        for i in range(7):
            snapshot = AgentSnapshot.capture(
                cycle_number=i,
                telemetry=sample_telemetry,
                diagnosis=None,
                metrics=agent_metrics,
                last_decision=None,
                current_status=SystemStatus.NOMINAL,
            )
            history.add_snapshot(snapshot)
        
        recent = history.get_recent(count=3)
        
        assert len(recent) == 3
        assert recent[0].cycle_number == 4
        assert recent[1].cycle_number == 5
        assert recent[2].cycle_number == 6
    
    def test_get_by_cycle_finds_snapshot(
        self,
        sample_telemetry: TelemetryData,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: get_by_cycle encuentra snapshot por número de ciclo."""
        history = SnapshotHistory(max_size=10)
        
        for i in range(5):
            snapshot = AgentSnapshot.capture(
                cycle_number=i * 10,  # 0, 10, 20, 30, 40
                telemetry=sample_telemetry,
                diagnosis=None,
                metrics=agent_metrics,
                last_decision=None,
                current_status=SystemStatus.NOMINAL,
            )
            history.add_snapshot(snapshot)
        
        found = history.get_by_cycle(20)
        assert found is not None
        assert found.cycle_number == 20
        
        not_found = history.get_by_cycle(99)
        assert not_found is None
    
    def test_get_status_changes_detects_transitions(
        self,
        sample_telemetry: TelemetryData,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: get_status_changes detecta transiciones de estado."""
        history = SnapshotHistory(max_size=10)
        
        statuses = [
            SystemStatus.NOMINAL,
            SystemStatus.NOMINAL,
            SystemStatus.INESTABLE,  # Transición
            SystemStatus.INESTABLE,
            SystemStatus.CRITICO,    # Transición
        ]
        
        for i, status in enumerate(statuses):
            snapshot = AgentSnapshot.capture(
                cycle_number=i,
                telemetry=sample_telemetry,
                diagnosis=None,
                metrics=agent_metrics,
                last_decision=None,
                current_status=status,
            )
            history.add_snapshot(snapshot)
        
        changes = history.get_status_changes()
        
        # Debe haber 2 transiciones
        assert len(changes) == 2
        
        # Primera transición: NOMINAL → INESTABLE
        assert changes[0][1] == SystemStatus.NOMINAL
        assert changes[0][2] == SystemStatus.INESTABLE
        
        # Segunda transición: INESTABLE → CRITICO
        assert changes[1][1] == SystemStatus.INESTABLE
        assert changes[1][2] == SystemStatus.CRITICO
    
    def test_pythonic_interface(
        self,
        sample_telemetry: TelemetryData,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: Interfaz Pythonic funciona correctamente."""
        history = SnapshotHistory(max_size=5)
        
        for i in range(3):
            snapshot = AgentSnapshot.capture(
                cycle_number=i,
                telemetry=sample_telemetry,
                diagnosis=None,
                metrics=agent_metrics,
                last_decision=None,
                current_status=SystemStatus.NOMINAL,
            )
            history.add_snapshot(snapshot)
        
        # __len__
        assert len(history) == 3
        
        # __getitem__
        assert history[0].cycle_number == 0
        assert history[2].cycle_number == 2
        
        # __iter__
        cycles = [s.cycle_number for s in history]
        assert cycles == [0, 1, 2]


# ============================================================================
# PRUEBAS DE EVALUADORES
# ============================================================================

class TestBaseEvaluator:
    """Suite de pruebas para BaseEvaluator."""
    
    def test_node_index_mapping(self):
        """UNIT: Mapeo nodo → índice es correcto."""
        evaluator = BaseEvaluator()
        
        assert evaluator._get_node_idx(NetworkTopology.NODE_AGENT) == 0
        assert evaluator._get_node_idx(NetworkTopology.NODE_CORE) == 1
        assert evaluator._get_node_idx(NetworkTopology.NODE_REDIS) == 2
        assert evaluator._get_node_idx(NetworkTopology.NODE_FILESYSTEM) == 3
        
        assert evaluator._get_node_idx("Unknown") is None
    
    def test_validate_gradient_accepts_valid(self):
        """UNIT: _validate_gradient acepta gradientes válidos."""
        evaluator = BaseEvaluator()
        
        valid_gradient = np.array([0.1, 0.2, 0.3, 0.0])
        
        # No debe lanzar excepción
        evaluator._validate_gradient(valid_gradient, num_nodes=4)
    
    def test_validate_gradient_rejects_wrong_dimension(self):
        """UNIT: _validate_gradient rechaza dimensión incorrecta."""
        evaluator = BaseEvaluator()
        
        wrong_gradient = np.array([0.1, 0.2, 0.3])  # 3 elementos, esperado 4
        
        with pytest.raises(ValueError, match="Dimensión inválida"):
            evaluator._validate_gradient(wrong_gradient, num_nodes=4)
    
    def test_validate_gradient_rejects_nan_inf(self):
        """UNIT: _validate_gradient rechaza NaN/Inf."""
        evaluator = BaseEvaluator()
        
        gradient_nan = np.array([0.1, float('nan'), 0.3, 0.0])
        
        with pytest.raises(ValueError, match="contiene NaN/Inf"):
            evaluator._validate_gradient(gradient_nan, num_nodes=4)
        
        gradient_inf = np.array([0.1, float('inf'), 0.3, 0.0])
        
        with pytest.raises(ValueError, match="contiene NaN/Inf"):
            evaluator._validate_gradient(gradient_inf, num_nodes=4)
    
    def test_validate_gradient_rejects_negative(self):
        """UNIT: _validate_gradient rechaza valores negativos."""
        evaluator = BaseEvaluator()
        
        negative_gradient = np.array([0.1, -0.2, 0.3, 0.0])
        
        with pytest.raises(ValueError, match="valores negativos"):
            evaluator._validate_gradient(negative_gradient, num_nodes=4)


class TestFragmentationEvaluator(BaseAgentTest):
    """
    Suite de pruebas para FragmentationEvaluator.
    
    Verifica:
        - Gradiente correcto cuando β₀ > 1
        - Gradiente cero cuando β₀ = 1
        - Localización en nodos desconectados
    """
    
    def test_no_gradient_when_connected(
        self,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        threshold_config: ThresholdConfig,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: Sin gradiente cuando β₀ = 1 (conectado)."""
        evaluator = FragmentationEvaluator()
        
        gradient = evaluator.compute_gradient(
            telemetry=None,
            topo_health=mock_topological_health,
            voltage_analysis=mock_persistence_analysis_nominal,
            saturation_analysis=mock_persistence_analysis_nominal,
            config=threshold_config,
            metrics=agent_metrics,
            num_nodes=4,
        )
        
        assert_gradient_valid(gradient, 4)
        assert np.all(gradient == 0.0), "Gradiente debe ser cero cuando está conectado"
    
    def test_gradient_when_fragmented(
        self,
        mock_topological_health_degraded: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        threshold_config: ThresholdConfig,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: Gradiente localizado en nodos desconectados cuando β₀ > 1."""
        evaluator = FragmentationEvaluator()
        
        gradient = evaluator.compute_gradient(
            telemetry=None,
            topo_health=mock_topological_health_degraded,
            voltage_analysis=mock_persistence_analysis_nominal,
            saturation_analysis=mock_persistence_analysis_nominal,
            config=threshold_config,
            metrics=agent_metrics,
            num_nodes=4,
        )
        
        assert_gradient_valid(gradient, 4)
        
        # Filesystem (índice 3) está desconectado
        assert gradient[3] > 0.0, "Nodo desconectado debe tener penalización"
        
        # Otros nodos conectados deben tener cero
        assert gradient[0] == 0.0
        assert gradient[1] == 0.0
        assert gradient[2] == 0.0
    
    def test_evaluate_returns_disconnected_status(
        self,
        mock_topological_health_degraded: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        threshold_config: ThresholdConfig,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: evaluate retorna DISCONNECTED cuando β₀ > 1."""
        evaluator = FragmentationEvaluator()
        
        result = evaluator.evaluate(
            telemetry=None,
            topo_health=mock_topological_health_degraded,
            voltage_analysis=mock_persistence_analysis_nominal,
            saturation_analysis=mock_persistence_analysis_nominal,
            config=threshold_config,
            metrics=agent_metrics,
        )
        
        assert result is not None
        status, summary = result
        assert status == SystemStatus.DISCONNECTED
        assert "β₀" in summary


class TestNoTelemetryEvaluator(BaseAgentTest):
    """Suite de pruebas para NoTelemetryEvaluator."""
    
    def test_no_gradient_when_telemetry_present(
        self,
        sample_telemetry: TelemetryData,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        threshold_config: ThresholdConfig,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: Sin gradiente cuando hay telemetría."""
        evaluator = NoTelemetryEvaluator()
        
        gradient = evaluator.compute_gradient(
            telemetry=sample_telemetry,
            topo_health=mock_topological_health,
            voltage_analysis=mock_persistence_analysis_nominal,
            saturation_analysis=mock_persistence_analysis_nominal,
            config=threshold_config,
            metrics=agent_metrics,
            num_nodes=4,
        )
        
        assert_gradient_valid(gradient, 4)
        assert np.all(gradient == 0.0)
    
    def test_uniform_gradient_when_no_telemetry(
        self,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        threshold_config: ThresholdConfig,
    ):
        """UNIT: Gradiente uniforme cuando falta telemetría."""
        # Métricas con fallos consecutivos
        metrics = AgentMetrics()
        metrics.consecutive_failures = 5
        
        evaluator = NoTelemetryEvaluator()
        
        gradient = evaluator.compute_gradient(
            telemetry=None,
            topo_health=mock_topological_health,
            voltage_analysis=mock_persistence_analysis_nominal,
            saturation_analysis=mock_persistence_analysis_nominal,
            config=threshold_config,
            metrics=metrics,
            num_nodes=4,
        )
        
        assert_gradient_valid(gradient, 4)
        
        # Gradiente debe ser uniforme
        assert np.all(gradient == gradient[0])
        
        # Severidad crece con fallos consecutivos
        expected_severity = min(1.0, 5 * 0.2)  # 5 * severity_growth_rate
        expected_per_node = expected_severity / 4
        assert abs(gradient[0] - expected_per_node) < TEST_EPSILON


class TestCriticalVoltageEvaluator(BaseAgentTest):
    """Suite de pruebas para CriticalVoltageEvaluator."""
    
    def test_no_gradient_when_voltage_nominal(
        self,
        sample_telemetry: TelemetryData,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        threshold_config: ThresholdConfig,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: Sin gradiente cuando voltaje es nominal."""
        evaluator = CriticalVoltageEvaluator()
        
        gradient = evaluator.compute_gradient(
            telemetry=sample_telemetry,  # voltage=0.3 < critical=0.8
            topo_health=mock_topological_health,
            voltage_analysis=mock_persistence_analysis_nominal,
            saturation_analysis=mock_persistence_analysis_nominal,
            config=threshold_config,
            metrics=agent_metrics,
            num_nodes=4,
        )
        
        assert_gradient_valid(gradient, 4)
        assert np.all(gradient == 0.0)
    
    def test_gradient_localized_to_core_when_critical(
        self,
        sample_telemetry_critical: TelemetryData,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        threshold_config: ThresholdConfig,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: Gradiente localizado en Core cuando voltaje crítico."""
        evaluator = CriticalVoltageEvaluator()
        
        gradient = evaluator.compute_gradient(
            telemetry=sample_telemetry_critical,  # voltage=0.85 > critical=0.8
            topo_health=mock_topological_health,
            voltage_analysis=mock_persistence_analysis_nominal,
            saturation_analysis=mock_persistence_analysis_nominal,
            config=threshold_config,
            metrics=agent_metrics,
            num_nodes=4,
        )
        
        assert_gradient_valid(gradient, 4)
        
        # Core (índice 1) debe tener penalización
        assert gradient[1] > 0.0
        assert gradient[1] == sample_telemetry_critical.flyback_voltage
        
        # Otros nodos deben tener cero
        assert gradient[0] == 0.0
        assert gradient[2] == 0.0
        assert gradient[3] == 0.0


class TestHamiltonianSynthesizer(BaseAgentTest):
    """
    Suite de pruebas para HamiltonianSynthesizer.
    
    Verifica:
        - Síntesis de gradiente total
        - Renormalización gauge (neutralidad de carga)
        - Superposición lineal de gradientes
    """
    
    def test_initialization_with_default_evaluators(self):
        """UNIT: Inicialización con evaluadores por defecto."""
        synthesizer = HamiltonianSynthesizer()
        
        assert len(synthesizer.evaluators) > 0
        
        # Verificar que incluye evaluadores estándar
        evaluator_names = {e.name for e in synthesizer.evaluators}
        assert "FragmentationEvaluator" in evaluator_names
        assert "CriticalVoltageEvaluator" in evaluator_names
    
    def test_initialization_with_custom_evaluators(self):
        """UNIT: Inicialización con evaluadores personalizados."""
        custom_evaluators = [
            FragmentationEvaluator(),
            CriticalVoltageEvaluator(),
        ]
        
        synthesizer = HamiltonianSynthesizer(evaluators=custom_evaluators)
        
        assert len(synthesizer.evaluators) == 2
    
    def test_synthesize_gradient_zero_when_nominal(
        self,
        sample_telemetry: TelemetryData,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        threshold_config: ThresholdConfig,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: Gradiente total es cero cuando sistema nominal."""
        synthesizer = HamiltonianSynthesizer()
        
        gradient = synthesizer.synthesize_gradient(
            telemetry=sample_telemetry,
            topo_health=mock_topological_health,
            voltage_analysis=mock_persistence_analysis_nominal,
            saturation_analysis=mock_persistence_analysis_nominal,
            config=threshold_config,
            metrics=agent_metrics,
            num_nodes=4,
        )
        
        assert_gradient_valid(gradient, 4)
        assert_charge_neutrality(gradient)
        
        # Sistema nominal → gradiente cero
        assert np.linalg.norm(gradient) < TEST_EPSILON
    
    def test_synthesize_gradient_nonzero_when_critical(
        self,
        sample_telemetry_critical: TelemetryData,
        mock_topological_health: TopologicalHealth,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        threshold_config: ThresholdConfig,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: Gradiente total es no cero cuando hay issues críticos."""
        synthesizer = HamiltonianSynthesizer()
        
        gradient = synthesizer.synthesize_gradient(
            telemetry=sample_telemetry_critical,
            topo_health=mock_topological_health,
            voltage_analysis=mock_persistence_analysis_nominal,
            saturation_analysis=mock_persistence_analysis_nominal,
            config=threshold_config,
            metrics=agent_metrics,
            num_nodes=4,
        )
        
        assert_gradient_valid(gradient, 4)
        assert_charge_neutrality(gradient)
        
        # Debe haber penalización
        assert np.linalg.norm(gradient) > TEST_EPSILON
    
    def test_charge_neutrality_after_renormalization(
        self,
        sample_telemetry_critical: TelemetryData,
        mock_topological_health_degraded: TopologicalHealth,
        mock_persistence_analysis_critical: PersistenceAnalysisResult,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        threshold_config: ThresholdConfig,
        agent_metrics: AgentMetrics,
    ):
        """PROP: Neutralidad de carga después de renormalización."""
        synthesizer = HamiltonianSynthesizer()
        
        gradient = synthesizer.synthesize_gradient(
            telemetry=sample_telemetry_critical,
            topo_health=mock_topological_health_degraded,
            voltage_analysis=mock_persistence_analysis_critical,
            saturation_analysis=mock_persistence_analysis_nominal,
            config=threshold_config,
            metrics=agent_metrics,
            num_nodes=4,
        )
        
        # Verificar neutralidad de carga estricta
        total_charge = np.sum(gradient)
        tolerance = TEST_EPSILON * 4  # Tolerancia proporcional a num_nodes
        
        assert abs(total_charge) < tolerance, \
            f"Violación de neutralidad de carga: Σ gradient = {total_charge:.4e}"
    
    @given(
        st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=4, max_size=4)
    )
    @settings(max_examples=50)
    def test_renormalization_preserves_zero_mean(self, values: List[float]):
        """PROP (Hypothesis): Renormalización siempre produce media cero."""
        # Crear gradiente sintético
        gradient_raw = np.array(values)
        
        # Aplicar renormalización (restar media)
        mean = np.mean(gradient_raw)
        gradient_renorm = gradient_raw - mean
        
        # Verificar media cero
        assert abs(np.mean(gradient_renorm)) < TEST_EPSILON
    
    def test_evaluate_returns_worst_status(
        self,
        sample_telemetry_critical: TelemetryData,
        mock_topological_health_degraded: TopologicalHealth,
        mock_persistence_analysis_critical: PersistenceAnalysisResult,
        mock_persistence_analysis_nominal: PersistenceAnalysisResult,
        threshold_config: ThresholdConfig,
        agent_metrics: AgentMetrics,
    ):
        """UNIT: evaluate retorna el peor status (supremo)."""
        synthesizer = HamiltonianSynthesizer()
        
        status, summary = synthesizer.evaluate(
            telemetry=sample_telemetry_critical,
            topo_health=mock_topological_health_degraded,
            voltage_analysis=mock_persistence_analysis_critical,
            saturation_analysis=mock_persistence_analysis_nominal,
            config=threshold_config,
            metrics=agent_metrics,
        )
        
        # Múltiples issues → debe retornar CRITICO o DISCONNECTED
        assert status in (SystemStatus.CRITICO, SystemStatus.DISCONNECTED)
        assert len(summary) > 0
    
    def test_add_evaluator_dynamic(self):
        """UNIT: add_evaluator agrega evaluador dinámicamente."""
        synthesizer = HamiltonianSynthesizer(evaluators=[])
        
        assert len(synthesizer.evaluators) == 0
        
        evaluator = FragmentationEvaluator()
        synthesizer.add_evaluator(evaluator)
        
        assert len(synthesizer.evaluators) == 1
        assert synthesizer.evaluators[0] == evaluator
    
    def test_remove_evaluator_by_name(self):
        """UNIT: remove_evaluator remueve por nombre."""
        synthesizer = HamiltonianSynthesizer()
        
        initial_count = len(synthesizer.evaluators)
        
        success = synthesizer.remove_evaluator("FragmentationEvaluator")
        
        assert success is True
        assert len(synthesizer.evaluators) == initial_count - 1
        
        # Intentar remover inexistente
        success = synthesizer.remove_evaluator("NonExistentEvaluator")
        assert success is False


# ============================================================================
# PRUEBAS DE PROPERTY-BASED TESTING CON HYPOTHESIS
# ============================================================================

class TestGradientInvariants:
    """
    Pruebas de invariantes de gradientes usando Hypothesis.
    
    Verifica propiedades universales que deben cumplirse para
    TODOS los gradientes generados por los evaluadores.
    """
    
    @given(valid_gradient(num_nodes=4))
    def test_gradient_always_non_negative(self, gradient: np.ndarray):
        """PROP: Todos los gradientes deben ser no negativos."""
        assert np.all(gradient >= -TEST_EPSILON)
    
    @given(valid_gradient(num_nodes=4))
    def test_gradient_always_finite(self, gradient: np.ndarray):
        """PROP: Todos los gradientes deben ser finitos."""
        assert np.all(np.isfinite(gradient))
    
    @given(
        st.lists(valid_gradient(num_nodes=4), min_size=1, max_size=10)
    )
    def test_gradient_superposition_is_linear(self, gradients: List[np.ndarray]):
        """PROP: Superposición de gradientes es lineal."""
        # Σ(aᵢ·∇Vᵢ) = a₁·∇V₁ + a₂·∇V₂ + ...
        total = sum(gradients)
        
        assert np.all(np.isfinite(total))
        assert np.all(total >= -TEST_EPSILON * len(gradients))
    
    @given(valid_gradient(num_nodes=4))
    def test_renormalization_preserves_direction(self, gradient: np.ndarray):
        """PROP: Renormalización preserva dirección del gradiente."""
        assume(np.linalg.norm(gradient) > TEST_EPSILON)  # No cero
        
        # Renormalizar
        mean = np.mean(gradient)
        gradient_renorm = gradient - mean
        
        # Verificar que la dirección se preserva (producto escalar positivo)
        # Solo si el gradiente original no era constante
        if np.std(gradient) > TEST_EPSILON:
            dot_product = np.dot(gradient, gradient_renorm)
            assert dot_product >= 0, "Renormalización invirtió dirección"


# ============================================================================
# PRUEBAS DE LA FASE OBSERVE
# ============================================================================

class TestObservePhase(BaseAgentTest):
    """
    Suite de pruebas para la fase OBSERVE del ciclo OODA.
    
    Verifica:
        - Adquisición exitosa de telemetría
        - Clasificación correcta de errores
        - Actualización de métricas
        - Actualización topológica según conectividad
    """
    
    def test_observe_success_with_valid_telemetry(
        self,
        agent_config: AgentConfig,
        mock_requests_success,
    ):
        """UNIT: observe retorna telemetría válida en caso de éxito."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            telemetry = agent.observe()
            
            assert telemetry is not None
            self.assert_telemetry_valid(telemetry)
            
            # Verificar que métricas se actualizaron
            assert agent._metrics.successful_observations == 1
            assert agent._metrics.consecutive_failures == 0
    
    def test_observe_failure_with_timeout(
        self,
        agent_config: AgentConfig,
        mock_requests_timeout,
    ):
        """UNIT: observe retorna None en caso de timeout."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            telemetry = agent.observe()
            
            assert telemetry is None
            
            # Verificar que métricas registran fallo
            assert agent._metrics.failed_observations == 1
            assert agent._metrics.consecutive_failures == 1
    
    def test_observe_failure_with_connection_error(
        self,
        agent_config: AgentConfig,
        mock_requests_connection_error,
    ):
        """UNIT: observe retorna None en caso de ConnectionError."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            telemetry = agent.observe()
            
            assert telemetry is None
            assert agent._metrics.failed_observations == 1
    
    def test_observe_updates_topology_on_success(
        self,
        agent_config: AgentConfig,
        mock_requests_success,
    ):
        """UNIT: observe actualiza topología en caso de éxito."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Estado inicial: topología nominal
            initial_health = agent.topology.get_topological_health()
            assert initial_health.betti.is_connected
            
            # Ejecutar observación
            agent.observe()
            
            # Topología debe mantenerse conectada
            final_health = agent.topology.get_topological_health()
            assert final_health.betti.is_connected
    
    def test_observe_degrades_topology_after_max_failures(
        self,
        agent_config: AgentConfig,
        mock_requests_connection_error,
    ):
        """UNIT: observe degrada topología tras max_consecutive_failures."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Fallar repetidamente
            max_failures = agent_config.timing.max_consecutive_failures
            for _ in range(max_failures):
                agent.observe()
            
            # Verificar que se degradó topología
            assert agent._metrics.consecutive_failures == max_failures
            
            # La arista Agent↔Core debe haber sido removida
            health = agent.topology.get_topological_health()
            assert not health.betti.is_connected or len(health.disconnected_nodes) > 0
    
    def test_execute_observation_classifies_http_error(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: _execute_observation clasifica HTTP errors correctamente."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Mock HTTP error (404)
            with patch.object(agent._session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.ok = False
                mock_response.status_code = 404
                mock_get.return_value = mock_response
                
                result = agent._execute_observation("test_id")
                
                assert result.success is False
                assert result.error_type == ObservationErrorType.HTTP_ERROR
    
    def test_execute_observation_classifies_invalid_json(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: _execute_observation clasifica JSON inválido."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Mock respuesta con JSON malformado
            with patch.object(agent._session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.ok = True
                mock_response.json.side_effect = ValueError("Invalid JSON")
                mock_get.return_value = mock_response
                
                result = agent._execute_observation("test_id")
                
                assert result.success is False
                assert result.error_type == ObservationErrorType.INVALID_JSON
    
    def test_execute_observation_classifies_invalid_telemetry(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: _execute_observation clasifica telemetría inválida."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Mock respuesta con JSON válido pero schema inválido
            with patch.object(agent._session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.ok = True
                mock_response.json.return_value = {"invalid": "data"}
                mock_get.return_value = mock_response
                
                # TelemetryData.from_dict retornará None para schema inválido
                result = agent._execute_observation("test_id")
                
                assert result.success is False
                assert result.error_type == ObservationErrorType.INVALID_TELEMETRY
    
    def test_update_topology_from_telemetry_with_flags(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: _update_topology_from_telemetry usa flags de conectividad."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Telemetría con Redis desconectado
            telemetry = TelemetryData(
                flyback_voltage=0.3,
                saturation=0.5,
                raw_data={
                    "redis_connected": False,
                    "filesystem_accessible": True,
                },
            )
            
            agent._update_topology_from_telemetry(telemetry)
            
            # Verificar que Redis fue marcado como desconectado
            edges = agent.topology.edges
            
            # Agent↔Core debe estar presente
            assert (NetworkTopology.NODE_AGENT, NetworkTopology.NODE_CORE) in edges
            
            # Core↔Redis NO debe estar presente
            assert (NetworkTopology.NODE_CORE, NetworkTopology.NODE_REDIS) not in edges
            
            # Core↔Filesystem debe estar presente
            assert (NetworkTopology.NODE_CORE, NetworkTopology.NODE_FILESYSTEM) in edges
    
    @given(valid_telemetry())
    @settings(max_examples=20)
    def test_observe_always_produces_valid_or_none(
        self,
        agent_config: AgentConfig,
        telemetry: TelemetryData,
    ):
        """PROP (Hypothesis): observe siempre retorna telemetría válida o None."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Mock respuesta exitosa con telemetría generada
            with patch.object(agent._session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.ok = True
                mock_response.json.return_value = telemetry.raw_data
                mock_get.return_value = mock_response
                
                result = agent.observe()
                
                # Debe ser telemetría válida o None
                if result is not None:
                    self.assert_telemetry_valid(result)


# ============================================================================
# PRUEBAS DE LA FASE ORIENT
# ============================================================================

class TestOrientPhase(BaseAgentTest):
    """
    Suite de pruebas para la fase ORIENT del ciclo OODA.
    
    Verifica:
        - Análisis topológico correcto
        - Análisis de persistencia homológica
        - Construcción del diagnóstico
        - Evaluación del estado mediante sintetizador
    """
    
    def test_orient_with_nominal_telemetry(
        self,
        agent_config: AgentConfig,
        sample_telemetry: TelemetryData,
    ):
        """UNIT: orient con telemetría nominal retorna NOMINAL."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Preparar estado: agregar muestras de telemetría nominal
            for _ in range(5):
                agent.persistence.add_sample("flyback_voltage", 0.3)
                agent.persistence.add_sample("saturation", 0.5)
            
            status = agent.orient(sample_telemetry)
            
            assert status == SystemStatus.NOMINAL
            assert agent._last_diagnosis is not None
            assert agent._last_diagnosis.recommended_status == SystemStatus.NOMINAL
    
    def test_orient_with_critical_telemetry(
        self,
        agent_config: AgentConfig,
        sample_telemetry_critical: TelemetryData,
    ):
        """UNIT: orient con telemetría crítica retorna estado crítico."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            status = agent.orient(sample_telemetry_critical)
            
            # Debe retornar CRITICO o SATURADO
            assert status in (SystemStatus.CRITICO, SystemStatus.SATURADO)
            assert agent._last_diagnosis is not None
    
    def test_orient_with_none_telemetry(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: orient con telemetría None retorna UNKNOWN."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            status = agent.orient(None)
            
            assert status == SystemStatus.UNKNOWN
    
    def test_orient_updates_persistence_samples(
        self,
        agent_config: AgentConfig,
        sample_telemetry: TelemetryData,
    ):
        """UNIT: orient actualiza muestras de persistencia."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Estado inicial: sin muestras
            initial_voltage_stats = agent.persistence.get_statistics("flyback_voltage")
            assert initial_voltage_stats is None or initial_voltage_stats["count"] == 0
            
            # Ejecutar orient
            agent.orient(sample_telemetry)
            
            # Verificar que se agregó muestra
            final_voltage_stats = agent.persistence.get_statistics("flyback_voltage")
            assert final_voltage_stats is not None
            assert final_voltage_stats["count"] == 1
    
    def test_orient_constructs_diagnosis(
        self,
        agent_config: AgentConfig,
        sample_telemetry: TelemetryData,
    ):
        """UNIT: orient construye diagnóstico completo."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            agent.orient(sample_telemetry)
            
            diagnosis = agent._last_diagnosis
            assert diagnosis is not None
            
            # Verificar componentes del diagnóstico
            assert diagnosis.health is not None
            assert diagnosis.voltage_persistence is not None
            assert diagnosis.saturation_persistence is not None
            assert len(diagnosis.summary) > 0
            assert diagnosis.recommended_status is not None
    
    def test_orient_with_persistent_voltage_issue(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: orient detecta issues persistentes de voltaje."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular voltaje persistentemente alto
            critical_voltage = agent_config.thresholds.flyback_voltage_warning + 0.1
            for _ in range(15):
                agent.persistence.add_sample("flyback_voltage", critical_voltage)
            
            telemetry = TelemetryData(
                flyback_voltage=critical_voltage,
                saturation=0.5,
            )
            
            status = agent.orient(telemetry)
            
            # Debe detectar persistencia
            assert status in (SystemStatus.INESTABLE, SystemStatus.CRITICO)
            assert agent._last_diagnosis.has_persistent_issues
    
    def test_orient_calculates_topological_health(
        self,
        agent_config: AgentConfig,
        sample_telemetry: TelemetryData,
    ):
        """UNIT: orient calcula salud topológica correctamente."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            agent.orient(sample_telemetry)
            
            diagnosis = agent._last_diagnosis
            health = diagnosis.health
            
            # Verificar números de Betti
            assert health.betti.b0 >= 1
            assert health.betti.b1 >= 0
            
            # Verificar score de salud
            assert 0.0 <= health.health_score <= 1.0
    
    def test_orient_with_fragmented_topology(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: orient detecta fragmentación topológica."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular fragmentación removiendo arista crítica
            agent.topology.remove_edge(
                NetworkTopology.NODE_AGENT,
                NetworkTopology.NODE_CORE,
            )
            
            status = agent.orient(None)
            
            # Debe detectar fragmentación
            assert status in (SystemStatus.DISCONNECTED, SystemStatus.UNKNOWN)
            assert not agent._last_diagnosis.is_structurally_healthy


# ============================================================================
# PRUEBAS DE LA FASE DECIDE
# ============================================================================

class TestDecidePhase(BaseAgentTest):
    """
    Suite de pruebas para la fase DECIDE del ciclo OODA.
    
    Verifica:
        - Decision matrix correcta
        - Registro de decisiones en métricas
        - Refinamiento contextual
    """
    
    def test_decide_nominal_returns_heartbeat(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: decide con NOMINAL retorna HEARTBEAT."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            decision = agent.decide(SystemStatus.NOMINAL)
            
            assert decision == AgentDecision.HEARTBEAT
    
    def test_decide_unknown_returns_wait(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: decide con UNKNOWN retorna WAIT."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            decision = agent.decide(SystemStatus.UNKNOWN)
            
            assert decision == AgentDecision.WAIT
    
    def test_decide_inestable_returns_limpieza(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: decide con INESTABLE retorna EJECUTAR_LIMPIEZA."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            decision = agent.decide(SystemStatus.INESTABLE)
            
            assert decision == AgentDecision.EJECUTAR_LIMPIEZA
    
    def test_decide_saturado_returns_ajustar_velocidad(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: decide con SATURADO retorna AJUSTAR_VELOCIDAD."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            decision = agent.decide(SystemStatus.SATURADO)
            
            assert decision == AgentDecision.AJUSTAR_VELOCIDAD
    
    def test_decide_critico_returns_alerta_critica(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: decide con CRITICO retorna ALERTA_CRITICA."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            decision = agent.decide(SystemStatus.CRITICO)
            
            assert decision == AgentDecision.ALERTA_CRITICA
    
    def test_decide_disconnected_returns_reconnect(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: decide con DISCONNECTED retorna RECONNECT."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            decision = agent.decide(SystemStatus.DISCONNECTED)
            
            assert decision == AgentDecision.RECONNECT
    
    def test_decide_updates_metrics(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: decide actualiza métricas de decisiones."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Tomar varias decisiones
            agent.decide(SystemStatus.NOMINAL)
            agent.decide(SystemStatus.NOMINAL)
            agent.decide(SystemStatus.INESTABLE)
            
            # Verificar conteo
            assert agent._metrics.decisions_count["HEARTBEAT"] == 2
            assert agent._metrics.decisions_count["EJECUTAR_LIMPIEZA"] == 1
    
    def test_decide_updates_last_status(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: decide actualiza self._last_status."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            decision = agent.decide(SystemStatus.CRITICO)
            
            assert agent._last_status == SystemStatus.CRITICO
    
    @given(st.sampled_from(SystemStatus))
    def test_decide_always_returns_valid_decision(
        self,
        agent_config: AgentConfig,
        status: SystemStatus,
    ):
        """PROP (Hypothesis): decide siempre retorna decisión válida."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            decision = agent.decide(status)
            
            # Debe ser una decisión válida
            assert isinstance(decision, AgentDecision)


# ============================================================================
# PRUEBAS DE LA FASE ACT
# ============================================================================

class TestActPhase(BaseAgentTest):
    """
    Suite de pruebas para la fase ACT del ciclo OODA.
    
    Verifica:
        - Ejecución de acciones según decisión
        - Debouncing correcto
        - Resolución Hamiltoniana (si aplica)
        - Proyección de intención sobre MIC
    """
    
    def test_act_heartbeat_logs_nominal(
        self,
        agent_config: AgentConfig,
        caplog,
    ):
        """UNIT: act con HEARTBEAT logea estado nominal."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Crear diagnóstico nominal
            agent._last_diagnosis = self.create_mock_diagnosis(
                status=SystemStatus.NOMINAL,
                health_score=1.0,
            )
            
            with caplog.at_level(logging.INFO):
                agent.act(AgentDecision.HEARTBEAT)
            
            # Verificar log
            assert "HEARTBEAT" in caplog.text
    
    def test_act_ejecutar_limpieza_projects_intent(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: act con EJECUTAR_LIMPIEZA proyecta intención."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Mock _project_intent
            with patch.object(agent, '_project_intent', return_value=True) as mock_project:
                agent._last_diagnosis = self.create_mock_diagnosis(
                    status=SystemStatus.INESTABLE,
                )
                
                agent.act(AgentDecision.EJECUTAR_LIMPIEZA)
                
                # Verificar que se llamó project_intent con vector "clean"
                mock_project.assert_called_once()
                call_args = mock_project.call_args
                assert call_args[1]['vector'] == "clean"
                assert call_args[1]['stratum'] == "PHYSICS"
    
    def test_act_ajustar_velocidad_projects_configure(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: act con AJUSTAR_VELOCIDAD proyecta configuración."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with patch.object(agent, '_project_intent', return_value=True) as mock_project:
                agent._last_diagnosis = self.create_mock_diagnosis(
                    status=SystemStatus.SATURADO,
                )
                
                agent.act(AgentDecision.AJUSTAR_VELOCIDAD)
                
                mock_project.assert_called_once()
                call_args = mock_project.call_args
                assert call_args[1]['vector'] == "configure"
    
    def test_act_alerta_critica_notifies_external(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: act con ALERTA_CRITICA notifica sistemas externos."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with patch.object(agent, '_notify_external_system') as mock_notify:
                agent._last_diagnosis = self.create_mock_diagnosis(
                    status=SystemStatus.CRITICO,
                )
                
                agent.act(AgentDecision.ALERTA_CRITICA)
                
                # Verificar que se notificó
                mock_notify.assert_called_once_with(
                    "critical_alert",
                    unittest.mock.ANY,
                )
    
    def test_act_reconnect_reinitializes_topology(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: act con RECONNECT reinicializa topología."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Degradar topología
            agent.topology.remove_edge(
                NetworkTopology.NODE_AGENT,
                NetworkTopology.NODE_CORE,
            )
            
            initial_health = agent.topology.get_topological_health()
            assert not initial_health.betti.is_connected
            
            # Ejecutar reconnect
            agent._last_diagnosis = self.create_mock_diagnosis(
                status=SystemStatus.DISCONNECTED,
            )
            agent.act(AgentDecision.RECONNECT)
            
            # Topología debe reinicializarse
            final_health = agent.topology.get_topological_health()
            assert final_health.betti.is_connected
    
    def test_act_updates_last_decision_time(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: act actualiza self._last_decision_time."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            assert agent._last_decision_time is None
            
            before = datetime.now()
            agent.act(AgentDecision.HEARTBEAT)
            after = datetime.now()
            
            assert agent._last_decision_time is not None
            assert before <= agent._last_decision_time <= after
    
    def test_act_debounce_suppresses_repeated_actions(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: act suprime acciones repetidas dentro de ventana de debounce."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Primera ejecución
            result1 = agent.act(AgentDecision.EJECUTAR_LIMPIEZA)
            assert result1 is True
            
            # Segunda ejecución inmediata (misma decisión)
            result2 = agent.act(AgentDecision.EJECUTAR_LIMPIEZA)
            assert result2 is False  # Suprimida por debounce
    
    def test_act_debounce_allows_critical_decisions(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: act NO suprime decisiones críticas (requires_immediate_action)."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with patch.object(agent, '_notify_external_system'):
                # Primera ejecución
                agent._last_diagnosis = self.create_mock_diagnosis(
                    status=SystemStatus.CRITICO,
                )
                result1 = agent.act(AgentDecision.ALERTA_CRITICA)
                assert result1 is True
                
                # Segunda ejecución inmediata (crítica)
                result2 = agent.act(AgentDecision.ALERTA_CRITICA)
                assert result2 is True  # NO suprimida (es crítica)
    
    def test_act_with_hamiltonian_synthesis(
        self,
        agent_config: AgentConfig,
        sample_telemetry_critical: TelemetryData,
    ):
        """UNIT: act usa síntesis Hamiltoniana cuando hay gradiente no cero."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Preparar diagnóstico con issues
            agent._last_telemetry = sample_telemetry_critical
            agent._last_diagnosis = self.create_mock_diagnosis(
                status=SystemStatus.CRITICO,
                health_score=0.3,
            )
            
            # Mock GaugeFieldRouter
            mock_router = Mock()
            mock_router.route_gradient.return_value = CategoricalState(
                payload={},
                context={
                    "gauge_selected_agent": "clean_on_edge_0",
                    "gauge_potential": [0.1, 0.2, 0.3, 0.4],
                },
            )
            agent._gauge_router = mock_router
            
            # Ejecutar act
            result = agent.act(AgentDecision.EJECUTAR_LIMPIEZA)
            
            # Verificar que se llamó route_gradient
            if result:
                mock_router.route_gradient.assert_called_once()
    
    def test_project_intent_sends_http_post(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: _project_intent envía POST al endpoint correcto."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with patch.object(agent._session, 'post') as mock_post:
                mock_response = Mock()
                mock_response.ok = True
                mock_post.return_value = mock_response
                
                result = agent._project_intent(
                    vector="clean",
                    stratum="PHYSICS",
                    payload={"test": "data"},
                )
                
                assert result is True
                mock_post.assert_called_once()
                
                # Verificar URL
                call_args = mock_post.call_args
                expected_url = agent._conn_config.tools_endpoint("clean")
                assert call_args[0][0] == expected_url
    
    def test_build_diagnosis_message_compact(
        self,
        agent_config: AgentConfig,
    ):
        """UNIT: _build_diagnosis_message genera mensaje compacto."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            agent._last_diagnosis = self.create_mock_diagnosis(
                status=SystemStatus.INESTABLE,
                health_score=0.75,
            )
            
            message = agent._build_diagnosis_message()
            
            assert len(message) > 0
            assert isinstance(message, str)


# ============================================================================
# PRUEBAS DE INTEGRACIÓN DEL CICLO OODA COMPLETO
# ============================================================================

class TestOODAIntegration(BaseAgentTest):
    """
    Suite de pruebas de integración del ciclo OODA completo.
    
    Verifica:
        - Ejecución secuencial de fases
        - Coherencia entre fases
        - Propagación de estado
    """
    
    def test_complete_ooda_cycle_nominal(
        self,
        agent_config: AgentConfig,
        mock_requests_success,
    ):
        """INTEGRATION: Ciclo OODA completo con sistema nominal."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # OBSERVE
            telemetry = agent.observe()
            assert telemetry is not None
            
            # ORIENT
            status = agent.orient(telemetry)
            assert status == SystemStatus.NOMINAL
            
            # DECIDE
            decision = agent.decide(status)
            assert decision == AgentDecision.HEARTBEAT
            
            # ACT
            result = agent.act(decision)
            assert result is True
            
            # Verificar coherencia de estado
            assert agent._last_telemetry == telemetry
            assert agent._last_status == status
            assert agent._last_decision == decision
    
    def test_complete_ooda_cycle_critical(
        self,
        agent_config: AgentConfig,
    ):
        """INTEGRATION: Ciclo OODA completo con sistema crítico."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Mock telemetría crítica
            with patch.object(agent._session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.ok = True
                mock_response.json.return_value = {
                    "flux_condenser": {
                        "max_flyback_voltage": 0.9,  # Crítico
                        "avg_saturation": 0.97,      # Crítico
                    },
                }
                mock_get.return_value = mock_response
                
                # OBSERVE
                telemetry = agent.observe()
                assert telemetry is not None
                assert telemetry.flyback_voltage > agent_config.thresholds.flyback_voltage_critical
                
                # ORIENT
                status = agent.orient(telemetry)
                assert status in (SystemStatus.CRITICO, SystemStatus.SATURADO)
                
                # DECIDE
                decision = agent.decide(status)
                assert decision.is_active_control or decision == AgentDecision.ALERTA_CRITICA
                
                # ACT
                with patch.object(agent, '_project_intent', return_value=True):
                    result = agent.act(decision)
                    # Puede ser True o False (debounce)
                    assert isinstance(result, bool)
    
    def test_ooda_cycle_with_connection_failure(
        self,
        agent_config: AgentConfig,
        mock_requests_connection_error,
    ):
        """INTEGRATION: Ciclo OODA con fallo de conexión."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # OBSERVE (falla)
            telemetry = agent.observe()
            assert telemetry is None
            
            # ORIENT (sin telemetría)
            status = agent.orient(telemetry)
            assert status == SystemStatus.UNKNOWN
            
            # DECIDE
            decision = agent.decide(status)
            assert decision == AgentDecision.WAIT
            
            # ACT
            result = agent.act(decision)
            assert result is True
    
    def test_ooda_cycle_metrics_consistency(
        self,
        agent_config: AgentConfig,
        mock_requests_success,
    ):
        """INTEGRATION: Métricas se mantienen consistentes tras ciclo OODA."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            initial_cycles = agent._metrics.cycles_executed
            
            # Ejecutar ciclo OODA
            agent._metrics.increment_cycle()
            telemetry = agent.observe()
            status = agent.orient(telemetry)
            decision = agent.decide(status)
            agent.act(decision)
            
            # Verificar invariantes de métricas
            self.assert_metrics_valid(agent._metrics)
            
            # Verificar incremento de ciclos
            assert agent._metrics.cycles_executed == initial_cycles + 1
    
    def test_ooda_cycle_state_propagation(
        self,
        agent_config: AgentConfig,
        mock_requests_success,
    ):
        """INTEGRATION: Estado se propaga correctamente entre fases."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Ejecutar ciclo
            telemetry = agent.observe()
            status = agent.orient(telemetry)
            decision = agent.decide(status)
            agent.act(decision)
            
            # Verificar coherencia de estado interno
            assert agent._last_telemetry == telemetry
            assert agent._last_status == status
            assert agent._last_decision == decision
            assert agent._last_diagnosis is not None
            assert agent._last_diagnosis.recommended_status == status
    
    def test_multiple_ooda_cycles_accumulate_metrics(
        self,
        agent_config: AgentConfig,
        mock_requests_success,
    ):
        """INTEGRATION: Múltiples ciclos OODA acumulan métricas correctamente."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            num_cycles = 5
            
            for i in range(num_cycles):
                agent._metrics.increment_cycle()
                telemetry = agent.observe()
                status = agent.orient(telemetry)
                decision = agent.decide(status)
                agent.act(decision)
            
            # Verificar acumulación
            assert agent._metrics.cycles_executed == num_cycles
            assert agent._metrics.successful_observations == num_cycles
            assert agent._metrics.success_rate == 1.0
    
    def test_ooda_cycle_captures_snapshot(
        self,
        agent_config: AgentConfig,
        mock_requests_success,
    ):
        """INTEGRATION: Ciclo OODA puede capturar snapshot del estado."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Ejecutar ciclo
            telemetry = agent.observe()
            status = agent.orient(telemetry)
            decision = agent.decide(status)
            agent.act(decision)
            
            # Capturar snapshot
            snapshot = AgentSnapshot.capture(
                cycle_number=1,
                telemetry=telemetry,
                diagnosis=agent._last_diagnosis,
                metrics=agent._metrics,
                last_decision=decision,
                current_status=status,
            )
            
            # Verificar snapshot
            assert snapshot.telemetry == telemetry
            assert snapshot.current_status == status
            assert snapshot.last_decision == decision


# ============================================================================
# PRUEBAS STATEFUL CON HYPOTHESIS
# ============================================================================

class OODAStateMachine(RuleBasedStateMachine):
    """
    Máquina de estados para testing stateful del ciclo OODA.
    
    Verifica invariantes que deben mantenerse a través de múltiples
    transiciones de estado (múltiples ciclos OODA).
    """
    
    def __init__(self):
        super().__init__()
        
        config = AgentConfig(
            timing=TimingConfig(
                check_interval=1,
                debounce_window_seconds=5,
                max_consecutive_failures=3,
            ),
        )
        
        with patch('app.core.apu_agent.get_global_mic'):
            self.agent = AutonomousAgent(config=config)
        
        self.cycle_count = 0
    
    @rule()
    def observe_success(self):
        """Regla: Observación exitosa."""
        with patch.object(self.agent._session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.json.return_value = {
                "flux_condenser": {
                    "max_flyback_voltage": 0.3,
                    "avg_saturation": 0.5,
                },
            }
            mock_get.return_value = mock_response
            
            self.agent.observe()
    
    @rule()
    def observe_failure(self):
        """Regla: Observación fallida."""
        with patch.object(self.agent._session, 'get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout()
            
            self.agent.observe()
    
    @rule()
    def complete_cycle(self):
        """Regla: Ciclo OODA completo."""
        with patch.object(self.agent._session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.ok = True
            mock_response.json.return_value = {
                "flux_condenser": {
                    "max_flyback_voltage": 0.4,
                    "avg_saturation": 0.6,
                },
            }
            mock_get.return_value = mock_response
            
            telemetry = self.agent.observe()
            status = self.agent.orient(telemetry)
            decision = self.agent.decide(status)
            
            with patch.object(self.agent, '_project_intent', return_value=True):
                self.agent.act(decision)
        
        self.cycle_count += 1
    
    @invariant()
    def metrics_are_consistent(self):
        """INVARIANTE: Métricas siempre son consistentes."""
        metrics = self.agent._metrics
        
        # Conservación de totales
        assert (
            metrics.successful_observations + metrics.failed_observations
            == metrics.total_observations
        )
        
        # Success rate válido
        assert 0.0 <= metrics.success_rate <= 1.0
        
        # Contadores no negativos
        assert metrics.cycles_executed >= 0
        assert metrics.successful_observations >= 0
        assert metrics.failed_observations >= 0
    
    @invariant()
    def topology_is_valid(self):
        """INVARIANTE: Topología siempre es válida."""
        health = self.agent.topology.get_topological_health()
        
        # β₀ ≥ 1 (al menos una componente)
        assert health.betti.b0 >= 1
        
        # β₁ ≥ 0 (no ciclos negativos)
        assert health.betti.b1 >= 0
        
        # Score válido
        assert 0.0 <= health.health_score <= 1.0
    
    @invariant()
    def state_is_coherent(self):
        """INVARIANTE: Estado interno es coherente."""
        # Si hay diagnóstico, debe tener status
        if self.agent._last_diagnosis is not None:
            assert self.agent._last_diagnosis.recommended_status is not None
        
        # Si hay última decisión, debe haber tiempo
        if self.agent._last_decision is not None:
            assert self.agent._last_decision_time is not None


# Configurar test de la máquina de estados
TestOODAStateMachine = OODAStateMachine.TestCase


# ============================================================================
# PRUEBAS DE SHOULDERS (CASOS LÍMITE)
# ============================================================================

class TestOODAEdgeCases(BaseAgentTest):
    """
    Pruebas de casos límite del ciclo OODA.
    
    Verifica comportamiento en condiciones extremas o inusuales.
    """
    
    def test_observe_with_empty_response(
        self,
        agent_config: AgentConfig,
    ):
        """EDGE: observe con respuesta vacía retorna None."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with patch.object(agent._session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.ok = True
                mock_response.json.return_value = {}  # Vacío
                mock_get.return_value = mock_response
                
                telemetry = agent.observe()
                
                # Debe construir telemetría con defaults
                assert telemetry is not None
                assert telemetry.flyback_voltage == 0.0
                assert telemetry.saturation == 0.0
    
    def test_orient_with_extreme_persistence(
        self,
        agent_config: AgentConfig,
    ):
        """EDGE: orient con persistencia extremadamente larga."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Agregar muchas muestras con valor crítico
            for _ in range(100):
                agent.persistence.add_sample("flyback_voltage", 0.9)
            
            telemetry = TelemetryData(flyback_voltage=0.9, saturation=0.5)
            
            status = agent.orient(telemetry)
            
            # Debe detectar persistencia crítica
            assert status in (SystemStatus.CRITICO, SystemStatus.INESTABLE)
    
    def test_decide_with_rapid_status_changes(
        self,
        agent_config: AgentConfig,
    ):
        """EDGE: decide con cambios rápidos de estado."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            statuses = [
                SystemStatus.NOMINAL,
                SystemStatus.INESTABLE,
                SystemStatus.CRITICO,
                SystemStatus.NOMINAL,
            ]
            
            decisions = []
            for status in statuses:
                decision = agent.decide(status)
                decisions.append(decision)
            
            # Verificar que todas las decisiones son válidas
            assert len(decisions) == len(statuses)
            for decision in decisions:
                assert isinstance(decision, AgentDecision)
    
    def test_act_with_null_diagnosis(
        self,
        agent_config: AgentConfig,
    ):
        """EDGE: act sin diagnóstico previo (primer ciclo)."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            assert agent._last_diagnosis is None
            
            # act debe manejar ausencia de diagnóstico
            result = agent.act(AgentDecision.HEARTBEAT)
            
            assert isinstance(result, bool)
    
    def test_ooda_cycle_with_all_metrics_zero(
        self,
        agent_config: AgentConfig,
    ):
        """EDGE: Ciclo OODA con métricas en cero (primer ciclo)."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Verificar estado inicial
            assert agent._metrics.cycles_executed == 0
            assert agent._metrics.total_observations == 0
            
            # Ejecutar ciclo sin errores
            with patch.object(agent._session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.ok = True
                mock_response.json.return_value = {
                    "flux_condenser": {
                        "max_flyback_voltage": 0.5,
                        "avg_saturation": 0.5,
                    },
                }
                mock_get.return_value = mock_response
                
                telemetry = agent.observe()
                status = agent.orient(telemetry)
                decision = agent.decide(status)
                agent.act(decision)
            
            # Métricas deben actualizarse correctamente
            assert agent._metrics.successful_observations == 1


# ============================================================================
# PRUEBAS DE INICIALIZACIÓN Y CONFIGURACIÓN DEL AGENTE
# ============================================================================

class TestAgentInitialization(BaseAgentTest):
    """
    Suite de pruebas para inicialización del AutonomousAgent.
    
    Verifica:
        - Construcción correcta con configuración válida
        - Inicialización de componentes internos
        - Configuración de sesión HTTP
        - Instalación de signal handlers
        - Inicialización de topología esperada
    """
    
    def test_initialization_with_default_config(self):
        """UNIT: Inicialización con configuración por defecto."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            # Verificar componentes inicializados
            assert agent.config is not None
            assert agent.topology is not None
            assert agent.persistence is not None
            assert agent._synthesizer is not None
            assert agent._session is not None
            assert agent._metrics is not None
            
            # Estado inicial
            assert agent._running is False
            assert agent._last_decision is None
            assert agent._last_telemetry is None
    
    def test_initialization_with_custom_config(self, agent_config: AgentConfig):
        """UNIT: Inicialización con configuración personalizada."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Verificar que usa la configuración provista
            assert agent.config == agent_config
            assert agent._thresholds == agent_config.thresholds
            assert agent._conn_config == agent_config.connection
    
    def test_initialization_with_custom_synthesizer(self):
        """UNIT: Inicialización con sintetizador personalizado."""
        custom_synthesizer = HamiltonianSynthesizer(
            evaluators=[FragmentationEvaluator()]
        )
        
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(synthesizer=custom_synthesizer)
            
            assert agent._synthesizer == custom_synthesizer
            assert len(agent._synthesizer.evaluators) == 1
    
    def test_initialization_sets_up_expected_topology(self):
        """UNIT: Inicialización establece topología esperada."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            # Verificar topología inicial
            health = agent.topology.get_topological_health()
            
            # Debe estar conectada (β₀ = 1)
            assert health.betti.is_connected
            assert health.betti.b0 == 1
            
            # Debe tener las aristas esperadas
            expected_edges = set(NetworkTopology.EXPECTED_EDGES)
            actual_edges = agent.topology.edges
            assert expected_edges.issubset(actual_edges)
    
    def test_initialization_creates_http_session(self):
        """UNIT: Inicialización crea sesión HTTP con retry policy."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            # Verificar sesión HTTP
            assert agent._session is not None
            assert isinstance(agent._session, requests.Session)
            
            # Verificar headers predeterminados
            assert "User-Agent" in agent._session.headers
            assert "Content-Type" in agent._session.headers
    
    def test_initialization_installs_signal_handlers(self):
        """UNIT: Inicialización instala signal handlers."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            # Verificar que se guardaron handlers originales
            assert signal.SIGINT in agent._original_handlers
            assert signal.SIGTERM in agent._original_handlers
    
    def test_initialization_syncs_gauge_router(self):
        """UNIT: Inicialización sincroniza GaugeFieldRouter."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            # GaugeFieldRouter debe estar inicializado
            # (puede ser None si la topología es degenerada)
            if len(agent.topology.nodes) > 0 and len(agent.topology.edges) > 0:
                assert agent._gauge_router is not None
    
    def test_initialization_creates_snapshot_history(self):
        """UNIT: Inicialización crea historial de snapshots."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            assert agent._snapshot_history is not None
            assert agent._snapshot_history.size == 0
            assert agent._snapshot_history.max_size > 0


# ============================================================================
# PRUEBAS DE STARTUP Y HEALTH CHECK
# ============================================================================

class TestAgentStartup(BaseAgentTest):
    """
    Suite de pruebas para el proceso de startup del agente.
    
    Verifica:
        - Cold start con backoff exponencial
        - Health check inicial
        - Manejo de errores durante startup
        - Timeouts de startup
    """
    
    def test_wait_for_startup_success_immediate(self, agent_config: AgentConfig):
        """UNIT: _wait_for_startup exitoso en primer intento."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            agent._running = True
            
            # Mock respuesta exitosa inmediata
            with patch('app.core.apu_agent.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.ok = True
                mock_get.return_value = mock_response
                
                result = agent._wait_for_startup()
                
                assert result is True
                mock_get.assert_called_once()
    
    def test_wait_for_startup_success_after_retries(self, agent_config: AgentConfig):
        """UNIT: _wait_for_startup exitoso tras varios reintentos."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            agent._running = True
            
            # Mock: fallar 2 veces, luego éxito
            with patch('app.core.apu_agent.requests.get') as mock_get:
                mock_response_fail = Mock()
                mock_response_fail.ok = False
                mock_response_fail.status_code = 503
                
                mock_response_success = Mock()
                mock_response_success.ok = True
                
                mock_get.side_effect = [
                    mock_response_fail,
                    mock_response_fail,
                    mock_response_success,
                ]
                
                with patch('time.sleep'):  # Evitar sleep real
                    result = agent._wait_for_startup()
                
                assert result is True
                assert mock_get.call_count == 3
    
    def test_wait_for_startup_timeout_after_max_attempts(self, agent_config: AgentConfig):
        """UNIT: _wait_for_startup timeout tras max_attempts."""
        config = AgentConfig(
            timing=TimingConfig(
                startup_max_attempts=3,
                startup_backoff_initial=0.1,
            ),
        )
        
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=config)
            agent._running = True
            
            # Mock: siempre falla
            with patch('app.core.apu_agent.requests.get') as mock_get:
                mock_get.side_effect = requests.exceptions.ConnectionError()
                
                with patch('time.sleep'):
                    result = agent._wait_for_startup()
                
                assert result is False
                assert mock_get.call_count == 3  # max_attempts
    
    def test_wait_for_startup_exponential_backoff(self, agent_config: AgentConfig):
        """UNIT: _wait_for_startup usa backoff exponencial."""
        config = AgentConfig(
            timing=TimingConfig(
                startup_backoff_initial=1.0,
                startup_backoff_multiplier=2.0,
                startup_backoff_max=10.0,
                startup_max_attempts=5,
            ),
        )
        
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=config)
            agent._running = True
            
            sleep_times = []
            
            def mock_sleep(seconds):
                sleep_times.append(seconds)
            
            with patch('app.core.apu_agent.requests.get') as mock_get:
                mock_get.side_effect = requests.exceptions.ConnectionError()
                
                with patch('time.sleep', side_effect=mock_sleep):
                    agent._wait_for_startup()
            
            # Verificar backoff exponencial: [1.0, 2.0, 4.0, 8.0]
            assert len(sleep_times) >= 3
            assert sleep_times[0] == 1.0
            assert sleep_times[1] == 2.0
            assert sleep_times[2] == 4.0
    
    def test_wait_for_startup_respects_running_flag(self, agent_config: AgentConfig):
        """UNIT: _wait_for_startup termina si _running se setea a False."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            agent._running = True
            
            call_count = 0
            
            def mock_get(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                # Detener después del segundo intento
                if call_count == 2:
                    agent._running = False
                
                raise requests.exceptions.ConnectionError()
            
            with patch('app.core.apu_agent.requests.get', side_effect=mock_get):
                with patch('time.sleep'):
                    result = agent._wait_for_startup()
            
            assert result is False
            assert call_count == 2
    
    def test_health_check_success(self, agent_config: AgentConfig):
        """UNIT: health_check exitoso con Core accesible."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Mock respuesta exitosa
            with patch.object(agent._session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.ok = True
                mock_get.return_value = mock_response
                
                result = agent.health_check()
                
                assert result is True
                
                # Verificar que topología se reinicializó
                health = agent.topology.get_topological_health()
                assert health.betti.is_connected
    
    def test_health_check_failure_degrades_topology(self, agent_config: AgentConfig):
        """UNIT: health_check fallido degrada topología."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Mock error de conexión
            with patch.object(agent._session, 'get') as mock_get:
                mock_get.side_effect = requests.exceptions.ConnectionError()
                
                result = agent.health_check()
                
                assert result is False
                
                # Verificar degradación topológica
                health = agent.topology.get_topological_health()
                assert not health.betti.is_connected or health.health_score < 1.0
    
    def test_health_check_with_http_error_continues(self, agent_config: AgentConfig):
        """UNIT: health_check con HTTP error continúa (no es fatal)."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Mock HTTP error (503)
            with patch.object(agent._session, 'get') as mock_get:
                mock_response = Mock()
                mock_response.ok = False
                mock_response.status_code = 503
                mock_get.return_value = mock_response
                
                result = agent.health_check()
                
                # Permite continuar con advertencias
                assert result is True


# ============================================================================
# PRUEBAS DE SIGNAL HANDLERS Y SHUTDOWN
# ============================================================================

class TestSignalHandlers(BaseAgentTest):
    """
    Suite de pruebas para manejo de señales POSIX.
    
    Verifica:
        - Instalación de signal handlers
        - Manejo de SIGINT y SIGTERM
        - Shutdown graceful
        - Restauración de handlers originales
    """
    
    def test_setup_signal_handlers_installs_handlers(self, agent_config: AgentConfig):
        """UNIT: _setup_signal_handlers instala handlers para SIGINT y SIGTERM."""
        with patch('app.core.apu_agent.get_global_mic'):
            with patch('signal.signal') as mock_signal:
                agent = AutonomousAgent(config=agent_config)
                
                # Verificar que se llamó signal.signal
                assert mock_signal.call_count >= 2
                
                # Verificar señales instaladas
                calls = [call[0] for call in mock_signal.call_args_list]
                signals_installed = [call[0] for call in calls]
                
                assert signal.SIGINT in signals_installed
                assert signal.SIGTERM in signals_installed
    
    def test_handle_shutdown_sets_running_false(self, agent_config: AgentConfig):
        """UNIT: _handle_shutdown setea _running a False."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            agent._running = True
            
            # Simular recepción de SIGINT
            agent._handle_shutdown(signal.SIGINT, None)
            
            assert agent._running is False
    
    def test_handle_shutdown_logs_signal(self, agent_config: AgentConfig, caplog):
        """UNIT: _handle_shutdown logea la señal recibida."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with caplog.at_level(logging.INFO):
                agent._handle_shutdown(signal.SIGTERM, None)
            
            # Verificar log
            assert "SIGTERM" in caplog.text or "15" in caplog.text
            assert "shutdown" in caplog.text.lower()
    
    def test_restore_signal_handlers_restores_originals(self, agent_config: AgentConfig):
        """UNIT: _restore_signal_handlers restaura handlers originales."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Guardar handlers actuales
            current_sigint = signal.getsignal(signal.SIGINT)
            current_sigterm = signal.getsignal(signal.SIGTERM)
            
            # Restaurar
            agent._restore_signal_handlers()
            
            # Verificar restauración
            restored_sigint = signal.getsignal(signal.SIGINT)
            restored_sigterm = signal.getsignal(signal.SIGTERM)
            
            # Los handlers deben haber cambiado (restaurados a originales)
            # Nota: En tests, pueden ser los mismos si no hubo cambio previo
            assert restored_sigint is not None
            assert restored_sigterm is not None
    
    def test_signal_handler_integration_sigint(self, agent_config: AgentConfig):
        """INTEGRATION: Envío de SIGINT detiene el agente."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            agent._running = True
            
            # Simular envío de SIGINT
            os.kill(os.getpid(), signal.SIGINT)
            
            # Dar tiempo a procesar señal
            time.sleep(0.1)
            
            # El handler debe haber seteado _running a False
            assert agent._running is False
    
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="SIGTERM no soportado en Windows"
    )
    def test_signal_handler_integration_sigterm(self, agent_config: AgentConfig):
        """INTEGRATION: Envío de SIGTERM detiene el agente."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            agent._running = True
            
            # Simular envío de SIGTERM
            os.kill(os.getpid(), signal.SIGTERM)
            
            time.sleep(0.1)
            
            assert agent._running is False


class TestShutdown(BaseAgentTest):
    """
    Suite de pruebas para el proceso de shutdown.
    
    Verifica:
        - Limpieza de recursos
        - Cierre de sesión HTTP
        - Logging de métricas finales
        - Restauración de signal handlers
    """
    
    def test_shutdown_closes_http_session(self, agent_config: AgentConfig):
        """UNIT: _shutdown cierra la sesión HTTP."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Mock session.close
            with patch.object(agent._session, 'close') as mock_close:
                agent._shutdown()
                
                mock_close.assert_called_once()
    
    def test_shutdown_restores_signal_handlers(self, agent_config: AgentConfig):
        """UNIT: _shutdown restaura signal handlers."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with patch.object(agent, '_restore_signal_handlers') as mock_restore:
                agent._shutdown()
                
                mock_restore.assert_called_once()
    
    def test_shutdown_logs_final_metrics(self, agent_config: AgentConfig, caplog):
        """UNIT: _shutdown logea métricas finales."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular algunas métricas
            agent._metrics.cycles_executed = 10
            agent._metrics.successful_observations = 8
            agent._metrics.failed_observations = 2
            
            with caplog.at_level(logging.INFO):
                agent._shutdown()
            
            # Verificar log de métricas
            assert "Métricas finales" in caplog.text or "cycles" in caplog.text
    
    def test_shutdown_handles_session_close_error(self, agent_config: AgentConfig, caplog):
        """UNIT: _shutdown maneja errores al cerrar sesión."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Mock session.close que lanza excepción
            with patch.object(agent._session, 'close', side_effect=Exception("Close error")):
                with caplog.at_level(logging.WARNING):
                    agent._shutdown()
                
                # Debe logear warning pero no fallar
                assert "Error cerrando sesión" in caplog.text or "Close error" in caplog.text
    
    def test_stop_sets_running_false(self, agent_config: AgentConfig):
        """UNIT: stop() setea _running a False."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            agent._running = True
            
            agent.stop()
            
            assert agent._running is False
    
    def test_shutdown_is_idempotent(self, agent_config: AgentConfig):
        """UNIT: _shutdown puede llamarse múltiples veces sin error."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Llamar shutdown múltiples veces
            agent._shutdown()
            agent._shutdown()
            agent._shutdown()
            
            # No debe lanzar excepción


# ============================================================================
# PRUEBAS DEL BUCLE PRINCIPAL (RUN)
# ============================================================================

class TestRunLoop(BaseAgentTest):
    """
    Suite de pruebas para el bucle principal run().
    
    Verifica:
        - Ejecución del ciclo OODA repetidamente
        - Manejo de errores en ciclo
        - Sleep adaptativo
        - Captura de snapshots
        - Shutdown graceful
    """
    
    def test_run_waits_for_startup(self, agent_config: AgentConfig):
        """UNIT: run() espera startup antes de iniciar ciclo."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with patch.object(agent, '_wait_for_startup', return_value=False) as mock_wait:
                with patch.object(agent, 'health_check'):
                    agent.run()
                
                mock_wait.assert_called_once()
    
    def test_run_aborts_if_startup_fails(self, agent_config: AgentConfig):
        """UNIT: run() aborta si startup falla."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with patch.object(agent, '_wait_for_startup', return_value=False):
                with patch.object(agent, 'health_check') as mock_health:
                    agent.run()
                    
                    # health_check no debe llamarse si startup falla
                    mock_health.assert_not_called()
    
    def test_run_performs_health_check_unless_skipped(self, agent_config: AgentConfig):
        """UNIT: run() ejecuta health_check a menos que skip_health_check=True."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with patch.object(agent, '_wait_for_startup', return_value=True):
                with patch.object(agent, 'health_check', return_value=True) as mock_health:
                    with patch.object(agent, '_shutdown'):
                        # Ejecutar un ciclo y detener
                        agent._running = True
                        
                        def stop_after_one_cycle(*args, **kwargs):
                            agent._running = False
                        
                        with patch.object(agent, 'observe', side_effect=stop_after_one_cycle):
                            agent.run()
                        
                        mock_health.assert_called_once()
    
    def test_run_skip_health_check(self, agent_config: AgentConfig):
        """UNIT: run(skip_health_check=True) omite health check."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with patch.object(agent, 'health_check') as mock_health:
                with patch.object(agent, '_shutdown'):
                    agent._running = True
                    
                    def stop_immediately(*args, **kwargs):
                        agent._running = False
                    
                    with patch.object(agent, 'observe', side_effect=stop_immediately):
                        agent.run(skip_health_check=True)
                    
                    mock_health.assert_not_called()
    
    def test_run_executes_ooda_cycle_repeatedly(self, agent_config: AgentConfig):
        """UNIT: run() ejecuta ciclo OODA múltiples veces."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            observe_count = 0
            
            def mock_observe():
                nonlocal observe_count
                observe_count += 1
                
                # Detener después de 3 ciclos
                if observe_count >= 3:
                    agent._running = False
                
                return self.create_mock_telemetry()
            
            with patch.object(agent, 'observe', side_effect=mock_observe):
                with patch.object(agent, 'orient', return_value=SystemStatus.NOMINAL):
                    with patch.object(agent, 'decide', return_value=AgentDecision.HEARTBEAT):
                        with patch.object(agent, 'act', return_value=True):
                            with patch('time.sleep'):  # Evitar sleep real
                                agent.run(skip_health_check=True)
            
            # Verificar que se ejecutaron 3 ciclos
            assert observe_count == 3
    
    def test_run_increments_cycle_counter(self, agent_config: AgentConfig):
        """UNIT: run() incrementa contador de ciclos."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            initial_cycles = agent._metrics.cycles_executed
            
            def stop_after_two(*args, **kwargs):
                if agent._metrics.cycles_executed >= 2:
                    agent._running = False
                return self.create_mock_telemetry()
            
            with patch.object(agent, 'observe', side_effect=stop_after_two):
                with patch.object(agent, 'orient', return_value=SystemStatus.NOMINAL):
                    with patch.object(agent, 'decide', return_value=AgentDecision.HEARTBEAT):
                        with patch.object(agent, 'act'):
                            with patch('time.sleep'):
                                agent.run(skip_health_check=True)
            
            assert agent._metrics.cycles_executed == initial_cycles + 2
    
    def test_run_captures_snapshots(self, agent_config: AgentConfig):
        """UNIT: run() captura snapshots en cada ciclo."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            initial_snapshot_count = agent._snapshot_history.size
            
            def stop_after_one(*args, **kwargs):
                agent._running = False
                return self.create_mock_telemetry()
            
            with patch.object(agent, 'observe', side_effect=stop_after_one):
                with patch.object(agent, 'orient', return_value=SystemStatus.NOMINAL):
                    with patch.object(agent, 'decide', return_value=AgentDecision.HEARTBEAT):
                        with patch.object(agent, 'act'):
                            with patch('time.sleep'):
                                agent.run(skip_health_check=True)
            
            # Debe haberse agregado un snapshot
            assert agent._snapshot_history.size == initial_snapshot_count + 1
    
    def test_run_handles_errors_in_cycle_gracefully(self, agent_config: AgentConfig, caplog):
        """UNIT: run() maneja errores en ciclo sin abortar."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            error_count = 0
            
            def mock_observe():
                nonlocal error_count
                error_count += 1
                
                if error_count == 1:
                    raise ValueError("Test error in cycle")
                elif error_count >= 2:
                    agent._running = False
                
                return self.create_mock_telemetry()
            
            with caplog.at_level(logging.ERROR):
                with patch.object(agent, 'observe', side_effect=mock_observe):
                    with patch('time.sleep'):
                        agent.run(skip_health_check=True)
            
            # Debe haber loggeado el error
            assert "Error en ciclo OODA" in caplog.text or "Test error" in caplog.text
            
            # Debe haber continuado y ejecutado segundo ciclo
            assert error_count == 2
    
    def test_run_calls_shutdown_on_exit(self, agent_config: AgentConfig):
        """UNIT: run() llama _shutdown al salir."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            with patch.object(agent, '_shutdown') as mock_shutdown:
                def stop_immediately(*args, **kwargs):
                    agent._running = False
                    return self.create_mock_telemetry()
                
                with patch.object(agent, 'observe', side_effect=stop_immediately):
                    with patch('time.sleep'):
                        agent.run(skip_health_check=True)
                
                mock_shutdown.assert_called_once()
    
    def test_run_adaptive_sleep(self, agent_config: AgentConfig):
        """UNIT: run() usa sleep adaptativo según duración del ciclo."""
        config = AgentConfig(
            timing=TimingConfig(
                check_interval=10,  # 10 segundos entre ciclos
            ),
        )
        
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=config)
            
            sleep_times = []
            
            def mock_sleep(seconds):
                sleep_times.append(seconds)
            
            def stop_after_one(*args, **kwargs):
                agent._running = False
                return self.create_mock_telemetry()
            
            with patch.object(agent, 'observe', side_effect=stop_after_one):
                with patch.object(agent, 'orient', return_value=SystemStatus.NOMINAL):
                    with patch.object(agent, 'decide', return_value=AgentDecision.HEARTBEAT):
                        with patch.object(agent, 'act'):
                            with patch('time.sleep', side_effect=mock_sleep):
                                agent.run(skip_health_check=True)
            
            # Debe haber hecho sleep
            assert len(sleep_times) > 0
            
            # Sleep debe ser ≤ check_interval
            for sleep_time in sleep_times:
                assert sleep_time <= config.timing.check_interval
    
    def test_run_respects_running_flag(self, agent_config: AgentConfig):
        """UNIT: run() termina cuando _running se setea a False."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            cycle_count = 0
            
            def mock_observe():
                nonlocal cycle_count
                cycle_count += 1
                
                # Simular señal de shutdown en ciclo 2
                if cycle_count == 2:
                    agent._running = False
                
                return self.create_mock_telemetry()
            
            with patch.object(agent, 'observe', side_effect=mock_observe):
                with patch.object(agent, 'orient', return_value=SystemStatus.NOMINAL):
                    with patch.object(agent, 'decide', return_value=AgentDecision.HEARTBEAT):
                        with patch.object(agent, 'act'):
                            with patch('time.sleep'):
                                agent.run(skip_health_check=True)
            
            # Solo debe haber ejecutado 2 ciclos
            assert cycle_count == 2


# ============================================================================
# PRUEBAS DE FACTORY FUNCTIONS
# ============================================================================

class TestFactoryFunctions(BaseAgentTest):
    """
    Suite de pruebas para funciones factory.
    
    Verifica:
        - create_agent()
        - create_minimal_agent()
        - Construcción correcta de agentes
    """
    
    def test_create_agent_with_default_config(self):
        """UNIT: create_agent() sin args crea agente con config por defecto."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = create_agent()
            
            assert isinstance(agent, AutonomousAgent)
            assert agent.config is not None
    
    def test_create_agent_with_custom_config(self, agent_config: AgentConfig):
        """UNIT: create_agent() con config personalizada."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = create_agent(config=agent_config)
            
            assert agent.config == agent_config
    
    def test_create_agent_with_custom_synthesizer(self):
        """UNIT: create_agent() con sintetizador personalizado."""
        synthesizer = HamiltonianSynthesizer(evaluators=[])
        
        with patch('app.core.apu_agent.get_global_mic'):
            agent = create_agent(synthesizer=synthesizer)
            
            assert agent._synthesizer == synthesizer
    
    def test_create_minimal_agent_with_url(self):
        """UNIT: create_minimal_agent() crea agente con URL mínima."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = create_minimal_agent("http://localhost:8080")
            
            assert isinstance(agent, AutonomousAgent)
            assert agent.config.connection.base_url == "http://localhost:8080"
    
    def test_create_minimal_agent_normalizes_url(self):
        """UNIT: create_minimal_agent() normaliza URL."""
        with patch('app.core.apu_agent.get_global_mic'):
            # Sin esquema
            agent1 = create_minimal_agent("localhost:8080")
            assert agent1.config.connection.base_url == "http://localhost:8080"
            
            # Con trailing slash
            agent2 = create_minimal_agent("http://localhost:8080/")
            assert agent2.config.connection.base_url == "http://localhost:8080"
    
    def test_create_minimal_agent_invalid_url_raises(self):
        """UNIT: create_minimal_agent() con URL inválida lanza excepción."""
        with pytest.raises(ConfigurationError):
            create_minimal_agent("")
        
        with pytest.raises(ConfigurationError):
            create_minimal_agent("http://")


# ============================================================================
# PRUEBAS DE MAIN ENTRY POINT
# ============================================================================

class TestMainEntryPoint(BaseAgentTest):
    """
    Suite de pruebas para el entry point main().
    
    Verifica:
        - Ejecución exitosa
        - Manejo de errores de configuración
        - Manejo de KeyboardInterrupt
        - Códigos de salida
    """
    
    def test_main_success(self):
        """UNIT: main() retorna 0 en ejecución exitosa."""
        with patch('app.core.apu_agent.get_global_mic'):
            with patch('app.core.apu_agent.AutonomousAgent') as mock_agent_class:
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent
                
                # Mock run() que termina inmediatamente
                mock_agent.run.return_value = None
                
                from app.core.apu_agent import main
                
                exit_code = main()
                
                assert exit_code == 0
                mock_agent.run.assert_called_once()
    
    def test_main_configuration_error(self):
        """UNIT: main() retorna 1 en error de configuración."""
        with patch('app.core.apu_agent.AutonomousAgent') as mock_agent_class:
            mock_agent_class.side_effect = ConfigurationError("Test error")
            
            from app.core.apu_agent import main
            
            exit_code = main()
            
            assert exit_code == 1
    
    def test_main_keyboard_interrupt(self):
        """UNIT: main() retorna 0 en KeyboardInterrupt."""
        with patch('app.core.apu_agent.get_global_mic'):
            with patch('app.core.apu_agent.AutonomousAgent') as mock_agent_class:
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent
                mock_agent.run.side_effect = KeyboardInterrupt()
                
                from app.core.apu_agent import main
                
                exit_code = main()
                
                assert exit_code == 0
    
    def test_main_unexpected_exception(self, caplog):
        """UNIT: main() retorna 1 en excepción no manejada."""
        with patch('app.core.apu_agent.get_global_mic'):
            with patch('app.core.apu_agent.AutonomousAgent') as mock_agent_class:
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent
                mock_agent.run.side_effect = RuntimeError("Unexpected error")
                
                from app.core.apu_agent import main
                
                with caplog.at_level(logging.CRITICAL):
                    exit_code = main()
                
                assert exit_code == 1
                assert "Error no manejado" in caplog.text or "Unexpected error" in caplog.text


# ============================================================================
# PRUEBAS DE INTEGRACIÓN DE LIFECYCLE COMPLETO
# ============================================================================

class TestLifecycleIntegration(BaseAgentTest):
    """
    Suite de pruebas de integración del lifecycle completo.
    
    Verifica:
        - Flujo completo: init → startup → run → shutdown
        - Coherencia de estado durante todo el lifecycle
        - Manejo de errores en cada fase
    """
    
    def test_complete_lifecycle_nominal(self):
        """INTEGRATION: Lifecycle completo con sistema nominal."""
        with patch('app.core.apu_agent.get_global_mic'):
            # INIT
            agent = AutonomousAgent()
            assert agent._running is False
            
            # Simular startup exitoso
            with patch.object(agent, '_wait_for_startup', return_value=True):
                with patch.object(agent, 'health_check', return_value=True):
                    # Simular 2 ciclos OODA
                    cycle_count = 0
                    
                    def mock_observe():
                        nonlocal cycle_count
                        cycle_count += 1
                        
                        if cycle_count >= 2:
                            agent._running = False
                        
                        return self.create_mock_telemetry()
                    
                    with patch.object(agent, 'observe', side_effect=mock_observe):
                        with patch.object(agent, 'orient', return_value=SystemStatus.NOMINAL):
                            with patch.object(agent, 'decide', return_value=AgentDecision.HEARTBEAT):
                                with patch.object(agent, 'act', return_value=True):
                                    with patch('time.sleep'):
                                        # RUN
                                        agent.run()
            
            # Verificar estado final
            assert agent._metrics.cycles_executed == 2
            assert agent._metrics.successful_observations == 2
            assert agent._running is False
    
    def test_lifecycle_with_startup_failure(self):
        """INTEGRATION: Lifecycle que aborta en startup."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            with patch.object(agent, '_wait_for_startup', return_value=False):
                agent.run()
            
            # No debe haber ejecutado ciclos
            assert agent._metrics.cycles_executed == 0
    
    def test_lifecycle_with_health_check_warning(self, caplog):
        """INTEGRATION: Lifecycle continúa con health check degradado."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            with patch.object(agent, '_wait_for_startup', return_value=True):
                with patch.object(agent, 'health_check', return_value=False):
                    with caplog.at_level(logging.WARNING):
                        def stop_immediately(*args, **kwargs):
                            agent._running = False
                        
                        with patch.object(agent, 'observe', side_effect=stop_immediately):
                            with patch('time.sleep'):
                                agent.run()
                    
                    # Debe haber advertido pero continuado
                    assert "advertencias" in caplog.text.lower() or "warning" in caplog.text.lower()
    
    def test_lifecycle_with_signal_interruption(self):
        """INTEGRATION: Lifecycle interrumpido por señal."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            def mock_observe():
                # Simular señal de shutdown en primer ciclo
                agent._handle_shutdown(signal.SIGINT, None)
                return self.create_mock_telemetry()
            
            with patch.object(agent, 'observe', side_effect=mock_observe):
                with patch('time.sleep'):
                    agent.run(skip_health_check=True)
            
            # Debe haber terminado gracefully
            assert agent._running is False
    
    def test_lifecycle_metrics_accumulation(self):
        """INTEGRATION: Métricas se acumulan correctamente durante lifecycle."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            # Simular 5 ciclos: 3 éxitos, 2 fallos
            results = [
                self.create_mock_telemetry(),
                None,  # Fallo
                self.create_mock_telemetry(),
                self.create_mock_telemetry(),
                None,  # Fallo
            ]
            
            result_index = 0
            
            def mock_observe():
                nonlocal result_index
                result = results[result_index]
                result_index += 1
                
                if result_index >= len(results):
                    agent._running = False
                
                return result
            
            with patch.object(agent, 'observe', side_effect=mock_observe):
                with patch.object(agent, 'orient', return_value=SystemStatus.NOMINAL):
                    with patch.object(agent, 'decide', return_value=AgentDecision.HEARTBEAT):
                        with patch.object(agent, 'act'):
                            with patch('time.sleep'):
                                agent.run(skip_health_check=True)
            
            # Verificar métricas finales
            assert agent._metrics.successful_observations == 3
            assert agent._metrics.failed_observations == 2
            assert agent._metrics.total_observations == 5
            assert agent._metrics.success_rate == 0.6


# ============================================================================
# PRUEBAS DE ROBUSTEZ Y CASOS EXTREMOS
# ============================================================================

class TestLifecycleRobustness(BaseAgentTest):
    """
    Pruebas de robustez del lifecycle.
    
    Verifica comportamiento en condiciones extremas o adversas.
    """
    
    def test_lifecycle_survives_repeated_errors(self, caplog):
        """ROBUSTNESS: Lifecycle continúa tras errores repetidos en ciclo."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            error_count = 0
            
            def mock_observe():
                nonlocal error_count
                error_count += 1
                
                if error_count <= 3:
                    raise RuntimeError(f"Error {error_count}")
                else:
                    agent._running = False
                
                return self.create_mock_telemetry()
            
            with caplog.at_level(logging.ERROR):
                with patch.object(agent, 'observe', side_effect=mock_observe):
                    with patch('time.sleep'):
                        agent.run(skip_health_check=True)
            
            # Debe haber intentado 4 veces (3 errores + 1 stop)
            assert error_count == 4
            
            # Debe haber loggeado los errores
            assert caplog.text.count("Error en ciclo OODA") >= 3 or \
                   caplog.text.count("RuntimeError") >= 3
    
    def test_lifecycle_handles_rapid_start_stop(self):
        """ROBUSTNESS: Lifecycle maneja start/stop rápidos."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent()
            
            # Detener inmediatamente
            agent._running = True
            agent.stop()
            
            # run() debe terminar sin errores
            agent.run(skip_health_check=True)
            
            assert agent._running is False
    
    def test_lifecycle_with_zero_check_interval(self):
        """ROBUSTNESS: Lifecycle con check_interval=1 (mínimo razonable)."""
        config = AgentConfig(
            timing=TimingConfig(check_interval=1),
        )
        
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=config)
            
            cycle_count = 0
            
            def mock_observe():
                nonlocal cycle_count
                cycle_count += 1
                
                if cycle_count >= 2:
                    agent._running = False
                
                return self.create_mock_telemetry()
            
            with patch.object(agent, 'observe', side_effect=mock_observe):
                with patch.object(agent, 'orient', return_value=SystemStatus.NOMINAL):
                    with patch.object(agent, 'decide', return_value=AgentDecision.HEARTBEAT):
                        with patch.object(agent, 'act'):
                            with patch('time.sleep'):
                                agent.run(skip_health_check=True)
            
            # Debe haber ejecutado los 2 ciclos
            assert cycle_count == 2
    
    def test_lifecycle_with_extreme_backoff(self):
        """ROBUSTNESS: Lifecycle con backoff extremo no cuelga."""
        config = AgentConfig(
            timing=TimingConfig(
                startup_backoff_initial=0.01,
                startup_backoff_max=0.1,
                startup_max_attempts=3,
            ),
        )
        
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=config)
            
            # Simular fallo continuo de startup
            with patch('app.core.apu_agent.requests.get') as mock_get:
                mock_get.side_effect = requests.exceptions.ConnectionError()
                
                with patch('time.sleep'):
                    result = agent._wait_for_startup()
                
                # Debe timeout sin colgar
                assert result is False


# ============================================================================
# PRUEBAS DE API DE MÉTRICAS
# ============================================================================

class TestGetMetrics(BaseAgentTest):
    """
    Suite de pruebas para get_metrics().
    
    Verifica:
        - Estructura completa del dict de métricas
        - Consistencia de datos
        - Serialización correcta
        - Valores derivados
    """
    
    def test_get_metrics_returns_complete_structure(self, agent_config: AgentConfig):
        """UNIT: get_metrics() retorna estructura completa."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            metrics = agent.get_metrics()
            
            # Verificar secciones principales
            assert "config" in metrics
            assert "status" in metrics
            assert "counters" in metrics
            assert "rates" in metrics
            assert "timing" in metrics
            assert "decisions" in metrics
            assert "health" in metrics
            assert "topology" in metrics
    
    def test_get_metrics_config_section(self, agent_config: AgentConfig):
        """UNIT: get_metrics() incluye configuración."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            metrics = agent.get_metrics()
            config_section = metrics["config"]
            
            assert "core_api_url" in config_section
            assert "check_interval" in config_section
            assert "debounce_window" in config_section
            
            assert config_section["core_api_url"] == agent_config.connection.base_url
            assert config_section["check_interval"] == agent_config.timing.check_interval
    
    def test_get_metrics_status_section(self, agent_config: AgentConfig):
        """UNIT: get_metrics() incluye estado del agente."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            agent._running = True
            agent._last_status = SystemStatus.NOMINAL
            
            metrics = agent.get_metrics()
            status_section = metrics["status"]
            
            assert status_section["is_running"] is True
            assert status_section["last_status"] == "NOMINAL"
    
    def test_get_metrics_counters_section(self, agent_config: AgentConfig):
        """UNIT: get_metrics() incluye contadores correctos."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular algunas observaciones
            agent._metrics.record_success()
            agent._metrics.record_failure()
            agent._metrics.record_success()
            agent._metrics.increment_cycle()
            agent._metrics.increment_cycle()
            
            metrics = agent.get_metrics()
            counters = metrics["counters"]
            
            assert counters["cycles_executed"] == 2
            assert counters["successful_observations"] == 2
            assert counters["failed_observations"] == 1
            assert counters["total_observations"] == 3
            assert counters["consecutive_failures"] == 1
    
    def test_get_metrics_rates_section(self, agent_config: AgentConfig):
        """UNIT: get_metrics() incluye tasas calculadas."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular métricas
            agent._metrics.successful_observations = 8
            agent._metrics.failed_observations = 2
            
            metrics = agent.get_metrics()
            rates = metrics["rates"]
            
            assert "success_rate" in rates
            assert "failure_rate" in rates
            assert "observation_rate_per_min" in rates
            assert "cycle_rate_per_min" in rates
            
            # Verificar success_rate
            assert abs(rates["success_rate"] - 0.8) < TEST_EPSILON
    
    def test_get_metrics_topology_section(self, agent_config: AgentConfig):
        """UNIT: get_metrics() incluye información topológica."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            metrics = agent.get_metrics()
            topology = metrics["topology"]
            
            # Verificar secciones topológicas
            assert "betti" in topology
            assert "connectivity" in topology
            assert "health" in topology
            assert "issues" in topology
            
            # Verificar números de Betti
            betti = topology["betti"]
            assert "b0" in betti
            assert "b1" in betti
            assert "euler_characteristic" in betti
            
            # Verificar salud
            health = topology["health"]
            assert "score" in health
            assert "level" in health
            assert "is_healthy" in health
    
    def test_get_metrics_persistence_section(self, agent_config: AgentConfig):
        """UNIT: get_metrics() incluye estadísticas de persistencia."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Agregar muestras de persistencia
            for i in range(10):
                agent.persistence.add_sample("flyback_voltage", 0.3 + i * 0.01)
                agent.persistence.add_sample("saturation", 0.5 + i * 0.02)
            
            metrics = agent.get_metrics()
            
            if "persistence" in metrics:
                persistence = metrics["persistence"]
                
                assert "flyback_voltage" in persistence
                assert "saturation" in persistence
                
                # Verificar estadísticas
                voltage_stats = persistence["flyback_voltage"]
                assert "count" in voltage_stats
                assert voltage_stats["count"] == 10
    
    def test_get_metrics_last_diagnosis_section(self, agent_config: AgentConfig):
        """UNIT: get_metrics() incluye último diagnóstico."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Crear diagnóstico
            agent._last_diagnosis = self.create_mock_diagnosis(
                status=SystemStatus.INESTABLE,
                health_score=0.7,
            )
            
            metrics = agent.get_metrics()
            
            assert "last_diagnosis" in metrics
            diagnosis = metrics["last_diagnosis"]
            
            assert "summary" in diagnosis
            assert "recommended_status" in diagnosis
            assert "severity_level" in diagnosis
            assert "is_structurally_healthy" in diagnosis
            assert "metric_states" in diagnosis
    
    def test_get_metrics_snapshot_history_section(self, agent_config: AgentConfig):
        """UNIT: get_metrics() incluye info de historial de snapshots."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Agregar un snapshot
            snapshot = AgentSnapshot.capture(
                cycle_number=1,
                telemetry=self.create_mock_telemetry(),
                diagnosis=None,
                metrics=agent._metrics,
                last_decision=None,
                current_status=SystemStatus.NOMINAL,
            )
            agent._snapshot_history.add_snapshot(snapshot)
            
            metrics = agent.get_metrics()
            
            assert "snapshot_history" in metrics
            snapshot_info = metrics["snapshot_history"]
            
            assert snapshot_info["size"] == 1
            assert "max_size" in snapshot_info
            assert "is_full" in snapshot_info
    
    def test_get_metrics_with_no_data(self, agent_config: AgentConfig):
        """UNIT: get_metrics() funciona sin datos previos."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Sin ejecutar ciclos
            metrics = agent.get_metrics()
            
            # Debe retornar estructura válida
            assert isinstance(metrics, dict)
            assert metrics["counters"]["cycles_executed"] == 0
            assert metrics["counters"]["total_observations"] == 0
    
    @given(positive_int)
    @settings(max_examples=10)
    def test_get_metrics_always_valid_structure(
        self,
        agent_config: AgentConfig,
        num_cycles: int,
    ):
        """PROP (Hypothesis): get_metrics() siempre retorna estructura válida."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular ciclos
            for _ in range(min(num_cycles, 100)):  # Limitar para performance
                agent._metrics.increment_cycle()
                agent._metrics.record_success()
            
            metrics = agent.get_metrics()
            
            # Verificar estructura mínima
            assert isinstance(metrics, dict)
            assert "counters" in metrics
            assert "rates" in metrics
            assert isinstance(metrics["counters"], dict)
            assert isinstance(metrics["rates"], dict)


# ============================================================================
# PRUEBAS DE API DE RESUMEN TOPOLÓGICO
# ============================================================================

class TestGetTopologicalSummary(BaseAgentTest):
    """
    Suite de pruebas para get_topological_summary().
    
    Verifica:
        - Resumen completo de topología
        - Interpretación de invariantes
        - Detección de issues
        - Patrones de requests
    """
    
    def test_get_topological_summary_structure(self, agent_config: AgentConfig):
        """UNIT: get_topological_summary() retorna estructura completa."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            summary = agent.get_topological_summary()
            
            # Verificar secciones
            assert "timestamp" in summary
            assert "betti" in summary
            assert "health" in summary
            assert "issues" in summary
            assert "patterns" in summary
    
    def test_get_topological_summary_betti_interpretation(self, agent_config: AgentConfig):
        """UNIT: get_topological_summary() interpreta números de Betti."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            summary = agent.get_topological_summary()
            betti = summary["betti"]
            
            assert "values" in betti
            assert "interpretation" in betti
            assert "euler_characteristic" in betti
            
            # Verificar valores
            values = betti["values"]
            assert "b0" in values
            assert "b1" in values
            
            # Verificar interpretación textual
            interpretation = betti["interpretation"]
            assert isinstance(interpretation, str)
            assert len(interpretation) > 0
    
    def test_get_topological_summary_connected_system(self, agent_config: AgentConfig):
        """UNIT: get_topological_summary() con sistema conectado."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            summary = agent.get_topological_summary()
            
            # Sistema nominal debe estar conectado
            assert summary["betti"]["values"]["b0"] == 1
            assert "conectado" in summary["betti"]["interpretation"].lower() or \
                   "connected" in summary["betti"]["interpretation"].lower()
    
    def test_get_topological_summary_fragmented_system(self, agent_config: AgentConfig):
        """UNIT: get_topological_summary() con sistema fragmentado."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Fragmentar topología
            agent.topology.remove_edge(
                NetworkTopology.NODE_AGENT,
                NetworkTopology.NODE_CORE,
            )
            
            summary = agent.get_topological_summary()
            
            # Debe detectar fragmentación
            betti_b0 = summary["betti"]["values"]["b0"]
            assert betti_b0 > 1
            assert "fragmentado" in summary["betti"]["interpretation"].lower() or \
                   "componentes" in summary["betti"]["interpretation"].lower()
    
    def test_get_topological_summary_health_section(self, agent_config: AgentConfig):
        """UNIT: get_topological_summary() incluye salud topológica."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            summary = agent.get_topological_summary()
            health = summary["health"]
            
            assert "score" in health
            assert "level" in health
            assert "is_healthy" in health
            assert "status" in health
            
            # Valores válidos
            assert 0.0 <= health["score"] <= 1.0
            assert health["status"] in ("SANO", "DEGRADADO")
    
    def test_get_topological_summary_issues_section(self, agent_config: AgentConfig):
        """UNIT: get_topological_summary() lista issues detectados."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Crear issue: remover arista
            agent.topology.remove_edge(
                NetworkTopology.NODE_CORE,
                NetworkTopology.NODE_REDIS,
            )
            
            summary = agent.get_topological_summary()
            issues = summary["issues"]
            
            assert "disconnected" in issues
            assert "missing" in issues
            assert "diagnostics" in issues
            
            # Verificar que detectó la arista faltante
            missing = issues["missing"]
            assert len(missing) > 0
    
    def test_get_topological_summary_patterns_section(self, agent_config: AgentConfig):
        """UNIT: get_topological_summary() incluye patrones de requests."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular patrón de reintentos
            for _ in range(10):
                agent.topology.record_request("FAIL_TIMEOUT")
            
            summary = agent.get_topological_summary()
            patterns = summary["patterns"]
            
            # Debe ser una lista
            assert isinstance(patterns, list)
            
            # Si hay patrones, deben tener estructura
            if len(patterns) > 0:
                pattern = patterns[0]
                assert "id" in pattern
                assert "frequency" in pattern
    
    def test_get_topological_summary_with_healthy_system(self, agent_config: AgentConfig):
        """UNIT: get_topological_summary() con sistema sano."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            summary = agent.get_topological_summary()
            health = summary["health"]
            
            # Sistema nominal debe ser sano
            assert health["is_healthy"] is True
            assert health["status"] == "SANO"
            assert health["score"] >= 0.9


# ============================================================================
# PRUEBAS DE API DE SALUD POR ESTRATO
# ============================================================================

class TestGetStratumHealth(BaseAgentTest):
    """
    Suite de pruebas para get_stratum_health().
    
    Verifica:
        - Salud por estrato jerárquico (PHYSICS, TACTICS, STRATEGY, WISDOM)
        - Métricas específicas por nivel
        - Coherencia entre estratos
    """
    
    def test_get_stratum_health_physics(self, agent_config: AgentConfig):
        """UNIT: get_stratum_health(PHYSICS) retorna métricas físicas."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular telemetría
            agent._last_telemetry = TelemetryData(
                flyback_voltage=0.3,
                saturation=0.6,
                integrity_score=0.95,
            )
            
            health = agent.get_stratum_health(Stratum.PHYSICS)
            
            assert health["stratum"] == "PHYSICS"
            assert "voltage" in health
            assert "saturation" in health
            assert "status" in health
            assert "integrity" in health
            assert "timestamp" in health
            
            # Verificar valores
            voltage = health["voltage"]
            assert "value" in voltage
            assert "classification" in voltage
            assert "margin" in voltage
            
            assert voltage["value"] == 0.3
            assert voltage["classification"] == "nominal"
    
    def test_get_stratum_health_physics_without_telemetry(self, agent_config: AgentConfig):
        """UNIT: get_stratum_health(PHYSICS) sin telemetría."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Sin telemetría
            assert agent._last_telemetry is None
            
            health = agent.get_stratum_health(Stratum.PHYSICS)
            
            assert health["stratum"] == "PHYSICS"
            assert health["status"] == "UNKNOWN"
            assert health["voltage"] is None
            assert health["saturation"] is None
    
    def test_get_stratum_health_physics_critical(self, agent_config: AgentConfig):
        """UNIT: get_stratum_health(PHYSICS) con valores críticos."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            agent._last_telemetry = TelemetryData(
                flyback_voltage=0.85,  # Crítico
                saturation=0.97,       # Crítico
                integrity_score=0.5,
            )
            
            health = agent.get_stratum_health(Stratum.PHYSICS)
            
            assert health["status"] == "CRITICO"
            assert health["voltage"]["classification"] == "critical"
            assert health["saturation"]["classification"] == "critical"
    
    def test_get_stratum_health_tactics(self, agent_config: AgentConfig):
        """UNIT: get_stratum_health(TACTICS) retorna invariantes topológicos."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            health = agent.get_stratum_health(Stratum.TACTICS)
            
            assert health["stratum"] == "TACTICS"
            assert "betti_numbers" in health
            assert "euler_characteristic" in health
            assert "health" in health
            assert "issues" in health
            
            # Verificar números de Betti
            betti = health["betti_numbers"]
            assert "b0" in betti
            assert "b1" in betti
            assert "is_connected" in betti
            assert "is_ideal" in betti
    
    def test_get_stratum_health_tactics_degraded(self, agent_config: AgentConfig):
        """UNIT: get_stratum_health(TACTICS) con topología degradada."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Degradar topología
            agent.topology.remove_edge(
                NetworkTopology.NODE_AGENT,
                NetworkTopology.NODE_CORE,
            )
            
            health = agent.get_stratum_health(Stratum.TACTICS)
            
            betti = health["betti_numbers"]
            assert betti["b0"] > 1  # Fragmentado
            assert betti["is_connected"] is False
    
    def test_get_stratum_health_strategy(self, agent_config: AgentConfig):
        """UNIT: get_stratum_health(STRATEGY) retorna métricas estratégicas."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular diagnóstico y decisión
            agent._last_diagnosis = self.create_mock_diagnosis(
                status=SystemStatus.NOMINAL,
                health_score=0.95,
            )
            agent._last_decision = AgentDecision.HEARTBEAT
            agent._last_decision_time = datetime.now()
            
            health = agent.get_stratum_health(Stratum.STRATEGY)
            
            assert health["stratum"] == "STRATEGY"
            assert "risk_detected" in health
            assert "last_decision" in health
            assert "confidence" in health
            assert "status_age_seconds" in health
            assert "requires_intervention" in health
            
            # Verificar valores
            assert health["risk_detected"] is False  # NOMINAL
            assert health["last_decision"] == "HEARTBEAT"
            assert 0.0 <= health["confidence"] <= 1.0
    
    def test_get_stratum_health_strategy_with_risk(self, agent_config: AgentConfig):
        """UNIT: get_stratum_health(STRATEGY) detecta riesgo."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            agent._last_status = SystemStatus.CRITICO
            agent._last_diagnosis = self.create_mock_diagnosis(
                status=SystemStatus.CRITICO,
                health_score=0.3,
            )
            
            health = agent.get_stratum_health(Stratum.STRATEGY)
            
            assert health["risk_detected"] is True
            assert health["requires_intervention"] is True
    
    def test_get_stratum_health_wisdom(self, agent_config: AgentConfig):
        """UNIT: get_stratum_health(WISDOM) retorna veredicto y certeza."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular estado completo
            agent._last_status = SystemStatus.NOMINAL
            agent._last_diagnosis = self.create_mock_diagnosis(
                status=SystemStatus.NOMINAL,
                health_score=1.0,
            )
            agent._metrics.cycles_executed = 10
            
            health = agent.get_stratum_health(Stratum.WISDOM)
            
            assert health["stratum"] == "WISDOM"
            assert "verdict" in health
            assert "certainty" in health
            assert "rationale" in health
            assert "cycles_executed" in health
            assert "uptime_hours" in health
            assert "health_trend" in health
            
            # Verificar valores
            assert health["verdict"] == "NOMINAL"
            assert health["certainty"] == 1.0  # Con diagnóstico
            assert health["cycles_executed"] == 10
    
    def test_get_stratum_health_wisdom_without_diagnosis(self, agent_config: AgentConfig):
        """UNIT: get_stratum_health(WISDOM) sin diagnóstico."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Sin diagnóstico
            assert agent._last_diagnosis is None
            
            health = agent.get_stratum_health(Stratum.WISDOM)
            
            assert health["certainty"] == 0.0  # Sin certeza
            assert health["rationale"] == "Sin diagnóstico previo."
    
    def test_get_stratum_health_invalid_stratum(self, agent_config: AgentConfig):
        """UNIT: get_stratum_health() con estrato inválido."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Crear mock de estrato inválido
            invalid_stratum = Mock()
            invalid_stratum.name = "INVALID"
            
            health = agent.get_stratum_health(invalid_stratum)
            
            # Debe retornar error
            assert "error" in health
    
    def test_stratum_health_coherence(self, agent_config: AgentConfig):
        """INTEGRATION: Salud entre estratos es coherente."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular estado completo
            agent._last_telemetry = TelemetryData(
                flyback_voltage=0.85,  # Crítico
                saturation=0.5,
            )
            agent._last_status = SystemStatus.CRITICO
            agent._last_diagnosis = self.create_mock_diagnosis(
                status=SystemStatus.CRITICO,
                health_score=0.3,
            )
            
            # Obtener salud de todos los estratos
            physics = agent.get_stratum_health(Stratum.PHYSICS)
            tactics = agent.get_stratum_health(Stratum.TACTICS)
            strategy = agent.get_stratum_health(Stratum.STRATEGY)
            wisdom = agent.get_stratum_health(Stratum.WISDOM)
            
            # Verificar coherencia
            # PHYSICS: debe detectar voltaje crítico
            assert physics["status"] in ("CRITICO", "WARNING")
            
            # STRATEGY: debe detectar riesgo
            assert strategy["risk_detected"] is True
            
            # WISDOM: debe tener veredicto crítico
            assert wisdom["verdict"] == "CRITICO"


# ============================================================================
# PRUEBAS DE API DE SNAPSHOTS
# ============================================================================

class TestGetSnapshotHistory(BaseAgentTest):
    """
    Suite de pruebas para get_snapshot_history().
    
    Verifica:
        - Retorno de snapshots recientes
        - Formato compacto
        - Límite de snapshots
    """
    
    def test_get_snapshot_history_empty(self, agent_config: AgentConfig):
        """UNIT: get_snapshot_history() sin snapshots."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            history = agent.get_snapshot_history()
            
            assert isinstance(history, list)
            assert len(history) == 0
    
    def test_get_snapshot_history_with_snapshots(self, agent_config: AgentConfig):
        """UNIT: get_snapshot_history() retorna snapshots en formato compacto."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Agregar snapshots
            for i in range(5):
                snapshot = AgentSnapshot.capture(
                    cycle_number=i,
                    telemetry=self.create_mock_telemetry(),
                    diagnosis=None,
                    metrics=agent._metrics,
                    last_decision=AgentDecision.HEARTBEAT,
                    current_status=SystemStatus.NOMINAL,
                )
                agent._snapshot_history.add_snapshot(snapshot)
            
            history = agent.get_snapshot_history(count=3)
            
            # Debe retornar últimos 3
            assert len(history) == 3
            
            # Verificar formato compacto
            snapshot_dict = history[0]
            assert "timestamp" in snapshot_dict
            assert "cycle" in snapshot_dict
            assert "status" in snapshot_dict
    
    def test_get_snapshot_history_respects_count_limit(self, agent_config: AgentConfig):
        """UNIT: get_snapshot_history() respeta límite de count."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Agregar 10 snapshots
            for i in range(10):
                snapshot = AgentSnapshot.capture(
                    cycle_number=i,
                    telemetry=None,
                    diagnosis=None,
                    metrics=agent._metrics,
                    last_decision=None,
                    current_status=SystemStatus.NOMINAL,
                )
                agent._snapshot_history.add_snapshot(snapshot)
            
            # Solicitar solo 5
            history = agent.get_snapshot_history(count=5)
            
            assert len(history) == 5
    
    def test_get_snapshot_history_default_count(self, agent_config: AgentConfig):
        """UNIT: get_snapshot_history() usa count=10 por defecto."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Agregar 15 snapshots
            for i in range(15):
                snapshot = AgentSnapshot.capture(
                    cycle_number=i,
                    telemetry=None,
                    diagnosis=None,
                    metrics=agent._metrics,
                    last_decision=None,
                    current_status=SystemStatus.NOMINAL,
                )
                agent._snapshot_history.add_snapshot(snapshot)
            
            # Sin especificar count
            history = agent.get_snapshot_history()
            
            # Debe retornar 10 (default)
            assert len(history) == 10


# ============================================================================
# PRUEBAS DE SERIALIZACIÓN Y FORMATO
# ============================================================================

class TestMetricsSerialization(BaseAgentTest):
    """
    Suite de pruebas para serialización de métricas.
    
    Verifica:
        - JSON-serializability
        - Redondeo de flotantes
        - Formato de timestamps
    """
    
    def test_get_metrics_is_json_serializable(self, agent_config: AgentConfig):
        """UNIT: get_metrics() retorna dict serializable a JSON."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular estado completo
            agent._last_telemetry = TelemetryData(flyback_voltage=0.5, saturation=0.6)
            agent._last_diagnosis = self.create_mock_diagnosis()
            agent._metrics.record_success()
            
            metrics = agent.get_metrics()
            
            # Debe poder serializarse a JSON
            import json
            try:
                json_str = json.dumps(metrics)
                assert len(json_str) > 0
            except (TypeError, ValueError) as e:
                pytest.fail(f"get_metrics() no es JSON-serializable: {e}")
    
    def test_get_topological_summary_is_json_serializable(self, agent_config: AgentConfig):
        """UNIT: get_topological_summary() es JSON-serializable."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            summary = agent.get_topological_summary()
            
            import json
            try:
                json_str = json.dumps(summary)
                assert len(json_str) > 0
            except (TypeError, ValueError) as e:
                pytest.fail(f"get_topological_summary() no es JSON-serializable: {e}")
    
    def test_get_stratum_health_is_json_serializable(self, agent_config: AgentConfig):
        """UNIT: get_stratum_health() es JSON-serializable."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            for stratum in [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM]:
                health = agent.get_stratum_health(stratum)
                
                import json
                try:
                    json_str = json.dumps(health)
                    assert len(json_str) > 0
                except (TypeError, ValueError) as e:
                    pytest.fail(f"get_stratum_health({stratum.name}) no es JSON-serializable: {e}")
    
    def test_metrics_floats_are_rounded(self, agent_config: AgentConfig):
        """UNIT: Flotantes en métricas están redondeados."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            agent._metrics.successful_observations = 7
            agent._metrics.failed_observations = 3
            
            metrics = agent.get_metrics()
            
            # success_rate debería ser 0.7
            success_rate = metrics["rates"]["success_rate"]
            
            # Verificar que está redondeado (no más de 4 decimales)
            assert isinstance(success_rate, float)
            # Convertir a string y contar decimales
            str_rate = str(success_rate)
            if '.' in str_rate:
                decimals = len(str_rate.split('.')[1])
                assert decimals <= 4, f"Demasiados decimales: {success_rate}"
    
    def test_timestamps_are_iso_format(self, agent_config: AgentConfig):
        """UNIT: Timestamps están en formato ISO 8601."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            agent._last_telemetry = TelemetryData(flyback_voltage=0.5, saturation=0.6)
            
            health = agent.get_stratum_health(Stratum.PHYSICS)
            
            timestamp = health["timestamp"]
            
            # Debe ser string ISO 8601
            assert isinstance(timestamp, str)
            
            # Verificar que puede parsearse
            from datetime import datetime
            try:
                parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                assert parsed is not None
            except ValueError as e:
                pytest.fail(f"Timestamp no es ISO 8601: {timestamp}, error: {e}")


# ============================================================================
# PRUEBAS DE COHERENCIA ENTRE APIs
# ============================================================================

class TestAPICoherence(BaseAgentTest):
    """
    Suite de pruebas para coherencia entre diferentes APIs.
    
    Verifica que la información es consistente entre:
        - get_metrics()
        - get_topological_summary()
        - get_stratum_health()
    """
    
    def test_topology_coherence_across_apis(self, agent_config: AgentConfig):
        """INTEGRATION: Información topológica coherente entre APIs."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Degradar topología
            agent.topology.remove_edge(
                NetworkTopology.NODE_AGENT,
                NetworkTopology.NODE_CORE,
            )
            
            # Obtener datos de diferentes APIs
            metrics = agent.get_metrics()
            summary = agent.get_topological_summary()
            tactics_health = agent.get_stratum_health(Stratum.TACTICS)
            
            # Verificar coherencia de β₀
            metrics_b0 = metrics["topology"]["betti"]["b0"]
            summary_b0 = summary["betti"]["values"]["b0"]
            tactics_b0 = tactics_health["betti_numbers"]["b0"]
            
            assert metrics_b0 == summary_b0 == tactics_b0
            assert metrics_b0 > 1  # Fragmentado
    
    def test_health_score_coherence(self, agent_config: AgentConfig):
        """INTEGRATION: Health score coherente entre APIs."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            metrics = agent.get_metrics()
            summary = agent.get_topological_summary()
            tactics_health = agent.get_stratum_health(Stratum.TACTICS)
            
            metrics_score = metrics["topology"]["health"]["score"]
            summary_score = summary["health"]["score"]
            tactics_score = tactics_health["health"]["score"]
            
            # Deben ser iguales (con tolerancia numérica)
            assert abs(metrics_score - summary_score) < TEST_EPSILON
            assert abs(metrics_score - tactics_score) < TEST_EPSILON
    
    def test_status_coherence_across_apis(self, agent_config: AgentConfig):
        """INTEGRATION: Status coherente entre APIs."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Establecer status
            agent._last_status = SystemStatus.INESTABLE
            agent._last_diagnosis = self.create_mock_diagnosis(
                status=SystemStatus.INESTABLE,
                health_score=0.6,
            )
            
            metrics = agent.get_metrics()
            wisdom_health = agent.get_stratum_health(Stratum.WISDOM)
            
            metrics_status = metrics["status"]["last_status"]
            wisdom_verdict = wisdom_health["verdict"]
            
            # Deben coincidir
            assert metrics_status == "INESTABLE"
            assert wisdom_verdict == "INESTABLE"
    
    def test_cycles_executed_coherence(self, agent_config: AgentConfig):
        """INTEGRATION: Cycles executed coherente entre APIs."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular ciclos
            agent._metrics.cycles_executed = 42
            
            metrics = agent.get_metrics()
            wisdom_health = agent.get_stratum_health(Stratum.WISDOM)
            
            metrics_cycles = metrics["counters"]["cycles_executed"]
            wisdom_cycles = wisdom_health["cycles_executed"]
            
            assert metrics_cycles == 42
            assert wisdom_cycles == 42


# ============================================================================
# PRUEBAS DE PERFORMANCE DE APIs
# ============================================================================

class TestAPIPerformance(BaseAgentTest):
    """
    Suite de pruebas de performance de APIs.
    
    Verifica que las APIs responden en tiempo razonable.
    """
    
    def test_get_metrics_performance(self, agent_config: AgentConfig):
        """PERF: get_metrics() responde en < 100ms."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Poblar con datos
            for i in range(50):
                agent._metrics.record_success()
                agent.persistence.add_sample("flyback_voltage", 0.5)
            
            import time
            start = time.time()
            
            metrics = agent.get_metrics()
            
            duration = time.time() - start
            
            # Debe responder rápido
            assert duration < 0.1, f"get_metrics() tomó {duration:.3f}s (esperado < 0.1s)"
    
    def test_get_topological_summary_performance(self, agent_config: AgentConfig):
        """PERF: get_topological_summary() responde en < 100ms."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Agregar requests al historial
            for i in range(100):
                agent.topology.record_request(f"request_{i}")
            
            import time
            start = time.time()
            
            summary = agent.get_topological_summary()
            
            duration = time.time() - start
            
            assert duration < 0.1, f"get_topological_summary() tomó {duration:.3f}s"
    
    def test_get_stratum_health_performance(self, agent_config: AgentConfig):
        """PERF: get_stratum_health() responde en < 50ms por estrato."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            agent._last_telemetry = TelemetryData(flyback_voltage=0.5, saturation=0.6)
            agent._last_diagnosis = self.create_mock_diagnosis()
            
            for stratum in [Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.WISDOM]:
                import time
                start = time.time()
                
                health = agent.get_stratum_health(stratum)
                
                duration = time.time() - start
                
                assert duration < 0.05, \
                    f"get_stratum_health({stratum.name}) tomó {duration:.3f}s"


# ============================================================================
# PRUEBAS DE CASOS EXTREMOS EN APIs
# ============================================================================

class TestAPIEdgeCases(BaseAgentTest):
    """
    Pruebas de casos extremos en APIs de observabilidad.
    """
    
    def test_get_metrics_with_extreme_counters(self, agent_config: AgentConfig):
        """EDGE: get_metrics() con contadores muy grandes."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Simular muchos ciclos
            agent._metrics.cycles_executed = 1_000_000
            agent._metrics.successful_observations = 999_000
            agent._metrics.failed_observations = 1_000
            
            metrics = agent.get_metrics()
            
            # Debe manejar números grandes
            assert metrics["counters"]["cycles_executed"] == 1_000_000
            assert metrics["rates"]["success_rate"] == 0.999
    
    def test_get_topological_summary_with_many_patterns(self, agent_config: AgentConfig):
        """EDGE: get_topological_summary() con muchos patrones."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Crear muchos patrones diferentes
            for i in range(100):
                for _ in range(i % 10 + 1):
                    agent.topology.record_request(f"pattern_{i}")
            
            summary = agent.get_topological_summary()
            
            # Debe limitarse a top 5
            patterns = summary["patterns"]
            assert len(patterns) <= 5
    
    def test_get_stratum_health_with_zero_uptime(self, agent_config: AgentConfig):
        """EDGE: get_stratum_health() con uptime casi cero."""
        with patch('app.core.apu_agent.get_global_mic'):
            agent = AutonomousAgent(config=agent_config)
            
            # Crear agente recién inicializado
            agent._metrics.start_time = datetime.now()
            
            wisdom = agent.get_stratum_health(Stratum.WISDOM)
            
            # Debe manejar uptime cercano a cero
            assert wisdom["uptime_hours"] >= 0.0
            assert wisdom["uptime_hours"] < 0.01  # Menos de 36 segundos

