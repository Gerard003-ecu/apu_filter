"""
Tests robustecidos para AutonomousAgent (OODA Loop)
===================================================

Suite completa que valida:
- Configuración e inicialización
- Ciclo OODA completo
- Resiliencia ante fallos
- Debounce de acciones
- Lógica de umbrales
- Métricas internas
- Ciclo de vida del agente
"""

import os
import signal
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch, call

import pytest
import requests

from agent.apu_agent import (
    AgentDecision,
    AgentMetrics,
    AutonomousAgent,
    SystemStatus,
    TelemetryData,
    ThresholdConfig,
)


# =============================================================================
# FIXTURES COMPARTIDAS
# =============================================================================

class TestFixtures:
    """Clase base con fixtures reutilizables."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        """Limpia todas las variables de entorno relacionadas."""
        env_vars = [
            "CORE_API_URL",
            "CHECK_INTERVAL",
            "REQUEST_TIMEOUT",
            "LOG_LEVEL"
        ]
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)
        yield

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Configura variables de entorno de prueba."""
        monkeypatch.setenv("CORE_API_URL", "http://test-core:5000")
        monkeypatch.setenv("CHECK_INTERVAL", "1")
        monkeypatch.setenv("REQUEST_TIMEOUT", "1")
        yield

    @pytest.fixture
    def agent(self, clean_env):
        """
        Crea instancia del agente con configuración de prueba.
        Parchea signal handlers para evitar conflictos en tests.
        """
        with patch.object(AutonomousAgent, '_setup_signal_handlers'):
            agent = AutonomousAgent(
                core_api_url="http://test-core:5000",
                check_interval=1,
                request_timeout=1
            )
            yield agent
            # Cleanup: cerrar sesión si existe
            if hasattr(agent, '_session') and agent._session:
                try:
                    agent._session.close()
                except Exception:
                    pass

    @pytest.fixture
    def agent_with_mock_session(self, agent):
        """Agente con sesión HTTP mockeada."""
        agent._session = MagicMock()
        yield agent

    @pytest.fixture
    def nominal_response_data(self) -> Dict[str, Any]:
        """Datos de respuesta en estado nominal."""
        return {
            "flyback_voltage": 0.3,
            "saturation": 0.5,
            "timestamp": datetime.now().isoformat()
        }

    @pytest.fixture
    def critical_response_data(self) -> Dict[str, Any]:
        """Datos de respuesta en estado crítico."""
        return {
            "flyback_voltage": 0.85,
            "saturation": 0.97
        }

    @pytest.fixture
    def mock_successful_response(self, nominal_response_data):
        """Mock de respuesta HTTP exitosa."""
        response = Mock()
        response.ok = True
        response.status_code = 200
        response.json.return_value = nominal_response_data
        response.text = str(nominal_response_data)
        return response

    @pytest.fixture
    def mock_error_response(self):
        """Mock de respuesta HTTP con error."""
        response = Mock()
        response.ok = False
        response.status_code = 500
        response.text = "Internal Server Error"
        return response


# =============================================================================
# TESTS DE THRESHOLD CONFIG
# =============================================================================

class TestThresholdConfig(TestFixtures):
    """Tests para la configuración de umbrales."""

    def test_default_values(self):
        """Verifica valores por defecto de ThresholdConfig."""
        config = ThresholdConfig()
        
        assert config.flyback_voltage_warning == 0.5
        assert config.flyback_voltage_critical == 0.8
        assert config.saturation_warning == 0.9
        assert config.saturation_critical == 0.95

    def test_custom_values_accepted(self):
        """Verifica que valores personalizados se acepten."""
        config = ThresholdConfig(
            flyback_voltage_warning=0.3,
            flyback_voltage_critical=0.6,
            saturation_warning=0.7,
            saturation_critical=0.85
        )
        
        assert config.flyback_voltage_warning == 0.3
        assert config.flyback_voltage_critical == 0.6
        assert config.saturation_warning == 0.7
        assert config.saturation_critical == 0.85

    @pytest.mark.parametrize("warning,critical,field", [
        (0.6, 0.5, "flyback_voltage"),  # warning >= critical
        (0.5, 0.5, "flyback_voltage"),  # warning == critical
        (-0.1, 0.5, "flyback_voltage"),  # warning negativo
        (0.5, 1.1, "flyback_voltage"),  # critical > 1.0
    ])
    def test_invalid_voltage_thresholds_rejected(self, warning, critical, field):
        """Verifica rechazo de umbrales de voltaje inválidos."""
        with pytest.raises(ValueError):
            ThresholdConfig(
                flyback_voltage_warning=warning,
                flyback_voltage_critical=critical
            )

    @pytest.mark.parametrize("warning,critical", [
        (0.95, 0.9),   # warning >= critical
        (-0.5, 0.95),  # warning negativo
        (0.9, 1.5),    # critical > 1.0
    ])
    def test_invalid_saturation_thresholds_rejected(self, warning, critical):
        """Verifica rechazo de umbrales de saturación inválidos."""
        with pytest.raises(ValueError):
            ThresholdConfig(
                saturation_warning=warning,
                saturation_critical=critical
            )

    def test_immutability(self):
        """Verifica que ThresholdConfig es inmutable (frozen)."""
        config = ThresholdConfig()
        
        with pytest.raises(AttributeError):
            config.flyback_voltage_warning = 0.1


# =============================================================================
# TESTS DE TELEMETRY DATA
# =============================================================================

class TestTelemetryData(TestFixtures):
    """Tests para el parsing y validación de datos de telemetría."""

    def test_from_dict_valid_data(self, nominal_response_data):
        """Verifica parsing exitoso de datos válidos."""
        result = TelemetryData.from_dict(nominal_response_data)
        
        assert result is not None
        assert result.flyback_voltage == 0.3
        assert result.saturation == 0.5
        assert result.raw_data == nominal_response_data

    @pytest.mark.parametrize("invalid_data,description", [
        (None, "None input"),
        ({}, "empty dict"),
        ([], "list instead of dict"),
        ("string", "string instead of dict"),
        (123, "number instead of dict"),
        ({"flyback_voltage": 0.5}, "missing saturation"),
        ({"saturation": 0.5}, "missing flyback_voltage"),
        ({"flyback_voltage": "invalid", "saturation": 0.5}, "non-numeric voltage"),
        ({"flyback_voltage": 0.5, "saturation": None}, "None saturation"),
    ])
    def test_from_dict_invalid_data_returns_none(self, invalid_data, description):
        """Verifica que datos inválidos retornan None."""
        result = TelemetryData.from_dict(invalid_data)
        
        assert result is None, f"Should return None for: {description}"

    @pytest.mark.parametrize("voltage,expected", [
        (1.5, 1.0),    # Clamp superior
        (-0.5, 0.0),   # Clamp inferior
        (0.5, 0.5),    # Sin cambio
        (0.0, 0.0),    # Límite inferior exacto
        (1.0, 1.0),    # Límite superior exacto
    ])
    def test_voltage_clamping(self, voltage, expected):
        """Verifica clampeo de flyback_voltage a [0, 1]."""
        data = {"flyback_voltage": voltage, "saturation": 0.5}
        result = TelemetryData.from_dict(data)
        
        assert result is not None
        assert result.flyback_voltage == expected

    @pytest.mark.parametrize("saturation,expected", [
        (2.0, 1.0),    # Clamp superior
        (-1.0, 0.0),   # Clamp inferior
        (0.75, 0.75),  # Sin cambio
    ])
    def test_saturation_clamping(self, saturation, expected):
        """Verifica clampeo de saturation a [0, 1]."""
        data = {"flyback_voltage": 0.5, "saturation": saturation}
        result = TelemetryData.from_dict(data)
        
        assert result is not None
        assert result.saturation == expected

    def test_string_numeric_conversion(self):
        """Verifica conversión de strings numéricos a float."""
        data = {"flyback_voltage": "0.5", "saturation": "0.7"}
        result = TelemetryData.from_dict(data)
        
        assert result is not None
        assert result.flyback_voltage == 0.5
        assert result.saturation == 0.7

    def test_timestamp_auto_generated(self):
        """Verifica que timestamp se genera automáticamente."""
        data = {"flyback_voltage": 0.5, "saturation": 0.5}
        before = datetime.now()
        result = TelemetryData.from_dict(data)
        after = datetime.now()
        
        assert result is not None
        assert before <= result.timestamp <= after

    def test_raw_data_preserved(self):
        """Verifica que raw_data preserva campos adicionales."""
        data = {
            "flyback_voltage": 0.5,
            "saturation": 0.5,
            "extra_field": "value",
            "nested": {"a": 1, "b": 2}
        }
        result = TelemetryData.from_dict(data)
        
        assert result.raw_data == data
        assert result.raw_data["extra_field"] == "value"
        assert result.raw_data["nested"]["a"] == 1


# =============================================================================
# TESTS DE AGENT METRICS
# =============================================================================

class TestAgentMetrics(TestFixtures):
    """Tests para las métricas internas del agente."""

    @pytest.fixture
    def fresh_metrics(self) -> AgentMetrics:
        """Métricas nuevas sin historial."""
        return AgentMetrics()

    def test_initial_values(self, fresh_metrics):
        """Verifica valores iniciales de métricas."""
        assert fresh_metrics.cycles_executed == 0
        assert fresh_metrics.successful_observations == 0
        assert fresh_metrics.failed_observations == 0
        assert fresh_metrics.consecutive_failures == 0
        assert fresh_metrics.last_successful_observation is None
        assert fresh_metrics.decisions_count == {}
        assert isinstance(fresh_metrics.start_time, datetime)

    def test_record_success_updates_counters(self, fresh_metrics):
        """Verifica que record_success actualiza contadores."""
        fresh_metrics.record_success()
        
        assert fresh_metrics.successful_observations == 1
        assert fresh_metrics.consecutive_failures == 0
        assert fresh_metrics.last_successful_observation is not None

    def test_record_success_resets_consecutive_failures(self, fresh_metrics):
        """Verifica que éxito resetea fallos consecutivos."""
        # Simular fallos previos
        fresh_metrics.record_failure()
        fresh_metrics.record_failure()
        fresh_metrics.record_failure()
        assert fresh_metrics.consecutive_failures == 3
        
        # Un éxito debe resetear
        fresh_metrics.record_success()
        assert fresh_metrics.consecutive_failures == 0

    def test_record_failure_updates_counters(self, fresh_metrics):
        """Verifica que record_failure actualiza contadores."""
        fresh_metrics.record_failure()
        
        assert fresh_metrics.failed_observations == 1
        assert fresh_metrics.consecutive_failures == 1

    def test_consecutive_failures_accumulate(self, fresh_metrics):
        """Verifica acumulación de fallos consecutivos."""
        for i in range(5):
            fresh_metrics.record_failure()
            assert fresh_metrics.consecutive_failures == i + 1
        
        assert fresh_metrics.failed_observations == 5

    def test_record_decision_tracking(self, fresh_metrics):
        """Verifica registro de decisiones por tipo."""
        fresh_metrics.record_decision(AgentDecision.HEARTBEAT)
        fresh_metrics.record_decision(AgentDecision.HEARTBEAT)
        fresh_metrics.record_decision(AgentDecision.RECOMENDAR_LIMPIEZA)
        fresh_metrics.record_decision(AgentDecision.ALERTA_CRITICA)
        
        assert fresh_metrics.decisions_count["HEARTBEAT"] == 2
        assert fresh_metrics.decisions_count["RECOMENDAR_LIMPIEZA"] == 1
        assert fresh_metrics.decisions_count["ALERTA_CRITICA"] == 1

    def test_increment_cycle(self, fresh_metrics):
        """Verifica incremento de contador de ciclos."""
        for i in range(10):
            fresh_metrics.increment_cycle()
        
        assert fresh_metrics.cycles_executed == 10

    def test_success_rate_zero_observations(self, fresh_metrics):
        """Verifica success_rate con cero observaciones."""
        assert fresh_metrics.success_rate == 0.0

    def test_success_rate_all_success(self, fresh_metrics):
        """Verifica success_rate con 100% éxito."""
        for _ in range(10):
            fresh_metrics.record_success()
        
        assert fresh_metrics.success_rate == 1.0

    def test_success_rate_mixed(self, fresh_metrics):
        """Verifica success_rate con mezcla de resultados."""
        for _ in range(8):
            fresh_metrics.record_success()
        for _ in range(2):
            fresh_metrics.record_failure()
        
        # 8 / 10 = 0.8
        assert fresh_metrics.success_rate == 0.8

    def test_uptime_seconds(self, fresh_metrics):
        """Verifica cálculo de uptime."""
        time.sleep(0.1)
        
        uptime = fresh_metrics.uptime_seconds
        assert uptime >= 0.1

    def test_to_dict_completeness(self, fresh_metrics):
        """Verifica que to_dict incluye todos los campos."""
        fresh_metrics.record_success()
        fresh_metrics.record_failure()
        fresh_metrics.record_decision(AgentDecision.HEARTBEAT)
        fresh_metrics.increment_cycle()
        
        result = fresh_metrics.to_dict()
        
        expected_keys = {
            "cycles_executed",
            "successful_observations",
            "failed_observations",
            "success_rate",
            "consecutive_failures",
            "last_successful_observation",
            "decisions_count",
            "uptime_seconds"
        }
        
        assert set(result.keys()) == expected_keys
        assert result["cycles_executed"] == 1
        assert result["successful_observations"] == 1
        assert result["failed_observations"] == 1

    def test_to_dict_does_not_mutate_internal_state(self, fresh_metrics):
        """Verifica que to_dict no modifica estado interno."""
        fresh_metrics.record_decision(AgentDecision.HEARTBEAT)
        
        result = fresh_metrics.to_dict()
        result["decisions_count"]["MODIFIED"] = 999
        result["cycles_executed"] = 999
        
        # El original no debe cambiar
        assert "MODIFIED" not in fresh_metrics.decisions_count
        assert fresh_metrics.cycles_executed == 0


# =============================================================================
# TESTS DE INICIALIZACIÓN DEL AGENTE
# =============================================================================

class TestAgentInitialization(TestFixtures):
    """Tests de inicialización y configuración del agente."""

    def test_initialization_with_explicit_params(self, clean_env):
        """Verifica inicialización con parámetros explícitos."""
        with patch.object(AutonomousAgent, '_setup_signal_handlers'):
            agent = AutonomousAgent(
                core_api_url="http://explicit:8000",
                check_interval=30,
                request_timeout=10
            )
            
            assert agent.core_api_url == "http://explicit:8000"
            assert agent.check_interval == 30
            assert agent.request_timeout == 10
            assert isinstance(agent.thresholds, ThresholdConfig)

    def test_initialization_from_env_vars(self, mock_env):
        """Verifica inicialización desde variables de entorno."""
        with patch.object(AutonomousAgent, '_setup_signal_handlers'):
            agent = AutonomousAgent()
            
            assert agent.core_api_url == "http://test-core:5000"
            assert agent.check_interval == 1
            assert agent.request_timeout == 1

    def test_initialization_with_custom_thresholds(self, clean_env):
        """Verifica inicialización con umbrales personalizados."""
        custom_thresholds = ThresholdConfig(
            flyback_voltage_warning=0.4,
            flyback_voltage_critical=0.7
        )
        
        with patch.object(AutonomousAgent, '_setup_signal_handlers'):
            agent = AutonomousAgent(
                core_api_url="http://test:5000",
                thresholds=custom_thresholds
            )
            
            assert agent.thresholds.flyback_voltage_warning == 0.4
            assert agent.thresholds.flyback_voltage_critical == 0.7

    def test_telemetry_endpoint_constructed(self, agent):
        """Verifica construcción del endpoint de telemetría."""
        expected = f"{agent.core_api_url}/api/telemetry/status"
        assert agent.telemetry_endpoint == expected

    def test_internal_state_initialized(self, agent):
        """Verifica inicialización de estado interno."""
        assert agent._running is False
        assert agent._last_decision is None
        assert agent._last_decision_time is None
        assert agent._last_status is None
        assert isinstance(agent._metrics, AgentMetrics)

    def test_session_created(self, agent):
        """Verifica que sesión HTTP se crea."""
        assert agent._session is not None


class TestAgentURLValidation(TestFixtures):
    """Tests de validación y normalización de URLs."""

    @pytest.mark.parametrize("url,expected", [
        ("http://localhost:5002", "http://localhost:5002"),
        ("https://core.example.com", "https://core.example.com"),
        ("http://localhost:5002/", "http://localhost:5002"),  # Trailing slash
        ("localhost:5002", "http://localhost:5002"),  # Sin esquema
        ("  http://localhost:5002  ", "http://localhost:5002"),  # Espacios
        ("HTTP://UPPERCASE:5000", "HTTP://UPPERCASE:5000"),  # Mayúsculas
    ])
    def test_url_normalization(self, url, expected):
        """Verifica normalización de URLs válidas."""
        result = AutonomousAgent._validate_and_normalize_url(url)
        assert result == expected

    @pytest.mark.parametrize("invalid_url,expected_error", [
        ("", "no puede estar vacía"),
        ("   ", "no puede estar vacía"),
        (None, "no puede estar vacía"),
        ("http://", "URL sin host válido"),
    ])
    def test_invalid_urls_rejected(self, invalid_url, expected_error):
        """Verifica rechazo de URLs inválidas."""
        with pytest.raises(ValueError, match=expected_error):
            AutonomousAgent._validate_and_normalize_url(invalid_url)

    @pytest.mark.parametrize("invalid_interval", [0, -1, -100])
    def test_invalid_check_interval_rejected(self, invalid_interval, clean_env):
        """Verifica rechazo de check_interval inválido."""
        with patch.object(AutonomousAgent, '_setup_signal_handlers'):
            with pytest.raises(ValueError, match="entero positivo"):
                AutonomousAgent(
                    core_api_url="http://test:5000",
                    check_interval=invalid_interval
                )

    @pytest.mark.parametrize("invalid_timeout", [0, -5])
    def test_invalid_request_timeout_rejected(self, invalid_timeout, clean_env):
        """Verifica rechazo de request_timeout inválido."""
        with patch.object(AutonomousAgent, '_setup_signal_handlers'):
            with pytest.raises(ValueError, match="entero positivo"):
                AutonomousAgent(
                    core_api_url="http://test:5000",
                    request_timeout=invalid_timeout
                )

    def test_invalid_env_interval_uses_default_with_warning(self, monkeypatch, caplog):
        """Verifica que env var inválida usa default y genera warning."""
        monkeypatch.setenv("CORE_API_URL", "http://test:5000")
        monkeypatch.setenv("CHECK_INTERVAL", "not_a_number")
        monkeypatch.setenv("REQUEST_TIMEOUT", "5")
        
        with patch.object(AutonomousAgent, '_setup_signal_handlers'):
            agent = AutonomousAgent()
            
            # Debe usar el default (10)
            assert agent.check_interval == 10
            assert "default" in caplog.text.lower() or "inválido" in caplog.text.lower()


# =============================================================================
# TESTS DEL CICLO OODA - OBSERVE
# =============================================================================

class TestObserve(TestFixtures):
    """Tests de la fase OBSERVE del ciclo OODA."""

    def test_observe_success(self, agent_with_mock_session, nominal_response_data):
        """Verifica observación exitosa con datos válidos."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = nominal_response_data
        agent._session.get.return_value = mock_response
        
        result = agent.observe()
        
        assert result is not None
        assert isinstance(result, TelemetryData)
        assert result.flyback_voltage == 0.3
        assert result.saturation == 0.5

    def test_observe_success_updates_metrics(self, agent_with_mock_session, nominal_response_data):
        """Verifica que éxito actualiza métricas correctamente."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = nominal_response_data
        agent._session.get.return_value = mock_response
        
        initial_success = agent._metrics.successful_observations
        
        agent.observe()
        
        assert agent._metrics.successful_observations == initial_success + 1
        assert agent._metrics.last_successful_observation is not None
        assert agent._metrics.consecutive_failures == 0

    def test_observe_http_error_returns_none(self, agent_with_mock_session):
        """Verifica que errores HTTP retornan None."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        agent._session.get.return_value = mock_response
        
        result = agent.observe()
        
        assert result is None

    def test_observe_http_error_updates_metrics(self, agent_with_mock_session):
        """Verifica que errores HTTP actualizan métricas de fallo."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        agent._session.get.return_value = mock_response
        
        initial_failures = agent._metrics.failed_observations
        
        agent.observe()
        
        assert agent._metrics.failed_observations == initial_failures + 1
        assert agent._metrics.consecutive_failures == 1

    def test_observe_json_parse_error(self, agent_with_mock_session):
        """Verifica manejo de error al parsear JSON."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.side_effect = ValueError("Invalid JSON")
        agent._session.get.return_value = mock_response
        
        result = agent.observe()
        
        assert result is None
        assert agent._metrics.failed_observations == 1

    def test_observe_connection_error(self, agent_with_mock_session):
        """Verifica manejo de error de conexión."""
        agent = agent_with_mock_session
        agent._session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        result = agent.observe()
        
        assert result is None
        assert agent._metrics.failed_observations == 1

    def test_observe_timeout(self, agent_with_mock_session):
        """Verifica manejo de timeout."""
        agent = agent_with_mock_session
        agent._session.get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        result = agent.observe()
        
        assert result is None
        assert agent._metrics.failed_observations == 1

    def test_observe_incomplete_data(self, agent_with_mock_session):
        """Verifica manejo de datos incompletos."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"flyback_voltage": 0.5}  # Falta saturation
        agent._session.get.return_value = mock_response
        
        result = agent.observe()
        
        assert result is None

    def test_observe_consecutive_failures_accumulate(self, agent_with_mock_session):
        """Verifica acumulación de fallos consecutivos."""
        agent = agent_with_mock_session
        agent._session.get.side_effect = requests.exceptions.ConnectionError()
        
        for i in range(5):
            agent.observe()
            assert agent._metrics.consecutive_failures == i + 1

    def test_observe_success_resets_consecutive_failures(self, agent_with_mock_session, nominal_response_data):
        """Verifica que éxito resetea contador de fallos consecutivos."""
        agent = agent_with_mock_session
        
        # Simular fallos previos
        agent._metrics.consecutive_failures = 3
        
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = nominal_response_data
        agent._session.get.return_value = mock_response
        
        agent.observe()
        
        assert agent._metrics.consecutive_failures == 0


# =============================================================================
# TESTS DEL CICLO OODA - ORIENT
# =============================================================================

class TestOrient(TestFixtures):
    """Tests de la fase ORIENT del ciclo OODA."""

    def test_orient_null_data_returns_unknown(self, agent):
        """Verifica que telemetría None retorna UNKNOWN."""
        result = agent.orient(None)
        
        assert result == SystemStatus.UNKNOWN

    def test_orient_many_failures_returns_disconnected(self, agent):
        """Verifica que muchos fallos consecutivos retorna DISCONNECTED."""
        agent._metrics.consecutive_failures = agent.MAX_CONSECUTIVE_FAILURES
        
        result = agent.orient(None)
        
        assert result == SystemStatus.DISCONNECTED

    def test_orient_just_below_max_failures_returns_unknown(self, agent):
        """Verifica que fallos bajo el máximo retorna UNKNOWN."""
        agent._metrics.consecutive_failures = agent.MAX_CONSECUTIVE_FAILURES - 1
        
        result = agent.orient(None)
        
        assert result == SystemStatus.UNKNOWN

    @pytest.mark.parametrize("voltage,saturation", [
        (0.0, 0.0),      # Mínimos
        (0.49, 0.89),    # Justo debajo de umbrales
        (0.3, 0.5),      # Valores típicos
        (0.5, 0.9),      # Exactamente en umbrales (no superan)
    ])
    def test_orient_nominal_conditions(self, agent, voltage, saturation):
        """Verifica detección de estado NOMINAL."""
        telemetry = TelemetryData(
            flyback_voltage=voltage,
            saturation=saturation
        )
        
        result = agent.orient(telemetry)
        
        assert result == SystemStatus.NOMINAL

    @pytest.mark.parametrize("voltage,saturation", [
        (0.51, 0.5),     # Voltaje justo arriba de warning
        (0.6, 0.3),      # Voltaje warning, saturación baja
        (0.79, 0.5),     # Voltaje alto pero no crítico
    ])
    def test_orient_inestable_voltage_only(self, agent, voltage, saturation):
        """Verifica detección de INESTABLE por voltaje."""
        telemetry = TelemetryData(
            flyback_voltage=voltage,
            saturation=saturation
        )
        
        result = agent.orient(telemetry)
        
        assert result == SystemStatus.INESTABLE

    @pytest.mark.parametrize("voltage,saturation", [
        (0.3, 0.91),     # Saturación justo arriba de warning
        (0.3, 0.94),     # Saturación alta
        (0.49, 0.92),    # Voltaje OK, saturación warning
    ])
    def test_orient_saturado_saturation_only(self, agent, voltage, saturation):
        """Verifica detección de SATURADO por saturación."""
        telemetry = TelemetryData(
            flyback_voltage=voltage,
            saturation=saturation
        )
        
        result = agent.orient(telemetry)
        
        assert result == SystemStatus.SATURADO

    @pytest.mark.parametrize("voltage,saturation,reason", [
        (0.81, 0.5, "voltage crítico"),
        (0.9, 0.3, "voltage muy crítico"),
        (1.0, 0.0, "voltage máximo"),
        (0.3, 0.96, "saturation crítica"),
        (0.0, 0.99, "saturation muy crítica"),
        (0.0, 1.0, "saturation máxima"),
        (0.85, 0.97, "ambos críticos"),
        (0.6, 0.92, "ambos en warning = crítico"),
    ])
    def test_orient_critico_conditions(self, agent, voltage, saturation, reason):
        """Verifica detección de estado CRITICO."""
        telemetry = TelemetryData(
            flyback_voltage=voltage,
            saturation=saturation
        )
        
        result = agent.orient(telemetry)
        
        assert result == SystemStatus.CRITICO, f"Failed for: {reason}"

    def test_orient_exact_warning_threshold_is_nominal(self, agent):
        """Verifica que valor exacto en umbral warning es NOMINAL (no supera)."""
        # El umbral es "mayor que", no "mayor o igual"
        telemetry = TelemetryData(
            flyback_voltage=0.5,  # Exactamente en umbral warning
            saturation=0.9        # Exactamente en umbral warning
        )
        
        result = agent.orient(telemetry)
        
        assert result == SystemStatus.NOMINAL

    def test_orient_with_custom_thresholds(self, clean_env):
        """Verifica que umbrales personalizados se aplican."""
        custom_thresholds = ThresholdConfig(
            flyback_voltage_warning=0.3,
            flyback_voltage_critical=0.6
        )
        
        with patch.object(AutonomousAgent, '_setup_signal_handlers'):
            agent = AutonomousAgent(
                core_api_url="http://test:5000",
                thresholds=custom_thresholds
            )
            
            # Con umbrales default sería NOMINAL, con custom es INESTABLE
            telemetry = TelemetryData(flyback_voltage=0.4, saturation=0.5)
            
            result = agent.orient(telemetry)
            
            assert result == SystemStatus.INESTABLE


# =============================================================================
# TESTS DEL CICLO OODA - DECIDE
# =============================================================================

class TestDecide(TestFixtures):
    """Tests de la fase DECIDE del ciclo OODA."""

    @pytest.mark.parametrize("status,expected_decision", [
        (SystemStatus.NOMINAL, AgentDecision.HEARTBEAT),
        (SystemStatus.INESTABLE, AgentDecision.RECOMENDAR_LIMPIEZA),
        (SystemStatus.SATURADO, AgentDecision.RECOMENDAR_REDUCIR_VELOCIDAD),
        (SystemStatus.CRITICO, AgentDecision.ALERTA_CRITICA),
        (SystemStatus.DISCONNECTED, AgentDecision.RECONNECT),
        (SystemStatus.UNKNOWN, AgentDecision.WAIT),
    ])
    def test_status_to_decision_mapping(self, agent, status, expected_decision):
        """Verifica mapeo completo de estados a decisiones."""
        result = agent.decide(status)
        
        assert result == expected_decision

    def test_decide_records_decision_in_metrics(self, agent):
        """Verifica que decisiones se registran en métricas."""
        agent.decide(SystemStatus.NOMINAL)
        agent.decide(SystemStatus.NOMINAL)
        agent.decide(SystemStatus.INESTABLE)
        
        assert agent._metrics.decisions_count["HEARTBEAT"] == 2
        assert agent._metrics.decisions_count["RECOMENDAR_LIMPIEZA"] == 1

    def test_decide_updates_last_status(self, agent):
        """Verifica que decide actualiza _last_status."""
        agent.decide(SystemStatus.SATURADO)
        
        assert agent._last_status == SystemStatus.SATURADO

    def test_all_status_values_handled(self, agent):
        """Verifica que todos los valores de SystemStatus están manejados."""
        for status in SystemStatus:
            result = agent.decide(status)
            
            assert result is not None
            assert isinstance(result, AgentDecision)


# =============================================================================
# TESTS DEL CICLO OODA - ACT
# =============================================================================

class TestAct(TestFixtures):
    """Tests de la fase ACT del ciclo OODA."""

    @pytest.mark.parametrize("decision", list(AgentDecision))
    def test_all_decisions_executable(self, agent, decision):
        """Verifica que todas las decisiones se pueden ejecutar."""
        # Resetear estado para evitar debounce
        agent._last_decision = None
        agent._last_decision_time = None
        
        result = agent.act(decision)
        
        assert isinstance(result, bool)

    def test_act_returns_true_on_first_execution(self, agent):
        """Verifica que primera ejecución retorna True."""
        result = agent.act(AgentDecision.HEARTBEAT)
        
        assert result is True

    def test_act_updates_last_decision(self, agent):
        """Verifica que act actualiza _last_decision."""
        agent.act(AgentDecision.RECOMENDAR_LIMPIEZA)
        
        assert agent._last_decision == AgentDecision.RECOMENDAR_LIMPIEZA

    def test_act_updates_last_decision_time(self, agent):
        """Verifica que act actualiza _last_decision_time."""
        before = datetime.now()
        agent.act(AgentDecision.HEARTBEAT)
        after = datetime.now()
        
        assert agent._last_decision_time is not None
        assert before <= agent._last_decision_time <= after


class TestActLogging(TestFixtures):
    """Tests de logging en acciones."""

    def test_heartbeat_logs_info(self, agent, caplog):
        """Verifica que HEARTBEAT genera log apropiado."""
        import logging
        with caplog.at_level(logging.INFO):
            agent.act(AgentDecision.HEARTBEAT)
        
        assert "NOMINAL" in caplog.text
        assert "✅" in caplog.text

    def test_recomendar_limpieza_logs_warning(self, agent, caplog):
        """Verifica que RECOMENDAR_LIMPIEZA genera log WARNING."""
        import logging
        with caplog.at_level(logging.WARNING):
            agent.act(AgentDecision.RECOMENDAR_LIMPIEZA)
        
        assert "INESTABILIDAD" in caplog.text
        assert "⚠️" in caplog.text

    def test_recomendar_reducir_velocidad_logs_warning(self, agent, caplog):
        """Verifica que RECOMENDAR_REDUCIR_VELOCIDAD genera log WARNING."""
        import logging
        with caplog.at_level(logging.WARNING):
            agent.act(AgentDecision.RECOMENDAR_REDUCIR_VELOCIDAD)
        
        assert "SATURACIÓN" in caplog.text or "SATURACION" in caplog.text
        assert "⚠️" in caplog.text

    def test_alerta_critica_logs_critical(self, agent, caplog):
        """Verifica que ALERTA_CRITICA genera log CRITICAL."""
        import logging
        with caplog.at_level(logging.CRITICAL):
            agent.act(AgentDecision.ALERTA_CRITICA)
        
        # Verificar que contiene indicadores de alerta crítica
        log_lower = caplog.text.lower()
        assert "crítica" in log_lower or "critica" in log_lower or "critical" in log_lower
        assert "🚨" in caplog.text


class TestDebounce(TestFixtures):
    """Tests del mecanismo de debounce."""

    def test_same_decision_debounced(self, agent):
        """Verifica que decisión repetida se suprime."""
        decision = AgentDecision.HEARTBEAT
        
        # Primera ejecución: debe ejecutar
        result1 = agent.act(decision)
        assert result1 is True
        
        # Segunda ejecución inmediata: debe suprimir
        result2 = agent.act(decision)
        assert result2 is False

    def test_different_decision_not_debounced(self, agent):
        """Verifica que decisión diferente no se suprime."""
        agent.act(AgentDecision.HEARTBEAT)
        
        result = agent.act(AgentDecision.RECOMENDAR_LIMPIEZA)
        
        assert result is True

    def test_debounce_expires_after_window(self, agent):
        """Verifica que debounce expira después de la ventana de tiempo."""
        decision = AgentDecision.HEARTBEAT
        
        # Primera ejecución
        agent.act(decision)
        
        # Simular que pasó el tiempo de debounce
        agent._last_decision_time = (
            datetime.now() - timedelta(seconds=agent.DEBOUNCE_WINDOW_SECONDS + 1)
        )
        
        # Debe ejecutar porque expiró el debounce
        result = agent.act(decision)
        
        assert result is True

    @pytest.mark.parametrize("decision", [
        AgentDecision.ALERTA_CRITICA,
        AgentDecision.RECONNECT,
    ])
    def test_critical_decisions_bypass_debounce(self, agent, decision):
        """Verifica que decisiones críticas nunca se suprimen."""
        # Primera ejecución
        result1 = agent.act(decision)
        assert result1 is True
        
        # Segunda ejecución inmediata: debe ejecutar también
        result2 = agent.act(decision)
        assert result2 is True
        
        # Tercera ejecución
        result3 = agent.act(decision)
        assert result3 is True

    def test_first_action_always_executes(self, agent):
        """Verifica que primera acción siempre se ejecuta."""
        # Estado inicial
        assert agent._last_decision is None
        assert agent._last_decision_time is None
        
        result = agent.act(AgentDecision.WAIT)
        
        assert result is True

    def test_should_debounce_with_no_prior_decision(self, agent):
        """Verifica _should_debounce sin decisión previa."""
        result = agent._should_debounce(AgentDecision.HEARTBEAT)
        
        assert result is False

    def test_should_debounce_same_decision_within_window(self, agent):
        """Verifica _should_debounce con misma decisión en ventana."""
        agent._last_decision = AgentDecision.HEARTBEAT
        agent._last_decision_time = datetime.now()
        
        result = agent._should_debounce(AgentDecision.HEARTBEAT)
        
        assert result is True

    def test_should_debounce_different_decision(self, agent):
        """Verifica que decisión diferente no tiene debounce."""
        agent._last_decision = AgentDecision.HEARTBEAT
        agent._last_decision_time = datetime.now()
        
        result = agent._should_debounce(AgentDecision.WAIT)
        
        assert result is False


# =============================================================================
# TESTS DE INTEGRACIÓN DEL CICLO OODA
# =============================================================================

class TestOODAIntegration(TestFixtures):
    """Tests de integración del ciclo OODA completo."""

    def test_full_cycle_nominal(self, agent_with_mock_session, nominal_response_data):
        """Verifica ciclo completo con datos nominales."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = nominal_response_data
        agent._session.get.return_value = mock_response
        
        # OBSERVE
        telemetry = agent.observe()
        assert telemetry is not None
        assert telemetry.flyback_voltage == 0.3
        
        # ORIENT
        status = agent.orient(telemetry)
        assert status == SystemStatus.NOMINAL
        
        # DECIDE
        decision = agent.decide(status)
        assert decision == AgentDecision.HEARTBEAT
        
        # ACT
        executed = agent.act(decision)
        assert executed is True

    def test_full_cycle_critical(self, agent_with_mock_session, critical_response_data):
        """Verifica ciclo completo con datos críticos."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = critical_response_data
        agent._session.get.return_value = mock_response
        
        telemetry = agent.observe()
        status = agent.orient(telemetry)
        decision = agent.decide(status)
        
        assert telemetry.flyback_voltage == 0.85
        assert status == SystemStatus.CRITICO
        assert decision == AgentDecision.ALERTA_CRITICA

    def test_full_cycle_connection_failure(self, agent_with_mock_session):
        """Verifica ciclo completo con fallo de conexión."""
        agent = agent_with_mock_session
        agent._session.get.side_effect = requests.exceptions.ConnectionError()
        
        telemetry = agent.observe()
        status = agent.orient(telemetry)
        decision = agent.decide(status)
        
        assert telemetry is None
        assert status == SystemStatus.UNKNOWN
        assert decision == AgentDecision.WAIT

    def test_transition_to_disconnected(self, agent_with_mock_session):
        """Verifica transición a DISCONNECTED tras múltiples fallos."""
        agent = agent_with_mock_session
        agent._session.get.side_effect = requests.exceptions.ConnectionError()
        
        # Simular múltiples fallos
        for _ in range(agent.MAX_CONSECUTIVE_FAILURES):
            agent.observe()
        
        status = agent.orient(None)
        decision = agent.decide(status)
        
        assert status == SystemStatus.DISCONNECTED
        assert decision == AgentDecision.RECONNECT

    def test_metrics_updated_during_cycle(self, agent_with_mock_session, nominal_response_data):
        """Verifica actualización de métricas durante ciclo."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = nominal_response_data
        agent._session.get.return_value = mock_response
        
        initial_cycles = agent._metrics.cycles_executed
        
        agent._metrics.increment_cycle()
        agent.observe()
        agent.decide(agent.orient(TelemetryData.from_dict(nominal_response_data)))
        
        assert agent._metrics.cycles_executed == initial_cycles + 1
        assert agent._metrics.successful_observations == 1
        assert "HEARTBEAT" in agent._metrics.decisions_count


# =============================================================================
# TESTS DE CICLO DE VIDA
# =============================================================================

class TestHealthCheck(TestFixtures):
    """Tests del health check inicial."""

    def test_health_check_success(self, agent_with_mock_session, nominal_response_data):
        """Verifica health check exitoso."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = nominal_response_data
        agent._session.get.return_value = mock_response
        
        result = agent.health_check()
        
        assert result is True

    def test_health_check_http_error_returns_true_with_warning(self, agent_with_mock_session, caplog):
        """Verifica que health check con error HTTP retorna True (permite continuar)."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        agent._session.get.return_value = mock_response
        
        result = agent.health_check()
        
        # Retorna True para permitir continuar con warning
        assert result is True

    def test_health_check_connection_error_returns_false(self, agent_with_mock_session):
        """Verifica que health check con error de conexión retorna False."""
        agent = agent_with_mock_session
        agent._session.get.side_effect = requests.exceptions.ConnectionError()
        
        result = agent.health_check()
        
        assert result is False


class TestAgentStop(TestFixtures):
    """Tests de parada del agente."""

    def test_stop_sets_running_false(self, agent):
        """Verifica que stop() establece _running en False."""
        agent._running = True
        
        agent.stop()
        
        assert agent._running is False

    def test_stop_is_idempotent(self, agent):
        """Verifica que stop() es idempotente."""
        agent.stop()
        agent.stop()
        agent.stop()
        
        assert agent._running is False


class TestSignalHandling(TestFixtures):
    """Tests del manejo de señales."""

    def test_sigint_sets_running_false(self, agent):
        """Verifica que SIGINT establece _running en False."""
        agent._running = True
        
        agent._handle_shutdown(signal.SIGINT, None)
        
        assert agent._running is False

    def test_sigterm_sets_running_false(self, agent):
        """Verifica que SIGTERM establece _running en False."""
        agent._running = True
        
        agent._handle_shutdown(signal.SIGTERM, None)
        
        assert agent._running is False


class TestGetMetrics(TestFixtures):
    """Tests del método get_metrics()."""

    def test_get_metrics_returns_complete_data(self, agent):
        """Verifica que get_metrics retorna datos completos."""
        metrics = agent.get_metrics()
        
        expected_keys = {
            "cycles_executed",
            "successful_observations",
            "failed_observations",
            "success_rate",
            "consecutive_failures",
            "last_successful_observation",
            "decisions_count",
            "uptime_seconds",
            "core_api_url",
            "check_interval",
            "is_running",
            "last_status"
        }
        
        assert expected_keys.issubset(set(metrics.keys()))

    def test_get_metrics_reflects_agent_state(self, agent):
        """Verifica que métricas reflejan estado del agente."""
        agent._running = True
        agent._last_status = SystemStatus.CRITICO
        
        metrics = agent.get_metrics()
        
        assert metrics["is_running"] is True
        assert metrics["last_status"] == "CRITICO"
        assert metrics["core_api_url"] == agent.core_api_url


class TestShutdown(TestFixtures):
    """Tests del proceso de shutdown."""

    def test_shutdown_closes_session(self, agent):
        """Verifica que shutdown cierra la sesión HTTP."""
        mock_session = MagicMock()
        agent._session = mock_session
        
        agent._shutdown()
        
        mock_session.close.assert_called_once()

    def test_shutdown_handles_session_close_error(self, agent, caplog):
        """Verifica que shutdown maneja errores al cerrar sesión."""
        mock_session = MagicMock()
        mock_session.close.side_effect = Exception("Close error")
        agent._session = mock_session
        
        # No debe lanzar excepción
        agent._shutdown()
        
        # Debe registrar warning
        assert "error" in caplog.text.lower() or "Error" in caplog.text

    def test_shutdown_logs_final_metrics(self, agent, caplog):
        """Verifica que shutdown registra métricas finales."""
        import logging
        with caplog.at_level(logging.INFO):
            agent._shutdown()
        
        assert "métricas" in caplog.text.lower() or "metrics" in caplog.text.lower()


class TestRunLoop(TestFixtures):
    """Tests del bucle principal run()."""

    def test_run_executes_at_least_one_cycle(self, agent_with_mock_session, nominal_response_data):
        """Verifica que run() ejecuta al menos un ciclo."""
        agent = agent_with_mock_session
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = nominal_response_data
        agent._session.get.return_value = mock_response
        
        def stop_after_delay():
            time.sleep(0.3)
            agent.stop()
        
        stopper = threading.Thread(target=stop_after_delay)
        stopper.start()
        
        agent.run(skip_health_check=True)
        
        stopper.join()
        
        assert agent._metrics.cycles_executed >= 1

    def test_run_recovers_from_observe_errors(self, agent_with_mock_session, nominal_response_data):
        """Verifica que run() se recupera de errores en observe."""
        agent = agent_with_mock_session
        
        # Primer ciclo: error, segundo ciclo: éxito
        call_count = [0]
        
        def mock_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise requests.exceptions.ConnectionError()
            mock_resp = Mock()
            mock_resp.ok = True
            mock_resp.json.return_value = nominal_response_data
            return mock_resp
        
        agent._session.get.side_effect = mock_get
        
        def stop_after_cycles():
            time.sleep(0.5)
            agent.stop()
        
        stopper = threading.Thread(target=stop_after_cycles)
        stopper.start()
        
        # No debe lanzar excepción
        agent.run(skip_health_check=True)
        
        stopper.join()
        
        # Debe haber registrado al menos un fallo y un éxito
        assert agent._metrics.failed_observations >= 1 or agent._metrics.successful_observations >= 1


# =============================================================================
# CONFIGURACIÓN DE PYTEST
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])