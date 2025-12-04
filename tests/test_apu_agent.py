"""
Tests para AutonomousAgent (OODA Loop)
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import os
import requests

from agent.apu_agent import (
    AutonomousAgent,
    SystemStatus,
    AgentDecision,
    TelemetryData,
    ThresholdConfig
)

class TestAutonomousAgent:

    @pytest.fixture
    def mock_env(self):
        """Mock variables de entorno básicas"""
        with patch.dict(os.environ, {
            "CORE_API_URL": "http://test-core:5000",
            "CHECK_INTERVAL": "1",
            "REQUEST_TIMEOUT": "1"
        }):
            yield

    @pytest.fixture
    def agent(self, mock_env):
        """Instancia un agente con configuración de prueba"""
        return AutonomousAgent(
            core_api_url="http://test-core:5000",
            check_interval=1,
            request_timeout=1
        )

    # 1. Test de Inicialización
    def test_initialization_config(self, mock_env):
        """Verificar carga correcta de configuración"""
        agent = AutonomousAgent()
        assert agent.core_api_url == "http://test-core:5000"
        assert agent.check_interval == 1
        assert agent.request_timeout == 1
        assert isinstance(agent.thresholds, ThresholdConfig)

    def test_invalid_url_validation(self):
        """Verificar validación de URL"""
        # Test static method directly to bypass __init__ defaults logic
        with pytest.raises(ValueError, match="no puede estar vacía"):
            AutonomousAgent._validate_and_normalize_url("")

        with pytest.raises(ValueError, match="no puede estar vacía"):
            AutonomousAgent._validate_and_normalize_url(None)

        with pytest.raises(ValueError, match="URL sin host válido"):
            AutonomousAgent._validate_and_normalize_url("http://")

    # 2. Test del Ciclo OODA (Mockeado)
    @patch('agent.apu_agent.requests.Session')
    def test_ooda_loop_flow(self, mock_session_cls, agent):
        """
        Simula un ciclo completo:
        1. Observe (Data crítica)
        2. Orient (Detecta CRITICO)
        3. Decide (ALERTA_CRITICA)
        4. Act (Logging)
        """
        # Setup Mock de requests
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "flyback_voltage": 0.9,  # Crítico (>0.8)
            "saturation": 0.5
        }

        agent._session.get = Mock(return_value=mock_response)

        # Execute steps manually
        telemetry = agent.observe()
        status = agent.orient(telemetry)
        decision = agent.decide(status)
        executed = agent.act(decision)

        # Assertions
        assert telemetry.flyback_voltage == 0.9
        assert status == SystemStatus.CRITICO
        assert decision == AgentDecision.ALERTA_CRITICA
        assert executed is True

    # 3. Test de Resiliencia
    def test_resilience_connection_error(self, agent):
        """Simula error de conexión en observe()"""
        agent._session.get = Mock(side_effect=requests.exceptions.ConnectionError("Connection refused"))

        telemetry = agent.observe()

        assert telemetry is None
        assert agent._metrics.failed_observations == 1

    def test_resilience_malformed_data(self, agent):
        """Simula respuesta JSON inválida"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.side_effect = ValueError("Invalid JSON")
        agent._session.get = Mock(return_value=mock_response)

        telemetry = agent.observe()
        assert telemetry is None

    # 4. Test de Debounce
    def test_debounce_logic(self, agent):
        """Verificar que acciones repetitivas se suprimen"""
        decision = AgentDecision.RECOMENDAR_LIMPIEZA

        # Primera llamada: Debe ejecutar
        assert agent.act(decision) is True

        # Segunda llamada inmediata: Debe suprimir
        assert agent.act(decision) is False

        # Simular paso del tiempo
        agent._last_decision_time = datetime.now() - timedelta(minutes=2)

        # Tercera llamada tras tiempo: Debe ejecutar
        assert agent.act(decision) is True

    def test_critical_alerts_bypass_debounce(self, agent):
        """Alertas críticas nunca deben ser debounced"""
        decision = AgentDecision.ALERTA_CRITICA

        assert agent.act(decision) is True
        assert agent.act(decision) is True  # Should still run

    # 5. Tests de Thresholds
    def test_orient_logic_boundaries(self, agent):
        """Verificar lógica de clasificación de estados"""

        # Caso Nominal
        t_nominal = TelemetryData(flyback_voltage=0.1, saturation=0.1)
        assert agent.orient(t_nominal) == SystemStatus.NOMINAL

        # Caso Warning (Inestable)
        t_warn_volt = TelemetryData(flyback_voltage=0.6, saturation=0.1)
        assert agent.orient(t_warn_volt) == SystemStatus.INESTABLE

        # Caso Crítico
        t_crit = TelemetryData(flyback_voltage=0.85, saturation=0.1)
        assert agent.orient(t_crit) == SystemStatus.CRITICO

    # 6. Test de Métricas
    def test_metrics_collection(self, agent):
        """Verificar que las métricas se actualizan"""
        initial_cycles = agent._metrics.cycles_executed
        agent._metrics.increment_cycle()
        assert agent._metrics.cycles_executed == initial_cycles + 1

        agent._metrics.record_decision(AgentDecision.HEARTBEAT)
        assert agent._metrics.decisions_count['HEARTBEAT'] == 1
