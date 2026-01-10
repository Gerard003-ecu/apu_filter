"""
Tests robustecidos para AutonomousAgent (OODA Loop con Topología)
=================================================================

Suite actualizada que valida:
- Ciclo OODA con integración topológica
- Lógica de decisión basada en homología persistente
- Detección de bucles de reintentos
- Métricas enriquecidas
- Configuración y validación de datos (Tests restaurados)
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from agent.apu_agent import (
    AgentDecision,
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
# FIXTURES Y MOCKS
# =============================================================================


class TestFixtures:
    """Clase base con fixtures reutilizables."""

    @pytest.fixture
    def clean_env(self, monkeypatch):
        """Limpia variables de entorno."""
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
    def mock_env(self, monkeypatch):
        """Configura variables de entorno de prueba."""
        monkeypatch.setenv("CORE_API_URL", "http://test-core:5000")
        monkeypatch.setenv("CHECK_INTERVAL", "1")
        monkeypatch.setenv("REQUEST_TIMEOUT", "1")
        yield

    @pytest.fixture
    def mock_topology(self):
        """Mock de SystemTopology."""
        with patch("agent.apu_agent.SystemTopology") as mock:
            instance = mock.return_value
            # Default healthy state
            instance.get_topological_health.return_value = TopologicalHealth(
                betti=BettiNumbers(b0=1, b1=0),
                disconnected_nodes=frozenset(),
                missing_edges=frozenset(),
                request_loops=tuple(),
                health_score=1.0,
                level=HealthLevel.HEALTHY,
                diagnostics={},
            )
            instance.calculate_betti_numbers.return_value = BettiNumbers(b0=1, b1=0)
            # Fix: update_connectivity must return a tuple (count, warnings)
            instance.update_connectivity.return_value = (3, [])
            yield instance

    @pytest.fixture
    def mock_persistence(self):
        """Mock de PersistenceHomology."""
        with patch("agent.apu_agent.PersistenceHomology") as mock:
            instance = mock.return_value
            # Default stable state
            instance.analyze_persistence.return_value = PersistenceAnalysisResult(
                state=MetricState.STABLE,
                intervals=tuple(),
                feature_count=0,
                noise_count=0,
                active_count=0,
                max_lifespan=0,
                total_persistence=0,
                metadata={},
            )
            yield instance

    @pytest.fixture
    def agent(self, clean_env, mock_topology, mock_persistence):
        """
        Agente configurado con mocks topológicos.
        """
        with patch.object(AutonomousAgent, "_setup_signal_handlers"):
            agent = AutonomousAgent(
                core_api_url="http://test-core:5000", check_interval=1, request_timeout=1
            )
            # Mock session
            agent._session = MagicMock()
            yield agent
            if hasattr(agent, "_session") and agent._session:
                agent._session.close()

    @pytest.fixture
    def nominal_response_data(self) -> Dict[str, Any]:
        """Datos de respuesta nominal para pruebas."""
        return {
            "flyback_voltage": 0.3,
            "saturation": 0.5,
            "timestamp": datetime.now().isoformat(),
            "redis_connected": True,
            "filesystem_accessible": True,
        }


# =============================================================================
# TESTS: THRESHOLD CONFIG (Restaurados)
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
            saturation_critical=0.85,
        )

        assert config.flyback_voltage_warning == 0.3
        assert config.flyback_voltage_critical == 0.6
        assert config.saturation_warning == 0.7
        assert config.saturation_critical == 0.85

    @pytest.mark.parametrize(
        "warning,critical",
        [
            (0.6, 0.5),  # warning >= critical
            (0.5, 0.5),  # warning == critical
            (-0.1, 0.5),  # warning negativo
            (0.5, 1.1),  # critical > 1.0
        ],
    )
    def test_invalid_voltage_thresholds_rejected(self, warning, critical):
        """Verifica rechazo de umbrales de voltaje inválidos."""
        with pytest.raises(ValueError):
            ThresholdConfig(
                flyback_voltage_warning=warning, flyback_voltage_critical=critical
            )


# =============================================================================
# TESTS: TELEMETRY DATA (Restaurados)
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

    @pytest.mark.parametrize("invalid_data", [None, [], "string", 123])
    def test_from_dict_invalid_data_returns_none(self, invalid_data):
        """Verifica que tipos de datos inválidos retornan None."""
        result = TelemetryData.from_dict(invalid_data)
        assert result is None

    @pytest.mark.parametrize(
        "input_data,expected_flyback,expected_saturation",
        [
            ({}, 0.0, 0.0),
            ({"flyback_voltage": 0.5}, 0.5, 0.0),
            ({"saturation": 0.5}, 0.0, 0.5),
            ({"metrics": {"flyback_voltage": 0.7}}, 0.7, 0.0),
            ({"flux_condenser.max_flyback_voltage": 0.8}, 0.8, 0.0),
        ],
    )
    def test_from_dict_missing_data_returns_defaults(
        self, input_data, expected_flyback, expected_saturation
    ):
        """Verifica que datos faltantes retornan defaults (0.0/IDLE)."""
        result = TelemetryData.from_dict(input_data)
        assert result is not None
        assert result.flyback_voltage == expected_flyback
        assert result.saturation == expected_saturation

    @pytest.mark.parametrize(
        "voltage,expected",
        [
            (1.5, 1.0),  # Clamp superior
            (-0.5, 0.0),  # Clamp inferior
            (0.5, 0.5),  # Sin cambio
        ],
    )
    def test_voltage_clamping(self, voltage, expected):
        """Verifica clampeo de flyback_voltage a [0, 1]."""
        data = {"flyback_voltage": voltage, "saturation": 0.5}
        result = TelemetryData.from_dict(data)

        assert result is not None
        assert result.flyback_voltage == expected


# =============================================================================
# TESTS: AGENT INITIALIZATION (Restaurados)
# =============================================================================


class TestAgentInitialization(TestFixtures):
    """Tests de inicialización y configuración del agente."""

    def test_initialization_with_explicit_params(
        self, clean_env, mock_topology, mock_persistence
    ):
        """Verifica inicialización con parámetros explícitos."""
        with patch.object(AutonomousAgent, "_setup_signal_handlers"):
            agent = AutonomousAgent(
                core_api_url="http://explicit:8000", check_interval=30, request_timeout=10
            )

            assert agent.core_api_url == "http://explicit:8000"
            assert agent.check_interval == 30
            assert agent.request_timeout == 10
            assert isinstance(agent.thresholds, ThresholdConfig)

    def test_initialization_from_env_vars(self, mock_env, mock_topology, mock_persistence):
        """Verifica inicialización desde variables de entorno."""
        with patch.object(AutonomousAgent, "_setup_signal_handlers"):
            agent = AutonomousAgent()

            assert agent.core_api_url == "http://test-core:5000"
            assert agent.check_interval == 1
            assert agent.request_timeout == 1


# =============================================================================
# TESTS: OBSERVE (Integración Topológica)
# =============================================================================


class TestObserve(TestFixtures):
    """Tests para la fase OBSERVE con lógica de tracking."""

    def test_observe_generates_request_id_and_records_it(self, agent, nominal_response_data):
        """Verifica que cada request genera un ID y lo registra en topología."""
        mock_resp = Mock(ok=True)
        mock_resp.json.return_value = nominal_response_data
        agent._session.get.return_value = mock_resp

        agent.observe()

        # Verificar que se registró un request ID
        agent.topology.record_request.assert_called()
        args, _ = agent.topology.record_request.call_args
        request_id = args[0]
        assert request_id.startswith("obs_")

    def test_observe_success_updates_connectivity(self, agent, nominal_response_data):
        """Verifica que éxito actualiza conectividad en topología."""
        mock_resp = Mock(ok=True)
        mock_resp.json.return_value = nominal_response_data
        agent._session.get.return_value = mock_resp

        agent.observe()

        agent.topology.update_connectivity.assert_called()
        # Debe incluir Agent->Core y los subsistemas reportados
        call_args = agent.topology.update_connectivity.call_args[0][0]
        assert ("Agent", "Core") in call_args
        assert ("Core", "Redis") in call_args

    def test_observe_failure_records_fail_event(self, agent):
        """Verifica que fallo registra evento de error en topología."""
        agent._session.get.side_effect = requests.exceptions.ConnectionError()

        agent.observe()

        # Debe registrar el fallo como un request para detección de patrones
        agent.topology.record_request.assert_called()
        args = agent.topology.record_request.call_args[0]
        assert "FAIL_CONNECTION_ERROR" in args[0]

    def test_observe_degrades_topology_on_repeated_failures(self, agent):
        """Verifica que múltiples fallos cortan la conexión Agent-Core."""
        agent._session.get.side_effect = requests.exceptions.Timeout()

        # Simular fallos hasta el límite
        for _ in range(agent.MAX_CONSECUTIVE_FAILURES):
            agent.observe()

        agent.topology.remove_edge.assert_called_with("Agent", "Core")


# =============================================================================
# TESTS: ORIENT (Lógica Topológica Robusta)
# =============================================================================


class TestOrient(TestFixtures):
    """Tests para la fase ORIENT con prioridades topológicas."""

    def test_orient_priority_1_fragmentation(self, agent, mock_topology):
        """Prioridad 1: Fragmentación Topológica (b0 > 1) -> DISCONNECTED."""
        # Simular sistema fragmentado
        mock_topology.get_topological_health.return_value = TopologicalHealth(
            betti=BettiNumbers(b0=2, b1=0),  # Fragmentado
            disconnected_nodes=frozenset({"Redis"}),
            missing_edges=frozenset(),
            request_loops=tuple(),
            health_score=0.5,
            level=HealthLevel.CRITICAL,
        )

        status = agent.orient(TelemetryData(flyback_voltage=0.1, saturation=0.1))

        assert status == SystemStatus.DISCONNECTED
        assert "Fragmentación" in agent._last_diagnosis.summary

    def test_orient_priority_2_safety_net_voltage(self, agent):
        """Prioridad 2: Safety Net (Voltaje Crítico Instantáneo) -> CRITICO."""
        # Topología sana
        agent.topology.get_topological_health.return_value.betti = BettiNumbers(b0=1, b1=0)

        # Voltaje dispara safety net
        telemetry = TelemetryData(flyback_voltage=0.9, saturation=0.1)  # > 0.8 critical

        status = agent.orient(telemetry)

        assert status == SystemStatus.CRITICO
        assert "Instantáneo" in agent._last_diagnosis.summary

    def test_orient_priority_4_persistence_instability(self, agent, mock_persistence):
        """Prioridad 4: Inestabilidad Persistente (Feature/Critical) -> INESTABLE."""

        # Configurar persistencia para reportar FEATURE en voltaje
        def side_effect(metric, **kwargs):
            if metric == "flyback_voltage":
                return PersistenceAnalysisResult(
                    state=MetricState.FEATURE,
                    intervals=tuple(),
                    feature_count=1,
                    noise_count=0,
                    active_count=0,
                    max_lifespan=10,
                    total_persistence=50,
                    metadata={},
                )
            return PersistenceAnalysisResult(
                state=MetricState.STABLE,
                intervals=tuple(),
                feature_count=0,
                noise_count=0,
                active_count=0,
                max_lifespan=0,
                total_persistence=0,
                metadata={},
            )

        mock_persistence.analyze_persistence.side_effect = side_effect

        status = agent.orient(TelemetryData(flyback_voltage=0.4, saturation=0.2))

        assert status == SystemStatus.INESTABLE
        assert "Inestabilidad" in agent._last_diagnosis.summary

    def test_orient_priority_5_retry_loops(self, agent, mock_topology):
        """Prioridad 5: Loops de Reintentos de Error -> INESTABLE."""
        # Simular loop de errores
        loop_info = RequestLoopInfo(
            request_id="FAIL_TIMEOUT", count=6, first_seen=0, last_seen=10
        )

        mock_topology.get_topological_health.return_value = TopologicalHealth(
            betti=BettiNumbers(b0=1, b1=0),
            disconnected_nodes=frozenset(),
            missing_edges=frozenset(),
            request_loops=(loop_info,),  # Loop presente
            health_score=0.8,
            level=HealthLevel.DEGRADED,
        )

        status = agent.orient(TelemetryData(flyback_voltage=0.2, saturation=0.2))

        assert status == SystemStatus.INESTABLE
        assert "Patrón de Reintentos" in agent._last_diagnosis.summary

    def test_orient_nominal_with_noise_ignored(self, agent, mock_persistence):
        """Verifica que el RUIDO topológico es ignorado (Inmunidad)."""
        # Configurar persistencia para reportar NOISE
        mock_persistence.analyze_persistence.return_value = PersistenceAnalysisResult(
            state=MetricState.NOISE,
            intervals=tuple(),
            feature_count=0,
            noise_count=5,
            active_count=0,
            max_lifespan=2,
            total_persistence=10,
            metadata={},
        )

        status = agent.orient(TelemetryData(flyback_voltage=0.55, saturation=0.2))
        # 0.55 es warning, pero si el análisis dice NOISE, debe ser NOMINAL
        # Nota: La lógica actual en _evaluate_system_state usa el estado de persistencia
        # para decidir. Si es NOISE, cae al final -> NOMINAL.

        assert status == SystemStatus.NOMINAL


# =============================================================================
# TESTS: ACT (Diagnósticos Enriquecidos)
# =============================================================================


class TestAct(TestFixtures):
    """Tests para la fase ACT con mensajes enriquecidos."""

    def test_act_logs_rich_diagnosis(self, agent, caplog):
        """Verifica que el log incluye detalles topológicos."""
        # Preparar un diagnóstico rico
        diagnosis = TopologicalDiagnosis(
            health=TopologicalHealth(
                betti=BettiNumbers(b0=1, b1=1),  # Ciclo
                disconnected_nodes=frozenset(),
                missing_edges=frozenset(),
                request_loops=tuple(),
                health_score=0.8,
                level=HealthLevel.DEGRADED,
                diagnostics={},
            ),
            voltage_persistence=PersistenceAnalysisResult(
                state=MetricState.FEATURE,
                intervals=tuple(),
                feature_count=1,
                noise_count=0,
                active_count=0,
                max_lifespan=10,
                total_persistence=10,
                metadata={},
            ),
            saturation_persistence=PersistenceAnalysisResult(
                state=MetricState.STABLE,
                intervals=tuple(),
                feature_count=0,
                noise_count=0,
                active_count=0,
                max_lifespan=0,
                total_persistence=0,
                metadata={},
            ),
            summary="Test Diagnosis Summary",
            recommended_status=SystemStatus.INESTABLE,
        )
        agent._last_diagnosis = diagnosis

        import logging

        with caplog.at_level(logging.WARNING):
            agent.act(AgentDecision.RECOMENDAR_LIMPIEZA)

        # Verificar contenido del log
        assert "Test Diagnosis Summary" in caplog.text
        assert "[β₀=1]" in caplog.text
        assert "health=0.80" in caplog.text

    def test_debounce_logic_respects_critical_alerts(self, agent):
        """Verifica que ALERTA_CRITICA salta el debounce."""
        decision = AgentDecision.ALERTA_CRITICA

        assert agent.act(decision) is True
        # Inmediatamente otra vez
        assert agent.act(decision) is True


# =============================================================================
# TESTS: INTEGRACIÓN Y MÉTRICAS
# =============================================================================


class TestIntegration(TestFixtures):
    def test_get_metrics_includes_topology(self, agent):
        """Verifica que get_metrics incluye sección de topología."""
        metrics = agent.get_metrics()

        assert "topology" in metrics
        assert "persistence" in metrics
        assert metrics["topology"]["betti_b0"] == 1
        assert "flyback_voltage" in metrics["persistence"]

    def test_initialization_sets_expected_topology(self, agent):
        """Verifica que al iniciar se define la topología esperada."""
        agent.topology.update_connectivity.assert_called()
        # Se llama en __init__ para establecer Agent->Core, Core->Redis, etc.
        # Verificamos que se llamó al menos una vez
        assert agent.topology.update_connectivity.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
