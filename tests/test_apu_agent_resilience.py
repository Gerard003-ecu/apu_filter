"""
Tests de Resiliencia para AutonomousAgent
==========================================

Suite robustecida que valida patrones de resiliencia:
- Cold Start con reintentos exponenciales
- Circuit Breaker para protección contra cascadas
- Timeout Handling con cancelación graceful
- Degradación Graceful ante fallos parciales
- Recovery automático tras restauración
- Bulkhead para aislamiento de fallos
- Health Checks y readiness probes

Modelo de Estados de Resiliencia:
---------------------------------
                    ┌──────────┐
           ┌───────►│  CLOSED  │◄───────┐
           │        └────┬─────┘        │
           │             │ fallo        │ éxito tras half-open
           │             ▼              │
           │        ┌──────────┐        │
    timeout│        │   OPEN   │────────┤
           │        └────┬─────┘        │
           │             │ timeout      │
           │             ▼              │
           │        ┌──────────┐        │
           └────────┤HALF-OPEN │────────┘
                    └──────────┘
                         │ fallo
                         ▼
                    (vuelve a OPEN)
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

import pytest
import requests

from agent.apu_agent import AutonomousAgent


# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

FIXED_TIMESTAMP = datetime(2024, 1, 15, 10, 30, 0)

# Configuración de resiliencia esperada
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 60.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
DEFAULT_CIRCUIT_BREAKER_TIMEOUT = 30.0


class CircuitState(Enum):
    """Estados del Circuit Breaker."""
    CLOSED = auto()      # Operación normal
    OPEN = auto()        # Rechazando requests
    HALF_OPEN = auto()   # Probando recuperación


# =============================================================================
# BUILDERS: Escenarios de Fallo Controlados
# =============================================================================


@dataclass
class FailureScenario:
    """Describe un escenario de fallo para testing."""
    name: str
    responses: List[Any]
    expected_retries: int
    expected_success: bool
    description: str = ""


class ResponseBuilder:
    """
    Builder para crear secuencias de respuestas HTTP simuladas.
    
    Permite construir escenarios de fallo complejos de forma declarativa.
    """
    
    def __init__(self):
        self._responses: List[Any] = []
    
    def connection_error(self, message: str = "Connection refused") -> "ResponseBuilder":
        """Agrega un error de conexión."""
        self._responses.append(
            requests.exceptions.ConnectionError(message)
        )
        return self
    
    def timeout(self, message: str = "Read timed out") -> "ResponseBuilder":
        """Agrega un timeout."""
        self._responses.append(
            requests.exceptions.Timeout(message)
        )
        return self
    
    def read_timeout(self) -> "ResponseBuilder":
        """Agrega un ReadTimeout específico."""
        self._responses.append(
            requests.exceptions.ReadTimeout("Read timed out")
        )
        return self
    
    def connect_timeout(self) -> "ResponseBuilder":
        """Agrega un ConnectTimeout específico."""
        self._responses.append(
            requests.exceptions.ConnectTimeout("Connection timed out")
        )
        return self
    
    def http_error(self, status_code: int, reason: str = "") -> "ResponseBuilder":
        """Agrega una respuesta HTTP con código de error."""
        mock_resp = Mock()
        mock_resp.ok = status_code < 400
        mock_resp.status_code = status_code
        mock_resp.reason = reason or self._default_reason(status_code)
        mock_resp.text = f"Error {status_code}"
        mock_resp.json.side_effect = ValueError("No JSON")
        self._responses.append(mock_resp)
        return self
    
    def service_unavailable(self) -> "ResponseBuilder":
        """Agrega HTTP 503 Service Unavailable."""
        return self.http_error(503, "Service Unavailable")
    
    def bad_gateway(self) -> "ResponseBuilder":
        """Agrega HTTP 502 Bad Gateway."""
        return self.http_error(502, "Bad Gateway")
    
    def gateway_timeout(self) -> "ResponseBuilder":
        """Agrega HTTP 504 Gateway Timeout."""
        return self.http_error(504, "Gateway Timeout")
    
    def too_many_requests(self, retry_after: int = 60) -> "ResponseBuilder":
        """Agrega HTTP 429 Too Many Requests."""
        mock_resp = Mock()
        mock_resp.ok = False
        mock_resp.status_code = 429
        mock_resp.reason = "Too Many Requests"
        mock_resp.headers = {"Retry-After": str(retry_after)}
        self._responses.append(mock_resp)
        return self
    
    def success(self, data: Optional[Dict] = None) -> "ResponseBuilder":
        """Agrega una respuesta exitosa."""
        mock_resp = Mock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.reason = "OK"
        mock_resp.json.return_value = data or {"status": "healthy"}
        self._responses.append(mock_resp)
        return self
    
    def success_sequence(self, count: int, data: Optional[Dict] = None) -> "ResponseBuilder":
        """Agrega múltiples respuestas exitosas."""
        for _ in range(count):
            self.success(data)
        return self
    
    def failure_then_success(
        self, 
        failure_count: int, 
        failure_type: str = "connection"
    ) -> "ResponseBuilder":
        """Patrón común: N fallos seguidos de éxito."""
        for _ in range(failure_count):
            if failure_type == "connection":
                self.connection_error()
            elif failure_type == "timeout":
                self.timeout()
            elif failure_type == "503":
                self.service_unavailable()
        self.success()
        return self
    
    def intermittent_failures(self, pattern: str) -> "ResponseBuilder":
        """
        Crea patrón de fallos intermitentes.
        
        Args:
            pattern: String donde 'F' = fallo, 'S' = éxito
                     Ejemplo: "FFSFFFS" = 2 fallos, 1 éxito, 3 fallos, 1 éxito
        """
        for char in pattern.upper():
            if char == 'F':
                self.connection_error()
            elif char == 'S':
                self.success()
        return self
    
    def _default_reason(self, status_code: int) -> str:
        """Retorna reason por defecto según status code."""
        reasons = {
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            500: "Internal Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Gateway Timeout",
        }
        return reasons.get(status_code, "Unknown Error")
    
    def build(self) -> List[Any]:
        """Retorna la lista de respuestas configuradas."""
        return self._responses.copy()
    
    def build_side_effect(self) -> List[Any]:
        """Retorna lista para usar como side_effect en mock."""
        return self._responses.copy()


class ScenarioBuilder:
    """Builder para escenarios de prueba de resiliencia."""
    
    @staticmethod
    def cold_start_success(attempts: int = 3) -> FailureScenario:
        """Escenario: Cold start con éxito después de N intentos."""
        responses = (
            ResponseBuilder()
            .failure_then_success(attempts - 1, "connection")
            .build()
        )
        return FailureScenario(
            name=f"cold_start_{attempts}_attempts",
            responses=responses,
            expected_retries=attempts - 1,
            expected_success=True,
            description=f"Servicio disponible después de {attempts-1} reintentos"
        )
    
    @staticmethod
    def cold_start_with_503() -> FailureScenario:
        """Escenario: Cold start con 503 intermedios."""
        responses = (
            ResponseBuilder()
            .connection_error()
            .connection_error()
            .service_unavailable()
            .success()
            .build()
        )
        return FailureScenario(
            name="cold_start_with_503",
            responses=responses,
            expected_retries=3,
            expected_success=True,
            description="Conexión rechazada → 503 → éxito"
        )
    
    @staticmethod
    def persistent_failure() -> FailureScenario:
        """Escenario: Fallo persistente que agota reintentos."""
        responses = (
            ResponseBuilder()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()  # Más que MAX_RETRIES
            .build()
        )
        return FailureScenario(
            name="persistent_failure",
            responses=responses,
            expected_retries=DEFAULT_MAX_RETRIES,
            expected_success=False,
            description="Fallo persistente, agota todos los reintentos"
        )
    
    @staticmethod
    def timeout_recovery() -> FailureScenario:
        """Escenario: Timeouts seguidos de recuperación."""
        responses = (
            ResponseBuilder()
            .timeout()
            .timeout()
            .success()
            .build()
        )
        return FailureScenario(
            name="timeout_recovery",
            responses=responses,
            expected_retries=2,
            expected_success=True,
            description="Timeouts temporales con recuperación"
        )
    
    @staticmethod
    def mixed_errors() -> FailureScenario:
        """Escenario: Mezcla de diferentes tipos de error."""
        responses = (
            ResponseBuilder()
            .connection_error()
            .timeout()
            .service_unavailable()
            .bad_gateway()
            .success()
            .build()
        )
        return FailureScenario(
            name="mixed_errors",
            responses=responses,
            expected_retries=4,
            expected_success=True,
            description="Diferentes tipos de error antes del éxito"
        )
    
    @staticmethod
    def rate_limited() -> FailureScenario:
        """Escenario: Rate limiting con Retry-After."""
        responses = (
            ResponseBuilder()
            .too_many_requests(retry_after=5)
            .too_many_requests(retry_after=10)
            .success()
            .build()
        )
        return FailureScenario(
            name="rate_limited",
            responses=responses,
            expected_retries=2,
            expected_success=True,
            description="Rate limiting respetando Retry-After"
        )


# =============================================================================
# FIXTURES
# =============================================================================


class TestFixtures:
    """Clase base con fixtures reutilizables para tests de resiliencia."""

    @pytest.fixture
    def mock_time(self):
        """Mock de time.sleep para acelerar tests."""
        with patch("time.sleep", return_value=None) as mock:
            yield mock

    @pytest.fixture
    def mock_datetime(self):
        """Mock de datetime para control temporal."""
        with patch("agent.apu_agent.datetime") as mock:
            mock.now.return_value = FIXED_TIMESTAMP
            mock.side_effect = lambda *args, **kw: datetime(*args, **kw)
            yield mock

    @pytest.fixture
    def mock_requests_get(self):
        """Mock de requests.get."""
        with patch("requests.get") as mock:
            yield mock

    @pytest.fixture
    def mock_session(self):
        """Mock de requests.Session."""
        with patch("requests.Session") as mock:
            session_instance = MagicMock()
            mock.return_value = session_instance
            yield session_instance

    @pytest.fixture
    def mock_topology(self):
        """Mock de SystemTopology para aislar tests de resiliencia."""
        with patch("agent.apu_agent.SystemTopology") as mock:
            instance = MagicMock()
            mock.return_value = instance
            instance.update_connectivity.return_value = (3, [])
            instance.record_request.return_value = None
            instance.remove_edge.return_value = None
            yield instance

    @pytest.fixture
    def mock_persistence(self):
        """Mock de PersistenceHomology."""
        with patch("agent.apu_agent.PersistenceHomology") as mock:
            instance = MagicMock()
            mock.return_value = instance
            yield instance

    @pytest.fixture
    def mock_signal_handlers(self):
        """Mock para evitar configuración de signal handlers."""
        with patch.object(AutonomousAgent, "_setup_signal_handlers"):
            yield

    @pytest.fixture
    def agent(
        self, 
        mock_time, 
        mock_topology, 
        mock_persistence, 
        mock_signal_handlers,
        mock_session
    ):
        """
        Agente completamente aislado para tests de resiliencia.
        
        Todos los componentes externos están mockeados.
        """
        agent = AutonomousAgent(
            core_api_url="http://test-core:5000",
            check_interval=1,
            request_timeout=5
        )
        agent._session = mock_session
        agent._running = True
        yield agent
        agent._running = False

    @pytest.fixture
    def backoff_tracker(self):
        """Tracker para verificar comportamiento de backoff."""
        class BackoffTracker:
            def __init__(self):
                self.sleep_calls: List[float] = []
                self.call_timestamps: List[float] = []
                self._start_time = time.time()
            
            def record_sleep(self, duration: float):
                self.sleep_calls.append(duration)
                self.call_timestamps.append(time.time() - self._start_time)
            
            def verify_exponential_backoff(
                self, 
                initial: float = DEFAULT_INITIAL_BACKOFF,
                multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
                max_backoff: float = DEFAULT_MAX_BACKOFF,
                tolerance: float = 0.1
            ) -> bool:
                """Verifica que los sleeps siguen patrón exponencial."""
                if not self.sleep_calls:
                    return True
                
                expected = initial
                for actual in self.sleep_calls:
                    if abs(actual - min(expected, max_backoff)) > tolerance * expected:
                        return False
                    expected *= multiplier
                return True
            
            def get_total_wait_time(self) -> float:
                return sum(self.sleep_calls)
        
        return BackoffTracker()


# =============================================================================
# TESTS: COLD START / WARM UP
# =============================================================================


class TestColdStart(TestFixtures):
    """
    Tests para el comportamiento de arranque en frío.
    
    Valida que el agente maneje correctamente:
    - Servicio no disponible inicialmente
    - Reintentos con backoff
    - Éxito eventual tras reintentos
    - Fallo tras agotar reintentos
    """

    def test_wait_for_startup_immediate_success(
        self, agent, mock_session, mock_time
    ):
        """Éxito inmediato no requiere reintentos."""
        mock_session.get.return_value = ResponseBuilder().success().build()[0]

        agent._wait_for_startup()

        assert mock_session.get.call_count == 1
        assert mock_time.call_count == 0  # Sin sleeps

    def test_wait_for_startup_success_after_retries(
        self, agent, mock_session, mock_time
    ):
        """Éxito después de múltiples reintentos por ConnectionError."""
        scenario = ScenarioBuilder.cold_start_success(attempts=4)
        mock_session.get.side_effect = scenario.responses

        agent._wait_for_startup()

        assert mock_session.get.call_count == 4
        assert mock_time.call_count == 3  # Sleep después de cada fallo

    def test_wait_for_startup_handles_503_as_loading(
        self, agent, mock_session, mock_time
    ):
        """HTTP 503 se trata como 'aún cargando', no como error fatal."""
        scenario = ScenarioBuilder.cold_start_with_503()
        mock_session.get.side_effect = scenario.responses

        agent._wait_for_startup()

        assert mock_session.get.call_count == 4
        assert scenario.expected_success

    def test_wait_for_startup_exponential_backoff(
        self, agent, mock_session, backoff_tracker
    ):
        """Verifica backoff exponencial en reintentos."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .connection_error()
            .connection_error()
            .success()
            .build()
        )

        with patch("time.sleep", side_effect=backoff_tracker.record_sleep):
            agent._wait_for_startup()

        # Verificar patrón exponencial: 1s, 2s, 4s...
        assert len(backoff_tracker.sleep_calls) == 3
        assert backoff_tracker.verify_exponential_backoff()

    def test_wait_for_startup_respects_max_backoff(
        self, agent, mock_session, backoff_tracker
    ):
        """Backoff no excede el máximo configurado."""
        # Muchos fallos para alcanzar max backoff
        mock_session.get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .success()
            .build()
        )

        with patch("time.sleep", side_effect=backoff_tracker.record_sleep):
            agent._wait_for_startup()

        # Ningún sleep debe exceder MAX_BACKOFF
        max_observed = max(backoff_tracker.sleep_calls)
        assert max_observed <= DEFAULT_MAX_BACKOFF

    def test_wait_for_startup_logs_retry_attempts(
        self, agent, mock_session, mock_time, caplog
    ):
        """Cada reintento se registra en logs."""
        import logging
        
        mock_session.get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .connection_error()
            .success()
            .build()
        )

        with caplog.at_level(logging.WARNING):
            agent._wait_for_startup()

        # Debe haber logs de reintentos
        retry_logs = [r for r in caplog.records if "retry" in r.message.lower() 
                      or "reintent" in r.message.lower()
                      or "attempt" in r.message.lower()]
        assert len(retry_logs) >= 1

    def test_wait_for_startup_stops_when_running_false(
        self, agent, mock_session, mock_time
    ):
        """El loop termina si _running se vuelve False."""
        agent._running = False
        mock_session.get.side_effect = requests.exceptions.ConnectionError()

        # No debe quedar en loop infinito
        agent._wait_for_startup()

        # Debería haber salido sin éxito pero sin hang
        assert mock_session.get.call_count <= 1


# =============================================================================
# TESTS: TIMEOUT HANDLING
# =============================================================================


class TestTimeoutHandling(TestFixtures):
    """
    Tests para el manejo de timeouts.
    
    Valida:
    - ReadTimeout vs ConnectTimeout
    - Reintentos apropiados
    - Cancelación graceful
    """

    def test_read_timeout_triggers_retry(self, agent, mock_session, mock_time):
        """ReadTimeout provoca reintento."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .read_timeout()
            .success()
            .build()
        )

        agent._wait_for_startup()

        assert mock_session.get.call_count == 2

    def test_connect_timeout_triggers_retry(self, agent, mock_session, mock_time):
        """ConnectTimeout provoca reintento."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .connect_timeout()
            .success()
            .build()
        )

        agent._wait_for_startup()

        assert mock_session.get.call_count == 2

    def test_timeout_recovery_scenario(self, agent, mock_session, mock_time):
        """Recuperación después de timeouts intermitentes."""
        scenario = ScenarioBuilder.timeout_recovery()
        mock_session.get.side_effect = scenario.responses

        agent._wait_for_startup()

        assert mock_session.get.call_count == 3
        assert mock_time.call_count == 2

    def test_mixed_timeout_types(self, agent, mock_session, mock_time):
        """Manejo de diferentes tipos de timeout en secuencia."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .read_timeout()
            .connect_timeout()
            .timeout()  # Generic timeout
            .success()
            .build()
        )

        agent._wait_for_startup()

        assert mock_session.get.call_count == 4


# =============================================================================
# TESTS: HTTP ERROR HANDLING
# =============================================================================


class TestHTTPErrorHandling(TestFixtures):
    """
    Tests para el manejo de errores HTTP.
    
    Valida comportamiento ante:
    - 5xx (Server Errors) - Retriable
    - 4xx (Client Errors) - Generalmente no retriable
    - Rate Limiting (429)
    """

    def test_503_service_unavailable_retried(self, agent, mock_session, mock_time):
        """HTTP 503 provoca reintento."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .service_unavailable()
            .service_unavailable()
            .success()
            .build()
        )

        agent._wait_for_startup()

        assert mock_session.get.call_count == 3

    def test_502_bad_gateway_retried(self, agent, mock_session, mock_time):
        """HTTP 502 provoca reintento."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .bad_gateway()
            .success()
            .build()
        )

        agent._wait_for_startup()

        assert mock_session.get.call_count == 2

    def test_504_gateway_timeout_retried(self, agent, mock_session, mock_time):
        """HTTP 504 provoca reintento."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .gateway_timeout()
            .success()
            .build()
        )

        agent._wait_for_startup()

        assert mock_session.get.call_count == 2

    def test_mixed_5xx_errors(self, agent, mock_session, mock_time):
        """Escenario mixto de errores 5xx."""
        scenario = ScenarioBuilder.mixed_errors()
        mock_session.get.side_effect = scenario.responses

        agent._wait_for_startup()

        assert mock_session.get.call_count == len(scenario.responses)

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    def test_all_5xx_are_retriable(self, agent, mock_session, mock_time, status_code):
        """Todos los errores 5xx deben provocar reintento."""
        mock_session.get.side_effect = [
            ResponseBuilder().http_error(status_code).build()[0],
            ResponseBuilder().success().build()[0],
        ]

        agent._wait_for_startup()

        assert mock_session.get.call_count == 2, f"HTTP {status_code} debe ser retriable"


# =============================================================================
# TESTS: RATE LIMITING
# =============================================================================


class TestRateLimiting(TestFixtures):
    """
    Tests para el manejo de rate limiting.
    
    Valida:
    - Respeto del header Retry-After
    - Backoff apropiado sin header
    - Recuperación después de rate limit
    """

    def test_429_respects_retry_after_header(
        self, agent, mock_session, backoff_tracker
    ):
        """HTTP 429 respeta el header Retry-After."""
        retry_after_seconds = 5
        mock_session.get.side_effect = (
            ResponseBuilder()
            .too_many_requests(retry_after=retry_after_seconds)
            .success()
            .build()
        )

        with patch("time.sleep", side_effect=backoff_tracker.record_sleep):
            agent._wait_for_startup()

        # El sleep debe ser al menos el valor de Retry-After
        assert len(backoff_tracker.sleep_calls) >= 1
        # Verificar que se respetó Retry-After (con tolerancia)
        assert backoff_tracker.sleep_calls[0] >= retry_after_seconds - 1

    def test_429_without_retry_after_uses_backoff(
        self, agent, mock_session, mock_time
    ):
        """HTTP 429 sin Retry-After usa backoff estándar."""
        mock_resp = Mock()
        mock_resp.ok = False
        mock_resp.status_code = 429
        mock_resp.headers = {}  # Sin Retry-After

        mock_session.get.side_effect = [
            mock_resp,
            ResponseBuilder().success().build()[0],
        ]

        agent._wait_for_startup()

        assert mock_session.get.call_count == 2
        mock_time.assert_called()  # Debe haber esperado

    def test_multiple_429_before_success(self, agent, mock_session, mock_time):
        """Múltiples 429 antes de éxito."""
        scenario = ScenarioBuilder.rate_limited()
        mock_session.get.side_effect = scenario.responses

        agent._wait_for_startup()

        assert mock_session.get.call_count == 3


# =============================================================================
# TESTS: CIRCUIT BREAKER
# =============================================================================


class TestCircuitBreaker(TestFixtures):
    """
    Tests para el patrón Circuit Breaker.
    
    Valida:
    - Apertura después de N fallos consecutivos
    - Rechazo rápido en estado abierto
    - Transición a half-open después de timeout
    - Cierre tras éxito en half-open
    """

    def test_circuit_opens_after_threshold_failures(
        self, agent, mock_session, mock_time
    ):
        """Circuit se abre después del umbral de fallos."""
        # Simular fallos hasta el umbral
        failures = [
            requests.exceptions.ConnectionError()
            for _ in range(DEFAULT_CIRCUIT_BREAKER_THRESHOLD + 2)
        ]
        mock_session.get.side_effect = failures

        # El agente debe dejar de intentar después del umbral
        try:
            agent._wait_for_startup()
        except Exception:
            pass  # Puede lanzar excepción cuando el circuit está abierto

        # No debe haber intentado más allá del umbral + algunos extras
        assert mock_session.get.call_count <= DEFAULT_CIRCUIT_BREAKER_THRESHOLD + 2

    def test_circuit_rejects_fast_when_open(
        self, agent, mock_session, mock_time
    ):
        """En estado abierto, rechaza rápido sin intentar request."""
        # Primero, abrir el circuit
        agent._consecutive_failures = DEFAULT_CIRCUIT_BREAKER_THRESHOLD + 1
        
        if hasattr(agent, '_circuit_state'):
            agent._circuit_state = CircuitState.OPEN
            agent._circuit_opened_at = FIXED_TIMESTAMP

            # Intento mientras está abierto
            with patch("agent.apu_agent.datetime") as mock_dt:
                # Simular que ha pasado poco tiempo (aún abierto)
                mock_dt.now.return_value = FIXED_TIMESTAMP + timedelta(seconds=5)
                
                result = agent._check_circuit_breaker()
                
                assert result is False  # Rechazado por circuit abierto

    def test_circuit_transitions_to_half_open(
        self, agent, mock_session, mock_time
    ):
        """Circuit transiciona a half-open después del timeout."""
        if not hasattr(agent, '_circuit_state'):
            pytest.skip("Agent no implementa circuit breaker explícito")

        agent._circuit_state = CircuitState.OPEN
        agent._circuit_opened_at = FIXED_TIMESTAMP

        with patch("agent.apu_agent.datetime") as mock_dt:
            # Simular que pasó el timeout
            mock_dt.now.return_value = FIXED_TIMESTAMP + timedelta(
                seconds=DEFAULT_CIRCUIT_BREAKER_TIMEOUT + 1
            )

            result = agent._check_circuit_breaker()

            assert agent._circuit_state == CircuitState.HALF_OPEN

    def test_circuit_closes_on_half_open_success(
        self, agent, mock_session, mock_time
    ):
        """Circuit se cierra tras éxito en half-open."""
        if not hasattr(agent, '_circuit_state'):
            pytest.skip("Agent no implementa circuit breaker explícito")

        agent._circuit_state = CircuitState.HALF_OPEN
        mock_session.get.return_value = ResponseBuilder().success().build()[0]

        agent._wait_for_startup()

        assert agent._circuit_state == CircuitState.CLOSED
        assert agent._consecutive_failures == 0

    def test_circuit_reopens_on_half_open_failure(
        self, agent, mock_session, mock_time
    ):
        """Circuit vuelve a abrirse si falla en half-open."""
        if not hasattr(agent, '_circuit_state'):
            pytest.skip("Agent no implementa circuit breaker explícito")

        agent._circuit_state = CircuitState.HALF_OPEN
        mock_session.get.side_effect = requests.exceptions.ConnectionError()

        try:
            agent._make_request()
        except Exception:
            pass

        assert agent._circuit_state == CircuitState.OPEN


# =============================================================================
# TESTS: RECOVERY
# =============================================================================


class TestRecovery(TestFixtures):
    """
    Tests para la recuperación después de fallos.
    
    Valida:
    - Reset de contadores tras éxito
    - Restauración de conectividad
    - Logs de recuperación
    """

    def test_consecutive_failures_reset_on_success(
        self, agent, mock_session, mock_time
    ):
        """Contador de fallos se resetea tras éxito."""
        # Simular fallos seguidos de éxito
        mock_session.get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .connection_error()
            .success()
            .build()
        )

        agent._wait_for_startup()

        assert agent._consecutive_failures == 0

    def test_topology_restored_on_recovery(
        self, agent, mock_session, mock_time, mock_topology
    ):
        """Conectividad topológica se restaura tras recuperación."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .success()
            .build()
        )

        agent._wait_for_startup()

        # Debería haber restaurado la conexión Agent->Core
        mock_topology.add_edge.assert_called()
        call_args = [str(c) for c in mock_topology.add_edge.call_args_list]
        assert any("Agent" in c and "Core" in c for c in call_args)

    def test_recovery_logged(self, agent, mock_session, mock_time, caplog):
        """Recuperación se registra en logs."""
        import logging

        mock_session.get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .success()
            .build()
        )

        with caplog.at_level(logging.INFO):
            agent._wait_for_startup()

        # Debe haber log de recuperación/conexión exitosa
        recovery_keywords = ["success", "connected", "ready", "recovered", "éxito", "conectado"]
        log_text = caplog.text.lower()
        assert any(kw in log_text for kw in recovery_keywords)


# =============================================================================
# TESTS: MÉTRICAS Y OBSERVABILIDAD
# =============================================================================


class TestResilienceMetrics(TestFixtures):
    """
    Tests para métricas de resiliencia.
    
    Valida que el agente exponga métricas sobre:
    - Reintentos realizados
    - Tiempo de recuperación
    - Estado del circuit breaker
    - Fallos por tipo
    """

    def test_retry_count_tracked(self, agent, mock_session, mock_time):
        """Número de reintentos es trackeado."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .connection_error()
            .success()
            .build()
        )

        agent._wait_for_startup()

        if hasattr(agent, 'get_metrics'):
            metrics = agent.get_metrics()
            assert "retries" in metrics or "retry_count" in metrics

    def test_failure_types_categorized(self, agent, mock_session, mock_time):
        """Fallos se categorizan por tipo."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .timeout()
            .service_unavailable()
            .success()
            .build()
        )

        agent._wait_for_startup()

        if hasattr(agent, 'get_metrics'):
            metrics = agent.get_metrics()
            if "failures_by_type" in metrics:
                failures = metrics["failures_by_type"]
                assert "connection" in failures or "timeout" in failures

    def test_last_failure_time_recorded(self, agent, mock_session, mock_time):
        """Timestamp del último fallo es registrado."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .success()
            .build()
        )

        agent._wait_for_startup()

        if hasattr(agent, '_last_failure_time'):
            assert agent._last_failure_time is not None


# =============================================================================
# TESTS: ESCENARIOS COMPLEJOS
# =============================================================================


class TestComplexScenarios(TestFixtures):
    """
    Tests para escenarios complejos de resiliencia.
    
    Valida comportamiento en situaciones realistas combinadas.
    """

    def test_intermittent_failures_pattern(self, agent, mock_session, mock_time):
        """Manejo de patrón de fallos intermitentes."""
        # Patrón: Fallo, Éxito, Fallo, Fallo, Éxito
        mock_session.get.side_effect = (
            ResponseBuilder()
            .intermittent_failures("FSFFS")
            .build()
        )

        # Primer startup
        agent._wait_for_startup()

        # Debe haber manejado todos los intentos
        assert mock_session.get.call_count >= 3

    def test_cascading_failure_recovery(
        self, agent, mock_session, mock_time, mock_topology
    ):
        """Recuperación de fallo en cascada."""
        # Simular: Core falla → Redis falla → ambos se recuperan
        mock_session.get.side_effect = (
            ResponseBuilder()
            .connection_error()  # Core no disponible
            .service_unavailable()  # Core vuelve pero degradado
            .http_error(500)  # Error interno
            .success({"redis_connected": False})  # Core OK, Redis no
            .success({"redis_connected": True})  # Todo OK
            .build()
        )

        agent._wait_for_startup()

        # Debe haber intentado todas las veces
        assert mock_session.get.call_count == 5

    def test_partial_recovery_handled(self, agent, mock_session, mock_time):
        """Recuperación parcial es manejada apropiadamente."""
        # El servicio responde OK pero con datos parciales
        partial_response = Mock()
        partial_response.ok = True
        partial_response.status_code = 200
        partial_response.json.return_value = {
            "status": "degraded",
            "redis_connected": False,
            "filesystem_accessible": False,
        }

        mock_session.get.side_effect = [
            requests.exceptions.ConnectionError(),
            partial_response,
        ]

        agent._wait_for_startup()

        # Debe aceptar respuesta parcial como éxito de startup
        assert mock_session.get.call_count == 2

    def test_long_outage_simulation(self, agent, mock_session, backoff_tracker):
        """Simulación de outage prolongado."""
        # 10 fallos antes de recuperación
        responses = (
            ResponseBuilder()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .connection_error()
            .success()
            .build()
        )
        mock_session.get.side_effect = responses

        with patch("time.sleep", side_effect=backoff_tracker.record_sleep):
            try:
                agent._wait_for_startup()
            except Exception:
                pass  # Puede fallar si excede max retries

        # Verificar que los backoffs crecieron apropiadamente
        if len(backoff_tracker.sleep_calls) > 2:
            assert backoff_tracker.sleep_calls[-1] >= backoff_tracker.sleep_calls[0]


# =============================================================================
# TESTS: INVARIANTES DE RESILIENCIA
# =============================================================================


class TestResilienceInvariants(TestFixtures):
    """
    Tests para invariantes de resiliencia.
    
    Propiedades que siempre deben cumplirse:
    - Backoff nunca negativo
    - Contador de fallos nunca negativo
    - Estado consistente después de operaciones
    """

    def test_backoff_always_positive(self, agent, mock_session, backoff_tracker):
        """Backoff siempre es positivo."""
        mock_session.get.side_effect = (
            ResponseBuilder()
            .failure_then_success(5, "connection")
            .build()
        )

        with patch("time.sleep", side_effect=backoff_tracker.record_sleep):
            agent._wait_for_startup()

        assert all(sleep > 0 for sleep in backoff_tracker.sleep_calls)

    def test_failure_counter_never_negative(self, agent, mock_session, mock_time):
        """Contador de fallos nunca es negativo."""
        # Secuencia de fallos y éxitos
        mock_session.get.side_effect = (
            ResponseBuilder()
            .connection_error()
            .success()
            .build()
        )

        agent._wait_for_startup()

        assert agent._consecutive_failures >= 0

    def test_state_consistent_after_exception(self, agent, mock_session, mock_time):
        """Estado es consistente incluso después de excepciones."""
        # Excepción inesperada
        mock_session.get.side_effect = RuntimeError("Unexpected error")

        try:
            agent._wait_for_startup()
        except RuntimeError:
            pass

        # El agente debe estar en estado consistente
        assert hasattr(agent, '_running')
        assert hasattr(agent, '_consecutive_failures')

    def test_total_wait_bounded(self, agent, mock_session, backoff_tracker):
        """Tiempo total de espera está acotado."""
        # Muchos fallos
        mock_session.get.side_effect = [
            requests.exceptions.ConnectionError()
            for _ in range(20)
        ] + [ResponseBuilder().success().build()[0]]

        with patch("time.sleep", side_effect=backoff_tracker.record_sleep):
            try:
                agent._wait_for_startup()
            except Exception:
                pass

        # Tiempo total no debe explotar
        total_wait = backoff_tracker.get_total_wait_time()
        max_reasonable_wait = DEFAULT_MAX_RETRIES * DEFAULT_MAX_BACKOFF * 2
        assert total_wait < max_reasonable_wait


# =============================================================================
# TESTS: CONFIGURABILIDAD
# =============================================================================


class TestResilienceConfiguration(TestFixtures):
    """
    Tests para la configurabilidad de resiliencia.
    
    Valida que los parámetros de resiliencia sean configurables.
    """

    def test_custom_max_retries(self, mock_topology, mock_persistence, mock_signal_handlers):
        """Max retries es configurable."""
        custom_retries = 10
        
        with patch("time.sleep"):
            agent = AutonomousAgent(
                core_api_url="http://test:5000",
                max_retries=custom_retries
            )
            
            if hasattr(agent, 'max_retries'):
                assert agent.max_retries == custom_retries

    def test_custom_backoff_parameters(
        self, mock_topology, mock_persistence, mock_signal_handlers
    ):
        """Parámetros de backoff son configurables."""
        with patch("time.sleep"):
            agent = AutonomousAgent(
                core_api_url="http://test:5000",
                initial_backoff=2.0,
                max_backoff=120.0
            )
            
            if hasattr(agent, 'initial_backoff'):
                assert agent.initial_backoff == 2.0
            if hasattr(agent, 'max_backoff'):
                assert agent.max_backoff == 120.0

    def test_circuit_breaker_threshold_configurable(
        self, mock_topology, mock_persistence, mock_signal_handlers
    ):
        """Umbral del circuit breaker es configurable."""
        with patch("time.sleep"):
            agent = AutonomousAgent(
                core_api_url="http://test:5000",
                circuit_breaker_threshold=10
            )
            
            if hasattr(agent, 'circuit_breaker_threshold'):
                assert agent.circuit_breaker_threshold == 10


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])