from unittest.mock import MagicMock, patch

import requests

from agent.apu_agent import AutonomousAgent


class TestAgentResilience:
    """Suite de pruebas para la resiliencia del agente."""

    @patch("time.sleep", return_value=None)  # Saltar sleep para acelerar tests
    @patch("requests.get")
    def test_wait_for_startup_cold_start(self, mock_get, mock_sleep):
        """
        Prueba que _wait_for_startup maneja ConnectionRefusedError (Cold Start)
        y reintenta hasta tener éxito.
        """
        agent = AutonomousAgent()

        # Escenario:
        # 1. ConnectionError (Refused)
        # 2. ConnectionError (Refused)
        # 3. HTTP 503 (Service Unavailable - still loading)
        # 4. HTTP 200 OK (Success)

        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            requests.exceptions.ConnectionError("Connection refused"),
            MagicMock(ok=False, status_code=503),
            MagicMock(ok=True, status_code=200),
        ]

        # Necesitamos controlar el bucle _running, o iterará por siempre si la lógica falla.
        # Sin embargo, _wait_for_startup termina cuando response.ok es True.

        agent._running = True
        agent._wait_for_startup()

        # Aserciones
        assert mock_get.call_count == 4
        assert mock_sleep.call_count == 3  # Debe dormir después de cada fallo
