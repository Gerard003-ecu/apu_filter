from unittest.mock import MagicMock, patch

import requests

from agent.apu_agent import AutonomousAgent


class TestAgentResilience:
    @patch("time.sleep", return_value=None)  # Skip sleep to speed up tests
    @patch("requests.get")
    def test_wait_for_startup_cold_start(self, mock_get, mock_sleep):
        """
        Test that _wait_for_startup handles ConnectionRefusedError (Cold Start)
        and retries until success.
        """
        agent = AutonomousAgent()

        # Scenario:
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

        # We need to control the _running loop, or it will loop forever if logic fails.
        # However, _wait_for_startup loops while self._running is True.
        # We can't easily break the loop from outside without threading or side effects.
        # But wait, the method returns when response.ok is True.

        agent._running = True
        agent._wait_for_startup()

        # Assertions
        assert mock_get.call_count == 4
        assert mock_sleep.call_count == 3  # Should sleep after each failure
