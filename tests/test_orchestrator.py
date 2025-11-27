import unittest
from unittest.mock import MagicMock, patch

import agent.orchestrator as orchestrator


class TestOrchestrator(unittest.TestCase):
    @patch("agent.orchestrator.requests.get")
    def test_check_core_health_success(self, mock_get):
        # Mock a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response

        with self.assertLogs(level="INFO") as log:
            orchestrator.check_core_health()
            self.assertTrue(
                any("Core service is healthy" in output for output in log.output)
            )

    @patch("agent.orchestrator.requests.get")
    def test_check_core_health_warning(self, mock_get):
        # Mock a non-200 response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with self.assertLogs(level="WARNING") as log:
            orchestrator.check_core_health()
            self.assertTrue(
                any(
                    "Core service returned non-200 status" in output for output in log.output
                )
            )

    @patch("agent.orchestrator.requests.get")
    def test_check_core_health_failure(self, mock_get):
        # Mock an exception
        mock_get.side_effect = orchestrator.requests.RequestException("Connection refused")

        with self.assertLogs(level="ERROR") as log:
            orchestrator.check_core_health()
            self.assertTrue(
                any("Failed to connect to core service" in output for output in log.output)
            )


if __name__ == "__main__":
    unittest.main()
