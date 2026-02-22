
import pytest
from app.topology_viz import extract_anomaly_data, AnomalyData

class TestSprintIntegration:
    def test_extract_anomaly_data_with_list(self):
        """Test that extract_anomaly_data handles cycles as a list of strings."""
        details = {
            "cycles": ["A -> B -> A", "C -> D -> C"]
        }
        # The function expects {"details": ...} structure or just the dict?
        # Looking at code: extract_anomaly_data(analysis_result: Dict[str, Any])
        #   details = analysis_result.get("details")

        analysis_result = {"details": details}
        anomaly_data = extract_anomaly_data(analysis_result)

        assert "A" in anomaly_data.nodes_in_cycles
        assert "B" in anomaly_data.nodes_in_cycles
        assert "C" in anomaly_data.nodes_in_cycles
        assert "D" in anomaly_data.nodes_in_cycles

    def test_extract_anomaly_data_with_dict(self):
        """Test that extract_anomaly_data handles cycles as a dict (legacy)."""
        details = {
            "cycles": {"list": ["X -> Y -> X"]}
        }
        analysis_result = {"details": details}
        anomaly_data = extract_anomaly_data(analysis_result)

        assert "X" in anomaly_data.nodes_in_cycles
        assert "Y" in anomaly_data.nodes_in_cycles

    def test_extract_anomaly_data_with_unknown_format(self):
        """Test that extract_anomaly_data handles unknown formats gracefully."""
        details = {
            "cycles": 123 # Invalid
        }
        analysis_result = {"details": details}
        anomaly_data = extract_anomaly_data(analysis_result)
        assert len(anomaly_data.nodes_in_cycles) == 0
