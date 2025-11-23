"""
Suite de pruebas robusta para el pipeline director (PipelineDirector).

Incluye:
- Pruebas unitarias para el orquestador
- Pruebas de integración con los Steps
- Verificación de la inyección de telemetría
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.pipeline_director import (
    PipelineDirector,
    ProcessingStep,
    LoadDataStep,
    MergeDataStep,
    CalculateCostsStep,
    FinalMergeStep,
    BuildOutputStep,
    ProcessingThresholds
)
from app.telemetry import TelemetryContext
from tests.test_data import TEST_CONFIG

class MockStep(ProcessingStep):
    def __init__(self, config, thresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        telemetry.start_step("mock_step")
        context["mock_executed"] = True
        telemetry.end_step("mock_step", "success")
        return context

class TestPipelineDirector(unittest.TestCase):
    def setUp(self):
        self.config = TEST_CONFIG.copy()
        self.telemetry = TelemetryContext()
        self.director = PipelineDirector(self.config, self.telemetry)

    def test_initialization(self):
        """Verify initialization loads thresholds."""
        assert isinstance(self.director.thresholds, ProcessingThresholds)

    def test_execute_custom_recipe(self):
        """Verify that a custom recipe executes the steps."""
        self.director.STEP_REGISTRY["mock_step"] = MockStep
        self.config["pipeline_recipe"] = [{"step": "mock_step"}]

        context = {}
        result = self.director.execute(context)

        assert result.get("mock_executed") is True
        assert len(self.telemetry.steps) == 1
        assert self.telemetry.steps[0]["step"] == "mock_step"

    @patch("app.pipeline_director.LoadDataStep")
    @patch("app.pipeline_director.MergeDataStep")
    @patch("app.pipeline_director.CalculateCostsStep")
    @patch("app.pipeline_director.FinalMergeStep")
    @patch("app.pipeline_director.BuildOutputStep")
    def test_default_pipeline_execution(self, MockBuild, MockFinal, MockCalc, MockMerge, MockLoad):
        """Verify default pipeline execution order."""
        # Setup mocks behavior
        mock_load_instance = MockLoad.return_value
        mock_load_instance.execute.return_value = {"updated": "context"}

        mock_merge_instance = MockMerge.return_value
        mock_merge_instance.execute.return_value = {}

        mock_calc_instance = MockCalc.return_value
        mock_calc_instance.execute.return_value = {}

        mock_final_instance = MockFinal.return_value
        mock_final_instance.execute.return_value = {}

        mock_build_instance = MockBuild.return_value
        mock_build_instance.execute.return_value = {}

        # Remove pipeline_recipe to force default path
        if "pipeline_recipe" in self.config:
            del self.config["pipeline_recipe"]

        director = PipelineDirector(self.config, self.telemetry)

        # Override the registry with our mocks
        director.STEP_REGISTRY = {
            "load_data": MockLoad,
            "merge_data": MockMerge,
            "calculate_costs": MockCalc,
            "final_merge": MockFinal,
            "build_output": MockBuild,
        }

        director.execute({"initial": "ctx"})

        assert MockLoad.called
        assert MockMerge.called
        assert MockCalc.called
        assert MockFinal.called
        assert MockBuild.called

    def test_error_handling(self):
        """Verify that errors in steps are recorded in telemetry."""
        class ErrorStep(ProcessingStep):
            def __init__(self, config, thresholds):
                pass
            def execute(self, context, telemetry):
                raise ValueError("Step failed")

        self.director.STEP_REGISTRY["error_step"] = ErrorStep
        self.config["pipeline_recipe"] = [{"step": "error_step"}]

        with self.assertRaises(ValueError):
            self.director.execute({})

        assert len(self.telemetry.errors) > 0
        assert self.telemetry.errors[0]["step"] == "error_step"

class TestProcessingSteps(unittest.TestCase):
    """Test individual steps signature updates."""

    def setUp(self):
        self.config = TEST_CONFIG.copy()
        self.thresholds = ProcessingThresholds()
        self.telemetry = TelemetryContext()

    @patch("app.pipeline_director.PresupuestoProcessor")
    @patch("app.pipeline_director.InsumosProcessor")
    @patch("app.pipeline_director.DataFluxCondenser")
    @patch("app.pipeline_director.FileValidator")
    def test_load_data_step(self, MockFileVal, MockCondenser, MockInsumos, MockPresupuesto):
        """Test LoadDataStep with telemetry."""
        step = LoadDataStep(self.config, self.thresholds)

        # Setup mocks
        MockFileVal.return_value.validate_file_exists.return_value = (True, None)
        MockPresupuesto.return_value.process.return_value = pd.DataFrame({"A": [1]})
        MockInsumos.return_value.process.return_value = pd.DataFrame({"B": [2]})
        MockCondenser.return_value.stabilize.return_value = pd.DataFrame({"C": [3]})

        context = {
            "presupuesto_path": "p.csv",
            "apus_path": "a.csv",
            "insumos_path": "i.csv"
        }

        step.execute(context, self.telemetry)

        assert "load_data" in [s["step"] for s in self.telemetry.steps]
        # Check specific metric recorded
        assert "load_data.presupuesto_rows" in self.telemetry.metrics

    def test_merge_data_step_signature(self):
        """Test MergeDataStep accepts telemetry."""
        step = MergeDataStep(self.config, self.thresholds)
        # Mock logic inside execute to avoid full execution
        with patch("app.pipeline_director.DataMerger") as MockMerger:
            MockMerger.return_value.merge_apus_with_insumos.return_value = pd.DataFrame()
            context = {"df_apus_raw": pd.DataFrame(), "df_insumos": pd.DataFrame()}
            step.execute(context, self.telemetry)
            assert "merge_data" in [s["step"] for s in self.telemetry.steps]

if __name__ == "__main__":
    unittest.main()
