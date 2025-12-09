"""
Suite de pruebas robusta para el pipeline director (PipelineDirector).

Incluye:
- Pruebas unitarias para el orquestador
- Pruebas de integración con los Steps
- Verificación de la inyección de telemetría
- Pruebas de los validadores robustos
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
import numpy as np
from pathlib import Path
import os

from app.pipeline_director import (
    LoadDataStep,
    MergeDataStep,
    PipelineDirector,
    ProcessingStep,
    ProcessingThresholds,
    DataValidator,
    FileValidator,
    PresupuestoProcessor,
    ColumnNames,
    DataMerger,
    PipelineDirector
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

class TestDataValidator(unittest.TestCase):
    """Pruebas para los métodos robustecidos de DataValidator"""

    def test_validate_dataframe_not_empty(self):
        # Caso 1: None
        valid, error = DataValidator.validate_dataframe_not_empty(None, "test")
        self.assertFalse(valid)
        self.assertIn("None", error)

        # Caso 2: No es DataFrame
        valid, error = DataValidator.validate_dataframe_not_empty("not a df", "test")
        self.assertFalse(valid)
        self.assertIn("no es un DataFrame", error)

        # Caso 3: Vacío
        valid, error = DataValidator.validate_dataframe_not_empty(pd.DataFrame(), "test")
        self.assertFalse(valid)
        self.assertIn("vacío", error)

        # Caso 4: Solo nulos
        df_nulls = pd.DataFrame({'A': [None, None], 'B': [np.nan, np.nan]})
        valid, error = DataValidator.validate_dataframe_not_empty(df_nulls, "test")
        self.assertFalse(valid)
        self.assertIn("solo valores nulos", error)

        # Caso 5: Válido
        df_valid = pd.DataFrame({'A': [1]})
        valid, error = DataValidator.validate_dataframe_not_empty(df_valid, "test")
        self.assertTrue(valid)
        self.assertIsNone(error)

    def test_validate_required_columns(self):
        df = pd.DataFrame({'Col1': [1], 'Col2': [2]})

        # Caso 1: Válido
        valid, error = DataValidator.validate_required_columns(df, ['Col1'], "test")
        self.assertTrue(valid)

        # Caso 2: Case insensitive
        valid, error = DataValidator.validate_required_columns(df, ['col1'], "test")
        self.assertTrue(valid)

        # Caso 3: Faltante
        valid, error = DataValidator.validate_required_columns(df, ['Col3'], "test")
        self.assertFalse(valid)
        self.assertIn("Faltan columnas", error)

        # Caso 4: Input inválido
        valid, error = DataValidator.validate_required_columns(None, ['Col1'], "test")
        self.assertFalse(valid)

    def test_detect_and_log_duplicates(self):
        df = pd.DataFrame({
            'ID': [1, 2, 2, 3],
            'Val': ['a', 'b', 'b', 'c']
        })

        # Caso 1: Detectar y eliminar (default keep='first')
        df_clean = DataValidator.detect_and_log_duplicates(df, ['ID'], "test")
        self.assertEqual(len(df_clean), 3)
        self.assertEqual(df_clean.iloc[1]['ID'], 2)

        # Caso 2: Columna inexistente
        df_clean = DataValidator.detect_and_log_duplicates(df, ['Missing'], "test")
        self.assertEqual(len(df_clean), 4) # No hace nada

        # Caso 3: Input inválido
        df_clean = DataValidator.detect_and_log_duplicates(None, ['ID'], "test")
        self.assertTrue(df_clean.empty)

class TestFileValidator(unittest.TestCase):
    """Pruebas para los métodos robustecidos de FileValidator"""

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.stat')
    @patch('os.access')
    def test_validate_file_exists(self, mock_access, mock_stat, mock_is_file, mock_exists):
        # Setup mocks
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_access.return_value = True
        mock_stat.return_value.st_size = 100

        # Caso 1: Válido
        valid, error = FileValidator.validate_file_exists("data.csv", "test")
        self.assertTrue(valid)

        # Caso 2: No existe
        mock_exists.return_value = False
        valid, error = FileValidator.validate_file_exists("data.csv", "test")
        self.assertFalse(valid)
        self.assertIn("no encontrado", error)
        mock_exists.return_value = True

        # Caso 3: No es archivo
        mock_is_file.return_value = False
        valid, error = FileValidator.validate_file_exists("data_dir", "test")
        self.assertFalse(valid)
        self.assertIn("no es un archivo", error)
        mock_is_file.return_value = True

        # Caso 4: Sin permisos
        mock_access.return_value = False
        valid, error = FileValidator.validate_file_exists("data.csv", "test")
        self.assertFalse(valid)
        self.assertIn("Sin permisos", error)
        mock_access.return_value = True

        # Caso 5: Tamaño muy pequeño
        mock_stat.return_value.st_size = 5
        valid, error = FileValidator.validate_file_exists("data.csv", "test", min_size=10)
        self.assertFalse(valid)
        self.assertIn("demasiado pequeño", error)

class TestPresupuestoProcessorRobustness(unittest.TestCase):
    def setUp(self):
        self.config = TEST_CONFIG.copy()
        self.thresholds = ProcessingThresholds()
        self.profile = {"loader_params": {}}
        self.processor = PresupuestoProcessor(self.config, self.thresholds, self.profile)

    def test_clean_phantom_rows_robust(self):
        df = pd.DataFrame({
            'A': ['val', '', 'nan', None, ' '],
            'B': [1, np.nan, np.nan, np.nan, np.nan]
        })

        cleaned = self.processor._clean_phantom_rows(df)
        # Debería quedar solo la fila 0 ('val', 1)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]['A'], 'val')

    def test_process_presupuesto_error_handling(self):
        # Caso: load_data retorna None o error
        with patch('app.pipeline_director.load_data') as mock_load:
            mock_load.return_value = None
            result = self.processor.process("path.csv")
            self.assertTrue(result.empty)

class TestDataMergerRobustness(unittest.TestCase):
    def setUp(self):
        self.thresholds = ProcessingThresholds()
        self.merger = DataMerger(self.thresholds)

    def test_merge_apus_with_insumos_validation(self):
        df_apus = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ['Material A'],
            'other': [1]
        })
        df_insumos = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ['Material A'],
            'price': [100]
        })

        # Caso normal
        merged = self.merger.merge_apus_with_insumos(df_apus, df_insumos)
        self.assertEqual(len(merged), 1)
        self.assertIn('price', merged.columns)

        # Caso: df_insumos vacío
        merged_empty = self.merger.merge_apus_with_insumos(df_apus, pd.DataFrame())
        self.assertEqual(len(merged_empty), 1)
        self.assertNotIn('price', merged_empty.columns)

        # Caso: Columnas faltantes
        df_bad = pd.DataFrame({'wrong': [1]})
        merged_bad = self.merger.merge_apus_with_insumos(df_bad, df_insumos)
        self.assertTrue(merged_bad.equals(df_bad))

    def test_merge_with_presupuesto_robust(self):
        df_presupuesto = pd.DataFrame({
            ColumnNames.CODIGO_APU: ['A1', 'A2'],
            'qty': [10, 20]
        })
        df_costos = pd.DataFrame({
            ColumnNames.CODIGO_APU: ['A1', 'A2'],
            'cost': [100, 200]
        })

        # Caso normal
        merged = self.merger.merge_with_presupuesto(df_presupuesto, df_costos)
        self.assertEqual(len(merged), 2)
        self.assertIn('cost', merged.columns)

        # Caso duplicados (debe manejar MergeError o advertir)
        df_costos_dup = pd.DataFrame({
            ColumnNames.CODIGO_APU: ['A1', 'A1'],
            'cost': [100, 100]
        })

        with self.assertLogs(level='WARNING'):
            merged_dup = self.merger.merge_with_presupuesto(df_presupuesto, df_costos_dup)
            # Debería retornar resultado aunque haya duplicados (join m:1 o 1:m)
            self.assertGreaterEqual(len(merged_dup), 2)

class TestPipelineDirectorExecution(unittest.TestCase):
    def setUp(self):
        self.config = TEST_CONFIG.copy()
        self.telemetry = TelemetryContext()
        self.director = PipelineDirector(self.config, self.telemetry)

    def test_execute_robustness(self):
        # Caso: Contexto inválido
        with self.assertRaises(ValueError):
            self.director.execute(None)

        # Caso: Step falla con error crítico
        self.director.STEP_REGISTRY["fail_step"] = MockStep
        self.config["pipeline_recipe"] = [{"step": "fail_step"}]

        with patch.object(MockStep, 'execute', side_effect=Exception("Critical Fail")):
            with self.assertRaises(RuntimeError):
                self.director.execute({})

            # Verificar que se registró en telemetría
            self.assertTrue(any(e['step'] == 'fail_step' for e in self.telemetry.errors))

class TestProcessingSteps(unittest.TestCase):
    """Test individual steps with robust logic."""

    def setUp(self):
        self.config = TEST_CONFIG.copy()
        self.thresholds = ProcessingThresholds()
        self.telemetry = TelemetryContext()

    @patch("app.pipeline_director.PresupuestoProcessor")
    @patch("app.pipeline_director.InsumosProcessor")
    @patch("app.pipeline_director.DataFluxCondenser")
    @patch("app.pipeline_director.FileValidator")
    def test_load_data_step_robust(self, MockFileVal, MockCondenser, MockInsumos, MockPresupuesto):
        """Test LoadDataStep handles empty results gracefully."""
        step = LoadDataStep(self.config, self.thresholds)

        # Setup mocks to return empty DF
        MockFileVal.return_value.validate_file_exists.return_value = (True, None)
        MockPresupuesto.return_value.process.return_value = pd.DataFrame() # Vacío

        context = {
            "presupuesto_path": "p.csv",
            "apus_path": "a.csv",
            "insumos_path": "i.csv",
        }

        # Expect failure due to empty dataframe validation
        with self.assertRaises(ValueError):
            step.execute(context, self.telemetry)

        self.assertTrue(any("vacío" in e['message'] for e in self.telemetry.errors))

if __name__ == "__main__":
    unittest.main()
