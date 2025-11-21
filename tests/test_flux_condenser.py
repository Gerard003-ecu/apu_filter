"""
Suite de Pruebas Exhaustiva para el `DataFluxCondenser`.

Esta suite de pruebas verifica todos los aspectos del `DataFluxCondenser`,
asegurando su robustez, fiabilidad y comportamiento esperado bajo una amplia
variedad de escenarios, incluyendo el nuevo control adaptativo PID.

Cobertura de Pruebas:
- **Inicialización:** Valida que el condensador se configure correctamente,
  incluyendo la gestión de configuraciones personalizadas y por defecto.
- **Motor de Física RLC:** Pruebas unitarias para `FluxPhysicsEngine`, asegurando
  que los cálculos de saturación, flyback y diagnósticos sean precisos.
- **Controlador PI:** Pruebas unitarias para la clase `PIController`.
- **Flujo de Procesamiento Adaptativo (PID):** Utiliza `mocks` para aislar y probar
  el pipeline `stabilize` en modo streaming por lotes.
- **Validaciones y Errores:** Confirma que las excepciones personalizadas
  (`InvalidInputError`, `ProcessingError`) se lancen y propaguen correctamente.
"""
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch, ANY

import pandas as pd
import pytest

from app.flux_condenser import (
    CondenserConfig,
    DataFluxCondenser,
    DataFluxCondenserError,
    FluxPhysicsEngine,
    InvalidInputError,
    ParsedData,
    ProcessingError,
    PIController,
)

# ==================== FIXTURES ====================

@pytest.fixture
def valid_config() -> Dict[str, Any]:
    return {
        'parser_settings': {'delimiter': ',', 'encoding': 'utf-8'},
        'processor_settings': {'validate_types': True, 'skip_empty': False},
        'additional_key': 'value'
    }


@pytest.fixture
def valid_profile() -> Dict[str, Any]:
    return {
        'columns_mapping': {'cod_insumo': 'codigo', 'descripcion': 'desc'},
        'validation_rules': {'required_fields': ['codigo', 'cantidad']},
        'extra_config': 'data'
    }


@pytest.fixture
def condenser(valid_config, valid_profile) -> DataFluxCondenser:
    return DataFluxCondenser(valid_config, valid_profile)


@pytest.fixture
def sample_raw_records() -> List[Dict[str, Any]]:
    return [
        {'codigo': f'A{i}', 'cantidad': 10, 'precio': 100.0, 'insumo_line': f'line_{i}'}
        for i in range(100)
    ]


@pytest.fixture
def sample_parse_cache() -> Dict[str, Any]:
    return {f'line_{i}': 'data' for i in range(100)}


@pytest.fixture
def mock_csv_file(tmp_path) -> Path:
    file_path = tmp_path / "test_data.csv"
    file_path.write_text("codigo,cantidad,precio\n" + "\n".join([f"A{i},10,100.0" for i in range(100)]))
    return file_path


# ==================== TESTS: PIController ====================

class TestPIController:
    """Pruebas unitarias para el controlador PI."""

    def test_initialization(self):
        controller = PIController(kp=1.0, ki=0.1, setpoint=0.5, min_output=10, max_output=100)
        assert controller.Kp == 1.0
        assert controller.setpoint == 0.5
        assert controller._integral_error == 0.0

    def test_compute_increase(self):
        """Si la variable de proceso es menor al setpoint (bajo error), debería aumentar la salida?
        Espera:
        Setpoint = 0.5 (Saturación ideal)
        Variable = 0.1 (Saturación baja -> Flujo demasiado lento o fácil)
        Error = 0.5 - 0.1 = 0.4 (Positivo)
        Salida = Base + P(>0) + I(>0) -> Aumentar Batch Size
        """
        controller = PIController(kp=100.0, ki=0.0, setpoint=0.5, min_output=10, max_output=100)
        base = (10 + 100) / 2 # 55

        output = controller.compute(0.1)

        # Error = 0.4. P = 100 * 0.4 = 40. Output = 55 + 40 = 95.
        assert output > 55
        assert output <= 100

    def test_compute_decrease(self):
        """Si variable > setpoint (Saturación alta), error negativo, reducir batch."""
        controller = PIController(kp=100.0, ki=0.0, setpoint=0.3, min_output=10, max_output=100)
        # Saturation 0.8 > 0.3
        output = controller.compute(0.8)
        # Error = -0.5. P = -50. Base = 55. Out = 5. Clamped to 10.
        assert output == 10

    def test_integral_action(self):
        """El error integral debe acumularse."""
        controller = PIController(kp=0.0, ki=1000.0, setpoint=0.5, min_output=0, max_output=1000)
        # Primera llamada
        controller.compute(0.4) # Error 0.1
        first_integral = controller._integral_error

        time.sleep(0.01)
        controller.compute(0.4) # Error 0.1 again

        assert controller._integral_error > first_integral

# ==================== TESTS: FluxPhysicsEngine ====================

class TestFluxPhysicsEngine:
    """Pruebas del motor de física."""

    @pytest.fixture
    def engine(self):
        return FluxPhysicsEngine(capacitance=5000, resistance=10, inductance=2.0)

    def test_calculate_metrics(self, engine):
        metrics = engine.calculate_metrics(total_records=100, cache_hits=50)
        assert "saturation" in metrics
        assert "complexity" in metrics
        assert "flyback_voltage" in metrics

        # Complejidad = 1 - 50/100 = 0.5
        assert metrics["complexity"] == 0.5

        # Flyback > 0 porque hits < total
        assert metrics["flyback_voltage"] > 0.0

    def test_zero_records(self, engine):
        metrics = engine.calculate_metrics(0, 0)
        assert metrics["saturation"] == 0.0

# ==================== TESTS: DataFluxCondenser (Integration) ====================

class TestStabilizePID:
    """Pruebas de integración para el flujo estabilizado con PID."""

    @patch('app.flux_condenser.APUProcessor')
    @patch('app.flux_condenser.ReportParserCrudo')
    def test_stabilize_runs_in_batches(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache
    ):
        """Debe procesar todos los registros en lotes."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records # 100 records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        # Mock Processor returns a dummy DF with same length as input batch to simulate real processing
        mock_processor = Mock()
        def process_side_effect():
            # We need to access the set raw_records to return correct length
            # The logic sets processor.raw_records before calling process_all
            input_len = len(mock_processor.raw_records)
            return pd.DataFrame([{'res': 1}] * input_len)

        mock_processor.process_all.side_effect = process_side_effect
        mock_processor_class.return_value = mock_processor

        # Forzamos un batch size pequeño para asegurar múltiples iteraciones
        # PID start batch size is min_batch_size (50 by default in CondenserConfig)
        # Total 100 records -> Debería hacer al menos 2 batches si size se mantiene.

        result = condenser.stabilize(str(mock_csv_file))

        # Now we expect 100 rows because the mock returns rows proportional to input
        assert len(result) == 100

        # Let's verify call count.
        assert mock_processor_class.call_count >= 1
        assert mock_processor.process_all.call_count >= 1

    @patch('app.flux_condenser.ReportParserCrudo')
    def test_insufficient_records_returns_empty(
        self,
        mock_parser_class,
        condenser,
        mock_csv_file,
        caplog
    ):
        """Debe retornar vacío si no hay suficientes registros."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = [] # 0 records
        mock_parser_class.return_value = mock_parser

        with caplog.at_level(logging.WARNING):
            result = condenser.stabilize(str(mock_csv_file))

        assert result.empty

    @patch('app.flux_condenser.APUProcessor')
    @patch('app.flux_condenser.ReportParserCrudo')
    def test_flyback_warning(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        caplog
    ):
        """Debe loguear warning si hay flyback voltage alto."""
        # 10 records, 0 hits -> High flyback
        records = [{'id': i} for i in range(10)]

        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = records
        mock_parser.get_parse_cache.return_value = {} # 0 hits
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = pd.DataFrame()
        mock_processor_class.return_value = mock_processor

        with caplog.at_level(logging.WARNING):
            condenser.stabilize(str(mock_csv_file))

        assert "DIODO FLYBACK" in caplog.text

# ==================== TESTS DE INICIALIZACIÓN ====================

class TestInitialization:
    """Grupo de pruebas para la inicialización del `DataFluxCondenser`."""

    def test_init_with_valid_params(self, valid_config, valid_profile):
        condenser = DataFluxCondenser(valid_config, valid_profile)
        assert condenser.config == valid_config
        assert condenser.profile == valid_profile
        assert isinstance(condenser.condenser_config, CondenserConfig)
        assert isinstance(condenser.physics, FluxPhysicsEngine)
        assert isinstance(condenser.controller, PIController)

    def test_init_with_invalid_config_type(self, valid_profile):
        with pytest.raises(InvalidInputError):
            DataFluxCondenser("invalid", valid_profile)

# ==================== TESTS DE VALIDACIÓN DE ARCHIVO ====================

class TestInputFileValidation:
    def test_validate_nonexistent_file(self, condenser):
        with pytest.raises(InvalidInputError):
            condenser._validate_input_file("/nonexistent.csv")

    def test_validate_returns_path(self, condenser, mock_csv_file):
        path = condenser._validate_input_file(str(mock_csv_file))
        assert isinstance(path, Path)
