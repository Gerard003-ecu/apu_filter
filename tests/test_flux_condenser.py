"""
Suite de Pruebas Exhaustiva para el `DataFluxCondenser`.

Esta suite de pruebas verifica todos los aspectos del `DataFluxCondenser`,
asegurando su robustez, fiabilidad y comportamiento esperado bajo una amplia
variedad de escenarios, incluyendo el nuevo control adaptativo PID y el Modelo Energético Escalar.

Cobertura de Pruebas:
- **Inicialización:** Valida que el condensador se configure correctamente,
  incluyendo la gestión de configuraciones personalizadas y por defecto.
- **Motor de Física RLC (Energético):** Pruebas unitarias para `FluxPhysicsEngine`, asegurando
  que los cálculos de Energía Potencial, Cinética y Potencia Disipada sean precisos.
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

# ==================== TESTS: FluxPhysicsEngine (Energía Escalar) ====================

class TestFluxPhysicsEngine:
    """Pruebas del motor de física con el nuevo modelo de energía escalar."""

    @pytest.fixture
    def engine(self):
        return FluxPhysicsEngine(capacitance=5000, resistance=10, inductance=2.0)

    def test_calculate_energy_metrics(self, engine):
        """Verifica que se calculen las métricas de energía correctamente."""
        # Caso: 100 records, 50 hits (50% calidad)
        metrics = engine.calculate_metrics(total_records=100, cache_hits=50)

        assert "potential_energy" in metrics
        assert "kinetic_energy" in metrics
        assert "dissipated_power" in metrics

        # Verificamos coherencia básica
        assert metrics["potential_energy"] > 0
        assert metrics["kinetic_energy"] > 0
        assert metrics["dissipated_power"] > 0

        # Si la calidad es 50% (0.5), complexity es 0.5.
        # kinetic_energy = 0.5 * L * I^2 = 0.5 * 2.0 * 0.5^2 = 0.25
        assert math.isclose(metrics["kinetic_energy"], 0.25)

    def test_calculate_metrics_ideal_flow(self, engine):
        """Caso flujo ideal (100% hits) -> Máxima energía cinética, mínima disipación."""
        metrics = engine.calculate_metrics(total_records=100, cache_hits=100)

        # I = 1.0
        # Kinetic = 0.5 * 2.0 * 1.0^2 = 1.0 J
        assert math.isclose(metrics["kinetic_energy"], 1.0)

        # Noise I = 0.0 -> Dissipated Power = 0.0
        assert metrics["dissipated_power"] == 0.0

    def test_calculate_metrics_dirty_flow(self, engine):
        """Caso flujo sucio (0% hits) -> Máxima disipación."""
        metrics = engine.calculate_metrics(total_records=100, cache_hits=0)

        # I = 0.0 -> Kinetic = 0.0
        assert metrics["kinetic_energy"] == 0.0

        # Noise I = 1.0. Dynamic R será alta. Power será alto.
        assert metrics["dissipated_power"] > 10.0 # R base es 10, dynamic es > 10

    def test_zero_records(self, engine):
        metrics = engine.calculate_metrics(0, 0)
        assert metrics["saturation"] == 0.0
        assert metrics["potential_energy"] == 0.0
        assert metrics["kinetic_energy"] == 0.0
        assert metrics["dissipated_power"] == 0.0

    def test_system_diagnosis_energy(self, engine):
        """Prueba los diagnósticos basados en energía."""
        # Caso normal
        metrics = {
            "potential_energy": 500,
            "kinetic_energy": 1.0,
            "flyback_voltage": 0.0
        }
        diag = engine.get_system_diagnosis(metrics)
        assert "EQUILIBRIO" in diag

        # Caso baja inercia
        metrics["kinetic_energy"] = 0.05
        diag = engine.get_system_diagnosis(metrics)
        assert "ESTANCADO" in diag

        # Caso sobrecarga de presión
        metrics["kinetic_energy"] = 1.0
        metrics["potential_energy"] = 2000
        diag = engine.get_system_diagnosis(metrics)
        assert "SOBRECARGA" in diag

# ==================== TESTS: DataFluxCondenser (Integration) ====================

class TestStabilizePID:
    """Pruebas de integración para el flujo estabilizado con PID y Energía."""

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
            input_len = len(mock_processor.raw_records)
            return pd.DataFrame([{'res': 1}] * input_len)

        mock_processor.process_all.side_effect = process_side_effect
        mock_processor_class.return_value = mock_processor

        result = condenser.stabilize(str(mock_csv_file))

        assert len(result) == 100
        assert mock_processor_class.call_count >= 1
        assert mock_processor.process_all.call_count >= 1

    @patch('app.flux_condenser.APUProcessor')
    @patch('app.flux_condenser.ReportParserCrudo')
    def test_thermal_breaker_activation(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        caplog,
        sample_raw_records
    ):
        """
        Verifica que el 'Disyuntor Térmico' se active si la potencia disipada es alta.
        Simularemos una alta disipación mockeando 'physics.calculate_metrics' indirectamente
        o manipulando los datos para que physics calcule alta disipación.
        """
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records # 100 records
        # Cache vacía -> 0 hits -> Alta fricción -> Alta disipación
        mock_parser.get_parse_cache.return_value = {}
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = pd.DataFrame([{'a':1}] * 10) # Dummy
        mock_processor_class.return_value = mock_processor

        # Ejecutamos
        with caplog.at_level(logging.WARNING):
            condenser.stabilize(str(mock_csv_file))

        # Verificamos que se haya logueado el sobrecalentamiento
        # FluxPhysicsEngine con 0 hits en 100 records y R=10 genera mucha disipación
        # Dissipated Power = (1.0^2) * (10 * (1 + 1.0 * 5)) = 1 * 60 = 60 Watts > 50.0 Threshold
        assert "SOBRECALENTAMIENTO" in caplog.text
        assert "Frenando forzosamente" in caplog.text

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
        """Debe loguear warning si hay flyback voltage alto (compatibilidad)."""
        # 10 records, 0 hits -> High flyback likely
        records = [{'id': i} for i in range(10)]

        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = records
        mock_parser.get_parse_cache.return_value = {} # 0 hits
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = pd.DataFrame()
        mock_processor_class.return_value = mock_processor

        # Forzar lógica de flyback en el physics real o confiar en que 10 records y 0 hits genera flyback > 0.8?
        # Con 10 records: I=0. dt = log1p(10) ~= 2.39. delta_i = 1.
        # V_L = 2.0 * (1/2.39) = 0.83 > 0.8. Debería dispararse.

        with caplog.at_level(logging.WARNING):
            condenser.stabilize(str(mock_csv_file))

        # El log puede ser el nuevo "Pico de inestabilidad" o el del physics check.
        # En el código actualizado:
        # if metrics["flyback_voltage"] > 0.8: log warning
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
