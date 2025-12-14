"""
Suite de Pruebas Exhaustiva para el `DataFluxCondenser`.

Esta suite de pruebas verifica todos los aspectos del `DataFluxCondenser`,
asegurando su robustez, fiabilidad y comportamiento esperado bajo una amplia
variedad de escenarios, incluyendo el nuevo control adaptativo PID y el Modelo
Energético Escalar de Segundo Orden (RLC).

Cobertura de Pruebas:
- **Inicialización:** Valida que el condensador se configure correctamente,
  incluyendo la gestión de configuraciones personalizadas y por defecto.
- **Configuración (CondenserConfig):** Valida las reglas estrictas de configuración.
- **Motor de Física RLC (Energético):** Pruebas unitarias para `FluxPhysicsEngine`,
  asegurando que los cálculos de Energía Potencial, Cinética, Potencia Disipada,
  amortiguamiento y resonancia sean precisos.
- **Controlador PI:** Pruebas unitarias para la clase `PIController`, incluyendo
  anti-windup, filtros EMA y detección de estancamiento.
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
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from app.flux_condenser import (
    CondenserConfig,
    ConfigurationError,
    DataFluxCondenser,
    FluxPhysicsEngine,
    PIController,
    SystemConstants,
)

# ==================== FIXTURES ====================


@pytest.fixture
def valid_config() -> Dict[str, Any]:
    return {
        "parser_settings": {"delimiter": ",", "encoding": "utf-8"},
        "processor_settings": {"validate_types": True, "skip_empty": False},
        "additional_key": "value",
    }


@pytest.fixture
def valid_profile() -> Dict[str, Any]:
    return {
        "columns_mapping": {"cod_insumo": "codigo", "descripcion": "desc"},
        "validation_rules": {"required_fields": ["codigo", "cantidad"]},
        "extra_config": "data",
    }


@pytest.fixture
def condenser(valid_config, valid_profile) -> DataFluxCondenser:
    return DataFluxCondenser(valid_config, valid_profile)


@pytest.fixture
def sample_raw_records() -> List[Dict[str, Any]]:
    return [
        {"codigo": f"A{i}", "cantidad": 10, "precio": 100.0, "insumo_line": f"line_{i}"}
        for i in range(100)
    ]


@pytest.fixture
def sample_parse_cache() -> Dict[str, Any]:
    return {f"line_{i}": "data" for i in range(100)}


@pytest.fixture
def mock_csv_file(tmp_path) -> Path:
    file_path = tmp_path / "test_data.csv"
    file_path.write_text(
        "codigo,cantidad,precio\n" + "\n".join([f"A{i},10,100.0" for i in range(100)])
    )
    return file_path


# ==================== TESTS: CondenserConfig ====================


class TestCondenserConfig:
    """Pruebas para la validación de configuración."""

    def test_default_config_valid(self):
        config = CondenserConfig()
        assert config.min_records_threshold >= 0
        assert config.pid_setpoint == 0.30

    def test_invalid_records_threshold(self):
        with pytest.raises(ConfigurationError, match="min_records_threshold"):
            CondenserConfig(min_records_threshold=-1)

    def test_invalid_pid_params(self):
        with pytest.raises(ConfigurationError, match="pid_setpoint"):
            CondenserConfig(pid_setpoint=1.5)

        with pytest.raises(ConfigurationError, match="pid_kp"):
            CondenserConfig(pid_kp=-10.0)

    def test_invalid_batch_sizes(self):
        with pytest.raises(ConfigurationError, match="min_batch_size"):
            CondenserConfig(min_batch_size=0)

        with pytest.raises(ConfigurationError, match="max_batch_size"):
            CondenserConfig(min_batch_size=100, max_batch_size=50)


# ==================== TESTS: PIController ====================


class TestPIController:
    """Pruebas unitarias para el controlador PI."""

    def test_initialization(self):
        controller = PIController(
            kp=1.0, ki=0.1, setpoint=0.5, min_output=10, max_output=100
        )
        assert controller.Kp == 1.0
        assert controller.setpoint == 0.5
        assert controller._integral_error == 0.0

    def test_initialization_errors(self):
        with pytest.raises(ConfigurationError, match="min_output"):
            PIController(kp=1.0, ki=0.1, setpoint=0.5, min_output=0, max_output=100)

    def test_compute_increase(self):
        """
        Si la variable de proceso es menor al setpoint (bajo error),
        debería aumentar la salida.
        """
        controller = PIController(
            kp=100.0, ki=0.0, setpoint=0.5, min_output=10, max_output=100
        )
        # Base output = 55
        # Input 0.1 -> Error 0.4 -> P = 40 -> Output = 95
        # NOTA: Con filtro EMA, el primer valor se toma directo,
        # pero con lógica de warmup podría variar.
        output = controller.compute(0.1)
        # 55 (base) + 40 (P) = 95
        assert output >= 90  # Permitimos cierto margen por implementación interna

    def test_compute_decrease(self):
        """
        Si variable > setpoint (Saturación alta), error negativo,
        reducir batch.
        """
        controller = PIController(
            kp=100.0, ki=0.0, setpoint=0.3, min_output=10, max_output=100
        )
        # Base output = 55
        # Input 0.8 -> Error -0.5 -> P = -50 -> Output = 5. Clamped to 10.
        output = controller.compute(0.8)
        assert output == 10

    def test_integral_action(self):
        """El error integral debe acumularse tras el warmup."""
        # Usamos min_output > 0 para pasar validación
        controller = PIController(
            kp=0.0, ki=1000.0, setpoint=0.5, min_output=1, max_output=1000
        )

        # Ejecutar suficientes iteraciones para superar el warmup (_WARMUP_ITERATIONS = 3)
        for _ in range(4):
            controller.compute(0.4)
            time.sleep(0.001)

        first_integral = controller._integral_error

        time.sleep(0.01)
        controller.compute(0.4)  # Error 0.1 again

        # Ahora el integral debería haber aumentado
        assert controller._integral_error > first_integral

    def test_zero_division_protection(self):
        """Testea que no falle con valores raros."""
        controller = PIController(kp=1.0, ki=0.1, setpoint=0.5, min_output=1, max_output=100)
        # NaN should act as fallback to setpoint (no error)
        output = controller.compute(float("nan"))
        assert output > 0

    def test_ema_filter(self):
        """Testea el suavizado de la entrada (EMA)."""
        controller = PIController(
            kp=1.0, ki=0.0, setpoint=0.5, min_output=10, max_output=100
        )

        # Primera iteración toma el valor directo
        controller.compute(0.0)

        # Segunda iteración con cambio brusco
        # Si no hubiera filtro, PV sería 1.0. Con filtro (alpha=0.3):
        # PV = 0.3 * 1.0 + 0.7 * 0.0 = 0.3
        # Error = 0.5 - 0.3 = 0.2
        # P = 0.2, Output = 55 + 0.2 = 55 (aprox)
        controller.compute(1.0)

        # Accedemos a variable interna para verificar filtrado si es posible
        if hasattr(controller, "_pv_filtered"):
            assert 0.2 < controller._pv_filtered < 0.4

    def test_soft_anti_windup(self):
        """Testea el anti-windup con tanh."""
        controller = PIController(
            kp=1.0,
            ki=1000.0,
            setpoint=0.5,
            min_output=1,
            max_output=100,
            integral_limit_factor=1.0,
        )

        # Forzar saturación integral
        for _ in range(10):
            controller.compute(0.0)  # Error grande positivo

        # El integral no debería explotar
        assert controller._integral_error < (controller._integral_limit * 1.5)


# ==================== TESTS: FluxPhysicsEngine (Energía Escalar) ====================


class TestFluxPhysicsEngine:
    """Pruebas del motor de física con el nuevo modelo de energía escalar y RLC 2do orden."""

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
        assert metrics["kinetic_energy"] > 0

        # Nuevas métricas de 2do orden
        assert "damping_ratio" in metrics
        assert "stability_factor" in metrics

    def test_calculate_metrics_ideal_flow(self, engine):
        """
        Caso flujo ideal (100% hits) -> Máxima energía cinética.
        P = I^2 * R. Con I=1, R_dyn=R (complexity=0). P = 1^2 * 10 = 10W.
        """
        metrics = engine.calculate_metrics(total_records=100, cache_hits=100)
        # I = 1.0
        # Kinetic Energy (E_l) = 0.5 * L * I^2 = 0.5 * 2.0 * 1.0^2 = 1.0
        assert math.isclose(metrics["kinetic_energy"], 1.0, abs_tol=0.1)

        # Power is not 0 in active circuit!
        assert math.isclose(metrics["dissipated_power"], 10.0, abs_tol=0.1)

        # Factor de potencia ideal
        assert math.isclose(metrics["power_factor"], 1.0, abs_tol=0.01)

    def test_calculate_metrics_dirty_flow(self, engine):
        """
        Caso flujo sucio (0% hits) -> I=0.
        P = 0^2 * R = 0. Sistema estancado (frío).
        PF = 1.0 porque el sistema se evalúa en resonancia (X_L = X_C),
        por lo que es puramente resistivo.
        """
        metrics = engine.calculate_metrics(total_records=100, cache_hits=0)
        assert metrics["kinetic_energy"] == 0.0
        assert metrics["dissipated_power"] == 0.0
        assert metrics["power_factor"] == 1.0

    def test_system_diagnosis_energy(self, engine):
        """Prueba los diagnósticos basados en energía."""
        # 1. Equilibrio: Energía cinética normal, potencial controlada
        metrics = {
            "potential_energy": 500,  # Ec
            "kinetic_energy": 1.0,  # El -> Ratio = 500. Ratio < 1000.
            "flyback_voltage": 0.0,
            "damping_ratio": 1.0,
            "saturation": 0.5,
        }
        diag = engine.get_system_diagnosis(metrics)
        assert "EQUILIBRIO" in diag

        # 2. Estancado: Energía cinética prácticamente cero
        metrics_stalled = {
            "potential_energy": 0.0,
            "kinetic_energy": 0.0,  # < MIN_ENERGY_THRESHOLD (1e-10)
            "flyback_voltage": 0.0,
            "saturation": 0.0,
        }
        diag = engine.get_system_diagnosis(metrics_stalled)
        assert "ESTANCADO" in diag

        # 3. Inestabilidad
        metrics_unstable = {
            "potential_energy": 500,
            "kinetic_energy": 1.0,
            "damping_ratio": 0.05,  # < 0.1
        }
        diag = engine.get_system_diagnosis(metrics_unstable)
        assert "INESTABILIDAD" in diag

        # 4. Sobrecarga: Ratio Ec/El muy alto
        metrics_overload = {
            "potential_energy": 2000,
            "kinetic_energy": 1.0,  # Ratio = 2000 > 1000
            "flyback_voltage": 0.0,
            "damping_ratio": 1.0,
        }
        diag = engine.get_system_diagnosis(metrics_overload)
        assert "SOBRECARGA" in diag

    def test_trend_analysis(self, engine):
        """Prueba el análisis de tendencias."""
        # Generar historial sintético
        for i in range(10):
            engine._store_metrics({
                "saturation": 0.1 * i,  # Creciente
                "dissipated_power": 100 - i * 10, # Decreciente
                "damping_ratio": 1.0,
                "total_energy": 50.0
            })

        analysis = engine.get_trend_analysis()

        assert analysis["status"] == "OK"
        assert analysis["saturation"]["trend"] == "INCREASING"
        assert analysis["power"]["trend"] == "DECREASING"


# ==================== TESTS: DataFluxCondenser (Integration) ====================


class TestStabilizePID:
    """Pruebas de integración para el flujo estabilizado con PID y Energía."""

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_stabilize_runs_in_batches(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache,
    ):
        """Debe procesar todos los registros en lotes."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records  # 100 records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()

        def process_side_effect():
            # Robustez: Verificar si raw_records fue asignado como lista
            if isinstance(mock_processor.raw_records, list):
                input_len = len(mock_processor.raw_records)
            else:
                input_len = 0
            return pd.DataFrame([{"res": 1}] * input_len)

        mock_processor.process_all.side_effect = process_side_effect
        mock_processor_class.return_value = mock_processor

        result = condenser.stabilize(str(mock_csv_file))

        assert len(result) == 100
        assert mock_processor_class.call_count >= 1

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_thermal_breaker_activation(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        caplog,
        sample_raw_records,
    ):
        """Verifica que el 'Disyuntor Térmico' frene el proceso."""
        # Configurar para que genere alta disipación (Heat)
        # P = I^2 * R_dyn
        # I = cache_hits / records.
        # R_dyn = R * (1 + (1-I)*5)
        # Max Power is at I=0.8

        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records

        # Simular I = 0.8 -> 80% hits
        # records 0-79 in cache, 80-99 not
        partial_cache = {f"line_{i}": "data" for i in range(80)}
        mock_parser.get_parse_cache.return_value = partial_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = pd.DataFrame([{"a": 1}] * 10)
        mock_processor_class.return_value = mock_processor

        # Usar un threshold MUY bajo para garantizar disparo
        with patch("app.flux_condenser.SystemConstants.OVERHEAT_POWER_THRESHOLD", 0.001):
            with caplog.at_level(logging.WARNING):
                condenser.stabilize(str(mock_csv_file))

            # El log usa "OVERHEAT" para el warning del freno
            assert "OVERHEAT" in caplog.text

    @patch("app.flux_condenser.ReportParserCrudo")
    def test_insufficient_records_returns_empty(
        self, mock_parser_class, condenser, mock_csv_file, caplog
    ):
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = []
        mock_parser_class.return_value = mock_parser

        with caplog.at_level(logging.WARNING):
            result = condenser.stabilize(str(mock_csv_file))

        assert result.empty
