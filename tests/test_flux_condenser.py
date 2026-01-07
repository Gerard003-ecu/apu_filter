"""
Suite de Pruebas para el `DataFluxCondenser` - Versión Refinada.

Cobertura actualizada para los métodos refinados:
- PIController con EMA y anti-windup por back-calculation
- FluxPhysicsEngine con Betti numbers correctos y entropía termodinámica
- DataFluxCondenser con telemetría integrada y BatchResult
"""

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from app.flux_condenser import (
    BatchResult,
    CondenserConfig,
    ConfigurationError,
    DataFluxCondenser,
    FluxPhysicsEngine,
    InvalidInputError,
    ParsedData,
    PIController,
    ProcessingError,
    ProcessingStats,
    SystemConstants,
)


# ==================== FIXTURES ====================


@pytest.fixture
def valid_config() -> Dict[str, Any]:
    """Configuración válida para el procesador."""
    return {
        "parser_settings": {"delimiter": ",", "encoding": "utf-8"},
        "processor_settings": {"validate_types": True, "skip_empty": False},
    }


@pytest.fixture
def valid_profile() -> Dict[str, Any]:
    """Perfil válido de mapeo de columnas."""
    return {
        "columns_mapping": {"cod_insumo": "codigo", "descripcion": "desc"},
        "validation_rules": {"required_fields": ["codigo", "cantidad"]},
    }


@pytest.fixture
def default_condenser_config() -> CondenserConfig:
    """Configuración por defecto del condensador."""
    return CondenserConfig()


@pytest.fixture
def condenser(valid_config, valid_profile) -> DataFluxCondenser:
    """Instancia de condensador con configuración válida."""
    return DataFluxCondenser(valid_config, valid_profile)


@pytest.fixture
def sample_raw_records() -> List[Dict[str, Any]]:
    """Registros de ejemplo para pruebas."""
    return [
        {"codigo": f"A{i}", "cantidad": 10, "precio": 100.0, "insumo_line": f"line_{i}"}
        for i in range(100)
    ]


@pytest.fixture
def sample_parse_cache() -> Dict[str, Any]:
    """Caché de parseo de ejemplo."""
    return {f"line_{i}": "data" for i in range(100)}


@pytest.fixture
def mock_csv_file(tmp_path) -> Path:
    """Archivo CSV temporal para pruebas."""
    file_path = tmp_path / "test_data.csv"
    content = "codigo,cantidad,precio\n" + "\n".join(
        [f"A{i},10,100.0" for i in range(100)]
    )
    file_path.write_text(content)
    return file_path


@pytest.fixture
def small_csv_file(tmp_path) -> Path:
    """Archivo CSV pequeño para pruebas de validación."""
    file_path = tmp_path / "small_data.csv"
    file_path.write_text("codigo,cantidad\nA1,10\n")
    return file_path


# ==================== TESTS: CondenserConfig ====================


class TestCondenserConfig:
    """Pruebas para la validación de configuración."""

    def test_default_config_is_valid(self):
        """La configuración por defecto debe ser válida."""
        config = CondenserConfig()

        assert config.min_records_threshold >= 0
        assert 0.0 < config.pid_setpoint < 1.0
        assert config.pid_kp > 0
        assert config.pid_ki >= 0
        assert config.min_batch_size > 0
        assert config.max_batch_size >= config.min_batch_size
        assert config.system_capacitance > 0
        assert config.base_resistance >= 0
        assert config.system_inductance > 0

    def test_custom_valid_config(self):
        """Configuración personalizada válida."""
        config = CondenserConfig(
            min_records_threshold=10,
            pid_setpoint=0.5,
            pid_kp=1000.0,
            pid_ki=50.0,
            min_batch_size=25,
            max_batch_size=2500,
        )

        assert config.min_records_threshold == 10
        assert config.pid_setpoint == 0.5

    def test_invalid_min_records_threshold(self):
        """Threshold negativo debe fallar."""
        with pytest.raises(ConfigurationError, match="min_records_threshold"):
            CondenserConfig(min_records_threshold=-1)

    def test_invalid_pid_setpoint_too_high(self):
        """Setpoint >= 1.0 debe fallar."""
        with pytest.raises(ConfigurationError, match="pid_setpoint"):
            CondenserConfig(pid_setpoint=1.0)

    def test_invalid_pid_setpoint_too_low(self):
        """Setpoint <= 0.0 debe fallar."""
        with pytest.raises(ConfigurationError, match="pid_setpoint"):
            CondenserConfig(pid_setpoint=0.0)

    def test_invalid_pid_kp_negative(self):
        """Kp negativo debe fallar."""
        with pytest.raises(ConfigurationError, match="pid_kp"):
            CondenserConfig(pid_kp=-10.0)

    def test_invalid_min_batch_size_zero(self):
        """min_batch_size = 0 debe fallar."""
        with pytest.raises(ConfigurationError, match="min_batch_size"):
            CondenserConfig(min_batch_size=0)

    def test_invalid_batch_size_inversion(self):
        """min_batch_size > max_batch_size debe fallar."""
        with pytest.raises(ConfigurationError, match="min_batch_size"):
            CondenserConfig(min_batch_size=100, max_batch_size=50)

    def test_invalid_capacitance_zero(self):
        """Capacitancia <= 0 debe fallar."""
        with pytest.raises(ConfigurationError, match="system_capacitance"):
            CondenserConfig(system_capacitance=0)

    def test_invalid_inductance_zero(self):
        """Inductancia <= 0 debe fallar."""
        with pytest.raises(ConfigurationError, match="system_inductance"):
            CondenserConfig(system_inductance=0)


# ==================== TESTS: PIController ====================


class TestPIController:
    """Pruebas unitarias para el controlador PI refinado."""

    @pytest.fixture
    def basic_controller(self) -> PIController:
        """Controlador básico para pruebas."""
        return PIController(
            kp=100.0,
            ki=10.0,
            setpoint=0.5,
            min_output=10,
            max_output=100,
            integral_limit_factor=2.0,
        )

    def test_initialization_valid(self):
        """Inicialización con parámetros válidos."""
        controller = PIController(
            kp=100.0,
            ki=10.0,
            setpoint=0.5,
            min_output=10,
            max_output=100,
        )

        assert controller.Kp == 100.0
        assert controller.Ki == 10.0
        assert controller.setpoint == 0.5
        assert controller.min_output == 10
        assert controller.max_output == 100
        assert controller._integral_error == 0.0
        assert controller._filtered_pv is None
        assert controller._iteration_count == 0

    def test_initialization_with_ki_zero(self):
        """Ki = 0 es válido (controlador P puro)."""
        controller = PIController(
            kp=100.0,
            ki=0.0,
            setpoint=0.5,
            min_output=10,
            max_output=100,
        )
        assert controller.Ki == 0.0

    def test_initialization_invalid_kp_negative(self):
        """Kp negativo debe fallar."""
        with pytest.raises(ConfigurationError, match="Kp"):
            PIController(kp=-1.0, ki=10.0, setpoint=0.5, min_output=10, max_output=100)

    def test_initialization_invalid_kp_zero(self):
        """Kp = 0 debe fallar."""
        with pytest.raises(ConfigurationError, match="Kp"):
            PIController(kp=0.0, ki=10.0, setpoint=0.5, min_output=10, max_output=100)

    def test_initialization_invalid_ki_negative(self):
        """Ki negativo debe fallar."""
        with pytest.raises(ConfigurationError, match="Ki"):
            PIController(kp=100.0, ki=-1.0, setpoint=0.5, min_output=10, max_output=100)

    def test_initialization_invalid_min_output_zero(self):
        """min_output <= 0 debe fallar."""
        with pytest.raises(ConfigurationError, match="min_output"):
            PIController(kp=100.0, ki=10.0, setpoint=0.5, min_output=0, max_output=100)

    def test_initialization_invalid_output_range(self):
        """min_output >= max_output debe fallar."""
        with pytest.raises(ConfigurationError, match="Rango de salida"):
            PIController(kp=100.0, ki=10.0, setpoint=0.5, min_output=100, max_output=50)

    def test_initialization_invalid_setpoint_bounds(self):
        """setpoint fuera de (0, 1) debe fallar."""
        with pytest.raises(ConfigurationError, match="setpoint"):
            PIController(kp=100.0, ki=10.0, setpoint=1.5, min_output=10, max_output=100)

        with pytest.raises(ConfigurationError, match="setpoint"):
            PIController(kp=100.0, ki=10.0, setpoint=-0.1, min_output=10, max_output=100)

    def test_compute_output_increases_when_pv_low(self, basic_controller):
        """
        Cuando PV < setpoint, el error es positivo y la salida debe aumentar.
        """
        # setpoint = 0.5, PV = 0.1 -> error = 0.4
        # output_center = (10 + 100) / 2 = 55
        # P = 100 * 0.4 = 40
        # Output ≈ 55 + 40 = 95
        output = basic_controller.compute(0.1)

        assert output > basic_controller._output_center
        assert output <= basic_controller.max_output

    def test_compute_output_decreases_when_pv_high(self, basic_controller):
        """
        Cuando PV > setpoint, el error es negativo y la salida debe disminuir.
        """
        # setpoint = 0.5, PV = 0.9 -> error = -0.4
        # P = 100 * (-0.4) = -40
        # Output ≈ 55 - 40 = 15
        output = basic_controller.compute(0.9)

        assert output < basic_controller._output_center
        assert output >= basic_controller.min_output

    def test_compute_output_at_setpoint(self, basic_controller):
        """
        Cuando PV = setpoint, el error es ~0 y la salida debe estar cerca del centro.
        """
        output = basic_controller.compute(0.5)

        # Debería estar cerca del centro del rango
        assert abs(output - basic_controller._output_center) < 10

    def test_compute_clamps_to_max(self, basic_controller):
        """La salida debe estar limitada por max_output."""
        # Error muy grande positivo
        output = basic_controller.compute(0.0)
        assert output <= basic_controller.max_output

    def test_compute_clamps_to_min(self, basic_controller):
        """La salida debe estar limitada por min_output."""
        # Error muy grande negativo
        output = basic_controller.compute(1.0)
        assert output >= basic_controller.min_output

    def test_ema_filter_smooths_input(self, basic_controller):
        """El filtro EMA debe suavizar cambios bruscos en la entrada."""
        # Primera llamada: _filtered_pv se inicializa con el valor
        basic_controller.compute(0.2)
        first_filtered = basic_controller._filtered_pv
        assert first_filtered is not None
        assert math.isclose(first_filtered, 0.2, abs_tol=0.01)

        # Segunda llamada con cambio brusco
        basic_controller.compute(0.8)
        second_filtered = basic_controller._filtered_pv

        # El valor filtrado debe estar entre 0.2 y 0.8
        # EMA: new = alpha * measurement + (1 - alpha) * old
        # Con alpha = 0.3: new = 0.3 * 0.8 + 0.7 * 0.2 = 0.24 + 0.14 = 0.38
        assert 0.2 < second_filtered < 0.8
        assert math.isclose(second_filtered, 0.38, abs_tol=0.05)

    def test_integral_accumulates_over_time(self):
        """El término integral debe acumularse con errores persistentes."""
        controller = PIController(
            kp=10.0,
            ki=100.0,
            setpoint=0.5,
            min_output=10,
            max_output=1000,
        )

        # Ejecutar varias iteraciones con el mismo error
        for _ in range(5):
            controller.compute(0.3)  # error = 0.2
            time.sleep(0.01)

        # El integral debe haber acumulado
        assert controller._integral_error > 0

        # Guardar valor actual
        integral_after_5 = controller._integral_error

        # Más iteraciones
        for _ in range(5):
            controller.compute(0.3)
            time.sleep(0.01)

        # El integral debe seguir creciendo
        assert controller._integral_error > integral_after_5

    def test_anti_windup_back_calculation_upper(self):
        """
        Anti-windup por back-calculation previene saturación integral superior.
        """
        controller = PIController(
            kp=10.0,
            ki=5000.0,
            setpoint=0.5,
            min_output=10,
            max_output=100,
            integral_limit_factor=1.0,
        )

        # Forzar saturación con error persistente grande
        for _ in range(100):
            output = controller.compute(0.0)  # error = 0.5
            time.sleep(0.001)

        # La salida debe estar saturada en el máximo
        assert output == controller.max_output

        # El integral debe estar limitado
        assert controller._integral_error <= controller._integral_limit

    def test_anti_windup_back_calculation_lower(self):
        """
        Anti-windup por back-calculation previene saturación integral inferior.
        """
        controller = PIController(
            kp=10.0,
            ki=5000.0,
            setpoint=0.5,
            min_output=10,
            max_output=100,
            integral_limit_factor=1.0,
        )

        # Forzar saturación con error negativo persistente
        for _ in range(100):
            output = controller.compute(1.0)  # error = -0.5
            time.sleep(0.001)

        # La salida debe estar saturada en el mínimo
        assert output == controller.min_output

        # El integral debe estar limitado (valor negativo)
        assert controller._integral_error >= -controller._integral_limit

    def test_lyapunov_metric_converging(self):
        """
        El exponente de Lyapunov debe ser negativo cuando el sistema converge.
        """
        controller = PIController(
            kp=50.0,
            ki=5.0,
            setpoint=0.5,
            min_output=10,
            max_output=100,
        )

        # Simular convergencia hacia setpoint
        pv = 0.1
        for _ in range(20):
            controller.compute(pv)
            # Simular que el sistema responde y PV se acerca al setpoint
            pv = pv + (0.5 - pv) * 0.2
            time.sleep(0.001)

        lyapunov = controller.get_lyapunov_exponent()

        # Con convergencia, el exponente debe ser negativo o cercano a cero
        assert lyapunov < 0.5  # Permitimos algo de margen

    def test_get_stability_analysis_structure(self, basic_controller):
        """Verifica la estructura del análisis de estabilidad."""
        # Ejecutar algunas iteraciones
        for i in range(10):
            basic_controller.compute(0.3 + i * 0.02)
            time.sleep(0.001)

        analysis = basic_controller.get_stability_analysis()

        assert "status" in analysis
        assert analysis["status"] == "OPERATIONAL"
        assert "stability_class" in analysis
        assert "convergence" in analysis
        assert "lyapunov_exponent" in analysis
        assert "error_variance" in analysis
        assert "integral_saturation" in analysis
        assert "iterations" in analysis

    def test_get_stability_analysis_insufficient_data(self):
        """Con datos insuficientes, debe indicarlo."""
        controller = PIController(
            kp=100.0, ki=10.0, setpoint=0.5, min_output=10, max_output=100
        )

        analysis = controller.get_stability_analysis()

        assert analysis["status"] == "INSUFFICIENT_DATA"

    def test_get_diagnostics_structure(self, basic_controller):
        """Verifica la estructura del diagnóstico completo."""
        basic_controller.compute(0.3)

        diagnostics = basic_controller.get_diagnostics()

        assert "status" in diagnostics
        assert "control_metrics" in diagnostics
        assert "stability_analysis" in diagnostics
        assert "parameters" in diagnostics

        # Verificar sub-estructuras
        assert "iteration" in diagnostics["control_metrics"]
        assert "current_integral" in diagnostics["control_metrics"]
        assert "Kp" in diagnostics["parameters"]
        assert "Ki" in diagnostics["parameters"]

    def test_reset_clears_state(self, basic_controller):
        """Reset debe limpiar el estado pero preservar historial."""
        # Acumular estado
        for _ in range(10):
            basic_controller.compute(0.3)
            time.sleep(0.001)

        assert basic_controller._integral_error != 0.0
        assert basic_controller._iteration_count > 0

        # Guardar tamaño del historial
        history_size = len(basic_controller._error_history)

        # Reset
        basic_controller.reset()

        # Verificar estado limpio
        assert basic_controller._integral_error == 0.0
        assert basic_controller._last_time is None
        assert basic_controller._last_error is None
        assert basic_controller._filtered_pv is None
        assert basic_controller._iteration_count == 0

        # El historial se preserva para análisis post-mortem
        assert len(basic_controller._error_history) == history_size

    def test_get_state_serializable(self, basic_controller):
        """El estado debe ser serializable."""
        basic_controller.compute(0.4)

        state = basic_controller.get_state()

        # Verificar estructura
        assert "parameters" in state
        assert "state" in state
        assert "diagnostics" in state

        # Verificar que los valores son serializables (no objetos complejos)
        import json
        json_str = json.dumps(state)  # No debe lanzar excepción
        assert len(json_str) > 0

    def test_handles_nan_input_gracefully(self, basic_controller):
        """NaN en entrada debe manejarse sin crash."""
        # No debe lanzar excepción
        output = basic_controller.compute(float("nan"))

        # Debe retornar un valor válido
        assert basic_controller.min_output <= output <= basic_controller.max_output

    def test_handles_inf_input_gracefully(self, basic_controller):
        """Infinito en entrada debe manejarse sin crash."""
        output = basic_controller.compute(float("inf"))
        assert basic_controller.min_output <= output <= basic_controller.max_output

        output = basic_controller.compute(float("-inf"))
        assert basic_controller.min_output <= output <= basic_controller.max_output


# ==================== TESTS: FluxPhysicsEngine ====================


class TestFluxPhysicsEngine:
    """Pruebas del motor de física RLC refinado."""

    @pytest.fixture
    def engine(self) -> FluxPhysicsEngine:
        """Motor de física con parámetros por defecto."""
        return FluxPhysicsEngine(
            capacitance=5000.0,
            resistance=10.0,
            inductance=2.0,
        )

    def test_initialization_valid(self, engine):
        """Inicialización con parámetros válidos."""
        assert engine.C == 5000.0
        assert engine.R == 10.0
        assert engine.L == 2.0
        assert engine._omega_0 > 0
        assert engine._zeta > 0
        assert engine._Q > 0

    def test_initialization_calculates_derived_params(self, engine):
        """Verifica cálculo de parámetros derivados."""
        # omega_0 = 1 / sqrt(L * C)
        expected_omega = 1.0 / math.sqrt(2.0 * 5000.0)
        assert math.isclose(engine._omega_0, expected_omega, rel_tol=0.01)

        # alpha = R / (2L)
        expected_alpha = 10.0 / (2.0 * 2.0)
        assert math.isclose(engine._alpha, expected_alpha, rel_tol=0.01)

        # Q = sqrt(L/C) / R
        expected_Q = math.sqrt(2.0 / 5000.0) / 10.0
        assert math.isclose(engine._Q, expected_Q, rel_tol=0.01)

    def test_initialization_invalid_capacitance_zero(self):
        """Capacitancia <= 0 debe fallar."""
        with pytest.raises(ConfigurationError, match="Capacitancia"):
            FluxPhysicsEngine(capacitance=0, resistance=10, inductance=2)

    def test_initialization_invalid_capacitance_negative(self):
        """Capacitancia negativa debe fallar."""
        with pytest.raises(ConfigurationError, match="Capacitancia"):
            FluxPhysicsEngine(capacitance=-100, resistance=10, inductance=2)

    def test_initialization_invalid_resistance_negative(self):
        """Resistencia negativa debe fallar."""
        with pytest.raises(ConfigurationError, match="Resistencia"):
            FluxPhysicsEngine(capacitance=5000, resistance=-1, inductance=2)

    def test_initialization_invalid_inductance_zero(self):
        """Inductancia <= 0 debe fallar."""
        with pytest.raises(ConfigurationError, match="Inductancia"):
            FluxPhysicsEngine(capacitance=5000, resistance=10, inductance=0)

    def test_calculate_metrics_returns_all_keys(self, engine):
        """Verifica que calculate_metrics retorna todas las claves esperadas."""
        metrics = engine.calculate_metrics(
            total_records=100,
            cache_hits=50,
            error_count=5,
            processing_time=1.0,
        )

        expected_keys = [
            "saturation",
            "complexity",
            "current_I",
            "potential_energy",
            "kinetic_energy",
            "total_energy",
            "dissipated_power",
            "flyback_voltage",
            "dynamic_resistance",
            "damping_ratio",
            "damping_type",
            "resonant_frequency_hz",
            "quality_factor",
            "time_constant",
            "entropy_shannon",
            "entropy_rate",
            "entropy_ratio",
            "is_thermal_death",
            "betti_0",
            "betti_1",
            "graph_vertices",
            "graph_edges",
        ]

        for key in expected_keys:
            assert key in metrics, f"Falta clave: {key}"

    def test_calculate_metrics_ideal_flow(self, engine):
        """
        Flujo ideal (100% hits): máxima eficiencia, complejidad = 0.
        """
        metrics = engine.calculate_metrics(
            total_records=100,
            cache_hits=100,
        )

        # I = cache_hits / total = 1.0
        assert math.isclose(metrics["current_I"], 1.0, abs_tol=0.01)

        # Complejidad = 1 - I = 0
        assert math.isclose(metrics["complexity"], 0.0, abs_tol=0.01)

        # Energía cinética = 0.5 * L * I² = 0.5 * 2.0 * 1.0² = 1.0
        assert math.isclose(metrics["kinetic_energy"], 1.0, abs_tol=0.1)

        # Potencia disipada = I² * R_dyn = 1.0² * 10 = 10 W
        # (R_dyn = R cuando complexity = 0)
        assert math.isclose(metrics["dissipated_power"], 10.0, abs_tol=1.0)

    def test_calculate_metrics_zero_hits(self, engine):
        """
        Flujo sin hits (0%): I = 0, sistema estancado.
        """
        metrics = engine.calculate_metrics(
            total_records=100,
            cache_hits=0,
        )

        # I = 0
        assert metrics["current_I"] == 0.0

        # Complejidad máxima = 1.0
        assert metrics["complexity"] == 1.0

        # Energía cinética = 0 (I = 0)
        assert metrics["kinetic_energy"] == 0.0

        # Potencia disipada = 0 (I = 0)
        assert metrics["dissipated_power"] == 0.0

    def test_calculate_metrics_partial_hits(self, engine):
        """
        Flujo parcial: valores intermedios.
        """
        metrics = engine.calculate_metrics(
            total_records=100,
            cache_hits=50,
        )

        # I = 0.5
        assert math.isclose(metrics["current_I"], 0.5, abs_tol=0.01)

        # Complejidad = 0.5
        assert math.isclose(metrics["complexity"], 0.5, abs_tol=0.01)

        # Energía cinética = 0.5 * L * I² = 0.5 * 2 * 0.25 = 0.25
        assert math.isclose(metrics["kinetic_energy"], 0.25, abs_tol=0.05)

    def test_calculate_metrics_zero_records_returns_zero_metrics(self, engine):
        """Con 0 registros, retorna métricas cero."""
        metrics = engine.calculate_metrics(
            total_records=0,
            cache_hits=0,
        )

        assert metrics["saturation"] == 0.0
        assert metrics["complexity"] == 1.0
        assert metrics["current_I"] == 0.0
        assert metrics["kinetic_energy"] == 0.0

    def test_calculate_metrics_saturation_bounds(self, engine):
        """La saturación debe estar en [0, 1]."""
        # Probar con diferentes condiciones
        for hits in [0, 50, 100]:
            metrics = engine.calculate_metrics(
                total_records=100,
                cache_hits=hits,
                processing_time=0.1,
            )
            assert 0.0 <= metrics["saturation"] <= 1.0

    def test_calculate_metrics_dynamic_resistance(self, engine):
        """
        Resistencia dinámica aumenta con la complejidad.
        R_dyn = R * (1 + complexity * FACTOR)
        """
        # Flujo ideal: R_dyn = R
        metrics_ideal = engine.calculate_metrics(total_records=100, cache_hits=100)
        assert math.isclose(
            metrics_ideal["dynamic_resistance"],
            engine.R,
            abs_tol=0.1,
        )

        # Flujo sin hits: R_dyn = R * (1 + 1 * 5) = R * 6
        metrics_zero = engine.calculate_metrics(total_records=100, cache_hits=0)
        expected_r_dyn = engine.R * (1 + SystemConstants.COMPLEXITY_RESISTANCE_FACTOR)
        assert math.isclose(
            metrics_zero["dynamic_resistance"],
            expected_r_dyn,
            abs_tol=1.0,
        )

    def test_calculate_system_entropy_zero_errors(self, engine):
        """Entropía con 0 errores."""
        entropy = engine.calculate_system_entropy(
            total_records=100,
            error_count=0,
            processing_time=1.0,
        )

        # Con 0 errores, entropía de Shannon = 0
        assert entropy["shannon_entropy"] == 0.0
        assert entropy["is_thermal_death"] is False

    def test_calculate_system_entropy_all_errors(self, engine):
        """Entropía con 100% errores."""
        entropy = engine.calculate_system_entropy(
            total_records=100,
            error_count=100,
            processing_time=1.0,
        )

        # Con todos errores, entropía de Shannon = 0 (un solo estado)
        assert entropy["shannon_entropy"] == 0.0

    def test_calculate_system_entropy_half_errors(self, engine):
        """Entropía máxima con 50% errores."""
        entropy = engine.calculate_system_entropy(
            total_records=100,
            error_count=50,
            processing_time=1.0,
        )

        # Entropía de Shannon máxima = 1 bit cuando p = 0.5
        assert math.isclose(entropy["shannon_entropy"], 1.0, abs_tol=0.01)

    def test_calculate_system_entropy_zero_records(self, engine):
        """Entropía con 0 registros retorna ceros."""
        entropy = engine.calculate_system_entropy(
            total_records=0,
            error_count=0,
            processing_time=1.0,
        )

        assert entropy["shannon_entropy"] == 0.0
        assert entropy["configurational_entropy"] == 0.0
        assert entropy["is_thermal_death"] is False

    def test_betti_numbers_connected_graph(self, engine):
        """
        Betti-0 = 1 para grafo conexo, Betti-1 = ciclos.
        """
        # Ejecutar métricas para construir grafo
        metrics = engine.calculate_metrics(
            total_records=100,
            cache_hits=50,
        )

        # El grafo de métricas debe existir
        betti_0 = metrics["betti_0"]
        betti_1 = metrics["betti_1"]

        # Betti-0 >= 1 (al menos un componente)
        assert betti_0 >= 1

        # Betti-1 >= 0 (puede o no haber ciclos)
        assert betti_1 >= 0

        # Verificar fórmula de Euler: β₁ = E - V + β₀
        V = metrics["graph_vertices"]
        E = metrics["graph_edges"]
        if V > 0:
            expected_betti_1 = max(0, E - V + betti_0)
            assert betti_1 == expected_betti_1

    def test_betti_numbers_empty_graph(self, engine):
        """Grafo vacío tiene Betti-0 = 0."""
        # Con 0 registros, no hay grafo
        metrics = engine.calculate_metrics(total_records=0, cache_hits=0)

        assert metrics["betti_0"] == 0
        assert metrics["betti_1"] == 0
        assert metrics["graph_vertices"] == 0
        assert metrics["graph_edges"] == 0

    def test_get_system_diagnosis_normal(self, engine):
        """Diagnóstico para estado normal."""
        metrics = {
            "saturation": 0.5,
            "damping_ratio": 1.0,
            "potential_energy": 100,
            "kinetic_energy": 100,
            "dissipated_power": 10,
            "entropy_ratio": 0.2,
            "is_thermal_death": False,
            "betti_0": 1,
            "betti_1": 0,
        }

        diagnosis = engine.get_system_diagnosis(metrics)

        assert diagnosis["state"] == "NORMAL"
        assert diagnosis["entropy"] == "LOW"

    def test_get_system_diagnosis_saturated(self, engine):
        """Diagnóstico para estado saturado."""
        metrics = {
            "saturation": 0.98,
            "damping_ratio": 1.0,
            "potential_energy": 100,
            "kinetic_energy": 100,
            "dissipated_power": 10,
            "entropy_ratio": 0.2,
            "is_thermal_death": False,
            "betti_0": 1,
            "betti_1": 0,
        }

        diagnosis = engine.get_system_diagnosis(metrics)
        assert diagnosis["state"] == "SATURATED"

    def test_get_system_diagnosis_idle(self, engine):
        """Diagnóstico para estado inactivo."""
        metrics = {
            "saturation": 0.01,
            "damping_ratio": 1.0,
            "potential_energy": 0,
            "kinetic_energy": 0,
            "dissipated_power": 0,
            "entropy_ratio": 0.0,
            "is_thermal_death": False,
            "betti_0": 0,
            "betti_1": 0,
        }

        diagnosis = engine.get_system_diagnosis(metrics)
        assert diagnosis["state"] == "IDLE"

    def test_get_system_diagnosis_overheating(self, engine):
        """Diagnóstico para sobrecalentamiento."""
        metrics = {
            "saturation": 0.5,
            "damping_ratio": 1.0,
            "potential_energy": 100,
            "kinetic_energy": 100,
            "dissipated_power": 100,  # > OVERHEAT_POWER_THRESHOLD (50)
            "entropy_ratio": 0.2,
            "is_thermal_death": False,
            "betti_0": 1,
            "betti_1": 0,
        }

        diagnosis = engine.get_system_diagnosis(metrics)
        assert diagnosis["state"] == "OVERHEATING"

    def test_get_system_diagnosis_thermal_death(self, engine):
        """Diagnóstico para muerte térmica."""
        metrics = {
            "saturation": 0.5,
            "damping_ratio": 1.0,
            "potential_energy": 100,
            "kinetic_energy": 100,
            "dissipated_power": 10,
            "entropy_ratio": 0.95,
            "is_thermal_death": True,
            "betti_0": 1,
            "betti_1": 0,
        }

        diagnosis = engine.get_system_diagnosis(metrics)
        assert diagnosis["state"] == "THERMAL_DEATH"
        assert diagnosis["entropy"] == "HIGH"

    def test_get_system_diagnosis_topology_disconnected(self, engine):
        """Diagnóstico topológico para grafo desconexo."""
        metrics = {
            "saturation": 0.5,
            "damping_ratio": 1.0,
            "potential_energy": 100,
            "kinetic_energy": 100,
            "dissipated_power": 10,
            "entropy_ratio": 0.2,
            "is_thermal_death": False,
            "betti_0": 3,  # Múltiples componentes
            "betti_1": 0,
        }

        diagnosis = engine.get_system_diagnosis(metrics)
        assert diagnosis["topology"] == "DISCONNECTED"

    def test_get_system_diagnosis_topology_cyclic(self, engine):
        """Diagnóstico topológico para grafo con ciclos."""
        metrics = {
            "saturation": 0.5,
            "damping_ratio": 1.0,
            "potential_energy": 100,
            "kinetic_energy": 100,
            "dissipated_power": 10,
            "entropy_ratio": 0.2,
            "is_thermal_death": False,
            "betti_0": 1,
            "betti_1": 2,  # Ciclos detectados
        }

        diagnosis = engine.get_system_diagnosis(metrics)
        assert diagnosis["topology"] == "CYCLIC"

    def test_get_trend_analysis_insufficient_data(self, engine):
        """Análisis de tendencia con datos insuficientes."""
        analysis = engine.get_trend_analysis()

        assert analysis["status"] == "INSUFFICIENT_DATA"
        assert analysis["samples"] == 0

    def test_get_trend_analysis_with_data(self, engine):
        """Análisis de tendencia con datos suficientes."""
        # Generar historial creciente
        for i in range(10):
            engine._store_metrics({
                "saturation": 0.1 * i,
                "dissipated_power": 100 - i * 10,
                "entropy_ratio": 0.05 * i,
            })

        analysis = engine.get_trend_analysis()

        assert analysis["status"] == "OK"
        assert analysis["samples"] == 10

        # Saturation creciente
        assert "saturation" in analysis
        assert analysis["saturation"]["trend"] == "INCREASING"

        # Power decreciente
        assert "dissipated_power" in analysis
        assert analysis["dissipated_power"]["trend"] == "DECREASING"

    def test_state_evolution_updates(self, engine):
        """El estado del sistema evoluciona con cada cálculo."""
        initial_state = engine._state.copy()

        engine.calculate_metrics(
            total_records=100,
            cache_hits=80,
            processing_time=0.1,
        )

        # El estado debe haber cambiado
        # (puede ser el mismo si dt es muy pequeño, pero el historial crece)
        assert len(engine._state_history) >= 1


# ==================== TESTS: DataFluxCondenser ====================


class TestDataFluxCondenserInitialization:
    """Pruebas de inicialización del condensador."""

    def test_initialization_with_defaults(self, valid_config, valid_profile):
        """Inicialización con configuración por defecto."""
        condenser = DataFluxCondenser(valid_config, valid_profile)

        assert condenser.config == valid_config
        assert condenser.profile == valid_profile
        assert condenser.condenser_config is not None
        assert condenser.physics is not None
        assert condenser.controller is not None

    def test_initialization_with_custom_config(self, valid_config, valid_profile):
        """Inicialización con configuración personalizada."""
        custom_config = CondenserConfig(
            pid_kp=500.0,
            pid_ki=25.0,
            pid_setpoint=0.4,
        )

        condenser = DataFluxCondenser(
            valid_config,
            valid_profile,
            condenser_config=custom_config,
        )

        assert condenser.condenser_config.pid_kp == 500.0
        assert condenser.condenser_config.pid_setpoint == 0.4

    def test_initialization_with_empty_config(self):
        """Inicialización con configuraciones vacías."""
        condenser = DataFluxCondenser({}, {})

        assert condenser.config == {}
        assert condenser.profile == {}

    def test_initialization_with_none_config(self):
        """Inicialización con None como configuraciones."""
        condenser = DataFluxCondenser(None, None)

        assert condenser.config == {}
        assert condenser.profile == {}


class TestDataFluxCondenserValidation:
    """Pruebas de validación de entrada."""

    def test_validate_input_file_valid(self, condenser, mock_csv_file):
        """Archivo válido debe pasar validación."""
        path = condenser._validate_input_file(str(mock_csv_file))
        assert path.exists()
        assert path.is_file()

    def test_validate_input_file_not_exists(self, condenser, tmp_path):
        """Archivo inexistente debe fallar."""
        fake_path = tmp_path / "nonexistent.csv"

        with pytest.raises(InvalidInputError, match="no existe"):
            condenser._validate_input_file(str(fake_path))

    def test_validate_input_file_is_directory(self, condenser, tmp_path):
        """Directorio en lugar de archivo debe fallar."""
        with pytest.raises(InvalidInputError, match="no es un archivo"):
            condenser._validate_input_file(str(tmp_path))

    def test_validate_input_file_invalid_extension(self, condenser, tmp_path):
        """Extensión inválida debe fallar."""
        invalid_file = tmp_path / "data.xyz"
        invalid_file.write_text("data")

        with pytest.raises(InvalidInputError, match="Extensión no soportada"):
            condenser._validate_input_file(str(invalid_file))

    def test_validate_input_file_too_small(self, condenser, tmp_path):
        """Archivo muy pequeño debe fallar."""
        tiny_file = tmp_path / "tiny.csv"
        tiny_file.write_text("a")  # Menos de MIN_FILE_SIZE_BYTES

        with pytest.raises(InvalidInputError, match="muy pequeño"):
            condenser._validate_input_file(str(tiny_file))

    def test_stabilize_empty_path_raises_error(self, condenser):
        """Ruta vacía debe lanzar InvalidInputError."""
        with pytest.raises(InvalidInputError, match="requerido"):
            condenser.stabilize("")

    def test_stabilize_none_path_raises_error(self, condenser):
        """Ruta None debe lanzar InvalidInputError."""
        with pytest.raises(InvalidInputError):
            condenser.stabilize(None)


class TestDataFluxCondenserProcessing:
    """Pruebas del procesamiento principal."""

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_stabilize_processes_all_records(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache,
    ):
        """Debe procesar todos los registros."""
        # Increase batch size to process all in one go to match mock behavior
        condenser.condenser_config = CondenserConfig(min_batch_size=200)

        # Configurar mocks
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = pd.DataFrame(
            [{"result": i} for i in range(len(sample_raw_records))]
        )
        mock_processor_class.return_value = mock_processor

        result = condenser.stabilize(str(mock_csv_file))

        assert len(result) == 100
        assert mock_parser.parse_to_raw.called
        assert mock_processor.process_all.called

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_stabilize_handles_empty_records(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
    ):
        """Debe manejar registros vacíos correctamente."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = []
        mock_parser_class.return_value = mock_parser

        result = condenser.stabilize(str(mock_csv_file))

        assert result.empty
        assert not mock_processor_class.called

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_stabilize_with_progress_callback(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache,
    ):
        """Debe invocar callback de progreso."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = pd.DataFrame([{"a": 1}])
        mock_processor_class.return_value = mock_processor

        progress_calls = []

        def on_progress(stats: ProcessingStats):
            progress_calls.append(stats)

        condenser.stabilize(str(mock_csv_file), on_progress=on_progress)

        assert len(progress_calls) > 0
        assert all(isinstance(s, ProcessingStats) for s in progress_calls)

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_stabilize_with_metrics_callback(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache,
    ):
        """Debe invocar callback de métricas físicas."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = pd.DataFrame([{"a": 1}])
        mock_processor_class.return_value = mock_processor

        metrics_calls = []

        def progress_callback(metrics: Dict[str, Any]):
            metrics_calls.append(metrics)

        condenser.stabilize(
            str(mock_csv_file),
            progress_callback=progress_callback,
        )

        assert len(metrics_calls) > 0
        # Verificar que las métricas tienen las claves esperadas
        assert "saturation" in metrics_calls[0]

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_stabilize_with_telemetry(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache,
    ):
        """Debe registrar eventos en telemetría."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = pd.DataFrame([{"a": 1}])
        mock_processor_class.return_value = mock_processor

        mock_telemetry = Mock()

        condenser.stabilize(str(mock_csv_file), telemetry=mock_telemetry)

        # Debe llamar record_event al menos para start y complete
        assert mock_telemetry.record_event.call_count >= 2

        # Verificar eventos esperados
        call_args = [call[0][0] for call in mock_telemetry.record_event.call_args_list]
        assert "stabilization_start" in call_args
        assert "stabilization_complete" in call_args

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_thermal_breaker_triggers_on_high_power(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache,
        caplog,
    ):
        """Freno de emergencia se activa con alta potencia disipada."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = pd.DataFrame([{"a": 1}])
        mock_processor_class.return_value = mock_processor

        # Usar threshold muy bajo para garantizar disparo
        with patch.object(
            SystemConstants, "OVERHEAT_POWER_THRESHOLD", 0.001
        ):
            with caplog.at_level(logging.WARNING):
                condenser.stabilize(str(mock_csv_file))

            assert "OVERHEAT" in caplog.text
            assert condenser._emergency_brake_count > 0

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_stabilize_updates_statistics(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache,
    ):
        """Debe actualizar estadísticas de procesamiento."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = pd.DataFrame(
            [{"a": 1}] * 10
        )
        mock_processor_class.return_value = mock_processor

        condenser.stabilize(str(mock_csv_file))

        stats = condenser.get_processing_stats()

        assert "statistics" in stats
        assert stats["statistics"]["total_records"] == 100
        assert stats["statistics"]["processing_time"] > 0


class TestDataFluxCondenserBatchProcessing:
    """Pruebas del procesamiento por lotes."""

    def test_process_single_batch_success(
        self, condenser, sample_raw_records, sample_parse_cache
    ):
        """Batch exitoso retorna BatchResult correcto."""
        with patch.object(condenser, "_rectify_signal") as mock_rectify:
            mock_rectify.return_value = pd.DataFrame([{"a": 1}, {"a": 2}])

            result = condenser._process_single_batch(
                sample_raw_records[:10],
                sample_parse_cache,
            )

            assert isinstance(result, BatchResult)
            assert result.success is True
            assert result.records_processed == 2
            assert result.dataframe is not None
            assert len(result.dataframe) == 2

    def test_process_single_batch_empty(
        self, condenser, sample_parse_cache
    ):
        """Batch vacío retorna fallo."""
        result = condenser._process_single_batch([], sample_parse_cache)

        assert result.success is False
        assert "vacío" in result.error_message.lower()

    def test_process_single_batch_error_handling(
        self, condenser, sample_raw_records, sample_parse_cache
    ):
        """Errores en procesamiento son capturados."""
        with patch.object(condenser, "_rectify_signal") as mock_rectify:
            mock_rectify.side_effect = Exception("Error de prueba")

            result = condenser._process_single_batch(
                sample_raw_records[:10],
                sample_parse_cache,
            )

            assert result.success is False
            assert "Error de prueba" in result.error_message

    def test_consolidate_results_empty(self, condenser):
        """Consolidar lista vacía retorna DataFrame vacío."""
        result = condenser._consolidate_results([])
        assert result.empty

    def test_consolidate_results_filters_empty_dfs(self, condenser):
        """Consolida correctamente filtrando DataFrames vacíos."""
        batches = [
            pd.DataFrame([{"a": 1}]),
            pd.DataFrame(),
            pd.DataFrame([{"a": 2}]),
            None,
        ]

        result = condenser._consolidate_results(batches)

        assert len(result) == 2
        assert list(result["a"]) == [1, 2]

    def test_consolidate_results_respects_limit(self, condenser):
        """Respeta el límite máximo de batches."""
        # Crear más batches que el límite
        with patch.object(
            SystemConstants, "MAX_BATCHES_TO_CONSOLIDATE", 5
        ):
            batches = [
                pd.DataFrame([{"a": i}]) for i in range(10)
            ]

            result = condenser._consolidate_results(batches)

            # Solo debe consolidar los primeros 5
            assert len(result) == 5


class TestDataFluxCondenserHealthMonitoring:
    """Pruebas del monitoreo de salud del sistema."""

    def test_get_processing_stats_structure(self, condenser):
        """Verifica estructura de estadísticas."""
        stats = condenser.get_processing_stats()

        assert "statistics" in stats
        assert "controller" in stats
        assert "physics" in stats
        assert "emergency_brakes" in stats

    def test_get_system_health_healthy(self, condenser):
        """Sistema saludable sin problemas."""
        # Simular estado saludable
        condenser._start_time = time.time()
        condenser._emergency_brake_count = 0
        condenser._stats.total_records = 100
        condenser._stats.processed_records = 100
        condenser._stats.failed_batches = 0
        condenser._stats.total_batches = 10

        health = condenser.get_system_health()

        assert health["health"] == "HEALTHY"
        assert len(health["issues"]) == 0
        assert health["processing_efficiency"] == 1.0

    def test_get_system_health_degraded_by_emergency_brakes(self, condenser):
        """Sistema degradado por múltiples frenos de emergencia."""
        condenser._start_time = time.time()
        condenser._emergency_brake_count = 10  # > 5

        health = condenser.get_system_health()

        assert health["health"] == "DEGRADED"
        assert any("frenos de emergencia" in issue for issue in health["issues"])

    def test_get_system_health_degraded_by_failed_batches(self, condenser):
        """Sistema degradado por alta tasa de fallos."""
        condenser._start_time = time.time()
        condenser._emergency_brake_count = 0
        condenser._stats.total_batches = 100
        condenser._stats.failed_batches = 20  # 20% > 10%

        health = condenser.get_system_health()

        assert health["health"] == "DEGRADED"
        assert any("tasa de fallos" in issue for issue in health["issues"])


class TestDataFluxCondenserErrorHandling:
    """Pruebas de manejo de errores."""

    @patch("app.flux_condenser.ReportParserCrudo")
    def test_parser_error_raises_processing_error(
        self, mock_parser_class, condenser, mock_csv_file
    ):
        """Error en parser se propaga como ProcessingError."""
        mock_parser_class.side_effect = Exception("Parser failed")

        with pytest.raises(ProcessingError, match="inicializando parser"):
            condenser.stabilize(str(mock_csv_file))

    @patch("app.flux_condenser.ReportParserCrudo")
    def test_parse_to_raw_error_raises_processing_error(
        self, mock_parser_class, condenser, mock_csv_file
    ):
        """Error en parse_to_raw se propaga como ProcessingError."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.side_effect = Exception("Parse failed")
        mock_parser_class.return_value = mock_parser

        with pytest.raises(ProcessingError, match="extrayendo datos"):
            condenser.stabilize(str(mock_csv_file))

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_max_failed_batches_raises_error(
        self,
        mock_parser_class,
        mock_processor_class,
        valid_config,
        valid_profile,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache,
    ):
        """Exceder máximo de batches fallidos lanza error."""
        config = CondenserConfig(
            max_failed_batches=2,
            enable_partial_recovery=False,
        )
        condenser = DataFluxCondenser(valid_config, valid_profile, config)

        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        # Hacer que el procesador siempre falle
        mock_processor = Mock()
        mock_processor.process_all.side_effect = Exception("Always fails")
        mock_processor_class.return_value = mock_processor

        with pytest.raises(ProcessingError, match="batches fallidos"):
            condenser.stabilize(str(mock_csv_file))

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_partial_recovery_continues_on_failures(
        self,
        mock_parser_class,
        mock_processor_class,
        valid_config,
        valid_profile,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache,
        caplog,
    ):
        """Con recuperación parcial, continúa después de fallos."""
        config = CondenserConfig(
            max_failed_batches=1,
            enable_partial_recovery=True,
        )
        condenser = DataFluxCondenser(valid_config, valid_profile, config)

        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        # Alternar entre éxito y fallo
        call_count = [0]

        def alternating_process():
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise Exception("Simulated failure")
            return pd.DataFrame([{"a": 1}])

        mock_processor = Mock()
        mock_processor.process_all.side_effect = alternating_process
        mock_processor_class.return_value = mock_processor

        with caplog.at_level(logging.WARNING):
            result = condenser.stabilize(str(mock_csv_file))

        # Debe completar con resultados parciales
        assert not result.empty
        assert "recuperación parcial" in caplog.text

    @patch("app.flux_condenser.APUProcessor")
    @patch("app.flux_condenser.ReportParserCrudo")
    def test_telemetry_records_error_on_failure(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
    ):
        """Telemetría registra errores."""
        mock_parser_class.side_effect = Exception("Parser failed")
        mock_telemetry = Mock()

        with pytest.raises(ProcessingError):
            condenser.stabilize(str(mock_csv_file), telemetry=mock_telemetry)

        # Verificar que se registró el error
        error_calls = [
            call for call in mock_telemetry.record_event.call_args_list
            if call[0][0] == "stabilization_error"
        ]
        assert len(error_calls) == 1


class TestParsedDataAndBatchResult:
    """Pruebas para estructuras de datos auxiliares."""

    def test_parsed_data_is_immutable(self):
        """ParsedData debe ser inmutable (NamedTuple)."""
        data = ParsedData(
            raw_records=[{"a": 1}],
            parse_cache={"key": "value"},
        )

        assert data.raw_records == [{"a": 1}]
        assert data.parse_cache == {"key": "value"}

        # NamedTuple no permite asignación
        with pytest.raises(AttributeError):
            data.raw_records = []

    def test_batch_result_default_values(self):
        """BatchResult tiene valores por defecto correctos."""
        result = BatchResult(success=True)

        assert result.success is True
        assert result.dataframe is None
        assert result.records_processed == 0
        assert result.error_message == ""
        assert result.metrics is None

    def test_batch_result_with_all_fields(self):
        """BatchResult acepta todos los campos."""
        df = pd.DataFrame([{"a": 1}])
        result = BatchResult(
            success=True,
            dataframe=df,
            records_processed=1,
            error_message="",
            metrics={"saturation": 0.5},
        )

        assert result.success is True
        assert len(result.dataframe) == 1
        assert result.records_processed == 1
        assert result.metrics["saturation"] == 0.5


class TestProcessingStats:
    """Pruebas para estadísticas de procesamiento."""

    def test_add_batch_stats_success(self):
        """Agregar estadísticas de batch exitoso."""
        stats = ProcessingStats()
        stats.total_records = 100

        stats.add_batch_stats(
            batch_size=10,
            saturation=0.5,
            power=15.0,
            flyback=0.1,
            kinetic=0.5,
            success=True,
        )

        assert stats.total_batches == 1
        assert stats.processed_records == 10
        assert stats.failed_records == 0
        assert stats.avg_batch_size == 10.0
        assert stats.avg_saturation == 0.5
        assert stats.max_dissipated_power == 15.0

    def test_add_batch_stats_failure(self):
        """Agregar estadísticas de batch fallido."""
        stats = ProcessingStats()

        stats.add_batch_stats(
            batch_size=10,
            saturation=0.8,
            power=50.0,
            flyback=0.5,
            kinetic=1.0,
            success=False,
        )

        assert stats.total_batches == 1
        assert stats.processed_records == 0
        assert stats.failed_records == 10
        assert stats.failed_batches == 1

    def test_add_batch_stats_running_averages(self):
        """Promedios se calculan correctamente."""
        stats = ProcessingStats()

        stats.add_batch_stats(10, 0.2, 10, 0.1, 0.5, True)
        stats.add_batch_stats(20, 0.4, 20, 0.2, 1.0, True)
        stats.add_batch_stats(30, 0.6, 30, 0.3, 1.5, True)

        assert stats.total_batches == 3
        assert math.isclose(stats.avg_batch_size, 20.0, abs_tol=0.1)
        assert math.isclose(stats.avg_saturation, 0.4, abs_tol=0.01)
        assert stats.max_dissipated_power == 30.0
        assert stats.max_flyback_voltage == 0.3


# ==================== TESTS: Edge Cases ====================


class TestEdgeCases:
    """Pruebas de casos límite y condiciones extremas."""

    def test_very_large_batch_count(self, condenser):
        """Manejo de gran cantidad de batches."""
        batches = [pd.DataFrame([{"a": i}]) for i in range(1000)]
        result = condenser._consolidate_results(batches)
        assert len(result) == 1000

    def test_physics_engine_extreme_values(self):
        """Motor de física con valores extremos."""
        engine = FluxPhysicsEngine(
            capacitance=1e6,  # Muy grande
            resistance=0.001,  # Muy pequeña
            inductance=100.0,  # Grande
        )

        metrics = engine.calculate_metrics(
            total_records=1000000,
            cache_hits=500000,
        )

        # Debe retornar valores finitos
        assert math.isfinite(metrics["saturation"])
        assert math.isfinite(metrics["kinetic_energy"])
        assert math.isfinite(metrics["damping_ratio"])

    def test_controller_rapid_oscillation(self):
        """Controlador con entrada oscilante."""
        controller = PIController(
            kp=100.0,
            ki=10.0,
            setpoint=0.5,
            min_output=10,
            max_output=100,
        )

        outputs = []
        for i in range(20):
            # Alternar entre valores extremos
            pv = 0.1 if i % 2 == 0 else 0.9
            output = controller.compute(pv)
            outputs.append(output)

        # El EMA debe suavizar las oscilaciones
        # Los outputs no deberían oscilar tan violentamente como la entrada
        output_variance = sum((o - sum(outputs)/len(outputs))**2 for o in outputs) / len(outputs)

        # La varianza de salida debe ser menor que la de entrada
        # (entrada oscila entre 10 y 100 equivalentes, ~2000 varianza)
        assert output_variance < 2000

    def test_controller_long_running_stability(self):
        """Controlador mantiene estabilidad en ejecución larga."""
        controller = PIController(
            kp=50.0,
            ki=5.0,
            setpoint=0.5,
            min_output=10,
            max_output=100,
        )

        # Simular convergencia
        pv = 0.1
        for _ in range(100):
            output = controller.compute(pv)
            # Simular respuesta del sistema
            pv = pv + (0.5 - pv) * 0.1
            time.sleep(0.001)

        # Verificar estabilidad
        analysis = controller.get_stability_analysis()
        assert analysis["status"] == "OPERATIONAL"
        # El sistema debería estar convergiendo o estable
        assert analysis["stability_class"] in [
            "ASYMPTOTICALLY_STABLE",
            "MARGINALLY_STABLE",
        ]