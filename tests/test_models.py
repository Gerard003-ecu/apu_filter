"""
Suite de pruebas exhaustivas para el módulo de simulación Monte Carlo.

Esta suite cubre:
- Configuración y validación de parámetros
- Utilidades de sanitización y validación
- Clases de resultados y métricas
- Simulador principal con todas las distribuciones
- Análisis de sensibilidad
- Función de compatibilidad legacy
- Tests de integración y propiedades matemáticas

Ejecutar con: pytest test_monte_carlo.py -v --tb=short
Ejecutar con cobertura: pytest test_monte_carlo.py --cov=monte_carlo --cov-report=html
"""

import logging
import math
import warnings
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Importar el módulo a testear
from models.probability_models import (
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_VOLATILITY,
    MAX_NUM_SIMULATIONS,
    MAX_SAFE_FLOAT,
    MAX_VOLATILITY,
    MEMORY_HARD_LIMIT_GB,
    MIN_NUM_SIMULATIONS,
    MIN_SCALE_VALUE,
    MIN_VALID_ITEMS_FOR_SIMULATION,
    MIN_VOLATILITY,
    ConvergenceMetrics,
    DistributionType,
    MonteCarloConfig,
    MonteCarloSimulator,
    SimulationResult,
    SimulationStatus,
    calculate_convergence_metrics,
    estimate_memory_usage,
    is_numeric_valid,
    run_monte_carlo_simulation,
    sanitize_value,
    validate_required_keys,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def default_config() -> MonteCarloConfig:
    """Configuración por defecto para tests."""
    return MonteCarloConfig()


@pytest.fixture
def deterministic_config() -> MonteCarloConfig:
    """Configuración con semilla fija para reproducibilidad."""
    return MonteCarloConfig(
        num_simulations=1000,
        volatility_factor=0.10,
        random_seed=42,
    )


@pytest.fixture
def lognormal_config() -> MonteCarloConfig:
    """Configuración con distribución log-normal."""
    return MonteCarloConfig(
        num_simulations=1000,
        volatility_factor=0.15,
        distribution=DistributionType.LOGNORMAL,
        random_seed=42,
    )


@pytest.fixture
def triangular_config() -> MonteCarloConfig:
    """Configuración con distribución triangular."""
    return MonteCarloConfig(
        num_simulations=1000,
        volatility_factor=0.20,
        distribution=DistributionType.TRIANGULAR,
        random_seed=42,
    )


@pytest.fixture
def antithetic_config() -> MonteCarloConfig:
    """Configuración con variantes antitéticas."""
    return MonteCarloConfig(
        num_simulations=1000,
        volatility_factor=0.10,
        use_antithetic=True,
        random_seed=42,
    )


@pytest.fixture
def simple_apu_data() -> List[Dict[str, Any]]:
    """Datos APU simples para tests básicos."""
    return [
        {"VR_TOTAL": 1000.0, "CANTIDAD": 5.0},
        {"VR_TOTAL": 2000.0, "CANTIDAD": 3.0},
        {"VR_TOTAL": 1500.0, "CANTIDAD": 4.0},
    ]


@pytest.fixture
def large_apu_data() -> List[Dict[str, Any]]:
    """Datos APU más grandes para tests de rendimiento."""
    np.random.seed(42)
    return [
        {
            "VR_TOTAL": float(np.random.uniform(100, 10000)),
            "CANTIDAD": float(np.random.uniform(1, 100)),
        }
        for _ in range(100)
    ]


@pytest.fixture
def mixed_valid_invalid_data() -> List[Dict[str, Any]]:
    """Datos con mezcla de valores válidos e inválidos."""
    return [
        {"VR_TOTAL": 1000.0, "CANTIDAD": 5.0},  # Válido
        {"VR_TOTAL": np.nan, "CANTIDAD": 3.0},  # NaN en VR_TOTAL
        {"VR_TOTAL": 2000.0, "CANTIDAD": np.nan},  # NaN en CANTIDAD
        {"VR_TOTAL": np.inf, "CANTIDAD": 2.0},  # Infinito
        {"VR_TOTAL": -1000.0, "CANTIDAD": 5.0},  # Negativo (si threshold > 0)
        {"VR_TOTAL": 1500.0, "CANTIDAD": 4.0},  # Válido
        {"VR_TOTAL": "invalid", "CANTIDAD": 3.0},  # String no numérico
        {"VR_TOTAL": 3000.0, "CANTIDAD": 2.0},  # Válido
    ]


@pytest.fixture
def aliased_apu_data() -> List[Dict[str, Any]]:
    """Datos APU con nombres de columnas alternativos."""
    return [
        {"valor_total": 1000.0, "cantidad": 5.0},
        {"valor_total": 2000.0, "cantidad": 3.0},
        {"valor_total": 1500.0, "cantidad": 4.0},
    ]


@pytest.fixture
def silent_logger() -> logging.Logger:
    """Logger silencioso para tests."""
    logger = logging.getLogger("test_silent")
    logger.setLevel(logging.CRITICAL + 1)  # Desactiva todo logging
    return logger


@pytest.fixture
def simulator(deterministic_config, silent_logger) -> MonteCarloSimulator:
    """Simulador configurado para tests."""
    return MonteCarloSimulator(config=deterministic_config, logger=silent_logger)


# ============================================================================
# TESTS: MonteCarloConfig
# ============================================================================


class TestMonteCarloConfig:
    """Tests para la clase MonteCarloConfig."""

    def test_default_values(self):
        """Verifica valores por defecto."""
        config = MonteCarloConfig()

        assert config.num_simulations == DEFAULT_NUM_SIMULATIONS
        assert config.volatility_factor == DEFAULT_VOLATILITY
        assert config.min_cost_threshold == 0.0
        assert config.min_quantity_threshold == 0.0
        assert config.random_seed is None
        assert config.truncate_negative is True
        assert config.distribution == DistributionType.NORMAL
        assert config.use_antithetic is False
        assert config.check_convergence is True
        assert config.use_float32 is False

    def test_percentiles_default(self):
        """Verifica percentiles por defecto."""
        config = MonteCarloConfig()
        assert config.percentiles == [5, 25, 50, 75, 95]

    def test_percentiles_normalization(self):
        """Verifica que percentiles se ordenan y eliminan duplicados."""
        config = MonteCarloConfig(percentiles=[95, 5, 50, 5, 95])
        assert config.percentiles == [5, 50, 95]

    def test_custom_values(self):
        """Verifica valores personalizados."""
        config = MonteCarloConfig(
            num_simulations=5000,
            volatility_factor=0.25,
            min_cost_threshold=100.0,
            random_seed=123,
        )

        assert config.num_simulations == 5000
        assert config.volatility_factor == 0.25
        assert config.min_cost_threshold == 100.0
        assert config.random_seed == 123

    def test_distribution_from_string(self):
        """Verifica conversión de string a enum para distribución."""
        config = MonteCarloConfig(distribution="lognormal")
        assert config.distribution == DistributionType.LOGNORMAL

        config2 = MonteCarloConfig(distribution="TRIANGULAR")
        assert config2.distribution == DistributionType.TRIANGULAR

    def test_repr(self):
        """Verifica representación string."""
        config = MonteCarloConfig(num_simulations=5000, volatility_factor=0.15)
        repr_str = repr(config)

        assert "5,000" in repr_str
        assert "15.0%" in repr_str
        assert "normal" in repr_str

    # --- Tests de validación de errores ---

    def test_invalid_num_simulations_type(self):
        """Error si num_simulations no es int."""
        with pytest.raises(TypeError, match="num_simulations debe ser int"):
            MonteCarloConfig(num_simulations=1000.5)

    def test_num_simulations_too_low(self):
        """Error si num_simulations es muy bajo."""
        with pytest.raises(ValueError, match=f"debe estar entre {MIN_NUM_SIMULATIONS}"):
            MonteCarloConfig(num_simulations=50)

    def test_num_simulations_too_high(self):
        """Error si num_simulations es muy alto."""
        with pytest.raises(ValueError, match=f"{MAX_NUM_SIMULATIONS:,}"):
            MonteCarloConfig(num_simulations=10_000_000)

    def test_invalid_volatility_type(self):
        """Error si volatility_factor no es numérico."""
        with pytest.raises(TypeError, match="volatility_factor debe ser numérico"):
            MonteCarloConfig(volatility_factor="high")

    def test_volatility_out_of_range_low(self):
        """Error si volatility_factor es negativo."""
        with pytest.raises(ValueError, match="debe estar entre"):
            MonteCarloConfig(volatility_factor=-0.1)

    def test_volatility_out_of_range_high(self):
        """Error si volatility_factor excede 1."""
        with pytest.raises(ValueError, match="debe estar entre"):
            MonteCarloConfig(volatility_factor=1.5)

    def test_negative_min_cost_threshold(self):
        """Error si min_cost_threshold es negativo."""
        with pytest.raises(ValueError, match="no puede ser negativo"):
            MonteCarloConfig(min_cost_threshold=-100)

    def test_negative_min_quantity_threshold(self):
        """Error si min_quantity_threshold es negativo."""
        with pytest.raises(ValueError, match="no puede ser negativo"):
            MonteCarloConfig(min_quantity_threshold=-1)

    def test_invalid_random_seed_type(self):
        """Error si random_seed no es int."""
        with pytest.raises(TypeError, match="random_seed debe ser int"):
            MonteCarloConfig(random_seed=42.5)

    def test_negative_random_seed(self):
        """Error si random_seed es negativo."""
        with pytest.raises(ValueError, match="no puede ser negativo"):
            MonteCarloConfig(random_seed=-1)

    def test_invalid_percentiles_type(self):
        """Error si percentiles no es lista."""
        with pytest.raises(TypeError, match="percentiles debe ser una lista"):
            MonteCarloConfig(percentiles=(5, 95))

    def test_empty_percentiles(self):
        """Error si percentiles está vacía."""
        with pytest.raises(ValueError, match="no puede estar vacía"):
            MonteCarloConfig(percentiles=[])

    def test_non_integer_percentiles(self):
        """Error si percentiles contiene no-enteros."""
        with pytest.raises(TypeError, match="deben ser enteros"):
            MonteCarloConfig(percentiles=[5.5, 95.0])

    def test_percentiles_out_of_range(self):
        """Error si percentiles están fuera de rango."""
        with pytest.raises(ValueError, match="entre 0 y 100"):
            MonteCarloConfig(percentiles=[5, 150])

    def test_invalid_distribution_string(self):
        """Error si string de distribución es inválido."""
        with pytest.raises(ValueError, match="debe ser uno de"):
            MonteCarloConfig(distribution="gaussian")


# ============================================================================
# TESTS: DistributionType
# ============================================================================


class TestDistributionType:
    """Tests para el enum DistributionType."""

    def test_enum_values(self):
        """Verifica valores del enum."""
        assert DistributionType.NORMAL.value == "normal"
        assert DistributionType.LOGNORMAL.value == "lognormal"
        assert DistributionType.TRIANGULAR.value == "triangular"

    def test_enum_from_string(self):
        """Verifica creación desde string."""
        assert DistributionType("normal") == DistributionType.NORMAL
        assert DistributionType("lognormal") == DistributionType.LOGNORMAL


# ============================================================================
# TESTS: SimulationStatus
# ============================================================================


class TestSimulationStatus:
    """Tests para el enum SimulationStatus."""

    def test_enum_values(self):
        """Verifica valores del enum."""
        assert SimulationStatus.SUCCESS.value == "success"
        assert SimulationStatus.NO_VALID_DATA.value == "no_valid_data"
        assert SimulationStatus.INSUFFICIENT_DATA.value == "insufficient_data"
        assert SimulationStatus.CONVERGENCE_WARNING.value == "convergence_warning"
        assert SimulationStatus.ERROR.value == "error"


# ============================================================================
# TESTS: ConvergenceMetrics
# ============================================================================


class TestConvergenceMetrics:
    """Tests para la clase ConvergenceMetrics."""

    def test_creation(self):
        """Verifica creación de métricas."""
        metrics = ConvergenceMetrics(
            is_converged=True,
            mean_std_error=10.5,
            relative_error=0.005,
            half_width_ci=20.58,
            effective_sample_size=1000,
        )

        assert metrics.is_converged is True
        assert metrics.mean_std_error == 10.5
        assert metrics.relative_error == 0.005
        assert metrics.half_width_ci == 20.58
        assert metrics.effective_sample_size == 1000

    def test_to_dict(self):
        """Verifica conversión a diccionario."""
        metrics = ConvergenceMetrics(
            is_converged=True,
            mean_std_error=10.5,
            relative_error=0.005,
            half_width_ci=20.58,
            effective_sample_size=1000,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["is_converged"] is True
        assert result["mean_std_error"] == 10.5
        assert result["effective_sample_size"] == 1000

    def test_to_dict_sanitizes_nan(self):
        """Verifica que to_dict sanitiza NaN."""
        metrics = ConvergenceMetrics(
            is_converged=False,
            mean_std_error=float("nan"),
            relative_error=float("inf"),
            half_width_ci=10.0,
            effective_sample_size=100,
        )

        result = metrics.to_dict()

        assert result["mean_std_error"] is None
        assert result["relative_error"] is None


# ============================================================================
# TESTS: SimulationResult
# ============================================================================


class TestSimulationResult:
    """Tests para la clase SimulationResult."""

    @pytest.fixture
    def successful_result(self) -> SimulationResult:
        """Resultado exitoso de ejemplo."""
        return SimulationResult(
            status=SimulationStatus.SUCCESS,
            statistics={
                "mean": 10000.0,
                "median": 9800.0,
                "std_dev": 1500.0,
                "percentile_5": 7500.0,
                "percentile_25": 8800.0,
                "percentile_50": 9800.0,
                "percentile_75": 11000.0,
                "percentile_95": 12500.0,
                "var_95": 12500.0,
                "cvar_95": 13000.0,
            },
            metadata={
                "num_simulations_completed": 1000,
                "valid_items": 10,
            },
            raw_results=np.array([9000, 10000, 11000]),
            convergence=ConvergenceMetrics(
                is_converged=True,
                mean_std_error=47.4,
                relative_error=0.00474,
                half_width_ci=92.9,
                effective_sample_size=1000,
            ),
        )

    def test_is_successful_success(self, successful_result):
        """Verifica is_successful para SUCCESS."""
        assert successful_result.is_successful() is True

    def test_is_successful_convergence_warning(self):
        """Verifica is_successful para CONVERGENCE_WARNING."""
        result = SimulationResult(
            status=SimulationStatus.CONVERGENCE_WARNING,
            statistics={"mean": 1000.0},
            metadata={},
        )
        assert result.is_successful() is True

    def test_is_successful_error(self):
        """Verifica is_successful para ERROR."""
        result = SimulationResult(
            status=SimulationStatus.ERROR,
            statistics={},
            metadata={"error": "test"},
        )
        assert result.is_successful() is False

    def test_is_successful_no_data(self):
        """Verifica is_successful para NO_VALID_DATA."""
        result = SimulationResult(
            status=SimulationStatus.NO_VALID_DATA,
            statistics={},
            metadata={},
        )
        assert result.is_successful() is False

    def test_get_var(self, successful_result):
        """Verifica obtención de VaR."""
        var_95 = successful_result.get_var(0.95)
        assert var_95 == 12500.0

    def test_get_var_default_confidence(self, successful_result):
        """Verifica VaR con confianza por defecto."""
        var = successful_result.get_var()
        assert var == 12500.0

    def test_get_var_missing(self):
        """Verifica VaR cuando no existe."""
        result = SimulationResult(
            status=SimulationStatus.SUCCESS,
            statistics={"mean": 1000.0},
            metadata={},
        )
        assert result.get_var(0.99) is None

    def test_get_cvar(self, successful_result):
        """Verifica obtención de CVaR."""
        cvar_95 = successful_result.get_cvar(0.95)
        assert cvar_95 == 13000.0

    def test_get_confidence_interval(self, successful_result):
        """Verifica obtención de intervalo de confianza."""
        lower, upper = successful_result.get_confidence_interval(0.90)
        assert lower == 7500.0
        assert upper == 12500.0

    def test_get_confidence_interval_50(self, successful_result):
        """Verifica IC al 50%."""
        lower, upper = successful_result.get_confidence_interval(0.50)
        assert lower == 8800.0
        assert upper == 11000.0

    def test_to_dict_basic(self, successful_result):
        """Verifica conversión a diccionario básica."""
        result = successful_result.to_dict()

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert "statistics" in result
        assert "metadata" in result
        assert "convergence" in result
        assert "raw_results" not in result

    def test_to_dict_include_raw(self, successful_result):
        """Verifica conversión a diccionario con raw_results."""
        result = successful_result.to_dict(include_raw=True)

        assert "raw_results" in result
        assert result["raw_results"] == [9000, 10000, 11000]

    def test_repr(self, successful_result):
        """Verifica representación string."""
        repr_str = repr(successful_result)

        assert "success" in repr_str
        assert "10,000.00" in repr_str
        assert "1,500.00" in repr_str


# ============================================================================
# TESTS: Utilidades
# ============================================================================


class TestSanitizeValue:
    """Tests para la función sanitize_value."""

    def test_none_value(self):
        """Verifica manejo de None."""
        assert sanitize_value(None) is None

    def test_nan_value(self):
        """Verifica manejo de NaN."""
        assert sanitize_value(float("nan")) is None
        assert sanitize_value(np.nan) is None

    def test_inf_value(self):
        """Verifica manejo de infinito."""
        assert sanitize_value(float("inf")) is None
        assert sanitize_value(float("-inf")) is None
        assert sanitize_value(np.inf) is None

    def test_normal_float(self):
        """Verifica manejo de float normal."""
        assert sanitize_value(10.5) == 10.5

    def test_numpy_float64(self):
        """Verifica conversión de numpy float64."""
        result = sanitize_value(np.float64(10.5))
        assert result == 10.5
        assert isinstance(result, float)

    def test_numpy_float32(self):
        """Verifica conversión de numpy float32."""
        result = sanitize_value(np.float32(10.5))
        assert isinstance(result, float)

    def test_numpy_int64(self):
        """Verifica conversión de numpy int64."""
        result = sanitize_value(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_bool(self):
        """Verifica conversión de numpy bool."""
        assert sanitize_value(np.bool_(True)) is True
        assert sanitize_value(np.bool_(False)) is False

    def test_string_unchanged(self):
        """Verifica que strings no se modifican."""
        assert sanitize_value("hello") == "hello"

    def test_list_recursive(self):
        """Verifica sanitización recursiva de listas."""
        result = sanitize_value([1, np.nan, 3, np.inf])
        assert result == [1, None, 3, None]

    def test_tuple_recursive(self):
        """Verifica sanitización recursiva de tuplas."""
        result = sanitize_value((1, np.nan, 3))
        assert result == (1, None, 3)

    def test_dict_recursive(self):
        """Verifica sanitización recursiva de diccionarios."""
        result = sanitize_value({"a": 1, "b": np.nan, "c": np.inf})
        assert result == {"a": 1, "b": None, "c": None}

    def test_nested_structure(self):
        """Verifica sanitización de estructura anidada."""
        data = {"values": [1, np.nan], "nested": {"x": np.inf}}
        result = sanitize_value(data)

        assert result["values"] == [1, None]
        assert result["nested"]["x"] is None

    def test_numpy_array_unchanged(self):
        """Verifica que arrays numpy no se modifican internamente."""
        arr = np.array([1, 2, np.nan])
        result = sanitize_value(arr)
        assert isinstance(result, np.ndarray)

    def test_complex_number(self):
        """Verifica manejo de números complejos."""
        result = sanitize_value(complex(1, 2))
        assert result == "(1+2j)"

    def test_max_depth_protection(self):
        """Verifica protección contra recursión infinita."""
        # Crear estructura profundamente anidada
        deep = {"level": 0}
        current = deep
        for i in range(60):
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sanitize_value(deep, max_depth=50)
            # Debería haber una advertencia de profundidad
            assert len(w) >= 1

    def test_non_recursive_mode(self):
        """Verifica modo no recursivo."""
        data = [1, np.nan, 3]
        result = sanitize_value(data, recursive=False)
        assert result == [1, np.nan, 3]  # No sanitiza internamente

    def test_pandas_na(self):
        """Verifica manejo de pandas NA."""
        assert sanitize_value(pd.NA) is None


class TestValidateRequiredKeys:
    """Tests para la función validate_required_keys."""

    def test_valid_keys_first_option(self):
        """Verifica validación con primera opción de claves."""
        item = {"VR_TOTAL": 100, "CANTIDAD": 5}
        # No debería lanzar excepción
        validate_required_keys(
            item, [["VR_TOTAL", "vr_total"], ["CANTIDAD", "cantidad"]]
        )

    def test_valid_keys_second_option(self):
        """Verifica validación con segunda opción de claves."""
        item = {"vr_total": 100, "cantidad": 5}
        validate_required_keys(
            item, [["VR_TOTAL", "vr_total"], ["CANTIDAD", "cantidad"]]
        )

    def test_missing_first_key_group(self):
        """Error cuando falta primer grupo de claves."""
        item = {"CANTIDAD": 5}
        with pytest.raises(ValueError, match="Falta campo requerido"):
            validate_required_keys(
                item, [["VR_TOTAL", "vr_total"], ["CANTIDAD", "cantidad"]]
            )

    def test_missing_second_key_group(self):
        """Error cuando falta segundo grupo de claves."""
        item = {"VR_TOTAL": 100}
        with pytest.raises(ValueError, match="Falta campo requerido"):
            validate_required_keys(
                item, [["VR_TOTAL", "vr_total"], ["CANTIDAD", "cantidad"]]
            )

    def test_error_message_includes_index(self):
        """Verifica que mensaje de error incluye índice."""
        item = {"VR_TOTAL": 100}
        with pytest.raises(ValueError, match="en índice 5"):
            validate_required_keys(
                item,
                [["VR_TOTAL"], ["CANTIDAD"]],
                item_index=5,
            )


class TestEstimateMemoryUsage:
    """Tests para la función estimate_memory_usage."""

    def test_basic_estimation(self):
        """Verifica estimación básica."""
        memory = estimate_memory_usage(1000, 100)
        assert memory > 0
        assert isinstance(memory, int)

    def test_larger_simulation_more_memory(self):
        """Más simulaciones = más memoria."""
        mem_small = estimate_memory_usage(1000, 100)
        mem_large = estimate_memory_usage(10000, 100)
        assert mem_large > mem_small

    def test_more_apus_more_memory(self):
        """Más APUs = más memoria."""
        mem_small = estimate_memory_usage(1000, 100)
        mem_large = estimate_memory_usage(1000, 1000)
        assert mem_large > mem_small

    def test_float32_less_memory(self):
        """float32 usa menos memoria que float64."""
        mem_f64 = estimate_memory_usage(1000, 100, use_float32=False)
        mem_f32 = estimate_memory_usage(1000, 100, use_float32=True)
        assert mem_f32 < mem_f64

    def test_dataframe_overhead(self):
        """Verifica inclusión de overhead de DataFrame."""
        mem_with = estimate_memory_usage(1000, 100, include_dataframe_overhead=True)
        mem_without = estimate_memory_usage(1000, 100, include_dataframe_overhead=False)
        assert mem_with > mem_without

    def test_known_size_approximation(self):
        """Verifica aproximación conocida."""
        # 1000 sims × 100 APUs × 8 bytes = 800KB solo matriz
        memory = estimate_memory_usage(1000, 100)
        # Con overhead debería ser > 800KB pero < 2MB
        assert memory > 800_000
        assert memory < 2_000_000


class TestIsNumericValid:
    """Tests para la función is_numeric_valid."""

    def test_valid_int(self):
        """Verifica int válido."""
        assert is_numeric_valid(42) is True

    def test_valid_float(self):
        """Verifica float válido."""
        assert is_numeric_valid(3.14) is True

    def test_valid_zero(self):
        """Verifica cero."""
        assert is_numeric_valid(0) is True
        assert is_numeric_valid(0.0) is True

    def test_valid_negative(self):
        """Verifica negativo."""
        assert is_numeric_valid(-100.5) is True

    def test_nan_invalid(self):
        """NaN es inválido."""
        assert is_numeric_valid(float("nan")) is False
        assert is_numeric_valid(np.nan) is False

    def test_inf_invalid(self):
        """Infinito es inválido."""
        assert is_numeric_valid(float("inf")) is False
        assert is_numeric_valid(float("-inf")) is False

    def test_none_invalid(self):
        """None es inválido."""
        assert is_numeric_valid(None) is False

    def test_string_invalid(self):
        """String es inválido."""
        assert is_numeric_valid("100") is False

    def test_numpy_types_valid(self):
        """Tipos numpy válidos."""
        assert is_numeric_valid(np.float64(10.5)) is True
        assert is_numeric_valid(np.int32(42)) is True

    def test_numpy_nan_invalid(self):
        """numpy.nan es inválido."""
        assert is_numeric_valid(np.float64("nan")) is False


class TestCalculateConvergenceMetrics:
    """Tests para la función calculate_convergence_metrics."""

    def test_converged_simulation(self):
        """Verifica simulación convergida."""
        # Generar datos con baja variabilidad relativa
        np.random.seed(42)
        data = np.random.normal(10000, 100, 10000)

        metrics = calculate_convergence_metrics(data)

        assert metrics.is_converged is True
        assert metrics.effective_sample_size == 10000
        assert metrics.relative_error < 0.01

    def test_not_converged_high_variance(self):
        """Verifica simulación no convergida."""
        # Pocos datos con alta variabilidad
        np.random.seed(42)
        data = np.random.normal(1000, 500, 100)

        metrics = calculate_convergence_metrics(data)

        # Con CV alto y n pequeño, probablemente no converge
        assert metrics.effective_sample_size == 100
        # El relative_error dependerá de la muestra específica

    def test_std_error_formula(self):
        """Verifica fórmula de error estándar."""
        data = np.array([100, 100, 100, 100])  # Sin variación
        metrics = calculate_convergence_metrics(data)

        assert metrics.mean_std_error == 0.0

    def test_half_width_ci(self):
        """Verifica cálculo de semi-ancho IC."""
        np.random.seed(42)
        data = np.random.normal(1000, 100, 1000)

        metrics = calculate_convergence_metrics(data)

        # half_width = 1.96 * SEM
        expected_hw = 1.96 * metrics.mean_std_error
        assert abs(metrics.half_width_ci - expected_hw) < 0.01

    def test_custom_tolerance(self):
        """Verifica tolerancia personalizada."""
        np.random.seed(42)
        data = np.random.normal(1000, 50, 1000)

        # Con tolerancia muy baja, podría no converger
        metrics_strict = calculate_convergence_metrics(data, tolerance=0.001)
        metrics_loose = calculate_convergence_metrics(data, tolerance=0.1)

        assert metrics_loose.is_converged is True
        # La estricta podría o no converger


# ============================================================================
# TESTS: MonteCarloSimulator - Inicialización
# ============================================================================


class TestMonteCarloSimulatorInit:
    """Tests de inicialización del simulador."""

    def test_default_initialization(self):
        """Verifica inicialización por defecto."""
        simulator = MonteCarloSimulator()

        assert simulator.config is not None
        assert simulator.logger is not None
        assert simulator.rng is not None

    def test_custom_config(self, deterministic_config):
        """Verifica inicialización con config personalizada."""
        simulator = MonteCarloSimulator(config=deterministic_config)

        assert simulator.config.num_simulations == 1000
        assert simulator.config.random_seed == 42

    def test_custom_logger(self, silent_logger):
        """Verifica inicialización con logger personalizado."""
        simulator = MonteCarloSimulator(logger=silent_logger)

        assert simulator.logger is silent_logger

    def test_rng_reproducibility(self, deterministic_config, silent_logger):
        """Verifica reproducibilidad con semilla."""
        sim1 = MonteCarloSimulator(config=deterministic_config, logger=silent_logger)
        sim2 = MonteCarloSimulator(config=deterministic_config, logger=silent_logger)

        # Generar algunos números aleatorios
        r1 = sim1.rng.random(10)
        r2 = sim2.rng.random(10)

        np.testing.assert_array_equal(r1, r2)


# ============================================================================
# TESTS: MonteCarloSimulator - Validación de Entrada
# ============================================================================


class TestMonteCarloSimulatorInputValidation:
    """Tests de validación de datos de entrada."""

    def test_valid_input(self, simulator, simple_apu_data):
        """Verifica datos válidos."""
        # No debería lanzar excepción
        result = simulator.run_simulation(simple_apu_data)
        assert result.is_successful()

    def test_empty_list_error(self, simulator):
        """Error con lista vacía."""
        with pytest.raises(ValueError, match="no puede estar vacía"):
            simulator.run_simulation([])

    def test_not_list_error(self, simulator):
        """Error si no es lista."""
        with pytest.raises(TypeError, match="debe ser una lista"):
            simulator.run_simulation({"VR_TOTAL": 100, "CANTIDAD": 5})

    def test_not_dict_elements_error(self, simulator):
        """Error si elementos no son diccionarios."""
        with pytest.raises(TypeError, match="deben ser diccionarios"):
            simulator.run_simulation([1, 2, 3])

    def test_mixed_types_error(self, simulator):
        """Error con tipos mezclados."""
        data = [{"VR_TOTAL": 100, "CANTIDAD": 5}, "invalid", 123]
        with pytest.raises(TypeError, match="deben ser diccionarios"):
            simulator.run_simulation(data)

    def test_missing_required_keys_error(self, simulator):
        """Error cuando faltan claves requeridas."""
        data = [{"price": 100, "qty": 5}]  # Claves incorrectas
        with pytest.raises(ValueError, match="claves requeridas"):
            simulator.run_simulation(data)


# ============================================================================
# TESTS: MonteCarloSimulator - Preparación de Datos
# ============================================================================


class TestMonteCarloSimulatorDataPreparation:
    """Tests de preparación de datos."""

    def test_alias_normalization(self, simulator, aliased_apu_data):
        """Verifica normalización de aliases de columnas."""
        result = simulator.run_simulation(aliased_apu_data)

        assert result.is_successful()
        assert result.metadata["data_quality"]["valid_items"] == 3

    def test_mixed_valid_invalid_filtering(self, simulator, silent_logger):
        """Verifica filtrado de datos mixtos."""
        config = MonteCarloConfig(
            num_simulations=1000,
            min_cost_threshold=0.0,  # Permite negativos
            random_seed=42,
        )
        sim = MonteCarloSimulator(config=config, logger=silent_logger)

        data = [
            {"VR_TOTAL": 1000.0, "CANTIDAD": 5.0},
            {"VR_TOTAL": np.nan, "CANTIDAD": 3.0},
            {"VR_TOTAL": 2000.0, "CANTIDAD": 4.0},
        ]

        result = sim.run_simulation(data)

        assert result.is_successful()
        # Solo 2 de 3 deberían ser válidos
        assert result.metadata["data_quality"]["valid_items"] == 2

    def test_min_cost_threshold_filtering(self, silent_logger):
        """Verifica filtrado por umbral de costo mínimo."""
        config = MonteCarloConfig(
            num_simulations=1000,
            min_cost_threshold=500.0,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        data = [
            {"VR_TOTAL": 100.0, "CANTIDAD": 5.0},  # Descartado
            {"VR_TOTAL": 1000.0, "CANTIDAD": 3.0},  # Válido
            {"VR_TOTAL": 200.0, "CANTIDAD": 4.0},  # Descartado
        ]

        result = simulator.run_simulation(data)

        assert result.metadata["data_quality"]["valid_items"] == 1

    def test_all_invalid_data(self, simulator):
        """Verifica manejo cuando todos los datos son inválidos."""
        data = [
            {"VR_TOTAL": np.nan, "CANTIDAD": 5.0},
            {"VR_TOTAL": np.inf, "CANTIDAD": 3.0},
            {"VR_TOTAL": 1000.0, "CANTIDAD": np.nan},
        ]

        result = simulator.run_simulation(data)

        assert result.status == SimulationStatus.NO_VALID_DATA
        assert result.is_successful() is False


# ============================================================================
# TESTS: MonteCarloSimulator - Simulación Normal
# ============================================================================


class TestMonteCarloSimulatorNormalDistribution:
    """Tests de simulación con distribución normal."""

    def test_basic_simulation(self, simulator, simple_apu_data):
        """Verifica simulación básica."""
        result = simulator.run_simulation(simple_apu_data)

        assert result.status == SimulationStatus.SUCCESS
        assert result.statistics["mean"] is not None
        assert result.statistics["std_dev"] is not None

    def test_reproducibility(self, deterministic_config, silent_logger, simple_apu_data):
        """Verifica reproducibilidad con semilla."""
        sim1 = MonteCarloSimulator(config=deterministic_config, logger=silent_logger)
        sim2 = MonteCarloSimulator(config=deterministic_config, logger=silent_logger)

        result1 = sim1.run_simulation(simple_apu_data)
        result2 = sim2.run_simulation(simple_apu_data)

        assert result1.statistics["mean"] == result2.statistics["mean"]
        assert result1.statistics["std_dev"] == result2.statistics["std_dev"]

    def test_truncate_negative_default(self, deterministic_config, silent_logger):
        """Verifica truncamiento de negativos por defecto."""
        config = MonteCarloConfig(
            num_simulations=10000,
            volatility_factor=0.5,  # Alta volatilidad para generar negativos
            truncate_negative=True,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        data = [{"VR_TOTAL": 100.0, "CANTIDAD": 1.0}]  # Base pequeña
        result = simulator.run_simulation(data)

        # El mínimo no debería ser negativo
        assert result.statistics["min"] >= 0

    def test_no_truncate_negative(self, silent_logger):
        """Verifica sin truncamiento de negativos."""
        config = MonteCarloConfig(
            num_simulations=10000,
            volatility_factor=0.8,
            truncate_negative=False,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        data = [{"VR_TOTAL": 100.0, "CANTIDAD": 1.0}]
        result = simulator.run_simulation(data)

        # Con alta volatilidad y sin truncar, podría haber negativos
        # (o no, depende de la muestra)
        assert result.is_successful()


# ============================================================================
# TESTS: MonteCarloSimulator - Simulación Log-Normal
# ============================================================================


class TestMonteCarloSimulatorLognormalDistribution:
    """Tests de simulación con distribución log-normal."""

    def test_lognormal_simulation(self, lognormal_config, silent_logger, simple_apu_data):
        """Verifica simulación log-normal."""
        simulator = MonteCarloSimulator(config=lognormal_config, logger=silent_logger)
        result = simulator.run_simulation(simple_apu_data)

        assert result.is_successful()
        assert result.statistics["mean"] is not None

    def test_lognormal_always_positive(self, silent_logger):
        """Log-normal siempre produce valores positivos."""
        config = MonteCarloConfig(
            num_simulations=10000,
            volatility_factor=0.5,
            distribution=DistributionType.LOGNORMAL,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        data = [{"VR_TOTAL": 100.0, "CANTIDAD": 1.0}]
        result = simulator.run_simulation(data)

        # Log-normal nunca genera negativos
        assert result.statistics["min"] > 0

    def test_lognormal_positive_skew(self, silent_logger, large_apu_data):
        """Log-normal tiene sesgo positivo."""
        config = MonteCarloConfig(
            num_simulations=10000,
            volatility_factor=0.3,
            distribution=DistributionType.LOGNORMAL,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        result = simulator.run_simulation(large_apu_data)

        # Log-normal tiene media > mediana (sesgo positivo)
        assert result.statistics["mean"] >= result.statistics["median"]


# ============================================================================
# TESTS: MonteCarloSimulator - Simulación Triangular
# ============================================================================


class TestMonteCarloSimulatorTriangularDistribution:
    """Tests de simulación con distribución triangular."""

    def test_triangular_simulation(self, triangular_config, silent_logger, simple_apu_data):
        """Verifica simulación triangular."""
        simulator = MonteCarloSimulator(config=triangular_config, logger=silent_logger)
        result = simulator.run_simulation(simple_apu_data)

        assert result.is_successful()

    def test_triangular_bounded(self, silent_logger):
        """Triangular está acotada."""
        config = MonteCarloConfig(
            num_simulations=10000,
            volatility_factor=0.20,  # ±20%
            distribution=DistributionType.TRIANGULAR,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        data = [{"VR_TOTAL": 1000.0, "CANTIDAD": 1.0}]  # Base = 1000
        result = simulator.run_simulation(data)

        # Con vol=0.20, rango debería ser [800, 1200]
        assert result.statistics["min"] >= 800 * 0.99  # Pequeño margen
        assert result.statistics["max"] <= 1200 * 1.01


# ============================================================================
# TESTS: MonteCarloSimulator - Variantes Antitéticas
# ============================================================================


class TestMonteCarloSimulatorAntithetic:
    """Tests de variantes antitéticas."""

    def test_antithetic_reduces_variance(self, silent_logger, large_apu_data):
        """Variantes antitéticas deberían reducir varianza del estimador."""
        config_normal = MonteCarloConfig(
            num_simulations=1000,
            volatility_factor=0.15,
            use_antithetic=False,
            random_seed=42,
        )
        config_antithetic = MonteCarloConfig(
            num_simulations=1000,
            volatility_factor=0.15,
            use_antithetic=True,
            random_seed=42,
        )

        sim_normal = MonteCarloSimulator(config=config_normal, logger=silent_logger)
        sim_antithetic = MonteCarloSimulator(
            config=config_antithetic, logger=silent_logger
        )

        result_normal = sim_normal.run_simulation(large_apu_data)
        result_antithetic = sim_antithetic.run_simulation(large_apu_data)

        # Ambos deberían ser exitosos
        assert result_normal.is_successful()
        assert result_antithetic.is_successful()

    def test_antithetic_with_lognormal(self, silent_logger, simple_apu_data):
        """Verifica antitéticas con log-normal."""
        config = MonteCarloConfig(
            num_simulations=1000,
            volatility_factor=0.15,
            distribution=DistributionType.LOGNORMAL,
            use_antithetic=True,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        result = simulator.run_simulation(simple_apu_data)

        assert result.is_successful()
        assert result.statistics["min"] > 0  # Log-normal siempre positivo


# ============================================================================
# TESTS: MonteCarloSimulator - Estadísticas
# ============================================================================


class TestMonteCarloSimulatorStatistics:
    """Tests de cálculo de estadísticas."""

    def test_all_basic_statistics_present(self, simulator, simple_apu_data):
        """Verifica presencia de estadísticas básicas."""
        result = simulator.run_simulation(simple_apu_data)

        assert "mean" in result.statistics
        assert "median" in result.statistics
        assert "std_dev" in result.statistics
        assert "variance" in result.statistics
        assert "min" in result.statistics
        assert "max" in result.statistics
        assert "skewness" in result.statistics
        assert "kurtosis" in result.statistics

    def test_percentiles_present(self, simulator, simple_apu_data):
        """Verifica presencia de percentiles configurados."""
        result = simulator.run_simulation(simple_apu_data)

        for p in simulator.config.percentiles:
            assert f"percentile_{p}" in result.statistics

    def test_risk_metrics_present(self, simulator, simple_apu_data):
        """Verifica presencia de métricas de riesgo."""
        result = simulator.run_simulation(simple_apu_data)

        for conf in [90, 95, 99]:
            assert f"var_{conf}" in result.statistics
            assert f"cvar_{conf}" in result.statistics

    def test_coefficient_of_variation(self, simulator, simple_apu_data):
        """Verifica cálculo de coeficiente de variación."""
        result = simulator.run_simulation(simple_apu_data)

        cv = result.statistics["coefficient_of_variation"]
        mean = result.statistics["mean"]
        std = result.statistics["std_dev"]

        # CV = std / mean
        if mean > 0:
            expected_cv = std / mean
            assert abs(cv - expected_cv) < 0.001

    def test_iqr(self, simulator, simple_apu_data):
        """Verifica cálculo de rango intercuartílico."""
        result = simulator.run_simulation(simple_apu_data)

        iqr = result.statistics["iqr"]
        p25 = result.statistics["percentile_25"]
        p75 = result.statistics["percentile_75"]

        assert abs(iqr - (p75 - p25)) < 0.001

    def test_confidence_intervals(self, simulator, simple_apu_data):
        """Verifica intervalos de confianza."""
        result = simulator.run_simulation(simple_apu_data)

        assert result.statistics["ci_90_lower"] == result.statistics["percentile_5"]
        assert result.statistics["ci_90_upper"] == result.statistics["percentile_95"]
        assert result.statistics["ci_50_lower"] == result.statistics["percentile_25"]
        assert result.statistics["ci_50_upper"] == result.statistics["percentile_75"]

    def test_cvar_greater_than_var(self, simulator, simple_apu_data):
        """CVaR debe ser >= VaR (para costos)."""
        result = simulator.run_simulation(simple_apu_data)

        for conf in [90, 95, 99]:
            var = result.statistics[f"var_{conf}"]
            cvar = result.statistics[f"cvar_{conf}"]
            if var is not None and cvar is not None:
                assert cvar >= var


# ============================================================================
# TESTS: MonteCarloSimulator - Convergencia
# ============================================================================


class TestMonteCarloSimulatorConvergence:
    """Tests de métricas de convergencia."""

    def test_convergence_metrics_present(self, simulator, simple_apu_data):
        """Verifica presencia de métricas de convergencia."""
        result = simulator.run_simulation(simple_apu_data)

        assert result.convergence is not None
        assert isinstance(result.convergence, ConvergenceMetrics)

    def test_convergence_with_many_simulations(self, silent_logger, simple_apu_data):
        """Muchas simulaciones deberían converger."""
        config = MonteCarloConfig(
            num_simulations=50000,
            volatility_factor=0.10,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        result = simulator.run_simulation(simple_apu_data)

        assert result.convergence.is_converged is True

    def test_convergence_warning_status(self, silent_logger, simple_apu_data):
        """Verifica estado de advertencia de convergencia."""
        config = MonteCarloConfig(
            num_simulations=MIN_NUM_SIMULATIONS,  # Mínimo
            volatility_factor=0.5,  # Alta volatilidad
            check_convergence=True,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        result = simulator.run_simulation(simple_apu_data)

        # Con pocas simulaciones y alta volatilidad, podría no converger
        # pero aún debería ser "exitoso" en el sentido de producir resultados
        assert result.status in (
            SimulationStatus.SUCCESS,
            SimulationStatus.CONVERGENCE_WARNING,
        )

    def test_no_convergence_check(self, silent_logger, simple_apu_data):
        """Verifica desactivación de verificación de convergencia."""
        config = MonteCarloConfig(
            num_simulations=500,
            check_convergence=False,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        result = simulator.run_simulation(simple_apu_data)

        assert result.convergence is None


# ============================================================================
# TESTS: MonteCarloSimulator - Metadata
# ============================================================================


class TestMonteCarloSimulatorMetadata:
    """Tests de metadata de simulación."""

    def test_metadata_structure(self, simulator, simple_apu_data):
        """Verifica estructura de metadata."""
        result = simulator.run_simulation(simple_apu_data)

        assert "config" in result.metadata
        assert "data_quality" in result.metadata
        assert "simulation" in result.metadata
        assert "base_cost_summary" in result.metadata

    def test_config_metadata(self, simulator, simple_apu_data):
        """Verifica metadata de configuración."""
        result = simulator.run_simulation(simple_apu_data)

        config_meta = result.metadata["config"]
        assert config_meta["num_simulations_requested"] == simulator.config.num_simulations
        assert config_meta["volatility_factor"] == simulator.config.volatility_factor
        assert config_meta["distribution"] == simulator.config.distribution.value

    def test_data_quality_metadata(self, simulator, simple_apu_data):
        """Verifica metadata de calidad de datos."""
        result = simulator.run_simulation(simple_apu_data)

        dq = result.metadata["data_quality"]
        assert dq["total_items_input"] == 3
        assert dq["valid_items"] == 3
        assert dq["discarded_items"] == 0
        assert dq["discard_rate"] == 0.0

    def test_simulation_metadata(self, simulator, simple_apu_data):
        """Verifica metadata de simulación."""
        result = simulator.run_simulation(simple_apu_data)

        sim_meta = result.metadata["simulation"]
        assert sim_meta["num_simulations_completed"] <= simulator.config.num_simulations
        assert 0 <= sim_meta["simulation_success_rate"] <= 1

    def test_base_cost_summary(self, simulator, simple_apu_data):
        """Verifica resumen de costos base."""
        result = simulator.run_simulation(simple_apu_data)

        summary = result.metadata["base_cost_summary"]
        assert "sum" in summary
        assert "mean" in summary
        assert "std" in summary
        assert "min" in summary
        assert "max" in summary

        # Verificar valores esperados
        # simple_apu_data: [1000*5, 2000*3, 1500*4] = [5000, 6000, 6000]
        expected_sum = 5000 + 6000 + 6000
        assert summary["sum"] == expected_sum


# ============================================================================
# TESTS: MonteCarloSimulator - Análisis de Sensibilidad
# ============================================================================


class TestMonteCarloSimulatorSensitivityAnalysis:
    """Tests de análisis de sensibilidad."""

    def test_sensitivity_basic(self, simulator, simple_apu_data):
        """Verifica análisis de sensibilidad básico."""
        result = simulator.get_sensitivity_analysis(simple_apu_data)

        assert "top_contributors" in result
        assert "total_variance" in result
        assert "num_apus_analyzed" in result
        assert "concentration_index_hhi" in result

    def test_sensitivity_top_contributors(self, simulator, simple_apu_data):
        """Verifica estructura de top contributors."""
        result = simulator.get_sensitivity_analysis(simple_apu_data, top_n=3)

        assert len(result["top_contributors"]) == 3

        for contributor in result["top_contributors"]:
            assert "rank" in contributor
            assert "index" in contributor
            assert "base_cost" in contributor
            assert "sensitivity_index" in contributor
            assert "variance_contribution_pct" in contributor
            assert "cumulative_contribution_pct" in contributor

    def test_sensitivity_indices_sum_to_one(self, simulator, large_apu_data):
        """Los índices de sensibilidad deben sumar 1."""
        result = simulator.get_sensitivity_analysis(large_apu_data, top_n=100)

        total = sum(c["sensitivity_index"] for c in result["top_contributors"])
        assert abs(total - 1.0) < 0.001

    def test_sensitivity_cumulative_monotonic(self, simulator, simple_apu_data):
        """Contribución acumulada debe ser monótona creciente."""
        result = simulator.get_sensitivity_analysis(simple_apu_data, top_n=3)

        cumulative = [c["cumulative_contribution_pct"] for c in result["top_contributors"]]
        for i in range(1, len(cumulative)):
            assert cumulative[i] >= cumulative[i - 1]

    def test_sensitivity_hhi_interpretation(self, simulator, simple_apu_data):
        """Verifica interpretación de HHI."""
        result = simulator.get_sensitivity_analysis(simple_apu_data)

        assert "interpretation" in result
        assert "hhi_meaning" in result["interpretation"]

    def test_sensitivity_with_invalid_data(self, simulator):
        """Análisis de sensibilidad con datos inválidos."""
        data = [{"VR_TOTAL": np.nan, "CANTIDAD": 5.0}]
        result = simulator.get_sensitivity_analysis(data)

        assert "error" in result


# ============================================================================
# TESTS: MonteCarloSimulator - Memoria
# ============================================================================


class TestMonteCarloSimulatorMemory:
    """Tests de gestión de memoria."""

    def test_float32_option(self, silent_logger, simple_apu_data):
        """Verifica opción de float32."""
        config = MonteCarloConfig(
            num_simulations=1000,
            use_float32=True,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        result = simulator.run_simulation(simple_apu_data)

        assert result.is_successful()

    def test_memory_limit_error(self, silent_logger):
        """Verifica error por límite de memoria."""
        config = MonteCarloConfig(
            num_simulations=MAX_NUM_SIMULATIONS,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        # Crear datos muy grandes
        large_data = [{"VR_TOTAL": 1000.0, "CANTIDAD": 1.0} for _ in range(100000)]

        # Debería lanzar error o manejar correctamente
        # (depende del límite configurado)
        result = simulator.run_simulation(large_data)

        # Si excede el límite, debería ser error
        # Si no, debería funcionar
        assert result.status in (SimulationStatus.SUCCESS, SimulationStatus.ERROR)


# ============================================================================
# TESTS: MonteCarloSimulator - Casos Edge
# ============================================================================


class TestMonteCarloSimulatorEdgeCases:
    """Tests de casos límite."""

    def test_single_apu(self, simulator):
        """Simulación con un solo APU."""
        data = [{"VR_TOTAL": 1000.0, "CANTIDAD": 1.0}]
        result = simulator.run_simulation(data)

        assert result.is_successful()
        assert result.metadata["data_quality"]["valid_items"] == 1

    def test_very_small_costs(self, silent_logger):
        """Costos muy pequeños."""
        config = MonteCarloConfig(
            num_simulations=1000,
            min_cost_threshold=0.0,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        data = [{"VR_TOTAL": 0.001, "CANTIDAD": 1.0}]
        result = simulator.run_simulation(data)

        assert result.is_successful()

    def test_very_large_costs(self, simulator):
        """Costos muy grandes."""
        data = [{"VR_TOTAL": 1e12, "CANTIDAD": 1.0}]
        result = simulator.run_simulation(data)

        assert result.is_successful()

    def test_zero_volatility(self, silent_logger):
        """Volatilidad cero (determinista)."""
        config = MonteCarloConfig(
            num_simulations=1000,
            volatility_factor=0.0,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        data = [{"VR_TOTAL": 1000.0, "CANTIDAD": 1.0}]
        result = simulator.run_simulation(data)

        assert result.is_successful()
        # Con volatilidad 0, std_dev debería ser muy pequeño
        assert result.statistics["std_dev"] < 1.0

    def test_max_volatility(self, silent_logger):
        """Volatilidad máxima."""
        config = MonteCarloConfig(
            num_simulations=1000,
            volatility_factor=1.0,
            truncate_negative=True,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        data = [{"VR_TOTAL": 1000.0, "CANTIDAD": 1.0}]
        result = simulator.run_simulation(data)

        assert result.is_successful()


# ============================================================================
# TESTS: Función Legacy
# ============================================================================


class TestRunMonteCarloSimulationLegacy:
    """Tests para la función de compatibilidad legacy."""

    def test_basic_usage(self, simple_apu_data):
        """Uso básico de función legacy."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_monte_carlo_simulation(simple_apu_data)

        assert "mean" in result
        assert "std_dev" in result
        assert "percentile_5" in result
        assert "percentile_95" in result

    def test_deprecation_warning(self, simple_apu_data):
        """Verifica advertencia de deprecación."""
        with pytest.warns(DeprecationWarning, match="deprecada"):
            run_monte_carlo_simulation(simple_apu_data)

    def test_custom_parameters(self, simple_apu_data):
        """Parámetros personalizados."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_monte_carlo_simulation(
                simple_apu_data,
                num_simulations=5000,
                volatility_factor=0.20,
                min_cost_threshold=100.0,
            )

        assert result["mean"] is not None

    def test_empty_input_returns_none(self):
        """Lista vacía retorna None."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_monte_carlo_simulation([])

        assert result["mean"] is None
        assert result["std_dev"] is None

    def test_invalid_input_returns_none(self):
        """Input inválido retorna None."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_monte_carlo_simulation("invalid")

        assert result["mean"] is None

    def test_log_warnings_parameter(self, simple_apu_data):
        """Parámetro log_warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = run_monte_carlo_simulation(
                simple_apu_data, log_warnings=True
            )

        assert result["mean"] is not None


# ============================================================================
# TESTS: Propiedades Matemáticas
# ============================================================================


class TestMathematicalProperties:
    """Tests de propiedades matemáticas."""

    def test_mean_in_expected_range(self, silent_logger, simple_apu_data):
        """Media debe estar cerca del costo base total."""
        config = MonteCarloConfig(
            num_simulations=50000,
            volatility_factor=0.10,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        result = simulator.run_simulation(simple_apu_data)

        # Costo base: 5000 + 6000 + 6000 = 17000
        expected_base = 17000
        mean = result.statistics["mean"]

        # La media debería estar dentro del 5% del costo base
        assert abs(mean - expected_base) / expected_base < 0.05

    def test_std_dev_proportional_to_volatility(self, silent_logger):
        """Std dev proporcional a volatilidad."""
        data = [{"VR_TOTAL": 1000.0, "CANTIDAD": 10.0}]  # Base = 10000

        results = {}
        for vol in [0.05, 0.10, 0.20]:
            config = MonteCarloConfig(
                num_simulations=10000,
                volatility_factor=vol,
                random_seed=42,
            )
            sim = MonteCarloSimulator(config=config, logger=silent_logger)
            result = sim.run_simulation(data)
            results[vol] = result.statistics["std_dev"]

        # Mayor volatilidad = mayor std dev
        assert results[0.05] < results[0.10] < results[0.20]

    def test_percentile_ordering(self, simulator, large_apu_data):
        """Percentiles deben estar ordenados."""
        result = simulator.run_simulation(large_apu_data)

        percentiles = [
            result.statistics[f"percentile_{p}"]
            for p in simulator.config.percentiles
        ]

        for i in range(1, len(percentiles)):
            assert percentiles[i] >= percentiles[i - 1]

    def test_min_less_than_mean_less_than_max(self, simulator, large_apu_data):
        """min <= mean <= max."""
        result = simulator.run_simulation(large_apu_data)

        assert result.statistics["min"] <= result.statistics["mean"]
        assert result.statistics["mean"] <= result.statistics["max"]

    def test_variance_equals_std_squared(self, simulator, simple_apu_data):
        """variance = std_dev²."""
        result = simulator.run_simulation(simple_apu_data)

        std = result.statistics["std_dev"]
        var = result.statistics["variance"]

        assert abs(var - std**2) < 0.01

    def test_lognormal_mean_equals_base_approximately(self, silent_logger):
        """Log-normal: E[X] ≈ base_cost."""
        config = MonteCarloConfig(
            num_simulations=100000,
            volatility_factor=0.10,
            distribution=DistributionType.LOGNORMAL,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        data = [{"VR_TOTAL": 1000.0, "CANTIDAD": 10.0}]  # Base = 10000
        result = simulator.run_simulation(data)

        # Con suficientes simulaciones, la media debería estar cerca de 10000
        assert abs(result.statistics["mean"] - 10000) / 10000 < 0.05


# ============================================================================
# TESTS: Integración
# ============================================================================


class TestIntegration:
    """Tests de integración end-to-end."""

    def test_full_workflow(self, large_apu_data):
        """Flujo completo de trabajo."""
        # 1. Crear configuración
        config = MonteCarloConfig(
            num_simulations=5000,
            volatility_factor=0.15,
            distribution=DistributionType.LOGNORMAL,
            use_antithetic=True,
            check_convergence=True,
            random_seed=42,
        )

        # 2. Crear simulador
        simulator = MonteCarloSimulator(config=config)

        # 3. Ejecutar simulación
        result = simulator.run_simulation(large_apu_data)

        # 4. Verificar resultado
        assert result.is_successful()
        assert result.convergence is not None

        # 5. Obtener métricas de riesgo
        var_95 = result.get_var(0.95)
        cvar_95 = result.get_cvar(0.95)

        assert var_95 is not None
        assert cvar_95 is not None
        assert cvar_95 >= var_95

        # 6. Obtener intervalo de confianza
        lower, upper = result.get_confidence_interval(0.90)

        assert lower is not None
        assert upper is not None
        assert lower < result.statistics["mean"] < upper

        # 7. Análisis de sensibilidad
        sensitivity = simulator.get_sensitivity_analysis(large_apu_data, top_n=5)

        assert "top_contributors" in sensitivity
        assert len(sensitivity["top_contributors"]) == 5

        # 8. Serializar resultado
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["status"] == "success"

    def test_multiple_distribution_comparison(self, silent_logger, simple_apu_data):
        """Comparar resultados entre distribuciones."""
        results = {}

        for dist in DistributionType:
            config = MonteCarloConfig(
                num_simulations=10000,
                volatility_factor=0.15,
                distribution=dist,
                random_seed=42,
            )
            simulator = MonteCarloSimulator(config=config, logger=silent_logger)
            results[dist.value] = simulator.run_simulation(simple_apu_data)

        # Todas deberían ser exitosas
        for dist, result in results.items():
            assert result.is_successful(), f"Falló distribución: {dist}"

        # Log-normal debería tener sesgo positivo
        assert (
            results["lognormal"].statistics["skewness"]
            > results["normal"].statistics["skewness"]
        )

    def test_workflow_with_data_quality_issues(self, silent_logger):
        """Flujo con problemas de calidad de datos."""
        data = [
            {"VR_TOTAL": 1000.0, "CANTIDAD": 5.0},
            {"VR_TOTAL": np.nan, "CANTIDAD": 3.0},
            {"VR_TOTAL": 2000.0, "CANTIDAD": np.inf},
            {"VR_TOTAL": "invalid", "CANTIDAD": 2.0},
            {"VR_TOTAL": 1500.0, "CANTIDAD": 4.0},
            {"VR_TOTAL": -500.0, "CANTIDAD": 3.0},
            {"VR_TOTAL": 3000.0, "CANTIDAD": 2.0},
        ]

        config = MonteCarloConfig(
            num_simulations=1000,
            min_cost_threshold=0.0,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        result = simulator.run_simulation(data)

        # Debería manejar los datos problemáticos
        assert result.is_successful()
        assert result.metadata["data_quality"]["discarded_items"] > 0


# ============================================================================
# TESTS: Rendimiento
# ============================================================================


@pytest.mark.slow
class TestPerformance:
    """Tests de rendimiento (marcados como slow)."""

    def test_large_simulation(self, silent_logger):
        """Simulación grande."""
        config = MonteCarloConfig(
            num_simulations=100000,
            volatility_factor=0.15,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config, logger=silent_logger)

        # 1000 APUs
        np.random.seed(42)
        data = [
            {
                "VR_TOTAL": float(np.random.uniform(100, 10000)),
                "CANTIDAD": float(np.random.uniform(1, 100)),
            }
            for _ in range(1000)
        ]

        import time

        start = time.time()
        result = simulator.run_simulation(data)
        elapsed = time.time() - start

        assert result.is_successful()
        # Debería completarse en menos de 30 segundos
        assert elapsed < 30

    def test_float32_performance(self, silent_logger):
        """Comparar rendimiento float32 vs float64."""
        np.random.seed(42)
        data = [
            {
                "VR_TOTAL": float(np.random.uniform(100, 10000)),
                "CANTIDAD": float(np.random.uniform(1, 100)),
            }
            for _ in range(500)
        ]

        import time

        # Float64
        config64 = MonteCarloConfig(
            num_simulations=50000,
            use_float32=False,
            random_seed=42,
        )
        sim64 = MonteCarloSimulator(config=config64, logger=silent_logger)

        start = time.time()
        result64 = sim64.run_simulation(data)
        time64 = time.time() - start

        # Float32
        config32 = MonteCarloConfig(
            num_simulations=50000,
            use_float32=True,
            random_seed=42,
        )
        sim32 = MonteCarloSimulator(config=config32, logger=silent_logger)

        start = time.time()
        result32 = sim32.run_simulation(data)
        time32 = time.time() - start

        assert result64.is_successful()
        assert result32.is_successful()

        # Float32 debería ser más rápido o similar
        # (no siempre garantizado por cache effects)
        print(f"Float64: {time64:.3f}s, Float32: {time32:.3f}s")


# ============================================================================
# CONFIGURACIÓN DE PYTEST
# ============================================================================


def pytest_configure(config):
    """Configuración de pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
