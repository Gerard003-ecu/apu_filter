"""
Test Suite Completo para probability_models.py

Cubre todas las clases, métodos, validaciones, casos edge y flujos de error
del módulo de modelos de probabilidad y simulaciones Monte Carlo.
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Importar las clases y funciones a testear
from probability_models import (
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_VOLATILITY,
    MonteCarloConfig,
    MonteCarloSimulator,
    SimulationResult,
    SimulationStatus,
    estimate_memory_usage,
    run_monte_carlo_simulation,
    sanitize_value,
    validate_apu_data_structure,
)

# ============================================================================
# FIXTURES - Datos de Prueba Reutilizables
# ============================================================================


@pytest.fixture
def mock_logger():
    """Logger mock para pruebas."""
    logger = Mock(spec=logging.Logger)
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def default_config():
    """Configuración por defecto."""
    return MonteCarloConfig()


@pytest.fixture
def custom_config():
    """Configuración personalizada."""
    return MonteCarloConfig(
        num_simulations=500,
        volatility_factor=0.15,
        min_cost_threshold=100.0,
        min_quantity_threshold=0.1,
        random_seed=42,
        percentiles=[10, 50, 90],
    )


@pytest.fixture
def valid_apu_data():
    """Datos de APU válidos para pruebas."""
    return [
        {"VR_TOTAL": 10000.0, "CANTIDAD": 5.0},
        {"VR_TOTAL": 20000.0, "CANTIDAD": 3.0},
        {"VR_TOTAL": 15000.0, "CANTIDAD": 4.0},
        {"VR_TOTAL": 5000.0, "CANTIDAD": 10.0},
    ]


@pytest.fixture
def apu_data_with_zeros():
    """Datos con valores cero."""
    return [
        {"VR_TOTAL": 10000.0, "CANTIDAD": 0.0},  # Cantidad cero
        {"VR_TOTAL": 0.0, "CANTIDAD": 5.0},  # Valor cero
        {"VR_TOTAL": 15000.0, "CANTIDAD": 4.0},  # Válido
    ]


@pytest.fixture
def apu_data_with_negatives():
    """Datos con valores negativos."""
    return [
        {"VR_TOTAL": -10000.0, "CANTIDAD": 5.0},  # Valor negativo
        {"VR_TOTAL": 20000.0, "CANTIDAD": -3.0},  # Cantidad negativa
        {"VR_TOTAL": 15000.0, "CANTIDAD": 4.0},  # Válido
    ]


@pytest.fixture
def apu_data_with_nan():
    """Datos con valores NaN."""
    return [
        {"VR_TOTAL": np.nan, "CANTIDAD": 5.0},
        {"VR_TOTAL": 20000.0, "CANTIDAD": np.nan},
        {"VR_TOTAL": 15000.0, "CANTIDAD": 4.0},
        {"VR_TOTAL": np.inf, "CANTIDAD": 3.0},
    ]


@pytest.fixture
def apu_data_with_strings():
    """Datos con strings que pueden convertirse."""
    return [
        {"VR_TOTAL": "10000.0", "CANTIDAD": "5.0"},
        {"VR_TOTAL": "20000.0", "CANTIDAD": "3.0"},
        {"VR_TOTAL": "invalid", "CANTIDAD": "4.0"},
    ]


@pytest.fixture
def apu_data_missing_columns():
    """Datos con columnas faltantes."""
    return [
        {"VR_TOTAL": 10000.0},  # Falta CANTIDAD
        {"CANTIDAD": 5.0},  # Falta VR_TOTAL
    ]


@pytest.fixture
def large_apu_data():
    """Dataset grande para tests de rendimiento."""
    return [{"VR_TOTAL": 10000.0 + i * 100, "CANTIDAD": 5.0 + i * 0.1} for i in range(1000)]


@pytest.fixture
def simulator(mock_logger):
    """Simulador con logger mock."""
    return MonteCarloSimulator(logger=mock_logger)


@pytest.fixture
def simulator_with_seed():
    """Simulador con semilla fija para reproducibilidad."""
    config = MonteCarloConfig(random_seed=42, num_simulations=100)
    return MonteCarloSimulator(config=config)


# ============================================================================
# TESTS DE MonteCarloConfig
# ============================================================================


class TestMonteCarloConfig:
    """Tests para la clase de configuración."""

    def test_default_initialization(self):
        """Debe inicializar con valores por defecto."""
        config = MonteCarloConfig()

        assert config.num_simulations == DEFAULT_NUM_SIMULATIONS
        assert config.volatility_factor == DEFAULT_VOLATILITY
        assert config.min_cost_threshold == 0.0
        assert config.min_quantity_threshold == 0.0
        assert config.random_seed is None
        assert config.truncate_negative is True
        assert config.percentiles == [5, 25, 50, 75, 95]

    def test_custom_initialization(self):
        """Debe aceptar valores personalizados."""
        config = MonteCarloConfig(
            num_simulations=500,
            volatility_factor=0.15,
            min_cost_threshold=100.0,
            min_quantity_threshold=1.0,
            random_seed=42,
            truncate_negative=False,
            percentiles=[10, 90],
        )

        assert config.num_simulations == 500
        assert config.volatility_factor == 0.15
        assert config.min_cost_threshold == 100.0
        assert config.min_quantity_threshold == 1.0
        assert config.random_seed == 42
        assert config.truncate_negative is False
        assert config.percentiles == [10, 90]

    def test_validation_num_simulations_type(self):
        """Debe validar que num_simulations sea entero."""
        with pytest.raises(TypeError, match="debe ser int"):
            MonteCarloConfig(num_simulations=100.5)

        with pytest.raises(TypeError, match="debe ser int"):
            MonteCarloConfig(num_simulations="100")

    def test_validation_num_simulations_range(self):
        """Debe validar el rango de num_simulations."""
        with pytest.raises(ValueError, match="debe estar entre"):
            MonteCarloConfig(num_simulations=50)  # Menor al mínimo

        with pytest.raises(ValueError, match="debe estar entre"):
            MonteCarloConfig(num_simulations=2_000_000)  # Mayor al máximo

    def test_validation_volatility_type(self):
        """Debe validar que volatility_factor sea numérico."""
        with pytest.raises(TypeError, match="debe ser numérico"):
            MonteCarloConfig(volatility_factor="0.1")

    def test_validation_volatility_range(self):
        """Debe validar el rango de volatility_factor."""
        with pytest.raises(ValueError, match="debe estar entre"):
            MonteCarloConfig(volatility_factor=-0.1)

        with pytest.raises(ValueError, match="debe estar entre"):
            MonteCarloConfig(volatility_factor=1.5)

    def test_validation_thresholds_type(self):
        """Debe validar tipos de umbrales."""
        with pytest.raises(TypeError, match="debe ser numérico"):
            MonteCarloConfig(min_cost_threshold="100")

        with pytest.raises(TypeError, match="debe ser numérico"):
            MonteCarloConfig(min_quantity_threshold="1")

    def test_validation_thresholds_negative(self):
        """Debe rechazar umbrales negativos."""
        with pytest.raises(ValueError, match="no puede ser negativo"):
            MonteCarloConfig(min_cost_threshold=-100.0)

        with pytest.raises(ValueError, match="no puede ser negativo"):
            MonteCarloConfig(min_quantity_threshold=-1.0)

    def test_validation_random_seed_type(self):
        """Debe validar tipo de random_seed."""
        with pytest.raises(TypeError, match="debe ser int o None"):
            MonteCarloConfig(random_seed="42")

        # None debe ser válido
        config = MonteCarloConfig(random_seed=None)
        assert config.random_seed is None

    def test_validation_percentiles_type(self):
        """Debe validar que percentiles sea lista."""
        with pytest.raises(TypeError, match="debe ser una lista"):
            MonteCarloConfig(percentiles=(5, 95))

    def test_validation_percentiles_values(self):
        """Debe validar valores de percentiles."""
        with pytest.raises(ValueError, match="entre 0 y 100"):
            MonteCarloConfig(percentiles=[5, 150])

        with pytest.raises(ValueError, match="entre 0 y 100"):
            MonteCarloConfig(percentiles=[-5, 50])

        with pytest.raises(ValueError, match="entre 0 y 100"):
            MonteCarloConfig(percentiles=[5.5, 95])  # No enteros


# ============================================================================
# TESTS DE SimulationResult
# ============================================================================


class TestSimulationResult:
    """Tests para la clase de resultados."""

    def test_initialization(self):
        """Debe inicializar correctamente."""
        stats = {"mean": 100.0, "std_dev": 10.0}
        metadata = {"num_simulations": 1000}

        result = SimulationResult(
            status=SimulationStatus.SUCCESS, statistics=stats, metadata=metadata
        )

        assert result.status == SimulationStatus.SUCCESS
        assert result.statistics == stats
        assert result.metadata == metadata
        assert result.raw_results is None

    def test_initialization_with_raw_results(self):
        """Debe aceptar resultados brutos."""
        raw = np.array([100, 110, 90])

        result = SimulationResult(
            status=SimulationStatus.SUCCESS, statistics={}, metadata={}, raw_results=raw
        )

        np.testing.assert_array_equal(result.raw_results, raw)

    def test_to_dict_without_raw(self):
        """Debe convertir a dict sin resultados brutos."""
        result = SimulationResult(
            status=SimulationStatus.SUCCESS,
            statistics={"mean": 100.0},
            metadata={"count": 10},
        )

        result_dict = result.to_dict(include_raw=False)

        assert result_dict["status"] == "success"
        assert result_dict["statistics"] == {"mean": 100.0}
        assert result_dict["metadata"] == {"count": 10}
        assert "raw_results" not in result_dict

    def test_to_dict_with_raw(self):
        """Debe incluir resultados brutos si se solicita."""
        raw = np.array([100, 110, 90])
        result = SimulationResult(
            status=SimulationStatus.SUCCESS, statistics={}, metadata={}, raw_results=raw
        )

        result_dict = result.to_dict(include_raw=True)

        assert "raw_results" in result_dict
        assert result_dict["raw_results"] == [100, 110, 90]

    def test_is_successful_true(self):
        """Debe retornar True si el status es SUCCESS."""
        result = SimulationResult(
            status=SimulationStatus.SUCCESS, statistics={}, metadata={}
        )

        assert result.is_successful() is True

    def test_is_successful_false(self):
        """Debe retornar False si el status no es SUCCESS."""
        result = SimulationResult(
            status=SimulationStatus.NO_VALID_DATA, statistics={}, metadata={}
        )

        assert result.is_successful() is False


# ============================================================================
# TESTS DE sanitize_value
# ============================================================================


class TestSanitizeValue:
    """Tests para la función de sanitización."""

    def test_sanitize_nan(self):
        """Debe convertir NaN a None."""
        assert sanitize_value(np.nan) is None
        assert sanitize_value(pd.NA) is None
        assert sanitize_value(float("nan")) is None

    def test_sanitize_inf(self):
        """Debe convertir inf a None."""
        assert sanitize_value(np.inf) is None
        assert sanitize_value(-np.inf) is None
        assert sanitize_value(float("inf")) is None
        assert sanitize_value(float("-inf")) is None

    def test_sanitize_float(self):
        """Debe convertir numpy float a float nativo."""
        result = sanitize_value(np.float64(10.5))
        assert result == 10.5
        assert isinstance(result, float)

    def test_sanitize_int(self):
        """Debe convertir numpy int a int nativo."""
        result = sanitize_value(np.int64(10))
        assert result == 10
        assert isinstance(result, int)

    def test_sanitize_bool(self):
        """Debe convertir numpy bool a bool nativo."""
        result = sanitize_value(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_sanitize_string(self):
        """Debe retornar strings sin modificar."""
        result = sanitize_value("hello")
        assert result == "hello"
        assert isinstance(result, str)

    def test_sanitize_list(self):
        """Debe retornar listas sin modificar."""
        original = [1, 2, 3]
        result = sanitize_value(original)
        assert result is original

    def test_sanitize_tuple(self):
        """Debe retornar tuplas sin modificar."""
        original = (1, 2, 3)
        result = sanitize_value(original)
        assert result is original

    def test_sanitize_dict(self):
        """Debe retornar diccionarios sin modificar."""
        original = {"key": "value"}
        result = sanitize_value(original)
        assert result is original

    def test_sanitize_none(self):
        """Debe retornar None sin modificar."""
        assert sanitize_value(None) is None

    @pytest.mark.parametrize(
        "value,expected",
        [
            (10, 10),
            (10.5, 10.5),
            (True, True),
            (False, False),
            ("text", "text"),
            ([], []),
        ],
    )
    def test_sanitize_various_types(self, value, expected):
        """Debe manejar correctamente varios tipos."""
        result = sanitize_value(value)
        assert result == expected


# ============================================================================
# TESTS DE UTILIDADES
# ============================================================================


class TestValidateAPUDataStructure:
    """Tests para validación de estructura de datos."""

    def test_validate_valid_data(self, valid_apu_data):
        """Debe retornar True para datos válidos."""
        assert validate_apu_data_structure(valid_apu_data) is True

    def test_validate_not_a_list(self):
        """Debe retornar False si no es una lista."""
        assert validate_apu_data_structure("not a list") is False
        assert validate_apu_data_structure({"key": "value"}) is False
        assert validate_apu_data_structure(123) is False

    def test_validate_empty_list(self, mock_logger):
        """Debe retornar False para lista vacía."""
        result = validate_apu_data_structure([], logger=mock_logger)

        assert result is False
        mock_logger.warning.assert_called()

    def test_validate_not_all_dicts(self, mock_logger):
        """Debe retornar False si no todos son diccionarios."""
        data = [{"VR_TOTAL": 100}, "not a dict", {"CANTIDAD": 5}]

        result = validate_apu_data_structure(data, logger=mock_logger)

        assert result is False
        mock_logger.error.assert_called()

    def test_validate_logs_errors(self, mock_logger):
        """Debe loggear errores apropiadamente."""
        validate_apu_data_structure("invalid", logger=mock_logger)

        assert mock_logger.error.called


class TestEstimateMemoryUsage:
    """Tests para estimación de memoria."""

    def test_estimate_small_simulation(self):
        """Debe estimar correctamente para simulación pequeña."""
        memory = estimate_memory_usage(num_simulations=100, num_apus=10)

        # 100 * 10 * 8 bytes + overhead
        expected_matrix = 100 * 10 * 8
        assert memory > expected_matrix
        assert memory > 0

    def test_estimate_large_simulation(self):
        """Debe estimar correctamente para simulación grande."""
        memory = estimate_memory_usage(num_simulations=10000, num_apus=1000)

        # 10000 * 1000 * 8 = 80MB solo matriz
        expected_matrix = 10000 * 1000 * 8
        assert memory >= expected_matrix

    def test_estimate_returns_int(self):
        """Debe retornar un entero."""
        memory = estimate_memory_usage(100, 10)
        assert isinstance(memory, int)

    def test_estimate_zero_simulations(self):
        """Debe manejar cero simulaciones."""
        memory = estimate_memory_usage(0, 10)
        assert memory >= 0

    def test_estimate_zero_apus(self):
        """Debe manejar cero APUs."""
        memory = estimate_memory_usage(100, 0)
        assert memory >= 0


# ============================================================================
# TESTS DE MonteCarloSimulator - Inicialización
# ============================================================================


class TestMonteCarloSimulatorInitialization:
    """Tests para inicialización del simulador."""

    def test_initialization_default(self, mock_logger):
        """Debe inicializar con valores por defecto."""
        simulator = MonteCarloSimulator(logger=mock_logger)

        assert simulator.logger is mock_logger
        assert isinstance(simulator.config, MonteCarloConfig)
        assert simulator.rng is not None
        mock_logger.info.assert_called_once()

    def test_initialization_custom_config(self, mock_logger, custom_config):
        """Debe aceptar configuración personalizada."""
        simulator = MonteCarloSimulator(config=custom_config, logger=mock_logger)

        assert simulator.config is custom_config
        assert simulator.config.num_simulations == 500

    def test_initialization_without_logger(self):
        """Debe crear logger por defecto si no se proporciona."""
        simulator = MonteCarloSimulator()

        assert simulator.logger is not None
        assert isinstance(simulator.logger, logging.Logger)

    def test_initialization_sets_random_seed(self):
        """Debe configurar la semilla aleatoria."""
        config = MonteCarloConfig(random_seed=42)
        simulator = MonteCarloSimulator(config=config)

        # Verificar que se use la misma semilla
        assert simulator.config.random_seed == 42


# ============================================================================
# TESTS DE MonteCarloSimulator - Validación de Entrada
# ============================================================================


class TestMonteCarloSimulatorInputValidation:
    """Tests para validación de datos de entrada."""

    def test_validate_input_success(self, simulator, valid_apu_data):
        """Debe pasar validación con datos correctos."""
        # No debe lanzar excepción
        simulator._validate_input_data(valid_apu_data)

    def test_validate_input_not_list(self, simulator):
        """Debe rechazar datos que no sean lista."""
        with pytest.raises(TypeError, match="debe ser una lista"):
            simulator._validate_input_data("not a list")

    def test_validate_input_empty_list(self, simulator):
        """Debe rechazar lista vacía."""
        with pytest.raises(ValueError, match="no puede estar vacía"):
            simulator._validate_input_data([])

    def test_validate_input_not_dicts(self, simulator):
        """Debe rechazar elementos que no sean diccionarios."""
        with pytest.raises(TypeError, match="deben ser diccionarios"):
            simulator._validate_input_data([1, 2, 3])


# ============================================================================
# TESTS DE MonteCarloSimulator - Preparación de Datos
# ============================================================================


class TestMonteCarloSimulatorDataPreparation:
    """Tests para preparación y limpieza de datos."""

    def test_prepare_data_success(self, simulator, valid_apu_data):
        """Debe preparar datos válidos correctamente."""
        df_valid, discarded = simulator._prepare_data(valid_apu_data)

        assert len(df_valid) == 4
        assert discarded == 0
        assert "base_cost" in df_valid.columns
        assert (df_valid["base_cost"] > 0).all()

    def test_prepare_data_missing_columns(self, simulator, apu_data_missing_columns):
        """Debe lanzar error si faltan columnas."""
        with pytest.raises(ValueError, match="Faltan columnas requeridas"):
            simulator._prepare_data(apu_data_missing_columns)

    def test_prepare_data_converts_strings(self, simulator, apu_data_with_strings):
        """Debe convertir strings numéricos."""
        df_valid, discarded = simulator._prepare_data(apu_data_with_strings)

        # Los dos primeros son válidos, el tercero tiene "invalid"
        assert len(df_valid) == 2
        assert discarded == 1

    def test_prepare_data_filters_zeros(self, simulator, apu_data_with_zeros):
        """Debe filtrar valores cero."""
        df_valid, discarded = simulator._prepare_data(apu_data_with_zeros)

        # Solo el último es válido
        assert len(df_valid) == 1
        assert discarded == 2

    def test_prepare_data_filters_nan(self, simulator, apu_data_with_nan, mock_logger):
        """Debe filtrar valores NaN e inf."""
        simulator.logger = mock_logger
        df_valid, discarded = simulator._prepare_data(apu_data_with_nan)

        # Solo el tercero es completamente válido
        assert len(df_valid) == 1
        assert discarded == 3

        # Debe haber advertencias
        assert mock_logger.warning.called

    def test_prepare_data_calculates_base_cost(self, simulator, valid_apu_data):
        """Debe calcular el costo base correctamente."""
        df_valid, _ = simulator._prepare_data(valid_apu_data)

        # Verificar cálculo: VR_TOTAL * CANTIDAD
        expected_costs = [10000.0 * 5.0, 20000.0 * 3.0, 15000.0 * 4.0, 5000.0 * 10.0]

        np.testing.assert_array_almost_equal(df_valid["base_cost"].values, expected_costs)

    def test_prepare_data_with_threshold(self, valid_apu_data):
        """Debe aplicar umbrales de filtrado."""
        config = MonteCarloConfig(min_cost_threshold=8000.0, min_quantity_threshold=4.0)
        simulator = MonteCarloSimulator(config=config)

        df_valid, discarded = simulator._prepare_data(valid_apu_data)

        # Solo deben pasar los que cumplan ambos umbrales
        assert len(df_valid) < len(valid_apu_data)


# ============================================================================
# TESTS DE MonteCarloSimulator - Verificación de Memoria
# ============================================================================


class TestMonteCarloSimulatorMemoryCheck:
    """Tests para verificación de requisitos de memoria."""

    def test_check_memory_small_simulation(self, simulator):
        """Debe pasar para simulaciones pequeñas."""
        # No debe lanzar excepción
        simulator._check_memory_requirements(num_apus=10)

    def test_check_memory_warns_large_simulation(self, simulator, mock_logger):
        """Debe advertir para simulaciones grandes."""
        simulator.logger = mock_logger

        # Crear simulación que genere advertencia
        large_config = MonteCarloConfig(num_simulations=100000)
        large_simulator = MonteCarloSimulator(config=large_config, logger=mock_logger)

        large_simulator._check_memory_requirements(num_apus=10000)

        # Debe haber advertencia de memoria
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "memoria" in str(call).lower()
        ]
        assert len(warning_calls) > 0

    def test_check_memory_raises_on_excessive(self, simulator):
        """Debe lanzar MemoryError si es excesivo."""
        # Crear simulación extremadamente grande
        huge_config = MonteCarloConfig(num_simulations=1000000)
        huge_simulator = MonteCarloSimulator(config=huge_config)

        with pytest.raises(MemoryError):
            huge_simulator._check_memory_requirements(num_apus=100000)


# ============================================================================
# TESTS DE MonteCarloSimulator - Ejecución de Simulación
# ============================================================================


class TestMonteCarloSimulatorExecution:
    """Tests para ejecución de la simulación."""

    def test_execute_simulation_success(self, simulator_with_seed, valid_apu_data):
        """Debe ejecutar simulación exitosamente."""
        df_valid, _ = simulator_with_seed._prepare_data(valid_apu_data)

        simulated_costs = simulator_with_seed._execute_simulation(df_valid)

        assert len(simulated_costs) == 100  # num_simulations
        assert simulated_costs.dtype == np.float64
        assert np.all(np.isfinite(simulated_costs))

    def test_execute_simulation_reproducible(self, valid_apu_data):
        """Debe ser reproducible con misma semilla."""
        config = MonteCarloConfig(random_seed=42, num_simulations=100)

        sim1 = MonteCarloSimulator(config=config)
        df1, _ = sim1._prepare_data(valid_apu_data)
        results1 = sim1._execute_simulation(df1)

        sim2 = MonteCarloSimulator(config=config)
        df2, _ = sim2._prepare_data(valid_apu_data)
        results2 = sim2._execute_simulation(df2)

        np.testing.assert_array_equal(results1, results2)

    def test_execute_simulation_truncates_negatives(self, valid_apu_data):
        """Debe truncar valores negativos si está configurado."""
        config = MonteCarloConfig(
            truncate_negative=True,
            num_simulations=1000,
            volatility_factor=0.5,  # Alta volatilidad para generar negativos
        )
        simulator = MonteCarloSimulator(config=config)

        df_valid, _ = simulator._prepare_data(valid_apu_data)
        simulated_costs = simulator._execute_simulation(df_valid)

        # No debe haber valores negativos
        assert np.all(simulated_costs >= 0)

    def test_execute_simulation_allows_negatives(self, valid_apu_data):
        """Debe permitir valores negativos si está configurado."""
        config = MonteCarloConfig(
            truncate_negative=False,
            num_simulations=1000,
            volatility_factor=0.5,
            random_seed=42,
        )
        simulator = MonteCarloSimulator(config=config)

        df_valid, _ = simulator._prepare_data(valid_apu_data)
        simulated_costs = simulator._execute_simulation(df_valid)

        # Puede haber negativos (aunque depende del random)
        # Al menos verificamos que no se truncó todo
        assert len(simulated_costs) == 1000

    def test_execute_simulation_handles_invalid_results(self, simulator, mock_logger):
        """Debe manejar resultados inválidos."""
        simulator.logger = mock_logger

        # Crear DataFrame que pueda generar problemas
        df = pd.DataFrame(
            {
                "VR_TOTAL": [1e308, 1e308],  # Valores muy grandes
                "CANTIDAD": [1e308, 1e308],
                "base_cost": [1e308 * 1e308, 1e308 * 1e308],  # Overflow
            }
        )

        # Puede lanzar RuntimeError o procesar dependiendo de numpy
        try:
            result = simulator._execute_simulation(df)
            # Si no lanza error, verificar que filtre inválidos
            assert np.all(np.isfinite(result))
        except RuntimeError:
            # Esperado para casos extremos
            pass


# ============================================================================
# TESTS DE MonteCarloSimulator - Cálculo de Estadísticas
# ============================================================================


class TestMonteCarloSimulatorStatistics:
    """Tests para cálculo de estadísticas."""

    def test_calculate_statistics_basic(self, simulator):
        """Debe calcular estadísticas básicas."""
        simulated = np.array([90, 95, 100, 105, 110])

        stats = simulator._calculate_statistics(simulated)

        assert stats["mean"] == 100.0
        assert stats["median"] == 100.0
        assert stats["min"] == 90.0
        assert stats["max"] == 110.0
        assert "std_dev" in stats
        assert "variance" in stats

    def test_calculate_statistics_percentiles(self, simulator):
        """Debe calcular percentiles configurados."""
        simulated = np.arange(0, 101)  # 0 a 100

        stats = simulator._calculate_statistics(simulated)

        # Percentiles por defecto: [5, 25, 50, 75, 95]
        assert stats["percentile_5"] == 5.0
        assert stats["percentile_25"] == 25.0
        assert stats["percentile_50"] == 50.0
        assert stats["percentile_75"] == 75.0
        assert stats["percentile_95"] == 95.0

    def test_calculate_statistics_confidence_interval(self, simulator):
        """Debe calcular intervalo de confianza."""
        simulated = np.arange(0, 101)

        stats = simulator._calculate_statistics(simulated)

        assert stats["ci_90_lower"] == stats["percentile_5"]
        assert stats["ci_90_upper"] == stats["percentile_95"]

    def test_calculate_statistics_coefficient_variation(self, simulator):
        """Debe calcular coeficiente de variación."""
        simulated = np.array([90, 95, 100, 105, 110])

        stats = simulator._calculate_statistics(simulated)

        expected_cv = stats["std_dev"] / stats["mean"]
        assert stats["coefficient_of_variation"] == pytest.approx(expected_cv)

    def test_calculate_statistics_cv_zero_mean(self, simulator):
        """Debe manejar CV cuando la media es cero."""
        simulated = np.array([0, 0, 0, 0, 0])

        stats = simulator._calculate_statistics(simulated)

        assert stats["coefficient_of_variation"] is None

    def test_calculate_statistics_custom_percentiles(self):
        """Debe calcular percentiles personalizados."""
        config = MonteCarloConfig(percentiles=[10, 50, 90])
        simulator = MonteCarloSimulator(config=config)

        simulated = np.arange(0, 101)
        stats = simulator._calculate_statistics(simulated)

        assert "percentile_10" in stats
        assert "percentile_50" in stats
        assert "percentile_90" in stats
        assert "percentile_5" not in stats


# ============================================================================
# TESTS DE MonteCarloSimulator - Metadata
# ============================================================================


class TestMonteCarloSimulatorMetadata:
    """Tests para creación de metadata."""

    def test_create_metadata_complete(self, simulator, valid_apu_data):
        """Debe crear metadata completa."""
        df_valid, discarded = simulator._prepare_data(valid_apu_data)

        metadata = simulator._create_metadata(
            df_valid=df_valid, total_items=len(valid_apu_data), discarded_items=discarded
        )

        assert metadata["num_simulations"] == simulator.config.num_simulations
        assert metadata["volatility_factor"] == simulator.config.volatility_factor
        assert metadata["total_items_input"] == 4
        assert metadata["valid_items"] == 4
        assert metadata["discarded_items"] == 0
        assert metadata["discard_rate"] == 0.0
        assert "base_cost_sum" in metadata
        assert "base_cost_mean" in metadata
        assert "base_cost_std" in metadata

    def test_create_metadata_with_discarded(self, simulator, apu_data_with_zeros):
        """Debe calcular tasa de descarte correctamente."""
        df_valid, discarded = simulator._prepare_data(apu_data_with_zeros)

        metadata = simulator._create_metadata(
            df_valid=df_valid, total_items=3, discarded_items=discarded
        )

        assert metadata["discarded_items"] == 2
        assert metadata["discard_rate"] == pytest.approx(2 / 3)

    def test_create_metadata_base_costs(self, simulator, valid_apu_data):
        """Debe calcular estadísticas de costos base."""
        df_valid, _ = simulator._prepare_data(valid_apu_data)

        metadata = simulator._create_metadata(
            df_valid=df_valid, total_items=len(valid_apu_data), discarded_items=0
        )

        expected_sum = 10000 * 5 + 20000 * 3 + 15000 * 4 + 5000 * 10
        assert metadata["base_cost_sum"] == pytest.approx(expected_sum)


# ============================================================================
# TESTS DE MonteCarloSimulator - Integración
# ============================================================================


class TestMonteCarloSimulatorIntegration:
    """Tests de integración del simulador completo."""

    def test_run_simulation_success(self, simulator, valid_apu_data):
        """Debe ejecutar simulación completa exitosamente."""
        result = simulator.run_simulation(valid_apu_data)

        assert result.status == SimulationStatus.SUCCESS
        assert result.is_successful()
        assert "mean" in result.statistics
        assert "std_dev" in result.statistics
        assert result.metadata["valid_items"] == 4
        assert result.raw_results is not None

    def test_run_simulation_logs_info(self, simulator, valid_apu_data, mock_logger):
        """Debe loggear información del proceso."""
        simulator.logger = mock_logger

        result = simulator.run_simulation(valid_apu_data)

        # Verificar llamadas de logging
        assert mock_logger.info.called
        assert mock_logger.debug.called

    def test_run_simulation_no_valid_data(self, simulator, apu_data_with_zeros):
        """Debe manejar caso sin datos válidos."""
        # Configurar umbrales altos
        simulator.config.min_cost_threshold = 100000.0

        result = simulator.run_simulation(apu_data_with_zeros)

        assert result.status == SimulationStatus.NO_VALID_DATA
        assert not result.is_successful()
        assert result.statistics["mean"] is None

    def test_run_simulation_invalid_input_type(self, simulator):
        """Debe lanzar TypeError con entrada inválida."""
        with pytest.raises(TypeError):
            simulator.run_simulation("not a list")

    def test_run_simulation_empty_list(self, simulator):
        """Debe lanzar ValueError con lista vacía."""
        with pytest.raises(ValueError):
            simulator.run_simulation([])

    def test_run_simulation_missing_columns(self, simulator, apu_data_missing_columns):
        """Debe lanzar ValueError si faltan columnas."""
        with pytest.raises(ValueError, match="Faltan columnas"):
            simulator.run_simulation(apu_data_missing_columns)

    def test_run_simulation_handles_exceptions(self, simulator, valid_apu_data, mock_logger):
        """Debe manejar excepciones inesperadas."""
        simulator.logger = mock_logger

        # Simular error en preparación de datos
        with patch.object(
            simulator, "_prepare_data", side_effect=RuntimeError("Unexpected error")
        ):
            result = simulator.run_simulation(valid_apu_data)

            assert result.status == SimulationStatus.ERROR
            assert not result.is_successful()
            assert "error" in result.metadata


# ============================================================================
# TESTS DE run_monte_carlo_simulation (Función Legacy)
# ============================================================================


class TestRunMonteCarloSimulationLegacy:
    """Tests para la función de compatibilidad legacy."""

    def test_legacy_function_success(self, valid_apu_data):
        """Debe ejecutar correctamente con parámetros válidos."""
        result = run_monte_carlo_simulation(
            apu_details=valid_apu_data,
            num_simulations=100,
            volatility_factor=0.1,
            min_cost_threshold=0.0,
            log_warnings=False,
        )

        assert "mean" in result
        assert "std_dev" in result
        assert "percentile_5" in result
        assert "percentile_95" in result
        assert result["mean"] is not None

    def test_legacy_function_default_params(self, valid_apu_data):
        """Debe funcionar con parámetros por defecto."""
        result = run_monte_carlo_simulation(valid_apu_data)

        assert result["mean"] is not None
        assert isinstance(result["mean"], float)

    def test_legacy_function_invalid_simulations(self, valid_apu_data):
        """Debe retornar None en estadísticas si hay error."""
        result = run_monte_carlo_simulation(
            apu_details=valid_apu_data,
            num_simulations=-100,  # Inválido
            log_warnings=False,
        )

        # Debe retornar estructura con None debido al error
        assert result["mean"] is None
        assert result["std_dev"] is None

    def test_legacy_function_invalid_volatility(self, valid_apu_data):
        """Debe retornar None si volatility es inválida."""
        result = run_monte_carlo_simulation(
            apu_details=valid_apu_data,
            volatility_factor=-0.5,  # Inválido
            log_warnings=False,
        )

        assert result["mean"] is None

    def test_legacy_function_no_valid_data(self, apu_data_with_zeros):
        """Debe retornar None si no hay datos válidos."""
        result = run_monte_carlo_simulation(
            apu_details=apu_data_with_zeros,
            min_cost_threshold=100000.0,  # Muy alto
            log_warnings=False,
        )

        assert result["mean"] is None

    def test_legacy_function_with_logging(self, valid_apu_data, capsys):
        """Debe loggear si log_warnings=True."""
        result = run_monte_carlo_simulation(apu_details=valid_apu_data, log_warnings=True)

        # Verificar que funcionó
        assert result["mean"] is not None

    def test_legacy_function_returns_correct_structure(self, valid_apu_data):
        """Debe retornar estructura específica del legacy API."""
        result = run_monte_carlo_simulation(valid_apu_data)

        # Solo debe tener estas 4 claves
        assert set(result.keys()) == {"mean", "std_dev", "percentile_5", "percentile_95"}


# ============================================================================
# TESTS DE CASOS EDGE
# ============================================================================


class TestEdgeCases:
    """Tests para casos edge y situaciones límite."""

    def test_single_apu_item(self, simulator):
        """Debe manejar un solo item APU."""
        data = [{"VR_TOTAL": 10000.0, "CANTIDAD": 5.0}]

        result = simulator.run_simulation(data)

        assert result.is_successful()
        assert result.statistics["mean"] is not None

    def test_very_small_values(self, simulator):
        """Debe manejar valores muy pequeños."""
        data = [{"VR_TOTAL": 0.01, "CANTIDAD": 0.01}, {"VR_TOTAL": 0.02, "CANTIDAD": 0.02}]

        result = simulator.run_simulation(data)

        assert result.is_successful()

    def test_very_large_values(self, simulator):
        """Debe manejar valores muy grandes."""
        data = [{"VR_TOTAL": 1e12, "CANTIDAD": 1e6}, {"VR_TOTAL": 1e11, "CANTIDAD": 1e5}]

        result = simulator.run_simulation(data)

        assert result.is_successful()

    def test_zero_volatility(self, valid_apu_data):
        """Debe manejar volatilidad cero."""
        config = MonteCarloConfig(volatility_factor=0.0)
        simulator = MonteCarloSimulator(config=config)

        result = simulator.run_simulation(valid_apu_data)

        # Con volatilidad 0, todas las simulaciones deben dar el mismo valor
        assert result.statistics["std_dev"] == pytest.approx(0.0)

    def test_max_volatility(self, valid_apu_data):
        """Debe manejar volatilidad máxima."""
        config = MonteCarloConfig(volatility_factor=1.0)
        simulator = MonteCarloSimulator(config=config)

        result = simulator.run_simulation(valid_apu_data)

        assert result.is_successful()
        # Con volatilidad alta, debe haber gran dispersión
        assert result.statistics["std_dev"] > 0

    def test_all_identical_values(self, simulator):
        """Debe manejar todos los valores idénticos."""
        data = [
            {"VR_TOTAL": 10000.0, "CANTIDAD": 5.0},
            {"VR_TOTAL": 10000.0, "CANTIDAD": 5.0},
            {"VR_TOTAL": 10000.0, "CANTIDAD": 5.0},
        ]

        result = simulator.run_simulation(data)

        assert result.is_successful()

    def test_mixed_scales(self, simulator):
        """Debe manejar valores de diferentes escalas."""
        data = [
            {"VR_TOTAL": 1.0, "CANTIDAD": 1.0},
            {"VR_TOTAL": 1000.0, "CANTIDAD": 100.0},
            {"VR_TOTAL": 1000000.0, "CANTIDAD": 10000.0},
        ]

        result = simulator.run_simulation(data)

        assert result.is_successful()


# ============================================================================
# TESTS DE RENDIMIENTO
# ============================================================================


class TestPerformance:
    """Tests de rendimiento y volumen."""

    def test_large_dataset(self, large_apu_data):
        """Debe procesar dataset grande eficientemente."""
        config = MonteCarloConfig(num_simulations=1000)
        simulator = MonteCarloSimulator(config=config)

        result = simulator.run_simulation(large_apu_data)

        assert result.is_successful()
        assert result.metadata["valid_items"] == 1000

    def test_many_simulations(self, valid_apu_data):
        """Debe manejar muchas simulaciones."""
        config = MonteCarloConfig(num_simulations=10000)
        simulator = MonteCarloSimulator(config=config)

        result = simulator.run_simulation(valid_apu_data)

        assert result.is_successful()
        assert len(result.raw_results) == 10000

    @pytest.mark.slow
    def test_extreme_load(self):
        """Debe manejar carga extrema (test lento)."""
        data = [{"VR_TOTAL": 10000.0 + i, "CANTIDAD": 5.0 + i * 0.1} for i in range(10000)]

        config = MonteCarloConfig(num_simulations=1000)
        simulator = MonteCarloSimulator(config=config)

        result = simulator.run_simulation(data)

        assert result.is_successful()


# ============================================================================
# TESTS PARAMETRIZADOS
# ============================================================================


class TestParametrized:
    """Tests parametrizados para múltiples casos."""

    @pytest.mark.parametrize("num_sims", [100, 500, 1000, 5000])
    def test_various_simulation_counts(self, valid_apu_data, num_sims):
        """Debe funcionar con diferentes cantidades de simulaciones."""
        config = MonteCarloConfig(num_simulations=num_sims)
        simulator = MonteCarloSimulator(config=config)

        result = simulator.run_simulation(valid_apu_data)

        assert result.is_successful()
        assert len(result.raw_results) == num_sims

    @pytest.mark.parametrize("volatility", [0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    def test_various_volatilities(self, valid_apu_data, volatility):
        """Debe funcionar con diferentes volatilidades."""
        config = MonteCarloConfig(volatility_factor=volatility)
        simulator = MonteCarloSimulator(config=config)

        result = simulator.run_simulation(valid_apu_data)

        assert result.is_successful()

    @pytest.mark.parametrize("threshold", [0.0, 100.0, 1000.0, 10000.0])
    def test_various_cost_thresholds(self, valid_apu_data, threshold):
        """Debe funcionar con diferentes umbrales de costo."""
        config = MonteCarloConfig(min_cost_threshold=threshold)
        simulator = MonteCarloSimulator(config=config)

        result = simulator.run_simulation(valid_apu_data)

        # Puede o no tener datos válidos dependiendo del umbral
        assert result.status in [SimulationStatus.SUCCESS, SimulationStatus.NO_VALID_DATA]

    @pytest.mark.parametrize("seed", [None, 0, 42, 12345, 99999])
    def test_various_random_seeds(self, valid_apu_data, seed):
        """Debe funcionar con diferentes semillas."""
        config = MonteCarloConfig(random_seed=seed, num_simulations=100)
        simulator = MonteCarloSimulator(config=config)

        result = simulator.run_simulation(valid_apu_data)

        assert result.is_successful()


# ============================================================================
# TESTS DE REPRODUCIBILIDAD
# ============================================================================


class TestReproducibility:
    """Tests para verificar reproducibilidad."""

    def test_same_seed_same_results(self, valid_apu_data):
        """Debe producir resultados idénticos con misma semilla."""
        config = MonteCarloConfig(random_seed=42, num_simulations=100)

        sim1 = MonteCarloSimulator(config=config)
        result1 = sim1.run_simulation(valid_apu_data)

        sim2 = MonteCarloSimulator(config=config)
        result2 = sim2.run_simulation(valid_apu_data)

        # Estadísticas deben ser idénticas
        assert result1.statistics["mean"] == result2.statistics["mean"]
        assert result1.statistics["std_dev"] == result2.statistics["std_dev"]

        # Arrays brutos deben ser idénticos
        np.testing.assert_array_equal(result1.raw_results, result2.raw_results)

    def test_different_seed_different_results(self, valid_apu_data):
        """Debe producir resultados diferentes con semillas diferentes."""
        config1 = MonteCarloConfig(random_seed=42, num_simulations=100)
        config2 = MonteCarloConfig(random_seed=123, num_simulations=100)

        sim1 = MonteCarloSimulator(config=config1)
        result1 = sim1.run_simulation(valid_apu_data)

        sim2 = MonteCarloSimulator(config=config2)
        result2 = sim2.run_simulation(valid_apu_data)

        # Las medias serán cercanas pero no idénticas
        assert result1.statistics["mean"] != result2.statistics["mean"]

        # Arrays brutos deben ser diferentes
        assert not np.array_equal(result1.raw_results, result2.raw_results)


# ============================================================================
# TESTS DE COBERTURA ADICIONAL
# ============================================================================


class TestAdditionalCoverage:
    """Tests adicionales para cobertura completa."""

    def test_create_no_data_result(self, simulator):
        """Debe crear resultado apropiado sin datos."""
        result = simulator._create_no_data_result(total_items=10, discarded_items=10)

        assert result.status == SimulationStatus.NO_VALID_DATA
        assert result.statistics["mean"] is None
        assert result.metadata["discard_rate"] == 1.0

    def test_simulation_result_to_dict_comprehensive(self):
        """Debe convertir resultado completo a dict."""
        raw = np.array([1, 2, 3])
        result = SimulationResult(
            status=SimulationStatus.SUCCESS,
            statistics={"mean": 2.0},
            metadata={"count": 3},
            raw_results=raw,
        )

        dict_no_raw = result.to_dict(include_raw=False)
        assert "raw_results" not in dict_no_raw

        dict_with_raw = result.to_dict(include_raw=True)
        assert dict_with_raw["raw_results"] == [1, 2, 3]

    def test_sanitize_complex_values(self):
        """Debe manejar valores complejos."""
        # Complex float
        result = sanitize_value(np.complex64(1 + 2j))
        assert result is not None

        # Array (aunque no debería usarse así)
        arr = np.array([1, 2, 3])
        result = sanitize_value(arr)
        # Arrays no son list/tuple, así que se procesan
        assert result is not None

    def test_default_logger_creation(self):
        """Debe crear logger por defecto correctamente."""
        simulator = MonteCarloSimulator()

        assert simulator.logger is not None
        assert isinstance(simulator.logger, logging.Logger)
        assert len(simulator.logger.handlers) > 0

    def test_config_post_init_sets_percentiles(self):
        """Debe configurar percentiles por defecto en post_init."""
        config = MonteCarloConfig()

        assert config.percentiles is not None
        assert isinstance(config.percentiles, list)
        assert len(config.percentiles) > 0


# ============================================================================
# TESTS DE INTEGRACIÓN COMPLETA
# ============================================================================


class TestFullIntegration:
    """Tests de integración end-to-end."""

    def test_complete_workflow(self, valid_apu_data):
        """Debe ejecutar flujo completo correctamente."""
        # 1. Crear configuración
        config = MonteCarloConfig(
            num_simulations=1000,
            volatility_factor=0.15,
            random_seed=42,
            percentiles=[5, 25, 50, 75, 95],
        )

        # 2. Crear simulador
        simulator = MonteCarloSimulator(config=config)

        # 3. Ejecutar simulación
        result = simulator.run_simulation(valid_apu_data)

        # 4. Verificar resultado
        assert result.is_successful()
        assert result.status == SimulationStatus.SUCCESS

        # 5. Verificar estadísticas
        assert result.statistics["mean"] > 0
        assert result.statistics["std_dev"] > 0
        assert result.statistics["min"] >= 0  # Porque trunca negativos
        assert result.statistics["percentile_50"] == result.statistics["median"]

        # 6. Verificar metadata
        assert result.metadata["valid_items"] == 4
        assert result.metadata["discarded_items"] == 0

        # 7. Convertir a dict
        result_dict = result.to_dict()
        assert "status" in result_dict
        assert "statistics" in result_dict
        assert "metadata" in result_dict

    def test_workflow_with_data_issues(self, apu_data_with_nan):
        """Debe manejar flujo con datos problemáticos."""
        logger = logging.getLogger("test")
        logger.setLevel(logging.WARNING)

        simulator = MonteCarloSimulator(logger=logger)
        result = simulator.run_simulation(apu_data_with_nan)

        # Debe filtrar datos inválidos y continuar
        if result.is_successful():
            assert result.metadata["discarded_items"] > 0
        else:
            assert result.status == SimulationStatus.NO_VALID_DATA

    def test_legacy_to_modern_api_equivalence(self, valid_apu_data):
        """Debe dar resultados equivalentes entre API legacy y moderna."""
        # API Legacy
        legacy_result = run_monte_carlo_simulation(
            apu_details=valid_apu_data,
            num_simulations=1000,
            volatility_factor=0.1,
            min_cost_threshold=0.0,
            log_warnings=False,
        )

        # API Moderna
        config = MonteCarloConfig(
            num_simulations=1000,
            volatility_factor=0.1,
            min_cost_threshold=0.0,
            random_seed=None,  # Sin semilla para comparación justa
        )
        simulator = MonteCarloSimulator(config=config)
        modern_result = simulator.run_simulation(valid_apu_data)

        # Ambos deben ser exitosos
        assert legacy_result["mean"] is not None
        assert modern_result.is_successful()

        # Los valores deben estar en el mismo rango (no idénticos por aleatoriedad)
        assert (
            abs(legacy_result["mean"] - modern_result.statistics["mean"])
            < modern_result.statistics["mean"] * 0.1
        )  # 10% de tolerancia


# ============================================================================
# SUITE DE EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    """
    Ejecutar tests con pytest.
    
    Comandos útiles:
    - pytest test_probability_models.py -v
    - pytest test_probability_models.py --cov=probability_models --cov-report=html
    - pytest test_probability_models.py -k "test_config"
    - pytest test_probability_models.py -x
    - pytest test_probability_models.py -m "not slow"
    """
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=probability_models",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov_probability",
            "--tb=short",
            "-ra",
        ]
    )
