"""
Suite de Pruebas Exhaustiva para el `DataFluxCondenser`.

Esta suite de pruebas verifica todos los aspectos del `DataFluxCondenser`,
asegurando su robustez, fiabilidad y comportamiento esperado bajo una amplia
variedad de escenarios.

Cobertura de Pruebas:
- **Inicialización:** Valida que el condensador se configure correctamente,
  incluyendo la gestión de configuraciones personalizadas y por defecto.
- **Validación de Entradas:** Comprueba que el manejo de rutas de archivos
  (existentes, inexistentes, directorios, etc.) sea seguro.
- **Flujo de Procesamiento:** Utiliza `mocks` para aislar y probar cada
  etapa del pipeline (`_absorb_and_filter`, `_rectify_signal`), verificando
  la correcta interacción con `ReportParserCrudo` y `APUProcessor`.
- **Motor de Física:** Pruebas unitarias para `FluxPhysicsEngine`, asegurando
  que los cálculos de saturación y los estados de estabilidad sean precisos,
  ahora incluyendo el cálculo de voltaje Flyback.
- **Manejo de Errores:** Confirma que las excepciones personalizadas
  (`InvalidInputError`, `ProcessingError`) se lancen y propaguen
  correctamente.
- **Casos Límite (Edge Cases):** Evalúa el comportamiento con datasets
  vacíos, muy grandes, y con datos problemáticos (e.g., columnas nulas).
- **Telemetría y Logging:** Verifica que los mensajes de log, especialmente
  los de telemetría física y advertencias de flyback, se generen como se espera.

Metodología:
- **Fixtures de Pytest:** Se utilizan extensivamente para crear un entorno de
  pruebas limpio y reutilizable para cada test.
- **Mocks y Patching:** La librería `unittest.mock` se usa para aislar el
  `DataFluxCondenser` de sus dependencias, permitiendo pruebas unitarias
  enfocadas en su lógica de orquestación.
- **Pruebas Parametrizadas:** `pytest.mark.parametrize` se emplea para
  reducir código duplicado y probar múltiples variaciones de un mismo
  escenario de forma eficiente.
"""
import logging
import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

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
)

# ==================== FIXTURES ====================

@pytest.fixture
def valid_config() -> Dict[str, Any]:
    """
    Proporciona una configuración de sistema válida y completa.

    Returns:
        Dict[str, Any]: Un diccionario con las claves requeridas para
            simular una configuración real.
    """
    return {
        'parser_settings': {
            'delimiter': ',',
            'encoding': 'utf-8'
        },
        'processor_settings': {
            'validate_types': True,
            'skip_empty': False
        },
        'additional_key': 'value'
    }


@pytest.fixture
def valid_profile() -> Dict[str, Any]:
    """
    Proporciona un perfil de procesamiento válido y completo.

    Returns:
        Dict[str, Any]: Un diccionario que simula un perfil de archivo
            con mapeo de columnas y reglas de validación.
    """
    return {
        'columns_mapping': {
            'cod_insumo': 'codigo',
            'descripcion': 'desc'
        },
        'validation_rules': {
            'required_fields': ['codigo', 'cantidad']
        },
        'extra_config': 'data'
    }


@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """
    Proporciona una configuración mínima para probar el modo tolerante.

    Returns:
        Dict[str, Any]: Diccionario sin las claves requeridas.
    """
    return {'some_key': 'value'}


@pytest.fixture
def minimal_profile() -> Dict[str, Any]:
    """
    Proporciona un perfil mínimo para probar el modo tolerante.

    Returns:
        Dict[str, Any]: Diccionario sin las claves requeridas.
    """
    return {'another_key': 'data'}


@pytest.fixture
def custom_condenser_config() -> CondenserConfig:
    """
    Crea una instancia de `CondenserConfig` con valores no predeterminados.

    Returns:
        CondenserConfig: Configuración con umbrales y niveles de log
            personalizados para pruebas específicas.
    """
    return CondenserConfig(
        min_records_threshold=5,
        enable_strict_validation=True,
        log_level="DEBUG"
    )


@pytest.fixture
def default_condenser_config() -> CondenserConfig:
    """
    Crea una instancia de `CondenserConfig` con sus valores por defecto.

    Returns:
        CondenserConfig: Configuración por defecto.
    """
    return CondenserConfig()


@pytest.fixture
def condenser(valid_config, valid_profile) -> DataFluxCondenser:
    """
    Crea una instancia estándar del `DataFluxCondenser` para pruebas.

    Utiliza la configuración y perfil válidos.

    Args:
        valid_config (Dict[str, Any]): Fixture con configuración válida.
        valid_profile (Dict[str, Any]): Fixture con perfil válido.

    Returns:
        DataFluxCondenser: Una instancia lista para usar.
    """
    return DataFluxCondenser(valid_config, valid_profile)


@pytest.fixture
def sample_raw_records() -> List[Dict[str, Any]]:
    """
    Genera una lista de registros crudos de ejemplo.

    Returns:
        List[Dict[str, Any]]: Datos que simulan la salida del
            `ReportParserCrudo`.
    """
    return [
        {'codigo': 'A001', 'cantidad': 10, 'precio': 100.0},
        {'codigo': 'A002', 'cantidad': 5, 'precio': 50.0},
        {'codigo': 'A003', 'cantidad': 8, 'precio': 80.0}
    ]


@pytest.fixture
def sample_parse_cache() -> Dict[str, Any]:
    """
    Genera un diccionario de caché de parseo de ejemplo.

    Returns:
        Dict[str, Any]: Metadatos que simulan la caché del
            `ReportParserCrudo`.
    """
    return {
        'total_lines': 100,
        'skipped_lines': 5,
        'headers': ['codigo', 'cantidad', 'precio'],
        'file_encoding': 'utf-8'
    }


@pytest.fixture
def sample_dataframe(sample_raw_records) -> pd.DataFrame:
    """
    Crea un DataFrame de pandas a partir de los registros de ejemplo.

    Args:
        sample_raw_records (List[Dict[str, Any]]): Fixture con datos crudos.

    Returns:
        pd.DataFrame: DataFrame que simula la salida final del condensador.
    """
    return pd.DataFrame(sample_raw_records)


@pytest.fixture
def mock_csv_file(tmp_path) -> Path:
    """
    Crea un archivo CSV temporal en disco para pruebas de lectura.

    Utiliza el fixture `tmp_path` de pytest para gestionar archivos temporales.

    Args:
        tmp_path (Path): El directorio temporal proporcionado por pytest.

    Returns:
        Path: La ruta al archivo CSV temporal creado.
    """
    file_path = tmp_path / "test_data.csv"
    file_path.write_text(
        "codigo,cantidad,precio\n"
        "A001,10,100.0\n"
        "A002,5,50.0\n"
    )
    return file_path


# ==================== TESTS DE INICIALIZACIÓN ====================

class TestInitialization:
    """Grupo de pruebas para la inicialización del `DataFluxCondenser`."""

    def test_init_with_valid_params(self, valid_config, valid_profile):
        """
        Verifica que el condensador se inicialice correctamente con parámetros
        válidos y completos.
        """
        condenser = DataFluxCondenser(valid_config, valid_profile)

        assert condenser.config == valid_config
        assert condenser.profile == valid_profile
        assert isinstance(condenser.condenser_config, CondenserConfig)
        assert condenser.logger.name == "DataFluxCondenser"

    def test_init_with_custom_condenser_config(
        self,
        valid_config,
        valid_profile,
        custom_condenser_config
    ):
        """
        Asegura que se pueda pasar una `CondenserConfig` personalizada y que
        sus valores se apliquen correctamente.
        """
        condenser = DataFluxCondenser(
            valid_config,
            valid_profile,
            custom_condenser_config
        )

        assert condenser.condenser_config.min_records_threshold == 5
        assert condenser.condenser_config.enable_strict_validation is True
        assert condenser.condenser_config.log_level == "DEBUG"

    def test_init_with_default_condenser_config(self, valid_config, valid_profile):
        """
        Verifica que se utilice la `CondenserConfig` por defecto si no se
        proporciona una explícitamente.
        """
        condenser = DataFluxCondenser(valid_config, valid_profile)

        assert condenser.condenser_config.min_records_threshold == 1
        assert condenser.condenser_config.enable_strict_validation is True
        assert condenser.condenser_config.log_level == "INFO"
        # Verificar defaults RLC
        assert condenser.condenser_config.system_capacitance == 5000.0
        assert condenser.condenser_config.system_inductance == 2.0

    def test_init_with_invalid_config_type(self, valid_profile):
        """
        Prueba que la inicialización falle con un `InvalidInputError` si el
        parámetro `config` no es un diccionario.
        """
        with pytest.raises(InvalidInputError, match="deben ser diccionarios válidos"):
            DataFluxCondenser("invalid_config", valid_profile)

    def test_init_with_invalid_profile_type(self, valid_config):
        """
        Prueba que la inicialización falle con `InvalidInputError` si `profile`
        no es un diccionario.
        """
        with pytest.raises(InvalidInputError, match="deben ser diccionarios válidos"):
            DataFluxCondenser(valid_config, None)

    def test_init_with_none_params(self):
        """
        Asegura que la inicialización falle si `config` y `profile` son `None`.
        """
        with pytest.raises(InvalidInputError):
            DataFluxCondenser(None, None)

    def test_init_with_minimal_config_warns(
        self,
        minimal_config,
        minimal_profile,
        caplog
    ):
        """
        Verifica el modo tolerante: si faltan claves requeridas en config o
        profile, se debe registrar una advertencia en lugar de fallar.
        """
        with caplog.at_level(logging.WARNING):
            condenser = DataFluxCondenser(minimal_config, minimal_profile)

        assert condenser is not None
        assert "Claves faltantes en config" in caplog.text
        assert "Claves faltantes en profile" in caplog.text

    def test_logger_configuration(self, valid_config, valid_profile):
        """
        Comprueba que el nivel del logger se configure correctamente según
        lo especificado en `CondenserConfig`.
        """
        custom_config = CondenserConfig(log_level="DEBUG")
        condenser = DataFluxCondenser(valid_config, valid_profile, custom_config)

        assert condenser.logger.level == logging.DEBUG

    def test_physics_engine_initialization(self, valid_config, valid_profile):
        """Verifica que el motor de física se inicialice con parámetros RLC correctos."""
        condenser = DataFluxCondenser(valid_config, valid_profile)

        assert isinstance(condenser.physics, FluxPhysicsEngine)
        assert condenser.physics.L == 2.0  # Default inductance


# ==================== TESTS DE VALIDACIÓN DE ARCHIVO ====================

class TestInputFileValidation:
    """Grupo de pruebas para el método de validación de archivos de entrada."""

    def test_validate_existing_csv_file(self, condenser, mock_csv_file):
        """
        Verifica que un archivo `.csv` existente y válido pase la validación.
        """
        result = condenser._validate_input_file(str(mock_csv_file))

        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == '.csv'

    def test_validate_txt_file(self, condenser, tmp_path):
        """Asegura que los archivos `.txt` también sean aceptados."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("test data")

        result = condenser._validate_input_file(str(txt_file))
        assert result.suffix == '.txt'

    def test_validate_nonexistent_file(self, condenser):
        """
        Prueba que se lance `InvalidInputError` si el archivo no existe.
        """
        with pytest.raises(InvalidInputError, match="El archivo no existe"):
            condenser._validate_input_file("/path/to/nonexistent.csv")

    def test_validate_directory_path(self, condenser, tmp_path):
        """
        Prueba que se lance `InvalidInputError` si la ruta es un directorio.
        """
        with pytest.raises(InvalidInputError, match="La ruta no es un archivo"):
            condenser._validate_input_file(str(tmp_path))

    def test_validate_empty_string(self, condenser):
        """
        Asegura que una cadena vacía como ruta de archivo cause un error.
        """
        with pytest.raises(InvalidInputError, match="debe ser una cadena no vacía"):
            condenser._validate_input_file("")

    def test_validate_none_path(self, condenser):
        """Asegura que `None` como ruta de archivo cause un error."""
        with pytest.raises(InvalidInputError, match="debe ser una cadena no vacía"):
            condenser._validate_input_file(None)

    def test_validate_unusual_extension_warns(self, condenser, tmp_path, caplog):
        """
        Verifica que se registre una advertencia si el archivo tiene una
        extensión no estándar (e.g., `.xlsx`).
        """
        unusual_file = tmp_path / "data.xlsx"
        unusual_file.write_text("data")

        with caplog.at_level(logging.WARNING):
            condenser._validate_input_file(str(unusual_file))

        assert "Extensión inusual detectada" in caplog.text

    def test_validate_returns_path_object(self, condenser, mock_csv_file):
        """
        Confirma que el método de validación retorne un objeto `pathlib.Path`.
        """
        result = condenser._validate_input_file(str(mock_csv_file))
        assert isinstance(result, Path)


# ==================== TESTS DE FILTRADO (ABSORB AND FILTER) ====================

class TestAbsorbAndFilter:
    """
    Grupo de pruebas para el método `_absorb_and_filter`, que interactúa
    con `ReportParserCrudo`.
    """

    @patch('app.flux_condenser.ReportParserCrudo')
    def test_successful_parsing(
        self,
        mock_parser_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache
    ):
        """
        Simula un parseo exitoso y verifica que los datos se estructuren
        correctamente en `ParsedData`.
        """
        # Configurar mock
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        # Ejecutar
        result = condenser._absorb_and_filter(mock_csv_file)

        # Verificar
        assert isinstance(result, ParsedData)
        assert result.raw_records == sample_raw_records
        assert result.parse_cache == sample_parse_cache

        # Verificar que el mock fue llamado con los argumentos de palabra clave correctos
        mock_parser_class.assert_called_once()
        args, kwargs = mock_parser_class.call_args
        assert args[0] == str(mock_csv_file)
        assert kwargs.get('profile') == condenser.profile
        assert isinstance(kwargs.get('config'), dict)

    @patch('app.flux_condenser.ReportParserCrudo')
    def test_parser_returns_none_records(
        self,
        mock_parser_class,
        condenser,
        mock_csv_file,
        caplog
    ):
        """Debe manejar cuando parser retorna None en raw_records."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = None
        mock_parser.get_parse_cache.return_value = {}
        mock_parser_class.return_value = mock_parser

        with caplog.at_level(logging.WARNING):
            result = condenser._absorb_and_filter(mock_csv_file)

        assert result.raw_records == []
        assert "Parser retornó None" in caplog.text

    @patch('app.flux_condenser.ReportParserCrudo')
    def test_parser_returns_none_cache(
        self,
        mock_parser_class,
        condenser,
        mock_csv_file,
        caplog
    ):
        """Debe manejar cuando parser retorna None en cache."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = []
        mock_parser.get_parse_cache.return_value = None
        mock_parser_class.return_value = mock_parser

        with caplog.at_level(logging.WARNING):
            result = condenser._absorb_and_filter(mock_csv_file)

        assert result.parse_cache == {}
        assert "Cache retornó None" in caplog.text

    @patch('app.flux_condenser.ReportParserCrudo')
    def test_parser_raises_exception(
        self,
        mock_parser_class,
        condenser,
        mock_csv_file
    ):
        """Debe propagar excepciones del parser como ProcessingError."""
        mock_parser_class.side_effect = ValueError("Parser error")

        with pytest.raises(ProcessingError, match="Error durante el filtrado"):
            condenser._absorb_and_filter(mock_csv_file)

    @patch('app.flux_condenser.ReportParserCrudo')
    def test_parser_method_fails(
        self,
        mock_parser_class,
        condenser,
        mock_csv_file
    ):
        """Debe manejar cuando falla parse_to_raw()."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.side_effect = IOError("File read error")
        mock_parser_class.return_value = mock_parser

        with pytest.raises(ProcessingError, match="Error durante el filtrado"):
            condenser._absorb_and_filter(mock_csv_file)


# ==================== TESTS DE VALIDACIÓN DE DATOS PARSEADOS ====================

class TestValidateParsedData:
    """Pruebas de validación de datos parseados."""

    def test_valid_parsed_data(self, condenser, sample_raw_records, sample_parse_cache):
        """Debe validar datos correctos."""
        parsed_data = ParsedData(sample_raw_records, sample_parse_cache)

        result = condenser._validate_parsed_data(parsed_data)

        assert result is True

    def test_empty_records_below_threshold(self, condenser):
        """Debe retornar False si registros vacíos están bajo el umbral."""
        parsed_data = ParsedData([], {})

        result = condenser._validate_parsed_data(parsed_data)

        assert result is False

    def test_records_below_custom_threshold(self, valid_config, valid_profile):
        """Debe respetar umbral personalizado."""
        custom_config = CondenserConfig(min_records_threshold=5)
        condenser = DataFluxCondenser(valid_config, valid_profile, custom_config)

        parsed_data = ParsedData([{'a': 1}, {'b': 2}], {})  # Solo 2 registros

        result = condenser._validate_parsed_data(parsed_data)

        assert result is False

    def test_records_meet_custom_threshold(self, valid_config, valid_profile):
        """Debe pasar si cumple el umbral personalizado."""
        custom_config = CondenserConfig(min_records_threshold=3)
        condenser = DataFluxCondenser(valid_config, valid_profile, custom_config)

        records = [{'a': i} for i in range(5)]  # 5 registros
        parsed_data = ParsedData(records, {})

        result = condenser._validate_parsed_data(parsed_data)

        assert result is True

    def test_invalid_raw_records_type(self, condenser):
        """Debe fallar si raw_records no es lista."""
        parsed_data = ParsedData(
            raw_records="not a list",  # type: ignore
            parse_cache={}
        )

        with pytest.raises(ProcessingError, match="raw_records debe ser lista"):
            condenser._validate_parsed_data(parsed_data)

    def test_invalid_parse_cache_type(self, condenser):
        """Debe fallar si parse_cache no es dict."""
        parsed_data = ParsedData(
            raw_records=[],
            parse_cache="not a dict"  # type: ignore
        )

        with pytest.raises(ProcessingError, match="parse_cache debe ser dict"):
            condenser._validate_parsed_data(parsed_data)

    def test_warning_on_insufficient_records(self, condenser, caplog):
        """Debe advertir cuando registros son insuficientes."""
        parsed_data = ParsedData([], {})

        with caplog.at_level(logging.WARNING):
            condenser._validate_parsed_data(parsed_data)

        assert "Registros insuficientes" in caplog.text


# ==================== TESTS DE RECTIFICACIÓN ====================

class TestRectifySignal:
    """Pruebas del proceso de rectificación con APUProcessor."""

    @patch('app.flux_condenser.APUProcessor')
    def test_successful_processing(
        self,
        mock_processor_class,
        condenser,
        sample_raw_records,
        sample_parse_cache,
        sample_dataframe
    ):
        """Debe procesar correctamente y retornar DataFrame."""
        parsed_data = ParsedData(sample_raw_records, sample_parse_cache)

        mock_processor = Mock()
        mock_processor.process_all.return_value = sample_dataframe
        mock_processor_class.return_value = mock_processor

        result = condenser._rectify_signal(parsed_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)

        # Verificar que el mock fue llamado con los argumentos correctos
        mock_processor_class.assert_called_once_with(
            config=condenser.config,
            profile=condenser.profile,
            parse_cache=sample_parse_cache
        )

        # Verificar que raw_records se asignó antes de llamar a process_all
        assert mock_processor.raw_records == sample_raw_records
        mock_processor.process_all.assert_called_once()

    @patch('app.flux_condenser.APUProcessor')
    def test_processor_returns_wrong_type(
        self,
        mock_processor_class,
        condenser,
        sample_raw_records,
        sample_parse_cache
    ):
        """Debe fallar si APUProcessor no retorna DataFrame."""
        parsed_data = ParsedData(sample_raw_records, sample_parse_cache)

        mock_processor = Mock()
        mock_processor.process_all.return_value = "not a dataframe"
        mock_processor_class.return_value = mock_processor

        with pytest.raises(ProcessingError, match="debe retornar DataFrame"):
            condenser._rectify_signal(parsed_data)

    @patch('app.flux_condenser.APUProcessor')
    def test_processor_raises_exception(
        self,
        mock_processor_class,
        condenser,
        sample_raw_records,
        sample_parse_cache
    ):
        """Debe propagar excepciones como ProcessingError."""
        parsed_data = ParsedData(sample_raw_records, sample_parse_cache)

        mock_processor_class.side_effect = RuntimeError("Processing failed")

        with pytest.raises(ProcessingError, match="Error durante la rectificación"):
            condenser._rectify_signal(parsed_data)

    @patch('app.flux_condenser.APUProcessor')
    def test_processor_method_fails(
        self,
        mock_processor_class,
        condenser,
        sample_raw_records,
        sample_parse_cache
    ):
        """Debe manejar cuando process_all() falla."""
        parsed_data = ParsedData(sample_raw_records, sample_parse_cache)

        mock_processor = Mock()
        mock_processor.process_all.side_effect = KeyError("Missing column")
        mock_processor_class.return_value = mock_processor

        with pytest.raises(ProcessingError, match="Error durante la rectificación"):
            condenser._rectify_signal(parsed_data)


# ==================== TESTS DE VALIDACIÓN DE SALIDA ====================

class TestValidateOutput:
    """Pruebas de validación del DataFrame de salida."""

    def test_valid_dataframe(self, condenser, sample_dataframe):
        """Debe validar DataFrame correcto sin errores."""
        condenser._validate_output(sample_dataframe)  # No debe lanzar excepción

    def test_empty_dataframe_warns(self, condenser, caplog):
        """Debe advertir sobre DataFrame vacío."""
        df_empty = pd.DataFrame()

        with caplog.at_level(logging.WARNING):
            condenser._validate_output(df_empty)

        assert "DataFrame vacío generado" in caplog.text

    def test_dataframe_with_null_columns_warns(self, condenser, caplog):
        """Debe advertir sobre columnas completamente nulas."""
        df = pd.DataFrame({
            'col_valid': [1, 2, 3],
            'col_null': [None, None, None]
        })

        with caplog.at_level(logging.WARNING):
            condenser._validate_output(df)

        assert "Columnas completamente nulas" in caplog.text
        assert "col_null" in caplog.text

    def test_invalid_type_raises_error(self, condenser):
        """Debe fallar si la salida no es DataFrame."""
        with pytest.raises(ProcessingError, match="La salida debe ser DataFrame"):
            condenser._validate_output("not a dataframe")

    def test_none_output_raises_error(self, condenser):
        """Debe fallar si la salida es None."""
        with pytest.raises(ProcessingError, match="La salida debe ser DataFrame"):
            condenser._validate_output(None)

    def test_strict_validation_disabled(self, valid_config, valid_profile):
        """Debe omitir validaciones estrictas si están deshabilitadas."""
        custom_config = CondenserConfig(enable_strict_validation=False)
        condenser = DataFluxCondenser(valid_config, valid_profile, custom_config)

        df_empty = pd.DataFrame()

        # No debe lanzar advertencias
        condenser._validate_output(df_empty)


# ==================== TESTS: FluxPhysicsEngine (RLC) ====================

class TestFluxPhysicsEngine:
    """Pruebas unitarias para el motor de física RLC."""

    @pytest.fixture
    def physics_engine(self):
        """Instancia del motor con valores base RLC."""
        return FluxPhysicsEngine(capacitance=5000.0, resistance=10.0, inductance=2.0)

    @pytest.mark.parametrize("total, hits, expected_saturation", [
        (100, 100, 0.0020),  # Complejidad 0, carga mínima -> Saturación baja
        (100, 0, 0.0003),    # Complejidad 1 (max), carga mínima -> Saturación muy baja (carga lenta)
        (10000, 5000, 0.0555), # Carga alta, complejidad media -> Saturación moderada
    ])
    def test_calculate_metrics_saturation(self, physics_engine, total, hits, expected_saturation):
        """Debe calcular saturación correctamente (approx)."""
        metrics = physics_engine.calculate_metrics(total, hits)
        # La saturación es sensible a la exponencial, validamos rangos o approx
        assert metrics["saturation"] == pytest.approx(expected_saturation, abs=0.001)

    def test_flyback_voltage_calculation(self, physics_engine):
        """Debe calcular voltaje de flyback cuando hay caída de calidad."""
        # Caso ideal: 100% hits -> i=1.0, delta_i=0 -> V=0
        metrics_ideal = physics_engine.calculate_metrics(100, 100)
        assert metrics_ideal["flyback_voltage"] == 0.0

        # Caso colapso: 0% hits -> i=0.0, delta_i=1.0 -> V alto
        metrics_crash = physics_engine.calculate_metrics(100, 0)
        assert metrics_crash["flyback_voltage"] > 0.0

        # Validación numérica manual:
        # dt = ln(1 + 100) ≈ 4.615
        # delta_i = 1.0
        # V = L * (1.0 / 4.615) = 2.0 * 0.216 ≈ 0.433
        expected_v = 2.0 * (1.0 / math.log1p(100))
        assert metrics_crash["flyback_voltage"] == pytest.approx(expected_v, abs=0.001)

    def test_get_system_diagnosis_flyback(self, physics_engine):
        """Debe retornar diagnósticos de peligro inductivo."""
        # Peligro Alto (>0.5)
        metrics_high = {"saturation": 0.1, "complexity": 0.5, "flyback_voltage": 0.6}
        diag = physics_engine.get_system_diagnosis(metrics_high)
        assert "⚡ PELIGRO" in diag

        # Advertencia (>0.1)
        metrics_med = {"saturation": 0.1, "complexity": 0.5, "flyback_voltage": 0.2}
        diag = physics_engine.get_system_diagnosis(metrics_med)
        assert "⚠️ ADVERTENCIA" in diag

    def test_get_system_diagnosis_saturation(self, physics_engine):
        """Debe retornar diagnósticos de flujo normales si el voltaje es bajo."""
        # Flujo Laminar
        metrics_lam = {"saturation": 0.2, "complexity": 0.1, "flyback_voltage": 0.0}
        assert "FLUJO LAMINAR" in physics_engine.get_system_diagnosis(metrics_lam)

        # Flujo Turbulento
        metrics_tur = {"saturation": 0.8, "complexity": 0.1, "flyback_voltage": 0.0}
        assert "FLUJO TURBULENTO" in physics_engine.get_system_diagnosis(metrics_tur)


# ==================== TESTS DE INTEGRACIÓN (STABILIZE) ====================

class TestStabilize:
    """Pruebas de integración del flujo completo."""

    @patch('app.flux_condenser.APUProcessor')
    @patch('app.flux_condenser.ReportParserCrudo')
    def test_complete_successful_flow(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache,
        sample_dataframe
    ):
        """Debe ejecutar el flujo completo exitosamente."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = sample_dataframe
        mock_processor_class.return_value = mock_processor

        # Execute
        result = condenser.stabilize(str(mock_csv_file))

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_dataframe)
        mock_parser_class.assert_called_once()
        mock_processor_class.assert_called_once()

    @patch('app.flux_condenser.APUProcessor')
    @patch('app.flux_condenser.ReportParserCrudo')
    def test_telemetry_logging(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_dataframe,
        caplog
    ):
        """Debe registrar la telemetría física (RLC) durante un flujo exitoso."""
        # Setup: 100 registros crudos, 80 cacheados
        raw_records = [{'id': i} for i in range(100)]
        parse_cache = {f'line_{i}': 'data' for i in range(80)}

        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = raw_records
        mock_parser.get_parse_cache.return_value = parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = sample_dataframe
        mock_processor_class.return_value = mock_processor

        with caplog.at_level(logging.INFO):
            condenser.stabilize(str(mock_csv_file))

        # Validar logs actualizados
        assert "🧲 [TELEMETRÍA RLC]" in caplog.text
        assert "Pico(L):" in caplog.text
        assert "Sat(C):" in caplog.text

    @patch('app.flux_condenser.APUProcessor')
    @patch('app.flux_condenser.ReportParserCrudo')
    def test_flyback_diode_activation(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        caplog
    ):
        """Debe activar el diodo (warning) si hay un colapso de calidad."""
        # Usamos pocos registros para aumentar la pendiente di/dt
        # dt = log1p(10) ≈ 2.39, delta_i=1.0, V = 2.0 * (1/2.39) ≈ 0.83 > 0.8
        raw_records = [{'id': i} for i in range(10)]
        parse_cache = {} # 0 hits

        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = raw_records
        mock_parser.get_parse_cache.return_value = parse_cache
        mock_parser_class.return_value = mock_parser

        # Mocks para que pase la fase 2 aunque sea un desastre
        mock_processor = Mock()
        mock_processor.process_all.return_value = pd.DataFrame()
        mock_processor_class.return_value = mock_processor

        with caplog.at_level(logging.WARNING):
            condenser.stabilize(str(mock_csv_file))

        assert "🛡️ [DIODO FLYBACK ACTIVADO]" in caplog.text

    def test_nonexistent_file_raises_error(self, condenser):
        """Debe fallar con archivo inexistente."""
        with pytest.raises(InvalidInputError, match="El archivo no existe"):
            condenser.stabilize("/nonexistent/file.csv")

    @patch('app.flux_condenser.ReportParserCrudo')
    def test_empty_parsed_data_returns_empty_df(
        self,
        mock_parser_class,
        condenser,
        mock_csv_file,
        caplog
    ):
        """Debe retornar DataFrame vacío si no hay registros."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = []
        mock_parser.get_parse_cache.return_value = {}
        mock_parser_class.return_value = mock_parser

        with caplog.at_level(logging.WARNING):
            result = condenser.stabilize(str(mock_csv_file))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "Registros insuficientes" in caplog.text

    @patch('app.flux_condenser.ReportParserCrudo')
    def test_parser_failure_raises_processing_error(
        self,
        mock_parser_class,
        condenser,
        mock_csv_file
    ):
        """Debe propagar errores del parser."""
        mock_parser_class.side_effect = IOError("Read error")

        with pytest.raises(ProcessingError, match="Error durante el filtrado"):
            condenser.stabilize(str(mock_csv_file))

    @patch('app.flux_condenser.APUProcessor')
    @patch('app.flux_condenser.ReportParserCrudo')
    def test_processor_failure_raises_error(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache
    ):
        """Debe propagar errores del procesador."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor_class.side_effect = ValueError("Processing error")

        with pytest.raises(ProcessingError, match="Error durante la rectificación"):
            condenser.stabilize(str(mock_csv_file))

    @patch('app.flux_condenser.APUProcessor')
    @patch('app.flux_condenser.ReportParserCrudo')
    def test_unexpected_error_wrapped(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache
    ):
        """Debe envolver errores inesperados en ProcessingError."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.side_effect = ZeroDivisionError("Unexpected")
        mock_processor_class.return_value = mock_processor

        match_str = "Error durante la rectificación con APUProcessor: Unexpected"
        with pytest.raises(ProcessingError, match=match_str):
            condenser.stabilize(str(mock_csv_file))

    @patch('app.flux_condenser.APUProcessor')
    @patch('app.flux_condenser.ReportParserCrudo')
    def test_logging_messages(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache,
        sample_dataframe,
        caplog
    ):
        """Debe registrar mensajes apropiados durante el flujo."""
        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = sample_raw_records
        mock_parser.get_parse_cache.return_value = sample_parse_cache
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = sample_dataframe
        mock_processor_class.return_value = mock_processor

        with caplog.at_level(logging.INFO):
            condenser.stabilize(str(mock_csv_file))

        assert "⚡ [FÍSICA] Energizando circuito" in caplog.text
        assert "✅ [ÉXITO] Descarga estabilizada" in caplog.text


# ==================== TESTS DE ESTADÍSTICAS ====================

class TestGetProcessingStats:
    """Pruebas del método de estadísticas."""

    def test_returns_valid_structure(self, condenser):
        """Debe retornar estructura válida de estadísticas."""
        stats = condenser.get_processing_stats()

        assert 'condenser_config' in stats
        assert 'config_keys' in stats
        assert 'profile_keys' in stats

    def test_condenser_config_in_stats(self, valid_config, valid_profile):
        """Debe incluir configuración del condensador."""
        custom_config = CondenserConfig(
            min_records_threshold=10,
            enable_strict_validation=False
        )
        condenser = DataFluxCondenser(valid_config, valid_profile, custom_config)

        stats = condenser.get_processing_stats()

        assert stats['condenser_config']['min_records_threshold'] == 10
        assert stats['condenser_config']['strict_validation'] is False

    def test_config_keys_listed(self, condenser):
        """Debe listar las claves de config."""
        stats = condenser.get_processing_stats()

        assert isinstance(stats['config_keys'], list)
        assert 'parser_settings' in stats['config_keys']
        assert 'processor_settings' in stats['config_keys']

    def test_profile_keys_listed(self, condenser):
        """Debe listar las claves de profile."""
        stats = condenser.get_processing_stats()

        assert isinstance(stats['profile_keys'], list)
        assert 'columns_mapping' in stats['profile_keys']
        assert 'validation_rules' in stats['profile_keys']


# ==================== TESTS DE CONFIGURACIÓN ====================

class TestCondenserConfig:
    """Pruebas de la clase CondenserConfig."""

    def test_default_values(self):
        """Debe tener valores por defecto correctos."""
        config = CondenserConfig()

        assert config.min_records_threshold == 1
        assert config.enable_strict_validation is True
        assert config.log_level == "INFO"

    def test_custom_values(self):
        """Debe aceptar valores personalizados."""
        config = CondenserConfig(
            min_records_threshold=100,
            enable_strict_validation=False,
            log_level="ERROR"
        )

        assert config.min_records_threshold == 100
        assert config.enable_strict_validation is False
        assert config.log_level == "ERROR"

    def test_immutability(self):
        """Debe ser inmutable (frozen)."""
        config = CondenserConfig()

        with pytest.raises(AttributeError):
            config.min_records_threshold = 999

    def test_is_dataclass(self):
        """Debe ser un dataclass."""
        from dataclasses import is_dataclass
        assert is_dataclass(CondenserConfig)


# ==================== TESTS DE EXCEPCIONES PERSONALIZADAS ====================

class TestCustomExceptions:
    """Pruebas de las excepciones personalizadas."""

    def test_dataflux_condenser_error_inheritance(self):
        """DataFluxCondenserError debe heredar de Exception."""
        assert issubclass(DataFluxCondenserError, Exception)

    def test_invalid_input_error_inheritance(self):
        """InvalidInputError debe heredar de DataFluxCondenserError."""
        assert issubclass(InvalidInputError, DataFluxCondenserError)

    def test_processing_error_inheritance(self):
        """ProcessingError debe heredar de DataFluxCondenserError."""
        assert issubclass(ProcessingError, DataFluxCondenserError)

    def test_exception_messages(self):
        """Las excepciones deben preservar mensajes."""
        try:
            raise InvalidInputError("Test message")
        except InvalidInputError as e:
            assert str(e) == "Test message"

    def test_exception_chaining(self):
        """Debe soportar encadenamiento de excepciones."""
        original = ValueError("Original error")

        try:
            raise ProcessingError("Processing failed") from original
        except ProcessingError as e:
            assert e.__cause__ == original


# ==================== TESTS DE PARSEDDATA ====================

class TestParsedData:
    """Pruebas de la estructura ParsedData."""

    def test_creation(self, sample_raw_records, sample_parse_cache):
        """Debe crear instancia correctamente."""
        parsed_data = ParsedData(sample_raw_records, sample_parse_cache)

        assert parsed_data.raw_records == sample_raw_records
        assert parsed_data.parse_cache == sample_parse_cache

    def test_is_named_tuple(self):
        """Debe ser un NamedTuple."""
        from typing import get_type_hints
        hints = get_type_hints(ParsedData)

        assert 'raw_records' in hints
        assert 'parse_cache' in hints

    def test_immutability(self, sample_raw_records, sample_parse_cache):
        """Debe ser inmutable."""
        parsed_data = ParsedData(sample_raw_records, sample_parse_cache)

        with pytest.raises(AttributeError):
            parsed_data.raw_records = []

    def test_tuple_unpacking(self, sample_raw_records, sample_parse_cache):
        """Debe soportar desempaquetado de tupla."""
        parsed_data = ParsedData(sample_raw_records, sample_parse_cache)

        records, cache = parsed_data

        assert records == sample_raw_records
        assert cache == sample_parse_cache


# ==================== TESTS PARAMETRIZADOS ====================

class TestParametrizedScenarios:
    """Pruebas parametrizadas para múltiples escenarios."""

    @pytest.mark.parametrize("threshold,records_count,expected", [
        (1, 0, False),
        (1, 1, True),
        (5, 3, False),
        (5, 5, True),
        (10, 15, True),
    ])
    def test_threshold_validation(
        self,
        valid_config,
        valid_profile,
        threshold,
        records_count,
        expected
    ):
        """Debe validar diferentes umbrales correctamente."""
        config = CondenserConfig(min_records_threshold=threshold)
        condenser = DataFluxCondenser(valid_config, valid_profile, config)

        records = [{'id': i} for i in range(records_count)]
        parsed_data = ParsedData(records, {})

        result = condenser._validate_parsed_data(parsed_data)
        assert result == expected

    @pytest.mark.parametrize("extension", ['.csv', '.CSV', '.txt', '.TXT'])
    def test_valid_file_extensions(self, condenser, tmp_path, extension):
        """Debe aceptar extensiones válidas (case-insensitive)."""
        file = tmp_path / f"data{extension}"
        file.write_text("test")

        result = condenser._validate_input_file(str(file))
        assert result.exists()

    @pytest.mark.parametrize("log_level", ['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    def test_log_levels(self, valid_config, valid_profile, log_level):
        """Debe aceptar diferentes niveles de log."""
        config = CondenserConfig(log_level=log_level)
        condenser = DataFluxCondenser(valid_config, valid_profile, config)

        assert condenser.condenser_config.log_level == log_level


# ==================== TESTS DE EDGE CASES ====================

class TestEdgeCases:
    """Pruebas de casos límite y situaciones extremas."""

    @patch('app.flux_condenser.APUProcessor')
    @patch('app.flux_condenser.ReportParserCrudo')
    def test_very_large_dataset(
        self,
        mock_parser_class,
        mock_processor_class,
        condenser,
        mock_csv_file
    ):
        """Debe manejar datasets muy grandes."""
        large_records = [{'id': i, 'value': i*10} for i in range(100000)]
        large_df = pd.DataFrame(large_records)

        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = large_records
        mock_parser.get_parse_cache.return_value = {'total': 100000}
        mock_parser_class.return_value = mock_parser

        mock_processor = Mock()
        mock_processor.process_all.return_value = large_df
        mock_processor_class.return_value = mock_processor

        result = condenser.stabilize(str(mock_csv_file))

        assert len(result) == 100000

    def test_dataframe_with_all_null_columns(self, condenser, caplog):
        """Debe detectar y advertir sobre todas las columnas nulas."""
        df = pd.DataFrame({
            'col1': [None, None, None],
            'col2': [None, None, None]
        })

        with caplog.at_level(logging.WARNING):
            condenser._validate_output(df)

        assert "col1" in caplog.text
        assert "col2" in caplog.text

    def test_dataframe_with_mixed_types(self, condenser):
        """Debe aceptar DataFrame con tipos mixtos."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c'],
            'float_col': [1.1, 2.2, 3.3],
            'bool_col': [True, False, True]
        })

        condenser._validate_output(df)  # No debe fallar

    @patch('app.flux_condenser.ReportParserCrudo')
    def test_unicode_in_file_path(
        self,
        mock_parser_class,
        condenser,
        tmp_path
    ):
        """Debe manejar rutas con caracteres Unicode."""
        unicode_file = tmp_path / "archivo_ñ_测试.csv"
        unicode_file.write_text("data")

        mock_parser = Mock()
        mock_parser.parse_to_raw.return_value = []
        mock_parser.get_parse_cache.return_value = {}
        mock_parser_class.return_value = mock_parser

        result = condenser.stabilize(str(unicode_file))
        assert isinstance(result, pd.DataFrame)


# ==================== CONFIGURACIÓN DE PYTEST ====================

def pytest_configure(config):
    """Configuración de pytest."""
    config.addinivalue_line(
        "markers", "slow: marca tests lentos"
    )
    config.addinivalue_line(
        "markers", "integration: tests de integración"
    )


# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=flux_condenser", "--cov-report=html"])
