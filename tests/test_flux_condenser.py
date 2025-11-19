"""
Test Suite para DataFluxCondenser
Cobertura completa de lógica, validaciones, manejo de errores y casos edge.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from app.flux_condenser import (
    CondenserConfig,
    DataFluxCondenser,
    DataFluxCondenserError,
    InvalidInputError,
    ParsedData,
    ProcessingError,
)

# ==================== FIXTURES ====================

@pytest.fixture
def valid_config() -> Dict[str, Any]:
    """Configuración válida del sistema."""
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
    """Perfil válido de procesamiento."""
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
    """Configuración mínima sin claves requeridas (modo tolerante)."""
    return {'some_key': 'value'}


@pytest.fixture
def minimal_profile() -> Dict[str, Any]:
    """Perfil mínimo sin claves requeridas."""
    return {'another_key': 'data'}


@pytest.fixture
def custom_condenser_config() -> CondenserConfig:
    """Configuración personalizada del condensador."""
    return CondenserConfig(
        min_records_threshold=5,
        enable_strict_validation=True,
        log_level="DEBUG"
    )


@pytest.fixture
def default_condenser_config() -> CondenserConfig:
    """Configuración por defecto del condensador."""
    return CondenserConfig()


@pytest.fixture
def condenser(valid_config, valid_profile) -> DataFluxCondenser:
    """Instancia básica del condensador."""
    return DataFluxCondenser(valid_config, valid_profile)


@pytest.fixture
def sample_raw_records() -> List[Dict[str, Any]]:
    """Datos crudos de ejemplo."""
    return [
        {'codigo': 'A001', 'cantidad': 10, 'precio': 100.0},
        {'codigo': 'A002', 'cantidad': 5, 'precio': 50.0},
        {'codigo': 'A003', 'cantidad': 8, 'precio': 80.0}
    ]


@pytest.fixture
def sample_parse_cache() -> Dict[str, Any]:
    """Cache de parseo de ejemplo."""
    return {
        'total_lines': 100,
        'skipped_lines': 5,
        'headers': ['codigo', 'cantidad', 'precio'],
        'file_encoding': 'utf-8'
    }


@pytest.fixture
def sample_dataframe(sample_raw_records) -> pd.DataFrame:
    """DataFrame de ejemplo."""
    return pd.DataFrame(sample_raw_records)


@pytest.fixture
def mock_csv_file(tmp_path) -> Path:
    """Crea un archivo CSV temporal."""
    file_path = tmp_path / "test_data.csv"
    file_path.write_text(
        "codigo,cantidad,precio\n"
        "A001,10,100.0\n"
        "A002,5,50.0\n"
    )
    return file_path


# ==================== TESTS DE INICIALIZACIÓN ====================

class TestInitialization:
    """Pruebas de inicialización del condensador."""

    def test_init_with_valid_params(self, valid_config, valid_profile):
        """Debe inicializar correctamente con parámetros válidos."""
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
        """Debe aceptar configuración personalizada del condensador."""
        condenser = DataFluxCondenser(
            valid_config,
            valid_profile,
            custom_condenser_config
        )

        assert condenser.condenser_config.min_records_threshold == 5
        assert condenser.condenser_config.enable_strict_validation is True
        assert condenser.condenser_config.log_level == "DEBUG"

    def test_init_with_default_condenser_config(self, valid_config, valid_profile):
        """Debe usar configuración por defecto si no se especifica."""
        condenser = DataFluxCondenser(valid_config, valid_profile)

        assert condenser.condenser_config.min_records_threshold == 1
        assert condenser.condenser_config.enable_strict_validation is True
        assert condenser.condenser_config.log_level == "INFO"

    def test_init_with_invalid_config_type(self, valid_profile):
        """Debe fallar si config no es diccionario."""
        with pytest.raises(InvalidInputError, match="deben ser diccionarios válidos"):
            DataFluxCondenser("invalid_config", valid_profile)

    def test_init_with_invalid_profile_type(self, valid_config):
        """Debe fallar si profile no es diccionario."""
        with pytest.raises(InvalidInputError, match="deben ser diccionarios válidos"):
            DataFluxCondenser(valid_config, None)

    def test_init_with_none_params(self):
        """Debe fallar si ambos parámetros son None."""
        with pytest.raises(InvalidInputError):
            DataFluxCondenser(None, None)

    def test_init_with_minimal_config_warns(
        self,
        minimal_config,
        minimal_profile,
        caplog
    ):
        """Debe advertir sobre claves faltantes pero no fallar (modo tolerante)."""
        with caplog.at_level(logging.WARNING):
            condenser = DataFluxCondenser(minimal_config, minimal_profile)

        assert condenser is not None
        assert "Claves faltantes en config" in caplog.text
        assert "Claves faltantes en profile" in caplog.text

    def test_logger_configuration(self, valid_config, valid_profile):
        """Debe configurar el logger con el nivel correcto."""
        custom_config = CondenserConfig(log_level="DEBUG")
        condenser = DataFluxCondenser(valid_config, valid_profile, custom_config)

        assert condenser.logger.level == logging.DEBUG


# ==================== TESTS DE VALIDACIÓN DE ARCHIVO ====================

class TestInputFileValidation:
    """Pruebas de validación de archivos de entrada."""

    def test_validate_existing_csv_file(self, condenser, mock_csv_file):
        """Debe validar archivo CSV existente."""
        result = condenser._validate_input_file(str(mock_csv_file))

        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == '.csv'

    def test_validate_txt_file(self, condenser, tmp_path):
        """Debe aceptar archivos .txt."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("test data")

        result = condenser._validate_input_file(str(txt_file))
        assert result.suffix == '.txt'

    def test_validate_nonexistent_file(self, condenser):
        """Debe fallar si el archivo no existe."""
        with pytest.raises(InvalidInputError, match="El archivo no existe"):
            condenser._validate_input_file("/path/to/nonexistent.csv")

    def test_validate_directory_path(self, condenser, tmp_path):
        """Debe fallar si la ruta es un directorio."""
        with pytest.raises(InvalidInputError, match="La ruta no es un archivo"):
            condenser._validate_input_file(str(tmp_path))

    def test_validate_empty_string(self, condenser):
        """Debe fallar con cadena vacía."""
        with pytest.raises(InvalidInputError, match="debe ser una cadena no vacía"):
            condenser._validate_input_file("")

    def test_validate_none_path(self, condenser):
        """Debe fallar con None."""
        with pytest.raises(InvalidInputError, match="debe ser una cadena no vacía"):
            condenser._validate_input_file(None)

    def test_validate_unusual_extension_warns(self, condenser, tmp_path, caplog):
        """Debe advertir sobre extensiones inusuales."""
        unusual_file = tmp_path / "data.xlsx"
        unusual_file.write_text("data")

        with caplog.at_level(logging.WARNING):
            condenser._validate_input_file(str(unusual_file))

        assert "Extensión inusual detectada" in caplog.text

    def test_validate_returns_path_object(self, condenser, mock_csv_file):
        """Debe retornar objeto Path."""
        result = condenser._validate_input_file(str(mock_csv_file))
        assert isinstance(result, Path)


# ==================== TESTS DE FILTRADO (ABSORB AND FILTER) ====================

class TestAbsorbAndFilter:
    """Pruebas del proceso de filtrado con ReportParserCrudo."""

    @patch('app.flux_condenser.ReportParserCrudo')
    def test_successful_parsing(
        self,
        mock_parser_class,
        condenser,
        mock_csv_file,
        sample_raw_records,
        sample_parse_cache
    ):
        """Debe parsear correctamente y retornar ParsedData."""
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


# ==================== TESTS: FluxPhysicsEngine ====================

class TestFluxPhysicsEngine:
    """Pruebas unitarias para el motor de física."""

    # Importar math y FluxPhysicsEngine aquí para mantener el test aislado
    import math
    from app.flux_condenser import FluxPhysicsEngine

    @pytest.fixture
    def physics_engine(self):
        """Instancia del motor con valores base."""
        return self.FluxPhysicsEngine(capacitance=1000.0, resistance=10.0)

    @pytest.mark.parametrize("load_size, complexity, expected_saturation", [
        (100, 0.0, 0.00995),
        (5000, 0.0, 0.3934),
        (10000, 0.5, 0.4865),
        (20000, 1.0, 0.6321),
    ])
    def test_calculate_saturation(self, physics_engine, load_size, complexity, expected_saturation):
        """Debe calcular la saturación correctamente."""
        saturation = physics_engine.calculate_saturation(load_size, complexity)
        assert saturation == pytest.approx(expected_saturation, abs=1e-4)

    @pytest.mark.parametrize("saturation, expected_status", [
        (0.29, "FLUJO LAMINAR (Estable)"),
        (0.69, "FLUJO TRANSITORIO (Carga Media)"),
        (0.7, "FLUJO TURBULENTO (Alta Saturación)"),
    ])
    def test_get_stability_status(self, physics_engine, saturation, expected_status):
        """Debe retornar el estado de estabilidad correcto."""
        assert physics_engine.get_stability_status(saturation) == expected_status

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
        """Debe registrar la telemetría física durante un flujo exitoso."""
        # Setup: 100 registros crudos, 80 cacheados -> complejidad 0.2
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

        assert "📊 [TELEMETRÍA]" in caplog.text
        assert "Complejidad: 0.20" in caplog.text
        assert "Estado: FLUJO LAMINAR" in caplog.text

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
        assert "La carga no generó señal válida" in caplog.text

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

        # CORRECCIÓN: Ajustar a los nuevos mensajes de log.
        assert "⚡ [FÍSICA] Iniciando ciclo de estabilización" in caplog.text
        assert "✅ [ÉXITO] Flujo estabilizado" in caplog.text
        assert "registros procesados" in caplog.text


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
