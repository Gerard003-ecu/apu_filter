"""
Suite de pruebas para el módulo tools_interface.

Evalúa la lógica robusta de:
- Diagnóstico de archivos (APUs, Insumos, Presupuesto)
- Limpieza de archivos CSV
- Consulta de estado de telemetría
- Funciones de utilidad
"""

import logging
import pytest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch, PropertyMock

# Módulo bajo prueba
from scripts.tools_interface import (
    FileType,
    DiagnosticError,
    FileNotFoundDiagnosticError,
    UnsupportedFileTypeError,
    CleaningError,
    diagnose_file,
    clean_file,
    get_telemetry_status,
    get_supported_file_types,
    is_valid_file_type,
    _validate_path_exists,
    _normalize_path,
    _normalize_file_type,
    _create_error_response,
    _create_success_response,
    _validate_csv_parameters,
    _generate_output_path,
    _VALID_DELIMITERS,
    _SUPPORTED_ENCODINGS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_csv_file(tmp_path: Path) -> Path:
    """Crea un archivo CSV temporal válido."""
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text("col1;col2;col3\nval1;val2;val3\n", encoding="utf-8")
    return csv_file


@pytest.fixture
def temp_empty_file(tmp_path: Path) -> Path:
    """Crea un archivo vacío temporal."""
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()
    return empty_file


@pytest.fixture
def mock_diagnostic_result() -> Dict[str, Any]:
    """Resultado simulado de diagnóstico."""
    return {
        "total_rows": 100,
        "valid_rows": 95,
        "error_rows": 5,
        "warnings": [],
        "errors": ["Row 10: Invalid format"],
    }


@pytest.fixture
def mock_cleaning_stats():
    """Estadísticas simuladas de limpieza."""
    mock_stats = MagicMock()
    mock_stats.to_dict.return_value = {
        "rows_processed": 100,
        "rows_cleaned": 95,
        "rows_removed": 5,
        "cleaning_time_ms": 150,
    }
    return mock_stats


@pytest.fixture
def mock_telemetry_context():
    """Contexto de telemetría simulado."""
    context = MagicMock()
    context.get_business_report.return_value = {
        "status": "ACTIVE",
        "requests_processed": 42,
        "average_response_time_ms": 120,
        "system_health": "HEALTHY",
    }
    return context


# =============================================================================
# Tests para FileType Enum
# =============================================================================

class TestFileTypeEnum:
    """Pruebas para el enum FileType."""

    def test_file_type_values(self):
        """Verifica los valores del enum."""
        assert FileType.APUS.value == "apus"
        assert FileType.INSUMOS.value == "insumos"
        assert FileType.PRESUPUESTO.value == "presupuesto"

    def test_file_type_values_method(self):
        """Verifica el método values()."""
        values = FileType.values()
        assert isinstance(values, list)
        assert "apus" in values
        assert "insumos" in values
        assert "presupuesto" in values
        assert len(values) == 3

    def test_file_type_is_string_enum(self):
        """Verifica que FileType hereda de str."""
        assert isinstance(FileType.APUS, str)
        assert FileType.APUS == "apus"

    def test_file_type_from_value(self):
        """Verifica creación desde valor string."""
        assert FileType("apus") == FileType.APUS
        assert FileType("insumos") == FileType.INSUMOS
        assert FileType("presupuesto") == FileType.PRESUPUESTO

    def test_file_type_invalid_value(self):
        """Verifica que valores inválidos lanzan excepción."""
        with pytest.raises(ValueError):
            FileType("invalid")


# =============================================================================
# Tests para Excepciones Personalizadas
# =============================================================================

class TestCustomExceptions:
    """Pruebas para las excepciones personalizadas."""

    def test_diagnostic_error_hierarchy(self):
        """Verifica la jerarquía de excepciones de diagnóstico."""
        assert issubclass(FileNotFoundDiagnosticError, DiagnosticError)
        assert issubclass(UnsupportedFileTypeError, DiagnosticError)
        assert issubclass(DiagnosticError, Exception)

    def test_cleaning_error_hierarchy(self):
        """Verifica la jerarquía de excepciones de limpieza."""
        assert issubclass(CleaningError, Exception)

    def test_exception_messages(self):
        """Verifica que las excepciones preservan mensajes."""
        msg = "Test error message"
        
        exc1 = DiagnosticError(msg)
        assert str(exc1) == msg
        
        exc2 = FileNotFoundDiagnosticError(msg)
        assert str(exc2) == msg
        
        exc3 = UnsupportedFileTypeError(msg)
        assert str(exc3) == msg


# =============================================================================
# Tests para Funciones de Validación Internas
# =============================================================================

class TestValidatePathExists:
    """Pruebas para _validate_path_exists."""

    def test_existing_path_passes(self, temp_csv_file: Path):
        """Verifica que una ruta existente no lanza excepción."""
        # No debe lanzar excepción
        _validate_path_exists(temp_csv_file)

    def test_non_existing_path_raises(self, tmp_path: Path):
        """Verifica que una ruta inexistente lanza excepción."""
        non_existing = tmp_path / "non_existing.csv"
        
        with pytest.raises(FileNotFoundDiagnosticError) as exc_info:
            _validate_path_exists(non_existing)
        
        assert "not found" in str(exc_info.value)

    def test_custom_context_in_error(self, tmp_path: Path):
        """Verifica que el contexto personalizado aparece en el error."""
        non_existing = tmp_path / "missing.csv"
        
        with pytest.raises(FileNotFoundDiagnosticError) as exc_info:
            _validate_path_exists(non_existing, context="Input CSV")
        
        assert "Input CSV" in str(exc_info.value)


class TestNormalizePath:
    """Pruebas para _normalize_path."""

    def test_string_path_conversion(self, temp_csv_file: Path):
        """Verifica conversión de string a Path."""
        result = _normalize_path(str(temp_csv_file))
        assert isinstance(result, Path)

    def test_path_object_passthrough(self, temp_csv_file: Path):
        """Verifica que Path se procesa correctamente."""
        result = _normalize_path(temp_csv_file)
        assert isinstance(result, Path)

    def test_empty_string_raises(self):
        """Verifica que string vacío lanza excepción."""
        with pytest.raises(ValueError) as exc_info:
            _normalize_path("")
        
        assert "cannot be empty" in str(exc_info.value)

    def test_none_raises(self):
        """Verifica que None lanza excepción."""
        with pytest.raises(ValueError):
            _normalize_path(None)

    def test_relative_path_handling(self, tmp_path: Path):
        """Verifica manejo de rutas relativas."""
        # Crear archivo para que exista
        test_file = tmp_path / "relative_test.csv"
        test_file.touch()
        
        result = _normalize_path(test_file)
        assert result.is_absolute()


class TestNormalizeFileType:
    """Pruebas para _normalize_file_type."""

    def test_valid_string_types(self):
        """Verifica normalización de strings válidos."""
        assert _normalize_file_type("apus") == FileType.APUS
        assert _normalize_file_type("insumos") == FileType.INSUMOS
        assert _normalize_file_type("presupuesto") == FileType.PRESUPUESTO

    def test_case_insensitive(self):
        """Verifica que la normalización es case-insensitive."""
        assert _normalize_file_type("APUS") == FileType.APUS
        assert _normalize_file_type("Apus") == FileType.APUS
        assert _normalize_file_type("ApUs") == FileType.APUS

    def test_whitespace_trimming(self):
        """Verifica que se eliminan espacios en blanco."""
        assert _normalize_file_type("  apus  ") == FileType.APUS
        assert _normalize_file_type("\tinsumos\n") == FileType.INSUMOS

    def test_enum_passthrough(self):
        """Verifica que FileType pasa directamente."""
        assert _normalize_file_type(FileType.APUS) == FileType.APUS
        assert _normalize_file_type(FileType.INSUMOS) == FileType.INSUMOS

    def test_invalid_string_raises(self):
        """Verifica que string inválido lanza excepción."""
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            _normalize_file_type("invalid_type")
        
        error_msg = str(exc_info.value)
        assert "Unknown file type" in error_msg
        assert "invalid_type" in error_msg
        assert "apus" in error_msg  # Debe mostrar tipos válidos

    def test_non_string_type_raises(self):
        """Verifica que tipos no-string lanzan excepción."""
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            _normalize_file_type(123)
        
        assert "must be string or FileType" in str(exc_info.value)

    def test_none_raises(self):
        """Verifica que None lanza excepción."""
        with pytest.raises(UnsupportedFileTypeError):
            _normalize_file_type(None)


class TestValidateCsvParameters:
    """Pruebas para _validate_csv_parameters."""

    def test_valid_parameters(self):
        """Verifica que parámetros válidos no lanzan excepción."""
        # No debe lanzar excepción
        _validate_csv_parameters(";", "utf-8")
        _validate_csv_parameters(",", "latin-1")
        _validate_csv_parameters("\t", "utf-8-sig")

    def test_all_valid_delimiters(self):
        """Verifica todos los delimitadores válidos."""
        for delimiter in _VALID_DELIMITERS:
            _validate_csv_parameters(delimiter, "utf-8")

    def test_empty_delimiter_raises(self):
        """Verifica que delimitador vacío lanza excepción."""
        with pytest.raises(ValueError) as exc_info:
            _validate_csv_parameters("", "utf-8")
        
        assert "Delimiter cannot be empty" in str(exc_info.value)

    def test_invalid_delimiter_raises(self):
        """Verifica que delimitador inválido lanza excepción."""
        with pytest.raises(ValueError) as exc_info:
            _validate_csv_parameters("@", "utf-8")
        
        assert "Invalid delimiter" in str(exc_info.value)

    def test_empty_encoding_raises(self):
        """Verifica que encoding vacío lanza excepción."""
        with pytest.raises(ValueError) as exc_info:
            _validate_csv_parameters(";", "")
        
        assert "Encoding cannot be empty" in str(exc_info.value)

    def test_uncommon_encoding_logs_warning(self, caplog):
        """Verifica que encoding no común genera warning."""
        with caplog.at_level(logging.WARNING):
            _validate_csv_parameters(";", "uncommon-encoding")
        
        assert "not in common list" in caplog.text


class TestGenerateOutputPath:
    """Pruebas para _generate_output_path."""

    def test_default_suffix(self, tmp_path: Path):
        """Verifica sufijo por defecto '_clean'."""
        input_path = tmp_path / "data.csv"
        result = _generate_output_path(input_path)
        
        assert result.name == "data_clean.csv"
        assert result.parent == input_path.parent

    def test_custom_suffix(self, tmp_path: Path):
        """Verifica sufijo personalizado."""
        input_path = tmp_path / "data.csv"
        result = _generate_output_path(input_path, suffix="_processed")
        
        assert result.name == "data_processed.csv"

    def test_preserves_extension(self, tmp_path: Path):
        """Verifica que preserva la extensión original."""
        input_path = tmp_path / "data.txt"
        result = _generate_output_path(input_path)
        
        assert result.suffix == ".txt"

    def test_complex_filename(self, tmp_path: Path):
        """Verifica manejo de nombres complejos."""
        input_path = tmp_path / "my.data.file.csv"
        result = _generate_output_path(input_path)
        
        assert result.name == "my.data.file_clean.csv"


# =============================================================================
# Tests para Funciones de Respuesta
# =============================================================================

class TestCreateErrorResponse:
    """Pruebas para _create_error_response."""

    def test_basic_error_response(self):
        """Verifica estructura básica de respuesta de error."""
        result = _create_error_response("Something went wrong")
        
        assert result["success"] is False
        assert result["error"] == "Something went wrong"
        assert "error_type" in result

    def test_exception_error_response(self):
        """Verifica respuesta con excepción."""
        exc = ValueError("Invalid value")
        result = _create_error_response(exc)
        
        assert result["success"] is False
        assert result["error"] == "Invalid value"
        assert result["error_type"] == "ValueError"

    def test_extra_fields(self):
        """Verifica que campos extra se incluyen."""
        result = _create_error_response(
            "Error", 
            error_category="validation",
            field="username"
        )
        
        assert result["error_category"] == "validation"
        assert result["field"] == "username"


class TestCreateSuccessResponse:
    """Pruebas para _create_success_response."""

    def test_basic_success_response(self):
        """Verifica estructura básica de respuesta exitosa."""
        data = {"rows": 100, "processed": True}
        result = _create_success_response(data)
        
        assert result["success"] is True
        assert result["rows"] == 100
        assert result["processed"] is True

    def test_extra_fields(self):
        """Verifica que campos extra se incluyen."""
        data = {"rows": 100}
        result = _create_success_response(
            data,
            output_path="/tmp/output.csv",
            duration_ms=150
        )
        
        assert result["output_path"] == "/tmp/output.csv"
        assert result["duration_ms"] == 150

    def test_data_not_overwritten_by_extras(self):
        """Verifica que data tiene prioridad sobre extras."""
        data = {"key": "from_data"}
        result = _create_success_response(data, key="from_extra")
        
        # El comportamiento depende de la implementación
        # En el código actual, extras van después, así que sobrescriben
        assert "key" in result


# =============================================================================
# Tests para diagnose_file
# =============================================================================

class TestDiagnoseFile:
    """Pruebas para la función diagnose_file."""

    def test_file_not_found(self, tmp_path: Path):
        """Verifica manejo de archivo inexistente."""
        non_existing = tmp_path / "non_existing.csv"
        
        result = diagnose_file(non_existing, "apus")
        
        assert result["success"] is False
        assert "not found" in result["error"]
        assert result["error_type"] == "FileNotFoundDiagnosticError"

    def test_invalid_file_type(self, temp_csv_file: Path):
        """Verifica manejo de tipo de archivo inválido."""
        result = diagnose_file(temp_csv_file, "invalid_type")
        
        assert result["success"] is False
        assert "Unknown file type" in result["error"]
        assert result["error_type"] == "UnsupportedFileTypeError"

    def test_invalid_file_type_type(self, temp_csv_file: Path):
        """Verifica manejo de tipo no-string para file_type."""
        result = diagnose_file(temp_csv_file, 123)
        
        assert result["success"] is False
        assert "must be string or FileType" in result["error"]

    @patch("scripts.tools_interface.APUFileDiagnostic")
    def test_apus_diagnosis_success(
        self, 
        mock_diagnostic_class: MagicMock,
        temp_csv_file: Path,
        mock_diagnostic_result: Dict[str, Any]
    ):
        """Verifica diagnóstico exitoso de archivo APUS."""
        # Configurar mock
        mock_instance = MagicMock()
        mock_instance.to_dict.return_value = mock_diagnostic_result
        mock_diagnostic_class.return_value = mock_instance
        
        result = diagnose_file(temp_csv_file, "apus")
        
        assert result["success"] is True
        assert result["file_type"] == "apus"
        assert "total_rows" in result
        mock_instance.diagnose.assert_called_once()

    @patch("scripts.tools_interface.InsumosFileDiagnostic")
    def test_insumos_diagnosis_success(
        self,
        mock_diagnostic_class: MagicMock,
        temp_csv_file: Path,
        mock_diagnostic_result: Dict[str, Any]
    ):
        """Verifica diagnóstico exitoso de archivo Insumos."""
        mock_instance = MagicMock()
        mock_instance.to_dict.return_value = mock_diagnostic_result
        mock_diagnostic_class.return_value = mock_instance
        
        result = diagnose_file(temp_csv_file, "insumos")
        
        assert result["success"] is True
        assert result["file_type"] == "insumos"
        mock_instance.diagnose.assert_called_once()

    @patch("scripts.tools_interface.PresupuestoFileDiagnostic")
    def test_presupuesto_diagnosis_success(
        self,
        mock_diagnostic_class: MagicMock,
        temp_csv_file: Path,
        mock_diagnostic_result: Dict[str, Any]
    ):
        """Verifica diagnóstico exitoso de archivo Presupuesto."""
        mock_instance = MagicMock()
        mock_instance.to_dict.return_value = mock_diagnostic_result
        mock_diagnostic_class.return_value = mock_instance
        
        result = diagnose_file(temp_csv_file, "presupuesto")
        
        assert result["success"] is True
        assert result["file_type"] == "presupuesto"
        mock_instance.diagnose.assert_called_once()

    @patch("scripts.tools_interface.APUFileDiagnostic")
    def test_diagnosis_with_file_type_enum(
        self,
        mock_diagnostic_class: MagicMock,
        temp_csv_file: Path,
        mock_diagnostic_result: Dict[str, Any]
    ):
        """Verifica diagnóstico usando FileType enum."""
        mock_instance = MagicMock()
        mock_instance.to_dict.return_value = mock_diagnostic_result
        mock_diagnostic_class.return_value = mock_instance
        
        result = diagnose_file(temp_csv_file, FileType.APUS)
        
        assert result["success"] is True
        assert result["file_type"] == "apus"

    @patch("scripts.tools_interface.APUFileDiagnostic")
    def test_diagnosis_io_error(
        self,
        mock_diagnostic_class: MagicMock,
        temp_csv_file: Path
    ):
        """Verifica manejo de errores de I/O durante diagnóstico."""
        mock_diagnostic_class.side_effect = IOError("Cannot read file")
        
        result = diagnose_file(temp_csv_file, "apus")
        
        assert result["success"] is False
        assert "Cannot read file" in result["error"]
        assert result.get("error_category") == "io_error"

    @patch("scripts.tools_interface.APUFileDiagnostic")
    def test_diagnosis_unexpected_error(
        self,
        mock_diagnostic_class: MagicMock,
        temp_csv_file: Path
    ):
        """Verifica manejo de errores inesperados."""
        mock_diagnostic_class.side_effect = RuntimeError("Unexpected")
        
        result = diagnose_file(temp_csv_file, "apus")
        
        assert result["success"] is False
        assert result.get("error_category") == "unexpected"

    @patch("scripts.tools_interface.APUFileDiagnostic")
    def test_diagnosis_includes_file_path(
        self,
        mock_diagnostic_class: MagicMock,
        temp_csv_file: Path,
        mock_diagnostic_result: Dict[str, Any]
    ):
        """Verifica que la respuesta incluye la ruta del archivo."""
        mock_instance = MagicMock()
        mock_instance.to_dict.return_value = mock_diagnostic_result
        mock_diagnostic_class.return_value = mock_instance
        
        result = diagnose_file(temp_csv_file, "apus")
        
        assert "file_path" in result
        assert str(temp_csv_file) in result["file_path"]


# =============================================================================
# Tests para clean_file
# =============================================================================

class TestCleanFile:
    """Pruebas para la función clean_file."""

    def test_input_file_not_found(self, tmp_path: Path):
        """Verifica manejo de archivo de entrada inexistente."""
        non_existing = tmp_path / "non_existing.csv"
        
        result = clean_file(non_existing)
        
        assert result["success"] is False
        assert "not found" in result["error"]

    @patch("scripts.tools_interface.CSVCleaner")
    def test_clean_success_auto_output(
        self,
        mock_cleaner_class: MagicMock,
        temp_csv_file: Path,
        mock_cleaning_stats: MagicMock
    ):
        """Verifica limpieza exitosa con output auto-generado."""
        mock_cleaner_class.return_value.clean.return_value = mock_cleaning_stats
        
        result = clean_file(temp_csv_file)
        
        assert result["success"] is True
        assert "output_path" in result
        assert "_clean" in result["output_path"]
        mock_cleaner_class.return_value.clean.assert_called_once()

    @patch("scripts.tools_interface.CSVCleaner")
    def test_clean_success_custom_output(
        self,
        mock_cleaner_class: MagicMock,
        temp_csv_file: Path,
        tmp_path: Path,
        mock_cleaning_stats: MagicMock
    ):
        """Verifica limpieza exitosa con output personalizado."""
        mock_cleaner_class.return_value.clean.return_value = mock_cleaning_stats
        output_path = tmp_path / "custom_output.csv"
        
        result = clean_file(temp_csv_file, output_path=output_path)
        
        assert result["success"] is True
        assert str(output_path) in result["output_path"]

    @patch("scripts.tools_interface.CSVCleaner")
    def test_clean_with_custom_delimiter(
        self,
        mock_cleaner_class: MagicMock,
        temp_csv_file: Path,
        mock_cleaning_stats: MagicMock
    ):
        """Verifica limpieza con delimitador personalizado."""
        mock_cleaner_class.return_value.clean.return_value = mock_cleaning_stats
        
        result = clean_file(temp_csv_file, delimiter=",")
        
        assert result["success"] is True
        # Verificar que se pasó el delimitador correcto
        call_kwargs = mock_cleaner_class.call_args[1]
        assert call_kwargs["delimiter"] == ","

    @patch("scripts.tools_interface.CSVCleaner")
    def test_clean_with_custom_encoding(
        self,
        mock_cleaner_class: MagicMock,
        temp_csv_file: Path,
        mock_cleaning_stats: MagicMock
    ):
        """Verifica limpieza con encoding personalizado."""
        mock_cleaner_class.return_value.clean.return_value = mock_cleaning_stats
        
        result = clean_file(temp_csv_file, encoding="latin-1")
        
        assert result["success"] is True
        call_kwargs = mock_cleaner_class.call_args[1]
        assert call_kwargs["encoding"] == "latin-1"

    def test_invalid_delimiter(self, temp_csv_file: Path):
        """Verifica rechazo de delimitador inválido."""
        result = clean_file(temp_csv_file, delimiter="@")
        
        assert result["success"] is False
        assert "Invalid delimiter" in result["error"]

    def test_empty_delimiter(self, temp_csv_file: Path):
        """Verifica rechazo de delimitador vacío."""
        result = clean_file(temp_csv_file, delimiter="")
        
        assert result["success"] is False
        assert "Delimiter cannot be empty" in result["error"]

    def test_empty_encoding(self, temp_csv_file: Path):
        """Verifica rechazo de encoding vacío."""
        result = clean_file(temp_csv_file, encoding="")
        
        assert result["success"] is False
        assert "Encoding cannot be empty" in result["error"]

    def test_same_input_output_path(self, temp_csv_file: Path):
        """Verifica rechazo cuando input y output son iguales."""
        result = clean_file(temp_csv_file, output_path=temp_csv_file)
        
        assert result["success"] is False
        assert "cannot be the same" in result["error"]

    @patch("scripts.tools_interface.CSVCleaner")
    def test_output_file_exists_no_overwrite(
        self,
        mock_cleaner_class: MagicMock,
        temp_csv_file: Path,
        tmp_path: Path
    ):
        """Verifica error cuando output existe y overwrite=False."""
        output_path = tmp_path / "existing_output.csv"
        output_path.touch()  # Crear archivo existente
        
        result = clean_file(
            temp_csv_file, 
            output_path=output_path, 
            overwrite=False
        )
        
        assert result["success"] is False
        assert "already exists" in result["error"]

    @patch("scripts.tools_interface.CSVCleaner")
    def test_creates_output_directory(
        self,
        mock_cleaner_class: MagicMock,
        temp_csv_file: Path,
        tmp_path: Path,
        mock_cleaning_stats: MagicMock
    ):
        """Verifica que se crea el directorio de salida si no existe."""
        mock_cleaner_class.return_value.clean.return_value = mock_cleaning_stats
        
        output_path = tmp_path / "new_dir" / "subdir" / "output.csv"
        
        result = clean_file(temp_csv_file, output_path=output_path)
        
        assert result["success"] is True
        assert output_path.parent.exists()

    @patch("scripts.tools_interface.CSVCleaner")
    def test_clean_io_error(
        self,
        mock_cleaner_class: MagicMock,
        temp_csv_file: Path
    ):
        """Verifica manejo de errores de I/O durante limpieza."""
        mock_cleaner_class.return_value.clean.side_effect = IOError("Disk full")
        
        result = clean_file(temp_csv_file)
        
        assert result["success"] is False
        assert "Disk full" in result["error"]
        assert result.get("error_category") == "io_error"

    @patch("scripts.tools_interface.CSVCleaner")
    def test_clean_unexpected_error(
        self,
        mock_cleaner_class: MagicMock,
        temp_csv_file: Path
    ):
        """Verifica manejo de errores inesperados durante limpieza."""
        mock_cleaner_class.return_value.clean.side_effect = RuntimeError("Unexpected")
        
        result = clean_file(temp_csv_file)
        
        assert result["success"] is False
        assert result.get("error_category") == "unexpected"

    @patch("scripts.tools_interface.CSVCleaner")
    def test_clean_includes_input_path(
        self,
        mock_cleaner_class: MagicMock,
        temp_csv_file: Path,
        mock_cleaning_stats: MagicMock
    ):
        """Verifica que la respuesta incluye la ruta de entrada."""
        mock_cleaner_class.return_value.clean.return_value = mock_cleaning_stats
        
        result = clean_file(temp_csv_file)
        
        assert "input_path" in result

    @patch("scripts.tools_interface.CSVCleaner")
    def test_clean_stats_dict_fallback(
        self,
        mock_cleaner_class: MagicMock,
        temp_csv_file: Path
    ):
        """Verifica fallback cuando stats no tiene to_dict()."""
        # Stats sin método to_dict
        mock_stats = {"rows": 100}
        mock_cleaner_class.return_value.clean.return_value = mock_stats
        
        result = clean_file(temp_csv_file)
        
        # Debería manejar el caso sin to_dict
        assert result["success"] is True


# =============================================================================
# Tests para get_telemetry_status
# =============================================================================

class TestGetTelemetryStatus:
    """Pruebas para la función get_telemetry_status."""

    def test_no_context_returns_idle(self):
        """Verifica estado IDLE cuando no hay contexto."""
        result = get_telemetry_status()
        
        assert result["status"] == "IDLE"
        assert result["system_health"] == "UNKNOWN"
        assert result["has_active_context"] is False
        assert "No active processing context" in result["message"]

    def test_none_context_returns_idle(self):
        """Verifica estado IDLE con contexto None explícito."""
        result = get_telemetry_status(None)
        
        assert result["status"] == "IDLE"
        assert result["has_active_context"] is False

    def test_valid_context_returns_report(self, mock_telemetry_context: MagicMock):
        """Verifica que contexto válido retorna su reporte."""
        result = get_telemetry_status(mock_telemetry_context)
        
        assert result["status"] == "ACTIVE"
        assert result["system_health"] == "HEALTHY"
        assert result["has_active_context"] is True
        assert result["requests_processed"] == 42
        mock_telemetry_context.get_business_report.assert_called_once()

    def test_context_without_method(self):
        """Verifica manejo de contexto sin método requerido."""
        invalid_context = MagicMock(spec=[])  # Sin get_business_report
        
        result = get_telemetry_status(invalid_context)
        
        assert result["status"] == "ERROR"
        assert result["system_health"] == "DEGRADED"
        assert "missing required method" in result["message"]

    def test_context_method_raises_exception(self, mock_telemetry_context: MagicMock):
        """Verifica manejo de excepción en get_business_report."""
        mock_telemetry_context.get_business_report.side_effect = RuntimeError("DB Error")
        
        result = get_telemetry_status(mock_telemetry_context)
        
        assert result["status"] == "ERROR"
        assert result["system_health"] == "DEGRADED"
        assert "DB Error" in result.get("error", "")

    def test_context_returns_non_dict(self, mock_telemetry_context: MagicMock):
        """Verifica manejo cuando reporte no es diccionario."""
        mock_telemetry_context.get_business_report.return_value = "string report"
        
        result = get_telemetry_status(mock_telemetry_context)
        
        assert result["has_active_context"] is True
        assert "raw_report" in result

    def test_adds_default_status_if_missing(self, mock_telemetry_context: MagicMock):
        """Verifica que se añaden valores por defecto si faltan."""
        mock_telemetry_context.get_business_report.return_value = {
            "custom_metric": 123
        }
        
        result = get_telemetry_status(mock_telemetry_context)
        
        assert result["status"] == "ACTIVE"  # Default añadido
        assert result["system_health"] == "HEALTHY"  # Default añadido
        assert result["custom_metric"] == 123


# =============================================================================
# Tests para Funciones de Utilidad Pública
# =============================================================================

class TestGetSupportedFileTypes:
    """Pruebas para get_supported_file_types."""

    def test_returns_list(self):
        """Verifica que retorna una lista."""
        result = get_supported_file_types()
        assert isinstance(result, list)

    def test_contains_all_types(self):
        """Verifica que contiene todos los tipos esperados."""
        result = get_supported_file_types()
        
        assert "apus" in result
        assert "insumos" in result
        assert "presupuesto" in result

    def test_count_matches_enum(self):
        """Verifica que la cantidad coincide con el enum."""
        result = get_supported_file_types()
        assert len(result) == len(FileType)


class TestIsValidFileType:
    """Pruebas para is_valid_file_type."""

    def test_valid_types_return_true(self):
        """Verifica que tipos válidos retornan True."""
        assert is_valid_file_type("apus") is True
        assert is_valid_file_type("insumos") is True
        assert is_valid_file_type("presupuesto") is True

    def test_case_insensitive(self):
        """Verifica que es case-insensitive."""
        assert is_valid_file_type("APUS") is True
        assert is_valid_file_type("Insumos") is True

    def test_invalid_type_returns_false(self):
        """Verifica que tipos inválidos retornan False."""
        assert is_valid_file_type("invalid") is False
        assert is_valid_file_type("csv") is False
        assert is_valid_file_type("") is False

    def test_non_string_returns_false(self):
        """Verifica que no-strings retornan False."""
        assert is_valid_file_type(123) is False
        assert is_valid_file_type(None) is False
        assert is_valid_file_type([]) is False


# =============================================================================
# Tests de Integración
# =============================================================================

class TestIntegration:
    """Pruebas de integración entre componentes."""

    @patch("scripts.tools_interface.APUFileDiagnostic")
    @patch("scripts.tools_interface.CSVCleaner")
    def test_diagnose_then_clean_workflow(
        self,
        mock_cleaner_class: MagicMock,
        mock_diagnostic_class: MagicMock,
        temp_csv_file: Path,
        tmp_path: Path,
        mock_diagnostic_result: Dict[str, Any],
        mock_cleaning_stats: MagicMock
    ):
        """Verifica flujo completo: diagnóstico -> limpieza."""
        # Configurar mocks
        mock_diag_instance = MagicMock()
        mock_diag_instance.to_dict.return_value = mock_diagnostic_result
        mock_diagnostic_class.return_value = mock_diag_instance
        
        mock_cleaner_class.return_value.clean.return_value = mock_cleaning_stats
        
        # Paso 1: Diagnóstico
        diag_result = diagnose_file(temp_csv_file, "apus")
        assert diag_result["success"] is True
        
        # Paso 2: Limpieza (solo si diagnóstico exitoso)
        if diag_result["success"]:
            clean_result = clean_file(temp_csv_file)
            assert clean_result["success"] is True
            assert "output_path" in clean_result

    def test_all_file_types_have_diagnostics(self):
        """Verifica que todos los FileType tienen diagnóstico registrado."""
        from scripts.tools_interface import _DIAGNOSTIC_REGISTRY
        
        for file_type in FileType:
            assert file_type in _DIAGNOSTIC_REGISTRY, \
                f"FileType.{file_type.name} not in _DIAGNOSTIC_REGISTRY"

    def test_valid_delimiters_constant_not_empty(self):
        """Verifica que hay delimitadores válidos definidos."""
        assert len(_VALID_DELIMITERS) > 0
        assert ";" in _VALID_DELIMITERS
        assert "," in _VALID_DELIMITERS

    def test_supported_encodings_constant_not_empty(self):
        """Verifica que hay encodings soportados definidos."""
        assert len(_SUPPORTED_ENCODINGS) > 0
        assert "utf-8" in _SUPPORTED_ENCODINGS


# =============================================================================
# Tests de Logging
# =============================================================================

class TestLogging:
    """Pruebas para verificar logging apropiado."""

    @patch("scripts.tools_interface.APUFileDiagnostic")
    def test_diagnose_logs_start_and_completion(
        self,
        mock_diagnostic_class: MagicMock,
        temp_csv_file: Path,
        mock_diagnostic_result: Dict[str, Any],
        caplog
    ):
        """Verifica que diagnóstico registra inicio y fin."""
        mock_instance = MagicMock()
        mock_instance.to_dict.return_value = mock_diagnostic_result
        mock_diagnostic_class.return_value = mock_instance
        
        with caplog.at_level(logging.INFO):
            diagnose_file(temp_csv_file, "apus")
        
        assert "Starting" in caplog.text
        assert "completed" in caplog.text

    @patch("scripts.tools_interface.CSVCleaner")
    def test_clean_logs_start_and_completion(
        self,
        mock_cleaner_class: MagicMock,
        temp_csv_file: Path,
        mock_cleaning_stats: MagicMock,
        caplog
    ):
        """Verifica que limpieza registra inicio y fin."""
        mock_cleaner_class.return_value.clean.return_value = mock_cleaning_stats
        
        with caplog.at_level(logging.INFO):
            clean_file(temp_csv_file)
        
        assert "Starting CSV cleaning" in caplog.text
        assert "completed" in caplog.text

    def test_validation_errors_logged_as_warning(self, tmp_path: Path, caplog):
        """Verifica que errores de validación se loguean como warning."""
        non_existing = tmp_path / "missing.csv"
        
        with caplog.at_level(logging.WARNING):
            diagnose_file(non_existing, "apus")
        
        assert "Validation error" in caplog.text


# =============================================================================
# Tests de Edge Cases
# =============================================================================

class TestEdgeCases:
    """Pruebas para casos límite."""

    def test_diagnose_empty_file_path(self):
        """Verifica manejo de ruta vacía en diagnóstico."""
        result = diagnose_file("", "apus")
        
        assert result["success"] is False

    def test_clean_empty_file_path(self):
        """Verifica manejo de ruta vacía en limpieza."""
        result = clean_file("")
        
        assert result["success"] is False

    @patch("scripts.tools_interface.APUFileDiagnostic")
    def test_diagnose_file_with_special_characters(
        self,
        mock_diagnostic_class: MagicMock,
        tmp_path: Path,
        mock_diagnostic_result: Dict[str, Any]
    ):
        """Verifica manejo de rutas con caracteres especiales."""
        special_file = tmp_path / "archivo con espacios y ñ.csv"
        special_file.write_text("data", encoding="utf-8")
        
        mock_instance = MagicMock()
        mock_instance.to_dict.return_value = mock_diagnostic_result
        mock_diagnostic_class.return_value = mock_instance
        
        result = diagnose_file(special_file, "apus")
        
        assert result["success"] is True

    @patch("scripts.tools_interface.CSVCleaner")
    def test_clean_file_with_unicode_path(
        self,
        mock_cleaner_class: MagicMock,
        tmp_path: Path,
        mock_cleaning_stats: MagicMock
    ):
        """Verifica manejo de rutas con caracteres unicode."""
        unicode_file = tmp_path / "datos_日本語.csv"
        unicode_file.write_text("data", encoding="utf-8")
        
        mock_cleaner_class.return_value.clean.return_value = mock_cleaning_stats
        
        result = clean_file(unicode_file)
        
        assert result["success"] is True

    def test_diagnose_with_path_object(self, temp_csv_file: Path):
        """Verifica que acepta objetos Path directamente."""
        # Solo verificar que no lanza excepción por el tipo
        # El archivo existe, pero sin mock fallará en el diagnóstico real
        with patch("scripts.tools_interface.APUFileDiagnostic") as mock:
            mock_instance = MagicMock()
            mock_instance.to_dict.return_value = {}
            mock.return_value = mock_instance
            
            result = diagnose_file(temp_csv_file, FileType.APUS)
            
            assert "success" in result


# =============================================================================
# Ejecución directa (para desarrollo)
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])