# tests/test_clean_csv.py
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ajustar path para importar el módulo
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.clean_csv import CleaningStats, CSVCleaner, SkipReason, main

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_dir():
    """Crea un directorio temporal para las pruebas."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_csv_path(temp_dir):
    """Crea un archivo CSV válido de muestra."""
    csv_path = temp_dir / "sample.csv"
    content = """Name;Age;City
John Doe;30;New York
Jane Smith;25;Los Angeles
Bob Johnson;35;Chicago"""

    csv_path.write_text(content, encoding="utf-8")
    return csv_path


@pytest.fixture
def problematic_csv_path(temp_dir):
    """Crea un CSV con diferentes tipos de problemas."""
    csv_path = temp_dir / "problematic.csv"
    content = (
        "Name;Age;City\n"
        "John Doe;30;New York\n"
        "# This is a comment\n"
        "Jane Smith;25;Los Angeles\n"
        "\n"
        "Bob Johnson;35;Chicago;Extra Column\n"
        "   ;   ;   \n"
        "Alice;28;Boston\n"
    )
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


@pytest.fixture
def empty_csv_path(temp_dir):
    """Crea un archivo CSV vacío."""
    csv_path = temp_dir / "empty.csv"
    csv_path.write_text("", encoding="utf-8")
    return csv_path


@pytest.fixture
def header_only_csv_path(temp_dir):
    """Crea un CSV solo con encabezado."""
    csv_path = temp_dir / "header_only.csv"
    csv_path.write_text("Name;Age;City\n", encoding="utf-8")
    return csv_path


@pytest.fixture
def duplicate_headers_csv_path(temp_dir):
    """Crea un CSV con encabezados duplicados."""
    csv_path = temp_dir / "duplicate_headers.csv"
    content = """Name;Age;Name;City
John;30;Doe;NYC"""
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


@pytest.fixture
def large_csv_path(temp_dir):
    """Crea un CSV grande para probar límites."""
    csv_path = temp_dir / "large.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Col1;Col2;Col3\n")
        for i in range(1000):
            f.write(f"Value{i};Data{i};Info{i}\n")
    return csv_path


@pytest.fixture
def comma_delimited_csv_path(temp_dir):
    """Crea un CSV con delimitador de coma."""
    csv_path = temp_dir / "comma_delimited.csv"
    content = """Name,Age,City
John Doe,30,New York
Jane Smith,25,Los Angeles"""
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


@pytest.fixture
def output_path(temp_dir):
    """Ruta para archivo de salida."""
    return temp_dir / "output.csv"


# ============================================================================
# TESTS DE CleaningStats
# ============================================================================


class TestCleaningStats:
    """Pruebas para la clase CleaningStats."""

    def test_initialization(self):
        """Verifica la inicialización correcta de estadísticas."""
        stats = CleaningStats()

        assert stats.rows_written == 0
        assert stats.rows_skipped == 0
        assert len(stats.skip_reasons) == len(SkipReason)
        assert all(count == 0 for count in stats.skip_reasons.values())

    def test_record_written(self):
        """Verifica el registro de filas escritas."""
        stats = CleaningStats()

        stats.record_written()
        assert stats.rows_written == 1

        stats.record_written()
        stats.record_written()
        assert stats.rows_written == 3

    def test_record_skip(self):
        """Verifica el registro de filas saltadas."""
        stats = CleaningStats()

        stats.record_skip(SkipReason.EMPTY)
        assert stats.rows_skipped == 1
        assert stats.skip_reasons[SkipReason.EMPTY] == 1

        stats.record_skip(SkipReason.COMMENT)
        stats.record_skip(SkipReason.COMMENT)
        assert stats.rows_skipped == 3
        assert stats.skip_reasons[SkipReason.COMMENT] == 2

    def test_multiple_skip_reasons(self):
        """Verifica el registro de múltiples razones de salto."""
        stats = CleaningStats()

        stats.record_skip(SkipReason.EMPTY)
        stats.record_skip(SkipReason.COMMENT)
        stats.record_skip(SkipReason.INCONSISTENT_DELIMITERS)
        stats.record_skip(SkipReason.EMPTY)

        assert stats.rows_skipped == 4
        assert stats.skip_reasons[SkipReason.EMPTY] == 2
        assert stats.skip_reasons[SkipReason.COMMENT] == 1
        assert stats.skip_reasons[SkipReason.INCONSISTENT_DELIMITERS] == 1


# ============================================================================
# TESTS DE VALIDACIÓN
# ============================================================================


class TestValidations:
    """Pruebas para el método _validate_inputs."""

    def test_nonexistent_input_file(self, temp_dir, output_path):
        """Verifica error cuando el archivo de entrada no existe."""
        nonexistent = temp_dir / "nonexistent.csv"

        cleaner = CSVCleaner(input_path=str(nonexistent), output_path=str(output_path))

        with pytest.raises(FileNotFoundError, match="no existe"):
            cleaner._validate_inputs()

    def test_input_is_directory(self, temp_dir, output_path):
        """Verifica error cuando la entrada es un directorio."""
        cleaner = CSVCleaner(input_path=str(temp_dir), output_path=str(output_path))

        with pytest.raises(ValueError, match="no es un archivo"):
            cleaner._validate_inputs()

    def test_empty_input_file(self, empty_csv_path, output_path):
        """Verifica error cuando el archivo está vacío."""
        cleaner = CSVCleaner(input_path=str(empty_csv_path), output_path=str(output_path))

        with pytest.raises(ValueError, match="está vacío"):
            cleaner._validate_inputs()

    def test_output_already_exists_no_overwrite(self, sample_csv_path, temp_dir):
        """Verifica error cuando el archivo de salida existe sin overwrite."""
        output = temp_dir / "existing.csv"
        output.write_text("existing content", encoding="utf-8")

        cleaner = CSVCleaner(
            input_path=str(sample_csv_path), output_path=str(output), overwrite=False
        )

        with pytest.raises(ValueError, match="ya existe"):
            cleaner._validate_inputs()

    def test_output_already_exists_with_overwrite(self, sample_csv_path, temp_dir):
        """Verifica que con overwrite=True no hay error."""
        output = temp_dir / "existing.csv"
        output.write_text("existing content", encoding="utf-8")

        cleaner = CSVCleaner(
            input_path=str(sample_csv_path), output_path=str(output), overwrite=True
        )

        # No debe lanzar excepción
        cleaner._validate_inputs()

    def test_nonexistent_output_directory(self, sample_csv_path, temp_dir):
        """Verifica error cuando el directorio de salida no existe."""
        output = temp_dir / "nonexistent_dir" / "output.csv"

        cleaner = CSVCleaner(input_path=str(sample_csv_path), output_path=str(output))

        with pytest.raises(ValueError, match="directorio de salida no existe"):
            cleaner._validate_inputs()

    def test_empty_delimiter(self, sample_csv_path, output_path):
        """Verifica error con delimitador vacío."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path), output_path=str(output_path), delimiter=""
        )

        with pytest.raises(ValueError, match="no puede estar vacío"):
            cleaner._validate_inputs()

    def test_multi_character_delimiter(self, sample_csv_path, output_path):
        """Verifica error con delimitador de múltiples caracteres."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path), output_path=str(output_path), delimiter=";;"
        )

        with pytest.raises(ValueError, match="un solo carácter"):
            cleaner._validate_inputs()

    def test_same_input_output_file(self, sample_csv_path):
        """Verifica error cuando entrada y salida son el mismo archivo."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path), output_path=str(sample_csv_path)
        )

        with pytest.raises(ValueError, match="no pueden ser el mismo"):
            cleaner._validate_inputs()

    @patch("scripts.clean_csv.logger")
    def test_unusual_delimiter_warning(self, mock_logger, sample_csv_path, output_path):
        """Verifica advertencia con delimitador inusual."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path), output_path=str(output_path), delimiter="@"
        )

        cleaner._validate_inputs()

        # Debe haber generado una advertencia
        mock_logger.warning.assert_called()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "inusual" in warning_message.lower()

    def test_valid_delimiters_no_warning(self, sample_csv_path, output_path):
        """Verifica que delimitadores válidos no generen advertencia."""
        valid_delimiters = [";", ",", "\t", "|"]

        for delimiter in valid_delimiters:
            cleaner = CSVCleaner(
                input_path=str(sample_csv_path),
                output_path=str(output_path),
                delimiter=delimiter,
            )
            # No debe lanzar excepción
            cleaner._validate_inputs()


# ============================================================================
# TESTS DE PROCESAMIENTO DE ENCABEZADO
# ============================================================================


class TestProcessHeader:
    """Pruebas para el método _process_header."""

    def test_process_valid_header(self, sample_csv_path):
        """Verifica el procesamiento correcto de un encabezado válido."""
        cleaner = CSVCleaner(str(sample_csv_path), "dummy.csv")
        header_line = "Name;Age;City"

        cleaner._process_header(header_line)

        assert cleaner.expected_delimiter_count == 2
        assert cleaner.stats.rows_written == 0  # No debe modificar estadísticas

    def test_empty_header_raises_error(self, temp_dir):
        """Verifica que un encabezado completamente vacío lanza un error."""
        cleaner = CSVCleaner("dummy_in.csv", "dummy_out.csv")

        with pytest.raises(ValueError, match="El encabezado del CSV está vacío"):
            cleaner._process_header("")

    def test_blank_header_raises_error(self, temp_dir):
        """Verifica que un encabezado con solo espacios y delimitadores lanza error."""
        cleaner = CSVCleaner("dummy_in.csv", "dummy_out.csv")

        with pytest.raises(ValueError, match="contiene solo espacios en blanco"):
            cleaner._process_header("   ;   ;   ")

    @patch("scripts.clean_csv.logger")
    def test_duplicate_headers_warning(self, mock_logger, duplicate_headers_csv_path):
        """Verifica que se emite una advertencia con encabezados duplicados."""
        cleaner = CSVCleaner(str(duplicate_headers_csv_path), "dummy.csv")
        header_line = "Name;Age;Name;City"

        cleaner._process_header(header_line)

        mock_logger.warning.assert_called()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "duplicados" in warning_message


# ============================================================================
# TESTS DE VALIDACIÓN DE LÍNEAS
# ============================================================================


class TestShouldSkipLine:
    """Pruebas para el método _should_skip_line."""

    @pytest.fixture
    def cleaner(self, sample_csv_path):
        """Prepara un limpiador con el encabezado ya procesado."""
        cleaner = CSVCleaner(str(sample_csv_path), "dummy.csv")
        # Establecer el conteo esperado de delimitadores
        cleaner._process_header("Name;Age;City")
        return cleaner

    def test_valid_line_not_skipped(self, cleaner):
        """Verifica que una línea válida no se salta."""
        line = "John Doe;30;New York"
        result = cleaner._should_skip_line(line, line_num=2)
        assert result is None

    def test_empty_line_skipped(self, cleaner):
        """Verifica que una línea completamente vacía se salta."""
        line = ""
        result = cleaner._should_skip_line(line, line_num=2)
        assert result == SkipReason.EMPTY

    def test_whitespace_line_skipped(self, cleaner):
        """Verifica que una línea con solo espacios se salta."""
        line = "      "
        result = cleaner._should_skip_line(line, line_num=2)
        assert result == SkipReason.EMPTY

    def test_blank_fields_line_skipped(self, cleaner):
        """Verifica que una línea con campos en blanco se salta."""
        line = "   ;   ;   "
        result = cleaner._should_skip_line(line, line_num=2)
        assert result == SkipReason.WHITESPACE_ONLY

    def test_comment_line_skipped(self, cleaner):
        """Verifica que una línea de comentario se salta."""
        line = "# This is a comment"
        result = cleaner._should_skip_line(line, line_num=2)
        assert result == SkipReason.COMMENT

    def test_inconsistent_delimiters_strict_mode(self, cleaner):
        """Verifica que delimitadores inconsistentes se saltan en modo estricto."""
        cleaner.strict_mode = True
        line = "John;30;NYC;Extra"
        result = cleaner._should_skip_line(line, line_num=2)
        assert result == SkipReason.INCONSISTENT_DELIMITERS

    def test_inconsistent_delimiters_non_strict_mode(self, cleaner):
        """Verifica que delimitadores inconsistentes NO se saltan en modo no estricto."""
        cleaner.strict_mode = False
        line = "John;30;NYC;Extra"
        result = cleaner._should_skip_line(line, line_num=2)
        assert result is None

    def test_comment_with_leading_spaces(self, cleaner):
        """Verifica que un comentario con espacios al inicio se detecta."""
        line = "  # Comment with spaces"
        result = cleaner._should_skip_line(line, line_num=2)
        assert result == SkipReason.COMMENT


# ============================================================================
# TESTS DE LIMPIEZA COMPLETA
# ============================================================================


class TestCleanMethod:
    """Pruebas para el método clean (integración)."""

    def test_clean_preserves_original_format(self, temp_dir, output_path):
        """Verifica que el limpiador preserva el formato original exacto."""
        input_content = (
            '"Name";"Age";"City"\n'
            ' "John Doe" ; 30 ; "New York" \n'
            "# Comment to be skipped\n"
            '"Jane Smith";25;"Los Angeles"\n'
        )
        input_path = temp_dir / "input.csv"
        input_path.write_text(input_content, encoding="utf-8")

        cleaner = CSVCleaner(str(input_path), str(output_path))
        stats = cleaner.clean()

        assert stats.rows_written == 2
        assert stats.rows_skipped == 1

        output_content = output_path.read_text(encoding="utf-8")

        expected_output = (
            '"Name";"Age";"City"\n'
            ' "John Doe" ; 30 ; "New York" \n'
            '"Jane Smith";25;"Los Angeles"\n'
        )

        assert output_content == expected_output

    def test_clean_problematic_csv(self, problematic_csv_path, output_path):
        """Verifica la limpieza de un CSV con varios problemas."""
        cleaner = CSVCleaner(str(problematic_csv_path), str(output_path))
        stats = cleaner.clean()

        assert output_path.exists()
        assert stats.rows_written == 3
        assert stats.rows_skipped == 4

        assert stats.skip_reasons[SkipReason.COMMENT] == 1
        assert stats.skip_reasons[SkipReason.EMPTY] == 1
        assert stats.skip_reasons[SkipReason.INCONSISTENT_DELIMITERS] == 1
        assert stats.skip_reasons[SkipReason.WHITESPACE_ONLY] == 1

        # Verificar contenido
        output_content = output_path.read_text(encoding="utf-8")
        expected_content = (
            "Name;Age;City\n"
            "John Doe;30;New York\n"
            "Jane Smith;25;Los Angeles\n"
            "Alice;28;Boston\n"
        )
        assert output_content == expected_content

    def test_clean_with_comma_delimiter(self, comma_delimited_csv_path, output_path):
        """Verifica la limpieza con delimitador de coma."""
        cleaner = CSVCleaner(
            input_path=str(comma_delimited_csv_path),
            output_path=str(output_path),
            delimiter=",",
        )
        stats = cleaner.clean()
        assert stats.rows_written == 2

        original_content = comma_delimited_csv_path.read_text(encoding="utf-8")
        output_content = output_path.read_text(encoding="utf-8")
        assert original_content == output_content

    def test_clean_header_only_csv(self, header_only_csv_path, output_path):
        """Verifica la limpieza de un CSV solo con encabezado."""
        cleaner = CSVCleaner(str(header_only_csv_path), str(output_path))
        stats = cleaner.clean()
        assert stats.rows_written == 0

        output_content = output_path.read_text(encoding="utf-8")
        assert output_content == "Name;Age;City\n"

    def test_clean_non_strict_mode(self, problematic_csv_path, output_path):
        """Verifica que el modo no estricto permite delimitadores inconsistentes."""
        cleaner = CSVCleaner(
            input_path=str(problematic_csv_path),
            output_path=str(output_path),
            strict_mode=False,
        )
        stats = cleaner.clean()

        # La línea con delimitadores extra ahora debe ser incluida
        assert stats.rows_written == 4
        assert stats.skip_reasons[SkipReason.INCONSISTENT_DELIMITERS] == 0

        output_content = output_path.read_text(encoding="utf-8")
        assert "Bob Johnson;35;Chicago;Extra Column" in output_content

    def test_statistics_accuracy(self, problematic_csv_path, output_path):
        """Verifica la precisión de las estadísticas."""
        cleaner = CSVCleaner(
            input_path=str(problematic_csv_path),
            output_path=str(output_path),
        )
        stats = cleaner.clean()

        total_from_reasons = sum(stats.skip_reasons.values())
        assert total_from_reasons == stats.rows_skipped

        assert stats.skip_reasons[SkipReason.COMMENT] == 1
        assert stats.skip_reasons[SkipReason.EMPTY] == 1
        assert stats.skip_reasons[SkipReason.WHITESPACE_ONLY] == 1
        assert stats.skip_reasons[SkipReason.INCONSISTENT_DELIMITERS] == 1


# ============================================================================
# TESTS DE MANEJO DE ERRORES
# ============================================================================


class TestErrorHandling:
    """Pruebas para manejo de errores."""

    def test_permission_denied_raises_ioerror(self, temp_dir):
        """Verifica que un error de permisos lanza IOError."""
        input_file = temp_dir / "no_read.csv"
        input_file.write_text("header\ndata", encoding="utf-8")
        output_file = temp_dir / "output.csv"

        import platform

        if platform.system() != "Windows":
            input_file.chmod(0o000)

            cleaner = CSVCleaner(str(input_file), str(output_file))

            with pytest.raises(PermissionError):
                cleaner.clean()

            input_file.chmod(0o644)


# ============================================================================
# TESTS DE LÍNEA DE COMANDOS (main)
# ============================================================================


class TestMainFunction:
    """Pruebas para la función main."""

    def test_main_help(self):
        """Verifica que --help funciona."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["clean_csv.py", "--help"]):
                main()

        assert exc_info.value.code == 0

    def test_main_success(self, sample_csv_path, output_path):
        """Verifica ejecución exitosa desde main."""
        with patch("sys.argv", ["clean_csv.py", str(sample_csv_path), str(output_path)]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0
            assert output_path.exists()

    def test_main_with_delimiter_option(self, comma_delimited_csv_path, output_path):
        """Verifica opción de delimitador desde CLI."""
        with patch(
            "sys.argv",
            ["clean_csv.py", str(comma_delimited_csv_path), str(output_path), "-d", ","],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_with_overwrite_option(self, sample_csv_path, temp_dir):
        """Verifica opción de sobrescritura."""
        output = temp_dir / "existing.csv"
        output.write_text("old content", encoding="utf-8")

        with patch("sys.argv", ["clean_csv.py", str(sample_csv_path), str(output), "-o"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

            # Verificar que se sobrescribió
            content = output.read_text()
            assert "old content" not in content

    def test_main_with_verbose_option(self, sample_csv_path, output_path):
        """Verifica opción verbose."""
        with patch(
            "sys.argv", ["clean_csv.py", str(sample_csv_path), str(output_path), "-v"]
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_with_no_strict_option(self, problematic_csv_path, output_path):
        """Verifica opción no-strict."""
        with patch(
            "sys.argv",
            ["clean_csv.py", str(problematic_csv_path), str(output_path), "--no-strict"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    def test_main_file_not_found_error(self, temp_dir):
        """Verifica error de archivo no encontrado desde CLI."""
        with patch(
            "sys.argv",
            [
                "clean_csv.py",
                str(temp_dir / "nonexistent.csv"),
                str(temp_dir / "output.csv"),
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    def test_main_keyboard_interrupt(self, sample_csv_path, output_path):
        """Verifica manejo de Ctrl+C."""
        with patch("sys.argv", ["clean_csv.py", str(sample_csv_path), str(output_path)]):
            with patch.object(CSVCleaner, "clean", side_effect=KeyboardInterrupt()):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 130


# ============================================================================
# TESTS DE INTEGRACIÓN AVANZADOS
# ============================================================================


class TestAdvancedIntegration:
    """Pruebas de integración más complejas."""

    def test_special_characters_in_data(self, temp_dir, output_path):
        """Verifica manejo de caracteres especiales."""
        input_file = temp_dir / "special_chars.csv"
        content = """Name;Description
Test;"Quote ""inside"" quote"
Test2;Line with; semicolon
Test3;Emoji 😀"""
        input_file.write_text(content, encoding="utf-8")

        cleaner = CSVCleaner(input_path=str(input_file), output_path=str(output_path))

        stats = cleaner.clean()

        assert stats.rows_written == 2

    def test_mixed_line_endings(self, temp_dir, output_path):
        """Verifica manejo de diferentes finales de línea."""
        input_file = temp_dir / "mixed_endings.csv"
        # Mezclar \n y \r\n
        content = "Name;Age\nJohn;30\r\nJane;25\n"
        input_file.write_bytes(content.encode("utf-8"))

        cleaner = CSVCleaner(input_path=str(input_file), output_path=str(output_path))

        stats = cleaner.clean()

        assert stats.rows_written == 2

    def test_unicode_normalization(self, temp_dir, output_path):
        """Verifica manejo de caracteres Unicode."""
        input_file = temp_dir / "unicode.csv"
        content = """Name;City
José;São Paulo
François;Montréal
北京;中国"""
        input_file.write_text(content, encoding="utf-8")

        cleaner = CSVCleaner(input_path=str(input_file), output_path=str(output_path))

        stats = cleaner.clean()

        assert stats.rows_written == 3

        # Verificar que los caracteres se mantuvieron
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "José" in content
        assert "北京" in content

    def test_very_long_lines(self, temp_dir, output_path):
        """Verifica manejo de líneas muy largas."""
        input_file = temp_dir / "long_lines.csv"
        long_value = "x" * 10000

        with open(input_file, "w", encoding="utf-8") as f:
            f.write("Col1;Col2;Col3\n")
            f.write(f"{long_value};data;data\n")

        cleaner = CSVCleaner(input_path=str(input_file), output_path=str(output_path))

        stats = cleaner.clean()

        assert stats.rows_written == 1


# ============================================================================
# TESTS DE PERFORMANCE
# ============================================================================


class TestPerformance:
    """Pruebas de rendimiento."""

    def test_large_file_processing(self, temp_dir, output_path):
        """Verifica procesamiento eficiente de archivos grandes."""
        import time

        input_file = temp_dir / "very_large.csv"

        # Crear archivo con 10,000 filas
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("Col1;Col2;Col3;Col4;Col5\n")
            for i in range(10000):
                f.write(f"Val{i};Data{i};Info{i};Test{i};More{i}\n")

        cleaner = CSVCleaner(input_path=str(input_file), output_path=str(output_path))

        start_time = time.time()
        stats = cleaner.clean()
        elapsed_time = time.time() - start_time

        assert stats.rows_written == 10000
        # Debe procesar en menos de 5 segundos
        assert elapsed_time < 5.0


# ============================================================================
# TESTS DE CONFIGURACIÓN
# ============================================================================


class TestConfiguration:
    """Pruebas de diferentes configuraciones."""

    @pytest.mark.parametrize("delimiter", [";", ",", "\t", "|"])
    def test_different_delimiters(self, temp_dir, delimiter):
        """Prueba con diferentes delimitadores."""
        input_file = temp_dir / f"delim_{ord(delimiter)}.csv"
        output_file = temp_dir / f"out_{ord(delimiter)}.csv"

        # Crear CSV con el delimitador específico
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(f"Name{delimiter}Age{delimiter}City\n")
            f.write(f"John{delimiter}30{delimiter}NYC\n")

        cleaner = CSVCleaner(
            input_path=str(input_file), output_path=str(output_file), delimiter=delimiter
        )

        stats = cleaner.clean()

        assert stats.rows_written == 1

    @pytest.mark.parametrize("encoding", ["utf-8", "latin-1", "cp1252"])
    def test_different_encodings(self, temp_dir, encoding):
        """Prueba con diferentes encodings."""
        input_file = temp_dir / f"enc_{encoding}.csv"
        output_file = temp_dir / f"out_{encoding}.csv"

        try:
            input_file.write_text("Name;City\nTest;Test\n", encoding=encoding)

            cleaner = CSVCleaner(
                input_path=str(input_file), output_path=str(output_file), encoding=encoding
            )

            stats = cleaner.clean()

            assert stats.rows_written == 1
        except LookupError:
            pytest.skip(f"Encoding {encoding} no disponible")


# ============================================================================
# CONFIGURACIÓN DE PYTEST
# ============================================================================

if __name__ == "__main__":
    pytest.main(
        [__file__, "-v", "--tb=short", "--cov=scripts.clean_csv", "--cov-report=html"]
    )
