# tests/test_clean_csv.py
import pytest
import csv
import sys
from pathlib import Path
from io import StringIO
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Ajustar path para importar el módulo
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.clean_csv import (
    CSVCleaner,
    CleaningStats,
    SkipReason,
    main
)


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
    
    csv_path.write_text(content, encoding='utf-8')
    return csv_path


@pytest.fixture
def problematic_csv_path(temp_dir):
    """Crea un CSV con diferentes tipos de problemas."""
    csv_path = temp_dir / "problematic.csv"
    content = """Name;Age;City
John Doe;30;New York
# This is a comment
Jane Smith;25;Los Angeles
;;
Bob Johnson;35;Chicago;Extra Column
   ;   ;   
Alice;28;Boston
# Another comment
"""
    csv_path.write_text(content, encoding='utf-8')
    return csv_path


@pytest.fixture
def empty_csv_path(temp_dir):
    """Crea un archivo CSV vacío."""
    csv_path = temp_dir / "empty.csv"
    csv_path.write_text("", encoding='utf-8')
    return csv_path


@pytest.fixture
def header_only_csv_path(temp_dir):
    """Crea un CSV solo con encabezado."""
    csv_path = temp_dir / "header_only.csv"
    csv_path.write_text("Name;Age;City\n", encoding='utf-8')
    return csv_path


@pytest.fixture
def duplicate_headers_csv_path(temp_dir):
    """Crea un CSV con encabezados duplicados."""
    csv_path = temp_dir / "duplicate_headers.csv"
    content = """Name;Age;Name;City
John;30;Doe;NYC"""
    csv_path.write_text(content, encoding='utf-8')
    return csv_path


@pytest.fixture
def large_csv_path(temp_dir):
    """Crea un CSV grande para probar límites."""
    csv_path = temp_dir / "large.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
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
    csv_path.write_text(content, encoding='utf-8')
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
        stats.record_skip(SkipReason.INCONSISTENT_COLUMNS)
        stats.record_skip(SkipReason.EMPTY)
        
        assert stats.rows_skipped == 4
        assert stats.skip_reasons[SkipReason.EMPTY] == 2
        assert stats.skip_reasons[SkipReason.COMMENT] == 1
        assert stats.skip_reasons[SkipReason.INCONSISTENT_COLUMNS] == 1


# ============================================================================
# TESTS DE VALIDACIÓN
# ============================================================================

class TestValidations:
    """Pruebas para el método _validate_inputs."""
    
    def test_nonexistent_input_file(self, temp_dir, output_path):
        """Verifica error cuando el archivo de entrada no existe."""
        nonexistent = temp_dir / "nonexistent.csv"
        
        cleaner = CSVCleaner(
            input_path=str(nonexistent),
            output_path=str(output_path)
        )
        
        with pytest.raises(FileNotFoundError, match="no existe"):
            cleaner._validate_inputs()
    
    def test_input_is_directory(self, temp_dir, output_path):
        """Verifica error cuando la entrada es un directorio."""
        cleaner = CSVCleaner(
            input_path=str(temp_dir),
            output_path=str(output_path)
        )
        
        with pytest.raises(ValueError, match="no es un archivo"):
            cleaner._validate_inputs()
    
    def test_empty_input_file(self, empty_csv_path, output_path):
        """Verifica error cuando el archivo está vacío."""
        cleaner = CSVCleaner(
            input_path=str(empty_csv_path),
            output_path=str(output_path)
        )
        
        with pytest.raises(ValueError, match="está vacío"):
            cleaner._validate_inputs()
    
    def test_output_already_exists_no_overwrite(self, sample_csv_path, temp_dir):
        """Verifica error cuando el archivo de salida existe sin overwrite."""
        output = temp_dir / "existing.csv"
        output.write_text("existing content", encoding='utf-8')
        
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path=str(output),
            overwrite=False
        )
        
        with pytest.raises(ValueError, match="ya existe"):
            cleaner._validate_inputs()
    
    def test_output_already_exists_with_overwrite(self, sample_csv_path, temp_dir):
        """Verifica que con overwrite=True no hay error."""
        output = temp_dir / "existing.csv"
        output.write_text("existing content", encoding='utf-8')
        
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path=str(output),
            overwrite=True
        )
        
        # No debe lanzar excepción
        cleaner._validate_inputs()
    
    def test_nonexistent_output_directory(self, sample_csv_path, temp_dir):
        """Verifica error cuando el directorio de salida no existe."""
        output = temp_dir / "nonexistent_dir" / "output.csv"
        
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path=str(output)
        )
        
        with pytest.raises(ValueError, match="directorio de salida no existe"):
            cleaner._validate_inputs()
    
    def test_empty_delimiter(self, sample_csv_path, output_path):
        """Verifica error con delimitador vacío."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path=str(output_path),
            delimiter=""
        )
        
        with pytest.raises(ValueError, match="no puede estar vacío"):
            cleaner._validate_inputs()
    
    def test_multi_character_delimiter(self, sample_csv_path, output_path):
        """Verifica error con delimitador de múltiples caracteres."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path=str(output_path),
            delimiter=";;"
        )
        
        with pytest.raises(ValueError, match="un solo carácter"):
            cleaner._validate_inputs()
    
    def test_same_input_output_file(self, sample_csv_path):
        """Verifica error cuando entrada y salida son el mismo archivo."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path=str(sample_csv_path)
        )
        
        with pytest.raises(ValueError, match="no pueden ser el mismo"):
            cleaner._validate_inputs()
    
    @patch('scripts.clean_csv.logger')
    def test_unusual_delimiter_warning(self, mock_logger, sample_csv_path, output_path):
        """Verifica advertencia con delimitador inusual."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path=str(output_path),
            delimiter="@"
        )
        
        cleaner._validate_inputs()
        
        # Debe haber generado una advertencia
        mock_logger.warning.assert_called()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "inusual" in warning_message.lower()
    
    def test_valid_delimiters_no_warning(self, sample_csv_path, output_path):
        """Verifica que delimitadores válidos no generen advertencia."""
        valid_delimiters = [';', ',', '\t', '|']
        
        for delimiter in valid_delimiters:
            cleaner = CSVCleaner(
                input_path=str(sample_csv_path),
                output_path=str(output_path),
                delimiter=delimiter
            )
            # No debe lanzar excepción
            cleaner._validate_inputs()


# ============================================================================
# TESTS DE LECTURA DE ENCABEZADO
# ============================================================================

class TestReadHeader:
    """Pruebas para el método _read_header."""
    
    def test_read_valid_header(self, sample_csv_path):
        """Verifica lectura correcta de encabezado válido."""
        with open(sample_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            cleaner = CSVCleaner(
                input_path=str(sample_csv_path),
                output_path="dummy.csv"
            )
            
            header = cleaner._read_header(reader)
            
            assert header == ['Name', 'Age', 'City']
    
    def test_empty_csv_raises_error(self, temp_dir):
        """Verifica error con CSV completamente vacío."""
        empty_file = temp_dir / "empty.csv"
        empty_file.write_text("", encoding='utf-8')
        
        with open(empty_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            cleaner = CSVCleaner(
                input_path=str(empty_file),
                output_path="dummy.csv"
            )
            
            with pytest.raises(ValueError, match="vacío"):
                cleaner._read_header(reader)
    
    def test_blank_header_raises_error(self, temp_dir):
        """Verifica error con encabezado en blanco."""
        blank_header = temp_dir / "blank_header.csv"
        blank_header.write_text(";;;\n", encoding='utf-8')
        
        with open(blank_header, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            cleaner = CSVCleaner(
                input_path=str(blank_header),
                output_path="dummy.csv"
            )
            
            with pytest.raises(ValueError, match="vacío"):
                cleaner._read_header(reader)
    
    @patch('scripts.clean_csv.logger')
    def test_duplicate_headers_warning(self, mock_logger, duplicate_headers_csv_path):
        """Verifica advertencia con encabezados duplicados."""
        with open(duplicate_headers_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            cleaner = CSVCleaner(
                input_path=str(duplicate_headers_csv_path),
                output_path="dummy.csv"
            )
            
            header = cleaner._read_header(reader)
            
            # Debe haber generado advertencia
            mock_logger.warning.assert_called()
            warning_message = mock_logger.warning.call_args[0][0]
            assert "duplicados" in warning_message.lower()


# ============================================================================
# TESTS DE VALIDACIÓN DE FILAS
# ============================================================================

class TestShouldSkipRow:
    """Pruebas para el método _should_skip_row."""
    
    def test_valid_row_not_skipped(self, sample_csv_path):
        """Verifica que una fila válida no se salta."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path="dummy.csv"
        )
        
        row = ['John Doe', '30', 'New York']
        result = cleaner._should_skip_row(row, num_columns=3, line_num=2)
        
        assert result is None
    
    def test_empty_row_skipped(self, sample_csv_path):
        """Verifica que una fila vacía se salta."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path="dummy.csv"
        )
        
        row = []
        result = cleaner._should_skip_row(row, num_columns=3, line_num=2)
        
        assert result == SkipReason.EMPTY
    
    def test_blank_fields_row_skipped(self, sample_csv_path):
        """Verifica que una fila con campos en blanco se salta."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path="dummy.csv"
        )
        
        row = ['  ', '  ', '  ']
        result = cleaner._should_skip_row(row, num_columns=3, line_num=2)
        
        assert result == SkipReason.EMPTY
    
    def test_comment_row_skipped(self, sample_csv_path):
        """Verifica que una fila de comentario se salta."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path="dummy.csv"
        )
        
        row = ['# This is a comment', 'data', 'data']
        result = cleaner._should_skip_row(row, num_columns=3, line_num=2)
        
        assert result == SkipReason.COMMENT
    
    def test_inconsistent_columns_strict_mode(self, sample_csv_path):
        """Verifica que columnas inconsistentes se saltan en modo estricto."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path="dummy.csv",
            strict_mode=True
        )
        
        row = ['John', '30', 'NYC', 'Extra']
        result = cleaner._should_skip_row(row, num_columns=3, line_num=2)
        
        assert result == SkipReason.INCONSISTENT_COLUMNS
    
    def test_inconsistent_columns_non_strict_mode(self, sample_csv_path):
        """Verifica que columnas inconsistentes NO se saltan en modo no estricto."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path="dummy.csv",
            strict_mode=False
        )
        
        row = ['John', '30', 'NYC', 'Extra']
        result = cleaner._should_skip_row(row, num_columns=3, line_num=2)
        
        assert result is None
    
    def test_comment_with_spaces(self, sample_csv_path):
        """Verifica detección de comentario con espacios."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path="dummy.csv"
        )
        
        row = ['  # Comment with spaces', 'data', 'data']
        result = cleaner._should_skip_row(row, num_columns=3, line_num=2)
        
        assert result == SkipReason.COMMENT


# ============================================================================
# TESTS DE LIMPIEZA COMPLETA
# ============================================================================

class TestCleanMethod:
    """Pruebas para el método clean (integración)."""
    
    def test_clean_valid_csv(self, sample_csv_path, output_path):
        """Verifica limpieza exitosa de CSV válido."""
        cleaner = CSVCleaner(
            input_path=str(sample_csv_path),
            output_path=str(output_path)
        )
        
        stats = cleaner.clean()
        
        assert output_path.exists()
        assert stats.rows_written == 3
        assert stats.rows_skipped == 0
        
        # Verificar contenido
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            rows = list(reader)
            
        assert len(rows) == 4  # header + 3 data rows
        assert rows[0] == ['Name', 'Age', 'City']
    
    def test_clean_problematic_csv(self, problematic_csv_path, output_path):
        """Verifica limpieza de CSV con problemas."""
        cleaner = CSVCleaner(
            input_path=str(problematic_csv_path),
            output_path=str(output_path)
        )
        
        stats = cleaner.clean()
        
        assert output_path.exists()
        assert stats.rows_written == 3  # Solo las filas válidas
        assert stats.rows_skipped > 0
        assert stats.skip_reasons[SkipReason.COMMENT] == 2
        assert stats.skip_reasons[SkipReason.EMPTY] >= 1
        assert stats.skip_reasons[SkipReason.INCONSISTENT_COLUMNS] >= 1
    
    def test_clean_with_comma_delimiter(self, comma_delimited_csv_path, output_path):
        """Verifica limpieza con delimitador de coma."""
        cleaner = CSVCleaner(
            input_path=str(comma_delimited_csv_path),
            output_path=str(output_path),
            delimiter=','
        )
        
        stats = cleaner.clean()
        
        assert stats.rows_written == 2
        
        # Verificar delimitador correcto en salida
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert ',' in content
        assert ';' not in content
    
    def test_clean_header_only_csv(self, header_only_csv_path, output_path):
        """Verifica limpieza de CSV solo con encabezado."""
        cleaner = CSVCleaner(
            input_path=str(header_only_csv_path),
            output_path=str(output_path)
        )
        
        stats = cleaner.clean()
        
        assert stats.rows_written == 0
        
        # Verificar que el encabezado se escribió
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            rows = list(reader)
        
        assert len(rows) == 1
        assert rows[0] == ['Name', 'Age', 'City']
    
    def test_clean_large_csv(self, large_csv_path, output_path):
        """Verifica limpieza de CSV grande."""
        cleaner = CSVCleaner(
            input_path=str(large_csv_path),
            output_path=str(output_path)
        )
        
        stats = cleaner.clean()
        
        assert stats.rows_written == 1000
        assert output_path.exists()
    
    def test_clean_non_strict_mode(self, problematic_csv_path, output_path):
        """Verifica modo no estricto permite columnas inconsistentes."""
        cleaner = CSVCleaner(
            input_path=str(problematic_csv_path),
            output_path=str(output_path),
            strict_mode=False
        )
        
        stats = cleaner.clean()
        
        # En modo no estricto, algunas filas inconsistentes se aceptan
        assert stats.rows_written >= 3
    
    def test_clean_with_different_encoding(self, temp_dir, output_path):
        """Verifica limpieza con diferentes encodings."""
        # Crear archivo con encoding latin-1
        input_file = temp_dir / "latin1.csv"
        content = "Name;City\nJosé;São Paulo\n"
        input_file.write_text(content, encoding='latin-1')
        
        cleaner = CSVCleaner(
            input_path=str(input_file),
            output_path=str(output_path),
            encoding='latin-1'
        )
        
        stats = cleaner.clean()
        
        assert stats.rows_written == 1
    
    def test_statistics_accuracy(self, problematic_csv_path, output_path):
        """Verifica precisión de las estadísticas."""
        cleaner = CSVCleaner(
            input_path=str(problematic_csv_path),
            output_path=str(output_path),
            verbose=True
        )
        
        stats = cleaner.clean()
        
        # Verificar que la suma es coherente
        total_skip_reasons = sum(stats.skip_reasons.values())
        assert total_skip_reasons == stats.rows_skipped
        
        # Verificar que se registraron todas las razones
        assert stats.skip_reasons[SkipReason.COMMENT] > 0
        assert stats.skip_reasons[SkipReason.EMPTY] > 0


# ============================================================================
# TESTS DE MANEJO DE ERRORES
# ============================================================================

class TestErrorHandling:
    """Pruebas para manejo de errores."""
    
    def test_permission_denied_input(self, temp_dir, output_path):
        """Verifica manejo de error de permisos en lectura."""
        input_file = temp_dir / "no_read.csv"
        input_file.write_text("Name;Age\nJohn;30\n", encoding='utf-8')
        
        # Simular archivo sin permisos de lectura (solo en Unix)
        import platform
        if platform.system() != 'Windows':
            input_file.chmod(0o000)
            
            cleaner = CSVCleaner(
                input_path=str(input_file),
                output_path=str(output_path)
            )
            
            with pytest.raises(IOError):
                cleaner.clean()
            
            # Restaurar permisos para cleanup
            input_file.chmod(0o644)
    
    def test_csv_parsing_error(self, temp_dir, output_path):
        """Verifica manejo de errores de parsing CSV."""
        # Crear archivo con formato inválido
        input_file = temp_dir / "invalid.csv"
        # CSV con comillas sin cerrar puede causar problemas
        content = 'Name;Age\n"Unclosed quote;30\n'
        input_file.write_text(content, encoding='utf-8')
        
        cleaner = CSVCleaner(
            input_path=str(input_file),
            output_path=str(output_path)
        )
        
        # Puede o no lanzar excepción dependiendo del parser
        # pero no debe crashear
        try:
            cleaner.clean()
        except (ValueError, RuntimeError):
            pass


# ============================================================================
# TESTS DE LÍNEA DE COMANDOS (main)
# ============================================================================

class TestMainFunction:
    """Pruebas para la función main."""
    
    def test_main_help(self):
        """Verifica que --help funciona."""
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['clean_csv.py', '--help']):
                main()
        
        assert exc_info.value.code == 0
    
    def test_main_success(self, sample_csv_path, output_path):
        """Verifica ejecución exitosa desde main."""
        with patch('sys.argv', [
            'clean_csv.py',
            str(sample_csv_path),
            str(output_path)
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 0
            assert output_path.exists()
    
    def test_main_with_delimiter_option(self, comma_delimited_csv_path, output_path):
        """Verifica opción de delimitador desde CLI."""
        with patch('sys.argv', [
            'clean_csv.py',
            str(comma_delimited_csv_path),
            str(output_path),
            '-d', ','
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 0
    
    def test_main_with_overwrite_option(self, sample_csv_path, temp_dir):
        """Verifica opción de sobrescritura."""
        output = temp_dir / "existing.csv"
        output.write_text("old content", encoding='utf-8')
        
        with patch('sys.argv', [
            'clean_csv.py',
            str(sample_csv_path),
            str(output),
            '-o'
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 0
            
            # Verificar que se sobrescribió
            content = output.read_text()
            assert "old content" not in content
    
    def test_main_with_verbose_option(self, sample_csv_path, output_path):
        """Verifica opción verbose."""
        with patch('sys.argv', [
            'clean_csv.py',
            str(sample_csv_path),
            str(output_path),
            '-v'
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 0
    
    def test_main_with_no_strict_option(self, problematic_csv_path, output_path):
        """Verifica opción no-strict."""
        with patch('sys.argv', [
            'clean_csv.py',
            str(problematic_csv_path),
            str(output_path),
            '--no-strict'
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 0
    
    def test_main_file_not_found_error(self, temp_dir):
        """Verifica error de archivo no encontrado desde CLI."""
        with patch('sys.argv', [
            'clean_csv.py',
            str(temp_dir / "nonexistent.csv"),
            str(temp_dir / "output.csv")
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 1
    
    def test_main_keyboard_interrupt(self, sample_csv_path, output_path):
        """Verifica manejo de Ctrl+C."""
        with patch('sys.argv', [
            'clean_csv.py',
            str(sample_csv_path),
            str(output_path)
        ]):
            with patch.object(
                CSVCleaner, 
                'clean', 
                side_effect=KeyboardInterrupt()
            ):
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
        input_file.write_text(content, encoding='utf-8')
        
        cleaner = CSVCleaner(
            input_path=str(input_file),
            output_path=str(output_path)
        )
        
        stats = cleaner.clean()
        
        assert stats.rows_written == 3
    
    def test_mixed_line_endings(self, temp_dir, output_path):
        """Verifica manejo de diferentes finales de línea."""
        input_file = temp_dir / "mixed_endings.csv"
        # Mezclar \n y \r\n
        content = "Name;Age\nJohn;30\r\nJane;25\n"
        input_file.write_bytes(content.encode('utf-8'))
        
        cleaner = CSVCleaner(
            input_path=str(input_file),
            output_path=str(output_path)
        )
        
        stats = cleaner.clean()
        
        assert stats.rows_written == 2
    
    def test_unicode_normalization(self, temp_dir, output_path):
        """Verifica manejo de caracteres Unicode."""
        input_file = temp_dir / "unicode.csv"
        content = """Name;City
José;São Paulo
François;Montréal
北京;中国"""
        input_file.write_text(content, encoding='utf-8')
        
        cleaner = CSVCleaner(
            input_path=str(input_file),
            output_path=str(output_path)
        )
        
        stats = cleaner.clean()
        
        assert stats.rows_written == 3
        
        # Verificar que los caracteres se mantuvieron
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'José' in content
        assert '北京' in content
    
    def test_very_long_lines(self, temp_dir, output_path):
        """Verifica manejo de líneas muy largas."""
        input_file = temp_dir / "long_lines.csv"
        long_value = "x" * 10000
        
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write("Col1;Col2;Col3\n")
            f.write(f"{long_value};data;data\n")
        
        cleaner = CSVCleaner(
            input_path=str(input_file),
            output_path=str(output_path)
        )
        
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
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write("Col1;Col2;Col3;Col4;Col5\n")
            for i in range(10000):
                f.write(f"Val{i};Data{i};Info{i};Test{i};More{i}\n")
        
        cleaner = CSVCleaner(
            input_path=str(input_file),
            output_path=str(output_path)
        )
        
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
    
    @pytest.mark.parametrize("delimiter", [';', ',', '\t', '|'])
    def test_different_delimiters(self, temp_dir, delimiter):
        """Prueba con diferentes delimitadores."""
        input_file = temp_dir / f"delim_{ord(delimiter)}.csv"
        output_file = temp_dir / f"out_{ord(delimiter)}.csv"
        
        # Crear CSV con el delimitador específico
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(f"Name{delimiter}Age{delimiter}City\n")
            f.write(f"John{delimiter}30{delimiter}NYC\n")
        
        cleaner = CSVCleaner(
            input_path=str(input_file),
            output_path=str(output_file),
            delimiter=delimiter
        )
        
        stats = cleaner.clean()
        
        assert stats.rows_written == 1
    
    @pytest.mark.parametrize("encoding", ['utf-8', 'latin-1', 'cp1252'])
    def test_different_encodings(self, temp_dir, encoding):
        """Prueba con diferentes encodings."""
        input_file = temp_dir / f"enc_{encoding}.csv"
        output_file = temp_dir / f"out_{encoding}.csv"
        
        try:
            input_file.write_text("Name;City\nTest;Test\n", encoding=encoding)
            
            cleaner = CSVCleaner(
                input_path=str(input_file),
                output_path=str(output_file),
                encoding=encoding
            )
            
            stats = cleaner.clean()
            
            assert stats.rows_written == 1
        except LookupError:
            pytest.skip(f"Encoding {encoding} no disponible")


# ============================================================================
# CONFIGURACIÓN DE PYTEST
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=scripts.clean_csv", "--cov-report=html"])