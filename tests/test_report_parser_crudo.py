"""
Suite de pruebas completa para ReportParserCrudo.
Cubre funcionalidad, casos edge, errores y rendimiento.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Importar las clases a probar
from app.report_parser_crudo import (
    APUContext,
    FileReadError,
    ParserConfig,
    ReportParserCrudo,
)

# =====================================================================
# FIXTURES Y DATOS DE PRUEBA
# =====================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Crea un directorio temporal para las pruebas."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_apu_content_lines() -> str:
    """Contenido de ejemplo con formato de líneas."""
    return """
REMATE CON PINTURA...;;;;; UNIDAD: ML
;;;;; ITEM: 1,1
DESCRIPCION;UND;CANT.;...
LAMINA DE 1.22...;UND;0,33;14,04;174.928,81;65.403,35
"""


@pytest.fixture
def sample_invalid_content() -> str:
    """Contenido inválido para probar manejo de errores."""
    return """
    Este es un archivo de texto plano sin formato APU.
    No contiene información estructurada.
    Solo texto normal sin códigos ni items.
    Lorem ipsum dolor sit amet.
    """


# =====================================================================
# PRUEBAS DE CONFIGURACIÓN
# =====================================================================


class TestParserConfig:
    """Pruebas para la clase ParserConfig simplificada."""

    def test_default_config_creation(self):
        """Verifica que la configuración por defecto se crea correctamente."""
        config = ParserConfig()
        assert config.encodings == ["utf-8", "latin1", "cp1252", "iso-8859-1"]
        assert config.default_unit == "UND"
        assert config.max_lines_to_process == 100000

    def test_custom_config_creation(self):
        """Verifica la creación con parámetros personalizados."""
        config = ParserConfig(
            encodings=["utf-8"], default_unit="UN", max_lines_to_process=5000
        )
        assert config.encodings == ["utf-8"]
        assert config.default_unit == "UN"
        assert config.max_lines_to_process == 5000


# =====================================================================
# PRUEBAS DE APUContext
# =====================================================================


class TestAPUContext:
    """Pruebas para la clase APUContext."""

    def test_valid_apu_context_creation(self):
        """Verifica creación correcta de contexto APU."""
        context = APUContext(
            apu_code="TEST-001",
            apu_desc="Descripción de prueba",
            apu_unit="M3",
            source_line=1,
        )
        assert context.apu_code == "TEST-001"
        assert context.apu_desc == "Descripción de prueba"
        assert context.apu_unit == "M3"
        assert context.source_line == 1
        assert context.is_valid

    def test_apu_context_normalization(self):
        """Verifica normalización de valores."""
        context = APUContext(
            apu_code="  TEST-002  ",
            apu_desc="  Descripción  ",
            apu_unit="  m3  ",
            source_line=10,
        )
        assert context.apu_code == "TEST-002"
        assert context.apu_desc == "Descripción"
        assert context.apu_unit == "M3"

    def test_empty_apu_code_raises_error(self):
        """Verifica que código vacío lance error."""
        with pytest.raises(ValueError, match="El código del APU no puede estar vacío."):
            APUContext(apu_code="", apu_desc="Test", apu_unit="UN", source_line=1)

    def test_is_valid_property(self):
        """Verifica la propiedad is_valid."""
        valid_context = APUContext(
            apu_code="AB", apu_desc="Test", apu_unit="UN", source_line=1
        )
        assert valid_context.is_valid

        context = APUContext(apu_code="A ", apu_desc="Test", apu_unit="UN", source_line=1)
        assert not context.is_valid


# =====================================================================
# PRUEBAS DE ReportParserCrudo - INICIALIZACIÓN
# =====================================================================


class TestReportParserInitialization:
    """Pruebas de inicialización del parser."""

    def test_valid_file_initialization(self, temp_dir):
        """Verifica inicialización con archivo válido."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("Test content")

        parser = ReportParserCrudo(file_path)
        assert parser.file_path == file_path
        assert parser.config is not None
        assert not parser._parsed
        assert len(parser.raw_records) == 0

    def test_file_not_found_raises_error(self):
        """Verifica que archivo inexistente lance error."""
        with pytest.raises(FileNotFoundError):
            ReportParserCrudo("/path/to/nonexistent/file.txt")

    def test_directory_path_raises_error(self, temp_dir):
        """Verifica que pasar directorio lance error."""
        with pytest.raises(ValueError):
            ReportParserCrudo(temp_dir)

    def test_empty_file_raises_error(self, temp_dir):
        """Verifica que archivo vacío lance error."""
        file_path = temp_dir / "empty.txt"
        file_path.touch()

        with pytest.raises(ValueError):
            ReportParserCrudo(file_path)


# =====================================================================
# PRUEBAS DE LECTURA DE ARCHIVOS
# =====================================================================


class TestFileReading:
    """Pruebas de lectura de archivos con diferentes encodings."""

    def test_read_utf8_file(self, temp_dir):
        """Verifica lectura de archivo UTF-8."""
        file_path = temp_dir / "utf8.txt"
        content = "ITEM: TEST-001\nDescripción: Ñoño está aquí"
        file_path.write_text(content, encoding="utf-8")

        parser = ReportParserCrudo(file_path)
        read_content = parser._read_file_safely()

        assert "TEST-001" in read_content
        assert "Ñoño" in read_content
        assert parser.stats["encoding_used"] == "utf-8"

    def test_read_latin1_file(self, temp_dir):
        """Verifica lectura de archivo Latin-1."""
        file_path = temp_dir / "latin1.txt"
        content = "ITEM: TEST-002\nDescripción: Café"
        file_path.write_bytes(content.encode("latin1"))

        config = ParserConfig(encodings=["latin1", "utf-8"])
        parser = ReportParserCrudo(file_path, config=config)
        read_content = parser._read_file_safely()

        assert "TEST-002" in read_content
        assert parser.stats["encoding_used"] == "latin1"

    def test_all_encodings_fail(self, temp_dir):
        """Verifica error cuando todos los encodings fallan."""
        file_path = temp_dir / "bad.txt"
        file_path.write_text("test")

        config = ParserConfig(encodings=["invalid_encoding"])
        parser = ReportParserCrudo(file_path, config=config)

        with pytest.raises(FileReadError):
            parser._read_file_safely()


# =====================================================================
# PRUEBAS DE PARSING COMPLETO
# =====================================================================


class TestCompleteParsing:
    """Pruebas de flujo completo de parsing."""

    def test_parse_lines_format(self, temp_dir, sample_apu_content_lines):
        """Verifica parsing completo con formato de líneas."""
        file_path = temp_dir / "lines.txt"
        file_path.write_text(sample_apu_content_lines)

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        assert len(results) > 0
        assert results[0]["apu_code"] == "1.1"
        assert "LAMINA DE 1.22" in results[0]["insumo_line"]

    def test_parse_invalid_content(self, temp_dir, sample_invalid_content):
        """Verifica manejo de contenido inválido."""
        file_path = temp_dir / "invalid.txt"
        file_path.write_text(sample_invalid_content)

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        assert len(results) == 0
        assert parser.stats["insumos_extracted"] == 0

    def test_parse_already_parsed(self, temp_dir):
        """Verifica que no se re-procese archivo ya parseado."""
        file_path = temp_dir / "test.txt"
        file_path.write_text(
            "REMATE CON PINTURA...;;;;; UNIDAD: ML\n"
            ";;;;; ITEM: 1,1\n"
            "DESCRIPCION;UND;CANT.;..."
        )

        parser = ReportParserCrudo(file_path)

        results1 = parser.parse_to_raw()
        initial_count = len(results1)
        results2 = parser.parse_to_raw()

        assert len(results2) == initial_count
        assert parser._parsed is True


# =====================================================================
# PRUEBAS DE CASOS EDGE
# =====================================================================


class TestEdgeCases:
    """Pruebas de casos límite y especiales."""

    def test_only_whitespace_file(self, temp_dir):
        """Verifica manejo de archivo solo con espacios."""
        file_path = temp_dir / "whitespace.txt"
        file_path.write_text("   \n\n   \t\t   \n   ")

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()
        assert len(results) == 0
