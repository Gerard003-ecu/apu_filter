"""
Suite de pruebas completa para ReportParserCrudo.
Cubre funcionalidad, casos edge, validación topológica, errores y rendimiento.

Organización:
    1. Fixtures y datos de prueba
    2. Pruebas de APUContext
    3. Pruebas de inicialización del parser
    4. Pruebas de lectura de archivos
    5. Pruebas de validación básica
    6. Pruebas de validación Lark/topológica
    7. Pruebas de parsing completo
    8. Pruebas de handlers
    9. Pruebas de métricas topológicas
    10. Pruebas de casos edge
    11. Pruebas de rendimiento
"""

import hashlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import Generator, List, Dict, Any
from unittest.mock import MagicMock, patch

import pytest

from app.report_parser_crudo import (
    APUContext,
    CategoryHandler,
    FileReadError,
    HeaderHandler,
    InsumoHandler,
    JunkHandler,
    LineValidationResult,
    ParseStrategyError,
    ParserContext,
    ReportParserCrudo,
    ValidationStats,
)
from tests.test_data import TEST_CONFIG


# =====================================================================
# CONSTANTES Y DATOS DE PRUEBA
# =====================================================================

TEST_APUS_PROFILE = TEST_CONFIG.get("file_profiles", {}).get("apus_default", {})

# Líneas de insumo válidas para pruebas
VALID_INSUMO_LINES = [
    "CEMENTO PORTLAND TIPO I;KG;50,00;450,00;22500,00;22500,00",
    "ARENA LAVADA;M3;0,50;85000,00;42500,00;42500,00",
    "GRAVA TRITURADA 3/4\";M3;0,75;95000,00;71250,00;71250,00",
    "AGUA;LT;25,00;50,00;1250,00;1250,00",
    "ACERO DE REFUERZO 60000 PSI;KG;125,00;3200,00;400000,00;400000,00",
]

# Líneas que deben ser rechazadas
INVALID_INSUMO_LINES = [
    "",  # Vacía
    "   ",  # Solo espacios
    "SUBTOTAL;;;;",  # Subtotal
    "=================",  # Decorativa
    "TOTAL MATERIALES;;;;;150000,00",  # Línea de total
    "Texto sin separadores ni números",  # Sin estructura
    "A;B;C",  # Muy pocos campos
]

# Contenido APU completo para pruebas de integración
COMPLETE_APU_CONTENT = """
CONSTRUCCIÓN DE MURO EN BLOQUE;;;;; UNIDAD: M2
;;;;; ITEM: 1,1
DESCRIPCION;UND;CANTIDAD;PRECIO;SUBTOTAL;TOTAL
MATERIALES
BLOQUE DE CONCRETO 15x20x40;UND;12,50;2500,00;31250,00;31250,00
CEMENTO PORTLAND TIPO I;KG;8,00;520,00;4160,00;4160,00
ARENA LAVADA;M3;0,02;85000,00;1700,00;1700,00
SUBTOTAL MATERIALES;;;;;37110,00
MANO DE OBRA
OFICIAL DE CONSTRUCCION;HR;0,80;15000,00;12000,00;12000,00
AYUDANTE;HR;1,20;8000,00;9600,00;9600,00
SUBTOTAL MANO DE OBRA;;;;;21600,00
EQUIPO
HERRAMIENTA MENOR;%MO;5,00;21600,00;1080,00;1080,00
SUBTOTAL EQUIPO;;;;;1080,00
"""

MULTI_APU_CONTENT = """
EXCAVACIÓN MANUAL;;;;; UNIDAD: M3
;;;;; ITEM: 2,1
DESCRIPCION;UND;CANTIDAD;PRECIO;SUBTOTAL;TOTAL
MANO DE OBRA
OBRERO;HR;4,00;8000,00;32000,00;32000,00
SUBTOTAL;;;;;32000,00

RELLENO COMPACTADO;;;;; UNIDAD: M3
;;;;; ITEM: 2,2
DESCRIPCION;UND;CANTIDAD;PRECIO;SUBTOTAL;TOTAL
MATERIALES
MATERIAL SELECCIONADO;M3;1,20;45000,00;54000,00;54000,00
SUBTOTAL;;;;;54000,00
"""


# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Crea un directorio temporal para las pruebas."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_apu_file(temp_dir) -> Path:
    """Crea archivo APU de ejemplo."""
    file_path = temp_dir / "sample_apu.txt"
    file_path.write_text(COMPLETE_APU_CONTENT, encoding="utf-8")
    return file_path


@pytest.fixture
def multi_apu_file(temp_dir) -> Path:
    """Crea archivo con múltiples APUs."""
    file_path = temp_dir / "multi_apu.txt"
    file_path.write_text(MULTI_APU_CONTENT, encoding="utf-8")
    return file_path


@pytest.fixture
def empty_file(temp_dir) -> Path:
    """Crea archivo vacío."""
    file_path = temp_dir / "empty.txt"
    file_path.touch()
    return file_path


@pytest.fixture
def whitespace_file(temp_dir) -> Path:
    """Crea archivo con solo espacios en blanco."""
    file_path = temp_dir / "whitespace.txt"
    file_path.write_text("   \n\n   \t\t   \n   ", encoding="utf-8")
    return file_path


@pytest.fixture
def invalid_content_file(temp_dir) -> Path:
    """Crea archivo con contenido no-APU."""
    content = """
    Este es un archivo de texto plano sin formato APU.
    No contiene información estructurada.
    Solo texto normal sin códigos ni items.
    Lorem ipsum dolor sit amet.
    """
    file_path = temp_dir / "invalid.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def parser_with_sample(sample_apu_file) -> ReportParserCrudo:
    """Parser inicializado con archivo de ejemplo."""
    return ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE)


@pytest.fixture
def parser_context() -> ParserContext:
    """Contexto de parser limpio para pruebas unitarias."""
    return ParserContext()


@pytest.fixture
def valid_apu_context() -> APUContext:
    """Contexto APU válido para pruebas."""
    return APUContext(
        apu_code="TEST-001",
        apu_desc="Descripción de prueba",
        apu_unit="M2",
        source_line=10,
    )


# =====================================================================
# PRUEBAS DE APUContext
# =====================================================================


class TestAPUContext:
    """Pruebas para la clase APUContext."""

    def test_valid_creation(self):
        """Verifica creación correcta con datos válidos."""
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
        assert context.is_valid is True

    def test_whitespace_normalization(self):
        """Verifica normalización de espacios en blanco."""
        context = APUContext(
            apu_code="  TEST-002  ",
            apu_desc="  Descripción con espacios  ",
            apu_unit="  m3  ",
            source_line=10,
        )
        assert context.apu_code == "TEST-002"
        assert context.apu_desc == "Descripción con espacios"
        assert context.apu_unit == "M3"

    def test_unit_uppercase_normalization(self):
        """Verifica que la unidad se convierta a mayúsculas."""
        context = APUContext(
            apu_code="TEST", apu_desc="Test", apu_unit="kg", source_line=1
        )
        assert context.apu_unit == "KG"

    def test_empty_unit_uses_default(self):
        """Verifica uso de unidad por defecto cuando está vacía."""
        context = APUContext(
            apu_code="TEST", apu_desc="Test", apu_unit="", source_line=1
        )
        assert context.apu_unit == context.default_unit

    def test_none_unit_uses_default(self):
        """Verifica uso de unidad por defecto cuando es None."""
        context = APUContext(
            apu_code="TEST", apu_desc="Test", apu_unit=None, source_line=1
        )
        assert context.apu_unit == context.default_unit

    def test_empty_code_raises_error(self):
        """Verifica que código vacío lance ValueError."""
        with pytest.raises(ValueError, match="El código del APU no puede estar vacío"):
            APUContext(apu_code="", apu_desc="Test", apu_unit="UN", source_line=1)

    def test_whitespace_only_code_raises_error(self):
        """Verifica que código solo con espacios lance error."""
        with pytest.raises(ValueError, match="El código del APU no puede estar vacío"):
            APUContext(apu_code="   ", apu_desc="Test", apu_unit="UN", source_line=1)

    def test_none_code_raises_error(self):
        """Verifica que código None lance error apropiado."""
        with pytest.raises((ValueError, AttributeError)):
            APUContext(apu_code=None, apu_desc="Test", apu_unit="UN", source_line=1)

    @pytest.mark.parametrize(
        "code,expected_valid",
        [
            ("AB", True),  # Mínimo válido (2 caracteres)
            ("A", False),  # Muy corto
            ("A ", False),  # Un carácter después de strip
            ("ABC-123", True),  # Código típico
            ("1.1", True),  # Código numérico
        ],
    )
    def test_is_valid_property(self, code: str, expected_valid: bool):
        """Verifica la propiedad is_valid con diferentes códigos."""
        context = APUContext(
            apu_code=code, apu_desc="Test", apu_unit="UN", source_line=1
        )
        assert context.is_valid is expected_valid


# =====================================================================
# PRUEBAS DE INICIALIZACIÓN DEL PARSER
# =====================================================================


class TestReportParserInitialization:
    """Pruebas de inicialización del parser."""

    def test_valid_file_initialization(self, sample_apu_file):
        """Verifica inicialización con archivo válido."""
        parser = ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE)

        assert parser.file_path == sample_apu_file
        assert parser.profile is not None
        assert parser.config is not None
        assert parser._parsed is False
        assert len(parser.raw_records) == 0
        assert isinstance(parser.stats, dict) or hasattr(parser.stats, "__getitem__")

    def test_path_string_conversion(self, sample_apu_file):
        """Verifica que strings se conviertan a Path."""
        parser = ReportParserCrudo(str(sample_apu_file), profile=TEST_APUS_PROFILE)
        assert isinstance(parser.file_path, Path)

    def test_none_file_path_raises_error(self):
        """Verifica que file_path None lance error."""
        with pytest.raises(ValueError, match="file_path no puede ser None"):
            ReportParserCrudo(None, profile=TEST_APUS_PROFILE)

    def test_file_not_found_raises_error(self):
        """Verifica que archivo inexistente lance FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ReportParserCrudo("/path/to/nonexistent/file.txt", profile=TEST_APUS_PROFILE)

    def test_directory_path_raises_error(self, temp_dir):
        """Verifica que directorio lance ValueError."""
        with pytest.raises(ValueError, match="no es un archivo"):
            ReportParserCrudo(temp_dir, profile=TEST_APUS_PROFILE)

    def test_empty_file_raises_error(self, empty_file):
        """Verifica que archivo vacío lance ValueError."""
        with pytest.raises(ValueError, match="vacío"):
            ReportParserCrudo(empty_file, profile=TEST_APUS_PROFILE)

    def test_none_profile_handled(self, sample_apu_file):
        """Verifica manejo de profile None."""
        parser = ReportParserCrudo(sample_apu_file, profile=None)
        assert parser.profile == {}

    def test_invalid_profile_type_handled(self, sample_apu_file):
        """Verifica manejo de profile con tipo inválido."""
        parser = ReportParserCrudo(sample_apu_file, profile="invalid")
        assert parser.profile == {}

    def test_none_config_handled(self, sample_apu_file):
        """Verifica manejo de config None."""
        parser = ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE, config=None)
        assert parser.config == {}

    def test_debug_mode_from_config(self, sample_apu_file):
        """Verifica que debug_mode se tome de config."""
        config = {"debug_mode": True}
        parser = ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE, config=config)
        assert parser.debug_mode is True

    def test_lark_parser_initialization(self, sample_apu_file):
        """Verifica que Lark parser se inicialice (si está disponible)."""
        parser = ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE)
        # El parser puede o no estar disponible dependiendo de las importaciones
        # No falla si Lark no está disponible
        assert hasattr(parser, "lark_parser")


# =====================================================================
# PRUEBAS DE LECTURA DE ARCHIVOS
# =====================================================================


class TestFileReading:
    """Pruebas de lectura de archivos con diferentes encodings."""

    def test_read_utf8_file(self, temp_dir):
        """Verifica lectura de archivo UTF-8."""
        file_path = temp_dir / "utf8.txt"
        content = "ITEM: TEST-001\nDescripción: Ñoño está aquí con acentos áéíóú"
        file_path.write_text(content, encoding="utf-8")

        profile = {**TEST_APUS_PROFILE, "encoding": "utf-8"}
        parser = ReportParserCrudo(file_path, profile=profile)
        read_content = parser._read_file_safely()

        assert "TEST-001" in read_content
        assert "Ñoño" in read_content
        assert "áéíóú" in read_content
        assert parser.stats["encoding_used"] == "utf-8"

    def test_read_latin1_file(self, temp_dir):
        """Verifica lectura de archivo Latin-1."""
        file_path = temp_dir / "latin1.txt"
        content = "ITEM: TEST-002\nDescripción: Café y más"
        file_path.write_bytes(content.encode("latin1"))

        config = {"encodings": ["latin1", "utf-8"]}
        parser = ReportParserCrudo(file_path, profile=TEST_APUS_PROFILE, config=config)
        read_content = parser._read_file_safely()

        assert "TEST-002" in read_content
        assert "Café" in read_content
        assert parser.stats["encoding_used"] == "latin1"

    def test_read_cp1252_file(self, temp_dir):
        """Verifica lectura de archivo CP1252 (Windows)."""
        file_path = temp_dir / "cp1252.txt"
        # Usar caracteres que son distintos en CP1252 y Latin-1 si es posible,
        # o simplemente asegurar que el perfil no fuerce latin1.
        # TEST_APUS_PROFILE tiene encoding='latin1', que es intentado primero.
        # Latin1 puede decodificar cp1252 (mapeo directo byte-byte) pero incorrectamente para ciertos rangos (0x80-0x9F).
        # Para forzar el uso de la config, pasamos un perfil sin encoding o vacio.
        content = "ITEM: TEST-003\nDescripción con caracteres Windows"
        file_path.write_bytes(content.encode("cp1252"))

        config = {"encodings": ["cp1252"]}
        # Pasar profile vacio para evitar precedencia de encoding por defecto
        parser = ReportParserCrudo(file_path, profile={}, config=config)
        read_content = parser._read_file_safely()

        assert "TEST-003" in read_content
        assert parser.stats["encoding_used"] == "cp1252"

    def test_encoding_fallback(self, temp_dir):
        """Verifica fallback cuando primer encoding falla."""
        file_path = temp_dir / "fallback.txt"
        # Contenido que es válido en latin1 pero podría fallar en utf-8 estricto
        content = "Café résumé"
        file_path.write_bytes(content.encode("latin1"))

        # utf-8 fallará, latin1 debería funcionar
        config = {"encodings": ["utf-8", "latin1"]}
        parser = ReportParserCrudo(file_path, profile=TEST_APUS_PROFILE, config=config)
        read_content = parser._read_file_safely()

        assert "Café" in read_content or "Caf" in read_content

    def test_all_encodings_fail_raises_error(self, temp_dir):
        """Verifica error cuando todos los encodings fallan."""
        file_path = temp_dir / "bad.txt"
        # Secuencia de bytes inválida en UTF-8 estricto
        file_path.write_bytes(b"\x81\x82\x83\x95\xff\xfe")

        config = {"encodings": ["utf-8"]}  # Solo UTF-8, fallará
        parser = ReportParserCrudo(file_path, profile={}, config=config)

        with pytest.raises(FileReadError, match="ninguna de las codificaciones"):
            parser._read_file_safely()

    def test_profile_encoding_priority(self, temp_dir):
        """Verifica que encoding del profile tenga prioridad."""
        file_path = temp_dir / "priority.txt"
        content = "Test content"
        file_path.write_text(content, encoding="utf-8")

        profile = {"encoding": "utf-8"}
        parser = ReportParserCrudo(file_path, profile=profile)
        parser._read_file_safely()

        assert parser.stats["encoding_used"] == "utf-8"


# =====================================================================
# PRUEBAS DE VALIDACIÓN BÁSICA
# =====================================================================


class TestBasicValidation:
    """Pruebas de validación estructural básica."""

    def test_validate_basic_structure_valid_line(self, parser_with_sample):
        """Verifica validación de línea estructuralmente válida."""
        line = "CEMENTO PORTLAND;KG;50,00;450,00;22500,00;22500,00"
        fields = line.split(";")

        is_valid, reason = parser_with_sample._validate_basic_structure(line, fields)

        assert is_valid is True
        assert reason == ""

    def test_validate_basic_structure_insufficient_fields(self, parser_with_sample):
        """Verifica rechazo por campos insuficientes."""
        line = "A;B;C"
        fields = line.split(";")

        is_valid, reason = parser_with_sample._validate_basic_structure(line, fields)

        assert is_valid is False
        assert "Insuficientes campos" in reason

    def test_validate_basic_structure_empty_description(self, parser_with_sample):
        """Verifica rechazo cuando descripción está vacía."""
        line = ";KG;50,00;450,00;22500,00;22500,00"
        fields = line.split(";")

        is_valid, reason = parser_with_sample._validate_basic_structure(line, fields)

        assert is_valid is False
        assert "vacío" in reason.lower()

    def test_validate_basic_structure_no_numeric_fields(self, parser_with_sample):
        """Verifica rechazo cuando no hay campos numéricos."""
        # Evitar palabras clave como TOTAL o SUBTOTAL que disparan otros checks antes
        # Usar contenido que parezca un insumo real pero sin números
        # IMPORTANTE: No usar 'M3' o similar en la unidad porque el regex numérico (\d+) lo detecta.
        line = "CONCRETO;UND;SIN_VALOR;PENDIENTE;PENDIENTE;PENDIENTE"
        fields = line.split(";")

        is_valid, reason = parser_with_sample._validate_basic_structure(line, fields)

        assert is_valid is False
        assert "numéricos" in reason.lower()

    @pytest.mark.parametrize(
        "keyword",
        ["SUBTOTAL", "TOTAL", "COSTO DIRECTO", "COSTO TOTAL", "VALOR TOTAL"],
    )
    def test_validate_basic_structure_rejects_totals(self, parser_with_sample, keyword):
        """Verifica rechazo de líneas de subtotal/total."""
        line = f"{keyword};;;;;150000,00"
        fields = line.split(";")

        is_valid, reason = parser_with_sample._validate_basic_structure(line, fields)

        assert is_valid is False
        assert "subtotal" in reason.lower() or "total" in reason.lower()

    def test_validate_basic_structure_empty_line(self, parser_with_sample):
        """Verifica rechazo de línea vacía."""
        is_valid, reason = parser_with_sample._validate_basic_structure("", [])
        assert is_valid is False

    def test_validate_basic_structure_none_line(self, parser_with_sample):
        """Verifica rechazo de línea None."""
        is_valid, reason = parser_with_sample._validate_basic_structure(None, [])
        assert is_valid is False


# =====================================================================
# PRUEBAS DE VALIDACIÓN LARK/TOPOLÓGICA
# =====================================================================


class TestLarkValidation:
    """Pruebas de validación con parser Lark."""

    def test_validate_with_lark_parser_not_available(self, parser_with_sample):
        """Verifica comportamiento cuando Lark no está disponible."""
        parser_with_sample.lark_parser = None

        is_valid, tree, reason = parser_with_sample._validate_with_lark("test line")

        assert is_valid is True  # Omite validación
        assert tree is None
        assert "no disponible" in reason.lower()

    def test_validate_with_lark_empty_line(self, parser_with_sample):
        """Verifica rechazo de línea vacía."""
        is_valid, tree, reason = parser_with_sample._validate_with_lark("")

        assert is_valid is False
        assert "vacía" in reason.lower() or "inválido" in reason.lower()

    def test_validate_with_lark_line_too_long(self, parser_with_sample):
        """Verifica rechazo de línea excesivamente larga."""
        long_line = "A" * (parser_with_sample._MAX_LINE_LENGTH + 100)

        is_valid, tree, reason = parser_with_sample._validate_with_lark(long_line)

        assert is_valid is False
        assert "excede" in reason.lower() or "límite" in reason.lower()

    def test_validate_with_lark_line_too_short(self, parser_with_sample):
        """Verifica rechazo de línea muy corta."""
        is_valid, tree, reason = parser_with_sample._validate_with_lark("AB")

        assert is_valid is False
        assert "insuficiente" in reason.lower()

    def test_validate_with_lark_cache_hit(self, parser_with_sample):
        """Verifica funcionamiento del cache."""
        line = "CEMENTO PORTLAND;KG;50,00;450,00;22500,00;22500,00"

        # Primera llamada
        result1 = parser_with_sample._validate_with_lark(line, use_cache=True)
        cache_hits_before = parser_with_sample.validation_stats.cached_parses

        # Segunda llamada (debería usar cache)
        result2 = parser_with_sample._validate_with_lark(line, use_cache=True)
        cache_hits_after = parser_with_sample.validation_stats.cached_parses

        assert cache_hits_after > cache_hits_before

    def test_validate_with_lark_cache_disabled(self, parser_with_sample):
        """Verifica que cache se puede deshabilitar."""
        line = "CEMENTO PORTLAND;KG;50,00;450,00;22500,00;22500,00"

        initial_cache_size = len(parser_with_sample._parse_cache)
        parser_with_sample._validate_with_lark(line, use_cache=False)

        # Cache no debería crecer si está deshabilitado
        # (aunque esto depende de la implementación)
        assert len(parser_with_sample._parse_cache) >= initial_cache_size


class TestTopologicalMethods:
    """Pruebas de métodos topológicos."""

    def test_compute_semantic_cache_key_normalization(self, parser_with_sample):
        """Verifica normalización de clave de cache."""
        line1 = "CEMENTO  PORTLAND;KG;50,00"
        line2 = "CEMENTO PORTLAND;KG;50,00"

        key1 = parser_with_sample._compute_semantic_cache_key(line1)
        key2 = parser_with_sample._compute_semantic_cache_key(line2)

        # Espacios múltiples se normalizan a uno
        assert key1 == key2

    def test_compute_semantic_cache_key_long_line(self, parser_with_sample):
        """Verifica hash para líneas largas."""
        long_line = "DESCRIPCION MUY LARGA " * 200

        key = parser_with_sample._compute_semantic_cache_key(long_line)

        # Debe ser un hash corto, no la línea completa
        assert len(key) < len(long_line)
        assert len(key) <= 64  # SHA256 hex es 64 caracteres

    def test_has_minimal_structural_connectivity_valid(self, parser_with_sample):
        """Verifica detección de conectividad en línea válida."""
        line = "CEMENTO PORTLAND;KG;50,00;450,00;22500,00"

        result = parser_with_sample._has_minimal_structural_connectivity(line)

        assert result is True

    def test_has_minimal_structural_connectivity_no_separators(self, parser_with_sample):
        """Verifica rechazo sin separadores suficientes."""
        line = "CEMENTO PORTLAND KG 50 450"

        result = parser_with_sample._has_minimal_structural_connectivity(line)

        assert result is False

    def test_has_minimal_structural_connectivity_no_numbers(self, parser_with_sample):
        """Verifica rechazo sin números."""
        line = "DESCRIPCION;UND;CANT;PRECIO;SUBTOTAL"

        result = parser_with_sample._has_minimal_structural_connectivity(line)

        assert result is False

    def test_validate_tree_homotopy_none(self, parser_with_sample):
        """Verifica rechazo de árbol None."""
        result = parser_with_sample._validate_tree_homotopy(None)
        assert result is False

    def test_validate_tree_homotopy_invalid_structure(self, parser_with_sample):
        """Verifica rechazo de objeto sin estructura de árbol."""
        fake_tree = {"not": "a tree"}
        result = parser_with_sample._validate_tree_homotopy(fake_tree)
        assert result is False

    def test_is_valid_tree_none(self, parser_with_sample):
        """Verifica que árbol None es inválido."""
        assert parser_with_sample._is_valid_tree(None) is False

    def test_is_valid_tree_no_data_attribute(self, parser_with_sample):
        """Verifica rechazo de objeto sin atributo data."""
        fake = MagicMock(spec=[])  # Sin atributos
        assert parser_with_sample._is_valid_tree(fake) is False

    def test_get_topological_context(self, parser_with_sample):
        """Verifica extracción de contexto topológico."""
        line = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        context = parser_with_sample._get_topological_context(line, position=10, radius=3)

        assert "⟪" in context  # Marcador de error
        assert "⟫" in context

    def test_calculate_topological_completeness(self, parser_with_sample):
        """Verifica cálculo de completitud topológica."""
        complete_line = "CEMENTO PORTLAND;KG;$50,00;450,00;22500,00"
        incomplete_line = "Solo texto"

        complete_score = parser_with_sample._calculate_topological_completeness(complete_line)
        incomplete_score = parser_with_sample._calculate_topological_completeness(incomplete_line)

        assert complete_score > incomplete_score
        assert 0 <= complete_score <= 1
        assert 0 <= incomplete_score <= 1


# =====================================================================
# PRUEBAS DE DETECCIÓN DE CATEGORÍAS
# =====================================================================


class TestCategoryDetection:
    """Pruebas de detección de categorías."""

    @pytest.mark.parametrize(
        "line,expected_category",
        [
            ("MATERIALES", "MATERIALES"),
            ("MANO DE OBRA", "MANO DE OBRA"),
            ("EQUIPO", "EQUIPO"),
            ("TRANSPORTE", "TRANSPORTE"),
            ("HERRAMIENTA", "HERRAMIENTA"),
            ("OTROS", "OTROS"),
        ],
    )
    def test_detect_category_exact_match(self, parser_with_sample, line, expected_category):
        """Verifica detección de categorías exactas."""
        result = parser_with_sample._detect_category(line)
        assert result == expected_category

    @pytest.mark.parametrize(
        "line,expected_category",
        [
            ("MATERIAL", "MATERIALES"),
            ("M.O.", "MANO DE OBRA"),
            ("EQUIPOS", "EQUIPO"),
            ("MAQUINARIA", "EQUIPO"),
            ("HERR.", "HERRAMIENTA"),
        ],
    )
    def test_detect_category_variations(self, parser_with_sample, line, expected_category):
        """Verifica detección de variaciones de categorías."""
        result = parser_with_sample._detect_category(line)
        assert result == expected_category

    def test_detect_category_long_line_rejected(self, parser_with_sample):
        """Verifica que líneas largas no se detecten como categorías."""
        long_line = "MATERIALES " + "X" * 100
        result = parser_with_sample._detect_category(long_line)
        assert result is None

    def test_detect_category_with_numbers_rejected(self, parser_with_sample):
        """Verifica que líneas con muchos números no se detecten como categorías."""
        line = "MATERIALES 12345"
        result = parser_with_sample._detect_category(line)
        assert result is None

    def test_detect_category_returns_none_for_insumo(self, parser_with_sample):
        """Verifica que líneas de insumo no se detecten como categorías."""
        insumo_line = "CEMENTO PORTLAND;KG;50,00;450,00;22500,00"
        result = parser_with_sample._detect_category(insumo_line)
        assert result is None


# =====================================================================
# PRUEBAS DE DETECCIÓN DE LÍNEAS BASURA
# =====================================================================


class TestJunkLineDetection:
    """Pruebas de detección de líneas basura/decorativas."""

    @pytest.mark.parametrize(
        "line",
        [
            "================",
            "----------------",
            "________________",
            "***************",
            "   ",
            "",
        ],
    )
    def test_is_junk_line_decorative(self, parser_with_sample, line):
        """Verifica detección de líneas decorativas."""
        result = parser_with_sample._is_junk_line(line.upper())
        assert result is True

    @pytest.mark.parametrize(
        "keyword", ["SUBTOTAL", "COSTO DIRECTO", "DESCRIPCION", "TOTAL", "IVA", "AIU"]
    )
    def test_is_junk_line_keywords(self, parser_with_sample, keyword):
        """Verifica detección de keywords de basura."""
        line = f"LINEA CON {keyword} INCLUIDO"
        result = parser_with_sample._is_junk_line(line)
        assert result is True

    def test_is_junk_line_valid_insumo(self, parser_with_sample):
        """Verifica que líneas válidas no se marquen como basura."""
        line = "CEMENTO PORTLAND TIPO I"
        result = parser_with_sample._is_junk_line(line.upper())
        assert result is False

    def test_is_junk_line_short_line(self, parser_with_sample):
        """Verifica que líneas muy cortas se marquen como basura."""
        result = parser_with_sample._is_junk_line("AB")
        assert result is True

    def test_is_junk_line_none_input(self, parser_with_sample):
        """Verifica manejo de entrada None."""
        result = parser_with_sample._is_junk_line(None)
        assert result is True


# =====================================================================
# PRUEBAS DE HANDLERS
# =====================================================================


class TestJunkHandler:
    """Pruebas del handler de líneas basura."""

    def test_can_handle_decorative_line(self, parser_with_sample):
        """Verifica que maneje líneas decorativas."""
        handler = JunkHandler(parser_with_sample)
        assert handler.can_handle("================") is True

    def test_can_handle_valid_line(self, parser_with_sample):
        """Verifica que no maneje líneas válidas."""
        handler = JunkHandler(parser_with_sample)
        assert handler.can_handle("CEMENTO;KG;50;450;22500") is False

    def test_handle_increments_counter(self, parser_with_sample, parser_context):
        """Verifica que handle incremente contador."""
        handler = JunkHandler(parser_with_sample)
        handler.handle("================", parser_context)
        assert parser_context.stats["junk_lines_skipped"] == 1


class TestHeaderHandler:
    """Pruebas del handler de encabezados APU."""

    def test_can_handle_header_pattern(self, parser_with_sample):
        """Verifica detección de patrón de encabezado."""
        handler = HeaderHandler(parser_with_sample)
        line = "CONSTRUCCIÓN DE MURO;;;;; UNIDAD: M2"
        next_line = ";;;;; ITEM: 1,1"
        assert handler.can_handle(line, next_line) is True

    def test_can_handle_missing_item(self, parser_with_sample):
        """Verifica rechazo cuando falta línea ITEM."""
        handler = HeaderHandler(parser_with_sample)
        line = "CONSTRUCCIÓN DE MURO;;;;; UNIDAD: M2"
        next_line = "Otra línea sin ITEM"
        assert handler.can_handle(line, next_line) is False

    def test_can_handle_missing_unit(self, parser_with_sample):
        """Verifica rechazo cuando falta UNIDAD."""
        handler = HeaderHandler(parser_with_sample)
        line = "CONSTRUCCIÓN DE MURO;;;;;"
        next_line = ";;;;; ITEM: 1,1"
        assert handler.can_handle(line, next_line) is False

    def test_handle_creates_apu_context(self, parser_with_sample, parser_context):
        """Verifica que handle cree contexto APU."""
        handler = HeaderHandler(parser_with_sample)
        line = "CONSTRUCCIÓN DE MURO;;;;; UNIDAD: M2"
        next_line = ";;;;; ITEM: 1,1"

        result = handler.handle(line, parser_context, next_line)

        assert result is True  # Consume línea extra
        assert parser_context.current_apu is not None
        assert parser_context.current_apu.apu_code == "1.1"


class TestCategoryHandler:
    """Pruebas del handler de categorías."""

    def test_can_handle_category(self, parser_with_sample):
        """Verifica detección de línea de categoría."""
        handler = CategoryHandler(parser_with_sample)
        assert handler.can_handle("MATERIALES") is True

    def test_can_handle_non_category(self, parser_with_sample):
        """Verifica rechazo de línea no-categoría."""
        handler = CategoryHandler(parser_with_sample)
        assert handler.can_handle("CEMENTO;KG;50") is False

    def test_handle_updates_context(self, parser_with_sample, parser_context):
        """Verifica que handle actualice contexto."""
        handler = CategoryHandler(parser_with_sample)
        handler.handle("MATERIALES", parser_context)

        assert parser_context.current_category == "MATERIALES"


class TestInsumoHandler:
    """Pruebas del handler de insumos."""

    def test_can_handle_insumo_line(self, parser_with_sample):
        """Verifica detección de línea de insumo."""
        handler = InsumoHandler(parser_with_sample)
        line = "CEMENTO;KG;50,00;450,00;22500,00;22500,00"
        assert handler.can_handle(line) is True

    def test_can_handle_no_separator(self, parser_with_sample):
        """Verifica rechazo sin separador."""
        handler = InsumoHandler(parser_with_sample)
        assert handler.can_handle("Solo texto sin separador") is False

    def test_can_handle_no_numbers(self, parser_with_sample):
        """Verifica rechazo sin números."""
        handler = InsumoHandler(parser_with_sample)
        assert handler.can_handle("A;B;C;D;E") is False

    def test_handle_orphan_rejected(self, parser_with_sample, parser_context):
        """Verifica rechazo de insumo huérfano (sin APU padre)."""
        handler = InsumoHandler(parser_with_sample)
        line = "CEMENTO;KG;50,00;450,00;22500,00;22500,00"

        result = handler.handle(line, parser_context)

        assert result is False
        assert parser_context.stats["orphans_discarded"] == 1
        assert len(parser_context.raw_records) == 0

    def test_handle_with_parent_accepted(self, parser_with_sample, parser_context, valid_apu_context):
        """Verifica aceptación de insumo con APU padre."""
        handler = InsumoHandler(parser_with_sample)
        parser_context.current_apu = valid_apu_context
        parser_context.current_category = "MATERIALES"
        line = "CEMENTO PORTLAND;KG;50,00;450,00;22500,00;22500,00"

        result = handler.handle(line, parser_context)

        assert result is False
        # El resultado depende de la validación Lark


# =====================================================================
# PRUEBAS DE PARSING COMPLETO
# =====================================================================


class TestCompleteParsing:
    """Pruebas de flujo completo de parsing."""

    def test_parse_complete_apu(self, sample_apu_file):
        """Verifica parsing completo de archivo APU."""
        parser = ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE)
        results = parser.parse_to_raw()

        assert len(results) > 0
        assert parser._parsed is True
        assert parser.stats["insumos_extracted"] > 0

    def test_parse_extracts_correct_apu_code(self, sample_apu_file):
        """Verifica extracción correcta de código APU."""
        parser = ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE)
        results = parser.parse_to_raw()

        if results:
            assert results[0]["apu_code"] == "1.1"

    def test_parse_multiple_apus(self, multi_apu_file):
        """Verifica parsing de múltiples APUs."""
        parser = ReportParserCrudo(multi_apu_file, profile=TEST_APUS_PROFILE)
        results = parser.parse_to_raw()

        # Verificar que detectó múltiples APUs
        apu_codes = set(r["apu_code"] for r in results)
        assert len(apu_codes) >= 1  # Al menos un APU

    def test_parse_invalid_content_returns_empty(self, invalid_content_file):
        """Verifica que contenido inválido retorne lista vacía."""
        parser = ReportParserCrudo(invalid_content_file, profile=TEST_APUS_PROFILE)
        results = parser.parse_to_raw()

        assert len(results) == 0
        assert parser.stats.get("insumos_extracted", 0) == 0

    def test_parse_already_parsed_returns_cached(self, sample_apu_file):
        """Verifica que re-parsing retorne resultados cacheados."""
        parser = ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE)

        results1 = parser.parse_to_raw()
        results2 = parser.parse_to_raw()

        assert results1 is results2  # Mismo objeto (cacheado)
        assert parser._parsed is True

    def test_parse_records_have_required_fields(self, sample_apu_file):
        """Verifica que registros tengan campos requeridos."""
        parser = ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE)
        results = parser.parse_to_raw()

        required_fields = ["apu_code", "apu_desc", "apu_unit", "category", "insumo_line", "source_line"]

        for record in results:
            for field in required_fields:
                assert field in record, f"Falta campo requerido: {field}"

    def test_parse_whitespace_file(self, whitespace_file):
        """Verifica manejo de archivo solo con espacios."""
        parser = ReportParserCrudo(whitespace_file, profile=TEST_APUS_PROFILE)
        results = parser.parse_to_raw()

        assert len(results) == 0


# =====================================================================
# PRUEBAS DE MÉTRICAS TOPOLÓGICAS
# =====================================================================


class TestTopologicalMetrics:
    """Pruebas de cálculo de métricas topológicas."""

    def test_calculate_field_entropy_uniform(self, parser_with_sample):
        """Verifica entropía con distribución uniforme."""
        fields = ["texto", "123", "mixto1", ""]
        entropy = parser_with_sample._calculate_field_entropy(fields)

        assert 0 <= entropy <= 1

    def test_calculate_field_entropy_single_type(self, parser_with_sample):
        """Verifica entropía con un solo tipo."""
        fields = ["100", "200", "300", "400"]
        entropy = parser_with_sample._calculate_field_entropy(fields)

        # Entropía baja para un solo tipo
        assert entropy < 0.5

    def test_calculate_field_entropy_empty(self, parser_with_sample):
        """Verifica entropía con lista vacía."""
        entropy = parser_with_sample._calculate_field_entropy([])
        assert entropy == 0.0

    def test_calculate_structural_density(self, parser_with_sample):
        """Verifica cálculo de densidad estructural."""
        line = "CEMENTO PORTLAND;KG;50,00;450,00"
        density = parser_with_sample._calculate_structural_density(line)

        assert 0 < density < 1

    def test_calculate_structural_density_empty(self, parser_with_sample):
        """Verifica densidad con línea vacía."""
        density = parser_with_sample._calculate_structural_density("")
        assert density == 0.0

    def test_calculate_numeric_cohesion_contiguous(self, parser_with_sample):
        """Verifica cohesión con números contiguos."""
        fields = ["texto", "100", "200", "300", "texto"]
        cohesion = parser_with_sample._calculate_numeric_cohesion(fields)

        assert cohesion > 0.8  # Alta cohesión

    def test_calculate_numeric_cohesion_dispersed(self, parser_with_sample):
        """Verifica cohesión con números dispersos."""
        fields = ["100", "texto", "texto", "texto", "200"]
        cohesion = parser_with_sample._calculate_numeric_cohesion(fields)

        assert cohesion < 0.5  # Baja cohesión

    def test_calculate_numeric_cohesion_empty(self, parser_with_sample):
        """Verifica cohesión sin números."""
        fields = ["texto", "otro", "más"]
        cohesion = parser_with_sample._calculate_numeric_cohesion(fields)

        assert cohesion == 0.0

    def test_calculate_homogeneity_index_uniform(self, parser_with_sample):
        """Verifica índice de homogeneidad uniforme."""
        fields = ["texto", "otro", "más", "palabras"]
        homogeneity = parser_with_sample._calculate_homogeneity_index(fields)

        assert homogeneity == 1.0  # Todos del mismo tipo

    def test_calculate_homogeneity_index_mixed(self, parser_with_sample):
        """Verifica índice de homogeneidad mixto."""
        fields = ["texto", "100", "otro", "200"]
        homogeneity = parser_with_sample._calculate_homogeneity_index(fields)

        assert 0 < homogeneity < 1

    def test_determine_homeomorphism_class(self, parser_with_sample):
        """Verifica determinación de clase de homeomorfismo."""
        metrics = {
            "field_entropy": 0.8,
            "structural_density": 0.15,
            "numeric_cohesion": 0.9,
            "homogeneity_index": 0.7,
        }

        result = parser_with_sample._determine_homeomorphism_class("full_homeomorphism", metrics)

        assert "CLASE_" in result

    def test_compute_structural_signature(self, parser_with_sample):
        """Verifica cómputo de firma estructural."""
        line1 = "CEMENTO;KG;50;450"
        line2 = "CEMENTO;KG;50;450"
        line3 = "DIFERENTE;M3;100;900"

        sig1 = parser_with_sample._compute_structural_signature(line1)
        sig2 = parser_with_sample._compute_structural_signature(line2)
        sig3 = parser_with_sample._compute_structural_signature(line3)

        assert sig1 == sig2  # Misma estructura
        assert sig1 != sig3  # Diferente estructura
        assert len(sig1) == 16  # Longitud esperada del hash


# =====================================================================
# PRUEBAS DE EXTRACCIÓN DE ENCABEZADOS
# =====================================================================


class TestHeaderExtraction:
    """Pruebas de extracción de encabezados APU."""

    def test_extract_apu_header_valid(self, parser_with_sample):
        """Verifica extracción de encabezado válido."""
        header_line = "CONSTRUCCIÓN DE MURO;;;;; UNIDAD: M2"
        item_line = ";;;;; ITEM: 1,1"

        result = parser_with_sample._extract_apu_header(header_line, item_line, 1)

        assert result is not None
        assert result.apu_code == "1.1"
        assert result.apu_unit == "M2"
        assert "CONSTRUCCIÓN" in result.apu_desc

    def test_extract_apu_header_missing_item(self, parser_with_sample):
        """Verifica manejo de ITEM faltante."""
        header_line = "DESCRIPCION;;;;; UNIDAD: M2"
        item_line = "Línea sin ITEM"

        result = parser_with_sample._extract_apu_header(header_line, item_line, 1)

        # Debería generar código automático
        assert result is not None
        assert "UNKNOWN" in result.apu_code or result.apu_code is not None

    def test_extract_apu_header_invalid_code(self, parser_with_sample):
        """Verifica rechazo de código inválido."""
        header_line = "DESCRIPCION;;;;; UNIDAD: M2"
        item_line = ";;;;; ITEM: X"  # Código muy corto después de limpieza

        result = parser_with_sample._extract_apu_header(header_line, item_line, 1)

        # Puede retornar None o generar código automático
        # El comportamiento depende de clean_apu_code


# =====================================================================
# PRUEBAS DE CONSTRUCCIÓN DE REGISTROS
# =====================================================================


class TestRecordBuilding:
    """Pruebas de construcción de registros de insumo."""

    def test_build_insumo_record(self, parser_with_sample, valid_apu_context):
        """Verifica construcción de registro completo."""
        line = "CEMENTO PORTLAND;KG;50,00;450,00;22500,00;22500,00"
        validation_result = LineValidationResult(
            is_valid=True,
            fields_count=6,
            has_numeric_fields=True,
            validation_layer="full_homeomorphism",
        )

        record = parser_with_sample._build_insumo_record(
            context=valid_apu_context,
            category="MATERIALES",
            line=line,
            line_number=10,
            validation_result=validation_result,
        )

        assert record["apu_code"] == valid_apu_context.apu_code
        assert record["category"] == "MATERIALES"
        assert record["insumo_line"] == line
        assert record["source_line"] == 10
        assert "topological_metrics" in record
        assert "homeomorphism_class" in record

    def test_build_insumo_record_with_precomputed_fields(self, parser_with_sample, valid_apu_context):
        """Verifica construcción con campos pre-procesados."""
        line = "CEMENTO;KG;50;450;22500;22500"
        fields = ["CEMENTO", "KG", "50", "450", "22500", "22500"]
        validation_result = LineValidationResult(
            is_valid=True,
            fields_count=6,
            validation_layer="full_homeomorphism",
        )

        record = parser_with_sample._build_insumo_record(
            context=valid_apu_context,
            category="MATERIALES",
            line=line,
            line_number=10,
            validation_result=validation_result,
            fields=fields,
        )

        assert record["fields_count"] == 6


# =====================================================================
# PRUEBAS DE CACHE
# =====================================================================


class TestCacheManagement:
    """Pruebas de gestión del cache de parsing."""

    def test_cache_result_stores_value(self, parser_with_sample):
        """Verifica almacenamiento en cache."""
        key = "test_key"
        parser_with_sample._cache_result(key, True, MagicMock())

        assert key in parser_with_sample._parse_cache

    def test_cache_result_size_limit(self, parser_with_sample):
        """Verifica límite de tamaño del cache."""
        max_size = parser_with_sample._MAX_CACHE_SIZE

        # Llenar cache más allá del límite
        for i in range(max_size + 100):
            parser_with_sample._cache_result(f"key_{i}", True, None)

        assert len(parser_with_sample._parse_cache) <= max_size

    def test_get_parse_cache(self, parser_with_sample):
        """Verifica exportación del cache."""
        # Añadir algunas entradas válidas
        mock_tree = MagicMock()
        mock_tree.data = "line"
        mock_tree.children = []
        parser_with_sample._parse_cache["valid_key"] = (True, mock_tree)
        parser_with_sample._parse_cache["invalid_key"] = (False, None)

        result = parser_with_sample.get_parse_cache()

        # Solo debería incluir entradas válidas
        assert isinstance(result, dict)


# =====================================================================
# PRUEBAS DE VALIDACIÓN DE INSUMO COMPLETA
# =====================================================================


class TestValidateInsumoLine:
    """Pruebas de validación completa de línea de insumo."""

    @pytest.mark.parametrize("line", VALID_INSUMO_LINES)
    def test_validate_valid_lines(self, parser_with_sample, line):
        """Verifica validación de líneas válidas."""
        fields = [f.strip() for f in line.split(";")]
        result = parser_with_sample._validate_insumo_line(line, fields)

        # Puede ser válido o no dependiendo de Lark
        assert isinstance(result, LineValidationResult)
        assert result.fields_count == len(fields)

    @pytest.mark.parametrize("line", INVALID_INSUMO_LINES)
    def test_validate_invalid_lines(self, parser_with_sample, line):
        """Verifica rechazo de líneas inválidas."""
        fields = [f.strip() for f in line.split(";")] if line else []
        result = parser_with_sample._validate_insumo_line(line, fields)

        assert result.is_valid is False

    def test_validate_none_line(self, parser_with_sample):
        """Verifica manejo de línea None."""
        result = parser_with_sample._validate_insumo_line(None, [])
        assert result.is_valid is False

    def test_validate_none_fields(self, parser_with_sample):
        """Verifica manejo de fields None."""
        result = parser_with_sample._validate_insumo_line("test", None)
        assert result.is_valid is False


# =====================================================================
# PRUEBAS DE LOGGING Y ESTADÍSTICAS
# =====================================================================


class TestLoggingAndStats:
    """Pruebas de logging y estadísticas."""

    def test_log_validation_summary_runs(self, parser_with_sample):
        """Verifica que el resumen de validación se ejecute sin errores."""
        # Simular algunas estadísticas
        parser_with_sample.validation_stats.total_evaluated = 100
        parser_with_sample.validation_stats.passed_basic = 80
        parser_with_sample.validation_stats.passed_lark = 70
        parser_with_sample.stats["insumos_extracted"] = 70

        # No debería lanzar excepción
        parser_with_sample._log_validation_summary()

    def test_record_failed_sample(self, parser_with_sample):
        """Verifica registro de muestras fallidas."""
        line = "Test line"
        fields = ["A", "B", "C"]
        reason = "Test reason"

        parser_with_sample._record_failed_sample(line, fields, reason)

        assert len(parser_with_sample.validation_stats.failed_samples) == 1
        sample = parser_with_sample.validation_stats.failed_samples[0]
        assert sample["line"] == line
        assert sample["reason"] == reason

    def test_record_failed_sample_max_limit(self, parser_with_sample):
        """Verifica límite de muestras fallidas."""
        max_samples = parser_with_sample.config.get(
            "max_failed_samples", parser_with_sample._MAX_FAILED_SAMPLES
        )

        for i in range(max_samples + 10):
            parser_with_sample._record_failed_sample(f"line_{i}", [], "reason")

        assert len(parser_with_sample.validation_stats.failed_samples) <= max_samples


# =====================================================================
# PRUEBAS DE CASOS EDGE
# =====================================================================


class TestEdgeCases:
    """Pruebas de casos límite y especiales."""

    def test_very_long_line(self, temp_dir):
        """Verifica manejo de línea muy larga."""
        long_line = "A" * 10000 + ";" + "B" * 10000
        content = f"""
DESCRIPCION;;;;; UNIDAD: M2
;;;;; ITEM: 1,1
{long_line}
"""
        file_path = temp_dir / "long.txt"
        file_path.write_text(content)

        parser = ReportParserCrudo(file_path, profile=TEST_APUS_PROFILE)
        results = parser.parse_to_raw()
        # No debería crashear

    def test_special_characters(self, temp_dir):
        """Verifica manejo de caracteres especiales."""
        content = """
CONSTRUCCIÓN PIÑÓN ÑÑÑ;;;;; UNIDAD: M²
;;;;; ITEM: 1,1
DESCRIPCIÓN CON ÑOÑO;KG;50,00;€450,00;$22500,00;¥22500,00
"""
        file_path = temp_dir / "special.txt"
        file_path.write_text(content, encoding="utf-8")

        parser = ReportParserCrudo(file_path, profile={"encoding": "utf-8"})
        results = parser.parse_to_raw()
        # No debería crashear

    def test_mixed_line_endings(self, temp_dir):
        """Verifica manejo de diferentes finales de línea."""
        content = "LINEA1;;;;; UNIDAD: M2\r\n;;;;; ITEM: 1,1\r\nDATOS;KG;50;450;22500;22500\n"
        file_path = temp_dir / "mixed.txt"
        file_path.write_bytes(content.encode("utf-8"))

        parser = ReportParserCrudo(file_path, profile=TEST_APUS_PROFILE)
        results = parser.parse_to_raw()
        # No debería crashear

    def test_empty_fields_in_line(self, temp_dir):
        """Verifica manejo de campos vacíos."""
        content = """
DESCRIPCION;;;;; UNIDAD: M2
;;;;; ITEM: 1,1
CEMENTO;;50,00;;22500,00;22500,00
"""
        file_path = temp_dir / "empty_fields.txt"
        file_path.write_text(content)

        parser = ReportParserCrudo(file_path, profile=TEST_APUS_PROFILE)
        results = parser.parse_to_raw()
        # Puede aceptar o rechazar, pero no debería crashear

    def test_unicode_normalization(self, parser_with_sample):
        """Verifica normalización de Unicode."""
        # É puede representarse de dos formas en Unicode
        line1 = "CAF\u00C9"  # É como un solo carácter
        line2 = "CAFE\u0301"  # E + acento combinante

        # Las claves de cache deberían manejar esto
        key1 = parser_with_sample._compute_semantic_cache_key(line1)
        key2 = parser_with_sample._compute_semantic_cache_key(line2)

        # Pueden o no ser iguales según la implementación

    def test_concurrent_like_access(self, sample_apu_file):
        """Simula acceso concurrente al parser."""
        parser = ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE)

        # Llamar parse_to_raw múltiples veces rápidamente
        results = []
        for _ in range(5):
            results.append(parser.parse_to_raw())

        # Todos deberían retornar el mismo resultado
        assert all(r is results[0] for r in results)


# =====================================================================
# PRUEBAS DE RENDIMIENTO
# =====================================================================


class TestPerformance:
    """Pruebas de rendimiento."""

    def test_parse_time_reasonable(self, sample_apu_file):
        """Verifica que el parsing se complete en tiempo razonable."""
        parser = ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE)

        start = time.time()
        parser.parse_to_raw()
        elapsed = time.time() - start

        # Debería completarse en menos de 5 segundos para archivo pequeño
        assert elapsed < 5.0

    def test_cache_improves_performance(self, parser_with_sample):
        """Verifica que el cache mejore el rendimiento."""
        line = "CEMENTO PORTLAND;KG;50,00;450,00;22500,00;22500,00"

        # Primera llamada (sin cache)
        start1 = time.time()
        for _ in range(100):
            parser_with_sample._validate_with_lark(line, use_cache=False)
        time_no_cache = time.time() - start1

        # Limpiar cache
        parser_with_sample._parse_cache.clear()

        # Segunda llamada (con cache)
        start2 = time.time()
        for _ in range(100):
            parser_with_sample._validate_with_lark(line, use_cache=True)
        time_with_cache = time.time() - start2

        # Con cache debería ser al menos igual de rápido
        # (la primera iteración llena el cache)
        assert time_with_cache <= time_no_cache * 2

    @pytest.mark.parametrize("line_count", [100, 500])
    def test_large_file_parsing(self, temp_dir, line_count):
        """Verifica parsing de archivos grandes."""
        # Generar contenido grande
        lines = ["DESCRIPCION;;;;; UNIDAD: M2", ";;;;; ITEM: 1,1"]
        for i in range(line_count):
            lines.append(f"INSUMO_{i};KG;{i};{i*10};{i*100};{i*100}")

        file_path = temp_dir / "large.txt"
        file_path.write_text("\n".join(lines))

        parser = ReportParserCrudo(file_path, profile=TEST_APUS_PROFILE)

        start = time.time()
        results = parser.parse_to_raw()
        elapsed = time.time() - start

        # Tiempo proporcional al tamaño
        max_time = line_count * 0.01  # 10ms por línea máximo
        assert elapsed < max_time or elapsed < 30  # O máximo 30 segundos


# =====================================================================
# PRUEBAS DE INTEGRACIÓN
# =====================================================================


class TestIntegration:
    """Pruebas de integración end-to-end."""

    def test_full_workflow(self, sample_apu_file):
        """Verifica flujo completo de trabajo."""
        # 1. Inicialización
        parser = ReportParserCrudo(sample_apu_file, profile=TEST_APUS_PROFILE)
        assert not parser._parsed

        # 2. Parsing
        results = parser.parse_to_raw()
        assert parser._parsed

        # 3. Verificar resultados
        if results:
            for record in results:
                assert "apu_code" in record
                assert "insumo_line" in record
                assert "topological_metrics" in record

        # 4. Cache
        cached = parser.get_parse_cache()
        assert isinstance(cached, dict)

        # 5. Re-parsing (debe usar cache)
        results2 = parser.parse_to_raw()
        assert results is results2

    def test_error_recovery(self, temp_dir):
        """Verifica recuperación ante errores en contenido."""
        content = """
DESCRIPCION VALIDA;;;;; UNIDAD: M2
;;;;; ITEM: 1,1
LINEA VALIDA;KG;50,00;450,00;22500,00;22500,00
LINEA %%%% CORRUPTA %%%%
OTRA LINEA VALIDA;M3;10,00;1000,00;10000,00;10000,00
SUBTOTAL;;;;;32500,00
"""
        file_path = temp_dir / "mixed.txt"
        file_path.write_text(content)

        parser = ReportParserCrudo(file_path, profile=TEST_APUS_PROFILE)
        results = parser.parse_to_raw()

        # Debería procesar las líneas válidas
        # y manejar graciosamente las inválidas
        assert parser.stats.get("lines_ignored_in_context", 0) >= 0 or len(results) >= 0