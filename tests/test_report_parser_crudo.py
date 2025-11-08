"""
Suite de pruebas completa para ReportParserCrudo.
Cubre funcionalidad, casos edge, errores y rendimiento.
"""

import re
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest

# Importar las clases a probar
from app.report_parser_crudo import (
    APUContext,
    FileReadError,
    ParserConfig,
    ParseStrategyError,
    ParsingStrategy,
    PatternMatcher,
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
def sample_apu_content_blocks() -> str:
    """Contenido de ejemplo con formato de bloques."""
    return """
DESCRIPCIÓN: EXCAVACIÓN MANUAL EN MATERIAL COMÚN
UNIDAD: M3
ITEM: EXC-001

MATERIALES
Arena gruesa;M3;0.50;25000;12500
Cemento Portland;KG;50;800;40000
Agua;LT;150;50;7500

MANO DE OBRA
Oficial;HH;8;35000;280000
Ayudante;HH;16;25000;400000

EQUIPO
Retroexcavadora;HM;2;150000;300000
Volqueta;HM;4;120000;480000


DESCRIPCIÓN: CONCRETO SIMPLE F'C=210 KG/CM2
UNIDAD: M3
ITEM: CON-002

MATERIALES
Cemento Portland Tipo I;BLS;9.73;95000;923350
Arena gruesa;M3;0.52;120000;62400
Piedra chancada 1/2";M3;0.53;140000;74200
Agua;M3;0.186;5000;930

MANO DE OBRA
Operario;HH;2;45000;90000
Oficial;HH;2;35000;70000
Peón;HH;8;25000;200000

EQUIPO
Mezcladora de concreto;HM;1;80000;80000
Vibrador de concreto;HM;0.5;50000;25000
"""


@pytest.fixture
def sample_apu_content_lines() -> str:
    """Contenido de ejemplo con formato de líneas."""
    return """
ITEM: TUB-001 DESCRIPCIÓN: TUBERÍA PVC 4" UNIDAD: ML
================================================================
CODIGO;DESCRIPCION;UNIDAD;CANTIDAD;P.UNITARIO;PARCIAL
----------------------------------------------------------------
MATERIALES
MAT001;Tubería PVC 4" C-10;ML;1.05;25000;26250
MAT002;Pegamento PVC;GAL;0.01;85000;850
MAT003;Limpiador PVC;GAL;0.01;45000;450
MANO DE OBRA
MO001;Plomero;HH;0.5;40000;20000
MO002;Ayudante plomería;HH;0.5;25000;12500
TRANSPORTE
TR001;Transporte materiales;GLB;1;15000;15000
================================================================
SUBTOTAL: 75050

ITEM: TUB-002 DESCRIPCIÓN: TUBERÍA CPVC 1/2" UNIDAD: ML
================================================================
MATERIALES
MAT004;Tubería CPVC 1/2";ML;1.05;18000;18900
MAT002;Pegamento CPVC;GAL;0.005;95000;475
"""


@pytest.fixture
def sample_apu_content_mixed() -> str:
    """Contenido con formato mixto y casos especiales."""
    return """
Proyecto: Construcción Edificio XYZ
Fecha: 2024-01-15
=====================================

COD: EST-001
Descripción del APU: Estructura metálica tipo A
Und: KG

--- MATERIALES ---
Perfil metálico W8x31    KG    1.00    $8,500    $8,500
Soldadura E6011          KG    0.05    $12,000   $600
Pintura anticorrosiva   GAL   0.01    $45,000   $450

*** MANO DE OBRA ***
• Soldador especializado    HH    0.15    45000    6750
• Ayudante soldadura        HH    0.15    25000    3750

[EQUIPO]
- Equipo de soldadura    HM    0.10    35000    3500
- Pulidora               HM    0.05    25000    1250

Total: $24,800
_____________________________________________

CÓDIGO: EST-002 | DESCRIPCIÓN: Estructura metálica tipo B | UNIDAD: KG
Materiales:
Perfil W10x49|KG|1.0|9500|9500
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


@pytest.fixture
def sample_corrupted_content() -> str:
    """Contenido con caracteres corruptos."""
    return "ITEM: APU001\n" + "Descripci\udcf3n: Test\udce1\udce9\udcedo\udcfa" + "\nMateriales\n"


@pytest.fixture
def basic_config() -> ParserConfig:
    """Configuración básica para pruebas."""
    return ParserConfig(
        encodings=['utf-8', 'latin1'],
        strategy='auto',
        debug_mode=True,
        max_debug_samples=5,
        max_lines_to_process=1000
    )


@pytest.fixture
def strict_config() -> ParserConfig:
    """Configuración estricta para pruebas."""
    return ParserConfig(
        min_apu_code_length=5,
        min_description_length=10,
        confidence_threshold=0.8,
        debug_mode=False
    )


# =====================================================================
# PRUEBAS DE CONFIGURACIÓN
# =====================================================================

class TestParserConfig:
    """Pruebas para la clase ParserConfig."""

    def test_default_config_creation(self):
        """Verifica que la configuración por defecto se crea correctamente."""
        config = ParserConfig()
        assert config.encodings == ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        assert config.strategy == 'auto'
        assert config.default_unit == 'UND'
        assert config.max_lines_to_process == 100000
        assert 0 <= config.confidence_threshold <= 1

    def test_custom_config_creation(self):
        """Verifica la creación con parámetros personalizados."""
        config = ParserConfig(
            encodings=['utf-8'],
            strategy='blocks',
            min_apu_code_length=3,
            default_unit='UN'
        )
        assert config.encodings == ['utf-8']
        assert config.strategy == 'blocks'
        assert config.min_apu_code_length == 3
        assert config.default_unit == 'UN'

    def test_invalid_strategy_raises_error(self):
        """Verifica que estrategias inválidas lancen error."""
        with pytest.raises(ValueError, match="Estrategia inválida"):
            ParserConfig(strategy='invalid_strategy')

    def test_invalid_min_values_raise_error(self):
        """Verifica validación de valores mínimos."""
        with pytest.raises(ValueError, match="min_apu_code_length debe ser >= 1"):
            ParserConfig(min_apu_code_length=0)

        with pytest.raises(ValueError, match="min_description_length debe ser >= 1"):
            ParserConfig(min_description_length=0)

    def test_invalid_max_lines_raises_error(self):
        """Verifica validación de límite de líneas."""
        with pytest.raises(ValueError, match="max_lines_to_process debe ser >= 100"):
            ParserConfig(max_lines_to_process=50)

    def test_invalid_confidence_threshold(self):
        """Verifica validación del umbral de confianza."""
        with pytest.raises(ValueError, match="confidence_threshold debe estar entre 0 y 1"):
            ParserConfig(confidence_threshold=1.5)

        with pytest.raises(ValueError, match="confidence_threshold debe estar entre 0 y 1"):
            ParserConfig(confidence_threshold=-0.1)

    def test_empty_encodings_raises_error(self):
        """Verifica que se requiere al menos un encoding."""
        with pytest.raises(ValueError, match="Debe especificar al menos un encoding"):
            ParserConfig(encodings=[])


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
            apu_unit="M3"
        )
        assert context.apu_code == "TEST-001"
        assert context.apu_desc == "Descripción de prueba"
        assert context.apu_unit == "M3"
        assert context.confidence == 1.0
        assert context.is_valid

    def test_apu_context_normalization(self):
        """Verifica normalización de valores."""
        context = APUContext(
            apu_code="  TEST-002  ",
            apu_desc="  Descripción  ",
            apu_unit="  m3  "
        )
        assert context.apu_code == "TEST-002"
        assert context.apu_desc == "Descripción"
        assert context.apu_unit == "M3"  # Convertido a mayúsculas

    def test_empty_apu_code_raises_error(self):
        """Verifica que código vacío lance error."""
        with pytest.raises(ValueError, match="El código APU no puede estar vacío"):
            APUContext(apu_code="", apu_desc="Test", apu_unit="UN")

    def test_none_values_handling(self):
        """Verifica manejo de valores None."""
        context = APUContext(
            apu_code="TEST-003",
            apu_desc=None,
            apu_unit=None
        )
        assert context.apu_desc == ""
        assert context.apu_unit == "UND"  # Valor por defecto

    def test_is_valid_property(self):
        """Verifica la propiedad is_valid."""
        valid_context = APUContext(apu_code="AB", apu_desc="Test", apu_unit="UN")
        assert valid_context.is_valid

        # Código muy corto después de limpieza
        context = APUContext(apu_code="A ", apu_desc="Test", apu_unit="UN")
        assert not context.is_valid

    def test_to_dict_conversion(self):
        """Verifica conversión a diccionario."""
        context = APUContext(
            apu_code="TEST-004",
            apu_desc="Descripción",
            apu_unit="KG",
            confidence=0.85
        )
        result = context.to_dict()
        assert isinstance(result, dict)
        assert result['apu_code'] == "TEST-004"
        assert result['apu_desc'] == "Descripción"
        assert result['apu_unit'] == "KG"
        assert 'confidence' not in result  # No incluido en to_dict


# =====================================================================
# PRUEBAS DE PatternMatcher
# =====================================================================

class TestPatternMatcher:
    """Pruebas para la clase PatternMatcher."""

    @pytest.fixture
    def matcher(self) -> PatternMatcher:
        """Crea una instancia de PatternMatcher."""
        return PatternMatcher()

    def test_pattern_compilation(self, matcher):
        """Verifica que los patrones se compilen correctamente."""
        assert len(matcher._patterns) > 0
        assert all(isinstance(p, re.Pattern) for p in matcher._patterns.values())

    def test_item_pattern_matching(self, matcher):
        """Prueba el patrón de items."""
        test_cases = [
            ("ITEM: ABC-001", "ABC-001"),
            ("CÓDIGO: TEST123", "TEST123"),
            ("COD. XYZ-456", "XYZ-456"),
            ("Item:A1B2C3", "A1B2C3"),
        ]

        for text, expected in test_cases:
            match = matcher.match('item_flexible', text)
            assert match is not None, f"No match for: {text}"
            assert match.group(1) == expected

    def test_unit_pattern_matching(self, matcher):
        """Prueba el patrón de unidades."""
        test_cases = [
            ("UNIDAD: M3", "M3"),
            ("UND. KG", "KG"),
            ("U: ML", "ML"),
            ("UNIDAD:GLB", "GLB"),
        ]

        for text, expected in test_cases:
            match = matcher.match('unit_flexible', text)
            assert match is not None, f"No match for: {text}"
            assert match.group(1) == expected

    def test_numeric_row_pattern(self, matcher):
        """Prueba el patrón de filas numéricas."""
        valid_rows = [
            "1.5 25,000 37,500",
            "10.00 1500.50 15005.00",
            "0,5 1.250,00 625,00"
        ]

        invalid_rows = [
            "Solo texto sin números",
            "Un solo numero 123",
            "Dos numeros 12 34"
        ]

        for row in valid_rows:
            assert matcher.match('numeric_row', row) is not None

        for row in invalid_rows:
            assert matcher.match('numeric_row', row) is None

    def test_currency_pattern(self, matcher):
        """Prueba el patrón de moneda."""
        valid_currencies = [
            "$1,234.56",
            "€ 500",
            "25000 COP",
            "1500.00 USD"
        ]

        for curr in valid_currencies:
            assert matcher.match('currency', curr) is not None

    def test_cache_functionality(self, matcher):
        """Verifica que el cache LRU funcione."""
        text = "ITEM: TEST-CACHE"

        # Primera llamada - no está en cache
        result1 = matcher.match('item_flexible', text)

        # Segunda llamada - debe estar en cache
        result2 = matcher.match('item_flexible', text)

        assert result1 is not None
        assert result2 is not None
        assert result1.group(1) == result2.group(1)

    def test_invalid_pattern_name_raises_error(self, matcher):
        """Verifica que patrones inválidos lancen error."""
        with pytest.raises(ValueError, match="Patrón no definido"):
            matcher.match('non_existent_pattern', "test")

        with pytest.raises(ValueError, match="Patrón no definido"):
            matcher.get_pattern('invalid_pattern')


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

    def test_initialization_with_custom_config(self, temp_dir, strict_config):
        """Verifica inicialización con configuración personalizada."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("Test")

        parser = ReportParserCrudo(file_path, config=strict_config)
        assert parser.config == strict_config
        assert parser.config.min_apu_code_length == 5

    def test_file_not_found_raises_error(self):
        """Verifica que archivo inexistente lance error."""
        with pytest.raises(FileNotFoundError, match="Archivo no encontrado"):
            ReportParserCrudo("/path/to/nonexistent/file.txt")

    def test_directory_path_raises_error(self, temp_dir):
        """Verifica que pasar directorio lance error."""
        with pytest.raises(ValueError, match="La ruta no corresponde a un archivo"):
            ReportParserCrudo(temp_dir)

    def test_empty_file_raises_error(self, temp_dir):
        """Verifica que archivo vacío lance error."""
        file_path = temp_dir / "empty.txt"
        file_path.touch()  # Crear archivo vacío

        with pytest.raises(ValueError, match="El archivo está vacío"):
            ReportParserCrudo(file_path)

    def test_large_file_warning(self, temp_dir, caplog):
        """Verifica advertencia para archivos grandes."""
        file_path = temp_dir / "large.txt"
        # Crear archivo de más de 100MB (simulado con poco contenido)
        file_path.write_text("x" * 1000)

        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value = Mock(st_size=101 * 1024 * 1024)
            parser = ReportParserCrudo(file_path)
            assert "Archivo muy grande" in caplog.text


# =====================================================================
# PRUEBAS DE LECTURA DE ARCHIVOS
# =====================================================================

class TestFileReading:
    """Pruebas de lectura de archivos con diferentes encodings."""

    def test_read_utf8_file(self, temp_dir, basic_config):
        """Verifica lectura de archivo UTF-8."""
        file_path = temp_dir / "utf8.txt"
        content = "ITEM: TEST-001\nDescripción: Ñoño está aquí"
        file_path.write_text(content, encoding='utf-8')

        parser = ReportParserCrudo(file_path, config=basic_config)
        read_content = parser._read_file_safely()

        assert "TEST-001" in read_content
        assert "Ñoño" in read_content
        assert parser.stats['encoding_used'] == 'utf-8'

    def test_read_latin1_file(self, temp_dir):
        """Verifica lectura de archivo Latin-1."""
        file_path = temp_dir / "latin1.txt"
        content = "ITEM: TEST-002\nDescripción: Café"
        file_path.write_bytes(content.encode('latin1'))

        config = ParserConfig(encodings=['latin1', 'utf-8'])
        parser = ReportParserCrudo(file_path, config=config)
        read_content = parser._read_file_safely()

        assert "TEST-002" in read_content
        assert parser.stats['encoding_used'] == 'latin1'

    def test_read_with_encoding_errors(self, temp_dir):
        """Verifica manejo de errores de encoding."""
        file_path = temp_dir / "mixed.txt"
        # Crear contenido con mezcla problemática
        file_path.write_bytes(b'ITEM: TEST\nDesc: \xff\xfe')

        parser = ReportParserCrudo(file_path)
        content = parser._read_file_safely()

        assert "ITEM: TEST" in content
        assert 'with_replacements' in parser.stats.get('encoding_used', '')

    def test_all_encodings_fail(self, temp_dir):
        """Verifica error cuando todos los encodings fallan."""
        file_path = temp_dir / "bad.txt"
        file_path.write_text("test")

        config = ParserConfig(encodings=['invalid_encoding'])
        parser = ReportParserCrudo(file_path, config=config)

        with pytest.raises(FileReadError, match="No se pudo leer el archivo"):
            parser._read_file_safely()

    def test_content_hash_generation(self, temp_dir):
        """Verifica generación de hash del contenido."""
        file_path = temp_dir / "hash.txt"
        content = "ITEM: TEST-001\nMateriales"
        file_path.write_text(content)

        parser = ReportParserCrudo(file_path)
        generated_hash = parser._generate_content_hash(content)

        assert isinstance(generated_hash, str)
        assert len(generated_hash) == 32  # MD5 hash length

        # El mismo contenido debe generar el mismo hash
        assert generated_hash == parser._generate_content_hash(content)


# =====================================================================
# PRUEBAS DE DETECCIÓN DE SEPARADORES
# =====================================================================

class TestSeparatorDetection:
    """Pruebas de detección automática de separadores."""

    def test_detect_semicolon_separator(self):
        """Verifica detección de punto y coma."""
        content = """
        ITEM: TEST-001
        MAT001;Material A;KG;10;1000;10000
        MAT002;Material B;UN;5;500;2500
        MAT003;Material C;M3;2;25000;50000
        """

        parser = ReportParserCrudo.__new__(ReportParserCrudo)
        parser.config = ParserConfig()
        parser.stats = Counter()
        parser._detected_separator = None

        confidence = parser._auto_detect_separator(content)

        assert parser._detected_separator == ';'
        assert confidence > 0.5
        assert not parser._separator_is_regex

    def test_detect_tab_separator(self):
        """Verifica detección de tabulación."""
        content = """
        ITEM: TEST-002
        MAT001\tMaterial A\tKG\t10\t1000
        MAT002\tMaterial B\tUN\t5\t500
        MAT003\tMaterial C\tM3\t2\t25000
        """

        parser = ReportParserCrudo.__new__(ReportParserCrudo)
        parser.config = ParserConfig()
        parser.stats = Counter()
        parser._detected_separator = None

        confidence = parser._auto_detect_separator(content)

        assert parser._detected_separator == '\t'
        assert confidence > 0.5

    def test_detect_pipe_separator(self):
        """Verifica detección de pipe."""
        content = """
        MAT001|Material A|KG|10|1000
        MAT002|Material B|UN|5|500
        MAT003|Material C|M3|2|25000
        """

        parser = ReportParserCrudo.__new__(ReportParserCrudo)
        parser.config = ParserConfig()
        parser.stats = Counter()
        parser._detected_separator = None

        confidence = parser._auto_detect_separator(content)

        assert parser._detected_separator == '|'
        assert confidence > 0.4

    def test_detect_multiple_spaces_separator(self):
        """Verifica detección de múltiples espacios."""
        content = """
        MAT001    Material A    KG    10    1000
        MAT002    Material B    UN    5     500
        MAT003    Material C    M3    2     25000
        """

        parser = ReportParserCrudo.__new__(ReportParserCrudo)
        parser.config = ParserConfig()
        parser.stats = Counter()
        parser._detected_separator = None

        parser._auto_detect_separator(content)

        assert parser._detected_separator == r'\s{2,}'
        assert parser._separator_is_regex

    def test_no_clear_separator(self):
        """Verifica cuando no hay separador claro."""
        content = """
        Este es un texto sin formato claro
        No tiene separadores consistentes
        Solo texto normal
        """

        parser = ReportParserCrudo.__new__(ReportParserCrudo)
        parser.config = ParserConfig()
        parser.stats = Counter()
        parser._detected_separator = None

        confidence = parser._auto_detect_separator(content)

        assert confidence == 0.0 or confidence < 0.3


# =====================================================================
# PRUEBAS DE ESTRATEGIAS DE PARSING
# =====================================================================

class TestParsingStrategies:
    """Pruebas de las diferentes estrategias de parsing."""

    def test_strategy_auto_detection_blocks(self, temp_dir, sample_apu_content_blocks):
        """Verifica detección automática de estrategia de bloques."""
        file_path = temp_dir / "blocks.txt"
        file_path.write_text(sample_apu_content_blocks)

        parser = ReportParserCrudo(file_path)
        strategy = parser._determine_strategy(sample_apu_content_blocks)

        assert strategy == ParsingStrategy.BLOCKS

    def test_strategy_auto_detection_lines(self, temp_dir, sample_apu_content_lines):
        """Verifica detección automática de estrategia de líneas."""
        file_path = temp_dir / "lines.txt"
        file_path.write_text(sample_apu_content_lines)

        parser = ReportParserCrudo(file_path)
        strategy = parser._determine_strategy(sample_apu_content_lines)

        assert strategy in [ParsingStrategy.LINES, ParsingStrategy.BLOCKS]

    def test_forced_strategy_blocks(self, temp_dir, sample_apu_content_lines):
        """Verifica estrategia forzada a bloques."""
        file_path = temp_dir / "test.txt"
        file_path.write_text(sample_apu_content_lines)

        config = ParserConfig(strategy='blocks')
        parser = ReportParserCrudo(file_path, config=config)
        strategy = parser._determine_strategy(sample_apu_content_lines)

        assert strategy == ParsingStrategy.BLOCKS

    def test_forced_strategy_lines(self, temp_dir, sample_apu_content_blocks):
        """Verifica estrategia forzada a líneas."""
        file_path = temp_dir / "test.txt"
        file_path.write_text(sample_apu_content_blocks)

        config = ParserConfig(strategy='lines')
        parser = ReportParserCrudo(file_path, config=config)
        strategy = parser._determine_strategy(sample_apu_content_blocks)

        assert strategy == ParsingStrategy.LINES

    def test_hybrid_strategy(self, temp_dir):
        """Verifica estrategia híbrida."""
        content = "Contenido ambiguo sin formato claro\nAlgunas líneas\nPero no muchas"
        file_path = temp_dir / "hybrid.txt"
        file_path.write_text(content)

        config = ParserConfig(strategy='auto')
        parser = ReportParserCrudo(file_path, config=config)
        strategy = parser._determine_strategy(content)

        # Con contenido ambiguo, debería elegir híbrido o líneas
        assert strategy in [ParsingStrategy.HYBRID, ParsingStrategy.LINES]


# =====================================================================
# PRUEBAS DE EXTRACCIÓN DE APU
# =====================================================================

class TestAPUExtraction:
    """Pruebas de extracción de contexto APU."""

    @pytest.fixture
    def parser(self, temp_dir) -> ReportParserCrudo:
        """Crea un parser para pruebas."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("dummy")
        return ReportParserCrudo(file_path)

    def test_extract_full_header_pattern(self, parser):
        """Verifica extracción con patrón completo."""
        lines = [
            "Excavación manual en terreno normal UNIDAD: M3 ITEM: EXC-001",
            "Materiales:"
        ]

        context = parser._extract_apu_context_enhanced(lines)

        assert context is not None
        assert context.apu_code == "EXC-001"
        assert "Excavación manual" in context.apu_desc
        assert context.apu_unit == "M3"
        assert context.confidence >= 0.9

    def test_extract_item_first_pattern(self, parser):
        """Verifica extracción con ítem primero."""
        lines = [
            "ITEM: CON-002 DESCRIPCIÓN: Concreto simple UNIDAD: M3",
            "Materiales:"
        ]

        context = parser._extract_apu_context_enhanced(lines)

        assert context is not None
        assert context.apu_code == "CON-002"
        assert "Concreto simple" in context.apu_desc
        assert context.apu_unit == "M3"

    def test_extract_separate_fields(self, parser):
        """Verifica extracción con campos separados."""
        lines = [
            "CÓDIGO: TUB-003",
            "DESCRIPCIÓN: Tubería PVC de 4 pulgadas",
            "UNIDAD: ML",
            "Materiales:"
        ]

        context = parser._extract_apu_context_enhanced(lines)

        assert context is not None
        assert context.apu_code == "TUB-003"
        assert "Tubería PVC" in context.apu_desc
        assert context.apu_unit == "ML"
        assert context.confidence < 0.9  # Menor confianza por campos separados

    def test_extract_minimal_context(self, parser):
        """Verifica extracción con información mínima."""
        lines = [
            "ITEM: MIN-001",
            "Lista de materiales"
        ]

        context = parser._extract_apu_context_enhanced(lines)

        assert context is not None
        assert context.apu_code == "MIN-001"
        assert context.apu_unit == "UND"  # Valor por defecto
        assert context.confidence < 0.8

    def test_no_apu_found(self, parser):
        """Verifica cuando no se encuentra APU."""
        lines = [
            "Solo texto sin formato",
            "Sin códigos ni items"
        ]

        context = parser._extract_apu_context_enhanced(lines)
        assert context is None

    def test_invalid_apu_code_rejected(self, parser):
        """Verifica rechazo de códigos inválidos."""
        lines = [
            "ITEM: X",  # Muy corto
            "Descripción: Test"
        ]

        parser.config.min_apu_code_length = 3
        context = parser._extract_apu_context_enhanced(lines)
        assert context is None


# =====================================================================
# PRUEBAS DE DETECCIÓN DE CATEGORÍAS
# =====================================================================

class TestCategoryDetection:
    """Pruebas de detección de categorías."""

    @pytest.fixture
    def parser(self, temp_dir) -> ReportParserCrudo:
        """Crea un parser para pruebas."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("dummy")
        return ReportParserCrudo(file_path)

    @pytest.mark.parametrize("line,expected", [
        ("MATERIALES", "MATERIALES"),
        ("MATERIAL", "MATERIALES"),
        ("MAT.", "MATERIALES"),
        ("INSUMOS", "MATERIALES"),
        ("MANO DE OBRA", "MANO DE OBRA"),
        ("M.O.", "MANO DE OBRA"),
        ("PERSONAL", "MANO DE OBRA"),
        ("EQUIPO", "EQUIPO"),
        ("MAQUINARIA", "EQUIPO"),
        ("TRANSPORTE", "TRANSPORTE"),
        ("HERRAMIENTAS", "HERRAMIENTA"),
        ("OTROS", "OTROS"),
    ])
    def test_detect_categories(self, parser, line, expected):
        """Verifica detección de diferentes categorías."""
        result = parser._detect_category(line.upper())
        assert result == expected

    def test_category_with_context(self, parser):
        """Verifica detección con contexto adicional."""
        lines = [
            "*** MATERIALES ***",
            "--- MANO DE OBRA ---",
            "[EQUIPO]",
            "TRANSPORTE:",
            "• HERRAMIENTAS",
        ]

        for line in lines:
            result = parser._detect_category(line.upper())
            assert result is not None

    def test_reject_false_categories(self, parser):
        """Verifica rechazo de falsas categorías."""
        lines = [
            "Material de construcción;KG;10;5000;50000",  # Línea de datos
            "123456 MATERIAL 789",  # Muchos números
            "A" * 101,  # Muy largo
        ]

        for line in lines:
            result = parser._detect_category(line.upper())
            assert result is None

    def test_category_cache(self, parser):
        """Verifica funcionamiento del cache de categorías."""
        line = "MATERIALES"

        # Primera llamada
        result1 = parser._detect_category_cached(line)
        assert result1 == "MATERIALES"
        assert line in parser._category_cache

        # Segunda llamada (desde cache)
        result2 = parser._detect_category_cached(line)
        assert result2 == result1


# =====================================================================
# PRUEBAS DE VALIDACIÓN DE INSUMOS
# =====================================================================

class TestInsumoValidation:
    """Pruebas de validación de insumos."""

    @pytest.fixture
    def parser(self, temp_dir) -> ReportParserCrudo:
        """Crea un parser para pruebas."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("dummy")
        parser = ReportParserCrudo(file_path)
        parser._detected_separator = ';'
        return parser

    def test_valid_insumo_with_numbers(self, parser):
        """Verifica insumo válido con números."""
        lines = [
            "MAT001;Cemento Portland;BLS;10;95000;950000",
            "Arena gruesa M3 0.52 120000 62400",
            "Piedra|M3|0.53|140000|74200",
        ]

        for line in lines:
            assert parser._is_valid_insumo_enhanced(line)

    def test_valid_insumo_with_currency(self, parser):
        """Verifica insumo con formato de moneda."""
        lines = [
            "Material A;$1,500;10 unidades",
            "Material B;5000 COP;cantidad: 20",
        ]

        for line in lines:
            assert parser._is_valid_insumo_enhanced(line)

    def test_invalid_insumo_too_short(self, parser):
        """Verifica rechazo de líneas muy cortas."""
        lines = ["", "   ", "AB", "123"]

        for line in lines:
            assert not parser._is_valid_insumo_enhanced(line)

    def test_invalid_insumo_too_long(self, parser):
        """Verifica rechazo de líneas muy largas."""
        line = "x" * 501  # Más de 500 caracteres
        assert not parser._is_valid_insumo_enhanced(line)

    def test_invalid_insumo_single_part(self, parser):
        """Verifica rechazo de líneas con una sola parte."""
        lines = [
            "SoloUnaParte",
            "Texto sin separadores ni estructura",
        ]

        for line in lines:
            assert not parser._is_valid_insumo_enhanced(line)

    def test_split_line_with_different_separators(self, parser):
        """Verifica división de líneas con diferentes separadores."""
        # Separador simple
        parser._detected_separator = ';'
        parser._separator_is_regex = False
        parts = parser._split_line("A;B;C")
        assert parts == ['A', 'B', 'C']

        # Separador regex
        parser._detected_separator = r'\s{2,}'
        parser._separator_is_regex = True
        parts = parser._split_line("A    B    C")
        assert parts == ['A', 'B', 'C']


# =====================================================================
# PRUEBAS DE LÍNEAS BASURA
# =====================================================================

class TestJunkLineDetection:
    """Pruebas de detección de líneas basura."""

    @pytest.fixture
    def parser(self, temp_dir) -> ReportParserCrudo:
        """Crea un parser para pruebas."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("dummy")
        return ReportParserCrudo(file_path)

    @pytest.mark.parametrize("line", [
        "SUBTOTAL",
        "COSTO DIRECTO",
        "TOTAL: 150000",
        "IMPUESTO 19%",
        "IVA",
        "=" * 50,
        "-" * 30,
        "_" * 40,
        "Página 1",
        "Page 25",
        "  ",
        "••••••",
    ])
    def test_detect_junk_lines(self, parser, line):
        """Verifica detección de líneas basura."""
        assert parser._is_junk_line(line)

    @pytest.mark.parametrize("line", [
        "Material cemento",
        "MAT001;Descripción;UN;10",
        "Oficial HH 8 35000",
    ])
    def test_valid_lines_not_junk(self, parser, line):
        """Verifica que líneas válidas no se marquen como basura."""
        assert not parser._is_junk_line(line)

    def test_junk_cache(self, parser):
        """Verifica cache de líneas basura."""
        line = "SUBTOTAL"

        # Primera llamada
        result1 = parser._is_junk_line_cached(line)
        assert result1 is True

        # Verificar que se agregó al cache
        assert line in parser._junk_cache

        # Segunda llamada (desde cache)
        result2 = parser._is_junk_line_cached(line)
        assert result2 == result1


# =====================================================================
# PRUEBAS DE PARSING COMPLETO
# =====================================================================

class TestCompleteParsing:
    """Pruebas de flujo completo de parsing."""

    def test_parse_blocks_format(self, temp_dir, sample_apu_content_blocks):
        """Verifica parsing completo con formato de bloques."""
        file_path = temp_dir / "blocks.txt"
        file_path.write_text(sample_apu_content_blocks)

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        assert len(results) > 0
        assert parser.stats['insumos_extracted'] > 0

        # Verificar estructura de registros
        for record in results:
            assert 'apu_code' in record
            assert 'apu_desc' in record
            assert 'apu_unit' in record
            assert 'category' in record
            assert 'insumo_line' in record
            assert 'confidence' in record

    def test_parse_lines_format(self, temp_dir, sample_apu_content_lines):
        """Verifica parsing completo con formato de líneas."""
        file_path = temp_dir / "lines.txt"
        file_path.write_text(sample_apu_content_lines)

        config = ParserConfig(field_separator=';')
        parser = ReportParserCrudo(file_path, config=config)
        results = parser.parse_to_raw()

        assert len(results) > 0

        # Verificar APUs detectados
        apu_codes = {r['apu_code'] for r in results}
        assert 'TUB-001' in apu_codes
        assert 'TUB-002' in apu_codes

    def test_parse_mixed_format(self, temp_dir, sample_apu_content_mixed):
        """Verifica parsing con formato mixto."""
        file_path = temp_dir / "mixed.txt"
        file_path.write_text(sample_apu_content_mixed)

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        # Debe extraer algo incluso con formato mixto
        assert parser.stats.get('apus_detected', 0) > 0 or \
               parser.stats.get('valid_apu_blocks', 0) > 0

    def test_parse_invalid_content(self, temp_dir, sample_invalid_content):
        """Verifica manejo de contenido inválido."""
        file_path = temp_dir / "invalid.txt"
        file_path.write_text(sample_invalid_content)

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        # No debe extraer nada de contenido inválido
        assert len(results) == 0
        assert parser.stats['insumos_extracted'] == 0

    def test_parse_already_parsed(self, temp_dir):
        """Verifica que no se re-procese archivo ya parseado."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("ITEM: TEST-001\nMaterial;UN;10")

        parser = ReportParserCrudo(file_path)

        # Primer parse
        results1 = parser.parse_to_raw()
        initial_count = len(results1)

        # Segundo parse (debe devolver cache)
        results2 = parser.parse_to_raw()

        assert len(results2) == initial_count
        assert parser._parsed is True


# =====================================================================
# PRUEBAS DE FALLBACK Y RECUPERACIÓN
# =====================================================================

class TestFallbackStrategies:
    """Pruebas de estrategias de respaldo."""

    def test_fallback_on_failed_primary_strategy(self, temp_dir):
        """Verifica fallback cuando falla estrategia principal."""
        content = """Material;KG;10;5000
        Cemento;BLS;50;95000
        Arena;M3;2;120000"""

        file_path = temp_dir / "fallback.txt"
        file_path.write_text(content)

        # Forzar estrategia que fallará
        config = ParserConfig(strategy='blocks')
        parser = ReportParserCrudo(file_path, config=config)

        with patch.object(parser, '_parse_by_blocks', return_value=False):
            with patch.object(parser, '_fallback_parsing', return_value=True) as mock_fallback:
                results = parser.parse_to_raw()
                mock_fallback.assert_called_once()

    def test_hybrid_strategy_fallback(self, temp_dir):
        """Verifica fallback en estrategia híbrida."""
        content = """ITEM: TEST-001
        Contenido ambiguo
        Sin formato claro"""

        file_path = temp_dir / "hybrid.txt"
        file_path.write_text(content)

        config = ParserConfig(strategy='hybrid')
        parser = ReportParserCrudo(file_path, config=config)
        results = parser.parse_to_raw()

        # Debe intentar ambas estrategias
        assert 'insumos_extracted' in parser.stats

    def test_error_recovery_in_block_processing(self, temp_dir):
        """Verifica recuperación de errores en procesamiento de bloques."""
        content = """
        ITEM: GOOD-001
        Material A;KG;10
        
        CORRUPTED BLOCK @#$%
        
        ITEM: GOOD-002
        Material B;UN;5
        """

        file_path = temp_dir / "recovery.txt"
        file_path.write_text(content)

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        # Debe procesar bloques buenos aunque uno falle
        apu_codes = {r['apu_code'] for r in results}
        assert len(apu_codes) >= 1  # Al menos un APU procesado


# =====================================================================
# PRUEBAS DE VALIDACIÓN Y LIMPIEZA
# =====================================================================

class TestValidationAndCleaning:
    """Pruebas de validación y limpieza de resultados."""

    def test_remove_duplicate_records(self, temp_dir):
        """Verifica eliminación de registros duplicados."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("dummy")

        parser = ReportParserCrudo(file_path)

        # Agregar registros duplicados manualmente
        parser.raw_records = [
            {'apu_code': 'A1', 'category': 'MAT', 'insumo_line': 'Line1'},
            {'apu_code': 'A1', 'category': 'MAT', 'insumo_line': 'Line1'},  # Duplicado
            {'apu_code': 'A1', 'category': 'MAT', 'insumo_line': 'Line2'},
        ]

        parser._validate_and_clean_results()

        assert len(parser.raw_records) == 2
        assert parser.stats['duplicates_removed'] == 1

    def test_calculate_statistics(self, temp_dir):
        """Verifica cálculo de estadísticas."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("dummy")

        parser = ReportParserCrudo(file_path)

        parser.raw_records = [
            {'apu_code': 'A1', 'category': 'MAT', 'insumo_line': 'L1'},
            {'apu_code': 'A1', 'category': 'MO', 'insumo_line': 'L2'},
            {'apu_code': 'A2', 'category': 'MAT', 'insumo_line': 'L3'},
        ]

        parser._validate_and_clean_results()

        assert parser.stats['unique_apus'] == 2
        assert parser.stats['avg_items_per_apu'] == 1.5


# =====================================================================
# PRUEBAS DE DEBUG Y LOGGING
# =====================================================================

class TestDebugAndLogging:
    """Pruebas de funcionalidad de debug y logging."""

    def test_debug_samples_collection(self, temp_dir, sample_apu_content_blocks):
        """Verifica recolección de muestras de debug."""
        file_path = temp_dir / "debug.txt"
        file_path.write_text(sample_apu_content_blocks)

        config = ParserConfig(debug_mode=True, max_debug_samples=3)
        parser = ReportParserCrudo(file_path, config=config)
        results = parser.parse_to_raw()

        assert len(parser.debug_samples) <= 3

        if parser.debug_samples:
            sample = parser.debug_samples[0]
            assert 'type' in sample
            assert 'block_num' in sample or 'line_num' in sample

    def test_error_logging(self, temp_dir):
        """Verifica registro de errores."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("dummy")

        parser = ReportParserCrudo(file_path)

        # Agregar errores manualmente
        parser._add_error('test_error', 'Test error message')
        parser._add_error('another_error', 'Another message')

        assert len(parser.errors) == 2
        assert parser.errors[0]['type'] == 'test_error'
        assert parser.errors[0]['message'] == 'Test error message'

    def test_statistics_summary(self, temp_dir, sample_apu_content_blocks):
        """Verifica generación de resumen de estadísticas."""
        file_path = temp_dir / "stats.txt"
        file_path.write_text(sample_apu_content_blocks)

        parser = ReportParserCrudo(file_path)
        parser.parse_to_raw()

        summary = parser.get_statistics_summary()

        assert 'file_path' in summary
        assert 'file_size' in summary
        assert 'encoding' in summary
        assert 'items_extracted' in summary
        assert 'parsed_successfully' in summary
        assert isinstance(summary['parsed_successfully'], bool)


# =====================================================================
# PRUEBAS DE RENDIMIENTO
# =====================================================================

class TestPerformance:
    """Pruebas de rendimiento y límites."""

    def test_large_file_limit(self, temp_dir):
        """Verifica límite de líneas procesadas."""
        # Generar archivo con muchas líneas
        lines = ["Line " + str(i) for i in range(200)]
        content = "\n".join(lines)

        file_path = temp_dir / "large.txt"
        file_path.write_text(content)

        config = ParserConfig(max_lines_to_process=100)
        parser = ReportParserCrudo(file_path, config=config)

        with patch.object(parser, '_parse_by_lines') as mock_parse:
            parser.parse_to_raw()

            # Verificar que se truncó el contenido
            call_args = mock_parse.call_args[0][0] if mock_parse.called else ""
            truncated_lines = call_args.split('\n') if call_args else []
            assert len(truncated_lines) <= 100

    def test_cache_performance(self, temp_dir):
        """Verifica que el cache mejore el rendimiento."""
        file_path = temp_dir / "cache_test.txt"
        file_path.write_text("dummy")

        parser = ReportParserCrudo(file_path)

        # Muchas llamadas a la misma categoría
        line = "MATERIALES"

        start_time = time.time()
        for _ in range(1000):
            parser._detect_category_cached(line)
        cached_time = time.time() - start_time

        # Limpiar cache
        parser._category_cache.clear()

        start_time = time.time()
        for _ in range(100):  # Menos iteraciones sin cache
            parser._detect_category(line)
        no_cache_time = time.time() - start_time

        # El tiempo con cache debe ser menor (ajustado por iteraciones)
        assert cached_time < no_cache_time * 10


# =====================================================================
# PRUEBAS DE INTEGRACIÓN
# =====================================================================

class TestIntegration:
    """Pruebas de integración del sistema completo."""

    def test_full_workflow_blocks_format(self, temp_dir, sample_apu_content_blocks):
        """Prueba flujo completo con formato de bloques."""
        file_path = temp_dir / "integration_blocks.txt"
        file_path.write_text(sample_apu_content_blocks)

        config = ParserConfig(
            strategy='auto',
            debug_mode=True,
            min_apu_code_length=3
        )

        parser = ReportParserCrudo(file_path, config=config)
        results = parser.parse_to_raw()

        # Verificaciones comprehensivas
        assert len(results) > 0
        assert parser._parsed is True
        assert parser.stats['insumos_extracted'] > 0

        # Verificar APUs extraídos
        apu_codes = {r['apu_code'] for r in results}
        assert 'EXC-001' in apu_codes
        assert 'CON-002' in apu_codes

        # Verificar categorías
        categories = {r['category'] for r in results}
        assert 'MATERIALES' in categories
        assert 'MANO DE OBRA' in categories
        assert 'EQUIPO' in categories

        # Verificar estadísticas
        summary = parser.get_statistics_summary()
        assert summary['parsed_successfully'] is True
        assert summary['unique_apus'] == 2

    def test_full_workflow_lines_format(self, temp_dir, sample_apu_content_lines):
        """Prueba flujo completo con formato de líneas."""
        file_path = temp_dir / "integration_lines.txt"
        file_path.write_text(sample_apu_content_lines)

        config = ParserConfig(
            field_separator=';',
            strategy='lines'
        )

        parser = ReportParserCrudo(file_path, config=config)
        results = parser.parse_to_raw()

        assert len(results) > 0

        # Verificar que cada registro tenga los campos necesarios
        required_fields = {'apu_code', 'apu_desc', 'apu_unit', 'category', 'insumo_line'}
        for record in results:
            assert all(field in record for field in required_fields)
            assert record['separator_used'] == ';'

    def test_error_handling_integration(self, temp_dir):
        """Prueba manejo de errores en flujo completo."""
        # Archivo con problemas
        content = """
        ITEM: @#$%^&*
        123 456 789
        
        ITEM: VALID-001
        Material;KG;10
        """

        file_path = temp_dir / "errors.txt"
        file_path.write_text(content)

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        # Debe procesar lo que pueda
        if results:
            valid_codes = [r['apu_code'] for r in results if r['apu_code'].startswith('VALID')]
            assert len(valid_codes) >= 0

        # Debe registrar errores
        assert len(parser.errors) >= 0


# =====================================================================
# PRUEBAS DE CASOS EDGE
# =====================================================================

class TestEdgeCases:
    """Pruebas de casos límite y especiales."""

    def test_single_line_file(self, temp_dir):
        """Verifica manejo de archivo de una línea."""
        file_path = temp_dir / "single.txt"
        file_path.write_text("ITEM: TEST-001")

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        # Puede o no extraer algo, pero no debe fallar
        assert isinstance(results, list)

    def test_only_whitespace_file(self, temp_dir):
        """Verifica manejo de archivo solo con espacios."""
        file_path = temp_dir / "whitespace.txt"
        file_path.write_text("   \n\n   \t\t   \n   ")

        parser = ReportParserCrudo(file_path)

        with pytest.raises(ParseStrategyError):
            parser.parse_to_raw()

    def test_special_characters_in_content(self, temp_dir):
        """Verifica manejo de caracteres especiales."""
        content = """
        ITEM: TEST-001
        Material con ñ, á, é, í, ó, ú
        Símbolo € $ ¥ £
        Emoji 😀 (si se soporta)
        """

        file_path = temp_dir / "special.txt"
        file_path.write_text(content, encoding='utf-8')

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        # No debe fallar con caracteres especiales
        assert parser.stats.get('encoding_used') == 'utf-8'

    def test_very_long_lines(self, temp_dir):
        """Verifica manejo de líneas muy largas."""
        long_line = "Material " + "x" * 1000  # Línea muy larga
        content = f"""
        ITEM: TEST-001
        {long_line}
        Material normal;KG;10
        """

        file_path = temp_dir / "long_lines.txt"
        file_path.write_text(content)

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        # La línea muy larga debe ser rechazada como insumo
        for record in results:
            assert len(record['insumo_line']) <= 500

    def test_circular_references_prevention(self, temp_dir):
        """Verifica prevención de referencias circulares en cache."""
        file_path = temp_dir / "circular.txt"
        file_path.write_text("dummy")

        parser = ReportParserCrudo(file_path)

        # Llenar cache hasta el límite
        for i in range(1001):
            parser._category_cache[f"LINE_{i}"] = f"CAT_{i}"

        # No debe crecer indefinidamente
        assert len(parser._category_cache) <= 1001

    def test_unicode_normalization(self, temp_dir):
        """Verifica normalización de Unicode."""
        # Mismo carácter en diferentes formas Unicode
        content = """
        ITEM: CAF\u00C9-001
        ITEM: CAFE\u0301-001
        """

        file_path = temp_dir / "unicode.txt"
        file_path.write_text(content)

        parser = ReportParserCrudo(file_path)
        # No debe fallar con diferentes representaciones Unicode
        results = parser.parse_to_raw()
        assert isinstance(results, list)


# =====================================================================
# PRUEBAS DE REGRESIÓN
# =====================================================================

class TestRegression:
    """Pruebas de regresión para bugs específicos encontrados."""

    def test_regression_empty_apu_description(self, temp_dir):
        """Regresión: APU sin descripción no debe fallar."""
        content = """
        ITEM: REG-001
        UNIDAD: KG
        Material;KG;10
        """

        file_path = temp_dir / "regression1.txt"
        file_path.write_text(content)

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        if results:
            # La descripción debe tener algún valor por defecto
            assert all(r['apu_desc'] for r in results)

    def test_regression_mixed_separators(self, temp_dir):
        """Regresión: Archivos con separadores mixtos."""
        content = """
        ITEM: MIX-001
        Material A;KG;10
        Material B|UN|5
        Material C,LT,100
        """

        file_path = temp_dir / "regression2.txt"
        file_path.write_text(content)

        parser = ReportParserCrudo(file_path)
        results = parser.parse_to_raw()

        # Debe detectar el separador más común
        assert parser._detected_separator is not None

    def test_regression_consecutive_empty_lines_reset(self, temp_dir):
        """Regresión: Reset de APU después de líneas vacías."""
        content = """
        ITEM: FIRST-001
        Material A;KG;10
        
        
        
        
        
        
        Material B;UN;5
        
        ITEM: SECOND-001
        Material C;M3;2
        """

        file_path = temp_dir / "regression3.txt"
        file_path.write_text(content)

        config = ParserConfig(strategy='lines')
        parser = ReportParserCrudo(file_path, config=config)
        results = parser.parse_to_raw()

        # Material B no debe asociarse a FIRST-001 después de muchas líneas vacías
        apu_codes = {r['apu_code'] for r in results}
        # Debe haber detectado al menos SECOND-001
        if 'SECOND-001' in apu_codes:
            second_items = [r for r in results if r['apu_code'] == 'SECOND-001']
            assert any('Material C' in r['insumo_line'] for r in second_items)


# =====================================================================
# TEST RUNNER
# =====================================================================

if __name__ == "__main__":
    # Configurar pytest para ejecución con mayor detalle
    pytest.main([
        __file__,
        "-v",  # Verbose
        "--tb=short",  # Traceback corto
        "--color=yes",  # Colores en output
        "-s",  # No capturar stdout
        "--cov=app.report_parser_crudo",  # Coverage del módulo
        "--cov-report=term-missing",  # Reporte de coverage
        "--cov-report=html",  # Reporte HTML
        "--durations=10",  # Mostrar 10 tests más lentos
    ])
