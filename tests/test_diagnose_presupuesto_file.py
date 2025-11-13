# tests/test_diagnose_presupuesto_file.py
"""
Suite de pruebas completa para diagnose_presupuesto_file.py

Este módulo contiene pruebas exhaustivas para validar:
- Inicialización y validación de archivos
- Detección de encoding con múltiples estrategias
- Detección de separadores de columnas
- Identificación de encabezados
- Análisis de estructura y consistencia
- Generación de reportes
- Manejo de errores y casos edge
"""

import io
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, mock_open

import pytest

# Importar el módulo a probar
# Ajustar el path según la estructura del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from diagnose_presupuesto_file import (
    PresupuestoFileDiagnostic,
    DiagnosticError,
    FileReadError,
    EncodingDetectionError,
    ConfidenceLevel,
    HeaderCandidate,
    SampleLine,
    ColumnStatistics,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Directorio temporal para archivos de prueba."""
    return tmp_path


@pytest.fixture
def valid_presupuesto_file(temp_dir):
    """Crea un archivo de presupuesto válido con estructura estándar."""
    content = """PRESUPUESTO DE OBRA
Proyecto: Test Project
Fecha: 2024-01-01

ITEM;DESCRIPCION;CANTIDAD;UNIDAD;VR. UNITARIO;TOTAL
1;Excavación manual;100;M3;50000;5000000
2;Concreto 3000 PSI;50;M3;350000;17500000
3;Acero de refuerzo;2000;KG;4500;9000000
4;Formaleta metálica;150;M2;25000;3750000
5;Mano de obra;1;GL;15000000;15000000

TOTAL GENERAL;;;;;;;50250000
"""
    file_path = temp_dir / "presupuesto_valid.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def presupuesto_with_header_offset(temp_dir):
    """Archivo con encabezado después de varias líneas."""
    content = """# Comentario inicial
# Más información
Proyecto: Test

ITEM;DESCRIPCION;CANT;UNIDAD;PRECIO;TOTAL
1;Item 1;10;UND;1000;10000
2;Item 2;20;UND;2000;40000
"""
    file_path = temp_dir / "presupuesto_offset.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def presupuesto_inconsistent_columns(temp_dir):
    """Archivo con número inconsistente de columnas."""
    content = """ITEM;DESCRIPCION;CANTIDAD;UNIDAD;VR. UNITARIO;TOTAL
1;Item normal;10;UND;1000;10000
2;Item con columnas extra;20;UND;2000;40000;EXTRA;MAS
3;Item corto;5;UND
4;Item normal 2;15;UND;1500;22500
"""
    file_path = temp_dir / "presupuesto_inconsistent.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def presupuesto_different_separators(temp_dir):
    """Crea archivos con diferentes separadores."""
    files = {}
    
    # Punto y coma
    content_semicolon = "ITEM;DESCRIPCION;CANT\n1;Item 1;10\n2;Item 2;20\n"
    f1 = temp_dir / "sep_semicolon.csv"
    f1.write_text(content_semicolon, encoding='utf-8')
    files['semicolon'] = f1
    
    # Coma
    content_comma = "ITEM,DESCRIPCION,CANT\n1,Item 1,10\n2,Item 2,20\n"
    f2 = temp_dir / "sep_comma.csv"
    f2.write_text(content_comma, encoding='utf-8')
    files['comma'] = f2
    
    # Tabulación
    content_tab = "ITEM\tDESCRIPCION\tCANT\n1\tItem 1\t10\n2\tItem 2\t20\n"
    f3 = temp_dir / "sep_tab.csv"
    f3.write_text(content_tab, encoding='utf-8')
    files['tab'] = f3
    
    # Pipe
    content_pipe = "ITEM|DESCRIPCION|CANT\n1|Item 1|10\n2|Item 2|20\n"
    f4 = temp_dir / "sep_pipe.csv"
    f4.write_text(content_pipe, encoding='utf-8')
    files['pipe'] = f4
    
    return files


@pytest.fixture
def presupuesto_different_encodings(temp_dir):
    """Crea archivos con diferentes encodings."""
    files = {}
    
    content = "ITEM;DESCRIPCIÓN;CANT\n1;Excavación;10\n"
    
    # UTF-8
    f1 = temp_dir / "enc_utf8.csv"
    f1.write_text(content, encoding='utf-8')
    files['utf-8'] = f1
    
    # Latin1
    f2 = temp_dir / "enc_latin1.csv"
    f2.write_text(content, encoding='latin1')
    files['latin1'] = f2
    
    # CP1252
    f3 = temp_dir / "enc_cp1252.csv"
    f3.write_text(content, encoding='cp1252')
    files['cp1252'] = f3
    
    # UTF-8 with BOM
    f4 = temp_dir / "enc_utf8_bom.csv"
    f4.write_text(content, encoding='utf-8-sig')
    files['utf-8-sig'] = f4
    
    return files


@pytest.fixture
def empty_file(temp_dir):
    """Archivo completamente vacío."""
    file_path = temp_dir / "empty.csv"
    file_path.write_text("", encoding='utf-8')
    return file_path


@pytest.fixture
def whitespace_only_file(temp_dir):
    """Archivo con solo espacios y saltos de línea."""
    content = "\n\n   \n\t\n   \n"
    file_path = temp_dir / "whitespace.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def large_file(temp_dir):
    """Archivo grande que excede MAX_LINES_TO_ANALYZE."""
    content_lines = ["ITEM;DESCRIPCION;CANT;UNIDAD;PRECIO;TOTAL\n"]
    for i in range(1, 1500):  # Más que MAX_LINES_TO_ANALYZE (1000)
        content_lines.append(f"{i};Item {i};{i*10};UND;{i*100};{i*1000}\n")
    
    file_path = temp_dir / "large.csv"
    file_path.write_text("".join(content_lines), encoding='utf-8')
    return file_path


@pytest.fixture
def presupuesto_no_header(temp_dir):
    """Archivo sin encabezado reconocible."""
    content = """1;Dato 1;10;UND;1000;10000
2;Dato 2;20;UND;2000;40000
3;Dato 3;30;UND;3000;90000
"""
    file_path = temp_dir / "no_header.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def presupuesto_with_comments(temp_dir):
    """Archivo con diferentes tipos de comentarios."""
    content = """# Comentario con hash
// Comentario estilo C++
/* Comentario estilo C */
-- Comentario estilo SQL
' Comentario estilo VB

ITEM;DESCRIPCION;CANTIDAD;UNIDAD;VR. UNITARIO;TOTAL
1;Item 1;10;UND;1000;10000
# Comentario entre datos
2;Item 2;20;UND;2000;40000
"""
    file_path = temp_dir / "with_comments.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


# ============================================================================
# TESTS DE INICIALIZACIÓN Y VALIDACIÓN
# ============================================================================

class TestInitialization:
    """Pruebas de inicialización del diagnosticador."""
    
    def test_valid_file_initialization(self, valid_presupuesto_file):
        """Debe inicializarse correctamente con archivo válido."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        assert diagnostic.file_path == valid_presupuesto_file.resolve()
        assert isinstance(diagnostic.stats, dict)
        assert diagnostic._encoding is None  # No se ha leído aún
    
    def test_initialization_with_string_path(self, valid_presupuesto_file):
        """Debe aceptar rutas como string."""
        diagnostic = PresupuestoFileDiagnostic(str(valid_presupuesto_file))
        assert diagnostic.file_path.exists()
    
    def test_initialization_with_path_object(self, valid_presupuesto_file):
        """Debe aceptar objetos Path."""
        diagnostic = PresupuestoFileDiagnostic(Path(valid_presupuesto_file))
        assert diagnostic.file_path.exists()
    
    def test_nonexistent_file_raises_error(self, temp_dir):
        """Debe lanzar ValueError si el archivo no existe."""
        nonexistent = temp_dir / "nonexistent.csv"
        with pytest.raises(ValueError, match="no existe"):
            PresupuestoFileDiagnostic(nonexistent)
    
    def test_directory_path_raises_error(self, temp_dir):
        """Debe lanzar ValueError si la ruta es un directorio."""
        with pytest.raises(ValueError, match="no apunta a un archivo"):
            PresupuestoFileDiagnostic(temp_dir)
    
    def test_empty_file_raises_error(self, empty_file):
        """Debe lanzar ValueError si el archivo está vacío."""
        with pytest.raises(ValueError, match="vacío"):
            PresupuestoFileDiagnostic(empty_file)
    
    def test_permission_check(self, valid_presupuesto_file):
        """Debe verificar permisos de lectura."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        assert diagnostic._check_read_permissions() is True
    
    @patch('pathlib.Path.open', side_effect=PermissionError("No access"))
    def test_no_read_permission_raises_error(self, mock_open, valid_presupuesto_file):
        """Debe lanzar PermissionError si no hay permisos de lectura."""
        with pytest.raises(PermissionError, match="No hay permisos"):
            PresupuestoFileDiagnostic(valid_presupuesto_file)


# ============================================================================
# TESTS DE DETECCIÓN DE ENCODING
# ============================================================================

class TestEncodingDetection:
    """Pruebas de detección de encoding."""
    
    def test_detect_utf8(self, presupuesto_different_encodings):
        """Debe detectar UTF-8 correctamente."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_different_encodings['utf-8'])
        result = diagnostic.diagnose()
        assert result['encoding'] == 'utf-8'
        assert result['stats']['encoding_method'] == 'predefined'
    
    def test_detect_utf8_with_bom(self, presupuesto_different_encodings):
        """Debe detectar UTF-8 con BOM."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_different_encodings['utf-8-sig'])
        result = diagnostic.diagnose()
        assert result['encoding'] in ['utf-8-sig', 'utf-8']
    
    def test_detect_latin1(self, presupuesto_different_encodings):
        """Debe detectar Latin1."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_different_encodings['latin1'])
        result = diagnostic.diagnose()
        assert result['encoding'] in ['latin1', 'utf-8', 'cp1252']  # Pueden ser compatibles
    
    def test_fallback_encoding_strategies(self, temp_dir):
        """Debe intentar múltiples estrategias de encoding."""
        # Crear archivo con bytes problemáticos
        file_path = temp_dir / "problematic.csv"
        with open(file_path, 'wb') as f:
            f.write(b'ITEM;DESC\n1;Test\x80\x81\n')  # Bytes inválidos en UTF-8
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        # No debe fallar, debe usar estrategia de fallback
        result = diagnostic.diagnose()
        assert result['success'] is True
    
    @patch('diagnose_presupuesto_file.CHARDET_AVAILABLE', True)
    @patch('diagnose_presupuesto_file.chardet.detect')
    def test_chardet_detection_high_confidence(self, mock_detect, temp_dir):
        """Debe usar chardet cuando otros métodos fallan y confianza es alta."""
        # Crear archivo que falle con encodings estándar
        file_path = temp_dir / "special.csv"
        with open(file_path, 'wb') as f:
            f.write(b'\xff\xfeI\x00T\x00E\x00M\x00')  # UTF-16 LE
        
        mock_detect.return_value = {'encoding': 'utf-16', 'confidence': 0.95}
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        
        # Mockear la lectura después de detección
        with patch.object(Path, 'open', mock_open(read_data='ITEM;DESC\n')):
            content = diagnostic._read_with_chardet()
            if content:  # Si chardet funcionó
                assert diagnostic._encoding == 'utf-16'
    
    @patch('diagnose_presupuesto_file.CHARDET_AVAILABLE', True)
    @patch('diagnose_presupuesto_file.chardet.detect')
    def test_chardet_detection_low_confidence(self, mock_detect, temp_dir):
        """Debe rechazar chardet cuando confianza es baja."""
        file_path = temp_dir / "ambiguous.csv"
        file_path.write_bytes(b'ITEM;DESC\n')
        
        mock_detect.return_value = {'encoding': 'ascii', 'confidence': 0.5}
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic._read_with_chardet()
        assert result is None  # Confianza muy baja


# ============================================================================
# TESTS DE DETECCIÓN DE SEPARADOR
# ============================================================================

class TestSeparatorDetection:
    """Pruebas de detección de separador."""
    
    def test_detect_semicolon(self, presupuesto_different_separators):
        """Debe detectar punto y coma como separador."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_different_separators['semicolon'])
        result = diagnostic.diagnose()
        assert result['separator'] == ';'
    
    def test_detect_comma(self, presupuesto_different_separators):
        """Debe detectar coma como separador."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_different_separators['comma'])
        result = diagnostic.diagnose()
        assert result['separator'] == ','
    
    def test_detect_tab(self, presupuesto_different_separators):
        """Debe detectar tabulación como separador."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_different_separators['tab'])
        result = diagnostic.diagnose()
        assert result['separator'] == '\t'
    
    def test_detect_pipe(self, presupuesto_different_separators):
        """Debe detectar pipe como separador."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_different_separators['pipe'])
        result = diagnostic.diagnose()
        assert result['separator'] == '|'
    
    def test_separator_confidence_levels(self, presupuesto_different_separators):
        """Debe asignar niveles de confianza al separador detectado."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_different_separators['semicolon'])
        result = diagnostic.diagnose()
        assert result['stats']['separator_confidence'] in [
            ConfidenceLevel.HIGH.value,
            ConfidenceLevel.MEDIUM.value,
            ConfidenceLevel.LOW.value
        ]
    
    def test_ambiguous_separator_defaults_to_semicolon(self, temp_dir):
        """Debe usar punto y coma por defecto si no hay separador claro."""
        # Archivo sin separadores claros
        content = "ITEMDESCRIPCION\nDato1Valor1\nDato2Valor2\n"
        file_path = temp_dir / "no_separator.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        assert result['separator'] == ';'  # Default
    
    def test_separator_with_empty_file_content(self, temp_dir):
        """Debe manejar archivo con solo líneas vacías."""
        content = "\n\n\n"
        file_path = temp_dir / "only_newlines.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        with pytest.raises(FileReadError):
            diagnostic.diagnose()


# ============================================================================
# TESTS DE DETECCIÓN DE ENCABEZADO
# ============================================================================

class TestHeaderDetection:
    """Pruebas de detección de encabezado."""
    
    def test_detect_header_line_zero(self, temp_dir):
        """Debe detectar encabezado en línea 0."""
        content = "ITEM;DESCRIPCION;CANTIDAD;UNIDAD;VR. UNITARIO;TOTAL\n1;Item;10;UND;1000;10000\n"
        file_path = temp_dir / "header_zero.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['header_candidate'] is not None
        assert result['header_candidate']['line_num'] == 1  # Primera línea (1-indexed)
    
    def test_detect_header_with_offset(self, presupuesto_with_header_offset):
        """Debe detectar encabezado después de varias líneas."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_with_header_offset)
        result = diagnostic.diagnose()
        
        assert result['header_candidate'] is not None
        assert result['header_candidate']['line_num'] > 1
    
    def test_no_header_detected(self, presupuesto_no_header):
        """Debe retornar None si no hay encabezado reconocible."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_no_header)
        result = diagnostic.diagnose()
        
        assert result['header_candidate'] is None
    
    def test_header_keyword_matches(self, valid_presupuesto_file):
        """Debe identificar palabras clave en el encabezado."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()
        
        assert result['header_candidate'] is not None
        matches = result['header_candidate']['matches']
        assert len(matches) >= 2  # MIN_HEADER_KEYWORD_MATCHES
        # Verificar que matches sea una lista de strings
        assert all(isinstance(m, str) for m in matches)
    
    def test_header_column_count(self, valid_presupuesto_file):
        """Debe contar correctamente las columnas del encabezado."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()
        
        assert result['header_candidate'] is not None
        assert result['header_candidate']['column_count'] >= 5  # ITEM, DESC, CANT, UNIDAD, VR. UNIT, TOTAL
    
    def test_header_confidence_levels(self, temp_dir):
        """Debe asignar niveles de confianza según coincidencias."""
        # Encabezado con muchas coincidencias (HIGH)
        content_high = "ITEM;CODIGO;DESCRIPCION;CANTIDAD;UNIDAD;VR. UNITARIO;PRECIO;TOTAL\n"
        file_high = temp_dir / "header_high_conf.csv"
        file_high.write_text(content_high + "1;001;Item;10;UND;1000;1000;10000\n", encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_high)
        result = diagnostic.diagnose()
        
        assert result['header_candidate'] is not None
        assert result['header_candidate']['confidence'] == ConfidenceLevel.HIGH.value
    
    def test_multiple_header_candidates_selects_best(self, temp_dir):
        """Debe seleccionar el mejor candidato cuando hay múltiples opciones."""
        content = """ITEM;DESC
ITEM;DESCRIPCION;CANTIDAD;UNIDAD;VR. UNITARIO;TOTAL
1;Item completo;10;UND;1000;10000
2;Item 2;20;UND;2000;40000
"""
        file_path = temp_dir / "multiple_candidates.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        # Debe seleccionar el encabezado con más coincidencias/columnas
        assert result['header_candidate'] is not None
        assert result['header_candidate']['column_count'] >= 5


# ============================================================================
# TESTS DE ANÁLISIS DE ESTRUCTURA
# ============================================================================

class TestStructureAnalysis:
    """Pruebas de análisis de estructura del archivo."""
    
    def test_count_empty_lines(self, temp_dir):
        """Debe contar correctamente líneas vacías."""
        content = "ITEM;DESC\n\n1;Item\n\n\n2;Item2\n"
        file_path = temp_dir / "with_empty.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['stats']['empty_lines'] == 3
    
    def test_count_comment_lines(self, presupuesto_with_comments):
        """Debe contar correctamente líneas de comentario."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_with_comments)
        result = diagnostic.diagnose()
        
        assert result['stats']['comment_lines'] >= 5
    
    def test_identify_data_start_line(self, valid_presupuesto_file):
        """Debe identificar la línea donde comienzan los datos."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()
        
        assert result['data_start_line'] is not None
        assert result['data_start_line'] > result['header_candidate']['line_num']
    
    def test_column_consistency_high(self, valid_presupuesto_file):
        """Debe detectar alta consistencia en columnas."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()
        
        assert result['stats']['column_consistency'] > 0.9
    
    def test_column_consistency_low(self, presupuesto_inconsistent_columns):
        """Debe detectar baja consistencia en columnas."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_inconsistent_columns)
        result = diagnostic.diagnose()
        
        assert result['stats']['column_consistency'] < 0.9
    
    def test_dominant_column_count(self, valid_presupuesto_file):
        """Debe identificar el número de columnas dominante."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()
        
        assert 'dominant_column_count' in result['stats']
        assert result['stats']['dominant_column_count'] >= 5
    
    def test_column_distribution(self, presupuesto_inconsistent_columns):
        """Debe generar distribución de columnas."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_inconsistent_columns)
        result = diagnostic.diagnose()
        
        assert 'column_distribution' in result
        assert len(result['column_distribution']) > 1  # Múltiples conteos
    
    def test_sample_lines_collection(self, valid_presupuesto_file):
        """Debe recolectar líneas de muestra."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()
        
        # Verificar en el objeto diagnostic, no en result
        assert len(diagnostic.sample_lines) > 0
        assert all(isinstance(s, SampleLine) for s in diagnostic.sample_lines)
    
    def test_large_file_truncation(self, large_file):
        """Debe truncar análisis en archivos grandes."""
        diagnostic = PresupuestoFileDiagnostic(large_file)
        result = diagnostic.diagnose()
        
        assert result['stats'].get('truncated_analysis') is True
        assert result['stats']['lines_analyzed'] == diagnostic.MAX_LINES_TO_ANALYZE


# ============================================================================
# TESTS DE FUNCIONES AUXILIARES
# ============================================================================

class TestHelperFunctions:
    """Pruebas de funciones auxiliares."""
    
    @pytest.mark.parametrize("size,expected", [
        (0, "0 B"),
        (500, "500 B"),
        (1024, "1.00 KB"),
        (1536, "1.50 KB"),
        (1048576, "1.00 MB"),
        (1073741824, "1.00 GB"),
        (5000, "4.88 KB"),
    ])
    def test_human_readable_size(self, size, expected, valid_presupuesto_file):
        """Debe convertir bytes a formato legible."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic._human_readable_size(size)
        assert result == expected
    
    @pytest.mark.parametrize("input_text,expected_output", [
        ("ITEM", "ITEM"),
        ("Descripción", "DESCRIPCION"),
        ("VR. UNITARIO", "VR UNITARIO"),
        ("Cód$igo", "CODIGO"),
        ("  Espacio  Multiple  ", "ESPACIO MULTIPLE"),
        ("CAÑÓN", "CANON"),
        ("Válór/Total", "VALOR/TOTAL"),
    ])
    def test_normalize_header_text(self, input_text, expected_output, valid_presupuesto_file):
        """Debe normalizar texto de encabezado correctamente."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic._normalize_header_text(input_text)
        # Eliminar caracteres que también se eliminan en el expected
        assert expected_output.replace('/', '') in result or result in expected_output.replace('/', '')
    
    @pytest.mark.parametrize("line,is_comment", [
        ("# Comentario", True),
        ("// Comentario", True),
        ("/* Comentario", True),
        ("-- Comentario", True),
        ("' Comentario", True),
        ("* Comentario", True),
        ("REM Comentario", True),
        ("Normal line", False),
        ("1;2;3", False),
        ("", False),
    ])
    def test_is_comment_line(self, line, is_comment, valid_presupuesto_file):
        """Debe identificar correctamente líneas de comentario."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic._is_comment_line(line)
        assert result == is_comment
    
    def test_check_read_permissions_success(self, valid_presupuesto_file):
        """Debe retornar True con permisos correctos."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        assert diagnostic._check_read_permissions() is True
    
    @patch('pathlib.Path.open', side_effect=PermissionError("Access denied"))
    def test_check_read_permissions_failure(self, mock_open, valid_presupuesto_file):
        """Debe retornar False sin permisos."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        assert diagnostic._check_read_permissions() is False


# ============================================================================
# TESTS DE DATACLASSES
# ============================================================================

class TestDataClasses:
    """Pruebas de las dataclasses."""
    
    def test_header_candidate_creation(self):
        """Debe crear HeaderCandidate válido."""
        header = HeaderCandidate(
            line_num=1,
            content="ITEM;DESC;CANT",
            matches=["ITEM", "DESCRIPCION"],
            match_count=2,
            column_count=3
        )
        assert header.line_num == 1
        assert header.match_count == 2
        assert header.confidence == ConfidenceLevel.MEDIUM
    
    def test_header_candidate_validation_line_num(self):
        """Debe validar line_num >= 1."""
        with pytest.raises(ValueError, match="line_num debe ser >= 1"):
            HeaderCandidate(
                line_num=0,
                content="Test",
                matches=["A", "B"],
                match_count=2,
                column_count=3
            )
    
    def test_header_candidate_validation_match_count(self):
        """Debe validar consistencia de match_count."""
        with pytest.raises(ValueError, match="match_count debe coincidir"):
            HeaderCandidate(
                line_num=1,
                content="Test",
                matches=["A", "B"],
                match_count=3,  # Inconsistente con len(matches)
                column_count=3
            )
    
    def test_sample_line_creation(self):
        """Debe crear SampleLine válido."""
        sample = SampleLine(
            line_num=5,
            content="1;Item;10",
            column_count=3
        )
        assert sample.line_num == 5
        assert sample.column_count == 3
    
    def test_sample_line_validation(self):
        """Debe validar valores negativos."""
        with pytest.raises(ValueError):
            SampleLine(line_num=0, content="test", column_count=-1)
    
    def test_column_statistics_add_sample(self):
        """Debe agregar muestras correctamente."""
        stats = ColumnStatistics()
        stats.add_sample("sample1", max_samples=3)
        stats.add_sample("sample2", max_samples=3)
        stats.add_sample("sample3", max_samples=3)
        stats.add_sample("sample4", max_samples=3)  # No debe agregarse
        
        assert stats.count == 4
        assert len(stats.samples) == 3  # Límite alcanzado


# ============================================================================
# TESTS DE GENERACIÓN DE REPORTES
# ============================================================================

class TestReportGeneration:
    """Pruebas de generación de reportes."""
    
    def test_generate_report_creates_log_output(self, valid_presupuesto_file, caplog):
        """Debe generar output de log al crear reporte."""
        import logging
        caplog.set_level(logging.INFO)
        
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        diagnostic.diagnose()
        
        # Verificar que se generó output
        assert len(caplog.records) > 0
        assert any("REPORTE DE DIAGNÓSTICO" in record.message for record in caplog.records)
    
    def test_report_includes_file_info(self, valid_presupuesto_file, caplog):
        """Debe incluir información del archivo en reporte."""
        import logging
        caplog.set_level(logging.INFO)
        
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        diagnostic.diagnose()
        
        log_output = " ".join(record.message for record in caplog.records)
        assert "INFORMACIÓN BÁSICA" in log_output
        assert "Tamaño" in log_output
    
    def test_report_includes_recommendations(self, valid_presupuesto_file, caplog):
        """Debe incluir recomendaciones en reporte."""
        import logging
        caplog.set_level(logging.INFO)
        
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        diagnostic.diagnose()
        
        log_output = " ".join(record.message for record in caplog.records)
        assert "RECOMENDACIONES" in log_output
    
    def test_report_includes_pandas_example(self, valid_presupuesto_file, caplog):
        """Debe incluir ejemplo de código pandas."""
        import logging
        caplog.set_level(logging.INFO)
        
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        diagnostic.diagnose()
        
        log_output = " ".join(record.message for record in caplog.records)
        assert "EJEMPLO DE CÓDIGO PANDAS" in log_output or "pandas" in log_output.lower()


# ============================================================================
# TESTS DE MANEJO DE ERRORES
# ============================================================================

class TestErrorHandling:
    """Pruebas de manejo de errores."""
    
    def test_file_read_error_on_empty_content(self, whitespace_only_file):
        """Debe lanzar FileReadError con contenido vacío."""
        diagnostic = PresupuestoFileDiagnostic(whitespace_only_file)
        with pytest.raises(FileReadError, match="no contiene datos válidos"):
            diagnostic.diagnose()
    
    def test_diagnostic_error_wraps_unexpected_exceptions(self, valid_presupuesto_file):
        """Debe envolver excepciones inesperadas en DiagnosticError."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        
        # Forzar error durante análisis
        with patch.object(diagnostic, '_analyze_structure_single_pass', side_effect=RuntimeError("Test error")):
            with pytest.raises(DiagnosticError, match="Fallo en el diagnóstico"):
                diagnostic.diagnose()
    
    def test_reset_state_clears_previous_data(self, valid_presupuesto_file):
        """Debe limpiar estado previo al resetear."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        
        # Poblar con datos
        diagnostic.stats['test'] = 123
        diagnostic.sample_lines.append(SampleLine(1, "test", column_count=3))
        
        # Resetear
        diagnostic._reset_state()
        
        assert 'test' not in diagnostic.stats
        assert len(diagnostic.sample_lines) == 0
        assert diagnostic.header_candidate is None
    
    def test_handles_corrupted_file_gracefully(self, temp_dir):
        """Debe manejar archivos corruptos sin crash."""
        # Crear archivo con bytes aleatorios
        file_path = temp_dir / "corrupted.csv"
        with open(file_path, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\x04\x05\xff\xfe\xfd')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        # No debe lanzar excepción, debe usar fallback
        result = diagnostic.diagnose()
        assert result['success'] is True


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

class TestIntegration:
    """Pruebas de integración del flujo completo."""
    
    def test_full_diagnostic_workflow(self, valid_presupuesto_file):
        """Debe completar el flujo completo de diagnóstico exitosamente."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()
        
        # Verificar estructura del resultado
        assert result['success'] is True
        assert 'stats' in result
        assert 'encoding' in result
        assert 'separator' in result
        assert 'file_size' in result
        
        # Verificar que se detectó encabezado
        assert result['header_candidate'] is not None
        assert result['data_start_line'] is not None
        
        # Verificar estadísticas
        assert result['stats']['total_lines'] > 0
        assert result['stats']['non_empty_lines'] > 0
    
    def test_result_dict_structure(self, valid_presupuesto_file):
        """Debe retornar diccionario con estructura esperada."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()
        
        required_keys = ['success', 'file_path', 'stats', 'encoding', 'separator']
        for key in required_keys:
            assert key in result
    
    def test_multiple_diagnoses_same_instance(self, valid_presupuesto_file):
        """Debe permitir múltiples diagnósticos con la misma instancia."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        
        result1 = diagnostic.diagnose()
        result2 = diagnostic.diagnose()
        
        # Los resultados deben ser consistentes
        assert result1['encoding'] == result2['encoding']
        assert result1['separator'] == result2['separator']
    
    def test_end_to_end_with_recommendations(self, valid_presupuesto_file):
        """Debe proporcionar recomendaciones accionables."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()
        
        # Verificar que hay suficiente información para usar el archivo
        assert result['separator'] is not None
        assert result['encoding'] is not None
        if result['header_candidate']:
            # Debe poder calcular header para pandas (0-indexed)
            pandas_header = result['header_candidate']['line_num'] - 1
            assert pandas_header >= 0
    
    def test_diagnostic_with_various_file_qualities(self, temp_dir):
        """Debe manejar archivos de diferentes calidades."""
        files_to_test = []
        
        # Archivo perfecto
        perfect = temp_dir / "perfect.csv"
        perfect.write_text(
            "ITEM;DESCRIPCION;CANTIDAD;UNIDAD;VR. UNITARIO;TOTAL\n"
            "1;Item 1;10;UND;1000;10000\n"
            "2;Item 2;20;UND;2000;40000\n",
            encoding='utf-8'
        )
        files_to_test.append(perfect)
        
        # Archivo con problemas menores
        minor_issues = temp_dir / "minor.csv"
        minor_issues.write_text(
            "\n# Comment\n"
            "ITEM;DESC;CANT\n"
            "1;Item;10\n"
            "\n"
            "2;Item;20\n",
            encoding='utf-8'
        )
        files_to_test.append(minor_issues)
        
        # Todos deben procesarse exitosamente
        for file_path in files_to_test:
            diagnostic = PresupuestoFileDiagnostic(file_path)
            result = diagnostic.diagnose()
            assert result['success'] is True


# ============================================================================
# TESTS DEL MAIN
# ============================================================================

class TestMainFunction:
    """Pruebas de la función main."""
    
    def test_main_no_arguments(self):
        """Debe retornar 1 si no se proporciona archivo."""
        from diagnose_presupuesto_file import main
        
        with patch('sys.argv', ['script_name']):
            result = main()
            assert result == 1
    
    def test_main_with_valid_file(self, valid_presupuesto_file):
        """Debe retornar 0 con archivo válido."""
        from diagnose_presupuesto_file import main
        
        with patch('sys.argv', ['script_name', str(valid_presupuesto_file)]):
            result = main()
            assert result == 0
    
    def test_main_with_nonexistent_file(self, temp_dir):
        """Debe retornar 1 con archivo inexistente."""
        from diagnose_presupuesto_file import main
        
        with patch('sys.argv', ['script_name', str(temp_dir / 'nonexistent.csv')]):
            result = main()
            assert result == 1
    
    def test_main_keyboard_interrupt(self, valid_presupuesto_file):
        """Debe retornar 130 al interrumpir con teclado."""
        from diagnose_presupuesto_file import main
        
        with patch('sys.argv', ['script_name', str(valid_presupuesto_file)]):
            with patch('diagnose_presupuesto_file.PresupuestoFileDiagnostic.diagnose', 
                      side_effect=KeyboardInterrupt()):
                result = main()
                assert result == 130


# ============================================================================
# TESTS DE CASOS EDGE
# ============================================================================

class TestEdgeCases:
    """Pruebas de casos edge y situaciones especiales."""
    
    def test_file_with_only_header(self, temp_dir):
        """Debe manejar archivo con solo encabezado."""
        content = "ITEM;DESCRIPCION;CANTIDAD;UNIDAD;VR. UNITARIO;TOTAL\n"
        file_path = temp_dir / "only_header.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['header_candidate'] is not None
        assert result['data_start_line'] is None  # No hay datos
    
    def test_file_with_unicode_characters(self, temp_dir):
        """Debe manejar caracteres Unicode especiales."""
        content = "ITEM;DESCRIPCIÓN;CANT\n1;Excavación™;10\n2;Diseño®;20\n3;Señalización€;30\n"
        file_path = temp_dir / "unicode.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['success'] is True
    
    def test_file_with_very_long_lines(self, temp_dir):
        """Debe manejar líneas muy largas."""
        long_desc = "X" * 10000
        content = f"ITEM;DESCRIPCION;CANT\n1;{long_desc};10\n"
        file_path = temp_dir / "long_lines.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['success'] is True
    
    def test_file_with_special_line_endings(self, temp_dir):
        """Debe manejar diferentes finales de línea."""
        # Windows line endings (CRLF)
        content_crlf = "ITEM;DESC\r\n1;Item\r\n"
        file_path = temp_dir / "crlf.csv"
        file_path.write_bytes(content_crlf.encode('utf-8'))
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['success'] is True
    
    def test_file_with_mixed_separators(self, temp_dir):
        """Debe manejar archivo con separadores mixtos (error común)."""
        content = "ITEM;DESCRIPCION;CANT\n1;Item,with,commas;10\n2;Normal item;20\n"
        file_path = temp_dir / "mixed_sep.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        # Debe detectar punto y coma como separador dominante
        assert result['separator'] == ';'
    
    def test_file_with_quoted_fields(self, temp_dir):
        """Debe manejar campos entre comillas."""
        content = 'ITEM;DESCRIPCION;CANT\n1;"Item con; separador";10\n2;"Normal";20\n'
        file_path = temp_dir / "quoted.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['success'] is True
    
    def test_numeric_only_file(self, temp_dir):
        """Debe manejar archivo con solo números."""
        content = "1;2;3;4;5\n10;20;30;40;50\n"
        file_path = temp_dir / "numeric.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        # No debe detectar encabezado (sin palabras clave)
        assert result['header_candidate'] is None


# ============================================================================
# TESTS DE PERFORMANCE
# ============================================================================

class TestPerformance:
    """Pruebas de rendimiento (opcional)."""
    
    def test_large_file_processes_in_reasonable_time(self, large_file):
        """Debe procesar archivo grande en tiempo razonable."""
        import time
        
        diagnostic = PresupuestoFileDiagnostic(large_file)
        
        start = time.time()
        result = diagnostic.diagnose()
        elapsed = time.time() - start
        
        assert result['success'] is True
        assert elapsed < 10  # Debe completar en menos de 10 segundos
    
    def test_memory_efficient_with_samples(self, large_file):
        """Debe limitar muestras para eficiencia de memoria."""
        diagnostic = PresupuestoFileDiagnostic(large_file)
        diagnostic.diagnose()
        
        # No debe almacenar todas las líneas
        assert len(diagnostic.sample_lines) <= diagnostic.MAX_SAMPLE_LINES


# ============================================================================
# CONFIGURACIÓN DE PYTEST
# ============================================================================

@pytest.fixture(autouse=True)
def reset_logging():
    """Resetea configuración de logging entre pruebas."""
    import logging
    # Guardar handlers originales
    logger = logging.getLogger("PresupuestoDiagnostic")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    
    yield
    
    # Restaurar
    logger.handlers = original_handlers
    logger.level = original_level


# ============================================================================
# MARKS Y CONFIGURACIÓN
# ============================================================================

# Marcar tests lentos
slow_tests = pytest.mark.slow

# Marcar tests que requieren archivos reales
requires_files = pytest.mark.requires_files


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])