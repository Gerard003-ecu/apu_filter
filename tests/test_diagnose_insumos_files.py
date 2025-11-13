# tests/test_diagnose_insumos_file.py
"""
Suite de pruebas completa para diagnose_insumos_file.py

Este módulo contiene pruebas exhaustivas para validar:
- Inicialización y validación de archivos
- Detección de encoding con múltiples estrategias
- Detección de separadores de columnas
- Identificación de grupos y estructura jerárquica
- Detección de encabezados por grupo
- Análisis de integridad y consistencia
- Manejo de grupos duplicados
- Generación de reportes especializados
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
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from diagnose_insumos_file import (
    InsumosFileDiagnostic,
    DiagnosticError,
    FileReadError,
    EncodingDetectionError,
    StructureError,
    ConfidenceLevel,
    GroupInfo,
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
def valid_insumos_file(temp_dir):
    """Crea un archivo de insumos válido con estructura estándar."""
    content = """G;MATERIALES DE CONSTRUCCION
CODIGO;DESCRIPCION;UND;VR. UNIT;TOTAL
MAT001;Cemento gris 50kg;BL;25000;25000
MAT002;Arena lavada m3;M3;45000;45000
MAT003;Gravilla m3;M3;55000;55000

G;MANO DE OBRA
CODIGO;DESCRIPCION;UND;VR. UNIT;TOTAL
MO001;Oficial;DIA;80000;80000
MO002;Ayudante;DIA;50000;50000

G;EQUIPO Y HERRAMIENTA
CODIGO;DESCRIPCION;UND;VR. UNIT;TOTAL
EQ001;Mezcladora;DIA;120000;120000
EQ002;Vibrador;DIA;90000;90000
EQ003;Andamios;M2;5000;5000
"""
    file_path = temp_dir / "insumos_valid.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def insumos_with_missing_headers(temp_dir):
    """Archivo con grupos pero sin encabezados."""
    content = """G;MATERIALES
MAT001;Cemento;BL;25000;25000
MAT002;Arena;M3;45000;45000

G;MANO DE OBRA
MO001;Oficial;DIA;80000;80000
"""
    file_path = temp_dir / "missing_headers.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def insumos_with_empty_groups(temp_dir):
    """Archivo con grupos vacíos (sin datos)."""
    content = """G;MATERIALES
CODIGO;DESCRIPCION;UND;VR. UNIT;TOTAL

G;MANO DE OBRA
CODIGO;DESCRIPCION;UND;VR. UNIT;TOTAL
MO001;Oficial;DIA;80000;80000

G;EQUIPO

G;TRANSPORTE
CODIGO;DESCRIPCION;UND;VR. UNIT;TOTAL
"""
    file_path = temp_dir / "empty_groups.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def insumos_with_duplicate_groups(temp_dir):
    """Archivo con nombres de grupos duplicados."""
    content = """G;MATERIALES
CODIGO;DESCRIPCION;UND;VR. UNIT
MAT001;Cemento;BL;25000

G;MANO DE OBRA
CODIGO;DESCRIPCION;UND;VR. UNIT
MO001;Oficial;DIA;80000

G;MATERIALES
CODIGO;DESCRIPCION;UND;VR. UNIT
MAT002;Arena;M3;45000

G;MATERIALES
CODIGO;DESCRIPCION;UND;VR. UNIT
MAT003;Gravilla;M3;55000
"""
    file_path = temp_dir / "duplicate_groups.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def insumos_inconsistent_columns(temp_dir):
    """Archivo con columnas inconsistentes en un grupo."""
    content = """G;MATERIALES
CODIGO;DESCRIPCION;UND;VR. UNIT;TOTAL
MAT001;Cemento normal;BL;25000;25000
MAT002;Arena con descripcion muy larga y columnas extra;M3;45000;45000;EXTRA1;EXTRA2
MAT003;Gravilla corta;M3
MAT004;Item normal;M3;55000;55000

G;MANO DE OBRA
CODIGO;DESCRIPCION;UND;VR. UNIT;TOTAL
MO001;Oficial;DIA;80000;80000
MO002;Ayudante;DIA;50000;50000
"""
    file_path = temp_dir / "inconsistent_columns.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def insumos_different_separators(temp_dir):
    """Crea archivos con diferentes separadores."""
    files = {}
    
    # Punto y coma
    content_semicolon = """G;MATERIALES
CODIGO;DESCRIPCION;UND
MAT001;Cemento;BL
"""
    f1 = temp_dir / "sep_semicolon.csv"
    f1.write_text(content_semicolon, encoding='utf-8')
    files['semicolon'] = f1
    
    # Coma
    content_comma = """G,MATERIALES
CODIGO,DESCRIPCION,UND
MAT001,Cemento,BL
"""
    f2 = temp_dir / "sep_comma.csv"
    f2.write_text(content_comma, encoding='utf-8')
    files['comma'] = f2
    
    # Tabulación
    content_tab = """G\tMATERIALES
CODIGO\tDESCRIPCION\tUND
MAT001\tCemento\tBL
"""
    f3 = temp_dir / "sep_tab.csv"
    f3.write_text(content_tab, encoding='utf-8')
    files['tab'] = f3
    
    # Pipe
    content_pipe = """G|MATERIALES
CODIGO|DESCRIPCION|UND
MAT001|Cemento|BL
"""
    f4 = temp_dir / "sep_pipe.csv"
    f4.write_text(content_pipe, encoding='utf-8')
    files['pipe'] = f4
    
    return files


@pytest.fixture
def insumos_different_encodings(temp_dir):
    """Crea archivos con diferentes encodings."""
    files = {}
    
    content = """G;MATERIALES
CÓDIGO;DESCRIPCIÓN;UND
MAT001;Excavación;M3
"""
    
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
    
    return files


@pytest.fixture
def insumos_with_comments(temp_dir):
    """Archivo con diferentes tipos de comentarios."""
    content = """# Archivo de insumos
// Proyecto: Test
/* Comentario multilínea */

G;MATERIALES
# Comentario dentro del grupo
CODIGO;DESCRIPCION;UND;VR. UNIT
MAT001;Cemento;BL;25000
// Más comentarios
MAT002;Arena;M3;45000

G;MANO DE OBRA
CODIGO;DESCRIPCION;UND;VR. UNIT
MO001;Oficial;DIA;80000
"""
    file_path = temp_dir / "with_comments.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


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
def large_insumos_file(temp_dir):
    """Archivo grande que excede MAX_LINES_TO_ANALYZE."""
    content_lines = []
    
    for group_num in range(1, 50):  # 50 grupos
        content_lines.append(f"G;GRUPO {group_num}\n")
        content_lines.append("CODIGO;DESCRIPCION;UND;VR. UNIT\n")
        
        for item_num in range(1, 50):  # 50 items por grupo
            content_lines.append(
                f"ITEM{group_num:02d}{item_num:03d};Descripcion {item_num};UND;{item_num * 1000}\n"
            )
    
    file_path = temp_dir / "large.csv"
    file_path.write_text("".join(content_lines), encoding='utf-8')
    return file_path


@pytest.fixture
def insumos_no_groups(temp_dir):
    """Archivo sin grupos (estructura inválida)."""
    content = """CODIGO;DESCRIPCION;UND;VR. UNIT
MAT001;Cemento;BL;25000
MAT002;Arena;M3;45000
MAT003;Gravilla;M3;55000
"""
    file_path = temp_dir / "no_groups.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def insumos_grupo_prefix_variants(temp_dir):
    """Archivo con diferentes variantes del prefijo de grupo."""
    content = """G;MATERIALES
CODIGO;DESCRIPCION;UND
MAT001;Cemento;BL

GRUPO;MANO DE OBRA
CODIGO;DESCRIPCION;UND
MO001;Oficial;DIA

G;EQUIPO
CODIGO;DESCRIPCION;UND
EQ001;Mezcladora;DIA
"""
    file_path = temp_dir / "prefix_variants.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def insumos_mixed_quality_groups(temp_dir):
    """Archivo con grupos de diferente calidad."""
    content = """G;GRUPO PERFECTO
CODIGO;DESCRIPCION;UND;VR. UNIT;TOTAL
MAT001;Item 1;UND;1000;1000
MAT002;Item 2;UND;2000;2000
MAT003;Item 3;UND;3000;3000

G;GRUPO SIN ENCABEZADO
MAT004;Item 4;UND;4000;4000

G;GRUPO INCONSISTENTE
CODIGO;DESCRIPCION;UND
MAT005;Item 5;UND;5000;5000;EXTRA
MAT006;Item 6
MAT007;Item 7;UND;7000;7000

G;GRUPO VACIO
CODIGO;DESCRIPCION;UND

G;GRUPO BUENO
CODIGO;DESCRIPCION;UND;VR. UNIT
MAT008;Item 8;UND;8000
MAT009;Item 9;UND;9000
"""
    file_path = temp_dir / "mixed_quality.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path


# ============================================================================
# TESTS DE INICIALIZACIÓN Y VALIDACIÓN
# ============================================================================

class TestInitialization:
    """Pruebas de inicialización del diagnosticador."""
    
    def test_valid_file_initialization(self, valid_insumos_file):
        """Debe inicializarse correctamente con archivo válido."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        assert diagnostic.file_path == valid_insumos_file.resolve()
        assert isinstance(diagnostic.stats, dict)
        assert diagnostic._encoding is None
        assert len(diagnostic.groups_found) == 0
    
    def test_initialization_with_string_path(self, valid_insumos_file):
        """Debe aceptar rutas como string."""
        diagnostic = InsumosFileDiagnostic(str(valid_insumos_file))
        assert diagnostic.file_path.exists()
    
    def test_initialization_with_path_object(self, valid_insumos_file):
        """Debe aceptar objetos Path."""
        diagnostic = InsumosFileDiagnostic(Path(valid_insumos_file))
        assert diagnostic.file_path.exists()
    
    def test_nonexistent_file_raises_error(self, temp_dir):
        """Debe lanzar ValueError si el archivo no existe."""
        nonexistent = temp_dir / "nonexistent.csv"
        with pytest.raises(ValueError, match="no existe"):
            InsumosFileDiagnostic(nonexistent)
    
    def test_directory_path_raises_error(self, temp_dir):
        """Debe lanzar ValueError si la ruta es un directorio."""
        with pytest.raises(ValueError, match="no apunta a un archivo"):
            InsumosFileDiagnostic(temp_dir)
    
    def test_empty_file_raises_error(self, empty_file):
        """Debe lanzar ValueError si el archivo está vacío."""
        with pytest.raises(ValueError, match="vacío"):
            InsumosFileDiagnostic(empty_file)
    
    def test_permission_check(self, valid_insumos_file):
        """Debe verificar permisos de lectura."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        assert diagnostic._check_read_permissions() is True
    
    @patch('pathlib.Path.open', side_effect=PermissionError("No access"))
    def test_no_read_permission_raises_error(self, mock_open, valid_insumos_file):
        """Debe lanzar PermissionError si no hay permisos de lectura."""
        with pytest.raises(PermissionError, match="No hay permisos"):
            InsumosFileDiagnostic(valid_insumos_file)
    
    def test_reset_state_initialization(self, valid_insumos_file):
        """Debe inicializar correctamente el estado."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        assert len(diagnostic.groups_found) == 0
        assert diagnostic.current_group is None
        assert len(diagnostic._group_names_seen) == 0


# ============================================================================
# TESTS DE DETECCIÓN DE GRUPOS
# ============================================================================

class TestGroupDetection:
    """Pruebas específicas de detección de grupos."""
    
    def test_detect_single_group(self, temp_dir):
        """Debe detectar un grupo simple."""
        content = """G;MATERIALES
CODIGO;DESCRIPCION;UND
MAT001;Cemento;BL
"""
        file_path = temp_dir / "single_group.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['stats']['groups_detected'] == 1
        assert len(result['groups_found']) == 1
        assert result['groups_found'][0]['name'] == 'MATERIALES'
    
    def test_detect_multiple_groups(self, valid_insumos_file):
        """Debe detectar múltiples grupos."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        assert result['stats']['groups_detected'] == 3
        group_names = [g['name'] for g in result['groups_found']]
        assert 'MATERIALES DE CONSTRUCCION' in group_names
        assert 'MANO DE OBRA' in group_names
        assert 'EQUIPO Y HERRAMIENTA' in group_names
    
    def test_detect_no_groups(self, insumos_no_groups):
        """Debe retornar 0 grupos si no hay estructura de grupos."""
        diagnostic = InsumosFileDiagnostic(insumos_no_groups)
        result = diagnostic.diagnose()
        
        assert result['stats']['groups_detected'] == 0
        assert len(result['groups_found']) == 0
    
    def test_detect_duplicate_group_names(self, insumos_with_duplicate_groups):
        """Debe detectar y manejar grupos duplicados."""
        diagnostic = InsumosFileDiagnostic(insumos_with_duplicate_groups)
        result = diagnostic.diagnose()
        
        assert result['stats']['duplicate_groups'] >= 2
        # Verificar que se renombraron los duplicados
        group_names = [g['name'] for g in result['groups_found']]
        assert 'MATERIALES' in group_names
        assert 'MATERIALES (2)' in group_names or 'MATERIALES (3)' in group_names
    
    def test_group_line_numbers(self, valid_insumos_file):
        """Debe registrar correctamente los números de línea de grupos."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        for group in result['groups_found']:
            assert group['line_num'] > 0
    
    def test_is_group_line_method(self, valid_insumos_file):
        """Debe identificar correctamente líneas de grupo."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        diagnostic.diagnose()
        
        # Probar con líneas de grupo
        is_group, name = diagnostic._is_group_line("G;MATERIALES")
        assert is_group is True
        assert name == "MATERIALES"
        
        # Probar con línea normal
        is_group, name = diagnostic._is_group_line("MAT001;Cemento;BL")
        assert is_group is False
        assert name is None
    
    def test_different_group_prefixes(self, insumos_grupo_prefix_variants):
        """Debe detectar diferentes variantes del prefijo de grupo."""
        diagnostic = InsumosFileDiagnostic(insumos_grupo_prefix_variants)
        result = diagnostic.diagnose()
        
        assert result['stats']['groups_detected'] >= 2
        group_names = [g['name'] for g in result['groups_found']]
        assert 'MATERIALES' in group_names
        # GRUPO; también debería detectarse si está en GROUP_PREFIXES
    
    def test_empty_group_name_ignored(self, temp_dir):
        """Debe ignorar líneas de grupo con nombre vacío."""
        content = """G;
CODIGO;DESCRIPCION
MAT001;Cemento

G;MATERIALES VALIDOS
CODIGO;DESCRIPCION
MAT002;Arena
"""
        file_path = temp_dir / "empty_group_name.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        # Solo debe contar el grupo válido
        assert result['stats']['groups_detected'] >= 1
        group_names = [g['name'] for g in result['groups_found']]
        assert 'MATERIALES VALIDOS' in group_names


# ============================================================================
# TESTS DE ANÁLISIS DE GRUPOS
# ============================================================================

class TestGroupAnalysis:
    """Pruebas de análisis detallado de grupos."""
    
    def test_groups_with_headers(self, valid_insumos_file):
        """Debe identificar grupos con encabezados."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        assert result['stats']['groups_with_headers'] >= 2
        
        for group in result['groups_found']:
            if group['has_header']:
                assert group['header_line'] is not None
    
    def test_groups_with_data(self, valid_insumos_file):
        """Debe identificar grupos con datos."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        assert result['stats']['groups_with_data'] >= 2
        
        for group in result['groups_found']:
            if group['has_data']:
                assert group['data_lines'] > 0
    
    def test_complete_groups(self, valid_insumos_file):
        """Debe identificar grupos completos (con encabezado y datos)."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        assert result['stats']['groups_complete'] >= 2
    
    def test_group_without_header(self, insumos_with_missing_headers):
        """Debe detectar grupos sin encabezado."""
        diagnostic = InsumosFileDiagnostic(insumos_with_missing_headers)
        result = diagnostic.diagnose()
        
        groups_without_headers = [
            g for g in result['groups_found'] 
            if not g['has_header']
        ]
        assert len(groups_without_headers) >= 1
    
    def test_empty_group_detection(self, insumos_with_empty_groups):
        """Debe detectar grupos vacíos (sin datos)."""
        diagnostic = InsumosFileDiagnostic(insumos_with_empty_groups)
        result = diagnostic.diagnose()
        
        empty_groups = [
            g for g in result['groups_found']
            if not g['has_data']
        ]
        assert len(empty_groups) >= 2
    
    def test_group_data_line_count(self, valid_insumos_file):
        """Debe contar correctamente las líneas de datos por grupo."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        for group in result['groups_found']:
            if group['name'] == 'MATERIALES DE CONSTRUCCION':
                assert group['data_lines'] >= 3
    
    def test_group_column_consistency(self, valid_insumos_file):
        """Debe calcular consistencia de columnas por grupo."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        for group in result['groups_found']:
            if group['has_data']:
                consistency = float(group['column_consistency'].strip('%')) / 100
                assert 0.0 <= consistency <= 1.0
    
    def test_group_dominant_columns(self, valid_insumos_file):
        """Debe identificar número dominante de columnas por grupo."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        for group in result['groups_found']:
            if group['has_data']:
                assert group['dominant_columns'] is not None
                assert group['dominant_columns'] > 0
    
    def test_inconsistent_group_columns(self, insumos_inconsistent_columns):
        """Debe detectar inconsistencias en columnas de un grupo."""
        diagnostic = InsumosFileDiagnostic(insumos_inconsistent_columns)
        result = diagnostic.diagnose()
        
        # Buscar el grupo MATERIALES que tiene inconsistencias
        materiales_group = next(
            (g for g in result['groups_found'] if 'MATERIALES' in g['name']),
            None
        )
        
        if materiales_group and materiales_group['has_data']:
            consistency = float(materiales_group['column_consistency'].strip('%')) / 100
            assert consistency < 1.0  # Debe tener inconsistencias


# ============================================================================
# TESTS DE VALIDACIÓN DE INTEGRIDAD
# ============================================================================

class TestIntegrityValidation:
    """Pruebas de validación de integridad de grupos."""
    
    def test_integrity_issues_detected(self, insumos_mixed_quality_groups):
        """Debe detectar problemas de integridad en grupos."""
        diagnostic = InsumosFileDiagnostic(insumos_mixed_quality_groups)
        result = diagnostic.diagnose()
        
        if 'integrity_issues' in result['stats']:
            assert result['stats']['integrity_issues'] > 0
    
    def test_group_confidence_levels(self, valid_insumos_file):
        """Debe asignar niveles de confianza a los grupos."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        for group in result['groups_found']:
            assert group['confidence'] in [
                ConfidenceLevel.HIGH.value,
                ConfidenceLevel.MEDIUM.value,
                ConfidenceLevel.LOW.value,
                ConfidenceLevel.NONE.value
            ]
    
    def test_overall_confidence_calculation(self, valid_insumos_file):
        """Debe calcular confianza general del archivo."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        assert 'overall_confidence' in result['stats']
        assert result['stats']['overall_confidence'] in [
            ConfidenceLevel.HIGH.value,
            ConfidenceLevel.MEDIUM.value,
            ConfidenceLevel.LOW.value,
            ConfidenceLevel.NONE.value
        ]
    
    def test_high_quality_file_high_confidence(self, valid_insumos_file):
        """Archivo de alta calidad debe tener alta confianza."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        # Debe tener confianza alta o media
        assert result['stats']['overall_confidence'] in [
            ConfidenceLevel.HIGH.value,
            ConfidenceLevel.MEDIUM.value
        ]
    
    def test_low_quality_file_low_confidence(self, insumos_with_missing_headers):
        """Archivo de baja calidad debe tener baja confianza."""
        diagnostic = InsumosFileDiagnostic(insumos_with_missing_headers)
        result = diagnostic.diagnose()
        
        # Confianza debe ser media o baja
        assert result['stats']['overall_confidence'] in [
            ConfidenceLevel.MEDIUM.value,
            ConfidenceLevel.LOW.value,
            ConfidenceLevel.NONE.value
        ]


# ============================================================================
# TESTS DE DETECCIÓN DE ENCODING Y SEPARADOR
# ============================================================================

class TestEncodingAndSeparator:
    """Pruebas de detección de encoding y separador."""
    
    def test_detect_utf8(self, insumos_different_encodings):
        """Debe detectar UTF-8 correctamente."""
        diagnostic = InsumosFileDiagnostic(insumos_different_encodings['utf-8'])
        result = diagnostic.diagnose()
        assert result['encoding'] == 'utf-8'
    
    def test_detect_semicolon_separator(self, insumos_different_separators):
        """Debe detectar punto y coma como separador."""
        diagnostic = InsumosFileDiagnostic(insumos_different_separators['semicolon'])
        result = diagnostic.diagnose()
        assert result['separator'] == ';'
    
    def test_detect_comma_separator(self, insumos_different_separators):
        """Debe detectar coma como separador."""
        diagnostic = InsumosFileDiagnostic(insumos_different_separators['comma'])
        result = diagnostic.diagnose()
        assert result['separator'] == ','
    
    def test_detect_tab_separator(self, insumos_different_separators):
        """Debe detectar tabulación como separador."""
        diagnostic = InsumosFileDiagnostic(insumos_different_separators['tab'])
        result = diagnostic.diagnose()
        assert result['separator'] == '\t'
    
    def test_separator_confidence(self, valid_insumos_file):
        """Debe asignar nivel de confianza al separador."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        assert 'separator_confidence' in result['stats']
        assert result['stats']['separator_confidence'] in [
            ConfidenceLevel.HIGH.value,
            ConfidenceLevel.MEDIUM.value,
            ConfidenceLevel.LOW.value,
            ConfidenceLevel.NONE.value
        ]


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
        (5000, "4.88 KB"),
    ])
    def test_human_readable_size(self, size, expected, valid_insumos_file):
        """Debe convertir bytes a formato legible."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic._human_readable_size(size)
        assert result == expected
    
    @pytest.mark.parametrize("input_text,should_contain", [
        ("CODIGO", "CODIGO"),
        ("Descripción", "DESCRIPCION"),
        ("VR. UNIT", "VR UNIT"),
        ("Cód$igo", "CODIGO"),
        ("  Espacio  Multiple  ", "ESPACIO MULTIPLE"),
    ])
    def test_normalize_text(self, input_text, should_contain, valid_insumos_file):
        """Debe normalizar texto correctamente."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic._normalize_text(input_text)
        assert should_contain in result or result in should_contain
    
    @pytest.mark.parametrize("line,is_comment", [
        ("# Comentario", True),
        ("// Comentario", True),
        ("/* Comentario", True),
        ("-- Comentario", True),
        ("' Comentario", True),
        ("Normal line", False),
        ("G;MATERIALES", False),
        ("", False),
    ])
    def test_is_comment_line(self, line, is_comment, valid_insumos_file):
        """Debe identificar correctamente líneas de comentario."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic._is_comment_line(line)
        assert result == is_comment
    
    def test_finalize_current_group(self, valid_insumos_file):
        """Debe finalizar grupos correctamente."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        
        # Crear un grupo de prueba
        diagnostic.current_group = GroupInfo(
            name="TEST",
            line_num=1
        )
        diagnostic.current_group.data_lines = 5
        
        # Finalizar
        diagnostic._finalize_current_group()
        
        # El grupo actual debe seguir existiendo (solo se hace logging)
        assert diagnostic.current_group is not None


# ============================================================================
# TESTS DE DATACLASSES
# ============================================================================

class TestDataClasses:
    """Pruebas de las dataclasses específicas."""
    
    def test_group_info_creation(self):
        """Debe crear GroupInfo válido."""
        group = GroupInfo(
            name="MATERIALES",
            line_num=1
        )
        assert group.name == "MATERIALES"
        assert group.line_num == 1
        assert group.data_lines == 0
        assert group.has_header is False
        assert group.has_data is False
    
    def test_group_info_validation_line_num(self):
        """Debe validar line_num >= 1."""
        with pytest.raises(ValueError, match="line_num debe ser >= 1"):
            GroupInfo(name="TEST", line_num=0)
    
    def test_group_info_validation_empty_name(self):
        """Debe validar que name no esté vacío."""
        with pytest.raises(ValueError, match="name no puede estar vacío"):
            GroupInfo(name="", line_num=1)
        
        with pytest.raises(ValueError, match="name no puede estar vacío"):
            GroupInfo(name="   ", line_num=1)
    
    def test_group_info_validation_negative_data_lines(self):
        """Debe validar data_lines >= 0."""
        with pytest.raises(ValueError, match="data_lines debe ser >= 0"):
            GroupInfo(name="TEST", line_num=1, data_lines=-1)
    
    def test_group_info_validation_header_line(self):
        """Debe validar que header_line sea posterior a line_num."""
        with pytest.raises(ValueError, match="header_line debe ser posterior"):
            GroupInfo(name="TEST", line_num=5, header_line=3)
    
    def test_group_info_properties(self):
        """Debe calcular propiedades correctamente."""
        group = GroupInfo(name="TEST", line_num=1, header_line=2)
        
        # Sin datos
        assert group.dominant_column_count is None
        assert group.column_consistency == 0.0
        assert group.has_header is True
        assert group.has_data is False
        
        # Agregar datos
        group.data_lines = 10
        group.column_counts[5] = 8
        group.column_counts[6] = 2
        
        assert group.dominant_column_count == 5
        assert group.column_consistency == 0.8
        assert group.has_data is True
    
    def test_header_candidate_with_group(self):
        """Debe crear HeaderCandidate con nombre de grupo."""
        header = HeaderCandidate(
            line_num=2,
            content="CODIGO;DESC;UND",
            matches=["CODIGO", "UND"],
            match_count=2,
            column_count=3,
            group_name="MATERIALES"
        )
        assert header.group_name == "MATERIALES"
    
    def test_sample_line_with_group(self):
        """Debe crear SampleLine con nombre de grupo."""
        sample = SampleLine(
            line_num=3,
            content="MAT001;Cemento;BL",
            group_name="MATERIALES",
            column_count=3
        )
        assert sample.group_name == "MATERIALES"
    
    def test_column_statistics_functionality(self):
        """Debe funcionar ColumnStatistics correctamente."""
        stats = ColumnStatistics()
        
        stats.add_sample("line1", max_samples=2)
        stats.add_sample("line2", max_samples=2)
        stats.add_sample("line3", max_samples=2)
        
        assert stats.count == 3
        assert len(stats.samples) == 2
        
        stats.percentage = 75.5
        assert stats.percentage == 75.5


# ============================================================================
# TESTS DE GENERACIÓN DE REPORTES
# ============================================================================

class TestReportGeneration:
    """Pruebas de generación de reportes."""
    
    def test_generate_report_creates_output(self, valid_insumos_file, caplog):
        """Debe generar output de log al crear reporte."""
        import logging
        caplog.set_level(logging.INFO)
        
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        diagnostic.diagnose()
        
        assert len(caplog.records) > 0
        assert any("REPORTE DE DIAGNÓSTICO" in record.message for record in caplog.records)
    
    def test_report_includes_group_analysis(self, valid_insumos_file, caplog):
        """Debe incluir análisis de grupos en reporte."""
        import logging
        caplog.set_level(logging.INFO)
        
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        diagnostic.diagnose()
        
        log_output = " ".join(record.message for record in caplog.records)
        assert "ANÁLISIS DE GRUPOS" in log_output or "GRUPOS" in log_output
    
    def test_report_includes_group_details(self, valid_insumos_file, caplog):
        """Debe incluir detalles de grupos individuales."""
        import logging
        caplog.set_level(logging.INFO)
        
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        diagnostic.diagnose()
        
        log_output = " ".join(record.message for record in caplog.records)
        assert "MATERIALES" in log_output or "MANO DE OBRA" in log_output
    
    def test_report_includes_python_example(self, valid_insumos_file, caplog):
        """Debe incluir ejemplo de código Python."""
        import logging
        caplog.set_level(logging.INFO)
        
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        diagnostic.diagnose()
        
        log_output = " ".join(record.message for record in caplog.records)
        assert "EJEMPLO" in log_output or "python" in log_output.lower()
    
    def test_report_shows_integrity_issues(self, insumos_mixed_quality_groups, caplog):
        """Debe mostrar problemas de integridad en reporte."""
        import logging
        caplog.set_level(logging.INFO)
        
        diagnostic = InsumosFileDiagnostic(insumos_mixed_quality_groups)
        diagnostic.diagnose()
        
        log_output = " ".join(record.message for record in caplog.records)
        # Puede mostrar warnings sobre grupos problemáticos
        has_warnings = any(
            record.levelname == 'WARNING' 
            for record in caplog.records
        )
        # O mencionar problemas en el reporte
        has_problems_mention = "problemas" in log_output.lower() or "issues" in log_output.lower()
        
        assert has_warnings or has_problems_mention


# ============================================================================
# TESTS DE MANEJO DE ERRORES
# ============================================================================

class TestErrorHandling:
    """Pruebas de manejo de errores."""
    
    def test_file_read_error_on_whitespace_only(self, whitespace_only_file):
        """Debe lanzar FileReadError con contenido solo espacios."""
        diagnostic = InsumosFileDiagnostic(whitespace_only_file)
        with pytest.raises(FileReadError, match="no contiene datos válidos"):
            diagnostic.diagnose()
    
    def test_diagnostic_error_wraps_exceptions(self, valid_insumos_file):
        """Debe envolver excepciones inesperadas en DiagnosticError."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        
        with patch.object(diagnostic, '_analyze_structure_hierarchical', 
                         side_effect=RuntimeError("Test error")):
            with pytest.raises(DiagnosticError, match="Fallo en el diagnóstico"):
                diagnostic.diagnose()
    
    def test_reset_state_clears_data(self, valid_insumos_file):
        """Debe limpiar estado al resetear."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        
        # Poblar con datos
        diagnostic.stats['test'] = 123
        diagnostic.groups_found.append(GroupInfo(name="TEST", line_num=1))
        diagnostic._group_names_seen.add("TEST")
        
        # Resetear
        diagnostic._reset_state()
        
        assert 'test' not in diagnostic.stats
        assert len(diagnostic.groups_found) == 0
        assert len(diagnostic._group_names_seen) == 0
    
    def test_handles_malformed_group_line(self, temp_dir):
        """Debe manejar líneas de grupo mal formadas."""
        content = """G;
G
GRUPO;
CODIGO;DESCRIPCION
MAT001;Cemento

G;GRUPO VALIDO
CODIGO;DESCRIPCION
MAT002;Arena
"""
        file_path = temp_dir / "malformed_groups.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        # Debe procesar sin errores, ignorando líneas inválidas
        assert result['success'] is True
    
    def test_handles_corrupted_file(self, temp_dir):
        """Debe manejar archivos corruptos gracefully."""
        file_path = temp_dir / "corrupted.csv"
        with open(file_path, 'wb') as f:
            f.write(b'\x00\x01\x02G;TEST\x03\x04\xff\xfe')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['success'] is True


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

class TestIntegration:
    """Pruebas de integración del flujo completo."""
    
    def test_full_diagnostic_workflow(self, valid_insumos_file):
        """Debe completar el flujo completo exitosamente."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        # Verificar estructura básica
        assert result['success'] is True
        assert 'stats' in result
        assert 'encoding' in result
        assert 'separator' in result
        assert 'groups_found' in result
        
        # Verificar que se detectaron grupos
        assert result['stats']['groups_detected'] > 0
        assert len(result['groups_found']) > 0
    
    def test_result_dict_structure(self, valid_insumos_file):
        """Debe retornar diccionario con estructura completa."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        required_keys = [
            'success', 'file_path', 'stats', 'encoding', 
            'separator', 'groups_found'
        ]
        for key in required_keys:
            assert key in result
    
    def test_groups_found_structure(self, valid_insumos_file):
        """Debe retornar grupos con estructura completa."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        if result['groups_found']:
            group = result['groups_found'][0]
            expected_keys = [
                'name', 'line_num', 'header_line', 'data_lines',
                'dominant_columns', 'column_consistency', 'confidence',
                'has_header', 'has_data'
            ]
            for key in expected_keys:
                assert key in group
    
    def test_multiple_diagnoses_consistent(self, valid_insumos_file):
        """Debe dar resultados consistentes en múltiples diagnósticos."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        
        result1 = diagnostic.diagnose()
        result2 = diagnostic.diagnose()
        
        assert result1['encoding'] == result2['encoding']
        assert result1['separator'] == result2['separator']
        assert result1['stats']['groups_detected'] == result2['stats']['groups_detected']
    
    def test_end_to_end_recommendations(self, valid_insumos_file):
        """Debe proporcionar información accionable."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        # Información suficiente para procesar
        assert result['separator'] is not None
        assert result['encoding'] is not None
        assert len(result['groups_found']) > 0
        
        for group in result['groups_found']:
            assert 'name' in group
            assert 'line_num' in group
    
    def test_various_file_qualities(self, temp_dir):
        """Debe manejar archivos de diferentes calidades."""
        files = []
        
        # Archivo perfecto
        perfect = temp_dir / "perfect.csv"
        perfect.write_text(
            "G;MATERIALES\n"
            "CODIGO;DESCRIPCION;UND;VR. UNIT\n"
            "MAT001;Cemento;BL;25000\n"
            "MAT002;Arena;M3;45000\n",
            encoding='utf-8'
        )
        files.append(perfect)
        
        # Archivo con problemas menores
        minor = temp_dir / "minor.csv"
        minor.write_text(
            "\n# Comment\n"
            "G;MATERIALES\n"
            "CODIGO;DESC\n"
            "MAT001;Cemento\n"
            "\n",
            encoding='utf-8'
        )
        files.append(minor)
        
        # Todos deben procesarse
        for file_path in files:
            diagnostic = InsumosFileDiagnostic(file_path)
            result = diagnostic.diagnose()
            assert result['success'] is True


# ============================================================================
# TESTS DEL MAIN
# ============================================================================

class TestMainFunction:
    """Pruebas de la función main."""
    
    def test_main_no_arguments(self):
        """Debe retornar 1 si no se proporciona archivo."""
        from diagnose_insumos_file import main
        
        with patch('sys.argv', ['script_name']):
            result = main()
            assert result == 1
    
    def test_main_with_valid_file(self, valid_insumos_file):
        """Debe retornar 0 con archivo válido."""
        from diagnose_insumos_file import main
        
        with patch('sys.argv', ['script_name', str(valid_insumos_file)]):
            result = main()
            assert result == 0
    
    def test_main_with_nonexistent_file(self, temp_dir):
        """Debe retornar 1 con archivo inexistente."""
        from diagnose_insumos_file import main
        
        with patch('sys.argv', ['script_name', str(temp_dir / 'nonexistent.csv')]):
            result = main()
            assert result == 1
    
    def test_main_keyboard_interrupt(self, valid_insumos_file):
        """Debe retornar 130 al interrumpir."""
        from diagnose_insumos_file import main
        
        with patch('sys.argv', ['script_name', str(valid_insumos_file)]):
            with patch('diagnose_insumos_file.InsumosFileDiagnostic.diagnose',
                      side_effect=KeyboardInterrupt()):
                result = main()
                assert result == 130


# ============================================================================
# TESTS DE CASOS EDGE
# ============================================================================

class TestEdgeCases:
    """Pruebas de casos edge y situaciones especiales."""
    
    def test_group_with_only_header(self, temp_dir):
        """Debe manejar grupo con solo encabezado."""
        content = """G;MATERIALES
CODIGO;DESCRIPCION;UND;VR. UNIT
"""
        file_path = temp_dir / "only_header.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['stats']['groups_detected'] == 1
        group = result['groups_found'][0]
        assert group['has_header'] is True
        assert group['has_data'] is False
    
    def test_consecutive_groups(self, temp_dir):
        """Debe manejar grupos consecutivos sin separación."""
        content = """G;GRUPO1
CODIGO;DESC
G;GRUPO2
CODIGO;DESC
G;GRUPO3
CODIGO;DESC
"""
        file_path = temp_dir / "consecutive.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['stats']['groups_detected'] == 3
    
    def test_unicode_in_group_names(self, temp_dir):
        """Debe manejar Unicode en nombres de grupos."""
        content = """G;MATERIALES™ Y EQUIPOS®
CODIGO;DESCRIPCIÓN;UND
MAT001;Excavación€;M3

G;MANO DE OBRA SEÑALIZACIÓN
CODIGO;DESCRIPCIÓN;UND
MO001;Peón;DIA
"""
        file_path = temp_dir / "unicode_groups.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['success'] is True
        assert result['stats']['groups_detected'] >= 2
    
    def test_very_long_group_names(self, temp_dir):
        """Debe manejar nombres de grupos muy largos."""
        long_name = "X" * 500
        content = f"""G;{long_name}
CODIGO;DESC
MAT001;Item
"""
        file_path = temp_dir / "long_name.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['success'] is True
        assert result['stats']['groups_detected'] == 1
    
    def test_mixed_line_endings(self, temp_dir):
        """Debe manejar diferentes finales de línea."""
        content = "G;MATERIALES\r\nCODIGO;DESC\r\nMAT001;Cemento\r\n"
        file_path = temp_dir / "crlf.csv"
        file_path.write_bytes(content.encode('utf-8'))
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['success'] is True
    
    def test_group_with_numeric_name(self, temp_dir):
        """Debe manejar grupos con nombres numéricos."""
        content = """G;12345
CODIGO;DESC
MAT001;Item
"""
        file_path = temp_dir / "numeric_name.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['stats']['groups_detected'] == 1
        assert result['groups_found'][0]['name'] == '12345'
    
    def test_group_with_special_characters(self, temp_dir):
        """Debe manejar caracteres especiales en nombres."""
        content = """G;MAT@#$%&*()
CODIGO;DESC
MAT001;Item
"""
        file_path = temp_dir / "special_chars.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['success'] is True
    
    def test_deeply_nested_comments(self, temp_dir):
        """Debe manejar comentarios intercalados con datos."""
        content = """G;MATERIALES
# Comentario
CODIGO;DESCRIPCION;UND
# Otro comentario
MAT001;Cemento;BL
// Comentario C++
MAT002;Arena;M3
/* Comentario */
"""
        file_path = temp_dir / "nested_comments.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        assert result['success'] is True
        # Los comentarios no deben contarse como datos
        group = result['groups_found'][0]
        assert group['data_lines'] == 2  # Solo MAT001 y MAT002


# ============================================================================
# TESTS DE PERFORMANCE
# ============================================================================

class TestPerformance:
    """Pruebas de rendimiento."""
    
    def test_large_file_reasonable_time(self, large_insumos_file):
        """Debe procesar archivo grande en tiempo razonable."""
        import time
        
        diagnostic = InsumosFileDiagnostic(large_insumos_file)
        
        start = time.time()
        result = diagnostic.diagnose()
        elapsed = time.time() - start
        
        assert result['success'] is True
        assert elapsed < 15  # Menos de 15 segundos
    
    def test_large_file_truncation(self, large_insumos_file):
        """Debe truncar análisis en archivos muy grandes."""
        diagnostic = InsumosFileDiagnostic(large_insumos_file)
        result = diagnostic.diagnose()
        
        if result['stats'].get('total_lines', 0) > diagnostic.MAX_LINES_TO_ANALYZE:
            assert result['stats'].get('truncated_analysis') is True
    
    def test_memory_efficiency(self, large_insumos_file):
        """Debe ser eficiente en uso de memoria."""
        diagnostic = InsumosFileDiagnostic(large_insumos_file)
        diagnostic.diagnose()
        
        # No debe almacenar todas las líneas
        assert len(diagnostic.sample_lines) <= diagnostic.MAX_SAMPLE_LINES
        
        # No debe almacenar muestras ilimitadas por grupo
        for group in diagnostic.groups_found:
            if hasattr(group, 'samples'):
                assert len(group.samples) <= diagnostic.MAX_SAMPLES_PER_GROUP


# ============================================================================
# TESTS ESPECÍFICOS DE ESTRUCTURA JERÁRQUICA
# ============================================================================

class TestHierarchicalStructure:
    """Pruebas específicas de estructura jerárquica."""
    
    def test_group_header_data_sequence(self, valid_insumos_file):
        """Debe respetar secuencia: grupo -> encabezado -> datos."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        for group in result['groups_found']:
            if group['has_header'] and group['has_data']:
                # Header debe estar después del grupo
                assert group['header_line'] > group['line_num']
    
    def test_multiple_groups_independence(self, valid_insumos_file):
        """Debe analizar grupos independientemente."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()
        
        # Cada grupo debe tener su propio análisis
        for group in result['groups_found']:
            # Cada grupo debe tener su propia info de columnas
            if group['has_data']:
                assert group['dominant_columns'] is not None
    
    def test_group_isolation(self, temp_dir):
        """Datos de un grupo no deben mezclarse con otro."""
        content = """G;GRUPO1
CODIGO;DESC;UND
MAT001;Item1;UND
MAT002;Item2;UND

G;GRUPO2
CODIGO;DESC
MAT003;Item3
"""
        file_path = temp_dir / "isolated_groups.csv"
        file_path.write_text(content, encoding='utf-8')
        
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        grupo1 = result['groups_found'][0]
        grupo2 = result['groups_found'][1]
        
        # Cada grupo debe tener conteo de columnas diferente
        assert grupo1['dominant_columns'] != grupo2['dominant_columns']


# ============================================================================
# CONFIGURACIÓN DE PYTEST
# ============================================================================

@pytest.fixture(autouse=True)
def reset_logging():
    """Resetea configuración de logging entre pruebas."""
    import logging
    logger = logging.getLogger("InsumosDiagnostic")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    
    yield
    
    logger.handlers = original_handlers
    logger.level = original_level


# ============================================================================
# MARKS
# ============================================================================

slow_tests = pytest.mark.slow
requires_files = pytest.mark.requires_files


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])