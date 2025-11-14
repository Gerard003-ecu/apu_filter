# tests/test_diagnose_presupuesto_file.py

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Añadir la ruta de scripts al path para importación
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.diagnose_presupuesto_file import ConfidenceLevel, PresupuestoFileDiagnostic

# --- Fixtures ---

@pytest.fixture
def valid_presupuesto_file(tmp_path):
    """Crea un archivo de presupuesto válido y más extenso."""
    content = """
# Metadatos del proyecto
Proyecto: Edificio Central
Fecha: 2024-01-01

ITEM;DESCRIPCION;CANTIDAD;UNIDAD;VR. UNITARIO;TOTAL
1;Excavación;100;M3;50000;5000000
2;Concreto;50;M3;350000;17500000
3;Acero;1500;KG;4000;6000000
4;Mano de Obra;1;GLB;10000000;10000000

TOTAL PRESUPUESTO: 38500000
"""
    file_path = tmp_path / "presupuesto_valid.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path

@pytest.fixture
def presupuesto_inconsistent_columns(tmp_path):
    """Archivo con número de columnas inconsistente."""
    content = """
ITEM;DESCRIPCION;CANTIDAD;TOTAL
1;Item A;10;1000
2;Item B con error;20;2000;EXTRA
3;Item C;30;3000
"""
    file_path = tmp_path / "presupuesto_inconsistent.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path

@pytest.fixture
def presupuesto_no_header(tmp_path):
    """Archivo sin un encabezado claro."""
    content = """
1;Excavación;100;M3;50000;5000000
2;Concreto;50;M3;350000;17500000
"""
    file_path = tmp_path / "presupuesto_no_header.csv"
    file_path.write_text(content, encoding='utf-8')
    return file_path

# --- Suite de Pruebas ---

class TestPresupuestoFileDiagnostic:
    """Pruebas para la clase PresupuestoFileDiagnostic."""

    def test_initialization_valid_file(self, valid_presupuesto_file):
        """Verifica que la inicialización con un archivo válido sea exitosa."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        assert diagnostic.file_path.exists()
        assert diagnostic.stats.file_size > 0

    def test_initialization_file_not_found(self, tmp_path):
        """Verifica que se lance un error si el archivo no existe."""
        with pytest.raises(ValueError, match="Archivo no encontrado"):
            PresupuestoFileDiagnostic(tmp_path / "non_existent.csv")

    @patch('scripts.diagnose_presupuesto_file.chardet.detect', return_value={'encoding': 'utf-8', 'confidence': 0.99})
    def test_diagnose_successful_run(self, mock_chardet, valid_presupuesto_file):
        """Prueba de integración de un diagnóstico exitoso."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()

        assert result.success
        assert result.stats.encoding.lower() == 'utf-8'
        assert result.stats.csv_delimiter == ';'
        assert result.header_candidate is not None
        assert result.header_candidate.line_num == 6
        assert result.stats.dominant_column_count == 6
        assert result.stats.column_consistency > 0.9

    def test_diagnose_inconsistent_columns(self, presupuesto_inconsistent_columns):
        """Verifica la detección de columnas inconsistentes."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_inconsistent_columns)
        result = diagnostic.diagnose()

        assert result.success
        assert result.stats.dominant_column_count == 4
        assert result.stats.column_consistency < 0.8  # Menor consistencia
        assert "Columnas inconsistentes" in " ".join(result.warnings)

    def test_diagnose_no_header(self, presupuesto_no_header):
        """Verifica el comportamiento cuando no se detecta encabezado."""
        diagnostic = PresupuestoFileDiagnostic(presupuesto_no_header)
        result = diagnostic.diagnose()

        assert result.success
        assert result.header_candidate is None
        assert "No se pudo detectar fila de encabezado" in result.warnings

    def test_header_detection_with_offset(self, valid_presupuesto_file):
        """Verifica que el encabezado se detecte aunque no esté en la primera línea."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()

        assert result.header_candidate is not None
        assert result.header_candidate.line_num == 6
        assert result.header_candidate.confidence == ConfidenceLevel.HIGH

    def test_comment_and_empty_line_handling(self, valid_presupuesto_file):
        """Verifica que las líneas de comentario y vacías se manejen correctamente."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()

        assert result.stats.comment_lines == 0  # Lógica actual no cuenta metadatos como comentarios
        assert result.stats.empty_lines == 1

    def test_total_line_detection(self, valid_presupuesto_file):
        """Verifica que las líneas de total/resumen se ignoren."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()

        assert result.stats.summary_lines_ignored == 1

    def test_overall_confidence_calculation(self, valid_presupuesto_file):
        """Verifica que la confianza general se calcule adecuadamente."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()

        assert result.stats.overall_confidence_score > 0.8
        assert result.stats.overall_confidence == ConfidenceLevel.HIGH.value

    def test_recommendations_generation(self, valid_presupuesto_file):
        """Verifica que se generen recomendaciones útiles."""
        diagnostic = PresupuestoFileDiagnostic(valid_presupuesto_file)
        result = diagnostic.diagnose()

        recommendations = " ".join(result.recommendations)
        assert "pandas" in recommendations
        assert f"header={result.header_candidate.line_num - 1}" in recommendations
        assert f"sep='{result.stats.csv_delimiter}'" in recommendations

    def test_main_function_integration(self, tmp_path, caplog):
        """Prueba la función main con un archivo simple y explícito."""
        import logging

        from scripts.diagnose_presupuesto_file import main

        # Crear un archivo simple que el Sniffer pueda manejar fácilmente
        content = "ITEM;DESCRIPCION;CANT\n1;Item A;10\n2;Item B;20\n"
        p = tmp_path / "simple_presupuesto.csv"
        p.write_text(content, encoding="utf-8")

        with patch.object(sys, 'argv', ['diagnose_presupuesto_file.py', str(p)]):
            with caplog.at_level(logging.INFO):
                return_code = main()

            assert return_code == 0
            log_output = caplog.text
            assert "REPORTE - DIAGNÓSTICO DE PRESUPUESTO" in log_output
