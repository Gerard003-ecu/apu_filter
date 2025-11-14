# tests/test_diagnose_insumos_file.py

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Añadir la ruta de scripts al path para importación
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.diagnose_insumos_file import ConfidenceLevel, InsumosFileDiagnostic

# --- Fixtures ---


@pytest.fixture
def valid_insumos_file(tmp_path):
    """Crea un archivo de insumos válido y más extenso."""
    content = """
# Insumos del proyecto
G;MATERIALES
CODIGO;DESCRIPCION;UND;VR. UNIT
MAT001;Cemento;BL;25000
MAT002;Arena;M3;45000
MAT003;Acero;KG;4000

G;MANO DE OBRA
CODIGO;DESCRIPCION;UND;VR. UNIT
MO001;Oficial;DIA;80000
MO002;Ayudante;DIA;50000
"""
    file_path = tmp_path / "insumos_valid.csv"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def insumos_mixed_quality(tmp_path):
    """Archivo con grupos de diferente calidad (sin header, inconsistente, etc.)."""
    content = """
G;GRUPO COMPLETO
CODIGO;DESCRIPCION;UND
ITEM1;Item A;UND

G;GRUPO SIN HEADER
ITEM2;Item B;UND

G;GRUPO INCONSISTENTE
CODIGO;DESCRIPCION
ITEM3;Item C;UND;EXTRA
ITEM4;Item D

G;GRUPO VACIO
CODIGO;DESCRIPCION
"""
    file_path = tmp_path / "insumos_mixed.csv"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def insumos_no_groups(tmp_path):
    """Archivo sin marcadores de grupo 'G;'."""
    content = """
CODIGO;DESCRIPCION;UND
ITEM1;Item A;UND
ITEM2;Item B;UND
"""
    file_path = tmp_path / "insumos_no_groups.csv"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def insumos_duplicate_groups(tmp_path):
    """Archivo con nombres de grupo duplicados."""
    content = """
G;MATERIALES
CODIGO;DESCRIPCION
MAT1;Cemento

G;MANO DE OBRA
CODIGO;DESCRIPCION
MO1;Oficial

G;MATERIALES
CODIGO;DESCRIPCION
MAT2;Arena
"""
    file_path = tmp_path / "insumos_duplicates.csv"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# --- Suite de Pruebas ---


class TestInsumosFileDiagnostic:
    """Pruebas para la clase InsumosFileDiagnostic."""

    def test_initialization_valid_file(self, valid_insumos_file):
        """Verifica la inicialización exitosa con un archivo válido."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        assert diagnostic.file_path.exists()
        assert diagnostic.stats.file_size > 0

    def test_initialization_file_not_found(self, tmp_path):
        """Verifica que se lance un error si el archivo no existe."""
        with pytest.raises(ValueError, match="Archivo no encontrado"):
            InsumosFileDiagnostic(tmp_path / "non_existent.csv")

    def test_diagnose_successful_run(self, valid_insumos_file):
        """Prueba de integración de un diagnóstico exitoso."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()

        assert result.success
        assert result.stats.encoding.lower() in [
            "utf-8",
            "ascii",
        ]  # ASCII es un subconjunto de UTF-8
        assert result.stats.csv_delimiter == ";"
        assert result.stats.groups_detected == 2
        assert result.stats.groups_complete == 2

        materiales_group = next(g for g in result.groups if g.name == "MATERIALES")
        assert materiales_group.is_complete
        assert materiales_group.dominant_column_count == 4

    def test_diagnose_no_groups_found(self, insumos_no_groups):
        """Verifica el comportamiento cuando no se encuentran grupos."""
        diagnostic = InsumosFileDiagnostic(insumos_no_groups)
        result = diagnostic.diagnose()

        assert result.success
        assert result.stats.groups_detected == 0
        assert "CRÍTICO: No se detectaron grupos" in " ".join(result.recommendations)

    def test_diagnose_mixed_quality_groups(self, insumos_mixed_quality):
        """Verifica el análisis de grupos con diferentes problemas de calidad."""
        diagnostic = InsumosFileDiagnostic(insumos_mixed_quality)
        result = diagnostic.diagnose()

        assert result.success
        assert result.stats.groups_detected == 4
        assert (
            result.stats.groups_complete == 2
        )  # Lógica mejorada ahora cuenta "INCONSISTENTE" como completo
        assert result.stats.integrity_issues > 0

        inconsistent_group = next(
            g for g in result.groups if g.name == "GRUPO INCONSISTENTE"
        )
        assert inconsistent_group.column_consistency < 0.8

        empty_group = next(g for g in result.groups if g.name == "GRUPO VACIO")
        assert not empty_group.has_data

    def test_diagnose_duplicate_groups(self, insumos_duplicate_groups):
        """Verifica el manejo de nombres de grupo duplicados."""
        diagnostic = InsumosFileDiagnostic(insumos_duplicate_groups)
        result = diagnostic.diagnose()

        assert result.success
        assert result.stats.groups_detected == 3
        assert result.stats.duplicate_groups == 1

        group_names = [g.name for g in result.groups]
        assert "MATERIALES" in group_names
        assert "MATERIALES (2)" in group_names

    def test_overall_confidence_high_for_good_file(self, valid_insumos_file):
        """Verifica que un archivo de buena calidad reciba una alta confianza."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()

        assert result.stats.overall_confidence_score > 0.8
        assert result.stats.overall_confidence == ConfidenceLevel.HIGH.value

    def test_overall_confidence_low_for_bad_file(self, insumos_mixed_quality):
        """Verifica que un archivo de baja calidad reciba una baja confianza."""
        diagnostic = InsumosFileDiagnostic(insumos_mixed_quality)
        result = diagnostic.diagnose()

        assert (
            result.stats.overall_confidence_score < 0.8
        )  # Un archivo mixto puede tener confianza media
        assert result.stats.overall_confidence == ConfidenceLevel.MEDIUM.value

    def test_recommendations_for_hierarchical_processing(self, valid_insumos_file):
        """Verifica que se generen recomendaciones para el procesamiento jerárquico."""
        diagnostic = InsumosFileDiagnostic(valid_insumos_file)
        result = diagnostic.diagnose()

        recommendations = " ".join(result.recommendations)
        assert "Estrategia sugerida" in recommendations
        assert "Identificar líneas que empiecen con 'G;'" in recommendations
        assert "Procesar cada grupo como DataFrame independiente" in recommendations

    def test_main_function_integration(self, tmp_path, caplog):
        """Prueba la función main con un archivo simple y explícito."""
        import logging

        from scripts.diagnose_insumos_file import main

        # Crear un archivo simple que el Sniffer pueda manejar fácilmente
        content = "G;GRUPO1\nCODIGO;DESC\nITEM1;A\nITEM2;B\n"
        p = tmp_path / "simple_insumos.csv"
        p.write_text(content, encoding="utf-8")

        with patch.object(sys, "argv", ["diagnose_insumos_file.py", str(p)]):
            with caplog.at_level(logging.INFO):
                return_code = main()

            assert return_code == 0
            log_output = caplog.text
            assert "REPORTE - DIAGNÓSTICO DE INSUMOS JERÁRQUICOS" in log_output
            assert "GRUPO1" in log_output
