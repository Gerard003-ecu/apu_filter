# tests/test_diagnose_apus_file.py

import logging
from pathlib import Path

import pytest

# Importar la clase y dataclasses a probar
from scripts.diagnose_apus_file import (
    APUFileDiagnostic,
    DiagnosticResult,
    FileStats,
    Pattern,
)

# --- Fixtures ---

@pytest.fixture
def sample_file_content() -> str:
    """Contenido de prueba realista."""
    return """
REMATE CON PINTURA    UNIDAD: ML    ITEM: 1,1
DESCRIPCION    UND    CANT.    PRECIO UNIT    VALOR TOTAL

MATERIALES
LAMINA DE ACERO    UND    0,33    174.928,81    65.403,35

MANO DE OBRA
M.O. CORTE Y DOBLEZ    UND    6,316    1.208,12    7.630,23

EQUIPO
EQUIPO Y HERRAMIENTA    UND    78.149,36    2%    1.562,98

SUBTOTAL OTROS    5.115,78
COSTO DIRECTO    79.712,00

REMATE CAL 22    UNIDAD: ML    ITEM: 1,2
MATERIALES
LAMINA DE 1.22 X 3.05    UND    0,27    174.928,81    54.145,39
"""

@pytest.fixture
def apus_file(tmp_path: Path, sample_file_content: str) -> Path:
    """Crea un archivo de APUs temporal y devuelve su ruta."""
    p = tmp_path / "apus.csv"
    p.write_text(sample_file_content, encoding="utf-8")
    return p

# --- Suite de Pruebas ---

class TestAPUFileDiagnostic:
    """Pruebas para APUFileDiagnostic usando archivos temporales reales."""

    def test_initialization_valid_path(self, apus_file: Path):
        """Verifica que la inicialización con una ruta válida sea correcta."""
        diagnostic = APUFileDiagnostic(apus_file)
        assert diagnostic.file_path.exists()
        assert isinstance(diagnostic.stats, FileStats)

    def test_diagnose_file_not_found(self, tmp_path: Path):
        """Verifica el comportamiento cuando el archivo no existe."""
        diagnostic = APUFileDiagnostic(tmp_path / "non_existent.csv")
        result = diagnostic.diagnose()
        assert not result.success
        assert "Archivo no encontrado" in result.errors[0]

    def test_diagnose_successful_run(self, apus_file: Path):
        """Prueba de integración de un diagnóstico exitoso."""
        diagnostic = APUFileDiagnostic(apus_file)
        result = diagnostic.diagnose()

        assert result.success
        assert result.stats.encoding.lower() in ['utf-8', 'ascii']
        assert result.stats.total_lines > 0
        assert len(result.patterns) > 0
        assert len(result.recommendations) > 0

    def test_diagnose_line_stats_analysis(self, apus_file: Path, sample_file_content: str):
        """Verifica que las estadísticas de líneas se calculen correctamente."""
        diagnostic = APUFileDiagnostic(apus_file)
        result = diagnostic.diagnose()
        stats = result.stats

        assert stats.total_lines == len(sample_file_content.splitlines())
        assert stats.empty_lines == 6
        assert stats.lines_with_item == 2
        assert stats.lines_with_unidad == 2
        assert stats.categories['MATERIALES'] == 2
        assert stats.categories['MANO DE OBRA'] == 1

    def test_diagnose_key_pattern_detection(self, apus_file: Path):
        """Verifica la detección de patrones clave como ITEM y UNIDAD."""
        diagnostic = APUFileDiagnostic(apus_file)
        result = diagnostic.diagnose()

        item_codes = [p for p in result.patterns if p.type == 'ITEM_CODE']
        units = [p for p in result.patterns if p.type == 'UNIT']

        assert len(item_codes) == 2
        assert item_codes[0].value == "1,1"
        assert all(u.value == "ML" for u in units)

    def test_diagnose_returns_valid_structure(self, apus_file: Path):
        """Verifica que diagnose() devuelva una estructura DiagnosticResult válida."""
        diagnostic = APUFileDiagnostic(apus_file)
        result = diagnostic.diagnose()

        assert isinstance(result, DiagnosticResult)
        assert isinstance(result.stats, FileStats)
        assert all(isinstance(p, Pattern) for p in result.patterns)

    def test_generate_diagnostic_report_logs_output(self, apus_file: Path, caplog):
        """Verifica que el reporte se genere correctamente en el log."""
        diagnostic = APUFileDiagnostic(apus_file)
        with caplog.at_level(logging.INFO):
            diagnostic.diagnose()

        log_output = caplog.text
        assert "REPORTE DE DIAGNÓSTICO AVANZADO - ARCHIVO APU" in log_output
        assert "RECOMENDACIONES" in log_output
        assert "Probablemente 'ITEM' marca inicio de cada APU" in log_output
