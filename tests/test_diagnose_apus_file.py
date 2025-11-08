# tests/test_diagnose_apus_file.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Importar la clase a probar
from scripts.diagnose_apus_file import APUFileDiagnostic

# --- Fixtures ---

@pytest.fixture
def sample_file_content() -> str:
    """
    Contenido de prueba realista que simula un archivo de APUs con:
    - Palabras clave (ITEM, UNIDAD, DESCRIPCIÓN)
    - Categorías (MATERIALES, MANO DE OBRA, etc.)
    - Separadores (espacios múltiples, punto y coma implícito)
    - Bloques separados por líneas vacías
    - Patrones estructurados
    """
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
def mock_path(sample_file_content: str):
    """
    Mock realista de un objeto Path que simula:
    - exists()
    - is_file()
    - stat().st_size
    - read_text()
    """
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.is_file.return_value = True
    mock_path.stat.return_value.st_size = len(sample_file_content.encode("utf-8"))
    mock_path.read_text.return_value = sample_file_content
    return mock_path


# --- Suite de Pruebas ---

class TestAPUFileDiagnostic:
    """
    Pruebas unitarias y de integración para APUFileDiagnostic.
    Verifican inicialización, análisis, robustez y reporte.
    """

    def test_initialization_valid_path(self, mock_path):
        """
        Verifica que la inicialización convierta correctamente la ruta y no procese aún.
        """
        diagnostic = APUFileDiagnostic(str(mock_path))

        # Se convierte a Path y se resuelve (aunque mock no cambie)
        assert isinstance(diagnostic.file_path, Path)
        assert diagnostic.stats == Counter()
        assert diagnostic.patterns_found == []
        assert diagnostic.sample_lines == []

    def test_diagnose_file_not_found(self, caplog):
        """
        Verifica que se registre error y retorne dict vacío si el archivo no existe.
        """
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        mock_path.is_file.return_value = False

        diagnostic = APUFileDiagnostic(mock_path)
        result = diagnostic.diagnose()

        assert "Archivo no encontrado" in caplog.text
        assert result == {}

    def test_diagnose_not_a_file(self, caplog):
        """
        Verifica que falle si la ruta existe pero no es un archivo.
        """
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = False

        diagnostic = APUFileDiagnostic(mock_path)
        result = diagnostic.diagnose()

        assert "Ruta no es un archivo" in caplog.text
        assert result == {}

    @patch("scripts.diagnose_apus_file.Path.read_text")
    def test_read_with_fallback_success_utf8(self, mock_read_text, sample_file_content):
        """
        Verifica que lea con utf-8 si está disponible.
        """
        mock_read_text.return_value = sample_file_content
        diagnostic = APUFileDiagnostic("fake_path.txt")

        content = diagnostic._read_with_fallback_encoding()

        assert content == sample_file_content
        assert diagnostic.stats["encoding"] == "utf-8"
        mock_read_text.assert_called_once_with(encoding="utf-8", errors="replace")

    @patch("scripts.diagnose_apus_file.Path.read_text")
    def test_read_with_fallback_utf8_fails_latin1_succeeds(self, mock_read_text, sample_file_content):
        """
        Verifica que intente encodings alternativos si utf-8 falla.
        """
        # Simulamos fallo en utf-8, éxito en latin1
        mock_read_text.side_effect = [
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"),
            sample_file_content,
        ]

        diagnostic = APUFileDiagnostic("fake_path.txt")
        content = diagnostic._read_with_fallback_encoding()

        assert content == sample_file_content
        assert diagnostic.stats["encoding"] == "latin1"
        assert mock_read_text.call_count == 2
        mock_read_text.assert_any_call(encoding="utf-8", errors="replace")
        mock_read_text.assert_any_call(encoding="latin1", errors="replace")

    @patch("scripts.diagnose_apus_file.Path.read_text")
    def test_read_with_fallback_all_encodings_fail(self, mock_read_text):
        """
        Verifica que devuelva None si todos los encodings fallan.
        """
        mock_read_text.side_effect = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad") for _ in range(4)

        diagnostic = APUFileDiagnostic("fake_path.txt")
        content = diagnostic._read_with_fallback_encoding()

        assert content is None
        assert "ninguno de los encodings soportados" in diagnostic._generate_diagnostic_report.__globals__["logger"].__dict__.get("name", "")

    def test_analyze_lines_correct_stats(self, sample_file_content):
        """
        Verifica que _analyze_lines calcule estadísticas correctamente.
        """
        diagnostic = APUFileDiagnostic("fake_path.txt")
        lines = sample_file_content.splitlines()

        with patch("scripts.diagnose_apus_file.Path.exists", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.is_file", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.read_text", return_value=sample_file_content):
            diagnostic.diagnose()  # Ejecutamos todo para que se llame _analyze_lines

        stats = diagnostic.stats

        assert stats["total_lines"] == len(lines)
        assert stats["empty_lines"] >= 5
        assert stats["non_empty_lines"] > 0
        assert stats["lines_with_ITEM"] == 2
        assert stats["lines_with_UNIDAD"] == 2
        assert stats["lines_with_DESCRIPCION"] == 1
        assert stats["category_MATERIALES"] == 2
        assert stats["category_MANO DE OBRA"] == 1
        assert stats["category_EQUIPO"] == 1
        assert stats["lines_with_multiple_spaces"] > 10  # Muchos espacios entre campos

    def test_analyze_structure_block_detection(self, sample_file_content):
        """
        Verifica que los bloques por doble salto sean detectados correctamente.
        """
        diagnostic = APUFileDiagnostic("fake_path.txt")

        with patch("scripts.diagnose_apus_file.Path.exists", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.is_file", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.read_text", return_value=sample_file_content):
            diagnostic.diagnose()

        stats = diagnostic.stats
        # Hay al menos 5 bloques: encabezado, MATERIALES, MANO DE OBRA, EQUIPO, REMATE CAL 22
        assert stats["blocks_by_double_newline"] >= 5

    def test_detect_patterns_item_and_unit(self, sample_file_content):
        """
        Verifica que detecte correctamente códigos ITEM y unidades.
        """
        diagnostic = APUFileDiagnostic("fake_path.txt")

        with patch("scripts.diagnose_apus_file.Path.exists", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.is_file", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.read_text", return_value=sample_file_content):
            diagnostic.diagnose()

        item_codes = [p for p in diagnostic.patterns_found if p["type"] == "ITEM_CODE"]
        units = [p for p in diagnostic.patterns_found if p["type"] == "UNIT"]

        assert len(item_codes) >= 2
        assert item_codes[0]["value"] == "1,1"
        assert item_codes[1]["value"] == "1,2"

        assert len(units) >= 2
        assert all(u["value"] == "ML" for u in units)

        assert diagnostic.stats["numeric_rows"] >= 4  # Múltiples filas numéricas

    def test_diagnose_returns_expected_structure(self, sample_file_content):
        """
        Prueba de integración: diagnose() debe devolver un dict con las claves correctas.
        """
        diagnostic = APUFileDiagnostic("fake_path.txt")

        with patch("scripts.diagnose_apus_file.Path.exists", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.is_file", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.read_text", return_value=sample_file_content):
            result = diagnostic.diagnose()

        assert isinstance(result, dict)
        assert set(result.keys()) == {"stats", "patterns", "samples"}
        assert isinstance(result["stats"], dict)
        assert isinstance(result["patterns"], list)
        assert isinstance(result["samples"], list)

        assert result["stats"]["total_lines"] > 0
        assert len(result["patterns"]) > 0
        assert len(result["samples"]) > 0

    def test_diagnose_multiple_calls_resets_state(self, sample_file_content):
        """
        Verifica que múltiples llamadas a diagnose() no acumulen estado.
        """
        diagnostic = APUFileDiagnostic("fake_path.txt")

        with patch("scripts.diagnose_apus_file.Path.exists", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.is_file", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.read_text", return_value=sample_file_content):
            result1 = diagnostic.diagnose()
            result2 = diagnostic.diagnose()

        # Los resultados deben ser idénticos, sin acumulación
        assert result1 == result2

    def test_generate_diagnostic_report_logs_output(self, sample_file_content, caplog):
        """
        Verifica que el reporte se genere correctamente usando logging.
        """
        diagnostic = APUFileDiagnostic("fake_path.txt")

        with patch("scripts.diagnose_apus_file.Path.exists", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.is_file", return_value=True), \
             patch("scripts.diagnose_apus_file.Path.read_text", return_value=sample_file_content):
            with caplog.at_level(logging.INFO):
                diagnostic.diagnose()

        # Verificar que se generaron secciones clave
        log_output = "\n".join(record.message for record in caplog.records)

        assert "REPORTE DE DIAGNÓSTICO DEL ARCHIVO APU" in log_output
        assert "ESTADÍSTICAS GENERALES" in log_output
        assert "SEPARADORES DETECTADOS" in log_output
        assert "PALABRAS CLAVE ENCONTRADAS" in log_output
        assert "RECOMENDACIONES" in log_output
        assert "El archivo usa PUNTO Y COMA" not in log_output  # No hay punto y coma
        assert "bloques separados por líneas vacías" in log_output
        assert "Se detectaron" in log_output  # ITEM encontrado