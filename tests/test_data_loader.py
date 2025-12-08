# tests/test_data_loader.py
"""
Pruebas unitarias para el módulo data_loader.py utilizando pytest.
Cubre casos de éxito, manejo de errores y validaciones para CSV, Excel y PDF.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Importar el módulo bajo prueba
from app.data_loader import (
    LoadStatus,
    load_data,
    load_from_csv,
    load_from_pdf,
    load_from_xlsx,
    FileMetadata,
    FileFormat,
    LoadResult,
    DataQualityMetrics
)

# ─────────────────────────────────────────────────────────────
# FIXTURES DE PYTEST
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_path_exists():
    """Fixture para mockear pathlib.Path.exists."""
    with patch("pathlib.Path.exists") as mock_exists:
        yield mock_exists


@pytest.fixture
def mock_path_is_file():
    """Fixture para mockear pathlib.Path.is_file."""
    with patch("pathlib.Path.is_file") as mock_is_file:
        yield mock_is_file


@pytest.fixture
def mock_path_stat():
    """Fixture para mockear pathlib.Path.stat."""
    with patch("pathlib.Path.stat") as mock_stat:
        stat_result = MagicMock()
        stat_result.st_size = 1024  # 1KB
        stat_result.st_mtime = 1600000000.0
        stat_result.st_mode = 0o100444  # File type + read permissions
        mock_stat.return_value = stat_result
        yield mock_stat


@pytest.fixture
def mock_file_open():
    """Fixture para mockear builtins.open (para validaciones de lectura)."""
    with patch("builtins.open") as mock_open:
        mock_file = MagicMock()
        # Simular lectura de bytes para verificación de firma
        mock_file.read.return_value = b"PK\x03\x04"  # Firma ZIP (Excel/OOXML)
        mock_open.return_value.__enter__.return_value = mock_file
        yield mock_open


# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA load_from_csv
# ─────────────────────────────────────────────────────────────


@patch("pandas.read_csv")
def test_load_from_csv_success(
    mock_read_csv, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open
):
    """Prueba la carga exitosa de un archivo CSV."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_read_csv.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("data.csv"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.CSV,
            exists=True,
            readable=True
        )

        with patch("app.data_loader._validate_file_path", return_value=Path("data.csv")):
            result = load_from_csv(Path("data.csv"), sep=";", encoding="utf-8")

    assert result.status == LoadStatus.SUCCESS
    assert not result.data.empty
    assert result.data.shape == (2, 2)


def test_load_from_csv_file_not_found(mock_path_exists, mock_path_is_file):
    """Prueba que load_from_csv devuelva FAILED cuando el archivo no existe."""
    mock_path_exists.return_value = False
    mock_path_is_file.return_value = False

    with patch("app.data_loader._validate_file_path", side_effect=FileNotFoundError("Archivo no encontrado")):
        result = load_from_csv(Path("missing.csv"))
        assert result.status == LoadStatus.FAILED
        assert "Archivo no encontrado" in result.error_message


@patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError("No columns to parse from file"))
def test_load_from_csv_empty_error(
    mock_read_csv, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open
):
    """Prueba el manejo de un archivo CSV vacío que lanza EmptyDataError."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("empty.csv"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.CSV,
            exists=True,
            readable=True
        )
        with patch("app.data_loader._validate_file_path", return_value=Path("empty.csv")):
            result = load_from_csv(Path("empty.csv"))

    assert result.status == LoadStatus.EMPTY

    # Check that warnings contain the error message
    warnings_text = " ".join(result.quality_metrics.warnings) if result.quality_metrics else ""
    error_text = str(result.error_message) if result.error_message else ""
    assert "No columns to parse" in warnings_text or "No columns to parse" in error_text


@patch("pandas.read_csv", side_effect=UnicodeDecodeError("utf-8", b"\\xff", 0, 1, "invalid"))
def test_load_from_csv_encoding_error(
    mock_read_csv, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open
):
    """Prueba el manejo de errores de codificación en CSV."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("bad_encoding.csv"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.CSV,
            exists=True,
            readable=True
        )
        with patch("app.data_loader._validate_file_path", return_value=Path("bad_encoding.csv")):
            # Force retry limit to avoid infinite loop
            result = load_from_csv(Path("bad_encoding.csv"), encoding="utf-8", max_retries=1)

    assert result.status == LoadStatus.FAILED
    assert "No se pudo cargar el CSV" in str(result.error_message)


# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA load_from_xlsx
# ─────────────────────────────────────────────────────────────


@patch("pandas.read_excel")
def test_load_from_xlsx_single_sheet_success(
    mock_read_excel, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open
):
    """Prueba la carga exitosa de una única hoja de Excel."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_read_excel.return_value = pd.DataFrame({"X": [10], "Y": [20]})

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("data.xlsx"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.EXCEL,
            exists=True,
            readable=True
        )
        with patch("app.data_loader._validate_file_path", return_value=Path("data.xlsx")):
            with patch("pandas.ExcelFile") as mock_excel_file:
                mock_excel_file.return_value.sheet_names = ["Hoja1"]
                result = load_from_xlsx(Path("data.xlsx"), sheet_name="Hoja1")

    assert result.status == LoadStatus.SUCCESS
    assert result.data.shape == (1, 2)


@patch("pandas.read_excel")
def test_load_from_xlsx_all_sheets_success(
    mock_read_excel, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open
):
    """Prueba que devuelva un dict de DataFrames cuando sheet_name es None."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    sheets = {"Sheet1": pd.DataFrame({"A": [1]}), "Sheet2": pd.DataFrame({"B": [2]})}
    mock_read_excel.return_value = sheets

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("data.xlsx"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.EXCEL,
            exists=True,
            readable=True
        )
        with patch("app.data_loader._validate_file_path", return_value=Path("data.xlsx")):
            with patch("pandas.ExcelFile") as mock_excel_file:
                mock_excel_file.return_value.sheet_names = ["Sheet1", "Sheet2"]
                result = load_from_xlsx(Path("data.xlsx"), sheet_name=None)

    assert result.status == LoadStatus.SUCCESS
    assert isinstance(result.data, dict)
    assert len(result.data) == 2


def test_load_from_xlsx_file_not_found(mock_path_exists, mock_path_is_file):
    """Prueba que devuelva FAILED si el archivo Excel no existe."""
    mock_path_exists.return_value = False
    mock_path_is_file.return_value = False

    with patch("app.data_loader._validate_file_path", side_effect=FileNotFoundError("Archivo no encontrado")):
        result = load_from_xlsx(Path("missing.xlsx"))
        assert result.status == LoadStatus.FAILED
        assert "Archivo no encontrado" in result.error_message


@patch("pandas.read_excel", side_effect=ValueError("No such sheet"))
def test_load_from_xlsx_sheet_not_found(
    mock_read_excel, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open
):
    """Prueba el manejo de una hoja de Excel inexistente."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("data.xlsx"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.EXCEL,
            exists=True,
            readable=True
        )
        with patch("app.data_loader._validate_file_path", return_value=Path("data.xlsx")):
            with patch("pandas.ExcelFile") as mock_excel_file:
                # Mock available sheets to be different from requested
                mock_excel_file.return_value.sheet_names = ["Sheet1"]
                result = load_from_xlsx(Path("data.xlsx"), sheet_name="NoExiste")

    # The validator now checks available sheets before calling read_excel if available_sheets are retrieved
    assert result.status == LoadStatus.FAILED
    # Error message contains 'no encontrada' (lowercase n) in recent versions of code or 'No encontrada'
    assert "no encontrada" in str(result.error_message).lower() or "no such sheet" in str(result.error_message).lower()


# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA load_from_pdf
# ─────────────────────────────────────────────────────────────


@patch("pdfplumber.open")
def test_load_from_pdf_no_tables(
    mock_pdf_open, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open, caplog
):
    """Prueba un PDF sin tablas; debe devolver un DataFrame vacío."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_page = MagicMock()
    mock_page.extract_tables.return_value = []
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("no_tables.pdf"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.PDF,
            exists=True,
            readable=True
        )
        with patch("app.data_loader._validate_file_path", return_value=Path("no_tables.pdf")):
            result = load_from_pdf(Path("no_tables.pdf"))

    assert result.status == LoadStatus.EMPTY
    assert result.data.empty


@patch("pdfplumber.open")
def test_load_from_pdf_with_dirty_tables(
    mock_pdf_open, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open
):
    """Prueba la extracción de tablas 'sucias' (con Nones y longitudes variables)."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_table = [
        ["Nombre", "Edad", "Ciudad"],
        ["Alice", "30", "Madrid"],
        ["Bob", None],  # Fila con longitud inconsistente
        ["Charlie", "25", "Barcelona"],
    ]
    mock_page = MagicMock()
    mock_page.extract_tables.return_value = [mock_table]
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("dirty_tables.pdf"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.PDF,
            exists=True,
            readable=True
        )
        with patch("app.data_loader._validate_file_path", return_value=Path("dirty_tables.pdf")):
            result = load_from_pdf(Path("dirty_tables.pdf"))

    assert result.status == LoadStatus.SUCCESS
    # Note: _source_page and _source_table columns are added, so shape changes
    assert result.data.shape[0] == 3
    assert result.data.shape[1] >= 3


@patch("pdfplumber.open")
def test_load_from_pdf_with_table_settings(
    mock_pdf_open, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open
):
    """Prueba que los 'table_settings' se pasen correctamente a extract_tables."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_page = MagicMock()
    mock_page.extract_tables.return_value = []
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf

    settings = {"vertical_strategy": "lines"}

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("some.pdf"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.PDF,
            exists=True,
            readable=True
        )
        with patch("app.data_loader._validate_file_path", return_value=Path("some.pdf")):
            load_from_pdf(Path("some.pdf"), table_settings=settings)

    mock_page.extract_tables.assert_called_once_with(settings)


def test_load_from_pdf_file_not_found(mock_path_exists, mock_path_is_file):
    """Prueba que devuelva FAILED si el PDF no existe."""
    mock_path_exists.return_value = False
    mock_path_is_file.return_value = False

    with patch("app.data_loader._validate_file_path", side_effect=FileNotFoundError("Archivo no encontrado")):
        result = load_from_pdf(Path("missing.pdf"))
        assert result.status == LoadStatus.FAILED


# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA LA FUNCIÓN FACTORY: load_data
# ─────────────────────────────────────────────────────────────


@patch("app.data_loader.load_from_csv")
def test_load_data_routes_to_csv(
    mock_loader, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open
):
    """Prueba que load_data llame a load_from_csv para archivos .csv."""
    # Setup mock return value to be a valid LoadResult object to avoid formatting errors
    mock_result = LoadResult(
        status=LoadStatus.SUCCESS,
        data=pd.DataFrame(),
        file_metadata=FileMetadata(Path("d.csv"), 100, 0.1, FileFormat.CSV, True, True),
        quality_metrics=DataQualityMetrics(total_rows=10, total_columns=2),
        load_time_seconds=0.5
    )
    mock_loader.return_value = mock_result

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("data.csv"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.CSV,
            exists=True,
            readable=True
        )
        with patch("app.data_loader._validate_file_path", return_value=Path("data.csv")):
            load_data(Path("data.csv"), sep=";")

    mock_loader.assert_called_once()


@patch("app.data_loader.load_from_xlsx")
def test_load_data_routes_to_xlsx(
    mock_loader, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open
):
    """Prueba que load_data llame a load_from_xlsx para archivos .xlsx."""
    mock_result = LoadResult(
        status=LoadStatus.SUCCESS,
        data=pd.DataFrame(),
        file_metadata=FileMetadata(Path("d.xlsx"), 100, 0.1, FileFormat.EXCEL, True, True),
        quality_metrics=DataQualityMetrics(total_rows=10, total_columns=2),
        load_time_seconds=0.5
    )
    mock_loader.return_value = mock_result

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("data.xlsx"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.EXCEL,
            exists=True,
            readable=True
        )
        with patch("app.data_loader._validate_file_path", return_value=Path("data.xlsx")):
            load_data(Path("data.xlsx"), sheet_name=0)

    mock_loader.assert_called_once()


@patch("app.data_loader.load_from_pdf")
def test_load_data_routes_to_pdf(
    mock_loader, mock_path_exists, mock_path_is_file, mock_path_stat, mock_file_open
):
    """Prueba que load_data llame a load_from_pdf para archivos .pdf."""
    mock_result = LoadResult(
        status=LoadStatus.SUCCESS,
        data=pd.DataFrame(),
        file_metadata=FileMetadata(Path("d.pdf"), 100, 0.1, FileFormat.PDF, True, True),
        quality_metrics=DataQualityMetrics(total_rows=10, total_columns=2),
        load_time_seconds=0.5
    )
    mock_loader.return_value = mock_result

    with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
        mock_meta.return_value = FileMetadata(
            path=Path("report.pdf"),
            size_bytes=1024,
            size_mb=0.001,
            format=FileFormat.PDF,
            exists=True,
            readable=True
        )
        with patch("app.data_loader._validate_file_path", return_value=Path("report.pdf")):
            load_data(Path("report.pdf"), page_range=range(1))

    mock_loader.assert_called_once()


def test_load_data_unsupported_format(mock_path_exists, mock_path_is_file, mock_path_stat):
    """Prueba que devuelva LoadResult con error para formatos no soportados."""
    path = Path("data.xyz")

    with patch("app.data_loader._validate_file_path", return_value=path):
        with patch("app.data_loader.FileMetadata.from_path") as mock_meta:
            mock_meta.return_value = FileMetadata(
                path=path,
                size_bytes=1024,
                size_mb=0.001,
                format=FileFormat.UNKNOWN,
                exists=True,
                readable=True
            )
            result = load_data(path)

    assert result.status == LoadStatus.FAILED
    assert "Formato de archivo no soportado" in str(result.error_message)


def test_load_data_file_not_found(mock_path_exists, mock_path_is_file):
    """Prueba que load_data devuelva FAILED cuando el archivo no existe."""
    with patch("app.data_loader._validate_file_path", side_effect=FileNotFoundError("Archivo no encontrado")):
        result = load_data(Path("missing.any"))

    assert result.status == LoadStatus.FAILED
    assert "Archivo no encontrado" in str(result.error_message)
