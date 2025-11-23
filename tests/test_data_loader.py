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
def mock_path_resolve():
    """Fixture for Path.resolve."""
    with patch("pathlib.Path.resolve") as mock_resolve:
        # Default behavior: return the path itself (as a mock that acts like a Path)
        def side_effect():
            return MagicMock(
                spec=Path,
                exists=lambda: True,
                is_file=lambda: True,
                stat=lambda: MagicMock(st_size=1024, st_mode=0o100444),
                suffix=".csv",
            )

        mock_resolve.side_effect = side_effect
        yield mock_resolve


# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA load_from_csv
# ─────────────────────────────────────────────────────────────


@patch("pandas.read_csv")
def test_load_from_csv_success(
    mock_read_csv, mock_path_exists, mock_path_is_file, mock_path_stat
):
    """Prueba la carga exitosa de un archivo CSV."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_read_csv.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    # Use a real Path object, mocks will intercept calls on it
    path = Path("data.csv")
    # We need to mock resolve specifically for the logic inside _validate_file_path
    with patch.object(Path, "resolve", return_value=path):
        result = load_from_csv(path, sep=";", encoding="utf-8")

    assert result.status == LoadStatus.SUCCESS
    assert not result.data.empty
    assert result.data.shape == (2, 2)


def test_load_from_csv_file_not_found(mock_path_exists, mock_path_is_file):
    """Prueba que load_from_csv maneje FileNotFoundError internamente o lo propague."""
    # In data_loader.py, load_from_csv calls _validate_file_path which raises FileNotFoundError.
    # load_from_csv DOES NOT wrap _validate_file_path in try-except, so it raises.
    mock_path_exists.return_value = False
    mock_path_is_file.return_value = False

    with patch.object(Path, "resolve", return_value=Path("missing.csv")):
        with pytest.raises(FileNotFoundError, match="Archivo no encontrado"):
            load_from_csv(Path("missing.csv"))


@patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError)
def test_load_from_csv_empty_error(
    mock_read_csv, mock_path_exists, mock_path_is_file, mock_path_stat
):
    """Prueba el manejo de un archivo CSV vacío que lanza EmptyDataError."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True

    path = Path("empty.csv")
    with patch.object(Path, "resolve", return_value=path):
        result = load_from_csv(path)

    assert result.status == LoadStatus.EMPTY
    assert "EmptyDataError" in str(result.error_message) or result.quality_metrics is None


@patch("pandas.read_csv", side_effect=UnicodeDecodeError("utf-8", b"\\xff", 0, 1, "invalid"))
def test_load_from_csv_encoding_error(
    mock_read_csv, mock_path_exists, mock_path_is_file, mock_path_stat
):
    """Prueba el manejo de errores de codificación en CSV."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True

    path = Path("bad_encoding.csv")
    with patch.object(Path, "resolve", return_value=path):
        # load_from_csv tries multiple encodings. If all fail, it returns FAILED.
        result = load_from_csv(path, encoding="utf-8")

    assert result.status == LoadStatus.FAILED
    assert "No se pudo cargar el CSV" in str(result.error_message)


# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA load_from_xlsx
# ─────────────────────────────────────────────────────────────


@patch("pandas.read_excel")
def test_load_from_xlsx_single_sheet_success(
    mock_read_excel, mock_path_exists, mock_path_is_file, mock_path_stat
):
    """Prueba la carga exitosa de una única hoja de Excel."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_read_excel.return_value = pd.DataFrame({"X": [10], "Y": [20]})

    path = Path("data.xlsx")
    with patch.object(Path, "resolve", return_value=path):
        result = load_from_xlsx(path, sheet_name="Hoja1")

    assert result.status == LoadStatus.SUCCESS
    assert result.data.shape == (1, 2)


@patch("pandas.read_excel")
def test_load_from_xlsx_all_sheets_success(
    mock_read_excel, mock_path_exists, mock_path_is_file, mock_path_stat
):
    """Prueba que devuelva un dict de DataFrames cuando sheet_name es None."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    sheets = {"Sheet1": pd.DataFrame({"A": [1]}), "Sheet2": pd.DataFrame({"B": [2]})}
    mock_read_excel.return_value = sheets

    path = Path("data.xlsx")
    with patch.object(Path, "resolve", return_value=path):
        result = load_from_xlsx(path, sheet_name=None)

    assert result.status == LoadStatus.SUCCESS
    assert isinstance(result.data, dict)
    assert len(result.data) == 2


def test_load_from_xlsx_file_not_found(mock_path_exists, mock_path_is_file):
    """Prueba que se lance FileNotFoundError si el archivo Excel no existe."""
    mock_path_exists.return_value = False
    mock_path_is_file.return_value = False

    with patch.object(Path, "resolve", return_value=Path("missing.xlsx")):
        with pytest.raises(FileNotFoundError):
            load_from_xlsx(Path("missing.xlsx"))


@patch("pandas.read_excel", side_effect=ValueError("No such sheet"))
def test_load_from_xlsx_sheet_not_found(
    mock_read_excel, mock_path_exists, mock_path_is_file, mock_path_stat
):
    """Prueba el manejo de una hoja de Excel inexistente."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True

    path = Path("data.xlsx")
    with patch.object(Path, "resolve", return_value=path):
        result = load_from_xlsx(path, sheet_name="NoExiste")

    assert result.status == LoadStatus.FAILED
    assert "Hoja 'NoExiste' no encontrada" in str(result.error_message)


# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA load_from_pdf
# ─────────────────────────────────────────────────────────────


@patch("pdfplumber.open")
def test_load_from_pdf_no_tables(
    mock_pdf_open, mock_path_exists, mock_path_is_file, mock_path_stat, caplog
):
    """Prueba un PDF sin tablas; debe devolver un DataFrame vacío y loguear."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True
    mock_page = MagicMock()
    mock_page.extract_tables.return_value = []
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf

    path = Path("no_tables.pdf")
    with patch.object(Path, "resolve", return_value=path):
        result = load_from_pdf(path)

    assert result.status == LoadStatus.EMPTY
    assert result.data.empty


@patch("pdfplumber.open")
def test_load_from_pdf_with_dirty_tables(
    mock_pdf_open, mock_path_exists, mock_path_is_file, mock_path_stat
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

    path = Path("dirty_tables.pdf")
    with patch.object(Path, "resolve", return_value=path):
        result = load_from_pdf(path)

    assert result.status == LoadStatus.SUCCESS
    assert result.data.shape == (3, 3)


@patch("pdfplumber.open")
def test_load_from_pdf_with_table_settings(
    mock_pdf_open, mock_path_exists, mock_path_is_file, mock_path_stat
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
    path = Path("some.pdf")
    with patch.object(Path, "resolve", return_value=path):
        load_from_pdf(path, table_settings=settings)

    mock_page.extract_tables.assert_called_once_with(settings)


def test_load_from_pdf_file_not_found(mock_path_exists, mock_path_is_file):
    """Prueba que se lance FileNotFoundError si el PDF no existe."""
    mock_path_exists.return_value = False
    mock_path_is_file.return_value = False

    with patch.object(Path, "resolve", return_value=Path("missing.pdf")):
        with pytest.raises(FileNotFoundError):
            load_from_pdf(Path("missing.pdf"))


# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA LA FUNCIÓN FACTORY: load_data
# ─────────────────────────────────────────────────────────────


@patch("app.data_loader.load_from_csv")
def test_load_data_routes_to_csv(
    mock_loader, mock_path_exists, mock_path_is_file, mock_path_stat
):
    """Prueba que load_data llame a load_from_csv para archivos .csv."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True

    path = Path("data.csv")
    with patch.object(Path, "resolve", return_value=path):
        load_data(path, sep=";")

    mock_loader.assert_called_once()


@patch("app.data_loader.load_from_xlsx")
def test_load_data_routes_to_xlsx(
    mock_loader, mock_path_exists, mock_path_is_file, mock_path_stat
):
    """Prueba que load_data llame a load_from_xlsx para archivos .xlsx."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True

    path = Path("data.xlsx")
    with patch.object(Path, "resolve", return_value=path):
        load_data(path, sheet_name=0)

    mock_loader.assert_called_once()


@patch("app.data_loader.load_from_pdf")
def test_load_data_routes_to_pdf(
    mock_loader, mock_path_exists, mock_path_is_file, mock_path_stat
):
    """Prueba que load_data llame a load_from_pdf para archivos .pdf."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True

    path = Path("report.pdf")
    with patch.object(Path, "resolve", return_value=path):
        load_data(path, page_range=range(1))

    mock_loader.assert_called_once()


def test_load_data_unsupported_format(mock_path_exists, mock_path_is_file, mock_path_stat):
    """Prueba que devuelva LoadResult con error para formatos no soportados."""
    mock_path_exists.return_value = True
    mock_path_is_file.return_value = True

    path = Path("data.xyz")
    with patch.object(Path, "resolve", return_value=path):
        result = load_data(path)

    assert result.status == LoadStatus.FAILED
    assert "Formato de archivo no soportado" in str(result.error_message)


def test_load_data_file_not_found(mock_path_exists, mock_path_is_file):
    """Prueba que load_data devuelva FAILED cuando el archivo no existe."""
    mock_path_exists.return_value = False
    mock_path_is_file.return_value = False

    with patch.object(Path, "resolve", return_value=Path("missing.any")):
        result = load_data(Path("missing.any"))

    assert result.status == LoadStatus.FAILED
    assert "Archivo no encontrado" in str(result.error_message)
