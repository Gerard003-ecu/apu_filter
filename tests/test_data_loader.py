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

# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA load_from_csv
# ─────────────────────────────────────────────────────────────

@patch("pandas.read_csv")
def test_load_from_csv_success(mock_read_csv, mock_path_exists):
    """Prueba la carga exitosa de un archivo CSV."""
    mock_path_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    df = load_from_csv("data.csv", sep=";", encoding="utf-8")

    mock_read_csv.assert_called_once_with(Path("data.csv"), sep=";", encoding="utf-8")
    assert not df.empty
    assert df.shape == (2, 2)

def test_load_from_csv_file_not_found(mock_path_exists):
    """Prueba que se lance FileNotFoundError si el CSV no existe."""
    mock_path_exists.return_value = False
    with pytest.raises(FileNotFoundError, match="Archivo no encontrado"):
        load_from_csv("missing.csv")

@patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError)
def test_load_from_csv_empty_error(mock_read_csv, mock_path_exists):
    """Prueba el manejo de un archivo CSV vacío que lanza EmptyDataError."""
    mock_path_exists.return_value = True
    with pytest.raises(pd.errors.EmptyDataError):
        load_from_csv("empty.csv")

@patch("pandas.read_csv", side_effect=UnicodeDecodeError("utf-8", b"\\xff", 0, 1, "invalid"))
def test_load_from_csv_encoding_error(mock_read_csv, mock_path_exists):
    """Prueba el manejo de errores de codificación en CSV."""
    mock_path_exists.return_value = True
    with pytest.raises(UnicodeDecodeError):
        load_from_csv("bad_encoding.csv", encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA load_from_xlsx
# ─────────────────────────────────────────────────────────────

@patch("pandas.read_excel")
def test_load_from_xlsx_single_sheet_success(mock_read_excel, mock_path_exists):
    """Prueba la carga exitosa de una única hoja de Excel."""
    mock_path_exists.return_value = True
    mock_read_excel.return_value = pd.DataFrame({"X": [10], "Y": [20]})

    df = load_from_xlsx("data.xlsx", sheet_name="Hoja1")

    mock_read_excel.assert_called_once_with(Path("data.xlsx"), sheet_name="Hoja1")
    assert df.shape == (1, 2)

@patch("pandas.read_excel")
def test_load_from_xlsx_all_sheets_success(mock_read_excel, mock_path_exists):
    """Prueba que devuelva un dict de DataFrames cuando sheet_name es None."""
    mock_path_exists.return_value = True
    sheets = {
        "Sheet1": pd.DataFrame({"A": [1]}),
        "Sheet2": pd.DataFrame({"B": [2]})
    }
    mock_read_excel.return_value = sheets

    data = load_from_xlsx("data.xlsx", sheet_name=None)

    assert isinstance(data, dict)
    assert len(data) == 2
    assert "Sheet1" in data
    pd.testing.assert_frame_equal(data["Sheet2"], sheets["Sheet2"])

def test_load_from_xlsx_file_not_found(mock_path_exists):
    """Prueba que se lance FileNotFoundError si el archivo Excel no existe."""
    mock_path_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        load_from_xlsx("missing.xlsx")

@patch("pandas.read_excel", side_effect=ValueError("No such sheet"))
def test_load_from_xlsx_sheet_not_found(mock_read_excel, mock_path_exists):
    """Prueba el manejo de una hoja de Excel inexistente."""
    mock_path_exists.return_value = True
    with pytest.raises(ValueError, match="Hoja 'NoExiste' no encontrada"):
        load_from_xlsx("data.xlsx", sheet_name="NoExiste")

# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA load_from_pdf
# ─────────────────────────────────────────────────────────────

@patch("pdfplumber.open")
def test_load_from_pdf_no_tables(mock_pdf_open, mock_path_exists, caplog):
    """Prueba un PDF sin tablas; debe devolver un DataFrame vacío y loguear."""
    mock_path_exists.return_value = True
    mock_page = MagicMock()
    mock_page.extract_tables.return_value = []
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf

    df = load_from_pdf("no_tables.pdf")

    assert df.empty
    assert "No se encontraron tablas en el PDF" in caplog.text

@patch("pdfplumber.open")
def test_load_from_pdf_with_dirty_tables(mock_pdf_open, mock_path_exists):
    """Prueba la extracción de tablas 'sucias' (con Nones y longitudes variables)."""
    mock_path_exists.return_value = True
    mock_table = [
        ["Nombre", "Edad", "Ciudad"],
        ["Alice", "30", "Madrid"],
        ["Bob", None],  # Fila con longitud inconsistente
        ["Charlie", "25", "Barcelona"]
    ]
    mock_page = MagicMock()
    mock_page.extract_tables.return_value = [mock_table]
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf

    df = load_from_pdf("dirty_tables.pdf")

    assert df.shape == (3, 3)  # 3 filas de datos, 3 columnas
    assert list(df.columns) == ["Nombre", "Edad", "Ciudad"]
    # Verifica que pandas haya llenado los valores faltantes con NaN (o None)
    assert pd.isna(df.iloc[1, 2])

@patch("pdfplumber.open")
def test_load_from_pdf_with_table_settings(mock_pdf_open, mock_path_exists):
    """Prueba que los 'table_settings' se pasen correctamente a extract_tables."""
    mock_path_exists.return_value = True
    mock_page = MagicMock()
    mock_page.extract_tables.return_value = []
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf

    settings = {"vertical_strategy": "lines"}
    load_from_pdf("some.pdf", table_settings=settings)

    mock_page.extract_tables.assert_called_once_with(settings)

def test_load_from_pdf_file_not_found(mock_path_exists):
    """Prueba que se lance FileNotFoundError si el PDF no existe."""
    mock_path_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        load_from_pdf("missing.pdf")

# ─────────────────────────────────────────────────────────────
# PRUEBAS PARA LA FUNCIÓN FACTORY: load_data
# ─────────────────────────────────────────────────────────────

@patch("app.data_loader.load_from_csv")
def test_load_data_routes_to_csv(mock_loader, mock_path_exists):
    """Prueba que load_data llame a load_from_csv para archivos .csv."""
    mock_path_exists.return_value = True
    load_data(Path("data.csv"), sep=";")
    mock_loader.assert_called_once_with(Path("data.csv"), sep=";")

@patch("app.data_loader.load_from_xlsx")
def test_load_data_routes_to_xlsx(mock_loader, mock_path_exists):
    """Prueba que load_data llame a load_from_xlsx para archivos .xlsx."""
    mock_path_exists.return_value = True
    load_data(Path("data.xlsx"), sheet_name=0)
    mock_loader.assert_called_once_with(Path("data.xlsx"), sheet_name=0)

@patch("app.data_loader.load_from_pdf")
def test_load_data_routes_to_pdf(mock_loader, mock_path_exists):
    """Prueba que load_data llame a load_from_pdf para archivos .pdf."""
    mock_path_exists.return_value = True
    load_data(Path("report.pdf"), page_range=range(1))
    mock_loader.assert_called_once_with(Path("report.pdf"), page_range=range(1))

def test_load_data_unsupported_format(mock_path_exists):
    """Prueba que se lance ValueError para formatos de archivo no soportados."""
    mock_path_exists.return_value = True
    with pytest.raises(ValueError, match="Formato de archivo no soportado"):
        load_data(Path("data.txt"))

def test_load_data_file_not_found(mock_path_exists):
    """Prueba que la función factory lance FileNotFoundError."""
    mock_path_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        load_data(Path("missing.any"))
