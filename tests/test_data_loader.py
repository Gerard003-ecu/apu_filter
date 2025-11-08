# app/tests/test_data_loader.py
"""
Pruebas unitarias para el módulo data_loader.py.

Cubre casos de éxito, manejo de errores, validaciones y comportamiento
con diferentes formatos (CSV, Excel, PDF).
"""

import unittest
from unittest import mock
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import logging
from pathlib import Path
import tempfile
import os

# Importar el módulo bajo prueba
from ..data_loader import (
    load_from_csv,
    load_from_xlsx,
    load_from_pdf,
    load_data,
)

# Para evitar logs molestos durante las pruebas
logging.disable(logging.CRITICAL)


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        """Configuración común antes de cada prueba."""
        self.valid_path = Path("dummy.csv")
        self.nonexistent_path = Path("not_exists.csv")

    # ─────────────────────────────────────────────────────────────
    # TEST: load_from_csv
    # ─────────────────────────────────────────────────────────────

    @patch("pandas.read_csv")
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_from_csv_success(self, mock_exists, mock_read_csv):
        """Prueba carga exitosa de CSV."""
        mock_read_csv.return_value = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df = load_from_csv("data.csv", sep=";", encoding="utf-8")
        mock_read_csv.assert_called_once_with(
            "data.csv", sep=";", encoding="utf-8"
        )
        self.assertFalse(df.empty)
        self.assertEqual(df.shape, (2, 2))

    @patch("pathlib.Path.exists", return_value=False)
    def test_load_from_csv_file_not_found(self):
        """Prueba que se lance FileNotFoundError si el archivo no existe."""
        with self.assertRaises(FileNotFoundError):
            load_from_csv("missing.csv")

    @patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError("Empty"))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_from_csv_empty(self, mock_exists, mock_read_csv):
        """Prueba manejo de archivo CSV vacío."""
        with self.assertRaises(pd.errors.EmptyDataError):
            load_from_csv("empty.csv")

    @patch("pandas.read_csv", side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_from_csv_encoding_error(self, mock_exists, mock_read_csv):
        """Prueba manejo de error de codificación."""
        with self.assertRaises(UnicodeDecodeError):
            load_from_csv("bad_encoding.csv", encoding="utf-8")

    # ─────────────────────────────────────────────────────────────
    # TEST: load_from_xlsx
    # ─────────────────────────────────────────────────────────────

    @patch("pandas.read_excel")
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_from_xlsx_success(self, mock_exists, mock_read_excel):
        """Prueba carga exitosa de Excel (hoja única)."""
        mock_read_excel.return_value = pd.DataFrame({"X": [10], "Y": [20]})
        df = load_from_xlsx("data.xlsx", sheet_name="Hoja1")
        mock_read_excel.assert_called_once_with(
            "data.xlsx", sheet_name="Hoja1"
        )
        self.assertEqual(df.shape, (1, 2))

    @patch("pandas.read_excel", return_value={"Sheet1": pd.DataFrame({"A": [1]})})
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_from_xlsx_multiple_sheets_returns_first(self, mock_exists, mock_read_excel):
        """Si se devuelven múltiples hojas, debe tomar la primera."""
        with self.assertLogs("app.data_loader", level="WARNING") as log:
            df = load_from_xlsx("data.xlsx")
            self.assertIn("múltiples hojas", log.output[0].lower())
        self.assertEqual(df.shape, (1, 1))

    @patch("pathlib.Path.exists", return_value=False)
    def test_load_from_xlsx_file_not_found(self):
        """Prueba que se lance FileNotFoundError si el archivo Excel no existe."""
        with self.assertRaises(FileNotFoundError):
            load_from_xlsx("missing.xlsx")

    @patch("pandas.read_excel", side_effect=ValueError("No such sheet"))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_from_xlsx_sheet_not_found(self, mock_exists, mock_read_excel):
        """Prueba manejo de hoja inexistente."""
        with self.assertRaises(ValueError):
            load_from_xlsx("data.xlsx", sheet_name="NoExiste")

    # ─────────────────────────────────────────────────────────────
    # TEST: load_from_pdf
    # ─────────────────────────────────────────────────────────────

    @patch("pdfplumber.open")
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_from_pdf_no_tables(self, mock_exists, mock_pdf_open):
        """Prueba PDF sin tablas: debe devolver DataFrame vacío."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(), MagicMock()]  # 2 páginas
        for page in mock_pdf.pages:
            page.extract_tables.return_value = []
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        with self.assertLogs("app.data_loader", level="WARNING") as log:
            df = load_from_pdf("no_tables.pdf")
            self.assertIn("no se encontraron tablas", log.output[0].lower())
        self.assertTrue(df.empty)

    @patch("pdfplumber.open")
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_from_pdf_with_tables(self, mock_exists, mock_pdf_open):
        """Prueba extracción exitosa de tablas del PDF."""
        # Simular una tabla: encabezado + 2 filas
        mock_table = [
            ["Nombre", "Edad"],
            ["Alice", "30"],
            ["Bob", "25"]
        ]
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = [mock_table]
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        df = load_from_pdf("with_tables.pdf")
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(list(df.columns), ["Nombre", "Edad"])

    @patch("pdfplumber.open", side_effect=Exception("PDF corrupto"))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_from_pdf_processing_error(self, mock_exists, mock_pdf_open):
        """Prueba manejo de error al abrir el PDF."""
        with self.assertRaises(Exception) as context:
            load_from_pdf("corrupt.pdf")
        self.assertIn("PDF corrupto", str(context.exception))

    @patch("pathlib.Path.exists", return_value=False)
    def test_load_from_pdf_file_not_found(self, mock_exists):
        """Prueba que se lance FileNotFoundError si el PDF no existe."""
        with self.assertRaises(FileNotFoundError):
            load_from_pdf("missing.pdf")

    @patch("pdfplumber.open")
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_from_pdf_with_page_range(self, mock_exists, mock_pdf_open):
        """Prueba que solo se procesen las páginas en el rango dado."""
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(), MagicMock(), MagicMock()]  # 3 páginas
        for page in mock_pdf.pages:
            page.extract_tables.return_value = []

        mock_pdf_open.return_value.__enter__.return_value = mock_pdf

        load_from_pdf("data.pdf", page_range=range(1, 2))  # Solo página 2
        self.assertEqual(len(mock_pdf.pages), 3)
        # Se debe haber procesado solo 1 página
        self.assertEqual(len(mock_pdf.pages[1].extract_tables.call_args_list), 1)
        self.assertEqual(len(mock_pdf.pages[0].extract_tables.call_args_list), 0)
        self.assertEqual(len(mock_pdf.pages[2].extract_tables.call_args_list), 0)

    # ─────────────────────────────────────────────────────────────
    # TEST: load_data (factory)
    # ─────────────────────────────────────────────────────────────

    @patch("app.data_loader.load_from_csv")
    def test_load_data_csv_calls_correct_loader(self, mock_loader):
        """Prueba que .csv llame a load_from_csv."""
        mock_loader.return_value = pd.DataFrame()
        df = load_data("data.csv", sep=";")
        mock_loader.assert_called_once_with("data.csv", sep=";")
        self.assertIsInstance(df, pd.DataFrame)

    @patch("app.data_loader.load_from_xlsx")
    def test_load_data_xlsx_calls_correct_loader(self, mock_loader):
        """Prueba que .xlsx llame a load_from_xlsx."""
        mock_loader.return_value = pd.DataFrame()
        df = load_data("data.xlsx", sheet_name=0)
        mock_loader.assert_called_once_with("data.xlsx", sheet_name=0)

    @patch("app.data_loader.load_from_pdf")
    def test_load_data_pdf_calls_correct_loader(self, mock_loader):
        """Prueba que .pdf llame a load_from_pdf."""
        mock_loader.return_value = pd.DataFrame()
        df = load_data("report.pdf", page_range=range(0, 1))
        mock_loader.assert_called_once_with("report.pdf", page_range=range(0, 1))

    def test_load_data_unsupported_format(self):
        """Prueba que se lance ValueError con extensión no soportada."""
        with self.assertRaises(ValueError) as context:
            load_data("data.txt")
        self.assertIn("no soportado", str(context.exception).lower())

    @patch("pathlib.Path.exists", return_value=False)
    def test_load_data_file_not_found(self, mock_exists):
        """Prueba que se lance FileNotFoundError si el archivo no existe."""
        with self.assertRaises(FileNotFoundError):
            load_data("missing.csv")

    # ─────────────────────────────────────────────────────────────
    # TEST: Casos extremos y tipos de entrada
    # ─────────────────────────────────────────────────────────────

    def test_load_data_with_pathlib_path(self):
        """Prueba que acepte Path de pathlib, no solo str."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"A;B\n1;2")  # Contenido CSV básico
            tmp_path = Path(tmp.name)

        try:
            with patch("pandas.read_csv", return_value=pd.DataFrame({"A": [1], "B": [2]})):
                df = load_data(tmp_path, sep=";")
                self.assertEqual(df.shape, (1, 2))
        finally:
            os.remove(tmp_path)

    def tearDown(self):
        """Limpieza después de cada prueba."""
        pass


if __name__ == "__main__":
    unittest.main()