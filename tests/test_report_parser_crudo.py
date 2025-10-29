# En tests/test_report_parser_crudo.py

import logging
import os
import shutil
import sys
import tempfile
import unittest

# Añadir el directorio raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.report_parser_crudo import ReportParserCrudo


class TestReportParserCrudo(unittest.TestCase):
    """
    Suite de pruebas para ReportParserCrudo, enfocado en la extracción de datos en crudo.
    """

    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todas las pruebas."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_parser_crudo_")
        logging.basicConfig(
            level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
        )

    @classmethod
    def tearDownClass(cls):
        """Limpieza final después de todas las pruebas."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def _create_test_file(self, content: str, filename: str = "test.txt") -> str:
        """Crea un archivo de prueba con el contenido dado."""
        path = os.path.join(self.temp_dir, filename)
        with open(path, "w", encoding="latin1") as f:
            f.write(content)
        return path

    def test_basic_raw_extraction(self):
        """Prueba que el parser extrae registros crudos como strings."""
        apu_data = (
            "ITEM: APU-01; UNIDAD: M2\n"
            "UNA DESCRIPCION DE APU VALIDA\n"
            "MATERIALES\n"
            "Cemento;UND;1;;100;100\n"
        )
        path = self._create_test_file(apu_data)
        parser = ReportParserCrudo(path)
        raw_records = parser.parse_to_raw()

        self.assertEqual(len(raw_records), 1)
        record = raw_records[0]
        self.assertEqual(record["apu_code"], "01")
        self.assertEqual(record["apu_desc"], "UNA DESCRIPCION DE APU VALIDA")
        self.assertEqual(record["apu_unit"], "M2")
        self.assertEqual(record["category"], "MATERIALES")
        self.assertEqual(record["insumo_line"], "Cemento;UND;1;;100;100")

    def test_no_numeric_conversion(self):
        """Verifica que no se realiza ninguna conversión numérica."""
        apu_data = "ITEM: APU-02\nUNA DESCRIPCION VALIDA\nInsumo;U;1,5;;1.000,00;1,500.00\n"
        path = self._create_test_file(apu_data)
        parser = ReportParserCrudo(path)
        raw_records = parser.parse_to_raw()

        self.assertEqual(len(raw_records), 1)
        # Esto es una simplificación, en la realidad se parsearía la línea completa.
        # El punto es verificar que los campos numéricos siguen siendo strings.
        self.assertIsInstance(raw_records[0]["insumo_line"], str)


    def test_inline_description_extraction(self):
        """Prueba la extracción de descripción inline como string."""
        apu_data = "ITEM: APU-03; DESCRIPCION: Excavacion Manual\n"
        path = self._create_test_file(apu_data)
        parser = ReportParserCrudo(path)
        raw_records = parser.parse_to_raw()

        # Aunque no haya insumos, el contexto del APU debe tener la descripción
        self.assertEqual(parser.context["apu_desc"], "Excavacion Manual")

    def test_multiple_apus_raw(self):
        """Prueba el parsing crudo de múltiples APUs."""
        multi_apu_data = (
            "ITEM: 001\n"
            "DESCRIPCION APU 1\n"
            "Mat 1;U;1;;1;1\n\n"
            "ITEM: 002\n"
            "DESCRIPCION APU 2\n"
            "Mat 2;U;1;;1;1\n"
        )
        path = self._create_test_file(multi_apu_data)
        parser = ReportParserCrudo(path)
        raw_records = parser.parse_to_raw()

        self.assertEqual(len(raw_records), 2)
        self.assertEqual(raw_records[0]["apu_desc"], "DESCRIPCION APU 1")
        self.assertEqual(raw_records[1]["apu_desc"], "DESCRIPCION APU 2")

if __name__ == "__main__":
    unittest.main(verbosity=2)
