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
    Pruebas unitarias para la clase `ReportParserCrudo`.

    Esta suite de pruebas se enfoca en la primera etapa del procesamiento: la
    extracción de datos en crudo de un archivo de reporte. Las pruebas validan
    que el parser pueda identificar correctamente los APUs, sus descripciones,
    unidades, categorías y líneas de insumos, manteniendo toda la información
    como cadenas de texto sin aplicar lógica de negocio.
    """

    @classmethod
    def setUpClass(cls):
        """
        Configura un directorio temporal para almacenar los archivos de prueba
        generados durante la ejecución.
        """
        cls.temp_dir = tempfile.mkdtemp(prefix="test_parser_crudo_")
        logging.basicConfig(
            level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
        )

    @classmethod
    def tearDownClass(cls):
        """
        Limpia y elimina el directorio temporal y su contenido al finalizar
        todas las pruebas.
        """
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def _create_test_file(self, content: str, filename: str = "test.txt") -> str:
        """
        Crea un archivo de prueba temporal con el contenido especificado.

        Args:
            content: El contenido de texto que se escribirá en el archivo.
            filename: El nombre del archivo a crear dentro del directorio temporal.

        Returns:
            La ruta completa al archivo creado.
        """
        path = os.path.join(self.temp_dir, filename)
        with open(path, "w", encoding="latin1") as f:
            f.write(content)
        return path

    def test_basic_raw_extraction(self):
        """
        Prueba la extracción básica de un único APU.

        Verifica que el parser pueda extraer correctamente el código, la
        descripción, la unidad, la categoría y la línea de insumo de un APU
        bien formado, manteniendo todos los datos como strings.
        """
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
        """
        Asegura que la extracción cruda no realice ninguna conversión numérica.

        Verifica que los valores que parecen números (con comas o puntos) se
        mantengan como cadenas de texto en el campo `insumo_line`.
        """
        apu_data = "ITEM: APU-02\nUNA DESCRIPCION VALIDA\nInsumo;U;1,5;;1.000,00;1,500.00\n"
        path = self._create_test_file(apu_data)
        parser = ReportParserCrudo(path)
        raw_records = parser.parse_to_raw()

        self.assertEqual(len(raw_records), 1)
        # Esto es una simplificación, en la realidad se parsearía la línea completa.
        # El punto es verificar que los campos numéricos siguen siendo strings.
        self.assertIsInstance(raw_records[0]["insumo_line"], str)


    def test_inline_description_extraction(self):
        """
        Prueba la capacidad del parser para extraer la descripción de un APU
        cuando se encuentra en la misma línea que el "ITEM:".
        """
        apu_data = "ITEM: APU-03; DESCRIPCION: Excavacion Manual\n"
        path = self._create_test_file(apu_data)
        parser = ReportParserCrudo(path)
        parser.parse_to_raw()

        # Aunque no haya insumos, el contexto del APU debe tener la descripción
        self.assertEqual(parser.context["apu_desc"], "Excavacion Manual")

    def test_multiple_apus_raw(self):
        """
        Prueba el correcto procesamiento de un archivo que contiene múltiples
        APUs, asegurando que cada uno se extraiga como un registro independiente.
        """
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
