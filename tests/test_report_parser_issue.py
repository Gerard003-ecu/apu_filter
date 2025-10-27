import os
import shutil
import sys
import tempfile
import unittest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.report_parser import ReportParser
from tests.test_data import APUS_DATA  # Importar los nuevos datos


class TestReportParserWithNewData(unittest.TestCase):
    """
    Suite de pruebas para ReportParser utilizando los nuevos datos de prueba centralizados.
    """

    @classmethod
    def setUpClass(cls):
        """
        Configura el entorno de prueba creando un archivo APU temporal
        con los nuevos datos.
        """
        # Crear un directorio temporal para los archivos de prueba
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_file_path = os.path.join(cls.temp_dir, "test_apus.csv")

        # Escribir los nuevos datos de APUS_DATA en el archivo temporal
        with open(cls.test_file_path, "w", encoding="latin1") as f:
            f.write(APUS_DATA)

        # Crear una instancia del parser y analizar el archivo
        cls.parser = ReportParser(cls.test_file_path)
        cls.df = cls.parser.parse()

        if cls.df.empty:
            raise ValueError("El análisis de los datos de prueba no produjo resultados.")

    @classmethod
    def tearDownClass(cls):
        """Limpia el directorio temporal después de las pruebas."""
        shutil.rmtree(cls.temp_dir)

    def test_dataframe_is_not_empty(self):
        """Verifica que el DataFrame resultante no esté vacío."""
        self.assertFalse(self.df.empty, "El DataFrame no debería estar vacío.")

    def test_finds_correct_number_of_apus(self):
        """
        Verifica que el parser identifica el número correcto de APUs únicos
        basado en los nuevos datos de prueba.
        """
        expected_apus = ["1.1", "1.2", "2.1", "3.1"]
        actual_apus = self.df["CODIGO_APU"].unique()
        self.assertCountEqual(
            actual_apus,
            expected_apus,
            f"Se esperaban los APUs {expected_apus} pero se encontraron {actual_apus}",
        )

    def test_specific_insumo_is_parsed_correctly(self):
        """
        Prueba que un insumo específico de los nuevos datos de prueba
        (TEJA TRAPEZOIDAL ROJA) se analiza correctamente.
        """
        # Buscar el insumo específico en el DataFrame
        insumo_desc = "TEJA TRAPEZOIDAL ROJA"
        insumo = self.df[self.df["DESCRIPCION_INSUMO"] == insumo_desc]

        # Verificar que el insumo fue encontrado
        self.assertFalse(
            insumo.empty, f"El insumo '{insumo_desc}' no fue encontrado."
        )

        # Realizar aserciones sobre los datos del insumo
        insumo_data = insumo.iloc[0]
        self.assertEqual(insumo_data["UNIDAD_INSUMO"], "M2")
        self.assertAlmostEqual(insumo_data["CANTIDAD_APU"], 1.05, places=2)
        self.assertAlmostEqual(insumo_data["PRECIO_UNIT_APU"], 47619, places=2)
        self.assertAlmostEqual(insumo_data["VALOR_TOTAL_APU"], 50000, places=2)
        self.assertEqual(insumo_data["CATEGORIA"], "MATERIALES")


if __name__ == "__main__":
    unittest.main()
