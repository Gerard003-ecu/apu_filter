import os
import sys
import unittest

# Añadir el directorio raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.report_parser import ReportParser

# Datos de prueba para el parser, utilizando formato delimitado por punto y coma.
APUS_TEST_DATA = """
ITEM: 1,1
CONSTRUCCION DE MURO EN LADRILLO ESTRUCTURAL
MATERIALES;;;;;
LAMINA DE 1.22 X 2.44 EN 6MM RH;M2;0,0420;-;37.000,00;1.554,00
PERFIL TUBULAR CUADRADO 2" X 2";ML;1,5000;;12.000,00;18.000,00
MANO DE OBRA;;;;;
AYUDANTE;HR;1,0;;10.000,00;10.000,00
OFICIAL;HR;1,0;;15.000,00;15.000,00
EQUIPO;;;;;
HERRAMIENTA MENOR;%;0.05;-;18000;900,00
ITEM: 1,2
PUNTO HIDRAULICO AGUA FRIA/CALIENTE
MATERIALES;;;;;
SOLDADURA PVC 1/4 GAL;UND;0,0200;-;50.000,00;1.000,00
INSUMO CON NUMERO GRANDE;UND;2,0;-;1 250 500,50;2 501 001,00
"""


class TestReportParserRefactored(unittest.TestCase):
    """
    Pruebas unitarias para la clase ReportParser refactorizada.
    """

    @classmethod
    def setUpClass(cls):
        """
        Configura el entorno de prueba una vez para toda la clase.
        Crea un archivo temporal con datos y una instancia del ReportParser.
        """
        cls.test_file_path = "test_apus_refactored.txt"
        with open(cls.test_file_path, "w", encoding="latin1") as f:
            f.write(APUS_TEST_DATA)

        cls.parser = ReportParser(cls.test_file_path)
        cls.df = cls.parser.parse()

        # Imprimir el DataFrame para depuración si es necesario
        print("\nDataFrame parseado para pruebas:")
        print(cls.df.to_string())

    @classmethod
    def tearDownClass(cls):
        """
        Limpia el entorno de prueba eliminando el archivo temporal.
        """
        os.remove(cls.test_file_path)

    def test_dataframe_not_empty(self):
        """Verifica que el DataFrame no está vacío."""
        self.assertFalse(self.df.empty, "El DataFrame no debería estar vacío.")

    def test_finds_correct_number_of_apus(self):
        """
        Verifica que el parser identifica el número correcto de APUs únicos.
        """
        apu_codes = self.df["CODIGO_APU"].unique()
        self.assertEqual(len(apu_codes), 2, "Deberían encontrarse exactamente 2 APUs.")
        self.assertIn("1,1", apu_codes)
        self.assertIn("1,2", apu_codes)

    def test_parses_insumo_with_hyphen_as_waste(self):
        """
        Verifica que un insumo con un guion en la columna de desperdicio
        se parsea correctamente.
        """
        insumo = self.df[self.df["DESCRIPCION_INSUMO"] == "LAMINA DE 1.22 X 2.44 EN 6MM RH"]
        self.assertEqual(len(insumo), 1, "Debería encontrarse un solo insumo de 'LAMINA'.")

        insumo_data = insumo.iloc[0]
        self.assertEqual(insumo_data["CODIGO_APU"], "1,1")
        self.assertEqual(insumo_data["UNIDAD_INSUMO"], "M2")
        self.assertAlmostEqual(insumo_data["CANTIDAD_APU"], 0.0420, places=4)
        self.assertAlmostEqual(insumo_data["PRECIO_UNIT_APU"], 37000.00, places=2)
        self.assertAlmostEqual(insumo_data["VALOR_TOTAL_APU"], 1554.00, places=2)
        self.assertEqual(insumo_data["CATEGORIA"], "MATERIALES")

    def test_parses_insumo_without_waste_column(self):
        """
        Verifica que un insumo sin la columna de desperdicio se parsea correctamente.
        """
        insumo = self.df[self.df["DESCRIPCION_INSUMO"] == 'PERFIL TUBULAR CUADRADO 2" X 2"']
        self.assertEqual(len(insumo), 1, "Debería encontrarse un solo insumo de 'PERFIL'.")

        insumo_data = insumo.iloc[0]
        self.assertEqual(insumo_data["CODIGO_APU"], "1,1")
        self.assertEqual(insumo_data["UNIDAD_INSUMO"], "ML")
        self.assertAlmostEqual(insumo_data["CANTIDAD_APU"], 1.5000, places=4)
        self.assertAlmostEqual(insumo_data["PRECIO_UNIT_APU"], 12000.00, places=2)
        self.assertAlmostEqual(insumo_data["VALOR_TOTAL_APU"], 18000.00, places=2)
        self.assertEqual(insumo_data["CATEGORIA"], "MATERIALES")

    def test_parses_number_with_spaces(self):
        """
        Verifica que un número con espacios como separadores de miles
        se convierte correctamente.
        """
        insumo = self.df[self.df["DESCRIPCION_INSUMO"] == "INSUMO CON NUMERO GRANDE"]
        self.assertEqual(len(insumo), 1, "Debería encontrarse el insumo con número grande.")

        insumo_data = insumo.iloc[0]
        self.assertEqual(insumo_data["CODIGO_APU"], "1,2")
        self.assertAlmostEqual(insumo_data["PRECIO_UNIT_APU"], 1250500.50, places=2)
        self.assertAlmostEqual(insumo_data["VALOR_TOTAL_APU"], 2501001.00, places=2)

    def test_assigns_correct_apu_and_category(self):
        """
        Verifica que los insumos se asignan al APU y categoría correctos.
        """
        insumo_soldadura = self.df[self.df["DESCRIPCION_INSUMO"] == "SOLDADURA PVC 1/4 GAL"]
        self.assertEqual(
            len(insumo_soldadura), 1, "Debería encontrarse un insumo de 'SOLDADURA PVC'."
        )

        insumo_data = insumo_soldadura.iloc[0]
        self.assertEqual(
            insumo_data["CODIGO_APU"], "1,2", "El insumo debería pertenecer al APU '1,2'."
        )
        self.assertEqual(
            insumo_data["CATEGORIA"], "MATERIALES", "La categoría debería ser 'MATERIALES'."
        )

    def test_parses_herramienta_menor(self):
        """
        Verifica que la línea de 'HERRAMIENTA MENOR' se parsea correctamente.
        """
        herramienta = self.df[self.df["DESCRIPCION_INSUMO"] == "HERRAMIENTA MENOR"]
        self.assertEqual(
            len(herramienta), 1, "Debería encontrarse un insumo de 'HERRAMIENTA MENOR'."
        )

        herramienta_data = herramienta.iloc[0]
        self.assertEqual(herramienta_data["CATEGORIA"], "EQUIPO")
        self.assertAlmostEqual(herramienta_data["VALOR_TOTAL_APU"], 900.00, places=2)


if __name__ == "__main__":
    unittest.main()
