import os
import unittest

from app.report_parser import ReportParser

# Datos de prueba representativos simulando el contenido de un archivo de APUs.
APUS_TEST_DATA = """
ITEM: 1,1
CONSTRUCCION DE MURO EN LADRILLO ESTRUCTURAL

MATERIALES
LAMINA DE 1.22 X 2.44 EN 6MM RH             M2      0,0420       37.000,00           1.554,00
PERFIL TUBULAR CUADRADO 2" X 2"             ML      1,5000       12.000,00          18.000,00

MANO DE OBRA
AYUDANTE                                    HR      1,0000       10.000,00          10.000,00
OFICIAL                                     HR      1,0000       15.000,00          15.000,00

EQUIPO Y HERRAMIENTA
EQUIPO Y HERRAMIENTA (MANO DE OBRA) 5% 1.250,00

ITEM: 1,2
PUNTO HIDRAULICO AGUA FRIA/CALIENTE
MATERIALES
SOLDADURA PVC 1/4 GAL                       UND     0,0200       50.000,00           1.000,00
LIMPIADOR PVC 1/4 GAL                       UND     0,0200       40.000,00             800,00
"""


class TestReportParser(unittest.TestCase):
    """
    Pruebas unitarias para la clase ReportParser.
    """

    def setUp(self):
        """
        Configura el entorno de prueba. Crea un archivo temporal con datos
        y una instancia del ReportParser.
        """
        self.test_file_path = "test_apus.txt"
        with open(self.test_file_path, "w", encoding="utf-8") as f:
            f.write(APUS_TEST_DATA)
        self.parser = ReportParser(self.test_file_path)
        self.df = self.parser.parse()

    def tearDown(self):
        """
        Limpia el entorno de prueba eliminando el archivo temporal.
        """
        os.remove(self.test_file_path)

    def test_finds_correct_number_of_apus(self):
        """
        Verifica que el parser identifica el número correcto de APUs únicos.
        """
        apu_codes = self.df["apu_code"].unique()
        self.assertEqual(len(apu_codes), 2, "Deberían encontrarse exactamente 2 APUs.")
        self.assertIn("1.1", apu_codes)
        self.assertIn("1.2", apu_codes)

    def test_parses_standard_insumo_correctly(self):
        """
        Verifica que un insumo estándar se parsea con los valores correctos.
        """
        # Buscar el insumo específico
        insumo = self.df[self.df["descripcion"].str.contains("lamina de 1.22")]
        self.assertEqual(len(insumo), 1, "Debería encontrarse un solo insumo de 'lamina'.")

        # Extraer la fila de datos
        insumo_data = insumo.iloc[0]

        # Verificar los valores numéricos
        self.assertAlmostEqual(insumo_data["cantidad"], 0.0420, places=4)
        self.assertAlmostEqual(insumo_data["precio_unitario"], 37000.00, places=2)
        self.assertAlmostEqual(insumo_data["precio_total"], 1554.00, places=2)
        self.assertEqual(insumo_data["unidad"], "M2")
        self.assertEqual(insumo_data["apu_code"], "1.1")

    def test_parses_special_case_herramienta_menor(self):
        """
        Verifica que el caso especial de 'Herramienta Menor' se parsea correctamente.
        """
        herramienta = self.df[self.df["descripcion"] == "equipo y herramienta (5%)"]
        self.assertEqual(
            len(herramienta), 1, "Debería encontrarse un solo insumo de 'Herramienta Menor'."
        )

        herramienta_data = herramienta.iloc[0]

        self.assertEqual(herramienta_data["categoria"], "EQUIPO Y HERRAMIENTA")
        self.assertEqual(herramienta_data["unidad"], "%")
        self.assertAlmostEqual(herramienta_data["cantidad"], 5.0, places=1)
        self.assertAlmostEqual(herramienta_data["precio_total"], 1250.00, places=2)
        self.assertEqual(herramienta_data["apu_code"], "1.1")

    def test_assigns_insumos_to_correct_apu_and_category(self):
        """
        Verifica que los insumos se asignan al APU y categoría correctos.
        """
        # Verificar un insumo del segundo APU
        insumo_soldadura = self.df[self.df["descripcion"].str.contains("soldadura pvc")]
        self.assertEqual(
            len(insumo_soldadura), 1, "Debería encontrarse un insumo de 'soldadura pvc'."
        )

        insumo_data = insumo_soldadura.iloc[0]

        self.assertEqual(
            insumo_data["apu_code"], "1.2", "El insumo debería pertenecer al APU '1.2'."
        )
        self.assertEqual(
            insumo_data["categoria"], "MATERIALES", "La categoría debería ser 'MATERIALES'."
        )


if __name__ == "__main__":
    unittest.main()
