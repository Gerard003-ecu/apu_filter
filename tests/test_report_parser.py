# En tests/test_report_parser.py

import os
import sys
import unittest
import pandas as pd
import shutil

# Añadir el directorio raíz del proyecto al sys.path para encontrar los módulos de la app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.report_parser import ReportParser

class TestNewReportParser(unittest.TestCase):
    """
    Suite de pruebas exhaustiva para la nueva y robusta implementación de ReportParser.
    """

    @classmethod
    def setUpClass(cls):
        """Crea un directorio temporal para los archivos de prueba."""
        cls.temp_dir = "temp_test_files_for_parser"
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        os.makedirs(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        """Elimina el directorio temporal después de que todas las pruebas han corrido."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def _create_test_file(self, filename: str, content: str) -> str:
        """
        Crea un archivo de prueba en el directorio temporal y devuelve su ruta.
        El contenido se codifica en 'latin1' para simular los archivos reales.
        """
        path = os.path.join(self.temp_dir, filename)
        with open(path, "w", encoding="latin1") as f:
            f.write(content)
        return path

    # Aquí se agregarán los métodos de prueba para cada escenario.

    def test_basic_parsing_and_context_detection(self):
        """
        Prueba el parsing básico y la correcta detección de contexto
        (descripción de APU, unidad y cambio de categoría).
        """
        apu_data = (
            "UNA DESCRIPCION DE APU VALIDA\n"
            "ITEM: APU-01; UNIDAD: M2\n"
            "MATERIALES\n"
            "Cemento;UND;1;;100;100\n"
            "Arena;M3;2;;50;100\n"
            "MANO DE OBRA\n"
            "Oficial;JOR;8;;20;160\n"
        )
        test_file = self._create_test_file("basic_parsing.txt", apu_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 3, "Deberían parsearse 3 insumos.")
        self.assertListEqual(
            list(df.columns),
            ['CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU', 'DESCRIPCION_INSUMO',
             'UNIDAD_INSUMO', 'CANTIDAD_APU', 'PRECIO_UNIT_APU', 'VALOR_TOTAL_APU',
             'CATEGORIA', 'RENDIMIENTO', 'FORMATO_ORIGEN', 'NORMALIZED_DESC']
        )

        # Verificar contexto del APU
        self.assertEqual(df["CODIGO_APU"].iloc[0], "01") # clean_apu_code elimina caracteres no numéricos
        self.assertEqual(df["DESCRIPCION_APU"].iloc[0], "UNA DESCRIPCION DE APU VALIDA")
        self.assertEqual(df["UNIDAD_APU"].iloc[0], "M2")

        # Verificar categorías
        self.assertEqual(df[df["DESCRIPCION_INSUMO"] == "Cemento"]["CATEGORIA"].iloc[0], "MATERIALES")
        self.assertEqual(df[df["DESCRIPCION_INSUMO"] == "Arena"]["CATEGORIA"].iloc[0], "MATERIALES")
        self.assertEqual(df[df["DESCRIPCION_INSUMO"] == "Oficial"]["CATEGORIA"].iloc[0], "MANO DE OBRA")

    def test_mano_de_obra_compleja_logic(self):
        """
        Verifica que la lógica para la mano de obra compleja (formato SAGUT)
        calcule correctamente la cantidad y el rendimiento.
        """
        mo_data = (
            "ITEM: 9901\n"
            "MANO DE OBRA\n"
            # Desc; Jornal Base; Prestaciones; Jornal Total; Rendimiento; Vr. Total
            "OFICIAL DE PRIMERA;80.000;1,75;140.000;0,5;70.000\n"
        )
        test_file = self._create_test_file("mo_compleja.txt", mo_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 1)
        insumo = df.iloc[0]

        # Cantidad = Vr. Total / Jornal Total = 70000 / 140000 = 0.5
        self.assertAlmostEqual(insumo["CANTIDAD_APU"], 0.5, places=4)
        self.assertAlmostEqual(insumo["RENDIMIENTO"], 0.5, places=4)
        self.assertAlmostEqual(insumo["PRECIO_UNIT_APU"], 140000, places=2)
        self.assertEqual(insumo["FORMATO_ORIGEN"], "MO_COMPLEJA")

    def test_mano_de_obra_simple_logic(self):
        """
        Verifica que la lógica para la mano de obra simple (formato CSV)
        calcule correctamente el rendimiento a partir de los valores.
        """
        mo_data = (
            "ITEM: 9902\n"
            "MANO DE OBRA\n"
            # Desc; ; Cantidad; ; Vr. Unitario; Vr. Total
            "AYUDANTE DE OBRA;;1;;120000;15000\n"
        )
        test_file = self._create_test_file("mo_simple.txt", mo_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 1)
        insumo = df.iloc[0]

        # Rendimiento = Vr. Unitario / Vr. Total = 120000 / 15000 = 8.0
        self.assertAlmostEqual(insumo["RENDIMIENTO"], 8.0, places=4)
        self.assertAlmostEqual(insumo["CANTIDAD_APU"], 1.0, places=4)
        self.assertEqual(insumo["FORMATO_ORIGEN"], "MO_SIMPLE")

    def test_ignore_garbage_lines(self):
        """
        Verifica que el parser ignora líneas basura (títulos, separadores, etc.)
        y no las procesa como datos válidos.
        """
        garbage_data = (
            "FORMATO DE ANÁLISIS DE PRECIOS UNITARIOS\n"
            "=========================================\n"
            "ITEM: 123\n"
            "PRESUPUESTO OFICIAL\n"
            "Un insumo valido;UND;1;;100;100\n"
            "------\n"
            "SUBTOTAL: 100\n"
        )
        test_file = self._create_test_file("garbage.txt", garbage_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 1, "Solo el insumo válido debería ser parseado.")
        self.assertEqual(df.iloc[0]["DESCRIPCION_INSUMO"], "Un insumo valido")

    def test_discard_insumo_with_zero_values(self):
        """
        Verifica que `_should_add_insumo` descarta correctamente un insumo
        cuando tanto la cantidad como el valor total son cero.
        """
        zero_value_data = (
            "ITEM: 456\n"
            "MATERIALES\n"
            "Insumo bueno;UND;1;;100;100\n"
            "Insumo malo;UND;0;;0;0\n"
            "Otro insumo bueno;UND;2;;50;100\n"
        )
        test_file = self._create_test_file("zero_value.txt", zero_value_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 2, "El insumo con valores cero no debería agregarse.")
        descripciones = df["DESCRIPCION_INSUMO"].tolist()
        self.assertIn("Insumo bueno", descripciones)
        self.assertNotIn("Insumo malo", descripciones)

    def test_prevent_data_contamination(self):
        """
        Prueba el bug original: Un APU válido seguido de un APU 'plantilla'
        sin un 'ITEM:' no debe contaminar los datos del APU válido.
        """
        contamination_data = (
            "DESCRIPCION DEL PRIMER APU\n"
            "ITEM: APU-VALIDO-1\n"
            "MATERIALES\n"
            "Cemento;UND;1;;100;100\n"
            "\n"
            # Este es un APU de plantilla, sin ITEM. Sus insumos no deben ser parseados.
            "DESCRIPCION DEL APU PLANTILLA\n"
            "Insumo fantasma;UND;10;;10;100\n"
            "Obrero fantasma;JOR;8;;20;160\n"
        )
        test_file = self._create_test_file("contamination.txt", contamination_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        # Esta prueba verifica que el parser es robusto contra la contaminación de datos.
        # Los insumos que aparecen después de una descripción pero sin un nuevo "ITEM:"
        # no deben ser asignados al APU anterior.
        self.assertEqual(len(df), 1, "Solo los insumos del APU con 'ITEM:' deben ser parseados.")

        self.assertEqual(df.iloc[0]["CODIGO_APU"], "1")
        self.assertEqual(df.iloc[0]["DESCRIPCION_INSUMO"], "Cemento")

if __name__ == "__main__":
    unittest.main(verbosity=2)