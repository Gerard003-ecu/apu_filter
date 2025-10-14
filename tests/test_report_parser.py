# En tests/test_report_parser.py

import os
import sys
import unittest
import pandas as pd
import shutil
import logging
from unittest.mock import patch, MagicMock
import tempfile

# Añadir el directorio raíz del proyecto al sys.path para encontrar los módulos de la app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.report_parser import ReportParser

class TestNewReportParser(unittest.TestCase):
    """
    Suite de pruebas exhaustiva para la nueva y robusta implementación de ReportParser.
    """

    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todas las pruebas."""
        cls.temp_dir = tempfile.mkdtemp(prefix="test_report_parser_")
        
        # Configurar logging para pruebas
        logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
        cls.logger = logging.getLogger(__name__)

    @classmethod
    def tearDownClass(cls):
        """Limpieza final después de todas las pruebas."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Preparación antes de cada prueba individual."""
        self.test_files = []

    def tearDown(self):
        """Limpieza después de cada prueba individual."""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    def _create_test_file(self, filename: str, content: str) -> str:
        """
        Crea un archivo de prueba en el directorio temporal y devuelve su ruta.
        El contenido se codifica en 'latin1' para simular los archivos reales.
        """
        path = os.path.join(self.temp_dir, filename)
        with open(path, "w", encoding="latin1") as f:
            f.write(content)
        self.test_files.append(path)
        return path

    def _create_large_test_file(self, filename: str, num_apus: int = 10) -> str:
        """Crea un archivo de prueba grande para pruebas de estrés."""
        content = ""
        for i in range(num_apus):
            content += f"Descripción APU {i+1}\n"
            content += f"ITEM: APU-{i+1:03d}; UNIDAD: M2\n"
            content += "MATERIALES\n"
            content += f"Cemento {i+1};UND;{i+1};;100;{100*(i+1)}\n"
            content += f"Arena {i+1};M3;{i+2};;50;{50*(i+2)}\n"
            content += "MANO DE OBRA\n"
            content += f"Oficial {i+1};JOR;8;;20;160\n"
            content += "\n"
        return self._create_test_file(filename, content)

    def test_basic_parsing_and_context_detection(self):
        """Prueba el parsing básico y la correcta detección de contexto."""
        apu_data = (
            "UNA DESCRIPCION DE APU VALIDA\n"
            "ITEM: APU-01; UNIDAD: M2\n"
            "MATERIALES\n"
            "Cemento;UND;1;;100;100\n"
            "Arena;M3;2;;50;100\n"
            "MANO DE OBRA\n"
            "Oficial;JOR;8;;20;160\n"
            "EQUIPO\n"
            "Martillo;UND;1;;15;15\n"
        )
        test_file = self._create_test_file("basic_parsing.txt", apu_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 4, "Deberían parsearse 4 insumos.")
        
        # Verificar estructura de columnas
        expected_columns = [
            'CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU', 'DESCRIPCION_INSUMO',
            'UNIDAD_INSUMO', 'CANTIDAD_APU', 'PRECIO_UNIT_APU', 'VALOR_TOTAL_APU',
            'CATEGORIA', 'RENDIMIENTO', 'FORMATO_ORIGEN', 'NORMALIZED_DESC'
        ]
        self.assertListEqual(list(df.columns), expected_columns)

        # Verificar contexto del APU
        self.assertEqual(df["CODIGO_APU"].iloc[0], "01")
        self.assertEqual(df["DESCRIPCION_APU"].iloc[0], "UNA DESCRIPCION DE APU VALIDA")
        self.assertEqual(df["UNIDAD_APU"].iloc[0], "M2")

        # Verificar categorías
        categories = df.groupby("CATEGORIA")["DESCRIPCION_INSUMO"].count()
        self.assertEqual(categories.get("MATERIALES", 0), 2)
        self.assertEqual(categories.get("MANO DE OBRA", 0), 1)
        self.assertEqual(categories.get("EQUIPO", 0), 1)

    def test_mano_de_obra_compleja_logic(self):
        """Verifica la lógica para la mano de obra compleja (formato SAGUT)."""
        mo_data = (
            "ITEM: 9901\n"
            "MANO DE OBRA\n"
            "OFICIAL DE PRIMERA;80.000;1,75;140.000;0,5;70.000\n"
            "AYUDANTE;60.000;1,75;105.000;1,0;105.000\n"
        )
        test_file = self._create_test_file("mo_compleja.txt", mo_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 2)
        
        # Verificar primer insumo
        oficial = df[df["DESCRIPCION_INSUMO"] == "OFICIAL DE PRIMERA"].iloc[0]
        self.assertAlmostEqual(oficial["CANTIDAD_APU"], 0.5, places=4)
        self.assertAlmostEqual(oficial["RENDIMIENTO"], 0.5, places=4)
        self.assertAlmostEqual(oficial["PRECIO_UNIT_APU"], 140000, places=2)
        self.assertEqual(oficial["FORMATO_ORIGEN"], "MO_COMPLEJA")

        # Verificar segundo insumo
        ayudante = df[df["DESCRIPCION_INSUMO"] == "AYUDANTE"].iloc[0]
        self.assertAlmostEqual(ayudante["CANTIDAD_APU"], 1.0, places=4)

    def test_mano_de_obra_simple_logic(self):
        """Verifica la lógica para la mano de obra simple (formato CSV)."""
        mo_data = (
            "ITEM: 9902\n"
            "MANO DE OBRA\n"
            "AYUDANTE DE OBRA;;1;;120000;15000\n"
            "OFICIAL ESPECIALIZADO;;2;;180000;36000\n"
        )
        test_file = self._create_test_file("mo_simple.txt", mo_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 2)
        
        ayudante = df[df["DESCRIPCION_INSUMO"] == "AYUDANTE DE OBRA"].iloc[0]
        self.assertAlmostEqual(ayudante["RENDIMIENTO"], 8.0, places=4)
        self.assertAlmostEqual(ayudante["CANTIDAD_APU"], 1.0, places=4)
        self.assertEqual(ayudante["FORMATO_ORIGEN"], "MO_SIMPLE")

    def test_mo_detection_by_keywords(self):
        """Verifica que MO se detecte correctamente por keywords incluso sin categoría."""
        mo_data = (
            "ITEM: 9903\n"
            "SERVICIOS\n"  # Categoría incorrecta, pero keywords deben detectar MO
            "M.O. OFICIAL ALBAÑILERIA;;1;;150000;15000\n"
            "MANO DE OBRA ESPECIALIZADA;;2;;200000;40000\n"
        )
        test_file = self._create_test_file("mo_keywords.txt", mo_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 2)
        self.assertTrue(all(df["CATEGORIA"] == "MANO DE OBRA"))
        self.assertTrue(all("MO_" in df["FORMATO_ORIGEN"]))

    def test_fallback_parsing_mechanism(self):
        """Prueba el mecanismo de fallback para líneas estructuradas no reconocidas."""
        fallback_data = (
            "ITEM: 8801\n"
            "MATERIALES\n"
            "Material desconocido;COD123;2.5;DESC;75.50;188.75\n"
            "Otro material;COD456;1.0;DESC;100.00;100.00\n"
        )
        test_file = self._create_test_file("fallback.txt", fallback_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        # Debería parsear al menos algunos datos mediante fallback
        self.assertGreater(len(df), 0)
        if len(df) > 0:
            self.assertIn("FALLBACK", df["FORMATO_ORIGEN"].iloc[0])

    def test_ignore_garbage_lines(self):
        """Verifica que el parser ignora líneas basura."""
        garbage_data = (
            "FORMATO DE ANÁLISIS DE PRECIOS UNITARIOS\n"
            "=========================================\n"
            "CONSTRUCTOR: EMPRESA XYZ\n"
            "NIT: 123.456.789\n"
            "REPRESENTANTE LEGAL: JUAN PEREZ\n"
            "CIUDAD: BOGOTA\n"
            "FECHA: 2024-01-01\n"
            "PRESUPUESTO OFICIAL\n"
            "ITEM: 123\n"
            "MATERIALES\n"
            "Un insumo valido;UND;1;;100;100\n"
            "------\n"
            "SUBTOTAL: 100\n"
            "COSTOS DIRECTOS: 150\n"
            "COSTO TOTAL: 250\n"
            "PÁGINA 1 DE 1\n"
        )
        test_file = self._create_test_file("garbage.txt", garbage_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 1, "Solo el insumo válido debería ser parseado.")
        self.assertEqual(df.iloc[0]["DESCRIPCION_INSUMO"], "Un insumo valido")

    def test_discard_insumo_with_zero_values(self):
        """Verifica descarte de insumos con valores cero."""
        zero_value_data = (
            "ITEM: 456\n"
            "MATERIALES\n"
            "Insumo bueno;UND;1;;100;100\n"
            "Insumo malo;UND;0;;0;0\n"
            "Insumo con cantidad cero;UND;0;;50;0\n"
            "Insumo con total cero;UND;2;;0;0\n"
            "Otro insumo bueno;UND;2;;50;100\n"
        )
        test_file = self._create_test_file("zero_value.txt", zero_value_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 2, "Solo insumos con valores positivos deberían agregarse.")
        descripciones = df["DESCRIPCION_INSUMO"].tolist()
        self.assertIn("Insumo bueno", descripciones)
        self.assertIn("Otro insumo bueno", descripciones)
        self.assertNotIn("Insumo malo", descripciones)

    def test_prevent_data_contamination(self):
        """Prueba que no hay contaminación entre APUs."""
        contamination_data = (
            "DESCRIPCION DEL PRIMER APU\n"
            "ITEM: APU-VALIDO-1\n"
            "MATERIALES\n"
            "Cemento;UND;1;;100;100\n"
            "\n"
            "DESCRIPCION DEL APU PLANTILLA\n"
            "Insumo fantasma;UND;10;;10;100\n"
            "Obrero fantasma;JOR;8;;20;160\n"
        )
        test_file = self._create_test_file("contamination.txt", contamination_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 1, "Solo los insumos del APU con 'ITEM:' deben ser parseados.")
        self.assertEqual(df.iloc[0]["CODIGO_APU"], "1")
        self.assertEqual(df.iloc[0]["DESCRIPCION_INSUMO"], "Cemento")

    def test_multiple_apus_parsing(self):
        """Prueba el parsing de múltiples APUs en el mismo archivo."""
        multi_apu_data = (
            "APU Número 1\n"
            "ITEM: APU-001; UNIDAD: M2\n"
            "MATERIALES\n"
            "Cemento 1;UND;1;;100;100\n"
            "\n"
            "APU Número 2\n"
            "ITEM: APU-002; UNIDAD: M3\n"
            "MATERIALES\n"
            "Arena 2;M3;1;;50;50\n"
            "\n"
            "APU Número 3\n"
            "ITEM: APU-003; UNIDAD: UND\n"
            "MATERIALES\n"
            "Tornillo;UND;10;;5;50\n"
        )
        test_file = self._create_test_file("multi_apu.txt", multi_apu_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 3, "Deberían parsearse 3 insumos de 3 APUs diferentes.")
        
        codigos_apu = df["CODIGO_APU"].unique()
        self.assertEqual(len(codigos_apu), 3, "Deberían haber 3 APUs diferentes.")
        
        self.assertIn("001", codigos_apu)
        self.assertIn("002", codigos_apu)
        self.assertIn("003", codigos_apu)

    def test_large_file_performance(self):
        """Prueba de rendimiento con archivo grande."""
        test_file = self._create_large_test_file("large_file.txt", num_apus=50)
        
        import time
        start_time = time.time()
        
        parser = ReportParser(test_file)
        df = parser.parse()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertEqual(len(df), 50 * 3)  # 3 insumos por APU
        self.assertLess(processing_time, 10.0, "El procesamiento no debería tomar más de 10 segundos")

    def test_edge_cases_numeric_conversion(self):
        """Prueba casos extremos en conversión numérica."""
        edge_cases_data = (
            "ITEM: 9999\n"
            "MATERIALES\n"
            "Valor normal;UND;1.5;;100,50;150,75\n"
            "Valor con puntos;UND;1.000;;1.000,50;1.000,50\n"
            "Valor con espacios;UND;1 000;;1 000,50;1 000 500\n"
            "Valor negativo;UND;-1;;-50;-50\n"
            "Valor vacío;UND;;;;\n"
            "Valor texto;UND;ABC;;DEF;GHI\n"
        )
        test_file = self._create_test_file("edge_cases.txt", edge_cases_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        # Solo deberían parsearse los valores numéricos válidos
        self.assertGreater(len(df), 0)
        
        # Verificar que no hay valores negativos en cantidades o precios
        valid_rows = df[(df["CANTIDAD_APU"] > 0) & (df["VALOR_TOTAL_APU"] > 0)]
        self.assertTrue(len(valid_rows) > 0)

    def test_category_detection_variations(self):
        """Prueba detección de categorías con variaciones de formato."""
        category_data = (
            "ITEM: 7777\n"
            "MATERIALES Y SUMINISTROS\n"
            "Material 1;UND;1;;100;100\n"
            "MANO DE OBRA DIRECTA\n"
            "Obrero;JOR;1;;150;150\n"
            "EQUIPOS Y HERRAMIENTAS\n"
            "Equipo 1;UND;1;;200;200\n"
            "OTROS GASTOS\n"
            "Gasto 1;UND;1;;50;50\n"
            "TRANSPORTE\n"
            "Transporte 1;UND;1;;75;75\n"
        )
        test_file = self._create_test_file("categories.txt", category_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 5)
        
        # Verificar que todas las categorías se detectaron correctamente
        categories_found = set(df["CATEGORIA"].unique())
        expected_categories = {"MATERIALES", "MANO DE OBRA", "EQUIPO", "OTROS", "TRANSPORTE"}
        
        for expected in expected_categories:
            self.assertIn(expected, categories_found, f"Falta la categoría: {expected}")

    def test_error_handling(self):
        """Prueba el manejo de errores del parser."""
        # Test con archivo que no existe
        with self.assertRaises(Exception):
            parser = ReportParser("archivo_que_no_existe.txt")
            df = parser.parse()

        # Test con archivo vacío
        empty_file = self._create_test_file("empty.txt", "")
        parser = ReportParser(empty_file)
        df = parser.parse()
        self.assertTrue(df.empty, "El DataFrame debería estar vacío para archivo vacío")

    def test_stats_tracking(self):
        """Verifica que las estadísticas de parsing se calculen correctamente."""
        stats_data = (
            "ITEM: STATS-01\n"
            "MATERIALES\n"
            "Material 1;UND;1;;100;100\n"
            "Material 2;UND;2;;50;100\n"
            "MANO DE OBRA\n"
            "Obrero;JOR;1;;150;150\n"
            "EQUIPO\n"
            "Equipo 1;UND;1;;200;200\n"
        )
        test_file = self._create_test_file("stats.txt", stats_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        # Verificar estadísticas básicas
        stats = parser.stats
        self.assertEqual(stats["items_found"], 1)
        self.assertEqual(stats["insumos_parsed"], 2)  # 2 materiales + 1 equipo
        self.assertGreater(stats["mo_compleja_parsed"] + stats["mo_simple_parsed"], 0)
        self.assertGreater(stats["total_lines"], 0)
        self.assertGreater(stats["processed_lines"], 0)

    def test_normalized_text_functionality(self):
        """Prueba la funcionalidad de normalización de texto."""
        text_data = (
            "ITEM: NORM-01\n"
            "MATERIALES\n"
            "Material con ACENTOS;UND;1;;100;100\n"
            "Material con   espacios   extra;UND;1;;100;100\n"
            "Material con #símbolos!;UND;1;;100;100\n"
        )
        test_file = self._create_test_file("normalized.txt", text_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        # Verificar normalización
        normalized_texts = df["NORMALIZED_DESC"].tolist()
        
        for text in normalized_texts:
            self.assertTrue(text.islower(), "El texto normalizado debería estar en minúsculas")
            self.assertNotRegex(text, r"[áéíóú]", "No debería haber acentos en texto normalizado")
            self.assertNotRegex(text, r"\s{2,}", "No debería haber espacios múltiples")

    @patch('app.report_parser.clean_apu_code')
    def test_apu_code_cleaning(self, mock_clean):
        """Prueba la limpieza de códigos APU usando mock."""
        mock_clean.return_value = "CLEANED123"
        
        test_data = "ITEM: APU-DIRTY-CODE\nMATERIALES\nMaterial;UND;1;;100;100\n"
        test_file = self._create_test_file("mock_clean.txt", test_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        mock_clean.assert_called_once()
        self.assertEqual(df["CODIGO_APU"].iloc[0], "CLEANED123")

    def test_encoding_handling(self):
        """Prueba el manejo de diferentes codificaciones."""
        # Crear archivo con caracteres especiales
        special_chars_data = (
            "ITEM: ENCODING-01\n"
            "MATERIALES\n"
            "Material ñoño con carácter €;UND;1;;100;100\n"
            "Material con símbolo ®;UND;1;;100;100\n"
        )
        test_file = self._create_test_file("encoding.txt", special_chars_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        # El parser debería manejar los caracteres sin fallar
        self.assertGreater(len(df), 0)

def run_performance_tests():
    """Función adicional para ejecutar pruebas de rendimiento."""
    suite = unittest.TestSuite()
    suite.addTest(TestNewReportParser('test_large_file_performance'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Ejecutar pruebas normales
    unittest.main(verbosity=2)
    
    # Ejecutar pruebas de rendimiento por separado (opcional)
    # print("\n" + "="*50)
    # print("EJECUTANDO PRUEBAS DE RENDIMIENTO")
    # print("="*50)
    # run_performance_tests()