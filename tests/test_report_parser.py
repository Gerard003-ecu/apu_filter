# En tests/test_report_parser.py

import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

# Añadir el directorio raíz del proyecto al sys.path
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
        logging.basicConfig(
            level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
        )
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
        El contenido se codifica en 'utf-8' para compatibilidad en pruebas.
        """
        path = os.path.join(self.temp_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self.test_files.append(path)
        return path

    def _create_parser_for_content(self, content: str, filename: str) -> ReportParser:
        """Helper para crear un parser a partir de contenido de string."""
        path = self._create_test_file(filename, content)
        return ReportParser(path)

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
        parser = self._create_parser_for_content(apu_data, "basic_parsing.txt")
        df = parser.parse()

        self.assertEqual(len(df), 4, "Deberían parsearse 4 insumos.")
        expected_columns = [
            'CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU', 'DESCRIPCION_INSUMO',
            'UNIDAD_INSUMO', 'CANTIDAD_APU', 'PRECIO_UNIT_APU', 'VALOR_TOTAL_APU',
            'CATEGORIA', 'RENDIMIENTO', 'FORMATO_ORIGEN', 'NORMALIZED_DESC'
        ]
        self.assertListEqual(list(df.columns), expected_columns)
        self.assertEqual(df["CODIGO_APU"].iloc[0], "01")
        self.assertEqual(df["DESCRIPCION_APU"].iloc[0], "UNA DESCRIPCION DE APU VALIDA")
        self.assertEqual(df["UNIDAD_APU"].iloc[0], "M2")
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
        parser = self._create_parser_for_content(mo_data, "mo_compleja.txt")
        df = parser.parse()
        self.assertEqual(len(df), 2)
        oficial = df[df["DESCRIPCION_INSUMO"] == "OFICIAL DE PRIMERA"].iloc[0]
        self.assertAlmostEqual(oficial["CANTIDAD_APU"], 0.5, places=4)
        self.assertAlmostEqual(oficial["RENDIMIENTO"], 0.5, places=4)
        self.assertAlmostEqual(oficial["PRECIO_UNIT_APU"], 140000, places=2)
        self.assertEqual(oficial["FORMATO_ORIGEN"], "MO_COMPLEJA")
        ayudante = df[df["DESCRIPCION_INSUMO"] == "AYUDANTE"].iloc[0]
        self.assertAlmostEqual(ayudante["CANTIDAD_APU"], 1.0, places=4)

    def test_mano_de_obra_simple_logic(self):
        """Verifica la lógica para la mano de obra simple (formato CSV)."""
        mo_data = (
            "ITEM: 9902\n"
            "MANO DE OBRA\n"
            "AYUDANTE DE OBRA;;0.125;;120000;15000\n"
            "OFICIAL ESPECIALIZADO;;0.2;;180000;36000\n"
        )
        parser = self._create_parser_for_content(mo_data, "mo_simple.txt")
        df = parser.parse()
        self.assertEqual(len(df), 2)
        ayudante = df[df["DESCRIPCION_INSUMO"] == "AYUDANTE DE OBRA"].iloc[0]
        self.assertAlmostEqual(ayudante["RENDIMIENTO"], 8.0, places=4)
        self.assertAlmostEqual(ayudante["CANTIDAD_APU"], 0.125, places=4)
        self.assertEqual(ayudante["FORMATO_ORIGEN"], "MO_SIMPLE")

    def test_mo_detection_by_keywords(self):
        """Verifica que MO se detecte correctamente por keywords incluso sin categoría."""
        mo_data = (
            "ITEM: 9903\n"
            "SERVICIOS\n"
            "M.O. OFICIAL ALBAÑILERIA;;1;;150000;150000\n"
            "MANO DE OBRA ESPECIALIZADA;;2;;200000;400000\n"
        )
        parser = self._create_parser_for_content(mo_data, "mo_keywords.txt")
        df = parser.parse()
        self.assertEqual(len(df), 2)
        self.assertTrue(df["FORMATO_ORIGEN"].str.contains("MO_").all())

    def test_fallback_parsing_mechanism(self):
        """Prueba el mecanismo de fallback para líneas estructuradas no reconocidas."""
        fallback_data = (
            "ITEM: 8801\n"
            "MATERIALES\n"
            "Material sin formato valido;ALGUN DATO;OTRO DATO\n"
            "Otro material;1;2;3;4;5\n"
        )
        parser = self._create_parser_for_content(fallback_data, "fallback.txt")
        df = parser.parse()
        self.assertEqual(len(df), 1)
        # La lógica correcta es que la segunda línea es un insumo general válido
        self.assertIn("INSUMO_GENERAL", df["FORMATO_ORIGEN"].iloc[0])

    def test_ignore_garbage_lines(self):
        """Verifica que el parser ignora líneas basura."""
        garbage_data = (
            "FORMATO DE ANÁLISIS DE PRECIOS UNITARIOS\n"
            "ITEM: 123\n"
            "Un insumo valido;UND;1;;100;100\n"
            "SUBTOTAL: 100\n"
        )
        parser = self._create_parser_for_content(garbage_data, "garbage.txt")
        df = parser.parse()
        self.assertEqual(len(df), 1)
        self.assertEqual(parser.stats["garbage_lines"], 2)

    def test_discard_insumo_with_zero_values(self):
        """Verifica descarte de insumos con valores cero."""
        zero_value_data = (
            "ITEM: 456\n"
            "MATERIALES\n"
            "Insumo bueno;UND;1;;100;100\n"
            "Insumo malo;UND;0;;0;0\n"
            "Insumo con cantidad cero;UND;0;;50;50\n"
            "Insumo con total cero;UND;2;;50;0\n"
            "Otro insumo bueno;UND;2;;50;100\n"
        )
        parser = self._create_parser_for_content(zero_value_data, "zero_value.txt")
        df = parser.parse()
        self.assertEqual(
            len(df), 4, "Deberían agregarse insumos con cantidad o valor positivo."
        )
        descripciones = df["DESCRIPCION_INSUMO"].tolist()
        self.assertIn("Insumo bueno", descripciones)
        self.assertIn("Otro insumo bueno", descripciones)
        self.assertIn("Insumo con total cero", descripciones)
        self.assertIn("Insumo con cantidad cero", descripciones)
        self.assertNotIn("Insumo malo", descripciones)

    def test_prevent_data_contamination(self):
        """Prueba que no hay contaminación entre APUs."""
        contamination_data = (
            "DESCRIPCION DEL PRIMER APU\n"
            "ITEM: APU-VALIDO-1\n"
            "Cemento;UND;1;;100;100\n"
            "\n"
            "DESCRIPCION DEL APU PLANTILLA\n"
            "Insumo fantasma;UND;10;;10;100\n"
        )
        parser = self._create_parser_for_content(
            contamination_data, "contamination.txt"
        )
        df = parser.parse()
        self.assertEqual(
            len(df), 1, "Solo los insumos del APU con 'ITEM:' deben ser parseados."
        )
        self.assertEqual(df.iloc[0]["CODIGO_APU"], "1")

    def test_multiple_apus_parsing(self):
        """Prueba el parsing de múltiples APUs en el mismo archivo."""
        multi_apu_data = (
            "APU 1\nITEM: 001\nMat 1;U;1;;1;1\n\n"
            "APU 2\nITEM: 002\nMat 2;U;1;;1;1\n\n"
            "APU 3\nITEM: 003\nMat 3;U;1;;1;1\n"
        )
        parser = self._create_parser_for_content(multi_apu_data, "multi_apu.txt")
        df = parser.parse()
        self.assertEqual(len(df["CODIGO_APU"].unique()), 3)

    def test_category_detection_variations(self):
        """Prueba detección de categorías con variaciones de formato."""
        category_data = (
            "ITEM: 7777\n"
            "MATERIALES Y SUMINISTROS\n"
            "Material 1;UND;1;;100;100\n"
            "MANO DE OBRA\n"
            "Obrero;JOR;1;;150;150\n"
        )
        parser = self._create_parser_for_content(category_data, "categories.txt")
        df = parser.parse()
        self.assertEqual(len(df), 2)
        categories_found = set(df["CATEGORIA"].unique())
        self.assertIn("MATERIALES", categories_found)
        self.assertIn("MANO DE OBRA", categories_found)

    def test_error_handling(self):
        """Prueba el manejo de errores del parser."""
        parser = ReportParser("archivo_que_no_existe.txt")
        df = parser.parse()
        self.assertTrue(df.empty)
        empty_file = self._create_test_file("empty.txt", "")
        parser = ReportParser(empty_file)
        df = parser.parse()
        self.assertTrue(df.empty)

    def test_stats_tracking(self):
        """Verifica que las estadísticas de parsing se calculen correctamente."""
        stats_data = (
            "ITEM: STATS-01\nMATERIALES\n"
            "Mat 1;U;1;;1;1\nMat 2;U;1;;1;1\n"
            "MANO DE OBRA\nObrero;J;1;;1;1\n"
            "EQUIPO\nEq 1;U;1;;1;1\n"
            "Línea de descripción\n"
        )
        parser = self._create_parser_for_content(stats_data, "stats.txt")
        parser.parse()
        stats = parser.stats
        self.assertEqual(stats["items_found"], 1)
        self.assertEqual(stats["insumos_parsed"], 3)
        self.assertEqual(stats["mo_simple_parsed"], 1)
        self.assertEqual(stats["unparsed_data_lines"], 0)

    @patch("app.report_parser.clean_apu_code")
    def test_apu_code_cleaning(self, mock_clean):
        """Prueba la limpieza de códigos APU usando mock."""
        mock_clean.return_value = "CLEANED123"
        parser = self._create_parser_for_content(
            "ITEM: APU-DIRTY\nMat;U;1;;1;1", "mock_clean.txt"
        )
        df = parser.parse()
        mock_clean.assert_called_once_with("APU-DIRTY")
        self.assertEqual(df["CODIGO_APU"].iloc[0], "CLEANED123")

    def test_encoding_handling(self):
        """Prueba el manejo de diferentes codificaciones."""
        special_chars_data = "ITEM: ENC-01\nMaterial ñoño con €;U;1;;1;1\n"
        parser = self._create_parser_for_content(special_chars_data, "encoding.txt")
        df = parser.parse()
        self.assertEqual(df.iloc[0]["DESCRIPCION_INSUMO"], "Material ñoño con €")

if __name__ == "__main__":
    unittest.main(verbosity=2)
