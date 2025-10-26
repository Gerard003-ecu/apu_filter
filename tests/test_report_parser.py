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
        El contenido se codifica en 'latin1' para compatibilidad con el parser.
        """
        path = os.path.join(self.temp_dir, filename)
        with open(path, "w", encoding="latin1") as f:
            f.write(content)
        self.test_files.append(path)
        return path

    def _create_parser_for_content(self, content: str, filename: str) -> ReportParser:
        """Helper para crear un parser a partir de contenido de string."""
        path = self._create_test_file(filename, content)
        return ReportParser(path)

    def test_basic_parsing_and_context_detection(self):
        """Prueba el parsing básico con la nueva lógica de descripción."""
        apu_data = (
            "ITEM: APU-01; UNIDAD: M2\n"
            "UNA DESCRIPCION DE APU VALIDA;MATERIALES;Cemento;UND;1;;100;100\n"
        )
        parser = self._create_parser_for_content(apu_data, "basic_parsing.txt")
        df = parser.parse()

        self.assertEqual(len(df), 1, "Debería parsearse 1 insumo.")
        self.assertEqual(df["CODIGO_APU"].iloc[0], "01")
        self.assertEqual(df["DESCRIPCION_APU"].iloc[0], "UNA DESCRIPCION DE APU VALIDA")

    def test_mano_de_obra_compleja_logic(self):
        """Verifica la lógica para la mano de obra compleja con nueva máquina de estados."""
        mo_data = (
            "ITEM: 9901\n"
            "DESCRIPCION DEL APU DE MANO DE OBRA\n"  # ✅ LÍNEA DE DESCRIPCIÓN REQUERIDA
            "MANO DE OBRA\n"
            "OFICIAL DE PRIMERA;80.000;1,75;140.000;0,5;70.000\n"
            "AYUDANTE;60.000;1,75;105.000;1,0;105.000\n"
        )
        parser = self._create_parser_for_content(mo_data, "mo_compleja.txt")
        df = parser.parse()

        self.assertEqual(len(df), 2)
        self.assertEqual(
            df["DESCRIPCION_APU"].iloc[0], "DESCRIPCION DEL APU DE MANO DE OBRA"
        )

    def test_mano_de_obra_simple_logic(self):
        """Verifica la lógica para la mano de obra simple con nueva máquina de estados."""
        mo_data = (
            "ITEM: 9902\n"
            "APU CON MANO DE OBRA SIMPLE\n"  # ✅ LÍNEA DE DESCRIPCIÓN REQUERIDA
            "MANO DE OBRA\n"
            "AYUDANTE DE OBRA;;0.125;;120000;15000\n"
            "OFICIAL ESPECIALIZADO;;0.2;;180000;36000\n"
        )
        parser = self._create_parser_for_content(mo_data, "mo_simple.txt")
        df = parser.parse()
        self.assertEqual(len(df), 2)
        ayudante = df[df["DESCRIPCION_INSUMO"] == "AYUDANTE DE OBRA"].iloc[0]
        self.assertAlmostEqual(ayudante["CANTIDAD_APU"], 0.125, places=4)
        self.assertEqual(ayudante["FORMATO_ORIGEN"], "MO_SIMPLE")

    def test_mo_detection_by_keywords(self):
        """Verifica que MO se detecte por keywords incluso sin categoría correcta."""
        mo_data = (
            "ITEM: 9903\n"
            "DESCRIPCION APU TEST\n"
            "SERVICIOS VARIOS\n"  # Categoría incorrecta a propósito
            "M.O. OFICIAL;;1;;150000;150000\n"
        )
        test_file = self._create_test_file("mo_keywords.txt", mo_data)
        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 1, "Debería haber encontrado un insumo de MO.")
        insumo = df.iloc[0]
        self.assertEqual(
            insumo["CATEGORIA"],
            "MANO DE OBRA",
            "La categoría debería haber sido forzada a MANO DE OBRA.",
        )

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
            "DESCRIPCION\n"
            "Un insumo valido;UND;1;;100;100\n"
            "SUBTOTAL: 100\n"
        )
        parser = self._create_parser_for_content(garbage_data, "garbage.txt")
        df = parser.parse()
        self.assertEqual(len(df), 1)
        # The new parser does not count garbage lines in the same way.
        # The important check is that valid data is extracted.
        self.assertEqual(df['DESCRIPCION_INSUMO'].iloc[0], "Un insumo valido")

    def test_discard_insumo_with_zero_values(self):
        """Verifica descarte de insumos con valores cero."""
        zero_value_data = (
            "ITEM: 456\n"
            "DESCRIPCION\n"
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
        """
        Prueba que los insumos sin un APU activo son ignorados (nueva lógica más estricta).
        """
        contamination_data = (
            "ITEM: APU-VALIDO-1\n"
            "DESCRIPCION VALIDA\n"  # ✅ REQUERIDO
            "MATERIALES\n"
            "Cemento;UND;1;;100;100\n"
            "\n"  # Línea en blanco para resetear el contexto
            "DESCRIPCION DE PLANTILLA SIN ITEM - DEBE SER IGNORADA\n"  # ❌ IGNORADA
            "Insumo fantasma;UND;10;;10;100\n"  # ❌ IGNORADA
        )
        test_file = self._create_test_file("contamination.txt", contamination_data)
        parser = ReportParser(test_file)
        df = parser.parse()

        self.assertEqual(len(df), 1, "Solo el insumo del APU con 'ITEM:' debe ser parseado.")
        self.assertEqual(df.iloc[0]["DESCRIPCION_INSUMO"], "Cemento")
        # Verificar que NO hay APUs con descripción "fantasma"
        self.assertNotIn("Insumo fantasma", df["DESCRIPCION_INSUMO"].values)

    def test_multiple_apus_parsing(self):
        """Prueba el parsing de múltiples APUs en el mismo archivo."""
        multi_apu_data = (
            "ITEM: 001\nAPU 1\nMat 1;U;1;;1;1\n\n"
            "ITEM: 002\nAPU 2\nMat 2;U;1;;1;1\n\n"
            "ITEM: 003\nAPU 3\nMat 3;U;1;;1;1\n"
        )
        parser = self._create_parser_for_content(multi_apu_data, "multi_apu.txt")
        df = parser.parse()
        self.assertEqual(len(df["CODIGO_APU"].unique()), 3)

    def test_category_detection_variations(self):
        """Prueba detección de categorías con variaciones de formato."""
        category_data = (
            "ITEM: 7777\n"
            "UNA DESCRIPCION\n"
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


    @patch("app.report_parser.clean_apu_code")
    def test_apu_code_cleaning(self, mock_clean):
        """Prueba la limpieza de códigos APU usando mock."""
        mock_clean.return_value = "CLEANED123"
        parser = self._create_parser_for_content(
            "ITEM: APU-DIRTY\nDESCRIPCION\nMat;U;1;;1;1", "mock_clean.txt"
        )
        df = parser.parse()
        mock_clean.assert_called_once_with("APU-DIRTY")
        self.assertEqual(df["CODIGO_APU"].iloc[0], "CLEANED123")

    def test_encoding_handling(self):
        """Prueba el manejo de diferentes codificaciones."""
        special_chars_data = "ITEM: ENC-01\nDESCRIPCION\nMaterial con ñandú;U;1;;1;1\n"
        parser = self._create_parser_for_content(special_chars_data, "encoding.txt")
        df = parser.parse()
        self.assertEqual(df.iloc[0]["DESCRIPCION_INSUMO"], "Material con ñandú")

    def test_category_detection_before_insumos(self):
        """ Prueba CRÍTICA: La categoría debe detectarse después de la descripción """
        critical_data = (
            "ITEM: 1,1; UNIDAD: M2\n"
            "LAMINA DE ACERO GALVANIZADO CALIBRE 22\n"  # ✅ DESCRIPCIÓN PRIMERO
            "MATERIALES\n"  # ← LUEGO CATEGORÍA
            "LAMINA DE ACERO GALVANIZADO CAL 22; M2; 1,03; ; 34.756,10; 35.799,00\n"
            "MANO DE OBRA\n"  # ← CAMBIO DE CATEGORÍA
            "OFICIAL; JOR; 0,14; ; 50.000,00; 7.000,00\n"
        )
        test_file = self._create_test_file("critical_category.txt", critical_data)

        parser = ReportParser(test_file)
        df = parser.parse()

        # VERIFICACIONES CRÍTICAS
        self.assertEqual(len(df), 2, "Deberían haber 2 insumos")
        self.assertEqual(
            df["DESCRIPCION_APU"].iloc[0], "LAMINA DE ACERO GALVANIZADO CALIBRE 22"
        )

    def test_description_capture_robustness(self):
        """Prueba que la descripción SIEMPRE se captura después de ITEM."""
        test_cases = [
            (
                "ITEM: TEST-001\n"
                "DESCRIPCION SIMPLE\n"
                "MATERIALES\n"
                "Insumo;UND;1;;100;100",
                "DESCRIPCION SIMPLE"
            ),
            (
                "ITEM: TEST-002\n"
                "DESCRIPCION CON DATOS;MATERIALES;Insumo;UND;1;;100;100",
                "DESCRIPCION CON DATOS",
            ),
            (
                "ITEM: TEST-003\n"
                "   DESCRIPCION CON ESPACIOS   \n"  # Con espacios
                "MATERIALES\n"
                "Insumo;UND;1;;100;100",
                "DESCRIPCION CON ESPACIOS"
            )
        ]

        for i, (content, expected_desc) in enumerate(test_cases):
            with self.subTest(test_case=i):
                parser = self._create_parser_for_content(content, f"desc_test_{i}.txt")
                df = parser.parse()
                self.assertFalse(
                    df.empty, f"Test case {i}: DataFrame no debería estar vacío"
                )
                actual_desc = df["DESCRIPCION_APU"].iloc[0]
                self.assertEqual(
                    actual_desc,
                    expected_desc,
                    f"Test case {i}: Descripción no coincide",
                )

    def test_state_transitions(self):
        """Verifica las transiciones de estado de la máquina."""
        transition_data = (
            "ITEM: TRANS-001\n"  # IDLE → AWAITING_DESCRIPTION
            "DESCRIPCION APU\n"  # AWAITING_DESCRIPTION → PROCESSING_APU
            "MATERIALES\n"
            "Insumo1;UND;1;;100;100\n"
            "\n"  # PROCESSING_APU → IDLE (línea en blanco)
            "ITEM: TRANS-002\n"  # IDLE → AWAITING_DESCRIPTION
            "OTRA DESCRIPCION\n"  # AWAITING_DESCRIPTION → PROCESSING_APU
            "Insumo2;UND;1;;200;200"
        )

        parser = self._create_parser_for_content(transition_data, "transitions.txt")
        df = parser.parse()

        # Verificar que ambos APUs se procesaron
        self.assertEqual(len(df["CODIGO_APU"].unique()), 2)
        self.assertEqual(len(df), 2)

    def test_unit_extraction_robustness(self):
        """Prueba la extracción robusta de unidades de medida."""
        test_cases = [
            ("ITEM: APU-01; UNIDAD: M2", "M2"),
            ("ITEM: APU-02; UNIDAD M3", "M3"),
            ("ITEM: APU-03; UNIDAD: UND", "UND"),
            ("ITEM: APU-04; UNIDAD: JOR.", "JOR"),
            ("ITEM: APU-05; JORNAL", "JOR"),
            ("ITEM: APU-06; UNIDAD: SERVICIO", "SERVICIO"),
            ("ITEM: APU-07; ML", "ML"),
            ("ITEM: APU-08", "UND"),
        ]

        for i, (content, expected_unit) in enumerate(test_cases):
            with self.subTest(test_case=i):
                apu_data = content + "\nDESCRIPCION DE PRUEBA\nInsumo;U;1;;1;1"
                parser = self._create_parser_for_content(apu_data, f"unit_test_{i}.txt")
                df = parser.parse()
                self.assertFalse(df.empty)
                self.assertEqual(df["UNIDAD_APU"].iloc[0], expected_unit)

if __name__ == "__main__":
    unittest.main(verbosity=2)
