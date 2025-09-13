import os
import unittest
from unittest.mock import patch

import pandas as pd

# Importar funciones para probarlas individualmente
from procesador_csv import (
    _cached_csv_processing,  # Importar para controlar el cache
    calculate_estimate,
    find_and_rename_columns,
    normalize_text,
    process_all_files,
    safe_read_csv,
)


class TestIndividualFunctions(unittest.TestCase):
    """Pruebas para funciones de utilidad individuales."""

    def test_safe_read_csv(self):
        # Prueba con un archivo que no existe
        self.assertIsNone(safe_read_csv("non_existent_file.csv"))
        # Prueba con un archivo con encoding correcto
        with open("test_encoding.csv", "w", encoding="latin1") as f:
            f.write("col1;col2\né;ñ")
        df = safe_read_csv("test_encoding.csv", delimiter=";")
        self.assertIsNotNone(df)
        self.assertEqual(df.shape, (1, 2))
        os.remove("test_encoding.csv")

    def test_normalize_text(self):
        s = pd.Series(["  Texto CON Acentos y Ñ  ", "  Múltiples   espacios  "])
        result = normalize_text(s)
        self.assertEqual(result.iloc[0], "texto con acentos y n")
        self.assertEqual(result.iloc[1], "multiples espacios")

    def test_find_and_rename_columns(self):
        df = pd.DataFrame(columns=["  ITEM ", "DESCRIPCION DEL APU", "  CANT. "])
        column_map = {
            "CODIGO_APU": ["item"],
            "DESCRIPCION_APU": ["descripcion"],
            "CANTIDAD_PRESUPUESTO": ["cantidad", "cant"],
        }
        renamed_df = find_and_rename_columns(df, column_map)
        self.assertIn("CODIGO_APU", renamed_df.columns)
        self.assertIn("DESCRIPCION_APU", renamed_df.columns)
        self.assertIn("CANTIDAD_PRESUPUESTO", renamed_df.columns)


class TestCSVProcessor(unittest.TestCase):
    """
    Clase de prueba actualizada para validar la lógica de `procesador_csv.py`.
    """

    @classmethod
    def setUpClass(cls):
        """Crear archivos de prueba una vez para toda la clase."""
        cls.presupuesto_data = (
            "ITEM;DESCRIPCION;UND;CANT.; VR. UNIT ; VR.TOTAL \n"
            "1;Actividad de Construcción 1;;;;\n"
            "1,1;Montaje de Estructura;ML;10; 155,00 ; 1550 \n"
            "1,2;Acabados Finales;M2;20; 225,00 ; 4500 \n"
        )
        cls.presupuesto_path = "test_presupuesto.csv"
        with open(cls.presupuesto_path, "w", encoding="latin1") as f:
            f.write(cls.presupuesto_data)

        cls.apus_data = (
            "REMATE CON PINTURA;;;;;ITEM:   1,1\n"
            "MATERIALES;;;;;\n"
            "Tornillo de Acero;UND; 10,0;;10,50;105,00\n"
            "MANO DE OBRA;;;;;\n"
            "Mano de Obra Especializada;HR; 2,5;;20,00;50,00\n"
            ";;;;\n"
            "REMATE DE ACERO;;;;;ITEM:   1,2\n"
            "MATERIALES;;;;;\n"
            "Pintura Anticorrosiva;GL; 5,0;;5,00;25,00\n"
            "MANO DE OBRA;;;;;\n"
            "Mano de Obra Especializada;HR; 10,0;;20,00;200,00\n"
        )
        cls.apus_path = "test_apus.csv"
        with open(cls.apus_path, "w", encoding="latin1") as f:
            f.write(cls.apus_data)

        cls.insumos_data = (
            "  G1  ;MATERIALES;;;;;\n"
            "  CODIGO  ;  DESCRIPCION  ;  UND  ;;  CANT.  ;  VR. UNIT.  ;\n"
            "INS-001;  Tornillo de Acero  ;UND;;;10,50;\n"
            "INS-003; pintura anticorrosiva ;GL;;;5,00;\n"
            "  G2  ;MANO DE OBRA;;;;;\n"
            "  CODIGO  ;  DESCRIPCION  ;  UND  ;;  CANT.  ;  VR. UNIT.  ;\n"
            "INS-002;Mano de Obra Especializada;HR;;;20,00;\n"
        )
        cls.insumos_path = "test_insumos.csv"
        with open(cls.insumos_path, "w", encoding="latin1") as f:
            f.write(cls.insumos_data)

        # Mock del config
        cls.test_config = {
            "presupuesto_column_map": {
                "CODIGO_APU": ["ITEM"],
                "DESCRIPCION_APU": ["DESCRIPCION"],
                "CANTIDAD_PRESUPUESTO": ["CANT."],
            },
            "category_keywords": {
                "MATERIALES": "MATERIALES",
                "MANO DE OBRA": "MANO DE OBRA",
                "EQUIPO": "EQUIPO",
                "OTROS": "OTROS",
            },
            "param_map": {
                "material": {"TST": "TEJA SENCILLA"},
                "tipo": {"CUBIERTA": "IZAJE MANUAL"},
            },
        }

    @classmethod
    def tearDownClass(cls):
        """Eliminar archivos de prueba al final."""
        os.remove(cls.presupuesto_path)
        os.remove(cls.apus_path)
        os.remove(cls.insumos_path)

    def setUp(self):
        """Limpiar el cache antes de cada prueba."""
        _cached_csv_processing.cache_clear()

    @patch("procesador_csv.config", new_callable=lambda: TestCSVProcessor.test_config)
    def test_process_all_files_structure_and_calculations(self, mock_config):
        """Prueba la función orquestadora `process_all_files`."""
        resultado = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path
        )

        self.assertIsInstance(resultado, dict)
        self.assertNotIn("error", resultado)
        self.assertIn("presupuesto", resultado)
        self.assertIn("insumos", resultado)
        self.assertIn("apus_detail", resultado)

        presupuesto_procesado = resultado["presupuesto"]
        self.assertEqual(len(presupuesto_procesado), 2)

        item1 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1,1"),
            None,
        )
        self.assertIsNotNone(item1)
        self.assertAlmostEqual(item1["VALOR_CONSTRUCCION_UN"], 155.0)

        item2 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1,2"),
            None,
        )
        self.assertIsNotNone(item2)
        self.assertAlmostEqual(item2["VALOR_CONSTRUCCION_UN"], 225.0)

    @patch("procesador_csv.config", new_callable=lambda: TestCSVProcessor.test_config)
    def test_caching_logic(self, mock_config):
        """Verifica que el sistema de caché funciona."""
        # Primera llamada (miss)
        process_all_files(self.presupuesto_path, self.apus_path, self.insumos_path)
        info1 = _cached_csv_processing.cache_info()
        self.assertEqual(info1.misses, 1)
        self.assertEqual(info1.hits, 0)

        # Segunda llamada (hit)
        process_all_files(self.presupuesto_path, self.apus_path, self.insumos_path)
        info2 = _cached_csv_processing.cache_info()
        self.assertEqual(info2.misses, 1)
        self.assertEqual(info2.hits, 1)

        # Llamada sin caché
        process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path, use_cache=False
        )
        info3 = _cached_csv_processing.cache_info()
        self.assertEqual(info3.currsize, 0)  # El cache debió ser limpiado

    @patch("procesador_csv.config", new_callable=lambda: TestCSVProcessor.test_config)
    def test_calculate_estimate(self, mock_config):
        """Prueba la función de estimación."""
        # Primero, procesar los archivos para llenar el data_store
        data_store = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path
        )

        # Caso 1: Faltan parámetros
        params_missing = {"tipo": "CUBIERTA"}
        result = calculate_estimate(params_missing, data_store)
        self.assertIn("error", result)
        self.assertIn("Parámetros requeridos faltantes", result["error"])

        # Caso 2: Estimación exitosa (simplificada)
        # Necesitamos un APU que coincida con la búsqueda
        data_store["all_apus"].append(
            {
                "CODIGO_APU": "M.O.1",
                "DESCRIPCION_APU": "MANO DE OBRA IZAJE MANUAL TEJA SENCILLA",
                "DESCRIPCION_INSUMO": "Ayudante",
                "CANTIDAD_APU": 8,
                "PRECIO_UNIT_APU": 10000,
                "VALOR_TOTAL_APU": 80000,
                "CATEGORIA": "MANO DE OBRA",
                "NORMALIZED_DESC": "ayudante",
            }
        )
        data_store["apus_detail"]["M.O.1"] = [
            {
                "Descripción": "Ayudante",
                "Cantidad": 8,
                "Vr Unitario": 10000,
                "Vr Total": 80000,
                "CATEGORIA": "MANO DE OBRA",
            }
        ]

        params_ok = {"tipo": "CUBIERTA", "material": "TST"}
        result = calculate_estimate(params_ok, data_store)
        self.assertNotIn("error", result)
        self.assertEqual(
            result["apu_encontrado"], "MANO DE OBRA IZAJE MANUAL TEJA SENCILLA"
        )
        self.assertAlmostEqual(result["valor_instalacion"], 80000)


if __name__ == "__main__":
    unittest.main()
