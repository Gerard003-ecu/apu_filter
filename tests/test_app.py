import io
import json
import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Importar la app de Flask y las funciones a probar
from app.app import create_app, user_sessions
from app.procesador_csv import (
    _cached_csv_processing,
    find_and_rename_columns,
    normalize_text,
    process_all_files,
    safe_read_dataframe,
)

# ======================================================================
# DATOS DE PRUEBA GLOBALES
# ======================================================================

PRESUPUESTO_DATA = (
    "ITEM;DESCRIPCION;UND;CANT.; VR. UNIT ; VR.TOTAL \n"
    "1,1;REMATE CON PINTURA DE FABRICA CAL 22 DE 120 CMTS CURVO;ML;10; 155,00 ; 1550 \n"
    "1,2;ACABADOS FINALES;M2;20; 225,00 ; 4500 \n"
    # Se ajusta la descripción para que la búsqueda por grupo funcione
    "1,3;INSTALACION TEJA SENCILLA CUBIERTA;M2;1;80000;80000\n"
    "1,4;SUMINISTRO TEJA SENCILLA;UN;1;50000;50000\n"
    "1,5;INGENIERO RESIDENTE;MES;1;15000;15000\n"
)

APUS_DATA = (
    "REMATE CON PINTURA DE FABRICA CAL 22 DE 120 CMTS CURVO\n"
    "ITEM: 1,1\n"
    "MATERIALES;;;;;\n"
    "Tornillo de Acero;UND;10,0;0;10,50;105,00\n"
    "MANO DE OBRA;;;;;\n"
    "M.O. CORTE Y DOBLEZ;UND;2,5;0;20,00;50,00\n"
    "\n"
    "ACABADOS FINALES\n"
    "ITEM: 1,2\n"
    "MATERIALES;;;;;\n"
    "Pintura Anticorrosiva;GL;5,0;0;5,00;25,00\n"
    "MANO DE OBRA;;;;;\n"
    "M.O. Mano de Obra Especializada;HR;10,0;0;20,00;200,00\n"
    "\n"
    "INSTALACION TEJA SENCILLA CUBIERTA\n"
    "ITEM: 1,3\n"
    "MANO DE OBRA;;;;;\n"
    "Ayudante;HR;8,0;0;10000,00;80000,00\n"
    "\n"
    "SUMINISTRO TEJA SENCILLA\n"
    "ITEM: 1,4\n"
    "MATERIALES;;;;;\n"
    "TEJA SENCILLA;M2;1,0;0;50000,00;50000,00\n"
    "\n"
    "INGENIERO RESIDENTE\n"
    "ITEM: 1,5\n"
    "MANO DE OBRA;;;;;\n"
    "INGENIERO RESIDENTE;MES;0,1;0;150000,00;15000,00\n"
    "\n"
    "INSTALACION CANAL CUADRILLA DE 5\n"
    "ITEM: 2,1\n"
    "MANO DE OBRA;;;;;\n"
    "Ayudante;HR;9,0;0;10000,00;90000,00\n"
    "\n"
    "INSTALACION CANAL CUADRILLA DE 3\n"
    "ITEM: 2,4\n"
    "MANO DE OBRA;;;;;\n"
    "Ayudante;HR;8,5;0;10000,00;85000,00\n"
    "\n"
    "INSTALACION CANAL LAMINA\n"
    "ITEM: 2,2\n"
    "MANO DE OBRA;;;;;\n"
    "Ayudante;HR;8,5;0;10000,00;85000,00\n"
    "\n"
    "INSTALACION PANEL TIPO SANDWICH\n"
    "ITEM: 2,3\n"
    "MANO DE OBRA;;;;;\n"
    "Ayudante;HR;9,5;0;10000,00;95000,00\n"
    "\n"
    "INSTALACION PANEL SANDWICH CUADRILLA DE 5\n"
    "ITEM: 2,5\n"
    "MANO DE OBRA;;;;;\n"
    "Ayudante;HR;10,0;0;10000,00;100000,00\n"
)

INSUMOS_DATA = (
    "  G1  ;MATERIALES;;;;;\n"
    "  CODIGO  ;  DESCRIPCION  ;  UND  ;;  CANT.  ;  VR. UNIT.  ;\n"
    "INS-001;  Tornillo de Acero  ;UND;;;10,50;\n"
    "INS-003; pintura anticorrosiva ;GL;;;5,00;\n"
    "  G2  ;MANO DE OBRA;;;;;\n"
    "  CODIGO  ;  DESCRIPCION  ;  UND  ;;  CANT.  ;  VR. UNIT.  ;\n"
    "INS-002;Mano de Obra Especializada;HR;;;20,00;\n"
)

TEST_CONFIG = {
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
        "tipo": {"CUBIERTA": "INSTALACION"},
    },
}

PRESUPUESTO_DATA_COMMA = """"ITEM","DESCRIPCION","UND","CANT.","VR. UNIT","VR.TOTAL"
"1,1","REMATE PINTURA FABRICA CAL 22 120CMTS CURVO","ML","10","155,00","1550"
"1,2","ACABADOS FINALES","M2","20","225,00","4500"
"1,3","M.O. INSTALACION CUBIERTA TST CUADRILLA 5","M2","1","80000","80000"
"1,4","SUMINISTRO TEJA SENCILLA","UN","1","50000","50000"
"""

INSUMOS_DATA_COMMA = (
    '"G1","MATERIALES"\n'
    '"CODIGO","DESCRIPCION","UND","CANT.","VR. UNIT."\n'
    '"INS-001","Tornillo de Acero","UND","","10.50"\n'
    '"INS-003","pintura anticorrosiva","GL","","5.00"\n'
    '"G2","MANO DE OBRA"\n'
    '"CODIGO","DESCRIPCION","UND","CANT.","VR. UNIT."\n'
    '"INS-002","Mano de Obra Especializada","HR","","20.00"\n'
)

class TestIndividualFunctions(unittest.TestCase):
    def test_safe_read_dataframe(self):
        self.assertIsNone(safe_read_dataframe("non_existent_file.csv"))
        with open("test_encoding.csv", "w", encoding="latin1") as f:
            f.write("col1;col2\né;ñ")
        df_csv = safe_read_dataframe("test_encoding.csv")
        self.assertIsNotNone(df_csv)
        self.assertEqual(df_csv.shape, (1, 2))
        os.remove("test_encoding.csv")
        df_to_excel = pd.DataFrame({"col1": ["é"], "col2": ["ñ"]})
        df_to_excel.to_excel("test.xlsx", index=False)
        df_xlsx = safe_read_dataframe("test.xlsx")
        self.assertIsNotNone(df_xlsx)
        self.assertEqual(df_xlsx.shape, (1, 2))
        os.remove("test.xlsx")

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
    @classmethod
    def setUpClass(cls):
        cls.presupuesto_path = "test_presupuesto.csv"
        with open(cls.presupuesto_path, "w", encoding="latin1") as f:
            f.write(PRESUPUESTO_DATA)
        cls.apus_path = "test_apus.csv"
        with open(cls.apus_path, "w", encoding="latin1") as f:
            f.write(APUS_DATA)
        cls.insumos_path = "test_insumos.csv"
        with open(cls.insumos_path, "w", encoding="latin1") as f:
            f.write(INSUMOS_DATA)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.presupuesto_path)
        os.remove(cls.apus_path)
        os.remove(cls.insumos_path)

    def setUp(self):
        _cached_csv_processing.cache_clear()

    @patch("app.procesador_csv.config", new_callable=lambda: TEST_CONFIG)
    def test_process_all_files_structure_and_calculations(self, mock_config):
        resultado = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path
        )
        self.assertIsInstance(resultado, dict)
        self.assertNotIn("error", resultado)
        presupuesto_procesado = resultado["presupuesto"]
        self.assertEqual(len(presupuesto_procesado), 5)
        item1 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1,1"), None
        )
        self.assertIsNotNone(item1)
        self.assertAlmostEqual(item1["VALOR_CONSTRUCCION_UN"], 155.0)
        item4 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1,4"), None
        )
        self.assertIsNotNone(item4)
        self.assertAlmostEqual(item4["VALOR_CONSTRUCCION_UN"], 50000.0)
        apu_detail_ing = resultado["apus_detail"]["1,5"]
        self.assertEqual(len(apu_detail_ing), 1)
        ing_item = apu_detail_ing[0]
        self.assertEqual(ing_item["DESCRIPCION"], "INGENIERO RESIDENTE")
        self.assertAlmostEqual(ing_item["CANTIDAD"], 0.1)
        self.assertAlmostEqual(ing_item["VALOR_UNITARIO"], 150000)
        self.assertAlmostEqual(ing_item["VALOR_TOTAL"], 15000)


class TestAppEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()
        os.makedirs(self.app.config["UPLOAD_FOLDER"], exist_ok=True)
        self.client = self.app.test_client()
        user_sessions.clear()

    def tearDown(self):
        self.app_context.pop()
        upload_folder = self.app.config["UPLOAD_FOLDER"]
        if os.path.exists(upload_folder):
            for root, dirs, files in os.walk(upload_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(upload_folder)

    def _get_test_file(self, filename, content):
        return (io.BytesIO(content.encode("latin1")), filename)

    @patch("app.procesador_csv.config", new_callable=lambda: TEST_CONFIG)
    @patch("app.estimator.config", new_callable=lambda: TEST_CONFIG)
    def test_get_estimate_with_session(self, mock_estimator_config, mock_processor_config):
        with self.client as c:
            data = {
                "presupuesto": self._get_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._get_test_file("apus.csv", APUS_DATA),
                "insumos": self._get_test_file("insumos.csv", INSUMOS_DATA),
            }
            c.post("/upload", data=data, content_type="multipart/form-data")
            estimate_params = {"tipo": "CUBIERTA", "material": "TST"}
            response = c.post("/api/estimate", json=estimate_params)
            self.assertEqual(response.status_code, 200)
            json_data = json.loads(response.data)
            self.assertAlmostEqual(json_data["valor_suministro"], 50000)
            self.assertAlmostEqual(json_data["valor_instalacion"], 80000)
            self.assertAlmostEqual(json_data["valor_construccion"], 130000)

if __name__ == "__main__":
    unittest.main()
