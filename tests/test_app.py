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
from app.app import app, user_sessions
from app.procesador_csv import (
    _cached_csv_processing,
    calculate_estimate,
    find_and_rename_columns,
    normalize_text,
    process_all_files,
    safe_read_csv,
)

# ======================================================================
# DATOS DE PRUEBA GLOBALES
# Mover los datos aquí los hace accesibles para todas las clases de prueba.
# ======================================================================

PRESUPUESTO_DATA = (
    "ITEM;DESCRIPCION;UND;CANT.; VR. UNIT ; VR.TOTAL \n"
    "1,1;Montaje de Estructura;ML;10; 155,00 ; 1550 \n"
    "1,2;Acabados Finales;M2;20; 225,00 ; 4500 \n"
    "1,3;MANO DE OBRA INSTALACION TEJA SENCILLA CUADRILLA DE 5;M2;1;80000;80000\n"
    "1,4;APU con Corte y Doblez;UN;1;15000;15000\n"
)

APUS_DATA = (
    "MANO DE OBRA INSTALACION TEJA SENCILLA CUADRILLA DE 5;;;;;ITEM:   1,3\n"
    "MANO DE OBRA;;;;;\n"
    "Ayudante;HR;8;;10000;80000\n"
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

APUS_DATA_COMMA = """"MANO DE OBRA INSTALACION TEJA SENCILLA CUADRILLA DE 5","","","","","ITEM:   1,3"
"MANO DE OBRA","","","","",""
"Ayudante","HR","8","","10000","80000"
"REMATE CON PINTURA","","","","","ITEM:   1,1"
"MATERIALES","","","","",""
"Tornillo de Acero","UND","10,0","","10,50","105,00"
"MANO DE OBRA","","","","",""
"Mano de Obra Especializada","HR","2,5","","20,00","50,00"
","","","",""
"REMATE DE ACERO","","","","","ITEM:   1,2"
"MATERIALES","","","","",""
"Pintura Anticorrosiva","GL","5,0","","5,00","25,00"
"MANO DE OBRA","","","","",""
"Mano de Obra Especializada","HR","10,0","","20,00","200,00"
"""

PRESUPUESTO_DATA_COMMA = """"ITEM","DESCRIPCION","UND","CANT.","VR. UNIT","VR.TOTAL"
"1,1","Montaje de Estructura","ML","10","155,00","1550"
"1,2","Acabados Finales","M2","20","225,00","4500"
"1,3","MANO DE OBRA INSTALACION TEJA SENCILLA CUADRILLA DE 5","M2","1","80000","80000"
"1,4","APU con Corte y Doblez","UN","1","15000","15000"
"""


# ======================================================================
# CLASES DE PRUEBA
# ======================================================================


class TestIndividualFunctions(unittest.TestCase):
    """Pruebas para funciones de utilidad individuales."""

    # ... (esta clase no necesita cambios)
    def test_safe_read_csv(self):
        self.assertIsNone(safe_read_csv("non_existent_file.csv"))
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
    """Clase de prueba actualizada para validar la lógica de `procesador_csv.py`."""

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
        self.assertEqual(len(presupuesto_procesado), 4)
        item1 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1,1"), None
        )
        self.assertIsNotNone(item1)
        self.assertAlmostEqual(item1["VALOR_CONSTRUCCION_UN"], 155.0)

        # Test for the new "CORTE Y DOBLEZ" case
        item4 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1,4"), None
        )
        self.assertIsNotNone(item4)
        # TODO: La lógica para manejar APUs no encontrados en el archivo apus.csv
        # necesita ser definida. Por ahora, el costo es 0.
        self.assertAlmostEqual(item4["VALOR_CONSTRUCCION_UN"], 0.0)

    @patch("app.procesador_csv.config", new_callable=lambda: TEST_CONFIG)
    def test_process_all_files_comma_delimited(self, mock_config):
        """Prueba el procesamiento de archivos CSV delimitados por comas."""
        presupuesto_path_comma = "test_presupuesto_comma.csv"
        apus_path_comma = "test_apus_comma.csv"

        with open(presupuesto_path_comma, "w", encoding="latin1") as f:
            f.write(PRESUPUESTO_DATA_COMMA)
        with open(apus_path_comma, "w", encoding="latin1") as f:
            f.write(APUS_DATA_COMMA)

        resultado = process_all_files(
            presupuesto_path_comma, apus_path_comma, self.insumos_path
        )

        os.remove(presupuesto_path_comma)
        os.remove(apus_path_comma)

        self.assertIsInstance(resultado, dict)
        self.assertNotIn("error", resultado)
        presupuesto_procesado = resultado["presupuesto"]
        self.assertEqual(len(presupuesto_procesado), 4)
        item1 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1,1"), None
        )
        self.assertIsNotNone(item1)
        self.assertAlmostEqual(item1["VALOR_CONSTRUCCION_UN"], 155.0)

    @patch("app.procesador_csv.config", new_callable=lambda: TEST_CONFIG)
    def test_caching_logic(self, mock_config):
        process_all_files(self.presupuesto_path, self.apus_path, self.insumos_path)
        info1 = _cached_csv_processing.cache_info()
        self.assertEqual(info1.misses, 1)
        process_all_files(self.presupuesto_path, self.apus_path, self.insumos_path)
        info2 = _cached_csv_processing.cache_info()
        self.assertEqual(info2.hits, 1)
        process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path, use_cache=False
        )
        info3 = _cached_csv_processing.cache_info()
        self.assertEqual(info3.currsize, 0)

    @patch("app.procesador_csv.config", new_callable=lambda: TEST_CONFIG)
    def test_calculate_estimate(self, mock_config):
        data_store = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path
        )
        params_ok = {"tipo": "CUBIERTA", "material": "TST", "cuadrilla": "5"}
        result = calculate_estimate(params_ok, data_store)
        self.assertNotIn("error", result)
        expected_apu = (
            "Suministro: N/A | Instalación: "
            "MANO DE OBRA INSTALACION TEJA SENCILLA CUADRILLA DE 5"
        )
        self.assertEqual(result["apu_encontrado"], expected_apu)
        self.assertAlmostEqual(result["valor_instalacion"], 80000)


class TestAppEndpoints(unittest.TestCase):
    """Pruebas para los endpoints de la aplicación Flask."""

    def setUp(self):
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret-key"
        app.config["UPLOAD_FOLDER"] = "test_uploads"
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        self.client = app.test_client()
        user_sessions.clear()

    def tearDown(self):
        upload_folder = app.config["UPLOAD_FOLDER"]
        if os.path.exists(upload_folder):
            for root, dirs, files in os.walk(upload_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(upload_folder)

    def _get_test_file(self, filename, content):
        return (io.BytesIO(content.encode("latin1")), filename)

    def test_01_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    @patch("app.procesador_csv.config", new_callable=lambda: TEST_CONFIG)
    def test_02_upload_success(self, mock_config):
        with self.client as c:
            data = {
                "presupuesto": self._get_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._get_test_file("apus.csv", APUS_DATA),
                "insumos": self._get_test_file("insumos.csv", INSUMOS_DATA),
            }
            response = c.post("/upload", data=data, content_type="multipart/form-data")
            self.assertEqual(response.status_code, 200)
            with c.session_transaction() as sess:
                self.assertIn("session_id", sess)

    @patch("app.procesador_csv.config", new_callable=lambda: TEST_CONFIG)
    def test_07_get_estimate_with_session(self, mock_config):
        with self.client as c:
            data = {
                "presupuesto": self._get_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._get_test_file("apus.csv", APUS_DATA),
                "insumos": self._get_test_file("insumos.csv", INSUMOS_DATA),
            }
            upload_response = c.post(
                "/upload", data=data, content_type="multipart/form-data"
            )
            self.assertEqual(upload_response.status_code, 200)

            estimate_params = {"tipo": "CUBIERTA", "material": "TST", "cuadrilla": "5"}
            response = c.post("/api/estimate", json=estimate_params)

            self.assertEqual(response.status_code, 200)
            json_data = json.loads(response.data)
            self.assertAlmostEqual(json_data["valor_instalacion"], 80000)

    @patch("app.procesador_csv.config", new_callable=lambda: TEST_CONFIG)
    def test_08_get_apu_detail_with_simulation(self, mock_config):
        with self.client as c:
            # Primero, sube los archivos para inicializar la sesión de datos
            data = {
                "presupuesto": self._get_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._get_test_file("apus.csv", APUS_DATA),
                "insumos": self._get_test_file("insumos.csv", INSUMOS_DATA),
            }
            upload_response = c.post(
                "/upload", data=data, content_type="multipart/form-data"
            )
            self.assertEqual(upload_response.status_code, 200)

            # Ahora, solicita el detalle de un APU específico
            # El código '1,1' se codifica como '1%2C1' en la URL
            response = c.get("/api/apu/1%2C1")

            self.assertEqual(response.status_code, 200)
            json_data = json.loads(response.data)

            # Verifica que la clave 'simulation' exista
            self.assertIn("simulation", json_data)

            # Verifica que los resultados de la simulación tengan las claves esperadas
            sim_results = json_data["simulation"]
            self.assertIn("mean", sim_results)
            self.assertIn("std_dev", sim_results)
            self.assertIn("percentile_5", sim_results)
            self.assertIn("percentile_95", sim_results)

            # Verifica que los valores de la simulación no sean cero (o None)
            self.assertNotEqual(sim_results["mean"], 0)


if __name__ == "__main__":
    unittest.main()
