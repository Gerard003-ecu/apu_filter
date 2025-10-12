import io
import json
import os
import sys
import unittest

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Importar la app de Flask y las funciones a probar
from app.app import create_app, user_sessions
from app.procesador_csv import (
    process_all_files,
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
    "M.O. CORTE Y DOBLEZ;1;1;1;1;50,00\n" # Formato corregido
    "\n"
    "ACABADOS FINALES\n"
    "ITEM: 1,2\n"
    "MATERIALES;;;;;\n"
    "Pintura Anticorrosiva;GL;5,0;0;5,00;25,00\n"
    "MANO DE OBRA;;;;;\n"
    "M.O. Mano de Obra Especializada;1;1;1;1;200,00\n" # Formato corregido
    "\n"
    "INSTALACION TEJA SENCILLA CUBIERTA\n"
    "ITEM: 1,3\n"
    "MANO DE OBRA;;;;;\n"
    # Formato: Desc;Jornal Base;Prestaciones;Jornal Total;Rendimiento;Valor Total
    "M.O. Ayudante;10000,00;0;10000,00;8,0;1250,00\n"
    "\n"
    "SUMINISTRO TEJA SENCILLA\n"
    "ITEM: 1,4\n"
    "MATERIALES;;;;;\n"
    "TEJA SENCILLA;M2;1,0;0;50000,00;50000,00\n"
    "\n"
    "CUADRILLA DE 4\n"
    "ITEM: 1,6;UNIDAD: DIA\n"
    "MANO DE OBRA;;;;;\n"
    "OFICIAL;DIA;1;0;120000,00;120000,00\n"
    "AYUDANTE;DIA;1;0;80000,00;80000,00\n"
    "\n"
    "INGENIERO RESIDENTE\n"
    "ITEM: 1,5\n"
    "MANO DE OBRA;;;;;\n"
    "INGENIERO RESIDENTE;150000;0;150000;10;15000,00\n" # Formato corregido
    "\n"
    "INSTALACION CANAL CUADRILLA DE 5\n"
    "ITEM: 2,1\n"
    "MANO DE OBRA;;;;;\n"
    "M.O. Ayudante;10000;0;10000;0,11;90000\n"
    "\n"
    "INSTALACION CANAL CUADRILLA DE 3\n"
    "ITEM: 2,4\n"
    "MANO DE OBRA;;;;;\n"
    "M.O. Ayudante;10000;0;10000;0,12;85000\n"
    "\n"
    "INSTALACION CANAL LAMINA\n"
    "ITEM: 2,2\n"
    "MANO DE OBRA;;;;;\n"
    "M.O. Ayudante;10000;0;10000;5,0;2000\n"
    "\n"
    "INSTALACION PANEL TIPO SANDWICH\n"
    "ITEM: 2,3\n"
    "MANO DE OBRA;;;;;\n"
    "M.O. Ayudante;10000,00;0;10000,00;2,0;5000,00\n"
    "\n"
    "INSTALACION PANEL SANDWICH CUADRILLA DE 5\n"
    "ITEM: 2,5\n"
    "MANO DE OBRA;;;;;\n"
    "M.O. Ayudante;10000;0;10000;0,1;100000\n"
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

    def test_process_all_files_structure_and_calculations(self):
        resultado = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
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
        # 'apus_detail' ahora es una lista, así que filtramos para encontrar el item
        apu_detail_completo = resultado["apus_detail"]
        apu_detail_ing_list = [
            item for item in apu_detail_completo if item.get("CODIGO_APU") == "1,5"
        ]
        self.assertEqual(len(apu_detail_ing_list), 1)
        ing_item = apu_detail_ing_list[0]
        # La descripción del insumo ahora viene de la columna 'DESCRIPCION_INSUMO'
        self.assertEqual(ing_item["DESCRIPCION_INSUMO"], "INGENIERO RESIDENTE")
        self.assertAlmostEqual(ing_item["CANTIDAD_APU"], 0.1)
        self.assertAlmostEqual(ing_item["PRECIO_UNIT_APU"], 150000)
        # El nombre final de la columna en el diccionario de salida es VR_TOTAL
        self.assertAlmostEqual(ing_item["VR_TOTAL"], 15000)


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

    def test_get_estimate_with_session(self):
        with self.client as c:
            data = {
                "presupuesto": self._get_test_file("presupuesto.csv", PRESUPUESTO_DATA),
                "apus": self._get_test_file("apus.csv", APUS_DATA),
                "insumos": self._get_test_file("insumos.csv", INSUMOS_DATA),
            }
            # Set the app config directly for the test
            c.application.config["APP_CONFIG"] = TEST_CONFIG
            c.post("/upload", data=data, content_type="multipart/form-data")

            # Usa los nuevos parámetros que incluyen la cuadrilla
            estimate_params = {"material": "TST", "cuadrilla": "4"}
            response = c.post("/api/estimate", json=estimate_params)
            self.assertEqual(response.status_code, 200)
            json_data = json.loads(response.data)

            # Los valores esperados deben coincidir con el cálculo en test_estimator
            self.assertAlmostEqual(json_data["valor_suministro"], 50000)
            self.assertAlmostEqual(json_data["valor_instalacion"], 25000)
            self.assertAlmostEqual(json_data["valor_construccion"], 75000)

if __name__ == "__main__":
    unittest.main()
