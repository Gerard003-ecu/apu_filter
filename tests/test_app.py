import io
import json
import os
import sys
import unittest

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Importar la app de Flask y las funciones a probar
from app.app import create_app, user_sessions
from tests.test_procesador_csv import (
    APUS_DATA,
    INSUMOS_DATA,
    PRESUPUESTO_DATA,
    TEST_CONFIG,
)


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
            self.assertAlmostEqual(json_data["valor_suministro"], 0.0)
            self.assertAlmostEqual(json_data["valor_instalacion"], 25000)
            self.assertAlmostEqual(json_data["valor_construccion"], 25000)

if __name__ == "__main__":
    unittest.main()
