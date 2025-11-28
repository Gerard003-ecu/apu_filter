"""
Tests para los endpoints de herramientas (Diagnóstico y Limpieza).
"""
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from app.app import create_app


class TestApiTools(unittest.TestCase):
    def setUp(self):
        self.app = create_app("testing")
        self.client = self.app.test_client()
        self.temp_dir = tempfile.mkdtemp()

        # Fake file content
        self.dummy_apu_csv = "ITEM;UNIDAD;DESCRIPCION\n1;m2;Test Item"
        self.dummy_dirty_csv = "Name;Age;City\nJohn;30;NYC\nBad;Line"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_telemetry_status(self):
        """Prueba el endpoint de telemetría."""
        response = self.client.get("/api/telemetry/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("status", data)
        # Check for message instead of system_health which is not in business report
        self.assertIn("message", data)

    def test_diagnose_endpoint_missing_file(self):
        """Prueba error si falta el archivo en diagnose."""
        response = self.client.post("/api/tools/diagnose", data={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "MISSING_FILE")

    def test_diagnose_endpoint_success(self):
        """Prueba ejecución exitosa de diagnose (mocking logic via integration)."""
        data = {
            "file": (io.BytesIO(self.dummy_apu_csv.encode("utf-8")), "apus.csv"),
            "type": "apus"
        }
        response = self.client.post(
            "/api/tools/diagnose",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        json_resp = response.get_json()
        self.assertTrue(json_resp.get("success"))
        # Verifica que tenga stats
        self.assertIn("stats", json_resp)

    def test_clean_endpoint_success(self):
        """Prueba ejecución exitosa de clean."""
        data = {
            "file": (io.BytesIO(self.dummy_dirty_csv.encode("utf-8")), "dirty.csv"),
            "delimiter": ";"
        }
        response = self.client.post(
            "/api/tools/clean",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        json_resp = response.get_json()
        self.assertTrue(json_resp.get("success"))
        # Should contain stats from Cleaner
        self.assertIn("rows_written", json_resp)
        self.assertIn("rows_skipped", json_resp)
        # Should have message about deleted temp file
        self.assertIn("message", json_resp)

    def test_clean_endpoint_missing_file(self):
        """Prueba error si falta archivo en clean."""
        response = self.client.post("/api/tools/clean", data={})
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
