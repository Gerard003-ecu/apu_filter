import os
import sys
import unittest
from unittest.mock import patch

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.estimator import calculate_estimate
from app.procesador_csv import process_all_files, _cached_csv_processing

# Reuse test data from test_app
from tests.test_app import APUS_DATA, INSUMOS_DATA, PRESUPUESTO_DATA, TEST_CONFIG


class TestEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presupuesto_path = "test_presupuesto_estimator.csv"
        with open(cls.presupuesto_path, "w", encoding="latin1") as f:
            f.write(PRESUPUESTO_DATA)
        cls.apus_path = "test_apus_estimator.csv"
        with open(cls.apus_path, "w", encoding="latin1") as f:
            f.write(APUS_DATA)
        cls.insumos_path = "test_insumos_estimator.csv"
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
    @patch("app.estimator.config", new_callable=lambda: TEST_CONFIG)
    def test_calculate_estimate_logic(self, mock_estimator_config, mock_processor_config):
        """
        Tests the refactored calculate_estimate function with the new search logic.
        """
        data_store = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path
        )

        # The search for 'TEJA SENCILLA' should now work on 'original_description'
        params_ok = {"tipo": "CUBIERTA", "material": "TST"}
        result = calculate_estimate(params_ok, data_store)

        self.assertNotIn("error", result)
        # Verify that the correct APUs were found based on the full description
        self.assertIn(
            "Suministro: SUMINISTRO TEJA SENCILLA", result["apu_encontrado"]
        )
        self.assertIn(
            "Instalación: INSTALACION TEJA SENCILLA CUBIERTA", result["apu_encontrado"]
        )
        self.assertAlmostEqual(result["valor_suministro"], 50000.0)
        self.assertAlmostEqual(result["valor_instalacion"], 80000.0)
        self.assertAlmostEqual(result["valor_construccion"], 130000.0)

        # Test case where no match is found
        params_fail = {"tipo": "CUBIERTA", "material": "ACERO INOXIDABLE"}
        result_fail = calculate_estimate(params_fail, data_store)
        self.assertAlmostEqual(result_fail["valor_suministro"], 0)
        self.assertAlmostEqual(result_fail["valor_instalacion"], 0)
        self.assertIn("No encontrado", result_fail["apu_encontrado"])


if __name__ == "__main__":
    unittest.main()