import os
import sys
import unittest
from unittest.mock import patch

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.estimator import calculate_estimate
from app.procesador_csv import _cached_csv_processing, process_all_files

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

    @patch("app.procesador_csv.config", new_callable=lambda: TEST_CONFIG)
    @patch("app.estimator.config", new_callable=lambda: TEST_CONFIG)
    def test_new_search_logic(self, mock_estimator_config, mock_processor_config):
        """
        Tests the new _find_best_match logic for installation APUs.
        """
        data_store = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path
        )

        # 1. Test Strict Search: material + cuadrilla
        # Should find "INSTALACION PANEL SANDWICH CUADRILLA DE 5" because all keywords match.
        params_strict = {"material": "PANEL SANDWICH", "cuadrilla": "5"}
        result_strict = calculate_estimate(params_strict, data_store)

        self.assertIn("Coincidencia estricta encontrada", result_strict["log"])
        self.assertIn(
            "Instalación: INSTALACION PANEL SANDWICH CUADRILLA DE 5",
            result_strict["apu_encontrado"],
        )
        self.assertAlmostEqual(result_strict["valor_instalacion"], 100000.0)

        # 2. Test Flexible Search: material + non-existent cuadrilla
        # Should find a flexible match on "canal" since "cuadrilla de 99" doesn't exist.
        params_flexible = {"material": "CANAL SOLO", "cuadrilla": "99"}
        result_flexible = calculate_estimate(params_flexible, data_store)

        self.assertIn("Coincidencia flexible encontrada", result_flexible["log"])
        self.assertNotIn("Coincidencia estricta encontrada", result_flexible["log"])
        # It should find the first "CANAL" APU, which is "INSTALACION CANAL CUADRILLA DE 5"
        self.assertIn(
            "Instalación: INSTALACION CANAL CUADRILLA DE 5",
            result_flexible["apu_encontrado"],
        )
        self.assertAlmostEqual(result_flexible["valor_instalacion"], 90000.0)

        # 3. Test Flexible Search for "PANEL SANDWICH"
        # The material is "PANEL SANDWICH", but the APU is "INSTALACION PANEL TIPO SANDWICH".
        # The strict search will now find a match for "panel" and "sandwich"
        params_sandwich = {"material": "PANEL SANDWICH", "cuadrilla": "0"} # No specific cuadrilla
        result_sandwich = calculate_estimate(params_sandwich, data_store)
        self.assertIn("Coincidencia estricta encontrada", result_sandwich["log"])
        self.assertIn(
            "Instalación: INSTALACION PANEL TIPO SANDWICH",
            result_sandwich["apu_encontrado"],
        )
        self.assertAlmostEqual(result_sandwich["valor_instalacion"], 95000.0)


if __name__ == "__main__":
    unittest.main()
