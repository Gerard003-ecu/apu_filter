import os
import sys
import unittest

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.estimator import calculate_estimate
from app.procesador_csv import process_all_files

# Reuse test data from test_app
from tests.test_procesador_csv import (
    APUS_DATA,
    INSUMOS_DATA,
    PRESUPUESTO_DATA,
    TEST_CONFIG,
)


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

    def test_calculate_estimate_logic_two_step(self):
        """
        Tests the refactored calculate_estimate function with the new two-step search logic.
        """
        data_store = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
        )

        # 1. Caso de prueba principal con la nueva lógica
        params = {"material": "TEJA", "cuadrilla": "1"}
        result = calculate_estimate(params, data_store, TEST_CONFIG)

        self.assertNotIn("error", result)
        self.assertIn("Suministro: SUMINISTRO TEJA TRAPEZOIDAL ROJA CAL.28", result["apu_encontrado"])
        self.assertIn("Tarea: INSTALACION TEJA TRAPEZOIDAL", result["apu_encontrado"])
        self.assertIn("Cuadrilla: CUADRILLA TIPO 1 (1 OF + 2 AYU)", result["apu_encontrado"])
        self.assertAlmostEqual(result["valor_suministro"], 50000.0)
        self.assertAlmostEqual(result["valor_instalacion"], 11760.0)
        self.assertAlmostEqual(result["valor_construccion"], 61760.0)
        self.assertAlmostEqual(result["rendimiento_m2_por_dia"], 25.0)

    def test_calculate_estimate_flexible_search(self):
        """
        Tests the flexible search logic and validates the final calculated value.
        """
        data_store = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
        )

        params = {"material": "TEJA", "cuadrilla": "2"}
        result = calculate_estimate(params, data_store, TEST_CONFIG)

        self.assertNotIn("error", result)
        self.assertAlmostEqual(result["valor_construccion"], 68160.0, places=2)


if __name__ == "__main__":
    unittest.main()
