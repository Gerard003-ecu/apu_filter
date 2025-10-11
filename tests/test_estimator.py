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
    def test_calculate_estimate_logic_two_step(self, mock_estimator_config, mock_processor_config):
        """
        Tests the refactored calculate_estimate function with the new two-step search logic.
        """
        data_store = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path
        )

        # 1. Caso de prueba principal con la nueva lógica
        params = {"material": "TST", "cuadrilla": "4"}
        result = calculate_estimate(params, data_store)

        self.assertNotIn("error", result)

        # Verificar que se encontraron los APUs correctos
        self.assertIn("Suministro: SUMINISTRO TEJA SENCILLA", result["apu_encontrado"])
        self.assertIn("Tarea: INSTALACION TEJA SENCILLA CUBIERTA", result["apu_encontrado"])
        self.assertIn("Cuadrilla: CUADRILLA DE 4", result["apu_encontrado"])

        # Verificar los valores calculados
        # APU Tarea (Rendimiento): TIEMPO = 0.125 días/un -> Rendimiento = 1 / 0.125 = 8 un/día
        # APU Cuadrilla (Costo): VALOR_CONSTRUCCION_UN = 120000 + 80000 = 200000 $/día
        # Costo Instalación = Costo Diario / Rendimiento = 200000 / 8 = 25000 $/un
        self.assertAlmostEqual(result["valor_suministro"], 50000.0)
        self.assertAlmostEqual(result["valor_instalacion"], 25000.0)
        self.assertAlmostEqual(result["valor_construccion"], 75000.0)
        self.assertAlmostEqual(result["tiempo_instalacion"], 0.125)

        # 2. Caso de prueba donde no se encuentra la cuadrilla
        params_no_cuadrilla = {"material": "PANEL TIPO SANDWICH", "cuadrilla": "99"}
        result_no_cuadrilla = calculate_estimate(params_no_cuadrilla, data_store)
        self.assertIn("--> No se encontró un APU para la cuadrilla especificada.", result_no_cuadrilla["log"])
        self.assertAlmostEqual(result_no_cuadrilla["valor_instalacion"], 0)
        # El rendimiento aún debe calcularse
        # TIEMPO = 0.5 -> Rendimiento = 2.0
        self.assertAlmostEqual(result_no_cuadrilla["tiempo_instalacion"], 0.5)

        # 3. Caso de prueba donde no se encuentra el APU de tarea
        params_no_task = {"material": "MATERIAL INEXISTENTE", "cuadrilla": "4"}
        result_no_task = calculate_estimate(params_no_task, data_store)
        self.assertIn("No se encontró APU de Tarea", result_no_task["log"])
        self.assertAlmostEqual(result_no_task["valor_instalacion"], 0)
        self.assertAlmostEqual(result_no_task["tiempo_instalacion"], 0)
        # El costo de la cuadrilla debe encontrarse
        self.assertIn("Cuadrilla: CUADRILLA DE 4", result_no_task["apu_encontrado"])


if __name__ == "__main__":
    unittest.main()
