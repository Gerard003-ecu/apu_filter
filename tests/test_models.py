import os
import sys
import unittest

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.probability_models import run_monte_carlo_simulation


class TestProbabilityModels(unittest.TestCase):
    def test_simulation_returns_correct_structure(self):
        """
        Verifies that the simulation result has the correct structure.
        """
        apu_details = [
            {"CATEGORIA": "MATERIALES", "VALOR_UNITARIO": 100, "CANTIDAD": 10},
            {"CATEGORIA": "MANO DE OBRA", "VALOR_UNITARIO": 50, "CANTIDAD": 5},
        ]
        result = run_monte_carlo_simulation(apu_details, num_simulations=10)

        self.assertIsInstance(result, dict)
        self.assertIn("mean", result)
        self.assertIn("std_dev", result)
        self.assertIn("percentile_5", result)
        self.assertIn("percentile_95", result)

    def test_simulation_values_are_reasonable(self):
        """
        Verifies that the simulation values are reasonable for a known input.
        """
        # Con un costo determinista de $1000 (100*10) y sin variaciones aleatorias
        # para la categoría 'OTROS', el resultado debe ser muy cercano a 1000.
        apu_details = [
            {"CATEGORIA": "OTROS", "VALOR_UNITARIO": 100, "CANTIDAD": 10},
        ]

        # Usamos un número alto de simulaciones para que la media se acerque al valor real
        result = run_monte_carlo_simulation(apu_details, num_simulations=10000)

        # El costo esperado (mean) debe ser muy cercano al costo determinista
        self.assertAlmostEqual(
            result["mean"], 1000, delta=1
        )  # Delta muy pequeño porque no hay aleatoriedad

        # Con un apu_details que sí tiene componentes aleatorios
        apu_details_random = [
            {"CATEGORIA": "MATERIALES", "VALOR_UNITARIO": 80, "CANTIDAD": 10},  # 800
            {"CATEGORIA": "MANO DE OBRA", "VALOR_UNITARIO": 20, "CANTIDAD": 10},  # 200
        ]

        result_random = run_monte_carlo_simulation(apu_details_random, num_simulations=1000)

        # El costo esperado debe estar cerca de 1000, pero con más margen por la aleatoriedad
        self.assertAlmostEqual(result_random["mean"], 1000, delta=100)

        # Los percentiles deben tener sentido
        self.assertLess(result_random["percentile_5"], result_random["mean"])
        self.assertGreater(result_random["percentile_95"], result_random["mean"])
        self.assertGreater(result_random["std_dev"], 0)

    def test_simulation_handles_empty_input(self):
        """
        Verifies that the simulation handles empty input gracefully.
        """
        apu_details = []
        result = run_monte_carlo_simulation(apu_details, num_simulations=10)

        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["std_dev"], 0)
        self.assertEqual(result["percentile_5"], 0)
        self.assertEqual(result["percentile_95"], 0)

# ======================================================================
# AÑADIR ESTE BLOQUE AL FINAL DEL ARCHIVO
# ======================================================================
if __name__ == '__main__':
    unittest.main()