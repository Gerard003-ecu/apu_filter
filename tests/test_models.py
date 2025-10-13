import os
import sys
import unittest
import numpy as np

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.probability_models import run_monte_carlo_simulation, sanitize_value


class TestProbabilityModels(unittest.TestCase):

    def test_sanitize_value(self):
        """Tests the sanitize_value helper function."""
        self.assertIsNone(sanitize_value(np.nan))
        self.assertEqual(sanitize_value(10), 10)
        self.assertEqual(sanitize_value(0.5), 0.5)

    def test_simulation_returns_correct_structure(self):
        """
        Verifies that the simulation result has the correct structure and types.
        """
        # The model now expects 'VR_TOTAL' instead of 'VALOR_UNITARIO'
        apu_details = [
            {"VR_TOTAL": 1000, "CANTIDAD": 10},
            {"VR_TOTAL": 250, "CANTIDAD": 5},
        ]
        result = run_monte_carlo_simulation(apu_details, num_simulations=10)

        self.assertIsInstance(result, dict)
        self.assertIn("mean", result)
        self.assertIn("std_dev", result)
        self.assertIn("percentile_5", result)
        self.assertIn("percentile_95", result)
        # Check that no value is NaN
        for key, value in result.items():
            self.assertIsNot(value, np.nan, f"Value for {key} should not be NaN")

    def test_simulation_values_are_reasonable(self):
        """
        Verifies that the simulation values are reasonable for a known input.
        """
        # The new simulation model expects 'VR_TOTAL' and introduces randomness.
        apu_details = [
            {"VR_TOTAL": 1000, "CANTIDAD": 10},
        ]

        # Use a high number of simulations for the mean to approach the expected value.
        result = run_monte_carlo_simulation(apu_details, num_simulations=10000)

        # The expected mean should be close to 1000.
        # A larger delta is needed due to the introduced randomness.
        self.assertAlmostEqual(result["mean"], 1000, delta=15)
        self.assertGreater(result["std_dev"], 0)

        # Test with multiple items
        apu_details_random = [
            {"VR_TOTAL": 800, "CANTIDAD": 10},
            {"VR_TOTAL": 200, "CANTIDAD": 10},
        ]
        result_random = run_monte_carlo_simulation(apu_details_random, num_simulations=1000)

        # The total cost is 1000. The mean of the simulation should be close to it.
        self.assertAlmostEqual(result_random["mean"], 1000, delta=100)

        # Percentiles should make sense
        self.assertLess(result_random["percentile_5"], result_random["mean"])
        self.assertGreater(result_random["percentile_95"], result_random["mean"])
        self.assertGreater(result_random["std_dev"], 0)

    def test_simulation_handles_empty_input(self):
        """
        Verifies that the simulation handles empty input gracefully.
        """
        apu_details = []
        result = run_monte_carlo_simulation(apu_details, num_simulations=10)
        expected = {'mean': 0, 'std_dev': 0, 'percentile_5': 0, 'percentile_95': 0}
        self.assertEqual(result, expected)

    def test_simulation_handles_zero_cost_input(self):
        """
        Verifies that the simulation handles input with zero total cost.
        """
        apu_details = [{"VR_TOTAL": 0, "CANTIDAD": 10}]
        result = run_monte_carlo_simulation(apu_details, num_simulations=100)
        expected = {'mean': 0, 'std_dev': 0, 'percentile_5': 0, 'percentile_95': 0}
        self.assertEqual(result, expected)

    def test_simulation_handles_missing_columns(self):
        """
        Verifies graceful handling of missing required columns.
        """
        apu_details = [{"CANTIDAD": 10}] # Missing VR_TOTAL
        result = run_monte_carlo_simulation(apu_details, num_simulations=10)
        expected = {'mean': 0, 'std_dev': 0, 'percentile_5': 0, 'percentile_95': 0}
        self.assertEqual(result, expected)

        apu_details_2 = [{"VR_TOTAL": 100}] # Missing CANTIDAD
        result_2 = run_monte_carlo_simulation(apu_details_2, num_simulations=10)
        self.assertEqual(result_2, expected)


if __name__ == "__main__":
    unittest.main()