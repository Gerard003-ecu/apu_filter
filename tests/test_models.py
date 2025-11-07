# tests/test_probability_models.py

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.probability_models import run_monte_carlo_simulation, sanitize_value


class TestProbabilityModels(unittest.TestCase):
    """
    Pruebas unitarias para los modelos de probabilidad.

    Esta suite valida la robustez, coherencia y correcta integración entre
    sanitize_value() y run_monte_carlo_simulation(), asegurando que:
    - Los valores NaN/inválidos se manejan con consistencia.
    - Las salidas son siempre estructuradas y seguras para serialización.
    - Los casos borde (vacíos, faltantes, ceros, tipos erróneos) no rompen el sistema.
    - La lógica estadística es razonable y estable.
    """

    # -----------------------------
    # TESTS PARA sanitize_value
    # -----------------------------

    def test_sanitize_value_handles_nan(self):
        """sanitize_value debe devolver None cuando recibe np.nan o pd.NA."""
        self.assertIsNone(sanitize_value(np.nan))
        self.assertIsNone(sanitize_value(pd.NA))
        self.assertIsNone(sanitize_value(float('nan')))

    def test_sanitize_value_handles_numeric_values(self):
        """sanitize_value debe devolver el valor numérico original como float."""
        self.assertEqual(sanitize_value(10), 10.0)
        self.assertEqual(sanitize_value(0), 0.0)
        self.assertEqual(sanitize_value(0.5), 0.5)
        self.assertEqual(sanitize_value(-100.7), -100.7)
        self.assertEqual(sanitize_value(np.int32(42)), 42.0)
        self.assertEqual(sanitize_value(np.float64(3.14)), 3.14)

    def test_sanitize_value_handles_non_numeric(self):
        """sanitize_value debe devolver valores no numéricos sin modificar."""
        self.assertEqual(sanitize_value("hello"), "hello")
        self.assertEqual(sanitize_value(True), True)
        self.assertEqual(sanitize_value(None), None)
        self.assertEqual(sanitize_value([1, 2, 3]), [1, 2, 3])

    # -----------------------------
    # TESTS PARA run_monte_carlo_simulation
    # -----------------------------

    def test_simulation_returns_correct_structure_with_valid_input(self):
        """
        Verifica que la salida tenga las claves esperadas y que los valores sean
        floats o None (nunca np.nan).
        """
        apu_details = [
            {"VR_TOTAL": 1000, "CANTIDAD": 10},
            {"VR_TOTAL": 250, "CANTIDAD": 5},
        ]
        result = run_monte_carlo_simulation(apu_details, num_simulations=100)

        # Estructura
        self.assertIsInstance(result, dict)
        expected_keys = {"mean", "std_dev", "percentile_5", "percentile_95"}
        self.assertEqual(set(result.keys()), expected_keys)

        # Tipos de valor: deben ser float o None (nunca np.nan)
        for key in expected_keys:
            value = result[key]
            self.assertTrue(
                isinstance(value, (float, type(None))),
                f"Valor de {key} debe ser float o None, no {type(value)}"
            )
            self.assertNotEqual(value, np.nan, f"Valor de {key} no debe ser np.nan")

    def test_simulation_handles_empty_input_list(self):
        """
        Una lista vacía de apu_details debe retornar None para todas las métricas.
        No debe retornar ceros, porque eso implica "costo cero", no "sin datos".
        """
        result = run_monte_carlo_simulation([], num_simulations=10)
        expected = {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        }
        self.assertEqual(result, expected)

    def test_simulation_handles_missing_required_columns(self):
        """
        Si faltan columnas requeridas ('VR_TOTAL' o 'CANTIDAD'), debe retornar None.
        No debe intentar inferir o rellenar.
        """
        # Falta VR_TOTAL
        result1 = run_monte_carlo_simulation([{"CANTIDAD": 10}], num_simulations=10)
        self.assertEqual(result1, {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        })

        # Falta CANTIDAD
        result2 = run_monte_carlo_simulation([{"VR_TOTAL": 100}], num_simulations=10)
        self.assertEqual(result2, {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        })

        # Faltan ambas
        result3 = run_monte_carlo_simulation([{"OTRO_CAMPO": 5}], num_simulations=10)
        self.assertEqual(result3, {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        })

    def test_simulation_handles_zero_or_negative_cost(self):
        """
        Costos base (VR_TOTAL * CANTIDAD) <= 0 deben ser ignorados.
        Si todos los costos son inválidos, debe retornar None.
        """
        # VR_TOTAL = 0
        result1 = run_monte_carlo_simulation([{"VR_TOTAL": 0, "CANTIDAD": 10}], num_simulations=100)
        self.assertEqual(result1, {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        })

        # CANTIDAD = 0
        result2 = run_monte_carlo_simulation([{"VR_TOTAL": 100, "CANTIDAD": 0}], num_simulations=100)
        self.assertEqual(result2, {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        })

        # VR_TOTAL negativo
        result3 = run_monte_carlo_simulation([{"VR_TOTAL": -50, "CANTIDAD": 1}], num_simulations=100)
        self.assertEqual(result3, {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        })

        # Mixto: uno válido, uno inválido
        result4 = run_monte_carlo_simulation([
            {"VR_TOTAL": 100, "CANTIDAD": 1},
            {"VR_TOTAL": 0, "CANTIDAD": 1}
        ], num_simulations=100)
        # Solo el primero cuenta → costo base = 100 → simulación debe tener media cercana a 100
        self.assertIsNotNone(result4['mean'])
        self.assertGreater(result4['mean'], 50)  # por la variabilidad
        self.assertLess(result4['mean'], 150)

    def test_simulation_with_valid_positive_costs(self):
        """
        Verifica que la simulación genere resultados estadísticamente razonables
        con entradas válidas.
        """
        # Un solo item: VR_TOTAL=1000, CANTIDAD=1 → base=1000
        apu_details = [{"VR_TOTAL": 1000, "CANTIDAD": 1}]
        result = run_monte_carlo_simulation(apu_details, num_simulations=5000)

        # La media debe estar cerca de 1000 (con margen por variabilidad)
        self.assertIsNotNone(result['mean'])
        self.assertGreater(result['mean'], 900)
        self.assertLess(result['mean'], 1100)

        # La desviación estándar debe ser positiva (por la volatilidad del 10%)
        self.assertGreater(result['std_dev'], 0)
        self.assertLess(result['std_dev'], 200)  # 10% de 1000 = 100, más ruido

        # Percentiles deben ser lógicos
        self.assertLess(result['percentile_5'], result['mean'])
        self.assertLess(result['mean'], result['percentile_95'])

    def test_simulation_with_multiple_items(self):
        """
        Verifica que múltiples APU se sumen correctamente en la simulación.
        """
        apu_details = [
            {"VR_TOTAL": 800, "CANTIDAD": 10},  # base = 8000
            {"VR_TOTAL": 200, "CANTIDAD": 10},  # base = 2000
        ]
        total_base_cost = 10000

        result = run_monte_carlo_simulation(apu_details, num_simulations=5000)

        self.assertIsNotNone(result['mean'])
        self.assertGreater(result['mean'], 9000)
        self.assertLess(result['mean'], 11000)
        self.assertGreater(result['std_dev'], 0)
        self.assertLess(result['percentile_5'], result['mean'])
        self.assertGreater(result['percentile_95'], result['mean'])

    def test_simulation_handles_invalid_input_types(self):
        """
        La función debe manejar entradas no listas sin lanzar excepciones.
        """
        # None
        result1 = run_monte_carlo_simulation(None, num_simulations=10)
        self.assertEqual(result1, {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        })

        # String
        result2 = run_monte_carlo_simulation("not a list", num_simulations=10)
        self.assertEqual(result2, {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        })

        # Número
        result3 = run_monte_carlo_simulation(123, num_simulations=10)
        self.assertEqual(result3, {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        })

    def test_simulation_throws_on_invalid_num_simulations(self):
        """
        Debe lanzar ValueError si num_simulations no es entero positivo.
        """
        with self.assertRaises(ValueError):
            run_monte_carlo_simulation([{"VR_TOTAL": 100, "CANTIDAD": 1}], num_simulations=0)

        with self.assertRaises(ValueError):
            run_monte_carlo_simulation([{"VR_TOTAL": 100, "CANTIDAD": 1}], num_simulations=-5)

        with self.assertRaises(ValueError):
            run_monte_carlo_simulation([{"VR_TOTAL": 100, "CANTIDAD": 1}], num_simulations="100")

        # Esto debe funcionar
        try:
            run_monte_carlo_simulation([{"VR_TOTAL": 100, "CANTIDAD": 1}], num_simulations=100)
        except ValueError:
            self.fail("num_simulations=100 debería ser válido")

    def test_simulation_throws_on_negative_volatility(self):
        """
        Debe lanzar ValueError si volatility_factor es negativo.
        """
        with self.assertRaises(ValueError):
            run_monte_carlo_simulation([{"VR_TOTAL": 100, "CANTIDAD": 1}], num_simulations=10, volatility_factor=-0.1)

        # Esto debe funcionar
        try:
            run_monte_carlo_simulation([{"VR_TOTAL": 100, "CANTIDAD": 1}], num_simulations=10, volatility_factor=0.0)
        except ValueError:
            self.fail("volatility_factor=0.0 debe ser válido")

    def test_simulation_with_custom_volatility_factor(self):
        """
        Verifica que el factor de volatilidad afecte la desviación estándar.
        """
        apu_details = [{"VR_TOTAL": 100, "CANTIDAD": 1}]  # base = 100

        # Baja volatilidad → baja desviación
        result_low = run_monte_carlo_simulation(apu_details, num_simulations=2000, volatility_factor=0.01)
        # Alta volatilidad → alta desviación
        result_high = run_monte_carlo_simulation(apu_details, num_simulations=2000, volatility_factor=0.2)

        self.assertGreater(result_high['std_dev'], result_low['std_dev'])
        self.assertGreater(result_high['std_dev'], 0)
        self.assertGreater(result_low['std_dev'], 0)

    def test_simulation_with_min_cost_threshold(self):
        """
        Verifica que el umbral mínimo funcione correctamente (aunque no es expuesto en API pública,
        el código interno lo usa — no es necesario exponerlo, pero debe funcionar).
        """
        # Un costo de 50, con umbral 100 → debe ignorarse
        apu_details = [{"VR_TOTAL": 50, "CANTIDAD": 2}]  # base = 100 → igual al umbral
        result = run_monte_carlo_simulation(apu_details, num_simulations=100, min_cost_threshold=100)
        # Como base = 100, no se ignora
        self.assertIsNotNone(result['mean'])

        apu_details2 = [{"VR_TOTAL": 49, "CANTIDAD": 2}]  # base = 98 < 100 → ignorado
        result2 = run_monte_carlo_simulation(apu_details2, num_simulations=100, min_cost_threshold=100)
        self.assertEqual(result2, {
            'mean': None,
            'std_dev': None,
            'percentile_5': None,
            'percentile_95': None
        })

    def test_simulation_handles_mixed_data_types_and_nan(self):
        """
        Verifica que el sistema maneje entradas con tipos mixtos y NaN sin fallar.
        """
        apu_details = [
            {"VR_TOTAL": "100", "CANTIDAD": 1},      # string numérico → debe parsear
            {"VR_TOTAL": np.nan, "CANTIDAD": 1},      # NaN → ignorado
            {"VR_TOTAL": 200, "CANTIDAD": "2"},       # string numérico → debe parsear
            {"VR_TOTAL": "abc", "CANTIDAD": 1},       # no numérico → ignorado
            {"VR_TOTAL": 150, "CANTIDAD": None},      # None → ignorado
        ]
        result = run_monte_carlo_simulation(apu_details, num_simulations=100)

        # Solo dos entradas válidas: 100*1 + 200*2 = 500
        self.assertIsNotNone(result['mean'])
        self.assertGreater(result['mean'], 400)
        self.assertLess(result['mean'], 600)

    def test_simulation_output_never_returns_numpy_nan(self):
        """
        Asegura que ninguna salida contenga np.nan — solo None o float.
        Esto es crítico para integraciones con JSON, APIs, etc.
        """
        apu_details = [{"VR_TOTAL": 100, "CANTIDAD": 1}]
        result = run_monte_carlo_simulation(apu_details, num_simulations=10)

        for key, value in result.items():
            self.assertFalse(
                isinstance(value, float) and np.isnan(value),
                f"Valor de {key} no debe ser np.nan (aunque sea float)"
            )


if __name__ == "__main__":
    unittest.main()