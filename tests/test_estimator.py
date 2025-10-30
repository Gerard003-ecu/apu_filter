import os
import sys
import unittest

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.estimator import calculate_estimate
from app.procesador_csv import process_all_files

# Importar los datos de prueba centralizados y realistas
from tests.test_data import (
    APUS_DATA,
    INSUMOS_DATA,
    PRESUPUESTO_DATA,
    TEST_CONFIG,
)


class TestEstimatorWithNewData(unittest.TestCase):
    """
    Pruebas de integración para la función `calculate_estimate`.

    Utiliza un conjunto de datos centralizado y realista para simular el
    proceso completo desde la carga de archivos hasta el cálculo de una
-    estimación. Valida la lógica de búsqueda de APUs (suministro, instalación,
    cuadrilla) y la precisión de los cálculos finales.
    """
    @classmethod
    def setUpClass(cls):
        """
        Prepara el entorno de prueba para toda la clase.

        Crea archivos temporales (presupuesto, APUs, insumos) a partir de los
        datos de prueba centralizados y los procesa una sola vez para
        optimizar el tiempo de ejecución de las pruebas. El `data_store`
        resultante se reutiliza en todos los tests de la clase.
        """
        cls.presupuesto_path = "test_presupuesto_estimator.csv"
        with open(cls.presupuesto_path, "w", encoding="latin1") as f:
            f.write(PRESUPUESTO_DATA)

        cls.apus_path = "test_apus_estimator.csv"
        with open(cls.apus_path, "w", encoding="latin1") as f:
            f.write(APUS_DATA)

        cls.insumos_path = "test_insumos_estimator.csv"
        with open(cls.insumos_path, "w", encoding="latin1") as f:
            f.write(INSUMOS_DATA)

        # Procesar los datos una sola vez para toda la clase de prueba
        cls.data_store = process_all_files(
            cls.presupuesto_path, cls.apus_path, cls.insumos_path, config=TEST_CONFIG
        )
        if "error" in cls.data_store:
            raise RuntimeError(f"La preparación de datos falló: {cls.data_store['error']}")


    @classmethod
    def tearDownClass(cls):
        """
        Limpia el entorno después de que todas las pruebas de la clase se hayan
        ejecutado, eliminando los archivos temporales creados.
        """
        os.remove(cls.presupuesto_path)
        os.remove(cls.apus_path)
        os.remove(cls.insumos_path)

    def test_calculate_estimate_logic_two_step(self):
        """
        Prueba el escenario ideal donde existen APUs claros y separados para
        suministro e instalación de un material.

        Verifica que `calculate_estimate` identifique correctamente cada APU,
        extraiga los valores correctos y calcule el costo total de construcción
        y el rendimiento esperado.
        """
        params = {"material": "TEJA", "cuadrilla": "1"}
        result = calculate_estimate(params, self.data_store, TEST_CONFIG)

        self.assertNotIn("error", result)
        # Verificar que se encontraron las descripciones correctas
        self.assertIn(
            "Suministro: SUMINISTRO TEJA TRAPEZOIDAL ROJA CAL.28",
            result["apu_encontrado"],
        )
        self.assertIn("Tarea: INSTALACION TEJA TRAPEZOIDAL", result["apu_encontrado"])
        self.assertIn(
            "Cuadrilla: CUADRILLA TIPO 1 (1 OF + 2 AYU)", result["apu_encontrado"]
        )

        # Verificar los valores calculados con los nuevos datos
        self.assertAlmostEqual(result["valor_suministro"], 52000.0, places=2)
        self.assertAlmostEqual(result["valor_instalacion"], 11760.0, places=2)
        self.assertAlmostEqual(result["valor_construccion"], 63760.0, places=2)
        self.assertAlmostEqual(result["rendimiento_m2_por_dia"], 25.0, places=2)

    def test_calculate_estimate_flexible_search(self):
        """
        Prueba la capacidad del estimador para manejar casos donde no existe un
        APU de instalación explícito para un material.

        Verifica que se encuentre un APU de suministro genérico y que el sistema
        cree una tarea de instalación sintética, reflejándose en los resultados
        finales.
        """
        params = {"material": "PINTURA", "cuadrilla": "1"} # Usando el APU de pintura
        result = calculate_estimate(params, self.data_store, TEST_CONFIG)

        self.assertNotIn("error", result, f"El cálculo falló: {result.get('error')}")

        # Verificar que se encontró el APU de suministro correcto (pintura)
        self.assertIn(
            "Suministro: APU DE SUMINISTRO PINTURA",
            result["apu_encontrado"],
        )
        # Verificar que la tarea de instalación se infiere correctamente
        self.assertIn(
            "Tarea: INSTALACION PINTURA ANTICORROSIVA", result["apu_encontrado"]
        )

        # Verificar el valor de construcción final
        # Suministro (65000) + Instalación (0, porque no hay tarea de instalación)
        self.assertAlmostEqual(result["valor_construccion"], 65000.0, places=2)
