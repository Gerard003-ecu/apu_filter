import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importar la app de Flask y las funciones a probar
from app.procesador_csv import (
    process_all_files,
)

# Importar los datos de prueba centralizados
from tests.test_data import (
    APUS_DATA,
    INSUMOS_DATA,
    PRESUPUESTO_DATA,
    TEST_CONFIG,
)


class TestCSVProcessorWithNewData(unittest.TestCase):
    """
    Pruebas de integración para la función `process_all_files`.

    Esta clase valida el flujo completo de procesamiento de los archivos CSV
    de entrada (presupuesto, APUs, insumos). Se asegura de que los datos se
    lean, procesen, combinen y validen correctamente, y de que se manejen
-    adecuadamente los casos de error como costos excesivos o duplicados.
    """
    @classmethod
    def setUpClass(cls):
        """
        Configura el entorno de prueba creando archivos temporales con datos
        realistas para ser utilizados en todas las pruebas de la clase.
        """
        cls.presupuesto_path = "test_presupuesto_new.csv"
        with open(cls.presupuesto_path, "w", encoding="latin1") as f:
            f.write(PRESUPUESTO_DATA)

        cls.apus_path = "test_apus_new.csv"
        with open(cls.apus_path, "w", encoding="latin1") as f:
            f.write(APUS_DATA)

        cls.insumos_path = "test_insumos_new.csv"
        with open(cls.insumos_path, "w", encoding="latin1") as f:
            f.write(INSUMOS_DATA)

    @classmethod
    def tearDownClass(cls):
        """
        Limpia el entorno de prueba eliminando los archivos temporales
        creados después de que todas las pruebas hayan finalizado.
        """
        os.remove(cls.presupuesto_path)
        os.remove(cls.apus_path)
        os.remove(cls.insumos_path)

    def test_process_all_files_structure_and_calculations(self):
        """
        Prueba el caso de éxito del procesamiento.

        Verifica que la estructura del `data_store` resultante sea la correcta,
        que no contenga errores y que los cálculos clave (como el valor de
        construcción de un APU) sean precisos según los datos de entrada.
        """
        resultado = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
        )

        self.assertIsInstance(resultado, dict)
        self.assertNotIn(
            "error", resultado, f"El procesamiento falló: {resultado.get('error')}"
        )

        presupuesto_procesado = resultado["presupuesto"]
        self.assertEqual(
            len(presupuesto_procesado), 4, "Deberían procesarse 4 ítems."
        )

        # Buscar el ítem 1.1 y verificar su valor de construcción
        item1_1 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1.1"), None
        )
        self.assertIsNotNone(item1_1, "El ítem 1.1 no fue encontrado.")
        self.assertAlmostEqual(item1_1["VALOR_CONSTRUCCION_UN"], 50000.0, places=2)

    def test_abnormally_high_cost_triggers_error(self):
        """
        Valida que el sistema detecte y rechace un presupuesto con costos
        totales anormalmente altos, previniendo errores por datos de entrada
        incorrectos.
        """
        # Datos con una cantidad extremadamente alta en el presupuesto
        presupuesto_alto = (
            "ITEM;DESCRIPCION;UND;CANT.;VR. UNIT;VR.TOTAL\n"
            "1.1;SUMINISTRO TEJA;M2;20000000;52000;1040000000000\n"
        )
        presupuesto_alto_path = "test_presupuesto_alto.csv"
        with open(presupuesto_alto_path, "w", encoding="latin1") as f:
            f.write(presupuesto_alto)

        APUS_ALTO = (
            "ITEM: 1.1; UNIDAD: M2\n"
            "SUMINISTRO TEJA TRAPEZOIDAL ROJA CAL.28\n"
            "MATERIALES\n"
            "TEJA TRAPEZOIDAL ROJA;M2;1.05;;47619;52000\n"
        )
        apus_alto_path = "test_apus_alto.csv"
        with open(apus_alto_path, "w", encoding="latin1") as f:
            f.write(APUS_ALTO)

        with self.assertLogs('app.procesador_csv', level='ERROR') as cm:
            resultado = process_all_files(
                presupuesto_alto_path, apus_alto_path, self.insumos_path, config=TEST_CONFIG
            )
            # Verificar que el log contiene el mensaje de error esperado
            self.assertTrue(any("COSTO TOTAL ANORMALMENTE ALTO" in msg for msg in cm.output))

        self.assertIn("error", resultado)
        self.assertIn("Costo total anormalmente alto", resultado["error"])

        # Limpiar los archivos temporales creados para esta prueba
        if os.path.exists(presupuesto_alto_path):
            os.remove(presupuesto_alto_path)
        if os.path.exists(apus_alto_path):
            os.remove(apus_alto_path)

    def test_cartesian_explosion_on_final_merge(self):
        """
        Prueba la salvaguarda contra explosiones cartesianas.

        Verifica que `process_all_files` retorne un error si detecta códigos de
        APU duplicados en los costos calculados, lo que podría llevar a un
        merge incorrecto y a datos inflados.
        """
        malformed_apu_costos = pd.DataFrame({
            'CODIGO_APU': ['1.1', '1.2', '1.1'], # '1.1' está duplicado
            'VALOR_CONSTRUCCION_UN': [100, 200, 150]
        })

        with patch('app.procesador_csv._calculate_apu_costs_and_metadata',
                   return_value=(malformed_apu_costos, pd.DataFrame(), pd.DataFrame())):
            with self.assertLogs("app.procesador_csv", level="ERROR") as cm:
                resultado = process_all_files(
                    self.presupuesto_path,
                    self.apus_path,
                    self.insumos_path,
                    TEST_CONFIG
                )
                # Verificar que se loguea el error de explosión cartesiana
                self.assertTrue(
                    any("EXPLOSIÓN CARTESIANA DETECTADA" in msg for msg in cm.output)
                )

        self.assertIn("error", resultado)
        self.assertIn("Explosión cartesiana detectada", resultado["error"])
