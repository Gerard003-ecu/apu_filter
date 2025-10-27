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
    @classmethod
    def setUpClass(cls):
        """Crea archivos temporales con los nuevos datos de prueba."""
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
        """Elimina los archivos temporales después de las pruebas."""
        os.remove(cls.presupuesto_path)
        os.remove(cls.apus_path)
        os.remove(cls.insumos_path)

    def test_process_all_files_structure_and_calculations(self):
        """
        Prueba la estructura y los cálculos del `process_all_files` con los nuevos datos.
        Verifica que el valor de construcción para un APU específico sea el esperado.
        """
        resultado = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
        )

        self.assertIsInstance(resultado, dict)
        self.assertNotIn("error", resultado, f"El procesamiento falló inesperadamente: {resultado.get('error')}")

        presupuesto_procesado = resultado["presupuesto"]
        self.assertEqual(len(presupuesto_procesado), 4, "Deberían procesarse 4 ítems del presupuesto.")

        # Buscar el ítem 1.1 y verificar su valor de construcción
        item1_1 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1.1"), None
        )
        self.assertIsNotNone(item1_1, "El ítem 1.1 no fue encontrado en el resultado.")
        self.assertAlmostEqual(item1_1["VALOR_CONSTRUCCION_UN"], 52000.0, places=2)

    def test_abnormally_high_cost_triggers_error(self):
        """
        Prueba que un costo anormalmente alto, proveniente de una cantidad
        excesiva en el presupuesto, activa un error de validación.
        """
        # Datos con una cantidad extremadamente alta en el presupuesto
        PRESUPUESTO_ALTO = (
            "ITEM;DESCRIPCION;UND;CANT.;VR. UNIT;VR.TOTAL\n"
            "1.1;SUMINISTRO TEJA;M2;20000000;52000;1040000000000\n" # Cantidad > 1e6, Costo Total > 1e12
        )
        presupuesto_alto_path = "test_presupuesto_alto.csv"
        with open(presupuesto_alto_path, "w", encoding="latin1") as f:
            f.write(PRESUPUESTO_ALTO)

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

        os.remove(presupuesto_alto_path)
        os.remove(apus_alto_path)

    def test_cartesian_explosion_on_final_merge(self):
        """
        Prueba que el merge final retorna un error si df_apu_costos tiene
        CODIGO_APU duplicados, previniendo una explosión cartesiana.
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
