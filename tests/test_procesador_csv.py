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

# ======================================================================
# DATOS DE PRUEBA GLOBALES
# ======================================================================

PRESUPUESTO_DATA = (
    "ITEM;DESCRIPCION;UND;CANT.; VR. UNIT ; VR.TOTAL \n"
    "1.1;SUMINISTRO TEJA TRAPEZOIDAL;M2;100; 50000; 5000000\n"
    "1.2;INSTALACION TEJA TRAPEZOIDAL;M2;100; 8000; 800000\n"
    "2.1;CUADRILLA TIPO 1 (1 OF + 2 AYU);DIA;1; 280000; 280000\n"
    "2.2;CUADRILLA TIPO 2 (1 OF + 4 AYU);DIA;1; 440000; 440000\n"
)

APUS_DATA = (
    "ITEM: 1.1; UNIDAD: M2\n"
    "SUMINISTRO TEJA TRAPEZOIDAL ROJA CAL.28\n"
    "MATERIALES\n"
    "TEJA TRAPEZOIDAL ROJA;M2;1.05;0;47619;50000\n"
    "\n"
    "ITEM: 1.2; UNIDAD: M2\n"
    "INSTALACION TEJA TRAPEZOIDAL\n"
    "MANO DE OBRA\n"
    "CUADRILLA TIPO 1;;0.04;;280000;11200\n"
    "EQUIPO\n"
    "HERRAMIENTA MENOR;%;0.05;;11200;560\n"
    "\n"
    "ITEM: 2.1; UNIDAD: DIA\n"
    "CUADRILLA TIPO 1 (1 OF + 2 AYU)\n"
    "MANO DE OBRA\n"
    "OFICIAL;JOR;1;0;120000;120000\n"
    "AYUDANTE;JOR;2;0;80000;160000\n"
    "\n"
    "ITEM: 2.2; UNIDAD: DIA\n"
    "CUADRILLA TIPO 2 (1 OF + 4 AYU)\n"
    "MANO DE OBRA\n"
    "OFICIAL;JOR;1;0;120000;120000\n"
    "AYUDANTE;JOR;4;0;80000;320000\n"
)

INSUMOS_DATA = (
    "  G1  ;MATERIALES;;;;;\n"
    "  CODIGO  ;  DESCRIPCION  ;  UND  ;;  CANT.  ;  VR. UNIT.  ;\n"
    "INS-001;  Tornillo de Acero  ;UND;;;10,50;\n"
    "INS-003; pintura anticorrosiva ;GL;;;5,00;\n"
    "  G2  ;MANO DE OBRA;;;;;\n"
    "  CODIGO  ;  DESCRIPCION  ;  UND  ;;  CANT.  ;  VR. UNIT.  ;\n"
    "INS-002;Mano de Obra Especializada;HR;;;20,00;\n"
)

TEST_CONFIG = {
    "presupuesto_column_map": {
        "CODIGO_APU": ["ITEM"],
        "DESCRIPCION_APU": ["DESCRIPCION"],
        "CANTIDAD_PRESUPUESTO": ["CANT."],
    },
    "category_keywords": {
        "MATERIALES": "MATERIALES",
        "MANO DE OBRA": "MANO DE OBRA",
        "EQUIPO": "EQUIPO",
        "OTROS": "OTROS",
    },
    "param_map": {
        "material": {"TEJA": "TEJA TRAPEZOIDAL"},
        "cuadrilla": {"1": "TIPO 1", "2": "TIPO 2"},
    },
}

class TestCSVProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presupuesto_path = "test_presupuesto.csv"
        with open(cls.presupuesto_path, "w", encoding="latin1") as f:
            f.write(PRESUPUESTO_DATA)
        cls.apus_path = "test_apus.csv"
        with open(cls.apus_path, "w", encoding="latin1") as f:
            f.write(APUS_DATA)
        cls.insumos_path = "test_insumos.csv"
        with open(cls.insumos_path, "w", encoding="latin1") as f:
            f.write(INSUMOS_DATA)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.presupuesto_path)
        os.remove(cls.apus_path)
        os.remove(cls.insumos_path)

    def test_process_all_files_structure_and_calculations(self):
        resultado = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
        )
        self.assertIsInstance(resultado, dict)
        self.assertNotIn("error", resultado)
        presupuesto_procesado = resultado["presupuesto"]
        self.assertEqual(len(presupuesto_procesado), 4)
        item1 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1,1"), None
        )
        self.assertIsNotNone(item1)
        self.assertAlmostEqual(item1["VALOR_CONSTRUCCION_UN"], 50000.0)

    def test_duplicate_codigo_apu_in_presupuesto(self):
        """
        Tests that if the budget file contains duplicate CODIGO_APU, only the
        first one is kept and a warning is logged.
        """
        PRESUPUESTO_DUPLICADO = (
            "ITEM;DESCRIPCION;UND;CANT.;VR. UNIT;VR.TOTAL\n"
            "1.1;Primera Desc;ML;10;100;1000\n"
            "1.2;Otra Desc;M2;20;200;4000\n"
            "1.1;Segunda Desc (Duplicado);ML;30;300;9000\n"
        )
        presupuesto_path = "test_presupuesto_duplicado.csv"
        with open(presupuesto_path, "w", encoding="latin1") as f:
            f.write(PRESUPUESTO_DUPLICADO)

        with self.assertLogs("app.procesador_csv", level="WARNING") as cm:
            resultado = process_all_files(
                presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
            )
            # Check for the specific warning message
            self.assertTrue(
                any("Se encontraron 2 filas duplicadas en presupuesto" in msg
                    for msg in cm.output)
            )

        # The result should not contain an error
        self.assertNotIn("error", resultado)

        # The processed budget should only have 2 unique items
        presupuesto_procesado = resultado["presupuesto"]
        self.assertEqual(len(presupuesto_procesado), 2)

        # Verify that the first item (1,1) was kept
        item1 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1,1"), None
        )
        self.assertIsNotNone(item1)
        self.assertEqual(item1["DESCRIPCION_APU"], "Primera Desc")

        os.remove(presupuesto_path)

    def test_duplicate_insumos_keeps_higher_price(self):
        """
        Tests that if a duplicate insumo is found, the one with the highest price is kept.
        """
        INSUMOS_DUPLICADO = (
            "G1;MATERIALES\n"
            "CODIGO;DESCRIPCION;UND;CANT.;VR. UNIT.\n"
            "INS-001;Tornillo de Acero;UND;;10.50\n"
            "INS-002;Tornillo de Acero;UND;;12.00\n"  # Duplicado con mayor precio
            "INS-003;Pintura;GL;;5.00\n"
        )
        insumos_path = "test_insumos_duplicado.csv"
        with open(insumos_path, "w", encoding="latin1") as f:
            f.write(INSUMOS_DUPLICADO)

        resultado = process_all_files(
            self.presupuesto_path, self.apus_path, insumos_path, config=TEST_CONFIG
        )

        self.assertNotIn("error", resultado)
        # raw_insumos_df is already a list of dicts, which is what we need to check
        insumos_list = resultado["raw_insumos_df"]

        # There should be only one "Tornillo de Acero"
        tornillo_entries = [
            item
            for item in insumos_list
            if item["DESCRIPCION_INSUMO"] == "Tornillo de Acero"
        ]
        self.assertEqual(len(tornillo_entries), 1)

        # The price should be the higher one
        self.assertAlmostEqual(tornillo_entries[0]["VR_UNITARIO_INSUMO"], 12.00)

        os.remove(insumos_path)

    def test_cartesian_explosion_on_final_merge(self):
        """
        Tests that the final merge returns an error if df_apu_costos has duplicate
        CODIGO_APU, preventing a cartesian explosion. This is tested by mocking
        the output of _calculate_apu_costs_and_metadata.
        """
        # Create a malformed df_apu_costos with duplicate CODIGO_APU
        malformed_apu_costos = pd.DataFrame({
            'CODIGO_APU': ['1,1', '1,2', '1,1'],
            'VALOR_CONSTRUCCION_UN': [100, 200, 150]
        })
        # Mock the helper function to return the malformed data
        with patch('app.procesador_csv._calculate_apu_costs_and_metadata',
                   return_value=(malformed_apu_costos, pd.DataFrame(), pd.DataFrame())):
            with self.assertLogs("app.procesador_csv", level="ERROR") as cm:
                resultado = process_all_files(
                    self.presupuesto_path,
                    self.apus_path,
                    self.insumos_path,
                    TEST_CONFIG
                )
                self.assertTrue(
                    any("EXPLOSIÓN CARTESIANA DETECTADA" in msg for msg in cm.output)
                )

        # The result should contain an error message
        self.assertIn("error", resultado)
        self.assertIn("Explosión cartesiana detectada", resultado["error"])

    def test_abnormally_high_cost_triggers_error(self):
        """
        Tests that if the total construction cost is abnormally high, an error is returned.
        """
        # Create data that will result in a very high cost
        PRESUPUESTO_ALTO = (
            "ITEM;DESCRIPCION;UND;CANT.;VR. UNIT;VR.TOTAL\n"
            "1,1;Costo Alto;M2;1000000;1000001;1000001000000\n" # > 1e12
        )
        APUS_ALTO = (
            "ITEM: 1,1\n"
            "Costo Alto\n"
            "MATERIALES\n"
            "Material Caro;UN;1;0;1000001;1000001\n"
        )
        presupuesto_path = "test_presupuesto_alto.csv"
        apus_path = "test_apus_alto.csv"
        with open(presupuesto_path, "w", encoding="latin1") as f:
            f.write(PRESUPUESTO_ALTO)
        with open(apus_path, "w", encoding="latin1") as f:
            f.write(APUS_ALTO)

        with self.assertLogs('app.procesador_csv', level='ERROR') as cm:
            resultado = process_all_files(
                presupuesto_path, apus_path, self.insumos_path, config=TEST_CONFIG
            )
            self.assertTrue(any("COSTO TOTAL ANORMALMENTE ALTO" in msg for msg in cm.output))

        self.assertIn("error", resultado)
        self.assertIn("Costo total anormalmente alto", resultado["error"])

        os.remove(presupuesto_path)
        os.remove(apus_path)
