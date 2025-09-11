import os
import unittest

import pandas as pd

from procesador_datos import process_files


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """
        Esta función se ejecuta antes de cada prueba.
        Crea archivos Excel falsos para usar en los tests.
        """
        # --- Crear DataFrame de Presupuesto Falso ---
        presupuesto_data = {
            "ÍTEM": ["APU-01", "APU-02"],
            "DESCRIPCIÓN": ["Actividad 1", "Actividad 2"],
            "UN.": ["m2", "m3"],
            "CANT.": [100, 50],
            "VR. UNITARIO": [0, 0],  # El valor se calculará
            "VR. PARCIAL": [0, 0],
        }
        # Agregamos 9 filas vacías al principio para simular el encabezado real
        df_presupuesto = pd.DataFrame(presupuesto_data)
        writer_presupuesto = pd.ExcelWriter("test_presupuesto.xlsx", engine="openpyxl")
        df_presupuesto.to_excel(
            writer_presupuesto, index=False, startrow=9, sheet_name="Sheet1"
        )
        writer_presupuesto.close()

        # --- Crear DataFrame de Insumos Falso ---
        insumos_data = {
            "CÓDIGO": ["INS-001", "INS-002", "INS-003"],
            "DESCRIPCIÓN": ["Material A", "Mano de Obra B", "Equipo C"],
            "UNIDAD": ["kg", "hr", "hr"],
            "VR.UNITARIO": [10, 20, 30],
        }
        # Agregamos 8 filas vacías para simular el encabezado
        df_insumos = pd.DataFrame(insumos_data)
        writer_insumos = pd.ExcelWriter("test_insumos.xlsx", engine="openpyxl")
        df_insumos.to_excel(
            writer_insumos, index=False, startrow=8, sheet_name="Sheet1"
        )
        writer_insumos.close()

        # --- Crear DataFrame de APUs Falso ---
        # La estructura de APU es más compleja, la simulamos como la lee el script
        apu_data = [
            ["APU: APU-01"],
            ["INS-001", "Material A", "kg", "", 2.5],  # Código, Desc, Und, '', Cantidad
            ["INS-002", "Mano de Obra B", "hr", "", 1.0],
            ["APU: APU-02"],
            ["INS-003", "Equipo C", "hr", "", 3.0],
            ["INS-001", "Material A", "kg", "", 5.0],
        ]
        df_apu = pd.DataFrame(apu_data)
        df_apu.to_excel("test_apus.xlsx", index=False, header=False)

    def tearDown(self):
        """
        Esta función se ejecuta después de cada prueba.
        Elimina los archivos falsos para mantener limpio el directorio.
        """
        os.remove("test_presupuesto.xlsx")
        os.remove("test_insumos.xlsx")
        os.remove("test_apus.xlsx")

    def test_process_files_integration(self):
        """
        Prueba la integración completa de la función process_files.
        """
        # Ejecutar la función con los archivos de prueba
        df_resultado = process_files(
            "test_presupuesto.xlsx", "test_apus.xlsx", "test_insumos.xlsx"
        )

        # --- Verificaciones (Assertions) ---
        self.assertIsNotNone(df_resultado)
        self.assertFalse(df_resultado.empty)

        # Verificar que las columnas esperadas estén en el resultado
        expected_columns = ["Código APU", "Descripción", "Valor Total", "ZONA"]
        self.assertListEqual(list(df_resultado.columns), expected_columns)

        # Verificar el número de filas
        self.assertEqual(len(df_resultado), 2)

        # Verificar cálculos específicos
        # APU-01: (2.5 * 10) + (1.0 * 20) = 25 + 20 = 45. Total = 45 * 100 = 4500
        # APU-02: (3.0 * 30) + (5.0 * 10) = 90 + 50 = 140. Total = 140 * 50 = 7000
        valor_total_apu01 = df_resultado[df_resultado["Código APU"] == "APU-01"][
            "Valor Total"
        ].iloc[0]
        valor_total_apu02 = df_resultado[df_resultado["Código APU"] == "APU-02"][
            "Valor Total"
        ].iloc[0]

        self.assertAlmostEqual(valor_total_apu01, 4500)
        self.assertAlmostEqual(valor_total_apu02, 7000)


if __name__ == "__main__":
    unittest.main()
