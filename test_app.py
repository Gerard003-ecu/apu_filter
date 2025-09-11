import os
import unittest

# Importamos la función específica del nuevo módulo que queremos probar
from procesador_csv import process_csv_files


class TestCSVProcessor(unittest.TestCase):
    """
    Clase de prueba para el módulo procesador_csv.py.
    Valida la lógica de procesamiento de los archivos CSV generados por SAGUT.
    """

    def setUp(self):
        """
        Se ejecuta antes de cada prueba. Crea archivos CSV temporales
        con datos simulados para asegurar que la prueba es aislada y repetible.
        """
        # --- 1. Datos de prueba como strings multilínea ---

        # Simula el archivo presupuesto.csv
        self.presupuesto_csv_data = (
            "ITEM;DESCRIPCION;UND;CANT.; VR. UNIT ; VR.TOTAL \n"
            "1,1;Instalacion de Teja;ML;10;0;0\n"
            "1,2;Remate especial;ML;5;0;0\n"
        )

        # Simula el archivo insumos.csv (con encabezados variables)
        self.insumos_csv_data = (
            ";;;;;;\n"
            "  G1  ;MATERIALES;;;;\n"
            "  CODIGO  ;  DESCRIPCION  ;  UND  ;;  CANT.  ;  VR. UNIT.  ;VR.TOTAL\n"
            "101;Teja Metalica;UND;;1;15.000,00;15.000,00\n"
            "102;Tornillo Autoperforante;UND;;1;500,00;500,00\n"
            "201;Mano de Obra Cuadrilla;HR;;1;20.000,00;20.000,00\n"
        )

        # Simula el archivo apus.csv (con su formato de reporte)
        self.apus_csv_data = (
            "REMATE CON PINTURA;;;;;  UNIDAD:   ML  \n"
            ";;;;;  ITEM:   1,1  \n"
            "DESCRIPCION;UND;CANT.;DESP.%;\n"
            "MATERIALES;;;;\n"
            "Teja Metalica;UND;2,0;;\n"
            "Tornillo Autoperforante;UND;10,0;;\n"
            "MANO DE OBRA;;;;\n"
            "Mano de Obra Cuadrilla;HR;0,5;;\n"
            ";;;COSTO DIRECTO;;\n"
            ";;;;;  ITEM:   1,2  \n"
            "Teja Metalica;UND;1,5;;\n"
        )

        # --- 2. Crear archivos temporales ---
        self.presupuesto_path = "test_presupuesto.csv"
        self.insumos_path = "test_insumos.csv"
        self.apus_path = "test_apus.csv"

        with open(self.presupuesto_path, "w", encoding="latin1") as f:
            f.write(self.presupuesto_csv_data)
        with open(self.insumos_path, "w", encoding="latin1") as f:
            f.write(self.insumos_csv_data)
        with open(self.apus_path, "w", encoding="latin1") as f:
            f.write(self.apus_csv_data)

    def tearDown(self):
        """
        Se ejecuta después de cada prueba. Elimina los archivos CSV temporales
        para no dejar basura en el directorio de trabajo.
        """
        os.remove(self.presupuesto_path)
        os.remove(self.insumos_path)
        os.remove(self.apus_path)

    def test_process_csv_files_integration(self):
        """
        Prueba la función principal `process_csv_files` con los datos simulados.
        """
        # Ejecutar la función bajo prueba
        df_resultado = process_csv_files(
            self.presupuesto_path, self.apus_path, self.insumos_path
        )

        # --- Verificaciones (Assertions) ---
        self.assertIsNotNone(
            df_resultado, "El DataFrame resultante no debería ser nulo."
        )
        self.assertFalse(
            df_resultado.empty, "El DataFrame resultante no debería estar vacío."
        )

        # Verificar que las columnas esperadas estén presentes
        expected_columns = ["Código APU", "Descripción", "Valor Total", "ZONA"]
        self.assertListEqual(
            list(df_resultado.columns),
            expected_columns,
            "Las columnas del DataFrame no son las esperadas.",
        )

        # Verificar el número de filas del resultado final
        self.assertEqual(
            len(df_resultado), 2, "El número de filas en el resultado es incorrecto."
        )

        # Verificar cálculos específicos para un APU
        # APU '1,1':
        #  - Teja: 2.0 * 15,000 = 30,000
        #  - Tornillo: 10.0 * 500 = 5,000
        #  - MO: 0.5 * 20,000 = 10,000
        #  - Vr. Unitario APU = 30,000 + 5,000 + 10,000 = 45,000
        #  - Cantidad Presupuesto = 10
        #  - Valor Total = 45,000 * 10 = 450,000
        valor_total_apu_1_1 = df_resultado[df_resultado["Código APU"] == "1,1"][
            "Valor Total"
        ].iloc[0]
        self.assertAlmostEqual(
            valor_total_apu_1_1,
            450000,
            "El cálculo del Valor Total para el APU '1,1' es incorrecto.",
        )


if __name__ == "__main__":
    unittest.main()
