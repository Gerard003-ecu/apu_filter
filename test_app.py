import os
import unittest

# Importamos la función orquestadora que ahora es el núcleo de nuestra lógica
from procesador_csv import process_all_files


class TestCSVProcessor(unittest.TestCase):
    """
    Clase de prueba actualizada para validar la nueva lógica de `procesador_csv.py`,
    que devuelve una estructura de datos compleja para el dashboard.
    """

    def setUp(self):
        """
        Crea archivos CSV temporales con datos de prueba antes de cada test.
        Estos datos simulan la estructura de los reportes de SAGUT, incluyendo
        columnas de precios en apus.csv para ser más realistas.
        """
        # --- Datos de Prueba para presupuesto.csv ---
        self.presupuesto_data = (
            "ITEM;DESCRIPCION;UND;CANT.; VR. UNIT ; VR.TOTAL \n"
            "1;Actividad de Construcción 1;;;;\n"
            "1,1;Montaje de Estructura;ML;10; 155,00 ; 1550 \n"
            "1,2;Acabados Finales;M2;20; 250,00 ; 5000 \n"
        )
        with open("test_presupuesto.csv", "w", encoding="latin1") as f:
            f.write(self.presupuesto_data)

        # --- Datos de Prueba para apus.csv (ACTUALIZADO) ---
        # Se añaden columnas de precio unitario y total para simular el formato real
        # y asegurar que la nueva lógica de parsing no falle.
        self.apus_data = (
            "REMATE CON PINTURA;;;;;ITEM:   1,1\n"
            "MATERIALES;;;;;\n"
            "Tornillo de Acero;UND; 10,0;;10,50;105,00\n"
            "MANO DE OBRA;;;;;\n"
            "Mano de Obra Especializada;HR; 2,5;;20,00;50,00\n"
            ";;;;\n"
            "REMATE DE ACERO;;;;;ITEM:   1,2\n"
            "MATERIALES;;;;;\n"
            "Pintura Anticorrosiva;GL; 5,0;;5,00;25,00\n"
            "MANO DE OBRA;;;;;\n"
            "Mano de Obra Especializada;HR; 10,0;;20,00;200,00\n"
        )
        with open("test_apus.csv", "w", encoding="latin1") as f:
            f.write(self.apus_data)

        # --- Datos de Prueba para insumos.csv ---
        self.insumos_data = (
            "  G1  ;MATERIALES;;;;;\n"
            "  CODIGO  ;  DESCRIPCION  ;  UND  ;;  CANT.  ;  VR. UNIT.  ;\n"
            "INS-001;  Tornillo de Acero  ;UND;;;10,50;\n"
            "INS-003; pintura anticorrosiva ;GL;;;5,00;\n"
            "  G2  ;MANO DE OBRA;;;;;\n"
            "  CODIGO  ;  DESCRIPCION  ;  UND  ;;  CANT.  ;  VR. UNIT.  ;\n"
            "INS-002;Mano de Obra Especializada;HR;;;20,00;\n"
        )
        with open("test_insumos.csv", "w", encoding="latin1") as f:
            f.write(self.insumos_data)

    def tearDown(self):
        """
        Elimina los archivos CSV temporales después de cada prueba.
        """
        os.remove("test_presupuesto.csv")
        os.remove("test_apus.csv")
        os.remove("test_insumos.csv")

    def test_process_all_files_structure_and_calculations(self):
        """
        Prueba la función orquestadora `process_all_files`, verificando tanto
        la estructura del diccionario resultante como la precisión de los cálculos.
        """
        # Llama a la función principal con los archivos de prueba
        resultado = process_all_files(
            "test_presupuesto.csv", "test_apus.csv", "test_insumos.csv"
        )

        # 1. Verificar que el resultado sea un diccionario y no contenga errores
        self.assertIsInstance(resultado, dict)
        self.assertNotIn("error", resultado)

        # 2. Verificar que las claves principales existan
        self.assertIn("presupuesto", resultado)
        self.assertIn("insumos", resultado)
        self.assertIn("apus_detail", resultado)

        # 3. Validar el contenido de la clave "presupuesto"
        presupuesto_procesado = resultado["presupuesto"]
        self.assertEqual(len(presupuesto_procesado), 2)

        # APU 1,1: (10 tornillos * $10.50) + (2.5 horas * $20.00) = 105 + 50 = $155 (Costo Unitario)
        # Valor Total Presupuesto 1,1: 10 ML * $155 = $1550
        item1 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1,1"),
            None,
        )
        self.assertIsNotNone(
            item1, "No se encontró el ítem 1,1 en el presupuesto procesado"
        )
        # El valor total ahora se calcula a partir de la cantidad y el costo de construcción
        valor_total_calculado1 = (
            item1["CANTIDAD_PRESUPUESTO"] * item1["VALOR_CONSTRUCCION_UN"]
        )
        self.assertAlmostEqual(valor_total_calculado1, 1550.0)

        # APU 1,2: (5 galones * $5.00) + (10 horas * $20.00) = 25 + 200 = $225 (Costo Unitario)
        # Valor Total Presupuesto 1,2: 20 M2 * $225 = $4500
        item2 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1,2"),
            None,
        )
        self.assertIsNotNone(
            item2, "No se encontró el ítem 1,2 en el presupuesto procesado"
        )
        # El valor total se calcula igual para el segundo ítem
        valor_total_calculado2 = (
            item2["CANTIDAD_PRESUPUESTO"] * item2["VALOR_CONSTRUCCION_UN"]
        )
        self.assertAlmostEqual(valor_total_calculado2, 4500.0)

        # 4. Validar la estructura de la clave "insumos"
        insumos_procesados = resultado["insumos"]
        self.assertIn("MATERIALES", insumos_procesados)
        self.assertIn("MANO DE OBRA", insumos_procesados)
        self.assertEqual(len(insumos_procesados["MATERIALES"]), 2)
        self.assertEqual(len(insumos_procesados["MANO DE OBRA"]), 1)

        # 5. Validar la estructura de la clave "apus_detail"
        apus_detalle = resultado["apus_detail"]
        self.assertIn("1,1", apus_detalle)
        self.assertIn("1,2", apus_detalle)
        self.assertEqual(len(apus_detalle["1,1"]), 2)  # Debe tener 2 insumos
        # Verificar que la categoría se asignó correctamente
        self.assertEqual(apus_detalle["1,1"][0]["CATEGORIA"], "MATERIALES")
        self.assertEqual(apus_detalle["1,1"][1]["CATEGORIA"], "MANO DE OBRA")


if __name__ == "__main__":
    unittest.main()
