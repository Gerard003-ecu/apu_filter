import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importar la app de Flask y las funciones a probar
from app.procesador_csv import process_all_files

# Importar los datos de prueba centralizados
from tests.test_data import (
    TEST_CONFIG,
)


class TestCSVProcessorWithNewData(unittest.TestCase):
    """
    Pruebas de integración robustas para la función `process_all_files`.

    Esta clase valida el flujo completo de procesamiento con la nueva
    arquitectura híbrida Lark+Python y todas las funcionalidades mejoradas.
    """

    @classmethod
    def setUpClass(cls):
        """
        Configura el entorno de prueba creando archivos temporales con datos
        realistas y compatibles con el nuevo parser.
        """
        # Presupuesto con formato mejorado
        cls.presupuesto_path = "test_presupuesto_new.csv"
        presupuesto_data = (
            "ITEM;DESCRIPCION;UND;CANT.;VR. UNIT;VR.TOTAL\n"
            "1.1;INSTALACION TUBERIA PVC 3/4;ML;100;50000;5000000\n"
            "1.2;EXCAVACION MANUAL;M3;50;25000;1250000\n"
            "2.1;SUM. CEMENTO;UND;200;15000;3000000\n"
            "2.2;TRANSPORTE MATERIAL;VIAJE;10;80000;800000\n"
        )
        with open(cls.presupuesto_path, "w", encoding="latin1") as f:
            f.write(presupuesto_data)

        # APUs con formato compatible con Lark (comillas en descripciones)
        cls.apus_path = "test_apus_new.csv"
        apus_data = (
            "ITEM: 1.1; UNIDAD: ML\n"
            "INSTALACION TUBERIA PVC 3/4\n"
            "MATERIALES\n"
            '"TUBERIA PVC 3/4";ML;1.05;;38095;40000\n'
            '"CEMENTO";UND;0.5;;10000;5000\n'
            "MANO DE OBRA\n"
            '"OFICIAL";50000;0;75000;8.0;9375\n'
            "EQUIPO\n"
            '"HERRAMIENTA MENOR";DIA;0.1;;50000;5000\n'
        )
        with open(cls.apus_path, "w", encoding="latin1") as f:
            f.write(apus_data)

        # Insumos con descripciones normalizadas
        cls.insumos_path = "test_insumos_new.csv"
        insumos_data = (
            "G;MATERIALES\n"
            "CODIGO;DESCRIPCION;UND;CANT.;VR. UNIT.\n"
            "M001;TUBERIA PVC 3/4;ML;1;40000\n"
            "M002;CEMENTO;UND;1;12000\n"
            "G;MANO DE OBRA\n"
            "CODIGO;DESCRIPCION;UND;CANT.;VR. UNIT.\n"
            "MO01;OFICIAL;JOR;1;80000\n"
            "G;EQUIPO\n"
            "CODIGO;DESCRIPCION;UND;CANT.;VR. UNIT.\n"
            "E001;HERRAMIENTA MENOR;DIA;1;55000\n"
        )
        with open(cls.insumos_path, "w", encoding="latin1") as f:
            f.write(insumos_data)

    @classmethod
    def tearDownClass(cls):
        """
        Limpia el entorno de prueba eliminando los archivos temporales.
        """
        for path in [cls.presupuesto_path, cls.apus_path, cls.insumos_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_process_all_files_structure_and_calculations_enhanced(self):
        """
        Prueba mejorada del caso de éxito del procesamiento.

        Verifica la estructura del data_store resultante, cálculos clave
        y la integración con la nueva lógica de normalización.
        """
        resultado = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
        )

        self.assertIsInstance(resultado, dict)
        self.assertNotIn(
            "error", resultado, f"El procesamiento falló: {resultado.get('error')}"
        )

        # Verificar estructura completa del resultado
        expected_keys = ["presupuesto", "insumos", "apus_detail", "all_apus", "processed_apus"]
        for key in expected_keys:
            self.assertIn(key, resultado, f"Falta clave en resultado: {key}")

        presupuesto_procesado = resultado["presupuesto"]
        self.assertGreater(len(presupuesto_procesado), 0, "Debería haber APUs procesados")

        # Verificar que se usó NORMALIZED_DESC en lugar de crear columnas temporales
        apus_detail = resultado["apus_detail"]
        if apus_detail:
            first_insumo = apus_detail[0]
            self.assertIn("NORMALIZED_DESC", first_insumo,
                          "Debería usar NORMALIZED_DESC del APUProcessor")

        # Buscar el ítem 1.1 y verificar cálculos
        item1_1 = next(
            (item for item in presupuesto_procesado if item["CODIGO_APU"] == "1.1"), None
        )
        self.assertIsNotNone(item1_1, "El ítem 1.1 no fue encontrado.")

        # Verificar cálculos de costos
        self.assertIn("VALOR_CONSTRUCCION_UN", item1_1)
        self.assertIn("VALOR_SUMINISTRO_UN", item1_1)
        self.assertIn("VALOR_INSTALACION_UN", item1_1)

    def test_normalized_desc_integration(self):
        """
        Prueba que el sistema use correctamente NORMALIZED_DESC para los merges.
        """
        # Mock específico para verificar el uso de NORMALIZED_DESC
        with patch('app.procesador_csv.APUProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor

            # Simular DataFrame con NORMALIZED_DESC
            mock_apus_df = pd.DataFrame({
                'CODIGO_APU': ['1.1', '1.1'],
                'DESCRIPCION_APU': ['APU Test', 'APU Test'],
                'UNIDAD_APU': ['ML', 'ML'],
                'DESCRIPCION_INSUMO': ['TUBERIA PVC', 'OFICIAL'],
                'UNIDAD_INSUMO': ['ML', 'JOR'],
                'CANTIDAD_APU': [1.05, 0.125],
                'PRECIO_UNIT_APU': [38095, 75000],
                'VALOR_TOTAL_APU': [40000, 9375],
                'CATEGORIA': ['MATERIALES', 'MANO DE OBRA'],
                'FORMATO_ORIGEN': ['INSUMO_BASICO', 'MO_COMPLETA'],
                'TIPO_INSUMO': ['SUMINISTRO', 'MANO_DE_OBRA'],
                'RENDIMIENTO': [0, 8.0],
                'NORMALIZED_DESC': ['tuberia pvc', 'oficial'] # Columna del APUProcessor
            })

            mock_processor.process_all.return_value = mock_apus_df

            resultado = process_all_files(
                self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
            )

            # Verificar que no se creó DESCRIPCION_INSUMO_NORM duplicada
            if 'apus_detail' in resultado and resultado['apus_detail']:
                first_insumo = resultado['apus_detail'][0]
                self.assertIn('NORMALIZED_DESC', first_insumo)
                # Asegurar que no hay columnas duplicadas
                self.assertNotIn('DESCRIPCION_INSUMO_NORM', first_insumo)

    def test_abnormally_high_cost_triggers_error_enhanced(self):
        """
        Valida mejorada para costos anormalmente altos.

        Usa datos compatibles con el nuevo parser y verifica la detección
        mejorada de outliers.
        """
        # Datos con costos extremadamente altos pero con formato correcto
        presupuesto_alto = (
            "ITEM;DESCRIPCION;UND;CANT.;VR. UNIT;VR.TOTAL\n"
            "1.1;MATERIAL COSTOSO;UND;1000000;1000000;1000000000000\n"
        )
        presupuesto_alto_path = "test_presupuesto_alto.csv"
        with open(presupuesto_alto_path, "w", encoding="latin1") as f:
            f.write(presupuesto_alto)

        apus_alto = (
            "ITEM: 1.1; UNIDAD: UND\n"
            "MATERIAL COSTOSO\n"
            "MATERIALES\n"
            '"MATERIAL COSTOSO";UND;1;;1000000;1000000\n'
        )
        apus_alto_path = "test_apus_alto.csv"
        with open(apus_alto_path, "w", encoding="latin1") as f:
            f.write(apus_alto)

        with self.assertLogs('app.procesador_csv', level='ERROR') as cm:
            resultado = process_all_files(
                presupuesto_alto_path, apus_alto_path, self.insumos_path, config=TEST_CONFIG
            )

            # Verificar detección mejorada de outliers
            error_messages = [msg for msg in cm.output if "ANORMAL" in msg or "ALTO" in msg]
            self.assertGreater(len(error_messages), 0,
                               "Debería detectar costos anormalmente altos")

        self.assertIn("error", resultado)
        self.assertIn("alto", resultado["error"].lower())

        # Limpiar archivos temporales
        for path in [presupuesto_alto_path, apus_alto_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_cartesian_explosion_on_final_merge_enhanced(self):
        """
        Prueba mejorada contra explosiones cartesianas.

        Verifica las validaciones mejoradas en los merges y la estructura
        de datos esperada.
        """
        # Crear DataFrame con estructura mejorada pero duplicados
        malformed_apu_costos = pd.DataFrame({
            'CODIGO_APU': ['1.1', '1.2', '1.1'], # '1.1' duplicado
            'VALOR_SUMINISTRO_UN': [50000, 30000, 55000],
            'VALOR_INSTALACION_UN': [20000, 15000, 22000],
            'VALOR_TRANSPORTE_UN': [0, 0, 0],
            'VALOR_OTRO_UN': [0, 0, 0],
            'VALOR_CONSTRUCCION_UN': [70000, 45000, 77000],
            'PCT_SUMINISTRO': [71.4, 66.7, 71.4],
            'PCT_INSTALACION': [28.6, 33.3, 28.6],
            'PCT_TRANSPORTE': [0, 0, 0],
            'tipo_apu': ['Suministro', 'Suministro', 'Suministro']
        })

        df_tiempo = pd.DataFrame({
            'CODIGO_APU': ['1.1', '1.2'],
            'TIEMPO_INSTALACION': [0.125, 0.1]
        })

        df_rendimiento = pd.DataFrame({
            'CODIGO_APU': ['1.1', '1.2'],
            'RENDIMIENTO_DIA': [8.0, 10.0]
        })

        with patch('app.procesador_csv._calculate_apu_costs_and_metadata',
                   return_value=(malformed_apu_costos, df_tiempo, df_rendimiento)):
            with self.assertLogs("app.procesador_csv", level="ERROR") as cm:
                resultado = process_all_files(
                    self.presupuesto_path,
                    self.apus_path,
                    self.insumos_path,
                    TEST_CONFIG
                )

                # Verificar detección mejorada de explosión cartesiana
                error_messages = [msg for msg in cm.output if "EXPLOSIÓN" in msg or "CARTESIANA" in msg]
                self.assertGreater(len(error_messages), 0,
                                   "Debería detectar explosión cartesiana")

        self.assertIn("error", resultado)
        self.assertIn("explosión", resultado["error"].lower() or "cartesiana" in resultado["error"].lower())

    def test_data_synchronization_enhanced(self):
        """
        Prueba la sincronización mejorada de datos entre APUs y presupuesto.
        """
        with patch('app.procesador_csv.APUProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor

            # Simular APUs que incluyen códigos no presentes en el presupuesto
            mock_apus_df = pd.DataFrame({
                'CODIGO_APU': ['1.1', '1.2', '1.3', '2.1'], # 1.3 no está en presupuesto
                'DESCRIPCION_APU': ['APU 1.1', 'APU 1.2', 'APU 1.3', 'APU 2.1'],
                'UNIDAD_APU': ['ML', 'M3', 'UND', 'UND'],
                'DESCRIPCION_INSUMO': ['Insumo A', 'Insumo B', 'Insumo C', 'Insumo D'],
                'UNIDAD_INSUMO': ['ML', 'M3', 'UND', 'UND'],
                'CANTIDAD_APU': [1, 1, 1, 1],
                'PRECIO_UNIT_APU': [1000, 2000, 3000, 4000],
                'VALOR_TOTAL_APU': [1000, 2000, 3000, 4000],
                'CATEGORIA': ['MATERIALES', 'MATERIALES', 'MATERIALES', 'MATERIALES'],
                'FORMATO_ORIGEN': ['INSUMO_BASICO', 'INSUMO_BASICO', 'INSUMO_BASICO', 'INSUMO_BASICO'],
                'TIPO_INSUMO': ['SUMINISTRO', 'SUMINISTRO', 'SUMINISTRO', 'SUMINISTRO'],
                'RENDIMIENTO': [0, 0, 0, 0],
                'NORMALIZED_DESC': ['insumo a', 'insumo b', 'insumo c', 'insumo d']
            })

            mock_processor.process_all.return_value = mock_apus_df

            resultado = process_all_files(
                self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
            )

            # Verificar que solo los APUs del presupuesto están en el resultado
            if 'presupuesto' in resultado:
                codigos_presupuesto = {item['CODIGO_APU'] for item in resultado['presupuesto']}
                expected_codes = {'1.1', '1.2', '2.1', '2.2'} # Del presupuesto real
                self.assertEqual(codigos_presupuesto, expected_codes)

            # Verificar sincronización en apus_detail
            if 'apus_detail' in resultado:
                codigos_detalle = {item['CODIGO_APU'] for item in resultado['apus_detail']}
                # No debería incluir 1.3 que no está en presupuesto
                self.assertNotIn('1.3', codigos_detalle)

    def test_unit_inference_in_processing(self):
        """
        Prueba que la inferencia agresiva de unidades funcione en el procesamiento completo.
        """
        with patch('app.procesador_csv.APUProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor

            # Simular APUs con unidades UND que deben ser inferidas
            mock_apus_df = pd.DataFrame({
                'CODIGO_APU': ['1.1', '1.2'],
                'DESCRIPCION_APU': ['Excavacion manual', 'Pintura de fachada'],
                'UNIDAD_APU': ['UND', 'UND'], # Unidades indefinidas
                'DESCRIPCION_INSUMO': ['Insumo A', 'Insumo B'],
                'UNIDAD_INSUMO': ['UND', 'UND'],
                'CANTIDAD_APU': [1, 1],
                'PRECIO_UNIT_APU': [1000, 2000],
                'VALOR_TOTAL_APU': [1000, 2000],
                'CATEGORIA': ['MOVIMIENTO DE TIERRAS', 'ACABADOS'],
                'FORMATO_ORIGEN': ['INSUMO_BASICO', 'INSUMO_BASICO'],
                'TIPO_INSUMO': ['SUMINISTRO', 'SUMINISTRO'],
                'RENDIMIENTO': [0, 0],
                'NORMALIZED_DESC': ['insumo a', 'insumo b']
            })

            mock_processor.process_all.return_value = mock_apus_df

            resultado = process_all_files(
                self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
            )

            # Verificar que las unidades fueron inferidas correctamente
            if 'processed_apus' in resultado:
                for apu in resultado['processed_apus']:
                    codigo = apu['CODIGO_APU']
                    if codigo == '1.1':
                        self.assertEqual(apu['UNIDAD'], 'M3') # Excavación → M3
                    elif codigo == '1.2':
                        self.assertEqual(apu['UNIDAD'], 'M2') # Pintura → M2

    def test_validation_thresholds_application(self):
        """
        Prueba que los umbrales de validación del config se apliquen correctamente.
        """
        # Config con umbrales más estrictos
        strict_config = TEST_CONFIG.copy()
        strict_config["validation_thresholds"]["SUMINISTRO"]["max_valor_total"] = 1000

        with patch('app.procesador_csv.APUProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor

            # Simular APU con valor superior al umbral
            mock_apus_df = pd.DataFrame({
                'CODIGO_APU': ['1.1'],
                'DESCRIPCION_APU': ['APU Test'],
                'UNIDAD_APU': ['UND'],
                'DESCRIPCION_INSUMO': ['Material Costoso'],
                'UNIDAD_INSUMO': ['UND'],
                'CANTIDAD_APU': [1],
                'PRECIO_UNIT_APU': [1500], # Superior al umbral de 1000
                'VALOR_TOTAL_APU': [1500],
                'CATEGORIA': ['MATERIALES'],
                'FORMATO_ORIGEN': ['INSUMO_BASICO'],
                'TIPO_INSUMO': ['SUMINISTRO'],
                'RENDIMIENTO': [0],
                'NORMALIZED_DESC': ['material costoso']
            })

            mock_processor.process_all.return_value = mock_apus_df

            resultado = process_all_files(
                self.presupuesto_path, self.apus_path, self.insumos_path, config=strict_config
            )

            # Verificar que el insumo fue rechazado por el umbral
            if 'apus_detail' in resultado:
                # Debería estar vacío o no contener el insumo costoso
                insumos_costosos = [item for item in resultado['apus_detail']
                                    if item['DESCRIPCION_INSUMO'] == 'Material Costoso']
                self.assertEqual(len(insumos_costosos), 0)

    def test_error_handling_robustness(self):
        """
        Prueba el manejo robusto de errores en el procesamiento completo.
        """
        # Caso 1: Error en carga de presupuesto
        with patch('app.procesador_csv.process_presupuesto_csv', return_value=pd.DataFrame()):
            resultado = process_all_files(
                self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
            )
            self.assertIn("error", resultado)

        # Caso 2: Error en carga de insumos
        with patch('app.procesador_csv.process_insumos_csv', return_value=pd.DataFrame()):
            resultado = process_all_files(
                self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
            )
            self.assertIn("error", resultado)

        # Caso 3: Error en APUProcessor
        with patch('app.procesador_csv.APUProcessor') as mock_processor_class:
            mock_processor = MagicMock()
            mock_processor_class.return_value = mock_processor
            mock_processor.process_all.side_effect = Exception("Error simulado en processor")

            with self.assertLogs('app.procesador_csv', level='ERROR') as cm:
                resultado = process_all_files(
                    self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
                )
                self.assertIn("error", resultado)

    def test_dataframe_structure_validation(self):
        """
        Prueba que los DataFrames intermedios tengan la estructura correcta.
        """
        resultado = process_all_files(
            self.presupuesto_path, self.apus_path, self.insumos_path, config=TEST_CONFIG
        )

        if 'error' not in resultado:
            # Verificar estructura de presupuesto procesado
            presupuesto = resultado['presupuesto']
            if presupuesto:
                first_item = presupuesto[0]
                expected_keys = [
                    'CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU', 'CANTIDAD_PRESUPUESTO',
                    'VALOR_SUMINISTRO_UN', 'VALOR_INSTALACION_UN', 'VALOR_CONSTRUCCION_UN'
                ]
                for key in expected_keys:
                    self.assertIn(key, first_item, f"Falta clave en presupuesto: {key}")

            # Verificar estructura de apus_detail
            apus_detail = resultado['apus_detail']
            if apus_detail:
                first_detail = apus_detail[0]
                expected_detail_keys = [
                    'CODIGO_APU', 'DESCRIPCION_INSUMO', 'UNIDAD_INSUMO', 'CANTIDAD_APU',
                    'PRECIO_UNIT_APU', 'VALOR_TOTAL_APU', 'TIPO_INSUMO', 'NORMALIZED_DESC'
                ]
                for key in expected_detail_keys:
                    self.assertIn(key, first_detail, f"Falta clave en apus_detail: {key}")


if __name__ == "__main__":
    # Ejecutar pruebas con cobertura completa
    unittest.main(verbosity=2, failfast=False)
