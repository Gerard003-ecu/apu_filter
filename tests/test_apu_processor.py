# En tests/test_apu_processor.py

import logging
import os
import sys
import unittest

# Añadir el directorio raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.apu_processor import APUProcessor


class TestAPUProcessor(unittest.TestCase):
    """
    Pruebas unitarias para la clase APUProcessor.

    Esta suite de pruebas se centra en validar la lógica de negocio implementada
    en APUProcessor, como la conversión de tipos, cálculos de campos derivados,
    inferencia de unidades y la exclusión de insumos no relevantes.
    """

    def setUp(self):
        """
        Configura el entorno de logging para las pruebas.
        """
        logging.basicConfig(
            level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
        )

    def test_numeric_conversion_and_calculations(self):
        """
        Verifica que la conversión de strings a números y los cálculos básicos
        se realicen correctamente.
        """
        raw_records = [
            {
                "apu_code": "APU-01",
                "apu_desc": "Prueba Numerica",
                "apu_unit": "M2",
                "category": "MATERIALES",
                "insumo_line": "Cemento;UND;1,5;;1.000,00;1.500,00",
            }
        ]
        processor = APUProcessor(raw_records)
        df = processor.process_all()

        self.assertEqual(len(df), 1)
        record = df.iloc[0]
        self.assertAlmostEqual(record["CANTIDAD_APU"], 1.5)
        self.assertAlmostEqual(record["PRECIO_UNIT_APU"], 1000.00)
        self.assertAlmostEqual(record["VALOR_TOTAL_APU"], 1500.00)

    def test_mo_rendimiento_calculation(self):
        """
        Prueba que el cálculo del rendimiento para insumos de mano de obra
        se ejecute correctamente cuando el campo está ausente.
        """
        raw_records = [
            {
                "apu_code": "APU-MO-01",
                "apu_desc": "Prueba MO",
                "apu_unit": "JOR",
                "category": "MANO DE OBRA",
                "insumo_line": "OFICIAL;;0.1;;150000;15000",
            }
        ]
        processor = APUProcessor(raw_records)
        df = processor.process_all()

        self.assertEqual(len(df), 1)
        # Rendimiento = 150000 / 15000 = 10
        self.assertAlmostEqual(df.iloc[0]["RENDIMIENTO"], 10.0)

    def test_unit_inference(self):
        """
        Valida la capacidad de inferir la unidad de un APU cuando se
        proporciona como 'UND' (indefinida), basándose en palabras clave.
        """
        raw_records = [
            {
                "apu_code": "APU-INF-01",
                "apu_desc": "Excavacion de material",
                "apu_unit": "UND",
                "category": "EQUIPO",
                "insumo_line": "Retroexcavadora;DIA;1;;500000;500000",
            }
        ]
        processor = APUProcessor(raw_records)
        df = processor.process_all()

        self.assertEqual(df.iloc[0]["UNIDAD_APU"], "M3") # "Excavacion" infiere M3

    def test_exclusion_of_metadata_insumos(self):
        """
        Asegura que los insumos que representan metadatos (como 'EQUIPO Y
        HERRAMIENTA') sean filtrados y no se incluyan en el resultado final.
        """
        raw_records = [
            {
                "apu_code": "APU-META-01",
                "apu_desc": "Prueba Metadatos",
                "apu_unit": "UND",
                "category": "OTROS",
                "insumo_line": "EQUIPO Y HERRAMIENTA MENOR;%;5;;;15000",
            },
            {
                "apu_code": "APU-META-01",
                "apu_desc": "Prueba Metadatos",
                "apu_unit": "UND",
                "category": "OTROS",
                "insumo_line": "Insumo Valido;UND;1;;1000;1000",
            },
        ]
        processor = APUProcessor(raw_records)
        df = processor.process_all()

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["DESCRIPCION_INSUMO"], "Insumo Valido")


if __name__ == "__main__":
    unittest.main(verbosity=2)
