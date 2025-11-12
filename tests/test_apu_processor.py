import unittest

import pandas as pd

from app.apu_processor import APUProcessor
from app.utils import calculate_unit_costs


class TestFixtures:
    @staticmethod
    def get_default_config():
        return {
            "keyword_maps": {
                "equipo": ["EQUIPO", "HERRAMIENTA", "MAQUINA"],
                "mano_de_obra": ["OFICIAL", "AYUDANTE", "PEON"],
                "transporte": ["TRANSPORTE", "VOLQUETA"],
                "suministro": ["CEMENTO", "ARENA"]
            },
            "validation_thresholds": {
                "MANO_DE_OBRA": {
                    "min_jornal": 50000,
                    "max_jornal": 10000000,
                    "min_rendimiento": 0.001,
                    "max_rendimiento": 1000
                }
            }
        }

    @staticmethod
    def get_sample_records():
        return [
            {
                'codigo_apu': '1.1', 'descripcion_apu': 'Muro', 'unidad_apu': 'M3',
                'lines': [
                    'OFICIAL;JOR;0.125;;180000;22500',
                    'CEMENTO;KG;350;1200;420000'
                ]
            },
            {
                'codigo_apu': '2.1', 'descripcion_apu': 'Excavacion', 'unidad_apu': 'M3',
                'lines': [
                    'PEON;JOR;0.5;;100000;50000'
                ]
            },
        ]

class TestAPUProcessor(unittest.TestCase):
    def setUp(self):
        self.config = TestFixtures.get_default_config()

    def test_process_all(self):
        records = TestFixtures.get_sample_records()
        processor = APUProcessor(records, self.config)
        df = processor.process_all()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 3)

    def test_integration_with_calculate_unit_costs(self):
        records = TestFixtures.get_sample_records()
        processor = APUProcessor(records, self.config)
        df = processor.process_all()

        # Renombrar columnas para que coincidan con lo que espera calculate_unit_costs
        df = df.rename(columns={
            'codigo_apu': 'CODIGO_APU',
            'descripcion_apu': 'DESCRIPCION_APU',
            'unidad_apu': 'UNIDAD_APU',
            'tipo_insumo': 'TIPO_INSUMO',
            'valor_total': 'VALOR_TOTAL_APU'
        })

        costs_df = calculate_unit_costs(df)
        self.assertFalse(costs_df.empty)
        self.assertIn('COSTO_UNITARIO_TOTAL', costs_df.columns)
        self.assertEqual(len(costs_df), 2)

if __name__ == '__main__':
    unittest.main()
