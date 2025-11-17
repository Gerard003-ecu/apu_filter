import unittest

import pandas as pd

from app.apu_processor import APUProcessor
from app.utils import calculate_unit_costs


class TestFixtures:
    @staticmethod
    def get_default_config():
        return {
            "apu_processor_rules": {
                "special_cases": {"TRANSPORTE": "TRANSPORTE"},
                "mo_keywords": ["OFICIAL", "AYUDANTE", "PEON", "CUADRILLA"],
                "equipo_keywords": ["EQUIPO", "HERRAMIENTA", "MAQUINA"],
            },
            "validation_thresholds": {
                "MANO_DE_OBRA": {
                    "min_jornal": 50000,
                    "max_jornal": 10000000,
                    "min_rendimiento": 0.001,
                    "max_rendimiento": 1000,
                }
            },
        }

    @staticmethod
    def get_sample_records():
        return [
            {
                "codigo_apu": "1.1",
                "descripcion_apu": "Muro",
                "unidad_apu": "M3",
                "lines": ["OFICIAL;JOR;0.125;;180000;22500", "CEMENTO;KG;350;1200;420000"],
            },
            {
                "codigo_apu": "2.1",
                "descripcion_apu": "Excavacion",
                "unidad_apu": "M3",
                "lines": ["PEON;JOR;0.5;;100000;50000"],
            },
        ]


class TestAPUProcessor(unittest.TestCase):
    def setUp(self):
        self.config = TestFixtures.get_default_config()
        # Perfil por defecto para pruebas que no dependen de un perfil específico
        self.default_profile = {"number_format": {"decimal_separator": "."}}

    def test_process_all(self):
        records = TestFixtures.get_sample_records()
        # CAMBIO: Pasar el perfil por defecto
        processor = APUProcessor(records, self.config, self.default_profile)
        df = processor.process_all()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 3)

    def test_integration_with_calculate_unit_costs(self):
        records = TestFixtures.get_sample_records()
        # CAMBIO: Pasar el perfil por defecto
        processor = APUProcessor(records, self.config, self.default_profile)
        df = processor.process_all()

        # Renombrar columnas para que coincidan con lo que espera calculate_unit_costs
        df = df.rename(
            columns={
                "codigo_apu": "CODIGO_APU",
                "descripcion_apu": "DESCRIPCION_APU",
                "unidad_apu": "UNIDAD_APU",
                "tipo_insumo": "TIPO_INSUMO",
                "valor_total": "VALOR_TOTAL_APU",
            }
        )

        costs_df = calculate_unit_costs(df)
        self.assertFalse(costs_df.empty)
        self.assertIn("COSTO_UNITARIO_TOTAL", costs_df.columns)
        self.assertEqual(len(costs_df), 2)

    def test_process_with_comma_decimal_separator(self):
        """
        Prueba que el procesador maneja correctamente los números con comas
        decimales cuando se especifica en la configuración.
        """
        # 1. Configuración de perfil con separador de coma
        config = TestFixtures.get_default_config()
        # El perfil ahora lleva la configuración específica del archivo
        comma_profile = {"number_format": {"decimal_separator": ","}}

        # 2. Datos de muestra con comas como decimales y puntos como miles
        comma_records = [
            {
                "codigo_apu": "3.1",
                "descripcion_apu": "Piso Industrial",
                "unidad_apu": "M2",
                "lines": [
                    # Formato: DESCRIPCION;UND;CANT;PRECIO;TOTAL
                    "CONCRETO ESPECIAL;M3;0,15;850.123,50;127.518,53",
                    # Formato MO: DESCRIPCION;UND;RENDIMIENTO;;JORNAL;TOTAL
                    "CUADRILLA PISOS;JOR;0,08;;250.000,00;20.000,00",
                ],
            }
        ]

        # 3. Procesar los datos, pasando el perfil con la coma
        processor = APUProcessor(comma_records, config, comma_profile)
        df = processor.process_all()

        # 4. Verificaciones
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)

        # Verificar el parseo correcto de 'CONCRETO'
        concreto_row = df[df["descripcion_insumo"] == "CONCRETO ESPECIAL"].iloc[0]
        self.assertAlmostEqual(concreto_row["cantidad"], 0.15)
        self.assertAlmostEqual(concreto_row["precio_unitario"], 850123.50)
        self.assertAlmostEqual(concreto_row["valor_total"], 127518.53)

        # Verificar el parseo correcto de 'CUADRILLA' (Mano de Obra)
        cuadrilla_row = df[df["descripcion_insumo"] == "CUADRILLA PISOS"].iloc[0]
        self.assertAlmostEqual(cuadrilla_row["rendimiento"], 0.08)
        self.assertAlmostEqual(cuadrilla_row["precio_unitario"], 250000.00)  # Jornal
        self.assertAlmostEqual(cuadrilla_row["cantidad"], 1.0 / 0.08)  # Cantidad calculada


if __name__ == "__main__":
    unittest.main()
