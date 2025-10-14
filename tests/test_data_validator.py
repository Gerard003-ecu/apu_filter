import os
import sys
import unittest

import pandas as pd

# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.data_validator import (
    _validate_extreme_costs,
    _validate_missing_descriptions,
    _validate_zero_quantity_with_cost,
    validate_and_clean_data,
)


class TestDataValidator(unittest.TestCase):

    def test_validate_extreme_costs(self):
        """
        Prueba que la función de costos extremos añada una alerta correctamente.
        """
        # Escenario 1: Un costo normal
        data_normal = [{'ITEM': '1,1', 'VALOR_CONSTRUCCION_UN': 100000}]
        resultado_normal = _validate_extreme_costs(data_normal)
        self.assertNotIn('alerta', resultado_normal[0])

        # Escenario 2: Un costo extremo
        data_extremo = [{'ITEM': '1,2', 'VALOR_CONSTRUCCION_UN': 60_000_000}]
        resultado_extremo = _validate_extreme_costs(data_extremo)
        self.assertIn('alerta', resultado_extremo[0])
        self.assertIn('excede el umbral', resultado_extremo[0]['alerta'])

    def test_validate_zero_quantity_with_cost(self):
        """
        Prueba que la función de cantidad cero recalcule la cantidad o añada una alerta.
        """
        # Escenario 1: Cantidad cero, pero se puede recalcular
        data_recalculable = [{
            'DESCRIPCION_INSUMO': 'Tornillo',
            'CANTIDAD': 0,
            'VALOR_TOTAL': 100,
            'VR_UNITARIO': 10
        }]
        resultado_recalculado = _validate_zero_quantity_with_cost(data_recalculable)
        self.assertAlmostEqual(resultado_recalculado[0]['CANTIDAD'], 10.0)
        self.assertIn('alerta', resultado_recalculado[0])
        self.assertIn('recalculada', resultado_recalculado[0]['alerta'])

        # Escenario 2: Cantidad cero, no se puede recalcular (precio unitario es cero)
        data_no_recalculable = [{
            'DESCRIPCION_INSUMO': 'Pintura',
            'CANTIDAD': 0,
            'VALOR_TOTAL': 100,
            'VR_UNITARIO': 0
        }]
        resultado_no_recalculado = _validate_zero_quantity_with_cost(data_no_recalculable)
        self.assertEqual(resultado_no_recalculado[0]['CANTIDAD'], 0)
        self.assertIn('alerta', resultado_no_recalculado[0])
        self.assertIn('No se puede recalcular', resultado_no_recalculado[0]['alerta'])

    def test_validate_missing_descriptions(self):
        """
        Prueba que la función de descripciones faltantes asigne un placeholder y una alerta.
        """
        # Crear un DataFrame de insumos de referencia para el fuzzy matching
        insumos_df = pd.DataFrame({
            'DESCRIPCION_INSUMO': ['Tornillo de acero 1/2"', 'Pintura anticorrosiva']
        })

        # Escenario 1: Descripción faltante (None)
        data_faltante = [{'DESCRIPCION_INSUMO': None}]
        resultado_faltante = _validate_missing_descriptions(data_faltante, insumos_df)
        self.assertEqual(
            resultado_faltante[0]['DESCRIPCION_INSUMO'], "Insumo sin descripción"
        )
        self.assertIn('alerta', resultado_faltante[0])

        # Escenario 2: Descripción presente
        data_ok = [{'DESCRIPCION_INSUMO': 'Tornillo de acero 1/2"'}]
        resultado_ok = _validate_missing_descriptions(data_ok, insumos_df)
        self.assertNotIn('alerta', resultado_ok[0])

    def test_validate_and_clean_data_integration(self):
        """
        Prueba la función orquestadora para asegurar que llama a las sub-funciones.
        """
        data_store = {
            'presupuesto': [{'ITEM': '1,1', 'VALOR_CONSTRUCCION_UN': 90_000_000}],
            'apus_detail': [{
                'DESCRIPCION_INSUMO': 'Tornillo',
                'CANTIDAD': 0,
                'VALOR_TOTAL': 100,
                'VR_UNITARIO': 10
            }],
            'raw_insumos_df': pd.DataFrame({'DESCRIPCION_INSUMO': ['Tornillo']})
        }

        resultado = validate_and_clean_data(data_store)

        # Verificar que se aplicaron las alertas de cada sub-función
        self.assertIn('alerta', resultado['presupuesto'][0])
        self.assertIn('alerta', resultado['apus_detail'][0])

if __name__ == '__main__':
    unittest.main()
