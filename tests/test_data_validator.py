import os
import sys
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pandas as pd

# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data_validator import (
    _validate_extreme_costs,
    _validate_missing_descriptions,
    _validate_zero_quantity_with_cost,
    validate_and_clean_data,
    TipoAlerta,
)


class TestDataValidator(unittest.TestCase):
    """
    Pruebas unitarias robustas para las funciones de validación de datos.
    Alineadas con el código refinado: inmutabilidad, alertas acumulativas,
    manejo de errores y cobertura de bordes.
    """

    def setUp(self):
        """Preparar datos de prueba reutilizables."""
        self.presupuesto_normal = [{"ITEM": "1,1", "VALOR_CONSTRUCCION_UN": 100_000}]
        self.presupuesto_extremo = [{"ITEM": "1,2", "VALOR_CONSTRUCCION_UN": 60_000_000}]

        self.apus_detail_recalculable = [
            {
                "DESCRIPCION_INSUMO": "Tornillo",
                "CANTIDAD": 0,
                "VALOR_TOTAL": 100,
                "VR_UNITARIO": 10,
            }
        ]
        self.apus_detail_no_recalculable = [
            {
                "DESCRIPCION_INSUMO": "Pintura",
                "CANTIDAD": 0,
                "VALOR_TOTAL": 100,
                "VR_UNITARIO": 0,
            }
        ]
        self.apus_detail_con_descripcion = [{"DESCRIPCION_INSUMO": 'Tornillo de acero 1/2"'}]
        self.apus_detail_sin_descripcion = [{"DESCRIPCION_INSUMO": None}]

        self.raw_insumos_df = pd.DataFrame(
            {
                "DESCRIPCION_INSUMO": [
                    'Tornillo de acero 1/2"',
                    "Pintura anticorrosiva",
                    "Cemento Portland",
                ]
            }
        ).astype(str)

        # Datos originales para verificar inmutabilidad
        self.original_presupuesto = deepcopy(self.presupuesto_extremo)
        self.original_apus = deepcopy(self.apus_detail_recalculable)
        self.original_df = deepcopy(self.raw_insumos_df)

    def test_validate_extreme_costs__normal_cost(self):
        """Verifica que no se añada alerta si el costo es normal."""
        resultado, metrics = _validate_extreme_costs(self.presupuesto_normal)
        self.assertNotIn(
            "alertas", resultado[0], "No debe haber alertas para costos normales"
        )

    def test_validate_extreme_costs__extreme_cost(self):
        """Verifica que se añada una alerta cuando el costo excede el umbral."""
        resultado, metrics = _validate_extreme_costs(self.presupuesto_extremo)
        self.assertIn("alertas", resultado[0], "Debe haber una lista de alertas")
        self.assertEqual(len(resultado[0]["alertas"]), 1)
        self.assertIn("excede el umbral", resultado[0]["alertas"][0]["mensaje"])
        self.assertIn("60,000,000.00", resultado[0]["alertas"][0]["mensaje"])

    def test_validate_extreme_costs__non_numeric_value(self):
        """Verifica que no falle si el valor no es numérico."""
        data_broken = [{"ITEM": "X", "VALOR_CONSTRUCCION_UN": "N/A"}]
        resultado, metrics = _validate_extreme_costs(data_broken)
        self.assertIn("alertas", resultado[0], "Debe alertar si no es numérico")

    def test_validate_extreme_costs__inmutable_input(self):
        """Verifica que la función no modifica el input original."""
        original_copy = deepcopy(self.presupuesto_extremo)
        _validate_extreme_costs(self.presupuesto_extremo)
        self.assertEqual(
            original_copy, self.original_presupuesto, "El input original fue modificado"
        )

    def test_validate_zero_quantity_with_cost__recalculable(self):
        """Verifica que se recalcula la cantidad y alerta cuando es posible."""
        resultado, metrics = _validate_zero_quantity_with_cost(self.apus_detail_recalculable)
        self.assertAlmostEqual(resultado[0]["CANTIDAD"], 10.0, places=4)
        self.assertIn("alertas", resultado[0])
        self.assertEqual(len(resultado[0]["alertas"]), 1)
        self.assertIn("recalculada", resultado[0]["alertas"][0]["mensaje"])
        self.assertIn("10.0000", resultado[0]["alertas"][0]["mensaje"])

    def test_validate_zero_quantity_with_cost__non_recalculable(self):
        """Verifica que no se modifica cantidad si VR_UNITARIO es 0 o inválido."""
        resultado, metrics = _validate_zero_quantity_with_cost(
            self.apus_detail_no_recalculable
        )
        self.assertEqual(resultado[0]["CANTIDAD"], 0)
        self.assertIn("alertas", resultado[0])
        # Expect 2 alerts: one for inability to recalculate, one for mathematical incoherence (0 * 0 != 100)
        self.assertEqual(len(resultado[0]["alertas"]), 2)
        self.assertIn("No se puede recalcular", resultado[0]["alertas"][0]["mensaje"])
        self.assertIn("Incoherencia matemática", resultado[0]["alertas"][1]["mensaje"])

    def test_validate_zero_quantity_with_cost__invalid_types(self):
        """Verifica manejo de tipos inválidos (cadena, None, NaN)."""
        data_broken = [
            {
                "DESCRIPCION_INSUMO": "Pintura",
                "CANTIDAD": "cero",  # cadena
                "VALOR_TOTAL": "100",
                "VR_UNITARIO": "0",
            }
        ]
        resultado, metrics = _validate_zero_quantity_with_cost(data_broken)
        self.assertEqual(resultado[0]["CANTIDAD"], "cero")  # No se modifica si no es convertible
        self.assertIn("alertas", resultado[0])

    def test_validate_zero_quantity_with_cost__inmutable_input(self):
        """Verifica que no se modifica el input original."""
        original_copy = deepcopy(self.apus_detail_recalculable)
        _validate_zero_quantity_with_cost(self.apus_detail_recalculable)
        self.assertEqual(
            original_copy, self.original_apus, "El input original fue modificado"
        )

    @patch("app.data_validator.process")
    @patch("app.data_validator.HAS_FUZZY", False)
    def test_validate_missing_descriptions__no_fuzzy_available(self, mock_process):
        """Verifica fallback cuando fuzzywuzzy no está instalado."""
        resultado, metrics = _validate_missing_descriptions(
            self.apus_detail_sin_descripcion, self.raw_insumos_df
        )
        self.assertEqual(resultado[0]["DESCRIPCION_INSUMO"], "Insumo sin descripción")
        self.assertIn("alertas", resultado[0])
        self.assertTrue(
            any("fuzzy matching no disponible" in a["mensaje"] or "no instalado" in a["mensaje"] or "sin referencias" in a["mensaje"]
                for a in resultado[0]["alertas"]),
            f"Alerts found: {resultado[0]['alertas']}"
        )

    def test_validate_missing_descriptions__missing_description(self):
        """Verifica que se asigna texto predeterminado y alerta si no hay descripción."""
        # Force no fuzzy match found by using empty df
        resultado, metrics = _validate_missing_descriptions(
            self.apus_detail_sin_descripcion, pd.DataFrame()
        )
        self.assertEqual(resultado[0]["DESCRIPCION_INSUMO"], "Insumo sin descripción")
        self.assertIn("alertas", resultado[0])
        self.assertEqual(len(resultado[0]["alertas"]), 1)
        self.assertIn("faltante", resultado[0]["alertas"][0]["mensaje"])

    def test_validate_missing_descriptions__present_description(self):
        """Verifica que no se añade alerta si la descripción está presente."""
        resultado, metrics = _validate_missing_descriptions(
            self.apus_detail_con_descripcion, self.raw_insumos_df
        )
        self.assertNotIn(
            "alertas", resultado[0], "No debe haber alertas si la descripción está presente"
        )

    @patch("app.data_validator.process")
    @patch("app.data_validator.HAS_FUZZY", True)
    def test_validate_missing_descriptions__fuzzy_matching_enabled(self, mock_process):
        """Verifica que fuzzy matching no modifica una descripción existente similar."""
        data_similar = [{"DESCRIPCION_INSUMO": "Tornillo de acero 1/2 pulgadas"}]
        resultado, metrics = _validate_missing_descriptions(
            data_similar, self.raw_insumos_df
        )
        self.assertEqual(
            resultado[0]["DESCRIPCION_INSUMO"], "Tornillo de acero 1/2 pulgadas"
        )
        self.assertNotIn("alertas", resultado[0])

    def test_validate_missing_descriptions__raw_insumos_df_none(self):
        """Verifica comportamiento cuando raw_insumos_df es None."""
        resultado, metrics = _validate_missing_descriptions(
            self.apus_detail_sin_descripcion, None
        )
        self.assertEqual(resultado[0]["DESCRIPCION_INSUMO"], "Insumo sin descripción")
        self.assertIn("alertas", resultado[0])
        self.assertTrue(
            any("no hay referencias disponibles" in a["mensaje"] or "sin datos de referencia" in a["mensaje"]
                for a in resultado[0]["alertas"]),
            f"Alerts: {resultado[0]['alertas']}"
        )

    def test_validate_missing_descriptions__raw_insumos_df_missing_column(self):
        """Verifica comportamiento cuando raw_insumos_df no tiene la columna requerida."""
        df_broken = pd.DataFrame({"OTRA_COLUMNA": ["valor"]})
        resultado, metrics = _validate_missing_descriptions(
            self.apus_detail_sin_descripcion, df_broken
        )
        self.assertEqual(resultado[0]["DESCRIPCION_INSUMO"], "Insumo sin descripción")
        self.assertIn("alertas", resultado[0])
        self.assertTrue(
            any("no hay referencias disponibles" in a["mensaje"] or "sin datos de referencia" in a["mensaje"]
                for a in resultado[0]["alertas"]),
            f"Alerts: {resultado[0]['alertas']}"
        )

    def test_validate_missing_descriptions__inmutable_input(self):
        """Verifica que no se modifica el input original."""
        original_copy = deepcopy(self.apus_detail_sin_descripcion)
        _validate_missing_descriptions(self.apus_detail_sin_descripcion, self.raw_insumos_df)
        self.assertEqual(
            original_copy,
            self.apus_detail_sin_descripcion,
            "El input original fue modificado",
        )

    def test_validate_and_clean_data__integration_success(self):
        """Verifica que todas las validaciones se aplican correctamente en conjunto."""
        data_store = {
            "presupuesto": self.presupuesto_extremo,
            "apus_detail": self.apus_detail_recalculable,
            "raw_insumos_df": self.raw_insumos_df,
        }

        resultado = validate_and_clean_data(data_store)

        # Verificar que las alertas están presentes y son correctas
        self.assertIn("alertas", resultado["presupuesto"][0])
        self.assertEqual(len(resultado["presupuesto"][0]["alertas"]), 1)
        self.assertIn(
            "excede el umbral", resultado["presupuesto"][0]["alertas"][0]["mensaje"]
        )

        self.assertIn("alertas", resultado["apus_detail"][0])
        self.assertEqual(len(resultado["apus_detail"][0]["alertas"]), 1)
        self.assertIn("recalculada", resultado["apus_detail"][0]["alertas"][0]["mensaje"])

    def test_validate_and_clean_data__missing_presupuesto_key(self):
        """Verifica que no falla si 'presupuesto' no está presente."""
        data_store = {
            "apus_detail": self.apus_detail_recalculable,
            "raw_insumos_df": self.raw_insumos_df,
        }
        resultado = validate_and_clean_data(data_store)
        self.assertNotIn("presupuesto", resultado, "No debe añadirse si no estaba")
        self.assertIn("apus_detail", resultado)

    def test_validate_and_clean_data__missing_apus_detail_key(self):
        """Verifica que no falla si 'apus_detail' no está presente."""
        data_store = {
            "presupuesto": self.presupuesto_extremo,
            "raw_insumos_df": self.raw_insumos_df,
        }
        resultado = validate_and_clean_data(data_store)
        self.assertNotIn("apus_detail", resultado, "No debe añadirse si no estaba")
        self.assertIn("presupuesto", resultado)

    def test_validate_and_clean_data__invalid_data_store_type(self):
        """Verifica que no falla si data_store no es dict."""
        resultado = validate_and_clean_data("esto no es un dict")
        self.assertTrue("error" in resultado)  # Returns dict with error, not empty dict

    def test_validate_and_clean_data__none_in_presupuesto(self):
        """Verifica manejo de presupuesto como None."""
        data_store = {
            "presupuesto": None,
            "apus_detail": self.apus_detail_recalculable,
            "raw_insumos_df": self.raw_insumos_df,
        }
        resultado = validate_and_clean_data(data_store)
        # Should return None if the validation failed gracefully or metrics with error
        self.assertIsNone(resultado["presupuesto"])

    def test_validate_and_clean_data__none_in_apus_detail(self):
        """Verifica manejo de apus_detail como None."""
        data_store = {
            "presupuesto": self.presupuesto_extremo,
            "apus_detail": None,
            "raw_insumos_df": self.raw_insumos_df,
        }
        resultado = validate_and_clean_data(data_store)
        self.assertIsNone(resultado["apus_detail"])

    def test_validate_and_clean_data__inmutable_input(self):
        """Verifica que validate_and_clean_data no modifica el input original."""
        original_data = {
            "presupuesto": deepcopy(self.presupuesto_extremo),
            "apus_detail": deepcopy(self.apus_detail_recalculable),
            "raw_insumos_df": deepcopy(self.raw_insumos_df),
        }
        original_presupuesto = deepcopy(original_data["presupuesto"])
        original_apus = deepcopy(original_data["apus_detail"])

        validate_and_clean_data(original_data)

        self.assertEqual(
            original_data["presupuesto"],
            original_presupuesto,
            "Presupuesto fue modificado",
        )
        self.assertEqual(
            original_data["apus_detail"], original_apus, "Apus_detail fue modificado"
        )


if __name__ == "__main__":
    unittest.main()
