import os
import sys
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd

# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data_validator import (
    AnomalyValidator,
    PyramidalValidator,
    TipoAlerta,
    _validar_coherencia_matematica,
    _validate_descriptions,
    _validate_extreme_costs,
    _validate_zero_quantity_with_cost,
    validate_and_clean_data,
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
        # Update assertion to match actual error message
        self.assertIn("Excesivo", resultado[0]["alertas"][0]["mensaje"])
        self.assertIn("60000000.0", resultado[0]["alertas"][0]["mensaje"])

    def test_validate_extreme_costs__non_numeric_value(self):
        """Verifica que no falle si el valor no es numérico. No debe generar alerta, solo ignorar."""
        data_broken = [{"ITEM": "X", "VALOR_CONSTRUCCION_UN": "N/A"}]
        resultado, metrics = _validate_extreme_costs(data_broken)
        # It just gracefully handles it, no alerts expected for this specific function logic unless changed
        self.assertNotIn("alertas", resultado[0])

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
        # Expect 2 alerts: one for mathematical incoherence (initially 0*10 != 100) and one for recalculation
        self.assertEqual(len(resultado[0]["alertas"]), 2)
        self.assertIn("Recalculado", resultado[0]["alertas"][1]["mensaje"])

    def test_validate_zero_quantity_with_cost__non_recalculable(self):
        """Verifica que no se modifica cantidad si VR_UNITARIO es 0 o inválido."""
        resultado, metrics = _validate_zero_quantity_with_cost(
            self.apus_detail_no_recalculable
        )
        self.assertEqual(resultado[0]["CANTIDAD"], 0)
        self.assertIn("alertas", resultado[0])
        # Expect alerts for mathematical incoherence
        self.assertTrue(len(resultado[0]["alertas"]) >= 1)
        self.assertIn("Esperado", resultado[0]["alertas"][0]["mensaje"])

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
        self.assertEqual(
            resultado[0]["CANTIDAD"], "cero"
        )  # No se modifica si no es convertible
        # No alerts expected as it skips invalid number checks
        self.assertNotIn("alertas", resultado[0])

    def test_validate_zero_quantity_with_cost__inmutable_input(self):
        """Verifica que no se modifica el input original."""
        original_copy = deepcopy(self.apus_detail_recalculable)
        _validate_zero_quantity_with_cost(self.apus_detail_recalculable)
        self.assertEqual(
            original_copy, self.original_apus, "El input original fue modificado"
        )

    def test_validate_descriptions__missing_description(self):
        """Verifica que se asigna texto predeterminado y alerta si no hay descripción."""
        resultado, metrics = _validate_descriptions(self.apus_detail_sin_descripcion)
        self.assertEqual(resultado[0]["DESCRIPCION_INSUMO"], "Insumo sin descripción")
        self.assertIn("alertas", resultado[0])
        self.assertEqual(len(resultado[0]["alertas"]), 1)
        self.assertIn("Descripción inválida", resultado[0]["alertas"][0]["mensaje"])

    def test_validate_descriptions__present_description(self):
        """Verifica que no se añade alerta si la descripción está presente."""
        resultado, metrics = _validate_descriptions(self.apus_detail_con_descripcion)
        self.assertNotIn(
            "alertas", resultado[0], "No debe haber alertas si la descripción está presente"
        )

    def test_validate_descriptions__inmutable_input(self):
        """Verifica que no se modifica el input original."""
        original_copy = deepcopy(self.apus_detail_sin_descripcion)
        _validate_descriptions(self.apus_detail_sin_descripcion)
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
        }

        resultado = validate_and_clean_data(data_store)

        # Verificar que las alertas están presentes y son correctas
        self.assertIn("alertas", resultado["presupuesto"][0])
        self.assertEqual(len(resultado["presupuesto"][0]["alertas"]), 1)
        self.assertIn(
            "Excesivo", resultado["presupuesto"][0]["alertas"][0]["mensaje"]
        )

        self.assertIn("alertas", resultado["apus_detail"][0])
        # Expect 2 alerts: incoherence + recalculation
        self.assertEqual(len(resultado["apus_detail"][0]["alertas"]), 2)
        self.assertIn("Recalculado", resultado["apus_detail"][0]["alertas"][1]["mensaje"])

    def test_validate_and_clean_data__missing_presupuesto_key(self):
        """Verifica que no falla si 'presupuesto' no está presente."""
        data_store = {
            "apus_detail": self.apus_detail_recalculable,
        }
        resultado = validate_and_clean_data(data_store)
        self.assertNotIn("presupuesto", resultado, "No debe añadirse si no estaba")
        self.assertIn("apus_detail", resultado)

    def test_validate_and_clean_data__missing_apus_detail_key(self):
        """Verifica que no falla si 'apus_detail' no está presente."""
        data_store = {
            "presupuesto": self.presupuesto_extremo,
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
        }
        resultado = validate_and_clean_data(data_store)
        self.assertEqual(resultado["presupuesto"], [])

    def test_validate_and_clean_data__none_in_apus_detail(self):
        """Verifica manejo de apus_detail como None."""
        data_store = {
            "presupuesto": self.presupuesto_extremo,
            "apus_detail": None,
        }
        resultado = validate_and_clean_data(data_store)
        self.assertEqual(resultado["apus_detail"], [])

    def test_validate_and_clean_data__inmutable_input(self):
        """Verifica que validate_and_clean_data no modifica el input original."""
        original_data = {
            "presupuesto": deepcopy(self.presupuesto_extremo),
            "apus_detail": deepcopy(self.apus_detail_recalculable),
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


class TestDataValidatorAdvanced(unittest.TestCase):
    """
    Pruebas para componentes avanzados V2: Validación Piramidal, Anomalías,
    Teoría de Control y Entropía.
    """

    def test_control_theory_validation(self):
        """Test the gain and sensitivity analysis in mathematical validation."""
        # Case 1: Stable System (Perfect Match)
        Q = 10.0
        P = 5.0
        VT = 50.0
        is_coherent, diff, msg, analysis = _validar_coherencia_matematica(Q, P, VT)
        self.assertTrue(is_coherent)
        self.assertLess(analysis['sensibilidad'], 20.0) # S = sqrt(10^2 + 5^2) = 11.18

        # Case 2: High Sensitivity (Large values)
        Q_high = 10000.0
        P_high = 5000.0
        VT_high = 50000000.0
        is_coherent, diff, msg, analysis = _validar_coherencia_matematica(Q_high, P_high, VT_high)
        self.assertTrue(is_coherent)
        self.assertGreater(analysis['sensibilidad'], 1000)

        # Case 3: Incoherent with High Sensitivity (Real Error)
        # With default tolerance of 1% (0.01), for 50M, tolerance is 500k.
        # To fail, error > 500k.
        VT_real_error = 55000000.0 # 10% error (5M > 500k)
        is_coherent, _, msg, _ = _validar_coherencia_matematica(Q_high, P_high, VT_real_error)
        self.assertFalse(is_coherent)
        self.assertIn("ALTA SENSIBILIDAD", msg)

    def test_pyramidal_validator(self):
        """Test the graph-based stability validation and SPOF detection."""
        validator = PyramidalValidator()

        # Create a stable pyramid: 1 APU -> 3 Insumos
        apus_df = pd.DataFrame([
            {"CODIGO": "APU-1", "DESCRIPCION": "Muro"},
            {"CODIGO": "APU-2", "DESCRIPCION": "Columna"},
            {"CODIGO": "APU-3", "DESCRIPCION": "Viga"}
        ])
        insumos_df = pd.DataFrame([
            # APU-1
            {"APU_CODIGO": "APU-1", "DESCRIPCION_INSUMO_NORM": "Ladrillo"},
            {"APU_CODIGO": "APU-1", "DESCRIPCION_INSUMO_NORM": "Cemento"},
            # APU-2 (Usa Cemento tambien)
            {"APU_CODIGO": "APU-2", "DESCRIPCION_INSUMO_NORM": "Cemento"},
            {"APU_CODIGO": "APU-2", "DESCRIPCION_INSUMO_NORM": "Hierro"},
            # APU-3 (Usa Cemento tambien)
            {"APU_CODIGO": "APU-3", "DESCRIPCION_INSUMO_NORM": "Cemento"}
        ])

        metrics = validator.validate_structure(apus_df, insumos_df)
        self.assertEqual(metrics.structure_load, 3)
        self.assertGreater(metrics.pyramid_stability_index, 0.0)

        # Check SPOF: Cemento should be a SPOF because it connects to 3 APUs
        # Umbral critico will be max(2, int(3*0.1)) = 2. Cemento has 3 connections.
        resiliencia = metrics.graf_analysis.get("resiliencia", {})
        spofs = resiliencia.get("puntos_fallo_unico", [])

        spof_insumos = [s["insumo"] for s in spofs]
        self.assertIn("Cemento", spof_insumos)

        # Ladrillo connects only to APU-1 (count 1 < 2), so NOT a SPOF
        self.assertNotIn("Ladrillo", spof_insumos)

        # Test Floating Node (Inverted Pyramid / Disconnected)
        apus_df_bad = pd.DataFrame([
            {"CODIGO": "APU-1", "DESCRIPCION": "Muro"},
            {"CODIGO": "APU-4", "DESCRIPCION": "Fantasma"}
        ])
        metrics_bad = validator.validate_structure(apus_df_bad, insumos_df)
        self.assertIn("APU-4", metrics_bad.floating_nodes)

    def test_anomaly_validator(self):
        """Test statistical anomaly detection."""
        validator = AnomalyValidator(config={"zscore_threshold": 2.0})

        # Create data with an obvious outlier
        data = [
            {"VALOR_CONSTRUCCION_UN": 100.0},
            {"VALOR_CONSTRUCCION_UN": 102.0},
            {"VALOR_CONSTRUCCION_UN": 98.0},
            {"VALOR_CONSTRUCCION_UN": 101.0},
            {"VALOR_CONSTRUCCION_UN": 10000.0} # Outlier
        ]

        marked_data, metrics = validator.detect_cost_anomalies(data)

        outliers = [d for d in marked_data if "anomalias" in d]
        self.assertEqual(len(outliers), 1)
        self.assertEqual(outliers[0]["VALOR_CONSTRUCCION_UN"], 10000.0)
        self.assertEqual(outliers[0]["anomalias"][0]["tipo"], "COSTO_ANOMALO")

    def test_quality_entropy_integration(self):
        """Test that the main orchestrator calculates Quality Entropy (S_data)."""
        data_store = {
            "presupuesto": [{"VALOR_CONSTRUCCION_UN": 100.0}],
            "apus_detail": []
        }

        result = validate_and_clean_data(data_store, aplicar_analisis_termico=True)

        self.assertIn("quality_entropy_analysis", result)
        qa = result["quality_entropy_analysis"]
        self.assertIn("quality_entropy_final", qa)
        self.assertIn("stability_status", qa)

        # Verify renaming: should NOT use "temperatura" keys from old proposal
        self.assertNotIn("temperatura_final", qa)
        self.assertIn("quality_entropy_final", qa)

    def test_alert_entropy_impact(self):
        """Test that alerts increase entropy."""
        # Data with many alerts
        data_bad = {
            "presupuesto": [{"VALOR_CONSTRUCCION_UN": -100.0}] * 10, # Negative costs
            "apus_detail": []
        }

        result = validate_and_clean_data(data_bad)
        entropy = result["quality_entropy_analysis"]["quality_entropy_final"]
        self.assertGreater(entropy, 50.0) # Should be high/chaotic
        self.assertEqual(result["quality_entropy_analysis"]["stability_status"], "CAOTICO")


if __name__ == "__main__":
    unittest.main()
