import unittest
import pandas as pd
import numpy as np
from app.data_validator import (
    _validar_coherencia_matematica,
    PyramidalValidator,
    AnomalyValidator,
    validate_and_clean_data,
    TipoAlerta
)

class TestDataValidatorV2(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
