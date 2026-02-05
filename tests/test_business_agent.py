"""
Pruebas Unitarias para Refinamientos de BusinessAgent
======================================================

Valida específicamente las mejoras implementadas:
1. Nueva fórmula de structural_coherence (decaimiento exponencial).
2. RiskChallenger con umbrales configurables y multi-nivel.
3. Operaciones Algebraicas robustas.
4. Validación extendida de DataFrames.
"""

import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from app.business_agent import (
    TopologicalMetricsBundle,
    RiskChallenger,
    AlgebraicOperations,
    BusinessAgent,
    ConstructionRiskReport
)
from app.constants import ColumnNames
from app.tools_interface import MICRegistry

class TestTopologicalMetricsBundle:
    def test_structural_coherence_formula(self):
        """Valida que la coherencia siga la nueva fórmula de decaimiento."""
        # Mock de un grafo con 10 nodos
        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 10

        # Caso ideal: beta_0=1, beta_1=0, stability=1.0
        bundle = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 1, "beta_1": 0},
            pyramid_stability=1.0,
            graph=mock_graph
        )
        # exp(0) * exp(0) * 1.0 = 1.0
        assert bundle.structural_coherence == pytest.approx(1.0)

        # Caso con fragmentación: beta_0=2
        bundle2 = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 2, "beta_1": 0},
            pyramid_stability=1.0,
            graph=mock_graph
        )
        # exp(-ln(2)*1) * exp(0) * 1.0 = 0.5
        assert bundle2.structural_coherence == pytest.approx(0.5)

        # Caso con ciclos: beta_1=1
        bundle3 = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 1, "beta_1": 1},
            pyramid_stability=1.0,
            graph=mock_graph
        )
        # lambda_1 = ln(2) / sqrt(10) ≈ 0.693 / 3.162 ≈ 0.219
        # exp(0) * exp(-0.219 * 1) * 1.0 ≈ 0.803
        lambda_1 = math.log(2) / math.sqrt(10)
        expected = math.exp(-lambda_1 * 1)
        assert bundle3.structural_coherence == pytest.approx(expected)

class TestRiskChallengerRefined:
    def test_init_with_custom_thresholds(self):
        custom = {"critical_stability": 0.5}
        challenger = RiskChallenger(custom)
        assert challenger.thresholds["critical_stability"] == 0.5
        assert challenger.thresholds["warning_stability"] == 0.85 # Default preserved

    def test_veto_critical_instability(self):
        challenger = RiskChallenger()
        # Ψ < 0.70 (default critical)
        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Baja",
            financial_risk_level="BAJO",
            details={"pyramid_stability": 0.5},
            strategic_narrative="Todo bien."
        )

        audited = challenger.challenge_verdict(report)

        assert audited.financial_risk_level == "RIESGO ESTRUCTURAL (CRÍTICO)"
        assert audited.integrity_score == 70.0 # 100 * (1 - 0.3)
        assert "ACTA DE DELIBERACIÓN" in audited.strategic_narrative

    def test_warning_suboptimal_stability(self):
        challenger = RiskChallenger()
        # 0.70 <= Ψ < 0.85 (default warning)
        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Baja",
            financial_risk_level="SAFE",
            details={"pyramid_stability": 0.8},
            strategic_narrative="Todo bien."
        )

        audited = challenger.challenge_verdict(report)

        assert audited.financial_risk_level == "RIESGO ESTRUCTURAL (SEVERO)"
        assert audited.integrity_score == 85.0 # 100 * (1 - 0.15)

    def test_cycle_density_warning(self):
        challenger = RiskChallenger()
        # beta_1 / n > 0.33
        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Baja",
            financial_risk_level="MODERADO",
            details={
                "pyramid_stability": 0.9,
                "topological_invariants": {
                    "betti_numbers": {"beta_1": 5},
                    "n_nodes": 10
                }
            },
            strategic_narrative="Todo bien."
        )

        audited = challenger.challenge_verdict(report)

        # 5/10 = 0.5 > 0.33 -> Alerta
        assert audited.integrity_score == 95.0 # Penalización leve
        assert "challenger_cycle_warning" in audited.details

class TestAlgebraicOperations:
    def test_safe_normalize(self):
        v = np.array([3.0, 4.0])
        normed = AlgebraicOperations.safe_normalize(v)
        assert np.linalg.norm(normed) == pytest.approx(1.0)
        assert normed[0] == 0.6

        # Vector zero
        v0 = np.array([0.0, 0.0])
        normed0 = AlgebraicOperations.safe_normalize(v0)
        assert np.linalg.norm(normed0) == pytest.approx(1.0)
        assert np.all(normed0 == 1.0 / np.sqrt(2))

    def test_weighted_geometric_mean(self):
        factors = [0.5, 0.5, 0.5]
        weights = [1.0, 1.0, 1.0]
        result = AlgebraicOperations.weighted_geometric_mean(factors, weights)
        assert result == pytest.approx(0.5)

        factors2 = [0.1, 0.9]
        weights2 = [1.0, 1.0]
        # sqrt(0.1 * 0.9) = sqrt(0.09) = 0.3
        result2 = AlgebraicOperations.weighted_geometric_mean(factors2, weights2)
        assert result2 == pytest.approx(0.3)

class TestBusinessAgentRefined:
    def test_validate_dataframes_with_diagnostics(self):
        mic = MagicMock(spec=MICRegistry)
        agent = BusinessAgent({}, mic)

        df_p = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["A1"],
            ColumnNames.DESCRIPCION_APU: ["Desc"],
            ColumnNames.VALOR_TOTAL: [100.0]
        })
        df_d = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["A1"],
            ColumnNames.DESCRIPCION_INSUMO: ["I1"],
            ColumnNames.CANTIDAD_APU: [1.0],
            ColumnNames.COSTO_INSUMO_EN_APU: [100.0]
        })

        valid, msg, diag = agent._validate_dataframes(df_p, df_d)

        assert valid is True
        assert diag is not None
        assert "row_counts" in diag
        assert diag["row_counts"]["presupuesto"] == 1

    def test_validate_dataframes_detects_outliers(self):
        mic = MagicMock(spec=MICRegistry)
        agent = BusinessAgent({}, mic)

        # Generar 20 filas, una muy alta
        vals = [100.0] * 19 + [10000.0]
        df_p = pd.DataFrame({
            ColumnNames.CODIGO_APU: [f"A{i}" for i in range(20)],
            ColumnNames.DESCRIPCION_APU: ["Desc"] * 20,
            ColumnNames.VALOR_TOTAL: vals
        })
        df_d = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["A0"],
            ColumnNames.DESCRIPCION_INSUMO: ["I1"],
            ColumnNames.CANTIDAD_APU: [1.0],
            ColumnNames.COSTO_INSUMO_EN_APU: [100.0]
        })

        valid, msg, diag = agent._validate_dataframes(df_p, df_d)
        assert "distribution_analysis" in diag
        assert diag["distribution_analysis"]["outlier_count"] == 1

if __name__ == "__main__":
    pytest.main([__file__])
