# -*- coding: utf-8 -*-
import pytest
import numpy as np
from math import isclose
from app.financial_engine import (
    FinancialConfig,
    CapitalAssetPricing,
    RiskQuantifier,
    RealOptionsAnalyzer,
    FinancialEngine,
    DistributionType,
    OptionModelType,
    calculate_volatility_from_returns
)

@pytest.fixture
def config():
    return FinancialConfig(
        risk_free_rate=0.04,
        market_premium=0.06,
        beta=1.0,
        tax_rate=0.3,
        cost_of_debt=0.08,
        debt_to_equity_ratio=1.0
    )

class TestRefinedCapitalAssetPricing:
    def test_beta_regularization(self):
        # Beta very low should be regularized
        low_beta_config = FinancialConfig(beta=0.01)
        capm = CapitalAssetPricing(low_beta_config)
        # 0.3 * 0.01 + 0.7 * 1.0 = 0.003 + 0.7 = 0.703
        assert isclose(low_beta_config.beta, 0.703)

    def test_calculate_ke_structural_adjustment(self, config):
        capm = CapitalAssetPricing(config)
        base_ke = capm.calculate_ke(structural_risk_adjustment=1.0)
        adj_ke = capm.calculate_ke(structural_risk_adjustment=1.2)
        assert adj_ke == pytest.approx(base_ke * 1.2)

        # Test max bound (Rf + 2*MRP = 0.04 + 2*0.06 = 0.16)
        extreme_ke = capm.calculate_ke(structural_risk_adjustment=2.0)
        assert extreme_ke <= 0.16000000000000003

    def test_calculate_wacc_topological_coherence(self, config):
        capm = CapitalAssetPricing(config)
        # base coherence = 1.0 -> penalty = 0
        wacc_base = capm.calculate_wacc(topological_coherence=1.0)
        # coherence = 0.5 -> penalty = 0.3 * (1 - 0.5) = 0.15. WACC_adj = WACC_base * 1.15
        wacc_adj = capm.calculate_wacc(topological_coherence=0.5)
        assert wacc_adj == pytest.approx(wacc_base * 1.15)

    def test_calculate_npv_certainty_equivalent(self, config):
        capm = CapitalAssetPricing(config)
        cash_flows = [1000, 1000, 1000]
        # With certainty_equivalent=1.0 and lambda=0.1
        # NPV = -0 + 1000*exp(-0.1)/(1+wacc) + 1000*exp(-0.2)/(1+wacc)^2 + 1000*exp(-0.3)/(1+wacc)^3
        npv = capm.calculate_npv(cash_flows, initial_investment=0, certainty_equivalent=1.0)

        wacc = capm.calculate_wacc()
        expected = (1000 * np.exp(-0.1) / (1+wacc) +
                    1000 * np.exp(-0.2) / (1+wacc)**2 +
                    1000 * np.exp(-0.3) / (1+wacc)**3)
        assert npv == pytest.approx(expected)

class TestRefinedRiskQuantifier:
    def test_cornish_fisher_expansion(self):
        rq = RiskQuantifier(DistributionType.NORMAL)
        mean, std = 100, 10
        # Normal
        var_norm, _ = rq.calculate_var(mean, std, confidence_level=0.95)

        # With positive skewness (right tail is fatter for costs)
        var_skew, metrics = rq.calculate_var(mean, std, confidence_level=0.95, skewness=1.0)
        assert var_skew > var_norm
        assert "Normal CF" in metrics["distribution"]

    def test_monte_carlo_metrics(self):
        rq = RiskQuantifier(DistributionType.NORMAL)
        res = rq.calculate_risk_metrics_monte_carlo(mean=100, std_dev=10, n_simulations=1000)
        assert "var_95" in res
        assert "cvar_95" in res
        assert res["cvar_95"] >= res["var_95"]
        assert "max_drawdown" in res

class TestRefinedRealOptionsAnalyzer:
    def test_black_scholes_valuation(self):
        roa = RealOptionsAnalyzer(OptionModelType.BLACK_SCHOLES)
        res = roa.value_option_to_wait(
            project_value=100, investment_cost=100, risk_free_rate=0.05, time_to_expire=1, volatility=0.2
        )
        assert res["model"] == "Black-Scholes-Merton"
        assert res["option_value"] > 0
        assert "delta" in res
        assert "gamma" in res

    def test_binomial_greeks(self):
        roa = RealOptionsAnalyzer(OptionModelType.BINOMIAL)
        res = roa.value_option_to_wait(
            project_value=100, investment_cost=100, risk_free_rate=0.05, time_to_expire=1, volatility=0.2, steps=50
        )
        assert "delta" in res
        assert "gamma" in res
        assert "vega" in res
        assert 0 <= res["delta"] <= 1

class TestRefinedFinancialEngine:
    def test_thermo_structural_volatility(self, config):
        engine = FinancialEngine(config)
        # Base vol = 0.2, Psi=1.5, Temp=25 -> no adjustment (if coherence=1, pressure=1)
        res = engine._calculate_thermo_structural_volatility(0.2, 1.5, 25.0)
        assert res["volatility_adjusted"] == pytest.approx(0.2)

        # High temperature -> increased vol
        res_hot = engine._calculate_thermo_structural_volatility(0.2, 1.0, 45.0)
        assert res_hot["volatility_adjusted"] > 0.2

        # Low stability -> increased vol
        res_unstable = engine._calculate_thermo_structural_volatility(0.2, 0.5, 25.0)
        assert res_unstable["volatility_adjusted"] > 0.2

    def test_robust_metrics(self, config):
        engine = FinancialEngine(config)
        res = engine.calculate_robust_metrics(npv=1000, cash_flows=[5000, 5000], investment=8000, volatility=0.2)
        assert "safety_margin" in res
        assert "robustness_index" in res
        assert "probability_of_breach" in res
        assert "rating" in res

    def test_analyze_project_full(self, config):
        engine = FinancialEngine(config)
        res = engine.analyze_project(
            initial_investment=1000,
            cash_flows=[400, 400, 400],
            cost_std_dev=100,
            project_volatility=0.2,
            pyramid_stability=1.2,
            system_temperature=35.0,
            structural_coherence=0.8
        )
        assert "physics_details" in res
        assert "robustness" in res
        assert "thermodynamics" in res
        assert "stability_class" in res["thermodynamics"]

class TestRefinedVolatilityUtils:
    def test_volatility_methods(self):
        returns = [0.01, -0.01, 0.02, -0.02, 0.015, -0.005]
        res_std = calculate_volatility_from_returns(returns, method="standard")
        res_ewma = calculate_volatility_from_returns(returns, method="ewma")
        res_garch = calculate_volatility_from_returns(returns, method="garch")

        assert res_std["volatility"] > 0
        assert res_ewma["volatility"] > 0
        assert res_garch["volatility"] > 0
        assert "normality_test" in res_std
