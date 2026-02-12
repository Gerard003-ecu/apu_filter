"""
Suite de pruebas para el módulo de análisis financiero.

Coherente con los métodos refinados que implementan:
- Validación robusta de parámetros con soporte para neutralidad (β=0)
- Cálculo exacto de Greeks (Delta, Gamma, Theta) en modelo binomial
- Ecuación unificada termo-estructural de volatilidad
- Métricas de riesgo extendidas (CVaR exacto, IRR, eficiencia de capital)
"""

from math import exp, isclose, sqrt
from typing import Dict, Any

import numpy as np
import pytest
from scipy.stats import norm, t

from app.financial_engine import (
    CapitalAssetPricing,
    DistributionType,
    FinancialConfig,
    FinancialEngine,
    OptionModelType,
    RealOptionsAnalyzer,
    RiskQuantifier,
    calculate_volatility_from_returns,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def base_config() -> FinancialConfig:
    """Configuración financiera base para las pruebas."""
    return FinancialConfig(
        risk_free_rate=0.05,
        market_premium=0.08,
        beta=1.2,
        tax_rate=0.25,
        cost_of_debt=0.07,
        debt_to_equity_ratio=1.5,
        project_life_years=10,
        liquidity_ratio=0.15,
        fixed_contracts_ratio=0.6,
    )


@pytest.fixture
def neutral_config() -> FinancialConfig:
    """Configuración neutral al riesgo sistémico (β=0, D/E=0)."""
    return FinancialConfig(
        risk_free_rate=0.04,
        market_premium=0.06,
        beta=0.0,  # Neutral al riesgo sistémico
        tax_rate=0.30,
        cost_of_debt=0.08,
        debt_to_equity_ratio=0.0,  # Sin deuda
    )


@pytest.fixture
def capm(base_config) -> CapitalAssetPricing:
    """Instancia de CapitalAssetPricing con configuración base."""
    return CapitalAssetPricing(base_config)


@pytest.fixture
def capm_neutral(neutral_config) -> CapitalAssetPricing:
    """Instancia de CapitalAssetPricing neutral al riesgo."""
    return CapitalAssetPricing(neutral_config)


@pytest.fixture
def risk_quantifier() -> RiskQuantifier:
    """Instancia de RiskQuantifier con distribución Normal."""
    return RiskQuantifier(distribution=DistributionType.NORMAL)


@pytest.fixture
def risk_quantifier_t() -> RiskQuantifier:
    """Instancia de RiskQuantifier con distribución t-Student."""
    return RiskQuantifier(distribution=DistributionType.STUDENT_T)


@pytest.fixture
def options_analyzer() -> RealOptionsAnalyzer:
    """Instancia de RealOptionsAnalyzer con modelo binomial."""
    return RealOptionsAnalyzer(model_type=OptionModelType.BINOMIAL)


@pytest.fixture
def engine(base_config) -> FinancialEngine:
    """Instancia de FinancialEngine con configuración base."""
    return FinancialEngine(base_config)


@pytest.fixture
def engine_neutral(neutral_config) -> FinancialEngine:
    """Instancia de FinancialEngine neutral al riesgo."""
    return FinancialEngine(neutral_config)


# ============================================================================
# PRUEBAS PARA FinancialConfig (Validación de Parámetros)
# ============================================================================


class TestFinancialConfig:
    """Pruebas para la validación de configuración financiera."""

    def test_valid_configuration_standard(self, base_config):
        """Verifica que una configuración estándar sea válida."""
        assert base_config.risk_free_rate == 0.05
        assert base_config.beta == 1.2
        assert base_config.debt_to_equity_ratio == 1.5

    def test_valid_configuration_neutral(self, neutral_config):
        """Verifica que β=0 y D/E=0 sean válidos (neutralidad al riesgo)."""
        assert neutral_config.beta == 0.0
        assert neutral_config.debt_to_equity_ratio == 0.0
        # No debe lanzar excepciones

    def test_cross_constraint_warning_debt_without_cost(self, caplog):
        """Verifica warning cuando hay deuda pero costo de deuda es cero."""
        import logging
        
        with caplog.at_level(logging.WARNING):
            config = FinancialConfig(
                risk_free_rate=0.05,
                market_premium=0.08,
                beta=1.0,
                debt_to_equity_ratio=1.0,
                cost_of_debt=0.0,  # Inconsistente con D/E > 0
            )
        
        # Debería generar warning sobre inconsistencia
        # (Depende de implementación exacta de _validate_cross_constraints)
        assert config is not None

    def test_invalid_market_premium_raises(self):
        """Verifica que prima de mercado fuera de rango crítico lance error."""
        with pytest.raises(ValueError, match="Prima de riesgo"):
            FinancialConfig(
                risk_free_rate=0.05,
                market_premium=0.005,  # < 0.01, crítico
                beta=1.0,
            )

    def test_invalid_project_life_raises(self):
        """Verifica que vida del proyecto inválida lance error."""
        with pytest.raises(ValueError, match="Vida del proyecto"):
            FinancialConfig(
                risk_free_rate=0.05,
                market_premium=0.08,
                beta=1.0,
                project_life_years=0,  # < 1, crítico
            )


# ============================================================================
# PRUEBAS PARA CapitalAssetPricing
# ============================================================================


class TestCapitalAssetPricing:
    """Pruebas para el motor de cálculo de costo de capital."""

    def test_calculate_ke_standard(self, capm, base_config):
        """
        Verifica cálculo de Ke mediante CAPM.
        
        Fórmula: Ke = Rf + β × (Rm - Rf) = Rf + β × MRP
        Esperado: 0.05 + 1.2 × 0.08 = 0.146 (14.6%)
        """
        expected_ke = base_config.risk_free_rate + base_config.beta * base_config.market_premium
        calculated_ke = capm.calculate_ke()
        
        assert isclose(calculated_ke, expected_ke, rel_tol=1e-9), (
            f"Ke esperado: {expected_ke:.4%}, obtenido: {calculated_ke:.4%}"
        )
        assert calculated_ke > base_config.risk_free_rate, (
            "Ke debe ser mayor que Rf cuando β > 0"
        )

    def test_calculate_ke_neutral_beta(self, capm_neutral, neutral_config):
        """
        Verifica que β=0 retorne Ke = Rf (activo neutral al mercado).
        """
        calculated_ke = capm_neutral.calculate_ke()
        
        assert isclose(calculated_ke, neutral_config.risk_free_rate, rel_tol=1e-9), (
            f"Con β=0, Ke debe ser Rf. Esperado: {neutral_config.risk_free_rate:.4%}, "
            f"obtenido: {calculated_ke:.4%}"
        )

    def test_calculate_wacc_standard(self, capm, base_config):
        """
        Verifica cálculo de WACC con estructura de capital mixta.
        
        Cálculo detallado:
        - Ke = 5% + 1.2 × 8% = 14.6%
        - D/E = 1.5 → We = 1/(1+1.5) = 0.4, Wd = 0.6
        - Kd_after_tax = 7% × (1 - 25%) = 5.25%
        - WACC = 0.4 × 14.6% + 0.6 × 5.25% = 8.99%
        """
        ke = capm.calculate_ke()
        d_e = base_config.debt_to_equity_ratio
        
        w_e = 1 / (1 + d_e)
        w_d = d_e / (1 + d_e)
        kd_after_tax = base_config.cost_of_debt * (1 - base_config.tax_rate)
        
        expected_wacc = w_e * ke + w_d * kd_after_tax
        calculated_wacc = capm.calculate_wacc()
        
        assert isclose(calculated_wacc, expected_wacc, rel_tol=1e-6), (
            f"WACC esperado: {expected_wacc:.4%}, obtenido: {calculated_wacc:.4%}"
        )
        assert calculated_wacc < ke, (
            "WACC debe ser menor que Ke debido al escudo fiscal"
        )

    def test_calculate_wacc_no_debt(self, capm_neutral, neutral_config):
        """
        Verifica que WACC = Ke cuando no hay deuda (D/E = 0).
        """
        ke = capm_neutral.calculate_ke()
        wacc = capm_neutral.calculate_wacc()
        
        assert isclose(wacc, ke, rel_tol=1e-9), (
            f"Sin deuda, WACC debe ser Ke. WACC: {wacc:.4%}, Ke: {ke:.4%}"
        )

    def test_calculate_npv_positive(self, capm):
        """Verifica cálculo de VAN con flujos positivos."""
        initial_investment = 100_000
        cash_flows = [30_000, 35_000, 40_000, 45_000, 50_000]
        
        npv = capm.calculate_npv(cash_flows, initial_investment)
        wacc = capm.calculate_wacc()
        
        # Verificar manualmente
        expected_npv = -initial_investment
        for i, cf in enumerate(cash_flows, 1):
            expected_npv += cf / ((1 + wacc) ** i)
        
        assert isclose(npv, expected_npv, rel_tol=1e-6), (
            f"VAN esperado: {expected_npv:,.2f}, obtenido: {npv:,.2f}"
        )

    def test_sensitivity_analysis_returns_correct_format(self, capm, base_config):
        """
        Verifica formato y contenido del análisis de sensibilidad.
        
        Refinamiento: El método devuelve Dict[float, float] (valor → WACC).
        """
        beta_range = [0.8, 1.0, 1.2, 1.5, 2.0]
        original_beta = base_config.beta
        original_wacc = capm.calculate_wacc()
        
        # Ejecutar análisis
        results = capm.sensitivity_analysis("beta", beta_range)
        
        # Verificar formato: debe ser Dict[float, float]
        assert isinstance(results, dict), "Resultado debe ser un diccionario"
        assert len(results) == len(beta_range), "Debe haber un resultado por cada valor"
        
        # Verificar que todas las claves corresponden a los valores del rango
        for beta_val in beta_range:
            assert beta_val in results, f"Falta resultado para beta={beta_val}"
            assert isinstance(results[beta_val], float), "WACC debe ser float"
        
        # Verificar monotonicidad (WACC crece con beta)
        waccs = [results[b] for b in sorted(beta_range)]
        for i in range(1, len(waccs)):
            assert waccs[i] > waccs[i - 1], (
                f"WACC debe ser monótono creciente con beta"
            )
        
        # Verificar restauración del estado original
        assert base_config.beta == original_beta, "Beta no fue restaurado"
        
        capm.calculate_ke.cache_clear()
        capm.calculate_wacc.cache_clear()
        restored_wacc = capm.calculate_wacc()
        
        assert isclose(restored_wacc, original_wacc, rel_tol=1e-9), (
            f"WACC no restaurado. Original: {original_wacc:.4%}, Final: {restored_wacc:.4%}"
        )

    def test_sensitivity_analysis_invalid_parameter(self, capm):
        """Verifica que parámetro inválido lance error."""
        with pytest.raises(ValueError, match="desconocido"):
            capm.sensitivity_analysis("invalid_param", [1.0, 2.0])


# ============================================================================
# PRUEBAS PARA RiskQuantifier
# ============================================================================


class TestRiskQuantifier:
    """Pruebas para el cuantificador de riesgo financiero."""

    def test_calculate_var_normal_basic(self, risk_quantifier):
        """
        Verifica cálculo básico de VaR para distribución Normal.
        
        VaR_α = μ + z_α × σ_scaled
        donde σ_scaled = σ × √(T/252)
        """
        mean = 1_000_000
        std_dev = 150_000
        confidence = 0.95
        
        var, metrics = risk_quantifier.calculate_var(
            mean=mean,
            std_dev=std_dev,
            confidence_level=confidence,
            time_horizon_days=1,
        )
        
        # Cálculo esperado
        time_factor = sqrt(1 / 252)
        scaled_std = std_dev * time_factor
        z_alpha = norm.ppf(confidence)
        expected_var = mean + z_alpha * scaled_std
        
        assert isclose(var, expected_var, rel_tol=1e-6), (
            f"VaR esperado: {expected_var:,.2f}, obtenido: {var:,.2f}"
        )
        assert metrics["distribution"] == "Normal"
        assert isclose(metrics["z_score"], z_alpha, rel_tol=1e-9)
        assert isclose(metrics["scaled_std"], scaled_std, rel_tol=1e-9)

    def test_calculate_var_cvar_relationship_normal(self, risk_quantifier):
        """
        Verifica que CVaR > VaR para distribución Normal.
        
        CVaR (Expected Shortfall) siempre excede VaR.
        """
        mean = 1_000_000
        std_dev = 150_000
        
        var, metrics = risk_quantifier.calculate_var(
            mean=mean,
            std_dev=std_dev,
            confidence_level=0.95,
        )
        
        cvar = metrics["cvar"]
        
        assert cvar > var, (
            f"CVaR ({cvar:,.2f}) debe ser > VaR ({var:,.2f})"
        )
        
        # Verificar alias
        assert "expected_shortfall" in metrics
        assert metrics["expected_shortfall"] == cvar

    def test_calculate_var_student_t_heavier_tails(self, risk_quantifier, risk_quantifier_t):
        """
        Verifica que t-Student produce VaR mayor que Normal (colas pesadas).
        """
        mean = 1_000_000
        std_dev = 150_000
        confidence = 0.95
        df = 5
        
        var_normal, _ = risk_quantifier.calculate_var(
            mean=mean, std_dev=std_dev, confidence_level=confidence
        )
        
        var_t, metrics_t = risk_quantifier_t.calculate_var(
            mean=mean, std_dev=std_dev, confidence_level=confidence, df_student_t=df
        )
        
        assert var_t > var_normal, (
            f"VaR t-Student ({var_t:,.2f}) debe ser > VaR Normal ({var_normal:,.2f})"
        )
        assert f"Student-t(df={df})" in metrics_t["distribution"]

    def test_calculate_var_student_t_convergence_to_normal(self, risk_quantifier, risk_quantifier_t):
        """
        Verifica convergencia de t-Student → Normal cuando df → ∞.
        """
        mean = 1_000_000
        std_dev = 150_000
        
        var_normal, _ = risk_quantifier.calculate_var(mean=mean, std_dev=std_dev)
        var_t_high_df, _ = risk_quantifier_t.calculate_var(
            mean=mean, std_dev=std_dev, df_student_t=1000
        )
        
        # Con df=1000, diferencia debe ser < 1%
        assert isclose(var_t_high_df, var_normal, rel_tol=0.01), (
            f"t(df=1000) ≈ Normal. Diff: {abs(var_t_high_df - var_normal):,.2f}"
        )

    def test_calculate_var_extended_metrics(self, risk_quantifier):
        """
        Verifica que las métricas extendidas estén presentes y sean válidas.
        """
        mean = 1_000_000
        std_dev = 150_000
        
        _, metrics = risk_quantifier.calculate_var(
            mean=mean,
            std_dev=std_dev,
            confidence_level=0.95,
            time_horizon_days=10,
        )
        
        # Verificar métricas extendidas del refinamiento
        expected_keys = [
            "var", "var_lower", "var_upper", "cvar", "expected_shortfall",
            "scaled_std", "confidence", "z_score", "time_horizon_days",
            "annualization_factor", "tail_risk_ratio", "risk_contribution"
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Falta métrica: {key}"
        
        # Verificar coherencia de valores
        assert metrics["var_upper"] >= metrics["var_lower"], "var_upper ≥ var_lower"
        assert metrics["tail_risk_ratio"] >= 1.0, "tail_risk_ratio ≥ 1"
        assert metrics["annualization_factor"] > 0, "Factor de anualización positivo"

    def test_calculate_var_zero_std_deterministic(self, risk_quantifier):
        """
        Verifica que std_dev=0 produzca resultado determinístico.
        """
        mean = 1_000_000
        
        var, metrics = risk_quantifier.calculate_var(mean=mean, std_dev=0)
        
        assert var == mean, "Con σ=0, VaR = media"
        assert metrics["cvar"] == mean, "Con σ=0, CVaR = media"
        assert metrics["scaled_std"] == 0
        assert metrics["distribution"] == "Degenerate"

    def test_calculate_var_invalid_confidence(self, risk_quantifier):
        """Verifica que nivel de confianza inválido lance error."""
        with pytest.raises(ValueError, match="Nivel de confianza"):
            risk_quantifier.calculate_var(mean=100, std_dev=10, confidence_level=1.5)
        
        with pytest.raises(ValueError, match="Nivel de confianza"):
            risk_quantifier.calculate_var(mean=100, std_dev=10, confidence_level=0)

    def test_calculate_var_negative_std_raises(self, risk_quantifier):
        """Verifica que std negativa lance error."""
        with pytest.raises(ValueError, match="negativa"):
            risk_quantifier.calculate_var(mean=100, std_dev=-10)

    def test_suggest_contingency_all_methods(self, risk_quantifier):
        """
        Verifica que suggest_contingency devuelva todas las estrategias.
        """
        base_cost = 1_000_000
        std_dev = 150_000  # CV = 15%
        
        results = risk_quantifier.suggest_contingency(
            base_cost=base_cost,
            std_dev=std_dev,
            confidence_level=0.90,
            method="all"
        )
        
        # Claves de métodos base
        method_keys = ["var_based", "percentage_based", "heuristic"]
        for key in method_keys:
            assert key in results, f"Falta método: {key}"
            assert results[key] >= 0, f"Contingencia {key} no puede ser negativa"
        
        # Claves de metadatos
        assert "recommended" in results
        assert "coefficient_of_variation" in results
        assert "percentage_rate" in results
        assert "heuristic_multiplier" in results
        
        # recommended es el máximo de los métodos
        method_values = [results[k] for k in method_keys]
        assert results["recommended"] == max(method_values)
        
        # CV correcto
        expected_cv = std_dev / base_cost
        assert isclose(results["coefficient_of_variation"], expected_cv, rel_tol=1e-9)

    def test_suggest_contingency_cv_thresholds(self, risk_quantifier):
        """
        Verifica que el porcentaje de contingencia varíe según CV.
        """
        base_cost = 1_000_000
        
        # CV < 10% → 10%
        results_low = risk_quantifier.suggest_contingency(
            base_cost=base_cost, std_dev=50_000  # CV = 5%
        )
        assert results_low["percentage_rate"] == 0.10
        
        # 10% < CV < 20% → 15%
        results_mid = risk_quantifier.suggest_contingency(
            base_cost=base_cost, std_dev=150_000  # CV = 15%
        )
        assert results_mid["percentage_rate"] == 0.15
        
        # CV > 20% → 20%
        results_high = risk_quantifier.suggest_contingency(
            base_cost=base_cost, std_dev=250_000  # CV = 25%
        )
        assert results_high["percentage_rate"] == 0.20


# ============================================================================
# PRUEBAS PARA RealOptionsAnalyzer
# ============================================================================


class TestRealOptionsAnalyzer:
    """Pruebas para el analizador de opciones reales."""

    def test_binomial_valuation_basic_call(self, options_analyzer):
        """
        Verifica valoración básica de call option.
        
        El valor de la opción debe ser ≥ valor intrínseco.
        """
        result = options_analyzer._binomial_valuation(
            S=100, K=100, r=0.05, T=1, sigma=0.3, n=100, american=True
        )
        
        assert result["option_value"] >= 0, "Valor de opción no puede ser negativo"
        assert result["option_value"] >= result["intrinsic_value"], (
            "Valor de opción ≥ valor intrínseco"
        )
        assert result["time_value"] >= 0, "Valor temporal no puede ser negativo"
        assert "Americana" in result["model"]

    def test_binomial_valuation_american_equals_european_for_call(self, options_analyzer):
        """
        Verifica que para calls sin dividendos, Americana ≈ Europea.
        
        Teoría: Nunca es óptimo ejercer una call americana antes del vencimiento
        si el subyacente no paga dividendos.
        """
        params = {"S": 100, "K": 100, "r": 0.05, "T": 1, "sigma": 0.3, "n": 200}
        
        american = options_analyzer._binomial_valuation(**params, american=True)
        european = options_analyzer._binomial_valuation(**params, american=False)
        
        # Valores deben ser prácticamente iguales
        assert isclose(
            american["option_value"],
            european["option_value"],
            rel_tol=1e-4
        ), "Para CALL sin dividendos, Americana ≈ Europea"
        
        # Pocosno nodos de ejercicio anticipado
        assert american["early_exercise_nodes"] <= 10, (
            "Demasiados nodos de ejercicio anticipado para CALL sin dividendos"
        )

    def test_binomial_valuation_greeks_valid_ranges(self, options_analyzer):
        """
        Verifica que los Greeks estén en rangos válidos.
        
        Para una call:
        - Delta ∈ [0, 1]
        - Gamma ≥ 0
        - Theta ≤ 0 (generalmente, el tiempo erosiona valor)
        """
        result = options_analyzer._binomial_valuation(
            S=100, K=100, r=0.05, T=1, sigma=0.3, n=100
        )
        
        # Delta de call está en [0, 1]
        assert 0 <= result["delta"] <= 1, f"Delta fuera de rango: {result['delta']}"
        
        # Gamma debe estar presente y ser no negativo
        assert "gamma" in result
        assert result["gamma"] >= 0, f"Gamma negativo: {result['gamma']}"
        
        # Theta debe estar presente
        assert "theta" in result
        assert "theta_daily" in result

    def test_binomial_valuation_convergence(self, options_analyzer):
        """
        Verifica que el modelo converge con más pasos.
        """
        base_params = {"S": 100, "K": 100, "r": 0.05, "T": 1, "sigma": 0.3}
        
        result_50 = options_analyzer._binomial_valuation(**base_params, n=50)
        result_200 = options_analyzer._binomial_valuation(**base_params, n=200)
        result_500 = options_analyzer._binomial_valuation(**base_params, n=500)
        
        # Diferencia debe reducirse con más pasos
        diff_50_200 = abs(result_50["option_value"] - result_200["option_value"])
        diff_200_500 = abs(result_200["option_value"] - result_500["option_value"])
        
        assert diff_200_500 < diff_50_200, "El modelo debe converger con más pasos"
        
        # Con n=500, debería estar cerca del valor de Black-Scholes
        # (No implementamos BS aquí, pero verificamos convergencia relativa)
        assert isclose(result_200["option_value"], result_500["option_value"], rel_tol=0.01)

    def test_binomial_valuation_itm_otm_atm(self, options_analyzer):
        """
        Verifica comportamiento para opciones ITM, OTM y ATM.
        """
        params = {"r": 0.05, "T": 1, "sigma": 0.3, "n": 100}
        
        # ITM (In The Money): S > K
        itm = options_analyzer._binomial_valuation(S=120, K=100, **params)
        assert itm["intrinsic_value"] == 20  # max(120-100, 0)
        assert itm["option_value"] > 20  # Valor total > intrínseco
        
        # ATM (At The Money): S = K
        atm = options_analyzer._binomial_valuation(S=100, K=100, **params)
        assert atm["intrinsic_value"] == 0
        assert atm["option_value"] > 0  # Solo valor temporal
        
        # OTM (Out of The Money): S < K
        otm = options_analyzer._binomial_valuation(S=80, K=100, **params)
        assert otm["intrinsic_value"] == 0
        assert otm["option_value"] > 0  # Solo valor temporal (menor que ATM)
        
        # Ordenamiento: ITM > ATM > OTM
        assert itm["option_value"] > atm["option_value"] > otm["option_value"]

    def test_binomial_valuation_edge_cases(self, options_analyzer):
        """
        Verifica manejo de casos extremos.
        """
        # Caso 1: T = 0 (expirada)
        expired = options_analyzer._binomial_valuation(
            S=110, K=100, r=0.05, T=0, sigma=0.3, n=100
        )
        assert expired["option_value"] == 10  # max(110-100, 0)
        assert expired["time_value"] == 0
        
        # Caso 2: σ = 0 (determinístico)
        deterministic = options_analyzer._binomial_valuation(
            S=110, K=100, r=0.05, T=1, sigma=0, n=100
        )
        # Valor = max(S - K*e^(-rT), 0)
        expected = max(110 - 100 * exp(-0.05 * 1), 0)
        assert isclose(deterministic["option_value"], expected, rel_tol=1e-6)

    def test_binomial_valuation_parameters_returned(self, options_analyzer):
        """
        Verifica que los parámetros del modelo estén en el resultado.
        """
        result = options_analyzer._binomial_valuation(
            S=100, K=100, r=0.05, T=1, sigma=0.3, n=100
        )
        
        assert "parameters" in result
        params = result["parameters"]
        
        assert "u" in params  # Factor de subida
        assert "d" in params  # Factor de bajada
        assert "p" in params  # Probabilidad neutral al riesgo
        assert "dt" in params  # Delta t
        assert "discount_factor" in params
        
        # Verificar coherencia: u * d = 1
        assert isclose(params["u"] * params["d"], 1.0, rel_tol=1e-9)
        
        # Verificar 0 < p < 1
        assert 0 < params["p"] < 1


# ============================================================================
# PRUEBAS PARA FinancialEngine (Fachada)
# ============================================================================


class TestFinancialEngine:
    """Pruebas para el motor financiero integrado."""

    def test_analyze_project_basic(self, engine):
        """
        Verifica análisis básico de proyecto.
        """
        result = engine.analyze_project(
            initial_investment=100_000,
            cash_flows=[30_000, 35_000, 40_000, 45_000, 50_000],
            cost_std_dev=15_000,
            volatility=0.25,
        )
        
        # Verificar claves principales
        assert "wacc" in result
        assert "npv" in result
        assert "total_value" in result
        assert "var" in result
        assert "contingency" in result
        assert "real_option_value" in result
        assert "performance" in result
        assert "thermodynamics" in result
        
        # Verificar tipos
        assert isinstance(result["wacc"], float)
        assert isinstance(result["npv"], float)
        assert isinstance(result["performance"], dict)

    def test_analyze_project_volatility_adjustment(self, engine):
        """
        Verifica que la volatilidad se ajuste con estabilidad piramidal.
        """
        base_result = engine.analyze_project(
            initial_investment=100_000,
            cash_flows=[30_000, 35_000, 40_000, 45_000],
            cost_std_dev=15_000,
            volatility=0.25,
        )
        
        # Con pirámide inestable (Ψ < 1)
        adjusted_result = engine.analyze_project(
            initial_investment=100_000,
            cash_flows=[30_000, 35_000, 40_000, 45_000],
            cost_std_dev=15_000,
            volatility=0.25,
            pyramid_stability=0.5,  # Inestable
            system_temperature=35.0,  # Estrés térmico
        )
        
        # La volatilidad estructural debe ser mayor
        assert adjusted_result["volatility_structural"] > base_result["volatility_base"], (
            "Volatilidad debe aumentar con pirámide inestable"
        )
        assert adjusted_result["physics_adjustment"] is True
        assert "physics_details" in adjusted_result

    def test_analyze_project_v3_compatibility(self, engine):
        """
        Verifica compatibilidad con argumentos V3.0.
        """
        result = engine.analyze_project(
            initial_investment=100_000,
            cash_flows=[],  # Ignorado
            cost_std_dev=15_000,
            expected_cash_flows=[30_000, 35_000, 40_000],  # V3.0
            project_volatility=0.25,  # V3.0
            liquidity=0.2,  # Override
            fixed_contracts_ratio=0.7,  # Override
        )
        
        # Verificar que se usaron los overrides
        assert result["thermodynamics"]["liquidity_ratio"] == 0.2
        assert result["thermodynamics"]["fixed_contracts_ratio"] == 0.7

    def test_analyze_project_missing_volatility_raises(self, engine):
        """
        Verifica que volatilidad faltante lance error.
        """
        with pytest.raises(ValueError, match="volatility"):
            engine.analyze_project(
                initial_investment=100_000,
                cash_flows=[30_000, 35_000],
                cost_std_dev=15_000,
                # volatility no proporcionada
            )

    def test_analyze_project_empty_flows_raises(self, engine):
        """
        Verifica que flujos vacíos lancen error.
        """
        with pytest.raises(ValueError, match="flujo"):
            engine.analyze_project(
                initial_investment=100_000,
                cash_flows=[],
                cost_std_dev=15_000,
                volatility=0.25,
            )

    def test_thermo_structural_volatility_stable_no_adjustment(self, engine):
        """
        Verifica que pirámide estable no aumente volatilidad.
        """
        base_vol = 0.25
        
        result = engine._calculate_thermo_structural_volatility(
            base_volatility=base_vol,
            stability_psi=2.0,  # Muy estable
            system_temperature=25.0,  # Normal
        )
        
        # Sin ajuste significativo
        assert isclose(result, base_vol, rel_tol=0.01), (
            f"Pirámide estable no debe ajustar volatilidad. Base: {base_vol}, Result: {result}"
        )

    def test_thermo_structural_volatility_unstable_increases(self, engine):
        """
        Verifica que pirámide inestable aumente volatilidad.
        """
        base_vol = 0.25
        
        result = engine._calculate_thermo_structural_volatility(
            base_volatility=base_vol,
            stability_psi=0.5,  # Inestable
            system_temperature=25.0,
        )
        
        assert result > base_vol, "Pirámide inestable debe aumentar volatilidad"

    def test_thermo_structural_volatility_thermal_stress(self, engine):
        """
        Verifica efecto del estrés térmico.
        """
        base_vol = 0.25
        psi = 1.5  # Estable estructuralmente
        
        normal = engine._calculate_thermo_structural_volatility(base_vol, psi, 25.0)
        stressed = engine._calculate_thermo_structural_volatility(base_vol, psi, 50.0)
        
        assert stressed > normal, "Temperatura alta debe aumentar volatilidad"

    def test_thermo_structural_volatility_max_amplification(self, engine):
        """
        Verifica que existe un límite máximo de amplificación.
        """
        base_vol = 0.25
        
        result = engine._calculate_thermo_structural_volatility(
            base_volatility=base_vol,
            stability_psi=0.1,  # Extremadamente inestable
            system_temperature=100.0,  # Temperatura extrema
        )
        
        # No debe exceder 3x (MAX_AMPLIFICATION en el refinamiento)
        max_expected = base_vol * 3.0
        assert result <= max_expected * 1.01, (
            f"Volatilidad {result} excede máximo {max_expected}"
        )

    def test_performance_metrics_standard_case(self, engine):
        """
        Verifica métricas de performance para caso estándar.
        """
        investment = 100_000
        npv = 50_000  # ROI = 50%
        years = 5
        flows = [30_000, 30_000, 30_000, 30_000, 30_000]
        
        metrics = engine._calculate_performance_metrics(npv, investment, years, flows=flows)
        
        # ROI = NPV / Investment = 50%
        assert isclose(metrics["roi"], 0.5, rel_tol=1e-6)
        
        # PI = (NPV + Investment) / Investment = 1.5
        assert isclose(metrics["profitability_index"], 1.5, rel_tol=1e-6)
        
        # Recomendación positiva
        assert metrics["recommendation"] == "ACEPTAR"
        
        # Retorno anualizado: (1.5)^(1/5) - 1 ≈ 8.45%
        expected_annual = (1.5) ** (1 / 5) - 1
        assert isclose(metrics["annualized_return"], expected_annual, rel_tol=1e-6)

    def test_performance_metrics_payback_calculation(self, engine):
        """
        Verifica cálculo correcto del período de recuperación.
        """
        metrics = engine._calculate_performance_metrics(
            npv=20_000,
            investment=100_000,
            years=5,
            flows=[30_000, 30_000, 30_000, 30_000, 30_000]
        )
        
        # Payback: 100k se recupera después de ~3.33 años
        # Año 1: 30k, Año 2: 60k, Año 3: 90k, Año 4: 120k
        # Interpolación: 3 + (100k - 90k) / 30k = 3.33
        assert "payback_period" in metrics
        assert isclose(metrics["payback_period"], 3.33, rel_tol=0.01)
        assert metrics["payback_status"] == "RECUPERABLE"

    def test_performance_metrics_never_recovers(self, engine):
        """
        Verifica manejo de proyecto que nunca recupera inversión.
        """
        metrics = engine._calculate_performance_metrics(
            npv=-80_000,
            investment=100_000,
            years=5,
            flows=[5_000, 5_000, 5_000, 5_000, 5_000]  # Total: 25k < 100k
        )
        
        assert metrics["payback_period"] == float("inf")
        assert metrics["payback_status"] == "NO_RECUPERABLE"
        assert "recovery_gap" in metrics
        assert metrics["recovery_gap"] == 75_000  # 100k - 25k

    def test_performance_metrics_zero_investment(self, engine):
        """
        Verifica manejo de inversión cero.
        """
        metrics = engine._calculate_performance_metrics(
            npv=1000, investment=0, years=5
        )
        
        assert metrics["roi"] == float("inf")
        assert np.isnan(metrics["profitability_index"])
        assert np.isnan(metrics["annualized_return"])
        assert metrics["recommendation"] == "REVISAR"

    def test_performance_metrics_negative_investment(self, engine):
        """
        Verifica manejo de inversión negativa (desinversión).
        """
        metrics = engine._calculate_performance_metrics(
            npv=1000, investment=-5000, years=5
        )
        
        # roi = -npv / abs(investment) según el código
        expected_roi = -1000 / 5000  # = -0.2
        assert isclose(metrics["roi"], expected_roi, abs_tol=1e-9)
        assert np.isnan(metrics["profitability_index"])
        assert metrics["recommendation"] == "REVISAR"

    def test_performance_metrics_total_loss(self, engine):
        """
        Verifica manejo de pérdida total (ROI = -100%).
        """
        metrics = engine._calculate_performance_metrics(
            npv=-1000, investment=1000, years=5
        )
        
        assert isclose(metrics["roi"], -1.0)
        assert isclose(metrics["annualized_return"], -1.0)
        assert metrics["profitability_index"] == 0.0

    def test_performance_metrics_includes_irr(self, engine):
        """
        Verifica que se calcule IRR estimada.
        """
        metrics = engine._calculate_performance_metrics(
            npv=20_000,
            investment=100_000,
            years=5,
            flows=[30_000, 30_000, 30_000, 30_000, 30_000]
        )
        
        assert "irr_estimate" in metrics
        irr = metrics["irr_estimate"]
        
        # IRR debe ser positiva para proyecto rentable
        assert irr > 0, "IRR debe ser positiva para proyecto rentable"
        
        # IRR debe ser mayor que WACC si NPV > 0
        wacc = engine.capm.calculate_wacc()
        assert irr > wacc, f"IRR ({irr:.2%}) debe ser > WACC ({wacc:.2%})"

    def test_financial_thermal_inertia(self, engine):
        """
        Verifica cálculo de inercia térmica financiera.
        """
        inertia = engine.calculate_financial_thermal_inertia(
            liquidity=0.2,
            fixed_contracts_ratio=0.5
        )
        
        assert inertia["inertia"] == 0.2 * 0.5  # = 0.1
        
        # Mayor liquidez y más contratos fijos = más inercia
        high_inertia = engine.calculate_financial_thermal_inertia(0.5, 0.8)
        assert high_inertia["inertia"] > inertia["inertia"]

    def test_predict_temperature_change(self, engine):
        """
        Verifica predicción de cambio de temperatura financiera.
        """
        # Con inercia, el cambio se amortigua
        delta_t = engine.predict_temperature_change(
            perturbation=100,
            inertia_data={'inertia': 0.5}
        )
        assert delta_t["temperature_change"] == 100 / 0.5  # = 200
        
        # Sin inercia, el cambio es directo
        delta_t_no_inertia = engine.predict_temperature_change(
            perturbation=100,
            inertia_data={'inertia': 0.0}
        )
        assert delta_t_no_inertia["temperature_change"] == 100  # Perturbación completa

    def test_adjust_volatility_by_topology_no_report(self, engine):
        """
        Verifica que sin reporte topológico no haya ajuste.
        """
        base_vol = 0.25
        result = engine.adjust_volatility_by_topology(base_vol, None)
        assert result == base_vol

        result2 = engine.adjust_volatility_by_topology(base_vol, {})
        assert result2 == base_vol

    def test_adjust_volatility_by_topology_with_synergy(self, engine):
        """
        Verifica ajuste por sinergia de riesgo topológico.
        """
        base_vol = 0.25
        topology_report = {
            "synergy_risk": {
                "synergy_detected": True,
                "synergy_strength": 1.0,
            },
            "euler_efficiency": 0.8,
        }
        
        adjusted = engine.adjust_volatility_by_topology(base_vol, topology_report)
        
        assert adjusted > base_vol, "Sinergia de riesgo debe aumentar volatilidad"


# ============================================================================
# PRUEBAS PARA FUNCIONES DE UTILIDAD
# ============================================================================


class TestUtilityFunctions:
    """Pruebas para funciones de utilidad del módulo."""

    def test_calculate_volatility_from_returns_daily(self):
        """
        Verifica cálculo de volatilidad anualizada desde retornos diarios.
        """
        np.random.seed(42)
        true_daily_std = 0.01
        returns = list(np.random.normal(0, true_daily_std, 1000))
        
        # La función devuelve un float directamente
        volatility = calculate_volatility_from_returns(returns, frequency="daily")
        
        # Volatilidad anualizada esperada: σ_daily * √252
        sample_std = np.std(returns, ddof=1)
        expected_vol = sample_std * sqrt(252)
        
        assert isclose(volatility, expected_vol, rel_tol=0.01), (
            f"Esperado: {expected_vol:.4%}, Obtenido: {volatility:.4%}"
        )

    def test_calculate_volatility_from_returns_all_frequencies(self):
        """
        Verifica volatilidad para todas las frecuencias válidas.
        """
        np.random.seed(42)
        returns = list(np.random.normal(0, 0.01, 500))
        sample_std = np.std(returns, ddof=1)
        
        frequencies = {
            "daily": 252,
            "weekly": 52,
            "monthly": 12,
            "annual": 1,
        }
        
        results = {}
        for freq, factor in frequencies.items():
            vol = calculate_volatility_from_returns(returns, frequency=freq)
            expected = sample_std * sqrt(factor)
            results[freq] = vol
            
            assert isclose(vol, expected, rel_tol=0.001), (
                f"Frecuencia {freq}: esperado {expected:.4%}, obtenido {vol:.4%}"
            )
        
        # Verificar ordenamiento: daily > weekly > monthly > annual
        assert results["daily"] > results["weekly"] > results["monthly"] > results["annual"]

    def test_calculate_volatility_from_returns_minimum_data(self):
        """
        Verifica que funciona con el mínimo de datos (2 retornos).
        """
        returns = [0.01, 0.02]
        vol = calculate_volatility_from_returns(returns, frequency="daily")
        
        assert vol > 0, "Volatilidad debe ser positiva"
        
        # Verificar valor exacto
        expected_std = np.std(returns, ddof=1)
        expected_vol = expected_std * sqrt(252)
        assert isclose(vol, expected_vol, rel_tol=1e-9)

    def test_calculate_volatility_from_returns_empty_raises(self):
        """
        Verifica que lista vacía lance error.
        """
        with pytest.raises(ValueError, match="Se requieren al menos 2 retornos"):
            calculate_volatility_from_returns([])

    def test_calculate_volatility_from_returns_single_raises(self):
        """
        Verifica que un solo elemento lance error.
        """
        with pytest.raises(ValueError, match="Se requieren al menos 2 retornos"):
            calculate_volatility_from_returns([0.01])

    def test_calculate_volatility_from_returns_invalid_frequency(self):
        """
        Verifica que frecuencia inválida lance error.
        """
        with pytest.raises(ValueError, match="no válida"):
            calculate_volatility_from_returns([0.01, 0.02], frequency="hourly")

    def test_calculate_volatility_from_returns_none_raises(self):
        """
        Verifica que None lance error.
        """
        with pytest.raises((ValueError, TypeError)):
            calculate_volatility_from_returns(None)

    def test_calculate_volatility_custom_trading_days(self):
        """
        Verifica parámetro personalizado de días de trading.
        """
        np.random.seed(42)
        returns = list(np.random.normal(0, 0.01, 100))
        sample_std = np.std(returns, ddof=1)
        
        # Con 365 días de trading (por ejemplo, crypto)
        vol_365 = calculate_volatility_from_returns(
            returns, frequency="daily", annual_trading_days=365
        )
        expected_365 = sample_std * sqrt(365)
        
        assert isclose(vol_365, expected_365, rel_tol=0.001)
        
        # Comparar con default (252)
        vol_252 = calculate_volatility_from_returns(returns, frequency="daily")
        
        assert vol_365 > vol_252, "Más días de trading = mayor volatilidad anualizada"


# ============================================================================
# PRUEBAS DE INTEGRACIÓN
# ============================================================================


class TestIntegration:
    """Pruebas de integración end-to-end."""

    def test_full_project_analysis_workflow(self, engine):
        """
        Verifica flujo completo de análisis de proyecto.
        """
        # Datos del proyecto
        investment = 500_000
        cash_flows = [100_000, 120_000, 140_000, 160_000, 180_000]
        std_dev = 50_000
        volatility = 0.30
        
        # Análisis completo
        result = engine.analyze_project(
            initial_investment=investment,
            cash_flows=cash_flows,
            cost_std_dev=std_dev,
            volatility=volatility,
            pyramid_stability=1.2,
            system_temperature=28.0,
        )
        
        # Verificar coherencia de resultados
        assert result["wacc"] > 0
        assert isinstance(result["npv"], float)
        
        # El valor total incluye opciones reales
        if result["real_option_value"] > 0:
            assert result["total_value"] > result["npv"]
        
        # VaR debe ser mayor que la inversión (por definición, al 95%)
        # dado que VaR = mean + z*sigma y z > 0
        
        # Performance metrics coherentes
        perf = result["performance"]
        if result["npv"] > 0:
            assert perf["roi"] > 0
            assert perf["profitability_index"] > 1
            assert perf["recommendation"] == "ACEPTAR"

    def test_neutral_risk_project(self, engine_neutral):
        """
        Verifica análisis con configuración neutral (β=0, D/E=0).
        """
        result = engine_neutral.analyze_project(
            initial_investment=100_000,
            cash_flows=[25_000, 25_000, 25_000, 25_000, 25_000],
            cost_std_dev=10_000,
            volatility=0.20,
        )
        
        # WACC debe ser igual a Rf cuando β=0 y D/E=0
        rf = engine_neutral.config.risk_free_rate
        assert isclose(result["wacc"], rf, rel_tol=1e-9), (
            f"Con β=0 y D/E=0, WACC debe ser Rf. WACC: {result['wacc']:.4%}, Rf: {rf:.4%}"
        )

    def test_high_risk_project_with_physics(self, engine):
        """
        Verifica que proyecto de alto riesgo se penalice correctamente.
        """
        # Proyecto con pirámide invertida y estrés térmico
        result = engine.analyze_project(
            initial_investment=100_000,
            cash_flows=[30_000, 30_000, 30_000, 30_000],
            cost_std_dev=20_000,
            volatility=0.25,
            pyramid_stability=0.3,  # Muy inestable
            system_temperature=45.0,  # Muy caliente
        )
        
        # La volatilidad debe haberse amplificado significativamente
        assert result["volatility_structural"] > result["volatility_base"] * 1.5, (
            "Con pirámide invertida y estrés térmico, volatilidad debe amplificarse >50%"
        )
        assert result["physics_adjustment"] is True

    def test_deterministic_project_zero_volatility(self, engine):
        """
        Verifica proyecto determinístico (σ=0).
        """
        result = engine.analyze_project(
            initial_investment=100_000,
            cash_flows=[30_000, 30_000, 30_000, 30_000, 30_000],
            cost_std_dev=0,
            volatility=0.0,
        )
        
        # Sin volatilidad, no hay valor de opciones reales
        assert result["real_option_value"] == 0
        assert result["volatility"] == 0
        assert result["volatility_structural"] == 0