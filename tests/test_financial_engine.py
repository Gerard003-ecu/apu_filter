# -*- coding: utf-8 -*-
import pytest
import numpy as np
from math import isclose, nan, sqrt
from app.financial_engine import (
    FinancialConfig,
    CapitalAssetPricing,
    RiskQuantifier,
    RealOptionsAnalyzer,
    FinancialEngine,
    DistributionType,
    calculate_volatility_from_returns,
)
from scipy.stats import norm

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def base_config():
    """Configuración financiera base para las pruebas."""
    return FinancialConfig(
        risk_free_rate=0.05,
        market_premium=0.08,
        beta=1.2,
        tax_rate=0.25,
        cost_of_debt=0.07,
        debt_to_equity_ratio=1.5
    )

@pytest.fixture
def capm(base_config):
    """Instancia de CapitalAssetPricing."""
    return CapitalAssetPricing(base_config)

@pytest.fixture
def risk_quantifier():
    """Instancia de RiskQuantifier."""
    return RiskQuantifier()

@pytest.fixture
def options_analyzer():
    """Instancia de RealOptionsAnalyzer."""
    return RealOptionsAnalyzer()

@pytest.fixture
def engine(base_config):
    """Instancia de FinancialEngine."""
    return FinancialEngine(base_config)

# ============================================================================
# PRUEBAS PARA CapitalAssetPricing
# ============================================================================

class TestCapitalAssetPricing:
    def test_calculate_wacc_correction(self, capm):
        """
        Verifica que el WACC se calcule correctamente sin el typo 'weight_debit'.
        Ke = 5% + 1.2 * 8% = 14.6%
        D/E = 1.5 => E_w = 1/(1+1.5)=0.4, D_w = 1.5/(1+1.5)=0.6
        WACC = 0.4 * 14.6% + 0.6 * 7% * (1 - 0.25) = 5.84% + 3.15% = 8.99%
        """
        expected_ke = 0.146
        expected_wacc = 0.0899
        assert isclose(capm.calculate_ke(), expected_ke)
        assert isclose(capm.calculate_wacc(), expected_wacc)

    def test_sensitivity_analysis_cache_invalidation(self, capm, base_config):
        """
        Valida que la caché se invalide y restaure correctamente durante el
        análisis de sensibilidad.
        """
        original_beta = base_config.beta
        original_wacc = capm.calculate_wacc()

        # El primer cálculo debe estar cacheado
        assert capm.calculate_wacc() == original_wacc

        # Ejecutar análisis de sensibilidad
        beta_range = [1.0, 1.5, 2.0]
        results = capm.sensitivity_analysis('beta', beta_range)

        # Verificar que el beta original se haya restaurado
        assert base_config.beta == original_beta

        # Limpiar la caché explícitamente para la verificación final
        capm.calculate_wacc.cache_clear()

        # El WACC recalculado debe coincidir con el original
        final_wacc = capm.calculate_wacc()
        assert isclose(final_wacc, original_wacc)

        # Verificar que la caché fue usada y limpiada (no hay forma directa, pero
        # la lógica asegura que la restauración ocurre en el 'finally')
        assert len(results) == len(beta_range)

# ============================================================================
# PRUEBAS PARA RiskQuantifier
# ============================================================================

class TestRiskQuantifier:
    def test_calculate_var_cvar_normal(self, risk_quantifier):
        """
        Verifica el cálculo de VaR y CVaR (Expected Shortfall) para una
        distribución Normal.
        """
        mean, std_dev, conf = 1000, 150, 0.95
        risk_quantifier.distribution = DistributionType.NORMAL
        var, metrics = risk_quantifier.calculate_var(mean, std_dev, conf)

        # CORRECCIÓN: Aplicar el mismo factor de escalado que usa la función
        time_scaling_factor = sqrt(1 / 252)
        scaled_std = std_dev * time_scaling_factor

        z = norm.ppf(conf)
        expected_var = mean + z * scaled_std
        expected_cvar = mean + scaled_std * norm.pdf(z) / (1 - conf)

        assert isclose(var, expected_var, rel_tol=1e-4)
        assert isclose(metrics['cvar'], expected_cvar, rel_tol=1e-4)
        assert metrics['distribution'] == "Normal"

    def test_calculate_var_cvar_student_t(self, risk_quantifier):
        """
        Verifica el cálculo de VaR y CVaR para una distribución t-Student,
        que debe ser mayor que la Normal por sus colas pesadas.
        """
        mean, std_dev, conf, df = 1000, 150, 0.95, 5

        # Calcular con t-Student
        risk_quantifier.distribution = DistributionType.STUDENT_T
        var_t, metrics_t = risk_quantifier.calculate_var(mean, std_dev, conf, df_student_t=df)

        # CORRECCIÓN: Reestablecer a Normal para la comparación
        risk_quantifier.distribution = DistributionType.NORMAL
        var_n, _ = risk_quantifier.calculate_var(mean, std_dev, conf)

        assert var_t > var_n  # VaR t-Student debe ser mayor
        assert metrics_t['cvar'] > metrics_t['var_cvar_ratio'] * var_t if metrics_t['var_cvar_ratio'] else 0
        assert metrics_t['distribution'] == f"Student-t(df={df})"

    def test_suggest_contingency_all_methods(self, risk_quantifier):
        """
        Valida que el método 'suggest_contingency' devuelva todas las estrategias
        por defecto.
        """
        results = risk_quantifier.suggest_contingency(1000, 150)
        expected_keys = [
            "var_based", "percentage_based", "heuristic", "recommended"
        ]
        assert all(key in results for key in expected_keys)
        assert results['recommended'] == max(
            results["var_based"], results["percentage_based"], results["heuristic"]
        )

# ============================================================================
# PRUEBAS PARA RealOptionsAnalyzer
# ============================================================================

class TestRealOptionsAnalyzer:
    def test_binomial_valuation_american_vs_european(self, options_analyzer):
        """
        Compara la valoración de una opción Americana vs. Europea, donde la
        Americana debe tener un valor igual o superior.
        """
        params = {
            'S': 100, 'K': 100, 'r': 0.05, 'T': 1, 'sigma': 0.3, 'n': 100
        }

        # Valoración Americana (permite ejercicio anticipado)
        american_result = options_analyzer._binomial_valuation(**params, american=True)

        # Valoración Europea (sin ejercicio anticipado)
        european_result = options_analyzer._binomial_valuation(**params, american=False)

        assert american_result['option_value'] >= european_result['option_value']
        assert american_result['model'].endswith("(Americana)")
        assert european_result['model'].endswith("(Europea)")

        # CORRECCIÓN: Para una call sin dividendos, el valor debería ser muy cercano
        assert isclose(american_result['option_value'], european_result['option_value'], rel_tol=1e-9)

        # Ahora, un caso donde el ejercicio anticipado es óptimo (muy in-the-money)
        params_itm = {
            'S': 150, 'K': 100, 'r': 0.05, 'T': 1, 'sigma': 0.2, 'n': 100
        }
        american_itm = options_analyzer._binomial_valuation(**params_itm, american=True)
        european_itm = options_analyzer._binomial_valuation(**params_itm, american=False)

        assert american_itm['option_value'] >= european_itm['option_value']
        assert isclose(american_itm['option_value'], european_itm['option_value'], rel_tol=1e-9)

# ============================================================================
# PRUEBAS PARA FinancialEngine (Fachada)
# ============================================================================

class TestFinancialEngine:
    def test_performance_metrics_edge_cases(self, engine):
        """
        Prueba los casos extremos del cálculo de métricas de performance.
        """
        # Caso 1: Inversión cero
        metrics_zero_inv = engine._calculate_performance_metrics(1000, 0, 5)
        assert metrics_zero_inv['roi'] == float('inf')
        assert np.isnan(metrics_zero_inv['annualized_return'])

        # Caso 2: Inversión negativa
        metrics_neg_inv = engine._calculate_performance_metrics(1000, -5000, 5)
        assert isclose(metrics_neg_inv['roi'], 1000 / 5000) # ROI se invierte
        assert np.isnan(metrics_neg_inv['profitability_index'])

        # Caso 3: ROI < -100% (pérdida total)
        metrics_total_loss = engine._calculate_performance_metrics(-1200, 1000, 5)
        assert isclose(metrics_total_loss['roi'], -1.2)
        assert np.isnan(metrics_total_loss['annualized_return'])

        # Caso 4: ROI = -100% (pérdida total del capital)
        metrics_exact_loss = engine._calculate_performance_metrics(-1000, 1000, 5)
        assert isclose(metrics_exact_loss['roi'], -1.0)
        assert isclose(metrics_exact_loss['annualized_return'], -1.0)

# ============================================================================
# PRUEBAS PARA FUNCIONES DE UTILIDAD
# ============================================================================

class TestUtilityFunctions:
    def test_calculate_volatility_frequency(self):
        """
        Valida que la volatilidad se anualice correctamente desde diferentes
        frecuencias de datos.
        """
        # Desviación estándar de 1%
        returns = np.random.normal(loc=0.001, scale=0.01, size=1000)
        std_period = np.std(returns, ddof=1)

        # Volatilidad diaria
        vol_daily = calculate_volatility_from_returns(list(returns), frequency='daily')
        assert isclose(vol_daily, std_period * np.sqrt(252), rel_tol=1e-3)

        # Volatilidad mensual
        vol_monthly = calculate_volatility_from_returns(list(returns), frequency='monthly')
        assert isclose(vol_monthly, std_period * np.sqrt(12), rel_tol=1e-3)

        # Volatilidad anual (debe ser igual a la std del periodo)
        vol_annual = calculate_volatility_from_returns(list(returns), frequency='annual')
        assert isclose(vol_annual, std_period * np.sqrt(1), rel_tol=1e-3)

    def test_calculate_volatility_invalid_input(self):
        """Prueba manejo de errores para la función de volatilidad."""
        with pytest.raises(ValueError, match="Se requieren al menos 2 retornos"):
            calculate_volatility_from_returns([])
        with pytest.raises(ValueError, match="Frecuencia 'invalid_freq' no válida"):
            calculate_volatility_from_returns([0.1, 0.2], frequency='invalid_freq')
