# -*- coding: utf-8 -*-
"""
Motor de Inteligencia Financiera para la Valoración de Proyectos de Construcción.

Este módulo integra principios de finanzas corporativas (CAPM, WACC), cuantificación
de riesgos (VaR) y valoración de opciones reales (Black-Scholes) para enriquecer
las estimaciones de costos técnicos con una capa de viabilidad económica.

MEJORAS IMPLEMENTADAS:
1. Validación robusta de parámetros financieros
2. Cálculo de VaR con distribuciones alternativas (Normal, Student-t)
3. Griegas completas para opciones reales
4. Modelo binomial como alternativa a Black-Scholes
5. Análisis de sensibilidad paramétrica
6. Caché para cálculos repetitivos
7. Métricas de performance integradas
"""

import logging
from dataclasses import dataclass, field
from math import exp, log, sqrt, pow
from typing import Optional, Tuple, Dict, List, Callable
from enum import Enum
from functools import lru_cache

import numpy as np
from scipy.stats import norm, t

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERACIONES Y TIPOS
# ============================================================================


class DistributionType(Enum):
    """Tipos de distribuciones para cálculo de riesgo."""
    NORMAL = "normal"
    STUDENT_T = "student_t"


class OptionModelType(Enum):
    """Modelos disponibles para valoración de opciones."""
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"


# ============================================================================
# CLASE DE CONFIGURACIÓN FINANCIERA (MEJORADA)
# ============================================================================


@dataclass
class FinancialConfig:
    """
    Almacena parámetros macroeconómicos y de proyecto con validación robusta.

    MEJORAS:
    1. Validación de rangos realistas
    2. Valores por defecto basados en datos históricos del sector
    3. Post-inicialización para coherencia de parámetros
    """

    risk_free_rate: float = 0.04
    market_premium: float = 0.06
    beta: float = 1.2
    tax_rate: float = 0.30
    cost_of_debt: float = 0.08
    debt_to_equity_ratio: float = 0.6
    project_life_years: int = 10  # Nueva: vida útil del proyecto

    def __post_init__(self):
        """Valida la coherencia de los parámetros financieros."""
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Valida que todos los parámetros estén en rangos realistas."""
        validations = [
            (self.risk_free_rate, 0.0, 0.15, "Tasa libre de riesgo"),
            (self.market_premium, 0.01, 0.20, "Prima de riesgo de mercado"),
            (self.beta, 0.1, 5.0, "Beta"),
            (self.tax_rate, 0.0, 0.50, "Tasa impositiva"),
            (self.cost_of_debt, 0.01, 0.30, "Costo de la deuda"),
            (self.debt_to_equity_ratio, 0.0, 5.0, "Razón Deuda/Capital"),
            (self.project_life_years, 1, 50, "Vida del proyecto")
        ]
        
        for value, min_val, max_val, name in validations:
            if not (min_val <= value <= max_val):
                logger.warning(
                    f"{name} ({value}) fuera de rango típico [{min_val}, {max_val}]"
                )


# ============================================================================
# MOTOR DE VALORACIÓN DE ACTIVOS (CAPM Y WACC - MEJORADO)
# ============================================================================


class CapitalAssetPricing:
    """
    Calcula el costo de capital utilizando el Modelo de Valoración de Activos de Capital
    (CAPM) y el Costo Promedio Ponderado de Capital (WACC).

    MEJORAS:
    1. Caché para cálculos repetitivos
    2. Métricas de sensibilidad
    3. Cálculo de VAN usando WACC
    """

    def __init__(self, config: FinancialConfig):
        if not isinstance(config, FinancialConfig):
            raise TypeError("El parámetro 'config' debe ser una instancia de FinancialConfig.")
        self.config = config
        self._ke_cache = None
        self._wacc_cache = None

    @lru_cache(maxsize=1)
    def calculate_ke(self) -> float:
        """
        Calcula el Costo del Equity (Ke) usando el modelo CAPM.
        
        Returns:
            El costo del equity como un valor decimal.
        """
        try:
            if self.config.beta < 0.1:
                logger.warning(f"Beta muy bajo ({self.config.beta}), riesgo subestimado")
            
            ke = self.config.risk_free_rate + self.config.beta * self.config.market_premium
            logger.info(f"Costo del Equity (Ke) calculado: {ke:.2%}")
            return ke
        except Exception as e:
            logger.error(f"Error calculando Ke: {e}")
            raise ValueError(f"Error en cálculo de Ke: {e}")

    @lru_cache(maxsize=1)
    def calculate_wacc(self) -> float:
        """
        Calcula el Costo Promedio Ponderado de Capital (WACC).
        
        Returns:
            El WACC como un valor decimal.
        """
        try:
            if self.config.debt_to_equity_ratio < 0:
                raise ValueError("La razón D/E no puede ser negativa")
            
            ke = self.calculate_ke()
            d_e_ratio = self.config.debt_to_equity_ratio
            
            # Calcular pesos
            weight_equity = 1 / (1 + d_e_ratio)
            weight_debt = d_e_ratio / (1 + d_e_ratio)
            
            # Calcular WACC con validación
            if weight_equity + weight_debit != 1.0:
                logger.warning(f"Pesos no suman 1: Equity={weight_equity:.3f}, Deuda={weight_debt:.3f}")
            
            wacc = (weight_equity * ke) + (
                weight_debt * self.config.cost_of_debt * (1 - self.config.tax_rate)
            )
            
            logger.info(f"WACC calculado: {wacc:.2%}")
            return wacc
        except ZeroDivisionError:
            logger.error("La razón D/E resultó en división por cero")
            raise
        except Exception as e:
            logger.error(f"Error calculando WACC: {e}")
            raise

    def calculate_npv(self, cash_flows: List[float], initial_investment: float = 0) -> float:
        """
        Calcula el Valor Presente Neto usando el WACC como tasa de descuento.
        
        Args:
            cash_flows: Lista de flujos de caja futuros
            initial_investment: Inversión inicial (negativa)
            
        Returns:
            El VAN del proyecto
        """
        try:
            wacc = self.calculate_wacc()
            npv = -initial_investment
            
            for i, cf in enumerate(cash_flows, 1):
                npv += cf / pow(1 + wacc, i)
            
            logger.info(f"VAN calculado: ${npv:,.2f} con WACC={wacc:.2%}")
            return npv
        except Exception as e:
            logger.error(f"Error calculando VAN: {e}")
            raise

    def sensitivity_analysis(self, parameter: str, range_values: List[float]) -> Dict[float, float]:
        """
        Análisis de sensibilidad del WACC a cambios en parámetros.
        
        Args:
            parameter: 'beta', 'cost_of_debt', 'debt_to_equity_ratio'
            range_values: Valores a evaluar
            
        Returns:
            Diccionario con valores del parámetro y WACC resultante
        """
        results = {}
        original_value = getattr(self.config, parameter)
        
        for value in range_values:
            setattr(self.config, parameter, value)
            # Invalidar caché
            self._ke_cache = None
            self._wacc_cache = None
            results[value] = self.calculate_wacc()
        
        # Restaurar valor original
        setattr(self.config, parameter, original_value)
        self._ke_cache = None
        self._wacc_cache = None
        
        return results


# ============================================================================
# CUANTIFICADOR DE RIESGOS (MEJORADO)
# ============================================================================


class RiskQuantifier:
    """
    Proporciona herramientas para medir el riesgo financiero.

    MEJORAS:
    1. Múltiples distribuciones (Normal, Student-t)
    2. VaR condicional (CVaR/Expected Shortfall)
    3. Escalado temporal
    4. Validación de parámetros robusta
    """

    def __init__(self, distribution: DistributionType = DistributionType.NORMAL):
        self.distribution = distribution

    def calculate_var(
        self,
        mean: float,
        std_dev: float,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1,
        df_student_t: int = 5  # Grados de libertad para Student-t
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el Valor en Riesgo (VaR) con múltiples distribuciones.
        
        Args:
            mean: Media de la distribución
            std_dev: Desviación estándar
            confidence_level: Nivel de confianza
            time_horizon_days: Horizonte temporal (escalado sqrt(T))
            df_student_t: Grados de libertad para distribución t-Student
            
        Returns:
            Tupla con (VaR, métricas adicionales)
        """
        # Validación robusta
        if std_dev < 0:
            raise ValueError("La desviación estándar no puede ser negativa")
        if not 0 < confidence_level < 1:
            raise ValueError(f"Nivel de confianza inválido: {confidence_level}")
        if time_horizon_days <= 0:
            raise ValueError(f"Horizonte temporal inválido: {time_horizon_days}")
        
        try:
            # Escalado temporal (sqrt(T) rule)
            scaled_std = std_dev * sqrt(time_horizon_days / 252)  # Asumiendo 252 días hábiles
            
            if self.distribution == DistributionType.NORMAL:
                z_score = norm.ppf(confidence_level)
                distribution_name = "Normal"
            elif self.distribution == DistributionType.STUDENT_T:
                z_score = t.ppf(confidence_level, df_student_t)
                distribution_name = f"Student-t(df={df_student_t})"
            else:
                raise ValueError(f"Distribución no soportada: {self.distribution}")
            
            var = mean + z_score * scaled_std
            
            # Calcular CVaR (Expected Shortfall)
            if self.distribution == DistributionType.NORMAL:
                cvar = mean - scaled_std * norm.pdf(z_score) / (1 - confidence_level)
            else:
                # Aproximación para t-Student
                cvar = mean - scaled_std * t.pdf(z_score, df_student_t) / (1 - confidence_level)
            
            metrics = {
                "distribution": distribution_name,
                "z_score": z_score,
                "scaled_std": scaled_std,
                "cvar": cvar,
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon_days
            }
            
            logger.info(
                f"VaR({confidence_level:.1%}, {time_horizon_days}d)={var:,.2f}, "
                f"CVaR={cvar:,.2f} usando {distribution_name}"
            )
            
            return var, metrics
            
        except Exception as e:
            logger.error(f"Error calculando VaR: {e}")
            raise

    def calculate_historical_var(
        self,
        returns: List[float],
        confidence_level: float = 0.95,
        window: int = None
    ) -> float:
        """
        Calcula VaR histórico (no paramétrico).
        
        Args:
            returns: Serie histórica de retornos
            confidence_level: Nivel de confianza
            window: Ventana móvil (opcional)
            
        Returns:
            VaR histórico
        """
        if not returns:
            raise ValueError("Lista de retornos vacía")
        
        try:
            if window and window < len(returns):
                # Usar ventana móvil
                recent_returns = returns[-window:]
            else:
                recent_returns = returns
            
            var = -np.percentile(recent_returns, (1 - confidence_level) * 100)
            logger.info(f"VaR histórico ({confidence_level:.1%}) = {var:.2%}")
            return var
            
        except Exception as e:
            logger.error(f"Error calculando VaR histórico: {e}")
            raise

    def suggest_contingency(
        self,
        base_cost: float,
        std_dev: float,
        confidence_level: float = 0.90,
        method: str = "var"
    ) -> Dict[str, float]:
        """
        Sugiere contingencia usando múltiples métodos.
        
        Args:
            base_cost: Costo base estimado
            std_dev: Desviación estándar del costo
            confidence_level: Nivel de confianza deseado
            method: 'var', 'percentage', o 'heuristic'
            
        Returns:
            Diccionario con diferentes estimaciones de contingencia
        """
        contingencies = {}
        
        if method == "var" or method == "all":
            var, _ = self.calculate_var(base_cost, std_dev, confidence_level)
            contingencies["var_based"] = max(0, var - base_cost)
        
        if method == "percentage" or method == "all":
            # Método porcentual estándar (10-20% para construcción)
            percentage = 0.15 if std_dev/base_cost > 0.1 else 0.10
            contingencies["percentage_based"] = base_cost * percentage
        
        if method == "heuristic" or method == "all":
            # Método heurístico: múltiplo de desviación estándar
            multiplier = 1.5 if std_dev/base_cost > 0.15 else 1.0
            contingencies["heuristic"] = multiplier * std_dev
        
        # Recomendación final (máximo de los métodos)
        if contingencies:
            contingencies["recommended"] = max(contingencies.values())
            logger.info(f"Contingencia recomendada: ${contingencies['recommended']:,.2f}")
        
        return contingencies


# ============================================================================
# ANALIZADOR DE OPCIONES REALES (MEJORADO)
# ============================================================================


class RealOptionsAnalyzer:
    """
    Valora la flexibilidad estratégica en proyectos de inversión.

    MEJORAS:
    1. Múltiples modelos (Black-Scholes, Binomial)
    2. Cálculo de griegas completas
    3. Análisis de sensibilidad
    4. Validación de parámetros robusta
    """

    def __init__(self, model_type: OptionModelType = OptionModelType.BLACK_SCHOLES):
        self.model_type = model_type

    def _validate_option_parameters(
        self,
        project_value: float,
        investment_cost: float,
        risk_free_rate: float,
        time_to_expire_years: float,
        volatility: float
    ) -> None:
        """Valida parámetros para modelos de opciones."""
        if project_value <= 0:
            raise ValueError(f"Valor del proyecto debe ser positivo: {project_value}")
        if investment_cost <= 0:
            raise ValueError(f"Costo de inversión debe ser positivo: {investment_cost}")
        if risk_free_rate < 0:
            raise ValueError(f"Tasa libre de riesgo no puede ser negativa: {risk_free_rate}")
        if time_to_expire_years <= 0:
            raise ValueError(f"Tiempo a expiración debe ser positivo: {time_to_expire_years}")
        if volatility <= 0:
            raise ValueError(f"Volatilidad debe ser positiva: {volatility}")

    def _calculate_black_scholes_greeks(
        self,
        S: float,  # Valor proyecto
        K: float,  # Costo inversión
        T: float,  # Tiempo
        r: float,  # Tasa libre riesgo
        sigma: float  # Volatilidad
    ) -> Dict[str, float]:
        """Calcula las griegas para el modelo Black-Scholes."""
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        # CDF y PDF de distribución normal
        N_d1 = norm.cdf(d1)
        n_d1 = norm.pdf(d1)
        
        greeks = {
            'delta': N_d1,  # Sensibilidad al precio del subyacente
            'gamma': n_d1 / (S * sigma * sqrt(T)),  # Sensibilidad del delta
            'vega': S * n_d1 * sqrt(T),  # Sensibilidad a la volatilidad
            'theta': -(S * n_d1 * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2),
            'rho': K * T * exp(-r * T) * norm.cdf(d2)  # Sensibilidad a tasa de interés
        }
        
        return greeks

    def value_option_to_wait(
        self,
        project_value: float,
        investment_cost: float,
        risk_free_rate: float,
        time_to_expire_years: float,
        volatility: float,
        dividend_yield: float = 0.0,  # Rendimiento por dividendos/beneficios tempranos
        steps: int = 100  # Pasos para modelo binomial
    ) -> Dict[str, float]:
        """
        Valora la "Opción de Esperar" usando múltiples modelos.
        
        Returns:
            Diccionario con valor de opción y métricas adicionales
        """
        try:
            self._validate_option_parameters(
                project_value, investment_cost, risk_free_rate,
                time_to_expire_years, volatility
            )
            
            if self.model_type == OptionModelType.BLACK_SCHOLES:
                return self._black_scholes_valuation(
                    project_value, investment_cost, risk_free_rate,
                    time_to_expire_years, volatility, dividend_yield
                )
            else:
                return self._binomial_valuation(
                    project_value, investment_cost, risk_free_rate,
                    time_to_expire_years, volatility, steps
                )
                
        except Exception as e:
            logger.error(f"Error en valoración de opción real: {e}")
            raise

    def _black_scholes_valuation(
        self,
        S: float, K: float, r: float, T: float, sigma: float, q: float = 0.0
    ) -> Dict[str, float]:
        """Valoración usando modelo Black-Scholes-Merton."""
        d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        # Valor de opción call europea con dividendos
        option_value = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        
        # Calcular griegas
        greeks = self._calculate_black_scholes_greeks(S, K, T, r, sigma)
        
        # Métricas adicionales
        intrinsic_value = max(S - K, 0)
        time_value = max(0, option_value - intrinsic_value)
        
        results = {
            'option_value': option_value,
            'intrinsic_value': intrinsic_value,
            'time_value': time_value,
            'model': 'Black-Scholes-Merton',
            'greeks': greeks,
            'moneyness': S / K,  # Razón valor/costo
            'break_even': K * exp(r * T)  # Punto de equilibrio
        }
        
        logger.info(f"Opción de esperar (BSM): ${option_value:,.2f}")
        logger.info(f"  Valor intrínseco: ${intrinsic_value:,.2f}")
        logger.info(f"  Valor tiempo: ${time_value:,.2f}")
        
        return results

    def _binomial_valuation(
        self,
        S: float, K: float, r: float, T: float, sigma: float, n: int = 100
    ) -> Dict[str, float]:
        """Valoración usando modelo binomial (Cox-Ross-Rubinstein)."""
        dt = T / n
        u = exp(sigma * sqrt(dt))  # Movimiento ascendente
        d = 1 / u  # Movimiento descendente
        p = (exp(r * dt) - d) / (u - d)  # Probabilidad neutral al riesgo
        
        # Árbol binomial para precios del subyacente
        prices = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(i + 1):
                prices[j, i] = S * (u ** (i - j)) * (d ** j)
        
        # Valoración backward induction
        values = np.zeros((n + 1, n + 1))
        values[:, n] = np.maximum(prices[:, n] - K, 0)
        
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                values[j, i] = exp(-r * dt) * (
                    p * values[j, i + 1] + (1 - p) * values[j + 1, i + 1]
                )
        
        option_value = values[0, 0]
        intrinsic_value = max(S - K, 0)
        
        results = {
            'option_value': option_value,
            'intrinsic_value': intrinsic_value,
            'time_value': option_value - intrinsic_value,
            'model': 'Binomial (CRR)',
            'steps': n,
            'u': u,
            'd': d,
            'p': p,
            'moneyness': S / K
        }
        
        logger.info(f"Opción de esperar (Binomial {n} pasos): ${option_value:,.2f}")
        
        return results

    def analyze_sensitivity(
        self,
        base_params: Dict[str, float],
        variable_param: str,
        range_values: List[float]
    ) -> Dict[float, Dict[str, float]]:
        """
        Análisis de sensibilidad del valor de la opción.
        
        Args:
            base_params: Parámetros base
            variable_param: Parámetro a variar
            range_values: Valores a evaluar
            
        Returns:
            Resultados para cada valor del parámetro
        """
        results = {}
        
        for value in range_values:
            params = base_params.copy()
            params[variable_param] = value
            
            try:
                result = self.value_option_to_wait(**params)
                results[value] = {
                    'option_value': result['option_value'],
                    'moneyness': result.get('moneyness', 0)
                }
            except Exception as e:
                logger.warning(f"Error con {variable_param}={value}: {e}")
                results[value] = {'error': str(e)}
        
        return results


# ============================================================================
# FACHADA PRINCIPAL (NUEVO)
# ============================================================================


class FinancialEngine:
    """
    Fachada principal que integra todos los componentes del motor financiero.
    
    MEJORAS:
    1. Interfaz unificada para todos los componentes
    2. Cálculo de métricas de performance
    3. Reportes integrados
    """
    
    def __init__(self, config: FinancialConfig):
        self.config = config
        self.capm_engine = CapitalAssetPricing(config)
        self.risk_quantifier = RiskQuantifier(DistributionType.NORMAL)
        self.options_analyzer = RealOptionsAnalyzer(OptionModelType.BLACK_SCHOLES)
        
    def analyze_project(
        self,
        initial_investment: float,
        expected_cash_flows: List[float],
        cost_std_dev: float,
        project_volatility: float
    ) -> Dict[str, any]:
        """
        Análisis completo de viabilidad de proyecto.
        
        Returns:
            Diccionario con todas las métricas de análisis
        """
        analysis = {}
        
        try:
            # 1. Análisis de costo de capital
            analysis['wacc'] = self.capm_engine.calculate_wacc()
            analysis['cost_of_equity'] = self.capm_engine.calculate_ke()
            
            # 2. Valoración del proyecto
            analysis['npv'] = self.capm_engine.calculate_npv(
                expected_cash_flows, initial_investment
            )
            
            # 3. Análisis de riesgo
            analysis['var'], var_metrics = self.risk_quantifier.calculate_var(
                mean=initial_investment,
                std_dev=cost_std_dev,
                confidence_level=0.95
            )
            analysis['var_metrics'] = var_metrics
            
            # 4. Contingencia recomendada
            analysis['contingency'] = self.risk_quantifier.suggest_contingency(
                base_cost=initial_investment,
                std_dev=cost_std_dev
            )
            
            # 5. Opciones reales (si VAN es positivo)
            if analysis['npv'] > 0:
                option_value = self.options_analyzer.value_option_to_wait(
                    project_value=analysis['npv'] + initial_investment,
                    investment_cost=initial_investment,
                    risk_free_rate=self.config.risk_free_rate,
                    time_to_expire_years=self.config.project_life_years,
                    volatility=project_volatility
                )
                analysis['real_option_value'] = option_value
                analysis['total_value'] = analysis['npv'] + option_value.get('option_value', 0)
            else:
                analysis['real_option_value'] = None
                analysis['total_value'] = analysis['npv']
            
            # 6. Métricas de performance
            analysis['performance_metrics'] = self._calculate_performance_metrics(
                analysis['npv'],
                initial_investment,
                len(expected_cash_flows)
            )
            
            logger.info(f"Análisis completado. VAN: ${analysis['npv']:,.2f}")
            
        except Exception as e:
            logger.error(f"Error en análisis de proyecto: {e}")
            raise
        
        return analysis
    
    def _calculate_performance_metrics(
        self,
        npv: float,
        investment: float,
        years: int
    ) -> Dict[str, float]:
        """Calcula métricas adicionales de performance."""
        roi = (npv / investment) if investment != 0 else 0
        annualized_return = ((1 + roi) ** (1 / years) - 1) if years > 0 else 0
        
        return {
            'roi': roi,
            'annualized_return': annualized_return,
            'npv_investment_ratio': npv / investment if investment != 0 else 0
        }
    
    def generate_report(self, analysis: Dict[str, any]) -> str:
        """Genera reporte de análisis en formato legible."""
        report = [
            "=" * 60,
            "INFORME DE VIABILIDAD FINANCIERA DEL PROYECTO",
            "=" * 60,
            f"\n1. COSTO DE CAPITAL:",
            f"   WACC: {analysis.get('wacc', 0):.2%}",
            f"   Costo de Equity (Ke): {analysis.get('cost_of_equity', 0):.2%}",
            f"\n2. VALORACIÓN:",
            f"   VAN del Proyecto: ${analysis.get('npv', 0):,.2f}",
        ]
        
        if 'real_option_value' in analysis and analysis['real_option_value']:
            opt_val = analysis['real_option_value'].get('option_value', 0)
            report.append(f"   Valor Opción Real: ${opt_val:,.2f}")
            report.append(f"   Valor Total (VAN + Opción): ${analysis.get('total_value', 0):,.2f}")
        
        report.extend([
            f"\n3. ANÁLISIS DE RIESGO:",
            f"   VaR (95% confianza): ${analysis.get('var', 0):,.2f}",
            f"   CVaR: ${analysis.get('var_metrics', {}).get('cvar', 0):,.2f}",
        ])
        
        if 'contingency' in analysis:
            cont = analysis['contingency'].get('recommended', 0)
            report.append(f"   Contingencia Recomendada: ${cont:,.2f}")
        
        if 'performance_metrics' in analysis:
            metrics = analysis['performance_metrics']
            report.extend([
                f"\n4. MÉTRICAS DE PERFORMANCE:",
                f"   ROI: {metrics.get('roi', 0):.2%}",
                f"   Retorno Anualizado: {metrics.get('annualized_return', 0):.2%}"
            ])
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


# ============================================================================
# FUNCIONES DE UTILIDAD (NUEVO)
# ============================================================================


def calculate_volatility_from_returns(returns: List[float], annual_trading_days: int = 252) -> float:
    """
    Calcula volatilidad anualizada a partir de retornos históricos.
    
    Args:
        returns: Retornos diarios/semanales/mensuales
        annual_trading_days: Días de trading por año
        
    Returns:
        Volatilidad anualizada
    """
    if not returns:
        raise ValueError("Lista de retornos vacía")
    
    returns_array = np.array(returns)
    std_daily = np.std(returns_array)
    volatility = std_daily * sqrt(annual_trading_days)
    
    logger.info(f"Volatilidad anualizada calculada: {volatility:.2%}")
    return volatility


def monte_carlo_simulation(
    initial_value: float,
    expected_return: float,
    volatility: float,
    time_years: float,
    n_simulations: int = 10000,
    random_seed: int = 42
) -> Dict[str, any]:
    """
    Simulación Monte Carlo para proyección de valores futuros.
    
    Returns:
        Estadísticas de la simulación
    """
    np.random.seed(random_seed)
    
    # Simulación geométrica browniana
    dt = 1/252  # Paso diario
    n_steps = int(time_years / dt)
    
    simulations = np.zeros((n_simulations, n_steps + 1))
    simulations[:, 0] = initial_value
    
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_simulations)
        simulations[:, t] = simulations[:, t-1] * np.exp(
            (expected_return - 0.5 * volatility**2) * dt + 
            volatility * sqrt(dt) * z
        )
    
    final_values = simulations[:, -1]
    
    stats = {
        'mean': np.mean(final_values),
        'median': np.median(final_values),
        'std': np.std(final_values),
        'percentile_5': np.percentile(final_values, 5),
        'percentile_95': np.percentile(final_values, 95),
        'probability_positive': np.mean(final_values > initial_value),
        'simulations': simulations
    }
    
    logger.info(f"Simulación Monte Carlo completada: {n_simulations:,} simulaciones")
    logger.info(f"  Valor esperado final: ${stats['mean']:,.2f}")
    logger.info(f"  Probabilidad de ganancia: {stats['probability_positive']:.1%}")
    
    return stats


# ============================================================================
# EJEMPLO DE USO (DEMOSTRACIÓN)
# ============================================================================


def ejemplo_uso_completo():
    """Ejemplo completo de uso del motor financiero."""
    print("\n" + "="*60)
    print("EJEMPLO: ANÁLISIS DE PROYECTO DE CONSTRUCCIÓN")
    print("="*60)
    
    # 1. Configuración
    config = FinancialConfig(
        risk_free_rate=0.04,
        market_premium=0.06,
        beta=1.3,
        tax_rate=0.30,
        cost_of_debt=0.07,
        debt_to_equity_ratio=0.7,
        project_life_years=5
    )
    
    # 2. Inicializar motor
    engine = FinancialEngine(config)
    
    # 3. Parámetros del proyecto
    inversion_inicial = 1_000_000
    flujos_esperados = [300_000, 350_000, 400_000, 450_000, 500_000]
    desviacion_costos = 150_000
    volatilidad_proyecto = 0.25
    
    # 4. Análisis completo
    try:
        analisis = engine.analyze_project(
            initial_investment=inversion_inicial,
            expected_cash_flows=flujos_esperados,
            cost_std_dev=desviacion_costos,
            project_volatility=volatilidad_proyecto
        )
        
        # 5. Generar reporte
        reporte = engine.generate_report(analisis)
        print(reporte)
        
        # 6. Análisis de sensibilidad adicional
        print("\nANÁLISIS DE SENSIBILIDAD DEL WACC A BETA:")
        sensibilidad = engine.capm_engine.sensitivity_analysis(
            'beta', [0.8, 1.0, 1.2, 1.4, 1.6]
        )
        for beta, wacc in sensibilidad.items():
            print(f"  Beta={beta:.1f}: WACC={wacc:.2%}")
        
    except Exception as e:
        print(f"Error en análisis: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Ejecutar ejemplo
    ejemplo_uso_completo()