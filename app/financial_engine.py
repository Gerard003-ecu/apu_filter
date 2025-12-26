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
8. Ajuste de volatilidad por topología (Riesgo Sistémico)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from math import exp, log, pow, sqrt
from typing import Any, Dict, List, Optional, Tuple

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
            (self.project_life_years, 1, 50, "Vida del proyecto"),
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
            raise TypeError(
                "El parámetro 'config' debe ser una instancia de FinancialConfig."
            )
        self.config = config

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

        CORRECCIONES:
        - Typo corregido: 'weight_debit' → 'weight_debt'
        - Añadida tolerancia numérica para validación

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

            # Validación con tolerancia numérica (precisión de punto flotante)
            if abs(weight_equity + weight_debt - 1.0) > 1e-10:
                logger.warning(
                    f"Inconsistencia numérica en pesos: "
                    f"Equity={weight_equity:.6f}, Deuda={weight_debt:.6f}"
                )

            # Costo de deuda después de impuestos
            after_tax_cost_of_debt = self.config.cost_of_debt * (1 - self.config.tax_rate)

            wacc = (weight_equity * ke) + (weight_debt * after_tax_cost_of_debt)

            logger.info(
                f"WACC calculado: {wacc:.2%} "
                f"(Ke={ke:.2%}, Kd_at={after_tax_cost_of_debt:.2%})"
            )
            return wacc

        except ZeroDivisionError:
            logger.error("División por cero en cálculo de pesos de capital")
            raise ValueError("Parámetros de estructura de capital inválidos")
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

    def sensitivity_analysis(
        self, parameter: str, range_values: List[float]
    ) -> Dict[float, float]:
        """
        Análisis de sensibilidad del WACC a cambios en parámetros.

        CORRECCIONES:
        - Uso correcto de cache_clear() para invalidar @lru_cache
        - Validación de parámetro existente
        - Manejo de errores por valor individual

        Args:
            parameter: 'beta', 'cost_of_debt', 'debt_to_equity_ratio', 'tax_rate'
            range_values: Valores a evaluar

        Returns:
            Diccionario con valores del parámetro y WACC resultante
        """
        # Validar que el parámetro existe
        if not hasattr(self.config, parameter):
            raise ValueError(f"Parámetro '{parameter}' no existe en FinancialConfig")

        results = {}
        original_value = getattr(self.config, parameter)

        try:
            for value in range_values:
                setattr(self.config, parameter, value)

                # Invalidar caché correctamente usando el método de lru_cache
                self.calculate_ke.cache_clear()
                self.calculate_wacc.cache_clear()

                try:
                    results[value] = self.calculate_wacc()
                except Exception as e:
                    logger.warning(f"Error con {parameter}={value}: {e}")
                    results[value] = float("nan")
        finally:
            # Restaurar valor original siempre (incluso si hay excepciones)
            setattr(self.config, parameter, original_value)
            self.calculate_ke.cache_clear()
            self.calculate_wacc.cache_clear()

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
        df_student_t: int = 5,
        trading_days_per_year: int = 252,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el Valor en Riesgo (VaR) con múltiples distribuciones.

        CORRECCIONES:
        - Fórmula VaR corregida: representa el costo máximo al nivel de confianza
        - CVaR (Expected Shortfall) con fórmula matemáticamente correcta
        - Parámetro configurable para días de trading

        Args:
            mean: Media de la distribución (costo esperado)
            std_dev: Desviación estándar
            confidence_level: Nivel de confianza (ej: 0.95 para 95%)
            time_horizon_days: Horizonte temporal en días
            df_student_t: Grados de libertad para distribución t-Student
            trading_days_per_year: Días hábiles por año

        Returns:
            Tupla con (VaR, métricas adicionales)
        """
        # Validación robusta
        if std_dev < 0:
            raise ValueError("La desviación estándar no puede ser negativa")
        if not 0 < confidence_level < 1:
            raise ValueError(f"Nivel de confianza debe estar en (0, 1): {confidence_level}")
        if time_horizon_days <= 0:
            raise ValueError(f"Horizonte temporal debe ser positivo: {time_horizon_days}")
        if df_student_t <= 2:
            raise ValueError(
                f"Grados de libertad deben ser > 2 para varianza finita: {df_student_t}"
            )

        try:
            # Escalado temporal usando regla de raíz cuadrada del tiempo
            time_scaling_factor = sqrt(time_horizon_days / trading_days_per_year)
            scaled_std = std_dev * time_scaling_factor

            if self.distribution == DistributionType.NORMAL:
                # Cuantil de la distribución normal estándar
                z_score = norm.ppf(confidence_level)
                distribution_name = "Normal"

                # VaR: Costo máximo esperado al nivel de confianza
                var = mean + z_score * scaled_std

                # CVaR (Expected Shortfall): E[X | X > VaR]
                # Para normal: ES = μ + σ * φ(z_α) / (1 - α)
                cvar = mean + scaled_std * norm.pdf(z_score) / (1 - confidence_level)

            elif self.distribution == DistributionType.STUDENT_T:
                z_score = t.ppf(confidence_level, df_student_t)
                distribution_name = f"Student-t(df={df_student_t})"

                var = mean + z_score * scaled_std

                # CVaR para t-Student: ajuste por colas pesadas
                # ES_t = μ + σ * (f_t(z_α) / (1-α)) * ((df + z_α²) / (df - 1))
                t_pdf_at_quantile = t.pdf(z_score, df_student_t)
                tail_adjustment = (df_student_t + z_score**2) / (df_student_t - 1)
                cvar = (
                    mean
                    + scaled_std
                    * t_pdf_at_quantile
                    / (1 - confidence_level)
                    * tail_adjustment
                )

            else:
                raise ValueError(f"Distribución no soportada: {self.distribution}")

            metrics = {
                "distribution": distribution_name,
                "z_score": z_score,
                "scaled_std": scaled_std,
                "cvar": cvar,
                "var_cvar_ratio": var / cvar if cvar != 0 else float("nan"),
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon_days,
                "time_scaling_factor": time_scaling_factor,
            }

            logger.info(
                f"VaR({confidence_level:.1%}, {time_horizon_days}d) = {var:,.2f}, "
                f"CVaR = {cvar:,.2f} [{distribution_name}]"
            )

            return var, metrics

        except Exception as e:
            logger.error(f"Error calculando VaR: {e}")
            raise

    def suggest_contingency(
        self,
        base_cost: float,
        std_dev: float,
        confidence_level: float = 0.90,
        method: str = "all",
    ) -> Dict[str, float]:
        """
        Sugiere contingencia usando múltiples métodos.

        CORRECCIONES:
        - Valor por defecto 'all' para calcular todos los métodos
        - Validación de método solicitado
        - Coeficiente de variación para decisiones heurísticas

        Args:
            base_cost: Costo base estimado (debe ser > 0)
            std_dev: Desviación estándar del costo
            confidence_level: Nivel de confianza deseado
            method: 'var', 'percentage', 'heuristic', o 'all'

        Returns:
            Diccionario con diferentes estimaciones de contingencia
        """
        valid_methods = {"var", "percentage", "heuristic", "all"}
        if method not in valid_methods:
            raise ValueError(f"Método '{method}' no válido. Opciones: {valid_methods}")

        if base_cost <= 0:
            raise ValueError(f"Costo base debe ser positivo: {base_cost}")

        contingencies = {}
        coefficient_of_variation = std_dev / base_cost if base_cost > 0 else 0

        calculate_all = method == "all"

        # Método VaR
        if calculate_all or method == "var":
            var, _ = self.calculate_var(base_cost, std_dev, confidence_level)
            contingencies["var_based"] = max(0, var - base_cost)

        # Método porcentual (estándares de industria de construcción)
        if calculate_all or method == "percentage":
            if coefficient_of_variation > 0.20:
                percentage = 0.20  # Alta incertidumbre
            elif coefficient_of_variation > 0.10:
                percentage = 0.15  # Incertidumbre moderada
            else:
                percentage = 0.10  # Baja incertidumbre
            contingencies["percentage_based"] = base_cost * percentage
            contingencies["percentage_rate"] = percentage

        # Método heurístico (múltiplo de desviación estándar)
        if calculate_all or method == "heuristic":
            if coefficient_of_variation > 0.20:
                multiplier = 2.0
            elif coefficient_of_variation > 0.15:
                multiplier = 1.5
            else:
                multiplier = 1.0
            contingencies["heuristic"] = multiplier * std_dev
            contingencies["heuristic_multiplier"] = multiplier

        # Recomendación final
        numeric_values = [
            v
            for k, v in contingencies.items()
            if isinstance(v, (int, float))
            and not k.endswith("_rate")
            and not k.endswith("_multiplier")
        ]

        if numeric_values:
            contingencies["recommended"] = max(numeric_values)
            contingencies["coefficient_of_variation"] = coefficient_of_variation
            logger.info(
                f"Contingencia recomendada: ${contingencies['recommended']:,.2f} "
                f"(CV={coefficient_of_variation:.1%})"
            )

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
        volatility: float,
    ) -> None:
        """Valida parámetros para modelos de opciones."""
        if project_value <= 0:
            raise ValueError(f"Valor del proyecto debe ser positivo: {project_value}")
        if investment_cost <= 0:
            raise ValueError(f"Costo de inversión debe ser positivo: {investment_cost}")
        if risk_free_rate < 0:
            raise ValueError(f"Tasa libre de riesgo no puede ser negativa: {risk_free_rate}")
        if time_to_expire_years <= 0:
            raise ValueError(
                f"Tiempo a expiración debe ser positivo: {time_to_expire_years}"
            )
        if volatility <= 0:
            raise ValueError(f"Volatilidad debe ser positiva: {volatility}")

    def _calculate_black_scholes_greeks(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> Dict[str, float]:
        """Calcula las griegas para el modelo Black-Scholes."""
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        N_d1, n_d1 = norm.cdf(d1), norm.pdf(d1)

        return {
            "delta": N_d1,
            "gamma": n_d1 / (S * sigma * sqrt(T)),
            "vega": S * n_d1 * sqrt(T),
            "theta": -(S * n_d1 * sigma) / (2 * sqrt(T))
            - r * K * exp(-r * T) * norm.cdf(d2),
            "rho": K * T * exp(-r * T) * norm.cdf(d2),
        }

    def value_option_to_wait(
        self,
        project_value: float,
        investment_cost: float,
        risk_free_rate: float,
        time_to_expire_years: float,
        volatility: float,
        dividend_yield: float = 0.0,
        binomial_steps: int = 100,
        is_american: bool = True,
    ) -> Dict[str, float]:
        """
        Valora la "Opción de Esperar" usando múltiples modelos.
        """
        try:
            self._validate_option_parameters(
                project_value,
                investment_cost,
                risk_free_rate,
                time_to_expire_years,
                volatility,
            )

            if self.model_type == OptionModelType.BLACK_SCHOLES and not is_american:
                return self._black_scholes_valuation(
                    project_value,
                    investment_cost,
                    risk_free_rate,
                    time_to_expire_years,
                    volatility,
                    dividend_yield,
                )
            else:
                return self._binomial_valuation(
                    project_value,
                    investment_cost,
                    risk_free_rate,
                    time_to_expire_years,
                    volatility,
                    binomial_steps,
                    is_american,
                )
        except Exception as e:
            logger.error(f"Error en valoración de opción real: {e}")
            raise

    def _black_scholes_valuation(
        self, S: float, K: float, r: float, T: float, sigma: float, q: float = 0.0
    ) -> Dict[str, float]:
        """Valoración usando modelo Black-Scholes-Merton (Europea)."""
        d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        option_value = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        greeks = self._calculate_black_scholes_greeks(S, K, T, r, sigma)

        intrinsic_value = max(S - K, 0)

        return {
            "option_value": option_value,
            "intrinsic_value": intrinsic_value,
            "time_value": max(0, option_value - intrinsic_value),
            "model": "Black-Scholes-Merton (Europea)",
            "greeks": greeks,
            "moneyness": S / K,
        }

    def _binomial_valuation(
        self,
        S: float,
        K: float,
        r: float,
        T: float,
        sigma: float,
        n: int = 100,
        american: bool = True,
    ) -> Dict[str, float]:
        """
        Valoración usando modelo binomial (Cox-Ross-Rubinstein).

        MEJORAS:
        - Soporte para opciones americanas (ejercicio anticipado)
        """
        dt = T / n
        u = exp(sigma * sqrt(dt))
        d = 1 / u
        p = (exp(r * dt) - d) / (u - d)

        if not 0 < p < 1:
            raise ValueError(f"Probabilidad neutral al riesgo fuera de rango: p={p:.4f}.")

        discount = exp(-r * dt)

        prices = np.zeros(n + 1)
        prices[0] = S * (d**n)
        for i in range(1, n + 1):
            prices[i] = prices[i - 1] * (u / d)

        option_values = np.maximum(prices - K, 0)  # Payoff de una Call

        early_exercise_count = 0
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                continuation = discount * (
                    p * option_values[j + 1] + (1 - p) * option_values[j]
                )

                if american:
                    price_at_node = S * (u**j) * (d ** (i - j))
                    exercise = max(price_at_node - K, 0)
                    if exercise > continuation:
                        option_values[j] = exercise
                        early_exercise_count += 1
                    else:
                        option_values[j] = continuation
                else:
                    option_values[j] = continuation

        delta = (
            (option_values[1] - option_values[0]) / (S * u - S * d)
            if n > 0
            else float("nan")
        )

        return {
            "option_value": option_values[0],
            "intrinsic_value": max(S - K, 0),
            "time_value": max(0, option_values[0] - max(S - K, 0)),
            "model": f"Binomial CRR ({'Americana' if american else 'Europea'})",
            "steps": n,
            "delta": delta,
            "p": p,
            "early_exercise_nodes": early_exercise_count if american else 0,
        }


# ============================================================================
# FACHADA PRINCIPAL (NUEVO)
# ============================================================================


class FinancialEngine:
    """
    Fachada principal que integra todos los componentes del motor financiero.
    """

    def __init__(self, config: FinancialConfig):
        self.config = config
        self.capm_engine = CapitalAssetPricing(config)
        self.risk_quantifier = RiskQuantifier(DistributionType.NORMAL)
        self.options_analyzer = RealOptionsAnalyzer(OptionModelType.BINOMIAL)

    def adjust_volatility_by_topology(
        self, base_volatility: float, topology_report: Dict[str, Any]
    ) -> float:
        """
        Ajusta la volatilidad basándose en la "Sinergia de Riesgo" topológica.

        Si se detecta 'synergy_detected' en el reporte, esto implica que los riesgos
        no son aislados, sino que interactúan multiplicativamente (Producto Cup).
        Se aplica un multiplicador de pánico a la volatilidad.

        Args:
            base_volatility (float): Volatilidad de mercado base.
            topology_report (Dict[str, Any]): Reporte del BusinessTopologicalAnalyzer.

        Returns:
            float: Volatilidad ajustada.
        """
        adjusted_volatility = base_volatility

        # Extraer bandera de sinergia
        synergy_info = {}
        if "details" in topology_report and "synergy_risk" in topology_report["details"]:
            synergy_info = topology_report["details"]["synergy_risk"]
        elif "synergy_risk" in topology_report:
            synergy_info = topology_report["synergy_risk"]

        is_risky = synergy_info.get("synergy_detected", False)

        if is_risky:
            # Factor de penalización por riesgo sistémico (1.2x = +20% volatilidad)
            penalty_factor = 1.2
            logger.warning(
                f"Sinergia de Riesgo detectada en Topología. Ajustando Volatilidad: "
                f"{base_volatility:.2%} -> {base_volatility * penalty_factor:.2%}"
            )
            adjusted_volatility *= penalty_factor

        return adjusted_volatility

    def analyze_project(
        self,
        initial_investment: float,
        expected_cash_flows: List[float],
        cost_std_dev: float,
        project_volatility: float,
        topology_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, any]:
        """
        Análisis completo de viabilidad de proyecto.

        Args:
            topology_report: Opcional, para ajuste de volatilidad por riesgo sistémico.
        """
        analysis = {}
        try:
            # Ajuste de volatilidad por topología
            final_volatility = project_volatility
            if topology_report:
                final_volatility = self.adjust_volatility_by_topology(
                    project_volatility, topology_report
                )
                analysis["volatility_adjusted"] = True
                analysis["original_volatility"] = project_volatility

            analysis["wacc"] = self.capm_engine.calculate_wacc()
            analysis["npv"] = self.capm_engine.calculate_npv(
                expected_cash_flows, initial_investment
            )
            analysis["volatility"] = final_volatility

            analysis["var"], var_metrics = self.risk_quantifier.calculate_var(
                mean=initial_investment, std_dev=cost_std_dev, confidence_level=0.95
            )
            analysis["var_metrics"] = var_metrics

            analysis["contingency"] = self.risk_quantifier.suggest_contingency(
                base_cost=initial_investment, std_dev=cost_std_dev
            )

            option_params = {
                "project_value": analysis["npv"] + initial_investment,
                "investment_cost": initial_investment,
                "risk_free_rate": self.config.risk_free_rate,
                "time_to_expire_years": self.config.project_life_years,
                "volatility": final_volatility,
            }
            if option_params["project_value"] > 0:
                analysis["real_option"] = self.options_analyzer.value_option_to_wait(
                    **option_params
                )
                analysis["total_value"] = analysis["npv"] + analysis["real_option"].get(
                    "option_value", 0
                )
            else:
                analysis["real_option"] = None
                analysis["total_value"] = analysis["npv"]

            analysis["performance"] = self._calculate_performance_metrics(
                analysis["npv"], initial_investment, len(expected_cash_flows)
            )

        except Exception as e:
            logger.error(f"Error en análisis de proyecto: {e}")
            raise

        return analysis

    def _calculate_performance_metrics(
        self, npv: float, investment: float, years: int
    ) -> Dict[str, float]:
        """
        Calcula métricas adicionales de performance.

        CORRECCIONES:
        - Manejo de inversión cero o negativa
        - Protección contra ROI < -100% para retorno anualizado
        """
        metrics = {}

        if investment > 0:
            roi = npv / investment
            pi = (npv + investment) / investment
            metrics["profitability_index"] = pi
            metrics["recommendation"] = "ACEPTAR" if pi > 1 else "RECHAZAR"
        elif investment < 0:
            logger.warning("Inversión inicial negativa, ROI invertido")
            roi = -npv / investment
            metrics["profitability_index"] = float("nan")
            metrics["recommendation"] = "REVISAR"  # Casuística no estándar
        else:
            roi = float("inf") if npv > 0 else (float("-inf") if npv < 0 else 0)
            metrics["profitability_index"] = float("nan")
            metrics["recommendation"] = "REVISAR"  # Indefinido
            logger.warning("Inversión inicial es cero, ROI y PI indefinidos")

        metrics["roi"] = roi

        if years > 0 and investment != 0:
            total_return = 1 + roi
            if total_return > 0:
                annualized_return = pow(total_return, 1 / years) - 1
            elif total_return == 0:
                annualized_return = -1.0
            else:
                annualized_return = float("nan")
                logger.warning(
                    f"ROI={roi:.2%} implica pérdida > 100%, anualización no definida"
                )
        else:
            annualized_return = float("nan")

        metrics["annualized_return"] = annualized_return

        return metrics


# ============================================================================
# FUNCIONES DE UTILIDAD (NUEVO)
# ============================================================================


def calculate_volatility_from_returns(
    returns: List[float], frequency: str = "daily", annual_trading_days: int = 252
) -> float:
    """
    Calcula volatilidad anualizada a partir de retornos históricos.

    MEJORAS:
    - Soporte para múltiples frecuencias de datos
    - Validación de tamaño mínimo de muestra
    """
    if not returns or len(returns) < 2:
        raise ValueError(f"Se requieren al menos 2 retornos. Recibidos: {len(returns)}")

    factors = {"daily": annual_trading_days, "weekly": 52, "monthly": 12, "annual": 1}
    if frequency not in factors:
        raise ValueError(
            f"Frecuencia '{frequency}' no válida. Opciones: {list(factors.keys())}"
        )

    std_period = np.std(np.array(returns), ddof=1)  # ddof=1 for sample std dev
    volatility = std_period * sqrt(factors[frequency])

    logger.info(
        f"Volatilidad anualizada: {volatility:.2%} (n={len(returns)}, freq={frequency})"
    )
    return volatility


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- DEMO ---
    config = FinancialConfig(beta=1.4)
    engine = FinancialEngine(config)

    analisis = engine.analyze_project(
        initial_investment=1e6,
        expected_cash_flows=[3e5, 3.5e5, 4e5, 4.5e5, 5e5],
        cost_std_dev=1.5e5,
        project_volatility=0.30,
    )

    print("\n--- INFORME DE VIABILIDAD ---")
    print(f"WACC: {analisis['wacc']:.2%}")
    print(f"VAN: ${analisis['npv']:,.2f}")
    print(f"Valor Total (VAN + Opción): ${analisis['total_value']:,.2f}")
    print(
        f"Contingencia Recomendada (VaR-based): ${analisis['contingency']['recommended']:,.2f}"
    )
    print(f"ROI: {analisis['performance']['roi']:.2%}")
    print(f"Opción Americana (Binomial): ${analisis['real_option']['option_value']:,.2f}")
    print("-----------------------------\n")
