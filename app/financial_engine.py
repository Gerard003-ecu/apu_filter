# -*- coding: utf-8 -*-
"""
Motor de Inteligencia Financiera para la Valoración de Proyectos de Construcción.

Este módulo integra principios de finanzas corporativas (CAPM, WACC), cuantificación
de riesgos (VaR) y valoración de opciones reales (Black-Scholes) para enriquecer
las estimaciones de costos técnicos con una capa de viabilidad económica.
"""

import logging
from dataclasses import dataclass, field
from math import exp, log, sqrt
from typing import Optional

from scipy.stats import norm

# Configuración del logger para este módulo
logger = logging.getLogger(__name__)


# ============================================================================
# CLASE DE CONFIGURACIÓN FINANCIERA
# ============================================================================


@dataclass
class FinancialConfig:
    """
    Almacena parámetros macroeconómicos y de proyecto con valores por defecto
    conservadores, representativos del sector de construcción.

    Estos valores pueden ser sobreescritos para ajustar el análisis a un
    escenario económico específico.

    Attributes:
        risk_free_rate (float): Tasa Libre de Riesgo (Rf). Generalmente, el rendimiento
                                de bonos gubernamentales a largo plazo. Default: 4.0%.
        market_premium (float): Prima de Riesgo de Mercado (Rm). El rendimiento adicional
                                esperado del mercado sobre la tasa libre de riesgo. Default: 6.0%.
        beta (float): Coeficiente Beta (β). Medida de la volatilidad de los retornos del
                      proyecto en relación con el mercado. >1 indica mayor volatilidad. Default: 1.2.
        tax_rate (float): Tasa Impositiva corporativa (T). Default: 30.0%.
        cost_of_debt (float): Costo de la Deuda (Kd). Tasa de interés que la empresa paga
                              por su financiamiento de deuda. Default: 8.0%.
        debt_to_equity_ratio (float): Razón Deuda/Capital (D/E). Proporción de la deuda
                                      respecto al capital en la estructura financiera. Default: 0.6.
    """

    risk_free_rate: float = 0.04
    market_premium: float = 0.06
    beta: float = 1.2
    tax_rate: float = 0.30
    cost_of_debt: float = 0.08
    debt_to_equity_ratio: float = 0.6


# ============================================================================
# MOTOR DE VALORACIÓN DE ACTIVOS (CAPM Y WACC)
# ============================================================================


class CapitalAssetPricing:
    """
    Calcula el costo de capital utilizando el Modelo de Valoración de Activos de Capital
    (CAPM) y el Costo Promedio Ponderado de Capital (WACC).
    """

    def __init__(self, config: FinancialConfig):
        """
        Inicializa el motor con una configuración financiera.

        Args:
            config: Una instancia de FinancialConfig.
        """
        if not isinstance(config, FinancialConfig):
            raise TypeError("El parámetro 'config' debe ser una instancia de FinancialConfig.")
        self.config = config

    def calculate_ke(self) -> float:
        """
        Calcula el Costo del Equity (Ke) usando el modelo CAPM.

        Fórmula: Ke = Rf + β * (Rm - Rf)

        Returns:
            El costo del equity como un valor decimal (ej. 0.11 para 11%).
        """
        try:
            ke = self.config.risk_free_rate + self.config.beta * (
                self.config.market_premium - self.config.risk_free_rate
            )
            logger.debug(f"Costo del Equity (Ke) calculado: {ke:.4f}")
            return ke
        except Exception as e:
            logger.error(f"Error calculando Ke: {e}")
            return 0.0

    def calculate_wacc(self) -> float:
        """
        Calcula el Costo Promedio Ponderado de Capital (WACC).

        El WACC representa la tasa de descuento que se debe utilizar para valorar
        los flujos de caja futuros de un proyecto.

        Fórmula:
        WACC = (E / (E + D)) * Ke + (D / (E + D)) * Kd * (1 - T)
        donde E/(E+D) y D/(E+D) son los pesos de equity y deuda.

        Returns:
            El WACC como un valor decimal (ej. 0.09 para 9%).
        """
        try:
            ke = self.calculate_ke()
            d_e_ratio = self.config.debt_to_equity_ratio

            # Calcular pesos
            weight_equity = 1 / (1 + d_e_ratio)
            weight_debt = d_e_ratio / (1 + d_e_ratio)

            # Calcular WACC
            wacc = (weight_equity * ke) + (
                weight_debt * self.config.cost_of_debt * (1 - self.config.tax_rate)
            )

            logger.debug(f"WACC calculado: {wacc:.4f}")
            return wacc
        except ZeroDivisionError:
            logger.error("La razón D/E no puede resultar en una división por cero.")
            return 0.0
        except Exception as e:
            logger.error(f"Error calculando WACC: {e}")
            return 0.0


# ============================================================================
# CUANTIFICADOR DE RIESGOS
# ============================================================================


class RiskQuantifier:
    """
    Proporciona herramientas para medir el riesgo financiero, como el Valor en
    Riesgo (VaR) y sugerencias de contingencia.
    """

    def calculate_var(
        self, mean: float, std_dev: float, confidence_level: float = 0.95
    ) -> float:
        """
        Calcula el Valor en Riesgo (VaR) paramétrico asumiendo una distribución normal.

        El VaR estima la pérdida máxima esperada de una inversión durante un
        período de tiempo, con un nivel de confianza dado.

        Args:
            mean (float): El costo esperado (media) de la distribución de costos.
            std_dev (float): La desviación estándar de los costos.
            confidence_level (float): El nivel de confianza (ej. 0.95 para 95%).

        Returns:
            El valor del VaR. Un VaR positivo indica que el costo podría exceder
            la media en esa cantidad.
        """
        if std_dev < 0:
            logger.warning("La desviación estándar no puede ser negativa. Se usará 0.")
            std_dev = 0.0

        try:
            # Z-score para el nivel de confianza (cola derecha)
            z_score = norm.ppf(confidence_level)
            var = mean + z_score * std_dev
            logger.debug(
                f"VaR ({confidence_level:.0%}) calculado: {var:.2f} (z-score: {z_score:.4f})"
            )
            return var
        except Exception as e:
            logger.error(f"Error calculando VaR: {e}")
            return mean  # Devolver la media como fallback

    def suggest_contingency(self, std_dev: float, multiplier: float = 1.5) -> float:
        """
        Sugiere un porcentaje de imprevistos basado en la volatilidad.

        Una regla heurística simple es usar un múltiplo de la desviación estándar
        como un colchón para contingencias.

        Args:
            std_dev (float): La desviación estándar de los costos.
            multiplier (float): El factor a multiplicar por la desviación estándar.

        Returns:
            El monto de contingencia sugerido.
        """
        if std_dev < 0:
            logger.warning("La desviación estándar no puede ser negativa. Se usará 0.")
            std_dev = 0.0

        contingency = multiplier * std_dev
        logger.debug(
            f"Contingencia sugerida ({multiplier}xσ): {contingency:.2f}"
        )
        return contingency


# ============================================================================
# ANALIZADOR DE OPCIONES REALES
# ============================================================================


class RealOptionsAnalyzer:
    """
    Valora la flexibilidad estratégica en proyectos de inversión utilizando
    el modelo Black-Scholes, adaptado para opciones reales.
    """

    def value_option_to_wait(
        self,
        project_value: float,
        investment_cost: float,
        risk_free_rate: float,
        time_to_expire_years: float,
        volatility: float,
    ) -> Optional[float]:
        """
        Valora la "Opción de Esperar" usando el modelo Black-Scholes.

        Esta opción representa el valor estratégico de poder posponer una decisión
        de inversión irreversible en un entorno de incertidumbre.

        Args:
            project_value (float): Valor presente de los flujos de caja esperados del
                                   proyecto (S). Análogo al precio del activo subyacente.
            investment_cost (float): Costo de la inversión inicial (K). Análogo al
                                     precio de ejercicio.
            risk_free_rate (float): Tasa libre de riesgo (r).
            time_to_expire_years (float): Tiempo hasta que la oportunidad de inversión
                                          desaparece (T).
            volatility (float): Volatilidad de los retornos del proyecto (σ), derivada
                                de la desviación estándar de los costos o precios.

        Returns:
            El valor de la opción de esperar, o None si los inputs son inválidos.
        """
        # Validar inputs para evitar errores matemáticos
        if any(
            v <= 0
            for v in [project_value, investment_cost, time_to_expire_years, volatility]
        ):
            logger.warning(
                "Inputs para Black-Scholes deben ser positivos. "
                f"S={project_value}, K={investment_cost}, T={time_to_expire_years}, σ={volatility}"
            )
            return None

        try:
            # Calcular d1 y d2
            d1 = (
                log(project_value / investment_cost)
                + (risk_free_rate + (volatility**2) / 2) * time_to_expire_years
            ) / (volatility * sqrt(time_to_expire_years))

            d2 = d1 - volatility * sqrt(time_to_expire_years)

            # Calcular el valor de la opción de compra (Call Option)
            option_value = project_value * norm.cdf(d1) - investment_cost * exp(
                -risk_free_rate * time_to_expire_years
            ) * norm.cdf(d2)

            logger.debug(f"Valor de la Opción de Esperar calculado: {option_value:.2f}")
            return option_value
        except ZeroDivisionError:
            logger.error("División por cero en cálculo de Black-Scholes (volatilidad o tiempo son cero).")
            return None
        except Exception as e:
            logger.error(f"Error calculando valor de opción real: {e}")
            return None
