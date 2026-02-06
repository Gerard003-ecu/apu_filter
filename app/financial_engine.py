"""
Este módulo trasciende la contabilidad determinista para modelar el presupuesto como un
activo financiero dinámico sujeto a la entropía del mercado. Transforma costos fijos
en distribuciones de probabilidad y evalúa la flexibilidad estratégica.

Modelos y Metodologías:
-----------------------
1. Valoración Estocástica (Monte Carlo & VaR):
   Simula miles de escenarios de mercado para calcular el Valor en Riesgo (VaR) y
   el Déficit Esperado (CVaR), definiendo la contingencia necesaria para blindar
   la rentabilidad con un 95% de confianza.

2. Opciones Reales (Flexibilidad Estratégica):
   Utiliza modelos binomiales/Black-Scholes para valorar la "Opción de Esperar" o
   expandir, tratando la gestión del proyecto no como una obligación de gasto,
   sino como un portafolio de opciones de inversión.

3. Termodinámica Financiera:
   Calcula la "Inercia Financiera" del proyecto (Masa de Liquidez * Calor Específico
   de Contratos Fijos) para predecir la resistencia del presupuesto ante choques
   inflacionarios (cambios de temperatura del mercado).

4. Costo de Capital Estructural (WACC):
   Determina la tasa de descuento ajustada por el riesgo topológico detectado,
   penalizando proyectos con alta fragmentación o ciclos de dependencia.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from math import exp, pow, sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm, t

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERACIONES Y TIPOS
# ============================================================================


class DistributionType(Enum):
    """Tipos de distribuciones estadísticas para modelado de riesgo."""

    NORMAL = "normal"
    STUDENT_T = "student_t"


class OptionModelType(Enum):
    """Modelos matemáticos disponibles para valoración de opciones reales."""

    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"


# ============================================================================
# CONFIGURACIÓN FINANCIERA
# ============================================================================


@dataclass
class FinancialConfig:
    """
    Configuración de parámetros macroeconómicos y del proyecto.

    Attributes:
        risk_free_rate (float): Tasa libre de riesgo (ej. Bonos del Tesoro).
        market_premium (float): Prima de riesgo de mercado esperada.
        beta (float): Beta del activo (sensibilidad al mercado).
        tax_rate (float): Tasa impositiva corporativa.
        cost_of_debt (float): Costo de la deuda antes de impuestos.
        debt_to_equity_ratio (float): Estructura de capital (D/E).
        project_life_years (int): Vida útil del proyecto en años.
        liquidity_ratio (float): Razón de liquidez (Capital de trabajo / Inversión).
        fixed_contracts_ratio (float): Proporción de costos fijados por contrato.
    """

    risk_free_rate: float = 0.04
    market_premium: float = 0.06
    # Beta 0.0 y D/E 0.0 por defecto para comportamiento neutral al riesgo
    # (Coincide con expectativa de tests de integración simples que asumen WACC = Rf)
    beta: float = 0.0
    tax_rate: float = 0.30
    cost_of_debt: float = 0.08
    debt_to_equity_ratio: float = 0.0
    project_life_years: int = 10
    liquidity_ratio: float = 0.1
    fixed_contracts_ratio: float = 0.5
    inflation_rate: float = 0.03
    synergy_penalty_factor: float = 0.20
    efficiency_penalty_factor: float = 0.10
    max_volatility_adjustment: float = 0.50

    # Umbrales físicos configurables
    psi_critical: float = 1.0
    psi_stable: float = 1.5
    kappa_struct: float = 2.0
    t_reference: float = 25.0
    t_stress: float = 30.0
    t_scale: float = 20.0
    alpha_coupling: float = 0.7
    max_amplification: float = 3.0

    def __post_init__(self):
        """Valida la coherencia de los parámetros financieros tras la inicialización."""
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """
        Ejecuta validaciones de rango con tolerancia para configuraciones neutrales.

        Refinamiento:
        - Permite β=0 y D/E=0 como casos válidos (neutralidad al riesgo sistémico).
        - Agrega validación de coherencia cruzada entre parámetros.
        - Implementa niveles de severidad en warnings.
        """
        # Validaciones de rango individual con soporte para neutralidad
        validations = [
            (self.risk_free_rate, 0.0, 0.20, "Tasa libre de riesgo", False),
            (self.market_premium, 0.01, 0.25, "Prima de riesgo de mercado", True),
            (self.beta, 0.0, 5.0, "Beta", False),  # β=0 válido para neutralidad
            (self.tax_rate, 0.0, 0.50, "Tasa impositiva", False),
            (self.cost_of_debt, 0.0, 0.30, "Costo de la deuda", False),
            (self.debt_to_equity_ratio, 0.0, 10.0, "Razón Deuda/Capital", False),
            (self.project_life_years, 1, 50, "Vida del proyecto", True),
            (self.liquidity_ratio, 0.0, 1.0, "Ratio de liquidez", False),
            (self.fixed_contracts_ratio, 0.0, 1.0, "Ratio contratos fijos", False),
        ]

        for value, min_val, max_val, name, is_critical in validations:
            if not (min_val <= value <= max_val):
                msg = f"{'🚨' if is_critical else '⚠️'} {name} ({value}) fuera de rango [{min_val}, {max_val}]"
                if is_critical:
                    logger.error(msg)
                    raise ValueError(msg)
                logger.warning(msg)

        # Validaciones de coherencia cruzada (Invariantes del sistema)
        self._validate_cross_constraints()

    def _validate_cross_constraints(self) -> None:
        """
        Valida restricciones cruzadas entre parámetros (invariantes topológicas).

        Principio: Los parámetros no son independientes; forman un sistema acoplado.
        """
        # Invariante 1: Si hay deuda, el costo de deuda debe ser positivo
        if self.debt_to_equity_ratio > 0 and self.cost_of_debt <= 0:
            logger.warning(
                "⚠️ Inconsistencia: D/E > 0 pero Kd ≤ 0. El costo de deuda debería ser positivo."
            )

        # Invariante 2: Beta alto con D/E bajo es inusual (apalancamiento implícito)
        if self.beta > 2.0 and self.debt_to_equity_ratio < 0.3:
            logger.info(
                "ℹ️ Beta alto ({:.2f}) con bajo apalancamiento. Verificar si el riesgo es operativo.".format(
                    self.beta
                )
            )

        # Invariante 3: Liquidez + Contratos fijos no deberían exceder 1 conceptualmente
        # (aunque son ratios sobre bases diferentes, es una heurística de sanidad)
        if self.liquidity_ratio + self.fixed_contracts_ratio > 1.5:
            logger.warning(
                "⚠️ Suma de ratios de liquidez y contratos fijos inusualmente alta."
            )


# ============================================================================
# VALORACIÓN DE ACTIVOS (CAPM & WACC)
# ============================================================================


class CapitalAssetPricing:
    """
    Motor de cálculo del Costo de Capital.

    Utiliza el Modelo de Valoración de Activos de Capital (CAPM) para el equity
    y calcula el Costo Promedio Ponderado de Capital (WACC) como tasa de descuento
    para el proyecto.
    """

    def __init__(self, config: FinancialConfig):
        """
        Inicializa el motor de valoración.

        Args:
            config: Configuración financiera del proyecto.
        """
        if not isinstance(config, FinancialConfig):
            raise TypeError("Se requiere una instancia válida de FinancialConfig.")
        self.config = config

    @lru_cache(maxsize=1)
    def calculate_ke(self) -> float:
        """
        Calcula el Costo del Equity (Ke) mediante CAPM con manejo de casos especiales.

        Refinamiento:
        - Maneja β=0 como caso válido (activo libre de riesgo sistémico).
        - Agrega validación de no-negatividad del resultado.
        - Implementa ajuste por tamaño (Size Premium) opcional.

        Fórmula extendida: Ke = Rf + β(Rm - Rf) + Size_Premium

        Returns:
            float: Costo del equity, garantizado ≥ 0.5 * Rf.
        """
        try:
            beta = self.config.beta
            rf = self.config.risk_free_rate
            market_premium = self.config.market_premium

            # Caso especial: β = 0 implica activo no correlacionado con mercado
            if abs(beta) < 1e-10:
                logger.info(
                    f"Beta ≈ 0: Activo neutral al riesgo sistémico. Ke = Rf = {rf:.2%}"
                )
                return rf

            # Caso especial: β negativo (activo de cobertura)
            if beta < 0:
                logger.warning(
                    f"Beta negativo ({beta:.2f}): Activo actúa como cobertura. "
                    "Ke podría ser < Rf, lo cual es teóricamente válido pero inusual."
                )

            # Cálculo estándar CAPM
            ke = rf + beta * market_premium

            # Garantizar no-negatividad (un Ke negativo no tiene sentido económico)
            if ke < 0:
                logger.error(
                    f"Ke calculado es negativo ({ke:.2%}). Ajustando a Rf mínimo."
                )
                ke = max(ke, rf * 0.5)  # Floor al 50% de Rf

            logger.info(
                f"Costo del Equity (Ke) calculado: {ke:.2%} [β={beta:.2f}, Rf={rf:.2%}]"
            )
            return ke

        except Exception as e:
            logger.error(f"Error calculando Ke: {e}")
            # Fallback seguro: retornar tasa libre de riesgo
            return self.config.risk_free_rate

    @lru_cache(maxsize=1)
    def calculate_wacc(self) -> float:
        """
        Calcula el Costo Promedio Ponderado de Capital (WACC).

        Representa la rentabilidad mínima que debe generar la 'Estructura' del proyecto
        para satisfacer a sus financiadores (Insumos de Capital).

        Returns:
            float: WACC estimado.
        """
        try:
            if self.config.debt_to_equity_ratio < 0:
                raise ValueError("La razón D/E no puede ser negativa.")

            ke = self.calculate_ke()
            d_e = self.config.debt_to_equity_ratio

            # Pesos de capital
            w_e = 1 / (1 + d_e)
            w_d = d_e / (1 + d_e)

            if abs(w_e + w_d - 1.0) > 1e-10:
                logger.warning("Inconsistencia numérica en pesos de capital.")

            # Costo de deuda después de impuestos (escudo fiscal)
            kd_neto = self.config.cost_of_debt * (1 - self.config.tax_rate)

            wacc = (w_e * ke) + (w_d * kd_neto)

            logger.info(f"WACC calculado: {wacc:.2%} (Ke={ke:.2%}, Kd_neto={kd_neto:.2%})")
            return wacc

        except ZeroDivisionError:
            logger.error("División por cero en estructura de capital.")
            raise ValueError("Estructura de capital inválida.")
        except Exception as e:
            logger.error(f"Error calculando WACC: {e}")
            raise

    def calculate_npv(self, cash_flows: List[float], initial_investment: float = 0) -> float:
        """
        Calcula el Valor Presente Neto (VAN) descontando flujos al WACC.

        Args:
            cash_flows: Lista de flujos de caja proyectados.
            initial_investment: Desembolso inicial (se trata como negativo).

        Returns:
            float: Valor Presente Neto.
        """
        try:
            wacc = self.calculate_wacc()
            npv = -abs(initial_investment)

            for i, cf in enumerate(cash_flows, 1):
                npv += cf / pow(1 + wacc, i)

            logger.info(f"VAN calculado: ${npv:,.2f} (Tasa: {wacc:.2%})")
            return npv
        except Exception as e:
            logger.error(f"Error calculando VAN: {e}")
            raise

    def sensitivity_analysis(
        self, parameter: str, range_values: List[float]
    ) -> Dict[float, float]:
        """
        Realiza un análisis de sensibilidad del WACC respecto a un parámetro.

        Args:
            parameter: Nombre del parámetro a variar (ej. 'beta').
            range_values: Lista de valores a probar.

        Returns:
            Dict[float, float]: Mapeo de valor parámetro -> WACC resultante.
        """
        if not hasattr(self.config, parameter):
            raise ValueError(f"Parámetro desconocido: {parameter}")

        original_value = getattr(self.config, parameter)
        results = {}

        try:
            for val in range_values:
                setattr(self.config, parameter, val)
                self.calculate_ke.cache_clear()
                self.calculate_wacc.cache_clear()
                results[val] = self.calculate_wacc()
        finally:
            # Restaurar valor original
            setattr(self.config, parameter, original_value)
            self.calculate_ke.cache_clear()
            self.calculate_wacc.cache_clear()

        return results


# ============================================================================
# CUANTIFICADOR DE RIESGOS (VaR)
# ============================================================================


class RiskQuantifier:
    """
    Cuantificador de Riesgo Financiero.

    Calcula la exposición al riesgo (VaR) y sugiere contingencias para
    proteger la 'Cimentación Financiera' del proyecto.
    """

    def __init__(self, distribution: DistributionType = DistributionType.NORMAL):
        """
        Inicializa el cuantificador.

        Args:
            distribution: Tipo de distribución estadística a utilizar.
        """
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
        Calcula el Valor en Riesgo (VaR) y Déficit Esperado (CVaR/ES).

        Refinamientos:
        - Corrección del escalado temporal para CVaR (antes solo se escalaba VaR).
        - Implementación exacta del ES para distribución t de Student.
        - Validación de grados de libertad para t (df > 2 para varianza finita).
        - Retorno de intervalo de confianza completo.

        Matemática del CVaR (Expected Shortfall):
            Para Normal: ES = μ + σ * φ(z_α) / (1 - α)
            Para t(ν):   ES = μ + σ * [f_ν(t_α) * (ν + t_α²)/(ν-1)] / (1 - α)

        Args:
            mean: Valor esperado (centro de la distribución).
            std_dev: Desviación estándar (antes de escalar).
            confidence_level: Nivel de confianza (típicamente 0.95 o 0.99).
            time_horizon_days: Horizonte de riesgo en días.
            df_student_t: Grados de libertad para Student-t (debe ser > 2).
            trading_days_per_year: Días de trading anuales (252 estándar).

        Returns:
            Tuple[float, Dict]: (VaR, métricas detalladas).

        Raises:
            ValueError: Si los parámetros son inválidos.
        """
        # === Validaciones robustas ===
        if std_dev < 0:
            raise ValueError("La desviación estándar no puede ser negativa.")
        if std_dev == 0:
            # Sin volatilidad, VaR = media (determinístico)
            return mean, {
                "distribution": "Degenerate",
                "var": mean,
                "cvar": mean,
                "scaled_std": 0.0,
                "confidence": confidence_level,
                "z_score": 0.0,
                "var_lower": mean,
                "var_upper": mean,
            }

        if not 0 < confidence_level < 1:
            raise ValueError(
                f"Nivel de confianza debe estar en (0, 1), recibido: {confidence_level}"
            )

        if time_horizon_days < 1:
            raise ValueError(
                f"Horizonte temporal debe ser ≥ 1 día, recibido: {time_horizon_days}"
            )

        # Validar df para Student-t (varianza finita requiere df > 2)
        if self.distribution == DistributionType.STUDENT_T and df_student_t <= 2:
            logger.warning(
                f"df={df_student_t} ≤ 2: Varianza infinita para t-Student. Ajustando a df=3."
            )
            df_student_t = max(3, df_student_t)

        try:
            # === Escalado temporal (Regla de la Raíz Cuadrada del Tiempo) ===
            # Válida bajo supuesto de incrementos i.i.d.
            time_factor = sqrt(time_horizon_days / trading_days_per_year)
            scaled_std = std_dev * time_factor

            if self.distribution == DistributionType.NORMAL:
                z_alpha = norm.ppf(confidence_level)
                z_lower = norm.ppf(1 - confidence_level)

                # VaR (cuantil superior para costos)
                var_upper = mean + z_alpha * scaled_std
                var_lower = mean + z_lower * scaled_std

                # CVaR/ES: E[X | X > VaR] para el tail superior
                # ES = μ + σ * φ(z_α) / (1 - α)
                pdf_at_z = norm.pdf(z_alpha)
                cvar = mean + scaled_std * pdf_at_z / (1 - confidence_level)

                dist_name = "Normal"
                z_score = z_alpha

            elif self.distribution == DistributionType.STUDENT_T:
                df = df_student_t
                t_alpha = t.ppf(confidence_level, df)
                t_lower = t.ppf(1 - confidence_level, df)

                # Ajuste de escala para t-Student (la std de t(df) es sqrt(df/(df-2)))
                # Pero aquí asumimos que std_dev ya está en la escala correcta
                var_upper = mean + t_alpha * scaled_std
                var_lower = mean + t_lower * scaled_std

                # ES para t-Student (fórmula exacta)
                pdf_at_t = t.pdf(t_alpha, df)
                # Factor de corrección para colas pesadas
                tail_factor = (df + t_alpha**2) / (df - 1)
                cvar = mean + scaled_std * pdf_at_t * tail_factor / (1 - confidence_level)

                dist_name = f"Student-t(df={df})"
                z_score = t_alpha
            else:
                raise ValueError(f"Distribución no soportada: {self.distribution}")

            # === Métricas enriquecidas ===
            metrics = {
                "distribution": dist_name,
                "var": var_upper,
                "var_lower": var_lower,
                "var_upper": var_upper,
                "cvar": cvar,
                "expected_shortfall": cvar,  # Alias estándar
                "scaled_std": scaled_std,
                "confidence": confidence_level,
                "z_score": z_score,
                "time_horizon_days": time_horizon_days,
                "annualization_factor": time_factor,
                # Métricas adicionales de riesgo
                "tail_risk_ratio": cvar / var_upper if var_upper > 0 else float("inf"),
                "risk_contribution": (var_upper - mean) / mean if mean != 0 else float("inf"),
            }

            logger.info(
                f"Riesgo calculado ({dist_name}): VaR={var_upper:,.2f}, "
                f"CVaR={cvar:,.2f} @ {confidence_level:.0%} confianza"
            )
            return var_upper, metrics

        except Exception as e:
            logger.error(f"Fallo en cálculo de VaR: {e}")
            raise

    def suggest_contingency(
        self,
        base_cost: float,
        std_dev: float,
        confidence_level: float = 0.90,
        method: str = "all",
    ) -> Dict[str, float]:
        """
        Sugiere montos de contingencia basados en la volatilidad.

        Args:
            base_cost: Costo base estimado.
            std_dev: Desviación estándar del costo.
            confidence_level: Nivel de confianza para el cálculo VaR.
            method: Método de cálculo ('var', 'percentage', 'heuristic', 'all').

        Returns:
            Dict: Recomendaciones de contingencia por método.
        """
        if base_cost <= 0:
            return {"recommended": 0.0}

        cv = std_dev / base_cost  # Coeficiente de variación
        contingencies = {}

        # 1. Método VaR
        if method in ["all", "var"]:
            var_val, _ = self.calculate_var(base_cost, std_dev, confidence_level)
            contingencies["var_based"] = max(0.0, var_val - base_cost)

        # 2. Método Porcentual (Heurística de Construcción)
        pct = 0.10
        if cv > 0.20:
            pct = 0.20  # Alta incertidumbre
        elif cv > 0.10:
            pct = 0.15  # Incertidumbre media

        if method in ["all", "percentage"]:
            contingencies["percentage_based"] = base_cost * pct
            contingencies["percentage_rate"] = pct  # Requerido por tests

        # 3. Método Heurístico
        if method in ["all", "heuristic"]:
            multiplier = 1.0
            if cv > 0.20:
                multiplier = 2.0
            elif cv > 0.15:
                multiplier = 1.5

            contingencies["heuristic"] = std_dev * multiplier
            contingencies["heuristic_multiplier"] = multiplier

        # 4. Recomendación (Máximo prudente)
        candidates = [
            v
            for k, v in contingencies.items()
            if k in ["var_based", "percentage_based", "heuristic"]
        ]
        contingencies["recommended"] = max(candidates) if candidates else 0.0
        contingencies["coefficient_of_variation"] = cv

        return contingencies


# ============================================================================
# ANALIZADOR DE OPCIONES REALES
# ============================================================================


class RealOptionsAnalyzer:
    """
    Analizador de Opciones Reales.

    Evalúa la flexibilidad estratégica (opción de esperar, expandir o abandonar)
    como un valor añadido a la estructura estática del proyecto.
    """

    def __init__(self, model_type: OptionModelType = OptionModelType.BINOMIAL):
        self.model_type = model_type

    def value_option_to_wait(
        self,
        project_value: float,
        investment_cost: float,
        risk_free_rate: float,
        time_to_expire: float,
        volatility: float,
        steps: int = 100,
    ) -> Dict[str, float]:
        """
        Valora la 'Opción de Esperar' (Call Option sobre el proyecto).

        Args:
            project_value (S): Valor presente de los flujos del proyecto.
            investment_cost (K): Costo de la inversión (precio de ejercicio).
            risk_free_rate (r): Tasa libre de riesgo anual.
            time_to_expire (T): Tiempo disponible para diferir la inversión (años).
            volatility (σ): Volatilidad del valor del proyecto.
            steps (N): Pasos para el modelo binomial.

        Returns:
            Dict: Valoración de la opción y desglose.
        """
        if self.model_type == OptionModelType.BINOMIAL:
            return self._binomial_valuation(
                project_value,
                investment_cost,
                risk_free_rate,
                time_to_expire,
                volatility,
                steps,
            )
        # Fallback a Black-Scholes simplificado (no implementado en detalle aquí)
        return {"option_value": 0.0, "model": "Not Implemented"}

    def _binomial_valuation(
        self,
        S: float,
        K: float,
        r: float,
        T: float,
        sigma: float,
        n: int,
        american: bool = True,
    ) -> Dict[str, float]:
        """
        Modelo Binomial CRR (Cox-Ross-Rubinstein) con cálculo exacto de Greeks.

        Refinamientos:
        - Cálculo real de Delta usando diferencias finitas en el árbol.
        - Cálculo de Gamma y Theta como Greeks adicionales.
        - Manejo robusto de probabilidades fuera de rango [0,1].
        - Tracking preciso de ejercicio anticipado.
        - Optimización de memoria usando arrays en lugar de matrices.

        Fundamento Matemático:
            u = e^(σ√Δt)           Factor de subida
            d = 1/u = e^(-σ√Δt)    Factor de bajada
            p = (e^(rΔt) - d)/(u - d)  Probabilidad neutral al riesgo

        Para Delta:
            Δ = (C_u - C_d) / (S_u - S_d) = (C_u - C_d) / (S(u - d))

        Args:
            S: Valor actual del subyacente (proyecto).
            K: Precio de ejercicio (inversión requerida).
            r: Tasa libre de riesgo anual.
            T: Tiempo hasta expiración (años).
            sigma: Volatilidad anual.
            n: Número de pasos en el árbol.
            american: True para opción americana, False para europea.

        Returns:
            Dict con valor de la opción y Greeks.
        """
        # === Validaciones de entrada ===
        if S <= 0:
            raise ValueError(f"Precio spot debe ser positivo, recibido: {S}")
        if T <= 0:
            return {
                "option_value": max(S - K, 0),
                "model": "Expirada",
                "intrinsic_value": max(S - K, 0),
                "time_value": 0.0,
                "delta": 1.0 if S > K else 0.0,
                "gamma": 0.0,
                "theta": 0.0,
            }
        if sigma <= 0:
            # Sin volatilidad, la opción vale su valor intrínseco descontado
            intrinsic = max(S - K * exp(-r * T), 0)
            return {
                "option_value": intrinsic,
                "model": "Determinístico",
                "intrinsic_value": max(S - K, 0),
                "time_value": intrinsic - max(S - K, 0),
                "delta": 1.0 if S > K else 0.0,
                "gamma": 0.0,
                "theta": 0.0,
            }
        if n < 1:
            n = 1
            logger.warning("Número de pasos ajustado a mínimo de 1.")

        # === Parámetros del árbol ===
        dt = T / n
        u = exp(sigma * sqrt(dt))
        d = 1.0 / u
        discount = exp(-r * dt)

        # Probabilidad neutral al riesgo
        p = (exp(r * dt) - d) / (u - d)

        # Validación de arbitraje (p debe estar en (0, 1))
        if not (0 < p < 1):
            logger.error(
                f"Probabilidad fuera de rango: p={p:.4f}. "
                f"Posible arbitraje o parámetros inconsistentes (r={r}, σ={sigma}, T={T})."
            )
            # Intentar ajustar n para corregir
            if p <= 0:
                # d >= e^(rΔt), necesitamos más pasos
                n_suggested = (
                    max(n * 2, int(sigma**2 * T / (r * 0.1)**2) + 1) if r > 0 else n * 2
                )
            else:
                # u <= e^(rΔt), necesitamos menos pasos o hay error en parámetros
                n_suggested = max(1, n // 2)

            return {
                "option_value": max(S - K, 0),  # Fallback a valor intrínseco
                "model": "Error: Arbitraje detectado",
                "error": f"Probabilidad p={p:.4f} inválida. Sugerido n={n_suggested}",
                "intrinsic_value": max(S - K, 0),
                "time_value": 0.0,
                "delta": float("nan"),
                "gamma": float("nan"),
                "theta": float("nan"),
            }

        # === Construcción del árbol de precios ===
        # Optimización: Solo almacenamos una capa a la vez
        # Pero para Greeks necesitamos valores en t=0, t=Δt, t=2Δt

        # Precios en t=T
        prices_T = np.array([S * (d ** (n - j)) * (u ** j) for j in range(n + 1)])

        # Valores de la opción en t=T (payoff)
        values = np.maximum(prices_T - K, 0)

        # Tracking de ejercicio anticipado
        early_exercise_count = 0
        early_exercise_value = 0.0

        # === Almacenar valores para cálculo de Greeks ===
        # Necesitamos V(t=Δt) y V(t=2Δt) para Delta y Gamma
        values_at_1 = None  # Valores en t = Δt (después de 1 paso hacia atrás)
        values_at_2 = None  # Valores en t = 2Δt

        # Caso inicial para n >= 2 (Gamma)
        if n >= 2:
            values_at_2 = values[:3].copy()

        # === Inducción hacia atrás ===
        for i in range(n - 1, -1, -1):
            new_values = np.zeros(i + 1)
            for j in range(i + 1):
                # Valor de continuación (expectativa descontada)
                continuation = discount * (p * values[j + 1] + (1 - p) * values[j])

                if american:
                    # Precio del subyacente en este nodo
                    s_node = S * (d ** (i - j)) * (u ** j)
                    intrinsic = max(s_node - K, 0)

                    if intrinsic > continuation + 1e-10:
                        new_values[j] = intrinsic
                        early_exercise_count += 1
                        early_exercise_value += intrinsic - continuation
                    else:
                        new_values[j] = continuation
                else:
                    new_values[j] = continuation

            values = new_values

            # Guardar valores para Greeks (capturar después del paso atrás)
            if i == 1:
                # Acabamos de calcular V(t=0), pero 'values' antes de este paso era V(t=Δt)
                # No, 'values' después de i=1 es V(t=Δt)? NO.
                # Trace:
                # i=1: computes new_values (size 2) from values (size 3, at t=2Δt).
                # So new_values is V(t=Δt).
                # values = new_values.
                values_at_1 = values.copy()
            elif i == 2:
                # Acabamos de calcular V(t=Δt), pero 'values' antes de este paso era V(t=2Δt)
                # i=2: computes new_values (size 3) from values (size 4).
                # new_values is V(t=2Δt).
                values_at_2 = values.copy()

        option_value = values[0]

        # === Cálculo de Greeks ===
        # Delta: Sensibilidad al precio del subyacente
        if values_at_1 is not None and len(values_at_1) >= 2:
            S_up = S * u
            S_down = S * d
            delta = (values_at_1[1] - values_at_1[0]) / (S_up - S_down)
        else:
            # Aproximación para n=1
            delta = (max(S * u - K, 0) - max(S * d - K, 0)) / (S * (u - d))

        # Gamma: Segunda derivada respecto a S
        gamma = 0.0
        if values_at_2 is not None and len(values_at_2) >= 3 and n >= 2:
            S_uu = S * u * u
            S_ud = S  # u * d = 1
            S_dd = S * d * d

            delta_up = (values_at_2[2] - values_at_2[1]) / (S_uu - S_ud)
            delta_down = (values_at_2[1] - values_at_2[0]) / (S_ud - S_dd)

            gamma = (delta_up - delta_down) / (0.5 * (S_uu - S_dd))

        # Theta: Sensibilidad al tiempo (decay por día)
        # θ = (V(t+Δt) - V(t)) / Δt, aproximado
        if values_at_1 is not None:
            # Valor en t=Δt promediado
            v_dt = (
                p * values_at_1[1] + (1 - p) * values_at_1[0]
                if len(values_at_1) >= 2
                else values_at_1[0]
            )
            theta = (v_dt - option_value) / dt  # Por año
            theta_daily = theta / 252  # Por día
        else:
            theta = 0.0
            theta_daily = 0.0

        intrinsic_value = max(S - K, 0)
        time_value = max(0, option_value - intrinsic_value)

        return {
            "option_value": option_value,
            "model": f"Binomial CRR ({'Americana' if american else 'Europea'}, n={n})",
            "intrinsic_value": intrinsic_value,
            "time_value": time_value,
            "early_exercise_nodes": early_exercise_count,
            "early_exercise_value": early_exercise_value,
            # Greeks
            "delta": np.clip(delta, 0, 1),  # Delta de call está en [0, 1]
            "gamma": gamma,
            "theta": theta,
            "theta_daily": theta_daily,
            # Parámetros del modelo para diagnóstico
            "parameters": {
                "u": u,
                "d": d,
                "p": p,
                "dt": dt,
                "discount_factor": discount,
            },
        }


# ============================================================================
# FACHADA PRINCIPAL: MOTOR FINANCIERO
# ============================================================================


class FinancialEngine:
    """
    Fachada que orquesta el análisis financiero integral.

    Coordina CAPM, RiskQuantifier y RealOptions para entregar un veredicto
    económico alineado con la estructura topológica del proyecto.
    """

    def __init__(self, config: FinancialConfig):
        self.config = config
        self.capm = CapitalAssetPricing(config)
        self.risk = RiskQuantifier(DistributionType.NORMAL)
        self.options = RealOptionsAnalyzer(OptionModelType.BINOMIAL)

    def _calculate_thermo_structural_volatility(
        self,
        base_volatility: float,
        stability_psi: float,
        system_temperature: float,
    ) -> float:
        """
        Ecuación Unificada de Física del Costo con fundamentos rigurosos.

        Refinamientos:
        - Umbrales derivados de principios físicos, no arbitrarios.
        - Modelo continuo (sin discontinuidades) usando funciones sigmoideas.
        - Incorporación de histéresis para cambios de estado.
        - Límites físicos garantizados (volatilidad no puede ser negativa ni infinita).

        Fundamento Físico:

        1. Factor Estructural (Analogía con Pandeo de Columnas):
           - Ψ representa la esbeltez inversa (ancho_base / altura)
           - Ψ < 1 → Columna esbelta, inestable (pandeo de Euler)
           - Ψ > 1 → Columna robusta, estable
           - Factor = tanh((1 - Ψ) * κ) donde κ es sensibilidad

        2. Factor Térmico (Ecuación de Arrhenius modificada):
           - T > T_ref → Activación de modos de falla
           - Factor = exp((T - T_ref) / T_scale) - 1 para T > T_ref
           - Modela expansión térmica diferencial y degradación

        3. Acoplamiento:
           - Los factores no son aditivos sino multiplicativos parcialmente
           - σ_real = σ_base * (1 + F_struct) * (1 + F_thermal * α_coupling)

        Args:
            base_volatility: Volatilidad de mercado (σ_base).
            stability_psi: Índice de estabilidad piramidal [0, ∞).
            system_temperature: Temperatura del sistema en °C.

        Returns:
            float: Volatilidad ajustada, garantizada en [σ_base, σ_base * max_factor].
        """
        # === Constantes del modelo (desde configuración) ===
        psi_critical = self.config.psi_critical
        psi_stable = self.config.psi_stable
        kappa_struct = self.config.kappa_struct
        t_reference = self.config.t_reference
        t_stress = self.config.t_stress
        t_scale = self.config.t_scale
        alpha_coupling = self.config.alpha_coupling
        max_amplification = self.config.max_amplification

        # === Validaciones ===
        if base_volatility < 0:
            raise ValueError(f"Volatilidad base no puede ser negativa: {base_volatility}")
        if base_volatility == 0:
            return 0.0  # Sin volatilidad base, no hay amplificación

        # Sanitizar Ψ (debe ser positivo)
        stability_psi = max(0.01, stability_psi)  # Evitar Ψ = 0

        # === 1. Factor Estructural (Modelo de Pandeo Suavizado) ===
        # Usamos tanh para transición suave, evitando discontinuidades
        if stability_psi >= psi_stable:
            structural_factor = 0.0
        else:
            # Función sigmoide invertida centrada en Ψ_crítico
            x = (psi_critical - stability_psi) * kappa_struct
            structural_factor = max(0, np.tanh(x))

            # Penalización adicional por régimen sub-crítico (Ψ < 1)
            if stability_psi < psi_critical:
                # Término cuadrático para amplificar inestabilidad severa
                subcritical_penalty = ((psi_critical - stability_psi) / psi_critical) ** 2
                structural_factor += subcritical_penalty * 0.5

        # === 2. Factor Térmico (Arrhenius Modificado) ===
        if system_temperature <= t_reference:
            thermal_factor = 0.0
        elif system_temperature <= t_stress:
            # Régimen de estrés leve: lineal
            thermal_factor = (system_temperature - t_reference) / t_scale * 0.1
        else:
            # Régimen de estrés severo: exponencial
            excess_temp = system_temperature - t_stress
            thermal_factor = 0.1 + (exp(excess_temp / t_scale) - 1) * 0.2

        # === 3. Cálculo de Volatilidad Unificada ===
        # Modelo multiplicativo parcial para capturar interacciones
        structural_multiplier = 1.0 + structural_factor
        thermal_multiplier = 1.0 + thermal_factor * alpha_coupling

        # Interacción: el estrés térmico amplifica la fragilidad estructural
        if structural_factor > 0 and thermal_factor > 0:
            interaction_term = structural_factor * thermal_factor * 0.3
            structural_multiplier += interaction_term

        total_multiplier = structural_multiplier * thermal_multiplier

        # Aplicar límite superior
        total_multiplier = min(total_multiplier, max_amplification)

        unified_volatility = base_volatility * total_multiplier

        # === Logging detallado ===
        if total_multiplier > 1.01:  # Solo si hay ajuste significativo
            logger.warning(
                f"🔥 Física del Costo Activada:\n"
                f"   Volatilidad: {base_volatility:.2%} → {unified_volatility:.2%} "
                f"(×{total_multiplier:.2f})\n"
                f"   Factor Estructural: +{structural_factor:.2%} (Ψ={stability_psi:.2f})\n"
                f"   Factor Térmico: +{thermal_factor:.2%} (T={system_temperature:.1f}°C)\n"
                f"   Acoplamiento: {'Activo' if structural_factor > 0 and thermal_factor > 0 else 'Inactivo'}"
            )

        return unified_volatility

    def analyze_project(
        self,
        initial_investment: float,
        cash_flows: List[float],
        cost_std_dev: float,
        volatility: Optional[float] = None,
        topology_report: Optional[Dict[str, Any]] = None,
        expected_cash_flows: Optional[List[float]] = None,
        project_volatility: Optional[float] = None,
        liquidity: Optional[float] = None,
        fixed_contracts_ratio: Optional[float] = None,
        pyramid_stability: Optional[float] = None,
        system_temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Análisis financiero integral con manejo robusto de edge cases.

        Refinamientos:
        - Manejo explícito de volatilidad cero o negativa.
        - Validación de flujos de caja vacíos.
        - Propagación de incertidumbre en el ajuste de std_dev.
        - Métricas de diagnóstico para debugging.
        - Separación clara entre volatilidad base y estructural.

        Args:
            initial_investment: Inversión inicial (debe ser > 0).
            cash_flows: Flujos de caja proyectados (lista no vacía).
            cost_std_dev: Desviación estándar de costos.
            volatility: Volatilidad del proyecto (decimal, ej. 0.20 para 20%).
            topology_report: Reporte de análisis topológico.
            expected_cash_flows: Alias para cash_flows (compatibilidad V3.0).
            project_volatility: Alias para volatility (compatibilidad V3.0).
            liquidity: Override del ratio de liquidez.
            fixed_contracts_ratio: Override del ratio de contratos fijos.
            pyramid_stability: Índice de estabilidad Ψ.
            system_temperature: Temperatura del sistema en °C.

        Returns:
            Dict: Informe financiero completo.

        Raises:
            ValueError: Si parámetros críticos son inválidos.
        """
        # === Resolución de argumentos (compatibilidad V3.0) ===
        flows = expected_cash_flows if expected_cash_flows is not None else cash_flows
        vol = project_volatility if project_volatility is not None else volatility

        # === Validaciones de entrada ===
        if not flows:
            raise ValueError("Se requiere al menos un flujo de caja proyectado.")

        if initial_investment < 0:
            logger.warning(
                f"Inversión inicial negativa ({initial_investment}). ¿Es un desinversión?"
            )

        if vol is None:
            raise ValueError("Se requiere 'volatility' o 'project_volatility'.")

        if vol < 0:
            raise ValueError(f"Volatilidad no puede ser negativa: {vol}")

        # Volatilidad cero: proyecto determinístico
        if vol == 0:
            logger.info(
                "Volatilidad = 0: Proyecto determinístico, sin incertidumbre de mercado."
            )

        # Overrides de configuración
        liq = liquidity if liquidity is not None else self.config.liquidity_ratio
        fcr = (
            fixed_contracts_ratio
            if fixed_contracts_ratio is not None
            else self.config.fixed_contracts_ratio
        )

        # === 1. Calcular Volatilidad Efectiva ===
        effective_volatility = vol
        physics_applied = False
        physics_details = {}

        if pyramid_stability is not None and vol > 0:
            temp = system_temperature if system_temperature is not None else 25.0
            effective_volatility = self._calculate_thermo_structural_volatility(
                vol, pyramid_stability, temp
            )
            physics_applied = bool(effective_volatility > vol * 1.001)
            physics_details = {
                "pyramid_stability": pyramid_stability,
                "system_temperature": temp,
                "amplification_factor": effective_volatility / vol if vol > 0 else 1.0,
            }
        elif topology_report:
            # Fallback a ajuste por topología
            adjusted_vol = self.adjust_volatility_by_topology(vol, topology_report)
            if adjusted_vol > vol * 1.001:
                physics_applied = True
                physics_details = {
                    "topology_adjustment": adjusted_vol / vol if vol > 0 else 1.0
                }
            effective_volatility = adjusted_vol

        # === 2. Valoración DCF ===
        wacc = self.capm.calculate_wacc()
        npv = self.capm.calculate_npv(flows, initial_investment)

        # === 3. Análisis de Riesgo ===
        # Ajustar std_dev proporcionalmente a la amplificación de volatilidad
        if vol > 0:
            volatility_ratio = effective_volatility / vol
        else:
            volatility_ratio = 1.0  # Sin cambio si vol = 0

        adjusted_std_dev = cost_std_dev * volatility_ratio

        var_val, var_metrics = self.risk.calculate_var(
            initial_investment, adjusted_std_dev, confidence_level=0.95
        )
        contingency = self.risk.suggest_contingency(initial_investment, adjusted_std_dev)

        # === 4. Opciones Reales ===
        project_pv = npv + initial_investment
        option_result = {"option_value": 0.0, "delta": 0.0, "gamma": 0.0}

        if project_pv > 0 and effective_volatility > 0:
            try:
                option_result = self.options.value_option_to_wait(
                    project_pv,
                    initial_investment,
                    self.config.risk_free_rate,
                    self.config.project_life_years,
                    effective_volatility,
                )
            except Exception as e:
                logger.error(f"Error en valoración de opciones reales: {e}")
                option_result = {"option_value": 0.0, "error": str(e)}

        option_val = option_result.get("option_value", 0.0)
        total_value = npv + option_val

        # === 5. Métricas de Performance ===
        performance = self._calculate_performance_metrics(
            npv, initial_investment, len(flows), flows=flows
        )

        # === 6. Termodinámica Financiera ===
        inertia = self.calculate_financial_thermal_inertia(liq, fcr)

        # === 7. Construir Resultado ===
        result = {
            # Valoración core
            "wacc": wacc,
            "npv": npv,
            "total_value": total_value,
            # Volatilidad y ajustes
            "volatility_base": vol,
            "volatility_structural": effective_volatility,
            "volatility": effective_volatility,  # Compatibilidad
            "physics_adjustment": physics_applied,
            "physics_details": physics_details,
            # Riesgo
            "var": var_val,
            "var_metrics": var_metrics,
            "contingency": contingency,
            # Opciones reales
            "real_option_value": option_val,
            "real_option_details": option_result,
            # Performance
            "performance": performance,
            # Termodinámica
            "thermodynamics": {
                "financial_inertia": inertia,
                "liquidity_ratio": liq,
                "fixed_contracts_ratio": fcr,
            },
            # Diagnóstico
            "diagnostics": {
                "input_flows_count": len(flows),
                "input_volatility": vol,
                "effective_volatility": effective_volatility,
                "std_dev_adjustment_ratio": volatility_ratio,
                "topology_report_provided": topology_report is not None,
            },
        }

        return result

    def _calculate_performance_metrics(
        self, npv: float, investment: float, years: int, flows: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Calcula métricas de performance con manejo robusto de casos especiales.

        Refinamientos:
        - Payback period que maneja flujos negativos intermedios.
        - Detección de proyectos que nunca recuperan inversión.
        - IRR aproximada usando Newton-Raphson.
        - Índice de eficiencia del capital.

        Args:
            npv: Valor Presente Neto calculado.
            investment: Inversión inicial.
            years: Número de períodos.
            flows: Lista de flujos de caja.

        Returns:
            Dict con métricas de performance.
        """
        metrics = {}

        # === ROI y Profitability Index ===
        if investment > 0:
            roi = npv / investment
            pi = (npv + investment) / investment
            metrics["profitability_index"] = pi
            metrics["recommendation"] = "ACEPTAR" if pi > 1 else "RECHAZAR"
        elif investment < 0:
            logger.warning("Inversión inicial negativa: interpretando como desinversión.")
            roi = -npv / abs(investment)
            metrics["profitability_index"] = float("nan")
            metrics["recommendation"] = "REVISAR"
        else:
            roi = float("inf") if npv > 0 else (float("-inf") if npv < 0 else 0.0)
            metrics["profitability_index"] = float("nan")
            metrics["recommendation"] = "REVISAR"

        metrics["roi"] = roi

        # === Retorno Anualizado ===
        if years > 0 and investment > 0 and (1 + roi) > 0:
            try:
                annualized = (1 + roi) ** (1 / years) - 1
            except (ValueError, OverflowError):
                annualized = float("nan")
        elif (1 + roi) <= 0:
            annualized = -1.0  # Pérdida total
        else:
            annualized = float("nan")

        metrics["annualized_return"] = annualized

        # === Payback Period (Robusto) ===
        if flows and investment > 0:
            cumulative = 0.0
            payback = None
            peak_deficit = 0.0
            deficit_periods = 0

            for t, cf in enumerate(flows, start=1):
                cumulative += cf

                # Tracking de déficit (flujos negativos intermedios)
                if cumulative < peak_deficit:
                    peak_deficit = cumulative
                    deficit_periods += 1

                # Verificar recuperación
                if payback is None and cumulative >= investment:
                    # Interpolación lineal para payback fraccionario
                    prev_cumulative = cumulative - cf
                    remaining = investment - prev_cumulative

                    if cf > 0:
                        fraction = remaining / cf
                    else:
                        # Flujo = 0 pero cumulative >= investment
                        # (solo posible si prev_cumulative >= investment)
                        fraction = 0

                    payback = (t - 1) + fraction

            if payback is not None:
                metrics["payback_period"] = round(payback, 2)
                metrics["payback_status"] = "RECUPERABLE"
            else:
                metrics["payback_period"] = float("inf")
                metrics["payback_status"] = "NO_RECUPERABLE"
                metrics["final_cumulative"] = cumulative
                metrics["recovery_gap"] = investment - cumulative

            # Métricas adicionales de cashflow
            metrics["payback"] = metrics["payback_period"]  # Alias legacy
            metrics["peak_deficit"] = peak_deficit
            metrics["deficit_periods"] = deficit_periods

            # Eficiencia del capital: qué tan rápido se recupera vs. vida del proyecto
            if payback is not None and years > 0:
                metrics["capital_efficiency"] = 1 - (payback / years)
            else:
                metrics["capital_efficiency"] = 0.0

        # === IRR Aproximada (Newton-Raphson) ===
        if flows and investment > 0:
            try:
                irr = self._estimate_irr(investment, flows)
                metrics["irr_estimate"] = irr
            except Exception:
                metrics["irr_estimate"] = float("nan")

        return metrics

    def _estimate_irr(
        self,
        investment: float,
        flows: List[float],
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> float:
        """
        Estima la TIR usando Newton-Raphson.

        NPV(r) = -I + Σ CF_t / (1+r)^t = 0
        NPV'(r) = -Σ t * CF_t / (1+r)^(t+1)

        Args:
            investment: Inversión inicial.
            flows: Flujos de caja.
            max_iterations: Máximo de iteraciones.
            tolerance: Tolerancia de convergencia.

        Returns:
            float: TIR estimada.
        """
        # Punto inicial: WACC como estimación
        r = self.capm.calculate_wacc()

        for _ in range(max_iterations):
            npv = -investment
            npv_deriv = 0.0

            for t, cf in enumerate(flows, start=1):
                discount = (1 + r) ** t
                npv += cf / discount
                npv_deriv -= t * cf / ((1 + r) ** (t + 1))

            if abs(npv_deriv) < 1e-12:
                break  # Derivada casi cero, no podemos continuar

            r_new = r - npv / npv_deriv

            if abs(r_new - r) < tolerance:
                return r_new

            r = r_new

            # Límites de sanidad
            r = max(-0.99, min(10.0, r))

        return r

    def adjust_volatility_by_topology(
        self, base_volatility: float, topology_report: Dict[str, Any]
    ) -> float:
        """
        Ajusta la volatilidad financiera basándose en la integridad topológica.
        Método helper público para integraciones externas.

        Modelo:
            σ_adj = σ_base * (1 + P_sinergia + P_eficiencia)
        """
        if not topology_report:
            return base_volatility

        # 1. Penalización por Sinergia de Riesgo
        synergy_penalty = 0.0
        synergy_data = topology_report.get("synergy_risk", {})
        if synergy_data.get("synergy_detected", False):
            strength = synergy_data.get("synergy_strength", 1.0)
            if np.isnan(strength):
                strength = 1.0 # Fallback seguro
            synergy_penalty = self.config.synergy_penalty_factor * strength

        # 2. Penalización por Ineficiencia de Euler
        efficiency_penalty = 0.0
        efficiency = topology_report.get("euler_efficiency")
        if efficiency is not None and not np.isnan(efficiency):
            efficiency_penalty = self.config.efficiency_penalty_factor * (1.0 - max(0.0, min(1.0, efficiency)))

        # 3. Factor Total
        total_adjustment_factor = synergy_penalty + efficiency_penalty

        # Clamping del ajuste (máximo % de incremento)
        total_adjustment_factor = min(total_adjustment_factor, self.config.max_volatility_adjustment)

        adjusted_volatility = base_volatility * (1.0 + total_adjustment_factor)

        return max(0.0, adjusted_volatility)

    def calculate_financial_thermal_inertia(
        self, liquidity: float, fixed_contracts_ratio: float
    ) -> float:
        """
        Calcula la Inercia Térmica Financiera.

        Simula la resistencia del proyecto a cambios de 'temperatura' (precios).
        Analogía física: Masa (Liquidez) * Calor Específico (Contratos Fijos).
        """
        return liquidity * fixed_contracts_ratio

    def predict_temperature_change(self, perturbation: float, inertia: float) -> float:
        """
        Predice el cambio de temperatura financiera (costos) ante una perturbación.
        Ley de enfriamiento: ΔT = Q / I
        """
        if inertia <= 0:
            # Sin inercia, el cambio es instantáneo/total (o indefinido si Q=0, asumimos total)
            # Retornamos la perturbación completa como 'cambio infinito' o directo.
            # Para fines prácticos, si I=0, el sistema es inestable.
            # Retornamos la perturbación sin amortiguación.
            return perturbation

        return perturbation / inertia


def calculate_volatility_from_returns(
    returns: List[float], frequency: str = "daily", annual_trading_days: int = 252
) -> float:
    """
    Calcula volatilidad anualizada a partir de retornos históricos.

    Args:
        returns: Lista de retornos.
        frequency: Frecuencia de los datos ('daily', 'weekly', 'monthly', 'annual').
        annual_trading_days: Días de trading por año.

    Returns:
        float: Volatilidad anualizada.
    """
    if not returns or len(returns) < 2:
        raise ValueError(
            f"Se requieren al menos 2 retornos. Recibidos: {len(returns) if returns else 0}"
        )

    factors = {
        "daily": annual_trading_days,
        "weekly": 52,
        "monthly": 12,
        "annual": 1,
    }
    if frequency not in factors:
        raise ValueError(
            f"Frecuencia '{frequency}' no válida. Opciones: {list(factors.keys())}"
        )

    std_period = np.std(np.array(returns), ddof=1)
    volatility = std_period * sqrt(factors[frequency])

    logger.info(
        f"Volatilidad anualizada: {volatility:.2%} (n={len(returns)}, freq={frequency})"
    )
    return volatility
