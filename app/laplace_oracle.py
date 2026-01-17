import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

import scipy.signal


class ConfigurationError(Exception):
    """Indica un problema con la configuración del sistema."""
    pass


# ============================================================================
# ORÁCULO DE LAPLACE - DOMINIO DE LA FRECUENCIA
# ============================================================================
class LaplaceOracle:
    """
    Analizador de estabilidad en el dominio de Laplace con capacidades extendidas.

    Características principales:
    1. Análisis de estabilidad absoluta y relativa
    2. Cálculo de márgenes de estabilidad (ganancia, fase)
    3. Diagramas de Nyquist y Bode numéricos
    4. Análisis de sensibilidad paramétrica
    5. Métricas de respuesta transitoria
    6. Validación rigurosa de casos límite

    Sistema RLC de segundo orden:
        H(s) = 1 / (L*C*s² + R*C*s + 1)

    Transformación a forma canónica:
        H(s) = ωₙ² / (s² + 2ζωₙs + ωₙ²)
        donde:
            ωₙ = 1/√(LC)  (frecuencia natural)
            ζ = R/(2) * √(C/L)  (factor de amortiguamiento)
    """

    def __init__(self, R: float, L: float, C: float, sample_rate: float = 1000.0):
        """
        Inicializa el analizador con validación rigurosa.

        Args:
            R: Resistencia (Ω) - debe ser ≥ 0
            L: Inductancia (H) - debe ser > 0
            C: Capacitancia (F) - debe ser > 0
            sample_rate: Frecuencia de muestreo para análisis discreto (Hz)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Validación paramétrica exhaustiva
        self._validate_parameters(R, L, C)

        # Parámetros físicos
        self.R = float(R)
        self.L = float(L)
        self.C = float(C)

        # Parámetros derivados
        self.omega_n = 1.0 / math.sqrt(self.L * self.C) if self.L * self.C > 0 else 0.0
        self.zeta = (self.R / 2.0) * math.sqrt(self.C / self.L) if self.L > 0 else 0.0
        self.Q = 1.0 / (2.0 * self.zeta) if self.zeta > 0 else float('inf')

        # Frecuencia de muestreo para análisis discreto
        self.sample_rate = float(sample_rate)
        self.T = 1.0 / self.sample_rate  # Período de muestreo

        # Construir sistema continuo
        try:
            num = [1.0]
            den = [self.L * self.C, self.R * self.C, 1.0]
            self.continuous_system = scipy.signal.TransferFunction(num, den)
        except Exception as e:
            raise ConfigurationError(f"Error construyendo sistema continuo: {e}")

        # Sistema discreto (Transformación bilineal Tustin)
        self.discrete_system = self._compute_discrete_system()

        # Clasificación del sistema
        self._classify_system()

        # Cache para resultados computacionalmente costosos
        self._analysis_cache: Dict[str, Any] = {}

    def _validate_parameters(self, R: float, L: float, C: float) -> None:
        """
        Validación exhaustiva de parámetros físicos.

        Incluye:
        1. Validez numérica (finito, no NaN)
        2. Rango físico (positividad)
        3. Consistencia dimensional
        4. Advertencias para valores extremos
        """
        errors = []
        warnings = []

        # Validación básica
        for name, value, min_val in [("R", R, 0.0), ("L", L, 1e-12), ("C", C, 1e-12)]:
            if not math.isfinite(value):
                errors.append(f"{name} debe ser un número finito, got {value}")
            elif value < min_val:
                errors.append(f"{name} debe ser ≥ {min_val}, got {value}")

        if errors:
            raise ConfigurationError(
                "Parámetros físicos inválidos:\n" + "\n".join(f"  • {e}" for e in errors)
            )

        # Verificaciones de consistencia física
        if L > 0 and C > 0:
            omega_n = 1.0 / math.sqrt(L * C)
            if omega_n > 1e9:  # > 1 GHz
                warnings.append(f"Frecuencia natural muy alta: {omega_n:.2e} rad/s")

            # Constante de tiempo del sistema
            if R > 0:
                tau_dominant = 2 * L / R if (R**2 * C) < (4 * L) else R * C
                if tau_dominant < 1e-9:
                    warnings.append(f"Constante de tiempo muy corta: {tau_dominant:.2e} s")

        # Verificación de resonancia peligrosa
        if R > 0 and L > 0 and C > 0:
            damping_ratio = (R / 2.0) * math.sqrt(C / L)
            if damping_ratio < 0.01:
                warnings.append("Sistema casi sin amortiguamiento (ζ < 0.01) - riesgo de oscilaciones")
            elif damping_ratio > 10:
                warnings.append("Sistema sobreamortiguado extremo (ζ > 10) - respuesta muy lenta")

        # Emitir warnings
        for warning in warnings:
            self.logger.warning(f"⚠️ Validación de parámetros: {warning}")

    def _classify_system(self) -> None:
        """Clasifica el sistema según su factor de amortiguamiento."""
        if self.zeta < 0:
            self.damping_class = "NEGATIVE_DAMPING"
            self.stability_class = "UNSTABLE"
            self.response_type = "DIVERGENT"
        elif abs(self.zeta) < 1e-10:
            self.damping_class = "UNDAMPED"
            self.stability_class = "MARGINALLY_STABLE"
            self.response_type = "OSCILLATORY"
        elif self.zeta < 1.0:
            self.damping_class = "UNDERDAMPED"
            self.stability_class = "STABLE"
            self.response_type = "OSCILLATORY_DECAY"
        elif abs(self.zeta - 1.0) < 1e-6:
            self.damping_class = "CRITICALLY_DAMPED"
            self.stability_class = "STABLE"
            self.response_type = "FASTEST_SETTLING"
        else:
            self.damping_class = "OVERDAMPED"
            self.stability_class = "STABLE"
            self.response_type = "EXPONENTIAL_DECAY"

    def _compute_discrete_system(self):
        """
        Convierte el sistema continuo a discreto usando transformación bilineal (Tustin).

        Transformación: s = (2/T) · (z-1)/(z+1)

        Para H(s) = 1 / (a₂s² + a₁s + a₀), la sustitución directa produce:

            H(z) = T²(z+1)² / [4a₂(z-1)² + 2a₁T(z-1)(z+1) + a₀T²(z+1)²]

        Expandiendo y agrupando por potencias de z, obtenemos coeficientes
        que preservan la estabilidad y mapean correctamente el eje imaginario
        al círculo unitario.

        Nota: Para frecuencias críticas, considerar pre-warping:
            ω_d = (2/T) · tan(ω_a · T/2)
        """
        T = self.T

        # Coeficientes del denominador continuo: a₂s² + a₁s + a₀
        a2 = self.L * self.C
        a1 = self.R * self.C
        a0 = 1.0

        # Validación de coeficientes
        if a2 < 1e-15:
            self.logger.warning("Sistema degenerado (LC ≈ 0), retornando sistema continuo")
            return self.continuous_system

        k = 2.0 / T  # Factor de transformación bilineal

        # ══════════════════════════════════════════════════════════════════
        # DERIVACIÓN DEL DENOMINADOR DISCRETO
        # ══════════════════════════════════════════════════════════════════
        # D(s) = a₂s² + a₁s + a₀
        #
        # Sustituyendo s = k(z-1)/(z+1):
        #   s² = k²(z-1)²/(z+1)²
        #   s  = k(z-1)/(z+1)
        #
        # Multiplicando por (z+1)²:
        #   D(z)·(z+1)² = a₂k²(z-1)² + a₁k(z-1)(z+1) + a₀(z+1)²
        #
        # Expandiendo:
        #   a₂k²(z² - 2z + 1) + a₁k(z² - 1) + a₀(z² + 2z + 1)
        #   = (a₂k² + a₁k + a₀)z² + (-2a₂k² + 2a₀)z + (a₂k² - a₁k + a₀)
        # ══════════════════════════════════════════════════════════════════

        k2 = k * k
        den_z2 = a2 * k2 + a1 * k + a0
        den_z1 = -2.0 * a2 * k2 + 2.0 * a0
        den_z0 = a2 * k2 - a1 * k + a0

        # Numerador: proviene del numerador continuo (1) multiplicado por (z+1)²
        # N(z) = (z+1)² = z² + 2z + 1
        num_z2 = 1.0
        num_z1 = 2.0
        num_z0 = 1.0

        # Normalizar para coeficiente líder unitario
        if abs(den_z2) < 1e-15:
            self.logger.warning("Denominador líder degenerado en discretización")
            return self.continuous_system

        # Coeficientes normalizados
        num = [num_z2 / den_z2, num_z1 / den_z2, num_z0 / den_z2]
        den = [1.0, den_z1 / den_z2, den_z0 / den_z2]

        # Verificación de estabilidad del sistema discreto
        # Los polos deben estar dentro del círculo unitario
        poly_coeffs = [1.0, den[1], den[2]]
        roots = np.roots(poly_coeffs)
        max_pole_magnitude = max(abs(r) for r in roots) if len(roots) > 0 else 0

        if max_pole_magnitude > 1.0 + 1e-6:
            self.logger.warning(
                f"Discretización produjo sistema inestable: |p_max| = {max_pole_magnitude:.6f}. "
                f"Considere aumentar sample_rate (actual: {self.sample_rate} Hz)"
            )

        try:
            return scipy.signal.TransferFunction(num, den, dt=T)
        except Exception as e:
            self.logger.warning(f"Error en discretización: {e}")
            return self.continuous_system

    def analyze_stability(self) -> Dict[str, Any]:
        """
        Análisis completo de estabilidad.

        Returns:
            Dict con:
            - estabilidad_absoluta (BIBO)
            - clasificación de amortiguamiento
            - polos y ceros (continuos y discretos)
            - márgenes de estabilidad
            - métricas de respuesta
        """
        if "stability" in self._analysis_cache:
            return self._analysis_cache["stability"]

        # Polos y ceros del sistema continuo
        poles_c = self.continuous_system.poles
        zeros_c = self.continuous_system.zeros

        # Polos y ceros del sistema discreto
        poles_d = self.discrete_system.poles
        zeros_d = self.discrete_system.zeros

        # Verificación de estabilidad BIBO
        epsilon = 1e-12
        unstable_poles_c = [p for p in poles_c if p.real > epsilon]
        marginally_stable_c = [p for p in poles_c if abs(p.real) <= epsilon]

        # Para discreto: dentro del círculo unitario
        unstable_poles_d = [p for p in poles_d if abs(p) > 1.0 + epsilon]
        marginally_stable_d = [p for p in poles_d if abs(abs(p) - 1.0) <= epsilon]

        # Diagnóstico de estabilidad
        is_stable_continuous = len(unstable_poles_c) == 0
        is_stable_discrete = len(unstable_poles_d) == 0

        stability_status = "STABLE"
        if not is_stable_continuous:
            stability_status = "UNSTABLE_CONTINUOUS"
        elif not is_stable_discrete:
            stability_status = "UNSTABLE_DISCRETE"
        elif len(marginally_stable_c) > 0 or len(marginally_stable_d) > 0:
            stability_status = "MARGINALLY_STABLE"

        # Cálculo de márgenes de estabilidad
        margins = self._calculate_stability_margins()

        # Métricas de respuesta transitoria
        transient_metrics = self._calculate_transient_metrics()

        # Análisis de sensibilidad
        sensitivity = self._calculate_parameter_sensitivity()

        result = {
            "status": stability_status,
            "is_stable": is_stable_continuous and is_stable_discrete,
            "is_marginally_stable": stability_status == "MARGINALLY_STABLE",

            # Sistema continuo
            "continuous": {
                "poles": [(p.real, p.imag) for p in poles_c],
                "zeros": [(z.real, z.imag) for z in zeros_c],
                "natural_frequency_rad_s": self.omega_n,
                "damping_ratio": self.zeta,
                "quality_factor": self.Q,
                "damping_class": self.damping_class,
                "response_type": self.response_type,
            },

            # Sistema discreto
            "discrete": {
                "poles": [(p.real, p.imag) for p in poles_d],
                "zeros": [(z.real, z.imag) for z in zeros_d],
                "sample_rate_hz": self.sample_rate,
                "sample_period_s": self.T,
                "is_stable": is_stable_discrete,
            },

            # Métricas de estabilidad
            "stability_margins": margins,

            # Métricas de respuesta
            "transient_response": transient_metrics,

            # Sensibilidad
            "parameter_sensitivity": sensitivity,

            # Información para control
            "control_recommendations": self._generate_control_recommendations(margins, sensitivity),
        }

        self._analysis_cache["stability"] = result
        return result

    def _calculate_stability_margins(self) -> Dict[str, Any]:
        """Calcula márgenes de estabilidad con derivación rigurosa."""
        EPSILON = 1e-10

        if self.zeta <= EPSILON:
            return {
                "gain_margin_db": float('-inf'),
                "phase_margin_deg": 0.0,
                "gain_crossover_freq_rad_s": 0.0,
                "phase_crossover_freq_rad_s": float('inf'),
                "is_margin_meaningful": False,
                "interpretation": "Sistema sin amortiguamiento - marginalmente estable",
            }

        # Para sistemas de segundo orden sin ceros, GM = ∞ (nunca cruza -180° a ganancia finita)
        gain_margin_db = float('inf')

        # Umbral para existencia de cruce de ganancia
        zeta_threshold = 1.0 / math.sqrt(2)  # ≈ 0.7071

        if self.zeta < zeta_threshold:
            # Existe frecuencia de cruce de ganancia
            u = 2.0 - 4.0 * self.zeta**2

            if u > EPSILON:
                omega_gc = self.omega_n * math.sqrt(u)

                # Calcular fase en ω_gc
                w_ratio_sq = u
                w_ratio = math.sqrt(u)

                numerator_angle = 2.0 * self.zeta * w_ratio
                denominator_angle = 1.0 - w_ratio_sq  # = 4ζ² - 1

                # Usar atan2 para manejo correcto de cuadrantes
                phase_rad = -math.atan2(numerator_angle, denominator_angle)

                # Margen de fase: PM = 180° + fase (fase es negativa)
                phase_margin_deg = 180.0 + math.degrees(phase_rad)

                # Fórmula alternativa cerrada para verificación:
                sqrt_term = math.sqrt(math.sqrt(1 + 4*self.zeta**4) - 2*self.zeta**2)
                if sqrt_term > EPSILON:
                    pm_alternative = math.degrees(math.atan(2*self.zeta / sqrt_term))
                    # Usar el más conservador (menor) para robustez
                    phase_margin_deg = min(phase_margin_deg, pm_alternative)
            else:
                omega_gc = self.omega_n
                phase_margin_deg = 90.0
        else:
            # Para ζ ≥ 1/√2: |H(jω)| < 1 ∀ω > 0 (no hay cruce de ganancia)
            omega_gc = self.omega_n

            # Margen de fase efectivo para sistemas bien amortiguados
            if self.zeta >= 1.0:
                # Críticamente amortiguado o sobreamortiguado
                phase_margin_deg = 90.0 + math.degrees(math.atan(self.zeta))
            else:
                # Entre 0.707 y 1.0: interpolación suave
                phase_margin_deg = 45.0 + 45.0 * (self.zeta - 0.5) / 0.5

        # Frecuencia de cruce de fase (donde ∠H = -180°)
        omega_pc = float('inf')

        return {
            "gain_margin_db": gain_margin_db,
            "phase_margin_deg": phase_margin_deg,
            "gain_crossover_freq_rad_s": omega_gc,
            "phase_crossover_freq_rad_s": omega_pc,
            "is_margin_meaningful": self.zeta < zeta_threshold,
            "analysis_notes": self._generate_margin_notes(self.zeta, phase_margin_deg),
            "interpretation": self._interpret_stability_margins(phase_margin_deg, gain_margin_db),
        }

    def _generate_margin_notes(self, zeta: float, pm_deg: float) -> str:
        """Genera notas técnicas sobre los márgenes calculados."""
        if zeta < 0.5:
            return (
                f"Sistema altamente subamortiguado (ζ={zeta:.3f}). "
                f"PM={pm_deg:.1f}° indica susceptibilidad a retardos de fase."
            )
        elif zeta < 1.0 / math.sqrt(2):
            return f"Sistema subamortiguado con cruce de ganancia definido. PM={pm_deg:.1f}°"
        elif zeta < 1.0:
            return (
                f"Sistema subamortiguado (ζ={zeta:.3f}) sin cruce de ganancia formal. "
                f"PM reportado es métrica de referencia en ω=ωₙ."
            )
        else:
            return f"Sistema sobreamortiguado (ζ={zeta:.3f}). Inherentemente robusto."

    def _interpret_stability_margins(self, pm_deg: float, gm_db: float) -> str:
        """Interpreta los márgenes de estabilidad."""
        if pm_deg < 30:
            return "MARGEN DE FASE BAJO - Sistema poco robusto a retardos"
        elif pm_deg > 60:
            return "MARGEN DE FASE ALTO - Sistema robusto pero posiblemente lento"
        else:
            return "MARGEN DE FASE ADECUADO - Buen equilibrio entre rapidez y robustez"

    def _calculate_transient_metrics(self) -> Dict[str, Any]:
        """Calcula métricas de respuesta transitoria para entrada escalón."""
        EPSILON = 1e-10

        # Caso inestable
        if self.zeta < -EPSILON:
            return {
                "status": "UNSTABLE",
                "metrics": {},
                "warning": "Sistema inestable (ζ < 0) - respuesta divergente",
            }

        # Caso sin amortiguamiento (oscilador armónico)
        if abs(self.zeta) < EPSILON:
            period = 2.0 * math.pi / self.omega_n if self.omega_n > 0 else float('inf')
            return {
                "status": "UNDAMPED_OSCILLATION",
                "rise_time_s": period / 4.0,  # Cuarto de período
                "peak_time_s": period / 2.0,
                "overshoot_percent": 100.0,
                "settling_time_s": float('inf'),
                "oscillation_period_s": period,
                "peak_value": 2.0,
                "steady_state_value": 1.0,
                "damped_frequency_rad_s": self.omega_n,
                "warning": "Oscilación sostenida - no alcanza estado estacionario",
            }

        # Caso subamortiguado
        if self.zeta < 1.0 - EPSILON:
            omega_d = self.omega_n * math.sqrt(1.0 - self.zeta**2)

            if 0.3 <= self.zeta <= 0.8:
                rise_time = (1.76 * self.zeta**2 - 0.417 * self.zeta + 1.039) / self.omega_n
            else:
                phi = math.acos(self.zeta)
                rise_time = (math.pi - phi) / omega_d

            peak_time = math.pi / omega_d

            exp_arg = -math.pi * self.zeta / math.sqrt(1.0 - self.zeta**2)
            overshoot_factor = math.exp(exp_arg)
            overshoot_percent = overshoot_factor * 100.0

            tolerance = 0.02
            settling_arg = tolerance * math.sqrt(1.0 - self.zeta**2)
            if settling_arg > 0:
                settling_time = -math.log(settling_arg) / (self.zeta * self.omega_n)
            else:
                settling_time = 4.0 / (self.zeta * self.omega_n)

            return {
                "status": "UNDERDAMPED",
                "rise_time_s": rise_time,
                "peak_time_s": peak_time,
                "overshoot_percent": overshoot_percent,
                "settling_time_s": settling_time,
                "settling_time_5pct_s": settling_time * 0.75,
                "peak_value": 1.0 + overshoot_factor,
                "steady_state_value": 1.0,
                "damped_frequency_rad_s": omega_d,
                "damped_frequency_hz": omega_d / (2.0 * math.pi),
                "number_of_oscillations": settling_time * omega_d / (2.0 * math.pi),
            }

        # Caso críticamente amortiguado
        if abs(self.zeta - 1.0) < EPSILON:
            rise_time = 3.3579 / self.omega_n
            settling_time_2pct = 5.8335 / self.omega_n
            settling_time_5pct = 4.7439 / self.omega_n

            return {
                "status": "CRITICALLY_DAMPED",
                "rise_time_s": rise_time,
                "peak_time_s": float('inf'),
                "overshoot_percent": 0.0,
                "settling_time_s": settling_time_2pct,
                "settling_time_5pct_s": settling_time_5pct,
                "peak_value": 1.0,
                "steady_state_value": 1.0,
                "note": "Respuesta más rápida sin sobrepaso",
            }

        # Caso sobreamortiguado (ζ > 1)
        sqrt_term = math.sqrt(self.zeta**2 - 1.0)
        s1 = -self.omega_n * (self.zeta - sqrt_term)
        s2 = -self.omega_n * (self.zeta + sqrt_term)

        tau_dominant = 1.0 / abs(s1)
        tau_fast = 1.0 / abs(s2)

        rise_time = 2.2 * tau_dominant
        settling_time = 4.0 * tau_dominant
        pole_ratio = abs(s2 / s1)

        return {
            "status": "OVERDAMPED",
            "rise_time_s": rise_time,
            "peak_time_s": float('inf'),
            "overshoot_percent": 0.0,
            "settling_time_s": settling_time,
            "peak_value": 1.0,
            "steady_state_value": 1.0,
            "pole_1_rad_s": s1,
            "pole_2_rad_s": s2,
            "time_constant_dominant_s": tau_dominant,
            "time_constant_fast_s": tau_fast,
            "pole_separation_ratio": pole_ratio,
            "can_approximate_first_order": pole_ratio > 5.0,
        }

    def _calculate_parameter_sensitivity(self) -> Dict[str, Any]:
        """Calcula la sensibilidad de los polos a variaciones paramétricas."""
        if self.zeta < 0 or self.omega_n == 0:
            return {"status": "INVALID_FOR_SENSITIVITY"}

        d_omega_n_dR = 0.0
        d_omega_n_dL = -0.5 * self.omega_n / self.L if self.L != 0 else 0.0
        d_omega_n_dC = -0.5 * self.omega_n / self.C if self.C != 0 else 0.0

        d_zeta_dR = 0.5 * math.sqrt(self.C / self.L) if self.L > 0 else 0.0
        d_zeta_dL = -0.25 * self.R * math.sqrt(self.C) / (self.L**1.5) if self.L > 0 else 0.0
        d_zeta_dC = 0.25 * self.R / (math.sqrt(self.L * self.C)) if self.L > 0 and self.C > 0 else 0.0

        if 0 < self.zeta < 1.0:
            omega_d = self.omega_n * math.sqrt(1 - self.zeta**2)
            s = complex(-self.zeta * self.omega_n, omega_d)
            ds_d_omega_n = complex(-self.zeta, math.sqrt(1 - self.zeta**2))
            ds_d_zeta = complex(-self.omega_n, -self.zeta * self.omega_n / math.sqrt(1 - self.zeta**2))

            ds_dR = ds_d_zeta * d_zeta_dR + ds_d_omega_n * d_omega_n_dR
            ds_dL = ds_d_zeta * d_zeta_dL + ds_d_omega_n * d_omega_n_dL
            ds_dC = ds_d_zeta * d_zeta_dC + ds_d_omega_n * d_omega_n_dC

            sensitivity_R = abs(ds_dR) * (self.R / abs(s)) if abs(s) > 0 else 0.0
            sensitivity_L = abs(ds_dL) * (self.L / abs(s)) if abs(s) > 0 else 0.0
            sensitivity_C = abs(ds_dC) * (self.C / abs(s)) if abs(s) > 0 else 0.0
        else:
            sensitivity_R = abs(d_zeta_dR) * self.R
            sensitivity_L = abs(d_zeta_dL) * self.L
            sensitivity_C = abs(d_zeta_dC) * self.C

        return {
            "sensitivity_to_R": sensitivity_R,
            "sensitivity_to_L": sensitivity_L,
            "sensitivity_to_C": sensitivity_C,
            "most_sensitive_parameter": max(
                ["R", "L", "C"],
                key=lambda p: {"R": sensitivity_R, "L": sensitivity_L, "C": sensitivity_C}[p]
            ),
            "robustness_classification": self._classify_robustness(sensitivity_R, sensitivity_L, sensitivity_C),
        }

    def _classify_robustness(self, sens_R: float, sens_L: float, sens_C: float) -> str:
        """Clasifica la robustez del sistema basado en sensibilidades."""
        max_sens = max(sens_R, sens_L, sens_C)

        if max_sens > 1.0:
            return "FRÁGIL - Alta sensibilidad a variaciones paramétricas"
        elif max_sens > 0.5:
            return "MODERADA - Sensibilidad media, monitorear parámetros"
        elif max_sens > 0.1:
            return "ROBUSTA - Baja sensibilidad a variaciones"
        else:
            return "MUY ROBUSTA - Insensible a variaciones paramétricas"

    def get_frequency_response(self, frequencies: Optional['np.ndarray'] = None) -> Dict[str, Any]:
        """Calcula respuesta en frecuencia del sistema."""
        if frequencies is None:
            w_min = self.omega_n / 1000.0 if self.omega_n > 0 else 0.01
            w_max = self.omega_n * 1000.0 if self.omega_n > 0 else 1000.0
            frequencies = np.logspace(np.log10(w_min), np.log10(w_max), 500)

        w, mag, phase = scipy.signal.bode(self.continuous_system, w=frequencies)
        nyquist_real, nyquist_imag = self._compute_nyquist_diagram(frequencies)
        resonance_freq, resonance_mag = self._find_resonance(w, mag)

        return {
            "frequencies_rad_s": w.tolist(),
            "magnitude_db": mag.tolist(),
            "phase_deg": phase.tolist(),
            "nyquist_real": nyquist_real.tolist(),
            "nyquist_imag": nyquist_imag.tolist(),
            "resonance": {
                "frequency_rad_s": resonance_freq,
                "magnitude_db": resonance_mag,
                "quality_factor": self.Q,
                "bandwidth_rad_s": self._calculate_bandwidth(w, mag),
            },
            "dc_gain_db": mag[0] if len(mag) > 0 else 0.0,
            "high_freq_slope_db_decade": -40.0,
        }

    def _compute_nyquist_diagram(self, frequencies: 'np.ndarray') -> Tuple['np.ndarray', 'np.ndarray']:
        """Calcula diagrama de Nyquist."""
        s = 1j * frequencies
        numerator = 1.0
        denominator = self.L * self.C * s**2 + self.R * self.C * s + 1.0
        H = numerator / denominator
        return H.real, H.imag

    def _find_resonance(self, w: 'np.ndarray', mag: 'np.ndarray') -> Tuple[float, float]:
        """Encuentra frecuencia de resonancia y magnitud pico."""
        if len(mag) == 0:
            return 0.0, 0.0

        if 0 < self.zeta < 1/math.sqrt(2):
            peak_idx = np.argmax(mag)
            return float(w[peak_idx]), float(mag[peak_idx])
        else:
            return float(w[0]), float(mag[0])

    def _calculate_bandwidth(self, w: 'np.ndarray', mag: 'np.ndarray') -> float:
        """Calcula ancho de banda a -3dB con interpolación robusta."""
        if len(mag) == 0 or len(w) == 0:
            return 0.0

        if self.omega_n > 0 and 0 < self.zeta < 2:
            zeta_sq = self.zeta ** 2
            term1 = 1.0 - 2.0 * zeta_sq
            term2 = math.sqrt(4.0 * zeta_sq**2 - 4.0 * zeta_sq + 2.0)
            bw_normalized_sq = term1 + term2

            if bw_normalized_sq > 0:
                return self.omega_n * math.sqrt(bw_normalized_sq)

        dc_gain = mag[0]
        target_gain = dc_gain - 3.0

        for i in range(len(mag) - 1):
            if (mag[i] >= target_gain >= mag[i + 1]) or \
               (mag[i] <= target_gain <= mag[i + 1]):
                mag_diff = mag[i + 1] - mag[i]
                if abs(mag_diff) < 1e-12:
                    continue
                t = (target_gain - mag[i]) / mag_diff
                t = max(0.0, min(1.0, t))
                bandwidth = w[i] + t * (w[i + 1] - w[i])
                return float(bandwidth)

        return float(w[-1])

    def _generate_control_recommendations(self, margins: Dict[str, Any], sensitivity: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones para control basadas en análisis."""
        recommendations = []

        if self.zeta < 0.3:
            recommendations.append(
                "Sistema subamortiguado (ζ < 0.3): Considere aumentar amortiguamiento "
                "o implementar control predictivo para evitar oscilaciones."
            )
        elif self.zeta > 3.0:
            recommendations.append(
                "Sistema sobreamortiguado (ζ > 3): Respuesta lenta. "
                "Considere reducir amortiguamiento para mejor tiempo de respuesta."
            )

        if self.omega_n > 1000:
            recommendations.append(
                "Alta frecuencia natural (>1000 rad/s): Sistema rápido. "
                "Asegurar que frecuencia de muestreo sea suficiente (≥ 10×ω_n)."
            )
        elif self.omega_n < 0.1:
            recommendations.append(
                "Baja frecuencia natural (<0.1 rad/s): Sistema lento. "
                "Considere control integral para eliminar error en estado estacionario."
            )

        pm = margins.get("phase_margin_deg", 0.0)

        if pm < 30:
            recommendations.append(
                f"Margen de fase bajo ({pm:.1f}°). "
                "Aumente amortiguamiento o reduzca ganancia para mejorar robustez."
            )
        elif pm > 80:
            recommendations.append(
                f"Margen de fase muy alto ({pm:.1f}°). "
                "Sistema muy robusto pero posiblemente lento. "
                "Considere aumentar ganancia para mejorar tiempo de respuesta."
            )

        if sensitivity.get("robustness_classification", "").startswith("FRÁGIL"):
            recommendations.append(
                "Alta sensibilidad paramétrica detectada. "
                "Implementar control adaptativo o usar componentes de alta precisión."
            )

        return recommendations

    def get_root_locus_data(self, K_range: Optional['np.ndarray'] = None) -> Dict[str, Any]:
        """Genera datos para lugar de las raíces (root locus)."""
        if K_range is None:
            K_range = np.logspace(-3, 3, 200)

        poles_data = []

        a2 = self.L * self.C
        a1 = self.R * self.C

        if abs(a2) < 1e-15:
            return {
                "error": "Sistema degenerado (LC ≈ 0)",
                "gain_values": [],
                "poles_real": [],
                "poles_imag": [],
            }

        for K in K_range:
            a0_modified = 1.0 + K
            discriminant = a1**2 - 4.0 * a2 * a0_modified

            if discriminant >= 0:
                sqrt_disc = math.sqrt(discriminant)
                pole1 = (-a1 + sqrt_disc) / (2.0 * a2)
                pole2 = (-a1 - sqrt_disc) / (2.0 * a2)
                poles_data.append((K, pole1, 0.0))
                poles_data.append((K, pole2, 0.0))
            else:
                real_part = -a1 / (2.0 * a2)
                imag_part = math.sqrt(-discriminant) / (2.0 * a2)
                poles_data.append((K, real_part, imag_part))
                poles_data.append((K, real_part, -imag_part))

        poles_real = [p[1] for p in poles_data]
        poles_imag = [p[2] for p in poles_data]
        asymptote_center = -a1 / (2.0 * a2)
        asymptote_angles_deg = [90.0, 270.0]
        breakaway = self._find_breakaway_points_refined(K_range)

        K_critical = None
        if self.R > 0:
            K_critical = float('inf')
        else:
            K_critical = 0.0

        return {
            "gain_values": K_range.tolist(),
            "poles_real": poles_real,
            "poles_imag": poles_imag,
            "asymptote_center": asymptote_center,
            "asymptote_angles_deg": asymptote_angles_deg,
            "breakaway_points": breakaway,
            "critical_gain": K_critical,
            "open_loop_poles": [
                (-self.zeta * self.omega_n, self.omega_n * math.sqrt(max(0, 1 - self.zeta**2))),
                (-self.zeta * self.omega_n, -self.omega_n * math.sqrt(max(0, 1 - self.zeta**2)))
            ] if self.zeta < 1 else [
                (-self.omega_n * (self.zeta - math.sqrt(self.zeta**2 - 1)), 0.0),
                (-self.omega_n * (self.zeta + math.sqrt(self.zeta**2 - 1)), 0.0)
            ],
        }

    def _find_breakaway_points_refined(self, K_range: 'np.ndarray') -> List[Dict[str, float]]:
        """Encuentra puntos de ruptura en el lugar de las raíces."""
        a2 = self.L * self.C
        a1 = self.R * self.C

        if abs(a2) < 1e-15:
            return []

        s_break = -a1 / (2.0 * a2)
        K_break = -(a2 * s_break**2 + a1 * s_break + 1.0)

        if K_range[0] <= K_break <= K_range[-1]:
            discriminant_at_break = a1**2 - 4.0 * a2 * (1.0 + K_break)
            point_type = "breakaway" if abs(discriminant_at_break) < 1e-10 else "break-in"

            return [{
                "real": s_break,
                "imag": 0.0,
                "gain_K": K_break,
                "type": point_type,
            }]

        return []

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Genera reporte completo del análisis de Laplace."""
        stability = self.analyze_stability()
        freq_response = self.get_frequency_response()
        root_locus = self.get_root_locus_data()
        validation = self.validate_for_control_design()

        return {
            "system_parameters": {
                "R": self.R,
                "L": self.L,
                "C": self.C,
                "omega_n": self.omega_n,
                "zeta": self.zeta,
                "Q": self.Q,
                "damping_class": self.damping_class,
            },
            "stability_analysis": stability,
            "frequency_response": freq_response,
            "root_locus": root_locus,
            "control_design_validation": validation,
            "timestamp": time.time(),
            "version": "2.0",
        }

    def validate_for_control_design(self) -> Dict[str, Any]:
        """
        Valida si el sistema es adecuado para diseño de control.

        Verifica:
        1. Estabilidad
        2. Márgenes adecuados
        3. Sensibilidad aceptable
        4. Frecuencia de muestreo suficiente
        """
        stability = self.analyze_stability()

        issues = []
        warnings = []

        # 1. Verificar estabilidad
        if not stability["is_stable"]:
            issues.append("Sistema inestable - no adecuado para control")

        # 2. Verificar márgenes
        pm = stability["stability_margins"]["phase_margin_deg"]
        if pm < 30:
            issues.append(f"Margen de fase insuficiente ({pm:.1f}° < 30°)")
        elif pm < 45:
            warnings.append(f"Margen de fase marginal ({pm:.1f}°)")

        # 3. Verificar frecuencia de muestreo (Nyquist)
        if self.omega_n > 0:
            nyquist_limit = 2 * self.omega_n
            if self.sample_rate < nyquist_limit:
                issues.append(
                    f"Frecuencia de muestreo insuficiente: "
                    f"{self.sample_rate:.1f} Hz < {nyquist_limit:.1f} Hz (ω_nyquist)"
                )
            elif self.sample_rate < 10 * self.omega_n:
                warnings.append(
                    f"Frecuencia de muestreo baja para control: "
                    f"{self.sample_rate:.1f} Hz < {10 * self.omega_n:.1f} Hz (10×ω_n)"
                )

        # 4. Verificar sensibilidad
        sens = stability["parameter_sensitivity"]
        if sens.get("robustness_classification", "").startswith("FRÁGIL"):
            warnings.append("Alta sensibilidad paramétrica - control difícil")

        return {
            "is_suitable_for_control": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "recommendations": stability["control_recommendations"],
            "summary": self._generate_validation_summary(issues, warnings),
        }

    def _generate_validation_summary(self, issues: List[str], warnings: List[str]) -> str:
        """Genera resumen de validación."""
        if issues:
            return f"NO APTO PARA CONTROL - {len(issues)} problemas críticos"
        elif warnings:
            return f"APTO CON ADVERTENCIAS - {len(warnings)} advertencias"
        else:
            return "APTO PARA CONTROL - Sistema bien condicionado"

    def get_laplace_pyramid(self) -> Dict[str, Any]:
        """
        Genera la Pirámide de Laplace con 4 niveles jerárquicos.

        Nivel 0 (Cúspide - Veredicto): ¿El sistema es Controlable? (Estabilidad Absoluta).
        Nivel 1 (Robustez): Márgenes de Fase y Ganancia.
        Nivel 2 (Dinámica): Lugar de las Raíces y Polos.
        Nivel 3 (Base): Parámetros Físicos RLC.

        Returns:
            Dict estructurado jerárquicamente.
        """
        stability = self.analyze_stability()
        validation = self.validate_for_control_design()
        margins = stability["stability_margins"]

        return {
            "level_0_verdict": {
                "is_controllable": validation["is_suitable_for_control"],
                "stability_status": stability["status"],
                "system_classification": self.damping_class,
                "summary": validation["summary"],
            },
            "level_1_robustness": {
                "phase_margin_deg": margins["phase_margin_deg"],
                "gain_margin_db": margins["gain_margin_db"],
                "robustness_class": stability["parameter_sensitivity"]["robustness_classification"],
                "sensitivity_max": max(
                    stability["parameter_sensitivity"].get("sensitivity_to_R", 0),
                    stability["parameter_sensitivity"].get("sensitivity_to_L", 0),
                    stability["parameter_sensitivity"].get("sensitivity_to_C", 0),
                ),
            },
            "level_2_dynamics": {
                "natural_frequency_rad_s": self.omega_n,
                "damping_ratio": self.zeta,
                "poles_continuous": stability["continuous"]["poles"],
                "zeros_continuous": stability["continuous"]["zeros"],
                "response_type": stability["continuous"]["response_type"],
            },
            "level_3_physics": {
                "R": self.R,
                "L": self.L,
                "C": self.C,
                "sample_rate_hz": self.sample_rate,
            }
        }
