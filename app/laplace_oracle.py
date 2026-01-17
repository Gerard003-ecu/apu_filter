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

        # Sistema discreto (Transformación bilineal Tustin con Pre-warping)
        self.discrete_system = self._compute_discrete_system()

        # Clasificación del sistema
        self._classify_system()

        # Cache para resultados computacionalmente costosos
        self._analysis_cache: Dict[str, Any] = {}

    def _validate_parameters(self, R: float, L: float, C: float) -> None:
        """
        Validación exhaustiva de parámetros físicos.
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
        Convierte sistema continuo a discreto usando transformación bilineal con pre-warping.

        Transformación corregida con pre-warping:
            s = ω₀ / tan(ω₀·T/2) · (z-1)/(z+1)

        donde ω₀ es la frecuencia crítica (ωₙ para sistemas resonantes).
        """
        T = self.T

        # Frecuencia crítica para pre-warping (frecuencia natural del sistema)
        omega_critical = self.omega_n if self.omega_n > 0 else 1.0 / T

        # Coeficiente de pre-warping
        if omega_critical * T < 1e-6:
            k = 2.0 / T  # Sin pre-warping para frecuencias bajas
        else:
            k = omega_critical / math.tan(omega_critical * T / 2.0)

        # Coeficientes del denominador continuo
        a2 = self.L * self.C
        a1 = self.R * self.C
        a0 = 1.0

        # Aplicar transformación bilineal: s = k*(z-1)/(z+1)
        # H(z) = b0 + b1*z⁻¹ + b2*z⁻² / (1 + a1*z⁻¹ + a2*z⁻²)

        # Denominador expandido: D(z) = (a2*k² + a1*k + a0)z² + (2*a0 - 2*a2*k²)z + (a2*k² - a1*k + a0)
        k2 = k * k

        den_z2 = a2 * k2 + a1 * k + a0
        den_z1 = 2.0 * a0 - 2.0 * a2 * k2
        den_z0 = a2 * k2 - a1 * k + a0

        # Numerador: N(z) = (z+1)² = z² + 2z + 1
        num_z2 = 1.0
        num_z1 = 2.0
        num_z0 = 1.0

        # Normalizar para coeficiente líder unitario en denominador
        if abs(den_z2) < 1e-15:
            self.logger.warning("Denominador degenerado en discretización, usando sistema continuo")
            return self.continuous_system

        # Coeficientes normalizados
        num = [num_z2 / den_z2, num_z1 / den_z2, num_z0 / den_z2]
        den = [1.0, den_z1 / den_z2, den_z0 / den_z2]

        # Validación de estabilidad
        if np:
            roots = np.roots(den)
            max_pole_magnitude = max(abs(r) for r in roots) if len(roots) > 0 else 0

            if max_pole_magnitude > 1.0 + 1e-6:
                self.logger.warning(
                    f"Sistema discreto inestable: |p|_max = {max_pole_magnitude:.3f}. "
                    f"Sample rate {self.sample_rate} Hz puede ser insuficiente."
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
        """Calcula márgenes de estabilidad con fórmulas exactas para sistemas de segundo orden."""

        # Para sistemas de segundo orden sin ceros:
        # H(s) = ωₙ²/(s² + 2ζωₙs + ωₙ²)

        # Margen de ganancia: siempre infinito para sistemas sin ceros en RHP
        gain_margin_db = float('inf')

        # Cálculo EXACTO del margen de fase
        if self.zeta <= 0:
            phase_margin_deg = 0.0
            omega_gc = 0.0
            is_meaningful = False
        else:
            # Frecuencia de cruce de ganancia: |H(jω)| = 1
            # Solución de |H(jω)|² = 1
            # ω⁴/ωₙ⁴ + (4ζ² - 2)ω²/ωₙ² + 1 = 1
            # ω²(ω²/ωₙ⁴ + (4ζ² - 2)/ωₙ²) = 0

            if self.zeta < 1 / math.sqrt(2):  # ζ < 0.707
                # Existe cruce de ganancia real (donde |H(jw)| = 1)
                # Solución exacta: w = wn * sqrt(2 - 4*zeta^2)
                term_inside = 2.0 - 4.0 * self.zeta**2
                # Protección contra errores de punto flotante cerca de 0.707
                omega_gc = self.omega_n * math.sqrt(max(0.0, term_inside))

                # Fase en ω_gc
                phase_at_gc = -math.atan2(
                    2 * self.zeta * omega_gc / self.omega_n,
                    1 - (omega_gc / self.omega_n)**2
                )
                phase_margin_deg = 180 + math.degrees(phase_at_gc)
                is_meaningful = True
            else:
                # No hay cruce de ganancia (|H(jω)| < 1 ∀ω)
                omega_gc = self.omega_n * math.sqrt(2 * self.zeta**2 - 1)

                # Margen de fase por definición en ω donde ganancia es máxima
                phase_at_omega_n = -math.pi / 2  # -90° en ω = ωₙ
                phase_margin_deg = 180 + math.degrees(phase_at_omega_n)
                is_meaningful = False

        # Frecuencia de cruce de fase (donde fase = -180°)
        # Para segundo orden, fase tiende a -180° solo cuando ω→∞
        omega_pc = float('inf')

        return {
            "gain_margin_db": gain_margin_db,
            "phase_margin_deg": phase_margin_deg,
            "gain_crossover_freq_rad_s": omega_gc,
            "phase_crossover_freq_rad_s": omega_pc,
            "is_margin_meaningful": is_meaningful,
            "derivation_method": "exact_second_order" if is_meaningful else "asymptotic_approximation",
            "notes": self._generate_margin_notes(self.zeta, phase_margin_deg),
            # Compatibility key
            "interpretation": self._generate_margin_notes(self.zeta, phase_margin_deg)
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
        """Calcula matriz de sensibilidad completa usando álgebra de derivadas parciales."""

        # Para sistema H(s) = 1/(LCs² + RCs + 1)
        # Polos: s = -ζωₙ ± jωₙ√(1-ζ²) para ζ<1

        if self.zeta < 0:
            return {"status": "UNSTABLE_SYSTEM", "sensitivity_matrix": {}, "robustness_classification": "UNSTABLE"}

        # Derivadas parciales de ωₙ y ζ
        d_omega_n_dL = -self.omega_n / (2 * self.L) if self.L > 0 else 0
        d_omega_n_dC = -self.omega_n / (2 * self.C) if self.C > 0 else 0
        d_omega_n_dR = 0.0

        d_zeta_dR = 0.5 * math.sqrt(self.C / self.L) if self.L > 0 else 0
        d_zeta_dL = -self.zeta / (2 * self.L) if self.L > 0 else 0
        d_zeta_dC = self.zeta / (2 * self.C) if self.C > 0 else 0

        # Sensibilidad de los polos (si son complejos conjugados)
        if 0 < self.zeta < 1:
            omega_d = self.omega_n * math.sqrt(1 - self.zeta**2)
            pole = complex(-self.zeta * self.omega_n, omega_d)

            # Derivadas del polo respecto a ωₙ y ζ
            dpole_d_omega_n = complex(-self.zeta, math.sqrt(1 - self.zeta**2))
            dpole_d_zeta = complex(-self.omega_n,
                                   -self.zeta * self.omega_n / math.sqrt(1 - self.zeta**2))

            # Cadena de derivadas para cada parámetro
            dpole_dR = dpole_d_zeta * d_zeta_dR
            dpole_dL = dpole_d_zeta * d_zeta_dL + dpole_d_omega_n * d_omega_n_dL
            dpole_dC = dpole_d_zeta * d_zeta_dC + dpole_d_omega_n * d_omega_n_dC

            # Sensibilidad normalizada (∂p/p)/(∂x/x)
            sens_R = abs(dpole_dR * self.R / pole) if abs(pole) > 0 else 0
            sens_L = abs(dpole_dL * self.L / pole) if abs(pole) > 0 else 0
            sens_C = abs(dpole_dC * self.C / pole) if abs(pole) > 0 else 0
        else:
            # Polos reales
            pole1 = -self.omega_n * (self.zeta - math.sqrt(abs(self.zeta**2 - 1)))
            pole2 = -self.omega_n * (self.zeta + math.sqrt(abs(self.zeta**2 - 1)))

            # Para simplificar, usar sensibilidad del polo dominante
            pole_dom = pole1 if abs(pole1) < abs(pole2) else pole2

            # Aproximación para polos reales
            sens_R = abs(d_zeta_dR * self.R / self.zeta) if self.zeta > 0 else 0
            sens_L = abs((d_zeta_dL * self.L / self.zeta) +
                         (d_omega_n_dL * self.L / self.omega_n)) if self.zeta > 0 else 0
            sens_C = abs((d_zeta_dC * self.C / self.zeta) +
                         (d_omega_n_dC * self.C / self.omega_n)) if self.zeta > 0 else 0

        # Matriz de sensibilidad completa
        sensitivity_matrix = {
            "to_R": {
                "omega_n": 0.0,
                "zeta": d_zeta_dR * self.R / self.zeta if self.zeta > 0 else 0,
                "pole_magnitude": sens_R,
                "pole_angle": 0.0 if self.zeta >= 1 else sens_R * 0.5
            },
            "to_L": {
                "omega_n": d_omega_n_dL * self.L / self.omega_n,
                "zeta": d_zeta_dL * self.L / self.zeta if self.zeta > 0 else 0,
                "pole_magnitude": sens_L,
                "pole_angle": 0.0 if self.zeta >= 1 else sens_L * 0.5
            },
            "to_C": {
                "omega_n": d_omega_n_dC * self.C / self.omega_n,
                "zeta": d_zeta_dC * self.C / self.zeta if self.zeta > 0 else 0,
                "pole_magnitude": sens_C,
                "pole_angle": 0.0 if self.zeta >= 1 else sens_C * 0.5
            }
        }

        # Número de condición (medida de robustez global)
        max_sens = max(sens_R, sens_L, sens_C)
        cond_number = max_sens / max(max_sens, 1e-6) if max_sens > 0 else 0.0
        robustness_class = self._classify_robustness_by_condition(cond_number)

        # Mapeo a claves legacy para compatibilidad
        most_sensitive = max(["R", "L", "C"],
                             key=lambda x: {"R": sens_R, "L": sens_L, "C": sens_C}[x])

        return {
            "sensitivity_matrix": sensitivity_matrix,
            "scalar_sensitivities": {
                "R": sens_R, "L": sens_L, "C": sens_C
            },
            "most_sensitive": most_sensitive,
            "condition_number": cond_number,
            "robustness_class": robustness_class,
            "recommendations": self._generate_sensitivity_recommendations(sens_R, sens_L, sens_C),

            # Backward compatibility keys
            "sensitivity_to_R": sens_R,
            "sensitivity_to_L": sens_L,
            "sensitivity_to_C": sens_C,
            "most_sensitive_parameter": most_sensitive,
            "robustness_classification": robustness_class,
        }

    def _classify_robustness_by_condition(self, cond_number: float) -> str:
        """Clasifica robustez basada en número de condición."""
        if cond_number > 100:
            return "FRÁGIL (MAL_CONDICIONADO) - Alta sensibilidad"
        elif cond_number > 10:
            return "MODERADO - Sensibilidad media"
        elif cond_number > 2:
            return "BIEN_CONDICIONADO - Baja sensibilidad"
        else:
            return "EXCELENTE - Muy robusto"

    def _generate_sensitivity_recommendations(self, sens_R, sens_L, sens_C):
        """Genera recomendaciones específicas basadas en sensibilidades."""
        rec = []

        if sens_R > max(sens_L, sens_C) * 2:
            rec.append("La resistencia R es el parámetro más crítico. Use resistores de alta precisión (±1% o mejor).")

        if sens_L > 0.5:
            rec.append("Alta sensibilidad a L. Considere inductores con núcleo fijo o implemente compensación adaptativa.")

        if sens_C > 0.5:
            rec.append("Alta sensibilidad a C. Use capacitores cerámicos NPO/C0G para baja deriva térmica.")

        return rec

    def get_frequency_response(self, frequencies: Optional['np.ndarray'] = None,
                               use_cache: bool = True) -> Dict[str, Any]:
        """Respuesta en frecuencia optimizada con caching y validación."""

        cache_key = f"freq_response_{hash(frequencies.tobytes()) if frequencies is not None else 'default'}"

        if use_cache and cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        # Generar rango de frecuencias logarítmico óptimo
        if frequencies is None:
            # Límites basados en propiedades del sistema
            w_min = min(self.omega_n / 1000.0, 1e-3) if self.omega_n > 0 else 1e-3
            w_max = max(self.omega_n * 1000.0, 1e3) if self.omega_n > 0 else 1e3

            # Número de puntos adaptativo
            n_points = min(1000, max(200, int(50 * math.log10(w_max / w_min))))
            frequencies = np.logspace(np.log10(w_min), np.log10(w_max), n_points)

        # Respuesta en frecuencia usando evaluación directa (más eficiente que scipy.signal.bode)
        s = 1j * frequencies

        # Evaluación directa: H(s) = 1/(LCs² + RCs + 1)
        denominator = self.L * self.C * s**2 + self.R * self.C * s + 1.0
        H = 1.0 / denominator

        magnitude_db = 20 * np.log10(np.abs(H))
        phase_deg = np.angle(H, deg=True)

        # Diagrama de Nyquist
        nyquist_real = H.real
        nyquist_imag = H.imag

        # Resonancia (solo para sistemas subamortiguados)
        resonance = self._find_resonance_analytical(frequencies, magnitude_db)

        # Ancho de banda
        bandwidth = self._calculate_bandwidth_robust(frequencies, magnitude_db)

        result = {
            "frequencies_rad_s": frequencies.tolist(),
            "magnitude_db": magnitude_db.tolist(),
            "phase_deg": phase_deg.tolist(),
            "nyquist_real": nyquist_real.tolist(),
            "nyquist_imag": nyquist_imag.tolist(),
            "resonance": resonance,
            "bandwidth_rad_s": bandwidth,
            "dc_gain_db": magnitude_db[0] if len(magnitude_db) > 0 else 0.0,
            "high_freq_asymptote_slope_db_dec": -40.0,  # -40 dB/década para 2º orden
            "cache_key": cache_key,
        }

        if use_cache:
            self._analysis_cache[cache_key] = result

        return result

    def _find_resonance_analytical(self, frequencies, magnitude_db):
        """Encuentra resonancia usando derivada analítica."""

        if self.zeta >= 1 / math.sqrt(2) or self.zeta <= 0:
            return {"frequency_rad_s": 0.0, "magnitude_db": magnitude_db[0], "exists": False}

        # Para sistema de 2º orden: frecuencia de resonancia ω_r = ωₙ√(1-2ζ²)
        omega_r = self.omega_n * math.sqrt(1 - 2 * self.zeta**2) if 1 - 2 * self.zeta**2 > 0 else 0

        # Magnitud en resonancia: |H(jω_r)| = 1/(2ζ√(1-ζ²))
        if 0 < self.zeta < 1:
            resonance_mag = 1.0 / (2 * self.zeta * math.sqrt(1 - self.zeta**2))
            resonance_mag_db = 20 * math.log10(resonance_mag)
        else:
            resonance_mag_db = 0.0

        return {
            "frequency_rad_s": omega_r,
            "magnitude_db": resonance_mag_db,
            "quality_factor": self.Q,
            "exists": True,
            "derivation": "analytical_second_order",
        }

    def _calculate_bandwidth_robust(self, frequencies, magnitude_db):
        """Cálculo robusto de ancho de banda con interpolación cúbica."""

        if len(magnitude_db) == 0:
            return 0.0

        dc_gain = magnitude_db[0]
        target_gain = dc_gain - 3.0  # -3 dB

        # Encontrar puntos donde cruza -3 dB
        crossings = []

        for i in range(len(magnitude_db) - 1):
            if (magnitude_db[i] >= target_gain >= magnitude_db[i + 1]) or \
               (magnitude_db[i] <= target_gain <= magnitude_db[i + 1]):
                # Interpolación cúbica spline local
                x = frequencies[i:i + 2]
                y = magnitude_db[i:i + 2]

                # Interpolación lineal robusta
                t = (target_gain - y[0]) / (y[1] - y[0])
                bandwidth = x[0] + t * (x[1] - x[0])
                crossings.append(bandwidth)

        if not crossings:
            return 0.0 if dc_gain < target_gain else frequencies[-1]

        return float(crossings[0])  # Primer cruce (menor frecuencia)

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
        """Genera datos para lugar de las raíces con derivación analítica correcta."""

        if K_range is None:
            # Rango adaptativo basado en propiedades del sistema
            K_max = max(100.0, 10.0 / abs(self.L * self.C)) if abs(self.L * self.C) > 0 else 100.0
            K_range = np.logspace(-3, math.log10(K_max), 300)

        # Sistema en lazo abierto: G(s) = 1/(LCs² + RCs + 1)
        # Ecuación característica: 1 + K·G(s) = 0
        # => LCs² + RCs + 1 + K = 0

        poles_real = []
        poles_imag = []
        breakpoints = []

        a = self.L * self.C
        b = self.R * self.C
        c_base = 1.0

        for K in K_range:
            c = c_base + K

            # Polos: s = [-b ± √(b² - 4ac)] / (2a)
            discriminant = b**2 - 4 * a * c

            if discriminant >= 0:
                sqrt_disc = math.sqrt(discriminant)
                s1 = (-b + sqrt_disc) / (2 * a)
                s2 = (-b - sqrt_disc) / (2 * a)
                poles_real.extend([s1, s2])
                poles_imag.extend([0.0, 0.0])
            else:
                real_part = -b / (2 * a)
                imag_part = math.sqrt(-discriminant) / (2 * a)
                poles_real.extend([real_part, real_part])
                poles_imag.extend([imag_part, -imag_part])

        # Puntos de ruptura: donde dK/ds = 0
        # K(s) = -(a s² + b s + c_base)
        # dK/ds = -(2a s + b) = 0 => s = -b/(2a)
        s_break = -b / (2 * a) if abs(a) > 0 else 0
        K_break = -(a * s_break**2 + b * s_break + c_base)

        if K_range[0] <= K_break <= K_range[-1]:
            breakpoints.append({
                "real": s_break,
                "imag": 0.0,
                "gain": K_break,
                "type": "breakaway" if a > 0 else "breakin",
            })

        # Asíntotas (para sistemas de orden n)
        n_poles = 2
        center = -b / (2 * a)
        angles = [90, 270]  # 180°/(n-m) = 180°/2 = 90° incrementos

        return {
            "gain_values": K_range.tolist(),
            "poles_real": poles_real,
            "poles_imag": poles_imag,
            "asymptote_center": center,
            "asymptote_angles_deg": angles,
            "breakaway_points": breakpoints,
            "departure_angles": self._calculate_departure_angles(),
            "critical_gain": self._find_critical_gain(K_range, poles_real, poles_imag),
            "system_order": n_poles,
        }

    def _calculate_departure_angles(self):
        """Calcula ángulos de partida de polos complejos."""
        if self.zeta >= 1 or self.zeta <= 0:
            return {}  # Polos reales

        # Para polos complejos p = -ζωₙ ± jωₙ√(1-ζ²)
        angle = math.degrees(math.atan(math.sqrt(1 - self.zeta**2) / self.zeta))

        return {
            "upper_pole": 180 - angle,
            "lower_pole": 180 + angle,
        }

    def _find_critical_gain(self, K_range, poles_real, poles_imag):
        """Encuentra ganancia crítica donde polos cruzan eje imaginario."""
        # En sistemas de 2do orden pasivos, la parte real es constante -R/2L o negativa.
        # Solo cruzaría si R < 0 (inestable).
        # Para R > 0, nunca cruza el eje imaginario (siempre estable).
        # Sin embargo, implementamos la lógica de detección de cruce de cero en parte real si hubiera.

        for i in range(len(poles_real) - 1):
            if poles_real[i] * poles_real[i+1] <= 0 and abs(poles_real[i]) > 1e-9:
                # Cruce de cero en parte real detected
                idx = i // 2 # Aproximado
                if idx < len(K_range):
                    return float(K_range[idx])

        return None

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
