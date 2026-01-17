import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto

try:
    import numpy as np
except ImportError:
    np = None

import scipy.signal


class ConfigurationError(Exception):
    """Indica un problema con la configuración del sistema."""
    pass


class DampingClass(Enum):
    """Clasificación del amortiguamiento del sistema."""
    NEGATIVE = auto()
    UNDAMPED = auto()
    UNDERDAMPED = auto()
    CRITICALLY_DAMPED = auto()
    OVERDAMPED = auto()


class StabilityStatus(Enum):
    """Estado de estabilidad del sistema."""
    STABLE = auto()
    MARGINALLY_STABLE = auto()
    UNSTABLE = auto()


@dataclass(frozen=True)
class SystemParameters:
    """Parámetros físicos validados del sistema RLC."""
    R: float
    L: float
    C: float

    def __post_init__(self):
        if self.L <= 0 or self.C <= 0:
            raise ConfigurationError("L y C deben ser estrictamente positivos")
        if self.R < 0:
            raise ConfigurationError("R no puede ser negativo")


class NumericalConstants:
    """Constantes numéricas centralizadas para consistencia."""
    EPSILON_ZERO = 1e-24          # Umbral para comparación con cero (ajustado para pF/pH)
    EPSILON_UNITY = 1e-6          # Umbral para comparación con 1
    EPSILON_STABILITY = 1e-9      # Umbral para análisis de estabilidad
    MIN_INDUCTANCE = 1e-15        # Inductancia mínima física (fH)
    MIN_CAPACITANCE = 1e-18       # Capacitancia mínima física (aF)
    MAX_FREQUENCY_RAD = 1e15      # Frecuencia máxima razonable (PHz)
    DEFAULT_SETTLING_TOLERANCE = 0.02  # 2% para tiempo de asentamiento
    NYQUIST_SAFETY_FACTOR = 10    # Factor de seguridad sobre Nyquist


class LaplaceOracle:
    """
    Analizador de estabilidad en el dominio de Laplace con capacidades extendidas.

    Sistema RLC de segundo orden en forma canónica:
        H(s) = ωₙ² / (s² + 2ζωₙs + ωₙ²)
    """

    def __init__(self, R: float, L: float, C: float, sample_rate: float = 1000.0):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._nc = NumericalConstants()

        # Validación paramétrica exhaustiva
        self._validate_parameters(R, L, C, sample_rate)

        # Parámetros físicos inmutables
        self._params = SystemParameters(R=float(R), L=float(L), C=float(C))

        self.R = self._params.R
        self.L = self._params.L
        self.C = self._params.C

        self._compute_derived_parameters()

        self.sample_rate = float(sample_rate)
        self.T = 1.0 / self.sample_rate

        self.continuous_system = self._build_continuous_system()
        self.discrete_system = self._compute_discrete_system()

        self._classify_system()

        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 50

    def _validate_parameters(self, R: float, L: float, C: float, sample_rate: float) -> None:
        errors = []
        warnings = []
        nc = self._nc

        param_specs = [
            ("R", R, 0.0, 1e9, "Ω"),
            ("L", L, nc.MIN_INDUCTANCE, 1e6, "H"),
            ("C", C, nc.MIN_CAPACITANCE, 1e3, "F"),
            ("sample_rate", sample_rate, 1.0, 1e12, "Hz"),
        ]

        for name, value, min_val, max_val, unit in param_specs:
            if not isinstance(value, (int, float)):
                errors.append(f"{name} debe ser numérico")
                continue
            if not math.isfinite(value):
                errors.append(f"{name} debe ser finito")
            elif value < min_val:
                errors.append(f"{name} = {value} {unit} < mínimo {min_val} {unit}")
            elif value > max_val:
                warnings.append(f"{name} = {value:.2e} {unit} excede rango típico")

        if errors:
            raise ConfigurationError(
                "Parámetros físicos inválidos:\n" + "\n".join(f"  • {e}" for e in errors)
            )

        if L > 0 and C > 0:
            omega_n = 1.0 / math.sqrt(L * C)
            f_n = omega_n / (2 * math.pi)

            nyquist_freq = sample_rate / 2

            if f_n > nyquist_freq:
                 msg = f"Frecuencia natural f_n = {f_n:.2e} Hz > f_Nyquist = {nyquist_freq:.2e} Hz. Aliasing severo."
                 self.logger.warning(msg)
                 errors.append(msg)
            elif f_n > nyquist_freq / nc.NYQUIST_SAFETY_FACTOR:
                warnings.append(
                    f"Frecuencia natural f_n = {f_n:.2e} Hz cercana a Nyquist."
                )

            if R > 0:
                zeta = (R / 2.0) * math.sqrt(C / L)
                if zeta < 0.01:
                    warnings.append(f"Sistema casi sin amortiguamiento (ζ = {zeta:.4f}).")
                elif zeta > 50:
                    warnings.append(f"Sistema extremadamente sobreamortiguado (ζ = {zeta:.1f}).")

        if errors:
            raise ConfigurationError(
                "Inconsistencias físicas detectadas:\n" + "\n".join(f"  • {e}" for e in errors)
            )

        for warning in warnings:
            self.logger.warning(f"⚠️ {warning}")

    def _compute_derived_parameters(self) -> None:
        nc = self._nc
        LC_product = self.L * self.C
        if LC_product > nc.EPSILON_ZERO:
            self.omega_n = 1.0 / math.sqrt(LC_product)
        else:
            self.omega_n = float('inf')
            self.logger.warning("Producto LC ≈ 0, ωₙ → ∞")

        if self.L > nc.EPSILON_ZERO:
            self.zeta = (self.R / 2.0) * math.sqrt(self.C / self.L)
        else:
            self.zeta = float('inf')

        if self.zeta > nc.EPSILON_ZERO:
            self.Q = 1.0 / (2.0 * self.zeta)
        else:
            self.Q = float('inf')

        if 0 < self.zeta < 1.0:
            self.omega_d = self.omega_n * math.sqrt(1.0 - self.zeta**2)
        else:
            self.omega_d = 0.0

    def _build_continuous_system(self) -> scipy.signal.TransferFunction:
        num = [1.0]
        den = [self.L * self.C, self.R * self.C, 1.0]
        try:
            return scipy.signal.TransferFunction(num, den)
        except Exception as e:
            raise ConfigurationError(f"Error construyendo sistema continuo: {e}")

    def _classify_system(self) -> None:
        """Clasifica el sistema. Expone damping_class como string para compatibilidad."""
        nc = self._nc

        if self.zeta < -nc.EPSILON_ZERO:
            self._damping_enum = DampingClass.NEGATIVE
            self.stability_status = StabilityStatus.UNSTABLE
            self.response_type = "DIVERGENT_OSCILLATORY"
        elif abs(self.zeta) < nc.EPSILON_ZERO:
            self._damping_enum = DampingClass.UNDAMPED
            self.stability_status = StabilityStatus.MARGINALLY_STABLE
            self.response_type = "SUSTAINED_OSCILLATION"
        elif self.zeta < 1.0 - nc.EPSILON_UNITY:
            self._damping_enum = DampingClass.UNDERDAMPED
            self.stability_status = StabilityStatus.STABLE
            self.response_type = "DAMPED_OSCILLATION"
        elif abs(self.zeta - 1.0) < nc.EPSILON_UNITY:
            self._damping_enum = DampingClass.CRITICALLY_DAMPED
            self.stability_status = StabilityStatus.STABLE
            self.response_type = "CRITICAL_EXPONENTIAL"
        else:
            self._damping_enum = DampingClass.OVERDAMPED
            self.stability_status = StabilityStatus.STABLE
            self.response_type = "OVERDAMPED_EXPONENTIAL"

        # Compatibilidad hacia atrás: damping_class como string
        self.damping_class = self._damping_enum.name

    def _compute_discrete_system(self) -> scipy.signal.TransferFunction:
        T = self.T
        nc = self._nc
        omega_critical = self.omega_n if self.omega_n > 0 and self.omega_n < nc.MAX_FREQUENCY_RAD else 1.0 / T
        omega_T_half = omega_critical * T / 2.0

        if omega_T_half < nc.EPSILON_ZERO:
            k = 2.0 / T
        elif omega_T_half > math.pi / 2 - 0.01:
            self.logger.error(f"ωT/2 = {omega_T_half:.3f} ≈ π/2. Sistema continuo demasiado rápido.")
            k = 2.0 / T
        else:
            k = omega_critical / math.tan(omega_T_half)

        a2 = self.L * self.C
        a1 = self.R * self.C
        a0 = 1.0

        k2 = k * k
        den_z2 = a2 * k2 + a1 * k + a0
        den_z1 = 2.0 * (a0 - a2 * k2)
        den_z0 = a2 * k2 - a1 * k + a0

        num_z2 = 1.0
        num_z1 = 2.0
        num_z0 = 1.0

        if abs(den_z2) < nc.EPSILON_ZERO:
            return scipy.signal.TransferFunction([1.0], [1.0], dt=T)

        num = np.array([num_z2, num_z1, num_z0]) / den_z2
        den = np.array([1.0, den_z1 / den_z2, den_z0 / den_z2])

        self._validate_discrete_stability(den)

        try:
            return scipy.signal.TransferFunction(num.tolist(), den.tolist(), dt=T)
        except Exception as e:
            self.logger.error(f"Error en discretización: {e}")
            return self.continuous_system

    def _validate_discrete_stability(self, den_coeffs: np.ndarray) -> None:
        if np is None: return
        try:
            roots = np.roots(den_coeffs)
            magnitudes = np.abs(roots)
            max_magnitude = np.max(magnitudes) if len(magnitudes) > 0 else 0

            if max_magnitude > 1.0 + self._nc.EPSILON_STABILITY:
                self.logger.error(f"⚠️ Sistema discreto INESTABLE: max|polo| = {max_magnitude:.6f} > 1.")
            elif max_magnitude > 0.99:
                self.logger.warning(f"Sistema discreto marginalmente estable: max|polo| = {max_magnitude:.6f}")
        except Exception as e:
            self.logger.warning(f"No se pudo validar estabilidad discreta: {e}")

    def _calculate_stability_margins(self) -> Dict[str, Any]:
        """Calcula márgenes de estabilidad (Lazo Abierto Implícito)."""
        nc = self._nc
        gain_margin_db = float('inf')
        omega_pc = float('inf')

        if self.zeta <= nc.EPSILON_ZERO:
            return {
                "gain_margin_db": gain_margin_db,
                "phase_margin_deg": 0.0,
                "gain_crossover_freq_rad_s": self.omega_n,
                "phase_crossover_freq_rad_s": omega_pc,
                "is_margin_meaningful": False,
                "regime": "UNDAMPED_OR_UNSTABLE",
                "interpretation": "Sistema sin amortiguamiento - PM = 0°"
            }

        term_sqrt = math.sqrt(4.0 * self.zeta**4 + 1.0)
        omega_gc = self.omega_n * math.sqrt(term_sqrt - 2.0 * self.zeta**2)
        pm_rad = math.atan2(2.0 * self.zeta * self.omega_n, omega_gc)
        phase_margin_deg = math.degrees(pm_rad)

        is_meaningful = True
        regime = "IMPLICIT_OPEN_LOOP"
        interpretation = self._generate_margin_interpretation(self.zeta, phase_margin_deg, regime)

        return {
            "gain_margin_db": gain_margin_db,
            "phase_margin_deg": phase_margin_deg,
            "gain_crossover_freq_rad_s": omega_gc,
            "phase_crossover_freq_rad_s": omega_pc,
            "is_margin_meaningful": is_meaningful,
            "regime": regime,
            "interpretation": interpretation,
            "derivation_method": "implicit_open_loop_type1",
            "notes": interpretation,
        }

    def _generate_margin_interpretation(self, zeta: float, pm_deg: float, regime: str) -> str:
        if zeta < 0.1:
            return f"Poco amortiguado (ζ={zeta:.3f}). PM bajo ({pm_deg:.1f}°)."
        elif zeta < 0.707:
            return f"Subamortiguado (ζ={zeta:.3f}). PM adecuado ({pm_deg:.1f}°)."
        else:
            return f"Bien amortiguado (ζ={zeta:.3f}). PM alto ({pm_deg:.1f}°)."

    def _calculate_transient_metrics(self) -> Dict[str, Any]:
        nc = self._nc
        if self.zeta < -nc.EPSILON_ZERO:
            return {"status": "UNSTABLE", "metrics": {}}
        if abs(self.zeta) < nc.EPSILON_ZERO:
            period = 2.0 * math.pi / self.omega_n if self.omega_n > 0 else float('inf')
            return {
                "status": "UNDAMPED", "rise_time_s": period / 4.0,
                "overshoot_percent": 100.0, "settling_time_s": float('inf')
            }

        if self.zeta < 1.0 - nc.EPSILON_UNITY:
            omega_d = self.omega_d
            phi = math.acos(self.zeta)
            rise_time = (math.pi - phi) / omega_d
            damping_factor = self.zeta / math.sqrt(1.0 - self.zeta**2)
            overshoot_percent = math.exp(-math.pi * damping_factor) * 100.0
            settling_time = 4.0 / (self.zeta * self.omega_n)
            return {
                "status": "UNDERDAMPED", "rise_time_s": rise_time,
                "overshoot_percent": overshoot_percent, "settling_time_s": settling_time,
                "peak_time_s": math.pi / omega_d,
            }

        if abs(self.zeta - 1.0) < nc.EPSILON_UNITY:
            return {
                "status": "CRITICALLY_DAMPED", "rise_time_s": 3.3579 / self.omega_n,
                "overshoot_percent": 0.0, "settling_time_s": 5.8335 / self.omega_n,
                "peak_time_s": float('inf'),
            }

        s1 = -self.omega_n * (self.zeta - math.sqrt(self.zeta**2 - 1.0))
        tau_slow = 1.0 / abs(s1)
        return {
            "status": "OVERDAMPED", "rise_time_s": 2.2 * tau_slow,
            "overshoot_percent": 0.0, "settling_time_s": 4.0 * tau_slow,
            "peak_time_s": float('inf'),
        }

    def _calculate_parameter_sensitivity(self) -> Dict[str, Any]:
        nc = self._nc
        if self.zeta < nc.EPSILON_ZERO:
            return {"status": "UNSTABLE", "robustness_classification": "NOT_APPLICABLE"}

        S_R_zeta = 1.0
        S_L_zeta = -0.5
        S_C_zeta = 0.5
        S_L_omega = -0.5
        S_C_omega = -0.5

        sens_R = abs(S_R_zeta)
        sens_L = math.sqrt(S_L_omega**2 + S_L_zeta**2)
        sens_C = math.sqrt(S_C_omega**2 + S_C_zeta**2)

        sensitivities = [sens_R, sens_L, sens_C]
        max_sens = max(sensitivities)

        if max_sens > 2.0: robustness_class = "FRAGILE"
        elif max_sens > 1.0: robustness_class = "SENSITIVE"
        elif max_sens < 0.5: robustness_class = "ROBUST"
        else: robustness_class = "MODERATE"

        param_sens = {"R": sens_R, "L": sens_L, "C": sens_C}
        most_sensitive = max(param_sens, key=param_sens.get)

        return {
            "scalar_sensitivities": param_sens,
            "most_sensitive_parameter": most_sensitive,
            "robustness_classification": robustness_class,
            "sensitivity_to_R": sens_R,
            "sensitivity_to_L": sens_L,
            "sensitivity_to_C": sens_C,
        }

    def get_frequency_response(
        self,
        frequencies: Optional[np.ndarray] = None,
        n_points: int = 500,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        if frequencies is not None:
            cache_key = f"freq_{hash(frequencies.tobytes())}"
        else:
            cache_key = f"freq_default_{n_points}"

        if use_cache and cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]["data"]

        if frequencies is None:
            w_min = max(self.omega_n / 1000.0, 1e-4)
            w_max = min(self.omega_n * 1000.0, 1e8)
            frequencies = np.logspace(np.log10(w_min), np.log10(w_max), n_points)

        w = frequencies
        LC = self.L * self.C
        RC = self.R * self.C

        real_part = 1.0 - LC * w**2
        imag_part = RC * w
        denom_sq = real_part**2 + imag_part**2
        denom_sq = np.maximum(denom_sq, 1e-30)

        magnitude = 1.0 / np.sqrt(denom_sq)
        magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-30))

        H_real = real_part / denom_sq
        H_imag = -imag_part / denom_sq
        phase_rad = np.arctan2(H_imag, H_real)
        phase_deg = np.degrees(np.unwrap(phase_rad))

        result = {
            "frequencies_rad_s": frequencies.tolist(),
            "magnitude_db": magnitude_db.tolist(),
            "phase_deg": phase_deg.tolist(),
            "nyquist_real": H_real.tolist(),
            "nyquist_imag": H_imag.tolist(),
            "resonance": self._find_resonance_analytical(),
            "bandwidth_rad_s": self._calculate_bandwidth_log_interp(frequencies, magnitude_db),
            "high_freq_rolloff_db_per_decade": -40.0,
        }

        self._analysis_cache[cache_key] = {"data": result, "timestamp": time.time()}
        self._prune_cache()
        return result

    def _find_resonance_analytical(self) -> Dict[str, Any]:
        nc = self._nc
        if self.zeta >= 0.707 or self.zeta <= nc.EPSILON_ZERO:
            return {"exists": False}

        omega_r = self.omega_n * math.sqrt(1.0 - 2.0 * self.zeta**2)
        mag = 1.0 / (2.0 * self.zeta * math.sqrt(1.0 - self.zeta**2))
        return {
            "exists": True,
            "frequency_rad_s": omega_r,
            "magnitude_db": 20 * math.log10(mag)
        }

    def _calculate_bandwidth_log_interp(self, freqs, mag_db) -> float:
        target = mag_db[0] - 3.0
        for i in range(len(mag_db) - 1):
            if mag_db[i] >= target > mag_db[i+1]:
                t = (target - mag_db[i]) / (mag_db[i+1] - mag_db[i])
                log_f = np.log10(freqs[i]) + t * (np.log10(freqs[i+1]) - np.log10(freqs[i]))
                return 10.0**log_f
        return 0.0

    def _prune_cache(self) -> None:
        if len(self._analysis_cache) > self._cache_max_size:
            self._analysis_cache.clear()

    def analyze_stability(self) -> Dict[str, Any]:
        cache_key = "stability_analysis"
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]["data"]

        margins = self._calculate_stability_margins()
        transient = self._calculate_transient_metrics()
        sensitivity = self._calculate_parameter_sensitivity()

        rec_params = {"margins": margins, "sensitivity": sensitivity}
        recommendations = self._generate_control_recommendations(rec_params)

        poles_c = self.continuous_system.poles
        zeros_c = self.continuous_system.zeros
        poles_d = self.discrete_system.poles
        zeros_d = self.discrete_system.zeros

        result = {
            "status": self.stability_status.name,
            "is_stable": self.stability_status == StabilityStatus.STABLE,
            "is_marginally_stable": self.stability_status == StabilityStatus.MARGINALLY_STABLE,
            "continuous": {
                "poles": [(complex(p).real, complex(p).imag) for p in poles_c],
                "zeros": [(complex(z).real, complex(z).imag) for z in zeros_c],
                "natural_frequency_rad_s": self.omega_n,
                "damping_ratio": self.zeta,
                "damping_class": self.damping_class, # String
            },
            "discrete": {
                "poles": [(complex(p).real, complex(p).imag) for p in poles_d],
                "zeros": [(complex(z).real, complex(z).imag) for z in zeros_d],
            },
            "stability_margins": margins,
            "transient_response": transient,
            "parameter_sensitivity": sensitivity,
            "control_recommendations": recommendations,
        }

        self._analysis_cache[cache_key] = {"data": result, "timestamp": time.time()}
        return result

    def _generate_control_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        recs = []
        margins = analysis_data.get("margins", {})
        sensitivity = analysis_data.get("sensitivity", {})

        pm = margins.get("phase_margin_deg", 0)
        if pm < 30: recs.append("Margen de fase bajo.")
        if self.zeta < 0.2: recs.append("Sistema muy subamortiguado.")
        if sensitivity.get("robustness_classification") == "FRAGILE": recs.append("Sistema frágil.")
        if not recs: recs.append("Sistema apto para control.")
        return recs

    def validate_for_control_design(self) -> Dict[str, Any]:
        stability = self.analyze_stability()
        margins = stability["stability_margins"]

        issues = []
        warnings = []

        if not stability["is_stable"]: issues.append("Sistema inestable")
        pm = margins.get("phase_margin_deg", 0)
        if pm < 30: issues.append(f"Margen de fase insuficiente ({pm:.1f}°)")
        elif pm < 45: warnings.append(f"Margen de fase marginal ({pm:.1f}°)")

        nyquist = self.sample_rate / 2
        if self.omega_n > nyquist: issues.append("Frecuencia de muestreo insuficiente (Aliasing)")
        elif self.omega_n > nyquist / 5: warnings.append("Frecuencia de muestreo baja")

        is_suitable = len(issues) == 0
        return {
            "is_suitable_for_control": is_suitable,
            "issues": issues,
            "warnings": warnings,
            "recommendations": stability["control_recommendations"],
            "summary": "APTO" if is_suitable else "NO APTO"
        }

    def get_root_locus_data(self, K_range: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if K_range is None: K_range = np.logspace(-3, 3, 300)
        poles_real = []
        poles_imag = []
        a, b, c_base = self.L * self.C, self.R * self.C, 1.0

        for K in K_range:
            c = c_base + K
            disc = b**2 - 4*a*c
            if disc >= 0:
                s1 = (-b + math.sqrt(disc)) / (2*a)
                s2 = (-b - math.sqrt(disc)) / (2*a)
                poles_real.extend([s1, s2])
                poles_imag.extend([0.0, 0.0])
            else:
                real = -b / (2*a)
                imag = math.sqrt(-disc) / (2*a)
                poles_real.extend([real, real])
                poles_imag.extend([imag, -imag])

        return {
            "gain_values": K_range.tolist(),
            "poles_real": poles_real,
            "poles_imag": poles_imag,
            "asymptote_angles_deg": [90, 270]
        }

    def get_comprehensive_report(self) -> Dict[str, Any]:
        return {
            "system_parameters": {
                "R": self.R, "L": self.L, "C": self.C,
                "omega_n": self.omega_n, "zeta": self.zeta
            },
            "stability_analysis": self.analyze_stability(),
            "frequency_response": self.get_frequency_response(),
            "root_locus": self.get_root_locus_data()
        }

    def get_laplace_pyramid(self) -> Dict[str, Any]:
        stability = self.analyze_stability()
        margins = stability["stability_margins"]
        sensitivity = stability["parameter_sensitivity"]

        is_controllable = (
            stability["is_stable"] and
            margins.get("phase_margin_deg", 0) > 30 and
            sensitivity.get("robustness_classification") != "FRAGILE"
        )

        return {
            "level_0_verdict": {
                "is_controllable": is_controllable,
                "stability_status": stability["status"],
            },
            "level_1_robustness": {
                "phase_margin_deg": margins["phase_margin_deg"],
                "gain_margin_db": margins["gain_margin_db"],
            },
            "level_2_dynamics": {
                "omega_n_rad_s": self.omega_n,
                "zeta": self.zeta,
                "poles": stability["continuous"]["poles"],
            },
            "level_3_physics": {
                "R_ohm": self.R,
                "L_henry": self.L,
                "C_farad": self.C,
            }
        }
