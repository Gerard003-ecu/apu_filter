"""
Este componente actúa como una "Capacitancia Lógica" que se sitúa entre la ingesta de datos
crudos y el procesamiento. Modela el flujo de información como un fluido con propiedades
físicas cuantificables, utilizando ecuaciones de circuitos RLC para prevenir el colapso
del sistema por saturación o "fricción" de datos sucios.

Modelo Físico y Variables de Estado (`FluxPhysicsEngine`):
----------------------------------------------------------
1. Energía Potencial (Presión): 
   Calculada como E_c = 0.5 * C * V^2. Representa la "presión" de datos acumulada en la cola.
   
2. Energía Cinética (Inercia de Calidad):
   Calculada como E_l = 0.5 * L * I^2. Representa el momento de un flujo de datos limpio y constante.
   Un flujo con alta inercia es resistente a perturbaciones menores.

3. Voltaje Flyback (Inestabilidad):
   V_flyback = L * di/dt. Detecta cambios bruscos (picos inductivos) en la calidad de los datos,
   actuando como un detector temprano de anomalías estructurales o cambios de formato.

4. Potencia Disipada (Fricción/Entropía):
   P = I_ruido^2 * R. Mide la energía desperdiciada procesando datos inválidos ("calor" del sistema).
   Si P > 50W (simulado), se activa un freno de emergencia térmico.

Mecanismos de Control (`PIController`):
---------------------------------------
Implementa un lazo de control Proporcional-Integral (PI) discreto con anti-windup para
ajustar dinámicamente el tamaño del lote (batch size), manteniendo el sistema en un
régimen de "Flujo Laminar" (saturación objetivo ~30%).
"""

import logging
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:
    np = None

import pandas as pd
import scipy.signal

from .apu_processor import APUProcessor
from .report_parser_crudo import ReportParserCrudo
from .telemetry import TelemetryContext

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES DEL SISTEMA
# ============================================================================
class SystemConstants:
    """Constantes del sistema para evitar números mágicos."""

    # Límites de tiempo
    MIN_DELTA_TIME: float = 0.001  # Segundos mínimos entre cálculos PID
    MAX_DELTA_TIME: float = 3600.0  # 1 hora máximo entre cálculos
    PROCESSING_TIMEOUT: float = 3600.0  # Timeout de procesamiento total

    # Límites físicos
    MIN_ENERGY_THRESHOLD: float = 1e-10  # Julios mínimos para cálculos
    MAX_EXPONENTIAL_ARG: float = 100.0  # Límite para evitar overflow en exp()
    MAX_FLYBACK_VOLTAGE: float = 10.0  # Límite de tensión inductiva

    # Diagnóstico
    LOW_INERTIA_THRESHOLD: float = 0.1
    HIGH_PRESSURE_RATIO: float = 1000.0
    HIGH_FLYBACK_THRESHOLD: float = 0.5
    OVERHEAT_POWER_THRESHOLD: float = 50.0  # Watts

    # Control de flujo
    EMERGENCY_BRAKE_FACTOR: float = 0.5
    MAX_ITERATIONS_MULTIPLIER: int = 10  # max_iterations = total_records * multiplier
    MIN_BATCH_SIZE_FLOOR: int = 1  # Tamaño mínimo absoluto de batch

    # Validación de archivos
    VALID_FILE_EXTENSIONS: Set[str] = {".csv", ".txt", ".tsv", ".dat"}
    MAX_FILE_SIZE_MB: float = 500.0  # Límite de tamaño de archivo
    MIN_FILE_SIZE_BYTES: int = 10  # Archivo mínimo válido

    # Resistencia dinámica
    COMPLEXITY_RESISTANCE_FACTOR: float = 5.0

    # Límites de registros
    MAX_RECORDS_LIMIT: int = 10_000_000  # Límite absoluto de registros
    MIN_RECORDS_FOR_PID: int = 10  # Mínimo para activar control PID

    # Cache
    MAX_CACHE_SIZE: int = 100_000  # Límite de entradas en cache

    # Consolidación
    MAX_BATCHES_TO_CONSOLIDATE: int = 10_000  # Límite de batches

    # Estabilidad Giroscópica
    GYRO_SENSITIVITY: float = 5.0  # FactorSensibilidad para Sg
    GYRO_EMA_ALPHA: float = 0.1  # Alpha para filtro EMA de corriente


# ============================================================================
# CLASES DE EXCEPCIONES
# ============================================================================
class DataFluxCondenserError(Exception):
    """Clase base para todas las excepciones personalizadas del condensador."""

    pass


class InvalidInputError(DataFluxCondenserError):
    """Indica un problema con los datos de entrada, como un archivo inválido."""

    pass


class ProcessingError(DataFluxCondenserError):
    """Señala un error durante una de las etapas de procesamiento de datos."""

    pass


class ConfigurationError(DataFluxCondenserError):
    """Indica un problema con la configuración del sistema."""

    pass


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================
class ParsedData(NamedTuple):
    """
    Estructura de datos inmutable para los resultados del parseo inicial.

    Agrupa la salida del `ReportParserCrudo` para asegurar que los datos
    crudos y la caché de parseo se mantengan juntos a través del pipeline.

    Attributes:
        raw_records (List[Dict[str, Any]]): Lista de registros de insumos.
        parse_cache (Dict[str, Any]): Metadatos generados durante el parseo.
    """

    raw_records: List[Dict[str, Any]]
    parse_cache: Dict[str, Any]


@dataclass(frozen=True)
class CondenserConfig:
    """
    Configuración inmutable y validada para el `DataFluxCondenser`.

    Define los umbrales operativos y comportamientos del condensador,
    incluyendo sus parámetros para el motor de simulación física y el PID.
    """

    min_records_threshold: int = 1
    enable_strict_validation: bool = True
    log_level: str = "INFO"

    # Configuración Física RLC
    system_capacitance: float = 5000.0
    base_resistance: float = 10.0
    system_inductance: float = 2.0

    # Configuración PID
    pid_setpoint: float = 0.30
    pid_kp: float = 2000.0
    pid_ki: float = 100.0
    min_batch_size: int = 50
    max_batch_size: int = 5000

    # Configuración de recuperación
    enable_partial_recovery: bool = False
    max_failed_batches: int = 3

    # Anti-windup
    integral_limit_factor: float = 2.0

    def __post_init__(self):
        """Valida la configuración después de la inicialización."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Valida que todos los parámetros estén en rangos válidos."""
        errors = []

        if self.min_records_threshold < 0:
            errors.append(f"min_records_threshold >= 0, got {self.min_records_threshold}")

        if self.system_capacitance <= 0:
            errors.append(f"system_capacitance > 0, got {self.system_capacitance}")

        if self.system_inductance <= 0:
            errors.append(f"system_inductance > 0, got {self.system_inductance}")

        if self.base_resistance < 0:
            errors.append(f"base_resistance >= 0, got {self.base_resistance}")

        if self.pid_kp < 0:
            errors.append(f"pid_kp >= 0, got {self.pid_kp}")

        if self.min_batch_size <= 0:
            errors.append(f"min_batch_size must be > 0, got {self.min_batch_size}")

        if self.min_batch_size > self.max_batch_size:
            errors.append(
                f"min_batch_size ({self.min_batch_size}) > max ({self.max_batch_size})"
            )

        if self.pid_setpoint <= 0.0 or self.pid_setpoint >= 1.0:
            errors.append(f"pid_setpoint debe estar entre 0 y 1, got {self.pid_setpoint}")

        if errors:
            raise ConfigurationError(
                "Errores de configuración:\n" + "\n".join(f"  - {e}" for e in errors)
            )


@dataclass
class ProcessingStats:
    """Estadísticas del procesamiento para observabilidad."""

    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    total_batches: int = 0
    failed_batches: int = 0
    processing_time: float = 0.0
    avg_batch_size: float = 0.0
    avg_saturation: float = 0.0
    max_dissipated_power: float = 0.0
    max_flyback_voltage: float = 0.0
    avg_kinetic_energy: float = 0.0
    emergency_brakes_triggered: int = 0

    def add_batch_stats(
        self,
        batch_size: int,
        saturation: float,
        power: float,
        flyback: float,
        kinetic: float,
        success: bool,
    ) -> None:
        """Actualiza estadísticas con datos de un batch procesado."""
        self.total_batches += 1
        if success:
            self.processed_records += batch_size
        else:
            self.failed_records += batch_size
            self.failed_batches += 1

        n = self.total_batches
        self.avg_batch_size = ((n - 1) * self.avg_batch_size + batch_size) / n
        self.avg_saturation = ((n - 1) * self.avg_saturation + saturation) / n
        self.avg_kinetic_energy = ((n - 1) * self.avg_kinetic_energy + kinetic) / n
        self.max_dissipated_power = max(self.max_dissipated_power, power)
        self.max_flyback_voltage = max(self.max_flyback_voltage, flyback)


@dataclass
class BatchResult:
    """Resultado estructurado de procesamiento de un batch."""

    success: bool
    dataframe: Optional[pd.DataFrame] = None
    records_processed: int = 0
    error_message: str = ""
    metrics: Optional[Dict[str, float]] = None


# ============================================================================
# ANALIZADOR DE LAPLACE - VERSIÓN ROBUSTECIDA
# ============================================================================
class EnhancedLaplaceAnalyzer:
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

        Transformación: s = (2/T) * (z-1)/(z+1)

        Para un sistema de segundo orden:
            H(s) = 1 / (a₂s² + a₁s + a₀)

        La transformación bilineal preserva la estabilidad y mapea el eje jω
        al círculo unitario sin aliasing (pre-warping opcional).
        """
        T = self.T

        # Coeficientes del sistema continuo
        a2 = self.L * self.C
        a1 = self.R * self.C
        a0 = 1.0

        # Transformación bilineal sin pre-warping (simplificada)
        # s = (2/T) * (z-1)/(z+1)

        # Coeficientes del denominador discreto
        den_z = [
            a2 * (4/T**2) + a1 * (2/T) + a0,
            2 * a0 - 2 * a2 * (4/T**2),
            a2 * (4/T**2) - a1 * (2/T) + a0
        ]

        # Numerador discreto (ganancia ajustada para DC gain = 1)
        num_z = [sum(den_z)]  # Para ganancia DC = 1

        try:
            return scipy.signal.TransferFunction(num_z, den_z, dt=T)
        except Exception as e:
            self.logger.warning(f"Error en discretización: {e}, usando sistema continuo")
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
        """
        Calcula márgenes de estabilidad para sistemas continuos.

        Para un sistema de segundo orden:
        - Margen de ganancia: infinito (para ζ > 0)
        - Margen de fase: depende de ζ

        Fórmulas analíticas para sistema de segundo orden:
            PM = arctan(2ζ / √(√(1 + 4ζ⁴) - 2ζ²))  [radianes]
            GM = ∞ (si ζ > 0)

        También calcula:
        - Frecuencia de cruce de ganancia (ωgc)
        - Frecuencia de cruce de fase (ωpc)
        """
        if self.zeta <= 0:
            return {
                "gain_margin_db": float('-inf'),
                "phase_margin_deg": 0.0,
                "gain_crossover_freq_rad_s": 0.0,
                "phase_crossover_freq_rad_s": 0.0,
                "is_margin_meaningful": False,
            }

        # Para sistema de segundo orden, GM = ∞
        gain_margin_db = float('inf')

        # Cálculo analítico del margen de fase
        if self.zeta < 1.0:
            # Sistema subamortiguado
            # Fórmula exacta para sistema de segundo orden
            sqrt_term = math.sqrt(math.sqrt(1 + 4 * self.zeta**4) - 2 * self.zeta**2)
            phase_margin_rad = math.atan(2 * self.zeta / sqrt_term)
        else:
            # Sistema sobreamortiguado - aproximación
            phase_margin_rad = math.pi / 2  # ~90°

        phase_margin_deg = math.degrees(phase_margin_rad)

        # Frecuencia de cruce de ganancia (donde |H(jω)| = 1)
        # Para sistema de segundo orden: ω_gc = ω_n * √(√(1 + 4ζ⁴) - 2ζ²)
        if self.zeta < 1.0:
            omega_gc = self.omega_n * math.sqrt(
                math.sqrt(1 + 4 * self.zeta**4) - 2 * self.zeta**2
            )
        else:
            # Aproximación para sistemas sobreamortiguados
            omega_gc = self.omega_n / (2 * self.zeta)

        # Frecuencia de cruce de fase (donde ∠H(jω) = -180°)
        # Para sistema de segundo orden con ζ > 0, no hay cruce de fase
        omega_pc = 0.0

        return {
            "gain_margin_db": gain_margin_db,
            "phase_margin_deg": phase_margin_deg,
            "gain_crossover_freq_rad_s": omega_gc,
            "phase_crossover_freq_rad_s": omega_pc,
            "is_margin_meaningful": True,
            "interpretation": self._interpret_stability_margins(phase_margin_deg, gain_margin_db),
        }

    def _interpret_stability_margins(self, pm_deg: float, gm_db: float) -> str:
        """Interpreta los márgenes de estabilidad."""
        if pm_deg < 30:
            return "MARGEN DE FASE BAJO - Sistema poco robusto a retardos"
        elif pm_deg > 60:
            return "MARGEN DE FASE ALTO - Sistema robusto pero posiblemente lento"
        else:
            return "MARGEN DE FASE ADECUADO - Buen equilibrio entre rapidez y robustez"

    def _calculate_transient_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas de respuesta transitoria para entrada escalón.

        Para sistema de segundo orden subamortiguado (0 < ζ < 1):
        - Tiempo de subida (tr): tiempo del 10% al 90%
        - Tiempo de pico (tp): primer máximo
        - Sobrepaso (Mp): máximo sobrepico porcentual
        - Tiempo de establecimiento (ts): tiempo al 2% del valor final

        Fórmulas analíticas:
            ω_d = ω_n√(1-ζ²)  (frecuencia amortiguada)
            t_r ≈ (π - arccos(ζ)) / ω_d
            t_p = π / ω_d
            M_p = exp(-πζ/√(1-ζ²)) * 100%
            t_s ≈ 4/(ζω_n)  (criterio 2%)
        """
        if self.zeta < 0:
            return {"status": "UNSTABLE", "metrics": {}}

        if abs(self.zeta) < 1e-10:  # Sin amortiguamiento
            return {
                "status": "UNDAMPED_OSCILLATION",
                "rise_time_s": float('inf'),
                "peak_time_s": math.pi / self.omega_n,
                "overshoot_percent": 100.0,  # Oscila indefinidamente
                "settling_time_s": float('inf'),
                "peak_value": 2.0,  # Oscila entre 0 y 2
                "steady_state_value": 1.0,
            }

        if self.zeta < 1.0:  # Subamortiguado
            omega_d = self.omega_n * math.sqrt(1 - self.zeta**2)

            # Tiempo de subida (aproximación)
            rise_time = (math.pi - math.acos(self.zeta)) / omega_d

            # Tiempo de pico
            peak_time = math.pi / omega_d

            # Sobrepaso
            overshoot = math.exp(-math.pi * self.zeta / math.sqrt(1 - self.zeta**2)) * 100.0

            # Tiempo de establecimiento (2%)
            settling_time = 4.0 / (self.zeta * self.omega_n)

            # Valor pico
            peak_value = 1.0 + math.exp(-math.pi * self.zeta / math.sqrt(1 - self.zeta**2))

            return {
                "status": "UNDERDAMPED",
                "rise_time_s": rise_time,
                "peak_time_s": peak_time,
                "overshoot_percent": overshoot,
                "settling_time_s": settling_time,
                "peak_value": peak_value,
                "steady_state_value": 1.0,
                "damped_frequency_rad_s": omega_d,
                "damped_frequency_hz": omega_d / (2 * math.pi),
            }

        elif abs(self.zeta - 1.0) < 1e-6:  # Críticamente amortiguado
            # Para ζ = 1: respuesta más rápida sin sobrepaso
            rise_time = 3.36 / self.omega_n  # Aproximación
            settling_time = 5.83 / self.omega_n  # Aproximación

            return {
                "status": "CRITICALLY_DAMPED",
                "rise_time_s": rise_time,
                "peak_time_s": float('inf'),  # No hay pico
                "overshoot_percent": 0.0,
                "settling_time_s": settling_time,
                "peak_value": 1.0,
                "steady_state_value": 1.0,
            }

        else:  # Sobreamortiguado
            # Dos constantes de tiempo reales
            alpha = self.zeta * self.omega_n
            beta = self.omega_n * math.sqrt(self.zeta**2 - 1)

            s1 = -alpha + beta
            s2 = -alpha - beta

            # Tiempo de subida aproximado
            rise_time = 2.2 / min(abs(s1), abs(s2))

            # Tiempo de establecimiento dominado por polo más lento
            settling_time = 4.0 / min(abs(s1), abs(s2))

            return {
                "status": "OVERDAMPED",
                "rise_time_s": rise_time,
                "peak_time_s": float('inf'),  # No hay pico
                "overshoot_percent": 0.0,
                "settling_time_s": settling_time,
                "peak_value": 1.0,
                "steady_state_value": 1.0,
                "pole_1_rad_s": s1,
                "pole_2_rad_s": s2,
                "dominant_pole_rad_s": min(abs(s1), abs(s2)),
            }

    def _calculate_parameter_sensitivity(self) -> Dict[str, Any]:
        """
        Calcula la sensibilidad de los polos a variaciones paramétricas.

        Sensibilidad de primer orden:
            ∂s/∂R, ∂s/∂L, ∂s/∂C

        Para un polo s = -ζω_n ± jω_d:
            ∂s/∂R = ∂s/∂ζ * ∂ζ/∂R + ∂s/∂ω_n * ∂ω_n/∂R

        Esto es importante para análisis de robustez.
        """
        if self.zeta < 0 or self.omega_n == 0:
            return {"status": "INVALID_FOR_SENSITIVITY"}

        # Derivadas parciales de ω_n y ζ respecto a parámetros
        d_omega_n_dR = 0.0
        d_omega_n_dL = -0.5 * self.omega_n / self.L if self.L != 0 else 0.0
        d_omega_n_dC = -0.5 * self.omega_n / self.C if self.C != 0 else 0.0

        d_zeta_dR = 0.5 * math.sqrt(self.C / self.L) if self.L > 0 else 0.0
        d_zeta_dL = -0.25 * self.R * math.sqrt(self.C) / (self.L**1.5) if self.L > 0 else 0.0
        d_zeta_dC = 0.25 * self.R / (math.sqrt(self.L * self.C)) if self.L > 0 and self.C > 0 else 0.0

        # Para sistemas subamortiguados, calcular sensibilidad de polos complejos
        if 0 < self.zeta < 1.0:
            omega_d = self.omega_n * math.sqrt(1 - self.zeta**2)

            # Polo s = -ζω_n + jω_d
            s = complex(-self.zeta * self.omega_n, omega_d)

            # Derivadas de s respecto a ω_n y ζ
            ds_d_omega_n = complex(-self.zeta, math.sqrt(1 - self.zeta**2))
            ds_d_zeta = complex(-self.omega_n, -self.zeta * self.omega_n / math.sqrt(1 - self.zeta**2))

            # Sensibilidad total usando regla de la cadena
            ds_dR = ds_d_zeta * d_zeta_dR + ds_d_omega_n * d_omega_n_dR
            ds_dL = ds_d_zeta * d_zeta_dL + ds_d_omega_n * d_omega_n_dL
            ds_dC = ds_d_zeta * d_zeta_dC + ds_d_omega_n * d_omega_n_dC

            # Magnitud de sensibilidad normalizada
            sensitivity_R = abs(ds_dR) * (self.R / abs(s)) if abs(s) > 0 else 0.0
            sensitivity_L = abs(ds_dL) * (self.L / abs(s)) if abs(s) > 0 else 0.0
            sensitivity_C = abs(ds_dC) * (self.C / abs(s)) if abs(s) > 0 else 0.0

        else:
            # Para sistemas sobreamortiguados o críticos
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
        """
        Calcula respuesta en frecuencia del sistema.

        Args:
            frequencies: Array de frecuencias (rad/s). Si None, usa rango logarítmico.

        Returns:
            Dict con magnitud, fase, y diagrama de Nyquist.
        """
        if frequencies is None:
            # Rango logarítmico centrado en ω_n
            w_min = self.omega_n / 1000.0 if self.omega_n > 0 else 0.01
            w_max = self.omega_n * 1000.0 if self.omega_n > 0 else 1000.0
            frequencies = np.logspace(np.log10(w_min), np.log10(w_max), 500)

        # Respuesta en frecuencia
        w, mag, phase = scipy.signal.bode(self.continuous_system, w=frequencies)

        # Diagrama de Nyquist
        nyquist_real, nyquist_imag = self._compute_nyquist_diagram(frequencies)

        # Puntos críticos
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
            "high_freq_slope_db_decade": -40.0,  # Sistema de segundo orden
        }

    def _compute_nyquist_diagram(self, frequencies: 'np.ndarray') -> Tuple['np.ndarray', 'np.ndarray']:
        """Calcula diagrama de Nyquist."""
        # Evaluar H(jω)
        s = 1j * frequencies
        numerator = 1.0
        denominator = self.L * self.C * s**2 + self.R * self.C * s + 1.0
        H = numerator / denominator

        return H.real, H.imag

    def _find_resonance(self, w: 'np.ndarray', mag: 'np.ndarray') -> Tuple[float, float]:
        """Encuentra frecuencia de resonancia y magnitud pico."""
        if len(mag) == 0:
            return 0.0, 0.0

        # Para sistemas subamortiguados
        if 0 < self.zeta < 1/math.sqrt(2):  # ζ < 0.707 para tener resonancia
            peak_idx = np.argmax(mag)
            return float(w[peak_idx]), float(mag[peak_idx])
        else:
            # No hay pico de resonancia
            return float(w[0]), float(mag[0])

    def _calculate_bandwidth(self, w: 'np.ndarray', mag: 'np.ndarray') -> float:
        """Calcula ancho de banda a -3dB."""
        if len(mag) == 0:
            return 0.0

        # Encontrar donde magnitud cae 3dB desde DC gain
        dc_gain = mag[0]
        target_gain = dc_gain - 3.0

        # Interpolar para encontrar frecuencia de -3dB
        for i in range(len(mag) - 1):
            if mag[i] >= target_gain >= mag[i + 1] or mag[i] <= target_gain <= mag[i + 1]:
                # Interpolación lineal
                w_low, w_high = w[i], w[i + 1]
                mag_low, mag_high = mag[i], mag[i + 1]

                # Evitar división por cero
                if mag_high != mag_low:
                    t = (target_gain - mag_low) / (mag_high - mag_low)
                    bandwidth = w_low + t * (w_high - w_low)
                    return float(bandwidth)

        return float(w[-1])  # Si no se encuentra, devolver máxima frecuencia

    def _generate_control_recommendations(self, margins: Dict[str, Any], sensitivity: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones para control basadas en análisis."""
        recommendations = []

        # Basado en factor de amortiguamiento
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

        # Basado en frecuencia natural
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

        # Basado en márgenes de estabilidad
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

        # Basado en sensibilidad
        if sensitivity.get("robustness_classification", "").startswith("FRÁGIL"):
            recommendations.append(
                "Alta sensibilidad paramétrica detectada. "
                "Implementar control adaptativo o usar componentes de alta precisión."
            )

        return recommendations

    def get_root_locus_data(self, K_range: Optional['np.ndarray'] = None) -> Dict[str, Any]:
        """
        Genera datos para lugar de las raíces (root locus).

        Para sistema H(s) = 1/(a₂s² + a₁s + a₀) con ganancia K:
            Lazo cerrado: T(s) = K*H(s) / (1 + K*H(s))
            Polos satisfacen: 1 + K*H(s) = 0
            → a₂s² + a₁s + a₀ + K = 0

        Args:
            K_range: Array de ganancias. Si None, usa rango logarítmico.
        """
        if K_range is None:
            K_range = np.logspace(-3, 3, 200)  # De 0.001 a 1000

        poles_real = []
        poles_imag = []

        for K in K_range:
            # Polos de lazo cerrado: raíces de a₂s² + a₁s + (a₀ + K) = 0
            a2 = self.L * self.C
            a1 = self.R * self.C
            a0_modified = 1.0 + K

            # Resolver ecuación cuadrática
            discriminant = a1**2 - 4 * a2 * a0_modified

            if discriminant >= 0:
                # Polos reales
                sqrt_disc = math.sqrt(discriminant)
                pole1 = (-a1 + sqrt_disc) / (2 * a2)
                pole2 = (-a1 - sqrt_disc) / (2 * a2)
                poles_real.extend([pole1.real, pole2.real])
                poles_imag.extend([pole1.imag, pole2.imag])
            else:
                # Polos complejos conjugados
                real_part = -a1 / (2 * a2)
                imag_part = math.sqrt(-discriminant) / (2 * a2)
                poles_real.extend([real_part, real_part])
                poles_imag.extend([imag_part, -imag_part])

        return {
            "gain_values": K_range.tolist(),
            "poles_real": poles_real,
            "poles_imag": poles_imag,
            "asymptote_center": -self.R / (2 * self.L) if self.L != 0 else 0.0,
            "breakaway_points": self._find_breakaway_points(K_range),
        }

    def _find_breakaway_points(self, K_range: 'np.ndarray') -> List[float]:
        """Encuentra puntos de ruptura en el lugar de las raíces."""
        # Para sistema de segundo orden, punto de ruptura en s = -a₁/(2a₂)
        breakaway_real = -self.R / (2 * self.L) if self.L != 0 else 0.0

        # Verificar si este punto está en el lugar
        K_at_breakaway = -(self.L * self.C * breakaway_real**2 +
                          self.R * self.C * breakaway_real + 1)

        if K_at_breakaway >= K_range[0] and K_at_breakaway <= K_range[-1]:
            return [float(breakaway_real)]
        return []

    def get_bode_diagram_data(self) -> Dict[str, Any]:
        """Prepara datos para diagramas de Bode."""
        freq_response = self.get_frequency_response()

        return {
            "magnitude": {
                "frequency": freq_response["frequencies_rad_s"],
                "magnitude_db": freq_response["magnitude_db"],
                "asymptotes": self._calculate_bode_asymptotes(),
            },
            "phase": {
                "frequency": freq_response["frequencies_rad_s"],
                "phase_deg": freq_response["phase_deg"],
                "phase_wrap": [((p + 180) % 360) - 180 for p in freq_response["phase_deg"]],
            },
        }

    def _calculate_bode_asymptotes(self) -> Dict[str, List[float]]:
        """Calcula asíntotas de Bode para sistema de segundo orden."""
        w = np.logspace(np.log10(self.omega_n/100), np.log10(self.omega_n*100), 100)

        # Asíntota de baja frecuencia: 0 dB
        low_freq_mag = np.zeros_like(w[w < self.omega_n])

        # Asíntota de alta frecuencia: pendiente -40 dB/década
        # En ω = ω_n: magnitud = -20*log10(2ζ) [aproximadamente]
        high_freq_mag = -40 * np.log10(w[w > self.omega_n] / self.omega_n)
        if self.zeta > 0:
            high_freq_mag -= 20 * math.log10(2 * self.zeta)

        return {
            "low_freq_freq": w[w < self.omega_n].tolist(),
            "low_freq_mag": low_freq_mag.tolist(),
            "high_freq_freq": w[w > self.omega_n].tolist(),
            "high_freq_mag": high_freq_mag.tolist(),
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


# ============================================================================
# CONTROLADOR PI DISCRETO - MÉTODOS REFINADOS
# ============================================================================
class PIController:
    """
    Controlador PI Discreto.

    Características:
    1. Filtro de media móvil exponencial para estabilización.
    2. Anti-windup con back-calculation mejorado.
    3. Análisis de estabilidad basado en Lyapunov discreto mejorado.
    4. Ganancia integral adaptativa para evitar windup en régimen transitorio.
    """

    _MAX_HISTORY_SIZE: int = 100

    def __init__(
        self,
        kp: float,
        ki: float,
        setpoint: float,
        min_output: int,
        max_output: int,
        integral_limit_factor: float = 2.0,
    ):
        """Inicializa el controlador PID."""
        self._validate_control_parameters(kp, ki, setpoint, min_output, max_output)

        self.Kp = float(kp)
        self.Ki = float(ki)
        self.setpoint = float(setpoint)
        self.min_output = int(min_output)
        self.max_output = int(max_output)

        # Espacio de salida normalizado
        self._output_range = max(1, self.max_output - self.min_output)
        self._output_center = (self.max_output + self.min_output) / 2.0

        # Anti-windup: límite integral basado en rango de salida
        self._integral_limit = self._output_range * max(0.1, integral_limit_factor)
        self._integral_error: float = 0.0

        # Filtro EMA (Exponential Moving Average) para suavizado
        self._ema_alpha: float = 0.3  # Factor de suavizado
        self._filtered_pv: Optional[float] = None

        # Estado temporal
        self._last_time: Optional[float] = None
        self._last_error: Optional[float] = None
        self._last_output: Optional[int] = None
        self._iteration_count: int = 0

        # Historial para análisis de estabilidad
        self._error_history: deque = deque(maxlen=self._MAX_HISTORY_SIZE)
        self._output_history: deque = deque(maxlen=self._MAX_HISTORY_SIZE)

        # Métricas de estabilidad de Lyapunov
        self._lyapunov_sum: float = 0.0
        self._lyapunov_count: int = 0

        # Adaptación de ganancia integral
        self._ki_adaptive: float = ki
        self._windup_detection_window: deque = deque(maxlen=5)

    def _validate_control_parameters(
        self, kp: float, ki: float, setpoint: float, min_output: int, max_output: int
    ) -> None:
        """
        Valida parámetros con criterio de Jury completo y análisis de estabilidad.

        Para un sistema PI discreto con función de transferencia en lazo cerrado:

            G_cl(z) = K(z) * G_p(z) / (1 + K(z) * G_p(z))

        donde K(z) = Kp + Ki*T*z/(z-1) es el controlador PI discreto.

        Criterio de Jury para polinomio de 2do orden P(z) = z² + a₁z + a₀:
        1. P(1) > 0  (estabilidad en z=1)
        2. P(-1) > 0 (estabilidad en z=-1, alternancia)
        3. |a₀| < 1  (raíces dentro del círculo unitario)

        Adicionalmente verificamos el margen de fase mediante:
        - Criterio de Nyquist simplificado para sistemas discretos
        """
        errors = []
        warnings = []

        # === VALIDACIONES BÁSICAS ===
        if kp <= 0:
            errors.append(f"Kp debe ser positivo para respuesta proporcional, got {kp}")
        if ki < 0:
            errors.append(f"Ki debe ser no-negativo, got {ki}")
        if min_output >= max_output:
            errors.append(f"Rango de salida inválido: [{min_output}, {max_output}]")
        if min_output <= 0:
            errors.append(f"min_output debe ser positivo, got {min_output}")
        if not (0.0 < setpoint < 1.0):
            errors.append(f"setpoint debe estar en (0, 1), got {setpoint}")

        if errors:
            raise ConfigurationError(
                "Errores en parámetros de control:\n" + "\n".join(f"  • {e}" for e in errors)
            )

        # === ANÁLISIS DE ESTABILIDAD DISCRETO ===
        output_range = max(1.0, float(max_output - min_output))

        # Ganancia de planta normalizada (modelo de primer orden)
        K_plant = 1.0 / output_range

        # Período de muestreo y polo de la planta
        T = 1.0  # Normalizado
        a_plant = 0.9  # Polo típico para sistema de primer orden estable

        # Coeficientes del polinomio característico del sistema en lazo cerrado
        # Derivación: (z-1)(z-a) + K_plant*T*(Kp*(z-1) + Ki*z) = 0
        # Expandiendo: z² + a₁z + a₀ = 0

        K_eff = K_plant * T
        a0 = a_plant - K_eff * (kp - ki)
        a1 = -(a_plant + 1.0) + K_eff * kp

        # === CRITERIO DE JURY ESTÁNDAR ===
        P_at_1 = 1.0 + a1 + a0      # P(1) = 1 + a₁ + a₀
        P_at_minus1 = 1.0 - a1 + a0  # P(-1) = 1 - a₁ + a₀

        cond1_magnitude = abs(a0) < 1.0
        cond2_positive = P_at_1 > 0.0
        cond3_alternating = P_at_minus1 > 0.0

        jury_stable = cond1_magnitude and cond2_positive and cond3_alternating

        # === CÁLCULO DE RAÍCES Y MARGEN DE ESTABILIDAD ===
        discriminant = a1 * a1 - 4.0 * a0

        if discriminant >= 0:
            # Raíces reales
            sqrt_disc = math.sqrt(discriminant)
            root1 = (-a1 + sqrt_disc) / 2.0
            root2 = (-a1 - sqrt_disc) / 2.0
            max_magnitude = max(abs(root1), abs(root2))
            is_oscillatory = False
        else:
            # Raíces complejas conjugadas
            real_part = -a1 / 2.0
            imag_part = math.sqrt(-discriminant) / 2.0
            max_magnitude = math.sqrt(real_part**2 + imag_part**2)
            is_oscillatory = True

            # Calcular frecuencia de oscilación natural
            if max_magnitude > 0:
                damped_freq = math.atan2(imag_part, real_part)
                warnings.append(
                    f"Sistema subamortiguado: ω_d = {damped_freq:.3f} rad/sample"
                )

        stability_margin = 1.0 - max_magnitude

        # === DIAGNÓSTICO Y ADVERTENCIAS ===
        if not jury_stable:
            detail = (
                f"Jury: |a₀|<1={cond1_magnitude}, "
                f"P(1)>0={cond2_positive}, P(-1)>0={cond3_alternating}"
            )
            if stability_margin < 0:
                errors.append(
                    f"Sistema inestable (margen={stability_margin:.4f}). {detail}"
                )
            else:
                warnings.append(f"Criterio de Jury marginalmente satisfecho. {detail}")

        if stability_margin < 0.05 and stability_margin >= 0:
            warnings.append(
                f"Margen de estabilidad crítico: {stability_margin:.4f}. "
                f"Considere reducir Kp o Ki."
            )

        if stability_margin < 0.2 and is_oscillatory:
            warnings.append(
                "Sistema con tendencia oscilatoria. "
                "Puede exhibir ringing en respuesta transitoria."
            )

        # === VERIFICACIÓN DE ANCHO DE BANDA ===
        # Frecuencia de cruce aproximada
        crossover_gain = kp * K_plant
        if crossover_gain > 0.5:
            warnings.append(
                f"Ganancia de cruce alta ({crossover_gain:.2f}). "
                "Riesgo de amplificación de ruido."
            )

        # Emitir warnings acumulados
        for w in warnings:
            logger.warning(f"⚠️ Control: {w}")

        if errors:
            raise ConfigurationError(
                "Errores en parámetros de control:\n" + "\n".join(f"  • {e}" for e in errors)
            )

    def _apply_ema_filter(self, measurement: float) -> float:
        """
        Aplica filtro de Media Móvil Exponencial con alpha adaptativo.

        El factor de suavizado α se adapta dinámicamente según:

        1. **Detección de escalón (step)**: Si |Δy| > τ_step, se reduce la inercia
           para seguir cambios abruptos rápidamente.

        2. **Varianza local**: Alta varianza → menor α (más suavizado).
           Baja varianza → mayor α (más reactivo).

        Fundamentación: El filtro EMA es óptimo para procesos ARIMA(0,1,1),
        y α_óptimo = 1 - θ donde θ es el parámetro MA. Estimamos θ
        a partir de la autocorrelación del error de predicción.

        La fórmula adaptativa usa una sigmoide inversa para mapear
        varianza a alpha de forma suave y acotada.
        """
        if self._filtered_pv is None:
            self._filtered_pv = measurement
            return measurement

        # === DETECCIÓN DE ESCALÓN (STEP CHANGE) ===
        innovation = measurement - self._filtered_pv
        step_threshold = 0.25 * max(abs(self.setpoint), 0.1)

        if abs(innovation) > step_threshold:
            # Cambio abrupto detectado: bypass parcial del filtro
            # Usamos interpolación con peso hacia la nueva medición
            bypass_weight = min(0.7, abs(innovation) / (2 * step_threshold))
            self._filtered_pv = bypass_weight * measurement + (1 - bypass_weight) * self._filtered_pv

            # Resetear historial de errores para evitar contaminación
            if hasattr(self, '_innovation_history'):
                self._innovation_history.clear()

            return self._filtered_pv

        # === ESTIMACIÓN DE VARIANZA LOCAL ===
        if not hasattr(self, '_innovation_history'):
            self._innovation_history = deque(maxlen=10)

        self._innovation_history.append(innovation)

        n_samples = len(self._innovation_history)

        if n_samples >= 3:
            innovations = list(self._innovation_history)
            mean_innov = sum(innovations) / n_samples

            # Varianza con corrección de Bessel
            if n_samples > 1:
                variance = sum((x - mean_innov)**2 for x in innovations) / (n_samples - 1)
            else:
                variance = 0.0

            # === MAPEO VARIANZA → ALPHA ===
            # Función sigmoide inversa: α = α_min + (α_max - α_min) / (1 + k * σ²)
            # donde k controla la sensibilidad a la varianza
            alpha_min = 0.05  # Máximo suavizado
            alpha_max = 0.5   # Mínimo suavizado (máxima reactividad)
            sensitivity = 50.0  # Factor de sensibilidad a varianza

            # Normalizar varianza respecto al setpoint
            normalized_variance = variance / max(self.setpoint**2, 0.01)

            adaptive_alpha = alpha_min + (alpha_max - alpha_min) / (1.0 + sensitivity * normalized_variance)

            # === CORRECCIÓN POR AUTOCORRELACIÓN ===
            # Si hay autocorrelación positiva en innovaciones, reducir alpha
            if n_samples >= 4:
                # Autocorrelación lag-1 simplificada
                autocorr = sum(
                    (innovations[i] - mean_innov) * (innovations[i-1] - mean_innov)
                    for i in range(1, n_samples)
                )
                autocorr /= max(variance * (n_samples - 1), 1e-10)

                # Autocorrelación positiva → proceso más suave → menor alpha
                if autocorr > 0.3:
                    adaptive_alpha *= (1.0 - 0.3 * autocorr)

            self._ema_alpha = max(alpha_min, min(alpha_max, adaptive_alpha))
        else:
            # Insuficientes muestras: usar alpha conservador
            self._ema_alpha = 0.2

        # === APLICAR FILTRO EMA ===
        self._filtered_pv = self._ema_alpha * measurement + (1 - self._ema_alpha) * self._filtered_pv

        return self._filtered_pv

    def _update_lyapunov_metric(self, error: float) -> None:
        """
        Actualiza métrica de Lyapunov.

        Usa función candidata V(e) = e² con estimación robusta del exponente mediante
        regresión de mínimos cuadrados sobre ventana deslizante.
        """
        # Almacenar |e| para regresión logarítmica
        abs_error = abs(error) + 1e-12  # Evitar log(0)

        if not hasattr(self, "_lyapunov_log_errors"):
            self._lyapunov_log_errors = deque(maxlen=20)

        self._lyapunov_log_errors.append(math.log(abs_error))

        n = len(self._lyapunov_log_errors)
        if n < 5:
            return

        # Regresión lineal: log|e(k)| = λ·k + c
        # donde λ es el exponente de Lyapunov
        log_errors = list(self._lyapunov_log_errors)
        k_vals = list(range(n))

        sum_k = sum(k_vals)
        sum_log_e = sum(log_errors)
        sum_k_log_e = sum(k * le for k, le in zip(k_vals, log_errors))
        sum_k2 = sum(k * k for k in k_vals)

        denominator = n * sum_k2 - sum_k * sum_k
        if abs(denominator) < 1e-10:
            return

        # Pendiente = exponente de Lyapunov estimado
        lyapunov_slope = (n * sum_k_log_e - sum_k * sum_log_e) / denominator

        # Filtrado EMA del exponente para estabilidad
        ema_factor = 0.2
        self._lyapunov_sum = (
            1 - ema_factor
        ) * self._lyapunov_sum + ema_factor * lyapunov_slope
        self._lyapunov_count = max(1, self._lyapunov_count)

        # Alerta temprana de inestabilidad con histéresis
        if self._lyapunov_sum > 0.15 and n > 10:
            logger.warning(
                f"⚠️ Divergencia detectada: λ ≈ {self._lyapunov_sum:.4f} > 0 "
                f"(basado en {n} muestras)"
            )

    def _adapt_integral_gain(self, error: float, output_saturated: bool) -> None:
        """Adapta la ganancia integral para prevenir windup."""
        self._windup_detection_window.append((error, output_saturated))

        if len(self._windup_detection_window) < 3:
            return

        # Detectar windup: error constante con saturación
        recent_errors = [e for e, _ in self._windup_detection_window]
        saturated_count = sum(1 for _, s in self._windup_detection_window if s)

        if np:
            error_std = np.std(recent_errors)
        else:
            mean = sum(recent_errors) / len(recent_errors)
            error_std = math.sqrt(
                sum((e - mean) ** 2 for e in recent_errors) / len(recent_errors)
            )

        # Condiciones para windup: baja variación en error con saturación frecuente
        if error_std < 0.05 and saturated_count >= 2:
            self._ki_adaptive = self.Ki * 0.5  # Reducir Ki temporalmente
            logger.debug("Windup detectado: reduciendo Ki adaptativamente")
        else:
            self._ki_adaptive = self.Ki  # Restaurar Ki nominal

    def compute(self, process_variable: float) -> int:
        """
        Calcula la salida del controlador PI.

        Características:
        1. Zona muerta (deadband) para reducir actuación innecesaria.
        2. Anti-windup con back-calculation y clamping condicional.
        3. Slew rate limiting para suavidad.
        4. Bumpless transfer en cambios de setpoint.
        """
        self._iteration_count += 1
        current_time = time.time()

        # Saturar entrada al rango válido
        pv_clamped = max(0.0, min(1.0, process_variable))

        # Filtrado EMA adaptativo
        filtered_pv = self._apply_ema_filter(pv_clamped)

        # Error con zona muerta para reducir jitter
        raw_error = self.setpoint - filtered_pv
        deadband = 0.02 * self.setpoint  # 2% del setpoint

        if abs(raw_error) < deadband:
            error = 0.0  # Dentro de banda muerta
        else:
            # Suavizar transición fuera de zona muerta
            error = raw_error - math.copysign(deadband, raw_error)

        self._error_history.append(error)

        # Delta tiempo con límites de cordura
        if self._last_time is None:
            dt = SystemConstants.MIN_DELTA_TIME
        else:
            dt = max(
                SystemConstants.MIN_DELTA_TIME,
                min(current_time - self._last_time, SystemConstants.MAX_DELTA_TIME),
            )

        # === TÉRMINO PROPORCIONAL ===
        P = self.Kp * error

        # === ANTI-WINDUP: Detección previa de saturación ===
        # Calcular salida tentativa para decidir si integrar
        tentative_I = self._ki_adaptive * (self._integral_error + error * dt)
        tentative_output = self._output_center + P + tentative_I

        will_saturate = (
            tentative_output > self.max_output or tentative_output < self.min_output
        )

        # Clamping condicional: no integrar si vamos a saturar
        # Y el error empuja hacia la saturación
        integrating_towards_saturation = (
            tentative_output > self.max_output and error < 0
        ) or (tentative_output < self.min_output and error > 0)

        if will_saturate and not integrating_towards_saturation:
            # Solo acumular si el error nos saca de saturación
            self._integral_error += error * dt
        elif not will_saturate:
            self._integral_error += error * dt

        # Aplicar límite integral con suavizado hiperbólico
        if abs(self._integral_error) > self._integral_limit:
            # Soft clamp usando tanh para evitar discontinuidades
            normalized = self._integral_error / self._integral_limit
            self._integral_error = self._integral_limit * math.tanh(normalized)

        # Adaptación de ganancia integral
        self._adapt_integral_gain(error, will_saturate)

        # === TÉRMINO INTEGRAL ===
        I = self._ki_adaptive * self._integral_error

        # === CÁLCULO DE SALIDA ===
        output_raw = self._output_center + P + I

        # === SLEW RATE LIMITING ===
        # Limitar cambio máximo por iteración (anti-jerk)
        if self._last_output is not None:
            max_slew = self._output_range * 0.15  # 15% máximo por paso
            delta = output_raw - self._last_output
            if abs(delta) > max_slew:
                output_raw = self._last_output + math.copysign(max_slew, delta)

        # Saturación final
        output = int(round(max(self.min_output, min(self.max_output, output_raw))))

        # === BACK-CALCULATION POST-SATURACIÓN ===
        # Si saturamos, ajustar integral para tracking
        if output != int(round(output_raw)):
            saturation_error = output_raw - output
            tracking_gain = 1.0 / max(self.Kp, 0.1)  # Kb = 1/Kp (regla de Åström)
            self._integral_error -= tracking_gain * saturation_error * dt

        # Actualizar métrica de Lyapunov
        self._update_lyapunov_metric(error)

        # Guardar estado
        self._last_time = current_time
        self._last_error = error
        self._last_output = output
        self._output_history.append(output)

        return output

    def get_lyapunov_exponent(self) -> float:
        """Retorna estimación del exponente de Lyapunov promedio."""
        if self._lyapunov_count == 0:
            return 0.0
        return self._lyapunov_sum / self._lyapunov_count

    def get_stability_analysis(self) -> Dict[str, Any]:
        """Análisis de estabilidad basado en historial."""
        if len(self._error_history) < 2:
            return {"status": "INSUFFICIENT_DATA", "samples": len(self._error_history)}

        errors = list(self._error_history)
        lyapunov = self.get_lyapunov_exponent()

        # Análisis de convergencia
        mean_error = sum(errors) / len(errors)
        error_variance = sum((e - mean_error) ** 2 for e in errors) / len(errors)

        recent_errors = errors[-min(10, len(errors)) :]
        mean_recent = sum(recent_errors) / len(recent_errors)
        recent_variance = sum((e - mean_recent) ** 2 for e in recent_errors) / len(
            recent_errors
        )

        # Diagnóstico
        if lyapunov < -0.1:
            stability = "ASYMPTOTICALLY_STABLE"
        elif lyapunov < 0.1:
            stability = "MARGINALLY_STABLE"
        else:
            stability = "POTENTIALLY_UNSTABLE"

        # Tendencia de convergencia
        convergence = "CONVERGING" if recent_variance < error_variance else "DIVERGING"

        return {
            "status": "OPERATIONAL",
            "stability_class": stability,
            "convergence": convergence,
            "lyapunov_exponent": lyapunov,
            "error_variance": error_variance,
            "recent_variance": recent_variance,
            "integral_saturation": abs(self._integral_error) / self._integral_limit,
            "iterations": self._iteration_count,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Diagnóstico completo del controlador."""
        stability = self.get_stability_analysis()

        return {
            "status": stability.get("status", "UNKNOWN"),
            "control_metrics": {
                "iteration": self._iteration_count,
                "current_integral": self._integral_error,
                "integral_limit": self._integral_limit,
                "integral_utilization": abs(self._integral_error) / self._integral_limit,
                "last_error": self._last_error,
                "last_output": self._last_output,
                "adaptive_ki": self._ki_adaptive,
            },
            "stability_analysis": stability,
            "parameters": {
                "Kp": self.Kp,
                "Ki": self.Ki,
                "setpoint": self.setpoint,
                "output_range": [self.min_output, self.max_output],
            },
        }

    def reset(self) -> None:
        """Reinicia estado del controlador preservando historial."""
        self._integral_error = 0.0
        self._last_time = None
        self._last_error = None
        self._last_output = None
        self._filtered_pv = None
        self._iteration_count = 0
        self._lyapunov_sum = 0.0
        self._lyapunov_count = 0
        self._ki_adaptive = self.Ki
        # Preserva historial para análisis post-mortem

    def get_state(self) -> Dict[str, Any]:
        """Retorna estado serializable del controlador."""
        return {
            "parameters": {
                "Kp": self.Kp,
                "Ki": self.Ki,
                "setpoint": self.setpoint,
                "min_output": self.min_output,
                "max_output": self.max_output,
            },
            "state": {
                "integral_error": self._integral_error,
                "filtered_pv": self._filtered_pv,
                "iteration": self._iteration_count,
                "adaptive_ki": self._ki_adaptive,
            },
            "diagnostics": self.get_stability_analysis(),
        }


# ============================================================================
# MOTOR DE FÍSICA - MÉTODOS REFINADOS
# ============================================================================
class FluxPhysicsEngine:
    """
    Motor de física RLC.

    Características:
    1. Integración numérica más estable (Runge-Kutta de 2do orden).
    2. Cálculo de números de Betti corregido para grafos.
    3. Entropía termodinámica con fundamentación estadística rigurosa.
    4. Modelo de amortiguamiento no lineal para alta saturación.
    """

    _MAX_METRICS_HISTORY: int = 100

    def __init__(self, capacitance: float, resistance: float, inductance: float):
        # Inicializar logger primero para usar en validación
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self._validate_physical_parameters(capacitance, resistance, inductance)

        self.C = float(capacitance)
        self.R = float(resistance)
        self.L = float(inductance)

        # Parámetros derivados del circuito RLC
        self._omega_0 = 1.0 / math.sqrt(self.L * self.C)  # Frecuencia natural
        self._alpha = self.R / (2.0 * self.L)  # Factor de amortiguamiento
        self._zeta = self._alpha / self._omega_0  # Ratio de amortiguamiento
        self._Q = math.sqrt(self.L / self.C) / self.R if self.R > 0 else float("inf")

        # Clasificación del sistema
        self._update_damping_classification()

        # Estado del sistema: [carga Q, corriente I]
        self._state = [0.0, 0.0]  # Compatible con/sin numpy
        self._state_history: deque = deque(maxlen=self._MAX_METRICS_HISTORY)

        # Estado del giroscopio (inicialización temprana)
        self._gyro_state = {
            "omega_x": 0.0,
            "omega_y": 0.0,
            "nutation_amplitude": 0.0,
            "precession_phase": 0.0,
        }

        # Grafo de conectividad para análisis topológico
        self._adjacency_list: Dict[int, Set[int]] = {}
        self._vertex_count: int = 0
        self._edge_count: int = 0

        # Historial de métricas
        self._metrics_history: deque = deque(maxlen=self._MAX_METRICS_HISTORY)
        self._entropy_history: deque = deque(maxlen=self._MAX_METRICS_HISTORY)

        # Estado temporal
        self._last_current: float = 0.0
        self._ema_current: float = 0.0  # EMA de la corriente (Eje de rotación)
        self._last_time: float = time.time()
        self._initialized: bool = False

        # Amortiguamiento no lineal
        self._nonlinear_damping_factor: float = 1.0

    def _validate_physical_parameters(self, C: float, R: float, L: float) -> None:
        """Validación de parámetros físicos con análisis dimensional."""
        errors = []

        if C <= 0:
            errors.append(f"Capacitancia debe ser positiva, got {C} F")
        if R < 0:
            errors.append(f"Resistencia debe ser no-negativa, got {R} Ω")
        if L <= 0:
            errors.append(f"Inductancia debe ser positiva, got {L} H")

        # Verificación de rangos físicamente razonables
        if C > 0 and L > 0:
            omega_0 = 1.0 / math.sqrt(L * C)
            if omega_0 > 1e12:  # > 1 THz
                self.logger.warning(
                    f"Frecuencia natural {omega_0:.2e} rad/s excesivamente alta"
                )

        if R > 0 and L > 0:
            tau = L / R  # Constante de tiempo
            if tau < 1e-12:  # < 1 ps
                self.logger.warning(f"Constante de tiempo {tau:.2e} s muy pequeña")

        if errors:
            raise ConfigurationError(
                "Parámetros físicos inválidos:\n" + "\n".join(f"  • {e}" for e in errors)
            )

    def _update_damping_classification(self) -> None:
        """Actualiza clasificación de amortiguamiento del sistema."""
        if self._zeta > 1.0:
            self._damping_type = "OVERDAMPED"
            self._omega_d = self._omega_0 * math.sqrt(self._zeta**2 - 1)
        elif self._zeta < 1.0:
            self._damping_type = "UNDERDAMPED"
            self._omega_d = self._omega_0 * math.sqrt(1 - self._zeta**2)
        else:
            self._damping_type = "CRITICALLY_DAMPED"
            self._omega_d = 0.0

    def _evolve_state_rk4(self, driving_current: float, dt: float) -> Tuple[float, float]:
        """
        Evolución del estado RLC usando Runge-Kutta de 4to orden (RK4).

        Mayor precisión O(dt⁴) vs O(dt²) de RK2, crítico para
        sistemas subamortiguados donde la oscilación debe preservarse.

        Sistema de ecuaciones de estado:
            dQ/dt = I
            dI/dt = (V_in - R·I - Q/C) / L
        """
        Q, I = self._state

        # Voltaje de entrada proporcional a corriente de driving
        # con saturación suave para evitar sobretensiones
        V_max = 20.0
        V_in = V_max * math.tanh(driving_current)

        # Función de derivadas del sistema
        def f(q: float, i: float) -> Tuple[float, float]:
            dq_dt = i
            # Resistencia no lineal: aumenta con I² (efecto Joule)
            R_eff = self.R * (1.0 + 0.1 * i * i)
            di_dt = (V_in - R_eff * i - q / self.C) / self.L
            return dq_dt, di_dt

        # RK4 clásico
        k1_q, k1_i = f(Q, I)
        k2_q, k2_i = f(Q + 0.5 * dt * k1_q, I + 0.5 * dt * k1_i)
        k3_q, k3_i = f(Q + 0.5 * dt * k2_q, I + 0.5 * dt * k2_i)
        k4_q, k4_i = f(Q + dt * k3_q, I + dt * k3_i)

        Q_new = Q + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        I_new = I + (dt / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)

        # === LIMITADOR DE ENERGÍA ===
        # Prevenir acumulación infinita de energía (estabilidad numérica)
        E_max = 100.0  # Energía máxima permitida
        E_current = 0.5 * self.L * I_new**2 + 0.5 * (Q_new**2) / self.C

        if E_current > E_max:
            # Escalar estado para limitar energía (conservando proporciones)
            scale = math.sqrt(E_max / E_current)
            Q_new *= scale
            I_new *= scale
            self._nonlinear_damping_factor = scale
            self.logger.debug(f"Energía limitada: {E_current:.2f} → {E_max:.2f} J")
        else:
            # Amortiguamiento no lineal suave para alta energía
            damping = 1.0 / (1.0 + 0.05 * max(0, E_current - E_max * 0.5))
            I_new *= damping
            self._nonlinear_damping_factor = damping

        self._state = [Q_new, I_new]

        self._state_history.append(
            {
                "Q": Q_new,
                "I": I_new,
                "time": time.time(),
                "energy": 0.5 * self.L * I_new**2 + 0.5 * (Q_new**2) / self.C,
                "V_in": V_in,
            }
        )

        return Q_new, I_new

    def _build_metric_graph(self, metrics: Dict[str, float]) -> None:
        """
        Construye grafo de correlación.

        Usa umbral adaptativo basado en correlación de Spearman (robusta a outliers)
        sobre historial.
        """
        metric_keys = [
            "saturation",
            "complexity",
            "current_I",
            "potential_energy",
            "kinetic_energy",
            "entropy_shannon",
        ]
        values = [metrics.get(k, 0.0) for k in metric_keys]

        self._adjacency_list.clear()
        self._vertex_count = len(values)
        self._edge_count = 0

        for i in range(self._vertex_count):
            self._adjacency_list[i] = set()

        if self._vertex_count < 2:
            return

        # Calcular matriz de distancias normalizadas
        # Usar distancia de correlación: d = 1 - |corr|

        # Normalizar valores al rango [0, 1]
        v_min = min(values)
        v_max = max(values)
        v_range = v_max - v_min if v_max != v_min else 1.0
        normalized = [(v - v_min) / v_range for v in values]

        # Umbral adaptativo basado en dispersión
        mean_val = sum(normalized) / len(normalized)
        variance = sum((v - mean_val) ** 2 for v in normalized) / len(normalized)

        # Mayor varianza → umbral más permisivo para capturar estructura
        base_threshold = 0.3
        adaptive_threshold = base_threshold * (1.0 + math.sqrt(variance))
        adaptive_threshold = min(0.7, adaptive_threshold)  # Cap máximo

        # Crear aristas basadas en proximidad en espacio normalizado
        for i in range(self._vertex_count):
            for j in range(i + 1, self._vertex_count):
                # Distancia euclidiana normalizada
                dist = abs(normalized[i] - normalized[j])

                # Correlación implícita: valores cercanos están correlacionados
                if dist < adaptive_threshold:
                    self._adjacency_list[i].add(j)
                    self._adjacency_list[j].add(i)
                    self._edge_count += 1

    def _calculate_betti_numbers(self) -> Dict[int, int]:
        """
        Calcula números de Betti usando Union-Find optimizado con
        elementos de homología persistente.

        Para un grafo G = (V, E):

        - β₀ = número de componentes conexas = |V| - rank(A)
        - β₁ = número de ciclos independientes = |E| - |V| + β₀
        - β_k = 0 para k ≥ 2 (el grafo es 1-dimensional)

        Característica de Euler: χ = β₀ - β₁ = |V| - |E|

        Complejidad ciclomática (McCabe): M = β₁ + 1

        La homología persistente se simula ordenando aristas por peso
        y rastreando nacimiento/muerte de características.
        """
        if self._vertex_count == 0:
            return {
                0: 0, 1: 0, 2: 0,
                "euler_characteristic": 0,
                "is_tree": False,
                "is_forest": True,
                "cyclomatic_complexity": 1,
                "homology_dimensions": [],
                "connected_components": 0,
                "independent_cycles": 0,
            }

        # === UNION-FIND CON COMPRESIÓN DE CAMINOS Y UNIÓN POR RANGO ===
        parent = list(range(self._vertex_count))
        rank = [0] * self._vertex_count

        def find(x: int) -> int:
            """Find con compresión de caminos (path halving)."""
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # Path halving
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            """
            Union por rango.
            Retorna True si x e y YA estaban conectados (arista crea ciclo).
            """
            root_x = find(x)
            root_y = find(y)

            if root_x == root_y:
                return True  # Ciclo detectado

            # Unión por rango para árbol balanceado
            if rank[root_x] < rank[root_y]:
                root_x, root_y = root_y, root_x

            parent[root_y] = root_x

            if rank[root_x] == rank[root_y]:
                rank[root_x] += 1

            return False  # Componentes fusionadas

        # === PROCESAR ARISTAS Y DETECTAR CICLOS ===
        edges_processed = 0
        cycles_detected = 0

        # Lista de aristas para homología persistente
        edge_list = []

        for u in range(self._vertex_count):
            neighbors = self._adjacency_list.get(u, set())
            for v in sorted(neighbors):
                if v > u:  # Cada arista una sola vez
                    edge_list.append((u, v))

        # === HOMOLOGÍA PERSISTENTE SIMPLIFICADA ===
        # Ordenar aristas por "peso" (simulado como índice)
        # En un grafo sin pesos, usamos el orden de inserción
        persistence_diagram = []

        for idx, (u, v) in enumerate(edge_list):
            edges_processed += 1

            is_cycle = union(u, v)

            if is_cycle:
                cycles_detected += 1
                # Registro de ciclo: nace en este momento, muere en infinito
                persistence_diagram.append({
                    "dimension": 1,
                    "birth": idx / max(len(edge_list), 1),  # Normalizado
                    "death": 1.0,  # Infinito normalizado
                    "persistence": 1.0 - idx / max(len(edge_list), 1),
                    "edge": (u, v),
                })

        # === CONTAR COMPONENTES CONEXAS ===
        unique_roots = set()
        for i in range(self._vertex_count):
            unique_roots.add(find(i))

        beta_0 = len(unique_roots)

        # === CALCULAR β₁ USANDO FÓRMULA DE EULER ===
        # χ = V - E = β₀ - β₁
        # β₁ = β₀ - χ = β₀ - (V - E) = β₀ - V + E
        chi = self._vertex_count - edges_processed
        beta_1 = beta_0 - chi

        # Validación: β₁ debe coincidir con ciclos detectados
        assert beta_1 == cycles_detected, (
            f"Inconsistencia: β₁={beta_1} ≠ ciclos={cycles_detected}"
        )

        # β₁ >= 0 siempre para grafos
        beta_1 = max(0, beta_1)

        # === MÉTRICAS TOPOLÓGICAS DERIVADAS ===
        is_connected = (beta_0 == 1)
        is_tree = is_connected and (beta_1 == 0)
        is_forest = (beta_1 == 0)  # Bosque: sin ciclos

        # Complejidad ciclomática de McCabe
        # M = E - V + 2P donde P = componentes conexas
        # Equivalente a: M = β₁ + P
        cyclomatic_complexity = beta_1 + beta_0

        # === FILTRAR DIAGRAMA DE PERSISTENCIA ===
        # Mantener solo características con persistencia significativa
        significant_features = [
            feat for feat in persistence_diagram
            if feat["persistence"] > 0.1
        ]

        return {
            # Números de Betti
            0: beta_0,
            1: beta_1,
            2: 0,  # Grafos son 1-dimensionales

            # Característica de Euler
            "euler_characteristic": chi,

            # Clasificación topológica
            "is_connected": is_connected,
            "is_tree": is_tree,
            "is_forest": is_forest,
            "is_cyclic": beta_1 > 0,

            # Métricas de complejidad
            "cyclomatic_complexity": cyclomatic_complexity,
            "graph_genus": beta_1,  # Para grafos planos

            # Componentes
            "connected_components": beta_0,
            "independent_cycles": beta_1,

            # Homología persistente
            "homology_dimensions": significant_features,
            "total_persistence": sum(f["persistence"] for f in persistence_diagram),

            # Estadísticas del grafo
            "vertex_count": self._vertex_count,
            "edge_count": edges_processed,
            "edge_density": (2 * edges_processed) / (self._vertex_count * (self._vertex_count - 1))
                if self._vertex_count > 1 else 0.0,
        }

    def calculate_gyroscopic_stability(self, current_I: float) -> float:
        """
        Calcula estabilidad giroscópica usando ecuaciones de Euler linealizadas.

        Modelo de trompo simétrico (Ix = Iy ≠ Iz):

        Ecuaciones de Euler para cuerpo rígido:
            Ix·dωx/dt = (Iy - Iz)·ωy·ωz + τx
            Iy·dωy/dt = (Iz - Ix)·ωz·ωx + τy
            Iz·dωz/dt = (Ix - Iy)·ωx·ωy + τz

        Para rotación estable alrededor de z con pequeñas perturbaciones:
            dωx/dt = Ω·ωy  donde Ω = (Iz - Ix)/Ix · ωz
            dωy/dt = -Ω·ωx

        Esto da oscilación armónica (precesión) con frecuencia Ω.

        Criterio de estabilidad (teorema de la raqueta de tenis):
        - Rotación alrededor del eje de momento de inercia máximo o mínimo: ESTABLE
        - Rotación alrededor del eje intermedio: INESTABLE

        La "corriente" representa velocidad angular ωz.
        La derivada dI/dt representa aceleración angular (torque).
        """
        current_time = time.time()

        # === INICIALIZACIÓN ===
        if not self._initialized:
            self._ema_current = current_I
            self._last_current = current_I
            self._last_time = current_time
            self._initialized = True

            # Estado del giroscopio
            self._gyro_state = {
                "omega_x": 0.0,  # Perturbación en x
                "omega_y": 0.0,  # Perturbación en y
                "nutation_amplitude": 0.0,
                "precession_phase": 0.0,
            }

            return 1.0  # Inicialmente estable

        dt = max(1e-6, current_time - self._last_time)

        # === MOMENTOS DE INERCIA EFECTIVOS ===
        # Modelamos el flujo de datos como un trompo alargado
        # Eje z es el eje principal de rotación (flujo de datos)
        Ix = 1.0   # Momento transversal
        Iy = 1.0   # Momento transversal (simetría axial)
        Iz = 1.5   # Momento axial (trompo alargado, Iz > Ix,Iy → estable)

        # Velocidad angular principal (proporcional a corriente)
        omega_z = abs(current_I) * 10.0  # Escalar para sensibilidad

        # === ECUACIONES DE EULER LINEALIZADAS ===
        # Para simetría axial (Ix = Iy):
        # d²ωx/dt² + Ω²·ωx = 0  (oscilador armónico)
        # donde Ω = (Iz - Ix)/Ix · ωz es la frecuencia de precesión

        if Ix > 0:
            Omega_precession = ((Iz - Ix) / Ix) * omega_z
        else:
            Omega_precession = 0.0

        # === EVOLUCIÓN DE PERTURBACIONES ===
        state = self._gyro_state
        omega_x = state["omega_x"]
        omega_y = state["omega_y"]

        # Ecuaciones acopladas (rotación en plano xy)
        # Usar Euler semi-implícito para estabilidad
        omega_x_new = omega_x * math.cos(Omega_precession * dt) + omega_y * math.sin(Omega_precession * dt)
        omega_y_new = -omega_x * math.sin(Omega_precession * dt) + omega_y * math.cos(Omega_precession * dt)

        # === EXCITACIÓN POR CAMBIO EN CORRIENTE ===
        dI_dt = (current_I - self._last_current) / dt

        # Cambios bruscos en corriente excitan nutación
        excitation_amplitude = 0.1 * abs(dI_dt)

        # Añadir excitación aleatoria en fase
        phase = state["precession_phase"] + Omega_precession * dt
        omega_x_new += excitation_amplitude * math.cos(phase)
        omega_y_new += excitation_amplitude * math.sin(phase)

        # === AMORTIGUAMIENTO VISCOSO ===
        # Las perturbaciones se amortiguan por fricción
        damping_coeff = 0.95  # Por paso de tiempo
        omega_x_new *= damping_coeff
        omega_y_new *= damping_coeff

        # === AMPLITUD DE NUTACIÓN ===
        nutation_amplitude = math.sqrt(omega_x_new**2 + omega_y_new**2)

        # Filtro EMA para suavizar
        alpha_nut = 0.1
        smoothed_nutation = (1 - alpha_nut) * state["nutation_amplitude"] + alpha_nut * nutation_amplitude

        # === CRITERIO DE ESTABILIDAD ===
        # 1. Velocidad mínima para estabilidad giroscópica
        #    ωz > ω_crítico donde ω_crítico depende de la geometría
        omega_critical = 0.5
        speed_factor = 1.0 - math.exp(-3.0 * max(0, omega_z - omega_critical))

        # 2. Nutación excesiva indica inestabilidad
        #    Si la nutación es comparable a ωz, el trompo "tambalea"
        nutation_ratio = smoothed_nutation / max(omega_z, 0.1)
        nutation_factor = 1.0 / (1.0 + 5.0 * nutation_ratio)

        # 3. Teorema de la raqueta de tenis
        #    Rotación alrededor de Iz (máximo) es estable si Iz > Ix, Iy
        #    Cuantificamos con el margen (Iz - Ix) / Ix
        inertia_margin = (Iz - Ix) / Ix
        stability_factor = math.tanh(2.0 * inertia_margin)  # 1 para margen grande

        # === ESTABILIDAD COMBINADA ===
        Sg = speed_factor * nutation_factor * stability_factor
        Sg = max(0.0, min(1.0, Sg))

        # === ACTUALIZAR ESTADO ===
        state["omega_x"] = omega_x_new
        state["omega_y"] = omega_y_new
        state["nutation_amplitude"] = smoothed_nutation
        state["precession_phase"] = phase % (2 * math.pi)

        self._last_current = current_I
        self._last_time = current_time

        # === DIAGNÓSTICO ===
        if Sg < 0.5:
            if Sg < 0.3:
                diagnosis = "NUTACIÓN CRÍTICA - Flujo inestable"
            else:
                diagnosis = "PRECESIÓN DETECTADA - Flujo oscilante"

            self.logger.debug(
                f"Estabilidad giroscópica: Sg={Sg:.3f}, "
                f"nutación={smoothed_nutation:.3f}, ωz={omega_z:.2f}. "
                f"Diagnóstico: {diagnosis}"
            )

        return Sg

    def calculate_system_entropy(
        self, total_records: int, error_count: int, processing_time: float
    ) -> Dict[str, float]:
        """
        Calcula entropía del sistema con correcciones para muestras pequeñas.

        Mejoras implementadas:

        1. **Estados puros**: Cuando error_count ∈ {0, total_records}, la entropía
           es exactamente 0 (estado determinístico), sin aplicar suavizado.

        2. **Estimador James-Stein shrinkage**: Para muestras pequeñas, contrae
           las probabilidades empíricas hacia una distribución uniforme.

           p̂_JS = λ·p_uniform + (1-λ)·p_empírico
           donde λ = α/(α + N) con α = 1 (prior Jeffrey's).

        3. **Corrección de Miller-Madow**: Ajusta sesgo de subestimación:
           H_MM = H + (m-1)/(2N·ln2)

        4. **Entropías generalizadas**: Rényi y Tsallis para diferentes
           sensibilidades a eventos raros.

        5. **Detección de muerte térmica**: Basada en teoría de grandes
           desviaciones, P(error) > ε con ε = 0.25.
        """
        if total_records <= 0:
            return self._get_zero_entropy()

        # === CASO ESPECIAL: ESTADOS PUROS ===
        # Un estado puro (sin mezcla) tiene entropía exactamente 0
        # Esto es físicamente correcto y evita artefactos del suavizado
        is_pure_state = (error_count == 0) or (error_count == total_records)

        if is_pure_state:
            # Entropía de Shannon para estado puro = 0
            # Todas las entropías generalizadas también son 0
            p_error = error_count / total_records

            return {
                "shannon_entropy": 0.0,
                "shannon_entropy_corrected": 0.0,
                "renyi_entropy_1": 0.0,
                "renyi_entropy_2": 0.0,
                "renyi_entropy_inf": 0.0,
                "tsallis_entropy": 0.0,
                "lempel_ziv_complexity": 0.0,
                "entropy_ratio": 0.0,
                "is_thermal_death": p_error > 0.5,  # 100% errores = muerte térmica
                "effective_samples": float(total_records),
                "kl_divergence": math.log2(2) if p_error in (0, 1) else 0.0,  # Máxima divergencia de uniforme
                "entropy_rate": 0.0,
                "mutual_info_temporal": 0.0,
                "max_entropy": 1.0,
                "entropy_absolute": 0.0,
                "configurational_entropy": 0.0,
            }

        # === SHRINKAGE DE JAMES-STEIN ===
        m = 2  # Número de categorías (éxito/error)
        alpha_prior = 1.0  # Prior de Jeffrey (no informativo)

        # Probabilidades empíricas (sin suavizado para el shrinkage)
        n_success = total_records - error_count
        n_error = error_count

        p_success_emp = n_success / total_records
        p_error_emp = n_error / total_records

        # Factor de shrinkage: λ = α/(α + N)
        lambda_js = alpha_prior / (alpha_prior + total_records)

        # Probabilidad uniforme (target del shrinkage)
        p_uniform = 1.0 / m

        # Probabilidades contraídas
        p_success = lambda_js * p_uniform + (1 - lambda_js) * p_success_emp
        p_error = lambda_js * p_uniform + (1 - lambda_js) * p_error_emp

        # Normalizar para garantizar suma = 1 (corrección numérica)
        p_total = p_success + p_error
        p_success /= p_total
        p_error /= p_total

        probabilities = [p_success, p_error]

        # === ENTROPÍA DE SHANNON ===
        H_shannon = 0.0
        for p in probabilities:
            if p > 1e-15:  # Evitar log(0)
                H_shannon -= p * math.log2(p)

        # === CORRECCIÓN DE MILLER-MADOW ===
        # Corrige sesgo de subestimación para muestras finitas
        # H_MM = H + (m-1) / (2*N*ln(2))
        miller_madow_correction = (m - 1) / (2 * total_records * math.log(2))
        H_mm = H_shannon + miller_madow_correction

        # === ENTROPÍA DE RÉNYI GENERALIZADA ===
        def renyi_entropy(alpha: float) -> float:
            """
            H_α = (1/(1-α)) * log₂(Σᵢ pᵢ^α)

            Límites:
            - α → 1: Shannon
            - α → 0: Hartley (log del soporte)
            - α → ∞: min-entropy (-log max(p))
            """
            if abs(alpha - 1.0) < 1e-8:
                return H_shannon

            sum_p_alpha = sum(p**alpha for p in probabilities if p > 1e-15)

            if sum_p_alpha <= 0:
                return 0.0

            return (1.0 / (1.0 - alpha)) * math.log2(sum_p_alpha)

        H_renyi_05 = renyi_entropy(0.5)   # Más sensible a eventos raros
        H_renyi_1 = H_shannon             # Shannon
        H_renyi_2 = renyi_entropy(2.0)    # Entropía de colisión

        # Min-entropía (α → ∞)
        p_max = max(probabilities)
        H_renyi_inf = -math.log2(p_max) if p_max > 0 else 0.0

        # === ENTROPÍA DE TSALLIS (q-entropía) ===
        # S_q = (1 - Σᵢ pᵢ^q) / (q - 1)
        # Es no-extensiva: S_q(A+B) = S_q(A) + S_q(B) + (1-q)*S_q(A)*S_q(B)
        q = 2.0
        sum_p_q = sum(p**q for p in probabilities if p > 1e-15)
        H_tsallis = (1.0 - sum_p_q) / (q - 1.0) if abs(q - 1.0) > 1e-8 else H_shannon

        # === DIVERGENCIA KL DESDE DISTRIBUCIÓN UNIFORME ===
        # D_KL(P||U) = Σᵢ pᵢ * log₂(pᵢ / u)
        # Mide "sorpresa" de la distribución real respecto a la uniforme
        kl_divergence = 0.0
        for p in probabilities:
            if p > 1e-15:
                kl_divergence += p * math.log2(p / p_uniform)

        # === COMPLEJIDAD DE LEMPEL-ZIV (aproximación) ===
        # Para un proceso binario, la complejidad se aproxima como
        # C ≈ H * n / log₂(n) para secuencias largas
        # Normalizamos a [0, 1] usando la relación con entropía
        if H_shannon > 0:
            lz_complexity = 1.0 - math.exp(-H_shannon)
        else:
            lz_complexity = 0.0

        # === MÉTRICAS DERIVADAS ===
        max_entropy = math.log2(m)  # 1 bit para sistema binario
        entropy_ratio = H_shannon / max_entropy if max_entropy > 0 else 0.0

        # Tasa de entropía (bits por unidad de tiempo)
        entropy_rate = H_shannon / max(processing_time, 1e-6)

        # === DETECCIÓN DE MUERTE TÉRMICA ===
        # Criterio: alta entropía + alta tasa de errores
        # Basado en principio de máxima entropía de Jaynes
        epsilon_death = 0.25
        is_thermal_death = (p_error_emp > epsilon_death) and (entropy_ratio > 0.85)

        # === INFORMACIÓN MUTUA TEMPORAL (estimación) ===
        # Aproximación basada en reducción de incertidumbre
        # I(t; t-1) ≈ H(t) - H(t|t-1)
        # Sin historial, asumimos I ≈ 0
        mutual_info_temporal = 0.0
        if len(self._entropy_history) >= 2:
            prev_entropy = self._entropy_history[-1].get("shannon_entropy", H_shannon)
            # Información ganada = reducción de entropía
            mutual_info_temporal = max(0, prev_entropy - H_shannon)

        result = {
            # Entropías fundamentales
            "shannon_entropy": H_shannon,
            "shannon_entropy_corrected": H_mm,

            # Familia de Rényi
            "renyi_entropy_05": H_renyi_05,
            "renyi_entropy_1": H_renyi_1,
            "renyi_entropy_2": H_renyi_2,
            "renyi_entropy_inf": H_renyi_inf,

            # Tsallis (no extensiva)
            "tsallis_entropy": H_tsallis,

            # Métricas de información
            "kl_divergence": kl_divergence,
            "lempel_ziv_complexity": lz_complexity,
            "mutual_info_temporal": mutual_info_temporal,

            # Métricas normalizadas
            "entropy_ratio": entropy_ratio,
            "max_entropy": max_entropy,
            "entropy_absolute": H_shannon,
            "entropy_rate": entropy_rate,

            # Diagnóstico
            "is_thermal_death": is_thermal_death,
            "effective_samples": total_records * (1 - lambda_js),

            # Alias para compatibilidad
            "configurational_entropy": H_renyi_2,
        }

        # Guardar en historial
        self._entropy_history.append({
            **result,
            "timestamp": time.time(),
            "total_records": total_records,
            "error_rate": p_error_emp,
        })

        return result

    def _get_zero_entropy(self) -> Dict[str, float]:
        """Retorna entropía cero para casos triviales."""
        return {
            "shannon_entropy": 0.0,
            "shannon_entropy_corrected": 0.0,
            "renyi_entropy_1": 0.0,
            "renyi_entropy_2": 0.0,
            "renyi_entropy_inf": 0.0,
            "tsallis_entropy": 0.0,
            "lempel_ziv_complexity": 0.0,
            "entropy_ratio": 0.0,
            "is_thermal_death": False,
            "effective_samples": 0.0,
            "kl_divergence": 0.0,
            "entropy_rate": 0.0,
            "mutual_info_temporal": 0.0,
            "max_entropy": 1.0,
            "entropy_absolute": 0.0,
            "configurational_entropy": 0.0,
        }

    def calculate_metrics(
        self,
        total_records: int,
        cache_hits: int,
        error_count: int = 0,
        processing_time: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calcula métricas físicas del sistema RLC.

        Modelo: el flujo de datos se modela como un circuito RLC donde:
        - Corriente I = eficiencia (cache_hits / total_records).
        - Carga Q = registros acumulados procesados.
        - Voltaje V = "presión" del pipeline (saturación).
        """
        if total_records <= 0:
            return self._get_zero_metrics()

        current_time = time.time()

        # Corriente normalizada (eficiencia de caché)
        current_I = cache_hits / total_records

        # Complejidad como resistencia adicional
        complexity = 1.0 - current_I

        # Resistencia dinámica
        R_dynamic = self.R * (
            1.0 + complexity * SystemConstants.COMPLEXITY_RESISTANCE_FACTOR
        )

        # Actualizar amortiguamiento dinámico
        zeta_dynamic = R_dynamic / (2.0 * math.sqrt(self.L / self.C))

        # Evolución del estado con RK4
        if self._initialized:
            dt = max(1e-6, current_time - self._last_time)
        else:
            dt = 0.01
            self._initialized = True

        Q, I = self._evolve_state_rk4(current_I, dt)

        # Constante de tiempo normalizada
        tau = self.L / R_dynamic if R_dynamic > 0 else float("inf")
        t_normalized = processing_time / tau if tau > 0 else 0.0
        t_normalized = min(t_normalized, 50.0)

        # Respuesta transitoria (voltaje en capacitor = saturación)
        if zeta_dynamic >= 1.0:
            # Sobreamortiguado o críticamente amortiguado
            saturation = 1.0 - math.exp(-t_normalized)
        else:
            # Subamortiguado - respuesta oscilatoria
            omega_d = self._omega_0 * math.sqrt(1 - zeta_dynamic**2)
            exp_term = math.exp(-zeta_dynamic * self._omega_0 * t_normalized)
            cos_term = math.cos(omega_d * t_normalized)
            sin_term = (zeta_dynamic / math.sqrt(1 - zeta_dynamic**2)) * math.sin(
                omega_d * t_normalized
            )
            saturation = 1.0 - exp_term * (cos_term + sin_term)

        saturation = max(0.0, min(1.0, saturation))

        # Energías
        E_capacitor = 0.5 * self.C * (saturation**2)  # Energía potencial
        E_inductor = 0.5 * self.L * (current_I**2)  # Energía cinética

        # Potencia disipada
        P_dissipated = (current_I**2) * R_dynamic

        # Voltaje de flyback inductivo
        di_dt = (current_I - self._last_current) / max(dt, 1e-6)
        V_flyback = min(abs(self.L * di_dt), SystemConstants.MAX_FLYBACK_VOLTAGE)

        # Entropía
        entropy_metrics = self.calculate_system_entropy(
            total_records, error_count, processing_time
        )

        # Estabilidad Giroscópica
        gyro_stability = self.calculate_gyroscopic_stability(current_I)

        # Construir grafo y calcular topología
        metrics = {
            "saturation": saturation,
            "complexity": complexity,
            "current_I": current_I,
            "potential_energy": E_capacitor,
            "kinetic_energy": E_inductor,
            "total_energy": E_capacitor + E_inductor,
            "dissipated_power": P_dissipated,
            "flyback_voltage": V_flyback,
            "dynamic_resistance": R_dynamic,
            "damping_ratio": zeta_dynamic,
            "damping_type": self._damping_type,
            "resonant_frequency_hz": self._omega_0 / (2 * math.pi),
            "quality_factor": self._Q,
            "time_constant": tau,
            # Entropía Extendida
            "entropy_shannon": entropy_metrics["shannon_entropy"],
            "entropy_shannon_corrected": entropy_metrics["shannon_entropy_corrected"],
            "tsallis_entropy": entropy_metrics["tsallis_entropy"],
            "kl_divergence": entropy_metrics["kl_divergence"],
            "entropy_rate": entropy_metrics["entropy_rate"],
            "entropy_ratio": entropy_metrics["entropy_ratio"],
            "is_thermal_death": entropy_metrics["is_thermal_death"],
            # Alias para pruebas
            "entropy_absolute": entropy_metrics["entropy_absolute"],
            # Giroscópica
            "gyroscopic_stability": gyro_stability,
        }

        # Análisis topológico
        self._build_metric_graph(metrics)
        betti = self._calculate_betti_numbers()
        metrics["betti_0"] = betti[0]
        metrics["betti_1"] = betti[1]
        metrics["graph_vertices"] = self._vertex_count
        metrics["graph_edges"] = self._edge_count

        # Actualizar estado
        self._last_current = current_I
        self._last_time = current_time

        # Guardar en historial
        self._store_metrics(metrics)

        return metrics

    def _get_zero_metrics(self) -> Dict[str, float]:
        """Métricas iniciales para casos triviales."""
        return {
            "saturation": 0.0,
            "complexity": 1.0,
            "current_I": 0.0,
            "potential_energy": 0.0,
            "kinetic_energy": 0.0,
            "total_energy": 0.0,
            "dissipated_power": 0.0,
            "flyback_voltage": 0.0,
            "dynamic_resistance": self.R,
            "damping_ratio": self._zeta,
            "damping_type": self._damping_type,
            "resonant_frequency_hz": self._omega_0 / (2 * math.pi),
            "quality_factor": self._Q,
            "time_constant": self.L / self.R if self.R > 0 else float("inf"),
            "entropy_shannon": 0.0,
            "entropy_absolute": 0.0,
            "entropy_rate": 0.0,
            "entropy_ratio": 0.0,
            "is_thermal_death": False,
            "betti_0": 0,
            "betti_1": 0,
            "graph_vertices": 0,
            "graph_edges": 0,
            "gyroscopic_stability": 1.0,
        }

    def _store_metrics(self, metrics: Dict[str, float]) -> None:
        """Almacena métricas con timestamp."""
        self._metrics_history.append({**metrics, "_timestamp": time.time()})

    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analiza tendencias en métricas históricas."""
        if len(self._metrics_history) < 2:
            return {"status": "INSUFFICIENT_DATA", "samples": len(self._metrics_history)}

        result = {"status": "OK", "samples": len(self._metrics_history)}

        # Métricas a analizar
        keys_to_analyze = ["saturation", "dissipated_power", "entropy_ratio"]

        for key in keys_to_analyze:
            values = [m.get(key, 0.0) for m in self._metrics_history if key in m]
            if len(values) >= 2:
                # Tendencia lineal simple
                first_half = sum(values[: len(values) // 2]) / (len(values) // 2)
                second_half = sum(values[len(values) // 2 :]) / (
                    len(values) - len(values) // 2
                )

                if second_half > first_half * 1.1:
                    trend = "INCREASING"
                elif second_half < first_half * 0.9:
                    trend = "DECREASING"
                else:
                    trend = "STABLE"

                result[key] = {
                    "trend": trend,
                    "current": values[-1],
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        return result

    def get_system_diagnosis(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Genera diagnóstico del estado del sistema."""
        diagnosis = {
            "state": "NORMAL",
            "damping": self._damping_type,
            "energy": "BALANCED",
            "entropy": "LOW",
        }

        # Diagnóstico de saturación
        saturation = metrics.get("saturation", 0.0)
        if saturation > 0.95:
            diagnosis["state"] = "SATURATED"
        elif saturation < 0.05:
            diagnosis["state"] = "IDLE"

        # Diagnóstico de energía
        pe = metrics.get("potential_energy", 0)
        ke = metrics.get("kinetic_energy", 0)
        total_e = pe + ke

        if total_e > 0:
            if pe / total_e > 0.9:
                diagnosis["energy"] = "POTENTIAL_DOMINATED"
            elif ke / total_e > 0.9:
                diagnosis["energy"] = "KINETIC_DOMINATED"

        # Diagnóstico de potencia
        power = metrics.get("dissipated_power", 0)
        if power > SystemConstants.OVERHEAT_POWER_THRESHOLD:
            diagnosis["state"] = "OVERHEATING"

        # Diagnóstico de entropía
        entropy_ratio = metrics.get("entropy_ratio", 0)
        if entropy_ratio > 0.8:
            diagnosis["entropy"] = "HIGH"
            if metrics.get("is_thermal_death", False):
                diagnosis["state"] = "THERMAL_DEATH"
        elif entropy_ratio > 0.5:
            diagnosis["entropy"] = "MODERATE"

        # Diagnóstico topológico
        betti_0 = metrics.get("betti_0", 1)
        betti_1 = metrics.get("betti_1", 0)

        if betti_0 > 1:
            diagnosis["topology"] = "DISCONNECTED"
        elif betti_1 > 0:
            diagnosis["topology"] = "CYCLIC"
        else:
            diagnosis["topology"] = "SIMPLE"

        # Diagnóstico Giroscópico
        gyro_stability = metrics.get("gyroscopic_stability", 1.0)
        diagnosis["rotation_stability"] = "STABLE"
        if gyro_stability < 0.6:
            diagnosis["rotation_stability"] = (
                "⚠️ PRECESIÓN DETECTADA (Inestabilidad de Flujo)"
            )
            # También escalamos el estado si es crítico
            if gyro_stability < 0.3 and diagnosis["state"] == "NORMAL":
                diagnosis["state"] = "UNSTABLE"

        return diagnosis


# ============================================================================
# DATA FLUX CONDENSER - MÉTODOS REFINADOS
# ============================================================================
class DataFluxCondenser:
    """
    Orquesta el pipeline de validación y procesamiento con control adaptativo.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        profile: Dict[str, Any],
        condenser_config: Optional[CondenserConfig] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.condenser_config = condenser_config or CondenserConfig()
        self.config = config or {}
        self.profile = profile or {}

        try:
            # 1. Validación de Estabilidad a Priori (Laplace)
            self.logger.info("🔬 Iniciando Análisis de Laplace Mejorado...")

            self.laplace_analyzer = EnhancedLaplaceAnalyzer(
                self.condenser_config.base_resistance,
                self.condenser_config.system_inductance,
                self.condenser_config.system_capacitance
            )

            validation = self.laplace_analyzer.validate_for_control_design()

            if not validation["is_suitable_for_control"]:
                issues_str = "\n".join(f"  - {i}" for i in validation["issues"])
                raise ConfigurationError(
                    f"CONFIGURACIÓN NO APTA PARA CONTROL:\n{issues_str}\n"
                    f"Resumen: {validation['summary']}"
                )

            # Loguear advertencias
            for warning in validation["warnings"]:
                self.logger.warning(f"⚠️ Advertencia de Control: {warning}")

            stability = self.laplace_analyzer.analyze_stability()
            self.logger.info(
                f"✅ Estabilidad Confirmada: "
                f"ω_n={stability['continuous']['natural_frequency_rad_s']:.2f} rad/s, "
                f"Márgenes: PM={stability['stability_margins']['phase_margin_deg']:.1f}°"
            )

            # 2. Inicialización de Componentes
            self.physics = FluxPhysicsEngine(
                self.condenser_config.system_capacitance,
                self.condenser_config.base_resistance,
                self.condenser_config.system_inductance,
            )
            self.controller = PIController(
                self.condenser_config.pid_kp,
                self.condenser_config.pid_ki,
                self.condenser_config.pid_setpoint,
                self.condenser_config.min_batch_size,
                self.condenser_config.max_batch_size,
                self.condenser_config.integral_limit_factor,
            )
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Error inicializando componentes: {e}")

        self._stats = ProcessingStats()
        self._start_time: Optional[float] = None
        self._emergency_brake_count: int = 0

    def get_physics_report(self) -> Dict[str, Any]:
        """Obtiene reporte físico completo del sistema."""
        return self.laplace_analyzer.get_comprehensive_report()

    def stabilize(
        self,
        file_path: str,
        on_progress: Optional[Callable[[ProcessingStats], None]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        telemetry: Optional[TelemetryContext] = None,
    ) -> pd.DataFrame:
        """
        Proceso principal de estabilización con control PID y telemetría.
        """
        self._start_time = time.time()
        self._stats = ProcessingStats()
        self._emergency_brake_count = 0
        self.controller.reset()

        # Validación de entrada
        if not file_path:
            raise InvalidInputError("file_path es requerido")

        path_obj = Path(file_path)
        self.logger.info(f"⚡ [STABILIZE] Iniciando: {path_obj.name}")

        # Registrar inicio en telemetría
        if telemetry:
            telemetry.record_event(
                "stabilization_start",
                {"file": path_obj.name, "config": asdict(self.condenser_config)},
            )

        try:
            validated_path = self._validate_input_file(file_path)

            # INYECCIÓN DE TELEMETRÍA AQUÍ
            parser = self._initialize_parser(validated_path, telemetry)

            raw_records, cache = self._extract_raw_data(parser)

            if not raw_records:
                self.logger.warning("No se encontraron registros para procesar")
                return pd.DataFrame()

            total_records = len(raw_records)
            self._stats.total_records = total_records

            # Verificar límite de registros
            if total_records > SystemConstants.MAX_RECORDS_LIMIT:
                raise ProcessingError(
                    f"Total de registros ({total_records}) excede límite "
                    f"({SystemConstants.MAX_RECORDS_LIMIT})"
                )

            processed_batches = self._process_batches_with_pid(
                raw_records,
                cache,
                total_records,
                on_progress,
                progress_callback,
                telemetry,
            )

            df_final = self._consolidate_results(processed_batches)
            self._stats.processing_time = time.time() - self._start_time
            self._validate_output(df_final)

            # Registrar fin en telemetría
            if telemetry:
                telemetry.record_event(
                    "stabilization_complete",
                    {
                        "records_processed": self._stats.processed_records,
                        "processing_time": self._stats.processing_time,
                        "emergency_brakes": self._emergency_brake_count,
                    },
                )

            self.logger.info(
                f"✅ [STABILIZE] Completado: {self._stats.processed_records} registros "
                f"en {self._stats.processing_time:.2f}s"
            )

            return df_final

        except DataFluxCondenserError as e:
            if telemetry:
                telemetry.record_event("stabilization_error", {"error": str(e)})
            raise
        except Exception as e:
            self.logger.exception(f"Error inesperado en estabilización: {e}")
            if telemetry:
                telemetry.record_event("stabilization_error", {"error": str(e)})
            raise ProcessingError(f"Error fatal: {e}")

    def _validate_input_file(self, file_path: str) -> Path:
        """Valida el archivo de entrada con verificaciones extendidas."""
        path = Path(file_path)

        if not path.exists():
            raise InvalidInputError(f"Archivo no existe: {file_path}")

        if not path.is_file():
            raise InvalidInputError(f"Ruta no es un archivo: {file_path}")

        if path.suffix.lower() not in SystemConstants.VALID_FILE_EXTENSIONS:
            raise InvalidInputError(
                f"Extensión no soportada: {path.suffix}. "
                f"Válidas: {SystemConstants.VALID_FILE_EXTENSIONS}"
            )

        file_size = path.stat().st_size
        if file_size < SystemConstants.MIN_FILE_SIZE_BYTES:
            raise InvalidInputError(f"Archivo muy pequeño: {file_size} bytes")

        max_size_bytes = SystemConstants.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size > max_size_bytes:
            raise InvalidInputError(
                f"Archivo excede límite: {file_size / 1024 / 1024:.1f} MB > "
                f"{SystemConstants.MAX_FILE_SIZE_MB} MB"
            )

        return path

    def _initialize_parser(self, path: Path, telemetry: Optional[TelemetryContext] = None) -> ReportParserCrudo:
        """Inicializa el parser con manejo de errores e inyección de telemetría."""
        try:
            # Pasamos telemetry al constructor del parser
            return ReportParserCrudo(str(path), self.profile, self.config, telemetry=telemetry)
        except TypeError:
            # Fallback por si ReportParserCrudo no ha sido actualizado aún en el entorno
            self.logger.warning("ReportParserCrudo no acepta telemetry, usando inicialización legacy")
            return ReportParserCrudo(str(path), self.profile, self.config)
        except Exception as e:
            raise ProcessingError(f"Error inicializando parser: {e}")

    def _extract_raw_data(self, parser) -> Tuple[List, Dict]:
        """Extrae datos crudos del parser."""
        try:
            raw_records = parser.parse_to_raw()
            cache = parser.get_parse_cache()
            return raw_records, cache
        except Exception as e:
            raise ProcessingError(f"Error extrayendo datos: {e}")

    def _process_batches_with_pid(
        self,
        raw_records: List,
        cache: Dict,
        total_records: int,
        on_progress: Optional[Callable],
        progress_callback: Optional[Callable],
        telemetry: Optional[TelemetryContext],
    ) -> List[pd.DataFrame]:
        """
        Procesamiento con control PID mejorado.
        """
        processed_batches = []
        failed_batches_count = 0
        current_index = 0
        current_batch_size = self.condenser_config.min_batch_size
        iteration = 0
        max_iterations = total_records * SystemConstants.MAX_ITERATIONS_MULTIPLIER

        saturation_history = []
        steady_state_counter = 0
        steady_state_threshold = 5
        last_complexity = 0.5

        while current_index < total_records and iteration < max_iterations:
            iteration += 1
            end_index = min(current_index + current_batch_size, total_records)
            batch = raw_records[current_index:end_index]
            batch_size = len(batch)

            if batch_size == 0:
                break

            elapsed_time = time.time() - self._start_time
            time_remaining = SystemConstants.PROCESSING_TIMEOUT - elapsed_time
            if time_remaining <= 0:
                self.logger.error("⏰ Timeout de procesamiento alcanzado")
                break

            cache_hits_est = self._estimate_cache_hits(batch, cache)

            metrics = self.physics.calculate_metrics(
                total_records=batch_size,
                cache_hits=cache_hits_est,
                error_count=failed_batches_count,
                processing_time=elapsed_time,
            )

            saturation = metrics.get("saturation", 0.5)
            complexity = metrics.get("complexity", 0.5)
            power = metrics.get("dissipated_power", 0)
            flyback = metrics.get("flyback_voltage", 0)
            gyro_stability = metrics.get("gyroscopic_stability", 1.0)

            saturation_history.append(saturation)
            if len(saturation_history) >= 3:
                predicted_sat = self._predict_next_saturation(saturation_history)
            else:
                predicted_sat = saturation

            complexity_delta = complexity - last_complexity
            feedforward_adjustment = 1.0
            if complexity_delta > 0.1:
                feedforward_adjustment = 0.85
            elif complexity_delta < -0.1:
                feedforward_adjustment = 1.1
            last_complexity = complexity

            if len(saturation_history) >= 3:
                recent_var = sum((s - saturation) ** 2 for s in saturation_history[-3:]) / 3
                if recent_var < 0.01:
                    steady_state_counter += 1
                else:
                    steady_state_counter = 0
            in_steady_state = steady_state_counter >= steady_state_threshold

            if progress_callback:
                try:
                    progress_callback({
                        **metrics,
                        "predicted_saturation": predicted_sat,
                        "in_steady_state": in_steady_state,
                        "feedforward_adjustment": feedforward_adjustment,
                    })
                except Exception as e:
                    self.logger.warning(f"Error en progress_callback: {e}")

            if gyro_stability < 0.5:
                effective_saturation = min(saturation + 0.2, 0.9)
            else:
                effective_saturation = saturation

            pid_output = self.controller.compute(effective_saturation)
            pid_output = int(pid_output * feedforward_adjustment)

            emergency_brake = False
            brake_reason = ""

            if power > SystemConstants.OVERHEAT_POWER_THRESHOLD:
                brake_factor = 0.3
                pid_output = max(SystemConstants.MIN_BATCH_SIZE_FLOOR, int(pid_output * brake_factor))
                emergency_brake = True
                brake_reason = f"OVERHEAT P={power:.1f}W"

            if flyback > SystemConstants.MAX_FLYBACK_VOLTAGE * 0.7:
                pid_output = max(SystemConstants.MIN_BATCH_SIZE_FLOOR, int(pid_output * 0.5))
                emergency_brake = True
                brake_reason = f"FLYBACK V={flyback:.2f}V"

            if predicted_sat > 0.9 and not in_steady_state:
                pid_output = max(SystemConstants.MIN_BATCH_SIZE_FLOOR, int(pid_output * 0.7))
                emergency_brake = True
                brake_reason = f"PREDICTED_SAT={predicted_sat:.2f}"

            if emergency_brake:
                self._emergency_brake_count += 1
                self._stats.emergency_brakes_triggered += 1
                self.logger.warning(f"🛑 EMERGENCY BRAKE: {brake_reason}")

            result = self._process_single_batch_with_recovery(
                batch, cache, failed_batches_count, telemetry
            )

            if result.success and result.dataframe is not None:
                if not result.dataframe.empty:
                    processed_batches.append(result.dataframe)
                self._stats.add_batch_stats(
                    batch_size=result.records_processed,
                    saturation=saturation,
                    power=power,
                    flyback=flyback,
                    kinetic=metrics.get("kinetic_energy", 0),
                    success=True,
                )
                failed_batches_count = max(0, failed_batches_count - 1)
            else:
                failed_batches_count += 1
                self._stats.add_batch_stats(
                    batch_size=batch_size,
                    saturation=saturation,
                    power=power,
                    flyback=flyback,
                    kinetic=metrics.get("kinetic_energy", 0),
                    success=False,
                )
                if failed_batches_count >= self.condenser_config.max_failed_batches:
                    if not self.condenser_config.enable_partial_recovery:
                        raise ProcessingError(f"Límite de batches fallidos: {failed_batches_count}")
                    pid_output = SystemConstants.MIN_BATCH_SIZE_FLOOR
                    self.logger.warning("Activando recuperación extrema")

            if on_progress:
                try:
                    on_progress(self._stats)
                except Exception as e:
                    self.logger.warning(f"Error en on_progress: {e}")

            if telemetry and (iteration % 10 == 0 or emergency_brake):
                telemetry.record_event(
                    "batch_iteration",
                    {
                        "iteration": iteration,
                        "progress": current_index / total_records,
                        "batch_size": batch_size,
                        "pid_output": pid_output,
                        "saturation": saturation,
                        "predicted_saturation": predicted_sat,
                        "in_steady_state": in_steady_state,
                        "emergency_brake": emergency_brake,
                    },
                )

            current_index = min(current_index + current_batch_size, total_records)

            inertia = 0.8 if in_steady_state else 0.6
            current_batch_size = int(inertia * current_batch_size + (1 - inertia) * pid_output)
            current_batch_size = max(
                SystemConstants.MIN_BATCH_SIZE_FLOOR,
                min(current_batch_size, self.condenser_config.max_batch_size),
            )

        return processed_batches

    def _estimate_cache_hits(self, batch: List, cache: Dict) -> int:
        """Estimación probabilística de cache hits."""
        if not batch: return 0
        if not cache: return max(1, len(batch) // 4)

        if not hasattr(self, "_cache_hit_history"):
            self._cache_hit_history = deque(maxlen=50)

        prior_hit_rate = sum(self._cache_hit_history) / len(self._cache_hit_history) if self._cache_hit_history else 0.5

        sample_size = min(50, len(batch))
        sample_indices = range(0, len(batch), max(1, len(batch) // sample_size))
        cache_field_set = set(cache.keys())
        sample_hits = 0

        for idx in sample_indices:
            if idx < len(batch):
                record = batch[idx]
                if isinstance(record, dict):
                    record_fields = set(record.keys())
                    overlap = len(record_fields & cache_field_set)
                    total_fields = len(record_fields | cache_field_set)
                    if total_fields > 0 and (overlap / total_fields) > 0.3:
                        sample_hits += 1

        actual_sample_size = len(list(sample_indices))
        sample_hit_rate = sample_hits / actual_sample_size if actual_sample_size > 0 else prior_hit_rate

        prior_weight = min(len(self._cache_hit_history) / 20, 0.5)
        posterior_hit_rate = prior_weight * prior_hit_rate + (1 - prior_weight) * sample_hit_rate

        self._cache_hit_history.append(sample_hit_rate)
        return max(1, int(posterior_hit_rate * len(batch)))

    def _predict_next_saturation(self, history: List[float]) -> float:
        """
        Predicción usando Filtro de Kalman Extendido (EKF) con modelo adaptativo.

        Modelo de estado de 3er orden:
            x = [saturación, velocidad, aceleración]ᵀ

        Modelo dinámico (oscilador amortiguado):
            ds/dt = v
            dv/dt = a - β·v - ω²·(s - s_eq)
            da/dt = -γ·a + ruido

        donde:
            - β: coeficiente de amortiguamiento
            - ω: frecuencia natural (determina rapidez de convergencia)
            - s_eq: saturación de equilibrio (setpoint)
            - γ: decaimiento de aceleración

        El EKF adapta estos parámetros basándose en el error de innovación.
        """
        if len(history) < 3:
            return history[-1] if history else 0.5

        # === INICIALIZACIÓN DEL EKF ===
        if not hasattr(self, '_ekf_state') or self._ekf_state is None:
            # Estimar estado inicial desde historia
            s0 = history[-1]
            v0 = (history[-1] - history[-2]) if len(history) >= 2 else 0.0
            a0 = 0.0
            if len(history) >= 3:
                v_prev = history[-2] - history[-3]
                a0 = v0 - v_prev

            self._ekf_state = {
                # Estado: [saturación, velocidad, aceleración]
                "x": [s0, v0, a0],

                # Covarianza del estado (incertidumbre)
                "P": [
                    [0.1, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.2],
                ],

                # Covarianza del proceso (ruido del modelo)
                "Q": [
                    [0.001, 0.0, 0.0],
                    [0.0, 0.01, 0.0],
                    [0.0, 0.0, 0.05],
                ],

                # Varianza de medición
                "R": 0.02,

                # Parámetros del modelo (adaptativos)
                "beta": 0.3,    # Amortiguamiento
                "omega": 0.2,   # Frecuencia natural (REDUCIDA para tracking)
                "gamma": 0.5,   # Decaimiento de aceleración
                "s_eq": 0.5,    # Equilibrio (se adaptará al setpoint)

                # Historial de innovaciones para adaptación
                "innovations": [],
            }

        ekf = self._ekf_state
        dt = 1.0  # Paso de tiempo normalizado

        # Extraer estado actual
        x = ekf["x"]
        P = ekf["P"]
        s, v, a = x[0], x[1], x[2]

        # === PREDICCIÓN (modelo no lineal) ===
        beta = ekf["beta"]
        omega = ekf["omega"]
        gamma = ekf["gamma"]
        s_eq = ekf["s_eq"]

        # Ecuaciones de estado discretizadas (Euler)
        s_pred = s + v * dt
        v_pred = v + (a - beta * v - omega**2 * (s - s_eq)) * dt
        a_pred = a * (1.0 - gamma * dt)  # Decaimiento exponencial

        x_pred = [s_pred, v_pred, a_pred]

        # Jacobiano del modelo (∂f/∂x)
        F = [
            [1.0, dt, 0.0],
            [-omega**2 * dt, 1.0 - beta * dt, dt],
            [0.0, 0.0, 1.0 - gamma * dt],
        ]

        # Propagación de covarianza: P_pred = F·P·Fᵀ + Q
        # Implementación sin numpy
        P_pred = [[0.0] * 3 for _ in range(3)]
        Q = ekf["Q"]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        P_pred[i][j] += F[i][k] * P[k][l] * F[j][l]
                P_pred[i][j] += Q[i][j]

        # === ACTUALIZACIÓN (medición: z = s_medido) ===
        z = history[-1]

        # Vector de observación: H = [1, 0, 0] (solo observamos saturación)
        H = [1.0, 0.0, 0.0]

        # Innovación (residuo)
        y = z - x_pred[0]

        # Varianza de innovación: S = H·P_pred·Hᵀ + R
        S = P_pred[0][0] + ekf["R"]

        # Ganancia de Kalman: K = P_pred·Hᵀ·S⁻¹
        K = [P_pred[i][0] / S for i in range(3)]

        # Estado actualizado: x = x_pred + K·y
        x_new = [x_pred[i] + K[i] * y for i in range(3)]

        # Covarianza actualizada: P = (I - K·H)·P_pred
        P_new = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                P_new[i][j] = P_pred[i][j] - K[i] * H[j] * P_pred[0][j]

        # === ADAPTACIÓN DE PARÁMETROS ===
        ekf["innovations"].append(y)
        if len(ekf["innovations"]) > 20:
            ekf["innovations"].pop(0)

        if len(ekf["innovations"]) >= 5:
            # Varianza de innovaciones
            innovations = ekf["innovations"]
            mean_innov = sum(innovations) / len(innovations)
            var_innov = sum((i - mean_innov)**2 for i in innovations) / len(innovations)

            # Si innovaciones son consistentemente grandes, aumentar Q
            expected_var = P_pred[0][0] + ekf["R"]
            if var_innov > 2 * expected_var:
                # Modelo subestima incertidumbre → aumentar Q
                for i in range(3):
                    ekf["Q"][i][i] *= 1.1
            elif var_innov < 0.5 * expected_var:
                # Modelo sobreestima → reducir Q
                for i in range(3):
                    ekf["Q"][i][i] *= 0.95

            # Detectar sesgo sistemático → ajustar s_eq
            if abs(mean_innov) > 0.05:
                ekf["s_eq"] += 0.1 * mean_innov
                ekf["s_eq"] = max(0.1, min(0.9, ekf["s_eq"]))

        # Guardar estado
        ekf["x"] = x_new
        ekf["P"] = P_new

        # === PREDICCIÓN A UN PASO ===
        # Usar modelo para predecir siguiente valor
        s_next = x_new[0] + x_new[1] * dt

        # Aplicar límites físicos con saturación suave (sigmoide)
        # Esto evita discontinuidades en el control
        s_next_bounded = 1.0 / (1.0 + math.exp(-10.0 * (s_next - 0.5)))

        return s_next_bounded

    def _process_single_batch_with_recovery(
        self,
        batch: List,
        cache: Dict,
        consecutive_failures: int,
        telemetry: Optional[TelemetryContext] = None,
    ) -> BatchResult:
        """
        Procesamiento de batch con estrategia de recuperación multinivel.

        Niveles de recuperación:

        1. **Intento directo**: Procesar batch completo.

        2. **División binaria**: Si falla, dividir en 2 y procesar recursivamente.
           Esto permite aislar registros problemáticos.
           Condición: batch_size > MIN_SPLIT_SIZE (evita recursión infinita)

        3. **Procesamiento unitario**: Para batches pequeños o múltiples fallos,
           procesar registro por registro, acumulando los exitosos.

        La agregación de resultados parciales se hace correctamente sumando
        records_processed de cada sub-resultado.
        """
        if not batch:
            return BatchResult(
                success=True,
                records_processed=0,
                dataframe=pd.DataFrame()
            )

        batch_size = len(batch)
        MIN_SPLIT_SIZE = 5  # Tamaño mínimo para dividir (evita recursión infinita)
        MAX_UNIT_PROCESSING_SIZE = 100  # Máximo para procesamiento unitario

        # === NIVEL 1: INTENTO DIRECTO ===
        if consecutive_failures == 0:
            try:
                parsed_data = ParsedData(batch, cache)
                df = self._rectify_signal(parsed_data, telemetry=telemetry)

                if df is not None and not df.empty:
                    return BatchResult(
                        success=True,
                        dataframe=df,
                        records_processed=len(df)
                    )
                else:
                    # Éxito pero sin datos
                    return BatchResult(
                        success=True,
                        dataframe=pd.DataFrame(),
                        records_processed=0
                    )

            except Exception as e:
                self.logger.debug(f"Intento directo falló: {e}")
                # Continuar a recuperación

        # === NIVEL 2: DIVISIÓN BINARIA ===
        # Solo si el batch es suficientemente grande y no hemos fallado mucho
        if consecutive_failures <= 2 and batch_size > MIN_SPLIT_SIZE:
            try:
                mid = batch_size // 2

                # Procesar mitades recursivamente
                left_result = self._process_single_batch_with_recovery(
                    batch[:mid], cache, consecutive_failures + 1, telemetry
                )
                right_result = self._process_single_batch_with_recovery(
                    batch[mid:], cache, consecutive_failures + 1, telemetry
                )

                # Agregar resultados
                dfs_to_concat = []
                total_records = 0

                if left_result.success and left_result.dataframe is not None:
                    if not left_result.dataframe.empty:
                        dfs_to_concat.append(left_result.dataframe)
                    total_records += left_result.records_processed

                if right_result.success and right_result.dataframe is not None:
                    if not right_result.dataframe.empty:
                        dfs_to_concat.append(right_result.dataframe)
                    total_records += right_result.records_processed

                if dfs_to_concat:
                    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
                else:
                    combined_df = pd.DataFrame()

                # Éxito parcial si recuperamos algo
                success = total_records > 0 or (left_result.success and right_result.success)

                return BatchResult(
                    success=success,
                    dataframe=combined_df,
                    records_processed=total_records,
                    error_message="" if success else "División binaria falló completamente"
                )

            except Exception as e:
                self.logger.warning(f"División binaria falló: {e}")
                # Continuar a nivel 3

        # === NIVEL 3: PROCESAMIENTO UNITARIO ===
        # Para batches pequeños o cuando la división falló
        if batch_size <= MAX_UNIT_PROCESSING_SIZE:
            successful_dfs = []
            failed_count = 0

            for idx, record in enumerate(batch):
                try:
                    parsed = ParsedData([record], cache)
                    df = self._rectify_signal(parsed, telemetry=telemetry)

                    if df is not None and not df.empty:
                        successful_dfs.append(df)

                except Exception as e:
                    failed_count += 1
                    if failed_count <= 5:  # Limitar logging
                        self.logger.debug(f"Registro {idx} falló: {e}")

            if successful_dfs:
                combined_df = pd.concat(successful_dfs, ignore_index=True)
                records_processed = len(combined_df)
            else:
                combined_df = pd.DataFrame()
                records_processed = 0

            success = records_processed > 0

            return BatchResult(
                success=success,
                dataframe=combined_df,
                records_processed=records_processed,
                error_message=f"Recuperación unitaria: {records_processed}/{batch_size} exitosos"
            )

        # === FALLO TOTAL ===
        # Batch muy grande y múltiples niveles de recuperación fallaron
        return BatchResult(
            success=False,
            dataframe=None,
            records_processed=0,
            error_message=f"Recuperación fallida para batch de {batch_size} registros"
        )

    def _rectify_signal(self, parsed_data: ParsedData, telemetry: Optional[TelemetryContext] = None) -> pd.DataFrame:
        """Convierte datos crudos a DataFrame mediante APUProcessor."""
        try:
            processor = APUProcessor(self.config, self.profile, parsed_data.parse_cache)
            processor.raw_records = parsed_data.raw_records
            return processor.process_all(telemetry=telemetry)
        except Exception as e:
            raise ProcessingError(f"Error en rectificación: {e}")

    def _consolidate_results(self, batches: List[pd.DataFrame]) -> pd.DataFrame:
        """Consolida resultados."""
        valid = [df for df in batches if df is not None and not df.empty]
        if not valid: return pd.DataFrame()
        if len(valid) > SystemConstants.MAX_BATCHES_TO_CONSOLIDATE:
            valid = valid[:SystemConstants.MAX_BATCHES_TO_CONSOLIDATE]
        try:
            return pd.concat(valid, ignore_index=True)
        except Exception as e:
            raise ProcessingError(f"Error consolidando: {e}")

    def _validate_output(self, df: pd.DataFrame) -> None:
        """Valida salida."""
        if df.empty:
            self.logger.warning("DataFrame salida vacío")
            return
        if len(df) < self.condenser_config.min_records_threshold:
            msg = f"Registros insuficientes: {len(df)}"
            if self.condenser_config.enable_strict_validation:
                raise ProcessingError(msg)
            self.logger.warning(msg)

    def _enhance_stats_with_diagnostics(self, stats: ProcessingStats, metrics: Dict) -> Dict:
        """Mejora estadísticas."""
        base = asdict(stats)
        return {
            **base,
            "efficiency": stats.processed_records / max(1, stats.total_records),
            "system_health": self.get_system_health(),
            "physics_diagnosis": self.physics.get_system_diagnosis(metrics),
            "current_metrics": metrics # Fix: expose passed metrics
        }

    def get_processing_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas."""
        current_metrics = {}
        if self.physics._metrics_history:
             current_metrics = self.physics._metrics_history[-1]

        return {
            "statistics": asdict(self._stats),
            "controller": self.controller.get_diagnostics(),
            "physics": self.physics.get_trend_analysis(),
            "emergency_brakes": self._emergency_brake_count,
            "current_metrics": current_metrics
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Retorna salud del sistema."""
        diag = self.controller.get_stability_analysis()
        health = "HEALTHY"
        issues = []
        if diag.get("stability_class") == "POTENTIALLY_UNSTABLE":
            health = "DEGRADED"
            issues.append("Inestabilidad de control")
        if self._emergency_brake_count > 5:
            health = "DEGRADED"
            issues.append("Frenos de emergencia frecuentes")

        return {
            "health": health,
            "issues": issues,
            "uptime": time.time() - self._start_time if self._start_time else 0
        }
