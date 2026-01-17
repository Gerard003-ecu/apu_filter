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
from .laplace_oracle import LaplaceOracle, ConfigurationError as OracleConfigurationError

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
        Aplica filtro de Media Móvil Exponencial con α adaptativo.

        ══════════════════════════════════════════════════════════════════
        TEORÍA DEL FILTRO EMA ADAPTATIVO
        ══════════════════════════════════════════════════════════════════

        El filtro EMA: ŷ[k] = α·y[k] + (1-α)·ŷ[k-1]

        Para procesos ARIMA(0,1,1), el α óptimo minimiza el MSE de predicción.
        Adaptamos α según:

        1. **Detección de escalón**: Cambios abruptos requieren α alto
           para seguimiento rápido.

        2. **Varianza del ruido**: Alta varianza → α bajo (más suavizado)
           Baja varianza → α alto (más reactivo)

        3. **Autocorrelación**: Correlación positiva en innovaciones indica
           que α es demasiado bajo (suavizado excesivo).

        Función de mapeo:
            α(σ²) = α_min + (α_max - α_min) / (1 + k·σ²_norm)

        donde σ²_norm es la varianza normalizada respecto al setpoint².
        ══════════════════════════════════════════════════════════════════
        """
        ALPHA_MIN = 0.05   # Máximo suavizado
        ALPHA_MAX = 0.50   # Mínimo suavizado
        SENSITIVITY = 50.0  # Sensibilidad a varianza

        if self._filtered_pv is None:
            self._filtered_pv = measurement
            if not hasattr(self, '_innovation_history'):
                self._innovation_history = deque(maxlen=10)
            return measurement

        # ══════════════════════════════════════════════════════════════════
        # DETECCIÓN DE CAMBIO ABRUPTO (STEP)
        # ══════════════════════════════════════════════════════════════════
        innovation = measurement - self._filtered_pv
        step_threshold = 0.20 * max(abs(self.setpoint), 0.05)

        if abs(innovation) > step_threshold:
            # Cambio abrupto: aplicar peso adaptativo hacia nueva medición
            # Usar función sigmoide para transición suave
            step_magnitude = abs(innovation) / step_threshold
            bypass_weight = 0.7 * (1.0 - math.exp(-step_magnitude + 1))
            bypass_weight = max(0.3, min(0.8, bypass_weight))

            self._filtered_pv = bypass_weight * measurement + (1 - bypass_weight) * self._filtered_pv

            # Limpiar historial para evitar sesgo por datos obsoletos
            if hasattr(self, '_innovation_history'):
                self._innovation_history.clear()

            return self._filtered_pv

        # ══════════════════════════════════════════════════════════════════
        # ESTIMACIÓN DE VARIANZA LOCAL Y ALPHA ADAPTATIVO
        # ══════════════════════════════════════════════════════════════════
        if not hasattr(self, '_innovation_history'):
            self._innovation_history = deque(maxlen=10)

        self._innovation_history.append(innovation)
        n_samples = len(self._innovation_history)

        if n_samples >= 3:
            innovations = list(self._innovation_history)
            mean_innov = sum(innovations) / n_samples

            # Varianza con corrección de Bessel para muestras pequeñas
            variance = sum((x - mean_innov)**2 for x in innovations) / max(1, n_samples - 1)

            # Normalizar varianza respecto a escala del setpoint
            reference_scale = max(self.setpoint**2, 0.0001)
            normalized_variance = variance / reference_scale

            # Mapeo varianza → alpha usando función racional
            adaptive_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) / (1.0 + SENSITIVITY * normalized_variance)

            # ══════════════════════════════════════════════════════════════
            # CORRECCIÓN POR AUTOCORRELACIÓN
            # ══════════════════════════════════════════════════════════════
            if n_samples >= 5 and variance > 1e-12:
                # Calcular autocorrelación lag-1 normalizada
                sum_product = sum(
                    (innovations[i] - mean_innov) * (innovations[i-1] - mean_innov)
                    for i in range(1, n_samples)
                )
                autocorr = sum_product / (variance * (n_samples - 1))
                autocorr = max(-1.0, min(1.0, autocorr))  # Clamp por estabilidad numérica

                # Autocorrelación positiva fuerte → reducir alpha (proceso más suave)
                # Autocorrelación negativa → puede indicar sobremuestreo
                if autocorr > 0.3:
                    adaptive_alpha *= (1.0 - 0.4 * autocorr)
                elif autocorr < -0.3:
                    adaptive_alpha *= (1.0 + 0.2 * abs(autocorr))

            self._ema_alpha = max(ALPHA_MIN, min(ALPHA_MAX, adaptive_alpha))
        else:
            # Muestras insuficientes: usar alpha conservador
            self._ema_alpha = 0.15

        # ══════════════════════════════════════════════════════════════════
        # APLICAR FILTRO EMA
        # ══════════════════════════════════════════════════════════════════
        self._filtered_pv = self._ema_alpha * measurement + (1.0 - self._ema_alpha) * self._filtered_pv

        return self._filtered_pv

    def _update_lyapunov_metric(self, error: float) -> None:
        """
        Actualiza estimación del exponente de Lyapunov usando regresión robusta.

        ══════════════════════════════════════════════════════════════════
        FUNDAMENTO TEÓRICO
        ══════════════════════════════════════════════════════════════════

        Para un sistema dinámico discreto, el exponente de Lyapunov λ caracteriza
        la tasa de divergencia/convergencia de trayectorias:

            |e(k)| ~ |e(0)| · exp(λ·k)

        Tomando logaritmos:
            log|e(k)| = log|e(0)| + λ·k

        Estimamos λ mediante regresión lineal de log|e| vs k en ventana deslizante.

        Interpretación:
            λ < 0  → Convergencia exponencial (estabilidad asintótica)
            λ ≈ 0  → Estabilidad marginal / ciclo límite
            λ > 0  → Divergencia (inestabilidad)

        ══════════════════════════════════════════════════════════════════
        IMPLEMENTACIÓN ROBUSTA
        ══════════════════════════════════════════════════════════════════

        - Regularización: evitar log(0) con offset ε
        - Filtrado EMA del exponente para reducir varianza
        - Umbral de alerta con histéresis para evitar falsos positivos
        ══════════════════════════════════════════════════════════════════
        """
        WINDOW_SIZE = 20
        MIN_SAMPLES = 5
        EMA_FACTOR = 0.15
        EPSILON_LOG = 1e-12  # Para evitar log(0)
        ALERT_THRESHOLD = 0.10  # Umbral de divergencia

        # Inicializar buffer si no existe
        if not hasattr(self, "_lyapunov_log_errors"):
            self._lyapunov_log_errors = deque(maxlen=WINDOW_SIZE)

        # Almacenar log|e| (con protección)
        abs_error = abs(error) + EPSILON_LOG
        self._lyapunov_log_errors.append(math.log(abs_error))

        n = len(self._lyapunov_log_errors)
        if n < MIN_SAMPLES:
            return

        # ══════════════════════════════════════════════════════════════════
        # REGRESIÓN LINEAL: log|e(k)| = λ·k + c
        # ══════════════════════════════════════════════════════════════════
        log_errors = list(self._lyapunov_log_errors)
        k_vals = list(range(n))

        # Sumas para mínimos cuadrados
        sum_k = n * (n - 1) / 2  # Σk = 0 + 1 + ... + (n-1)
        sum_k2 = n * (n - 1) * (2*n - 1) / 6  # Σk²
        sum_log_e = sum(log_errors)
        sum_k_log_e = sum(k * le for k, le in zip(k_vals, log_errors))

        denominator = n * sum_k2 - sum_k * sum_k

        if abs(denominator) < 1e-10:
            return  # Degenerado, no actualizar

        # Pendiente = exponente de Lyapunov instantáneo
        lambda_instant = (n * sum_k_log_e - sum_k * sum_log_e) / denominator

        # ══════════════════════════════════════════════════════════════════
        # FILTRADO EMA DEL EXPONENTE
        # ══════════════════════════════════════════════════════════════════
        # _lyapunov_sum almacena directamente el promedio EMA (no suma acumulada)
        self._lyapunov_sum = (1.0 - EMA_FACTOR) * self._lyapunov_sum + EMA_FACTOR * lambda_instant
        # Nota: _lyapunov_count ya no se usa para división

        # ══════════════════════════════════════════════════════════════════
        # SISTEMA DE ALERTA CON HISTÉRESIS
        # ══════════════════════════════════════════════════════════════════
        if not hasattr(self, '_lyapunov_alert_state'):
            self._lyapunov_alert_state = False

        # Histéresis: umbral de activación vs desactivación
        activate_threshold = ALERT_THRESHOLD
        deactivate_threshold = ALERT_THRESHOLD * 0.5

        if self._lyapunov_sum > activate_threshold and n >= 10:
            if not self._lyapunov_alert_state:
                self._lyapunov_alert_state = True
                logger.warning(
                    f"⚠️ Divergencia detectada: λ ≈ {self._lyapunov_sum:.4f} > {activate_threshold} "
                    f"(basado en {n} muestras). Sistema potencialmente inestable."
                )
        elif self._lyapunov_sum < deactivate_threshold:
            if self._lyapunov_alert_state:
                self._lyapunov_alert_state = False
                logger.info(f"✓ Sistema estabilizado: λ ≈ {self._lyapunov_sum:.4f}")

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
        Calcula la salida del controlador PI con mejoras robustas.

        ══════════════════════════════════════════════════════════════════
        ARQUITECTURA DEL CONTROLADOR
        ══════════════════════════════════════════════════════════════════

                     ┌───────-──┐
        r(t) ──(+)──>│   PI     │──> u(t) ──> [Saturación] ──> y
               │     │Controller│                             │
               │     └─────────-┘                             │
               │          ↑                                   │
               │     [Anti-windup]                            │
               │          │                                   │
               └──────────────────────────────────────────────┘
                         [Filtro EMA]

        Componentes:
        1. Filtro EMA adaptativo (pre-procesamiento)
        2. Zona muerta con transición suave
        3. Controlador PI: u = Kp·e + Ki·∫e·dt
        4. Anti-windup con back-calculation
        5. Slew rate limiting
        6. Saturación de salida

        ══════════════════════════════════════════════════════════════════
        ANTI-WINDUP: CLAMPING CONDICIONAL + BACK-CALCULATION
        ══════════════════════════════════════════════════════════════════

        Estrategia híbrida:
        1. Clamping condicional: No integrar si saturado Y error
           tiene mismo signo que saturación (empuja más hacia límite)

        2. Back-calculation: Después de saturar, ajustar integral
           para tracking: ∫e_new = ∫e - Kb·(u_raw - u_sat)·dt
           donde Kb = 1/Kp (regla de Åström)

        ══════════════════════════════════════════════════════════════════
        """
        self._iteration_count += 1
        current_time = time.time()

        # ══════════════════════════════════════════════════════════════════
        # PRE-PROCESAMIENTO
        # ══════════════════════════════════════════════════════════════════
        pv_clamped = max(0.0, min(1.0, process_variable))
        filtered_pv = self._apply_ema_filter(pv_clamped)

        # ══════════════════════════════════════════════════════════════════
        # CÁLCULO DE ERROR CON ZONA MUERTA SUAVE
        # ══════════════════════════════════════════════════════════════════
        raw_error = self.setpoint - filtered_pv
        deadband = 0.015 * max(abs(self.setpoint), 0.1)

        if abs(raw_error) < deadband:
            # Dentro de zona muerta: error efectivo = 0
            # PERO mantener integración pequeña para evitar offset
            error = 0.0
            # Aplicar decay suave al integral para prevenir acumulación
            self._integral_error *= 0.995
        elif abs(raw_error) < 2.0 * deadband:
            # Zona de transición: interpolación cuadrática suave
            # Evita discontinuidad en la derivada del error
            t = (abs(raw_error) - deadband) / deadband  # t ∈ [0, 1]
            smooth_factor = t * t * (3.0 - 2.0 * t)  # Hermite smoothstep
            error = math.copysign(smooth_factor * (abs(raw_error) - deadband), raw_error)
        else:
            # Fuera de zona muerta
            error = raw_error - math.copysign(deadband, raw_error)

        self._error_history.append(error)

        # ══════════════════════════════════════════════════════════════════
        # DELTA TIEMPO
        # ══════════════════════════════════════════════════════════════════
        if self._last_time is None:
            dt = SystemConstants.MIN_DELTA_TIME
        else:
            dt = max(
                SystemConstants.MIN_DELTA_TIME,
                min(current_time - self._last_time, SystemConstants.MAX_DELTA_TIME)
            )

        # ══════════════════════════════════════════════════════════════════
        # TÉRMINO PROPORCIONAL
        # ══════════════════════════════════════════════════════════════════
        P = self.Kp * error

        # ══════════════════════════════════════════════════════════════════
        # ANTI-WINDUP: CLAMPING CONDICIONAL
        # ══════════════════════════════════════════════════════════════════
        # Calcular salida tentativa para decidir si integrar
        tentative_integral = self._integral_error + error * dt
        tentative_I = self._ki_adaptive * tentative_integral
        tentative_output = self._output_center + P + tentative_I

        will_saturate_high = tentative_output > self.max_output
        will_saturate_low = tentative_output < self.min_output

        # Determinar si debemos integrar
        # Regla: integrar solo si:
        # 1. No hay saturación, O
        # 2. Hay saturación PERO el error tiene signo opuesto (nos saca de saturación)
        should_integrate = True

        if will_saturate_high and error > 0:
            # Saturación alta Y error positivo → empuja más alto → NO integrar
            should_integrate = False
        elif will_saturate_low and error < 0:
            # Saturación baja Y error negativo → empuja más bajo → NO integrar
            should_integrate = False

        if should_integrate:
            self._integral_error += error * dt

        # Límite integral con clamp suave (tanh)
        if abs(self._integral_error) > self._integral_limit:
            # Soft clamp: evita discontinuidad en la acción integral
            normalized = self._integral_error / self._integral_limit
            self._integral_error = self._integral_limit * math.tanh(normalized)

        # Adaptación de Ki para anti-windup adicional
        self._adapt_integral_gain(error, will_saturate_high or will_saturate_low)

        # ══════════════════════════════════════════════════════════════════
        # TÉRMINO INTEGRAL
        # ══════════════════════════════════════════════════════════════════
        I = self._ki_adaptive * self._integral_error

        # ══════════════════════════════════════════════════════════════════
        # SALIDA COMBINADA
        # ══════════════════════════════════════════════════════════════════
        output_raw = self._output_center + P + I

        # ══════════════════════════════════════════════════════════════════
        # SLEW RATE LIMITING (Anti-jerk)
        # ══════════════════════════════════════════════════════════════════
        if self._last_output is not None:
            max_slew = self._output_range * 0.12  # 12% máximo por paso
            delta = output_raw - self._last_output

            if abs(delta) > max_slew:
                output_raw = self._last_output + math.copysign(max_slew, delta)

        # ══════════════════════════════════════════════════════════════════
        # SATURACIÓN Y CUANTIZACIÓN
        # ══════════════════════════════════════════════════════════════════
        output_saturated = max(self.min_output, min(self.max_output, output_raw))
        output = int(round(output_saturated))

        # ══════════════════════════════════════════════════════════════════
        # BACK-CALCULATION POST-SATURACIÓN
        # ══════════════════════════════════════════════════════════════════
        saturation_error = output_raw - output_saturated

        if abs(saturation_error) > 0.5:  # Umbral para evitar ruido numérico
            # Kb = 1/Kp (regla de Åström para sistemas de primer orden)
            tracking_gain = 1.0 / max(self.Kp, 0.01)
            # Ajustar integral para reducir discontinuidad
            self._integral_error -= tracking_gain * saturation_error * dt

        # ══════════════════════════════════════════════════════════════════
        # ACTUALIZACIÓN DE MÉTRICAS Y ESTADO
        # ══════════════════════════════════════════════════════════════════
        self._update_lyapunov_metric(error)

        self._last_time = current_time
        self._last_error = error
        self._last_output = output
        self._output_history.append(output)

        return output

    def get_lyapunov_exponent(self) -> float:
        """
        Retorna estimación del exponente de Lyapunov.

        El valor retornado es el promedio EMA actualizado en cada iteración.
        No requiere división adicional ya que _lyapunov_sum almacena
        directamente el exponente filtrado.

        Returns:
            float: Exponente de Lyapunov estimado.
                   < 0: convergencia (estable)
                   ≈ 0: marginal
                   > 0: divergencia (inestable)
        """
        return self._lyapunov_sum

    def get_stability_analysis(self) -> Dict[str, Any]:
        """
        Análisis de estabilidad basado en historial de errores.

        Métricas calculadas:
        1. Exponente de Lyapunov: tasa de convergencia/divergencia
        2. Varianza del error: magnitud de oscilaciones
        3. Tendencia: comparación de varianza reciente vs histórica
        4. Utilización integral: proximidad a saturación anti-windup
        """
        if len(self._error_history) < 5:
            return {
                "status": "INSUFFICIENT_DATA",
                "samples": len(self._error_history),
                "message": f"Se requieren al menos 5 muestras, actual: {len(self._error_history)}"
            }

        errors = list(self._error_history)
        n = len(errors)
        lyapunov = self.get_lyapunov_exponent()

        # ══════════════════════════════════════════════════════════════════
        # ESTADÍSTICAS GLOBALES
        # ══════════════════════════════════════════════════════════════════
        mean_error = sum(errors) / n
        error_variance = sum((e - mean_error)**2 for e in errors) / n
        error_std = math.sqrt(error_variance)

        # Error absoluto medio
        mae = sum(abs(e) for e in errors) / n

        # ══════════════════════════════════════════════════════════════════
        # ESTADÍSTICAS RECIENTES (últimas 10 muestras)
        # ══════════════════════════════════════════════════════════════════
        n_recent = min(10, n)
        recent_errors = errors[-n_recent:]
        mean_recent = sum(recent_errors) / n_recent
        recent_variance = sum((e - mean_recent)**2 for e in recent_errors) / n_recent

        # ══════════════════════════════════════════════════════════════════
        # CLASIFICACIÓN DE ESTABILIDAD
        # ══════════════════════════════════════════════════════════════════
        if lyapunov < -0.15:
            stability = "ASYMPTOTICALLY_STABLE"
            stability_description = "Convergencia exponencial rápida"
        elif lyapunov < -0.05:
            stability = "STABLE"
            stability_description = "Convergencia moderada"
        elif lyapunov < 0.05:
            stability = "MARGINALLY_STABLE"
            stability_description = "Estabilidad marginal - posible ciclo límite"
        elif lyapunov < 0.15:
            stability = "WEAKLY_UNSTABLE"
            stability_description = "Divergencia lenta - ajustar ganancias"
        else:
            stability = "UNSTABLE"
            stability_description = "Divergencia exponencial - sistema inestable"

        # ══════════════════════════════════════════════════════════════════
        # ANÁLISIS DE TENDENCIA
        # ══════════════════════════════════════════════════════════════════
        variance_ratio = recent_variance / max(error_variance, 1e-12)

        if variance_ratio < 0.5:
            convergence = "CONVERGING_FAST"
        elif variance_ratio < 0.9:
            convergence = "CONVERGING"
        elif variance_ratio < 1.1:
            convergence = "STEADY"
        elif variance_ratio < 2.0:
            convergence = "DIVERGING"
        else:
            convergence = "DIVERGING_FAST"

        # ══════════════════════════════════════════════════════════════════
        # MÉTRICAS DE CONTROL
        # ══════════════════════════════════════════════════════════════════
        integral_utilization = abs(self._integral_error) / max(self._integral_limit, 1e-6)

        return {
            "status": "OPERATIONAL",
            "stability_class": stability,
            "stability_description": stability_description,
            "convergence": convergence,
            "lyapunov_exponent": lyapunov,
            "error_statistics": {
                "mean": mean_error,
                "variance": error_variance,
                "std_dev": error_std,
                "mae": mae,
            },
            "recent_statistics": {
                "mean": mean_recent,
                "variance": recent_variance,
                "samples": n_recent,
            },
            "trend_analysis": {
                "variance_ratio": variance_ratio,
                "interpretation": convergence,
            },
            "control_state": {
                "integral_utilization": integral_utilization,
                "integral_saturation_warning": integral_utilization > 0.8,
                "adaptive_ki_ratio": self._ki_adaptive / max(self.Ki, 1e-6),
            },
            "iterations": self._iteration_count,
            "total_samples": n,
        }


    # ============================================================================

    # ============================================================================

    # ============================================================================

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
        """
        Inicializa el orquestador con validación de estabilidad a priori.

        Secuencia de inicialización:
        1. Configuración de logging y parámetros base
        2. Análisis de Laplace para validación de estabilidad
        3. Inicialización de componentes (física, controlador)
        4. Setup de estructuras de estado

        Raises:
            ConfigurationError: Si la configuración no es apta para control
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuración con defaults seguros
        self.condenser_config = condenser_config or CondenserConfig()
        self.config = config if config is not None else {}
        self.profile = profile if profile is not None else {}

        # Estado de inicialización para diagnóstico
        self._initialization_status = {
            "laplace_validated": False,
            "physics_initialized": False,
            "controller_initialized": False,
            "timestamp": time.time(),
        }

        try:
            # ══════════════════════════════════════════════════════════════
            # FASE 1: ANÁLISIS DE ESTABILIDAD (Laplace)
            # ══════════════════════════════════════════════════════════════
            self.logger.info("🔬 Iniciando Análisis de Laplace Mejorado...")

            try:
                self.laplace_analyzer = LaplaceOracle(
                    R=self.condenser_config.base_resistance,
                    L=self.condenser_config.system_inductance,
                    C=self.condenser_config.system_capacitance,
                    sample_rate=getattr(self.condenser_config, 'sample_rate', 1000.0)
                )
            except OracleConfigurationError as e:
                raise ConfigurationError(str(e))

            validation = self.laplace_analyzer.validate_for_control_design()

            if not validation["is_suitable_for_control"]:
                issues_str = "\n".join(f"  • {issue}" for issue in validation["issues"])
                raise ConfigurationError(
                    f"CONFIGURACIÓN NO APTA PARA CONTROL:\n{issues_str}\n"
                    f"Resumen: {validation['summary']}\n"
                    f"Recomendaciones:\n" +
                    "\n".join(f"  → {r}" for r in validation.get("recommendations", []))
                )

            self._initialization_status["laplace_validated"] = True

            # Loguear advertencias con contexto
            for warning in validation["warnings"]:
                self.logger.warning(f"⚠️ Advertencia de Control: {warning}")

            stability = self.laplace_analyzer.analyze_stability()
            self.logger.info(
                f"✅ Estabilidad Confirmada: "
                f"ωₙ={stability['continuous']['natural_frequency_rad_s']:.2f} rad/s, "
                f"ζ={stability['continuous']['damping_ratio']:.3f}, "
                f"PM={stability['stability_margins']['phase_margin_deg']:.1f}°"
            )

            # Almacenar métricas de estabilidad para referencia
            self._stability_baseline = {
                "omega_n": stability['continuous']['natural_frequency_rad_s'],
                "zeta": stability['continuous']['damping_ratio'],
                "phase_margin": stability['stability_margins']['phase_margin_deg'],
                "damping_class": stability['continuous']['damping_class'],
            }

            # ══════════════════════════════════════════════════════════════
            # FASE 2: INICIALIZACIÓN DE COMPONENTES
            # ══════════════════════════════════════════════════════════════
            self.physics = FluxPhysicsEngine(
                self.condenser_config.system_capacitance,
                self.condenser_config.base_resistance,
                self.condenser_config.system_inductance,
            )
            self._initialization_status["physics_initialized"] = True

            self.controller = PIController(
                kp=self.condenser_config.pid_kp,
                ki=self.condenser_config.pid_ki,
                setpoint=self.condenser_config.pid_setpoint,
                min_output=self.condenser_config.min_batch_size,
                max_output=self.condenser_config.max_batch_size,
                integral_limit_factor=self.condenser_config.integral_limit_factor,
            )
            self._initialization_status["controller_initialized"] = True

        except ConfigurationError:
            raise
        except Exception as e:
            self.logger.exception(f"Error fatal en inicialización: {e}")
            raise ConfigurationError(
                f"Error inicializando componentes: {e}\n"
                f"Estado: {self._initialization_status}"
            )

        # ══════════════════════════════════════════════════════════════
        # FASE 3: ESTRUCTURAS DE ESTADO
        # ══════════════════════════════════════════════════════════════
        self._stats = ProcessingStats()
        self._start_time: Optional[float] = None
        self._emergency_brake_count: int = 0

        # Cache para predicciones (EKF)
        self._ekf_state: Optional[Dict[str, Any]] = None

        # Historial de métricas para análisis de tendencias
        self._metrics_history: deque = deque(maxlen=100)

        self.logger.info(
            f"✅ DataFluxCondenser inicializado: "
            f"batch_range=[{self.condenser_config.min_batch_size}, "
            f"{self.condenser_config.max_batch_size}]"
        )

    def get_physics_report(self) -> Dict[str, Any]:
        """
        Obtiene reporte físico completo del sistema.

        Incluye análisis de Laplace, respuesta en frecuencia,
        y validación para diseño de control.
        """
        try:
            report = self.laplace_analyzer.get_comprehensive_report()

            # Enriquecer con estado actual
            report["runtime_state"] = {
                "emergency_brakes": self._emergency_brake_count,
                "processed_records": self._stats.processed_records,
                "uptime_s": time.time() - self._start_time if self._start_time else 0,
            }

            return report

        except Exception as e:
            self.logger.error(f"Error generando reporte físico: {e}")
            return {
                "error": str(e),
                "system_parameters": {
                    "R": self.condenser_config.base_resistance,
                    "L": self.condenser_config.system_inductance,
                    "C": self.condenser_config.system_capacitance,
                }
            }

    def stabilize(
        self,
        file_path: str,
        on_progress: Optional[Callable[[ProcessingStats], None]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        telemetry: Optional[TelemetryContext] = None,
    ) -> pd.DataFrame:
        """
        Proceso principal de estabilización con control PID y telemetría.

        Pipeline de procesamiento:
        1. Validación de entrada
        2. Parsing de datos crudos
        3. Procesamiento por batches con control adaptativo
        4. Consolidación y validación de salida

        Args:
            file_path: Ruta al archivo de entrada
            on_progress: Callback para estadísticas de progreso
            progress_callback: Callback para métricas detalladas
            telemetry: Contexto de telemetría opcional

        Returns:
            DataFrame consolidado con datos procesados

        Raises:
            InvalidInputError: Si el archivo no es válido
            ProcessingError: Si ocurre error durante procesamiento
        """
        # Inicializar estado de sesión
        self._start_time = time.time()
        self._stats = ProcessingStats()
        self._emergency_brake_count = 0
        self._ekf_state = None  # Reset EKF para nueva sesión
        self.controller.reset()

        # Validación de entrada
        if not file_path:
            raise InvalidInputError("file_path es requerido y no puede estar vacío")

        path_obj = Path(file_path)
        self.logger.info(f"⚡ [STABILIZE] Iniciando: {path_obj.name}")

        # Contexto de telemetría con fallback
        telemetry_active = telemetry is not None

        if telemetry_active:
            telemetry.record_event(
                "stabilization_start",
                {
                    "file": path_obj.name,
                    "file_size_bytes": path_obj.stat().st_size if path_obj.exists() else 0,
                    "config": asdict(self.condenser_config),
                    "stability_baseline": self._stability_baseline,
                },
            )

        try:
            # ══════════════════════════════════════════════════════════════
            # FASE 1: VALIDACIÓN Y PARSING
            # ══════════════════════════════════════════════════════════════
            validated_path = self._validate_input_file(file_path)
            parser = self._initialize_parser(validated_path, telemetry)
            raw_records, cache = self._extract_raw_data(parser)

            if not raw_records:
                self.logger.warning("No se encontraron registros para procesar")
                if telemetry_active:
                    telemetry.record_event("stabilization_empty", {"reason": "no_records"})
                return pd.DataFrame()

            total_records = len(raw_records)
            self._stats.total_records = total_records

            # Verificar límites
            if total_records > SystemConstants.MAX_RECORDS_LIMIT:
                raise ProcessingError(
                    f"Total de registros ({total_records:,}) excede límite "
                    f"({SystemConstants.MAX_RECORDS_LIMIT:,}). "
                    f"Considere dividir el archivo."
                )

            self.logger.info(f"📊 Registros a procesar: {total_records:,}")

            # ══════════════════════════════════════════════════════════════
            # FASE 2: PROCESAMIENTO POR BATCHES
            # ══════════════════════════════════════════════════════════════
            processed_batches = self._process_batches_with_pid(
                raw_records=raw_records,
                cache=cache,
                total_records=total_records,
                on_progress=on_progress,
                progress_callback=progress_callback,
                telemetry=telemetry,
            )

            # ══════════════════════════════════════════════════════════════
            # FASE 3: CONSOLIDACIÓN Y VALIDACIÓN
            # ══════════════════════════════════════════════════════════════
            df_final = self._consolidate_results(processed_batches)
            self._stats.processing_time = time.time() - self._start_time

            self._validate_output(df_final)

            # Registrar éxito
            if telemetry_active:
                telemetry.record_event(
                    "stabilization_complete",
                    {
                        "records_input": total_records,
                        "records_output": len(df_final),
                        "records_processed": self._stats.processed_records,
                        "processing_time_s": self._stats.processing_time,
                        "throughput_records_per_s": (
                            self._stats.processed_records / max(0.001, self._stats.processing_time)
                        ),
                        "emergency_brakes": self._emergency_brake_count,
                        "batches_processed": len(processed_batches),
                        "efficiency": self._stats.processed_records / max(1, total_records),
                    },
                )

            self.logger.info(
                f"✅ [STABILIZE] Completado: {self._stats.processed_records:,} registros "
                f"en {self._stats.processing_time:.2f}s "
                f"({self._stats.processed_records / max(0.001, self._stats.processing_time):.0f} rec/s)"
            )

            return df_final

        except DataFluxCondenserError as e:
            if telemetry_active:
                telemetry.record_event(
                    "stabilization_error",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "progress": self._stats.processed_records / max(1, self._stats.total_records),
                    }
                )
            raise

        except Exception as e:
            self.logger.exception(f"Error inesperado en estabilización: {e}")
            if telemetry_active:
                telemetry.record_event(
                    "stabilization_fatal_error",
                    {"error_type": type(e).__name__, "error_message": str(e)}
                )
            raise ProcessingError(f"Error fatal en estabilización: {e}")

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
        Procesamiento con control PID mejorado y feedforward adaptativo.

        ══════════════════════════════════════════════════════════════════
        ARQUITECTURA DE CONTROL
        ══════════════════════════════════════════════════════════════════

                        ┌─────────────┐
        setpoint ──(+)──│     PI      │──┬──> batch_size
                   │    │ Controller  │  │
                   │    └─────────────┘  │
                   │           ↑         │
                   │    [Anti-windup]    │
                   │           │         │
                   │    ┌──────┴──────┐  │
                   │    │ Feedforward │<─┘
                   │    │ (Complexity)│
                   │    └─────────────┘
                   │           ↑
                   └───────────┤
                               │
        ┌──────────────────────┴──────────────────────┐
        │              PLANTA (Sistema)               │
        │  ┌─────────┐    ┌─────────┐    ┌─────────┐  │
        │  │ Physics │───>│  Batch  │───>│Saturation│ │
        │  │ Engine  │    │ Process │    │ Metrics │  │
        │  └─────────┘    └─────────┘    └─────────┘  │
        └─────────────────────────────────────────────┘

        Características:
        1. Control PI con anti-windup (del controlador)
        2. Feedforward basado en gradiente de complejidad
        3. Predicción de saturación con EKF
        4. Detección de estado estacionario con test estadístico
        5. Emergency brake multinivel

        ══════════════════════════════════════════════════════════════════
        """
        processed_batches: List[pd.DataFrame] = []
        failed_batches_count: int = 0
        current_index: int = 0
        current_batch_size: int = self.condenser_config.min_batch_size
        iteration: int = 0
        max_iterations: int = total_records * SystemConstants.MAX_ITERATIONS_MULTIPLIER

        # Estado para control avanzado
        saturation_history: deque = deque(maxlen=20)
        complexity_history: deque = deque(maxlen=10)
        steady_state_counter: int = 0
        STEADY_STATE_THRESHOLD: int = 7  # Iteraciones consecutivas

        # Estado para feedforward
        last_complexity: float = 0.5
        feedforward_integrator: float = 0.0
        FEEDFORWARD_GAIN: float = 0.15
        FEEDFORWARD_DECAY: float = 0.9

        while current_index < total_records and iteration < max_iterations:
            iteration += 1

            # ══════════════════════════════════════════════════════════════
            # EXTRACCIÓN DE BATCH
            # ══════════════════════════════════════════════════════════════
            end_index = min(current_index + current_batch_size, total_records)
            batch = raw_records[current_index:end_index]
            batch_size = len(batch)

            if batch_size == 0:
                break

            # Verificar timeout
            elapsed_time = time.time() - self._start_time
            time_remaining = SystemConstants.PROCESSING_TIMEOUT - elapsed_time

            if time_remaining <= 0:
                self.logger.error(
                    f"⏰ Timeout de procesamiento alcanzado ({SystemConstants.PROCESSING_TIMEOUT}s). "
                    f"Progreso: {current_index}/{total_records} ({100*current_index/total_records:.1f}%)"
                )
                break

            # Timeout warning anticipado
            if time_remaining < 60 and iteration % 10 == 0:
                self.logger.warning(
                    f"⏳ Tiempo restante bajo: {time_remaining:.0f}s. "
                    f"Considere reducir batch size."
                )

            # ══════════════════════════════════════════════════════════════
            # CÁLCULO DE MÉTRICAS FÍSICAS
            # ══════════════════════════════════════════════════════════════
            cache_hits_est = self._estimate_cache_hits(batch, cache)

            metrics = self.physics.calculate_metrics(
                total_records=batch_size,
                cache_hits=cache_hits_est,
                error_count=failed_batches_count,
                processing_time=elapsed_time,
            )

            saturation = metrics.get("saturation", 0.5)
            complexity = metrics.get("complexity", 0.5)
            power = metrics.get("dissipated_power", 0.0)
            flyback = metrics.get("flyback_voltage", 0.0)
            gyro_stability = metrics.get("gyroscopic_stability", 1.0)

            # Almacenar para historial
            saturation_history.append(saturation)
            complexity_history.append(complexity)
            self._metrics_history.append(metrics)

            # ══════════════════════════════════════════════════════════════
            # PREDICCIÓN DE SATURACIÓN (EKF)
            # ══════════════════════════════════════════════════════════════
            if len(saturation_history) >= 3:
                predicted_sat = self._predict_next_saturation(list(saturation_history))
            else:
                predicted_sat = saturation

            # ══════════════════════════════════════════════════════════════
            # FEEDFORWARD ADAPTATIVO
            # ══════════════════════════════════════════════════════════════
            # Modelo: feedforward compensa cambios en complejidad antes de que
            # afecten la saturación (control anticipativo)

            complexity_delta = complexity - last_complexity
            complexity_acceleration = 0.0

            if len(complexity_history) >= 3:
                # Segunda derivada de complejidad
                c = list(complexity_history)
                complexity_acceleration = c[-1] - 2*c[-2] + c[-3]

            # Integrador con decay para suavidad
            feedforward_integrator = (
                FEEDFORWARD_DECAY * feedforward_integrator +
                FEEDFORWARD_GAIN * (complexity_delta + 0.5 * complexity_acceleration)
            )

            # Limitar feedforward para evitar inestabilidad
            feedforward_integrator = max(-0.3, min(0.3, feedforward_integrator))

            # Factor de ajuste (1.0 = sin cambio)
            if complexity_delta > 0.05:
                # Complejidad aumentando → reducir batch
                feedforward_factor = 1.0 - abs(feedforward_integrator)
            elif complexity_delta < -0.05:
                # Complejidad disminuyendo → aumentar batch
                feedforward_factor = 1.0 + abs(feedforward_integrator)
            else:
                # Estable → relajar feedforward gradualmente
                feedforward_factor = 1.0 + 0.3 * feedforward_integrator

            feedforward_factor = max(0.7, min(1.3, feedforward_factor))
            last_complexity = complexity

            # ══════════════════════════════════════════════════════════════
            # DETECCIÓN DE ESTADO ESTACIONARIO
            # ══════════════════════════════════════════════════════════════
            # Usamos test de varianza con umbral adaptativo

            in_steady_state = False

            if len(saturation_history) >= 5:
                recent_sats = list(saturation_history)[-5:]
                mean_sat = sum(recent_sats) / len(recent_sats)
                variance = sum((s - mean_sat)**2 for s in recent_sats) / len(recent_sats)

                # Umbral adaptativo basado en el setpoint
                variance_threshold = 0.005 * (1.0 + abs(mean_sat - self.condenser_config.pid_setpoint))

                if variance < variance_threshold:
                    steady_state_counter += 1
                else:
                    # Reset parcial para histéresis
                    steady_state_counter = max(0, steady_state_counter - 2)

                in_steady_state = steady_state_counter >= STEADY_STATE_THRESHOLD

            # ══════════════════════════════════════════════════════════════
            # CALLBACK DE PROGRESO
            # ══════════════════════════════════════════════════════════════
            if progress_callback:
                try:
                    progress_callback({
                        **metrics,
                        "iteration": iteration,
                        "progress": current_index / total_records,
                        "predicted_saturation": predicted_sat,
                        "in_steady_state": in_steady_state,
                        "feedforward_factor": feedforward_factor,
                        "batch_size": batch_size,
                        "time_remaining_s": time_remaining,
                    })
                except Exception as e:
                    self.logger.debug(f"Error en progress_callback: {e}")

            # ══════════════════════════════════════════════════════════════
            # AJUSTE DE SATURACIÓN EFECTIVA
            # ══════════════════════════════════════════════════════════════
            # Compensar por inestabilidad giroscópica

            if gyro_stability < 0.5:
                # Baja estabilidad giroscópica → aumentar saturación percibida
                # para que el controlador reduzca batch size
                stability_penalty = 0.3 * (1.0 - gyro_stability / 0.5)
                effective_saturation = min(saturation + stability_penalty, 0.95)
            else:
                effective_saturation = saturation

            # ══════════════════════════════════════════════════════════════
            # CÓMPUTO DE CONTROL PI
            # ══════════════════════════════════════════════════════════════
            pid_output = self.controller.compute(effective_saturation)

            # Aplicar feedforward
            pid_output_adjusted = int(pid_output * feedforward_factor)

            # ══════════════════════════════════════════════════════════════
            # EMERGENCY BRAKE MULTINIVEL
            # ══════════════════════════════════════════════════════════════
            emergency_brake = False
            brake_reason = ""
            brake_severity = 1.0  # 1.0 = sin freno, < 1.0 = freno aplicado

            # Nivel 1: Sobrecalentamiento (potencia excesiva)
            if power > SystemConstants.OVERHEAT_POWER_THRESHOLD:
                overheat_ratio = power / SystemConstants.OVERHEAT_POWER_THRESHOLD
                brake_severity = min(brake_severity, 0.3 / overheat_ratio)
                emergency_brake = True
                brake_reason = f"OVERHEAT P={power:.1f}W (>{SystemConstants.OVERHEAT_POWER_THRESHOLD}W)"

            # Nivel 2: Flyback voltage (transitorios peligrosos)
            flyback_threshold = SystemConstants.MAX_FLYBACK_VOLTAGE * 0.7
            if flyback > flyback_threshold:
                flyback_ratio = flyback / flyback_threshold
                brake_severity = min(brake_severity, 0.5 / flyback_ratio)
                emergency_brake = True
                brake_reason = f"FLYBACK V={flyback:.2f}V (>{flyback_threshold:.2f}V)"

            # Nivel 3: Saturación predicha alta (preventivo)
            if predicted_sat > 0.92 and not in_steady_state:
                brake_severity = min(brake_severity, 0.7)
                emergency_brake = True
                brake_reason = f"PREDICTED_SAT={predicted_sat:.2f}"

            # Nivel 4: Fallos consecutivos
            if failed_batches_count >= 3:
                brake_severity = min(brake_severity, 0.5)
                emergency_brake = True
                brake_reason = f"CONSECUTIVE_FAILURES={failed_batches_count}"

            if emergency_brake:
                pid_output_adjusted = max(
                    SystemConstants.MIN_BATCH_SIZE_FLOOR,
                    int(pid_output_adjusted * brake_severity)
                )
                self._emergency_brake_count += 1
                self._stats.emergency_brakes_triggered += 1
                self.logger.warning(
                    f"🛑 EMERGENCY BRAKE [{self._emergency_brake_count}]: {brake_reason} "
                    f"→ batch_size reducido a {pid_output_adjusted}"
                )

            # ══════════════════════════════════════════════════════════════
            # PROCESAMIENTO DEL BATCH
            # ══════════════════════════════════════════════════════════════
            result = self._process_single_batch_with_recovery(
                batch=batch,
                cache=cache,
                consecutive_failures=failed_batches_count,
                telemetry=telemetry,
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

                # Reducir contador de fallos (con floor en 0)
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
                    if self.condenser_config.enable_partial_recovery:
                        pid_output_adjusted = SystemConstants.MIN_BATCH_SIZE_FLOOR
                        self.logger.warning(
                            f"⚠️ Activando recuperación extrema: "
                            f"{failed_batches_count} fallos consecutivos"
                        )
                    else:
                        raise ProcessingError(
                            f"Límite de batches fallidos alcanzado: {failed_batches_count}"
                        )

            # ══════════════════════════════════════════════════════════════
            # CALLBACKS Y TELEMETRÍA
            # ══════════════════════════════════════════════════════════════
            if on_progress:
                try:
                    on_progress(self._stats)
                except Exception as e:
                    self.logger.debug(f"Error en on_progress: {e}")

            if telemetry and (iteration % 10 == 0 or emergency_brake):
                telemetry.record_event(
                    "batch_iteration",
                    {
                        "iteration": iteration,
                        "progress": current_index / total_records,
                        "batch_size": batch_size,
                        "pid_output": pid_output_adjusted,
                        "saturation": saturation,
                        "predicted_saturation": predicted_sat,
                        "in_steady_state": in_steady_state,
                        "feedforward_factor": feedforward_factor,
                        "emergency_brake": emergency_brake,
                        "failed_batches": failed_batches_count,
                    },
                )

            # ══════════════════════════════════════════════════════════════
            # ACTUALIZACIÓN DE ÍNDICE Y BATCH SIZE
            # ══════════════════════════════════════════════════════════════
            current_index = end_index

            # Inercia adaptativa: mayor en estado estacionario
            if in_steady_state:
                inertia = 0.85
            elif emergency_brake:
                inertia = 0.3  # Respuesta rápida en emergencia
            else:
                inertia = 0.65

            # Filtro de primer orden para batch size
            current_batch_size = int(
                inertia * current_batch_size + (1.0 - inertia) * pid_output_adjusted
            )

            # Aplicar límites
            current_batch_size = max(
                SystemConstants.MIN_BATCH_SIZE_FLOOR,
                min(current_batch_size, self.condenser_config.max_batch_size)
            )

        # Log de resumen
        if iteration >= max_iterations:
            self.logger.warning(
                f"⚠️ Máximo de iteraciones alcanzado: {max_iterations}"
            )

        return processed_batches

    def _estimate_cache_hits(self, batch: List, cache: Dict) -> int:
        """
        Estimación bayesiana de cache hits con actualización incremental.

        ══════════════════════════════════════════════════════════════════
        MODELO BAYESIANO
        ══════════════════════════════════════════════════════════════════

        Utilizamos un modelo Beta-Binomial para la tasa de hits:

            Prior: p ~ Beta(α, β)
            Likelihood: k | n, p ~ Binomial(n, p)
            Posterior: p | k, n ~ Beta(α + k, β + n - k)

        donde:
            - p: probabilidad de cache hit
            - k: hits observados en muestra
            - n: tamaño de muestra

        La estimación puntual es la media posterior:
            E[p | datos] = (α + k) / (α + β + n)

        Inicializamos con prior no informativo Beta(1, 1) = Uniforme(0, 1),
        que se actualiza incrementalmente con cada batch.

        ══════════════════════════════════════════════════════════════════
        """
        if not batch:
            return 0

        # Prior uniforme si no hay historial
        if not cache:
            return max(1, len(batch) // 4)

        # Inicializar estado bayesiano
        if not hasattr(self, "_cache_bayesian_state"):
            self._cache_bayesian_state = {
                "alpha": 1.0,  # Prior Beta(1, 1)
                "beta": 1.0,
                "total_samples": 0,
            }

        state = self._cache_bayesian_state

        # ══════════════════════════════════════════════════════════════
        # MUESTREO ESTRATIFICADO
        # ══════════════════════════════════════════════════════════════
        # Muestrear uniformemente a través del batch para evitar sesgo

        max_sample_size = 50
        batch_len = len(batch)

        if batch_len <= max_sample_size:
            sample_indices = range(batch_len)
        else:
            # Muestreo sistemático
            step = batch_len / max_sample_size
            sample_indices = [int(i * step) for i in range(max_sample_size)]

        # Preparar conjunto de claves de cache
        cache_keys = set(cache.keys()) if isinstance(cache, dict) else set()

        sample_hits = 0
        sample_count = 0

        for idx in sample_indices:
            if idx >= batch_len:
                continue

            record = batch[idx]
            sample_count += 1

            if isinstance(record, dict):
                record_keys = set(record.keys())

                # Calcular overlap normalizado (Jaccard-like)
                intersection = len(record_keys & cache_keys)
                union = len(record_keys | cache_keys)

                if union > 0:
                    overlap_ratio = intersection / union

                    # Considerar hit si overlap > umbral
                    if overlap_ratio > 0.25:
                        sample_hits += 1

            elif hasattr(record, '__dict__'):
                # Para objetos, verificar atributos
                record_attrs = set(dir(record))
                if len(record_attrs & cache_keys) > 0:
                    sample_hits += 1

        if sample_count == 0:
            return max(1, batch_len // 4)

        # ══════════════════════════════════════════════════════════════
        # ACTUALIZACIÓN BAYESIANA
        # ══════════════════════════════════════════════════════════════

        # Actualizar parámetros de la Beta
        state["alpha"] += sample_hits
        state["beta"] += (sample_count - sample_hits)
        state["total_samples"] += sample_count

        # Limitar crecimiento de parámetros (ventana efectiva)
        MAX_EFFECTIVE_SAMPLES = 200
        if state["alpha"] + state["beta"] > MAX_EFFECTIVE_SAMPLES + 2:
            scale = MAX_EFFECTIVE_SAMPLES / (state["alpha"] + state["beta"] - 2)
            state["alpha"] = 1.0 + (state["alpha"] - 1.0) * scale
            state["beta"] = 1.0 + (state["beta"] - 1.0) * scale

        # Media posterior
        posterior_mean = state["alpha"] / (state["alpha"] + state["beta"])

        # Varianza posterior para diagnóstico
        posterior_var = (
            state["alpha"] * state["beta"] /
            ((state["alpha"] + state["beta"])**2 * (state["alpha"] + state["beta"] + 1))
        )

        # Estimación final
        estimated_hits = max(1, int(posterior_mean * batch_len))

        return estimated_hits

    def _predict_next_saturation(self, history: List[float]) -> float:
        """
        Predicción de saturación usando Filtro de Kalman Extendido (EKF).

        ══════════════════════════════════════════════════════════════════
        MODELO DE ESTADO
        ══════════════════════════════════════════════════════════════════

        Estado: x = [s, v, a]ᵀ
            - s: saturación
            - v: velocidad (ds/dt)
            - a: aceleración (d²s/dt²)

        Dinámica (oscilador amortiguado con equilibrio variable):
            ṡ = v
            v̇ = a - β·v - ω²·(s - s_eq)
            ȧ = -γ·a + w_a

        donde:
            β: coeficiente de amortiguamiento
            ω: frecuencia natural
            s_eq: punto de equilibrio (se adapta)
            γ: tasa de decaimiento de aceleración
            w_a: ruido de proceso

        Observación:
            z = s + v_z

        donde v_z es ruido de medición.

        ══════════════════════════════════════════════════════════════════
        IMPLEMENTACIÓN
        ══════════════════════════════════════════════════════════════════

        Usamos discretización de Euler con paso dt = 1.

        El filtro adapta los parámetros del modelo (β, ω, s_eq) basándose
        en las innovaciones para mejorar el tracking.

        ══════════════════════════════════════════════════════════════════
        """
        MIN_HISTORY = 3

        if len(history) < MIN_HISTORY:
            return history[-1] if history else 0.5

        # ══════════════════════════════════════════════════════════════
        # INICIALIZACIÓN DEL EKF
        # ══════════════════════════════════════════════════════════════
        if self._ekf_state is None:
            # Estimar condiciones iniciales desde historial
            s0 = history[-1]
            v0 = history[-1] - history[-2] if len(history) >= 2 else 0.0
            a0 = 0.0
            if len(history) >= 3:
                v_prev = history[-2] - history[-3]
                a0 = v0 - v_prev

            self._ekf_state = {
                # Estado
                "x": [s0, v0, a0],

                # Covarianza del estado (diagonal para simplicidad)
                "P": [
                    [0.05, 0.0, 0.0],
                    [0.0, 0.10, 0.0],
                    [0.0, 0.0, 0.05],
                ],

                # Covarianza del proceso
                "Q": [
                    [0.002, 0.0, 0.0],
                    [0.0, 0.02, 0.0],
                    [0.0, 0.0, 0.01],
                ],

                # Varianza de medición
                "R": 0.01,

                # Parámetros del modelo
                "beta": 0.4,     # Amortiguamiento
                "omega": 0.15,   # Frecuencia natural
                "gamma": 0.6,    # Decaimiento de aceleración
                "s_eq": 0.5,     # Equilibrio inicial

                # Historial de innovaciones
                "innovations": deque(maxlen=20),

                # Contador de iteraciones para adaptación
                "iteration": 0,
            }

        ekf = self._ekf_state
        ekf["iteration"] += 1
        dt = 1.0

        # Extraer estado y parámetros
        x = ekf["x"]
        P = ekf["P"]
        s, v, a = x[0], x[1], x[2]

        beta = ekf["beta"]
        omega = ekf["omega"]
        gamma = ekf["gamma"]
        s_eq = ekf["s_eq"]

        # ══════════════════════════════════════════════════════════════
        # PREDICCIÓN
        # ══════════════════════════════════════════════════════════════

        # Modelo no lineal discretizado
        s_pred = s + v * dt
        restoring_force = omega * omega * (s - s_eq)
        v_pred = v + (a - beta * v - restoring_force) * dt
        a_pred = a * (1.0 - gamma * dt)

        x_pred = [s_pred, v_pred, a_pred]

        # Jacobiano F = ∂f/∂x
        F = [
            [1.0, dt, 0.0],
            [-omega*omega*dt, 1.0 - beta*dt, dt],
            [0.0, 0.0, 1.0 - gamma*dt],
        ]

        # Propagación de covarianza: P_pred = F·P·Fᵀ + Q
        # Implementación explícita del producto matricial
        Q = ekf["Q"]
        P_pred = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        # Calcular F·P
        FP = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    FP[i][j] += F[i][k] * P[k][j]

        # Calcular (F·P)·Fᵀ + Q
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    P_pred[i][j] += FP[i][k] * F[j][k]  # F[j][k] = Fᵀ[k][j]
                P_pred[i][j] += Q[i][j]

        # ══════════════════════════════════════════════════════════════
        # ACTUALIZACIÓN
        # ══════════════════════════════════════════════════════════════

        z = history[-1]  # Medición actual

        # H = [1, 0, 0] → solo observamos saturación
        # Innovación
        y = z - x_pred[0]

        # Varianza de innovación: S = H·P_pred·Hᵀ + R = P_pred[0][0] + R
        S = P_pred[0][0] + ekf["R"]

        # Protección contra S muy pequeño
        if S < 1e-10:
            S = 1e-10

        # Ganancia de Kalman: K = P_pred·Hᵀ / S
        K = [P_pred[0][0] / S, P_pred[1][0] / S, P_pred[2][0] / S]

        # Estado actualizado
        x_new = [
            x_pred[0] + K[0] * y,
            x_pred[1] + K[1] * y,
            x_pred[2] + K[2] * y,
        ]

        # Covarianza actualizada: P = (I - K·H)·P_pred
        # Con H = [1, 0, 0], esto simplifica a:
        P_new = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for i in range(3):
            for j in range(3):
                P_new[i][j] = P_pred[i][j] - K[i] * P_pred[0][j]

        # Asegurar simetría y positividad
        for i in range(3):
            for j in range(i + 1, 3):
                avg = (P_new[i][j] + P_new[j][i]) / 2.0
                P_new[i][j] = avg
                P_new[j][i] = avg
            # Asegurar elementos diagonales positivos
            P_new[i][i] = max(1e-6, P_new[i][i])

        # ══════════════════════════════════════════════════════════════
        # ADAPTACIÓN DE PARÁMETROS
        # ══════════════════════════════════════════════════════════════

        ekf["innovations"].append(y)

        if len(ekf["innovations"]) >= 5:
            innovations = list(ekf["innovations"])
            n_innov = len(innovations)

            mean_innov = sum(innovations) / n_innov
            var_innov = sum((i - mean_innov)**2 for i in innovations) / n_innov

            # Varianza esperada de innovaciones
            expected_var = S

            # Ratio de consistencia
            nis = var_innov / max(expected_var, 1e-6)  # Normalized Innovation Squared

            # Adaptar Q si innovaciones son inconsistentes
            if nis > 2.0:
                # Subestimamos incertidumbre → aumentar Q
                q_scale = min(1.2, 1.0 + 0.1 * (nis - 2.0))
                for i in range(3):
                    ekf["Q"][i][i] *= q_scale
            elif nis < 0.3:
                # Sobreestimamos → reducir Q
                q_scale = max(0.85, 1.0 - 0.1 * (0.3 - nis))
                for i in range(3):
                    ekf["Q"][i][i] *= q_scale

            # Limitar Q para evitar divergencia
            for i in range(3):
                ekf["Q"][i][i] = max(1e-4, min(0.5, ekf["Q"][i][i]))

            # Adaptar s_eq si hay sesgo sistemático
            if abs(mean_innov) > 0.03:
                # El filtro predice sistemáticamente alto o bajo
                adaptation_rate = 0.1
                ekf["s_eq"] += adaptation_rate * mean_innov
                ekf["s_eq"] = max(0.1, min(0.9, ekf["s_eq"]))

            # Adaptar omega si hay oscilaciones
            if n_innov >= 8:
                # Detectar oscilaciones por cambios de signo
                sign_changes = sum(
                    1 for i in range(1, n_innov)
                    if innovations[i] * innovations[i-1] < 0
                )
                oscillation_freq = sign_changes / (n_innov - 1)

                if oscillation_freq > 0.6:
                    # Oscilando mucho → reducir omega (menos oscilatorio)
                    ekf["omega"] *= 0.95
                elif oscillation_freq < 0.2:
                    # Poco oscilatorio → aumentar omega
                    ekf["omega"] *= 1.03

                ekf["omega"] = max(0.05, min(0.5, ekf["omega"]))

        # Guardar estado
        ekf["x"] = x_new
        ekf["P"] = P_new

        # ══════════════════════════════════════════════════════════════
        # PREDICCIÓN A UN PASO ADELANTE
        # ══════════════════════════════════════════════════════════════

        s_next = x_new[0] + x_new[1] * dt

        # Asegurar límites físicos estrictos (Clamping simple para fidelidad de predicción)
        s_bounded = max(0.0, min(1.0, s_next))

        return s_bounded

    def _process_single_batch_with_recovery(
        self,
        batch: List,
        cache: Dict,
        consecutive_failures: int,
        telemetry: Optional[TelemetryContext] = None,
        _recursion_depth: int = 0,
    ) -> BatchResult:
        """
        Procesamiento de batch con estrategia de recuperación multinivel.

        ══════════════════════════════════════════════════════════════════
        NIVELES DE RECUPERACIÓN
        ══════════════════════════════════════════════════════════════════

        NIVEL 0: Intento directo
            - Procesar batch completo
            - Si éxito → retornar resultado
            - Si fallo → avanzar a nivel 1

        NIVEL 1: División binaria
            - Dividir batch en mitades
            - Procesar cada mitad recursivamente
            - Combinar resultados
            - Profundidad máxima limitada para evitar stack overflow

        NIVEL 2: Procesamiento unitario con cuarentena
            - Procesar registro por registro
            - Registros fallidos van a cuarentena
            - Retornar registros exitosos

        ══════════════════════════════════════════════════════════════════
        """
        MAX_RECURSION_DEPTH = 5
        MIN_SPLIT_SIZE = 3
        MAX_UNIT_PROCESSING_SIZE = 150

        if not batch:
            return BatchResult(
                success=True,
                records_processed=0,
                dataframe=pd.DataFrame()
            )

        batch_size = len(batch)

        # ══════════════════════════════════════════════════════════════
        # NIVEL 0: INTENTO DIRECTO
        # ══════════════════════════════════════════════════════════════

        if consecutive_failures == 0 and _recursion_depth == 0:
            try:
                parsed_data = ParsedData(batch, cache)
                df = self._rectify_signal(parsed_data, telemetry=telemetry)

                if df is not None:
                    return BatchResult(
                        success=True,
                        dataframe=df if not df.empty else pd.DataFrame(),
                        records_processed=len(df) if not df.empty else 0
                    )
                else:
                    return BatchResult(
                        success=True,
                        dataframe=pd.DataFrame(),
                        records_processed=0
                    )

            except Exception as e:
                self.logger.debug(
                    f"Nivel 0 falló para batch de {batch_size}: {type(e).__name__}"
                )
                # Continuar a recuperación

        # ══════════════════════════════════════════════════════════════
        # NIVEL 1: DIVISIÓN BINARIA
        # ══════════════════════════════════════════════════════════════

        can_split = (
            batch_size > MIN_SPLIT_SIZE and
            _recursion_depth < MAX_RECURSION_DEPTH and
            consecutive_failures <= 3
        )

        if can_split:
            try:
                mid = batch_size // 2

                # Procesar mitades con profundidad incrementada
                left_result = self._process_single_batch_with_recovery(
                    batch=batch[:mid],
                    cache=cache,
                    consecutive_failures=consecutive_failures + 1,
                    telemetry=telemetry,
                    _recursion_depth=_recursion_depth + 1,
                )

                right_result = self._process_single_batch_with_recovery(
                    batch=batch[mid:],
                    cache=cache,
                    consecutive_failures=consecutive_failures + 1,
                    telemetry=telemetry,
                    _recursion_depth=_recursion_depth + 1,
                )

                # Agregar resultados
                dfs_to_concat = []
                total_records = 0

                for result in [left_result, right_result]:
                    if result.success and result.dataframe is not None:
                        if not result.dataframe.empty:
                            dfs_to_concat.append(result.dataframe)
                        total_records += result.records_processed

                if dfs_to_concat:
                    try:
                        combined_df = pd.concat(dfs_to_concat, ignore_index=True)
                    except Exception as concat_error:
                        self.logger.warning(f"Error concatenando splits: {concat_error}")
                        # Intentar concatenación más robusta
                        combined_df = self._safe_concat(dfs_to_concat)
                else:
                    combined_df = pd.DataFrame()

                success = total_records > 0 or (left_result.success and right_result.success)

                return BatchResult(
                    success=success,
                    dataframe=combined_df,
                    records_processed=total_records,
                    error_message="" if success else "División binaria sin resultados"
                )

            except RecursionError:
                self.logger.error("Recursión máxima alcanzada en división binaria")
                # Fall through a nivel 2

            except Exception as e:
                self.logger.warning(f"División binaria falló: {e}")
                # Continuar a nivel 2

        # ══════════════════════════════════════════════════════════════
        # NIVEL 2: PROCESAMIENTO UNITARIO CON CUARENTENA
        # ══════════════════════════════════════════════════════════════

        if batch_size <= MAX_UNIT_PROCESSING_SIZE:
            successful_dfs = []
            quarantined_indices = []
            processed_count = 0

            for idx, record in enumerate(batch):
                try:
                    parsed = ParsedData([record], cache)
                    df = self._rectify_signal(parsed, telemetry=telemetry)

                    if df is not None and not df.empty:
                        successful_dfs.append(df)
                        processed_count += len(df)

                except Exception as e:
                    quarantined_indices.append(idx)

                    # Logging limitado para evitar spam
                    if len(quarantined_indices) <= 3:
                        self.logger.debug(
                            f"Registro {idx} en cuarentena: {type(e).__name__}"
                        )

            # Log de cuarentena si hay muchos
            if len(quarantined_indices) > 3:
                self.logger.debug(
                    f"Total registros en cuarentena: {len(quarantined_indices)}/{batch_size}"
                )

            if successful_dfs:
                combined_df = self._safe_concat(successful_dfs)
            else:
                combined_df = pd.DataFrame()

            success = processed_count > 0
            recovery_rate = processed_count / batch_size if batch_size > 0 else 0.0

            return BatchResult(
                success=success,
                dataframe=combined_df,
                records_processed=processed_count,
                error_message=(
                    f"Recuperación unitaria: {processed_count}/{batch_size} "
                    f"({100*recovery_rate:.1f}%) - {len(quarantined_indices)} en cuarentena"
                )
            )

        # ══════════════════════════════════════════════════════════════
        # FALLO TOTAL
        # ══════════════════════════════════════════════════════════════

        return BatchResult(
            success=False,
            dataframe=None,
            records_processed=0,
            error_message=(
                f"Recuperación fallida: batch_size={batch_size}, "
                f"depth={_recursion_depth}, failures={consecutive_failures}"
            )
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
        """
        Consolida resultados de múltiples batches con validación.

        Args:
            batches: Lista de DataFrames procesados

        Returns:
            DataFrame consolidado y validado
        """
        # Filtrar batches válidos
        valid_batches = [
            df for df in batches
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty
        ]

        if not valid_batches:
            self.logger.info("No hay batches válidos para consolidar")
            return pd.DataFrame()

        # Verificar límite de batches
        if len(valid_batches) > SystemConstants.MAX_BATCHES_TO_CONSOLIDATE:
            self.logger.warning(
                f"Truncando batches: {len(valid_batches)} → "
                f"{SystemConstants.MAX_BATCHES_TO_CONSOLIDATE}"
            )
            valid_batches = valid_batches[:SystemConstants.MAX_BATCHES_TO_CONSOLIDATE]

        # Estimar memoria requerida
        total_rows = sum(len(df) for df in valid_batches)
        avg_cols = sum(len(df.columns) for df in valid_batches) / len(valid_batches)

        self.logger.debug(
            f"Consolidando {len(valid_batches)} batches: "
            f"~{total_rows:,} filas, ~{avg_cols:.0f} columnas"
        )

        try:
            result = self._safe_concat(valid_batches)

            # Validación post-consolidación
            if not result.empty:
                # Eliminar duplicados si hay columna de ID
                id_columns = [col for col in result.columns if 'id' in col.lower()]
                if id_columns:
                    original_len = len(result)
                    result = result.drop_duplicates(subset=id_columns, keep='first')
                    if len(result) < original_len:
                        self.logger.info(
                            f"Eliminados {original_len - len(result)} duplicados"
                        )

                # Actualizar estadísticas
                self._stats.processed_records = len(result)

            return result

        except Exception as e:
            raise ProcessingError(f"Error consolidando resultados: {e}")

    def _validate_output(self, df: pd.DataFrame) -> None:
        """
        Valida el DataFrame de salida con múltiples criterios.

        Validaciones:
        1. DataFrame no vacío (warning o error según config)
        2. Mínimo de registros
        3. Columnas requeridas (si están definidas)
        4. Tipos de datos consistentes
        """
        if df.empty:
            msg = "DataFrame de salida está vacío"

            if self.condenser_config.enable_strict_validation:
                raise ProcessingError(msg)

            self.logger.warning(f"⚠️ {msg}")
            return

        n_records = len(df)
        n_columns = len(df.columns)

        # Verificar mínimo de registros
        min_threshold = self.condenser_config.min_records_threshold

        if n_records < min_threshold:
            msg = f"Registros insuficientes: {n_records} < {min_threshold}"

            if self.condenser_config.enable_strict_validation:
                raise ProcessingError(msg)

            self.logger.warning(f"⚠️ {msg}")

        # Verificar columnas requeridas (si están configuradas)
        required_columns = getattr(self.condenser_config, 'required_columns', None)

        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)

            if missing_columns:
                msg = f"Columnas requeridas faltantes: {missing_columns}"

                if self.condenser_config.enable_strict_validation:
                    raise ProcessingError(msg)

                self.logger.warning(f"⚠️ {msg}")

        # Verificar valores nulos excesivos
        null_ratio = df.isnull().sum().sum() / (n_records * n_columns)

        if null_ratio > 0.5:
            self.logger.warning(
                f"⚠️ Alto porcentaje de valores nulos: {100*null_ratio:.1f}%"
            )

        # Log de resumen
        self.logger.info(
            f"📋 Validación de salida: {n_records:,} registros, "
            f"{n_columns} columnas, {100*null_ratio:.1f}% nulos"
        )

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
        """
        Retorna estadísticas completas del procesamiento.

        Incluye:
        - Estadísticas base del pipeline
        - Diagnósticos del controlador
        - Análisis de tendencias de física
        - Métricas actuales del sistema
        """
        # Estadísticas base
        base_stats = asdict(self._stats)

        # Métricas actuales (última iteración)
        current_metrics = {}
        if self._metrics_history:
            current_metrics = dict(self._metrics_history[-1])

        # Tendencias de métricas
        trends = {}
        if len(self._metrics_history) >= 5:
            recent = list(self._metrics_history)[-5:]

            for key in ['saturation', 'power', 'complexity']:
                values = [m.get(key, 0) for m in recent if key in m]
                if values:
                    trends[f"{key}_trend"] = (values[-1] - values[0]) / len(values)
                    trends[f"{key}_mean"] = sum(values) / len(values)

        # Diagnósticos del controlador
        controller_diag = {}
        try:
            controller_diag = self.controller.get_diagnostics()
        except Exception as e:
            self.logger.debug(f"Error obteniendo diagnósticos de controlador: {e}")

        # Análisis de física
        physics_analysis = {}
        try:
            physics_analysis = self.physics.get_trend_analysis()
        except Exception as e:
            self.logger.debug(f"Error obteniendo análisis de física: {e}")

        return {
            "statistics": base_stats,
            "current_metrics": current_metrics,
            "trends": trends,
            "controller": controller_diag,
            "physics": physics_analysis,
            "emergency_brakes": self._emergency_brake_count,
            "ekf_state": {
                "active": self._ekf_state is not None,
                "iteration": self._ekf_state.get("iteration", 0) if self._ekf_state else 0,
                "equilibrium": self._ekf_state.get("s_eq", 0.5) if self._ekf_state else 0.5,
            },
            "timing": {
                "elapsed_s": time.time() - self._start_time if self._start_time else 0,
                "throughput_per_s": (
                    self._stats.processed_records /
                    max(0.001, time.time() - self._start_time)
                    if self._start_time else 0
                ),
            },
        }

    def get_system_health(self) -> Dict[str, Any]:
        """
        Evalúa la salud del sistema con múltiples indicadores.

        Niveles de salud:
        - HEALTHY: Todo funcionando correctamente
        - DEGRADED: Funcionando pero con advertencias
        - CRITICAL: Problemas serios que requieren atención
        - FAILED: Sistema en estado de fallo
        """
        issues = []
        warnings = []

        # ══════════════════════════════════════════════════════════════
        # EVALUACIÓN DEL CONTROLADOR
        # ══════════════════════════════════════════════════════════════
        try:
            controller_diag = self.controller.get_stability_analysis()
            stability_class = controller_diag.get("stability_class", "UNKNOWN")

            if stability_class == "UNSTABLE":
                issues.append("Control inestable: sistema divergente")
            elif stability_class == "POTENTIALLY_UNSTABLE":
                warnings.append("Control potencialmente inestable")
            elif stability_class == "MARGINALLY_STABLE":
                warnings.append("Estabilidad marginal del controlador")

            # Verificar utilización integral
            integral_util = controller_diag.get("integral_saturation", 0)
            if integral_util > 0.9:
                warnings.append(f"Saturación integral alta: {100*integral_util:.0f}%")

        except Exception as e:
            warnings.append(f"Error evaluando controlador: {e}")

        # ══════════════════════════════════════════════════════════════
        # EVALUACIÓN DE FRENOS DE EMERGENCIA
        # ══════════════════════════════════════════════════════════════

        if self._emergency_brake_count > 10:
            issues.append(
                f"Exceso de frenos de emergencia: {self._emergency_brake_count}"
            )
        elif self._emergency_brake_count > 5:
            warnings.append(
                f"Frenos de emergencia frecuentes: {self._emergency_brake_count}"
            )

        # ══════════════════════════════════════════════════════════════
        # EVALUACIÓN DE RENDIMIENTO
        # ══════════════════════════════════════════════════════════════

        if self._stats.total_records > 0:
            success_rate = self._stats.processed_records / self._stats.total_records

            if success_rate < 0.5:
                issues.append(f"Tasa de éxito muy baja: {100*success_rate:.1f}%")
            elif success_rate < 0.8:
                warnings.append(f"Tasa de éxito degradada: {100*success_rate:.1f}%")

        # ══════════════════════════════════════════════════════════════
        # EVALUACIÓN DEL EKF
        # ══════════════════════════════════════════════════════════════

        if self._ekf_state:
            ekf_iter = self._ekf_state.get("iteration", 0)
            if ekf_iter > 100:
                # Verificar convergencia del EKF
                P = self._ekf_state.get("P", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                trace_P = sum(P[i][i] for i in range(3))

                if trace_P > 1.0:
                    warnings.append(f"EKF con incertidumbre alta: tr(P)={trace_P:.3f}")

        # ══════════════════════════════════════════════════════════════
        # DETERMINACIÓN DE ESTADO DE SALUD
        # ══════════════════════════════════════════════════════════════

        if issues:
            health = "CRITICAL" if len(issues) >= 2 else "DEGRADED"
        elif warnings:
            health = "DEGRADED" if len(warnings) >= 3 else "HEALTHY"
        else:
            health = "HEALTHY"

        # Uptime
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "health": health,
            "issues": issues,
            "warnings": warnings,
            "uptime_s": uptime,
            "emergency_brakes": self._emergency_brake_count,
            "processed_ratio": (
                self._stats.processed_records / max(1, self._stats.total_records)
            ),
            "stability_baseline": self._stability_baseline,
            "recommendations": self._generate_health_recommendations(issues, warnings),
        }

    def _safe_concat(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenación robusta de DataFrames con manejo de esquemas inconsistentes.

        Estrategia:
        1. Identificar esquema común (intersección de columnas)
        2. Alinear DataFrames al esquema común
        3. Concatenar con manejo de tipos

        Args:
            dataframes: Lista de DataFrames a concatenar

        Returns:
            DataFrame concatenado
        """
        if not dataframes:
            return pd.DataFrame()

        if len(dataframes) == 1:
            return dataframes[0]

        # Filtrar DataFrames vacíos
        valid_dfs = [df for df in dataframes if df is not None and not df.empty]

        if not valid_dfs:
            return pd.DataFrame()

        if len(valid_dfs) == 1:
            return valid_dfs[0]

        try:
            # Intento directo
            return pd.concat(valid_dfs, ignore_index=True, sort=False)

        except Exception as e:
            self.logger.debug(f"Concatenación directa falló: {e}, intentando alineación")

            try:
                # Encontrar columnas comunes
                common_columns = set(valid_dfs[0].columns)
                for df in valid_dfs[1:]:
                    common_columns &= set(df.columns)

                if not common_columns:
                    self.logger.warning("No hay columnas comunes entre DataFrames")
                    # Usar unión en lugar de intersección
                    all_columns = set()
                    for df in valid_dfs:
                        all_columns |= set(df.columns)
                    common_columns = all_columns

                common_columns = sorted(common_columns)

                # Alinear cada DataFrame
                aligned_dfs = []
                for df in valid_dfs:
                    # Agregar columnas faltantes con NaN
                    for col in common_columns:
                        if col not in df.columns:
                            df = df.copy()
                            df[col] = pd.NA

                    aligned_dfs.append(df[list(common_columns)])

                return pd.concat(aligned_dfs, ignore_index=True, sort=False)

            except Exception as e2:
                self.logger.error(f"Concatenación con alineación falló: {e2}")

                # Último recurso: concatenar el primero válido
                return valid_dfs[0]

    def _generate_health_recommendations(
        self,
        issues: List[str],
        warnings: List[str]
    ) -> List[str]:
        """Genera recomendaciones basadas en problemas detectados."""
        recommendations = []

        # Recomendaciones por issues
        for issue in issues:
            if "inestable" in issue.lower():
                recommendations.append(
                    "Reducir ganancias del controlador (Kp, Ki) para mejorar estabilidad"
                )
            if "frenos de emergencia" in issue.lower():
                recommendations.append(
                    "Aumentar capacidad del sistema o reducir carga de trabajo"
                )
            if "tasa de éxito" in issue.lower():
                recommendations.append(
                    "Revisar calidad de datos de entrada y configuración del parser"
                )

        # Recomendaciones por warnings
        for warning in warnings:
            if "integral" in warning.lower():
                recommendations.append(
                    "Considerar ajustar integral_limit_factor o reducir Ki"
                )
            if "ekf" in warning.lower():
                recommendations.append(
                    "Reiniciar sesión para resetear estado del predictor"
                )

        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_recommendations = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique_recommendations.append(r)

        return unique_recommendations
