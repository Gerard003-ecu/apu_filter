"""
Módulo de Capacitancia Lógica para el procesamiento de flujos de datos.

Este módulo introduce el `DataFluxCondenser`, un componente de alto nivel que
actúa como una fachada estabilizadora para el pipeline de procesamiento de
Análisis de Precios Unitarios (APU). Su función principal es garantizar la
integridad, coherencia y estabilidad del flujo de datos antes de que ingrese
al núcleo del sistema.

Principios de Diseño:
- **Capacitancia Lógica:** Inspirado en los principios de un circuito RLC,
  el condensador "absorbe" datos crudos y los "descarga" de manera controlada,
  filtrando el ruido y la turbulencia.
- **Orquestación, no Implementación:** No contiene lógica de negocio de bajo
  nivel. En su lugar, orquesta componentes especializados como `ReportParserCrudo`
  (el "Guardia") y `APUProcessor` (el "Cirujano").
- **Telemetría Física:** Incorpora un `FluxPhysicsEngine` para calcular
  métricas de saturación, complejidad e inductancia (flyback), proporcionando
  una visión cuantitativa de la "salud" del flujo de datos entrante.
- **Control Adaptativo (PID):** Implementa un lazo de control Proporcional-Integral
  para ajustar dinámicamente el flujo de procesamiento (tamaño de lote) en función
  de la saturación y complejidad detectada, asegurando "Flujo Laminar".
- **Robustez y Tolerancia a Fallos:** Implementa validaciones estrictas en cada
  etapa y un manejo de errores detallado para prevenir la propagación de datos
  corruptos.
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
        Valida parámetros con criterios de estabilidad.

        Se basa en el Criterio de Jury para sistemas discretos.
        Para un sistema PI discreto con planta de primer orden G(z) = K/(z-a):
        - Estabilidad requiere que todos los polos estén dentro del círculo unitario.
        - Condición necesaria: |a - K*Kp| < 1 y Ki*T < 2*(1 + a - K*Kp).
        """
        errors = []

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

        # Criterio de Jury simplificado para sistema normalizado
        # Asumiendo planta con ganancia unitaria y polo en a ≈ 0.9
        a_plant = 0.9
        K_plant = 1.0

        # Condición 1: Estabilidad del lazo cerrado
        closed_loop_coeff = abs(a_plant - K_plant * kp)
        if closed_loop_coeff >= 1.0:
            logger.warning(
                f"Criterio de Jury: |a - K·Kp| = {closed_loop_coeff:.3f} ≥ 1.0 "
                f"indica posible inestabilidad"
            )

        # Condición 2: Margen integral
        # Para T_s = 1 (muestreo unitario normalizado)
        T_s = 1.0
        integral_margin = 2.0 * (1.0 + a_plant - K_plant * kp)
        if ki * T_s >= integral_margin:
            logger.warning(
                f"Ki·T = {ki * T_s:.3f} ≥ margen integral {integral_margin:.3f}, "
                f"riesgo de oscilaciones"
            )

        # Relación Ki/Kp para respuesta suave
        if kp > 0 and ki / kp > 0.5:
            logger.info(f"Ratio Ki/Kp = {ki / kp:.3f} > 0.5: respuesta integral dominante")

    def _apply_ema_filter(self, measurement: float) -> float:
        """
        Aplica filtro EMA con alpha adaptativo.

        Se basa en varianza normalizada y detección de cambios abruptos (step detection).
        """
        if self._filtered_pv is None:
            self._filtered_pv = measurement
            return measurement

        # Detectar cambio abrupto (step) para bypass del filtro
        if self._last_error is not None:
            step_threshold = 0.3 * abs(self.setpoint)
            if abs(measurement - self._filtered_pv) > step_threshold:
                # Cambio abrupto: reducir inercia del filtro
                self._filtered_pv = 0.5 * measurement + 0.5 * self._filtered_pv
                return self._filtered_pv

        # Alpha adaptativo basado en varianza del error
        if len(self._error_history) >= 3:
            recent_errors = list(self._error_history)[-5:]
            n = len(recent_errors)
            mean_error = sum(recent_errors) / n

            # Varianza con corrección de Bessel para muestras pequeñas
            if n > 1:
                error_variance = sum((e - mean_error) ** 2 for e in recent_errors) / (n - 1)
            else:
                error_variance = 0.0

            # Mapeo no lineal: varianza alta → alpha bajo (más suavizado)
            # Función sigmoide inversa para transición suave
            # Ajuste de escala: varianza típica de 0.01 es ruido bajo, 0.2 es alto
            normalized_var = min(error_variance * 10.0, 1.0)  # Escalar para sensibilidad

            adaptive_alpha = 0.1 + 0.4 / (1.0 + 5.0 * normalized_var)
            self._ema_alpha = max(0.05, min(0.6, adaptive_alpha))

        self._filtered_pv = (
            self._ema_alpha * measurement + (1 - self._ema_alpha) * self._filtered_pv
        )
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
        Calcula números de Betti para el complejo simplicial del grafo.

        Para grafos simples (complejos de dimensión 1):
        - β₀ = componentes conexas (dim del kernel de ∂₁)
        - β₁ = ciclos independientes = E - V + β₀ (por Euler-Poincaré)
        - β_k = 0 para k ≥ 2

        También calcula la característica de Euler: χ = β₀ - β₁
        """
        if self._vertex_count == 0:
            return {0: 0, 1: 0, 2: 0, "euler_characteristic": 0}

        # === CALCULAR β₀: COMPONENTES CONEXAS ===
        # Union-Find para eficiencia O(V·α(V))
        parent = list(range(self._vertex_count))
        rank = [0] * self._vertex_count

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])  # Compresión de caminos
            return parent[x]

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            # Union por rango
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        # Procesar todas las aristas
        for v in range(self._vertex_count):
            for neighbor in self._adjacency_list.get(v, set()):
                if neighbor > v:  # Evitar procesar dos veces
                    union(v, neighbor)

        # Contar componentes (raíces únicas)
        beta_0 = len(set(find(v) for v in range(self._vertex_count)))

        # === CALCULAR β₁: CICLOS INDEPENDIENTES ===
        # Por teorema de Euler-Poincaré para grafos: V - E + F = 2
        # Para grafos planares sin caras internas: β₁ = E - V + β₀
        beta_1 = max(0, self._edge_count - self._vertex_count + beta_0)

        # Característica de Euler
        euler_char = beta_0 - beta_1

        return {
            0: beta_0,
            1: beta_1,
            2: 0,
            "euler_characteristic": euler_char,
            "is_tree": beta_1 == 0 and beta_0 == 1,
            "is_forest": beta_1 == 0,
            "cyclomatic_complexity": beta_1 + 1,  # McCabe para grafos de flujo
        }

    def calculate_gyroscopic_stability(self, current_I: float) -> float:
        """
        Estabilidad Giroscópica basada en modelo de trompo simétrico.

        Usa el criterio de estabilidad de Routh para rotación:
        - ω (velocidad angular) ~ |I| (corriente como proxy de "rotación")
        - Precesión detectada cuando dω/dt causa desviación del eje

        Sg = exp(-κ|dI/dt|) · cos(θ_precesión) · factor_inercia
        """
        if not self._initialized:
            self._ema_current = current_I
            self._last_current = current_I
            self._last_time = time.time()
            self._initialized = True
            return 1.0

        current_time = time.time()
        dt = max(1e-6, current_time - self._last_time)

        # === DERIVADA DE LA CORRIENTE (aceleración angular) ===
        dI_dt = (current_I - self._last_current) / dt

        # === ACTUALIZAR EJE DE ROTACIÓN (EMA) ===
        alpha = SystemConstants.GYRO_EMA_ALPHA
        self._ema_current = alpha * current_I + (1 - alpha) * self._ema_current

        # === ÁNGULO DE PRECESIÓN ===
        # Desviación del eje actual respecto al eje estabilizado
        axis_deviation = current_I - self._ema_current
        precession_angle = math.atan2(axis_deviation, 1.0 + abs(self._ema_current))

        # === FACTOR DE ESTABILIDAD POR VELOCIDAD ===
        # Mayor velocidad angular → mayor estabilidad giroscópica
        # (efecto de conservación del momento angular)
        omega_normalized = abs(self._ema_current)
        inertia_factor = 1.0 - math.exp(-2.0 * omega_normalized)

        # === TÉRMINO DE NUTACIÓN ===
        # Oscilación rápida del eje - detectada por cambio en signo de dI/dt
        if not hasattr(self, "_last_dI_dt"):
            self._last_dI_dt = dI_dt
            nutation_factor = 1.0
        else:
            # Detección de cambio de signo (nutación)
            if self._last_dI_dt * dI_dt < 0:
                nutation_factor = 0.8  # Penalizar nutación
            else:
                nutation_factor = 1.0
            self._last_dI_dt = dI_dt

        # === CÁLCULO FINAL DE ESTABILIDAD ===
        sensitivity = SystemConstants.GYRO_SENSITIVITY

        # Componente exponencial por tasa de cambio
        exp_term = math.exp(-sensitivity * abs(dI_dt))

        # Componente de precesión
        precession_term = math.cos(precession_angle)

        # Estabilidad combinada
        Sg = exp_term * precession_term * inertia_factor * nutation_factor

        # Normalizar a [0, 1] con suavizado
        Sg_normalized = (Sg + 1.0) / 2.0
        Sg_normalized = max(0.0, min(1.0, Sg_normalized))

        # Actualizar estado
        self._last_current = current_I
        self._last_time = current_time

        return Sg_normalized

    def calculate_system_entropy(
        self, total_records: int, error_count: int, processing_time: float
    ) -> Dict[str, float]:
        """
        Entropía del sistema con estimadores robustos.

        1. Shannon con corrección de Horvitz-Thompson para muestreo.
        2. Rényi de orden 2 (entropía de colisión).
        3. Entropía condicional H(Error|Time).
        """
        if total_records <= 0:
            return self._get_zero_entropy()

        success_count = max(0, total_records - error_count)

        # Probabilidades con suavizado de Laplace (evita log(0))
        # Equivalente a prior uniforme Beta(1,1)
        alpha = 1  # Parámetro de suavizado
        p_success = (success_count + alpha) / (total_records + 2 * alpha)
        p_error = (error_count + alpha) / (total_records + 2 * alpha)

        # === ENTROPÍA DE SHANNON ===
        H_shannon = 0.0
        for p in [p_success, p_error]:
            if p > 0:
                H_shannon -= p * math.log2(p)

        # Corrección de Miller-Madow para sesgo de muestras finitas
        m = 2  # Número de categorías
        if total_records > m:
            H_shannon_corrected = H_shannon + (m - 1) / (2 * total_records * math.log(2))
        else:
            H_shannon_corrected = H_shannon

        # === ENTROPÍA DE RÉNYI (orden α=2) ===
        # H₂ = -log₂(Σpᵢ²) - más sensible a probabilidades dominantes
        sum_p_squared = p_success**2 + p_error**2
        H_renyi_2 = -math.log2(sum_p_squared) if sum_p_squared > 0 else 0.0

        # === ENTROPÍA DE TSALLIS (q=2) ===
        # Coincide con índice de Gini-Simpson
        q = 2.0
        H_tsallis = (1 - sum_p_squared) / (q - 1)

        # === DIVERGENCIA KL RESPECTO A UNIFORME ===
        uniform_p = 0.5
        D_kl = 0.0
        for p in [p_success, p_error]:
            if p > 0:
                D_kl += p * math.log2(p / uniform_p)

        # === TASA DE PRODUCCIÓN DE ENTROPÍA ===
        processing_time_safe = max(processing_time, 1e-6)
        entropy_rate = H_shannon / processing_time_safe

        # === INFORMACIÓN MUTUA TEMPORAL ===
        # Aproximación: cuánta información aporta el tiempo sobre el resultado
        # Usando historial de entropías
        mutual_info_temporal = 0.0
        if len(self._entropy_history) >= 2:
            prev_entropy = self._entropy_history[-1].get("shannon_entropy", H_shannon)
            # Cambio en entropía normalizado
            mutual_info_temporal = abs(H_shannon - prev_entropy) / max(
                H_shannon, prev_entropy, 0.01
            )

        # === DIAGNÓSTICO TERMODINÁMICO ===
        max_entropy = 1.0  # log₂(2) para sistema binario
        entropy_ratio = H_shannon / max_entropy

        # Detectar "muerte térmica" (máxima incertidumbre + tasa de error alta)
        is_thermal_death = entropy_ratio > 0.95 and error_count > total_records * 0.4

        result = {
            "shannon_entropy": H_shannon,
            "shannon_entropy_corrected": H_shannon_corrected,
            "renyi_entropy_2": H_renyi_2,
            "tsallis_entropy": H_tsallis,
            "kl_divergence": D_kl,
            "entropy_rate": entropy_rate,
            "entropy_ratio": entropy_ratio,
            "mutual_info_temporal": mutual_info_temporal,
            "max_entropy": max_entropy,
            "is_thermal_death": is_thermal_death,
            # Alias para compatibilidad
            "entropy_absolute": H_shannon,
            "configurational_entropy": H_renyi_2,  # Usar Rényi como configuracional
        }

        self._entropy_history.append(
            {
                **result,
                "timestamp": time.time(),
                "total_records": total_records,
                "error_rate": error_count / total_records,
            }
        )

        return result

    def _get_zero_entropy(self) -> Dict[str, float]:
        """Retorna entropía cero para casos triviales."""
        return {
            "shannon_entropy": 0.0,
            "shannon_entropy_corrected": 0.0,
            "tsallis_entropy": 0.0,
            "kl_divergence": 0.0,
            "entropy_absolute": 0.0,
            "configurational_entropy": 0.0,
            "entropy_rate": 0.0,
            "entropy_decay_time": 0.0,
            "max_entropy": 1.0,
            "entropy_ratio": 0.0,
            "is_thermal_death": False,
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
        except Exception as e:
            raise ConfigurationError(f"Error inicializando componentes: {e}")

        self._stats = ProcessingStats()
        self._start_time: Optional[float] = None
        self._emergency_brake_count: int = 0

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
            parser = self._initialize_parser(validated_path)
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

        # Verificar extensión
        if path.suffix.lower() not in SystemConstants.VALID_FILE_EXTENSIONS:
            raise InvalidInputError(
                f"Extensión no soportada: {path.suffix}. "
                f"Válidas: {SystemConstants.VALID_FILE_EXTENSIONS}"
            )

        # Verificar tamaño
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

    def _initialize_parser(self, path: Path) -> ReportParserCrudo:
        """Inicializa el parser con manejo de errores."""
        try:
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
        Procesamiento con control PID mejorado:
        1. Predicción anticipativa con Kalman
        2. Control feedforward basado en complejidad estimada
        3. Detección de régimen estacionario para optimización
        """
        processed_batches = []
        failed_batches_count = 0
        current_index = 0
        current_batch_size = self.condenser_config.min_batch_size
        iteration = 0
        max_iterations = total_records * SystemConstants.MAX_ITERATIONS_MULTIPLIER

        # Historiales para análisis
        saturation_history = []
        batch_size_history = []

        # Detección de régimen estacionario
        steady_state_counter = 0
        steady_state_threshold = 5

        # Control feedforward
        last_complexity = 0.5

        while current_index < total_records and iteration < max_iterations:
            iteration += 1

            # Determinar rango del batch
            end_index = min(current_index + current_batch_size, total_records)
            batch = raw_records[current_index:end_index]
            batch_size = len(batch)

            if batch_size == 0:
                break

            elapsed_time = time.time() - self._start_time

            # === VERIFICACIÓN DE TIMEOUT ===
            time_remaining = SystemConstants.PROCESSING_TIMEOUT - elapsed_time
            if time_remaining <= 0:
                self.logger.error("⏰ Timeout de procesamiento alcanzado")
                break

            # === ESTIMACIÓN DE CACHE HITS ===
            cache_hits_est = self._estimate_cache_hits(batch, cache)

            # === MÉTRICAS FÍSICAS ===
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

            # === PREDICCIÓN ANTICIPATIVA ===
            saturation_history.append(saturation)
            if len(saturation_history) >= 3:
                predicted_sat = self._predict_next_saturation(saturation_history)
            else:
                predicted_sat = saturation

            # === CONTROL FEEDFORWARD ===
            # Ajuste anticipativo basado en cambio de complejidad
            complexity_delta = complexity - last_complexity
            feedforward_adjustment = 1.0

            if complexity_delta > 0.1:
                # Complejidad aumentando: reducir batch preventivamente
                feedforward_adjustment = 0.85
            elif complexity_delta < -0.1:
                # Complejidad disminuyendo: podemos aumentar batch
                feedforward_adjustment = 1.1

            last_complexity = complexity

            # === DETECCIÓN DE RÉGIMEN ESTACIONARIO ===
            if len(saturation_history) >= 3:
                recent_var = sum((s - saturation) ** 2 for s in saturation_history[-3:]) / 3

                if recent_var < 0.01:  # Baja varianza
                    steady_state_counter += 1
                else:
                    steady_state_counter = 0

            in_steady_state = steady_state_counter >= steady_state_threshold

            # === CALLBACK DE MÉTRICAS ===
            if progress_callback:
                try:
                    progress_callback(
                        {
                            **metrics,
                            "predicted_saturation": predicted_sat,
                            "in_steady_state": in_steady_state,
                            "feedforward_adjustment": feedforward_adjustment,
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Error en progress_callback: {e}")

            # === CONTROL PID CON AJUSTES ===
            # Usar saturación efectiva considerando estabilidad giroscópica
            if gyro_stability < 0.5:
                effective_saturation = min(saturation + 0.2, 0.9)
                self.logger.debug(
                    f"Baja estabilidad giroscópica ({gyro_stability:.2f}): "
                    f"inflando saturación {saturation:.2f} → {effective_saturation:.2f}"
                )
            else:
                effective_saturation = saturation

            pid_output = self.controller.compute(effective_saturation)

            # Aplicar feedforward
            pid_output = int(pid_output * feedforward_adjustment)

            # === FRENOS DE EMERGENCIA ===
            emergency_brake = False
            brake_reason = ""

            if power > SystemConstants.OVERHEAT_POWER_THRESHOLD:
                brake_factor = 0.3
                pid_output = max(
                    SystemConstants.MIN_BATCH_SIZE_FLOOR, int(pid_output * brake_factor)
                )
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

            # === PROCESAMIENTO DEL BATCH ===
            result = self._process_single_batch_with_recovery(
                batch, cache, failed_batches_count
            )

            # Actualizar estadísticas
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
                        raise ProcessingError(
                            f"Límite de batches fallidos: {failed_batches_count}"
                        )
                    # Recuperación extrema
                    pid_output = SystemConstants.MIN_BATCH_SIZE_FLOOR
                    self.logger.warning("Activando recuperación extrema")

            # === CALLBACK DE PROGRESO ===
            if on_progress:
                try:
                    on_progress(self._stats)
                except Exception as e:
                    self.logger.warning(f"Error en on_progress: {e}")

            # === TELEMETRÍA ===
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

            # === ACTUALIZAR PARA SIGUIENTE ITERACIÓN ===
            current_index = end_index
            batch_size_history.append(current_batch_size)

            # Suavizado inercial del tamaño de batch
            # Mayor inercia en estado estacionario
            inertia = 0.8 if in_steady_state else 0.6
            current_batch_size = int(
                inertia * current_batch_size + (1 - inertia) * pid_output
            )

            # Aplicar límites
            current_batch_size = max(
                SystemConstants.MIN_BATCH_SIZE_FLOOR,
                min(current_batch_size, self.condenser_config.max_batch_size),
            )

        return processed_batches

    def _estimate_cache_hits(self, batch: List, cache: Dict) -> int:
        """
        Estimación probabilística de cache hits usando modelo Bayesiano
        con prior basado en historial de hits anteriores.
        """
        if not batch:
            return 0

        if not cache:
            # Sin caché: asumir miss rate alto
            return max(1, len(batch) // 4)

        # === PRIOR BASADO EN HISTORIAL ===
        if not hasattr(self, "_cache_hit_history"):
            self._cache_hit_history = deque(maxlen=50)

        if self._cache_hit_history:
            # Media móvil del hit rate histórico como prior
            prior_hit_rate = sum(self._cache_hit_history) / len(self._cache_hit_history)
        else:
            prior_hit_rate = 0.5  # Prior no informativo

        # === LIKELIHOOD POR MUESTREO ===
        sample_size = min(50, len(batch))
        sample_indices = range(0, len(batch), max(1, len(batch) // sample_size))

        cache_field_set = set(cache.keys())
        sample_hits = 0

        for idx in sample_indices:
            if idx < len(batch):
                record = batch[idx]
                if isinstance(record, dict):
                    # Hit si hay intersección significativa con campos de caché
                    record_fields = set(record.keys())
                    overlap = len(record_fields & cache_field_set)
                    total_fields = len(record_fields | cache_field_set)

                    if total_fields > 0:
                        # Jaccard similarity como proxy de hit
                        jaccard = overlap / total_fields
                        if jaccard > 0.3:  # Umbral de similitud
                            sample_hits += 1

        # Likelihood del sample
        actual_sample_size = len(list(sample_indices))
        if actual_sample_size > 0:
            sample_hit_rate = sample_hits / actual_sample_size
        else:
            sample_hit_rate = prior_hit_rate

        # === POSTERIOR BAYESIANO ===
        # Combinar prior y likelihood con pesos
        prior_weight = min(len(self._cache_hit_history) / 20, 0.5)  # Max 50% prior
        posterior_hit_rate = (
            prior_weight * prior_hit_rate + (1 - prior_weight) * sample_hit_rate
        )

        # Calcular hits estimados
        estimated_hits = int(posterior_hit_rate * len(batch))

        # Actualizar historial
        self._cache_hit_history.append(sample_hit_rate)

        return max(1, estimated_hits)

    def _predict_next_saturation(self, history: List[float]) -> float:
        """
        Predicción de saturación usando filtro de Kalman simplificado
        para mejor estimación en presencia de ruido.
        """
        if len(history) < 2:
            return history[-1] if history else 0.5

        # === INICIALIZACIÓN DEL FILTRO ===
        if not hasattr(self, "_kalman_state"):
            self._kalman_state = {
                "x": history[-1],  # Estado estimado
                "P": 1.0,  # Covarianza del error
                "Q": 0.01,  # Ruido del proceso
                "R": 0.1,  # Ruido de medición
            }

        ks = self._kalman_state

        # === PREDICCIÓN ===
        # Modelo: x(k+1) = x(k) + v(k) donde v(k) es tendencia
        # Estimar tendencia de las últimas muestras
        n = min(5, len(history))
        if n >= 2:
            trend = (history[-1] - history[-n]) / (n - 1)
        else:
            trend = 0.0

        # Limitar tendencia para estabilidad
        trend = max(-0.2, min(0.2, trend))

        x_pred = ks["x"] + trend
        P_pred = ks["P"] + ks["Q"]

        # === ACTUALIZACIÓN (Corrección) ===
        z = history[-1]  # Medición actual

        # Ganancia de Kalman
        K = P_pred / (P_pred + ks["R"])

        # Actualizar estado
        x_new = x_pred + K * (z - x_pred)
        P_new = (1 - K) * P_pred

        # Guardar estado
        ks["x"] = x_new
        ks["P"] = P_new

        # Adaptar ruido del proceso basado en error de predicción
        prediction_error = abs(z - x_pred)
        ks["Q"] = 0.9 * ks["Q"] + 0.1 * prediction_error**2

        # Predicción del próximo valor
        next_prediction = x_new + trend

        # Saturar al rango válido
        return max(0.0, min(1.0, next_prediction))

    def _process_single_batch_with_recovery(
        self, batch: List, cache: Dict, consecutive_failures: int
    ) -> BatchResult:
        """
        Procesamiento con estrategia de recuperación multinivel:
        1. Intento normal
        2. División binaria recursiva
        3. Procesamiento elemento por elemento
        4. Skip con logging
        """
        if not batch:
            return BatchResult(success=True, records_processed=0, dataframe=pd.DataFrame())

        batch_size = len(batch)

        # === NIVEL 0: INTENTO NORMAL ===
        if consecutive_failures == 0:
            try:
                parsed_data = ParsedData(batch, cache)
                df = self._rectify_signal(parsed_data)

                if df is None:
                    df = pd.DataFrame()

                return BatchResult(success=True, dataframe=df, records_processed=len(df))
            except Exception as e:
                self.logger.debug(f"Intento normal falló: {e}")
                # Continuar a recuperación

        # === NIVEL 1: DIVISIÓN BINARIA ===
        if consecutive_failures <= 2 and batch_size > 10:
            try:
                mid = batch_size // 2
                result_left = self._process_single_batch_with_recovery(
                    batch[:mid], cache, consecutive_failures + 1
                )
                result_right = self._process_single_batch_with_recovery(
                    batch[mid:], cache, consecutive_failures + 1
                )

                dfs = []
                records = 0

                if result_left.success and result_left.dataframe is not None:
                    dfs.append(result_left.dataframe)
                    records += result_left.records_processed

                if result_right.success and result_right.dataframe is not None:
                    dfs.append(result_right.dataframe)
                    records += result_right.records_processed

                if dfs:
                    combined = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
                    return BatchResult(
                        success=True, dataframe=combined, records_processed=records
                    )
            except Exception as e:
                self.logger.debug(f"División binaria falló: {e}")

        # === NIVEL 2: PROCESAMIENTO INDIVIDUAL ===
        if batch_size <= 100:
            successful_records = []

            for i, record in enumerate(batch):
                try:
                    parsed_single = ParsedData([record], cache)
                    df_single = self._rectify_signal(parsed_single)

                    if df_single is not None and not df_single.empty:
                        successful_records.append(df_single)
                except Exception:
                    # Skip silencioso en modo recuperación
                    continue

            if successful_records:
                combined = pd.concat(successful_records, ignore_index=True)
                return BatchResult(
                    success=True,
                    dataframe=combined,
                    records_processed=len(combined),
                    error_message=f"Recuperación parcial: {len(combined)}/{batch_size}",
                )

        # === NIVEL 3: FALLO TOTAL ===
        return BatchResult(
            success=False,
            error_message=f"Recuperación fallida para batch de {batch_size} registros",
            records_processed=0,
        )

    def _rectify_signal(self, parsed_data: ParsedData) -> pd.DataFrame:
        """Convierte datos crudos a DataFrame mediante APUProcessor."""
        try:
            processor = APUProcessor(self.config, self.profile, parsed_data.parse_cache)
            processor.raw_records = parsed_data.raw_records
            return processor.process_all()
        except Exception as e:
            raise ProcessingError(f"Error en rectificación: {e}")

    def _consolidate_results(self, batches: List[pd.DataFrame]) -> pd.DataFrame:
        """Consolida resultados con verificación de integridad."""
        if not batches:
            return pd.DataFrame()

        # Filtrar DataFrames vacíos
        valid_batches = [df for df in batches if df is not None and not df.empty]

        if not valid_batches:
            return pd.DataFrame()

        # Limitar número de batches para evitar memoria excesiva
        if len(valid_batches) > SystemConstants.MAX_BATCHES_TO_CONSOLIDATE:
            self.logger.warning(
                f"Truncando batches: {len(valid_batches)} > "
                f"{SystemConstants.MAX_BATCHES_TO_CONSOLIDATE}"
            )
            valid_batches = valid_batches[: SystemConstants.MAX_BATCHES_TO_CONSOLIDATE]

        try:
            return pd.concat(valid_batches, ignore_index=True)
        except Exception as e:
            raise ProcessingError(f"Error consolidando resultados: {e}")

    def _validate_output(self, df: pd.DataFrame) -> None:
        """Valida el DataFrame de salida."""
        if df.empty:
            self.logger.warning("DataFrame de salida vacío")
            return

        # Verificar registros mínimos si está configurado
        if len(df) < self.condenser_config.min_records_threshold:
            msg = (
                f"Registros insuficientes: {len(df)} < "
                f"{self.condenser_config.min_records_threshold}"
            )
            if self.condenser_config.enable_strict_validation:
                raise ProcessingError(msg)
            else:
                self.logger.warning(msg)

    def _enhance_stats_with_diagnostics(
        self, stats: ProcessingStats, metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Mejora las estadísticas con diagnóstico del sistema.
        """
        base_stats = asdict(stats)

        # Calcular eficiencia
        if stats.total_records > 0:
            efficiency = stats.processed_records / stats.total_records
        else:
            efficiency = 0.0

        # Diagnóstico del sistema
        system_health = self.get_system_health()
        physics_diagnosis = self.physics.get_system_diagnosis(metrics)

        enhanced = {
            **base_stats,
            "efficiency": efficiency,
            "throughput": stats.processed_records / max(stats.processing_time, 0.001),
            "system_health": system_health,
            "physics_diagnosis": physics_diagnosis,
            "current_metrics": {
                "saturation": metrics.get("saturation", 0),
                "complexity": metrics.get("complexity", 0),
                "gyroscopic_stability": metrics.get("gyroscopic_stability", 1.0),
                "entropy_ratio": metrics.get("entropy_ratio", 0),
            },
        }

        return enhanced

    def get_processing_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas completas de procesamiento."""
        return {
            "statistics": asdict(self._stats),
            "controller": self.controller.get_diagnostics(),
            "physics": self.physics.get_trend_analysis(),
            "emergency_brakes": self._emergency_brake_count,
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Retorna diagnóstico de salud del sistema."""
        controller_diag = self.controller.get_stability_analysis()
        physics_trend = self.physics.get_trend_analysis()

        # Determinar estado general
        health = "HEALTHY"
        issues = []

        if controller_diag.get("stability_class") == "POTENTIALLY_UNSTABLE":
            health = "DEGRADED"
            issues.append("Controlador potencialmente inestable")

        if self._emergency_brake_count > 5:
            health = "DEGRADED"
            issues.append(f"Múltiples frenos de emergencia: {self._emergency_brake_count}")

        if self._stats.failed_batches > self._stats.total_batches * 0.1:
            health = "DEGRADED"
            issues.append(
                f"Alta tasa de fallos: {self._stats.failed_batches}/{self._stats.total_batches}"
            )

        return {
            "health": health,
            "issues": issues,
            "controller_stability": controller_diag.get("stability_class", "UNKNOWN"),
            "processing_efficiency": (
                self._stats.processed_records / max(1, self._stats.total_records)
            ),
            "uptime": time.time() - self._start_time if self._start_time else 0,
        }
