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
from dataclasses import dataclass, asdict
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
    GYRO_EMA_ALPHA: float = 0.1    # Alpha para filtro EMA de corriente


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
    Controlador PI Discreto con:
    1. Filtro de media móvil exponencial para estabilización
    2. Anti-windup con back-calculation
    3. Análisis de estabilidad basado en Lyapunov discreto
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

    def _validate_control_parameters(
        self, kp: float, ki: float, setpoint: float,
        min_output: int, max_output: int
    ) -> None:
        """Validación de parámetros con criterios de estabilidad."""
        errors = []

        if kp <= 0:
            errors.append(f"Kp debe ser positivo para respuesta proporcional, got {kp}")
        if ki < 0:
            errors.append(f"Ki debe ser no-negativo, got {ki}")
        if min_output >= max_output:
            errors.append(
                f"Rango de salida inválido: [{min_output}, {max_output}]"
            )
        if min_output <= 0:
            errors.append(f"min_output debe ser positivo, got {min_output}")
        if not (0.0 < setpoint < 1.0):
            errors.append(f"setpoint debe estar en (0, 1), got {setpoint}")

        # Criterio de estabilidad para PI discreto:
        # Para evitar oscilaciones, la ganancia total no debe ser excesiva
        if kp > 0 and ki > 0:
            # Heurística: ratio Ki/Kp muy alto causa windup
            if ki / kp > 1.0:
                logger.warning(
                    f"Ratio Ki/Kp = {ki/kp:.2f} > 1.0 puede causar respuesta lenta"
                )

        if errors:
            raise ConfigurationError(
                "Errores en parámetros de control:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )

    def _apply_ema_filter(self, measurement: float) -> float:
        """Filtro de media móvil exponencial para suavizado de señal."""
        if self._filtered_pv is None:
            self._filtered_pv = measurement
        else:
            self._filtered_pv = (
                self._ema_alpha * measurement +
                (1 - self._ema_alpha) * self._filtered_pv
            )
        return self._filtered_pv

    def _update_lyapunov_metric(self, error: float) -> None:
        """
        Actualiza métrica de estabilidad de Lyapunov.

        Usa función de Lyapunov V(e) = e² y verifica que ΔV < 0
        para estabilidad asintótica.
        """
        if self._last_error is not None:
            V_current = error ** 2
            V_previous = self._last_error ** 2

            if V_previous > SystemConstants.MIN_ENERGY_THRESHOLD:
                # Tasa de cambio logarítmica (aproximación al exponente de Lyapunov)
                delta_V = V_current - V_previous
                if V_previous > 0:
                    lyapunov_rate = delta_V / V_previous
                    self._lyapunov_sum += lyapunov_rate
                    self._lyapunov_count += 1

    def compute(self, process_variable: float) -> int:
        """
        Calcula salida de control PI con anti-windup por back-calculation.

        Args:
            process_variable: Valor actual del proceso (0.0 - 1.0)

        Returns:
            Salida de control acotada al rango [min_output, max_output]
        """
        self._iteration_count += 1
        current_time = time.time()

        # Filtrado de señal
        filtered_pv = self._apply_ema_filter(
            max(0.0, min(1.0, process_variable))
        )

        # Cálculo de error
        error = self.setpoint - filtered_pv
        self._error_history.append(error)

        # Cálculo de delta tiempo
        if self._last_time is None:
            dt = SystemConstants.MIN_DELTA_TIME
        else:
            dt = max(
                SystemConstants.MIN_DELTA_TIME,
                min(current_time - self._last_time, SystemConstants.MAX_DELTA_TIME)
            )

        # Término proporcional
        P = self.Kp * error

        # Término integral con anti-windup
        integral_increment = error * dt
        proposed_integral = self._integral_error + integral_increment

        # Pre-cálculo de salida para anti-windup
        I_proposed = self.Ki * proposed_integral
        output_unbounded = self._output_center + P + I_proposed

        # Anti-windup por back-calculation
        if output_unbounded > self.max_output:
            # Saturación superior: limitar integral
            saturation_error = output_unbounded - self.max_output
            self._integral_error = proposed_integral - saturation_error / max(self.Ki, 1e-6)
        elif output_unbounded < self.min_output:
            # Saturación inferior: limitar integral
            saturation_error = self.min_output - output_unbounded
            self._integral_error = proposed_integral + saturation_error / max(self.Ki, 1e-6)
        else:
            self._integral_error = proposed_integral

        # Aplicar límite absoluto de integral
        self._integral_error = max(
            -self._integral_limit,
            min(self._integral_limit, self._integral_error)
        )

        # Cálculo final de salida
        I = self.Ki * self._integral_error
        output_raw = self._output_center + P + I
        output = int(round(max(self.min_output, min(self.max_output, output_raw))))

        # Actualizar métricas de estabilidad
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
            return {
                "status": "INSUFFICIENT_DATA",
                "samples": len(self._error_history)
            }

        errors = list(self._error_history)
        lyapunov = self.get_lyapunov_exponent()

        # Análisis de convergencia
        error_variance = sum((e - sum(errors)/len(errors))**2 for e in errors) / len(errors)
        recent_errors = errors[-min(10, len(errors)):]
        recent_variance = sum(
            (e - sum(recent_errors)/len(recent_errors))**2 for e in recent_errors
        ) / len(recent_errors)

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
            "iterations": self._iteration_count
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
                "last_output": self._last_output
            },
            "stability_analysis": stability,
            "parameters": {
                "Kp": self.Kp,
                "Ki": self.Ki,
                "setpoint": self.setpoint,
                "output_range": [self.min_output, self.max_output]
            }
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
        # Preserva historial para análisis post-mortem

    def get_state(self) -> Dict[str, Any]:
        """Retorna estado serializable del controlador."""
        return {
            "parameters": {
                "Kp": self.Kp,
                "Ki": self.Ki,
                "setpoint": self.setpoint,
                "min_output": self.min_output,
                "max_output": self.max_output
            },
            "state": {
                "integral_error": self._integral_error,
                "filtered_pv": self._filtered_pv,
                "iteration": self._iteration_count
            },
            "diagnostics": self.get_stability_analysis()
        }


# ============================================================================
# MOTOR DE FÍSICA - MÉTODOS REFINADOS
# ============================================================================
class FluxPhysicsEngine:
    """
    Motor de física RLC con:
    1. Modelo de circuito RLC serie coherente
    2. Cálculo correcto de números de Betti para grafos
    3. Entropía termodinámica basada en estados accesibles
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
        self._Q = math.sqrt(self.L / self.C) / self.R if self.R > 0 else float('inf')

        # Clasificación del sistema
        if self._zeta > 1.0:
            self._damping_type = "OVERDAMPED"
            self._omega_d = self._omega_0 * math.sqrt(self._zeta**2 - 1)
        elif self._zeta < 1.0:
            self._damping_type = "UNDERDAMPED"
            self._omega_d = self._omega_0 * math.sqrt(1 - self._zeta**2)
        else:
            self._damping_type = "CRITICALLY_DAMPED"
            self._omega_d = 0.0

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
                self.logger.warning(
                    f"Constante de tiempo {tau:.2e} s muy pequeña"
                )

        if errors:
            raise ConfigurationError(
                "Parámetros físicos inválidos:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )

    def _evolve_state(self, driving_current: float, dt: float) -> Tuple[float, float]:
        """
        Evoluciona el estado del sistema RLC usando integración semi-implícita.

        El sistema está descrito por:
        L(dI/dt) + RI + Q/C = V(t)
        dQ/dt = I

        Args:
            driving_current: Corriente de entrada normalizada
            dt: Paso de tiempo

        Returns:
            Tuple (carga, corriente) actualizados
        """
        Q, I = self._state

        # Método de Euler semi-implícito para estabilidad
        # Predictor
        dI_dt = (driving_current - self.R * I - Q / self.C) / self.L
        I_pred = I + dt * dI_dt

        # Corrector para Q
        Q_new = Q + dt * (I + I_pred) / 2.0

        # Corrector para I
        dI_dt_new = (driving_current - self.R * I_pred - Q_new / self.C) / self.L
        I_new = I + dt * (dI_dt + dI_dt_new) / 2.0

        # Actualizar estado
        self._state = [Q_new, I_new]
        self._state_history.append({
            'Q': Q_new,
            'I': I_new,
            'time': time.time()
        })

        return Q_new, I_new

    def _build_metric_graph(self, metrics: Dict[str, float]) -> None:
        """
        Construye grafo de correlación entre métricas para análisis topológico.

        Vertices: métricas individuales
        Edges: correlaciones fuertes entre métricas (|corr| > threshold)
        """
        # Seleccionar métricas clave para el grafo
        metric_keys = ['saturation', 'complexity', 'current_I', 'potential_energy']
        values = [metrics.get(k, 0.0) for k in metric_keys]

        self._adjacency_list.clear()
        self._vertex_count = len(values)
        self._edge_count = 0

        # Inicializar listas de adyacencia
        for i in range(self._vertex_count):
            self._adjacency_list[i] = set()

        # Crear aristas basadas en proximidad (umbral adaptativo)
        if len(values) > 1:
            value_range = max(values) - min(values) if max(values) != min(values) else 1.0
            threshold = 0.3 * value_range

            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    if abs(values[i] - values[j]) < threshold:
                        self._adjacency_list[i].add(j)
                        self._adjacency_list[j].add(i)
                        self._edge_count += 1

    def _calculate_betti_numbers(self) -> Dict[int, int]:
        """
        Calcula números de Betti para el grafo de métricas.

        Para un grafo:
        - β₀ = número de componentes conexas
        - β₁ = número de ciclos independientes = E - V + β₀ (fórmula de Euler)
        """
        if self._vertex_count == 0:
            return {0: 0, 1: 0}

        # Calcular componentes conexas con BFS
        visited = set()
        components = 0

        for start in range(self._vertex_count):
            if start not in visited:
                # BFS desde este vértice
                queue = deque([start])
                while queue:
                    v = queue.popleft()
                    if v not in visited:
                        visited.add(v)
                        for neighbor in self._adjacency_list.get(v, set()):
                            if neighbor not in visited:
                                queue.append(neighbor)
                components += 1

        beta_0 = components

        # β₁ = E - V + β₀ (característica de Euler para grafos)
        beta_1 = max(0, self._edge_count - self._vertex_count + beta_0)

        return {0: beta_0, 1: beta_1}

    def calculate_gyroscopic_stability(self, current_I: float) -> float:
        """
        Calcula la Estabilidad Giroscópica (Sg) del sistema.

        La estabilidad se define como la resistencia del sistema a cambios
        bruscos en su "eje de rotación" (flujo de corriente).

        Fórmula: Sg = 1.0 - tanh(|I - EMA_I| * Sensitivity)
        """
        # Inicializar EMA si es el primer valor
        if not self._initialized and self._ema_current == 0.0:
            self._ema_current = current_I

        # Actualizar EMA (Eje de Rotación)
        alpha = SystemConstants.GYRO_EMA_ALPHA
        self._ema_current = alpha * current_I + (1 - alpha) * self._ema_current

        # Calcular perturbación (Precesión/Wobble)
        perturbation = abs(current_I - self._ema_current)

        # Calcular Estabilidad Giroscópica (Sg)
        # tanh mapeta [0, inf) a [0, 1]
        # Si perturbación es 0, tanh(0) = 0, Sg = 1.0 (Estable)
        # Si perturbación es alta, tanh -> 1, Sg -> 0.0 (Inestable)
        sensitivity = SystemConstants.GYRO_SENSITIVITY
        sg = 1.0 - math.tanh(perturbation * sensitivity)

        return max(0.0, min(1.0, sg))

    def calculate_system_entropy(
        self,
        total_records: int,
        error_count: int,
        processing_time: float
    ) -> Dict[str, float]:
        """
        Calcula entropía del sistema con fundamento termodinámico.

        1. Entropía de Shannon: incertidumbre en distribución de errores
        2. Entropía configuracional: basada en microestados accesibles
        3. Entropía de tasa: producción de entropía por unidad de tiempo
        """
        if total_records <= 0:
            return self._get_zero_entropy()

        success_count = total_records - error_count

        # Entropía de Shannon (bits)
        shannon_entropy = 0.0
        for count in [success_count, error_count]:
            if count > 0:
                p = count / total_records
                shannon_entropy -= p * math.log2(p)

        # Entropía configuracional (Boltzmann)
        # Ω = número de microestados = C(total, errors)
        # Para evitar overflow, usamos aproximación de Stirling
        if 0 < error_count < total_records:
            # S = k_B * ln(Ω) ≈ k_B * [N*ln(N) - n*ln(n) - (N-n)*ln(N-n)]
            n = total_records
            k = error_count
            try:
                log_omega = (
                    n * math.log(n) -
                    k * math.log(k) -
                    (n - k) * math.log(n - k)
                )
                configurational_entropy = log_omega
            except (ValueError, ZeroDivisionError):
                configurational_entropy = 0.0
        else:
            configurational_entropy = 0.0

        # Entropía de tasa (producción por segundo)
        entropy_rate = shannon_entropy / max(processing_time, 0.001)

        # Entropía máxima posible (Binaria = 1.0 bit)
        max_entropy = 1.0

        # Diagnóstico de "muerte térmica" (máximo desorden)
        # H(p) > 0.8 indica alta incertidumbre/caos en el proceso
        entropy_ratio = shannon_entropy / max_entropy
        is_thermal_death = entropy_ratio > 0.8

        result = {
            "shannon_entropy": shannon_entropy,
            "entropy_absolute": shannon_entropy,  # Alias para compatibilidad de pruebas
            "configurational_entropy": configurational_entropy,
            "entropy_rate": entropy_rate,
            "max_entropy": max_entropy,
            "entropy_ratio": entropy_ratio,
            "is_thermal_death": is_thermal_death
        }

        self._entropy_history.append({
            **result,
            'timestamp': time.time()
        })

        return result

    def _get_zero_entropy(self) -> Dict[str, float]:
        """Retorna entropía cero para casos triviales."""
        return {
            "shannon_entropy": 0.0,
            "entropy_absolute": 0.0,
            "configurational_entropy": 0.0,
            "entropy_rate": 0.0,
            "max_entropy": 1.0,
            "entropy_ratio": 0.0,
            "is_thermal_death": False
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
        - Corriente I = eficiencia (cache_hits / total_records)
        - Carga Q = registros acumulados procesados
        - Voltaje V = "presión" del pipeline (saturación)
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

        # Evolución del estado
        if self._initialized:
            dt = max(1e-6, current_time - self._last_time)
        else:
            dt = 0.01
            self._initialized = True

        Q, I = self._evolve_state(current_I, dt)

        # Constante de tiempo normalizada
        tau = self.L / R_dynamic if R_dynamic > 0 else float('inf')
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
            sin_term = (zeta_dynamic / math.sqrt(1 - zeta_dynamic**2)) * math.sin(omega_d * t_normalized)
            saturation = 1.0 - exp_term * (cos_term + sin_term)

        saturation = max(0.0, min(1.0, saturation))

        # Energías
        E_capacitor = 0.5 * self.C * (saturation ** 2)  # Energía potencial
        E_inductor = 0.5 * self.L * (current_I ** 2)    # Energía cinética

        # Potencia disipada
        P_dissipated = (current_I ** 2) * R_dynamic

        # Voltaje de flyback inductivo
        di_dt = (current_I - self._last_current) / max(dt, 1e-6)
        V_flyback = min(
            abs(self.L * di_dt),
            SystemConstants.MAX_FLYBACK_VOLTAGE
        )

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
            # Entropía
            "entropy_shannon": entropy_metrics["shannon_entropy"],
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
            "time_constant": self.L / self.R if self.R > 0 else float('inf'),
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
        self._metrics_history.append({
            **metrics,
            "_timestamp": time.time()
        })

    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analiza tendencias en métricas históricas."""
        if len(self._metrics_history) < 2:
            return {
                "status": "INSUFFICIENT_DATA",
                "samples": len(self._metrics_history)
            }

        result = {
            "status": "OK",
            "samples": len(self._metrics_history)
        }

        # Métricas a analizar
        keys_to_analyze = ["saturation", "dissipated_power", "entropy_ratio"]

        for key in keys_to_analyze:
            values = [m.get(key, 0.0) for m in self._metrics_history if key in m]
            if len(values) >= 2:
                # Tendencia lineal simple
                first_half = sum(values[:len(values)//2]) / (len(values)//2)
                second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)

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
                    "max": max(values)
                }

        return result

    def get_system_diagnosis(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Genera diagnóstico del estado del sistema."""
        diagnosis = {
            "state": "NORMAL",
            "damping": self._damping_type,
            "energy": "BALANCED",
            "entropy": "LOW"
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
            diagnosis["rotation_stability"] = "⚠️ PRECESIÓN DETECTADA (Inestabilidad de Flujo)"
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
            telemetry.record_event("stabilization_start", {
                "file": path_obj.name,
                "config": asdict(self.condenser_config)
            })

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
                telemetry.record_event("stabilization_complete", {
                    "records_processed": self._stats.processed_records,
                    "processing_time": self._stats.processing_time,
                    "emergency_brakes": self._emergency_brake_count
                })

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
        Procesa lotes con control PID adaptativo.

        Implementa:
        1. Control de tamaño de batch basado en saturación
        2. Freno de emergencia por sobrecalentamiento
        3. Recuperación parcial ante fallos
        """
        processed_batches = []
        failed_batches_count = 0
        current_index = 0
        current_batch_size = self.condenser_config.min_batch_size
        iteration = 0
        max_iterations = total_records * SystemConstants.MAX_ITERATIONS_MULTIPLIER

        while current_index < total_records and iteration < max_iterations:
            iteration += 1

            # Determinar rango del batch
            end_index = min(current_index + current_batch_size, total_records)
            batch = raw_records[current_index:end_index]
            batch_size = len(batch)

            if batch_size == 0:
                break

            # Calcular tiempo transcurrido
            elapsed_time = time.time() - self._start_time

            # Verificar timeout global
            if elapsed_time > SystemConstants.PROCESSING_TIMEOUT:
                self.logger.error("Timeout de procesamiento alcanzado")
                break

            # Calcular métricas físicas
            metrics = self.physics.calculate_metrics(
                total_records=batch_size,
                cache_hits=int(batch_size * 0.7),  # Estimación
                error_count=0,
                processing_time=elapsed_time
            )

            # Callback de métricas
            if progress_callback:
                try:
                    progress_callback(metrics)
                except Exception as e:
                    self.logger.warning(f"Error en progress_callback: {e}")

            # Control PID
            saturation = metrics.get("saturation", 0.5)
            pid_output = self.controller.compute(saturation)

            # Freno de emergencia por potencia
            power = metrics.get("dissipated_power", 0)
            if power > SystemConstants.OVERHEAT_POWER_THRESHOLD:
                self.logger.warning(
                    f"🔥 OVERHEAT: P={power:.2f}W > {SystemConstants.OVERHEAT_POWER_THRESHOLD}W"
                )
                pid_output = max(
                    SystemConstants.MIN_BATCH_SIZE_FLOOR,
                    int(pid_output * SystemConstants.EMERGENCY_BRAKE_FACTOR)
                )
                self._emergency_brake_count += 1
                self._stats.emergency_brakes_triggered += 1

            # Procesar batch - REMOVED OUTER TRY/EXCEPT to let explicit errors bubble up if needed,
            # OR better: catch expected errors but respect stop logic.
            # _process_single_batch guarantees BatchResult return (no raise).

            result = self._process_single_batch(batch, cache)

            if result.success and result.dataframe is not None:
                processed_batches.append(result.dataframe)
                self._stats.add_batch_stats(
                    batch_size=result.records_processed,
                    saturation=saturation,
                    power=power,
                    flyback=metrics.get("flyback_voltage", 0),
                    kinetic=metrics.get("kinetic_energy", 0),
                    success=True
                )
            else:
                failed_batches_count += 1
                self._stats.add_batch_stats(
                    batch_size=batch_size,
                    saturation=saturation,
                    power=power,
                    flyback=metrics.get("flyback_voltage", 0),
                    kinetic=metrics.get("kinetic_energy", 0),
                    success=False
                )
                self.logger.error(f"Error procesando batch {iteration}: {result.error_message}")

                # Verificar límite de fallos
                if failed_batches_count >= self.condenser_config.max_failed_batches:
                    if not self.condenser_config.enable_partial_recovery:
                        # This raises ProcessingError which will be caught by stabilize
                        raise ProcessingError(
                            f"Límite de batches fallidos alcanzado: {failed_batches_count}"
                        )
                    else:
                        self.logger.warning(
                            f"Continuando con recuperación parcial "
                            f"({failed_batches_count} batches fallidos)"
                        )


            # Callback de progreso
            if on_progress:
                try:
                    on_progress(self._stats)
                except Exception as e:
                    self.logger.warning(f"Error en on_progress: {e}")

            # Telemetría de batch
            if telemetry and iteration % 10 == 0:
                telemetry.record_event("batch_processed", {
                    "iteration": iteration,
                    "progress": current_index / total_records,
                    "batch_size": batch_size,
                    "saturation": saturation
                })

            # Avanzar índice y actualizar tamaño de batch
            current_index = end_index
            current_batch_size = max(
                SystemConstants.MIN_BATCH_SIZE_FLOOR,
                pid_output
            )

        return processed_batches

    def _process_single_batch(self, batch: List, cache: Dict) -> BatchResult:
        """Procesa un lote individual con manejo de errores estructurado."""
        if not batch:
            return BatchResult(success=False, error_message="Batch vacío")

        try:
            parsed_data = ParsedData(batch, cache)
            df = self._rectify_signal(parsed_data)

            if df is None or df.empty:
                return BatchResult(
                    success=True,
                    dataframe=pd.DataFrame(),
                    records_processed=0
                )

            return BatchResult(
                success=True,
                dataframe=df,
                records_processed=len(df)
            )

        except Exception as e:
            # self.logger.error(f"Error en batch: {e}") # Caller logs it
            return BatchResult(
                success=False,
                error_message=str(e)
            )

    def _rectify_signal(self, parsed_data: ParsedData) -> pd.DataFrame:
        """Convierte datos crudos a DataFrame mediante APUProcessor."""
        try:
            processor = APUProcessor(
                self.config,
                self.profile,
                parsed_data.parse_cache
            )
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
            valid_batches = valid_batches[:SystemConstants.MAX_BATCHES_TO_CONSOLIDATE]

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

    def get_processing_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas completas de procesamiento."""
        return {
            "statistics": asdict(self._stats),
            "controller": self.controller.get_diagnostics(),
            "physics": self.physics.get_trend_analysis(),
            "emergency_brakes": self._emergency_brake_count
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
            issues.append(f"Alta tasa de fallos: {self._stats.failed_batches}/{self._stats.total_batches}")

        return {
            "health": health,
            "issues": issues,
            "controller_stability": controller_diag.get("stability_class", "UNKNOWN"),
            "processing_efficiency": (
                self._stats.processed_records / max(1, self._stats.total_records)
            ),
            "uptime": time.time() - self._start_time if self._start_time else 0
        }
