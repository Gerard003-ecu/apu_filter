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
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

# Manejo opcional de numpy para optimización matemática
try:
    import numpy as np
except ImportError:
    pass

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
        raw_records (List[Dict[str, Any]]): Lista de registros de insumos
            extraídos del archivo de APU, sin procesamiento profundo.
        parse_cache (Dict[str, Any]): Metadatos generados durante el parseo,
            útiles para optimizar el procesamiento posterior (e.g., líneas
            ya validadas por Lark).
    """

    raw_records: List[Dict[str, Any]]
    parse_cache: Dict[str, Any]


@dataclass(frozen=True)
class CondenserConfig:
    """
    Configuración inmutable y validada para el `DataFluxCondenser`.

    Define los umbrales operativos y comportamientos del condensador,
    incluyendo sus parámetros para el motor de simulación física y el PID.

    Attributes:
        min_records_threshold (int): Número mínimo de registros necesarios para
            considerar un archivo como válido para el procesamiento.
        enable_strict_validation (bool): Si es `True`, activa validaciones
            adicionales en el DataFrame de salida.
        log_level (str): Nivel de logging para la instancia del condensador.
        system_capacitance (float): Parámetro físico RLC (Faradios).
        base_resistance (float): Parámetro físico RLC (Ohmios).
        system_inductance (float): Parámetro físico RLC (Henrios).
        pid_setpoint (float): Objetivo de saturación (0.0-1.0).
        pid_kp (float): Ganancia Proporcional del PID.
        pid_ki (float): Ganancia Integral del PID.
        min_batch_size (int): Tamaño mínimo del lote de procesamiento.
        max_batch_size (int): Tamaño máximo del lote de procesamiento.
        enable_partial_recovery (bool): Permite continuar procesamiento si falla un batch.
        max_failed_batches (int): Máximo de batches que pueden fallar antes de abortar.
        integral_limit_factor (float): Factor para limitar la parte integral (anti-windup).
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
    integral_limit_factor: float = 2.0  # Múltiplo del rango de salida

    def __post_init__(self):
        """Valida la configuración después de la inicialización."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Valida que todos los parámetros estén en rangos válidos."""
        errors = []

        # Validar threshold
        if self.min_records_threshold < 0:
            errors.append(
                f"min_records_threshold debe ser >= 0, recibido: {self.min_records_threshold}"
            )

        # Validar parámetros físicos
        if self.system_capacitance <= 0:
            errors.append(
                f"system_capacitance debe ser > 0, recibido: {self.system_capacitance}"
            )

        if self.base_resistance <= 0:
            errors.append(f"base_resistance debe ser > 0, recibido: {self.base_resistance}")

        if self.system_inductance <= 0:
            errors.append(
                f"system_inductance debe ser > 0, recibido: {self.system_inductance}"
            )

        # Validar PID
        if not 0.0 <= self.pid_setpoint <= 1.0:
            errors.append(
                f"pid_setpoint debe estar en [0.0, 1.0], recibido: {self.pid_setpoint}"
            )

        if self.pid_kp < 0:
            errors.append(f"pid_kp debe ser >= 0, recibido: {self.pid_kp}")

        if self.pid_ki < 0:
            errors.append(f"pid_ki debe ser >= 0, recibido: {self.pid_ki}")

        # Validar batch sizes
        if self.min_batch_size <= 0:
            errors.append(f"min_batch_size debe ser > 0, recibido: {self.min_batch_size}")

        if self.max_batch_size <= 0:
            errors.append(f"max_batch_size debe ser > 0, recibido: {self.max_batch_size}")

        if self.min_batch_size > self.max_batch_size:
            errors.append(
                f"min_batch_size ({self.min_batch_size}) no puede ser mayor que "
                f"max_batch_size ({self.max_batch_size})"
            )

        # Validar recuperación
        if self.max_failed_batches < 0:
            errors.append(
                f"max_failed_batches debe ser >= 0, recibido: {self.max_failed_batches}"
            )

        if self.integral_limit_factor <= 0:
            errors.append(
                f"integral_limit_factor debe ser > 0, recibido: {self.integral_limit_factor}"
            )

        # Validar log level
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            errors.append(
                f"log_level debe ser uno de {valid_log_levels}, recibido: {self.log_level}"
            )

        if errors:
            raise ConfigurationError(
                "Errores de configuración detectados:\n"
                + "\n".join(f"  - {e}" for e in errors)
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

        # Promedios móviles
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
# CONTROLADOR PI DISCRETO
# ============================================================================
class PIController:
    """
    Implementación robusta de un Controlador PI Discreto.

    ROBUSTECIDO:
    - Validación exhaustiva en cada cómputo
    - Manejo de casos límite (NaN, Inf, valores extremos)
    - Anti-windup con múltiples estrategias
    - Historial de estados para diagnóstico
    - Reset seguro con preservación de configuración
    """

    # Constantes de clase
    _MAX_HISTORY_SIZE: int = 100
    _WARMUP_ITERATIONS: int = 3

    def __init__(
        self,
        kp: float,
        ki: float,
        setpoint: float,
        min_output: int,
        max_output: int,
        integral_limit_factor: float = 2.0,
    ):
        """
        Inicializa el controlador PI con validación exhaustiva.

        Args:
            kp: Ganancia Proporcional.
            ki: Ganancia Integral.
            setpoint: Punto de ajuste (objetivo).
            min_output: Salida mínima permitida.
            max_output: Salida máxima permitida.
            integral_limit_factor: Factor para limitar el término integral.
        """
        self._validate_parameters(
            kp, ki, setpoint, min_output, max_output, integral_limit_factor
        )

        # Parámetros inmutables después de validación
        self.Kp = float(kp)
        self.Ki = float(ki)
        self.setpoint = float(setpoint)
        self.min_output = int(min_output)
        self.max_output = int(max_output)

        # Cálculos derivados con protección
        self._base_output = (self.max_output + self.min_output) / 2.0
        self._output_range = max(
            1, self.max_output - self.min_output
        )  # Evitar división por 0
        self._integral_limit = self._output_range * max(0.1, integral_limit_factor)

        # Estado interno
        self._integral_error: float = 0.0
        self._last_time: Optional[float] = None
        self._last_error: Optional[float] = None
        self._iteration_count: int = 0
        self._in_warmup: bool = True

        # Historial para diagnóstico (circular buffer)
        self._history: List[Dict[str, Any]] = []
        self._consecutive_saturations: int = 0

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.debug(
            f"PIController inicializado: Kp={self.Kp}, Ki={self.Ki}, "
            f"SP={self.setpoint}, Range=[{self.min_output}, {self.max_output}]"
        )

    def _validate_parameters(
        self,
        kp: float,
        ki: float,
        setpoint: float,
        min_output: int,
        max_output: int,
        integral_limit_factor: float,
    ) -> None:
        """Valida todos los parámetros del controlador con mensajes detallados."""
        errors = []

        # Validar tipos
        for name, value, expected_type in [
            ("kp", kp, (int, float)),
            ("ki", ki, (int, float)),
            ("setpoint", setpoint, (int, float)),
            ("min_output", min_output, (int, float)),
            ("max_output", max_output, (int, float)),
            ("integral_limit_factor", integral_limit_factor, (int, float)),
        ]:
            if not isinstance(value, expected_type):
                errors.append(f"{name} debe ser numérico, recibido: {type(value).__name__}")
            elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                errors.append(f"{name} no puede ser NaN o Inf: {value}")

        if errors:
            raise ConfigurationError(
                "Tipos inválidos en PIController:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Validar rangos
        if kp < 0:
            errors.append(f"Kp debe ser >= 0, recibido: {kp}")

        if ki < 0:
            errors.append(f"Ki debe ser >= 0, recibido: {ki}")

        if not 0.0 <= setpoint <= 1.0:
            errors.append(f"setpoint debe estar en [0.0, 1.0], recibido: {setpoint}")

        if min_output < SystemConstants.MIN_BATCH_SIZE_FLOOR:
            errors.append(
                f"min_output debe ser >= {SystemConstants.MIN_BATCH_SIZE_FLOOR}, "
                f"recibido: {min_output}"
            )

        if max_output <= min_output:
            errors.append(f"max_output ({max_output}) debe ser > min_output ({min_output})")

        if integral_limit_factor <= 0:
            errors.append(
                f"integral_limit_factor debe ser > 0, recibido: {integral_limit_factor}"
            )

        if errors:
            raise ConfigurationError(
                "Parámetros inválidos del PIController:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def compute(self, process_variable: float) -> int:
        """
        Calcula la nueva salida de control con mejoras:
        - Filtro de media móvil exponencial para ruido.
        - Derivada de error para detección de oscilaciones.
        - Reset adaptativo del integrador.
        """
        self._iteration_count += 1

        # ==================== FILTRADO DE ENTRADA ====================
        # Filtro EMA para suavizar ruido (α=0.3)
        if not hasattr(self, "_pv_filtered"):
            self._pv_filtered = process_variable
        else:
            alpha = 0.3
            self._pv_filtered = alpha * process_variable + (1 - alpha) * self._pv_filtered

        process_variable = self._pv_filtered

        # Validación reforzada
        if math.isnan(process_variable) or math.isinf(process_variable):
            self.logger.warning("PV inválido, usando setpoint con decaimiento")
            process_variable = self.setpoint * (0.9**self._iteration_count)

        # ==================== CÁLCULO DE ERROR CON HISTÉRESIS ====================
        error = self.setpoint - process_variable

        # Histéresis para evitar oscilaciones menores al 1%
        if hasattr(self, "_last_error") and self._last_error is not None:
            error_change = abs(error - self._last_error)
            if error_change < 0.01:  # 1% de umbral
                error = self._last_error  # Mantener error anterior

        # ==================== CONTROL DERIVATIVO DISCRETO ====================
        current_time = time.time()

        if self._last_time is None:
            dt = SystemConstants.MIN_DELTA_TIME
            error_derivative = 0.0
        else:
            dt = current_time - self._last_time
            dt = max(SystemConstants.MIN_DELTA_TIME, min(dt, SystemConstants.MAX_DELTA_TIME))

            # Cálculo de derivada con filtro (evita ruido)
            if self._last_error is not None:
                raw_derivative = (error - self._last_error) / dt
                # Filtro de primer orden para derivada
                if not hasattr(self, "_derivative_filtered"):
                    self._derivative_filtered = raw_derivative
                else:
                    self._derivative_filtered = (
                        0.7 * raw_derivative + 0.3 * self._derivative_filtered
                    )
                error_derivative = self._derivative_filtered
            else:
                error_derivative = 0.0

        # ==================== ANTI-WINDUP MEJORADO ====================
        # Reset condicional del integrador si el error persiste por mucho tiempo
        if abs(self._integral_error) > self._integral_limit * 0.8:
            if abs(error) < 0.05:  # Error pequeño pero integrador grande
                self._integral_error *= 0.5  # Reducir integrador gradualmente
                self.logger.debug(f"Integrator bleed: {self._integral_error:.2f}")

        # Acumulación condicional del error
        integral_gain = self.Ki
        if abs(error) > 0.2:  # Error grande
            integral_gain *= 0.5  # Reducir ganancia integral para evitar sobreajuste

        self._integral_error += error * dt * integral_gain / self.Ki if self.Ki > 0 else 0

        # Limitar integrador con función soft-clipping
        if abs(self._integral_error) > self._integral_limit:
            excess_ratio = abs(self._integral_error) / self._integral_limit
            # Soft clipping: tanh
            self._integral_error = math.copysign(
                self._integral_limit * math.tanh(excess_ratio), self._integral_error
            )

        # ==================== CÁLCULO DE SALIDA ====================
        P = self.Kp * error
        I = self.Ki * self._integral_error

        # Término derivativo opcional (Kd=0.1*Kp por defecto)
        Kd = self.Kp * 0.1
        D = -Kd * error_derivative  # Negativo para estabilizar

        control_signal = self._base_output + P + I + D

        # ==================== DETECCIÓN DE OSCILACIONES ====================
        if len(self._history) >= 3:
            last_outputs = [h["output"] for h in self._history[-3:]]
            if len(set(last_outputs)) == 1 and abs(error) > 0.1:
                # Salida estancada pero error persistente
                self._integral_error *= 1.2  # Impulso al integrador
                self.logger.debug("Stall detection - boosting integral")

        # ==================== SATURACIÓN Y REDONDEO INTELIGENTE ====================
        output = (
            int(np.round(control_signal))
            if "np" in globals()
            else int(round(control_signal))
        )

        # Evitar cambios bruscos (limitación de slew rate)
        if hasattr(self, "_last_output") and self._last_output is not None:
            max_change = max(50, self._output_range * 0.1)  # Máximo 10% del rango o 50
            output = self._last_output + max(
                -max_change, min(max_change, output - self._last_output)
            )

        output = max(self.min_output, min(self.max_output, output))
        self._last_output = output

        # ==================== ACTUALIZACIÓN DE ESTADO ====================
        self._last_time = current_time
        self._last_error = error

        # Historial para análisis básico
        history_entry = {
            "iteration": self._iteration_count,
            "pv": process_variable,
            "error": error,
            "error_derivative": error_derivative,
            "P": P,
            "I": I,
            "D": D,
            "output": output,
            "integral": self._integral_error,
            "dt": dt,
            "control_signal_raw": control_signal,
        }

        self._history.append(history_entry)
        if len(self._history) > self._MAX_HISTORY_SIZE:
            self._history.pop(0)

        # Logging adaptativo
        log_freq = 20 if abs(error) < 0.05 else 5
        if self._iteration_count % log_freq == 0 or self._iteration_count <= 5:
            self.logger.debug(
                f"[PID #{self._iteration_count}] "
                f"PV={process_variable:.3f} | E={error:+.3f} | "
                f"P={P:+.0f} I={I:+.0f} D={D:+.0f} | "
                f"Out={output} (Δ={output - (self._last_output if hasattr(self, '_last_output') else 0)})"
            )

        return output

    def reset(self) -> None:
        """Resetea el estado interno preservando la configuración."""
        self._integral_error = 0.0
        self._last_time = None
        self._last_error = None
        self._iteration_count = 0
        self._in_warmup = True
        self._consecutive_saturations = 0
        self._history.clear()

        # Reset variables adicionales de estado
        if hasattr(self, "_pv_filtered"):
            del self._pv_filtered
        if hasattr(self, "_derivative_filtered"):
            del self._derivative_filtered
        if hasattr(self, "_last_output"):
            del self._last_output

        self.logger.debug("[PID] Controlador reseteado")

    def get_state(self) -> Dict[str, Any]:
        """Retorna el estado completo del controlador para observabilidad."""
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
                "iteration_count": self._iteration_count,
                "last_time": self._last_time,
                "last_error": self._last_error,
                "in_warmup": self._in_warmup,
                "consecutive_saturations": self._consecutive_saturations,
            },
            "limits": {
                "integral_limit": self._integral_limit,
                "base_output": self._base_output,
            },
            "history_size": len(self._history),
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Retorna diagnósticos del controlador basados en el historial."""
        if not self._history:
            return {"status": "NO_DATA", "message": "Sin historial disponible"}

        recent = self._history[-min(20, len(self._history)) :]

        avg_error = sum(h["error"] for h in recent) / len(recent)
        avg_output = sum(h["output"] for h in recent) / len(recent)
        output_variance = sum((h["output"] - avg_output) ** 2 for h in recent) / len(recent)

        # Detectar oscilaciones
        sign_changes = 0
        for i in range(1, len(recent)):
            if recent[i]["error"] * recent[i - 1]["error"] < 0:
                sign_changes += 1

        oscillating = sign_changes > len(recent) * 0.4

        return {
            "status": "OK" if not oscillating else "OSCILLATING",
            "avg_error": avg_error,
            "avg_output": avg_output,
            "output_variance": output_variance,
            "sign_changes": sign_changes,
            "is_oscillating": oscillating,
            "recommendation": (
                "Reducir Kp si hay oscilaciones"
                if oscillating
                else "Sistema estable"
                if abs(avg_error) < 0.1
                else "Aumentar Ki si error persistente"
            ),
        }


# ============================================================================
# MOTOR DE FÍSICA AVANZADO
# ============================================================================
class FluxPhysicsEngine:
    """
    Simula el comportamiento físico RLC basándose en la ENERGÍA.

    ROBUSTECIDO:
    - Validación exhaustiva de entradas.
    - Protección contra overflow/underflow.
    - Métricas normalizadas y sanitizadas.
    - Diagnóstico mejorado con múltiples niveles.
    - Historial de métricas para análisis de tendencias.
    """

    _MAX_METRICS_HISTORY: int = 100

    def __init__(self, capacitance: float, resistance: float, inductance: float):
        """
        Inicializa el motor de física con validación exhaustiva.

        Args:
            capacitance: Capacitancia (Faradios).
            resistance: Resistencia (Ohmios).
            inductance: Inductancia (Henrios).
        """
        # Validación ANTES de asignaciones
        self._init_warnings: List[str] = []
        self._validate_parameters(capacitance, resistance, inductance)

        self.C = float(capacitance)
        self.R = float(resistance)
        self.L = float(inductance)

        # Constantes derivadas pre-calculadas con protección
        self._tau_base = self.R * self.C

        # Frecuencia resonante: f_0 = 1 / (2π√(LC))
        lc_product = self.L * self.C
        self._resonant_freq = 1.0 / (2.0 * math.pi * math.sqrt(lc_product))

        # Factor de calidad del circuito: Q = (1/R) * √(L/C)
        self._quality_factor = (1.0 / self.R) * math.sqrt(self.L / self.C)

        # Estado temporal para derivadas
        self._last_current: float = 0.0
        self._last_time: float = time.time()
        self._initialized_temporal: bool = False

        # Historial para análisis de tendencias (usar deque para O(1) en ambos extremos)
        self._metrics_history: deque = deque(maxlen=self._MAX_METRICS_HISTORY)

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Procesar warnings acumulados durante validación
        for warning in self._init_warnings:
            self.logger.warning(f"Plausibilidad: {warning}")

        self.logger.info(
            f"Motor RLC inicializado: C={self.C:.2e}F, R={self.R:.2e}Ω, L={self.L:.2e}H | "
            f"τ={self._tau_base:.4f}s, f₀={self._resonant_freq:.4f}Hz, Q={self._quality_factor:.2f}"
        )

    def _validate_parameters(
        self, capacitance: float, resistance: float, inductance: float
    ) -> None:
        """
        Valida parámetros físicos con rangos plausibles.

        Rangos industriales típicos:
        - Capacitancia: 1e-12 F (pF) a 10 F (supercapacitores)
        - Resistencia: 1e-6 Ω (μΩ) a 1e9 Ω (GΩ)
        - Inductancia: 1e-9 H (nH) a 1e3 H (kH)

        Raises:
            ConfigurationError: Si los parámetros son inválidos.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Definición con rangos plausibles
        params_spec = [
            ("capacitance", capacitance, 1e-15, 1e2),
            ("resistance", resistance, 1e-6, 1e12),
            ("inductance", inductance, 1e-12, 1e6),
        ]

        for name, value, min_plausible, max_plausible in params_spec:
            # Validación de tipo
            if not isinstance(value, (int, float)):
                errors.append(f"{name} debe ser numérico, recibido: {type(value).__name__}")
                continue

            # Conversión segura a float para comparaciones
            try:
                float_val = float(value)
            except (ValueError, TypeError) as e:
                errors.append(f"{name} no convertible a float: {e}")
                continue

            # Validación de valores especiales IEEE 754
            if math.isnan(float_val):
                errors.append(f"{name} no puede ser NaN")
                continue
            if math.isinf(float_val):
                errors.append(f"{name} no puede ser Inf: {float_val}")
                continue

            # Validación de positividad estricta
            if float_val <= 0:
                errors.append(f"{name} debe ser > 0, recibido: {float_val}")
                continue

            # Advertencias de plausibilidad física (no bloquean)
            if float_val < min_plausible:
                warnings.append(
                    f"{name}={float_val:.2e} < mínimo plausible {min_plausible:.2e}"
                )
            elif float_val > max_plausible:
                warnings.append(
                    f"{name}={float_val:.2e} > máximo plausible {max_plausible:.2e}"
                )

        # Validación cruzada: factor de calidad Q razonable
        if not errors:
            try:
                Q = (1.0 / float(resistance)) * math.sqrt(
                    float(inductance) / float(capacitance)
                )
                if Q > 1000:
                    warnings.append(
                        f"Factor Q={Q:.1f} extremadamente alto (posible inestabilidad)"
                    )
                elif Q < 0.01:
                    warnings.append(
                        f"Factor Q={Q:.4f} extremadamente bajo (sistema sobreamortiguado)"
                    )
            except (ZeroDivisionError, ValueError):
                pass  # Ya capturado en validaciones anteriores

        if errors:
            raise ConfigurationError(
                "Parámetros físicos inválidos:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Almacenar warnings para el logger (se procesarán post-init)
        self._init_warnings = warnings

    def calculate_system_entropy(
        self, total_records: int, error_count: int, processing_time: float
    ) -> Dict[str, float]:
        """
        Calcula la Entropía del Sistema (S) basada en la Segunda Ley.
        La degradación de la energía útil aumenta la entropía del sistema.

        Args:
            total_records: Volumen total de masa/datos.
            error_count: Cantidad de 'fricción' o datos corruptos.
            processing_time: Tiempo transcurrido (t).

        Returns:
            Dict con entropía actual y estado de salud.
        """
        if total_records == 0:
            return {"entropy_absolute": 0.0, "entropy_rate": 0.0, "is_thermal_death": False}

        # La probabilidad de error (microestado de desorden)
        p_error = error_count / total_records

        # Entropía de Shannon (análogo a Boltzmann): S = -k * sum(p * log(p))
        # Usamos una constante k normalizada para el dominio de datos
        # k = 1 / ln(2) approx 1.4427 para normalizar a [0, 1] (Base 2)
        k_boltzmann_data = 1.442695

        if 0 < p_error < 1:
            entropy = -k_boltzmann_data * (
                p_error * math.log(p_error) + (1 - p_error) * math.log(1 - p_error)
            )
        else:
            entropy = 0.0  # Orden total o caos total (pero estático)

        # Tasa de generación de entropía (dS/dt)
        entropy_rate = entropy / max(processing_time, 0.001)

        return {
            "entropy_absolute": entropy,
            "entropy_rate": entropy_rate,  # Velocidad de degradación
            "is_thermal_death": entropy > 0.8,  # Umbral crítico de desorden
        }

    def calculate_metrics(
        self,
        total_records: int,
        cache_hits: int,
        error_count: int = 0,
        processing_time: float = 1.0,
    ) -> Dict[str, float]:
        """
        Modelo físico RLC de segundo orden corregido + Entropía.

        El sistema modela:
        - total_records: Análogo a "carga total" o demanda del sistema
        - cache_hits: Análogo a "flujo efectivo" o corriente útil
        - error_count: Cantidad de registros fallidos (fricción)
        - processing_time: Tiempo transcurrido

        Returns:
            Dict[str, float]: Métricas físicas del sistema.
        """
        # ================ VALIDACIÓN DE ENTRADA ================
        if not isinstance(total_records, (int, float)) or total_records <= 0:
            self.logger.debug(f"total_records inválido: {total_records}")
            return self._get_zero_metrics()

        if not isinstance(cache_hits, (int, float)):
            cache_hits = 0

        # Sanitización de cache_hits
        cache_hits = int(max(0, min(total_records, cache_hits)))
        total_records = int(total_records)

        try:
            # ================ PARÁMETROS DEL SISTEMA ================
            # Corriente normalizada (eficiencia del flujo) ∈ [0, 1]
            current_I = cache_hits / total_records

            # Factor de complejidad (pérdidas/ineficiencia) ∈ [0, 1]
            complexity = 1.0 - current_I

            # Resistencia dinámica: aumenta con complejidad
            R_dyn = self.R * (
                1.0 + complexity * SystemConstants.COMPLEXITY_RESISTANCE_FACTOR
            )

            # ================ PARÁMETROS RLC CORREGIDOS ================
            # Factor de amortiguamiento: ζ = R/(2) * √(C/L) = R / (2√(L/C))
            # Equivalente a: ζ = R / (2 * Z₀) donde Z₀ = √(L/C) es impedancia característica
            impedance_char = math.sqrt(self.L / self.C)
            damping_ratio = R_dyn / (2.0 * impedance_char)

            # Frecuencia natural angular: ωₙ = 1/√(LC)
            omega_n = 1.0 / math.sqrt(self.L * self.C)

            # Frecuencia amortiguada: ωd = ωₙ√(1-ζ²) solo si ζ < 1
            if damping_ratio < 1.0:
                omega_d = omega_n * math.sqrt(max(0.0, 1.0 - damping_ratio**2))
            else:
                omega_d = 0.0

            # ================ RESPUESTA AL ESCALÓN (SATURACIÓN) ================
            # Tiempo característico del sistema basado en registros procesados
            # Interpretación: cada registro representa un "quantum" temporal
            tau_effective = R_dyn * self.C
            t_normalized = float(total_records) / max(1.0, tau_effective * 1000.0)

            # Limitar t_normalized para evitar overflow en exp()
            t_normalized = min(t_normalized, 50.0)

            if damping_ratio < 1.0:
                # SUBAMORTIGUADO: oscilaciones decrecientes
                # y(t) = 1 - (e^(-ζωₙt) / √(1-ζ²)) * sin(ωd*t + φ)
                # donde φ = arccos(ζ)
                zeta_omega_t = damping_ratio * omega_n * t_normalized

                # Protección contra overflow
                if zeta_omega_t > 700:
                    exp_term = 0.0
                else:
                    exp_term = math.exp(-zeta_omega_t)

                sqrt_term = math.sqrt(max(1e-10, 1.0 - damping_ratio**2))

                # Fase correcta: φ = arccos(ζ)
                phase = math.acos(min(1.0, max(-1.0, damping_ratio)))

                sin_arg = omega_d * t_normalized + phase
                sin_term = math.sin(sin_arg)

                saturation_V = 1.0 - (exp_term / sqrt_term) * sin_term

            elif abs(damping_ratio - 1.0) < 0.05:
                # CRÍTICAMENTE AMORTIGUADO: respuesta más rápida sin oscilación
                # y(t) = 1 - (1 + ωₙt) * e^(-ωₙt)
                omega_t = omega_n * t_normalized

                if omega_t > 700:
                    saturation_V = 1.0
                else:
                    exp_term = math.exp(-omega_t)
                    saturation_V = 1.0 - (1.0 + omega_t) * exp_term

            else:
                # SOBREAMORTIGUADO: respuesta lenta sin oscilación
                # y(t) = 1 - (s₂/(s₂-s₁))e^(s₁t) + (s₁/(s₂-s₁))e^(s₂t)
                discriminant = math.sqrt(max(0.0, damping_ratio**2 - 1.0))
                s1 = -omega_n * (damping_ratio - discriminant)
                s2 = -omega_n * (damping_ratio + discriminant)

                # Protección división por cero
                s_diff = s2 - s1
                if abs(s_diff) < 1e-10:
                    saturation_V = 1.0 - math.exp(s1 * t_normalized)
                else:
                    # Protección overflow
                    exp_s1 = math.exp(max(-700, min(700, s1 * t_normalized)))
                    exp_s2 = math.exp(max(-700, min(700, s2 * t_normalized)))

                    A = s2 / s_diff
                    B = -s1 / s_diff
                    saturation_V = 1.0 - (A * exp_s1 + B * exp_s2)

            # Clamp saturación a [0, 1]
            saturation_V = max(0.0, min(1.0, saturation_V))

            # ================ ENERGÍAS FÍSICAS REALES ================
            # Energía almacenada en capacitor: E_C = ½CV²
            E_capacitor = 0.5 * self.C * (saturation_V**2)

            # Energía almacenada en inductor: E_L = ½LI²
            E_inductor = 0.5 * self.L * (current_I**2)

            # Energía total del sistema
            E_total = E_capacitor + E_inductor

            # Potencia disipada instantánea: P = I²R
            P_dissipated = (current_I**2) * R_dyn

            # Factor de potencia: cos(φ) = R/Z = P_real/P_aparente
            impedance_total = math.sqrt(
                R_dyn**2 + (omega_n * self.L - 1 / (omega_n * self.C)) ** 2
            )
            power_factor = R_dyn / max(1e-10, impedance_total)
            power_factor = max(0.0, min(1.0, power_factor))

            # ================ TENSIÓN DE FLYBACK (di/dt) ================
            current_time = time.time()

            if not self._initialized_temporal:
                self._last_current = current_I
                self._last_time = current_time
                self._initialized_temporal = True
                di_dt = 0.0
            else:
                dt = current_time - self._last_time
                dt = max(1e-6, dt)  # Mínimo 1μs para evitar división por cero

                di_dt = (current_I - self._last_current) / dt

                self._last_current = current_I
                self._last_time = current_time

            # Tensión inductiva: V_L = L * (di/dt)
            V_flyback = abs(self.L * di_dt)
            V_flyback = min(V_flyback, SystemConstants.MAX_FLYBACK_VOLTAGE)

            # ================ MÉTRICAS DE ESTABILIDAD CORREGIDAS ================
            # Factor de estabilidad: MAYOR amortiguamiento = MÁS estable
            # Normalizado: 0 = inestable (ζ→0), 1 = muy estable (ζ≥1)
            if damping_ratio >= 1.0:
                stability_factor = 1.0
            else:
                stability_factor = damping_ratio  # Lineal para ζ ∈ [0,1)

            # Margen de fase aproximado para sistema de 2do orden
            # PM ≈ arctan(2ζ / √(√(1+4ζ⁴) - 2ζ²)) en grados
            if damping_ratio > 0:
                inner = math.sqrt(
                    max(0.0, math.sqrt(1 + 4 * damping_ratio**4) - 2 * damping_ratio**2)
                )
                if inner > 1e-10:
                    phase_margin = math.degrees(math.atan(2 * damping_ratio / inner))
                else:
                    phase_margin = 90.0  # Sistema muy amortiguado
            else:
                phase_margin = 0.0  # Sistema marginal

            phase_margin = max(0.0, min(90.0, phase_margin))

            # ================ CLASIFICACIÓN DEL SISTEMA ================
            if damping_ratio < 0.7:
                system_type = "UNDERDAMPED"
            elif damping_ratio < 1.1:
                system_type = "CRITICALLY_DAMPED"
            else:
                system_type = "OVERDAMPED"

            # ================ CÁLCULO DE ENTROPÍA (TERMODINÁMICA) ================
            entropy_metrics = self.calculate_system_entropy(
                total_records, error_count, processing_time
            )

            # ================ CONSTRUIR RESULTADOS ================
            metrics = {
                # Métricas principales normalizadas
                "saturation": self._sanitize_metric(saturation_V, 0.0, 1.0),
                "complexity": self._sanitize_metric(complexity, 0.0, 1.0),
                "current_I": self._sanitize_metric(current_I, 0.0, 1.0),
                # Energías (en Joules)
                "potential_energy": self._sanitize_metric(E_capacitor, 0.0, 1e10),
                "kinetic_energy": self._sanitize_metric(E_inductor, 0.0, 1e10),
                "total_energy": self._sanitize_metric(E_total, 0.0, 1e10),
                # Potencia y voltaje
                "dissipated_power": self._sanitize_metric(P_dissipated, 0.0, 1e6),
                "flyback_voltage": self._sanitize_metric(
                    V_flyback, 0.0, SystemConstants.MAX_FLYBACK_VOLTAGE
                ),
                # Parámetros del sistema RLC
                "dynamic_resistance": self._sanitize_metric(R_dyn, 0.0, 1e9),
                "damping_ratio": self._sanitize_metric(damping_ratio, 0.0, 100.0),
                "natural_frequency": self._sanitize_metric(omega_n, 0.0, 1e9),
                "damped_frequency": self._sanitize_metric(omega_d, 0.0, 1e9),
                # Métricas de calidad
                "power_factor": self._sanitize_metric(power_factor, 0.0, 1.0),
                "stability_factor": self._sanitize_metric(stability_factor, 0.0, 1.0),
                "phase_margin": self._sanitize_metric(phase_margin, 0.0, 90.0),
                # Termodinámica
                "entropy_absolute": entropy_metrics["entropy_absolute"],
                "entropy_rate": entropy_metrics["entropy_rate"],
                "is_thermal_death": entropy_metrics["is_thermal_death"],
                # Clasificación
                "system_type": system_type,
            }

            # Almacenar en historial
            self._store_metrics(metrics)

            # Diagnóstico selectivo
            if damping_ratio < 0.3:
                self.logger.warning(
                    f"Sistema muy subamortiguado (ζ={damping_ratio:.3f}) - riesgo de oscilaciones"
                )
            elif damping_ratio > 5.0:
                self.logger.debug(
                    f"Sistema muy sobreamortiguado (ζ={damping_ratio:.2f}) - respuesta lenta"
                )

            return metrics

        except OverflowError as e:
            self.logger.error(f"Overflow en cálculo físico: {e}")
            return self._get_zero_metrics()
        except ValueError as e:
            self.logger.error(f"ValueError en modelo físico: {e}")
            return self._get_zero_metrics()
        except ZeroDivisionError as e:
            self.logger.error(f"División por cero en modelo físico: {e}")
            return self._get_zero_metrics()
        except Exception as e:
            self.logger.error(f"Error inesperado en modelo físico: {type(e).__name__}: {e}")
            return self._get_zero_metrics()

    def _sanitize_metric(self, value: float, min_val: float, max_val: float) -> float:
        """
        Sanitiza un valor de métrica asegurando que sea válido.

        Args:
            value: Valor a sanitizar.
            min_val: Valor mínimo permitido.
            max_val: Valor máximo permitido.

        Returns:
            float: Valor sanitizado.
        """
        if math.isnan(value) or math.isinf(value):
            self.logger.debug(f"Métrica inválida ({value}), reemplazando con 0.0")
            return 0.0

        return max(min_val, min(max_val, value))

    def _get_zero_metrics(self) -> Dict[str, float]:
        """
        Retorna métricas en estado cero/inicial (seguro y neutral).

        Returns:
            Dict[str, float]: Diccionario con valores por defecto seguros.
        """
        return {
            # Métricas principales
            "saturation": 0.0,
            "complexity": 1.0,  # Sin datos = máxima incertidumbre
            "current_I": 0.0,
            # Energías
            "potential_energy": 0.0,
            "kinetic_energy": 0.0,
            "total_energy": 0.0,
            # Potencia y voltaje
            "dissipated_power": 0.0,
            "flyback_voltage": 0.0,
            # Parámetros del sistema (valores nominales)
            "dynamic_resistance": self.R,
            "damping_ratio": 1.0,  # Críticamente amortiguado = estable por defecto
            "natural_frequency": self._resonant_freq * 2.0 * math.pi,
            "damped_frequency": 0.0,
            # Métricas de calidad (valores conservadores)
            "power_factor": 1.0,  # Ideal por defecto
            "stability_factor": 1.0,  # Estable por defecto
            "phase_margin": 45.0,  # Margen razonable
            # Clasificación
            "system_type": "INITIAL",
        }

    def _store_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Almacena métricas en historial con buffer circular eficiente.

        Usa deque con maxlen para O(1) en inserción y eliminación automática.

        Args:
            metrics: Diccionario de métricas a almacenar.
        """
        # Crear copia con timestamp para evitar mutaciones externas
        timestamped = {
            **{k: v for k, v in metrics.items() if k != "system_type"},
            "_timestamp": time.time(),
            "_system_type": metrics.get("system_type", "UNKNOWN"),
        }

        # deque con maxlen maneja automáticamente el límite
        self._metrics_history.append(timestamped)

    def get_system_diagnosis(self, metrics: Dict[str, float]) -> str:
        """
        Genera diagnóstico jerárquico del sistema.

        Args:
            metrics: Diccionario de métricas actuales.

        Returns:
            str: Diagnóstico textual con emoji indicador de severidad.
        """
        # Validación de entrada robusta
        if not isinstance(metrics, dict) or not metrics:
            return "❓ DIAGNÓSTICO NO DISPONIBLE (métricas inválidas o vacías)"

        try:
            # Extracción segura con valores por defecto conservadores
            def safe_get(key: str, default: float) -> float:
                val = metrics.get(key, default)
                if not isinstance(val, (int, float)):
                    return default
                if math.isnan(val) or math.isinf(val):
                    return default
                return float(val)

            ec = safe_get("potential_energy", 0.0)
            el = safe_get("kinetic_energy", 0.0)
            flyback = safe_get("flyback_voltage", 0.0)
            power = safe_get("dissipated_power", 0.0)
            saturation = safe_get("saturation", 0.0)
            damping = safe_get("damping_ratio", 1.0)
            total_energy = safe_get("total_energy", ec + el)

            # ═══════════════════════════════════════════════════════════════
            # DIAGNÓSTICO JERÁRQUICO (orden de criticidad descendente)
            # ═══════════════════════════════════════════════════════════════

            # NIVEL CRÍTICO (🔴)

            # 1. Sobrecalentamiento térmico
            if power > SystemConstants.OVERHEAT_POWER_THRESHOLD:
                return f"🔴 SOBRECALENTAMIENTO CRÍTICO (P={power:.2f}W > {SystemConstants.OVERHEAT_POWER_THRESHOLD}W)"

            # 2. Inestabilidad dinámica severa
            if damping < 0.1:
                return (
                    f"🔴 INESTABILIDAD SEVERA (ζ={damping:.3f} → oscilaciones divergentes)"
                )

            # 3. Sistema sin energía cinética (estancado)
            if el < SystemConstants.MIN_ENERGY_THRESHOLD and saturation < 0.1:
                return "🔴 SISTEMA ESTANCADO (sin inercia ni saturación)"

            # NIVEL ADVERTENCIA (🟠)

            # 4. Inestabilidad moderada
            if damping < 0.3:
                return f"🟠 INESTABILIDAD MODERADA (ζ={damping:.2f} → oscilaciones persistentes)"

            # 5. Sobrepresión energética (más potencial que cinética)
            if el > SystemConstants.MIN_ENERGY_THRESHOLD:
                energy_ratio = ec / el
                if energy_ratio > SystemConstants.HIGH_PRESSURE_RATIO:
                    return f"🟠 SOBRECARGA POTENCIAL (Ec/El={energy_ratio:.1f}x)"

            # 6. Pico inductivo significativo
            if flyback > SystemConstants.HIGH_FLYBACK_THRESHOLD:
                return f"⚡ PICO INDUCTIVO ALTO (V_L={flyback:.2f}V)"

            # NIVEL PRECAUCIÓN (🟡)

            # 7. Saturación muy alta
            if saturation > 0.95:
                return f"🟡 SATURACIÓN LÍMITE ({saturation:.1%})"

            # 8. Baja inercia operativa
            if el < SystemConstants.LOW_INERTIA_THRESHOLD and el > 0:
                return f"🟡 INERCIA BAJA (El={el:.4f}J)"

            # 9. Sistema subamortiguado (oscilatorio)
            if damping < 0.7:
                return f"🔵 RÉGIMEN OSCILATORIO (ζ={damping:.2f}, respuesta con overshoot)"

            # 10. Sistema sobreamortiguado (lento)
            if damping > 2.0:
                return f"🔵 RÉGIMEN LENTO (ζ={damping:.2f}, respuesta amortiguada)"

            # NIVEL OK (🟢)

            # Sistema en equilibrio saludable
            efficiency = saturation * 100
            return f"🟢 EQUILIBRIO NOMINAL (η={efficiency:.1f}%, ζ={damping:.2f})"

        except Exception as e:
            self.logger.error(f"Error en diagnóstico: {type(e).__name__}: {e}")
            return f"❓ ERROR DE DIAGNÓSTICO ({type(e).__name__})"

    def get_trend_analysis(self) -> Dict[str, Any]:
        """
        Analiza tendencias basadas en historial de métricas.

        Calcula estadísticas móviles y detecta patrones de comportamiento.

        Returns:
            Dict[str, Any]: Análisis de tendencias con estadísticas.
        """
        history_len = len(self._metrics_history)

        if history_len < 5:
            return {
                "status": "INSUFFICIENT_DATA",
                "samples": history_len,
                "message": f"Se requieren al menos 5 muestras, hay {history_len}",
            }

        try:
            # Tomar las últimas N muestras (máximo 20)
            sample_size = min(20, history_len)
            recent = list(self._metrics_history)[-sample_size:]

            def extract_series(key: str, default: float = 0.0) -> List[float]:
                """Extrae serie temporal de una métrica con manejo de errores."""
                series = []
                for m in recent:
                    val = m.get(key, default)
                    if isinstance(val, (int, float)) and not (
                        math.isnan(val) or math.isinf(val)
                    ):
                        series.append(float(val))
                    else:
                        series.append(default)
                return series

            def calc_stats(series: List[float]) -> Dict[str, float]:
                """Calcula estadísticas de una serie."""
                if not series:
                    return {
                        "current": 0.0,
                        "average": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "std": 0.0,
                    }

                n = len(series)
                avg = sum(series) / n
                variance = sum((x - avg) ** 2 for x in series) / n
                std = math.sqrt(variance)

                return {
                    "current": series[-1],
                    "average": avg,
                    "min": min(series),
                    "max": max(series),
                    "std": std,
                }

            def detect_trend(series: List[float], threshold: float = 0.05) -> str:
                """Detecta tendencia usando regresión lineal simple."""
                if len(series) < 3:
                    return "INSUFFICIENT"

                n = len(series)
                x_mean = (n - 1) / 2.0
                y_mean = sum(series) / n

                # Pendiente: Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
                numerator = sum((i - x_mean) * (series[i] - y_mean) for i in range(n))
                denominator = sum((i - x_mean) ** 2 for i in range(n))

                if abs(denominator) < 1e-10:
                    return "STABLE"

                slope = numerator / denominator

                # Normalizar pendiente por el rango de valores
                value_range = (
                    max(series) - min(series) if max(series) != min(series) else 1.0
                )
                normalized_slope = slope * n / value_range

                if normalized_slope > threshold:
                    return "INCREASING"
                elif normalized_slope < -threshold:
                    return "DECREASING"
                else:
                    return "STABLE"

            # Extraer series
            saturations = extract_series("saturation", 0.0)
            powers = extract_series("dissipated_power", 0.0)
            dampings = extract_series("damping_ratio", 1.0)
            energies = extract_series("total_energy", 0.0)

            # Calcular timestamps para tasa de muestreo
            timestamps = extract_series("_timestamp", time.time())
            if len(timestamps) >= 2:
                time_span = timestamps[-1] - timestamps[0]
                sample_rate = (len(timestamps) - 1) / max(0.001, time_span)
            else:
                time_span = 0.0
                sample_rate = 0.0

            return {
                "status": "OK",
                "samples": history_len,
                "window_size": sample_size,
                "time_span_seconds": round(time_span, 2),
                "sample_rate_hz": round(sample_rate, 3),
                "saturation": {
                    **calc_stats(saturations),
                    "trend": detect_trend(saturations, 0.03),
                },
                "power": {
                    **calc_stats(powers),
                    "trend": detect_trend(powers, 0.1),
                },
                "damping": {
                    **calc_stats(dampings),
                    "trend": detect_trend(dampings, 0.05),
                },
                "energy": {
                    **calc_stats(energies),
                    "trend": detect_trend(energies, 0.05),
                },
                # Alertas basadas en tendencias
                "alerts": self._generate_trend_alerts(saturations, powers, dampings),
            }

        except Exception as e:
            self.logger.error(f"Error en análisis de tendencias: {type(e).__name__}: {e}")
            return {
                "status": "ERROR",
                "samples": history_len,
                "error": str(e),
            }

    def _generate_trend_alerts(
        self, saturations: List[float], powers: List[float], dampings: List[float]
    ) -> List[str]:
        """Genera alertas basadas en análisis de tendencias."""
        alerts = []

        if len(saturations) >= 3:
            # Alerta si saturación está cayendo consistentemente
            if all(saturations[i] > saturations[i + 1] for i in range(-3, -1)):
                alerts.append("⚠️ Saturación en descenso sostenido")

            # Alerta si saturación muy alta y estable (posible cuello de botella)
            if min(saturations[-5:]) > 0.9:
                alerts.append("⚠️ Saturación persistentemente alta (>90%)")

        if len(powers) >= 3:
            # Alerta si potencia disipada creciendo
            if all(powers[i] < powers[i + 1] for i in range(-3, -1)):
                alerts.append("🔥 Potencia disipada en aumento")

        if len(dampings) >= 3:
            # Alerta si amortiguamiento cayendo (hacia inestabilidad)
            if all(dampings[i] > dampings[i + 1] for i in range(-3, -1)):
                if dampings[-1] < 0.5:
                    alerts.append("⚡ Sistema aproximándose a inestabilidad")

        return alerts


# ============================================================================
# DATA FLUX CONDENSER
# ============================================================================
class DataFluxCondenser:
    """
    Orquesta el pipeline de validación y procesamiento de archivos de APU.

    ROBUSTECIDO:
    - Validación exhaustiva en cada etapa.
    - Timeouts y límites de recursos.
    - Recuperación parcial mejorada.
    - Métricas de calidad de datos.
    - Logging estructurado con contexto.
    """

    REQUIRED_CONFIG_KEYS: Set[str] = {"parser_settings", "processor_settings"}
    REQUIRED_PROFILE_KEYS: Set[str] = {"columns_mapping", "validation_rules"}

    def __init__(
        self,
        config: Dict[str, Any],
        profile: Dict[str, Any],
        condenser_config: Optional[CondenserConfig] = None,
    ):
        """
        Inicializa el Condensador con validación exhaustiva.

        Args:
            config: Configuración global del sistema.
            profile: Perfil de procesamiento específico.
            condenser_config: Configuración específica del condensador.
        """
        # Inicializar logger primero (para usar en validaciones)
        self.logger = logging.getLogger(self.__class__.__name__)

        # ═══════════════════════════════════════════════════════════════════════
        # VALIDACIÓN DE CONFIGURACIÓN DEL CONDENSADOR
        # ═══════════════════════════════════════════════════════════════════════

        try:
            if condenser_config is None:
                self.logger.debug("Usando CondenserConfig por defecto")
                self.condenser_config = CondenserConfig()
            elif isinstance(condenser_config, CondenserConfig):
                self.condenser_config = condenser_config
            else:
                raise InvalidInputError(
                    f"condenser_config debe ser CondenserConfig o None, "
                    f"recibido: {type(condenser_config).__name__}"
                )
        except ConfigurationError as e:
            raise ConfigurationError(f"Error en configuración del condensador: {e}") from e

        # Ajustar nivel de log
        try:
            self.logger.setLevel(self.condenser_config.log_level.upper())
        except (ValueError, AttributeError):
            self.logger.setLevel(logging.INFO)

        # ═══════════════════════════════════════════════════════════════════════
        # VALIDACIÓN DE CONFIG Y PROFILE
        # ═══════════════════════════════════════════════════════════════════════

        self._validate_initialization_params(config, profile)

        # Copiar para evitar mutaciones externas
        self.config = dict(config) if config else {}
        self.profile = dict(profile) if profile else {}

        # ═══════════════════════════════════════════════════════════════════════
        # INICIALIZACIÓN DE COMPONENTES
        # ═══════════════════════════════════════════════════════════════════════

        try:
            self.physics = FluxPhysicsEngine(
                capacitance=self.condenser_config.system_capacitance,
                resistance=self.condenser_config.base_resistance,
                inductance=self.condenser_config.system_inductance,
            )
        except ConfigurationError as e:
            raise ConfigurationError(f"Error inicializando motor físico: {e}") from e

        try:
            self.controller = PIController(
                kp=self.condenser_config.pid_kp,
                ki=self.condenser_config.pid_ki,
                setpoint=self.condenser_config.pid_setpoint,
                min_output=self.condenser_config.min_batch_size,
                max_output=self.condenser_config.max_batch_size,
                integral_limit_factor=self.condenser_config.integral_limit_factor,
            )
        except ConfigurationError as e:
            raise ConfigurationError(f"Error inicializando controlador PID: {e}") from e

        # Estadísticas
        self._stats = ProcessingStats()
        self._start_time: Optional[float] = None

        self.logger.info(
            f"DataFluxCondenser inicializado | "
            f"PID: Kp={self.condenser_config.pid_kp}, Ki={self.condenser_config.pid_ki} | "
            f"Batch: [{self.condenser_config.min_batch_size}, {self.condenser_config.max_batch_size}]"
        )

    def _validate_initialization_params(
        self, config: Dict[str, Any], profile: Dict[str, Any]
    ) -> None:
        """
        Valida parámetros de inicialización con mensajes detallados.

        Args:
            config: Configuración global.
            profile: Perfil de procesamiento.

        Raises:
            InvalidInputError: Si hay errores críticos en los parámetros.
        """
        errors: List[str] = []
        self._init_warnings: List[str] = []

        # ═══════════════════════════════════════════════════════════════════
        # VALIDACIÓN DE CONFIG
        # ═══════════════════════════════════════════════════════════════════
        if config is None:
            errors.append("config no puede ser None")
        elif not isinstance(config, dict):
            errors.append(f"config debe ser dict, recibido: {type(config).__name__}")
        else:
            # Verificar claves requeridas
            missing_config = self.REQUIRED_CONFIG_KEYS - set(config.keys())
            if missing_config:
                self._init_warnings.append(f"Claves faltantes en config: {missing_config}")

            # Validar tipos de valores críticos
            if "parser_settings" in config:
                if not isinstance(config["parser_settings"], dict):
                    self._init_warnings.append(
                        f"parser_settings debe ser dict, recibido: "
                        f"{type(config['parser_settings']).__name__}"
                    )

            if "processor_settings" in config:
                if not isinstance(config["processor_settings"], dict):
                    self._init_warnings.append(
                        f"processor_settings debe ser dict, recibido: "
                        f"{type(config['processor_settings']).__name__}"
                    )

        # ═══════════════════════════════════════════════════════════════════
        # VALIDACIÓN DE PROFILE
        # ═══════════════════════════════════════════════════════════════════
        if profile is None:
            errors.append("profile no puede ser None")
        elif not isinstance(profile, dict):
            errors.append(f"profile debe ser dict, recibido: {type(profile).__name__}")
        else:
            # Verificar claves requeridas
            missing_profile = self.REQUIRED_PROFILE_KEYS - set(profile.keys())
            if missing_profile:
                self._init_warnings.append(f"Claves faltantes en profile: {missing_profile}")

            # Validar estructura de columns_mapping
            if "columns_mapping" in profile:
                mapping = profile["columns_mapping"]
                if not isinstance(mapping, dict):
                    self._init_warnings.append(
                        f"columns_mapping debe ser dict, recibido: {type(mapping).__name__}"
                    )
                elif not mapping:
                    self._init_warnings.append("columns_mapping está vacío")

            # Validar estructura de validation_rules
            if "validation_rules" in profile:
                rules = profile["validation_rules"]
                if not isinstance(rules, (dict, list)):
                    self._init_warnings.append(
                        f"validation_rules debe ser dict o list, recibido: {type(rules).__name__}"
                    )

        # Lanzar errores críticos
        if errors:
            raise InvalidInputError(
                "Errores de inicialización:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Registrar warnings (el logger ya existe en este punto)
        for warning in self._init_warnings:
            self.logger.warning(f"[INIT] {warning}")

    def stabilize(
        self,
        file_path: str,
        on_progress: Optional[Callable[[ProcessingStats], None]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        telemetry: Optional[TelemetryContext] = None,
    ) -> pd.DataFrame:
        """
        Proceso principal de estabilización con control PID.

        Orquesta el flujo: validar -> parsear -> extraer -> procesar lotes -> consolidar.

        Args:
            file_path: Ruta al archivo de APU a procesar.
            on_progress: Callback opcional para reportar progreso en tiempo real.
            telemetry: Contexto de telemetría opcional.

        Returns:
            pd.DataFrame: DataFrame con los datos procesados.

        Raises:
            InvalidInputError: Si el archivo es inválido.
            ProcessingError: Si ocurre un error durante el procesamiento.
        """
        self._start_time = time.time()

        # Resetear estado para nuevo procesamiento
        self._stats = ProcessingStats()
        self.controller.reset()

        # Validar entrada temprano
        if not file_path:
            raise InvalidInputError("file_path no puede ser vacío o None")

        path_obj = Path(file_path) if isinstance(file_path, str) else file_path

        self.logger.info(
            f"⚡ [STABILIZE] Iniciando procesamiento: {path_obj.name if hasattr(path_obj, 'name') else file_path}"
        )

        try:
            # ═══════════════════════════════════════════════════════════════════
            # FASE 1: Validación de archivo
            # ═══════════════════════════════════════════════════════════════════
            validated_path = self._validate_input_file(file_path)
            self._check_timeout("validación de archivo")

            # ═══════════════════════════════════════════════════════════════════
            # FASE 2: Inicialización del parser
            # ═══════════════════════════════════════════════════════════════════
            parser = self._initialize_parser(validated_path)
            self._check_timeout("inicialización del parser")

            # ═══════════════════════════════════════════════════════════════════
            # FASE 3: Extracción de datos crudos
            # ═══════════════════════════════════════════════════════════════════
            raw_records, cache = self._extract_raw_data(parser)
            self._check_timeout("extracción de datos")

            if not raw_records:
                self.logger.warning("[STABILIZE] Archivo sin registros válidos")
                return self._create_empty_result()

            total_records = len(raw_records)
            self._stats.total_records = total_records

            # ═══════════════════════════════════════════════════════════════════
            # FASE 4: Validación de umbral mínimo
            # ═══════════════════════════════════════════════════════════════════
            if total_records < self.condenser_config.min_records_threshold:
                self.logger.warning(
                    f"[STABILIZE] Registros insuficientes: {total_records} < "
                    f"{self.condenser_config.min_records_threshold}"
                )
                return self._create_empty_result()

            # Validar límite máximo
            if total_records > SystemConstants.MAX_RECORDS_LIMIT:
                self.logger.warning(
                    f"[STABILIZE] Excede límite de registros: {total_records} > "
                    f"{SystemConstants.MAX_RECORDS_LIMIT}. Truncando."
                )
                raw_records = raw_records[: SystemConstants.MAX_RECORDS_LIMIT]
                total_records = len(raw_records)

            # ═══════════════════════════════════════════════════════════════════
            # FASE 5: Procesamiento por lotes con PID
            # ═══════════════════════════════════════════════════════════════════
            processed_batches = self._process_batches_with_pid(
                raw_records,
                cache,
                total_records,
                on_progress,
                progress_callback=progress_callback,
                telemetry=telemetry,
            )
            self._check_timeout("procesamiento por lotes")

            # ═══════════════════════════════════════════════════════════════════
            # FASE 6: Consolidación de resultados
            # ═══════════════════════════════════════════════════════════════════
            df_final = self._consolidate_results(processed_batches)

            # ═══════════════════════════════════════════════════════════════════
            # FASE 7: Validación de salida
            # ═══════════════════════════════════════════════════════════════════
            self._validate_output(df_final)

            # Estadísticas finales
            self._stats.processing_time = time.time() - self._start_time
            self._log_final_stats()

            # Registrar métricas físicas finales en telemetría
            if telemetry:
                telemetry.record_metric(
                    "flux_condenser",
                    "max_dissipated_power",
                    self._stats.max_dissipated_power,
                )
                telemetry.record_metric(
                    "flux_condenser", "max_flyback_voltage", self._stats.max_flyback_voltage
                )
                telemetry.record_metric(
                    "flux_condenser", "avg_saturation", self._stats.avg_saturation
                )
                telemetry.record_metric(
                    "flux_condenser", "total_records", self._stats.total_records
                )
                telemetry.record_metric(
                    "flux_condenser", "processed_records", self._stats.processed_records
                )

            self.logger.info(
                f"✅ [STABILIZE] Completado en {self._stats.processing_time:.2f}s | "
                f"Procesados: {self._stats.processed_records}/{total_records}"
            )

            return df_final

        except (InvalidInputError, ProcessingError, ConfigurationError):
            # Re-lanzar errores conocidos
            raise
        except Exception as e:
            self.logger.exception(f"[STABILIZE] Error inesperado: {e}")
            raise ProcessingError(f"Error inesperado durante estabilización: {e}") from e
        finally:
            # Limpieza
            self._cleanup_after_processing()

    def _check_timeout(self, phase: str) -> None:
        """
        Verifica si se ha excedido el timeout de procesamiento.

        Args:
            phase: Nombre de la fase actual para logging.

        Raises:
            ProcessingError: Si se excede el timeout.
        """
        if self._start_time is None:
            return

        elapsed = time.time() - self._start_time
        if elapsed > SystemConstants.PROCESSING_TIMEOUT:
            raise ProcessingError(
                f"Timeout de procesamiento excedido en fase '{phase}': "
                f"{elapsed:.1f}s > {SystemConstants.PROCESSING_TIMEOUT}s"
            )

    def _cleanup_after_processing(self) -> None:
        """
        Limpieza después del procesamiento (exitoso o fallido).

        Libera referencias a objetos grandes y resetea estado temporal
        para permitir reutilización eficiente de memoria.
        """
        # Limpiar caché de claves normalizadas (puede ser grande)
        if hasattr(self, "_cache_keys_normalized"):
            self._cache_keys_normalized.clear()
            del self._cache_keys_normalized

        # Limpiar estado temporal del controlador de saturación
        if hasattr(self, "_last_saturation"):
            del self._last_saturation

        # Forzar garbage collection para objetos grandes si el procesamiento fue extenso
        if self._stats.total_records > 10000:
            import gc

            gc.collect()

        self.logger.debug("[CLEANUP] Limpieza post-procesamiento completada")

    def _create_empty_result(self) -> pd.DataFrame:
        """
        Crea un DataFrame vacío con metadatos de diagnóstico.

        Returns:
            pd.DataFrame: DataFrame vacío con atributos de contexto.
        """
        elapsed = time.time() - (self._start_time or time.time())
        self._stats.processing_time = elapsed

        # Crear DataFrame vacío con estructura base esperada
        df_empty = pd.DataFrame()

        # Agregar metadatos como atributos (accesibles vía df.attrs)
        df_empty.attrs["_condenser_metadata"] = {
            "status": "EMPTY_RESULT",
            "processing_time_seconds": round(elapsed, 3),
            "reason": self._determine_empty_reason(),
            "timestamp": time.time(),
            "condenser_version": getattr(self, "_version", "1.0.0"),
        }

        self.logger.info(
            f"[EMPTY_RESULT] Retornando DataFrame vacío. "
            f"Razón: {df_empty.attrs['_condenser_metadata']['reason']}"
        )

        return df_empty

    def _determine_empty_reason(self) -> str:
        """Determina la razón del resultado vacío basándose en el estado."""
        if self._stats.total_records == 0:
            return "NO_RECORDS_EXTRACTED"
        elif self._stats.total_records < self.condenser_config.min_records_threshold:
            return f"BELOW_THRESHOLD ({self._stats.total_records} < {self.condenser_config.min_records_threshold})"
        elif self._stats.failed_batches > 0 and self._stats.processed_records == 0:
            return "ALL_BATCHES_FAILED"
        else:
            return "UNKNOWN"

    def _validate_input_file(self, file_path: str) -> Path:
        """
        Valida el archivo de entrada exhaustivamente.

        Args:
            file_path: Ruta al archivo.

        Returns:
            Path: Objeto Path validado y resuelto.

        Raises:
            InvalidInputError: Si el archivo es inválido o inaccesible.
        """
        # ═══════════════════════════════════════════════════════════════════
        # VALIDACIÓN DE TIPO Y FORMATO
        # ═══════════════════════════════════════════════════════════════════
        if not isinstance(file_path, (str, Path)):
            raise InvalidInputError(
                f"file_path debe ser str o Path, recibido: {type(file_path).__name__}"
            )

        if isinstance(file_path, str):
            stripped = file_path.strip()
            if not stripped:
                raise InvalidInputError("file_path no puede ser una cadena vacía")
            # Detectar caracteres problemáticos
            if any(c in stripped for c in ["\x00", "\n", "\r"]):
                raise InvalidInputError("file_path contiene caracteres de control inválidos")
            path = Path(stripped)
        else:
            path = file_path

        # ═══════════════════════════════════════════════════════════════════
        # RESOLUCIÓN Y EXISTENCIA
        # ═══════════════════════════════════════════════════════════════════
        try:
            # Resolver enlaces simbólicos para obtener ruta real
            resolved_path = path.resolve()
        except (OSError, RuntimeError) as e:
            raise InvalidInputError(f"Error resolviendo ruta: {e}") from e

        if not resolved_path.exists():
            raise InvalidInputError(f"El archivo no existe: {path}")

        # Verificar que es archivo (no directorio, dispositivo, etc.)
        if not resolved_path.is_file():
            if resolved_path.is_dir():
                raise InvalidInputError(f"La ruta es un directorio, no un archivo: {path}")
            elif resolved_path.is_symlink():
                raise InvalidInputError(f"Enlace simbólico roto: {path}")
            else:
                raise InvalidInputError(f"La ruta no es un archivo regular: {path}")

        # ═══════════════════════════════════════════════════════════════════
        # PERMISOS DE ACCESO
        # ═══════════════════════════════════════════════════════════════════
        try:
            if not os.access(resolved_path, os.R_OK):
                raise InvalidInputError(f"Sin permisos de lectura: {path}")
        except OSError as e:
            raise InvalidInputError(f"Error verificando permisos: {e}") from e

        # Verificar que no está bloqueado (intento de apertura)
        header = b""
        try:
            with open(resolved_path, "rb") as f:
                # Leer solo los primeros bytes para verificar acceso
                header = f.read(16)
        except PermissionError as e:
            raise InvalidInputError(f"Archivo bloqueado o sin permisos: {e}") from e
        except IOError as e:
            raise InvalidInputError(f"Error de I/O al verificar archivo: {e}") from e

        # ═══════════════════════════════════════════════════════════════════
        # VALIDACIÓN DE TAMAÑO
        # ═══════════════════════════════════════════════════════════════════
        try:
            file_stat = resolved_path.stat()
            file_size = file_stat.st_size
            file_size_mb = file_size / (1024 * 1024)

            if file_size < SystemConstants.MIN_FILE_SIZE_BYTES:
                raise InvalidInputError(
                    f"Archivo demasiado pequeño ({file_size} bytes < "
                    f"{SystemConstants.MIN_FILE_SIZE_BYTES} bytes): {path}"
                )

            if file_size_mb > SystemConstants.MAX_FILE_SIZE_MB:
                self.logger.warning(
                    f"[VALIDATE] Archivo grande: {file_size_mb:.1f} MB > "
                    f"{SystemConstants.MAX_FILE_SIZE_MB} MB límite recomendado"
                )

            # Verificar que el archivo no fue modificado recientemente (posible escritura en curso)
            mtime = file_stat.st_mtime
            if time.time() - mtime < 1.0:  # Modificado hace menos de 1 segundo
                self.logger.warning(
                    f"[VALIDATE] Archivo modificado recientemente, "
                    f"posible escritura en curso: {path}"
                )

        except OSError as e:
            raise InvalidInputError(f"Error obteniendo información del archivo: {e}") from e

        # ═══════════════════════════════════════════════════════════════════
        # VALIDACIÓN DE EXTENSIÓN Y CONTENIDO
        # ═══════════════════════════════════════════════════════════════════
        suffix_lower = resolved_path.suffix.lower()
        if suffix_lower not in SystemConstants.VALID_FILE_EXTENSIONS:
            self.logger.warning(
                f"[VALIDATE] Extensión no estándar: '{suffix_lower}'. "
                f"Esperadas: {SystemConstants.VALID_FILE_EXTENSIONS}"
            )

        # Detectar si es archivo binario (puede indicar formato incorrecto)
        if header:
            # Bytes nulos indican archivo binario
            null_ratio = header.count(b"\x00") / len(header)
            if null_ratio > 0.3:
                self.logger.warning(
                    f"[VALIDATE] Archivo posiblemente binario (null ratio: {null_ratio:.1%})"
                )

        self.logger.debug(
            f"[VALIDATE] Archivo validado: {resolved_path} ({file_size_mb:.2f} MB)"
        )
        return resolved_path

    def _initialize_parser(self, validated_path: Path) -> ReportParserCrudo:
        """
        Inicializa el parser con manejo robusto de errores.

        Args:
            validated_path: Ruta validada del archivo.

        Returns:
            ReportParserCrudo: Instancia del parser inicializada.
        """
        if not isinstance(validated_path, Path):
            raise ProcessingError(
                f"validated_path debe ser Path, recibido: {type(validated_path).__name__}"
            )

        try:
            parser = ReportParserCrudo(
                file_path=str(validated_path),
                profile=self.profile,
                config=self.config,
            )

            self.logger.debug(f"[INIT_PARSER] Parser inicializado: {validated_path.name}")
            return parser

        except FileNotFoundError as e:
            raise InvalidInputError(f"Archivo no encontrado: {e}") from e
        except PermissionError as e:
            raise InvalidInputError(f"Permiso denegado: {e}") from e
        except Exception as e:
            raise ProcessingError(
                f"Error inicializando ReportParserCrudo: {type(e).__name__}: {e}"
            ) from e

    def _extract_raw_data(
        self, parser: ReportParserCrudo
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extrae datos crudos del parser con validación.

        Args:
            parser: Instancia del parser.

        Returns:
            Tuple[List, Dict]: Lista de registros y caché de parseo.
        """
        # Validación relajada para permitir Mocks en tests
        if not hasattr(parser, "parse_to_raw") or not hasattr(parser, "get_parse_cache"):
            raise ProcessingError(
                f"parser inválido, se esperaba interfaz ReportParserCrudo, "
                f"recibido: {type(parser).__name__}"
            )

        try:
            # Extraer registros crudos
            raw_records = parser.parse_to_raw()

            # Validar tipo de retorno
            if raw_records is None:
                self.logger.warning("[EXTRACT] parse_to_raw() retornó None")
                raw_records = []

            if not isinstance(raw_records, list):
                raise ProcessingError(
                    f"parse_to_raw() debe retornar list, recibido: {type(raw_records).__name__}"
                )

            # Obtener caché
            cache = parser.get_parse_cache()

            if cache is None:
                cache = {}
            elif not isinstance(cache, dict):
                self.logger.warning(
                    f"[EXTRACT] get_parse_cache() retornó {type(cache).__name__}, usando dict vacío"
                )
                cache = {}

            # Limitar tamaño de caché
            if len(cache) > SystemConstants.MAX_CACHE_SIZE:
                self.logger.warning(
                    f"[EXTRACT] Cache muy grande ({len(cache)} entradas), truncando a {SystemConstants.MAX_CACHE_SIZE}"
                )
                # Mantener las últimas entradas
                cache_items = list(cache.items())[-SystemConstants.MAX_CACHE_SIZE :]
                cache = dict(cache_items)

            self.logger.info(
                f"[EXTRACT] Extraídos {len(raw_records)} registros | "
                f"Cache: {len(cache)} entradas"
            )

            return raw_records, cache

        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error extrayendo datos crudos: {e}") from e

    def _process_batches_with_pid(
        self,
        raw_records: List[Dict[str, Any]],
        cache: Dict[str, Any],
        total_records: int,
        on_progress: Optional[Callable[[ProcessingStats], None]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        telemetry: Optional[TelemetryContext] = None,
    ) -> List[pd.DataFrame]:
        """
        Procesamiento por lotes con control PID adaptativo.

        Características:
        - Backoff exponencial para fallos con recuperación gradual.
        - Balanceo de carga adaptativo basado en throughput.
        - Predicción de tiempo restante con ventana móvil.
        - Protección contra pérdida de datos.

        Args:
            raw_records: Lista de registros crudos.
            cache: Caché de parseo.
            total_records: Número total de registros.
            on_progress: Callback para reportar progreso.
            telemetry: Contexto de telemetría opcional.

        Returns:
            List[pd.DataFrame]: Lista de DataFrames procesados.
        """
        # ═══════════════════════════════════════════════════════════════════
        # INICIALIZACIÓN DE ESTADO
        # ═══════════════════════════════════════════════════════════════════
        processed_batches: List[pd.DataFrame] = []
        current_index = 0
        current_batch_size = self.condenser_config.min_batch_size

        # Contadores de fallos
        failed_batches_count = 0
        consecutive_failures = 0
        skipped_records = 0

        # Historial para estadísticas (ventana fija)
        HISTORY_WINDOW = 10
        batch_times: deque = deque(maxlen=HISTORY_WINDOW)
        batch_sizes: deque = deque(maxlen=HISTORY_WINDOW)
        complexity_history: deque = deque(maxlen=HISTORY_WINDOW)

        # Inicializar estado de saturación
        self._last_saturation: float = 0.5  # Valor neutral inicial

        # Algoritmo de backoff
        backoff_factor = 1.0
        max_backoff_factor = 8.0
        min_backoff_batch = max(1, self.condenser_config.min_batch_size // 4)

        # Tiempo de inicio para métricas
        total_start_time = time.time()

        # ═══════════════════════════════════════════════════════════════════
        # CÁLCULO DE LÍMITE DE ITERACIONES DINÁMICO
        # ═══════════════════════════════════════════════════════════════════
        # Límite base: permite cierto overhead por reintentos
        base_iterations = int(
            math.ceil(total_records / max(1, self.condenser_config.min_batch_size))
        )
        # Factor de seguridad para reintentos y batches pequeños
        max_iterations = int(
            base_iterations * SystemConstants.MAX_ITERATIONS_MULTIPLIER * 1.5
        )
        # Límite absoluto para prevenir loops infinitos
        absolute_max = max(max_iterations, total_records * 3)

        self.logger.info(
            f"[PID_LOOP] Iniciando | Registros: {total_records:,} | "
            f"Batch inicial: {current_batch_size} | "
            f"Límite iteraciones: {max_iterations:,}"
        )

        iteration = 0
        while current_index < total_records and iteration < absolute_max:
            iteration += 1
            batch_start_time = time.time()

            # ═══════════════════════════════════════════════════════════════
            # VERIFICACIÓN DE TIMEOUT GLOBAL
            # ═══════════════════════════════════════════════════════════════
            self._check_timeout(f"batch {iteration}")

            # ═══════════════════════════════════════════════════════════════
            # PREDICCIÓN DE TIEMPO RESTANTE
            # ═══════════════════════════════════════════════════════════════
            if len(batch_times) >= 3 and len(batch_sizes) >= 3:
                # Calcular tiempo promedio por registro
                total_batch_time = sum(batch_times)
                total_batch_records = sum(batch_sizes)

                if total_batch_records > 0:
                    avg_time_per_record = total_batch_time / total_batch_records
                    records_remaining = total_records - current_index
                    estimated_time_remaining = records_remaining * avg_time_per_record

                    if iteration % 10 == 0:
                        progress_pct = (current_index / total_records) * 100
                        self.logger.info(
                            f"📈 Progreso: {current_index:,}/{total_records:,} "
                            f"({progress_pct:.1f}%) | ETA: {estimated_time_remaining:.1f}s"
                        )

            # ═══════════════════════════════════════════════════════════════
            # AJUSTE DINÁMICO DE BATCH SIZE BASADO EN RENDIMIENTO
            # ═══════════════════════════════════════════════════════════════
            if len(batch_times) >= 5:
                recent_times = list(batch_times)[-5:]
                recent_sizes = list(batch_sizes)[-5:]

                # Eficiencia = registros por segundo
                efficiencies = [
                    s / max(0.001, t) for t, s in zip(recent_times, recent_sizes)
                ]
                avg_efficiency = sum(efficiencies) / len(efficiencies)

                # Ajustar batch size según eficiencia
                if avg_efficiency < 10:  # < 10 rec/s: reducir
                    reduction = max(0.7, 1.0 - (10 - avg_efficiency) / 100)
                    current_batch_size = max(
                        min_backoff_batch, int(current_batch_size * reduction)
                    )
                    self.logger.debug(
                        f"Eficiencia baja ({avg_efficiency:.1f} rec/s), "
                        f"reduciendo batch a {current_batch_size}"
                    )
                elif (
                    avg_efficiency > 500 and consecutive_failures == 0
                ):  # > 500 rec/s: aumentar
                    current_batch_size = min(
                        self.condenser_config.max_batch_size, int(current_batch_size * 1.2)
                    )

            # ═══════════════════════════════════════════════════════════════
            # EXTRAER LOTE
            # ═══════════════════════════════════════════════════════════════
            end_index = min(current_index + current_batch_size, total_records)
            batch_records = raw_records[current_index:end_index]

            if not batch_records:
                self.logger.warning(f"[PID_LOOP] Batch vacío en índice {current_index}")
                current_index = end_index
                continue

            actual_batch_size = len(batch_records)

            # ═══════════════════════════════════════════════════════════════
            # CÁLCULO DE MÉTRICAS FÍSICAS
            # ═══════════════════════════════════════════════════════════════
            cache_hits = self._calculate_cache_hits(batch_records, cache)

            # Métricas sin factor de posición artificial
            # Usamos el estado global de fallos como proxy de entropía acumulada
            metrics = self.physics.calculate_metrics(
                total_records=actual_batch_size,
                cache_hits=cache_hits,
                error_count=self._stats.failed_records,
                processing_time=max(0.001, time.time() - self._start_time),
            )

            if progress_callback:
                progress_callback(metrics)

            if telemetry:
                # Registrar métricas físicas clave en el pasabordo
                for key, val in metrics.items():
                    # Prefix 'physics_' to avoid collisions if needed, or use namespace
                    if isinstance(val, (int, float)):
                        telemetry.record_metric("flux_condenser_physics", key, val)

            # Actualizar historial de complejidad
            current_complexity = metrics.get("complexity", 0.5)
            complexity_history.append(current_complexity)

            # ═══════════════════════════════════════════════════════════════
            # CONTROL PID CON SUAVIZADO
            # ═══════════════════════════════════════════════════════════════
            saturation = metrics.get("saturation", 0.5)

            # Suavizado exponencial para evitar cambios bruscos
            alpha = 0.3  # Factor de suavizado (0 = sin cambio, 1 = cambio completo)
            smoothed_saturation = alpha * saturation + (1 - alpha) * self._last_saturation

            # Limitar cambio máximo por iteración
            max_change = 0.2
            if abs(smoothed_saturation - self._last_saturation) > max_change:
                smoothed_saturation = self._last_saturation + math.copysign(
                    max_change, smoothed_saturation - self._last_saturation
                )

            pid_output = self.controller.compute(smoothed_saturation)
            self._last_saturation = smoothed_saturation

            # ═══════════════════════════════════════════════════════════════
            # FRENO DE EMERGENCIA POR SOBRECALENTAMIENTO
            # ═══════════════════════════════════════════════════════════════
            dissipated_power = metrics.get("dissipated_power", 0.0)
            if dissipated_power > SystemConstants.OVERHEAT_POWER_THRESHOLD:
                self.logger.warning(
                    f"🔥 [OVERHEAT] P={dissipated_power:.1f}W > "
                    f"{SystemConstants.OVERHEAT_POWER_THRESHOLD}W - Aplicando freno"
                )
                pid_output = max(
                    self.condenser_config.min_batch_size,
                    int(pid_output * SystemConstants.EMERGENCY_BRAKE_FACTOR),
                )
                self._stats.emergency_brakes_triggered += 1

            # ═══════════════════════════════════════════════════════════════
            # PROCESAR BATCH
            # ═══════════════════════════════════════════════════════════════
            batch_result = self._process_single_batch(
                batch_records, cache, current_index, end_index
            )

            batch_time = time.time() - batch_start_time

            # ═══════════════════════════════════════════════════════════════
            # MANEJO DE RESULTADO
            # ═══════════════════════════════════════════════════════════════
            if batch_result.success:
                # Reset de estado de fallo
                consecutive_failures = 0
                backoff_factor = max(1.0, backoff_factor * 0.5)  # Recuperación gradual

                if batch_result.dataframe is not None and not batch_result.dataframe.empty:
                    processed_batches.append(batch_result.dataframe)
                    self._stats.processed_records += batch_result.records_processed

                # Actualizar historial
                batch_times.append(batch_time)
                batch_sizes.append(actual_batch_size)

                # Calcular nuevo batch size basado en PID
                new_batch_size = int(pid_output)

            else:
                consecutive_failures += 1
                failed_batches_count += 1

                # Backoff exponencial con límite
                backoff_factor = min(max_backoff_factor, backoff_factor * 2.0)

                # Reducir batch size con jitter para evitar patrones repetitivos
                jitter = 0.9 + (hash(str(iteration)) % 20) / 100.0  # 0.9-1.09
                reduced_size = int(current_batch_size / backoff_factor * jitter)
                new_batch_size = max(min_backoff_batch, reduced_size)

                self.logger.warning(
                    f"[BATCH_FAIL] #{consecutive_failures} en índice {current_index}. "
                    f"Backoff: {backoff_factor:.1f}x, próximo batch: {new_batch_size}. "
                    f"Error: {batch_result.error_message[:100] if batch_result.error_message else 'N/A'}"
                )

                # Estrategia de recuperación por fallos múltiples
                if consecutive_failures >= 5:
                    # Registrar registros que serán saltados
                    skip_count = min(actual_batch_size, 10)
                    skipped_records += skip_count
                    self.logger.error(
                        f"[RECOVERY] {consecutive_failures} fallos consecutivos. "
                        f"Saltando {skip_count} registros problemáticos."
                    )

                    # Avanzar parcialmente (no saltar todo el batch)
                    current_index += skip_count
                    new_batch_size = min_backoff_batch
                    consecutive_failures = 3  # Reducir pero no resetear

            # ═══════════════════════════════════════════════════════════════
            # ACTUALIZAR ESTADÍSTICAS
            # ═══════════════════════════════════════════════════════════════
            self._stats.add_batch_stats(
                batch_size=actual_batch_size,
                saturation=smoothed_saturation,
                power=dissipated_power,
                flyback=metrics.get("flyback_voltage", 0.0),
                kinetic=metrics.get("kinetic_energy", 0.0),
                success=batch_result.success,
            )

            # ═══════════════════════════════════════════════════════════════
            # LOGGING ADAPTATIVO Y CALLBACK DE PROGRESO
            # ═══════════════════════════════════════════════════════════════
            log_frequency = max(1, min(50, total_records // 100))
            if iteration % log_frequency == 0 or iteration <= 3 or consecutive_failures > 0:
                throughput = actual_batch_size / max(0.001, batch_time)
                self.logger.info(
                    f"🔄 [{iteration}] Idx: {current_index:,}-{end_index:,} | "
                    f"Sat: {smoothed_saturation:.1%} | "
                    f"Time: {batch_time:.3f}s | "
                    f"Speed: {throughput:.0f} rec/s | "
                    f"Next: {new_batch_size:,}"
                )

            # Reportar progreso en tiempo real
            if on_progress:
                try:
                    on_progress(self._stats)
                except Exception as e:
                    self.logger.warning(f"⚠️ Error en callback de progreso: {e}")

            # ═══════════════════════════════════════════════════════════════
            # PREPARAR SIGUIENTE ITERACIÓN
            # ═══════════════════════════════════════════════════════════════
            current_index = end_index
            current_batch_size = max(
                self.condenser_config.min_batch_size,
                min(self.condenser_config.max_batch_size, new_batch_size),
            )

        # ═══════════════════════════════════════════════════════════════════
        # VALIDACIÓN DE TERMINACIÓN
        # ═══════════════════════════════════════════════════════════════════
        if iteration >= absolute_max:
            self.logger.error(
                f"[PID_LOOP] Límite de iteraciones alcanzado: {iteration}. "
                f"Procesados: {current_index}/{total_records}"
            )

        # ═══════════════════════════════════════════════════════════════════
        # RESUMEN FINAL
        # ═══════════════════════════════════════════════════════════════════
        total_time = time.time() - total_start_time
        overall_throughput = self._stats.processed_records / max(0.001, total_time)

        # Calcular complejidad promedio real
        avg_complexity = (
            sum(complexity_history) / len(complexity_history) if complexity_history else 0.5
        )

        self.logger.info(
            f"✅ [PID_LOOP] Completado en {total_time:.1f}s | "
            f"Throughput: {overall_throughput:.1f} rec/s | "
            f"Batches: {self._stats.total_batches} (fallidos: {failed_batches_count}) | "
            f"Saltados: {skipped_records} | "
            f"Complejidad promedio: {avg_complexity:.2f}"
        )

        return processed_batches

    def _process_single_batch(
        self,
        batch_records: List[Dict[str, Any]],
        cache: Dict[str, Any],
        start_idx: int,
        end_idx: int,
    ) -> BatchResult:
        """
        Procesa un batch individual con manejo de errores.

        Args:
            batch_records: Registros del batch.
            cache: Caché de parseo.
            start_idx: Índice inicial.
            end_idx: Índice final.

        Returns:
            BatchResult: Resultado del procesamiento del batch.
        """
        try:
            batch_data = ParsedData(batch_records, cache)
            df_batch = self._rectify_signal(batch_data)

            if not isinstance(df_batch, pd.DataFrame):
                return BatchResult(
                    success=False,
                    error_message=f"Resultado no es DataFrame: {type(df_batch).__name__}",
                    records_processed=0,
                )

            return BatchResult(
                success=True,
                dataframe=df_batch,
                records_processed=len(df_batch),
            )

        except Exception as e:
            return BatchResult(
                success=False,
                error_message=str(e),
                records_processed=0,
            )

    def _calculate_cache_hits(
        self,
        batch_records: List[Dict[str, Any]],
        cache: Dict[str, Any],
    ) -> int:
        """
        Cálculo eficiente de cache hits con múltiples estrategias.

        Estrategias de búsqueda (en orden de prioridad):
        1. Coincidencia exacta por clave canónica
        2. Coincidencia por hash de contenido
        3. Coincidencia parcial limitada (solo si cache es pequeño)

        Args:
            batch_records: Registros del batch actual.
            cache: Caché disponible.

        Returns:
            int: Número de hits en caché (entero, redondeado hacia arriba).
        """
        # Validaciones tempranas
        if not cache or not isinstance(cache, dict):
            return 0

        if not batch_records or not isinstance(batch_records, list):
            return 0

        cache_size = len(cache)
        batch_size = len(batch_records)

        # ═══════════════════════════════════════════════════════════════════
        # PREPARAR ÍNDICE DE CACHE (con invalidación)
        # ═══════════════════════════════════════════════════════════════════
        # Verificar si necesitamos reconstruir el índice
        cache_hash = hash(frozenset(cache.keys())) if cache_size < 10000 else id(cache)

        if (
            not hasattr(self, "_cache_index")
            or not hasattr(self, "_cache_hash")
            or self._cache_hash != cache_hash
        ):
            self._cache_hash = cache_hash
            self._cache_index = {
                "exact": set(),  # Claves normalizadas para búsqueda exacta
                "hashes": set(),  # Hashes conocidos
                "prefixes": set(),  # Prefijos de 20 chars (para búsqueda rápida)
            }

            for key, value in cache.items():
                if isinstance(key, str) and len(key) > 3:
                    normalized = key.lower().strip()
                    self._cache_index["exact"].add(normalized)

                    # Almacenar prefijo para búsqueda rápida
                    if len(normalized) >= 20:
                        self._cache_index["prefixes"].add(normalized[:20])

                # Indexar valores hash si existen
                if isinstance(value, dict) and "hash" in value:
                    h = value["hash"]
                    if isinstance(h, (str, int)):
                        self._cache_index["hashes"].add(str(h))

        # ═══════════════════════════════════════════════════════════════════
        # CONTAR HITS
        # ═══════════════════════════════════════════════════════════════════
        exact_hits = 0
        partial_hits = 0.0

        # Claves a buscar en registros
        search_keys = ("insumo_line", "line", "raw_line", "content", "text", "data", "key")

        for record in batch_records:
            if not isinstance(record, dict):
                continue

            found_hit = False

            # Estrategia 1: Búsqueda exacta por claves conocidas
            for key in search_keys:
                if key not in record:
                    continue

                content = record[key]
                if not isinstance(content, str) or len(content) <= 3:
                    continue

                normalized = content.lower().strip()

                if normalized in self._cache_index["exact"]:
                    exact_hits += 1
                    found_hit = True
                    break

            if found_hit:
                continue

            # Estrategia 2: Búsqueda por hash
            record_hash = record.get("hash")
            if record_hash is not None:
                str_hash = str(record_hash)
                if str_hash in self._cache_index["hashes"]:
                    exact_hits += 1
                    continue

            # Estrategia 3: Búsqueda por prefijo (solo si cache pequeño)
            if cache_size < 1000 and self._cache_index["prefixes"]:
                for key in search_keys:
                    if key not in record:
                        continue

                    content = record[key]
                    if not isinstance(content, str) or len(content) < 20:
                        continue

                    prefix = content.lower().strip()[:20]
                    if prefix in self._cache_index["prefixes"]:
                        partial_hits += 0.5
                        break

        # ═══════════════════════════════════════════════════════════════════
        # CALCULAR RESULTADO FINAL
        # ═══════════════════════════════════════════════════════════════════
        total_hits = exact_hits + partial_hits

        # Logging de diagnóstico para ratios anómalos
        if batch_size > 50:
            hit_ratio = total_hits / batch_size
            if hit_ratio < 0.05:
                self.logger.debug(
                    f"[CACHE] Hit ratio bajo: {hit_ratio:.1%} "
                    f"({total_hits:.1f}/{batch_size})"
                )
            elif hit_ratio > 0.95:
                self.logger.debug(
                    f"[CACHE] Hit ratio muy alto: {hit_ratio:.1%} - verificar cache"
                )

        # Redondear hacia arriba para no subestimar hits
        return int(math.ceil(total_hits))

    def _consolidate_results(self, processed_batches: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Consolida múltiples DataFrames con validación de esquema.

        Args:
            processed_batches: Lista de DataFrames de los lotes procesados.

        Returns:
            pd.DataFrame: DataFrame consolidado y validado.

        Raises:
            ProcessingError: Si hay error de consolidación irrecuperable.
        """
        if not processed_batches:
            self.logger.warning("[CONSOLIDATE] Sin batches para consolidar")
            return pd.DataFrame()

        if not isinstance(processed_batches, list):
            raise ProcessingError(
                f"processed_batches debe ser list, recibido: {type(processed_batches).__name__}"
            )

        # ═══════════════════════════════════════════════════════════════════
        # FILTRAR Y VALIDAR BATCHES
        # ═══════════════════════════════════════════════════════════════════
        valid_batches: List[pd.DataFrame] = []
        total_rows_before = 0
        column_sets: List[Set[str]] = []
        skipped_reasons: Dict[str, int] = {}

        for i, batch in enumerate(processed_batches):
            # Validar tipo
            if not isinstance(batch, pd.DataFrame):
                reason = f"no_dataframe_{type(batch).__name__}"
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                continue

            # Saltar vacíos
            if batch.empty:
                skipped_reasons["empty"] = skipped_reasons.get("empty", 0) + 1
                continue

            # Validar que tiene columnas
            if len(batch.columns) == 0:
                skipped_reasons["no_columns"] = skipped_reasons.get("no_columns", 0) + 1
                continue

            valid_batches.append(batch)
            total_rows_before += len(batch)
            column_sets.append(set(batch.columns))

        if skipped_reasons:
            self.logger.warning(f"[CONSOLIDATE] Batches saltados: {skipped_reasons}")

        if not valid_batches:
            self.logger.warning("[CONSOLIDATE] Todos los batches inválidos o vacíos")
            return pd.DataFrame()

        # ═══════════════════════════════════════════════════════════════════
        # VERIFICAR LÍMITE DE BATCHES
        # ═══════════════════════════════════════════════════════════════════
        if len(valid_batches) > SystemConstants.MAX_BATCHES_TO_CONSOLIDATE:
            self.logger.warning(
                f"[CONSOLIDATE] Demasiados batches ({len(valid_batches)}). "
                f"Submuestreando uniformemente a {SystemConstants.MAX_BATCHES_TO_CONSOLIDATE}"
            )
            # Submuestreo uniforme en lugar de truncar al final
            step = len(valid_batches) / SystemConstants.MAX_BATCHES_TO_CONSOLIDATE
            indices = [
                int(i * step) for i in range(SystemConstants.MAX_BATCHES_TO_CONSOLIDATE)
            ]
            valid_batches = [valid_batches[i] for i in indices]

        # ═══════════════════════════════════════════════════════════════════
        # ANALIZAR CONSISTENCIA DE ESQUEMAS
        # ═══════════════════════════════════════════════════════════════════
        if len(column_sets) > 1:
            # Encontrar columnas comunes y divergentes
            common_columns = column_sets[0].intersection(*column_sets[1:])
            all_columns = column_sets[0].union(*column_sets[1:])
            extra_columns = all_columns - common_columns

            if extra_columns:
                self.logger.info(
                    f"[CONSOLIDATE] Esquemas divergentes. "
                    f"Comunes: {len(common_columns)}, Extras: {len(extra_columns)}"
                )

                # Las columnas extras serán NaN donde no existan
                if len(extra_columns) > len(common_columns):
                    self.logger.warning(
                        "[CONSOLIDATE] Más columnas extras que comunes, "
                        "posible inconsistencia de datos"
                    )

        # ═══════════════════════════════════════════════════════════════════
        # CONCATENAR CON MANEJO DE MEMORIA
        # ═══════════════════════════════════════════════════════════════════
        try:
            # Estimar memoria requerida
            estimated_rows = sum(len(b) for b in valid_batches)
            avg_cols = sum(len(b.columns) for b in valid_batches) / len(valid_batches)

            # Si es muy grande, concatenar en chunks
            CHUNK_SIZE = 50
            if len(valid_batches) > CHUNK_SIZE:
                self.logger.info(
                    f"[CONSOLIDATE] Concatenación por chunks ({len(valid_batches)} batches)"
                )

                intermediate_results = []
                for i in range(0, len(valid_batches), CHUNK_SIZE):
                    chunk = valid_batches[i : i + CHUNK_SIZE]
                    chunk_df = pd.concat(chunk, ignore_index=True, sort=False)
                    intermediate_results.append(chunk_df)

                df_final = pd.concat(intermediate_results, ignore_index=True, sort=False)
            else:
                df_final = pd.concat(valid_batches, ignore_index=True, sort=False)

            # Validar resultado
            total_rows_after = len(df_final)

            if total_rows_after < total_rows_before * 0.95:
                self.logger.warning(
                    f"[CONSOLIDATE] Pérdida de filas durante concatenación: "
                    f"{total_rows_before} → {total_rows_after} "
                    f"({(1 - total_rows_after / total_rows_before) * 100:.1f}% perdido)"
                )

            self.logger.info(
                f"[CONSOLIDATE] {len(valid_batches)} batches → "
                f"{len(df_final):,} registros, {len(df_final.columns)} columnas"
            )

            return df_final

        except MemoryError as e:
            raise ProcessingError(
                f"Error de memoria consolidando {len(valid_batches)} batches "
                f"(~{estimated_rows:,} filas): {e}"
            ) from e
        except Exception as e:
            raise ProcessingError(
                f"Error consolidando resultados: {type(e).__name__}: {e}"
            ) from e

    def _rectify_signal(self, parsed_data: ParsedData) -> pd.DataFrame:
        """
        Convierte datos parseados usando APUProcessor.

        Args:
            parsed_data: Datos parseados crudos y cache.

        Returns:
            pd.DataFrame: DataFrame con los datos procesados.
        """
        if not isinstance(parsed_data, ParsedData):
            raise ProcessingError(
                f"parsed_data debe ser ParsedData, recibido: {type(parsed_data).__name__}"
            )

        if not parsed_data.raw_records:
            return pd.DataFrame()

        try:
            # Crear processor
            processor = APUProcessor(
                config=self.config,
                profile=self.profile,
                parse_cache=parsed_data.parse_cache,
            )

            # Asignar registros
            processor.raw_records = parsed_data.raw_records

            # Procesar
            df_result = processor.process_all()

            # Validar resultado
            if df_result is None:
                self.logger.warning("[RECTIFY] process_all() retornó None")
                return pd.DataFrame()

            if not isinstance(df_result, pd.DataFrame):
                raise ProcessingError(
                    f"process_all() debe retornar DataFrame, recibido: {type(df_result).__name__}"
                )

            return df_result

        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Error en rectificación: {e}") from e

    def _validate_output(self, df: pd.DataFrame) -> None:
        """
        Valida el DataFrame de salida con múltiples criterios de calidad.

        Args:
            df: DataFrame a validar.

        Raises:
            ProcessingError: Si el DataFrame no pasa validación estricta.
        """
        if not isinstance(df, pd.DataFrame):
            raise ProcessingError(
                f"Salida debe ser DataFrame, recibido: {type(df).__name__}"
            )

        # DataFrame vacío es válido (pero se registra)
        if df.empty:
            self.logger.warning("[VALIDATE_OUTPUT] DataFrame de salida vacío")
            return

        total_rows = len(df)
        total_cols = len(df.columns)

        # ═══════════════════════════════════════════════════════════════════
        # ANÁLISIS DE CALIDAD DE DATOS
        # ═══════════════════════════════════════════════════════════════════
        quality_issues: List[str] = []
        warnings: List[str] = []

        # Análisis por columna
        null_analysis = {}
        dtype_analysis = {}

        for col in df.columns:
            try:
                col_data = df[col]
                null_count = col_data.isnull().sum()
                null_ratio = null_count / total_rows

                null_analysis[col] = {
                    "count": int(null_count),
                    "ratio": round(null_ratio, 4),
                }

                # Clasificar problemas
                if null_ratio == 1.0:
                    quality_issues.append(f"Columna '{col}': 100% nulos")
                elif null_ratio > 0.9:
                    warnings.append(f"Columna '{col}': {null_ratio:.1%} nulos")

                # Analizar tipos de datos
                dtype_analysis[col] = str(col_data.dtype)

                # Detectar columnas con un solo valor único (posible constante o error)
                if null_ratio < 1.0:
                    nunique = col_data.nunique(dropna=True)
                    if nunique == 1 and total_rows > 10:
                        warnings.append(f"Columna '{col}': solo 1 valor único (constante)")
                    elif nunique == total_rows and total_rows > 100:
                        # Todos valores únicos - posible ID o timestamp
                        pass  # Esto es esperado para algunas columnas

            except Exception as e:
                warnings.append(f"Error analizando columna '{col}': {e}")

        # ═══════════════════════════════════════════════════════════════════
        # ANÁLISIS GLOBAL
        # ═══════════════════════════════════════════════════════════════════

        # Verificar filas duplicadas
        try:
            duplicate_count = df.duplicated().sum()
            duplicate_ratio = duplicate_count / total_rows
            if duplicate_ratio > 0.5:
                warnings.append(f"Alto ratio de duplicados: {duplicate_ratio:.1%}")
        except Exception:
            pass  # Algunos DataFrames no soportan duplicated()

        # Verificar uso de memoria
        try:
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            if memory_mb > 500:
                warnings.append(f"DataFrame grande en memoria: {memory_mb:.1f} MB")
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════════
        # REPORTAR RESULTADOS
        # ═══════════════════════════════════════════════════════════════════
        if warnings:
            self.logger.warning(
                f"[VALIDATE_OUTPUT] {len(warnings)} advertencias:\n  - "
                + "\n  - ".join(warnings[:10])  # Limitar a 10
            )

        if quality_issues:
            issue_msg = (
                f"[VALIDATE_OUTPUT] {len(quality_issues)} problemas de calidad:\n  - "
                + "\n  - ".join(quality_issues[:5])
            )

            if self.condenser_config.enable_strict_validation:
                self.logger.error(issue_msg)
                raise ProcessingError(
                    f"Validación estricta fallida: {len(quality_issues)} problemas de calidad"
                )
            else:
                self.logger.warning(issue_msg)

        # Resumen positivo si todo está bien
        if not quality_issues and not warnings:
            self.logger.debug(
                f"[VALIDATE_OUTPUT] OK: {total_rows:,} filas × {total_cols} columnas"
            )

    def _log_final_stats(self) -> None:
        """Registra estadísticas finales con formato estructurado."""
        success_rate = (
            self._stats.processed_records / self._stats.total_records * 100
            if self._stats.total_records > 0
            else 0.0
        )

        self.logger.info(
            f"\n{'═' * 70}\n"
            f"📊 ESTADÍSTICAS FINALES\n"
            f"{'═' * 70}\n"
            f"  Total registros:       {self._stats.total_records:,}\n"
            f"  Procesados:            {self._stats.processed_records:,} ({success_rate:.1f}%)\n"
            f"  Fallidos:              {self._stats.failed_records:,}\n"
            f"  Batches:               {self._stats.total_batches} (fallidos: {self._stats.failed_batches})\n"
            f"  Tiempo:                {self._stats.processing_time:.2f}s\n"
            f"  Batch promedio:        {self._stats.avg_batch_size:.0f}\n"
            f"  Saturación promedio:   {self._stats.avg_saturation:.1%}\n"
            f"  Potencia máx:          {self._stats.max_dissipated_power:.1f}W\n"
            f"  Frenos emergencia:     {self._stats.emergency_brakes_triggered}\n"
            f"{'═' * 70}"
        )

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas completas del procesamiento.

        Returns:
            Dict[str, Any]: Diccionario con configuración, estadísticas, estado del controlador y física.
        """
        success_rate = (
            self._stats.processed_records / self._stats.total_records
            if self._stats.total_records > 0
            else 0.0
        )

        return {
            "configuration": {
                "min_records_threshold": self.condenser_config.min_records_threshold,
                "strict_validation": self.condenser_config.enable_strict_validation,
                "partial_recovery": self.condenser_config.enable_partial_recovery,
                "batch_range": [
                    self.condenser_config.min_batch_size,
                    self.condenser_config.max_batch_size,
                ],
            },
            "statistics": {
                "total_records": self._stats.total_records,
                "processed_records": self._stats.processed_records,
                "failed_records": self._stats.failed_records,
                "total_batches": self._stats.total_batches,
                "failed_batches": self._stats.failed_batches,
                "processing_time": self._stats.processing_time,
                "avg_batch_size": self._stats.avg_batch_size,
                "avg_saturation": self._stats.avg_saturation,
                "max_dissipated_power": self._stats.max_dissipated_power,
                "emergency_brakes": self._stats.emergency_brakes_triggered,
                "success_rate": success_rate,
            },
            "controller": self.controller.get_state(),
            "controller_diagnostics": self.controller.get_diagnostics(),
            "physics_trends": self.physics.get_trend_analysis(),
        }

    def reset(self) -> None:
        """
        Resetea completamente el estado del condensador para reutilización.

        Limpia:
        - Estado del controlador PID
        - Estadísticas de procesamiento
        - Caches internos
        - Variables temporales
        """
        # Reset controlador PID
        self.controller.reset()

        # Reset estadísticas
        self._stats = ProcessingStats()
        self._start_time = None

        # Limpiar caches
        if hasattr(self, "_cache_keys_normalized"):
            self._cache_keys_normalized.clear()
            del self._cache_keys_normalized

        if hasattr(self, "_cache_index"):
            del self._cache_index

        if hasattr(self, "_cache_hash"):
            del self._cache_hash

        # Limpiar estado temporal
        if hasattr(self, "_last_saturation"):
            del self._last_saturation

        # Reset del motor de física también
        if hasattr(self.physics, "_last_current"):
            self.physics._last_current = 0.0
            self.physics._last_time = time.time()
            self.physics._initialized_temporal = False

        self.logger.info("[RESET] Condensador reseteado completamente")
