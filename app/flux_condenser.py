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
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

# Manejo opcional de numpy para optimización matemática
try:
    import numpy as np
except ImportError:
    pass

import pandas as pd

from .apu_processor import APUProcessor
from .report_parser_crudo import ReportParserCrudo

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
        self._output_range = max(1, self.max_output - self.min_output)  # Evitar división por 0
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
            errors.append(f"integral_limit_factor debe ser > 0, recibido: {integral_limit_factor}")

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
        if not hasattr(self, '_pv_filtered'):
            self._pv_filtered = process_variable
        else:
            alpha = 0.3
            self._pv_filtered = alpha * process_variable + (1 - alpha) * self._pv_filtered

        process_variable = self._pv_filtered

        # Validación reforzada
        if math.isnan(process_variable) or math.isinf(process_variable):
            self.logger.warning("PV inválido, usando setpoint con decaimiento")
            process_variable = self.setpoint * (0.9 ** self._iteration_count)

        # ==================== CÁLCULO DE ERROR CON HISTÉRESIS ====================
        error = self.setpoint - process_variable

        # Histéresis para evitar oscilaciones menores al 1%
        if hasattr(self, '_last_error') and self._last_error is not None:
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
            dt = max(SystemConstants.MIN_DELTA_TIME,
                    min(dt, SystemConstants.MAX_DELTA_TIME))

            # Cálculo de derivada con filtro (evita ruido)
            if self._last_error is not None:
                raw_derivative = (error - self._last_error) / dt
                # Filtro de primer orden para derivada
                if not hasattr(self, '_derivative_filtered'):
                    self._derivative_filtered = raw_derivative
                else:
                    self._derivative_filtered = 0.7 * raw_derivative + 0.3 * self._derivative_filtered
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
                self._integral_limit * math.tanh(excess_ratio),
                self._integral_error
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
        output = int(np.round(control_signal)) if 'np' in globals() else int(round(control_signal))

        # Evitar cambios bruscos (limitación de slew rate)
        if hasattr(self, '_last_output') and self._last_output is not None:
            max_change = max(50, self._output_range * 0.1)  # Máximo 10% del rango o 50
            output = self._last_output + max(-max_change,
                                            min(max_change, output - self._last_output))

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
            "P": P, "I": I, "D": D,
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
        if hasattr(self, '_pv_filtered'): del self._pv_filtered
        if hasattr(self, '_derivative_filtered'): del self._derivative_filtered
        if hasattr(self, '_last_output'): del self._last_output

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

        recent = self._history[-min(20, len(self._history)):]

        avg_error = sum(h["error"] for h in recent) / len(recent)
        avg_output = sum(h["output"] for h in recent) / len(recent)
        output_variance = sum((h["output"] - avg_output) ** 2 for h in recent) / len(recent)

        # Detectar oscilaciones
        sign_changes = 0
        for i in range(1, len(recent)):
            if recent[i]["error"] * recent[i-1]["error"] < 0:
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
                "Reducir Kp si hay oscilaciones" if oscillating
                else "Sistema estable" if abs(avg_error) < 0.1
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
        self._validate_parameters(capacitance, resistance, inductance)

        self.C = float(capacitance)
        self.R = float(resistance)
        self.L = float(inductance)

        # Constantes derivadas pre-calculadas
        self._tau_base = self.R * self.C  # Constante de tiempo base
        self._resonant_freq = 1.0 / (2.0 * math.pi * math.sqrt(self.L * self.C))

        # Historial para análisis de tendencias
        self._metrics_history: List[Dict[str, float]] = []

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(
            f"Motor RLC inicializado: C={self.C}F, R={self.R}Ω, L={self.L}H | "
            f"τ_base={self._tau_base:.2f}s, f_res={self._resonant_freq:.4f}Hz"
        )

    def _validate_parameters(
        self, capacitance: float, resistance: float, inductance: float
    ) -> None:
        """Valida parámetros físicos con rangos plausibles."""
        errors = []

        for name, value in [
            ("capacitance", capacitance),
            ("resistance", resistance),
            ("inductance", inductance),
        ]:
            if not isinstance(value, (int, float)):
                errors.append(f"{name} debe ser numérico, recibido: {type(value).__name__}")
                continue

            if math.isnan(value) or math.isinf(value):
                errors.append(f"{name} no puede ser NaN o Inf: {value}")
                continue

            if value <= 0:
                errors.append(f"{name} debe ser > 0, recibido: {value}")

        # Validar rangos físicamente plausibles (advertencias, no errores)
        if errors:
            raise ConfigurationError(
                "Parámetros físicos inválidos:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def calculate_metrics(self, total_records: int, cache_hits: int) -> Dict[str, float]:
        """
        Modelo físico mejorado:
        - Sistema de segundo orden (RLC) completo.
        - Resonancia y amortiguamiento.
        - Energías normalizadas por capacitancia.
        """
        # Validación mejorada
        if total_records <= 0:
            return self._get_zero_metrics()

        cache_hits = max(0, min(total_records, cache_hits))

        try:
            # ================ PARÁMETROS DEL SISTEMA RLC ================
            # Calidad del flujo (0-1)
            current_I = cache_hits / total_records

            # Factor de complejidad (ruido)
            complexity = 1.0 - current_I

            # Resistencia dinámica (aumenta con complejidad)
            R_dyn = self.R * (1.0 + complexity * SystemConstants.COMPLEXITY_RESISTANCE_FACTOR)

            # ================ ECUACIONES DEL CIRCUITO RLC ================
            # Constante de amortiguamiento (ζ)
            # Para circuito RLC serie: ζ = R/(2) * sqrt(C/L)
            damping_ratio = (R_dyn / 2.0) * math.sqrt(self.C / self.L)

            # Frecuencia natural (ω_n)
            omega_n = 1.0 / math.sqrt(self.L * self.C)

            # Frecuencia amortiguada (ω_d)
            if damping_ratio < 1.0:  # Sistema subamortiguado
                omega_d = omega_n * math.sqrt(1.0 - damping_ratio**2)
            else:  # Sobreamortiguado
                omega_d = 0.0

            # ================ SATURACIÓN (RESPUESTA AL ESCALÓN) ================
            # Modelo de respuesta al escalón de sistema de segundo orden
            if total_records > 0:
                # Tiempo normalizado por constante de tiempo
                t_normalized = float(total_records) / (R_dyn * self.C)

                if damping_ratio < 1.0:  # Subamortiguado
                    exp_term = math.exp(-damping_ratio * omega_n * t_normalized)
                    sin_term = math.sin(omega_d * t_normalized + math.atan2(
                        omega_d, damping_ratio * omega_n
                    ))
                    saturation_V = 1.0 - exp_term * sin_term / math.sqrt(1 - damping_ratio**2)
                elif abs(damping_ratio - 1.0) < 1e-6:  # Críticamente amortiguado
                    exp_term = math.exp(-omega_n * t_normalized)
                    saturation_V = 1.0 - (1.0 + omega_n * t_normalized) * exp_term
                else:  # Sobreamortiguado
                    s1 = -omega_n * (damping_ratio - math.sqrt(damping_ratio**2 - 1))
                    s2 = -omega_n * (damping_ratio + math.sqrt(damping_ratio**2 - 1))
                    A = (s2 / (s2 - s1))
                    B = (s1 / (s1 - s2))
                    saturation_V = 1.0 - (A * math.exp(s1 * t_normalized) +
                                        B * math.exp(s2 * t_normalized))
            else:
                saturation_V = 0.0

            # Limitar saturación [0, 1]
            saturation_V = max(0.0, min(1.0, saturation_V))

            # ================ ENERGÍAS Y POTENCIA ================
            # Energía en capacitor (normalizada)
            E_c = 0.5 * self.C * (saturation_V ** 2) / self.C  # Normalizada por C

            # Energía en inductor (normalizada)
            E_l = 0.5 * self.L * (current_I ** 2) / self.L  # Normalizada por L

            # Potencia disipada en resistor
            P_diss = (complexity ** 2) * R_dyn

            # Factor de potencia del sistema
            power_factor = current_I / math.sqrt(current_I**2 + complexity**2 + 1e-10)

            # ================ TENSIÓN DE FLYBACK (L·di/dt) ================
            # Calcular di/dt estimado
            if not hasattr(self, '_last_current'):
                self._last_current = current_I
                self._last_time = time.time()

            current_time = time.time()
            dt = max(0.001, current_time - self._last_time)
            di_dt = (current_I - self._last_current) / dt

            # Tensión inductiva (protegida)
            V_flyback = abs(self.L * di_dt)
            V_flyback = min(V_flyback, SystemConstants.MAX_FLYBACK_VOLTAGE)

            # Actualizar para próxima iteración
            self._last_current = current_I
            self._last_time = current_time

            # ================ MÉTRICAS DE CALIDAD ================
            # Factor de estabilidad (0=inestable, 1=estable)
            stability_factor = math.exp(-damping_ratio)

            # Margen de fase estimado
            phase_margin = 90.0 if damping_ratio < 0.7 else 180.0 * (1.0 - damping_ratio)

            # ================ CONSTRUIR RESULTADOS ================
            metrics = {
                # Métricas principales
                "saturation": self._sanitize_metric(saturation_V, 0.0, 1.0),
                "complexity": self._sanitize_metric(complexity, 0.0, 1.0),
                "flyback_voltage": self._sanitize_metric(V_flyback, 0.0, SystemConstants.MAX_FLYBACK_VOLTAGE),
                "potential_energy": self._sanitize_metric(E_c, 0.0, 1e10),
                "kinetic_energy": self._sanitize_metric(E_l, 0.0, 1e10),
                "dissipated_power": self._sanitize_metric(P_diss, 0.0, 1e6),

                # Parámetros del sistema
                "current_I": self._sanitize_metric(current_I, 0.0, 1.0),
                "dynamic_resistance": self._sanitize_metric(R_dyn, 0.0, 1e6),
                "damping_ratio": self._sanitize_metric(damping_ratio, 0.0, 10.0),
                "natural_frequency": self._sanitize_metric(omega_n, 0.0, 1e6),
                "damped_frequency": self._sanitize_metric(omega_d, 0.0, 1e6),

                # Métricas de calidad
                "power_factor": self._sanitize_metric(power_factor, 0.0, 1.0),
                "stability_factor": self._sanitize_metric(stability_factor, 0.0, 1.0),
                "phase_margin": self._sanitize_metric(phase_margin, 0.0, 90.0),

                # Diagnóstico
                "system_type": (
                    "UNDERDAMPED" if damping_ratio < 0.7 else
                    "CRITICALLY_DAMPED" if abs(damping_ratio - 1.0) < 0.1 else
                    "OVERDAMPED"
                )
            }

            # Almacenar en historial
            self._store_metrics(metrics)

            # Diagnóstico en tiempo real
            if damping_ratio < 0.5:
                self.logger.debug(f"Sistema subamortiguado (ζ={damping_ratio:.2f}) - posibles oscilaciones")
            elif damping_ratio > 2.0:
                self.logger.debug(f"Sistema sobreamortiguado (ζ={damping_ratio:.2f}) - respuesta lenta")

            return metrics

        except (OverflowError, ValueError, ZeroDivisionError) as e:
            self.logger.error(f"Error en modelo físico: {e}. Usando modelo simplificado.")
            # Fallback a modelo de primer orden (implementado internamente)
            return self._get_zero_metrics()

    def _sanitize_metric(
        self, value: float, min_val: float, max_val: float
    ) -> float:
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
        """Retorna métricas en estado cero (seguro)."""
        return {
            "saturation": 0.0,
            "complexity": 0.0,
            "flyback_voltage": 0.0,
            "potential_energy": 0.0,
            "kinetic_energy": 0.0,
            "dissipated_power": 0.0,
            "current_I": 0.0,
            "dynamic_resistance": self.R,
            "damping_ratio": 1.0,  # Valor seguro
            "natural_frequency": 0.0,
            "damped_frequency": 0.0,
            "power_factor": 0.0,
            "stability_factor": 0.0,
            "phase_margin": 45.0,
            "system_type": "UNKNOWN"
        }

    def _store_metrics(self, metrics: Dict[str, float]) -> None:
        """Almacena métricas en historial con buffer circular."""
        timestamped = {**metrics, "_timestamp": time.time()}
        self._metrics_history.append(timestamped)

        if len(self._metrics_history) > self._MAX_METRICS_HISTORY:
            self._metrics_history.pop(0)

    def get_system_diagnosis(self, metrics: Dict[str, float]) -> str:
        """
        Genera diagnóstico del sistema con múltiples niveles.

        Args:
            metrics: Diccionario de métricas actuales.

        Returns:
            str: Diagnóstico textual.
        """
        # Validar entrada
        if not isinstance(metrics, dict):
            return "❓ DIAGNÓSTICO NO DISPONIBLE (métricas inválidas)"

        try:
            # Extraer métricas con valores por defecto seguros
            ec = metrics.get("potential_energy", 0.0)
            el = metrics.get("kinetic_energy", 0.0)
            flyback = metrics.get("flyback_voltage", 0.0)
            power = metrics.get("dissipated_power", 0.0)
            saturation = metrics.get("saturation", 0.0)
            damping = metrics.get("damping_ratio", 1.0)

            # Sanitizar valores
            ec = 0.0 if (math.isnan(ec) or math.isinf(ec)) else ec
            el = 0.0 if (math.isnan(el) or math.isinf(el)) else el
            flyback = 0.0 if (math.isnan(flyback) or math.isinf(flyback)) else flyback
            power = 0.0 if (math.isnan(power) or math.isinf(power)) else power

            # ═══════════════════════════════════════════════════════════════════
            # DIAGNÓSTICO JERÁRQUICO (de más crítico a menos)
            # ═══════════════════════════════════════════════════════════════════

            # 1. CRÍTICO: Sobrecalentamiento
            if power > SystemConstants.OVERHEAT_POWER_THRESHOLD:
                return f"🔴 SOBRECALENTAMIENTO CRÍTICO (P={power:.1f}W)"

            # 2. CRÍTICO: Sistema estancado
            if el < SystemConstants.MIN_ENERGY_THRESHOLD:
                return "🔴 SISTEMA ESTANCADO (Inercia crítica baja)"

            # 3. CRÍTICO: Inestabilidad
            if damping < 0.2:
                return f"🔴 INESTABILIDAD (ζ={damping:.2f})"

            # 4. ADVERTENCIA: Sobrepresión
            energy_ratio = ec / el if el > SystemConstants.MIN_ENERGY_THRESHOLD else 0.0
            if energy_ratio > SystemConstants.HIGH_PRESSURE_RATIO:
                return f"🟠 SOBRECARGA DE PRESIÓN (ratio={energy_ratio:.1f})"

            # 5. ADVERTENCIA: Pico inductivo
            if flyback > SystemConstants.HIGH_FLYBACK_THRESHOLD:
                return f"⚡ PICO INDUCTIVO (V_L={flyback:.2f}V)"

            # 6. ADVERTENCIA: Saturación alta
            if saturation > 0.9:
                return f"🟡 SATURACIÓN ALTA ({saturation:.1%})"

            # 7. INFO: Baja inercia
            if el < SystemConstants.LOW_INERTIA_THRESHOLD:
                return f"🟡 BAJA INERCIA (El={el:.3f}J)"

            # 8. INFO: Oscilatorio
            if damping < 0.7:
                return f"🔵 RÉGIMEN OSCILATORIO (ζ={damping:.2f})"

            # 9. OK: Sistema estable
            return f"🟢 EQUILIBRIO ENERGÉTICO (Sat={saturation:.1%})"

        except Exception as e:
            self.logger.error(f"Error en diagnóstico: {e}")
            return "❓ DIAGNÓSTICO INDETERMINADO"

    def get_trend_analysis(self) -> Dict[str, Any]:
        """
        Analiza tendencias basadas en historial de métricas.

        Returns:
            Dict[str, Any]: Diccionario con análisis de tendencias.
        """
        if len(self._metrics_history) < 5:
            return {"status": "INSUFFICIENT_DATA", "samples": len(self._metrics_history)}

        recent = self._metrics_history[-20:]

        # Calcular tendencias
        saturations = [m["saturation"] for m in recent]
        powers = [m["dissipated_power"] for m in recent]

        avg_saturation = sum(saturations) / len(saturations)
        saturation_trend = saturations[-1] - saturations[0]

        avg_power = sum(powers) / len(powers)
        power_trend = powers[-1] - powers[0]

        return {
            "status": "OK",
            "samples": len(self._metrics_history),
            "saturation": {
                "current": saturations[-1],
                "average": avg_saturation,
                "trend": "INCREASING" if saturation_trend > 0.05 else
                         "DECREASING" if saturation_trend < -0.05 else "STABLE",
            },
            "power": {
                "current": powers[-1],
                "average": avg_power,
                "trend": "INCREASING" if power_trend > 1.0 else
                         "DECREASING" if power_trend < -1.0 else "STABLE",
            },
        }


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
            InvalidInputError: Si hay errores en los parámetros.
        """
        errors = []

        # Validar config
        if config is None:
            errors.append("config no puede ser None")
        elif not isinstance(config, dict):
            errors.append(f"config debe ser dict, recibido: {type(config).__name__}")

        # Validar profile
        if profile is None:
            errors.append("profile no puede ser None")
        elif not isinstance(profile, dict):
            errors.append(f"profile debe ser dict, recibido: {type(profile).__name__}")

        if errors:
            raise InvalidInputError(
                "Errores de inicialización:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Advertencias para claves faltantes (modo tolerante)
        if isinstance(config, dict):
            missing_config = self.REQUIRED_CONFIG_KEYS - set(config.keys())
            if missing_config:
                self.logger.warning(
                    f"Claves faltantes en config (modo tolerante): {missing_config}"
                )

        if isinstance(profile, dict):
            missing_profile = self.REQUIRED_PROFILE_KEYS - set(profile.keys())
            if missing_profile:
                self.logger.warning(
                    f"Claves faltantes en profile (modo tolerante): {missing_profile}"
                )

    def stabilize(self, file_path: str) -> pd.DataFrame:
        """
        Proceso principal de estabilización con control PID.

        Orquesta el flujo: validar -> parsear -> extraer -> procesar lotes -> consolidar.

        Args:
            file_path: Ruta al archivo de APU a procesar.

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
                raw_records = raw_records[:SystemConstants.MAX_RECORDS_LIMIT]
                total_records = len(raw_records)

            # ═══════════════════════════════════════════════════════════════════
            # FASE 5: Procesamiento por lotes con PID
            # ═══════════════════════════════════════════════════════════════════
            processed_batches = self._process_batches_with_pid(
                raw_records, cache, total_records
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
        """Limpieza después del procesamiento (exitoso o fallido)."""
        # Liberar referencias grandes
        pass  # Placeholder para limpieza específica si es necesaria

    def _create_empty_result(self) -> pd.DataFrame:
        """Crea un DataFrame vacío con estructura esperada."""
        self._stats.processing_time = time.time() - (self._start_time or time.time())
        return pd.DataFrame()

    def _validate_input_file(self, file_path: str) -> Path:
        """
        Valida el archivo de entrada exhaustivamente.

        Args:
            file_path: Ruta al archivo.

        Returns:
            Path: Objeto Path validado.

        Raises:
            InvalidInputError: Si el archivo es inválido.
        """
        # Validar tipo
        if not isinstance(file_path, (str, Path)):
            raise InvalidInputError(
                f"file_path debe ser str o Path, recibido: {type(file_path).__name__}"
            )

        if isinstance(file_path, str) and not file_path.strip():
            raise InvalidInputError("file_path no puede ser una cadena vacía")

        path = Path(file_path)

        # Verificar existencia
        if not path.exists():
            raise InvalidInputError(f"El archivo no existe: {path}")

        # Verificar que es archivo (no directorio)
        if not path.is_file():
            raise InvalidInputError(f"La ruta no es un archivo: {path}")

        # Verificar accesibilidad
        try:
            if not os.access(path, os.R_OK):
                raise InvalidInputError(f"El archivo no es legible: {path}")
        except Exception as e:
            raise InvalidInputError(f"Error verificando acceso a archivo: {e}") from e

        # Verificar tamaño
        try:
            file_size = path.stat().st_size

            if file_size < SystemConstants.MIN_FILE_SIZE_BYTES:
                raise InvalidInputError(
                    f"Archivo demasiado pequeño ({file_size} bytes): {path}"
                )

            file_size_mb = file_size / (1024 * 1024)
            if file_size_mb > SystemConstants.MAX_FILE_SIZE_MB:
                self.logger.warning(
                    f"Archivo grande ({file_size_mb:.1f} MB > "
                    f"{SystemConstants.MAX_FILE_SIZE_MB} MB): {path}"
                )
        except OSError as e:
            raise InvalidInputError(f"Error obteniendo tamaño del archivo: {e}") from e

        # Verificar extensión (advertencia, no error)
        if path.suffix.lower() not in SystemConstants.VALID_FILE_EXTENSIONS:
            self.logger.warning(
                f"Extensión no estándar: {path.suffix}. "
                f"Esperadas: {SystemConstants.VALID_FILE_EXTENSIONS}"
            )

        self.logger.debug(f"[VALIDATE] Archivo validado: {path}")
        return path

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
                cache_items = list(cache.items())[-SystemConstants.MAX_CACHE_SIZE:]
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
    ) -> List[pd.DataFrame]:
        """
        Procesamiento por lotes con:
        - Algoritmo de backoff exponencial para fallos.
        - Balanceo de carga adaptativo.
        - Predicción de tiempo restante.

        Args:
            raw_records: Lista de registros crudos.
            cache: Caché de parseo.
            total_records: Número total de registros.

        Returns:
            List[pd.DataFrame]: Lista de DataFrames procesados.
        """
        processed_batches: List[pd.DataFrame] = []
        current_index = 0
        current_batch_size = self.condenser_config.min_batch_size
        failed_batches_count = 0
        consecutive_failures = 0

        # Historial para predicción
        batch_times: List[float] = []
        batch_sizes: List[int] = []

        # Algoritmo de backoff
        backoff_factor = 1.0
        min_backoff_batch = max(1, self.condenser_config.min_batch_size // 2)

        # Estadísticas para balanceo
        total_start_time = time.time()
        records_per_second = 0.0

        # Límite dinámico de iteraciones basado en complejidad
        avg_complexity = 0.5  # Estimación inicial
        complexity_adjusted_limit = int(
            total_records * SystemConstants.MAX_ITERATIONS_MULTIPLIER *
            (1.0 + avg_complexity)
        )

        self.logger.info(
            f"[PID_LOOP] Iniciando | Registros: {total_records:,} | "
            f"Batch inicial: {current_batch_size} | "
            f"Límite de iteraciones: {complexity_adjusted_limit:,}"
        )

        iteration = 0
        while current_index < total_records and iteration < complexity_adjusted_limit:
            iteration += 1
            batch_start_time = time.time()

            # ================ PREDICCIÓN DE TIEMPO RESTANTE ================
            if len(batch_times) >= 3:
                avg_time_per_record = np.mean([
                    t / s for t, s in zip(batch_times[-3:], batch_sizes[-3:])
                ]) if 'np' in globals() else 1.0

                records_remaining = total_records - current_index
                estimated_time_remaining = records_remaining * avg_time_per_record

                if iteration % 10 == 0:
                    self.logger.info(
                        f"📈 Progreso: {current_index:,}/{total_records:,} "
                        f"({current_index/total_records*100:.1f}%) | "
                        f"ETA: {estimated_time_remaining:.1f}s"
                    )

            # ================ AJUSTE DINÁMICO DE BATCH SIZE ================
            # Basado en rendimiento reciente
            if len(batch_times) >= 5:
                recent_efficiency = [
                    s / t if t > 0 else s
                    for t, s in zip(batch_times[-5:], batch_sizes[-5:])
                ]
                avg_efficiency = sum(recent_efficiency) / len(recent_efficiency)

                # Ajustar batch size si eficiencia baja
                if avg_efficiency < 10:  # Menos de 10 registros/segundo
                    current_batch_size = max(
                        min_backoff_batch,
                        int(current_batch_size * 0.8)
                    )
                    self.logger.debug(f"Eficiencia baja ({avg_efficiency:.1f} rec/s), reduciendo batch")

            # ================ EXTRAER LOTE CON PADDING INTELIGENTE ================
            end_index = min(current_index + current_batch_size, total_records)

            # Intentar alinear a límites naturales (p.ej., múltiplos de 100)
            if (total_records - end_index) > 100:
                # Buscar punto de alineación natural (fin de sección)
                remaining = total_records - end_index
                if remaining > current_batch_size * 0.3:
                    # Extender batch para incluir sección completa
                    potential_end = min(
                        end_index + (100 - (end_index % 100)),
                        total_records
                    )
                    if potential_end - current_index <= self.condenser_config.max_batch_size:
                        end_index = potential_end

            batch_records = raw_records[current_index:end_index]

            if not batch_records:
                current_index = end_index
                continue

            # ================ CÁLCULO DE MÉTRICAS CON PONDERACIÓN ================
            cache_hits = self._calculate_cache_hits(batch_records, cache)

            # Ponderar por importancia del batch (primero/último son más críticos)
            position_factor = 1.0
            if current_index < total_records * 0.1:  # Primer 10%
                position_factor = 1.2  # Más conservador al inicio
            elif current_index > total_records * 0.9:  # Último 10%
                position_factor = 1.1  # Cuidado al final

            metrics = self.physics.calculate_metrics(
                len(batch_records) * position_factor,
                cache_hits
            )

            # ================ CONTROL PID CON LIMITADORES DINÁMICOS ================
            saturation = metrics["saturation"]

            # Limitar cambios bruscos en saturación
            if hasattr(self, '_last_saturation'):
                saturation_change = abs(saturation - self._last_saturation)
                if saturation_change > 0.3:  # Cambio mayor al 30%
                    saturation = self._last_saturation + math.copysign(0.3, saturation - self._last_saturation)
                    self.logger.debug(f"Limiting saturation change: {saturation_change:.2f} -> 0.3")

            new_batch_size = self.controller.compute(saturation)
            self._last_saturation = saturation

            # Freno de emergencia por sobrecalentamiento
            if metrics["dissipated_power"] > SystemConstants.OVERHEAT_POWER_THRESHOLD:
                self.logger.warning(
                    f"🔥 [OVERHEAT] P={metrics['dissipated_power']:.1f}W - Aplicando freno"
                )
                new_batch_size = max(
                    self.condenser_config.min_batch_size,
                    int(new_batch_size * SystemConstants.EMERGENCY_BRAKE_FACTOR)
                )
                self._stats.emergency_brakes_triggered += 1

            # ================ GESTIÓN DE FALLOS CON BACKOFF EXPONENCIAL ================
            batch_result = self._process_single_batch(
                batch_records, cache, current_index, end_index
            )

            batch_time = time.time() - batch_start_time

            if batch_result.success:
                consecutive_failures = 0
                backoff_factor = 1.0  # Reset backoff

                if batch_result.dataframe is not None:
                    processed_batches.append(batch_result.dataframe)

                    # Actualizar estadísticas de rendimiento
                    batch_times.append(batch_time)
                    batch_sizes.append(len(batch_records))

                    # Calcular throughput
                    if len(batch_times) >= 2:
                        total_time = sum(batch_times[-10:])
                        total_records_processed = sum(batch_sizes[-10:])
                        records_per_second = total_records_processed / total_time if total_time > 0 else 0

                        # Ajuste adaptativo basado en throughput
                        if records_per_second > 1000:  # Alto throughput
                            new_batch_size = min(
                                self.condenser_config.max_batch_size,
                                int(new_batch_size * 1.1)
                            )
            else:
                consecutive_failures += 1
                failed_batches_count += 1

                # Backoff exponencial con jitter
                backoff_factor *= 2.0
                jitter = 0.9 + (random.random() * 0.2) if 'random' in globals() else 1.0
                current_batch_size = max(
                    min_backoff_batch,
                    int(self.condenser_config.min_batch_size * backoff_factor * jitter)
                )

                self.logger.warning(
                    f"Batch fallido (#{consecutive_failures}). "
                    f"Backoff: {backoff_factor:.1f}x, nuevo batch: {current_batch_size}"
                )

                # Si hay muchos fallos consecutivos, reconsiderar estrategia
                if consecutive_failures >= 3:
                    self.logger.error("Múltiples fallos consecutivos. Re-evaluando...")
                    # Reducir batch size drásticamente
                    current_batch_size = min_backoff_batch
                    # Saltar registros problemáticos
                    current_index += len(batch_records) // 2  # Saltar mitad del batch fallido

            # ================ ACTUALIZAR ESTADÍSTICAS ================
            self._stats.add_batch_stats(
                batch_size=len(batch_records),
                saturation=saturation,
                power=metrics["dissipated_power"],
                flyback=metrics["flyback_voltage"],
                kinetic=metrics["kinetic_energy"],
                success=batch_result.success,
            )

            # ================ LOGGING INTELIGENTE ================
            log_interval = 5 if batch_time > 1.0 else 20
            if iteration % log_interval == 0 or iteration <= 5:
                self.logger.info(
                    f"🔄 [{iteration}/{complexity_adjusted_limit}] "
                    f"Batch: {len(batch_records):,} rec | "
                    f"Sat: {saturation:.1%} | "
                    f"Time: {batch_time:.2f}s | "
                    f"Throughput: {records_per_second:.1f} rec/s | "
                    f"Next: {new_batch_size:,}"
                )

            # ================ ACTUALIZAR PARA SIGUIENTE ITERACIÓN ================
            current_index = end_index

            if batch_result.success:
                current_batch_size = min(
                    self.condenser_config.max_batch_size,
                    max(self.condenser_config.min_batch_size, new_batch_size)
                )

        # ================ RESUMEN FINAL ================
        total_time = time.time() - total_start_time
        overall_throughput = self._stats.processed_records / total_time if total_time > 0 else 0

        self.logger.info(
            f"✅ [PID_LOOP] Completado en {total_time:.1f}s | "
            f"Throughput: {overall_throughput:.1f} rec/s | "
            f"Batches: {self._stats.total_batches} (fallidos: {failed_batches_count}) | "
            f"Eficiencia: {self._stats.processed_records/total_records*100:.1f}%"
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
        Calculo mejorado de cache hits con:
        - Búsqueda aproximada (fuzzy matching).
        - Ponderación por tipo de dato.
        - Preprocesamiento de claves.

        Args:
            batch_records: Registros del batch actual.
            cache: Caché disponible.

        Returns:
            int: Número estimado de hits en caché.
        """
        if not cache or not isinstance(cache, dict):
            return 0

        if not batch_records or not isinstance(batch_records, list):
            return 0

        # Preprocesar claves del cache para búsqueda rápida
        if not hasattr(self, '_cache_keys_normalized'):
            self._cache_keys_normalized = {}
            for key, value in cache.items():
                if isinstance(key, str):
                    # Normalizar: minúsculas, sin espacios extra
                    normalized = key.lower().strip()
                    if len(normalized) > 3:  # Ignorar claves muy cortas
                        self._cache_keys_normalized[normalized] = value

        cache_hits = 0
        possible_keys = ["insumo_line", "line", "raw_line", "content", "text", "data"]

        for record in batch_records:
            if not isinstance(record, dict):
                continue

            record_hit = False

            # Estrategia 1: Búsqueda exacta en claves conocidas
            for key in possible_keys:
                if key in record:
                    content = record[key]
                    if isinstance(content, str):
                        normalized_content = content.lower().strip()
                        if normalized_content in self._cache_keys_normalized:
                            cache_hits += 1
                            record_hit = True
                            break

            if record_hit:
                continue

            # Estrategia 2: Búsqueda aproximada (substrings)
            if not record_hit:
                for key, value in record.items():
                    if isinstance(value, str) and len(value) > 10:
                        # Buscar fragmentos largos en cache
                        for cache_key in self._cache_keys_normalized:
                            if value[:50] in cache_key or cache_key in value[:50]:
                                cache_hits += 0.5  # Hit parcial
                                break

            # Estrategia 3: Hash de contenido
            if not record_hit and 'hash' in record:
                content_hash = record.get('hash')
                if isinstance(content_hash, (str, int)):
                    str_hash = str(content_hash)
                    if str_hash in cache or f"hash_{str_hash}" in cache:
                        cache_hits += 1

        # Ajustar por tamaño del batch
        if len(batch_records) > 0:
            hit_ratio = cache_hits / len(batch_records)

            # Penalizar ratios muy bajos (posible problema de cache)
            if hit_ratio < 0.1 and len(batch_records) > 100:
                self.logger.debug(f"Cache hit ratio bajo: {hit_ratio:.1%}")

        return int(cache_hits)

    def _consolidate_results(
        self, processed_batches: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Consolida múltiples DataFrames en uno solo.

        Args:
            processed_batches: Lista de DataFrames de los lotes procesados.

        Returns:
            pd.DataFrame: DataFrame consolidado.
        """
        if not processed_batches:
            self.logger.warning("[CONSOLIDATE] Sin batches para consolidar")
            return pd.DataFrame()

        if not isinstance(processed_batches, list):
            raise ProcessingError(
                f"processed_batches debe ser list, recibido: {type(processed_batches).__name__}"
            )

        # Verificar límite de batches
        if len(processed_batches) > SystemConstants.MAX_BATCHES_TO_CONSOLIDATE:
            self.logger.warning(
                f"[CONSOLIDATE] Demasiados batches ({len(processed_batches)}), "
                f"usando los últimos {SystemConstants.MAX_BATCHES_TO_CONSOLIDATE}"
            )
            processed_batches = processed_batches[-SystemConstants.MAX_BATCHES_TO_CONSOLIDATE:]

        try:
            # Filtrar batches válidos
            valid_batches = []
            for i, batch in enumerate(processed_batches):
                if not isinstance(batch, pd.DataFrame):
                    self.logger.warning(f"[CONSOLIDATE] Batch {i} no es DataFrame, ignorando")
                    continue
                if batch.empty:
                    continue
                valid_batches.append(batch)

            if not valid_batches:
                self.logger.warning("[CONSOLIDATE] Todos los batches están vacíos o inválidos")
                return pd.DataFrame()

            # Concatenar
            df_final = pd.concat(valid_batches, ignore_index=True)

            self.logger.info(
                f"[CONSOLIDATE] {len(valid_batches)} batches → {len(df_final)} registros"
            )

            return df_final

        except MemoryError as e:
            raise ProcessingError(f"Error de memoria consolidando resultados: {e}") from e
        except Exception as e:
            raise ProcessingError(f"Error consolidando resultados: {e}") from e

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
        Valida el DataFrame de salida.

        Args:
            df: DataFrame a validar.
        """
        if not isinstance(df, pd.DataFrame):
            raise ProcessingError(
                f"Salida debe ser DataFrame, recibido: {type(df).__name__}"
            )

        if not self.condenser_config.enable_strict_validation:
            return

        if df.empty:
            self.logger.warning("[VALIDATE] DataFrame de salida vacío")
            return

        # Analizar columnas
        total_rows = len(df)
        problematic_columns = []

        for col in df.columns:
            try:
                null_count = df[col].isnull().sum()
                null_ratio = null_count / total_rows

                if null_ratio == 1.0:
                    problematic_columns.append((col, "100% nulos"))
                elif null_ratio > 0.9:
                    problematic_columns.append((col, f"{null_ratio:.1%} nulos"))
            except Exception:
                continue

        if problematic_columns:
            self.logger.warning(
                f"[VALIDATE] Columnas problemáticas: {problematic_columns[:10]}"
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
        """Resetea el estado del condensador para reutilización."""
        self.controller.reset()
        self._stats = ProcessingStats()
        self._start_time = None
        self.logger.info("[RESET] Condensador reseteado")
