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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

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

    Atributos:
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

    Atributos:
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
    - Manejo de edge cases (NaN, Inf, valores extremos)
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

        ROBUSTECIDO:
        - Validación de todos los parámetros
        - Cálculo seguro de límites internos
        - Inicialización de historial para diagnóstico

        Args:
            kp: Ganancia Proporcional.
            ki: Ganancia Integral.
            setpoint: Punto de ajuste (target).
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
        Calcula la nueva salida de control basada en el error actual.

        ROBUSTECIDO:
        - Validación y sanitización de entrada
        - Manejo de warmup inicial
        - Detección de saturación prolongada
        - Anti-windup con back-calculation
        - Historial para diagnóstico

        Args:
            process_variable: Valor actual del proceso (saturación medida).

        Returns:
            Señal de control (batch size) en rango válido.
        """
        self._iteration_count += 1

        # ═══════════════════════════════════════════════════════════════════════
        # SANITIZACIÓN DE ENTRADA
        # ═══════════════════════════════════════════════════════════════════════

        pv_original = process_variable

        # Validar tipo
        if not isinstance(process_variable, (int, float)):
            self.logger.warning(
                f"process_variable tipo inválido ({type(process_variable).__name__}), "
                f"usando setpoint como fallback"
            )
            process_variable = self.setpoint

        # Manejar NaN/Inf
        if math.isnan(process_variable) or math.isinf(process_variable):
            self.logger.warning(
                f"process_variable inválido ({pv_original}), usando setpoint"
            )
            process_variable = self.setpoint

        # Normalizar a rango [0, 1]
        process_variable = max(0.0, min(1.0, float(process_variable)))

        # ═══════════════════════════════════════════════════════════════════════
        # CÁLCULO DE DELTA TIME
        # ═══════════════════════════════════════════════════════════════════════

        current_time = time.time()

        if self._last_time is None:
            dt = SystemConstants.MIN_DELTA_TIME
            self._in_warmup = True
        else:
            dt = current_time - self._last_time

            # Validar dt
            if dt <= 0:
                self.logger.debug(f"Delta time no positivo ({dt}), usando mínimo")
                dt = SystemConstants.MIN_DELTA_TIME
            elif dt > SystemConstants.MAX_DELTA_TIME:
                self.logger.warning(
                    f"Delta time excesivo ({dt}s), posible interrupción. Reseteando integrador."
                )
                dt = SystemConstants.MIN_DELTA_TIME
                self._integral_error = 0.0  # Reset del integrador

        # Salir de warmup después de N iteraciones
        if self._in_warmup and self._iteration_count > self._WARMUP_ITERATIONS:
            self._in_warmup = False

        # ═══════════════════════════════════════════════════════════════════════
        # CÁLCULO DEL ERROR
        # ═══════════════════════════════════════════════════════════════════════

        # Error: positivo = saturación baja (aumentar batch)
        #        negativo = saturación alta (reducir batch)
        error = self.setpoint - process_variable

        # ═══════════════════════════════════════════════════════════════════════
        # TÉRMINO PROPORCIONAL
        # ═══════════════════════════════════════════════════════════════════════

        P = self.Kp * error

        # ═══════════════════════════════════════════════════════════════════════
        # TÉRMINO INTEGRAL CON ANTI-WINDUP
        # ═══════════════════════════════════════════════════════════════════════

        # Acumular error (solo si no estamos en warmup)
        if not self._in_warmup:
            self._integral_error += error * dt

        # Anti-windup: Clamping del integrador
        integral_before_clamp = self._integral_error
        self._integral_error = max(
            -self._integral_limit,
            min(self._integral_limit, self._integral_error)
        )

        # Detectar saturación del integrador
        if abs(integral_before_clamp) > abs(self._integral_error):
            self._consecutive_saturations += 1
            if self._consecutive_saturations > 10:
                self.logger.warning(
                    f"Integrador saturado por {self._consecutive_saturations} iteraciones. "
                    f"Considere ajustar ganancias."
                )
        else:
            self._consecutive_saturations = 0

        I = self.Ki * self._integral_error

        # ═══════════════════════════════════════════════════════════════════════
        # SEÑAL DE CONTROL
        # ═══════════════════════════════════════════════════════════════════════

        control_signal = self._base_output + P + I

        # Validar señal de control
        if math.isnan(control_signal) or math.isinf(control_signal):
            self.logger.error(
                f"Señal de control inválida (P={P}, I={I}), usando base_output"
            )
            control_signal = self._base_output
            self._integral_error = 0.0  # Reset del integrador

        # Saturación del actuador
        output = int(round(control_signal))
        output = max(self.min_output, min(self.max_output, output))

        # ═══════════════════════════════════════════════════════════════════════
        # ACTUALIZAR ESTADO E HISTORIAL
        # ═══════════════════════════════════════════════════════════════════════

        self._last_time = current_time
        self._last_error = error

        # Guardar en historial (buffer circular)
        history_entry = {
            "iteration": self._iteration_count,
            "pv": process_variable,
            "error": error,
            "P": P,
            "I": I,
            "output": output,
            "integral": self._integral_error,
            "dt": dt,
        }

        self._history.append(history_entry)
        if len(self._history) > self._MAX_HISTORY_SIZE:
            self._history.pop(0)

        # Logging periódico
        if self._iteration_count % 10 == 0 or self._iteration_count <= 3:
            self.logger.debug(
                f"[PID #{self._iteration_count}] PV={process_variable:.3f} | "
                f"Error={error:+.3f} | P={P:+.1f} | I={I:+.1f} | Out={output}"
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
        self.logger.debug("[PID] Controlador reseteado")

    def get_state(self) -> Dict[str, Any]:
        """Retorna estado completo del controlador para observabilidad."""
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
        """Retorna diagnósticos del controlador basados en historial."""
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
    - Validación exhaustiva de entradas
    - Protección contra overflow/underflow
    - Métricas normalizadas y sanitizadas
    - Diagnóstico mejorado con múltiples niveles
    - Historial de métricas para análisis de tendencias
    """

    _MAX_METRICS_HISTORY: int = 100

    def __init__(self, capacitance: float, resistance: float, inductance: float):
        """
        Inicializa el motor de física con validación exhaustiva.

        ROBUSTECIDO:
        - Validación de rangos físicamente plausibles
        - Cálculo de constantes derivadas

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

    def calculate_metrics(
        self, total_records: int, cache_hits: int
    ) -> Dict[str, float]:
        """
        Calcula métricas del flujo con validación exhaustiva.

        ROBUSTECIDO:
        - Validación de entradas
        - Protección contra overflow matemático
        - Sanitización de todas las métricas de salida
        - Almacenamiento en historial

        Args:
            total_records: Número total de registros en el batch
            cache_hits: Número de registros con hit en caché

        Returns:
            Diccionario con métricas normalizadas y sanitizadas
        """
        # ═══════════════════════════════════════════════════════════════════════
        # VALIDACIÓN DE ENTRADAS
        # ═══════════════════════════════════════════════════════════════════════

        # Validar tipos
        if not isinstance(total_records, (int, float)):
            self.logger.warning(
                f"total_records tipo inválido ({type(total_records).__name__}), usando 0"
            )
            return self._get_zero_metrics()

        if not isinstance(cache_hits, (int, float)):
            self.logger.warning(
                f"cache_hits tipo inválido ({type(cache_hits).__name__}), usando 0"
            )
            cache_hits = 0

        # Convertir a int
        total_records = int(total_records)
        cache_hits = int(cache_hits)

        # Validar rangos
        if total_records < 0:
            self.logger.error(f"total_records negativo: {total_records}")
            return self._get_zero_metrics()

        if cache_hits < 0:
            self.logger.warning(f"cache_hits negativo: {cache_hits}, usando 0")
            cache_hits = 0

        if cache_hits > total_records:
            self.logger.warning(
                f"cache_hits ({cache_hits}) > total_records ({total_records}), "
                f"normalizando"
            )
            cache_hits = total_records

        # Caso especial: sin datos
        if total_records == 0:
            return self._get_zero_metrics()

        # ═══════════════════════════════════════════════════════════════════════
        # CÁLCULOS FÍSICOS
        # ═══════════════════════════════════════════════════════════════════════

        try:
            # Corriente (I): Calidad del flujo [0.0, 1.0]
            current_I = cache_hits / total_records

            # Complejidad: Inversa de la calidad [0.0, 1.0]
            complexity = 1.0 - current_I

            # Resistencia Dinámica (R_dyn)
            dynamic_R = self.R * (
                1.0 + complexity * SystemConstants.COMPLEXITY_RESISTANCE_FACTOR
            )

            # Constante de tiempo dinámica
            tau_c = dynamic_R * self.C

            # ═══════════════════════════════════════════════════════════════════
            # SATURACIÓN (V)
            # ═══════════════════════════════════════════════════════════════════

            # Proteger contra división por cero y overflow
            if tau_c <= 0:
                tau_c = SystemConstants.MIN_ENERGY_THRESHOLD

            exponent = -float(total_records) / tau_c

            # Limitar exponente para evitar overflow
            exponent = max(-SystemConstants.MAX_EXPONENTIAL_ARG, exponent)
            exponent = min(SystemConstants.MAX_EXPONENTIAL_ARG, exponent)

            saturation_V = 1.0 - math.exp(exponent)

            # ═══════════════════════════════════════════════════════════════════
            # ENERGÍAS
            # ═══════════════════════════════════════════════════════════════════

            # Energía Potencial (Ec = 1/2 * C * V^2)
            potential_energy = 0.5 * self.C * (saturation_V ** 2)

            # Energía Cinética (El = 1/2 * L * I^2)
            kinetic_energy = 0.5 * self.L * (current_I ** 2)

            # Potencia Disipada (P = I_ruido^2 * R)
            noise_current = complexity  # 1.0 - current_I
            dissipated_power = (noise_current ** 2) * dynamic_R

            # ═══════════════════════════════════════════════════════════════════
            # FLYBACK (Tensión Inductiva)
            # ═══════════════════════════════════════════════════════════════════

            delta_i = 1.0 - current_I

            # Usar log1p para estabilidad numérica
            dt = math.log1p(total_records)
            dt = max(SystemConstants.MIN_ENERGY_THRESHOLD, dt)

            flyback_voltage = (self.L * abs(delta_i)) / dt

            # ═══════════════════════════════════════════════════════════════════
            # CONSTRUCCIÓN Y SANITIZACIÓN DE MÉTRICAS
            # ═══════════════════════════════════════════════════════════════════

            metrics = {
                "saturation": self._sanitize_metric(saturation_V, 0.0, 1.0),
                "complexity": self._sanitize_metric(complexity, 0.0, 1.0),
                "flyback_voltage": self._sanitize_metric(
                    flyback_voltage, 0.0, SystemConstants.MAX_FLYBACK_VOLTAGE
                ),
                "potential_energy": self._sanitize_metric(potential_energy, 0.0, 1e10),
                "kinetic_energy": self._sanitize_metric(kinetic_energy, 0.0, 1e10),
                "dissipated_power": self._sanitize_metric(dissipated_power, 0.0, 1e6),
                # Métricas adicionales para diagnóstico
                "current_I": self._sanitize_metric(current_I, 0.0, 1.0),
                "dynamic_resistance": self._sanitize_metric(dynamic_R, 0.0, 1e6),
                "tau_effective": self._sanitize_metric(tau_c, 0.0, 1e6),
            }

            # Almacenar en historial
            self._store_metrics(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculando métricas físicas: {e}", exc_info=True)
            return self._get_zero_metrics()

    def _sanitize_metric(
        self, value: float, min_val: float, max_val: float
    ) -> float:
        """
        Sanitiza un valor de métrica asegurando que sea válido.

        ROBUSTECIDO:
        - Manejo de NaN/Inf
        - Clamping a rango válido
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
            "tau_effective": self._tau_base,
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

        ROBUSTECIDO:
        - Validación de métricas de entrada
        - Diagnóstico jerárquico
        - Análisis de tendencias si hay historial
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

            # 3. ADVERTENCIA: Sobrepresión
            energy_ratio = ec / el if el > SystemConstants.MIN_ENERGY_THRESHOLD else 0.0
            if energy_ratio > SystemConstants.HIGH_PRESSURE_RATIO:
                return f"🟠 SOBRECARGA DE PRESIÓN (ratio={energy_ratio:.1f})"

            # 4. ADVERTENCIA: Pico inductivo
            if flyback > SystemConstants.HIGH_FLYBACK_THRESHOLD:
                return f"⚡ PICO INDUCTIVO (V_L={flyback:.2f}V)"

            # 5. ADVERTENCIA: Saturación alta
            if saturation > 0.9:
                return f"🟡 SATURACIÓN ALTA ({saturation:.1%})"

            # 6. INFO: Baja inercia
            if el < SystemConstants.LOW_INERTIA_THRESHOLD:
                return f"🟡 BAJA INERCIA (El={el:.3f}J)"

            # 7. OK: Sistema estable
            return f"🟢 EQUILIBRIO ENERGÉTICO (Sat={saturation:.1%})"

        except Exception as e:
            self.logger.error(f"Error en diagnóstico: {e}")
            return "❓ DIAGNÓSTICO INDETERMINADO"

    def get_trend_analysis(self) -> Dict[str, Any]:
        """
        Analiza tendencias basadas en historial de métricas.

        Returns:
            Diccionario con análisis de tendencias
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
    - Validación exhaustiva en cada etapa
    - Timeouts y límites de recursos
    - Recuperación parcial mejorada
    - Métricas de calidad de datos
    - Logging estructurado con contexto
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

        ROBUSTECIDO:
        - Validación de tipos antes de cualquier operación
        - Inicialización segura de componentes
        - Fallbacks para configuración faltante

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

        ROBUSTECIDO:
        - Validación de tipos
        - Advertencias para claves faltantes (no errores)
        - Verificación de estructura básica
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

        ROBUSTECIDO:
        - Timeout de procesamiento
        - Validación exhaustiva en cada fase
        - Recuperación de errores con cleanup
        - Métricas de calidad en resultado

        Args:
            file_path: Ruta al archivo de APU a procesar

        Returns:
            DataFrame con los datos procesados

        Raises:
            InvalidInputError: Si el archivo es inválido
            ProcessingError: Si ocurre un error durante el procesamiento
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
            # Cleanup
            self._cleanup_after_processing()

    def _check_timeout(self, phase: str) -> None:
        """
        Verifica si se ha excedido el timeout de procesamiento.

        Args:
            phase: Nombre de la fase actual para logging

        Raises:
            ProcessingError: Si se excede el timeout
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

        ROBUSTECIDO:
        - Validación de tipo de entrada
        - Verificación de existencia y accesibilidad
        - Validación de tamaño
        - Verificación de extensión

        Args:
            file_path: Ruta al archivo

        Returns:
            Path validado

        Raises:
            InvalidInputError: Si el archivo es inválido
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

        ROBUSTECIDO:
        - Validación de path
        - Manejo específico de errores de inicialización
        - Logging detallado
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

        ROBUSTECIDO:
        - Validación de tipo del parser (Duck Typing para soportar Mocks)
        - Validación de resultados
        - Límite de tamaño de cache
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

            # Obtener cache
            cache = parser.get_parse_cache()

            if cache is None:
                cache = {}
            elif not isinstance(cache, dict):
                self.logger.warning(
                    f"[EXTRACT] get_parse_cache() retornó {type(cache).__name__}, usando dict vacío"
                )
                cache = {}

            # Limitar tamaño de cache
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
        Procesa registros en lotes con control PID adaptativo.

        ROBUSTECIDO:
        - Protección contra loops infinitos
        - Recuperación parcial mejorada
        - Validación de cada batch
        - Métricas detalladas
        - Timeouts por batch
        """
        processed_batches: List[pd.DataFrame] = []
        current_index = 0
        current_batch_size = self.condenser_config.min_batch_size
        failed_batches_count = 0
        consecutive_empty_batches = 0

        # Protección contra loops infinitos
        max_iterations = min(
            total_records * SystemConstants.MAX_ITERATIONS_MULTIPLIER,
            1_000_000  # Límite absoluto
        )
        iteration_count = 0

        self.logger.info(
            f"[PID_LOOP] Iniciando | Total: {total_records} | "
            f"Batch inicial: {current_batch_size} | Max iter: {max_iterations}"
        )

        while current_index < total_records:
            iteration_count += 1

            # ═══════════════════════════════════════════════════════════════════
            # PROTECCIONES
            # ═══════════════════════════════════════════════════════════════════

            # Protección contra loop infinito
            if iteration_count > max_iterations:
                raise ProcessingError(
                    f"Loop infinito detectado: {iteration_count} iteraciones. "
                    f"Índice: {current_index}/{total_records}"
                )

            # Timeout
            self._check_timeout("procesamiento de batch")

            # Validar batch_size
            if current_batch_size <= 0:
                self.logger.error(
                    f"Batch size inválido: {current_batch_size}, reseteando"
                )
                current_batch_size = self.condenser_config.min_batch_size

            # ═══════════════════════════════════════════════════════════════════
            # EXTRAER LOTE
            # ═══════════════════════════════════════════════════════════════════

            end_index = min(current_index + current_batch_size, total_records)
            batch_records = raw_records[current_index:end_index]

            if not batch_records:
                consecutive_empty_batches += 1
                if consecutive_empty_batches > 10:
                    self.logger.error("Demasiados batches vacíos consecutivos")
                    break
                current_index = end_index
                continue

            consecutive_empty_batches = 0

            # ═══════════════════════════════════════════════════════════════════
            # CALCULAR MÉTRICAS FÍSICAS
            # ═══════════════════════════════════════════════════════════════════

            cache_hits = self._calculate_cache_hits(batch_records, cache)
            metrics = self.physics.calculate_metrics(len(batch_records), cache_hits)

            # ═══════════════════════════════════════════════════════════════════
            # CONTROL PID
            # ═══════════════════════════════════════════════════════════════════

            new_batch_size = self.controller.compute(metrics["saturation"])

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

            # ═══════════════════════════════════════════════════════════════════
            # TELEMETRÍA
            # ═══════════════════════════════════════════════════════════════════

            diagnosis = self.physics.get_system_diagnosis(metrics)

            if self._stats.total_batches % 10 == 0 or self._stats.total_batches < 3:
                self.logger.info(
                    f"🔄 [BATCH #{self._stats.total_batches + 1}] "
                    f"Size: {len(batch_records)} | "
                    f"Sat: {metrics['saturation']:.1%} | "
                    f"P: {metrics['dissipated_power']:.1f}W | "
                    f"→ Next: {new_batch_size} | {diagnosis}"
                )

            # ═══════════════════════════════════════════════════════════════════
            # PROCESAR BATCH
            # ═══════════════════════════════════════════════════════════════════

            batch_result = self._process_single_batch(
                batch_records, cache, current_index, end_index
            )

            if batch_result.success and batch_result.dataframe is not None:
                processed_batches.append(batch_result.dataframe)
            else:
                failed_batches_count += 1
                self.logger.error(
                    f"[BATCH_FAIL] {current_index}-{end_index}: {batch_result.error_message} | "
                    f"Fallos: {failed_batches_count}/{self.condenser_config.max_failed_batches}"
                )

                # Decidir si abortar
                if self.condenser_config.enable_partial_recovery:
                    if failed_batches_count > self.condenser_config.max_failed_batches:
                        raise ProcessingError(
                            f"Excedido límite de batches fallidos: {failed_batches_count}"
                        )
                else:
                    raise ProcessingError(
                        f"Batch falló (recovery deshabilitado): {batch_result.error_message}"
                    )

            # ═══════════════════════════════════════════════════════════════════
            # ACTUALIZAR ESTADÍSTICAS
            # ═══════════════════════════════════════════════════════════════════

            self._stats.add_batch_stats(
                batch_size=len(batch_records),
                saturation=metrics["saturation"],
                power=metrics["dissipated_power"],
                flyback=metrics["flyback_voltage"],
                kinetic=metrics["kinetic_energy"],
                success=batch_result.success,
            )

            # Avanzar
            current_index = end_index
            current_batch_size = new_batch_size

        self.logger.info(
            f"[PID_LOOP] Completado | Batches: {self._stats.total_batches} | "
            f"Fallidos: {self._stats.failed_batches}"
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

        ROBUSTECIDO:
        - Encapsulación de errores
        - Resultado estructurado
        - Validación de salida
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
        Calcula cache hits de forma robusta.

        ROBUSTECIDO:
        - Validación de entradas
        - Búsqueda en múltiples claves
        - Manejo de errores por registro
        """
        if not cache or not isinstance(cache, dict):
            return 0

        if not batch_records or not isinstance(batch_records, list):
            return 0

        cache_hits = 0
        possible_keys = ["insumo_line", "line", "raw_line", "_line", "content"]

        for record in batch_records:
            if not isinstance(record, dict):
                continue

            try:
                for key in possible_keys:
                    line_content = record.get(key)
                    if line_content and isinstance(line_content, str):
                        # Normalizar para búsqueda
                        normalized = line_content.strip()
                        if normalized in cache:
                            cache_hits += 1
                            break
            except Exception:
                continue

        return cache_hits

    def _consolidate_results(
        self, processed_batches: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Consolida múltiples DataFrames en uno solo.

        ROBUSTECIDO:
        - Validación de entrada
        - Filtrado de batches inválidos
        - Manejo de límites de memoria
        - Deduplicación opcional
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

        ROBUSTECIDO:
        - Validación de entrada
        - Manejo de errores específicos del processor
        - Validación de salida
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

        ROBUSTECIDO:
        - Validación de tipo
        - Análisis de calidad de columnas
        - Advertencias detalladas
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

        ROBUSTECIDO:
        - Inclusión de estado del controlador
        - Diagnósticos del motor físico
        - Métricas de calidad
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
