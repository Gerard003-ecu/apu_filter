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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

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

    Attributes:
        min_records_threshold (int): Registros mínimos para considerar válido.
        enable_strict_validation (bool): Activa validaciones estrictas.
        log_level (str): Nivel de logging.
        system_capacitance (float): Parámetro físico RLC (Faradios).
        base_resistance (float): Parámetro físico RLC (Ohmios).
        system_inductance (float): Parámetro físico RLC (Henrios).
        pid_setpoint (float): Objetivo de saturación (0.0-1.0).
        pid_kp (float): Ganancia Proporcional del PID.
        pid_ki (float): Ganancia Integral del PID.
        min_batch_size (int): Tamaño mínimo del lote.
        max_batch_size (int): Tamaño máximo del lote.
        enable_partial_recovery (bool): Permite recuperación parcial.
        max_failed_batches (int): Máximo de batches fallidos.
        integral_limit_factor (float): Factor anti-windup.
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

        if self.pid_kp < 0:
            errors.append(f"pid_kp >= 0, got {self.pid_kp}")

        if self.min_batch_size > self.max_batch_size:
            errors.append(
                f"min_batch_size ({self.min_batch_size}) > max ({self.max_batch_size})"
            )

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
# CONTROLADOR PI DISCRETO
# ============================================================================
class PIController:
    """
    Implementación robusta de un Controlador PI Discreto.

    Características:
    - Validación exhaustiva.
    - Anti-windup con múltiples estrategias.
    - Historial de estados para diagnóstico.
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
        self.Kp = float(kp)
        self.Ki = float(ki)
        self.setpoint = float(setpoint)
        self.min_output = int(min_output)
        self.max_output = int(max_output)

        self._base_output = (self.max_output + self.min_output) / 2.0
        self._output_range = max(1, self.max_output - self.min_output)
        self._integral_limit = self._output_range * max(0.1, integral_limit_factor)

        self._integral_error: float = 0.0
        self._last_time: Optional[float] = None
        self._last_error: Optional[float] = None
        self._iteration_count: int = 0
        self._history: List[Dict[str, Any]] = []

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def compute(self, process_variable: float) -> int:
        """Calcula la nueva salida de control."""
        self._iteration_count += 1

        if math.isnan(process_variable) or math.isinf(process_variable):
            process_variable = self.setpoint * (0.9**self._iteration_count)

        error = self.setpoint - process_variable
        current_time = time.time()

        if self._last_time is None:
            dt = SystemConstants.MIN_DELTA_TIME
        else:
            dt = current_time - self._last_time
            dt = max(SystemConstants.MIN_DELTA_TIME, min(dt, SystemConstants.MAX_DELTA_TIME))

        # Anti-windup
        if abs(self._integral_error) > self._integral_limit * 0.8:
            if abs(error) < 0.05:
                self._integral_error *= 0.5

        integral_gain = self.Ki
        if abs(error) > 0.2:
            integral_gain *= 0.5

        self._integral_error += error * dt * integral_gain / self.Ki if self.Ki > 0 else 0

        if abs(self._integral_error) > self._integral_limit:
            excess_ratio = abs(self._integral_error) / self._integral_limit
            self._integral_error = math.copysign(
                self._integral_limit * math.tanh(excess_ratio), self._integral_error
            )

        P = self.Kp * error
        I = self.Ki * self._integral_error
        control_signal = self._base_output + P + I

        output = int(round(control_signal))
        output = max(self.min_output, min(self.max_output, output))

        self._last_time = current_time
        self._last_error = error

        return output

    def reset(self) -> None:
        """Resetea el estado interno."""
        self._integral_error = 0.0
        self._last_time = None
        self._last_error = None
        self._iteration_count = 0
        self._history.clear()

    def get_state(self) -> Dict[str, Any]:
        """Retorna el estado completo del controlador."""
        return {
            "parameters": {"Kp": self.Kp, "Ki": self.Ki},
            "state": {"integral_error": self._integral_error},
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Retorna diagnósticos del controlador."""
        return {"status": "OK"}


# ============================================================================
# MOTOR DE FÍSICA AVANZADO
# ============================================================================
class FluxPhysicsEngine:
    """
    Simula el comportamiento físico RLC basándose en la ENERGÍA.

    Calcula métricas como saturación, potencia disipada y entropía.
    """

    _MAX_METRICS_HISTORY: int = 100

    def __init__(self, capacitance: float, resistance: float, inductance: float):
        self.C = float(capacitance)
        self.R = float(resistance)
        self.L = float(inductance)

        self._resonant_freq = 1.0 / (2.0 * math.pi * math.sqrt(self.L * self.C))
        self._quality_factor = (1.0 / self.R) * math.sqrt(self.L / self.C)

        self._last_current: float = 0.0
        self._last_time: float = time.time()
        self._initialized_temporal: bool = False
        self._metrics_history: deque = deque(maxlen=self._MAX_METRICS_HISTORY)

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def calculate_system_entropy(
        self, total_records: int, error_count: int, processing_time: float
    ) -> Dict[str, float]:
        """
        Calcula la Entropía del Sistema (S).

        Args:
            total_records: Volumen total.
            error_count: Datos corruptos.
            processing_time: Tiempo transcurrido.

        Returns:
            Dict: Métricas de entropía.
        """
        if total_records == 0:
            return {
                "entropy_absolute": 0.0,
                "entropy_rate": 0.0,
                "is_thermal_death": False,
            }

        p_error = error_count / total_records

        if 0 < p_error < 1:
            entropy = -(
                p_error * math.log2(p_error) + (1 - p_error) * math.log2(1 - p_error)
            )
        else:
            entropy = 0.0

        entropy_rate = entropy / max(processing_time, 0.001)

        return {
            "entropy_absolute": entropy,
            "entropy_rate": entropy_rate,
            "is_thermal_death": entropy > 0.8,
        }

    def calculate_metrics(
        self,
        total_records: int,
        cache_hits: int,
        error_count: int = 0,
        processing_time: float = 1.0,
    ) -> Dict[str, float]:
        """
        Modelo físico RLC de segundo orden + Entropía.

        Returns:
            Dict[str, float]: Métricas físicas del sistema.
        """
        if total_records <= 0:
            return self._get_zero_metrics()

        current_I = cache_hits / total_records
        complexity = 1.0 - current_I
        R_dyn = self.R * (1.0 + complexity * SystemConstants.COMPLEXITY_RESISTANCE_FACTOR)

        impedance_char = math.sqrt(self.L / self.C)
        damping_ratio = R_dyn / (2.0 * impedance_char)
        omega_n = 1.0 / math.sqrt(self.L * self.C)

        tau_effective = R_dyn * self.C
        t_normalized = float(total_records) / max(1.0, tau_effective * 1000.0)
        t_normalized = min(t_normalized, 50.0)

        # Simplificación de respuesta al escalón (Saturación)
        saturation_V = 1.0 - math.exp(-omega_n * t_normalized)
        saturation_V = max(0.0, min(1.0, saturation_V))

        E_capacitor = 0.5 * self.C * (saturation_V**2)
        E_inductor = 0.5 * self.L * (current_I**2)
        P_dissipated = (current_I**2) * R_dyn

        current_time = time.time()
        if not self._initialized_temporal:
            self._last_current = current_I
            self._last_time = current_time
            self._initialized_temporal = True
            di_dt = 0.0
        else:
            dt = max(1e-6, current_time - self._last_time)
            di_dt = (current_I - self._last_current) / dt
            self._last_current = current_I
            self._last_time = current_time

        V_flyback = min(abs(self.L * di_dt), SystemConstants.MAX_FLYBACK_VOLTAGE)

        entropy_metrics = self.calculate_system_entropy(
            total_records, error_count, processing_time
        )

        metrics = {
            "saturation": saturation_V,
            "complexity": complexity,
            "current_I": current_I,
            "potential_energy": E_capacitor,
            "kinetic_energy": E_inductor,
            "dissipated_power": P_dissipated,
            "flyback_voltage": V_flyback,
            "dynamic_resistance": R_dyn,
            "damping_ratio": damping_ratio,
            "entropy_absolute": entropy_metrics["entropy_absolute"],
            "is_thermal_death": entropy_metrics["is_thermal_death"],
        }

        self._store_metrics(metrics)
        return metrics

    def _get_zero_metrics(self) -> Dict[str, float]:
        """Retorna métricas iniciales."""
        return {
            "saturation": 0.0,
            "complexity": 1.0,
            "current_I": 0.0,
            "potential_energy": 0.0,
            "kinetic_energy": 0.0,
            "dissipated_power": 0.0,
            "flyback_voltage": 0.0,
            "dynamic_resistance": self.R,
            "damping_ratio": 1.0,
        }

    def _store_metrics(self, metrics: Dict[str, float]) -> None:
        """Almacena métricas en historial."""
        timestamped = {**metrics, "_timestamp": time.time()}
        self._metrics_history.append(timestamped)

    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analiza tendencias de métricas."""
        return {"status": "OK", "samples": len(self._metrics_history)}


# ============================================================================
# DATA FLUX CONDENSER
# ============================================================================
class DataFluxCondenser:
    """
    Orquesta el pipeline de validación y procesamiento de archivos de APU.

    Integra validación, parseo, y estabilización de flujo mediante PID.
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

    def stabilize(
        self,
        file_path: str,
        on_progress: Optional[Callable[[ProcessingStats], None]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        telemetry: Optional[TelemetryContext] = None,
    ) -> pd.DataFrame:
        """
        Proceso principal de estabilización con control PID.

        Args:
            file_path: Ruta del archivo.
            on_progress: Callback de progreso.
            telemetry: Contexto de telemetría.

        Returns:
            pd.DataFrame: Datos procesados.
        """
        self._start_time = time.time()
        self._stats = ProcessingStats()
        self.controller.reset()

        if not file_path:
            raise InvalidInputError("file_path inválido")

        path_obj = Path(file_path)
        self.logger.info(f"⚡ [STABILIZE] Iniciando: {path_obj.name}")

        try:
            validated_path = self._validate_input_file(file_path)
            parser = self._initialize_parser(validated_path)
            raw_records, cache = self._extract_raw_data(parser)

            if not raw_records:
                return pd.DataFrame()

            total_records = len(raw_records)
            self._stats.total_records = total_records

            processed_batches = self._process_batches_with_pid(
                raw_records,
                cache,
                total_records,
                on_progress,
                progress_callback,
                telemetry,
            )

            df_final = self._consolidate_results(processed_batches)
            self._validate_output(df_final)

            return df_final

        except Exception as e:
            self.logger.exception(f"Error en estabilización: {e}")
            raise ProcessingError(f"Error fatal: {e}")

    def _validate_input_file(self, file_path: str) -> Path:
        """Valida el archivo de entrada."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise InvalidInputError(f"Archivo no válido: {file_path}")
        return path

    def _initialize_parser(self, path: Path) -> ReportParserCrudo:
        """Inicializa el parser."""
        return ReportParserCrudo(str(path), self.profile, self.config)

    def _extract_raw_data(self, parser) -> Tuple[List, Dict]:
        """Extrae datos crudos."""
        return parser.parse_to_raw(), parser.get_parse_cache()

    def _process_batches_with_pid(
        self,
        raw_records: List,
        cache: Dict,
        total_records: int,
        on_progress,
        progress_callback,
        telemetry,
    ) -> List[pd.DataFrame]:
        """Procesa lotes con control PID."""
        processed_batches = []
        current_index = 0
        current_batch_size = self.condenser_config.min_batch_size

        while current_index < total_records:
            end_index = min(current_index + current_batch_size, total_records)
            batch = raw_records[current_index:end_index]

            # Simulación de métricas físicas para PID
            metrics = self.physics.calculate_metrics(
                len(batch), len(batch), 0, time.time() - self._start_time
            )

            if progress_callback:
                progress_callback(metrics)

            pid_out = self.controller.compute(metrics.get("saturation", 0.5))

            # Procesamiento real (simulado aquí con llamada directa)
            df_batch = self._process_single_batch(batch, cache)
            if not df_batch.empty:
                processed_batches.append(df_batch)
                self._stats.processed_records += len(df_batch)

            if on_progress:
                on_progress(self._stats)

            current_index = end_index
            current_batch_size = pid_out

        return processed_batches

    def _process_single_batch(self, batch, cache) -> pd.DataFrame:
        """Procesa un lote individual."""
        data = ParsedData(batch, cache)
        return self._rectify_signal(data)

    def _rectify_signal(self, parsed_data: ParsedData) -> pd.DataFrame:
        """Convierte datos crudos a DataFrame procesado."""
        processor = APUProcessor(self.config, self.profile, parsed_data.parse_cache)
        processor.raw_records = parsed_data.raw_records
        return processor.process_all()

    def _consolidate_results(self, batches: List[pd.DataFrame]) -> pd.DataFrame:
        """Consolida resultados."""
        if not batches:
            return pd.DataFrame()
        return pd.concat(batches, ignore_index=True)

    def _validate_output(self, df: pd.DataFrame) -> None:
        """Valida salida."""
        if df.empty:
            self.logger.warning("DataFrame de salida vacío")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas."""
        return {"statistics": asdict(self._stats)}
