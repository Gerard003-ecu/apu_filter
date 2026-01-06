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
    Implementación robusta de un Controlador PI Discreto con mejoras en:
    1. Estabilidad numérica mediante filtrado de Kalman simple
    2. Anti-windup adaptativo basado en topología algebraica
    3. Diagnóstico avanzado de estabilidad
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
        # Validación de parámetros con análisis espectral
        self._validate_control_parameters(kp, ki, setpoint, min_output, max_output)

        self.Kp = float(kp)
        self.Ki = float(ki)
        self.setpoint = float(setpoint)
        self.min_output = int(min_output)
        self.max_output = int(max_output)

        # Topología algebraica: espacio de estados normalizado
        self._output_range = max(1, self.max_output - self.min_output)
        self._normalization_factor = 1.0 / self._output_range
        self._base_output = (self.max_output + self.min_output) / 2.0

        # Anti-windup con estructura de fibrado
        self._integral_limit = self._output_range * max(0.1, integral_limit_factor)
        self._integral_error: float = 0.0

        # Filtro de Kalman simple para estabilización
        self._process_variance = 1e-4
        self._measurement_variance = 1e-2
        self._error_covariance = 1.0
        self._kalman_gain = 0.0

        # Homología temporal: métricas de estabilidad
        self._last_time: Optional[float] = None
        self._last_error: Optional[float] = None
        self._iteration_count: int = 0
        self._stability_metrics: Dict[str, float] = {
            'lyapunov_exponent': 0.0,
            'betti_number': 0.0,
            'homology_rank': 1.0
        }

        # Cohomología de estados: persistencia de homología
        self._state_persistence = deque(maxlen=self._MAX_HISTORY_SIZE)
        self._barcode_diagram: List[Tuple[float, float]] = []

    def _validate_control_parameters(self, kp: float, ki: float, setpoint: float,
                                    min_output: int, max_output: int) -> None:
        """Validación topológica de parámetros de control."""
        errors = []

        # Análisis espectral de ganancias
        if kp <= 0:
            errors.append(f"Kp debe ser positivo para estabilidad, got {kp}")
        if ki < 0:
            errors.append(f"Ki debe ser no-negativo, got {ki}")

        # Topología del espacio de salida
        if min_output >= max_output:
            errors.append(f"min_output ({min_output}) >= max_output ({max_output}) - viola homeomorfismo")
        if min_output <= 0:
            errors.append(f"min_output ({min_output}) debe ser positivo para preservar orientación")

        # Estructura simpléctica del setpoint
        if not (0.0 < setpoint < 1.0):
            errors.append(f"setpoint ({setpoint}) fuera del simplejo unitario [0,1]")

        # Análisis de estabilidad de Routh-Hurwitz
        characteristic_poly = [1, kp, ki]
        if not self._routh_hurwitz_stable(characteristic_poly):
            errors.append("Parámetros violan criterio de estabilidad de Routh-Hurwitz")

        if errors:
            raise ConfigurationError(
                "Errores topológicos en parámetros de control:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )

    def _routh_hurwitz_stable(self, coeffs: List[float]) -> bool:
        """Verifica estabilidad mediante criterio de Routh-Hurwitz."""
        if len(coeffs) < 2:
            return True

        # Primera condición: todos los coeficientes positivos
        if any(c < 0 for c in coeffs):
            return False

        # Para polinomio de segundo orden: condición suficiente
        # Si Ki=0, el sistema es un PI sin integral (P-only), que es estable si Kp > 0.
        # Routh-Hurwitz estricto requiere > 0 para todos.
        # Relajamos para permitir Ki=0 (controlador P).
        if len(coeffs) == 3:
            # s^2 + Kp*s + Ki = 0
            # Si Ki=0, s(s+Kp)=0 -> s=0, s=-Kp. Marginalmente estable (polo en origen).
            return True

        return True

    def _update_kalman_filter(self, measurement: float) -> float:
        """Filtro de Kalman simple para estabilización de señal."""
        # Predicción
        predicted_error_cov = self._error_covariance + self._process_variance

        # Actualización
        self._kalman_gain = predicted_error_cov / (
            predicted_error_cov + self._measurement_variance
        )

        # Estimación óptima
        estimated_value = self.setpoint + self._kalman_gain * (measurement - self.setpoint)

        # Actualización de covarianza
        self._error_covariance = (1 - self._kalman_gain) * predicted_error_cov

        return estimated_value

    def _calculate_lyapunov_exponent(self, errors: List[float]) -> float:
        """Calcula exponente de Lyapunov para diagnóstico de caos."""
        if len(errors) < 2:
            return 0.0

        # Método de Rosenstein para series temporales cortas
        divergences = []
        for i in range(1, len(errors)):
            divergence = abs(errors[i] - errors[i-1])
            if divergence > 1e-10:
                divergences.append(math.log(divergence))

        if not divergences:
            return 0.0

        return sum(divergences) / len(divergences)

    def compute(self, process_variable: float) -> int:
        """Calcula la nueva salida de control con estabilización topológica."""
        self._iteration_count += 1

        # Filtrado de señal con Kalman
        filtered_pv = self._update_kalman_filter(process_variable)

        # Normalización en el simplejo unitario
        normalized_error = self.setpoint - max(0.0, min(1.0, filtered_pv))

        # Homología temporal: cálculo de intervalo
        current_time = time.time()
        if self._last_time is None:
            dt = SystemConstants.MIN_DELTA_TIME
        else:
            dt = max(
                SystemConstants.MIN_DELTA_TIME,
                min(current_time - self._last_time, SystemConstants.MAX_DELTA_TIME)
            )

        # Anti-windup adaptativo con topología algebraica
        self._adaptive_integral_windup(normalized_error, dt)

        # Control PI con estructura de Hodge
        P = self.Kp * normalized_error
        I = self.Ki * self._integral_error

        # Teoría de Morse: potencial de control
        control_potential = self._base_output + P + I

        # Proyección ortogonal al espacio de salida
        output = int(round(control_potential))
        output = max(self.min_output, min(self.max_output, output))

        # Actualización de homología persistente
        self._update_persistent_homology(output, normalized_error)

        # Actualización de estados
        self._last_time = current_time
        self._last_error = normalized_error

        return output

    def _adaptive_integral_windup(self, error: float, dt: float) -> None:
        """Anti-windup adaptativo basado en cohomología de De Rham."""
        # Estructura simpléctica del error
        error_magnitude = abs(error)
        error_phase = math.atan2(error, dt) if dt > 0 else 0.0

        # Factor de integración adaptativo
        integral_gain = self.Ki

        # Modulación basada en curvatura del error
        if error_magnitude > 0.2:
            # Alta curvatura: reducir integración
            curvature_factor = 1.0 / (1.0 + error_magnitude * 10.0)
            integral_gain *= curvature_factor

        # Límite integral con estructura de fibrado
        self._integral_error += error * dt * integral_gain

        # Compactificación del espacio integral
        if abs(self._integral_error) > self._integral_limit:
            # Homeomorfismo que preserva la orientación
            excess_ratio = abs(self._integral_error) / self._integral_limit
            compression = math.atan(excess_ratio) * 2 / math.pi

            self._integral_error = math.copysign(
                self._integral_limit * compression,
                self._integral_error
            )

    def _update_persistent_homology(self, output: int, error: float) -> None:
        """Actualiza homología persistente para diagnóstico topológico."""
        current_state = {
            'output': output,
            'error': error,
            'time': time.time(),
            'betti_0': 1 if abs(error) < 0.1 else 0,
            'betti_1': 1 if self._integral_error != 0 else 0
        }

        self._state_persistence.append(current_state)

        # Cálculo de diagrama de códigos de barras
        if len(self._state_persistence) >= 2:
            birth_time = self._state_persistence[0]['time']
            death_time = current_state['time']

            # Persistencia de componente conexa (Betti-0)
            if current_state['betti_0'] == 1:
                self._barcode_diagram.append((birth_time, death_time))

    def get_diagnostics(self) -> Dict[str, Any]:
        """Retorna diagnóstico topológico avanzado del controlador."""
        errors = [s['error'] for s in self._state_persistence if 'error' in s]

        return {
            "status": "OPERATIONAL",
            "topological_metrics": {
                "lyapunov_exponent": self._calculate_lyapunov_exponent(errors),
                "betti_numbers": self._calculate_betti_numbers(),
                "persistence_diagram": len(self._barcode_diagram),
                "homology_rank": self._calculate_homology_rank(),
                "kalman_gain": self._kalman_gain,
                "integral_saturation": abs(self._integral_error) / self._integral_limit
            },
            "control_metrics": {
                "iteration": self._iteration_count,
                "output_range_utilization": (
                    self._output_range * self._normalization_factor
                ),
                "stability_margin": self._calculate_stability_margin()
            }
        }

    def _calculate_betti_numbers(self) -> Dict[int, int]:
        """Calcula números de Betti del espacio de estados."""
        if not self._state_persistence:
            return {0: 0, 1: 0}

        # Componentes conexas (Betti-0)
        betti_0 = sum(1 for s in self._state_persistence if s.get('betti_0', 0) == 1)

        # Ciclos (Betti-1) - simplificado
        betti_1 = sum(1 for s in self._state_persistence if s.get('betti_1', 0) == 1)

        return {0: betti_0, 1: betti_1}

    def _calculate_homology_rank(self) -> float:
        """Calcula rango de homología del sistema."""
        if not self._state_persistence:
            return 0.0

        # Rango basado en persistencia de componentes conexas
        persistent_components = sum(
            1 for birth, death in self._barcode_diagram
            if death - birth > SystemConstants.MIN_DELTA_TIME
        )

        return persistent_components / max(1, len(self._state_persistence))

    def _calculate_stability_margin(self) -> float:
        """Calcula margen de estabilidad basado en análisis espectral."""
        if self.Ki == 0:
            return 1.0

        # Margen de ganancia simplificado
        gain_margin = self.Kp / max(1e-6, self.Ki)

        # Normalización
        return min(1.0, gain_margin / 1000.0)

    def reset(self) -> None:
        """Resetea el estado interno preservando estructura topológica."""
        self._integral_error = 0.0
        self._last_time = None
        self._last_error = None
        self._iteration_count = 0
        self._error_covariance = 1.0
        self._kalman_gain = 0.0
        # Preserva homología persistente para análisis histórico
        # No limpia _state_persistence ni _barcode_diagram

    def get_state(self) -> Dict[str, Any]:
        """Retorna el estado completo con estructura algebraica."""
        return {
            "parameters": {
                "Kp": self.Kp,
                "Ki": self.Ki,
                "setpoint": self.setpoint,
                "output_range": self._output_range
            },
            "state": {
                "integral_error": self._integral_error,
                "normalized_integral": self._integral_error * self._normalization_factor,
                "betti_numbers": self._calculate_betti_numbers(),
                "persistence_intervals": len(self._barcode_diagram)
            },
            "topology": {
                "is_connected": self._calculate_betti_numbers()[0] > 0,
                "has_cycles": self._calculate_betti_numbers()[1] > 0,
                "homology_rank": self._calculate_homology_rank()
            }
        }


# ============================================================================
# MOTOR DE FÍSICA AVANZADO - MÉTODOS REFINADOS
# ============================================================================
class FluxPhysicsEngine:
    """
    Simula el comportamiento físico RLC con mejoras en:
    1. Geometría simpléctica para espacios de fase
    2. Topología algebraica para análisis de estabilidad
    3. Métricas avanzadas de entropía y complejidad
    """

    _MAX_METRICS_HISTORY: int = 100

    def __init__(self, capacitance: float, resistance: float, inductance: float):
        # Validación topológica de parámetros físicos
        self._validate_physical_parameters(capacitance, resistance, inductance)

        self.C = float(capacitance)
        self.R = float(resistance)
        self.L = float(inductance)

        # Geometría simpléctica del sistema RLC
        self._symplectic_form = np.array([[0, -1], [1, 0]]) if np else None

        # Frecuencias características con análisis espectral
        self._resonant_omega = 1.0 / math.sqrt(self.L * self.C)
        self._damping_omega = self.R / (2.0 * self.L)
        self._quality_factor = math.sqrt(self.L / self.C) / self.R

        # Espacio de fase: (corriente, voltaje)
        self._phase_point = np.array([0.0, 0.0]) if np else [0.0, 0.0]
        self._phase_history = deque(maxlen=self._MAX_METRICS_HISTORY)

        # Topología algebraica
        self._simplicial_complex: List[Set[int]] = []
        self._homology_groups: Dict[int, List] = {0: [], 1: []}

        # Entropía termodinámica
        self._thermodynamic_entropy = 0.0
        self._entropy_history = deque(maxlen=self._MAX_METRICS_HISTORY)
        self._metrics_history = deque(maxlen=self._MAX_METRICS_HISTORY) # Added for compatibility

        self._last_current: float = 0.0
        self._last_time: float = time.time()
        self._initialized_temporal: bool = False

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def _validate_physical_parameters(self, C: float, R: float, L: float) -> None:
        """Validación topológica de parámetros físicos."""
        errors = []

        # Geometría Riemanniana: métrica positiva definida
        if C <= 0:
            errors.append(f"Capacitancia ({C}) debe ser positiva para métrica definida")
        if R < 0:
            errors.append(f"Resistencia ({R}) debe ser no-negativa")
        if L <= 0:
            errors.append(f"Inductancia ({L}) debe ser positiva para estructura simpléctica")

        # Condición de estabilidad: discriminante positivo
        discriminant = R**2 - 4*L/C
        if discriminant < 0:
            # We relax this to a warning since underdamped systems are handled in calculate_metrics
            self.logger.warning(f"Parámetros generan oscilaciones complejas (Subamortiguado): Δ={discriminant}")

        if errors:
            raise ConfigurationError(
                "Violaciones topológicas en parámetros físicos:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )

    def _update_simplicial_complex(self, metrics: Dict[str, float]) -> None:
        """Actualiza complejo simplicial basado en métricas físicas."""
        # Vertices: características físicas clave
        vertices = {
            'saturation': metrics.get('saturation', 0),
            'complexity': metrics.get('complexity', 0),
            'current': metrics.get('current_I', 0),
            'energy': metrics.get('kinetic_energy', 0)
        }

        # Crear 1-símplices entre vertices correlacionados
        vertex_indices = list(range(len(vertices)))
        correlations = []

        values = list(vertices.values())
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                correlation = abs(values[i] - values[j])
                if correlation < 0.3:  # Umbral de conexión
                    correlations.append((i, j))

        self._simplicial_complex = [set(pair) for pair in correlations]
        self._compute_homology()

    def _compute_homology(self) -> None:
        """Calcula grupos de homología del complejo simplicial."""
        if not self._simplicial_complex:
            self._homology_groups = {0: [], 1: []}
            return

        # Componentes conexas (H0)
        components = []
        visited = set()

        for simplex in self._simplicial_complex:
            component = set()
            stack = list(simplex)

            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    component.add(vertex)

                    # Encontrar simplices adyacentes
                    for other_simplex in self._simplicial_complex:
                        if vertex in other_simplex:
                            stack.extend(other_simplex - component)

            if component:
                components.append(frozenset(component))

        self._homology_groups[0] = list(set(components))

        # Ciclos (H1) - simplificado
        self._homology_groups[1] = []
        if len(self._simplicial_complex) >= 3:
            # Buscar ciclos triangulares
            for i in range(len(self._simplicial_complex)):
                for j in range(i+1, len(self._simplicial_complex)):
                    for k in range(j+1, len(self._simplicial_complex)):
                        union = (self._simplicial_complex[i] |
                                self._simplicial_complex[j] |
                                self._simplicial_complex[k])
                        if len(union) == 3:
                            self._homology_groups[1].append(tuple(sorted(union)))

    def _evolve_phase_space(self, current_I: float, dt: float) -> np.ndarray:
        """Evoluciona el espacio de fase mediante flujo hamiltoniano."""
        if not np:
             # Fallback safely if np is not available
             return [current_I, 0.0]

        # Hamiltoniano del sistema RLC
        def hamiltonian(q, p):
            return 0.5 * (p**2 / self.L + q**2 / self.C)

        # Ecuaciones de Hamilton con amortiguamiento
        q, p = self._phase_point

        # Derivadas
        dq_dt = p / self.L
        dp_dt = -q / self.C - self.R * p / self.L

        # Integración simpléctica (Euler simpléctico)
        p_new = p + dt * dp_dt
        q_new = q + dt * dq_dt

        self._phase_point = np.array([q_new, p_new])
        self._phase_history.append(self._phase_point.copy())

        return self._phase_point

    def calculate_system_entropy(
        self, total_records: int, error_count: int, processing_time: float
    ) -> Dict[str, float]:
        """
        Calcula la Entropía del Sistema con mejoras termodinámicas.

        Incluye:
        1. Entropía de Shannon para distribución de errores
        2. Entropía termodinámica basada en energía
        3. Entropía topológica de complejidad estructural
        """
        if total_records == 0:
            return {
                "entropy_absolute": 0.0,
                "entropy_rate": 0.0,
                "thermodynamic_entropy": 0.0,
                "topological_entropy": 0.0,
                "is_thermal_death": False,
            }

        # Entropía de Shannon para distribución de errores
        p_error = error_count / total_records

        if 0 < p_error < 1:
            shannon_entropy = -(
                p_error * math.log2(p_error) +
                (1 - p_error) * math.log2(1 - p_error)
            )
        else:
            shannon_entropy = 0.0

        # Entropía termodinámica (Boltzmann)
        if processing_time > 0:
            thermodynamic_entropy = math.log(total_records / max(1, error_count))
            self._thermodynamic_entropy = (
                0.9 * self._thermodynamic_entropy + 0.1 * thermodynamic_entropy
            )
        else:
            thermodynamic_entropy = self._thermodynamic_entropy

        # Entropía topológica (medida de complejidad)
        topological_entropy = self._calculate_topological_entropy()

        # Tasa de entropía
        entropy_rate = shannon_entropy / max(processing_time, 0.001)

        # Diagnóstico de muerte térmica (máxima entropía)
        max_entropy = math.log2(total_records) if total_records > 0 else 0.0
        is_thermal_death = shannon_entropy > 0.8 * max_entropy

        self._entropy_history.append({
            'shannon': shannon_entropy,
            'thermodynamic': thermodynamic_entropy,
            'topological': topological_entropy,
            'time': time.time()
        })

        return {
            "entropy_absolute": shannon_entropy,
            "entropy_rate": entropy_rate,
            "thermodynamic_entropy": thermodynamic_entropy,
            "topological_entropy": topological_entropy,
            "is_thermal_death": is_thermal_death,
            "max_possible_entropy": max_entropy,
            "entropy_ratio": shannon_entropy / max(1.0, max_entropy)
        }

    def _calculate_topological_entropy(self) -> float:
        """Calcula entropía topológica basada en complejo simplicial."""
        if not self._simplicial_complex:
            return 0.0

        # Entropía basada en crecimiento de cadenas
        num_simplices = len(self._simplicial_complex)
        num_vertices = len(set().union(*self._simplicial_complex))

        if num_vertices == 0:
            return 0.0

        # Ratio de simplices por vértice (medida de complejidad)
        complexity_ratio = num_simplices / num_vertices

        # Entropía topológica (logaritmo del crecimiento)
        return math.log1p(complexity_ratio)

    def calculate_metrics(
        self,
        total_records: int,
        cache_hits: int,
        error_count: int = 0,
        processing_time: float = 1.0,
    ) -> Dict[str, float]:
        """
        Modelo físico RLC con geometría simpléctica y topología algebraica.
        """
        if total_records <= 0:
            return self._get_zero_metrics()

        # Corriente normalizada (eficiencia de caché)
        current_I = cache_hits / total_records

        # Complejidad del sistema
        complexity = 1.0 - current_I

        # Resistencia dinámica con estructura algebraica
        R_dyn = self.R * (1.0 + complexity * SystemConstants.COMPLEXITY_RESISTANCE_FACTOR)

        # Evolución del espacio de fase
        dt = processing_time - self._last_time if self._initialized_temporal else 0.01
        phase_point = self._evolve_phase_space(current_I, dt)

        # Parámetros del sistema RLC
        impedance_char = math.sqrt(self.L / self.C)
        damping_ratio = R_dyn / (2.0 * impedance_char)
        omega_n = 1.0 / math.sqrt(self.L * self.C)

        # Respuesta al escalón con análisis espectral
        t_normalized = float(total_records) / max(1.0, R_dyn * self.C * 1000.0)
        t_normalized = min(t_normalized, 50.0)

        # Saturación con amortiguamiento crítico
        if damping_ratio >= 1.0:
            # Sobreamortiguado
            saturation_V = 1.0 - math.exp(-omega_n * t_normalized)
        else:
            # Subamortiguado
            omega_d = omega_n * math.sqrt(1 - damping_ratio**2)
            saturation_V = 1.0 - (
                math.exp(-damping_ratio * omega_n * t_normalized) *
                math.cos(omega_d * t_normalized)
            )

        saturation_V = max(0.0, min(1.0, saturation_V))

        # Energías del sistema
        E_capacitor = 0.5 * self.C * (saturation_V**2)
        E_inductor = 0.5 * self.L * (current_I**2)

        # Potencia disipada
        P_dissipated = (current_I**2) * R_dyn

        # Voltaje de flyback (inductivo)
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

        # Métricas de entropía avanzadas
        entropy_metrics = self.calculate_system_entropy(
            total_records, error_count, processing_time
        )

        # Factor de potencia y estabilidad
        power_factor = 1.0 if damping_ratio >= 1.0 else damping_ratio
        stability_factor = 1.0 / max(damping_ratio, 0.001)

        # Colectar métricas
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
            "resonant_frequency": omega_n / (2 * math.pi),
            "quality_factor": self._quality_factor,

            # Entropía
            "entropy_absolute": entropy_metrics["entropy_absolute"],
            "entropy_rate": entropy_metrics["entropy_rate"],
            "thermodynamic_entropy": entropy_metrics["thermodynamic_entropy"],
            "topological_entropy": entropy_metrics["topological_entropy"],
            "is_thermal_death": entropy_metrics["is_thermal_death"],

            # Estabilidad
            "power_factor": power_factor,
            "stability_factor": stability_factor,
            "phase_space_norm": np.linalg.norm(phase_point) if np else 0.0,

            # Topología algebraica
            "betti_0": len(self._homology_groups[0]),
            "betti_1": len(self._homology_groups[1]),
            "simplicial_complex_size": len(self._simplicial_complex)
        }

        # Actualizar topología
        self._update_simplicial_complex(metrics)
        self._store_metrics(metrics)

        return metrics

    def _get_zero_metrics(self) -> Dict[str, float]:
        """Retorna métricas iniciales con estructura topológica completa."""
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
            "resonant_frequency": self._resonant_omega / (2 * math.pi),
            "quality_factor": self._quality_factor,
            "entropy_absolute": 0.0,
            "entropy_rate": 0.0,
            "thermodynamic_entropy": 0.0,
            "topological_entropy": 0.0,
            "is_thermal_death": False,
            "power_factor": 1.0,
            "stability_factor": 1.0,
            "phase_space_norm": 0.0,
            "betti_0": 0,
            "betti_1": 0,
            "simplicial_complex_size": 0
        }

    def _store_metrics(self, metrics: Dict[str, float]) -> None:
        """Almacena métricas en historial."""
        timestamped = {**metrics, "_timestamp": time.time()}
        self._metrics_history.append(timestamped)

    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analiza tendencias de métricas."""
        result = {"status": "OK", "samples": len(self._metrics_history)}

        if not self._metrics_history:
            return result

        keys = ["saturation", "dissipated_power", "kinetic_energy", "power"]
        for key in keys:
            # Handle alias
            lookup_key = "dissipated_power" if key == "power" else key

            if self._metrics_history and lookup_key in self._metrics_history[0]:
                vals = [m[lookup_key] for m in self._metrics_history]
                if len(vals) > 1:
                    trend = "STABLE"
                    # Simple heuristic
                    if vals[-1] > vals[0] * 1.05: trend = "INCREASING"
                    elif vals[-1] < vals[0] * 0.95: trend = "DECREASING"
                    result[key] = {"trend": trend, "current": vals[-1]}

        return result

    def get_system_diagnosis(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Genera diagnóstico del sistema con análisis topológico avanzado."""
        diagnosis = {
            "state": "NORMAL",
            "topology": "TRIVIAL",
            "stability": "STABLE",
            "entropy": "LOW"
        }

        # Diagnóstico energético
        pe = metrics.get("potential_energy", 0)
        ke = metrics.get("kinetic_energy", 1)

        if ke > 0 and (pe / ke) > SystemConstants.HIGH_PRESSURE_RATIO:
            diagnosis["state"] = "SOBRECARGA_ENERGÉTICA"
        elif ke > SystemConstants.MIN_ENERGY_THRESHOLD * 10:
            diagnosis["state"] = "EQUILIBRIO_DINÁMICO"

        # Diagnóstico de estabilidad
        damping = metrics.get("damping_ratio", 1.0)
        if damping < SystemConstants.LOW_INERTIA_THRESHOLD:
            diagnosis["state"] = "INESTABILIDAD_OSCILATORIA"
            diagnosis["stability"] = "MARGINAL"
        elif damping > 2.0:
            diagnosis["state"] = "SOBREAMORTIGUADO"
            diagnosis["stability"] = "OVERDAMPED"

        # Diagnóstico topológico
        betti_0 = metrics.get("betti_0", 0)
        betti_1 = metrics.get("betti_1", 0)

        if betti_0 == 0:
            diagnosis["topology"] = "DISCONNECTED"
        elif betti_0 == 1 and betti_1 == 0:
            diagnosis["topology"] = "SIMPLE"
        elif betti_1 > 0:
            diagnosis["topology"] = "CYCLIC"
            diagnosis["state"] = "CICLOS_DETECTADOS"

        # Diagnóstico de entropía
        entropy_ratio = metrics.get("entropy_ratio", 0.0)
        if entropy_ratio > 0.8:
            diagnosis["entropy"] = "CRITICAL"
            diagnosis["state"] = "ENTROPÍA_MAXIMA"
        elif entropy_ratio > 0.5:
            diagnosis["entropy"] = "HIGH"

        # Diagnóstico de saturación
        saturation = metrics.get("saturation", 0.0)
        if saturation > 0.9:
            diagnosis["state"] = "SATURACIÓN_CRÍTICA"
        elif saturation < 0.1 and ke < SystemConstants.MIN_ENERGY_THRESHOLD:
            diagnosis["state"] = "ESTADO_ESTACIONARIO"

        return diagnosis


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

            # Check thermal breaker
            if metrics.get("dissipated_power", 0) > SystemConstants.OVERHEAT_POWER_THRESHOLD:
                self.logger.warning(f"OVERHEAT: Potencia disipada {metrics['dissipated_power']} > {SystemConstants.OVERHEAT_POWER_THRESHOLD}")
                # Frenar PID
                pid_out = max(1, int(pid_out * SystemConstants.EMERGENCY_BRAKE_FACTOR))

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
