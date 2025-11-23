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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set

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

    # Límites físicos
    MIN_ENERGY_THRESHOLD: float = 1e-10  # Julios mínimos para cálculos
    MAX_EXPONENTIAL_ARG: float = 100.0   # Límite para evitar overflow en exp()

    # Diagnóstico
    LOW_INERTIA_THRESHOLD: float = 0.1
    HIGH_PRESSURE_RATIO: float = 1000.0
    HIGH_FLYBACK_THRESHOLD: float = 0.5
    OVERHEAT_POWER_THRESHOLD: float = 50.0  # Watts

    # Control de flujo
    EMERGENCY_BRAKE_FACTOR: float = 0.5
    MAX_ITERATIONS_MULTIPLIER: int = 10  # max_iterations = total_records * multiplier

    # Validación de archivos
    VALID_FILE_EXTENSIONS: Set[str] = {'.csv', '.txt', '.tsv'}

    # Resistencia dinámica
    COMPLEXITY_RESISTANCE_FACTOR: float = 5.0


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
            errors.append(f"min_records_threshold debe ser >= 0, recibido: {self.min_records_threshold}")

        # Validar parámetros físicos
        if self.system_capacitance <= 0:
            errors.append(f"system_capacitance debe ser > 0, recibido: {self.system_capacitance}")

        if self.base_resistance <= 0:
            errors.append(f"base_resistance debe ser > 0, recibido: {self.base_resistance}")

        if self.system_inductance <= 0:
            errors.append(f"system_inductance debe ser > 0, recibido: {self.system_inductance}")

        # Validar PID
        if not 0.0 <= self.pid_setpoint <= 1.0:
            errors.append(f"pid_setpoint debe estar en [0.0, 1.0], recibido: {self.pid_setpoint}")

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
            errors.append(f"max_failed_batches debe ser >= 0, recibido: {self.max_failed_batches}")

        if self.integral_limit_factor <= 0:
            errors.append(f"integral_limit_factor debe ser > 0, recibido: {self.integral_limit_factor}")

        # Validar log level
        valid_log_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"log_level debe ser uno de {valid_log_levels}, recibido: {self.log_level}")

        if errors:
            raise ConfigurationError(
                "Errores de configuración detectados:\n" + "\n".join(f"  - {e}" for e in errors)
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

    def add_batch_stats(self, batch_size: int, saturation: float, power: float, flyback: float, kinetic: float, success: bool) -> None:
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


# ============================================================================
# CONTROLADOR PI DISCRETO
# ============================================================================
class PIController:
    """
    Implementación robusta de un Controlador PI Discreto según la teoría de control.

    Objetivo: Mantener la saturación del sistema en un Setpoint (SP) estable,
    ajustando dinámicamente la variable de control (Tamaño del Batch).
    
    Mejoras implementadas:
    - Validación exhaustiva de parámetros
    - Anti-windup explícito con límites configurables
    - Protección contra delta_time inválido
    - Reset capability para reutilización
    - Logging detallado de estado
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        setpoint: float,
        min_output: int,
        max_output: int,
        integral_limit_factor: float = 2.0
    ):
        """
        Inicializa el controlador PI con validación de parámetros.
        
        Args:
            kp: Ganancia proporcional (debe ser >= 0)
            ki: Ganancia integral (debe ser >= 0)
            setpoint: Valor objetivo (debe estar en [0, 1])
            min_output: Salida mínima del actuador (debe ser > 0)
            max_output: Salida máxima del actuador (debe ser > min_output)
            integral_limit_factor: Factor para límites de anti-windup
        
        Raises:
            ConfigurationError: Si algún parámetro es inválido
        """
        self._validate_parameters(kp, ki, setpoint, min_output, max_output, integral_limit_factor)

        self.Kp = kp
        self.Ki = ki
        self.setpoint = setpoint

        self.min_output = min_output
        self.max_output = max_output

        # Calcular salida base (punto medio del rango)
        self._base_output = (self.max_output + self.min_output) / 2.0
        self._output_range = self.max_output - self.min_output

        # Anti-windup: Límites para el término integral
        self._integral_limit = self._output_range * integral_limit_factor

        # Estado interno
        self._integral_error = 0.0
        self._last_time: Optional[float] = None
        self._iteration_count = 0

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def _validate_parameters(
        self,
        kp: float,
        ki: float,
        setpoint: float,
        min_output: int,
        max_output: int,
        integral_limit_factor: float
    ) -> None:
        """Valida todos los parámetros del controlador."""
        errors = []

        if kp < 0:
            errors.append(f"Kp debe ser >= 0, recibido: {kp}")

        if ki < 0:
            errors.append(f"Ki debe ser >= 0, recibido: {ki}")

        if not 0.0 <= setpoint <= 1.0:
            errors.append(f"setpoint debe estar en [0.0, 1.0], recibido: {setpoint}")

        if min_output <= 0:
            errors.append(f"min_output debe ser > 0, recibido: {min_output}")

        if max_output <= min_output:
            errors.append(
                f"max_output ({max_output}) debe ser > min_output ({min_output})"
            )

        if integral_limit_factor <= 0:
            errors.append(f"integral_limit_factor debe ser > 0, recibido: {integral_limit_factor}")

        if errors:
            raise ConfigurationError(
                "Parámetros inválidos del PIController:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def compute(self, process_variable: float) -> int:
        """
        Calcula la nueva salida de control (u(t)) basada en el error actual.

        Ecuación Posicional Discreta con Anti-Windup:
        u(k) = base_output + Kp * e(k) + Ki * sum(e) * dt
        
        donde sum(e) está limitado para prevenir saturación del integrador.
        
        Args:
            process_variable: Valor actual del proceso (saturación medida)
        
        Returns:
            Señal de control (batch size) clampeada al rango válido
        
        Raises:
            ValueError: Si process_variable está fuera del rango válido
        """
        # Validar entrada
        if not isinstance(process_variable, (int, float)):
            raise ValueError(
                f"process_variable debe ser numérico, recibido: {type(process_variable)}"
            )

        if math.isnan(process_variable) or math.isinf(process_variable):
            self.logger.warning(
                f"process_variable inválido ({process_variable}), usando setpoint como fallback"
            )
            process_variable = self.setpoint

        # Normalizar a rango válido [0, 1]
        process_variable = max(0.0, min(1.0, process_variable))

        # Calcular delta de tiempo
        current_time = time.time()

        if self._last_time is None:
            # Primera iteración: no hay historia
            dt = SystemConstants.MIN_DELTA_TIME
        else:
            dt = current_time - self._last_time
            # Protección contra tiempo inválido
            if dt <= 0 or dt > 3600:  # Si dt > 1 hora, algo está mal
                self.logger.warning(f"Delta de tiempo anómalo: {dt}s, usando mínimo")
                dt = SystemConstants.MIN_DELTA_TIME

        # 1. Calcular Error (e(t))
        # Invertimos el signo: saturación alta -> error negativo -> reducir batch
        error = self.setpoint - process_variable

        # 2. Término Proporcional
        P = self.Kp * error

        # 3. Término Integral con Anti-Windup
        # Acumular error
        self._integral_error += error * dt

        # Aplicar límites de anti-windup (clamping del integrador)
        self._integral_error = max(
            -self._integral_limit,
            min(self._integral_limit, self._integral_error)
        )

        I = self.Ki * self._integral_error

        # 4. Señal de Control (u)
        control_signal = self._base_output + P + I

        # 5. Saturación del Actuador (Clamping de salida)
        output = int(round(control_signal))
        output = max(self.min_output, min(self.max_output, output))

        # Actualizar estado
        self._last_time = current_time
        self._iteration_count += 1

        # Logging cada 10 iteraciones para evitar spam
        if self._iteration_count % 10 == 0:
            self.logger.debug(
                f"[PID] Iter={self._iteration_count} | PV={process_variable:.3f} | "
                f"Error={error:.3f} | P={P:.1f} | I={I:.1f} | Out={output}"
            )

        return output

    def reset(self) -> None:
        """Resetea el estado interno del controlador para reutilización."""
        self._integral_error = 0.0
        self._last_time = None
        self._iteration_count = 0
        self.logger.debug("[PID] Controlador reseteado")

    def get_state(self) -> Dict[str, Any]:
        """Retorna el estado actual del controlador para observabilidad."""
        return {
            "integral_error": self._integral_error,
            "iteration_count": self._iteration_count,
            "last_time": self._last_time,
            "integral_limit": self._integral_limit
        }


# ============================================================================
# MOTOR DE FÍSICA AVANZADO
# ============================================================================
class FluxPhysicsEngine:
    """
    Simula el comportamiento físico RLC basándose en la ENERGÍA.

    Unifica Capacitancia e Inductancia bajo funciones escalares de Energía (Julios).
    - Energía Potencial (Ec): Presión acumulada por el volumen de datos.
    - Energía Cinética (El): Inercia de la calidad del flujo.
    - Energía Disipada (Er): Calor generado por la fricción de datos sucios.
    
    Mejoras implementadas:
    - Validación de parámetros físicos
    - Protección contra overflow matemático
    - Normalización de métricas
    - Diagnóstico basado en constantes nombradas
    - Manejo robusto de casos límite
    """

    def __init__(self, capacitance: float, resistance: float, inductance: float):
        """
        Inicializa el motor de física con validación de parámetros.
        
        Args:
            capacitance: Capacitancia del sistema (Faradios, debe ser > 0)
            resistance: Resistencia base (Ohmios, debe ser > 0)
            inductance: Inductancia del sistema (Henrios, debe ser > 0)
        
        Raises:
            ConfigurationError: Si algún parámetro es inválido
        """
        self._validate_parameters(capacitance, resistance, inductance)

        self.C = capacitance
        self.R = resistance
        self.L = inductance

        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        self.logger.info(
            f"Motor RLC inicializado: C={self.C}F, R={self.R}Ω, L={self.L}H"
        )

    def _validate_parameters(
        self,
        capacitance: float,
        resistance: float,
        inductance: float
    ) -> None:
        """Valida que los parámetros físicos sean válidos."""
        errors = []

        if capacitance <= 0:
            errors.append(f"capacitance debe ser > 0, recibido: {capacitance}")

        if resistance <= 0:
            errors.append(f"resistance debe ser > 0, recibido: {resistance}")

        if inductance <= 0:
            errors.append(f"inductance debe ser > 0, recibido: {inductance}")

        for param in [capacitance, resistance, inductance]:
            if math.isnan(param) or math.isinf(param):
                errors.append(f"Parámetro inválido (NaN o Inf): {param}")

        if errors:
            raise ConfigurationError(
                "Parámetros físicos inválidos:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def calculate_metrics(self, total_records: int, cache_hits: int) -> Dict[str, float]:
        """
        Calcula métricas vectoriales y escalares (energía) del flujo.
        
        Args:
            total_records: Número total de registros en el batch
            cache_hits: Número de registros con hit en caché
        
        Returns:
            Diccionario con métricas normalizadas del sistema
        """
        # Validar entradas
        if total_records < 0 or cache_hits < 0:
            self.logger.error(
                f"Parámetros negativos detectados: total_records={total_records}, "
                f"cache_hits={cache_hits}"
            )
            return self._get_zero_metrics()

        if cache_hits > total_records:
            self.logger.warning(
                f"cache_hits ({cache_hits}) > total_records ({total_records}), "
                f"normalizando a total_records"
            )
            cache_hits = total_records

        # Caso especial: sin datos
        if total_records == 0:
            return self._get_zero_metrics()

        try:
            # --- VARIABLES DE ESTADO ---
            # Corriente (I): Calidad del flujo (0.0 a 1.0)
            current_I = cache_hits / total_records
            current_I = max(0.0, min(1.0, current_I))  # Normalizar

            # Complejidad: Inversa a la corriente (fracción de datos sin caché)
            complexity = 1.0 - current_I

            # Resistencia Dinámica (R_dyn)
            dynamic_R = self.R * (1.0 + complexity * SystemConstants.COMPLEXITY_RESISTANCE_FACTOR)

            # Saturación (V): Ecuación de carga del condensador
            # V(t) = V_max * (1 - e^(-t/τ)) donde τ = R*C
            tau_c = dynamic_R * self.C

            # Prevenir overflow en exponencial
            exponent = -float(total_records) / tau_c if tau_c > 0 else -SystemConstants.MAX_EXPONENTIAL_ARG
            exponent = max(-SystemConstants.MAX_EXPONENTIAL_ARG, exponent)

            saturation_V = 1.0 - math.exp(exponent)
            saturation_V = max(0.0, min(1.0, saturation_V))  # Normalizar

            # --- CÁLCULOS DE ENERGÍA (ESCALARES) ---

            # 1. Energía Potencial (Ec = 1/2 * C * V^2)
            potential_energy = 0.5 * self.C * (saturation_V ** 2)

            # 2. Energía Cinética/Magnética (El = 1/2 * L * I^2)
            kinetic_energy = 0.5 * self.L * (current_I ** 2)

            # 3. Potencia Disipada (P = I_ruido^2 * R)
            noise_current = 1.0 - current_I
            dissipated_power = (noise_current ** 2) * dynamic_R

            # --- CÁLCULO DE FLYBACK (Tensión Inductiva) ---
            # V_L = L * (di/dt) -> Cambio en la calidad
            delta_i = 1.0 - current_I
            dt = math.log1p(total_records)  # log(1 + x) es más estable que log(x)

            flyback_voltage = (self.L * delta_i / dt) if dt > 0 else 0.0
            flyback_voltage = max(0.0, min(10.0, flyback_voltage))  # Limitar a rango razonable

            # Validar que no haya NaN o Inf en resultados
            metrics = {
                "saturation": saturation_V,
                "complexity": complexity,
                "flyback_voltage": flyback_voltage,
                "potential_energy": potential_energy,
                "kinetic_energy": kinetic_energy,
                "dissipated_power": dissipated_power
            }

            # Sanitizar métricas
            for key, value in metrics.items():
                if math.isnan(value) or math.isinf(value):
                    self.logger.warning(f"Métrica {key} inválida: {value}, reemplazando con 0.0")
                    metrics[key] = 0.0

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculando métricas físicas: {e}", exc_info=True)
            return self._get_zero_metrics()

    def _get_zero_metrics(self) -> Dict[str, float]:
        """Retorna un diccionario de métricas con valores en cero."""
        return {
            "saturation": 0.0,
            "complexity": 0.0,
            "flyback_voltage": 0.0,
            "potential_energy": 0.0,
            "kinetic_energy": 0.0,
            "dissipated_power": 0.0
        }

    def get_system_diagnosis(self, metrics: Dict[str, float]) -> str:
        """
        Genera diagnóstico del sistema basado en balance energético.
        
        Args:
            metrics: Diccionario de métricas del sistema
        
        Returns:
            Cadena con el diagnóstico del estado del sistema
        """
        try:
            ec = metrics.get("potential_energy", 0.0)
            el = metrics.get("kinetic_energy", 0.0)
            flyback = metrics.get("flyback_voltage", 0.0)

            # Prevenir división por cero
            if el < SystemConstants.MIN_ENERGY_THRESHOLD:
                return "🔴 SISTEMA ESTANCADO (Inercia crítica baja)"

            energy_ratio = ec / el

            # Diagnóstico jerárquico
            if energy_ratio > SystemConstants.HIGH_PRESSURE_RATIO:
                return "🟠 SOBRECARGA DE PRESIÓN (Riesgo de ruptura)"
            elif flyback > SystemConstants.HIGH_FLYBACK_THRESHOLD:
                return "⚡ PICO INDUCTIVO DETECTADO (Inestabilidad)"
            elif el < SystemConstants.LOW_INERTIA_THRESHOLD:
                return "🟡 BAJA INERCIA (Flujo débil)"
            else:
                return "🟢 EQUILIBRIO ENERGÉTICO (Estable)"

        except Exception as e:
            self.logger.error(f"Error en diagnóstico del sistema: {e}")
            return "❓ DIAGNÓSTICO INDETERMINADO"


# ============================================================================
# DATA FLUX CONDENSER
# ============================================================================
class DataFluxCondenser:
    """
    Orquesta el pipeline de validación y procesamiento de archivos de APU.

    Implementa una arquitectura de "Caja de Cristal" con control adaptativo PID.
    El sistema monitorea la "física" del procesamiento en tiempo real y ajusta
    la velocidad de ingestión (batch size) para mantener la estabilidad.
    
    Mejoras implementadas:
    - Validación exhaustiva de configuración
    - Protección contra loops infinitos
    - Recuperación parcial opcional ante fallos
    - Telemetría detallada con estadísticas
    - Manejo robusto de casos límite
    - Logging estructurado
    """

    REQUIRED_CONFIG_KEYS: Set[str] = {'parser_settings', 'processor_settings'}
    REQUIRED_PROFILE_KEYS: Set[str] = {'columns_mapping', 'validation_rules'}

    def __init__(
        self,
        config: Dict[str, Any],
        profile: Dict[str, Any],
        condenser_config: Optional[CondenserConfig] = None
    ):
        """
        Inicializa el Condensador con Motor RLC y Controlador PID.
        
        Args:
            config: Configuración del sistema (debe contener parser_settings, processor_settings)
            profile: Perfil de procesamiento (debe contener columns_mapping, validation_rules)
            condenser_config: Configuración específica del condensador (opcional)
        
        Raises:
            InvalidInputError: Si config o profile son inválidos
            ConfigurationError: Si condenser_config es inválido
        """
        self._validate_initialization_params(config, profile)

        self.config = config
        self.profile = profile

        # Inicializar configuración (esto validará los parámetros internamente)
        try:
            self.condenser_config = condenser_config or CondenserConfig()
        except ConfigurationError as e:
            raise ConfigurationError(f"Error en configuración del condensador: {e}") from e

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.condenser_config.log_level)

        # Inicializar Motor de Física RLC
        try:
            self.physics = FluxPhysicsEngine(
                capacitance=self.condenser_config.system_capacitance,
                resistance=self.condenser_config.base_resistance,
                inductance=self.condenser_config.system_inductance
            )
        except ConfigurationError as e:
            raise ConfigurationError(f"Error inicializando motor físico: {e}") from e

        # Inicializar Controlador PI
        try:
            self.controller = PIController(
                kp=self.condenser_config.pid_kp,
                ki=self.condenser_config.pid_ki,
                setpoint=self.condenser_config.pid_setpoint,
                min_output=self.condenser_config.min_batch_size,
                max_output=self.condenser_config.max_batch_size,
                integral_limit_factor=self.condenser_config.integral_limit_factor
            )
        except ConfigurationError as e:
            raise ConfigurationError(f"Error inicializando controlador PID: {e}") from e

        # Estadísticas de procesamiento
        self._stats = ProcessingStats()

        self.logger.info(
            f"DataFluxCondenser inicializado | "
            f"PID: Kp={self.condenser_config.pid_kp}, Ki={self.condenser_config.pid_ki} | "
            f"Batch: [{self.condenser_config.min_batch_size}, {self.condenser_config.max_batch_size}]"
        )

    def _validate_initialization_params(
        self,
        config: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> None:
        """
        Valida que config y profile sean diccionarios con las claves requeridas.
        
        Raises:
            InvalidInputError: Si la validación falla
        """
        if not isinstance(config, dict):
            raise InvalidInputError(
                f"config debe ser un diccionario, recibido: {type(config).__name__}"
            )

        if not isinstance(profile, dict):
            raise InvalidInputError(
                f"profile debe ser un diccionario, recibido: {type(profile).__name__}"
            )

        # Validar claves requeridas (modo warning, no error)
        missing_config_keys = self.REQUIRED_CONFIG_KEYS - set(config.keys())
        if missing_config_keys:
            self.logger.warning(
                f"Claves faltantes en config (modo tolerante): {missing_config_keys}"
            )

        missing_profile_keys = self.REQUIRED_PROFILE_KEYS - set(profile.keys())
        if missing_profile_keys:
            self.logger.warning(
                f"Claves faltantes en profile (modo tolerante): {missing_profile_keys}"
            )

    def stabilize(self, file_path: str) -> pd.DataFrame:
        """
        Proceso de Carga y Descarga CONTROLADO por PID.
        Procesa el archivo en flujo continuo (Streaming por Lotes Adaptativo).

        El sistema lee el archivo y lo divide en lotes cuyo tamaño es ajustado
        dinámicamente por el controlador PID basándose en la 'saturación' detectada
        en el lote anterior.
        
        Args:
            file_path: Ruta al archivo de APU a procesar
        
        Returns:
            DataFrame con los datos procesados
        
        Raises:
            InvalidInputError: Si el archivo es inválido
            ProcessingError: Si ocurre un error durante el procesamiento
        """
        start_time = time.time()
        path_obj = Path(file_path)

        # Resetear estadísticas y controlador
        self._stats = ProcessingStats()
        self.controller.reset()

        self.logger.info(
            f"⚡ [CONTROL ADAPTATIVO] Iniciando lazo de control para: {path_obj.name}"
        )

        try:
            validated_path = self._validate_input_file(file_path)

            # Fase 1: Inicializar el Parser (Guardia)
            parser = self._initialize_parser(validated_path)

            # Fase 2: Extract - Leer contenido crudo
            full_raw_records, full_cache = self._extract_raw_data(parser)

            if not full_raw_records:
                self.logger.warning("El archivo no contiene registros crudos válidos.")
                return pd.DataFrame()

            total_records = len(full_raw_records)
            self._stats.total_records = total_records

            # Fase 3: Validar umbral mínimo
            if total_records < self.condenser_config.min_records_threshold:
                self.logger.warning(
                    f"[VALIDACIÓN] Registros insuficientes: {total_records} < "
                    f"{self.condenser_config.min_records_threshold}"
                )
                return pd.DataFrame()

            # Fase 4: Procesamiento por lotes con control PID
            processed_batches = self._process_batches_with_pid(
                full_raw_records,
                full_cache,
                total_records
            )

            # Fase 5: Consolidar resultados
            df_final = self._consolidate_results(processed_batches)

            # Fase 6: Validar salida
            self._validate_output(df_final)

            # Registrar estadísticas finales
            self._stats.processing_time = time.time() - start_time
            self._log_final_stats()

            self.logger.info(
                f"✅ [ESTABILIZADO] Proceso completado en {self._stats.processing_time:.2f}s. "
                f"Registros procesados: {self._stats.processed_records}/{total_records}"
            )

            return df_final

        except (InvalidInputError, ProcessingError):
            # Re-lanzar errores conocidos
            raise
        except Exception as e:
            self.logger.exception(f"[ERROR CRÍTICO] Error inesperado: {e}")
            raise ProcessingError(f"Error inesperado durante estabilización: {e}") from e

    def _initialize_parser(self, validated_path: Path) -> ReportParserCrudo:
        """
        Inicializa el parser con manejo robusto de errores.
        
        Args:
            validated_path: Ruta validada al archivo
        
        Returns:
            Instancia de ReportParserCrudo configurada
        
        Raises:
            ProcessingError: Si falla la inicialización
        """
        try:
            parser = ReportParserCrudo(
                str(validated_path),
                profile=self.profile,
                config=self.config
            )
            self.logger.debug(f"Parser inicializado para: {validated_path.name}")
            return parser
        except Exception as e:
            raise ProcessingError(
                f"Error inicializando ReportParserCrudo: {e}"
            ) from e

    def _extract_raw_data(
        self,
        parser: ReportParserCrudo
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extrae datos crudos del parser con validación.
        
        Args:
            parser: Instancia del parser configurado
        
        Returns:
            Tupla (registros_crudos, cache)
        
        Raises:
            ProcessingError: Si falla la extracción
        """
        try:
            full_raw_records = parser.parse_to_raw()

            if not isinstance(full_raw_records, list):
                raise ProcessingError(
                    f"parse_to_raw() debe retornar lista, recibido: {type(full_raw_records).__name__}"
                )

            full_cache = parser.get_parse_cache() or {}

            if not isinstance(full_cache, dict):
                self.logger.warning(
                    f"get_parse_cache() retornó tipo inesperado: {type(full_cache).__name__}, "
                    f"usando dict vacío"
                )
                full_cache = {}

            self.logger.info(
                f"[EXTRACT] Extraídos {len(full_raw_records)} registros | "
                f"Cache: {len(full_cache)} entradas"
            )

            return full_raw_records, full_cache

        except Exception as e:
            raise ProcessingError(f"Error extrayendo datos crudos: {e}") from e

    def _process_batches_with_pid(
        self,
        full_raw_records: List[Dict[str, Any]],
        full_cache: Dict[str, Any],
        total_records: int
    ) -> List[pd.DataFrame]:
        """
        Procesa registros en lotes con control PID adaptativo.
        
        Args:
            full_raw_records: Lista completa de registros crudos
            full_cache: Cache de parseo completo
            total_records: Número total de registros
        
        Returns:
            Lista de DataFrames procesados (uno por batch)
        
        Raises:
            ProcessingError: Si se excede el límite de batches fallidos
        """
        processed_batches = []
        current_index = 0
        current_batch_size = self.condenser_config.min_batch_size
        failed_batches_count = 0

        # Protección contra loops infinitos
        max_iterations = total_records * SystemConstants.MAX_ITERATIONS_MULTIPLIER
        iteration_count = 0

        self.logger.info(
            f"[PID LOOP] Iniciando procesamiento | Total: {total_records} registros | "
            f"Batch inicial: {current_batch_size}"
        )

        while current_index < total_records:
            iteration_count += 1

            # Protección contra loop infinito
            if iteration_count > max_iterations:
                raise ProcessingError(
                    f"Excedido límite de iteraciones ({max_iterations}). "
                    f"Posible loop infinito detectado. "
                    f"Índice actual: {current_index}/{total_records}"
                )

            # Validar que batch_size sea válido
            if current_batch_size <= 0:
                self.logger.error(
                    f"Batch size inválido detectado: {current_batch_size}. "
                    f"Reseteando a mínimo: {self.condenser_config.min_batch_size}"
                )
                current_batch_size = self.condenser_config.min_batch_size

            # 1. Extraer lote actual
            end_index = min(current_index + current_batch_size, total_records)
            batch_records = full_raw_records[current_index:end_index]

            if not batch_records:
                self.logger.warning(
                    f"Batch vacío detectado en índice {current_index}, avanzando..."
                )
                current_index = end_index
                continue

            # 2. Calcular cache hits para el lote
            batch_cache_hits = self._calculate_cache_hits(batch_records, full_cache)

            # 3. Medir estado del sistema (Sensores)
            metrics = self.physics.calculate_metrics(len(batch_records), batch_cache_hits)

            # 4. Acción de control PID
            new_batch_size = self.controller.compute(metrics["saturation"])

            # 5. Protección basada en Energía Disipada (Diodo de Rueda Libre)
            if metrics["dissipated_power"] > SystemConstants.OVERHEAT_POWER_THRESHOLD:
                self.logger.warning(
                    f"🔥 [SOBRECALENTAMIENTO] Disipación alta "
                    f"({metrics['dissipated_power']:.1f}W). Aplicando freno de emergencia."
                )
                new_batch_size = int(new_batch_size * SystemConstants.EMERGENCY_BRAKE_FACTOR)
                new_batch_size = max(self.condenser_config.min_batch_size, new_batch_size)
                self._stats.emergency_brakes_triggered += 1

            # 6. Telemetría
            diagnosis = self.physics.get_system_diagnosis(metrics)
            self.logger.info(
                f"🔄 [PID] Batch #{self._stats.total_batches + 1} | "
                f"Size: {len(batch_records)} | "
                f"Sat: {metrics['saturation']:.2%} | "
                f"Ec: {metrics['potential_energy']:.0f}J | "
                f"El: {metrics['kinetic_energy']:.2f}J | "
                f"Disip: {metrics['dissipated_power']:.1f}W | "
                f"→ Next: {new_batch_size} | {diagnosis}"
            )

            # Advertencia de flyback
            if metrics["flyback_voltage"] > SystemConstants.HIGH_FLYBACK_THRESHOLD:
                self.logger.warning(
                    f"🛡️ [DIODO FLYBACK] Pico de inestabilidad detectado "
                    f"(V_L={metrics['flyback_voltage']:.2f}V) en batch {current_index}-{end_index}"
                )

            # 7. Procesar el lote
            batch_data = ParsedData(batch_records, full_cache)
            batch_success = False

            try:
                df_batch = self._rectify_signal(batch_data)
                processed_batches.append(df_batch)
                batch_success = True

            except ProcessingError as e:
                failed_batches_count += 1
                self.logger.error(
                    f"[ERROR] Batch {current_index}-{end_index} falló: {e} | "
                    f"Fallos acumulados: {failed_batches_count}/{self.condenser_config.max_failed_batches}"
                )

                # Decidir si abortar o continuar
                if self.condenser_config.enable_partial_recovery:
                    if failed_batches_count > self.condenser_config.max_failed_batches:
                        raise ProcessingError(
                            f"Excedido límite de batches fallidos "
                            f"({self.condenser_config.max_failed_batches}). Abortando."
                        ) from e
                    else:
                        self.logger.warning(
                            "[RECOVERY] Continuando con siguiente batch (modo recuperación parcial)"
                        )
                else:
                    # Modo estricto: un fallo aborta todo
                    raise

            # 8. Actualizar estadísticas
            self._stats.add_batch_stats(
                batch_size=len(batch_records),
                saturation=metrics["saturation"],
                power=metrics["dissipated_power"],
                flyback=metrics["flyback_voltage"],
                kinetic=metrics["kinetic_energy"],
                success=batch_success
            )

            # 9. Avanzar al siguiente batch
            current_index = end_index
            current_batch_size = new_batch_size

        self.logger.info(
            f"[PID LOOP] Completado | Batches procesados: {self._stats.total_batches} | "
            f"Batches fallidos: {self._stats.failed_batches}"
        )

        return processed_batches

    def _calculate_cache_hits(
        self,
        batch_records: List[Dict[str, Any]],
        full_cache: Dict[str, Any]
    ) -> int:
        """
        Calcula el número de cache hits para un batch de registros.
        
        Args:
            batch_records: Lista de registros del batch
            full_cache: Diccionario de cache completo
        
        Returns:
            Número de registros con hit en caché
        """
        if not full_cache:
            return 0

        cache_hits = 0
        for record in batch_records:
            # Intentar varias claves posibles para linkear con cache
            for key in ['insumo_line', 'line', 'raw_line', '_line']:
                line_content = record.get(key)
                if line_content and line_content in full_cache:
                    cache_hits += 1
                    break

        return cache_hits

    def _consolidate_results(self, processed_batches: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Consolida múltiples DataFrames de batches en uno solo.
        
        Args:
            processed_batches: Lista de DataFrames procesados
        
        Returns:
            DataFrame consolidado
        """
        if not processed_batches:
            self.logger.warning("[CONSOLIDATE] No hay batches para consolidar")
            return pd.DataFrame()

        try:
            # Filtrar batches vacíos
            non_empty_batches = [df for df in processed_batches if not df.empty]

            if not non_empty_batches:
                self.logger.warning("[CONSOLIDATE] Todos los batches están vacíos")
                return pd.DataFrame()

            df_final = pd.concat(non_empty_batches, ignore_index=True)

            self.logger.info(
                f"[CONSOLIDATE] Consolidados {len(non_empty_batches)} batches → "
                f"{len(df_final)} registros finales"
            )

            return df_final

        except Exception as e:
            raise ProcessingError(f"Error consolidando resultados: {e}") from e

    def _validate_input_file(self, file_path: str) -> Path:
        """
        Valida que el archivo de entrada exista y sea accesible.
        
        Args:
            file_path: Ruta al archivo
        
        Returns:
            Objeto Path validado
        
        Raises:
            InvalidInputError: Si el archivo es inválido
        """
        if not file_path or not isinstance(file_path, str):
            raise InvalidInputError(
                f"file_path debe ser una cadena no vacía, recibido: {type(file_path).__name__}"
            )

        path = Path(file_path)

        if not path.exists():
            raise InvalidInputError(f"El archivo no existe: {file_path}")

        if not path.is_file():
            raise InvalidInputError(f"La ruta no es un archivo válido: {file_path}")

        if path.suffix.lower() not in SystemConstants.VALID_FILE_EXTENSIONS:
            self.logger.warning(
                f"Extensión inusual detectada: {path.suffix}. "
                f"Se esperaba una de: {SystemConstants.VALID_FILE_EXTENSIONS}"
            )

        self.logger.debug(f"[VALIDACIÓN] Archivo validado: {path}")
        return path

    def _rectify_signal(self, parsed_data: ParsedData) -> pd.DataFrame:
        """
        Usa APUProcessor para convertir la señal filtrada en datos utilizables.
        
        Args:
            parsed_data: Datos parseados con cache
        
        Returns:
            DataFrame procesado
        
        Raises:
            ProcessingError: Si falla el procesamiento
        """
        try:
            # 1. Instanciar APUProcessor
            processor = APUProcessor(
                config=self.config,
                profile=self.profile,
                parse_cache=parsed_data.parse_cache
            )

            # 2. Pasar raw_records directamente
            processor.raw_records = parsed_data.raw_records

            # 3. Procesar
            df_result = processor.process_all()

            if not isinstance(df_result, pd.DataFrame):
                raise ProcessingError(
                    f"APUProcessor.process_all() debe retornar DataFrame, "
                    f"recibido: {type(df_result).__name__}"
                )

            return df_result

        except Exception as e:
            raise ProcessingError(
                f"Error durante la rectificación con APUProcessor: {e}"
            ) from e

    def _validate_output(self, df: pd.DataFrame) -> None:
        """
        Valida el DataFrame de salida antes de retornarlo.
        
        Args:
            df: DataFrame a validar
        
        Raises:
            ProcessingError: Si la validación falla críticamente
        """
        if not isinstance(df, pd.DataFrame):
            raise ProcessingError(
                f"La salida debe ser DataFrame, recibido: {type(df).__name__}"
            )

        if self.condenser_config.enable_strict_validation:
            if df.empty:
                self.logger.warning(
                    "[VALIDACIÓN] DataFrame vacío generado "
                    "(puede ser válido dependiendo del archivo)"
                )

            # Detectar columnas completamente nulas
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                self.logger.warning(
                    f"[VALIDACIÓN] Columnas completamente nulas detectadas: {null_columns}"
                )

            # Detectar columnas con alta proporción de nulos
            if not df.empty:
                high_null_cols = []
                for col in df.columns:
                    null_ratio = df[col].isnull().sum() / len(df)
                    if null_ratio > 0.9:  # >90% nulos
                        high_null_cols.append((col, f"{null_ratio:.1%}"))

                if high_null_cols:
                    self.logger.warning(
                        f"[VALIDACIÓN] Columnas con alta proporción de nulos: {high_null_cols}"
                    )

    def _log_final_stats(self) -> None:
        """Registra estadísticas finales del procesamiento."""
        self.logger.info(
            f"\n{'='*70}\n"
            f"📊 ESTADÍSTICAS FINALES\n"
            f"{'='*70}\n"
            f"  Registros totales:       {self._stats.total_records}\n"
            f"  Registros procesados:    {self._stats.processed_records}\n"
            f"  Registros fallidos:      {self._stats.failed_records}\n"
            f"  Batches totales:         {self._stats.total_batches}\n"
            f"  Batches fallidos:        {self._stats.failed_batches}\n"
            f"  Tiempo de proceso:       {self._stats.processing_time:.2f}s\n"
            f"  Tamaño promedio batch:   {self._stats.avg_batch_size:.0f}\n"
            f"  Saturación promedio:     {self._stats.avg_saturation:.2%}\n"
            f"  Potencia máx. disipada:  {self._stats.max_dissipated_power:.1f}W\n"
            f"  Frenos de emergencia:    {self._stats.emergency_brakes_triggered}\n"
            f"{'='*70}"
        )

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del último procesamiento.
        
        Returns:
            Diccionario con estadísticas completas
        """
        return {
            "condenser_config": {
                "min_records_threshold": self.condenser_config.min_records_threshold,
                "strict_validation": self.condenser_config.enable_strict_validation,
                "log_level": self.condenser_config.log_level,
                "pid_mode": True,
                "partial_recovery": self.condenser_config.enable_partial_recovery
            },
            "config_keys": list(self.config.keys()),
            "profile_keys": list(self.profile.keys()),
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
                "max_flyback_voltage": self._stats.max_flyback_voltage,
                "avg_kinetic_energy": self._stats.avg_kinetic_energy,
                "emergency_brakes_triggered": self._stats.emergency_brakes_triggered
            },
            "controller_state": self.controller.get_state(),
            "success_rate": (
                self._stats.processed_records / self._stats.total_records
                if self._stats.total_records > 0 else 0.0
            )
        }

    def reset(self) -> None:
        """Resetea el estado interno del condensador para reutilización."""
        self.controller.reset()
        self._stats = ProcessingStats()
        self.logger.info("[RESET] Condensador reseteado y listo para nuevo procesamiento")
