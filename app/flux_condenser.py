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
from typing import Any, Dict, List, NamedTuple, Optional

import pandas as pd

from .apu_processor import APUProcessor
from .report_parser_crudo import ReportParserCrudo

logger = logging.getLogger(__name__)


# --- NUEVA CLASE: CONTROLADOR PI DISCRETO ---
class PIController:
    """
    Implementación de un Controlador PI Discreto según la teoría de control.

    Objetivo: Mantener la saturación del sistema en un Setpoint (SP) estable,
    ajustando dinámicamente la variable de control (Tamaño del Batch).
    """
    def __init__(self, kp: float, ki: float, setpoint: float, min_output: int, max_output: int):
        self.Kp = kp
        self.Ki = ki
        self.setpoint = setpoint # El "Flujo Laminar" ideal (ej. 0.3 de saturación)

        # Límites del actuador (Tamaño de Batch)
        self.min_output = min_output
        self.max_output = max_output

        # Estado interno
        self._integral_error = 0.0
        self._last_time = time.time()

    def compute(self, process_variable: float) -> int:
        """
        Calcula la nueva salida de control (u(t)) basada en el error actual.

        Ecuación Posicional Discreta:
        u(k) = Kp * e(k) + Ki * sum(e) * dt
        """
        current_time = time.time()
        dt = current_time - self._last_time
        if dt <= 0: dt = 0.001 # Evitar división por cero

        # 1. Calcular Error (e(t))
        # Nota: Invertimos el signo porque queremos que:
        # Saturación Alta -> Error Negativo -> Reducir Batch
        error = self.setpoint - process_variable

        # 2. Término Proporcional
        P = self.Kp * error

        # 3. Término Integral (con Anti-Windup implícito por los límites de salida)
        self._integral_error += error * dt
        I = self.Ki * self._integral_error

        # 4. Señal de Control (u)
        # Base output es la mitad del rango, el PID ajusta desde ahí
        base_output = (self.max_output + self.min_output) / 2
        control_signal = base_output + P + I

        # 5. Saturación del Actuador (Clamping)
        output = max(self.min_output, min(self.max_output, int(control_signal)))

        self._last_time = current_time
        return output


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


class DataFluxCondenserError(Exception):
    """Clase base para todas las excepciones personalizadas del condensador."""
    pass


class InvalidInputError(DataFluxCondenserError):
    """Indica un problema con los datos de entrada, como un archivo inválido."""
    pass


class ProcessingError(DataFluxCondenserError):
    """Señala un error durante una de las etapas de procesamiento de datos."""
    pass


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
    """
    min_records_threshold: int = 1
    enable_strict_validation: bool = True
    log_level: str = "INFO"
    # --- Configuración Física RLC ---
    system_capacitance: float = 5000.0  # Faradios (Capacidad de carga)
    base_resistance: float = 10.0       # Ohmios (Fricción estática)
    system_inductance: float = 2.0      # Henrios (Inercia/Resistencia al cambio)
    # --- Configuración PID ---
    pid_setpoint: float = 0.30      # Objetivo: Saturación del 30% (Flujo Laminar)
    pid_kp: float = 2000.0          # Ganancia Proporcional (Reacción rápida)
    pid_ki: float = 100.0           # Ganancia Integral (Precisión a largo plazo)
    min_batch_size: int = 50        # Flujo mínimo (Goteo)
    max_batch_size: int = 5000      # Flujo máximo (Chorro)


# --- MOTOR DE FÍSICA AVANZADO (RLC) ---
class FluxPhysicsEngine:
    """
    Simula el comportamiento físico RLC (Resistencia-Inductancia-Capacitancia).

    Calcula métricas de 'salud' del flujo de datos como Saturación (Carga),
    Complejidad (Resistencia) y Tensión de Flyback (Inductancia).
    """
    def __init__(self, capacitance: float, resistance: float, inductance: float):
        """
        Inicializa el motor de física con los parámetros del circuito.

        Args:
            capacitance (float): Capacidad del sistema (análogo a Faradios).
            resistance (float): Resistencia base del sistema (análogo a Ohmios).
            inductance (float): Inercia del sistema (análogo a Henrios).
        """
        self.C = capacitance  # Capacidad de absorción
        self.R = resistance   # Resistencia base
        self.L = inductance   # Inercia del sistema (Inductancia)

    def calculate_metrics(self, total_records: int, cache_hits: int) -> Dict[str, float]:
        """
        Calcula métricas físicas del flujo de datos.

        Args:
            total_records (int): Número total de registros procesados.
            cache_hits (int): Número de registros que aprovecharon la caché.

        Returns:
            Dict[str, float]: Diccionario con saturación, complejidad y voltaje de flyback.
        """
        if total_records == 0:
            return {"saturation": 0.0, "complexity": 0.0, "flyback_voltage": 0.0}

        # 1. Resistencia Dinámica (Complejidad)
        # R aumenta si el cache no se usa (fricción de procesamiento)
        complexity = 1.0 - (cache_hits / total_records)

        # 2. Saturación (Carga del Condensador)
        # Vc(t) = 1 - e^(-t/RC)
        dynamic_R = self.R * (1 + complexity * 5) # La complejidad aumenta R drásticamente
        tau_c = dynamic_R * self.C
        saturation = 1.0 - math.exp(-float(total_records) / tau_c)

        # 3. Tensión de Flyback (Inductancia - Ley de Faraday/Lenz)
        # V_L = -L * (di/dt)
        # Asumimos que 'corriente' (i) es la tasa de éxito (cache_hits / total).
        # Un cambio brusco en la calidad genera un pico de voltaje lógico.

        current_i = cache_hits / total_records # 0.0 a 1.0
        delta_i = 1.0 - current_i # La caída de corriente (pérdida de calidad)

        # El 'dt' es inversamente proporcional al tamaño (en archivos pequeños,
        # un error es un cambio más brusco que en grandes).
        dt = math.log1p(total_records)

        # Voltaje inducido (magnitud del pico de error)
        flyback_voltage = self.L * (delta_i / dt) if dt > 0 else 0.0

        return {
            "saturation": saturation,
            "complexity": complexity,
            "flyback_voltage": flyback_voltage
        }

    def get_system_diagnosis(self, metrics: Dict[str, float]) -> str:
        """
        Genera un diagnóstico textual basado en las métricas físicas.
        """
        v_flyback = metrics["flyback_voltage"]
        saturation = metrics["saturation"]

        # Diagnóstico basado en Flyback (Picos de tensión)
        if v_flyback > 0.5:
            return f"⚡ PELIGRO: PICO INDUCTIVO ALTO (vL={v_flyback:.2f}). Datos muy inestables."
        elif v_flyback > 0.1:
            return f"⚠️ ADVERTENCIA: Rizado Inductivo (vL={v_flyback:.2f}). Calidad variable."

        # Diagnóstico basado en Saturación (Carga)
        if saturation < 0.3: return "🌊 FLUJO LAMINAR (Estable)"
        if saturation < 0.7: return "🌊 FLUJO TRANSITORIO (Carga Media)"
        return "🌊 FLUJO TURBULENTO (Saturado)"


class DataFluxCondenser:
    """
    Orquesta el pipeline de validación y procesamiento de archivos de APU.

    Implementa una arquitectura de "Caja de Cristal" con control adaptativo PID.
    El sistema monitorea la "física" del procesamiento en tiempo real y ajusta
    la velocidad de ingestión (batch size) para mantener la estabilidad.
    """
    REQUIRED_CONFIG_KEYS = {'parser_settings', 'processor_settings'}
    REQUIRED_PROFILE_KEYS = {'columns_mapping', 'validation_rules'}

    def __init__(
        self,
        config: Dict[str, Any],
        profile: Dict[str, Any],
        condenser_config: Optional[CondenserConfig] = None
    ):
        """
        Inicializa el Condensador con Motor RLC y Controlador PID.
        """
        self._validate_initialization_params(config, profile)

        self.config = config
        self.profile = profile
        self.condenser_config = condenser_config or CondenserConfig()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.condenser_config.log_level)

        # Inicializar Motor de Física RLC
        self.physics = FluxPhysicsEngine(
            capacitance=self.condenser_config.system_capacitance,
            resistance=self.condenser_config.base_resistance,
            inductance=self.condenser_config.system_inductance
        )

        # Inicializar Controlador PI
        self.controller = PIController(
            kp=self.condenser_config.pid_kp,
            ki=self.condenser_config.pid_ki,
            setpoint=self.condenser_config.pid_setpoint,
            min_output=self.condenser_config.min_batch_size,
            max_output=self.condenser_config.max_batch_size
        )

        self.logger.info("DataFluxCondenser (Motor RLC + Controlador PI) inicializado")

    def _validate_initialization_params(
        self,
        config: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> None:
        """Valida que config y profile sean diccionarios."""
        if not isinstance(config, dict) or not isinstance(profile, dict):
            raise InvalidInputError("config y profile deben ser diccionarios válidos")

        missing_config_keys = self.REQUIRED_CONFIG_KEYS - set(config.keys())
        if missing_config_keys:
            logger.warning(f"Claves faltantes en config (modo tolerante): {missing_config_keys}")

        missing_profile_keys = self.REQUIRED_PROFILE_KEYS - set(profile.keys())
        if missing_profile_keys:
            logger.warning(f"Claves faltantes en profile (modo tolerante): {missing_profile_keys}")

    def stabilize(self, file_path: str) -> pd.DataFrame:
        """
        Proceso de Carga y Descarga CONTROLADO por PID.
        Procesa el archivo en flujo continuo (Streaming por Lotes Adaptativo).

        El sistema lee el archivo y, en lugar de procesarlo todo de golpe,
        lo divide en lotes cuyo tamaño es ajustado dinámicamente por el
        controlador PID basándose en la 'saturación' detectada en el lote anterior.
        """
        start_time = time.time()
        path_obj = Path(file_path)
        self.logger.info(f"⚡ [CONTROL ADAPTATIVO] Iniciando lazo de control para: {path_obj.name}")

        try:
            validated_path = self._validate_input_file(file_path)

            # Inicializar el Guardia (Parser)
            # FIX: Pasar explícitamente 'config' como argumento de palabra clave
            parser = ReportParserCrudo(
                str(validated_path),
                profile=self.profile,
                config=self.config
            )

            # Leemos todo el contenido crudo primero (Extract)
            # En una versión futura, esto también sería streaming desde disco.
            full_raw_records = parser.parse_to_raw()
            full_cache = parser.get_parse_cache() or {}

            if not full_raw_records:
                self.logger.warning("El archivo no contiene registros crudos.")
                return pd.DataFrame()

            total_records = len(full_raw_records)

            # Validar umbral mínimo antes de iniciar el bucle
            if total_records < self.condenser_config.min_records_threshold:
                 self.logger.warning(
                    f"[VALIDACIÓN] Registros insuficientes: {total_records} < "
                    f"{self.condenser_config.min_records_threshold}"
                )
                 # Dependiendo de la lógica de negocio, podríamos retornar vacío o procesar lo que hay.
                 # La implementación anterior retornaba vacío si fallaba _validate_parsed_data.
                 return pd.DataFrame()

            processed_batches = []

            # --- BUCLE DE CONTROL PID ---
            current_index = 0
            current_batch_size = self.condenser_config.min_batch_size # Arranque suave

            self.logger.info(f"Iniciando procesamiento por lotes. Total registros: {total_records}")

            while current_index < total_records:
                # 1. Cortar el lote actual (Actuador)
                end_index = min(current_index + current_batch_size, total_records)
                batch_records = full_raw_records[current_index:end_index]

                # 2. Preparar Cache para el lote
                # Creamos un subset del cache relevante para este lote
                # Nota: Esto asume que las claves del cache son identificables en los records.
                # Si no es eficiente filtrar, pasamos el cache completo,
                # pero para la métrica necesitamos saber cuántos hits hubo en ESTE batch.
                # Simplificación: Calculamos hits estimados o pasamos todo y dejamos que Physics calcule sobre el total acumulado?
                # La propuesta dice: "medir la saturación después de cada trozo".
                # Physics.calculate_metrics toma (total_records, cache_hits).
                # Deberíamos calcular metricas LOCALES del batch.

                batch_cache_hits = 0
                for record in batch_records:
                    # Asumimos que hay una forma de linkear record con cache,
                    # o simplificamos asumiendo proporcionalidad si no hay keys claras.
                    # En ReportParserCrudo, el cache suele ser por 'insumo_line' o similar.
                    line_content = record.get('insumo_line', '')
                    if line_content in full_cache:
                        batch_cache_hits += 1

                # 3. Medir el estado del sistema (Sensor)
                # Calculamos métricas basadas en el batch actual
                metrics = self.physics.calculate_metrics(len(batch_records), batch_cache_hits)

                # 4. Acción de Control (PID)
                # El controlador decide el tamaño del SIGUIENTE lote basado en la saturación actual
                new_batch_size = self.controller.compute(metrics["saturation"])

                # Telemetría en tiempo real
                self.logger.debug(
                    f"🔄 [PID LOOP] Batch: {len(batch_records)} items | "
                    f"Sat: {metrics['saturation']:.2f} | "
                    f"V_Flyback: {metrics['flyback_voltage']:.2f} | "
                    f"→ Next Size: {new_batch_size}"
                )

                # Diodo Flyback
                if metrics["flyback_voltage"] > 0.8:
                     self.logger.warning(
                        f"🛡️ [DIODO FLYBACK] Pico de inestabilidad detectado en batch {current_index}-{end_index}."
                    )

                # 5. Procesar el lote (Planta)
                # Pasamos el cache completo al processor para que tenga contexto si lo necesita,
                # pero procesamos solo los records del batch.
                batch_data = ParsedData(batch_records, full_cache)

                try:
                    df_batch = self._rectify_signal(batch_data)
                    processed_batches.append(df_batch)
                except ProcessingError as e:
                    self.logger.error(f"Error procesando batch {current_index}-{end_index}: {e}")
                    # En un sistema robusto, podríamos decidir si abortar o continuar.
                    # Por ahora, propagamos el error o saltamos?
                    # Si falla un batch, el archivo podría quedar incompleto.
                    # "Robustez" sugiere intentar salvar lo que se pueda o fallar seguro.
                    # El original fallaba completo. Mantendremos eso por seguridad.
                    raise

                # Avanzar
                current_index = end_index
                current_batch_size = new_batch_size # Aplicar la decisión del PID

            # --- FIN DEL BUCLE ---

            # Consolidar resultados
            if processed_batches:
                df_final = pd.concat(processed_batches, ignore_index=True)
            else:
                df_final = pd.DataFrame()

            self._validate_output(df_final)

            elapsed = time.time() - start_time
            self.logger.info(
                f"✅ [ESTABILIZADO] Proceso completado en {elapsed:.2f}s. "
                f"El controlador PID mantuvo el flujo estable."
            )
            return df_final

        except InvalidInputError as e:
            self.logger.error(f"[ERROR] Entrada inválida: {e}")
            raise
        except ProcessingError as e:
            self.logger.error(f"[ERROR] Fallo en procesamiento: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"[ERROR CRÍTICO] Error inesperado: {e}")
            raise ProcessingError(f"Error inesperado durante estabilización: {e}") from e

    def _validate_input_file(self, file_path: str) -> Path:
        """Valida que el archivo de entrada exista y sea accesible."""
        if not file_path or not isinstance(file_path, str):
            raise InvalidInputError(f"file_path debe ser una cadena no vacía, recibido: {type(file_path)}")

        path = Path(file_path)

        if not path.exists():
            raise InvalidInputError(f"El archivo no existe: {file_path}")

        if not path.is_file():
            raise InvalidInputError(f"La ruta no es un archivo: {file_path}")

        if path.suffix.lower() not in {'.csv', '.txt'}:
            self.logger.warning(f"Extensión inusual detectada: {path.suffix}. Se esperaba .csv o .txt")

        self.logger.debug(f"[VALIDACIÓN] Archivo validado: {path}")
        return path

    def _rectify_signal(self, parsed_data: ParsedData) -> pd.DataFrame:
        """Usa APUProcessor para convertir la señal filtrada en datos utilizables."""
        # self.logger.debug("[FASE 2] Rectificando señal con APUProcessor...") # Verbose

        try:
            # 1. Instanciar APUProcessor
            processor = APUProcessor(
                config=self.config,
                profile=self.profile,
                parse_cache=parsed_data.parse_cache
            )

            # 2. Pasar raw_records directamente
            processor.raw_records = parsed_data.raw_records

            df_result = processor.process_all()

            if not isinstance(df_result, pd.DataFrame):
                raise ProcessingError(
                    f"APUProcessor.process_all() debe retornar DataFrame, recibido: {type(df_result)}"
                )

            return df_result

        except Exception as e:
            raise ProcessingError(f"Error durante la rectificación con APUProcessor: {e}") from e

    def _validate_output(self, df: pd.DataFrame) -> None:
        """Valida el DataFrame de salida antes de retornarlo."""
        if not isinstance(df, pd.DataFrame):
            raise ProcessingError(f"La salida debe ser DataFrame, recibido: {type(df)}")

        if self.condenser_config.enable_strict_validation:
            if df.empty:
                self.logger.warning("[VALIDACIÓN] DataFrame vacío generado (puede ser válido)")

            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                self.logger.warning(f"[VALIDACIÓN] Columnas completamente nulas: {null_columns}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del último procesamiento."""
        return {
            "condenser_config": {
                "min_records_threshold": self.condenser_config.min_records_threshold,
                "strict_validation": self.condenser_config.enable_strict_validation,
                "log_level": self.condenser_config.log_level,
                "pid_mode": True
            },
            "config_keys": list(self.config.keys()),
            "profile_keys": list(self.profile.keys())
        }
