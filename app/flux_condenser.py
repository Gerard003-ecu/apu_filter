"""
Módulo de Capacitancia Lógica para el procesamiento de flujos de datos.

Este módulo introduce el `DataFluxCondenser`, un componente de alto nivel que
actúa como una fachada estabilizadora para el pipeline de procesamiento de
Análisis de Precios Unitarios (APU). Su función principal es garantizar la
integridad, coherencia y estabilidad del flujo de datos antes de que ingrese
al núcleo del sistema.

Principios de Diseño:
- **Capacitancia Lógica:** Inspirado en los principios de un circuito RC,
  el condensador "absorbe" datos crudos y los "descarga" de manera controlada,
  filtrando el ruido y la turbulencia.
- **Orquestación, no Implementación:** No contiene lógica de negocio de bajo
  nivel. En su lugar, orquesta componentes especializados como `ReportParserCrudo`
  (el "Guardia") y `APUProcessor` (el "Cirujano").
- **Telemetría Física:** Incorpora un `FluxPhysicsEngine` para calcular
  métricas de saturación y complejidad, proporcionando una visión cuantitativa
  de la "salud" del flujo de datos entrante.
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
    incluyendo sus parámetros para el motor de simulación física.

    Atributos:
        min_records_threshold (int): Número mínimo de registros necesarios para
            considerar un archivo como válido para el procesamiento.
        enable_strict_validation (bool): Si es `True`, activa validaciones
            adicionales en el DataFrame de salida, como la detección de
            columnas nulas.
        log_level (str): Nivel de logging para la instancia del condensador.
        system_capacitance (float): Parámetro físico que representa la
            capacidad ideal del sistema para procesar registros en un ciclo.
            Análogo a la capacitancia en Faradios.
        base_resistance (float): Parámetro físico que representa la fricción
            o complejidad inherente del sistema. Análogo a la resistencia
            en Ohmios.
    """
    min_records_threshold: int = 1
    enable_strict_validation: bool = True
    log_level: str = "INFO"
    # Configuración Física
    system_capacitance: float = 5000.0
    base_resistance: float = 10.0


# --- NUEVA CLASE: MOTOR DE FÍSICA ---
class FluxPhysicsEngine:
    """
    Simula el comportamiento de un flujo de datos usando un análogo de circuito RC.

    Esta clase modela la ingesta de datos como la carga de un condensador a
    través de una resistencia. Permite cuantificar la "saturación" del sistema
    en función de la cantidad de datos (carga) y su complejidad inherente
    (resistencia).

    Atributos:
        C (float): La capacitancia del sistema, análoga a la capacidad de
            procesamiento de registros.
        R (float): La resistencia base del sistema, análoga a la complejidad
            mínima del procesamiento.
        tau (float): La constante de tiempo (R * C), que caracteriza la
            velocidad de "carga" o procesamiento del sistema.
    """
    def __init__(self, capacitance: float, resistance: float):
        """
        Inicializa el motor de física con los parámetros del circuito.

        Args:
            capacitance (float): Capacidad del sistema (análogo a Faradios).
            resistance (float): Resistencia base del sistema (análogo a Ohmios).
        """
        self.C = capacitance
        self.R = resistance
        self.tau = self.R * self.C

    def calculate_saturation(self, load_size: int, complexity_factor: float) -> float:
        """
        Calcula el índice de saturación del sistema para una carga de datos.

        La saturación se modela como el "voltaje" en un condensador que se
        carga. Una saturación de 1.0 representa un sistema completamente
        saturado. La resistencia dinámica aumenta con la complejidad de los
        datos, ralentizando la "carga".

        La fórmula utilizada es: V(t) = 1 - e^(-t / RC)

        Args:
            load_size (int): El número de registros a procesar (análogo al
                tiempo de carga `t`).
            complexity_factor (float): Un factor entre 0 y 1 que representa la
                complejidad adicional de los datos (e.g., porcentaje de
                líneas que requirieron un parseo complejo).

        Returns:
            float: El índice de saturación, un valor entre 0.0 y 1.0.
        """
        dynamic_R = self.R * (1 + complexity_factor)
        current_tau = dynamic_R * self.C
        t = float(load_size)
        if current_tau == 0:
            return 1.0 if t > 0 else 0.0
        saturation = 1.0 - math.exp(-t / current_tau)
        return saturation

    def get_stability_status(self, saturation: float) -> str:
        """
        Clasifica el estado del flujo de datos basado en su índice de saturación.

        Args:
            saturation (float): El índice de saturación calculado.

        Returns:
            str: Una descripción textual del estado del flujo (Laminar,
                 Transitorio o Turbulento).
        """
        if saturation < 0.3: return "FLUJO LAMINAR (Estable)"
        if saturation < 0.7: return "FLUJO TRANSITORIO (Carga Media)"
        return "FLUJO TURBULENTO (Alta Saturación)"


class DataFluxCondenser:
    """
    Orquesta el pipeline de validación y procesamiento de archivos de APU.

    Actúa como una fachada que encapsula la complejidad de interactuar con
    múltiples componentes (`ReportParserCrudo`, `APUProcessor`). Su objetivo es
    proporcionar una única interfaz (`stabilize`) para procesar un archivo de
    forma segura y robusta.

    El "Condensador" implementa una metáfora de circuito eléctrico:
    1.  **Carga (Absorb & Filter):** El `ReportParserCrudo` absorbe la "corriente"
        inicial, filtrando el ruido y generando una señal cruda.
    2.  **Estabilización (Telemetría):** El `FluxPhysicsEngine` mide la
        "tensión" y "resistencia" del flujo de datos, evaluando su estabilidad.
    3.  **Descarga (Rectify Signal):** El `APUProcessor` procesa la señal
        filtrada y la convierte en un DataFrame estructurado y útil.

    Atributos:
        config (Dict[str, Any]): Configuración global de la aplicación.
        profile (Dict[str, Any]): Perfil específico para el tipo de archivo.
        condenser_config (CondenserConfig): Configuración operativa del
            propio condensador.
        logger (logging.Logger): Instancia de logger para este componente.
        physics (FluxPhysicsEngine): Motor de simulación física para telemetría.
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
        Inicializa una nueva instancia del `DataFluxCondenser`.

        Args:
            config (Dict[str, Any]): El diccionario de configuración global de
                la aplicación, que contiene ajustes para los subcomponentes.
            profile (Dict[str, Any]): El perfil de procesamiento específico
                para el archivo, que define mapeos de columnas y reglas.
            condenser_config (Optional[CondenserConfig]): Una configuración
                específica para el condensador. Si es `None`, se utilizarán
                los valores por defecto de `CondenserConfig`.
        """
        self._validate_initialization_params(config, profile)

        self.config = config
        self.profile = profile
        self.condenser_config = condenser_config or CondenserConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.condenser_config.log_level)

        # Inicializar Motor de Física
        self.physics = FluxPhysicsEngine(
            capacitance=self.condenser_config.system_capacitance,
            resistance=self.condenser_config.base_resistance
        )
        self.logger.info("DataFluxCondenser (con Motor Físico) inicializado")

    def _validate_initialization_params(
        self,
        config: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> None:
        """
        Valida los parámetros de inicialización del condensador.

        Verifica que `config` y `profile` sean diccionarios. En modo tolerante,
        advierte sobre la ausencia de claves esperadas en lugar de lanzar un
        error.

        Args:
            config (Dict[str, Any]): Diccionario de configuración global.
            profile (Dict[str, Any]): Diccionario de perfil de archivo.

        Raises:
            InvalidInputError: Si `config` o `profile` no son diccionarios.
        """
        if not isinstance(config, dict) or not isinstance(profile, dict):
            raise InvalidInputError("config y profile deben ser diccionarios válidos")

        missing_config_keys = self.REQUIRED_CONFIG_KEYS - set(config.keys())
        if missing_config_keys:
            logger.warning(
                f"Claves faltantes en config (modo tolerante): {missing_config_keys}"
            )

        missing_profile_keys = self.REQUIRED_PROFILE_KEYS - set(profile.keys())
        if missing_profile_keys:
            logger.warning(
                f"Claves faltantes en profile (modo tolerante): {missing_profile_keys}"
            )

    def stabilize(self, file_path: str) -> pd.DataFrame:
        """
        Orquesta el ciclo completo de procesamiento de un archivo de APU.

        Este es el método principal y punto de entrada del condensador. Ejecuta
        la secuencia de validación, filtrado (absorción), telemetría física y
        procesamiento (rectificación) para transformar un archivo crudo en un
        DataFrame limpio y estructurado.

        Args:
            file_path (str): La ruta al archivo de APU que se va a procesar.

        Returns:
            pd.DataFrame: Un DataFrame de pandas con los datos de insumos
            procesados y validados. Retorna un DataFrame vacío si el archivo
            no contiene datos válidos o si el procesamiento no produce resultados.

        Raises:
            InvalidInputError: Si `file_path` no es válido (e.g., no existe,
                es un directorio).
            ProcessingError: Si ocurre un error irrecuperable en cualquier
                etapa del pipeline de procesamiento.
        """
        start_time = time.time()
        self.logger.info(f"⚡ [FÍSICA] Iniciando ciclo de estabilización para: {file_path}")

        try:
            validated_path = self._validate_input_file(file_path)
            parsed_data = self._absorb_and_filter(validated_path)

            if not self._validate_parsed_data(parsed_data):
                self.logger.warning(
                    "[ADVERTENCIA] La carga no generó señal válida (0 registros)"
                )
                return pd.DataFrame()

            # --- CÁLCULO FÍSICO ---
            total_records = len(parsed_data.raw_records)
            cache_size = len(parsed_data.parse_cache)
            complexity = 1.0 - (cache_size / total_records) if total_records > 0 else 0.0
            saturation = self.physics.calculate_saturation(total_records, complexity)
            status = self.physics.get_stability_status(saturation)

            self.logger.info(
                f"📊 [TELEMETRÍA] Saturación: {saturation:.4f} | "
                f"Complejidad: {complexity:.2f} | Estado: {status}"
            )
            # ----------------------

            df_stabilized = self._rectify_signal(parsed_data)
            self._validate_output(df_stabilized)

            elapsed = time.time() - start_time
            self.logger.info(
                f"✅ [ÉXITO] Flujo estabilizado en {elapsed:.2f}s: "
                f"{len(df_stabilized)} registros procesados"
            )
            return df_stabilized

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
        """
        Valida que el archivo de entrada exista y sea accesible.

        Args:
            file_path: Ruta al archivo a validar.

        Returns:
            Path validado.

        Raises:
            InvalidInputError: Si el archivo no existe o no es accesible.
        """
        if not file_path or not isinstance(file_path, str):
            raise InvalidInputError(
                f"file_path debe ser una cadena no vacía, recibido: {type(file_path)}"
            )

        path = Path(file_path)

        if not path.exists():
            raise InvalidInputError(f"El archivo no existe: {file_path}")

        if not path.is_file():
            raise InvalidInputError(f"La ruta no es un archivo: {file_path}")

        if path.suffix.lower() not in {'.csv', '.txt'}:
            self.logger.warning(
                f"Extensión inusual detectada: {path.suffix}. "
                "Se esperaba .csv o .txt"
            )

        self.logger.debug(f"[VALIDACIÓN] Archivo validado: {path}")
        return path

    def _absorb_and_filter(self, file_path: Path) -> ParsedData:
        """
        Usa ReportParserCrudo para filtrar el ruido de entrada.

        Args:
            file_path: Ruta validada al archivo.

        Returns:
            ParsedData con registros crudos y caché de parseo.

        Raises:
            ProcessingError: Si el parseo falla.
        """
        self.logger.debug("[FASE 1] Filtrando ruido con ReportParserCrudo...")

        try:
            # FIX: Pasar explícitamente 'config' como argumento de palabra clave
            parser = ReportParserCrudo(
                str(file_path),
                profile=self.profile,
                config=self.config
            )
            raw_records = parser.parse_to_raw()
            parse_cache = parser.get_parse_cache()

            # Validación de consistencia
            if raw_records is None:
                raw_records = []
                self.logger.warning(
                    "[FASE 1] Parser retornó None, se asume lista vacía"
                )

            if parse_cache is None:
                parse_cache = {}
                self.logger.warning(
                    "[FASE 1] Cache retornó None, se asume diccionario vacío"
                )

            parsed_data = ParsedData(
                raw_records=raw_records,
                parse_cache=parse_cache
            )

            self.logger.debug(
                f"[FASE 1] Filtrado completado: {len(raw_records)} registros extraídos"
            )
            return parsed_data

        except Exception as e:
            raise ProcessingError(
                f"Error durante el filtrado con ReportParserCrudo: {e}"
            ) from e

    def _validate_parsed_data(self, parsed_data: ParsedData) -> bool:
        """
        Valida que los datos parseados sean coherentes y suficientes.

        Args:
            parsed_data: Datos parseados a validar.

        Returns:
            True si los datos son válidos, False si están vacíos pero válidos.

        Raises:
            ProcessingError: Si los datos están corruptos.
        """
        if not isinstance(parsed_data.raw_records, list):
            raise ProcessingError(
                f"raw_records debe ser lista, recibido: {type(parsed_data.raw_records)}"
            )

        if not isinstance(parsed_data.parse_cache, dict):
            raise ProcessingError(
                f"parse_cache debe ser dict, recibido: {type(parsed_data.parse_cache)}"
            )

        records_count = len(parsed_data.raw_records)

        if records_count < self.condenser_config.min_records_threshold:
            self.logger.warning(
                f"[VALIDACIÓN] Registros insuficientes: {records_count} < "
                f"{self.condenser_config.min_records_threshold}"
            )
            return False

        self.logger.debug(f"[VALIDACIÓN] Datos parseados válidos: {records_count} registros")
        return True

    def _rectify_signal(self, parsed_data: ParsedData) -> pd.DataFrame:
        """
        Usa APUProcessor para convertir la señal filtrada en datos utilizables.

        Args:
            parsed_data: Datos parseados a procesar.

        Returns:
            DataFrame con datos procesados.

        Raises:
            ProcessingError: Si el procesamiento falla.
        """
        self.logger.debug("[FASE 2] Rectificando señal con APUProcessor...")

        try:
            # 1. Instanciar APUProcessor SIN raw_records
            processor = APUProcessor(
                config=self.config,
                profile=self.profile,
                parse_cache=parsed_data.parse_cache
            )

            # 2. Pasar raw_records directamente a process_all
            # NOTA: Asignamos manualmente raw_records porque el nuevo APUProcessor
            # espera que estén disponibles en self.raw_records o pasados de alguna forma.
            # Dado que el nuevo APUProcessor adaptativo usa self.raw_records,
            # debemos asignarlos antes de llamar a process_all.
            processor.raw_records = parsed_data.raw_records

            df_result = processor.process_all()

            if not isinstance(df_result, pd.DataFrame):
                raise ProcessingError(
                    f"APUProcessor.process_all() debe retornar DataFrame, "
                    f"recibido: {type(df_result)}"
                )

            self.logger.debug(
                f"[FASE 2] Rectificación completada: {len(df_result)} registros procesados"
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
            df: DataFrame a validar.

        Raises:
            ProcessingError: Si el DataFrame es inválido.
        """
        if not isinstance(df, pd.DataFrame):
            raise ProcessingError(
                f"La salida debe ser DataFrame, recibido: {type(df)}"
            )

        if self.condenser_config.enable_strict_validation:
            if df.empty:
                self.logger.warning(
                    "[VALIDACIÓN] DataFrame vacío generado (puede ser válido)"
                )

            # Validar que no haya columnas completamente nulas
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                self.logger.warning(
                    f"[VALIDACIÓN] Columnas completamente nulas: {null_columns}"
                )

        self.logger.debug(
            f"[VALIDACIÓN] Salida validada: {df.shape[0]} filas, {df.shape[1]} columnas"
        )

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del último procesamiento.

        Returns:
            Diccionario con estadísticas de procesamiento.
        """
        return {
            "condenser_config": {
                "min_records_threshold": self.condenser_config.min_records_threshold,
                "strict_validation": self.condenser_config.enable_strict_validation,
                "log_level": self.condenser_config.log_level
            },
            "config_keys": list(self.config.keys()),
            "profile_keys": list(self.profile.keys())
        }
