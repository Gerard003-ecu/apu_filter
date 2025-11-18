import logging
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

from .report_parser_crudo import ReportParserCrudo
from .apu_processor import APUProcessor
from .schemas import InsumoProcesado

logger = logging.getLogger(__name__)


class ParsedData(NamedTuple):
    """Estructura de datos resultante del parseo."""
    raw_records: List[Dict[str, Any]]
    parse_cache: Dict[str, Any]


class DataFluxCondenserError(Exception):
    """Excepción base para errores del condensador."""
    pass


class InvalidInputError(DataFluxCondenserError):
    """Error de validación de entrada."""
    pass


class ProcessingError(DataFluxCondenserError):
    """Error durante el procesamiento."""
    pass


@dataclass(frozen=True)
class CondenserConfig:
    """Configuración validada del condensador."""
    min_records_threshold: int = 1
    enable_strict_validation: bool = True
    log_level: str = "INFO"


class DataFluxCondenser:
    """
    Implementación de Capacitancia Lógica para el procesamiento de APUs.
    
    Actúa como un condensador y estabilizador de flujo que:
    1. Absorbe la carga bruta (archivos CSV crudos).
    2. Filtra el ruido (usando ReportParserCrudo como filtro pasa-bajos).
    3. Rectifica la señal (usando APUProcessor para estructurar datos).
    4. Descarga un flujo limpio y estable (DataFrame) al sistema principal.
    
    Attributes:
        config (Dict[str, Any]): Configuración del sistema.
        profile (Dict[str, Any]): Perfil de procesamiento.
        condenser_config (CondenserConfig): Configuración específica del condensador.
        
    Raises:
        InvalidInputError: Si los parámetros de entrada son inválidos.
        ProcessingError: Si falla el procesamiento de datos.
    """
    
    # Constantes
    REQUIRED_CONFIG_KEYS = {'parser_settings', 'processor_settings'}
    REQUIRED_PROFILE_KEYS = {'columns_mapping', 'validation_rules'}
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        profile: Dict[str, Any],
        condenser_config: Optional[CondenserConfig] = None
    ):
        """
        Inicializa el condensador con validación de dependencias.
        
        Args:
            config: Configuración general del sistema.
            profile: Perfil de procesamiento de APUs.
            condenser_config: Configuración específica del condensador.
            
        Raises:
            InvalidInputError: Si config o profile son inválidos.
        """
        self._validate_initialization_params(config, profile)
        
        self.config = config
        self.profile = profile
        self.condenser_config = condenser_config or CondenserConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.condenser_config.log_level)
        
        self.logger.info("DataFluxCondenser inicializado correctamente")

    def _validate_initialization_params(
        self, 
        config: Dict[str, Any], 
        profile: Dict[str, Any]
    ) -> None:
        """
        Valida que los parámetros de inicialización sean coherentes.
        
        Args:
            config: Configuración a validar.
            profile: Perfil a validar.
            
        Raises:
            InvalidInputError: Si falta alguna clave requerida.
        """
        if not isinstance(config, dict) or not isinstance(profile, dict):
            raise InvalidInputError(
                "config y profile deben ser diccionarios válidos"
            )
        
        # Validación flexible: advertir sobre claves faltantes sin bloquear
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
        Proceso de Carga y Descarga del Condensador.
        
        Ejecuta el ciclo completo de estabilización:
        1. Validación de entrada
        2. Filtrado de señal (parsing)
        3. Rectificación (procesamiento)
        4. Validación de salida
        
        Args:
            file_path: Ruta al archivo CSV a procesar.
            
        Returns:
            DataFrame estabilizado con datos procesados.
            
        Raises:
            InvalidInputError: Si el archivo no existe o es inválido.
            ProcessingError: Si falla el procesamiento.
        """
        self.logger.info(
            f"[INICIO] Ciclo de estabilización para: {file_path}"
        )
        
        try:
            # Validación de entrada
            validated_path = self._validate_input_file(file_path)
            
            # FASE 1: FILTRADO (El Guardia)
            parsed_data = self._absorb_and_filter(validated_path)
            
            if not self._validate_parsed_data(parsed_data):
                self.logger.warning(
                    "[ADVERTENCIA] La carga no generó señal válida (0 registros)"
                )
                return pd.DataFrame()

            # FASE 2: RECTIFICACIÓN (El Cirujano)
            df_stabilized = self._rectify_signal(parsed_data)
            
            # Validación de salida
            self._validate_output(df_stabilized)
            
            self.logger.info(
                f"[ÉXITO] Flujo estabilizado: {len(df_stabilized)} registros procesados"
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
        
        if not path.suffix.lower() in {'.csv', '.txt'}:
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
            parser = ReportParserCrudo(
                str(file_path), 
                self.profile, 
                self.config
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
            processor = APUProcessor(
                raw_records=parsed_data.raw_records,
                config=self.config,
                profile=self.profile,
                parse_cache=parsed_data.parse_cache
            )
            
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