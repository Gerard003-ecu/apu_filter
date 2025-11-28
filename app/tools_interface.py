"""
Interfaz de herramientas para la Matriz de Interacción Central (MIC).
Actúa como adaptador entre la API REST y los scripts de mantenimiento.

Este módulo proporciona funciones de alto nivel para:
- Diagnóstico de archivos (APUs, Insumos, Presupuesto)
- Limpieza de archivos CSV
- Consulta de estado de telemetría
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

from scripts.clean_csv import CSVCleaner
from scripts.diagnose_apus_file import APUFileDiagnostic
from scripts.diagnose_insumos_file import InsumosFileDiagnostic
from scripts.diagnose_presupuesto_file import PresupuestoFileDiagnostic

logger = logging.getLogger(__name__)


class FileType(str, Enum):
    """Tipos de archivo soportados para diagnóstico."""
    
    APUS = "apus"
    INSUMOS = "insumos"
    PRESUPUESTO = "presupuesto"
    
    @classmethod
    def values(cls) -> list[str]:
        """Retorna lista de valores válidos."""
        return [member.value for member in cls]


class DiagnosticError(Exception):
    """Excepción base para errores de diagnóstico."""
    pass


class FileNotFoundDiagnosticError(DiagnosticError):
    """El archivo a diagnosticar no existe."""
    pass


class UnsupportedFileTypeError(DiagnosticError):
    """Tipo de archivo no soportado."""
    pass


class CleaningError(Exception):
    """Excepción base para errores de limpieza."""
    pass


# Registro de diagnósticos: mapea tipo de archivo a su clase diagnóstica
_DIAGNOSTIC_REGISTRY: Dict[FileType, Type] = {
    FileType.APUS: APUFileDiagnostic,
    FileType.INSUMOS: InsumosFileDiagnostic,
    FileType.PRESUPUESTO: PresupuestoFileDiagnostic,
}

# Encodings comúnmente soportados
_SUPPORTED_ENCODINGS = frozenset({
    "utf-8", "utf-8-sig", "latin-1", "iso-8859-1", 
    "cp1252", "ascii", "utf-16", "utf-16-le", "utf-16-be"
})

# Delimitadores válidos para CSV
_VALID_DELIMITERS = frozenset({",", ";", "\t", "|", ":"})


def _validate_path_exists(path: Path, context: str = "File") -> None:
    """
    Valida que una ruta exista.
    
    Args:
        path: Ruta a validar
        context: Contexto para el mensaje de error
        
    Raises:
        FileNotFoundDiagnosticError: Si el archivo no existe
    """
    if not path.exists():
        raise FileNotFoundDiagnosticError(f"{context} not found: {path}")


def _normalize_path(file_path: Union[str, Path]) -> Path:
    """
    Normaliza una ruta a objeto Path.
    
    Args:
        file_path: Ruta como string o Path
        
    Returns:
        Path normalizado y resuelto
        
    Raises:
        ValueError: Si la ruta está vacía o es inválida
    """
    if not file_path:
        raise ValueError("File path cannot be empty")
    
    path = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Resolver rutas relativas si es posible
    try:
        return path.resolve() if path.exists() else path
    except OSError as e:
        raise ValueError(f"Invalid path '{file_path}': {e}")


def _normalize_file_type(file_type: Union[str, FileType]) -> FileType:
    """
    Normaliza el tipo de archivo a enum FileType.
    
    Args:
        file_type: Tipo como string o FileType
        
    Returns:
        FileType enum
        
    Raises:
        UnsupportedFileTypeError: Si el tipo no es válido
    """
    if isinstance(file_type, FileType):
        return file_type
    
    if not isinstance(file_type, str):
        raise UnsupportedFileTypeError(
            f"file_type must be string or FileType, got: {type(file_type).__name__}"
        )
    
    normalized = file_type.strip().lower()
    
    try:
        return FileType(normalized)
    except ValueError:
        valid_types = ", ".join(FileType.values())
        raise UnsupportedFileTypeError(
            f"Unknown file type: '{file_type}'. Valid types: {valid_types}"
        )


def _create_error_response(error: Union[str, Exception], **extras) -> Dict[str, Any]:
    """
    Crea una respuesta de error estandarizada.
    
    Args:
        error: Mensaje o excepción de error
        **extras: Campos adicionales para incluir
        
    Returns:
        Diccionario con estructura de error consistente
    """
    return {
        "success": False,
        "error": str(error),
        "error_type": type(error).__name__ if isinstance(error, Exception) else "Error",
        **extras
    }


def _create_success_response(data: Dict[str, Any], **extras) -> Dict[str, Any]:
    """
    Crea una respuesta exitosa estandarizada.
    
    Args:
        data: Datos de la respuesta
        **extras: Campos adicionales para incluir
        
    Returns:
        Diccionario con estructura de éxito consistente
    """
    response = {"success": True, **data, **extras}
    return response


def diagnose_file(
    file_path: Union[str, Path], 
    file_type: Union[str, FileType]
) -> Dict[str, Any]:
    """
    Ejecuta el diagnóstico apropiado según el tipo de archivo.

    Esta función actúa como dispatcher, seleccionando y ejecutando
    el diagnóstico correspondiente al tipo de archivo especificado.

    Args:
        file_path: Ruta al archivo a diagnosticar. Puede ser absoluta o relativa.
        file_type: Tipo de archivo. Valores válidos: 'apus', 'insumos', 'presupuesto'

    Returns:
        Diccionario con resultados del diagnóstico:
        - success (bool): Indica si el diagnóstico se ejecutó correctamente
        - Si success=True: Incluye datos del diagnóstico específico
        - Si success=False: Incluye 'error' y 'error_type'

    Examples:
        >>> result = diagnose_file("data/apus.csv", "apus")
        >>> if result["success"]:
        ...     print(f"Filas procesadas: {result.get('total_rows', 'N/A')}")
        
        >>> result = diagnose_file(Path("data/insumos.csv"), FileType.INSUMOS)
    """
    try:
        # Normalizar y validar entradas
        path = _normalize_path(file_path)
        normalized_type = _normalize_file_type(file_type)
        
        # Validar existencia del archivo
        _validate_path_exists(path, "File to diagnose")
        
        # Obtener clase diagnóstica del registro
        diagnostic_class = _DIAGNOSTIC_REGISTRY.get(normalized_type)
        
        if diagnostic_class is None:
            # Esto no debería ocurrir si el registro está sincronizado con FileType
            raise UnsupportedFileTypeError(
                f"No diagnostic registered for type: {normalized_type.value}"
            )
        
        # Ejecutar diagnóstico
        logger.info(f"Starting {normalized_type.value} diagnosis for: {path}")
        diagnostic = diagnostic_class(str(path))
        diagnostic.diagnose()
        
        # Obtener resultados
        result_data = diagnostic.to_dict()
        
        logger.info(f"Diagnosis completed for: {path}")
        return _create_success_response(
            result_data,
            file_type=normalized_type.value,
            file_path=str(path)
        )

    except (FileNotFoundDiagnosticError, UnsupportedFileTypeError) as e:
        logger.warning(f"Validation error in diagnose_file: {e}")
        return _create_error_response(e)
    
    except (OSError, IOError) as e:
        logger.error(f"I/O error during diagnosis: {e}", exc_info=True)
        return _create_error_response(e, error_category="io_error")
    
    except Exception as e:
        logger.error(
            f"Unexpected error executing diagnosis for {file_type}: {e}", 
            exc_info=True
        )
        return _create_error_response(e, error_category="unexpected")


def _validate_csv_parameters(delimiter: str, encoding: str) -> None:
    """
    Valida parámetros de limpieza CSV.
    
    Args:
        delimiter: Delimitador a validar
        encoding: Encoding a validar
        
    Raises:
        ValueError: Si algún parámetro es inválido
    """
    if not delimiter:
        raise ValueError("Delimiter cannot be empty")
    
    if delimiter not in _VALID_DELIMITERS:
        valid = ", ".join(repr(d) for d in sorted(_VALID_DELIMITERS))
        raise ValueError(
            f"Invalid delimiter '{delimiter}'. Supported: {valid}"
        )
    
    if not encoding:
        raise ValueError("Encoding cannot be empty")
    
    encoding_lower = encoding.lower().strip()
    if encoding_lower not in _SUPPORTED_ENCODINGS:
        # Advertencia pero no error - puede haber encodings válidos no listados
        logger.warning(
            f"Encoding '{encoding}' not in common list. "
            f"Proceeding anyway, but verify if issues arise."
        )


def _generate_output_path(input_path: Path, suffix: str = "_clean") -> Path:
    """
    Genera ruta de salida basada en la de entrada.
    
    Args:
        input_path: Ruta del archivo de entrada
        suffix: Sufijo a añadir antes de la extensión
        
    Returns:
        Path para el archivo de salida
    """
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def clean_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    delimiter: str = ";",
    encoding: str = "utf-8",
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Ejecuta la limpieza de un archivo CSV.

    Procesa el archivo de entrada aplicando reglas de limpieza y
    genera un archivo de salida con los datos normalizados.

    Args:
        input_path: Ruta al archivo CSV a limpiar.
        output_path: Ruta donde guardar el archivo limpio.
            Si es None, se genera automáticamente añadiendo '_clean' al nombre.
        delimiter: Delimitador de campos esperado. Por defecto ';'.
            Valores válidos: ',', ';', '\\t', '|', ':'
        encoding: Encoding del archivo. Por defecto 'utf-8'.
        overwrite: Si True, sobrescribe el archivo de salida si existe.
            Por defecto True.

    Returns:
        Diccionario con resultados de la limpieza:
        - success (bool): Indica si la limpieza fue exitosa
        - Si success=True:
            - output_path (str): Ruta del archivo generado
            - Estadísticas de limpieza (rows_processed, errors_found, etc.)
        - Si success=False:
            - error (str): Descripción del error
            - error_type (str): Tipo de excepción

    Raises:
        No levanta excepciones directamente; todos los errores se
        encapsulan en el diccionario de retorno.

    Examples:
        >>> result = clean_file("data/raw.csv")
        >>> if result["success"]:
        ...     print(f"Archivo limpio: {result['output_path']}")
        
        >>> result = clean_file(
        ...     "data/raw.csv",
        ...     output_path="data/processed.csv",
        ...     delimiter=",",
        ...     encoding="latin-1"
        ... )
    """
    try:
        # Normalizar ruta de entrada
        input_p = _normalize_path(input_path)
        _validate_path_exists(input_p, "Input file")
        
        # Validar parámetros CSV
        _validate_csv_parameters(delimiter, encoding)
        
        # Determinar ruta de salida
        if output_path is None:
            output_p = _generate_output_path(input_p)
            logger.debug(f"Auto-generated output path: {output_p}")
        else:
            output_p = _normalize_path(output_path)
        
        # Verificar que no sobrescribimos input accidentalmente
        if output_p.resolve() == input_p.resolve():
            raise ValueError(
                "Output path cannot be the same as input path. "
                "Use a different output path or let it auto-generate."
            )
        
        # Verificar/crear directorio de salida
        output_dir = output_p.parent
        if not output_dir.exists():
            logger.info(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar sobrescritura
        if output_p.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists and overwrite=False: {output_p}"
            )
        
        # Ejecutar limpieza
        logger.info(f"Starting CSV cleaning: {input_p} -> {output_p}")
        
        cleaner = CSVCleaner(
            input_path=str(input_p),
            output_path=str(output_p),
            delimiter=delimiter,
            encoding=encoding,
            overwrite=overwrite,
        )
        
        stats = cleaner.clean()
        
        # Construir respuesta
        result_data = stats.to_dict() if hasattr(stats, 'to_dict') else dict(stats)
        
        logger.info(f"CSV cleaning completed: {output_p}")
        
        return _create_success_response(
            result_data,
            output_path=str(output_p),
            input_path=str(input_p)
        )

    except (FileNotFoundDiagnosticError, ValueError, FileExistsError) as e:
        logger.warning(f"Validation error in clean_file: {e}")
        return _create_error_response(e)
    
    except (OSError, IOError) as e:
        logger.error(f"I/O error during cleaning: {e}", exc_info=True)
        return _create_error_response(e, error_category="io_error")
    
    except Exception as e:
        logger.error(f"Unexpected error executing cleaner: {e}", exc_info=True)
        return _create_error_response(e, error_category="unexpected")


def get_telemetry_status(
    telemetry_context: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Obtiene el estado actual del sistema (Vector de Estado).

    Proporciona información sobre el estado de procesamiento actual,
    ya sea desde un contexto de telemetría activo o un estado global.

    Args:
        telemetry_context: Objeto de contexto de telemetría opcional.
            Si se proporciona, debe tener método 'get_business_report()'.
            Si es None, retorna estado genérico del sistema.

    Returns:
        Diccionario con estado del sistema:
        - status (str): Estado actual ('ACTIVE', 'IDLE', 'ERROR')
        - message (str): Descripción del estado
        - system_health (str): Salud general ('HEALTHY', 'DEGRADED', 'UNKNOWN')
        - Campos adicionales según el contexto de telemetría

    Examples:
        >>> status = get_telemetry_status()
        >>> print(f"Sistema: {status['status']}")
        
        >>> status = get_telemetry_status(request.telemetry_context)
        >>> if status['system_health'] == 'HEALTHY':
        ...     print("Sistema operando normalmente")
    """
    # Estado por defecto cuando no hay contexto activo
    default_status: Dict[str, Any] = {
        "status": "IDLE",
        "message": "No active processing context",
        "system_health": "UNKNOWN",
        "has_active_context": False,
    }
    
    if telemetry_context is None:
        logger.debug("Telemetry status requested without active context")
        return default_status
    
    # Verificar que el contexto tiene el método esperado
    if not hasattr(telemetry_context, 'get_business_report'):
        logger.warning(
            f"Telemetry context missing 'get_business_report' method. "
            f"Context type: {type(telemetry_context).__name__}"
        )
        return {
            **default_status,
            "status": "ERROR",
            "message": "Invalid telemetry context: missing required method",
            "system_health": "DEGRADED",
        }
    
    try:
        report = telemetry_context.get_business_report()
        
        # Asegurar que el reporte es un diccionario
        if not isinstance(report, dict):
            logger.warning(
                f"Telemetry report is not a dict: {type(report).__name__}"
            )
            report = {"raw_report": str(report)}
        
        # Enriquecer con metadatos
        report["has_active_context"] = True
        report.setdefault("status", "ACTIVE")
        report.setdefault("system_health", "HEALTHY")
        
        return report
        
    except Exception as e:
        logger.error(
            f"Error retrieving telemetry report: {e}", 
            exc_info=True
        )
        return {
            **default_status,
            "status": "ERROR",
            "message": f"Failed to retrieve telemetry: {e}",
            "system_health": "DEGRADED",
            "error": str(e),
        }


# === Funciones de utilidad pública ===

def get_supported_file_types() -> list[str]:
    """
    Retorna lista de tipos de archivo soportados para diagnóstico.
    
    Returns:
        Lista de strings con tipos válidos
        
    Examples:
        >>> types = get_supported_file_types()
        >>> print(types)  # ['apus', 'insumos', 'presupuesto']
    """
    return FileType.values()


def is_valid_file_type(file_type: str) -> bool:
    """
    Verifica si un tipo de archivo es válido.
    
    Args:
        file_type: Tipo a verificar
        
    Returns:
        True si es válido, False en caso contrario
        
    Examples:
        >>> is_valid_file_type("apus")
        True
        >>> is_valid_file_type("unknown")
        False
    """
    try:
        _normalize_file_type(file_type)
        return True
    except (UnsupportedFileTypeError, TypeError):
        return False