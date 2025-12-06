"""
Interfaz de herramientas para la Matriz de Interacción Central (MIC).
Actúa como adaptador entre la API REST y los scripts de mantenimiento.

Este módulo proporciona funciones de alto nivel para:
- Diagnóstico de archivos (APUs, Insumos, Presupuesto)
- Limpieza de archivos CSV
- Consulta de estado de telemetría
"""

import codecs
import logging
import os
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

from scripts.clean_csv import CSVCleaner
from scripts.diagnose_apus_file import APUFileDiagnostic
from scripts.diagnose_insumos_file import InsumosFileDiagnostic
from scripts.diagnose_presupuesto_file import PresupuestoFileDiagnostic

logger = logging.getLogger(__name__)


# === Configuración y Constantes ===

MAX_FILE_SIZE_BYTES: Final[int] = 100 * 1024 * 1024  # 100 MB

SUPPORTED_ENCODINGS: Final[frozenset[str]] = frozenset({
    "utf-8", "utf-8-sig", "latin-1", "iso-8859-1",
    "cp1252", "ascii", "utf-16", "utf-16-le", "utf-16-be"
})

VALID_DELIMITERS: Final[frozenset[str]] = frozenset({",", ";", "\t", "|", ":"})

VALID_EXTENSIONS: Final[frozenset[str]] = frozenset({".csv", ".txt", ".tsv"})


# === Protocolos ===

@runtime_checkable
class TelemetryContextProtocol(Protocol):
    """Protocolo para contextos de telemetría."""

    def get_business_report(self) -> Dict[str, Any]:
        """Obtiene el reporte de negocio."""
        ...


@runtime_checkable
class DiagnosticProtocol(Protocol):
    """Protocolo para clases diagnósticas."""

    def diagnose(self) -> None:
        """Ejecuta el diagnóstico."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Retorna resultados como diccionario."""
        ...


# === Enums ===

class FileType(str, Enum):
    """Tipos de archivo soportados para diagnóstico."""

    APUS = "apus"
    INSUMOS = "insumos"
    PRESUPUESTO = "presupuesto"

    @classmethod
    def values(cls) -> List[str]:
        """Retorna lista de valores válidos."""
        return [member.value for member in cls]

    @classmethod
    def from_string(cls, value: str) -> "FileType":
        """
        Convierte un string a FileType de forma segura.

        Args:
            value: String a convertir

        Returns:
            FileType correspondiente

        Raises:
            ValueError: Si el valor no es válido
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value).__name__}")

        normalized = value.strip().lower()

        for member in cls:
            if member.value == normalized:
                return member

        valid = ", ".join(cls.values())
        raise ValueError(f"'{value}' is not valid. Options: {valid}")


# === Excepciones ===

class DiagnosticError(Exception):
    """Excepción base para errores de diagnóstico."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.details: Dict[str, Any] = details or {}


class FileNotFoundDiagnosticError(DiagnosticError):
    """El archivo a diagnosticar no existe."""
    pass


class UnsupportedFileTypeError(DiagnosticError):
    """Tipo de archivo no soportado."""
    pass


class FileValidationError(DiagnosticError):
    """Error de validación de archivo."""
    pass


class CleaningError(Exception):
    """Excepción base para errores de limpieza."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.details: Dict[str, Any] = details or {}


# === Registro de Diagnósticos ===

_DIAGNOSTIC_REGISTRY: Final[Dict[FileType, Type]] = {
    FileType.APUS: APUFileDiagnostic,
    FileType.INSUMOS: InsumosFileDiagnostic,
    FileType.PRESUPUESTO: PresupuestoFileDiagnostic,
}


def _get_diagnostic_class(file_type: FileType) -> Type:
    """
    Obtiene la clase diagnóstica para un tipo de archivo.

    Args:
        file_type: Tipo de archivo

    Returns:
        Clase diagnóstica correspondiente

    Raises:
        UnsupportedFileTypeError: Si no hay diagnóstico registrado
    """
    diagnostic_class = _DIAGNOSTIC_REGISTRY.get(file_type)

    if diagnostic_class is None:
        raise UnsupportedFileTypeError(
            f"No diagnostic registered for type: {file_type.value}",
            details={
                "file_type": file_type.value,
                "available_types": FileType.values()
            }
        )

    return diagnostic_class


# === Funciones de Validación de Rutas ===

def _validate_path_not_empty(file_path: Union[str, Path, None]) -> None:
    """
    Valida que la ruta no esté vacía o sea None.

    Args:
        file_path: Ruta a validar

    Raises:
        ValueError: Si la ruta está vacía o es None
    """
    if file_path is None:
        raise ValueError("File path cannot be None")

    if isinstance(file_path, str):
        if not file_path.strip():
            raise ValueError("File path cannot be empty string")
    elif isinstance(file_path, Path):
        if str(file_path) in ("", "."):
            raise ValueError("File path cannot be empty")
    else:
        raise ValueError(
            f"File path must be str or Path, got: {type(file_path).__name__}"
        )


def _normalize_path(file_path: Union[str, Path]) -> Path:
    """
    Normaliza una ruta a objeto Path resuelto.

    Args:
        file_path: Ruta como string o Path

    Returns:
        Path normalizado y expandido

    Raises:
        ValueError: Si la ruta está vacía o es inválida
    """
    _validate_path_not_empty(file_path)

    try:
        path = Path(file_path) if isinstance(file_path, str) else file_path

        # Expandir ~ a home directory
        path = path.expanduser()

        # Intentar resolver la ruta si es posible
        try:
            if path.exists() or path.parent.exists():
                path = path.resolve()
        except (OSError, RuntimeError):
            # Mantener path sin resolver si hay error de permisos o recursión
            pass

        return path

    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid path '{file_path}': {e}") from e
    except OSError as e:
        raise ValueError(f"OS error processing path '{file_path}': {e}") from e


def _validate_file_exists(path: Path) -> None:
    """
    Valida que un archivo exista y sea accesible para lectura.

    Args:
        path: Ruta del archivo

    Raises:
        FileNotFoundDiagnosticError: Si el archivo no existe
        FileValidationError: Si no es archivo o no es legible
    """
    if not path.exists():
        raise FileNotFoundDiagnosticError(
            f"File not found: {path}",
            details={
                "path": str(path),
                "parent_exists": path.parent.exists() if path.parent else False
            }
        )

    if not path.is_file():
        raise FileValidationError(
            f"Path exists but is not a file: {path}",
            details={
                "path": str(path),
                "is_directory": path.is_dir(),
                "is_symlink": path.is_symlink()
            }
        )

    if not os.access(path, os.R_OK):
        raise FileValidationError(
            f"File exists but is not readable (permission denied): {path}",
            details={"path": str(path)}
        )


def _validate_file_size(
    path: Path,
    max_size: int = MAX_FILE_SIZE_BYTES
) -> int:
    """
    Valida que el archivo no exceda el tamaño máximo.

    Args:
        path: Ruta del archivo
        max_size: Tamaño máximo en bytes

    Returns:
        Tamaño del archivo en bytes

    Raises:
        FileValidationError: Si excede el tamaño o está vacío
    """
    try:
        file_size = path.stat().st_size
    except OSError as e:
        raise FileValidationError(
            f"Cannot determine file size: {path}",
            details={"path": str(path), "error": str(e)}
        ) from e

    if file_size == 0:
        logger.warning(f"File is empty: {path}")
        # No lanzamos excepción, pero advertimos

    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        file_mb = file_size / (1024 * 1024)
        raise FileValidationError(
            f"File size ({file_mb:.2f} MB) exceeds maximum ({max_mb:.2f} MB): {path}",
            details={
                "path": str(path),
                "file_size_bytes": file_size,
                "file_size_mb": file_mb,
                "max_size_bytes": max_size,
                "max_size_mb": max_mb
            }
        )

    return file_size


def _validate_file_extension(
    path: Path,
    valid_extensions: frozenset[str] = VALID_EXTENSIONS
) -> str:
    """
    Valida que la extensión del archivo sea válida.

    Args:
        path: Ruta del archivo
        valid_extensions: Conjunto de extensiones válidas

    Returns:
        Extensión normalizada (lowercase)

    Raises:
        FileValidationError: Si la extensión no es válida
    """
    extension = path.suffix.lower()

    if not extension:
        raise FileValidationError(
            f"File has no extension: {path}",
            details={
                "path": str(path),
                "valid_extensions": sorted(valid_extensions)
            }
        )

    if extension not in valid_extensions:
        raise FileValidationError(
            f"Invalid file extension '{extension}'. "
            f"Expected: {sorted(valid_extensions)}",
            details={
                "path": str(path),
                "extension": extension,
                "valid_extensions": sorted(valid_extensions)
            }
        )

    return extension


# === Funciones de Validación de Parámetros CSV ===

def _validate_encoding(encoding: str) -> str:
    """
    Valida y normaliza un encoding usando codecs de Python.

    Args:
        encoding: Encoding a validar

    Returns:
        Encoding normalizado

    Raises:
        ValueError: Si el encoding es inválido o desconocido
    """
    if not encoding:
        raise ValueError("Encoding cannot be empty")

    if not isinstance(encoding, str):
        raise ValueError(
            f"Encoding must be string, got: {type(encoding).__name__}"
        )

    encoding_normalized = encoding.strip().lower()

    # Verificar que Python reconozca el encoding
    try:
        codec_info = codecs.lookup(encoding_normalized)
        # Usar el nombre canónico del codec
        canonical_name = codec_info.name
    except LookupError as e:
        raise ValueError(
            f"Unknown encoding: '{encoding}'. "
            f"Python cannot find a codec for this encoding."
        ) from e

    # Advertir si no está en la lista de encodings comunes
    if encoding_normalized not in SUPPORTED_ENCODINGS:
        logger.warning(
            f"Encoding '{encoding}' (canonical: '{canonical_name}') "
            f"is valid but not in common list. Proceeding anyway."
        )

    return encoding_normalized


def _validate_delimiter(delimiter: str) -> str:
    """
    Valida un delimitador CSV.

    Args:
        delimiter: Delimitador a validar

    Returns:
        Delimitador validado

    Raises:
        ValueError: Si el delimitador es inválido
    """
    if delimiter is None:
        raise ValueError("Delimiter cannot be None")

    if not isinstance(delimiter, str):
        raise ValueError(
            f"Delimiter must be string, got: {type(delimiter).__name__}"
        )

    if not delimiter:
        raise ValueError("Delimiter cannot be empty")

    if len(delimiter) != 1:
        raise ValueError(
            f"Delimiter must be single character, "
            f"got '{delimiter}' (length: {len(delimiter)})"
        )

    if delimiter not in VALID_DELIMITERS:
        valid_repr = ", ".join(repr(d) for d in sorted(VALID_DELIMITERS))
        raise ValueError(
            f"Invalid delimiter {repr(delimiter)}. Valid: {valid_repr}"
        )

    return delimiter


def _validate_csv_parameters(
    delimiter: str,
    encoding: str
) -> Tuple[str, str]:
    """
    Valida parámetros de procesamiento CSV.

    Args:
        delimiter: Delimitador a validar
        encoding: Encoding a validar

    Returns:
        Tupla (delimiter_validado, encoding_validado)

    Raises:
        ValueError: Si algún parámetro es inválido
    """
    validated_delimiter = _validate_delimiter(delimiter)
    validated_encoding = _validate_encoding(encoding)

    return validated_delimiter, validated_encoding


# === Funciones de Validación de Tipo de Archivo ===

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
            f"file_type must be string or FileType, "
            f"got: {type(file_type).__name__}",
            details={"received_type": type(file_type).__name__}
        )

    try:
        return FileType.from_string(file_type)
    except ValueError:
        valid_types = ", ".join(FileType.values())
        raise UnsupportedFileTypeError(
            f"Unknown file type: '{file_type}'. Valid: {valid_types}",
            details={
                "received": file_type,
                "valid_types": FileType.values()
            }
        )


# === Funciones de Respuesta ===

def _create_error_response(
    error: Union[str, Exception],
    **extras: Any
) -> Dict[str, Any]:
    """
    Crea una respuesta de error estandarizada.

    Args:
        error: Mensaje o excepción de error
        **extras: Campos adicionales para incluir

    Returns:
        Diccionario con estructura de error consistente
    """
    error_message = str(error)
    error_type = (
        type(error).__name__ if isinstance(error, Exception) else "Error"
    )

    response: Dict[str, Any] = {
        "success": False,
        "error": error_message,
        "error_type": error_type,
    }

    # Incluir detalles si la excepción los tiene
    if isinstance(error, (DiagnosticError, CleaningError)):
        if error.details:
            response["error_details"] = error.details

    # Agregar extras, filtrando valores None
    for key, value in extras.items():
        if value is not None:
            response[key] = value

    return response


def _create_success_response(
    data: Dict[str, Any],
    **extras: Any
) -> Dict[str, Any]:
    """
    Crea una respuesta exitosa estandarizada.

    Args:
        data: Datos de la respuesta
        **extras: Campos adicionales para incluir

    Returns:
        Diccionario con estructura de éxito consistente
    """
    if not isinstance(data, dict):
        logger.warning(
            f"Expected dict for data, got {type(data).__name__}. Converting."
        )
        data = {"data": data}

    response: Dict[str, Any] = {"success": True}
    response.update(data)

    for key, value in extras.items():
        if value is not None:
            response[key] = value

    return response


# === Funciones de Generación y Validación de Rutas de Salida ===

def _generate_output_path(
    input_path: Path,
    suffix: str = "_clean"
) -> Path:
    """
    Genera ruta de salida basada en la de entrada.

    Args:
        input_path: Ruta del archivo de entrada
        suffix: Sufijo a añadir antes de la extensión

    Returns:
        Path para el archivo de salida
    """
    if not suffix:
        suffix = "_clean"

    new_name = f"{input_path.stem}{suffix}{input_path.suffix}"
    return input_path.with_name(new_name)


def _validate_output_path(
    input_path: Path,
    output_path: Path,
    overwrite: bool
) -> None:
    """
    Valida que la ruta de salida sea válida.

    Args:
        input_path: Ruta de entrada
        output_path: Ruta de salida
        overwrite: Si se permite sobrescribir

    Raises:
        ValueError: Si la ruta de salida es igual a la de entrada
        FileExistsError: Si el archivo existe y overwrite=False
        PermissionError: Si no hay permisos de escritura
    """
    # Comparar rutas de forma segura
    try:
        input_resolved = input_path.resolve()

        if output_path.exists():
            output_resolved = output_path.resolve()
        else:
            # Para archivos que no existen, resolver el padre
            output_resolved = (
                output_path.parent.resolve() / output_path.name
                if output_path.parent.exists()
                else output_path
            )

        if input_resolved == output_resolved:
            raise ValueError(
                "Output path cannot be the same as input path. "
                "Use a different path or let it auto-generate."
            )
    except OSError as e:
        logger.warning(f"Could not fully resolve paths for comparison: {e}")

    # Verificar sobrescritura
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output file exists and overwrite=False: {output_path}"
            )

        if not os.access(output_path, os.W_OK):
            raise PermissionError(
                f"Output file exists but is not writable: {output_path}"
            )


def _ensure_output_directory(output_path: Path) -> None:
    """
    Asegura que el directorio de salida exista y sea escribible.

    Args:
        output_path: Ruta del archivo de salida

    Raises:
        OSError: Si no se puede crear el directorio
        PermissionError: Si no hay permisos de escritura
    """
    output_dir = output_path.parent

    if not output_dir.exists():
        logger.info(f"Creating output directory: {output_dir}")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Failed to create output directory '{output_dir}': {e}"
            ) from e

    if not os.access(output_dir, os.W_OK):
        raise PermissionError(
            f"Output directory is not writable: {output_dir}"
        )


# === Funciones de Extracción de Resultados ===

def _extract_diagnostic_result(diagnostic: Any) -> Dict[str, Any]:
    """
    Extrae resultados de un objeto diagnóstico de forma segura.

    Args:
        diagnostic: Objeto diagnóstico

    Returns:
        Diccionario con resultados
    """
    if hasattr(diagnostic, 'to_dict') and callable(diagnostic.to_dict):
        try:
            result = diagnostic.to_dict()
            if isinstance(result, dict):
                return result
            logger.warning(
                f"to_dict() returned {type(result).__name__}, expected dict"
            )
            return {"raw_result": result}
        except Exception as e:
            logger.warning(f"Error calling to_dict(): {e}")
            return {"diagnostic_completed": True, "to_dict_error": str(e)}

    logger.warning(
        f"Diagnostic {type(diagnostic).__name__} lacks to_dict() method"
    )
    return {"diagnostic_completed": True}


def _extract_cleaning_stats(stats: Any) -> Dict[str, Any]:
    """
    Extrae estadísticas de limpieza de forma segura.

    Args:
        stats: Objeto de estadísticas

    Returns:
        Diccionario con estadísticas
    """
    if stats is None:
        return {"cleaning_completed": True}

    if isinstance(stats, dict):
        return stats

    if hasattr(stats, 'to_dict') and callable(stats.to_dict):
        try:
            result = stats.to_dict()
            if isinstance(result, dict):
                return result
        except Exception as e:
            logger.warning(f"Error calling stats.to_dict(): {e}")

    # Intentar conversión a dict
    try:
        result = dict(stats)
        if isinstance(result, dict):
            return result
    except (TypeError, ValueError) as e:
        logger.warning(f"Could not convert stats to dict: {e}")

    return {"cleaning_completed": True, "raw_stats": str(stats)}


# === Funciones Principales ===

def diagnose_file(
    file_path: Union[str, Path],
    file_type: Union[str, FileType],
    *,
    validate_extension: bool = True,
    max_file_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ejecuta el diagnóstico apropiado según el tipo de archivo.

    Esta función actúa como dispatcher, seleccionando y ejecutando
    el diagnóstico correspondiente al tipo de archivo especificado.

    Args:
        file_path: Ruta al archivo a diagnosticar.
        file_type: Tipo de archivo ('apus', 'insumos', 'presupuesto').
        validate_extension: Si True, valida extensión del archivo.
        max_file_size: Tamaño máximo en bytes (None usa default).

    Returns:
        Diccionario con resultados:
        - success (bool): Si el diagnóstico fue exitoso
        - Si success=True: Datos del diagnóstico específico
        - Si success=False: error, error_type, error_details

    Examples:
        >>> result = diagnose_file("data/apus.csv", "apus")
        >>> if result["success"]:
        ...     print(f"Rows: {result.get('total_rows')}")
    """
    try:
        # Normalizar y validar entradas
        path = _normalize_path(file_path)
        normalized_type = _normalize_file_type(file_type)

        # Validaciones de archivo
        _validate_file_exists(path)

        if validate_extension:
            _validate_file_extension(path)

        effective_max = (
            max_file_size if max_file_size is not None
            else MAX_FILE_SIZE_BYTES
        )
        _validate_file_size(path, effective_max)

        # Obtener clase diagnóstica
        diagnostic_class = _get_diagnostic_class(normalized_type)

        # Ejecutar diagnóstico
        logger.info(
            f"Starting {normalized_type.value} diagnosis for: {path}"
        )

        diagnostic = diagnostic_class(str(path))
        diagnostic.diagnose()

        # Extraer resultados
        result_data = _extract_diagnostic_result(diagnostic)

        logger.info(f"Diagnosis completed successfully for: {path}")

        return _create_success_response(
            result_data,
            file_type=normalized_type.value,
            file_path=str(path)
        )

    except (
        FileNotFoundDiagnosticError,
        UnsupportedFileTypeError,
        FileValidationError
    ) as e:
        logger.warning(f"Validation error in diagnose_file: {e}")
        return _create_error_response(e)

    except ValueError as e:
        logger.warning(f"Value error in diagnose_file: {e}")
        return _create_error_response(e, error_category="validation")

    except PermissionError as e:
        logger.error(f"Permission error during diagnosis: {e}")
        return _create_error_response(e, error_category="permission")

    except (OSError, IOError) as e:
        logger.error(f"I/O error during diagnosis: {e}", exc_info=True)
        return _create_error_response(e, error_category="io_error")

    except Exception as e:
        logger.error(
            f"Unexpected error in diagnosis for {file_type}: {e}",
            exc_info=True
        )
        return _create_error_response(e, error_category="unexpected")


def clean_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    delimiter: str = ";",
    encoding: str = "utf-8",
    overwrite: bool = True,
    validate_extension: bool = True,
    max_file_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ejecuta la limpieza de un archivo CSV.

    Procesa el archivo de entrada aplicando reglas de limpieza y
    genera un archivo de salida con los datos normalizados.

    Args:
        input_path: Ruta al archivo CSV a limpiar.
        output_path: Ruta de salida (None genera automáticamente).
        delimiter: Delimitador de campos (default: ';').
        encoding: Encoding del archivo (default: 'utf-8').
        overwrite: Si sobrescribir archivo existente (default: True).
        validate_extension: Si validar extensión del archivo.
        max_file_size: Tamaño máximo en bytes (None usa default).

    Returns:
        Diccionario con resultados:
        - success (bool): Si la limpieza fue exitosa
        - Si success=True: output_path, input_path, estadísticas
        - Si success=False: error, error_type, error_details

    Examples:
        >>> result = clean_file("data/raw.csv")
        >>> if result["success"]:
        ...     print(f"Output: {result['output_path']}")
    """
    try:
        # Normalizar y validar entrada
        input_p = _normalize_path(input_path)
        _validate_file_exists(input_p)

        if validate_extension:
            _validate_file_extension(input_p)

        effective_max = (
            max_file_size if max_file_size is not None
            else MAX_FILE_SIZE_BYTES
        )
        _validate_file_size(input_p, effective_max)

        # Validar parámetros CSV
        validated_delimiter, validated_encoding = _validate_csv_parameters(
            delimiter, encoding
        )

        # Determinar ruta de salida
        if output_path is None:
            output_p = _generate_output_path(input_p)
            logger.debug(f"Auto-generated output path: {output_p}")
        else:
            output_p = _normalize_path(output_path)

        # Validar ruta de salida
        _validate_output_path(input_p, output_p, overwrite)
        _ensure_output_directory(output_p)

        # Ejecutar limpieza
        logger.info(
            f"Starting CSV cleaning: {input_p} -> {output_p} "
            f"(delimiter={repr(validated_delimiter)}, "
            f"encoding='{validated_encoding}')"
        )

        cleaner = CSVCleaner(
            input_path=str(input_p),
            output_path=str(output_p),
            delimiter=validated_delimiter,
            encoding=validated_encoding,
            overwrite=overwrite,
        )

        stats = cleaner.clean()

        # Extraer estadísticas
        result_data = _extract_cleaning_stats(stats)

        logger.info(f"CSV cleaning completed successfully: {output_p}")

        return _create_success_response(
            result_data,
            output_path=str(output_p),
            input_path=str(input_p)
        )

    except (FileNotFoundDiagnosticError, FileValidationError) as e:
        logger.warning(f"File validation error in clean_file: {e}")
        return _create_error_response(e)

    except (ValueError, FileExistsError) as e:
        logger.warning(f"Validation error in clean_file: {e}")
        return _create_error_response(e, error_category="validation")

    except PermissionError as e:
        logger.error(f"Permission error during cleaning: {e}")
        return _create_error_response(e, error_category="permission")

    except (OSError, IOError) as e:
        logger.error(f"I/O error during cleaning: {e}", exc_info=True)
        return _create_error_response(e, error_category="io_error")

    except Exception as e:
        logger.error(f"Unexpected error in cleaner: {e}", exc_info=True)
        return _create_error_response(e, error_category="unexpected")


def get_telemetry_status(
    telemetry_context: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Obtiene el estado actual del sistema (Vector de Estado).

    Args:
        telemetry_context: Objeto con método get_business_report().
            Si None, retorna estado genérico.

    Returns:
        Diccionario con estado del sistema:
        - status: 'ACTIVE', 'IDLE', o 'ERROR'
        - message: Descripción del estado
        - system_health: 'HEALTHY', 'DEGRADED', o 'UNKNOWN'
        - has_active_context: Si hay contexto activo

    Examples:
        >>> status = get_telemetry_status()
        >>> print(status['status'])  # 'IDLE'
    """
    base_status: Dict[str, Any] = {
        "status": "IDLE",
        "message": "No active processing context",
        "system_health": "UNKNOWN",
        "has_active_context": False,
    }

    if telemetry_context is None:
        logger.debug("Telemetry status requested without active context")
        return base_status.copy()

    # Verificar método requerido
    report_method = getattr(telemetry_context, 'get_business_report', None)

    if report_method is None:
        context_type = type(telemetry_context).__name__
        logger.warning(
            f"Telemetry context missing 'get_business_report'. "
            f"Type: {context_type}"
        )
        return {
            **base_status,
            "status": "ERROR",
            "message": (
                f"Invalid telemetry context ({context_type}): "
                f"missing get_business_report method"
            ),
            "system_health": "DEGRADED",
        }

    if not callable(report_method):
        logger.warning("get_business_report is not callable")
        return {
            **base_status,
            "status": "ERROR",
            "message": "get_business_report is not callable",
            "system_health": "DEGRADED",
        }

    try:
        report = report_method()

        # Validar y normalizar el reporte
        if report is None:
            logger.warning("Telemetry report returned None")
            report = {}
        elif not isinstance(report, dict):
            logger.warning(
                f"Telemetry report is {type(report).__name__}, not dict"
            )
            report = {"raw_report": str(report)}

        # Construir respuesta con valores por defecto
        result: Dict[str, Any] = {
            **base_status,
            **report,
            "has_active_context": True,
        }

        # Asegurar valores por defecto para campos críticos
        # Si report no trae status, sobrescribimos el IDLE de base_status con ACTIVE
        if "status" not in report:
            result["status"] = "ACTIVE"

        # Si report no trae system_health, sobrescribimos el UNKNOWN de base_status con HEALTHY
        if "system_health" not in report:
            result["system_health"] = "HEALTHY"

        result.setdefault("message", "Telemetry context active")

        return result

    except Exception as e:
        logger.error(f"Error retrieving telemetry report: {e}", exc_info=True)
        return {
            **base_status,
            "status": "ERROR",
            "message": f"Failed to retrieve telemetry: {e}",
            "system_health": "DEGRADED",
            "error": str(e),
            "error_type": type(e).__name__,
        }


# === Funciones de Utilidad Pública ===

def get_supported_file_types() -> List[str]:
    """
    Retorna lista de tipos de archivo soportados para diagnóstico.

    Returns:
        Lista de strings con tipos válidos

    Examples:
        >>> get_supported_file_types()
        ['apus', 'insumos', 'presupuesto']
    """
    return FileType.values()


def is_valid_file_type(file_type: Any) -> bool:
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
    if not isinstance(file_type, (str, FileType)):
        return False

    try:
        _normalize_file_type(file_type)
        return True
    except (UnsupportedFileTypeError, ValueError, TypeError):
        return False


def get_supported_delimiters() -> List[str]:
    """
    Retorna lista de delimitadores CSV soportados.

    Returns:
        Lista de delimitadores válidos
    """
    return sorted(VALID_DELIMITERS)


def get_supported_encodings() -> List[str]:
    """
    Retorna lista de encodings comúnmente soportados.

    Returns:
        Lista de encodings comunes
    """
    return sorted(SUPPORTED_ENCODINGS)


def validate_file_for_processing(
    file_path: Union[str, Path],
    *,
    check_extension: bool = True,
    max_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Pre-valida un archivo antes de procesarlo.

    Útil para validar archivos antes de diagnose_file o clean_file.

    Args:
        file_path: Ruta del archivo a validar
        check_extension: Si validar la extensión
        max_size: Tamaño máximo permitido en bytes

    Returns:
        Diccionario con resultado de validación:
        - valid (bool): True si el archivo es válido
        - path (str): Ruta normalizada
        - size (int): Tamaño en bytes (si válido)
        - extension (str): Extensión del archivo (si válido)
        - errors (list): Lista de errores (si no válido)

    Examples:
        >>> result = validate_file_for_processing("data/test.csv")
        >>> if result["valid"]:
        ...     print(f"Size: {result['size']} bytes")
    """
    errors: List[str] = []

    try:
        path = _normalize_path(file_path)
    except ValueError as e:
        return {
            "valid": False,
            "path": str(file_path),
            "errors": [str(e)],
        }

    # Validar existencia
    if not path.exists():
        errors.append(f"File does not exist: {path}")
    elif not path.is_file():
        errors.append(f"Path is not a file: {path}")
    elif not os.access(path, os.R_OK):
        errors.append(f"File is not readable: {path}")
    else:
        # Validar extensión
        if check_extension:
            extension = path.suffix.lower()
            if extension not in VALID_EXTENSIONS:
                errors.append(
                    f"Invalid extension '{extension}'. "
                    f"Valid: {sorted(VALID_EXTENSIONS)}"
                )

        # Validar tamaño
        try:
            size = path.stat().st_size
            effective_max = (
                max_size if max_size is not None
                else MAX_FILE_SIZE_BYTES
            )

            if size > effective_max:
                max_mb = effective_max / (1024 * 1024)
                file_mb = size / (1024 * 1024)
                errors.append(
                    f"File too large: {file_mb:.2f} MB (max: {max_mb:.2f} MB)"
                )
            elif size == 0:
                errors.append("File is empty")

        except OSError as e:
            errors.append(f"Cannot read file stats: {e}")
            size = 0

    if errors:
        return {
            "valid": False,
            "path": str(path),
            "errors": errors,
        }

    return {
        "valid": True,
        "path": str(path),
        "size": size,
        "extension": path.suffix.lower(),
    }