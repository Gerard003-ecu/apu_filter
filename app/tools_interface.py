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
import sys
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

# Integración del Motor Financiero
from .financial_engine import FinancialConfig, FinancialEngine

logger = logging.getLogger(__name__)


# === Configuración y Constantes ===

MAX_FILE_SIZE_BYTES: Final[int] = 100 * 1024 * 1024  # 100 MB

SUPPORTED_ENCODINGS: Final[frozenset[str]] = frozenset(
    {
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "iso-8859-1",
        "cp1252",
        "ascii",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
    }
)

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
            value: String a convertir.

        Returns:
            FileType correspondiente.

        Raises:
            ValueError: Si el valor no es válido.
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

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
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

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
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
        file_type: Tipo de archivo.

    Returns:
        Clase diagnóstica correspondiente.

    Raises:
        UnsupportedFileTypeError: Si no hay diagnóstico registrado.
    """
    diagnostic_class = _DIAGNOSTIC_REGISTRY.get(file_type)

    if diagnostic_class is None:
        raise UnsupportedFileTypeError(
            f"No diagnostic registered for type: {file_type.value}",
            details={"file_type": file_type.value, "available_types": FileType.values()},
        )

    return diagnostic_class


# === Funciones de Validación de Rutas ===


def _validate_path_not_empty(file_path: Union[str, Path, None]) -> None:
    """
    Valida que la ruta no esté vacía o sea None.

    Args:
        file_path: Ruta a validar.

    Raises:
        ValueError: Si la ruta está vacía o es None.
        TypeError: Si el tipo no es str ni Path.
    """
    if file_path is None:
        raise ValueError("File path cannot be None")

    if isinstance(file_path, str):
        # Validar string vacío o solo espacios
        if not file_path or not file_path.strip():
            raise ValueError("File path cannot be empty or whitespace-only string")
    elif isinstance(file_path, Path):
        # Path vacío se representa como "." internamente
        path_str = str(file_path)
        if path_str in ("", ".") or not path_str.strip():
            raise ValueError("File path cannot be empty")
    else:
        # Separar TypeError de ValueError para mejor semántica
        raise TypeError(f"File path must be str or Path, got: {type(file_path).__name__}")


def _normalize_path(file_path: Union[str, Path]) -> Path:
    """
    Normaliza una ruta a objeto Path resuelto.

    Args:
        file_path: Ruta como string o Path.

    Returns:
        Path normalizado y expandido.

    Raises:
        ValueError: Si la ruta está vacía o es inválida.
    """
    _validate_path_not_empty(file_path)

    try:
        path = Path(file_path) if isinstance(file_path, str) else file_path

        # Expandir ~ a home directory
        path = path.expanduser()

        # Estrategia de resolución en cascada
        resolved_path: Optional[Path] = None

        try:
            # Caso 1: El archivo existe - resolver directamente
            if path.exists():
                resolved_path = path.resolve(strict=True)
            # Caso 2: El parent existe - resolver parent y añadir nombre
            elif path.parent and path.parent != path and path.parent.exists():
                resolved_path = path.parent.resolve(strict=True) / path.name
            # Caso 3: Resolver sin verificar existencia
            else:
                resolved_path = path.resolve(strict=False)
        except OSError:
            # Permisos u otros errores de OS
            resolved_path = None
        except RuntimeError:
            # Recursión excesiva en symlinks
            resolved_path = None

        # Usar ruta resuelta o la original expandida
        return resolved_path if resolved_path is not None else path

    except TypeError as e:
        raise ValueError(f"Invalid path type for '{file_path}': {e}") from e
    except ValueError as e:
        raise ValueError(f"Invalid path value '{file_path}': {e}") from e
    except OSError as e:
        raise ValueError(f"OS error processing path '{file_path}': {e}") from e


def _safe_stat(path: Path) -> bool:
    """Helper para verificar si stat() es accesible."""
    try:
        path.stat()
        return True
    except OSError:
        return False


def _validate_file_exists(path: Path) -> None:
    """
    Valida que un archivo exista y sea accesible para lectura.

    Args:
        path: Ruta del archivo.

    Raises:
        FileNotFoundDiagnosticError: Si el archivo no existe.
        FileValidationError: Si no es archivo o no es legible.
    """
    # Verificar existencia primero
    try:
        exists = path.exists()
    except OSError as e:
        raise FileValidationError(
            f"Cannot verify path existence: {path}",
            details={"path": str(path), "error": str(e)},
        )

    if not exists:
        # Calcular parent_exists de forma segura
        parent_exists = False
        try:
            if path.parent and path.parent != path:
                parent_exists = path.parent.exists()
        except OSError:
            pass  # No crítico, solo informativo

        raise FileNotFoundDiagnosticError(
            f"File not found: {path}",
            details={
                "path": str(path),
                "parent_exists": parent_exists,
                "parent_path": str(path.parent) if path.parent else None,
            },
        )

    # Verificar que sea archivo (no directorio/symlink roto)
    try:
        is_file = path.is_file()
    except OSError as e:
        raise FileValidationError(
            f"Cannot determine if path is file: {path}",
            details={"path": str(path), "error": str(e)},
        )

    if not is_file:
        raise FileValidationError(
            f"Path exists but is not a file: {path}",
            details={
                "path": str(path),
                "is_directory": path.is_dir(),
                "is_symlink": path.is_symlink(),
                "is_mount": path.is_mount() if hasattr(path, "is_mount") else None,
            },
        )

    # Verificar permisos de lectura
    if not os.access(path, os.R_OK):
        raise FileValidationError(
            f"File exists but is not readable (permission denied): {path}",
            details={
                "path": str(path),
                "file_mode": oct(path.stat().st_mode) if _safe_stat(path) else None,
            },
        )


def _validate_file_size(path: Path, max_size: int = MAX_FILE_SIZE_BYTES) -> Tuple[int, bool]:
    """
    Valida que el archivo no exceda el tamaño máximo.

    Args:
        path: Ruta del archivo.
        max_size: Tamaño máximo en bytes (debe ser > 0).

    Returns:
        Tupla (tamaño_en_bytes, is_empty).

    Raises:
        FileValidationError: Si excede el tamaño o max_size inválido.
        ValueError: Si max_size <= 0.
    """
    # Validar max_size primero
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got: {max_size}")

    try:
        stat_result = path.stat()
        file_size = stat_result.st_size
    except OSError as e:
        raise FileValidationError(
            f"Cannot determine file size: {path}",
            details={"path": str(path), "error": str(e)},
        ) from e

    is_empty = file_size == 0

    if is_empty:
        logger.warning(
            f"File is empty (0 bytes): {path}. Processing may produce unexpected results."
        )

    if file_size > max_size:
        # Cálculo seguro de MB (max_size ya validado > 0)
        max_mb = max_size / (1024 * 1024)
        file_mb = file_size / (1024 * 1024)
        raise FileValidationError(
            f"File size ({file_mb:.2f} MB) exceeds maximum ({max_mb:.2f} MB): {path}",
            details={
                "path": str(path),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_mb, 2),
                "max_size_bytes": max_size,
                "max_size_mb": round(max_mb, 2),
                "excess_bytes": file_size - max_size,
            },
        )

    return file_size, is_empty


def _validate_file_extension(
    path: Path, valid_extensions: frozenset[str] = VALID_EXTENSIONS
) -> str:
    """
    Valida que la extensión del archivo sea válida.

    Args:
        path: Ruta del archivo.
        valid_extensions: Conjunto de extensiones válidas.

    Returns:
        Extensión normalizada (lowercase).

    Raises:
        FileValidationError: Si la extensión no es válida.
    """
    extension = path.suffix.lower()

    if not extension:
        raise FileValidationError(
            f"File has no extension: {path}",
            details={"path": str(path), "valid_extensions": sorted(valid_extensions)},
        )

    if extension not in valid_extensions:
        raise FileValidationError(
            f"Invalid file extension '{extension}'. Expected: {sorted(valid_extensions)}",
            details={
                "path": str(path),
                "extension": extension,
                "valid_extensions": sorted(valid_extensions),
            },
        )

    return extension


# === Funciones de Validación de Parámetros CSV ===


def _validate_encoding(encoding: str) -> str:
    """
    Valida y normaliza un encoding usando codecs de Python.

    Args:
        encoding: Encoding a validar.

    Returns:
        Nombre canónico del encoding (normalizado por Python).

    Raises:
        ValueError: Si el encoding es inválido o desconocido.
    """
    if encoding is None:
        raise ValueError("Encoding cannot be None")

    if not isinstance(encoding, str):
        raise ValueError(f"Encoding must be string, got: {type(encoding).__name__}")

    encoding_stripped = encoding.strip()

    if not encoding_stripped:
        raise ValueError("Encoding cannot be empty or whitespace-only")

    encoding_normalized = encoding_stripped.lower()

    # Verificar que Python reconozca el encoding
    try:
        codec_info = codecs.lookup(encoding_normalized)
        # Usar el nombre canónico del codec para consistencia
        canonical_name = codec_info.name
    except LookupError as e:
        # Sugerir encodings similares si es posible
        suggestions = [
            enc
            for enc in SUPPORTED_ENCODINGS
            if encoding_normalized in enc or enc in encoding_normalized
        ]
        suggestion_msg = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown encoding: '{encoding}'.{suggestion_msg}") from e

    # Verificar que el codec pueda realmente decodificar
    try:
        codecs.decode(b"test", encoding=encoding_normalized, errors="strict")
    except Exception as e:
        raise ValueError(
            f"Encoding '{encoding}' is registered but cannot decode: {e}"
        ) from e

    # Advertir si no está en la lista de encodings comunes
    if (
        canonical_name not in SUPPORTED_ENCODINGS
        and encoding_normalized not in SUPPORTED_ENCODINGS
    ):
        logger.warning(
            f"Encoding '{encoding}' (canonical: '{canonical_name}') "
            f"is valid but not in recommended list: {sorted(SUPPORTED_ENCODINGS)}. "
            f"Proceeding with caution."
        )

    # Retornar nombre canónico para consistencia
    return canonical_name


def _validate_delimiter(delimiter: str) -> str:
    """
    Valida un delimitador CSV.

    Args:
        delimiter: Delimitador a validar.

    Returns:
        Delimitador validado.

    Raises:
        ValueError: Si el delimitador es inválido.
    """
    if delimiter is None:
        raise ValueError("Delimiter cannot be None")

    if not isinstance(delimiter, str):
        raise ValueError(f"Delimiter must be string, got: {type(delimiter).__name__}")

    if not delimiter:
        raise ValueError("Delimiter cannot be empty")

    if len(delimiter) != 1:
        raise ValueError(
            f"Delimiter must be single character, "
            f"got '{delimiter}' (length: {len(delimiter)})"
        )

    if delimiter not in VALID_DELIMITERS:
        valid_repr = ", ".join(repr(d) for d in sorted(VALID_DELIMITERS))
        raise ValueError(f"Invalid delimiter {repr(delimiter)}. Valid: {valid_repr}")

    return delimiter


def _validate_csv_parameters(delimiter: str, encoding: str) -> Tuple[str, str]:
    """
    Valida parámetros de procesamiento CSV.

    Args:
        delimiter: Delimitador a validar.
        encoding: Encoding a validar.

    Returns:
        Tupla (delimiter_validado, encoding_validado).

    Raises:
        ValueError: Si algún parámetro es inválido.
    """
    validated_delimiter = _validate_delimiter(delimiter)
    validated_encoding = _validate_encoding(encoding)

    return validated_delimiter, validated_encoding


# === Funciones de Validación de Tipo de Archivo ===


def _normalize_file_type(file_type: Union[str, FileType]) -> FileType:
    """
    Normaliza el tipo de archivo a enum FileType.

    Args:
        file_type: Tipo como string o FileType.

    Returns:
        FileType enum.

    Raises:
        UnsupportedFileTypeError: Si el tipo no es válido.
    """
    if isinstance(file_type, FileType):
        return file_type

    if not isinstance(file_type, str):
        raise UnsupportedFileTypeError(
            f"file_type must be string or FileType, got: {type(file_type).__name__}",
            details={"received_type": type(file_type).__name__},
        )

    try:
        return FileType.from_string(file_type)
    except ValueError:
        valid_types = ", ".join(FileType.values())
        raise UnsupportedFileTypeError(
            f"Unknown file type: '{file_type}'. Valid: {valid_types}",
            details={"received": file_type, "valid_types": FileType.values()},
        )


# === Funciones de Respuesta ===


def _create_error_response(error: Union[str, Exception], **extras: Any) -> Dict[str, Any]:
    """
    Crea una respuesta de error estandarizada.

    Args:
        error: Mensaje o excepción de error.
        **extras: Campos adicionales para incluir.

    Returns:
        Diccionario con estructura de error consistente.
    """
    error_message = str(error)
    error_type = type(error).__name__ if isinstance(error, Exception) else "Error"

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


def _create_success_response(data: Dict[str, Any], **extras: Any) -> Dict[str, Any]:
    """
    Crea una respuesta exitosa estandarizada.

    Args:
        data: Datos de la respuesta.
        **extras: Campos adicionales para incluir.

    Returns:
        Diccionario con estructura de éxito consistente.

    Raises:
        ValueError: Si data contiene claves reservadas.
    """
    RESERVED_KEYS: frozenset[str] = frozenset({"success", "error", "error_type"})

    if not isinstance(data, dict):
        logger.warning(
            f"Expected dict for data, got {type(data).__name__}. Wrapping in 'result' key."
        )
        data = {"result": data}

    # Detectar colisiones con claves reservadas
    conflicting_keys = set(data.keys()) & RESERVED_KEYS
    if conflicting_keys:
        logger.warning(
            f"Data contains reserved keys {conflicting_keys}. "
            f"These will be prefixed with 'data_' to avoid conflicts."
        )
        data = {(f"data_{k}" if k in RESERVED_KEYS else k): v for k, v in data.items()}

    # Detectar colisiones entre data y extras
    extras_conflicting = set(data.keys()) & set(extras.keys())
    if extras_conflicting:
        logger.debug(f"Keys {extras_conflicting} in data will be overwritten by extras")

    response: Dict[str, Any] = {"success": True}
    response.update(data)

    for key, value in extras.items():
        if value is not None:
            response[key] = value

    return response


# === Funciones de Generación y Validación de Rutas de Salida ===


def _generate_output_path(input_path: Path, suffix: str = "_clean") -> Path:
    """
    Genera ruta de salida basada en la de entrada.

    Args:
        input_path: Ruta del archivo de entrada.
        suffix: Sufijo a añadir antes de la extensión.

    Returns:
        Path para el archivo de salida.
    """
    if not suffix:
        suffix = "_clean"

    new_name = f"{input_path.stem}{suffix}{input_path.suffix}"
    return input_path.with_name(new_name)


def _validate_output_path(input_path: Path, output_path: Path, overwrite: bool) -> None:
    """
    Valida que la ruta de salida sea válida.

    Args:
        input_path: Ruta de entrada.
        output_path: Ruta de salida.
        overwrite: Si se permite sobrescribir.

    Raises:
        ValueError: Si la ruta de salida es igual a la de entrada.
        FileExistsError: Si el archivo existe y overwrite=False.
        PermissionError: Si no hay permisos de escritura.
    """

    def _normalize_for_comparison(p: Path) -> str:
        """Normaliza ruta para comparación cross-platform."""
        try:
            resolved = p.resolve() if p.exists() else p.absolute()
            path_str = str(resolved)
            # En Windows, normalizar case
            if sys.platform == "win32":
                path_str = path_str.lower()
            return path_str
        except OSError:
            return str(p)

    # Comparar rutas de forma segura y cross-platform
    try:
        input_normalized = _normalize_for_comparison(input_path)
        output_normalized = _normalize_for_comparison(output_path)

        if input_normalized == output_normalized:
            raise ValueError(
                f"Output path cannot be the same as input path: '{input_path}'. "
                f"Use a different filename or directory."
            )
    except OSError as e:
        logger.warning(
            f"Could not fully resolve paths for comparison: {e}. "
            f"Proceeding with basic comparison."
        )
        # Comparación básica como fallback
        if str(input_path) == str(output_path):
            raise ValueError("Output path appears to be the same as input path.")

    # Verificar sobrescritura
    if output_path.exists():
        if not output_path.is_file():
            raise ValueError(f"Output path exists but is not a file: {output_path}")

        if not overwrite:
            raise FileExistsError(
                f"Output file already exists and overwrite=False: {output_path}. "
                f"Set overwrite=True or choose a different output path."
            )

        if not os.access(output_path, os.W_OK):
            raise PermissionError(f"Output file exists but is not writable: {output_path}")


def _ensure_output_directory(output_path: Path) -> None:
    """
    Asegura que el directorio de salida exista y sea escribible.

    Args:
        output_path: Ruta del archivo de salida.

    Raises:
        OSError: Si no se puede crear el directorio.
        PermissionError: Si no hay permisos de escritura.
    """
    output_dir = output_path.parent

    if not output_dir.exists():
        logger.info(f"Creating output directory: {output_dir}")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create output directory '{output_dir}': {e}") from e

    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {output_dir}")


# === Funciones de Extracción de Resultados ===


def _extract_diagnostic_result(diagnostic: Any) -> Dict[str, Any]:
    """
    Extrae resultados de un objeto diagnóstico de forma segura.

    Args:
        diagnostic: Objeto diagnóstico.

    Returns:
        Diccionario con resultados (siempre incluye 'diagnostic_completed').
    """
    base_result: Dict[str, Any] = {
        "diagnostic_completed": True,
        "diagnostic_class": type(diagnostic).__name__,
    }

    if not hasattr(diagnostic, "to_dict"):
        logger.warning(
            f"Diagnostic {type(diagnostic).__name__} lacks to_dict() method. "
            f"Returning minimal result."
        )
        return base_result

    if not callable(diagnostic.to_dict):
        logger.warning(
            f"Diagnostic {type(diagnostic).__name__}.to_dict is not callable. "
            f"Returning minimal result."
        )
        return {**base_result, "to_dict_not_callable": True}

    try:
        result = diagnostic.to_dict()

        if result is None:
            logger.warning(f"Diagnostic {type(diagnostic).__name__}.to_dict() returned None")
            return {**base_result, "to_dict_returned_none": True}

        if not isinstance(result, dict):
            logger.warning(
                f"to_dict() returned {type(result).__name__}, expected dict. "
                f"Wrapping result."
            )
            return {
                **base_result,
                "raw_result": result,
                "result_type": type(result).__name__,
            }

        # Validar que tenga al menos algún contenido útil
        if not result:
            logger.warning(
                f"Diagnostic {type(diagnostic).__name__}.to_dict() returned empty dict"
            )
            return {**base_result, "empty_result": True}

        # Merge con base_result, pero sin sobrescribir datos del diagnóstico
        return {**base_result, **result}

    except Exception as e:
        logger.warning(
            f"Error calling {type(diagnostic).__name__}.to_dict(): {e}", exc_info=True
        )
        return {
            **base_result,
            "diagnostic_completed": False,
            "to_dict_error": str(e),
            "to_dict_error_type": type(e).__name__,
        }


def _extract_cleaning_stats(stats: Any) -> Dict[str, Any]:
    """
    Extrae estadísticas de limpieza de forma segura.

    Args:
        stats: Objeto de estadísticas.

    Returns:
        Diccionario con estadísticas.
    """
    if stats is None:
        return {"cleaning_completed": True}

    if isinstance(stats, dict):
        return stats

    if hasattr(stats, "to_dict") and callable(stats.to_dict):
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
        - success (bool): Si el diagnóstico fue exitoso.
        - Si success=True: Datos del diagnóstico específico.
        - Si success=False: error, error_type, error_details.

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

        effective_max = max_file_size if max_file_size is not None else MAX_FILE_SIZE_BYTES
        file_size, is_empty = _validate_file_size(path, effective_max)

        # Warning adicional para archivos vacíos
        if is_empty:
            logger.warning(f"Diagnosing empty file: {path}. Results may be limited.")

        # Obtener clase diagnóstica
        diagnostic_class = _get_diagnostic_class(normalized_type)

        # Ejecutar diagnóstico con validación de instanciación
        logger.info(
            f"Starting {normalized_type.value} diagnosis for: {path} ({file_size} bytes)"
        )

        try:
            diagnostic = diagnostic_class(str(path))
        except TypeError as e:
            raise DiagnosticError(
                f"Failed to instantiate {diagnostic_class.__name__}: {e}",
                details={
                    "diagnostic_class": diagnostic_class.__name__,
                    "file_path": str(path),
                    "error": str(e),
                },
            ) from e

        # Verificar que diagnose() existe y es callable
        if not hasattr(diagnostic, "diagnose") or not callable(diagnostic.diagnose):
            raise DiagnosticError(
                f"Diagnostic class {diagnostic_class.__name__} "
                f"does not have a callable diagnose() method",
                details={"diagnostic_class": diagnostic_class.__name__},
            )

        try:
            diagnostic.diagnose()
        except Exception as e:
            raise DiagnosticError(
                f"Diagnosis execution failed: {e}",
                details={
                    "diagnostic_class": diagnostic_class.__name__,
                    "file_path": str(path),
                    "error_type": type(e).__name__,
                },
            ) from e

        # Extraer resultados
        result_data = _extract_diagnostic_result(diagnostic)

        logger.info(f"Diagnosis completed successfully for: {path}")

        return _create_success_response(
            result_data,
            file_type=normalized_type.value,
            file_path=str(path),
            file_size_bytes=file_size,
        )

    except DiagnosticError as e:
        logger.warning(f"Diagnostic error: {e}")
        return _create_error_response(e, error_category="diagnostic")

    except (FileNotFoundDiagnosticError, UnsupportedFileTypeError, FileValidationError) as e:
        logger.warning(f"Validation error in diagnose_file: {e}")
        return _create_error_response(e, error_category="validation")

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
        logger.error(f"Unexpected error in diagnosis for {file_type}: {e}", exc_info=True)
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
        - success (bool): Si la limpieza fue exitosa.
        - Si success=True: output_path, input_path, estadísticas.
        - Si success=False: error, error_type, error_details.

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

        effective_max = max_file_size if max_file_size is not None else MAX_FILE_SIZE_BYTES
        input_size, _ = _validate_file_size(input_p, effective_max)

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
            f"encoding='{validated_encoding}', "
            f"input_size={input_size} bytes)"
        )

        try:
            cleaner = CSVCleaner(
                input_path=str(input_p),
                output_path=str(output_p),
                delimiter=validated_delimiter,
                encoding=validated_encoding,
                overwrite=overwrite,
            )
        except TypeError as e:
            raise CleaningError(
                f"Failed to instantiate CSVCleaner: {e}", details={"error": str(e)}
            ) from e

        try:
            stats = cleaner.clean()
        except Exception as e:
            raise CleaningError(
                f"Cleaning execution failed: {e}",
                details={"error_type": type(e).__name__, "error": str(e)},
            ) from e

        # Verificar que el archivo de salida fue creado
        if not output_p.exists():
            raise CleaningError(
                f"Cleaning completed but output file was not created: {output_p}",
                details={"input_path": str(input_p), "output_path": str(output_p)},
            )

        # Obtener estadísticas del archivo de salida
        try:
            output_size = output_p.stat().st_size
        except OSError as e:
            logger.warning(f"Could not get output file size: {e}")
            output_size = None

        # Extraer estadísticas
        result_data = _extract_cleaning_stats(stats)

        # Añadir métricas de archivos
        result_data["input_size_bytes"] = input_size
        if output_size is not None:
            result_data["output_size_bytes"] = output_size
            if input_size > 0:
                result_data["size_reduction_pct"] = round(
                    (1 - output_size / input_size) * 100, 2
                )

        logger.info(
            f"CSV cleaning completed successfully: {output_p} "
            f"(output_size={output_size} bytes)"
        )

        return _create_success_response(
            result_data, output_path=str(output_p), input_path=str(input_p)
        )

    except CleaningError as e:
        logger.error(f"Cleaning error: {e}")
        return _create_error_response(e, error_category="cleaning")

    except (FileNotFoundDiagnosticError, FileValidationError) as e:
        logger.warning(f"File validation error in clean_file: {e}")
        return _create_error_response(e, error_category="validation")

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


def get_telemetry_status(telemetry_context: Optional[Any] = None) -> Dict[str, Any]:
    """
    Obtiene el estado actual del sistema (Vector de Estado).

    Args:
        telemetry_context: Objeto con método get_business_report().
            Si None, retorna estado genérico.

    Returns:
        Diccionario con estado del sistema:
        - status: 'ACTIVE', 'IDLE', o 'ERROR'.
        - message: Descripción del estado.
        - system_health: 'HEALTHY', 'DEGRADED', o 'UNKNOWN'.
        - has_active_context: Si hay contexto activo.

    Examples:
        >>> status = get_telemetry_status()
        >>> print(status['status'])  # 'IDLE'
    """
    # Estado base para contexto ausente
    IDLE_STATUS: Dict[str, Any] = {
        "status": "IDLE",
        "message": "No active processing context",
        "system_health": "UNKNOWN",
        "has_active_context": False,
    }

    if telemetry_context is None:
        logger.debug("Telemetry status requested without active context")
        return IDLE_STATUS.copy()

    context_type = type(telemetry_context).__name__

    # Verificar que el método exista
    if not hasattr(telemetry_context, "get_business_report"):
        logger.warning(
            f"Telemetry context ({context_type}) missing 'get_business_report' method"
        )
        return {
            "status": "ERROR",
            "message": f"Invalid telemetry context ({context_type}): missing get_business_report method",
            "system_health": "DEGRADED",
            "has_active_context": False,
            "error_type": "MissingMethodError",
        }

    report_method = getattr(telemetry_context, "get_business_report")

    if not callable(report_method):
        logger.warning(f"get_business_report on {context_type} is not callable")
        return {
            "status": "ERROR",
            "message": "get_business_report is not callable",
            "system_health": "DEGRADED",
            "has_active_context": False,
            "error_type": "NotCallableError",
        }

    try:
        report = report_method()

        # Normalizar el reporte
        if report is None:
            logger.warning(f"Telemetry report from {context_type} returned None")
            report = {}
        elif not isinstance(report, dict):
            logger.warning(
                f"Telemetry report is {type(report).__name__}, expected dict. Converting."
            )
            report = {"raw_report": report, "raw_report_type": type(report).__name__}

        # Construir respuesta con prioridad clara:
        # 1. Valores del reporte tienen prioridad
        # 2. Valores por defecto solo si no existen en reporte
        result: Dict[str, Any] = {
            "has_active_context": True,
        }

        # Añadir datos del reporte primero
        result.update(report)

        # Establecer defaults SOLO si no vienen en el reporte
        result.setdefault("status", "ACTIVE")
        result.setdefault("message", "Telemetry context active")
        result.setdefault("system_health", "HEALTHY")

        # Validar que status tenga valor válido
        valid_statuses = {"ACTIVE", "IDLE", "ERROR", "PENDING", "PROCESSING"}
        if result.get("status") not in valid_statuses:
            logger.warning(
                f"Unknown status '{result.get('status')}' in telemetry report. "
                f"Valid values: {valid_statuses}"
            )

        return result

    except Exception as e:
        logger.error(
            f"Error retrieving telemetry report from {context_type}: {e}", exc_info=True
        )
        return {
            "status": "ERROR",
            "message": f"Failed to retrieve telemetry: {e}",
            "system_health": "DEGRADED",
            "has_active_context": True,  # El contexto existe, solo falló
            "error": str(e),
            "error_type": type(e).__name__,
            "context_type": context_type,
        }


# === Funciones de Utilidad Pública ===


def get_supported_file_types() -> List[str]:
    """
    Retorna lista de tipos de archivo soportados para diagnóstico.

    Returns:
        Lista de strings con tipos válidos.

    Examples:
        >>> get_supported_file_types()
        ['apus', 'insumos', 'presupuesto']
    """
    return FileType.values()


def is_valid_file_type(file_type: Any) -> bool:
    """
    Verifica si un tipo de archivo es válido.

    Args:
        file_type: Tipo a verificar.

    Returns:
        True si es válido, False en caso contrario.

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
        Lista de delimitadores válidos.
    """
    return sorted(VALID_DELIMITERS)


def get_supported_encodings() -> List[str]:
    """
    Retorna lista de encodings comúnmente soportados.

    Returns:
        Lista de encodings comunes.
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
        file_path: Ruta del archivo a validar.
        check_extension: Si validar la extensión.
        max_size: Tamaño máximo permitido en bytes.

    Returns:
        Diccionario con resultado de validación:
        - valid (bool): True si el archivo es válido.
        - path (str): Ruta normalizada.
        - size (int): Tamaño en bytes (si válido).
        - extension (str): Extensión del archivo (si válido).
        - errors (list): Lista de errores (si no válido).

    Examples:
        >>> result = validate_file_for_processing("data/test.csv")
        >>> if result["valid"]:
        ...     print(f"Size: {result['size']} bytes")
    """
    errors: List[str] = []
    size: Optional[int] = None
    extension: Optional[str] = None

    # Normalizar ruta
    try:
        path = _normalize_path(file_path)
    except (ValueError, TypeError) as e:
        return {
            "valid": False,
            "path": str(file_path) if file_path is not None else None,
            "errors": [f"Invalid path: {e}"],
        }

    # Validar existencia y accesibilidad
    try:
        exists = path.exists()
    except OSError as e:
        errors.append(f"Cannot check file existence: {e}")
        exists = False

    if not exists:
        errors.append(f"File does not exist: {path}")
    else:
        try:
            is_file = path.is_file()
        except OSError as e:
            errors.append(f"Cannot determine if path is file: {e}")
            is_file = False

        if not is_file:
            errors.append(f"Path is not a file: {path}")
        else:
            # Verificar permisos
            if not os.access(path, os.R_OK):
                errors.append(f"File is not readable: {path}")
            else:
                # Solo validar extensión y tamaño si el archivo es accesible

                # Validar extensión
                extension = path.suffix.lower()
                if check_extension and extension not in VALID_EXTENSIONS:
                    errors.append(
                        f"Invalid extension '{extension}'. Valid: {sorted(VALID_EXTENSIONS)}"
                    )

                # Validar tamaño
                try:
                    size = path.stat().st_size
                    effective_max = max_size if max_size is not None else MAX_FILE_SIZE_BYTES

                    if effective_max <= 0:
                        errors.append(f"Invalid max_size: {effective_max}")
                    elif size > effective_max:
                        max_mb = effective_max / (1024 * 1024)
                        file_mb = size / (1024 * 1024)
                        errors.append(
                            f"File too large: {file_mb:.2f} MB (max: {max_mb:.2f} MB)"
                        )
                    elif size == 0:
                        # Warning, no error - archivos vacíos pueden ser válidos
                        logger.warning(f"File is empty: {path}")

                except OSError as e:
                    errors.append(f"Cannot read file stats: {e}")

    # Construir respuesta
    if errors:
        response: Dict[str, Any] = {
            "valid": False,
            "path": str(path),
            "errors": errors,
        }
        # Incluir información parcial si está disponible
        if size is not None:
            response["size"] = size
        if extension is not None:
            response["extension"] = extension
        return response

    # Éxito - size y extension garantizados no-None aquí
    assert size is not None, "size should be defined at this point"
    assert extension is not None, "extension should be defined at this point"

    return {
        "valid": True,
        "path": str(path),
        "size": size,
        "extension": extension,
        "size_mb": round(size / (1024 * 1024), 2),
    }


def analyze_financial_viability(
    amount: float, std_dev: float, time_years: int
) -> Dict[str, Any]:
    """
    Ejecuta un análisis de viabilidad financiera para un monto y riesgo dados.

    Args:
        amount (float): Monto de la inversión inicial.
        std_dev (float): Desviación estándar del costo.
        time_years (int): Horizonte del proyecto en años.

    Returns:
        Dict[str, Any]: Reporte financiero simplificado.
    """
    try:
        # Validación de parámetros
        if amount <= 0:
            raise ValueError("El monto debe ser un número positivo.")
        if std_dev < 0:
            raise ValueError("La desviación estándar no puede ser negativa.")
        if not 1 <= time_years <= 50:
            raise ValueError("El tiempo en años debe estar entre 1 y 50.")

        # Configuración estándar
        config = FinancialConfig(project_life_years=time_years)
        engine = FinancialEngine(config)

        # Simular flujos de caja (simplificación: se asume un retorno anual del 20%)
        # En un caso real, esto provendría de un modelo más complejo.
        annual_return = amount * 0.20
        cash_flows = [annual_return] * time_years

        # Volatilidad (simplificación: se asume un 30%)
        volatility = 0.30

        # Ejecutar análisis
        analysis = engine.analyze_project(
            initial_investment=amount,
            expected_cash_flows=cash_flows,
            cost_std_dev=std_dev,
            project_volatility=volatility,
        )

        # Construir respuesta simplificada
        report = {
            "success": True,
            "parameters": {
                "initial_investment": amount,
                "cost_std_dev": std_dev,
                "project_life_years": time_years,
            },
            "results": {
                "wacc": analysis.get("wacc"),
                "npv": analysis.get("npv"),
                "total_value_with_option": analysis.get("total_value"),
                "contingency_recommended": analysis.get("contingency", {}).get(
                    "recommended"
                ),
                "recommendation": analysis.get("performance", {}).get("recommendation"),
            },
        }
        return report

    except Exception as e:
        logger.error(f"Error en análisis financiero: {e}", exc_info=True)
        return _create_error_response(e, error_category="financial_analysis")
