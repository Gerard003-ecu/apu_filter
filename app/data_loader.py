# app/data_loader.py
"""
Módulo robusto para carga de datos desde diferentes formatos (CSV, Excel, PDF).
Incluye validaciones exhaustivas, manejo de errores, métricas y logging detallado.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Telemetry integration
try:
    from .telemetry import TelemetryContext
except ImportError:
    TelemetryContext = Any  # Fallback for typing if circular import

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES DE CONFIGURACIÓN
# ============================================================================
MAX_FILE_SIZE_MB = 500  # Tamaño máximo de archivo en MB
MAX_ROWS_WARNING = 1_000_000  # Advertencia si se cargan más de 1M filas
MAX_COLS_WARNING = 1000  # Advertencia si hay más de 1000 columnas
MIN_ROWS_WARNING = 1  # Advertencia si hay muy pocas filas

# Encodings a intentar en orden de prioridad
DEFAULT_ENCODINGS = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-16"]

# Delimitadores comunes para CSV
COMMON_DELIMITERS = [";", ",", "\t", "|"]

# Extensiones soportadas por tipo
SUPPORTED_CSV_EXTENSIONS = {".csv", ".txt", ".tsv"}
SUPPORTED_EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".xlsb"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}

ALL_SUPPORTED_EXTENSIONS = (
    SUPPORTED_CSV_EXTENSIONS | SUPPORTED_EXCEL_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS
)

# Configuración de validación
MAX_DUPLICATE_COLUMNS_ALLOWED = 0  # No permitir columnas duplicadas
MAX_NULL_PERCENTAGE_WARNING = 50  # Advertir si más del 50% de datos son nulos
MIN_VALID_ROWS_PERCENTAGE = 1  # Al menos 1% de filas deben ser válidas
MAX_PDF_PAGES = 500  # Límite de páginas a procesar


# ============================================================================
# ENUMS Y DATACLASSES
# ============================================================================
class FileFormat(Enum):
    """Formatos de archivo soportados"""

    CSV = "CSV"
    EXCEL = "EXCEL"
    PDF = "PDF"
    UNKNOWN = "UNKNOWN"


class LoadStatus(Enum):
    """Estados de carga de archivos"""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    EMPTY = "EMPTY"


@dataclass
class FileMetadata:
    """Metadatos de un archivo"""

    path: Path
    size_bytes: int
    size_mb: float
    format: FileFormat
    exists: bool
    readable: bool
    modified_time: Optional[datetime] = None
    is_symlink: bool = False
    error: Optional[str] = None

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "FileMetadata":
        """
        Crea metadatos desde una ruta de archivo.

        Args:
            path: Ruta al archivo (string o Path)

        Returns:
            FileMetadata: Objeto con los metadatos del archivo
        """
        # Normalizar path de forma segura
        try:
            if isinstance(path, str):
                path = Path(path)
            path = path.resolve()
        except (TypeError, ValueError, OSError) as e:
            logger.error(f"Ruta inválida: {path} - {e}")
            return cls(
                path=Path(str(path)) if path else Path("."),
                size_bytes=0,
                size_mb=0.0,
                format=FileFormat.UNKNOWN,
                exists=False,
                readable=False,
                error=f"Ruta inválida: {e}",
            )

        # Verificar existencia de forma segura
        try:
            exists = path.exists()
        except (OSError, PermissionError) as e:
            logger.warning(f"No se puede verificar existencia de {path}: {e}")
            exists = False

        # Verificar si es symlink
        is_symlink = False
        try:
            is_symlink = path.is_symlink()
            # Verificar symlink roto
            if is_symlink and not path.exists():
                logger.warning(f"Symlink roto detectado: {path}")
                return cls(
                    path=path,
                    size_bytes=0,
                    size_mb=0.0,
                    format=FileFormat.UNKNOWN,
                    exists=False,
                    readable=False,
                    is_symlink=True,
                    error="Symlink roto",
                )
        except (OSError, PermissionError):
            pass

        # Obtener tamaño y tiempo de modificación
        size_bytes = 0
        modified_time = None
        readable = False

        if exists:
            try:
                stat_info = path.stat()
                size_bytes = stat_info.st_size
                modified_time = datetime.fromtimestamp(stat_info.st_mtime)
            except (OSError, PermissionError, ValueError) as e:
                logger.warning(f"No se pudo obtener stat de {path}: {e}")
            except OverflowError:
                # Tiempo de modificación fuera de rango
                logger.warning(f"Tiempo de modificación inválido para {path}")

            # Verificar si es archivo y es legible
            try:
                readable = path.is_file()
                if readable:
                    # Verificar permisos de lectura intentando abrir el archivo
                    try:
                        with open(path, 'rb') as f:
                            f.read(1)  # Leer 1 byte para verificar
                    except (IOError, PermissionError):
                        readable = False
            except (OSError, PermissionError):
                readable = False

        size_mb = size_bytes / (1024 * 1024)

        # Determinar formato basado en extensión
        try:
            extension = path.suffix.lower() if path.suffix else ""
        except (AttributeError, TypeError):
            extension = ""

        if extension in SUPPORTED_CSV_EXTENSIONS:
            file_format = FileFormat.CSV
        elif extension in SUPPORTED_EXCEL_EXTENSIONS:
            file_format = FileFormat.EXCEL
        elif extension in SUPPORTED_PDF_EXTENSIONS:
            file_format = FileFormat.PDF
        else:
            file_format = FileFormat.UNKNOWN

        return cls(
            path=path,
            size_bytes=size_bytes,
            size_mb=size_mb,
            format=file_format,
            exists=exists,
            readable=readable,
            modified_time=modified_time,
            is_symlink=is_symlink,
        )


@dataclass
class DataQualityMetrics:
    """Métricas de calidad de datos cargados"""

    total_rows: int = 0
    total_columns: int = 0
    null_cells: int = 0
    null_percentage: float = 0.0
    duplicate_columns: List[str] = field(default_factory=list)
    empty_columns: List[str] = field(default_factory=list)
    columns_with_all_nulls: List[str] = field(default_factory=list)
    data_types: Dict[str, str] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def add_warning(self, warning: str) -> None:
        """Añade una advertencia a las métricas"""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte las métricas a diccionario"""
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "null_cells": self.null_cells,
            "null_percentage": round(self.null_percentage, 2),
            "duplicate_columns": self.duplicate_columns,
            "empty_columns": self.empty_columns,
            "columns_with_all_nulls": self.columns_with_all_nulls,
            "data_types": self.data_types,
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "warnings": self.warnings,
        }


@dataclass
class LoadResult:
    """Resultado de una operación de carga"""

    status: LoadStatus
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame], None]
    file_metadata: Optional[FileMetadata]
    quality_metrics: Optional[DataQualityMetrics]
    load_time_seconds: float
    encoding_used: Optional[str] = None
    delimiter_used: Optional[str] = None
    sheets_loaded: Optional[List[str]] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el resultado a diccionario de forma segura.

        Returns:
            Dict con información del resultado de carga
        """
        result = {
            "status": self.status.value if self.status else "UNKNOWN",
            "load_time_seconds": round(self.load_time_seconds, 3) if self.load_time_seconds else 0,
            "encoding_used": self.encoding_used,
            "delimiter_used": self.delimiter_used,
            "sheets_loaded": self.sheets_loaded,
            "error_message": self.error_message,
            "warnings": self.warnings if self.warnings else [],
        }

        # Manejar file_metadata de forma segura
        if self.file_metadata is not None:
            result["file_path"] = str(self.file_metadata.path)
            result["file_size_mb"] = round(self.file_metadata.size_mb, 2)
            result["format"] = self.file_metadata.format.value if self.file_metadata.format else "UNKNOWN"
        else:
            result["file_path"] = None
            result["file_size_mb"] = 0
            result["format"] = "UNKNOWN"

        # Manejar quality_metrics de forma segura
        if self.quality_metrics is not None:
            try:
                result["quality_metrics"] = self.quality_metrics.to_dict()
            except Exception as e:
                logger.warning(f"Error al serializar quality_metrics: {e}")
                result["quality_metrics"] = {"error": str(e)}
        else:
            result["quality_metrics"] = None

        # Información sobre los datos
        if self.data is None:
            result["has_data"] = False
            result["data_type"] = None
            result["data_shape"] = None
        elif isinstance(self.data, pd.DataFrame):
            result["has_data"] = not self.data.empty
            result["data_type"] = "DataFrame"
            result["data_shape"] = {"rows": len(self.data), "columns": len(self.data.columns)}
        elif isinstance(self.data, dict):
            result["has_data"] = any(
                isinstance(v, pd.DataFrame) and not v.empty
                for v in self.data.values()
            )
            result["data_type"] = "Dict[str, DataFrame]"
            result["data_shape"] = {
                sheet: {"rows": len(df), "columns": len(df.columns)}
                for sheet, df in self.data.items()
                if isinstance(df, pd.DataFrame)
            }
        else:
            result["has_data"] = self.data is not None
            result["data_type"] = type(self.data).__name__
            result["data_shape"] = None

        return result

    def add_warning(self, warning: str) -> None:
        """Agrega una advertencia al resultado."""
        if warning and warning not in self.warnings:
            self.warnings.append(warning)


# Tipos de uso común
PathType = Union[str, Path]
DataFrameOrDict = Union[pd.DataFrame, Dict[str, pd.DataFrame]]


# ============================================================================
# FUNCIONES AUXILIARES DE VALIDACIÓN
# ============================================================================
def _validate_file_path(path: PathType) -> Path:
    """
    Valida y normaliza una ruta de archivo.

    Args:
        path: Ruta como string o Path

    Returns:
        Path: Ruta normalizada y validada

    Raises:
        ValueError: Si la ruta es inválida o vacía
        FileNotFoundError: Si el archivo no existe
        PermissionError: Si no hay permisos de lectura
        IsADirectoryError: Si la ruta apunta a un directorio
    """
    # Validar que path no sea None o vacío
    if path is None:
        raise ValueError("La ruta del archivo no puede ser None")

    if isinstance(path, str):
        path = path.strip()
        if not path:
            raise ValueError("La ruta del archivo no puede estar vacía")

    # Convertir a Path y resolver
    try:
        path_obj = Path(path)
    except (TypeError, ValueError) as e:
        raise ValueError(f"No se puede crear Path desde '{path}': {e}")

    # Verificar caracteres problemáticos (principalmente en Windows)
    problematic_chars = ['<', '>', ':', '"', '|', '?', '*']
    path_str = str(path)
    for char in problematic_chars:
        if char in path_str and char != ':':  # ':' es válido en Windows para unidades
            # Solo advertir, no fallar (puede ser válido en algunos sistemas)
            logger.warning(f"La ruta contiene carácter potencialmente problemático: '{char}'")

    # Resolver ruta (convertir a absoluta, resolver symlinks)
    try:
        resolved_path = path_obj.resolve()
    except (OSError, RuntimeError) as e:
        # RuntimeError puede ocurrir con symlinks circulares
        raise ValueError(f"No se puede resolver la ruta '{path}': {e}")

    # Verificar existencia
    try:
        exists = resolved_path.exists()
    except (OSError, PermissionError) as e:
        raise PermissionError(f"No se puede acceder a la ruta '{resolved_path}': {e}")

    if not exists:
        # Proporcionar sugerencias útiles
        parent = resolved_path.parent
        parent_exists = parent.exists() if parent else False

        if parent_exists:
            # Buscar archivos similares en el directorio padre
            similar_files = []
            try:
                for f in parent.iterdir():
                    if f.is_file() and f.suffix == resolved_path.suffix:
                        similar_files.append(f.name)
            except (OSError, PermissionError):
                pass

            if similar_files:
                suggestion = f". Archivos similares encontrados: {similar_files[:5]}"
            else:
                suggestion = f". El directorio padre existe: {parent}"
        else:
            suggestion = f". El directorio padre tampoco existe: {parent}"

        raise FileNotFoundError(f"Archivo no encontrado: {resolved_path}{suggestion}")

    # Verificar que sea un archivo (no directorio)
    try:
        is_file = resolved_path.is_file()
    except (OSError, PermissionError) as e:
        raise PermissionError(f"No se puede verificar tipo de '{resolved_path}': {e}")

    if not is_file:
        if resolved_path.is_dir():
            raise IsADirectoryError(f"La ruta apunta a un directorio, no a un archivo: {resolved_path}")
        else:
            raise ValueError(f"La ruta no apunta a un archivo regular: {resolved_path}")

    # Verificar permisos de lectura intentando abrir el archivo
    try:
        with open(resolved_path, 'rb') as f:
            # Intentar leer un byte para verificar acceso real
            f.read(1)
    except PermissionError:
        raise PermissionError(f"No hay permisos de lectura para: {resolved_path}")
    except IOError as e:
        # Puede indicar archivo bloqueado u otro problema de I/O
        if "being used" in str(e).lower() or "locked" in str(e).lower():
            raise PermissionError(f"El archivo está bloqueado o en uso: {resolved_path}")
        raise ValueError(f"Error de I/O al acceder al archivo: {resolved_path} - {e}")

    return resolved_path


def _validate_file_size(
    metadata: FileMetadata,
    max_size_mb: float = MAX_FILE_SIZE_MB,
    allow_empty: bool = False,
) -> List[str]:
    """
    Valida que el tamaño del archivo sea razonable.

    Args:
        metadata: Metadatos del archivo
        max_size_mb: Tamaño máximo permitido en MB
        allow_empty: Si True, permite archivos vacíos (solo warning)

    Returns:
        List[str]: Lista de advertencias (vacía si no hay problemas)

    Raises:
        ValueError: Si el archivo excede el tamaño máximo
        ValueError: Si el archivo está vacío y allow_empty es False
    """
    warnings = []

    if metadata is None:
        raise ValueError("Metadatos del archivo son None")

    # Validar que tenemos información válida
    if not metadata.exists:
        raise FileNotFoundError(f"El archivo no existe: {metadata.path}")

    # Verificar tamaño máximo
    if metadata.size_mb > max_size_mb:
        raise ValueError(
            f"Archivo demasiado grande: {metadata.size_mb:.2f} MB. "
            f"Máximo permitido: {max_size_mb} MB. "
            f"Archivo: {metadata.path}"
        )

    # Verificar archivo vacío
    if metadata.size_bytes == 0:
        msg = f"El archivo está vacío (0 bytes): {metadata.path}"
        if allow_empty:
            logger.warning(msg)
            warnings.append(msg)
        else:
            raise ValueError(msg)

    # Advertencias para archivos muy pequeños (posiblemente corruptos)
    MIN_REASONABLE_SIZE_BYTES = 10  # Menos de 10 bytes es sospechoso
    if 0 < metadata.size_bytes < MIN_REASONABLE_SIZE_BYTES:
        msg = f"Archivo sospechosamente pequeño ({metadata.size_bytes} bytes): {metadata.path}"
        logger.warning(msg)
        warnings.append(msg)

    # Advertencia para archivos muy grandes (pero dentro del límite)
    LARGE_FILE_THRESHOLD_MB = max_size_mb * 0.8  # 80% del máximo
    if metadata.size_mb > LARGE_FILE_THRESHOLD_MB:
        msg = (
            f"Archivo grande ({metadata.size_mb:.2f} MB), "
            f"cerca del límite de {max_size_mb} MB: {metadata.path}"
        )
        logger.warning(msg)
        warnings.append(msg)

    return warnings


def _analyze_dataframe_quality(
    df: pd.DataFrame,
    include_sample_data: bool = False,
) -> DataQualityMetrics:
    """
    Analiza la calidad de un DataFrame y genera métricas detalladas.

    Args:
        df: DataFrame a analizar
        include_sample_data: Si True, incluye muestra de datos problemáticos

    Returns:
        DataQualityMetrics: Métricas de calidad comprehensivas
    """
    metrics = DataQualityMetrics()

    # Validar entrada
    if df is None:
        metrics.add_warning("DataFrame es None")
        return metrics

    if not isinstance(df, pd.DataFrame):
        metrics.add_warning(f"Entrada no es DataFrame: {type(df).__name__}")
        return metrics

    if df.empty:
        metrics.add_warning("DataFrame está vacío")
        return metrics

    # Manejar MultiIndex en columnas
    if isinstance(df.columns, pd.MultiIndex):
        metrics.add_warning(
            f"DataFrame tiene MultiIndex en columnas con {df.columns.nlevels} niveles"
        )
        # Aplanar para análisis
        try:
            flat_columns = ['_'.join(map(str, col)).strip('_') for col in df.columns.values]
            metrics.total_columns = len(flat_columns)
        except Exception as e:
            logger.warning(f"Error al aplanar MultiIndex: {e}")
            metrics.total_columns = len(df.columns)
    else:
        metrics.total_columns = len(df.columns)

    # Métricas básicas
    metrics.total_rows = len(df)

    # Análisis de nulos con manejo de errores
    try:
        null_count = df.isnull().sum().sum()
        total_cells = metrics.total_rows * metrics.total_columns
        metrics.null_cells = int(null_count)

        if total_cells > 0:
            metrics.null_percentage = (null_count / total_cells) * 100
        else:
            metrics.null_percentage = 0.0
    except Exception as e:
        logger.warning(f"Error al analizar nulos: {e}")
        metrics.add_warning(f"No se pudieron analizar valores nulos: {e}")

    # Columnas duplicadas (manejo seguro)
    try:
        if not isinstance(df.columns, pd.MultiIndex):
            # Convertir columnas a strings para comparación
            col_strings = [str(c) if c is not None else '' for c in df.columns]
            seen = set()
            duplicates = []
            for col in col_strings:
                if col in seen:
                    duplicates.append(col)
                seen.add(col)

            if duplicates:
                metrics.duplicate_columns = list(set(duplicates))
                metrics.add_warning(f"Columnas duplicadas encontradas: {metrics.duplicate_columns}")
    except Exception as e:
        logger.warning(f"Error al detectar columnas duplicadas: {e}")

    # Columnas con todos nulos
    try:
        all_null_mask = df.isnull().all()
        all_null_cols = df.columns[all_null_mask].tolist()
        if all_null_cols:
            metrics.columns_with_all_nulls = [str(c) for c in all_null_cols]
            metrics.add_warning(f"Columnas con todos valores nulos: {metrics.columns_with_all_nulls}")
    except Exception as e:
        logger.warning(f"Error al detectar columnas totalmente nulas: {e}")

    # Columnas sin nombre o con nombre inválido
    try:
        empty_cols = []
        for i, col in enumerate(df.columns):
            col_str = str(col) if col is not None else ''
            col_str = col_str.strip()

            if not col_str or col_str.lower() in ('nan', 'none', 'null', 'unnamed'):
                empty_cols.append(f"Column_{i}" if not col_str else col_str)

        if empty_cols:
            metrics.empty_columns = empty_cols
            metrics.add_warning(f"Columnas sin nombre válido encontradas: {len(empty_cols)}")
    except Exception as e:
        logger.warning(f"Error al detectar columnas vacías: {e}")

    # Tipos de datos (manejo seguro de tipos exóticos)
    try:
        data_types = {}
        for col in df.columns:
            col_str = str(col) if col is not None else f"col_{id(col)}"
            try:
                dtype_str = str(df[col].dtype)
            except Exception:
                dtype_str = "unknown"
            data_types[col_str] = dtype_str
        metrics.data_types = data_types
    except Exception as e:
        logger.warning(f"Error al obtener tipos de datos: {e}")
        metrics.data_types = {}

    # Uso de memoria (con fallback)
    try:
        memory_bytes = df.memory_usage(deep=True).sum()
        metrics.memory_usage_mb = memory_bytes / (1024 * 1024)
    except Exception as e:
        logger.warning(f"No se pudo calcular uso de memoria: {e}")
        # Estimación aproximada
        try:
            metrics.memory_usage_mb = (df.size * 8) / (1024 * 1024)  # Asumiendo 8 bytes por elemento
        except Exception:
            metrics.memory_usage_mb = 0.0

    # Advertencias de tamaño
    if metrics.total_rows > MAX_ROWS_WARNING:
        metrics.add_warning(
            f"DataFrame muy grande: {metrics.total_rows:,} filas "
            f"(umbral de advertencia: {MAX_ROWS_WARNING:,})"
        )

    if metrics.total_columns > MAX_COLS_WARNING:
        metrics.add_warning(
            f"Muchas columnas: {metrics.total_columns} "
            f"(umbral de advertencia: {MAX_COLS_WARNING})"
        )

    if metrics.total_rows < MIN_ROWS_WARNING:
        metrics.add_warning(
            f"Muy pocas filas: {metrics.total_rows} "
            f"(mínimo esperado: {MIN_ROWS_WARNING})"
        )

    # Advertencia de nulos
    if metrics.null_percentage > MAX_NULL_PERCENTAGE_WARNING:
        metrics.add_warning(
            f"Alto porcentaje de valores nulos: {metrics.null_percentage:.1f}% "
            f"(umbral: {MAX_NULL_PERCENTAGE_WARNING}%)"
        )

    # Análisis adicional: filas completamente duplicadas
    try:
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            pct_duplicates = (duplicate_rows / metrics.total_rows) * 100
            if pct_duplicates > 10:  # Más del 10% duplicado
                metrics.add_warning(
                    f"Alto porcentaje de filas duplicadas: {pct_duplicates:.1f}% "
                    f"({duplicate_rows:,} filas)"
                )
    except Exception as e:
        logger.debug(f"No se pudo analizar filas duplicadas: {e}")

    return metrics


def _clean_dataframe(
    df: pd.DataFrame,
    remove_empty_rows: bool = True,
    remove_empty_columns: bool = False,
    rename_duplicates: bool = True,
    preserve_index: bool = False,
) -> pd.DataFrame:
    """
    Limpia y normaliza un DataFrame de forma robusta.

    Args:
        df: DataFrame a limpiar
        remove_empty_rows: Si True, elimina filas completamente vacías
        remove_empty_columns: Si True, elimina columnas completamente vacías
        rename_duplicates: Si True, renombra columnas duplicadas
        preserve_index: Si True, preserva el índice original

    Returns:
        DataFrame limpio
    """
    if df is None:
        logger.warning("DataFrame de entrada es None, retornando DataFrame vacío")
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        logger.warning(f"Entrada no es DataFrame ({type(df).__name__}), retornando vacío")
        return pd.DataFrame()

    if df.empty:
        logger.debug("DataFrame de entrada está vacío")
        return df.copy()

    try:
        df_clean = df.copy()
    except Exception as e:
        logger.error(f"Error al copiar DataFrame: {e}")
        return df  # Retornar original si no se puede copiar

    rows_inicial = len(df_clean)
    cols_inicial = len(df_clean.columns)

    # Manejar MultiIndex en columnas
    if isinstance(df_clean.columns, pd.MultiIndex):
        logger.info("Aplanando MultiIndex en columnas")
        try:
            df_clean.columns = [
                '_'.join(str(level) for level in col if pd.notna(level) and str(level).strip())
                for col in df_clean.columns.values
            ]
        except Exception as e:
            logger.warning(f"Error al aplanar MultiIndex: {e}")
            df_clean.columns = [f"col_{i}" for i in range(len(df_clean.columns))]

    # Limpiar nombres de columnas
    try:
        nuevas_columnas = []
        for i, col in enumerate(df_clean.columns):
            if col is None or (isinstance(col, float) and pd.isna(col)):
                nuevas_columnas.append(f"Unnamed_{i}")
            else:
                col_str = str(col).strip()
                if not col_str or col_str.lower() in ('nan', 'none', 'null'):
                    nuevas_columnas.append(f"Unnamed_{i}")
                else:
                    nuevas_columnas.append(col_str)

        df_clean.columns = nuevas_columnas
    except Exception as e:
        logger.warning(f"Error al limpiar nombres de columnas: {e}")

    # Manejar columnas duplicadas
    if rename_duplicates:
        try:
            duplicated_mask = df_clean.columns.duplicated()
            if duplicated_mask.any():
                num_duplicates = duplicated_mask.sum()
                new_cols = []
                col_counts: Dict[str, int] = {}

                for col in df_clean.columns:
                    if col in col_counts:
                        col_counts[col] += 1
                        new_cols.append(f"{col}_{col_counts[col]}")
                    else:
                        col_counts[col] = 0
                        new_cols.append(col)

                df_clean.columns = new_cols
                logger.info(f"Se renombraron {num_duplicates} columnas duplicadas")
        except Exception as e:
            logger.warning(f"Error al renombrar columnas duplicadas: {e}")

    # Eliminar columnas completamente vacías
    if remove_empty_columns:
        try:
            empty_cols_mask = df_clean.isnull().all()
            empty_cols = df_clean.columns[empty_cols_mask].tolist()

            if empty_cols:
                df_clean = df_clean.drop(columns=empty_cols)
                logger.info(f"Se eliminaron {len(empty_cols)} columnas completamente vacías")
        except Exception as e:
            logger.warning(f"Error al eliminar columnas vacías: {e}")

    # Eliminar filas completamente vacías
    if remove_empty_rows:
        try:
            rows_before = len(df_clean)
            df_clean = df_clean.dropna(how='all')
            rows_removed = rows_before - len(df_clean)

            if rows_removed > 0:
                logger.info(f"Se eliminaron {rows_removed} filas completamente vacías")

                # Advertir si se eliminaron todas las filas
                if df_clean.empty:
                    logger.warning(
                        f"ADVERTENCIA: Todas las filas fueron eliminadas. "
                        f"El DataFrame original tenía {rows_inicial} filas."
                    )
        except Exception as e:
            logger.warning(f"Error al eliminar filas vacías: {e}")

    # Resetear índice si no se debe preservar
    if not preserve_index and not df_clean.empty:
        try:
            df_clean = df_clean.reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Error al resetear índice: {e}")

    # Log resumen
    rows_final = len(df_clean)
    cols_final = len(df_clean.columns)

    if rows_inicial != rows_final or cols_inicial != cols_final:
        logger.info(
            f"Limpieza completada: {rows_inicial}x{cols_inicial} -> {rows_final}x{cols_final}"
        )

    return df_clean


def _detect_csv_delimiter(
    path: Path,
    sample_size: int = 10,
    encodings_to_try: Optional[List[str]] = None,
    max_line_length: int = 10000,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Intenta detectar el delimitador y encoding de un archivo CSV.

    Args:
        path: Ruta al archivo CSV
        sample_size: Número de líneas a analizar
        encodings_to_try: Lista de encodings a probar
        max_line_length: Longitud máxima de línea a considerar

    Returns:
        Tuple[delimitador, encoding]: Delimitador y encoding detectados o (None, None)
    """
    if encodings_to_try is None:
        encodings_to_try = DEFAULT_ENCODINGS.copy()

    for encoding in encodings_to_try:
        try:
            lines = []
            with open(path, 'r', encoding=encoding, errors='strict') as f:
                for _ in range(sample_size):
                    try:
                        line = f.readline()
                        if not line:  # EOF
                            break
                        # Limitar longitud de línea
                        if len(line) > max_line_length:
                            line = line[:max_line_length]
                            logger.debug(f"Línea truncada a {max_line_length} caracteres")
                        lines.append(line)
                    except Exception as e:
                        logger.debug(f"Error leyendo línea: {e}")
                        break

            if not lines:
                continue

            # Contar ocurrencias de cada delimitador por línea
            delimiter_consistency: Dict[str, List[int]] = {
                delim: [] for delim in COMMON_DELIMITERS
            }

            for line in lines:
                # Ignorar líneas vacías o muy cortas
                if len(line.strip()) < 2:
                    continue

                for delim in COMMON_DELIMITERS:
                    count = line.count(delim)
                    delimiter_consistency[delim].append(count)

            # Evaluar consistencia de cada delimitador
            best_delimiter = None
            best_score = 0

            for delim, counts in delimiter_consistency.items():
                if not counts:
                    continue

                # Filtrar líneas con 0 ocurrencias
                non_zero_counts = [c for c in counts if c > 0]

                if not non_zero_counts:
                    continue

                # Calcular score basado en:
                # 1. Número promedio de ocurrencias (más es mejor)
                # 2. Consistencia (varianza baja es mejor)
                avg_count = sum(non_zero_counts) / len(non_zero_counts)

                # Solo considerar si hay al menos 1 ocurrencia promedio
                if avg_count < 1:
                    continue

                # Calcular varianza para medir consistencia
                if len(non_zero_counts) > 1:
                    variance = sum((c - avg_count) ** 2 for c in non_zero_counts) / len(non_zero_counts)
                    # Normalizar varianza
                    consistency_score = 1 / (1 + variance)
                else:
                    consistency_score = 0.5  # Solo una línea, menos confianza

                # Score combinado
                # Penalizar si no todas las líneas tienen el delimitador
                coverage = len(non_zero_counts) / len(counts) if counts else 0
                score = avg_count * consistency_score * coverage

                if score > best_score:
                    best_score = score
                    best_delimiter = delim

            if best_delimiter and best_score > 0.5:
                logger.debug(
                    f"Delimitador detectado: '{best_delimiter}' "
                    f"(score: {best_score:.2f}, encoding: {encoding})"
                )
                return best_delimiter, encoding

        except UnicodeDecodeError:
            logger.debug(f"Encoding {encoding} falló, probando siguiente...")
            continue
        except Exception as e:
            logger.debug(f"Error con encoding {encoding}: {e}")
            continue

    logger.debug("No se pudo detectar delimitador con confianza")
    return None, None


# ============================================================================
# FUNCIONES DE CARGA PRINCIPALES
# ============================================================================
def load_from_csv(
    path: PathType,
    sep: Optional[str] = None,
    encoding: Optional[str] = None,
    auto_detect: bool = True,
    max_retries: int = 20,  # Límite de combinaciones a probar
    **kwargs,
) -> LoadResult:
    """
    Carga un archivo CSV con detección automática de encoding y delimitador.

    Args:
        path: Ruta al archivo CSV
        sep: Delimitador (None para auto-detección)
        encoding: Encoding (None para intentar múltiples)
        auto_detect: Si True, intenta detectar delimitador y encoding
        max_retries: Número máximo de combinaciones a probar
        **kwargs: Argumentos adicionales para pd.read_csv

    Returns:
        LoadResult con el DataFrame y métricas
    """
    start_time = time.time()
    warnings_list: List[str] = []
    attempts_log: List[str] = []

    # Validar archivo
    try:
        path = _validate_file_path(path)
    except Exception as e:
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=None,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=str(e),
        )

    metadata = FileMetadata.from_path(path)

    try:
        size_warnings = _validate_file_size(metadata, allow_empty=True)
        warnings_list.extend(size_warnings)
    except ValueError as e:
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=str(e),
        )

    logger.info(f"Cargando CSV: {path} ({metadata.size_mb:.2f} MB)")

    # Limpiar kwargs de conflictos
    safe_kwargs = {k: v for k, v in kwargs.items() if k not in ('sep', 'encoding', 'delimiter')}

    if 'sep' in kwargs or 'delimiter' in kwargs:
        logger.warning("Parámetros 'sep' o 'delimiter' en kwargs serán ignorados. Use el parámetro 'sep' directo.")

    # Determinar encodings a probar
    if encoding:
        encodings_to_try = [encoding]
    else:
        encodings_to_try = DEFAULT_ENCODINGS.copy()

    # Determinar delimitadores a probar
    detected_delim = None
    detected_encoding = None

    if sep:
        delimiters_to_try = [sep]
    elif auto_detect:
        detected_delim, detected_encoding = _detect_csv_delimiter(path)
        if detected_delim:
            # Priorizar el delimitador detectado
            delimiters_to_try = [detected_delim] + [d for d in COMMON_DELIMITERS if d != detected_delim]
            if detected_encoding and detected_encoding not in encodings_to_try:
                encodings_to_try.insert(0, detected_encoding)
        else:
            delimiters_to_try = COMMON_DELIMITERS.copy()
    else:
        delimiters_to_try = [";"]  # Default

    # Intentar cargar con diferentes combinaciones
    last_error: Optional[Exception] = None
    attempts = 0

    for enc in encodings_to_try:
        for delim in delimiters_to_try:
            if attempts >= max_retries:
                logger.warning(f"Se alcanzó el límite de {max_retries} intentos")
                break

            attempts += 1
            attempt_info = f"intento {attempts}: encoding='{enc}', sep='{delim}'"

            try:
                logger.debug(f"Intentando cargar CSV: {attempt_info}")

                df = pd.read_csv(
                    path,
                    sep=delim,
                    encoding=enc,
                    on_bad_lines='warn',  # Python 3.9+ pandas
                    **safe_kwargs
                )

                # Verificar que la carga tenga sentido
                # Si solo hay una columna, probablemente el delimitador es incorrecto
                if len(df.columns) == 1 and len(df) > 0:
                    # Verificar si la única columna contiene el delimitador esperado
                    col_name = str(df.columns[0])
                    if any(d in col_name for d in COMMON_DELIMITERS if d != delim):
                        attempts_log.append(f"{attempt_info}: solo 1 columna, posible delimitador incorrecto")
                        continue

                # Limpiar y analizar
                df = _clean_dataframe(df)

                if df.empty:
                    logger.warning(f"CSV cargado pero está vacío: {path}")
                    quality_metrics = DataQualityMetrics()
                    quality_metrics.add_warning("Archivo CSV vacío después de limpieza")

                    result = LoadResult(
                        status=LoadStatus.EMPTY,
                        data=df,
                        file_metadata=metadata,
                        quality_metrics=quality_metrics,
                        load_time_seconds=time.time() - start_time,
                        encoding_used=enc,
                        delimiter_used=delim,
                    )
                    result.warnings = warnings_list
                    return result

                quality_metrics = _analyze_dataframe_quality(df)

                logger.info(
                    f"CSV cargado exitosamente: {quality_metrics.total_rows:,} filas, "
                    f"{quality_metrics.total_columns} columnas "
                    f"(encoding: {enc}, delimitador: '{delim}')"
                )

                if detected_delim and detected_delim != delim:
                    warnings_list.append(
                        f"Delimitador detectado ({detected_delim}) difiere del usado ({delim})"
                    )

                result = LoadResult(
                    status=LoadStatus.SUCCESS,
                    data=df,
                    file_metadata=metadata,
                    quality_metrics=quality_metrics,
                    load_time_seconds=time.time() - start_time,
                    encoding_used=enc,
                    delimiter_used=delim,
                )
                result.warnings = warnings_list
                return result

            except UnicodeDecodeError as e:
                last_error = e
                attempts_log.append(f"{attempt_info}: UnicodeDecodeError")
                continue

            except pd.errors.EmptyDataError as e:
                logger.warning(f"Archivo CSV vacío o mal formado: {path}")
                quality_metrics = DataQualityMetrics()
                quality_metrics.add_warning(str(e))

                result = LoadResult(
                    status=LoadStatus.EMPTY,
                    data=pd.DataFrame(),
                    file_metadata=metadata,
                    quality_metrics=quality_metrics,
                    load_time_seconds=time.time() - start_time,
                    error_message=str(e),
                )
                result.warnings = warnings_list
                return result

            except pd.errors.ParserError as e:
                last_error = e
                attempts_log.append(f"{attempt_info}: ParserError - {str(e)[:50]}")
                continue

            except Exception as e:
                last_error = e
                attempts_log.append(f"{attempt_info}: {type(e).__name__} - {str(e)[:50]}")
                continue

        if attempts >= max_retries:
            break

    # Si llegamos aquí, ninguna combinación funcionó
    error_details = "\n".join(attempts_log[-10:])  # Últimos 10 intentos
    error_msg = (
        f"No se pudo cargar el CSV después de {attempts} intentos "
        f"({len(encodings_to_try)} encodings × {len(delimiters_to_try)} delimitadores). "
        f"Último error: {last_error}\n"
        f"Últimos intentos:\n{error_details}"
    )
    logger.error(error_msg)

    return LoadResult(
        status=LoadStatus.FAILED,
        data=None,
        file_metadata=metadata,
        quality_metrics=None,
        load_time_seconds=time.time() - start_time,
        error_message=error_msg,
    )


def load_from_xlsx(
    path: PathType,
    sheet_name: Optional[Union[str, int, List[Union[str, int]]]] = 0,
    password: Optional[str] = None,
    **kwargs,
) -> LoadResult:
    """
    Carga un archivo Excel con manejo robusto de errores.

    Args:
        path: Ruta al archivo Excel
        sheet_name: Nombre/índice de hoja(s), None para todas, lista para múltiples
        password: Contraseña si el archivo está protegido (solo .xlsx)
        **kwargs: Argumentos adicionales para pd.read_excel

    Returns:
        LoadResult con DataFrame(s) y métricas
    """
    start_time = time.time()
    warnings_list: List[str] = []

    # Validar archivo
    try:
        path = _validate_file_path(path)
    except Exception as e:
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=None,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=str(e),
        )

    metadata = FileMetadata.from_path(path)

    try:
        size_warnings = _validate_file_size(metadata, allow_empty=True)
        warnings_list.extend(size_warnings)
    except ValueError as e:
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=str(e),
        )

    # Formatear información de hojas para logging
    if sheet_name is None:
        sheet_info = "todas las hojas"
    elif isinstance(sheet_name, (list, tuple)):
        sheet_info = f"hojas: {sheet_name}"
    else:
        sheet_info = f"hoja: {sheet_name}"

    logger.info(f"Cargando Excel: {path} ({metadata.size_mb:.2f} MB), {sheet_info}")

    # Primero, obtener lista de hojas disponibles
    available_sheets: List[str] = []
    try:
        excel_file = pd.ExcelFile(path)
        available_sheets = excel_file.sheet_names
        logger.debug(f"Hojas disponibles: {available_sheets}")
    except Exception as e:
        logger.warning(f"No se pudo obtener lista de hojas: {e}")

    # Validar que las hojas solicitadas existan
    if available_sheets and sheet_name is not None:
        sheets_to_check = [sheet_name] if not isinstance(sheet_name, list) else sheet_name

        for sheet in sheets_to_check:
            if isinstance(sheet, str) and sheet not in available_sheets:
                error_msg = (
                    f"Hoja '{sheet}' no encontrada. "
                    f"Hojas disponibles: {available_sheets}"
                )
                logger.error(error_msg)
                return LoadResult(
                    status=LoadStatus.FAILED,
                    data=None,
                    file_metadata=metadata,
                    quality_metrics=None,
                    load_time_seconds=time.time() - start_time,
                    error_message=error_msg,
                )
            elif isinstance(sheet, int) and (sheet < 0 or sheet >= len(available_sheets)):
                error_msg = (
                    f"Índice de hoja {sheet} fuera de rango. "
                    f"Rango válido: 0-{len(available_sheets)-1}"
                )
                logger.error(error_msg)
                return LoadResult(
                    status=LoadStatus.FAILED,
                    data=None,
                    file_metadata=metadata,
                    quality_metrics=None,
                    load_time_seconds=time.time() - start_time,
                    error_message=error_msg,
                )

    try:
        # Cargar datos
        read_kwargs = kwargs.copy()

        # Manejar contraseña (solo para openpyxl/.xlsx)
        if password:
            if path.suffix.lower() == '.xlsx':
                try:
                    from openpyxl import load_workbook
                    # Verificar que la contraseña funciona
                    wb = load_workbook(path, read_only=True, password=password)
                    wb.close()
                except ImportError:
                    warnings_list.append("openpyxl no instalado, no se puede usar contraseña")
                except Exception as e:
                    return LoadResult(
                        status=LoadStatus.FAILED,
                        data=None,
                        file_metadata=metadata,
                        quality_metrics=None,
                        load_time_seconds=time.time() - start_time,
                        error_message=f"Error con contraseña: {e}",
                    )
            else:
                warnings_list.append(
                    f"Contraseña proporcionada pero el formato {path.suffix} no la soporta"
                )

        data = pd.read_excel(path, sheet_name=sheet_name, **read_kwargs)

        # Procesar según el tipo de dato retornado
        if isinstance(data, dict):
            # Múltiples hojas
            sheets_loaded = list(data.keys())
            logger.info(f"Se cargaron {len(sheets_loaded)} hojas: {sheets_loaded}")

            # Limpiar y analizar cada hoja
            cleaned_data: Dict[str, pd.DataFrame] = {}
            metrics_by_sheet: Dict[str, DataQualityMetrics] = {}
            total_rows = 0
            total_cols = 0

            for sheet, df in data.items():
                sheet_str = str(sheet)

                try:
                    df_clean = _clean_dataframe(df)
                    cleaned_data[sheet_str] = df_clean

                    if not df_clean.empty:
                        metrics = _analyze_dataframe_quality(df_clean)
                        metrics_by_sheet[sheet_str] = metrics
                        total_rows += metrics.total_rows
                        total_cols = max(total_cols, metrics.total_columns)

                        for w in metrics.warnings:
                            warnings_list.append(f"[{sheet_str}] {w}")
                    else:
                        warnings_list.append(f"[{sheet_str}] Hoja vacía")

                except Exception as e:
                    logger.warning(f"Error procesando hoja '{sheet_str}': {e}")
                    cleaned_data[sheet_str] = pd.DataFrame()
                    warnings_list.append(f"[{sheet_str}] Error: {e}")

            # Crear métricas combinadas
            quality_metrics = DataQualityMetrics()
            quality_metrics.total_rows = total_rows
            quality_metrics.total_columns = total_cols
            quality_metrics.warnings = warnings_list

            # Determinar estado
            non_empty_sheets = [s for s, df in cleaned_data.items() if not df.empty]

            if not non_empty_sheets:
                logger.warning("Todas las hojas están vacías")
                return LoadResult(
                    status=LoadStatus.EMPTY,
                    data=cleaned_data,
                    file_metadata=metadata,
                    quality_metrics=quality_metrics,
                    load_time_seconds=time.time() - start_time,
                    sheets_loaded=sheets_loaded,
                )

            status = LoadStatus.SUCCESS
            if len(non_empty_sheets) < len(sheets_loaded):
                status = LoadStatus.PARTIAL_SUCCESS
                warnings_list.append(
                    f"Solo {len(non_empty_sheets)} de {len(sheets_loaded)} hojas tienen datos"
                )

            result = LoadResult(
                status=status,
                data=cleaned_data,
                file_metadata=metadata,
                quality_metrics=quality_metrics,
                load_time_seconds=time.time() - start_time,
                sheets_loaded=sheets_loaded,
            )
            result.warnings = warnings_list
            return result

        else:
            # Una sola hoja
            df = _clean_dataframe(data)

            # Determinar nombre de hoja para logging
            if isinstance(sheet_name, int) and available_sheets:
                actual_sheet_name = available_sheets[sheet_name] if sheet_name < len(available_sheets) else str(sheet_name)
            else:
                actual_sheet_name = str(sheet_name) if sheet_name is not None else "default"

            if df.empty:
                logger.warning(f"Hoja '{actual_sheet_name}' está vacía")
                quality_metrics = DataQualityMetrics()
                quality_metrics.add_warning(f"Hoja '{actual_sheet_name}' vacía")

                result = LoadResult(
                    status=LoadStatus.EMPTY,
                    data=df,
                    file_metadata=metadata,
                    quality_metrics=quality_metrics,
                    load_time_seconds=time.time() - start_time,
                    sheets_loaded=[actual_sheet_name],
                )
                result.warnings = warnings_list
                return result

            quality_metrics = _analyze_dataframe_quality(df)

            logger.info(
                f"Excel cargado exitosamente: {quality_metrics.total_rows:,} filas, "
                f"{quality_metrics.total_columns} columnas"
            )

            result = LoadResult(
                status=LoadStatus.SUCCESS,
                data=df,
                file_metadata=metadata,
                quality_metrics=quality_metrics,
                load_time_seconds=time.time() - start_time,
                sheets_loaded=[actual_sheet_name],
            )
            result.warnings = warnings_list
            return result

    except ValueError as e:
        error_msg = str(e)
        if "worksheet" in error_msg.lower() or "sheet" in error_msg.lower():
            error_msg = (
                f"Error de hoja: {e}. "
                f"Hojas disponibles: {available_sheets if available_sheets else 'desconocidas'}"
            )
        else:
            error_msg = f"Error de valor al leer Excel: {e}"

        logger.error(error_msg)
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )

    except PermissionError as e:
        error_msg = f"Archivo Excel bloqueado o sin permisos: {path}. Error: {e}"
        logger.error(error_msg)
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )

    except Exception as e:
        error_msg = f"Error inesperado al leer Excel {path}: {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)

        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )


def load_from_pdf(
    path: PathType,
    page_range: Optional[range] = None,
    pages: Optional[List[int]] = None,  # Alternativa: lista específica de páginas
    table_settings: Optional[Dict[str, Any]] = None,
    max_pages: int = MAX_PDF_PAGES,
    password: Optional[str] = None,
    **kwargs,
) -> LoadResult:
    """
    Extrae tablas de un PDF con manejo robusto de errores.

    Args:
        path: Ruta al archivo PDF
        page_range: Rango de páginas (ej: range(0, 5)) - 0-indexed
        pages: Lista específica de páginas a procesar (alternativa a page_range)
        table_settings: Configuración para extracción de tablas
        max_pages: Límite máximo de páginas a procesar
        password: Contraseña si el PDF está encriptado
        **kwargs: Argumentos para pdfplumber.open

    Returns:
        LoadResult con DataFrame y métricas
    """
    start_time = time.time()
    warnings_list: List[str] = []

    # Verificar disponibilidad de pdfplumber
    try:
        import pdfplumber
    except ImportError:
        error_msg = (
            "pdfplumber no está instalado. "
            "Instale con: pip install pdfplumber"
        )
        logger.error(error_msg)
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=None,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )

    # Validar archivo
    try:
        path = _validate_file_path(path)
    except Exception as e:
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=None,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=str(e),
        )

    metadata = FileMetadata.from_path(path)

    try:
        size_warnings = _validate_file_size(metadata, allow_empty=False)
        warnings_list.extend(size_warnings)
    except ValueError as e:
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=str(e),
        )

    logger.info(f"Extrayendo tablas del PDF: {path} ({metadata.size_mb:.2f} MB)")

    table_settings = table_settings or {}
    tables_data: List[pd.DataFrame] = []
    pages_processed = 0
    pages_with_tables = 0
    total_tables_found = 0
    extraction_errors: List[str] = []

    try:
        # Configurar kwargs para pdfplumber
        open_kwargs = {}
        if password:
            open_kwargs['password'] = password

        with pdfplumber.open(path, **open_kwargs) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF tiene {total_pages} páginas")

            # Validar y determinar páginas a procesar
            if pages is not None:
                # Lista específica de páginas
                pages_to_process_indices = []
                for p in pages:
                    if isinstance(p, int) and 0 <= p < total_pages:
                        pages_to_process_indices.append(p)
                    else:
                        warnings_list.append(f"Página {p} ignorada (fuera de rango 0-{total_pages-1})")

                if not pages_to_process_indices:
                    error_msg = f"Ninguna página válida especificada. Rango válido: 0-{total_pages-1}"
                    return LoadResult(
                        status=LoadStatus.FAILED,
                        data=None,
                        file_metadata=metadata,
                        quality_metrics=None,
                        load_time_seconds=time.time() - start_time,
                        error_message=error_msg,
                    )

                pages_to_process_indices = sorted(set(pages_to_process_indices))

            elif page_range is not None:
                # Validar y ajustar rango
                start_page = max(0, page_range.start)
                end_page = min(total_pages, page_range.stop)

                if start_page >= total_pages:
                    error_msg = f"Página inicial {page_range.start} excede total ({total_pages})"
                    return LoadResult(
                        status=LoadStatus.FAILED,
                        data=None,
                        file_metadata=metadata,
                        quality_metrics=None,
                        load_time_seconds=time.time() - start_time,
                        error_message=error_msg,
                    )

                if start_page != page_range.start or end_page != page_range.stop:
                    warnings_list.append(
                        f"Rango ajustado de {page_range.start}-{page_range.stop} "
                        f"a {start_page}-{end_page}"
                    )

                pages_to_process_indices = list(range(start_page, end_page))
                logger.info(f"Procesando páginas {start_page + 1} a {end_page}")
            else:
                # Todas las páginas
                pages_to_process_indices = list(range(total_pages))

            # Aplicar límite de páginas
            if len(pages_to_process_indices) > max_pages:
                warnings_list.append(
                    f"Limitando procesamiento a {max_pages} páginas "
                    f"(de {len(pages_to_process_indices)} solicitadas)"
                )
                pages_to_process_indices = pages_to_process_indices[:max_pages]

            # Procesar cada página
            for page_idx in pages_to_process_indices:
                page_num = page_idx + 1  # 1-indexed para logging
                pages_processed += 1

                try:
                    page = pdf.pages[page_idx]
                    extracted = page.extract_tables(table_settings)

                    if not extracted:
                        logger.debug(f"Página {page_num}: No se encontraron tablas")
                        continue

                    pages_with_tables += 1

                    for table_idx, table in enumerate(extracted):
                        if not table or len(table) == 0:
                            continue

                        total_tables_found += 1

                        try:
                            # Crear DataFrame con manejo robusto de headers
                            if len(table) > 1:
                                # Limpiar headers
                                raw_headers = table[0] if table[0] else []
                                headers = []
                                for j, h in enumerate(raw_headers):
                                    if h is None or (isinstance(h, str) and not h.strip()):
                                        headers.append(f"Col_{j}")
                                    else:
                                        headers.append(str(h).strip())

                                # Asegurar número correcto de headers
                                if table[1:]:  # Si hay datos
                                    max_cols = max(len(row) if row else 0 for row in table[1:])
                                    while len(headers) < max_cols:
                                        headers.append(f"Col_{len(headers)}")

                                df_table = pd.DataFrame(table[1:], columns=headers[:len(table[0]) if table[0] else 0])
                            else:
                                df_table = pd.DataFrame(table)

                            if not df_table.empty:
                                # Limpiar datos de la tabla
                                df_table = df_table.replace('', pd.NA)
                                df_table = df_table.dropna(how='all')

                                if not df_table.empty:
                                    # Agregar metadata de origen
                                    df_table['_source_page'] = page_num
                                    df_table['_source_table'] = table_idx + 1

                                    tables_data.append(df_table)
                                    logger.debug(
                                        f"Página {page_num}, tabla {table_idx + 1}: "
                                        f"{df_table.shape[0]} filas × {df_table.shape[1]} columnas"
                                    )

                        except Exception as e:
                            error_info = f"Error en tabla {table_idx + 1} de página {page_num}: {e}"
                            logger.warning(error_info)
                            extraction_errors.append(error_info)
                            continue

                except Exception as e:
                    error_info = f"Error procesando página {page_num}: {e}"
                    logger.warning(error_info)
                    extraction_errors.append(error_info)
                    continue

    except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as e:
        error_msg = f"PDF corrupto o mal formado: {e}"
        logger.error(error_msg)
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )

    except Exception as e:
        if "password" in str(e).lower() or "encrypted" in str(e).lower():
            error_msg = f"PDF encriptado. Proporcione contraseña: {e}"
        else:
            error_msg = f"Error al abrir o procesar PDF {path}: {type(e).__name__}: {e}"

        logger.error(error_msg, exc_info=True)
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )

    # Procesar resultados
    if extraction_errors:
        warnings_list.extend(extraction_errors[:10])  # Limitar warnings
        if len(extraction_errors) > 10:
            warnings_list.append(f"... y {len(extraction_errors) - 10} errores más")

    if not tables_data:
        logger.warning(
            f"No se encontraron tablas válidas en el PDF. "
            f"Páginas procesadas: {pages_processed}, páginas con tablas: {pages_with_tables}"
        )

        quality_metrics = DataQualityMetrics()
        quality_metrics.add_warning(
            f"No se encontraron tablas en {pages_processed} páginas procesadas"
        )
        quality_metrics.warnings.extend(warnings_list)

        return LoadResult(
            status=LoadStatus.EMPTY,
            data=pd.DataFrame(),
            file_metadata=metadata,
            quality_metrics=quality_metrics,
            load_time_seconds=time.time() - start_time,
        )

    # Combinar todas las tablas
    try:
        # Primero, alinear columnas si es posible
        all_columns = set()
        for df in tables_data:
            all_columns.update(df.columns)

        # Excluir columnas de metadata para alineación
        data_columns = [c for c in all_columns if not c.startswith('_source_')]

        combined_df = pd.concat(tables_data, ignore_index=True, sort=False)
        combined_df = _clean_dataframe(combined_df, remove_empty_rows=True)

        quality_metrics = _analyze_dataframe_quality(combined_df)
        quality_metrics.add_warning(
            f"Datos extraídos de {total_tables_found} tablas en "
            f"{pages_with_tables} de {pages_processed} páginas"
        )
        quality_metrics.warnings.extend(warnings_list)

        logger.info(
            f"PDF procesado exitosamente: {total_tables_found} tablas combinadas, "
            f"{quality_metrics.total_rows:,} filas, {quality_metrics.total_columns} columnas"
        )

        # Determinar estado
        status = LoadStatus.SUCCESS
        if extraction_errors:
            status = LoadStatus.PARTIAL_SUCCESS

        result = LoadResult(
            status=status,
            data=combined_df,
            file_metadata=metadata,
            quality_metrics=quality_metrics,
            load_time_seconds=time.time() - start_time,
        )
        result.warnings = warnings_list
        return result

    except Exception as e:
        error_msg = f"Error al combinar tablas del PDF: {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)

        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )


def load_data(
    path: PathType,
    format_hint: Optional[str] = None,
    validate_only: bool = False,
    telemetry_context: Optional[TelemetryContext] = None,
    **kwargs,
) -> LoadResult:
    """
    Función factory que carga datos según la extensión del archivo.
    Incluye validaciones exhaustivas y métricas detalladas.

    Args:
        path: Ruta al archivo
        format_hint: Sugerencia de formato ('csv', 'excel', 'xlsx', 'xls', 'pdf')
        validate_only: Si True, solo valida el archivo sin cargarlo
        telemetry_context: Contexto de telemetría para registro centralizado.
        **kwargs: Argumentos específicos del formato:
            - CSV: sep, encoding, auto_detect
            - Excel: sheet_name, password
            - PDF: page_range, pages, table_settings, password

    Returns:
        LoadResult con datos, métricas y metadatos

    Examples:
        >>> result = load_data("datos.csv")
        >>> if result.status == LoadStatus.SUCCESS:
        ...     df = result.data
        ...     print(result.quality_metrics.to_dict())

        >>> result = load_data("libro.xlsx", sheet_name=None)
        >>> if result.status == LoadStatus.SUCCESS:
        ...     for sheet, df in result.data.items():
        ...         print(f"{sheet}: {len(df)} filas")
    """
    start_time = time.time()

    logger.info("=" * 80)
    logger.info(f"Iniciando carga de datos desde: {path}")
    logger.info("=" * 80)

    if telemetry_context:
        telemetry_context.start_step("load_data", {"path": str(path), "format_hint": format_hint})

    # Validación inicial de path
    if path is None:
        error_msg = "La ruta del archivo no puede ser None"
        logger.error(error_msg)
        result = LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=None,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )
        if telemetry_context:
            telemetry_context.record_error("load_data", error_msg)
            telemetry_context.end_step("load_data", "failure", metadata=result.to_dict())
        return result

    try:
        # Validar y preparar ruta
        validated_path = _validate_file_path(path)
        metadata = FileMetadata.from_path(validated_path)

    except FileNotFoundError as e:
        error_msg = str(e)
        logger.error(error_msg)

        # Intentar crear metadata básica
        try:
            path_obj = Path(path) if not isinstance(path, Path) else path
            basic_metadata = FileMetadata(
                path=path_obj,
                size_bytes=0,
                size_mb=0,
                format=FileFormat.UNKNOWN,
                exists=False,
                readable=False,
                error="Archivo no encontrado",
            )
        except Exception:
            basic_metadata = None

        result = LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=basic_metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )
        if telemetry_context:
            telemetry_context.record_error("load_data", error_msg)
            telemetry_context.end_step("load_data", "failure", metadata=result.to_dict())
        return result

    except PermissionError as e:
        error_msg = f"Error de permisos: {e}"
        logger.error(error_msg)
        result = LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=None,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )
        if telemetry_context:
            telemetry_context.record_error("load_data", error_msg)
            telemetry_context.end_step("load_data", "failure", metadata=result.to_dict())
        return result

    except Exception as e:
        error_msg = f"Error validando archivo: {type(e).__name__}: {e}"
        logger.error(error_msg)
        result = LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=None,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )
        if telemetry_context:
            telemetry_context.record_error("load_data", error_msg)
            telemetry_context.end_step("load_data", "failure", metadata=result.to_dict())
        return result

    # Determinar formato
    file_format = metadata.format

    if format_hint:
        format_hint_normalized = format_hint.strip().upper()

        format_mapping = {
            'CSV': FileFormat.CSV,
            'TXT': FileFormat.CSV,
            'TSV': FileFormat.CSV,
            'EXCEL': FileFormat.EXCEL,
            'XLSX': FileFormat.EXCEL,
            'XLS': FileFormat.EXCEL,
            'XLSM': FileFormat.EXCEL,
            'XLSB': FileFormat.EXCEL,
            'PDF': FileFormat.PDF,
        }

        if format_hint_normalized in format_mapping:
            file_format = format_mapping[format_hint_normalized]
            if file_format != metadata.format:
                logger.info(
                    f"Formato sobrescrito por hint: {metadata.format.value} -> {file_format.value}"
                )
        else:
            logger.warning(
                f"Sugerencia de formato '{format_hint}' no reconocida. "
                f"Formatos válidos: {list(format_mapping.keys())}. "
                f"Usando detección automática: {metadata.format.value}"
            )

    # Validar formato soportado
    if file_format == FileFormat.UNKNOWN:
        extension = validated_path.suffix.lower()
        sorted_extensions = sorted(ALL_SUPPORTED_EXTENSIONS)
        error_msg = (
            f"Formato de archivo no soportado: '{extension}'. "
            f"Formatos soportados: {', '.join(sorted_extensions)}"
        )

        # Sugerencias para extensiones comunes no soportadas
        common_suggestions = {
            '.json': "Use json.load() o pd.read_json()",
            '.xml': "Use pd.read_xml() o lxml",
            '.parquet': "Use pd.read_parquet()",
            '.feather': "Use pd.read_feather()",
            '.pickle': "Use pd.read_pickle()",
            '.pkl': "Use pd.read_pickle()",
        }

        if extension in common_suggestions:
            error_msg += f"\nSugerencia: {common_suggestions[extension]}"

        logger.error(error_msg)
        result = LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )
        if telemetry_context:
            telemetry_context.record_error("load_data", error_msg)
            telemetry_context.end_step("load_data", "failure", metadata=result.to_dict())
        return result

    logger.info(
        f"Formato: {file_format.value}, Tamaño: {metadata.size_mb:.2f} MB"
    )

    # Si solo validación, retornar aquí
    if validate_only:
        try:
            warnings = _validate_file_size(metadata, allow_empty=True)

            quality_metrics = DataQualityMetrics()
            quality_metrics.warnings = warnings

            result = LoadResult(
                status=LoadStatus.SUCCESS,
                data=None,
                file_metadata=metadata,
                quality_metrics=quality_metrics,
                load_time_seconds=time.time() - start_time,
            )
            if telemetry_context:
                telemetry_context.end_step("load_data", "success", metadata=result.to_dict())
            return result
        except ValueError as e:
            result = LoadResult(
                status=LoadStatus.FAILED,
                data=None,
                file_metadata=metadata,
                quality_metrics=None,
                load_time_seconds=time.time() - start_time,
                error_message=str(e),
            )
            if telemetry_context:
                telemetry_context.record_error("load_data", str(e))
                telemetry_context.end_step("load_data", "failure", metadata=result.to_dict())
            return result

    # Filtrar kwargs según formato
    csv_params = {'sep', 'encoding', 'auto_detect', 'max_retries', 'header', 'names',
                  'index_col', 'usecols', 'dtype', 'skiprows', 'nrows', 'na_values'}
    excel_params = {'sheet_name', 'password', 'header', 'names', 'index_col',
                    'usecols', 'dtype', 'skiprows', 'nrows', 'na_values'}
    pdf_params = {'page_range', 'pages', 'table_settings', 'max_pages', 'password'}

    # Cargar según formato
    try:
        if file_format == FileFormat.CSV:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in csv_params}
            result = load_from_csv(validated_path, **filtered_kwargs)

        elif file_format == FileFormat.EXCEL:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in excel_params}
            result = load_from_xlsx(validated_path, **filtered_kwargs)

        elif file_format == FileFormat.PDF:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in pdf_params}
            result = load_from_pdf(validated_path, **filtered_kwargs)

        else:
            error_msg = f"Formato {file_format.value} detectado pero no implementado"
            logger.error(error_msg)
            result = LoadResult(
                status=LoadStatus.FAILED,
                data=None,
                file_metadata=metadata,
                quality_metrics=None,
                load_time_seconds=time.time() - start_time,
                error_message=error_msg,
            )
            if telemetry_context:
                telemetry_context.record_error("load_data", error_msg)
                telemetry_context.end_step("load_data", "failure", metadata=result.to_dict())
            return result

        # Advertir sobre kwargs ignorados
        all_valid_params = csv_params | excel_params | pdf_params
        ignored_params = set(kwargs.keys()) - all_valid_params
        if ignored_params:
            logger.warning(f"Parámetros ignorados (no aplicables): {ignored_params}")
            if hasattr(result, 'warnings'):
                result.warnings.append(f"Parámetros ignorados: {ignored_params}")

    except Exception as e:
        error_msg = f"Error inesperado durante la carga: {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        result = LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )
        if telemetry_context:
            telemetry_context.record_error("load_data", error_msg)
            telemetry_context.end_step("load_data", "failure", metadata=result.to_dict())
        return result

    # Logging final
    logger.info("=" * 80)
    logger.info(f"Carga completada - Estado: {result.status.value}")
    logger.info(f"Tiempo total: {result.load_time_seconds:.3f} segundos")

    if result.quality_metrics:
        logger.info(
            f"Filas: {result.quality_metrics.total_rows:,}, "
            f"Columnas: {result.quality_metrics.total_columns}"
        )
        if result.quality_metrics.warnings:
            for warning in result.quality_metrics.warnings[:5]:
                logger.warning(f"  - {warning}")
            if len(result.quality_metrics.warnings) > 5:
                logger.warning(f"  ... y {len(result.quality_metrics.warnings) - 5} advertencias más")

    if result.error_message:
        logger.error(f"Error: {result.error_message}")

    logger.info("=" * 80)

    if telemetry_context:
        status_str = "success" if result.status == LoadStatus.SUCCESS else "warning" if result.status == LoadStatus.PARTIAL_SUCCESS else "failure"

        # Registrar métricas
        if result.quality_metrics:
            telemetry_context.record_metric("loader", "rows_loaded", result.quality_metrics.total_rows)
            telemetry_context.record_metric("loader", "cols_loaded", result.quality_metrics.total_columns)
            telemetry_context.record_metric("loader", "file_size_mb", metadata.size_mb)

        telemetry_context.end_step("load_data", status_str, metadata=result.to_dict())

    return result


# ============================================================================
# FUNCIONES DE UTILIDAD ADICIONALES
# ============================================================================
def get_file_info(path: PathType) -> Dict[str, Any]:
    """
    Obtiene información detallada de un archivo sin cargarlo.

    Args:
        path: Ruta al archivo

    Returns:
        Diccionario con metadatos completos del archivo
    """
    result: Dict[str, Any] = {
        "path": str(path) if path else None,
        "exists": False,
        "size_mb": 0,
        "size_bytes": 0,
        "format": "UNKNOWN",
        "readable": False,
        "modified_time": None,
        "supported": False,
        "is_symlink": False,
        "error": None,
    }

    if not path:
        result["error"] = "Ruta vacía o None"
        return result

    try:
        path_obj = Path(path).resolve() if isinstance(path, str) else path.resolve()
        result["path"] = str(path_obj)
    except (TypeError, ValueError, OSError) as e:
        result["error"] = f"Ruta inválida: {e}"
        return result

    try:
        metadata = FileMetadata.from_path(path_obj)

        result.update({
            "exists": metadata.exists,
            "size_mb": round(metadata.size_mb, 4),
            "size_bytes": metadata.size_bytes,
            "format": metadata.format.value,
            "readable": metadata.readable,
            "is_symlink": metadata.is_symlink,
            "supported": metadata.format != FileFormat.UNKNOWN,
            "modified_time": metadata.modified_time.isoformat() if metadata.modified_time else None,
            "error": metadata.error,
        })

        # Información adicional
        if metadata.exists:
            result["extension"] = path_obj.suffix.lower()
            result["filename"] = path_obj.name
            result["parent_directory"] = str(path_obj.parent)

            # Verificar si excede límites
            if metadata.size_mb > MAX_FILE_SIZE_MB:
                result["size_warning"] = f"Excede límite de {MAX_FILE_SIZE_MB} MB"
            elif metadata.size_bytes == 0:
                result["size_warning"] = "Archivo vacío"

    except Exception as e:
        result["error"] = f"Error obteniendo información: {type(e).__name__}: {e}"
        logger.error(result["error"])

    return result


def validate_file_before_load(
    path: PathType,
    check_readable: bool = True,
    min_size_bytes: int = 1,
) -> Tuple[bool, List[str], List[str]]:
    """
    Valida un archivo exhaustivamente antes de intentar cargarlo.

    Args:
        path: Ruta al archivo
        check_readable: Si True, intenta abrir el archivo para verificar
        min_size_bytes: Tamaño mínimo requerido en bytes

    Returns:
        Tuple[bool, List[str], List[str]]: (es_valido, lista_errores, lista_advertencias)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Validar path
    if not path:
        errors.append("Ruta vacía o None")
        return False, errors, warnings

    try:
        path_obj = Path(path).resolve() if isinstance(path, str) else path.resolve()
    except Exception as e:
        errors.append(f"Ruta inválida: {e}")
        return False, errors, warnings

    # Verificar existencia
    try:
        if not path_obj.exists():
            errors.append(f"Archivo no existe: {path_obj}")

            # Sugerir archivos similares
            if path_obj.parent.exists():
                try:
                    similar = [
                        f.name for f in path_obj.parent.iterdir()
                        if f.is_file() and f.suffix == path_obj.suffix
                    ][:5]
                    if similar:
                        warnings.append(f"Archivos similares: {similar}")
                except Exception:
                    pass

            return False, errors, warnings
    except PermissionError:
        errors.append(f"Sin permisos para verificar: {path_obj}")
        return False, errors, warnings

    # Verificar que sea archivo
    if not path_obj.is_file():
        if path_obj.is_dir():
            errors.append(f"La ruta es un directorio, no un archivo: {path_obj}")
        else:
            errors.append(f"La ruta no es un archivo regular: {path_obj}")
        return False, errors, warnings

    # Obtener metadata
    try:
        metadata = FileMetadata.from_path(path_obj)
    except Exception as e:
        errors.append(f"Error obteniendo metadatos: {e}")
        return False, errors, warnings

    # Verificar formato
    if metadata.format == FileFormat.UNKNOWN:
        errors.append(f"Formato no soportado: {path_obj.suffix}")
        return False, errors, warnings

    # Verificar tamaño
    if metadata.size_bytes < min_size_bytes:
        if metadata.size_bytes == 0:
            errors.append("Archivo vacío (0 bytes)")
        else:
            errors.append(f"Archivo muy pequeño: {metadata.size_bytes} bytes (mínimo: {min_size_bytes})")
        return False, errors, warnings

    if metadata.size_mb > MAX_FILE_SIZE_MB:
        errors.append(
            f"Archivo demasiado grande: {metadata.size_mb:.2f} MB "
            f"(máximo: {MAX_FILE_SIZE_MB} MB)"
        )
        return False, errors, warnings

    # Advertencias de tamaño
    if metadata.size_mb > MAX_FILE_SIZE_MB * 0.8:
        warnings.append(
            f"Archivo grande ({metadata.size_mb:.2f} MB), "
            f"cerca del límite de {MAX_FILE_SIZE_MB} MB"
        )

    # Verificar lectura
    if check_readable:
        try:
            with open(path_obj, 'rb') as f:
                header = f.read(1024)  # Leer primeros 1KB

                # Verificar si parece ser binario corrupto
                null_count = header.count(b'\x00')
                if null_count > len(header) * 0.5:
                    warnings.append("Archivo parece contener muchos bytes nulos (posible corrupción)")

        except PermissionError:
            errors.append(f"Sin permisos de lectura: {path_obj}")
            return False, errors, warnings
        except IOError as e:
            errors.append(f"Error de I/O al leer: {e}")
            return False, errors, warnings

    # Verificaciones específicas por formato
    if metadata.format == FileFormat.PDF:
        try:
            with open(path_obj, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF'):
                    warnings.append("Archivo no comienza con firma PDF estándar")
        except Exception:
            pass

    elif metadata.format == FileFormat.EXCEL:
        try:
            with open(path_obj, 'rb') as f:
                header = f.read(4)
                # ZIP signature para xlsx, OLE signature para xls
                if not (header.startswith(b'PK') or header.startswith(b'\xD0\xCF\x11\xE0')):
                    warnings.append("Archivo no tiene firma Excel esperada")
        except Exception:
            pass

    return True, errors, warnings
