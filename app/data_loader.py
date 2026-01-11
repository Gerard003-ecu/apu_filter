# app/data_loader.py
"""
Este módulo actúa como la interfaz de frontera del sistema, responsable de la ingesta
robusta de datos desde fuentes heterogéneas (CSV, Excel, PDF). Su función no es solo
leer archivos, sino reducir la entropía de formato inicial, garantizando que solo
estructuras de datos normalizadas ingresen al pipeline de procesamiento.

Capacidades y Protocolos:
-------------------------
1. Detección Heurística de Formato:
   Implementa algoritmos de inferencia (`_detect_csv_delimiter`, `_detect_encoding`)
   para interpretar archivos sin metadatos explícitos, actuando como un traductor
   universal de dialectos de datos (encoding latino, separadores no estándar).

2. Minería de Datos en PDF:
   Utiliza `pdfplumber` para la extracción estructurada de tablas en documentos no
   estructurados, convirtiendo representaciones visuales en datos computables.

3. Telemetría de Ingesta:
   Integra `TelemetryContext` para registrar la "salud de entrada" (tamaño,
   integridad, tiempos de carga), proporcionando las primeras métricas para el
   Diagnóstico Termodinámico del sistema.

4. Validación de Integridad Física:
   Verifica la accesibilidad, permisos y consistencia de bytes (symlinks rotos,
   archivos bloqueados) antes de intentar el procesamiento lógico.
"""

import logging
import time
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

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
class HierarchyLevel(IntEnum):
    """
    Estratos de la Pirámide de Negocio.
    Fuente: topologia.md (Niveles de la Pirámide) [1]
    """

    ROOT = 0  # Proyecto Total (Ápice)
    STRATEGY = 1  # Capítulos (Pilares)
    TACTIC = 2  # APUs (Actividades)
    LOGISTICS = 3  # Insumos (Recursos Atómicos)


@dataclass
class HierarchicalData:
    """Contenedor de datos consciente de su posición topológica."""

    payload: pd.DataFrame
    level: HierarchyLevel
    lineage_hash: str


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
                        with open(path, "rb") as f:
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
    topological_analysis: Optional[Dict[str, Any]] = None
    homology_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el resultado a diccionario de forma segura.

        Returns:
            Dict con información del resultado de carga
        """
        result = {
            "status": self.status.value if self.status else "UNKNOWN",
            "load_time_seconds": round(self.load_time_seconds, 3)
            if self.load_time_seconds
            else 0,
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
            result["format"] = (
                self.file_metadata.format.value if self.file_metadata.format else "UNKNOWN"
            )
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

        # Manejar topological_analysis
        if self.topological_analysis:
            result["topological_analysis"] = self.topological_analysis

        # Manejar homology_analysis
        if self.homology_analysis:
            result["homology_analysis"] = self.homology_analysis

        # Información sobre los datos
        if self.data is None:
            result["has_data"] = False
            result["data_type"] = None
            result["data_shape"] = None
        elif isinstance(self.data, pd.DataFrame):
            result["has_data"] = not self.data.empty
            result["data_type"] = "DataFrame"
            result["data_shape"] = {
                "rows": len(self.data),
                "columns": len(self.data.columns),
            }
        elif isinstance(self.data, dict):
            result["has_data"] = any(
                isinstance(v, pd.DataFrame) and not v.empty for v in self.data.values()
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
# FUNCIONES AUXILIARES DE VALIDACIÓN Y TOPOLOGÍA
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
    problematic_chars = ["<", ">", ":", '"', "|", "?", "*"]
    path_str = str(path)
    for char in problematic_chars:
        if char in path_str and char != ":":  # ':' es válido en Windows para unidades
            # Solo advertir, no fallar (puede ser válido en algunos sistemas)
            logger.warning(
                f"La ruta contiene carácter potencialmente problemático: '{char}'"
            )

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
            raise IsADirectoryError(
                f"La ruta apunta a un directorio, no a un archivo: {resolved_path}"
            )
        else:
            raise ValueError(f"La ruta no apunta a un archivo regular: {resolved_path}")

    # Verificar permisos de lectura intentando abrir el archivo
    try:
        with open(resolved_path, "rb") as f:
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
        msg = (
            f"Archivo sospechosamente pequeño ({metadata.size_bytes} bytes): {metadata.path}"
        )
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


def _calculate_path_entropy(path: Path) -> float:
    """Calcula la entropía de Shannon de la ruta del archivo."""
    try:
        path_str = str(path)
        char_counts = {}
        total_chars = len(path_str)

        for char in path_str:
            char_counts[char] = char_counts.get(char, 0) + 1

        entropy = 0
        for count in char_counts.values():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)

        return entropy
    except:
        return 0.0


def _calculate_byte_entropy(data: bytes) -> float:
    """Calcula la entropía de una muestra de bytes."""
    if not data:
        return 0.0

    try:
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts[byte_counts > 0] / len(data)

        if len(probabilities) == 0:
            return 0.0

        entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = np.log2(len(probabilities))

        return entropy / max_entropy if max_entropy > 0 else 0.0
    except Exception:
        return 0.0


def _select_load_strategy(
    file_format: FileFormat,
    topological_analysis: Dict[str, Any]
) -> str:
    """Selecciona estrategia de carga basada en análisis topológico."""

    strategies = {
        FileFormat.CSV: {
            "default": "adaptive_csv",
            "low_entropy": "structured_csv",
            "high_depth": "robust_csv"
        },
        FileFormat.EXCEL: {
            "default": "full_excel",
            "large_file": "streaming_excel",
            "multi_sheet": "selective_excel"
        },
        FileFormat.PDF: {
            "default": "table_extraction",
            "high_entropy": "ocr_assisted",
            "structured": "direct_extraction"
        }
    }

    format_strategies = strategies.get(file_format, {"default": "standard"})

    # Heurísticas de selección
    if file_format == FileFormat.CSV:
        if topological_analysis.get("byte_entropy", 0.5) < 0.2:
            return format_strategies.get("low_entropy", "adaptive_csv")
        if topological_analysis.get("file_structure", {}).get("path_depth", 0) > 5:
            return format_strategies.get("high_depth", "robust_csv")

    return format_strategies.get("default", "standard")


def _analyze_dataframe_quality(
    df: pd.DataFrame,
    include_sample_data: bool = False,
) -> DataQualityMetrics:
    """
    Analiza la calidad usando métricas topológicas y teoría de información.

    Args:
        df: DataFrame a analizar
        include_sample_data: Si True, incluye muestra de datos problemáticos

    Returns:
        DataQualityMetrics: Métricas de calidad con dimensión topológica
    """
    metrics = DataQualityMetrics()

    if df is None or not isinstance(df, pd.DataFrame):
        metrics.add_warning("Entrada inválida para análisis de calidad")
        return metrics

    if df.empty:
        metrics.add_warning("DataFrame está vacío")
        return metrics

    # Dimensiones básicas (cardinalidad)
    metrics.total_rows = len(df)
    metrics.total_columns = len(df.columns)

    # Análisis de completitud (dimensión 0 - puntos)
    try:
        null_matrix = df.isnull()
        metrics.null_cells = int(null_matrix.sum().sum())
        total_cells = metrics.total_rows * metrics.total_columns

        if total_cells > 0:
            metrics.null_percentage = (metrics.null_cells / total_cells) * 100

            # Entropía de la distribución de nulos por columna
            null_by_col = null_matrix.sum(axis=0)
            if null_by_col.sum() > 0:
                p = null_by_col / null_by_col.sum()
                entropy_nulls = -np.sum(p * np.log2(p + 1e-10))
                metrics.add_warning(f"Entropía de distribución de nulos: {entropy_nulls:.3f}")
    except Exception as e:
        logger.warning(f"Error en análisis de nulos: {e}")

    # Análisis de dimensionalidad (estructura de columnas)
    try:
        # Cohesión de tipos de datos (dimensión 1 - líneas)
        type_groups = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            type_groups.setdefault(dtype, []).append(str(col))

        metrics.data_types = {dtype: len(cols) for dtype, cols in type_groups.items()}

        # Métrica de heterogeneidad de tipos
        if len(type_groups) > 1:
            heterogeneity = 1 - (max(len(c) for c in type_groups.values()) / metrics.total_columns)
            if heterogeneity > 0.7:
                metrics.add_warning(f"Alta heterogeneidad de tipos: {heterogeneity:.2f}")
    except Exception as e:
        logger.warning(f"Error en análisis de tipos: {e}")

    # Análisis de redundancia (dimensión 2 - superficies)
    try:
        # Detectar columnas duplicadas usando similitud de contenido
        duplicate_pairs = []
        columns = list(df.columns)

        # Limitación para evitar complejidad O(N^2) en dataframes anchos
        max_cols_check = min(len(columns), 100)

        for i in range(max_cols_check):
            for j in range(i + 1, max_cols_check):
                col1, col2 = columns[i], columns[j]
                try:
                    # Comparar series ignorando nulos
                    series1 = df[col1].dropna()
                    series2 = df[col2].dropna()

                    if len(series1) > 10 and len(series2) > 10:
                        # Coeficiente de correlación para numéricos
                        if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
                            if len(series1) == len(series2): # Simplificación para evitar alineación
                                corr = series1.corr(series2)
                                if abs(corr) > 0.95:
                                    duplicate_pairs.append((col1, col2, f"corr={corr:.3f}"))

                        # Similitud de strings para categóricos
                        elif pd.api.types.is_string_dtype(series1) and pd.api.types.is_string_dtype(series2):
                            sample1 = series1.sample(min(100, len(series1))).astype(str).values
                            sample2 = series2.sample(min(100, len(series2))).astype(str).values
                            # Comparación simple de conjuntos o valores directos
                            matches = sum(1 for a, b in zip(sample1, sample2) if a == b)
                            if matches / len(sample1) > 0.9:
                                duplicate_pairs.append((col1, col2, f"match={matches/len(sample1):.2f}"))
                except:
                    continue

        if duplicate_pairs:
            metrics.duplicate_columns = [f"{a} approx {b} ({reason})" for a, b, reason in duplicate_pairs[:5]]
            metrics.add_warning(f"Redundancia detectada en {len(duplicate_pairs)} pares de columnas")
    except Exception as e:
        logger.warning(f"Error en análisis de redundancia: {e}")

    # Análisis de integridad estructural
    try:
        # Columnas completamente nulas (dimensión colapsada)
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            metrics.columns_with_all_nulls = [str(c) for c in all_null_cols]
            metrics.add_warning(f"{len(all_null_cols)} columnas completamente nulas")

        # Columnas con baja variación (casi constantes)
        low_variance_cols = []
        for col in df.columns:
            if df[col].nunique(dropna=True) == 1 and len(df[col].dropna()) > 10:
                low_variance_cols.append(col)

        if low_variance_cols:
            metrics.add_warning(f"{len(low_variance_cols)} columnas con variación mínima")
    except Exception as e:
        logger.warning(f"Error en análisis de integridad: {e}")

    # Métricas de memoria con perspectiva topológica
    try:
        memory_bytes = df.memory_usage(deep=True, index=True).sum()
        metrics.memory_usage_mb = memory_bytes / (1024 * 1024)

        # Densidad de información (bits por celda no nula)
        total_cells = metrics.total_rows * metrics.total_columns
        non_null_cells = total_cells - metrics.null_cells
        if non_null_cells > 0:
            info_density = (memory_bytes * 8) / non_null_cells  # bits por celda
            if info_density > 100:  # Empírico
                metrics.add_warning(f"Alta densidad de información: {info_density:.1f} bits/celda")
    except Exception as e:
        logger.warning(f"Error en cálculo de memoria: {e}")
        # Fallback simple
        try:
             metrics.memory_usage_mb = (df.size * 8) / (1024 * 1024)
        except:
             metrics.memory_usage_mb = 0.0

    # Advertencias dimensionales
    if metrics.total_rows > MAX_ROWS_WARNING:
        metrics.add_warning(f"Cardinalidad alta: {metrics.total_rows:,} observaciones")

    if metrics.total_columns > MAX_COLS_WARNING:
        metrics.add_warning(f"Dimensionalidad alta: {metrics.total_columns} variables")

    if metrics.null_percentage > MAX_NULL_PERCENTAGE_WARNING:
        metrics.add_warning(f"Baja completitud: {metrics.null_percentage:.1f}% celdas nulas")

    # Análisis de conectividad (grafos de dependencia implícitos)
    try:
        if metrics.total_columns > 2 and metrics.total_rows > 10:
            # Matriz de correlación parcial para detectar relaciones
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 3:
                # Usar solo una muestra para evitar lentitud
                sample_corr = df[numeric_cols].sample(min(500, len(df))).corr().abs()

                # Contar correlaciones fuertes (> 0.8)
                strong_corrs = ((sample_corr > 0.8) & (sample_corr < 1.0)).sum().sum() / 2
                if strong_corrs > len(numeric_cols):
                    metrics.add_warning(f"Alta conectividad lineal: {int(strong_corrs)} correlaciones fuertes")
    except:
        pass

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
    (Mantiene implementación original robusta)
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
                "_".join(
                    str(level) for level in col if pd.notna(level) and str(level).strip()
                )
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
                if not col_str or col_str.lower() in ("nan", "none", "null"):
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
            df_clean = df_clean.dropna(how="all")
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

    return df_clean


def _detect_csv_delimiter(
    path: Path,
    sample_size: int = 20,
    encodings_to_try: Optional[List[str]] = None,
    max_line_length: int = 10000,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Detecta delimitador usando teoría de información y topología de datos.

    Args:
        path: Ruta al archivo CSV
        sample_size: Número de líneas a analizar
        encodings_to_try: Lista de encodings a probar
        max_line_length: Longitud máxima de línea a considerar

    Returns:
        Tuple[delimitador, encoding]: Delimitador y encoding detectados usando entropía de Shannon
    """
    if encodings_to_try is None:
        encodings_to_try = DEFAULT_ENCODINGS.copy()

    for encoding in encodings_to_try:
        try:
            lines = []
            with open(path, "r", encoding=encoding, errors="replace") as f:
                for _ in range(sample_size):
                    line = f.readline()
                    if not line:
                        break
                    if len(line) > max_line_length:
                        line = line[:max_line_length]
                    lines.append(line.strip())

            if len(lines) < 3:  # Mínimo para análisis estadístico
                continue

            # Métrica de entropía de Shannon para cada delimitador
            delimiter_scores = {}

            for delim in COMMON_DELIMITERS:
                # Calcular distribución de campos por línea
                field_counts = []
                for line in lines:
                    if not line or line.startswith(('#', '//', '--')):
                        continue

                    # Contar ocurrencias no vacías del delimitador
                    parts = [p for p in line.split(delim) if p.strip()]
                    if len(parts) > 1:  # Solo considerar líneas con al menos 2 campos
                        field_counts.append(len(parts))

                if not field_counts:
                    continue

                # Calcular métricas estadísticas
                mean_fields = np.mean(field_counts)
                std_fields = np.std(field_counts)

                # Coeficiente de variación (inverso = consistencia)
                if mean_fields > 0:
                    cv = std_fields / mean_fields
                    consistency_score = 1 / (1 + cv)
                else:
                    consistency_score = 0

                # Entropía de la distribución de campos
                unique_counts, count_freq = np.unique(field_counts, return_counts=True)
                prob_dist = count_freq / len(field_counts)
                entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))

                # Máxima entropía posible para esta distribución
                max_entropy = np.log2(len(unique_counts)) if len(unique_counts) > 1 else 1

                # Normalizar entropía (0-1)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

                # Puntuación compuesta basada en teoría de información
                # 1. Alta consistencia (baja varianza)
                # 2. Baja entropía (estructura regular) - Invertimos entropía para premiar orden
                # 3. Suficiente cobertura (porcentaje de líneas con el delimitador)
                coverage = len(field_counts) / len(lines)

                # Un buen delimitador produce un número fijo de columnas => Baja Entropía
                structure_score = 1.0 - normalized_entropy

                score = (consistency_score * 0.4 +
                        structure_score * 0.3 +
                        coverage * 0.3) * mean_fields

                delimiter_scores[delim] = {
                    'score': score,
                    'consistency': consistency_score,
                    'entropy': normalized_entropy,
                    'structure_score': structure_score,
                    'coverage': coverage,
                    'mean_fields': mean_fields
                }

            # Seleccionar mejor delimitador usando criterio topológico
            if delimiter_scores:
                # Filtrar delimitadores con cobertura mínima
                valid_delims = {d: s for d, s in delimiter_scores.items()
                              if s['coverage'] > 0.6 and s['mean_fields'] > 1}

                if valid_delims:
                    # Seleccionar por score compuesto
                    best_delim = max(valid_delims.items(),
                                   key=lambda x: x[1]['score'])

                    logger.debug(
                        f"Delimitador óptimo: '{best_delim[0]}' "
                        f"(score: {best_delim[1]['score']:.3f}, "
                        f"consistencia: {best_delim[1]['consistency']:.3f}, "
                        f"entropía: {best_delim[1]['entropy']:.3f}, "
                        f"encoding: {encoding})"
                    )

                    # Verificación adicional: el delimitador no debe ser contenido comúnmente en datos
                    sample_data = ' '.join(lines[:5])
                    if best_delim[0] in [';', ',']:
                        # Verificar que no sea un número con separador decimal
                        if re.search(r'\d' + re.escape(best_delim[0]) + r'\d', sample_data):
                            logger.debug(f"Delimitador '{best_delim[0]}' puede ser separador decimal")
                            continue

                    return best_delim[0], encoding

        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.debug(f"Error con encoding {encoding}: {e}")
            continue

    # Fallback: método de frecuencia simple
    logger.debug("Usando método de detección por frecuencia")
    for encoding in encodings_to_try:
        try:
            with open(path, "r", encoding=encoding, errors="replace") as f:
                content = f.read(5000)

            freq = {d: content.count(d) for d in COMMON_DELIMITERS}
            if freq:
                best_delim = max(freq.items(), key=lambda x: x[1])
                if best_delim[1] > 10:  # Mínimo umbral de ocurrencias
                    return best_delim[0], encoding
        except:
            continue

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
    safe_kwargs = {
        k: v for k, v in kwargs.items() if k not in ("sep", "encoding", "delimiter")
    }

    if "sep" in kwargs or "delimiter" in kwargs:
        logger.warning(
            "Parámetros 'sep' o 'delimiter' en kwargs serán ignorados. Use el parámetro 'sep' directo."
        )

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
            delimiters_to_try = [detected_delim] + [
                d for d in COMMON_DELIMITERS if d != detected_delim
            ]
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
                    on_bad_lines="warn",  # Python 3.9+ pandas
                    **safe_kwargs,
                )

                # Verificar que la carga tenga sentido
                # Si solo hay una columna, probablemente el delimitador es incorrecto
                if len(df.columns) == 1 and len(df) > 0:
                    # Verificar si la única columna contiene el delimitador esperado
                    col_name = str(df.columns[0])
                    if any(d in col_name for d in COMMON_DELIMITERS if d != delim):
                        attempts_log.append(
                            f"{attempt_info}: solo 1 columna, posible delimitador incorrecto"
                        )
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
                    f"Hoja '{sheet}' no encontrada. Hojas disponibles: {available_sheets}"
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
                    f"Rango válido: 0-{len(available_sheets) - 1}"
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
            if path.suffix.lower() == ".xlsx":
                try:
                    from openpyxl import load_workbook

                    # Verificar que la contraseña funciona
                    wb = load_workbook(path, read_only=True, password=password)
                    wb.close()
                except ImportError:
                    warnings_list.append(
                        "openpyxl no instalado, no se puede usar contraseña"
                    )
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
                actual_sheet_name = (
                    available_sheets[sheet_name]
                    if sheet_name < len(available_sheets)
                    else str(sheet_name)
                )
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
        error_msg = "pdfplumber no está instalado. Instale con: pip install pdfplumber"
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
            open_kwargs["password"] = password

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
                        warnings_list.append(
                            f"Página {p} ignorada (fuera de rango 0-{total_pages - 1})"
                        )

                if not pages_to_process_indices:
                    error_msg = f"Ninguna página válida especificada. Rango válido: 0-{total_pages - 1}"
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
                    error_msg = (
                        f"Página inicial {page_range.start} excede total ({total_pages})"
                    )
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
                                    max_cols = max(
                                        len(row) if row else 0 for row in table[1:]
                                    )
                                    while len(headers) < max_cols:
                                        headers.append(f"Col_{len(headers)}")

                                df_table = pd.DataFrame(
                                    table[1:],
                                    columns=headers[: len(table[0]) if table[0] else 0],
                                )
                            else:
                                df_table = pd.DataFrame(table)

                            if not df_table.empty:
                                # Limpiar datos de la tabla
                                df_table = df_table.replace("", pd.NA)
                                df_table = df_table.dropna(how="all")

                                if not df_table.empty:
                                    # Agregar metadata de origen
                                    df_table["_source_page"] = page_num
                                    df_table["_source_table"] = table_idx + 1

                                    tables_data.append(df_table)
                                    logger.debug(
                                        f"Página {page_num}, tabla {table_idx + 1}: "
                                        f"{df_table.shape[0]} filas × {df_table.shape[1]} columnas"
                                    )

                        except Exception as e:
                            error_info = (
                                f"Error en tabla {table_idx + 1} de página {page_num}: {e}"
                            )
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
        data_columns = [c for c in all_columns if not c.startswith("_source_")]

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
    Función factory con patrón estratégico y análisis topológico previo.

    Args:
        path: Ruta al archivo
        format_hint: Sugerencia de formato
        validate_only: Si True, solo valida el archivo
        telemetry_context: Contexto de telemetría
        **kwargs: Argumentos específicos del formato

    Returns:
        LoadResult con análisis topológico integrado
    """
    start_time = time.time()
    topological_analysis = {}

    logger.info("=" * 80)
    logger.info(f"Análisis topológico iniciado para: {path}")
    logger.info("=" * 80)

    if telemetry_context:
        telemetry_context.start_step(
            "load_data",
            {
                "path": str(path),
                "format_hint": format_hint,
                "phase": "topological_analysis"
            }
        )

    # Validación topológica previa
    try:
        validated_path = _validate_file_path(path)
        metadata = FileMetadata.from_path(validated_path)

        # Análisis de estructura de archivo
        topological_analysis["file_structure"] = {
            "path_depth": len(validated_path.parts),
            "extension": validated_path.suffix,
            "name_complexity": len(validated_path.stem) / 20,  # Normalizado
            "directory_entropy": _calculate_path_entropy(validated_path)
        }

    except Exception as e:
        error_msg = f"Fallo en análisis topológico inicial: {type(e).__name__}: {e}"
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
            telemetry_context.record_error("topological_analysis", error_msg)
            telemetry_context.end_step("load_data", "failure", {
                **result.to_dict(),
                "topological_analysis": topological_analysis
            })

        return result

    # Determinar formato con heurística mejorada
    file_format = metadata.format

    if format_hint:
        format_hint_normalized = format_hint.strip().upper()

        # Matriz de confusión de formatos (experiencia previa)
        format_confusion_matrix = {
            "CSV": {"confusions": ["TSV", "TXT"], "confidence": 0.9},
            "EXCEL": {"confusions": ["CSV", "PDF"], "confidence": 0.8},
            "PDF": {"confusions": ["EXCEL"], "confidence": 0.7}
        }

        format_mapping = {
            "CSV": FileFormat.CSV,
            "TXT": FileFormat.CSV,
            "TSV": FileFormat.CSV,
            "EXCEL": FileFormat.EXCEL,
            "XLSX": FileFormat.EXCEL,
            "XLS": FileFormat.EXCEL,
            "XLSM": FileFormat.EXCEL,
            "XLSB": FileFormat.EXCEL,
            "PDF": FileFormat.PDF,
        }

        if format_hint_normalized in format_mapping:
            suggested_format = format_mapping[format_hint_normalized]

            # Verificar si hay conflicto con detección automática
            if suggested_format != metadata.format:
                if metadata.format != FileFormat.UNKNOWN:
                    # Análisis de conflicto
                    topological_analysis["format_conflict"] = {
                        "detected": metadata.format.value,
                        "suggested": suggested_format.value,
                        "confidence": format_confusion_matrix.get(
                            suggested_format.value, {}
                        ).get("confidence", 0.5)
                    }

                    logger.warning(
                        f"Conflicto de formato: detectado={metadata.format.value}, "
                        f"sugerido={suggested_format.value}"
                    )

                # Priorizar sugerencia con registro
                file_format = suggested_format
                topological_analysis["format_override"] = True

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
            ".json": "Use json.load() o pd.read_json()",
            ".xml": "Use pd.read_xml() o lxml",
            ".parquet": "Use pd.read_parquet()",
            ".feather": "Use pd.read_feather()",
            ".pickle": "Use pd.read_pickle()",
            ".pkl": "Use pd.read_pickle()",
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
        result.topological_analysis = topological_analysis

        if telemetry_context:
            telemetry_context.record_error("load_data", error_msg)
            telemetry_context.end_step("load_data", "failure", {
                **result.to_dict(),
                "topological_analysis": topological_analysis
            })
        return result

    # Validación de integridad física con métricas topológicas
    try:
        size_warnings = _validate_file_size(metadata, allow_empty=validate_only)

        # Análisis de distribución de bytes (entropía del archivo)
        if metadata.size_bytes > 100:  # Solo para archivos no triviales
            try:
                with open(validated_path, 'rb') as f:
                    sample = f.read(min(1000, metadata.size_bytes))
                    byte_entropy = _calculate_byte_entropy(sample)
                    topological_analysis["byte_entropy"] = byte_entropy

                    if byte_entropy < 0.1:  # Archivo muy estructurado/constante
                        topological_analysis["low_entropy_warning"] = (
                            f"Entropía de bytes baja ({byte_entropy:.3f}), "
                            f"posible archivo binario o altamente estructurado"
                        )
            except:
                pass

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
            telemetry_context.record_error("file_validation", str(e))
            telemetry_context.end_step("load_data", "failure", {
                **result.to_dict(),
                "topological_analysis": topological_analysis
            })

        return result

    # Si solo validación, retornar análisis topológico
    if validate_only:
        quality_metrics = DataQualityMetrics()
        quality_metrics.warnings = size_warnings

        result = LoadResult(
            status=LoadStatus.SUCCESS,
            data=None,
            file_metadata=metadata,
            quality_metrics=quality_metrics,
            load_time_seconds=time.time() - start_time,
        )

        # Inyectar análisis topológico
        result.topological_analysis = topological_analysis

        if telemetry_context:
            telemetry_context.end_step("load_data", "success", {
                **result.to_dict(),
                "topological_analysis": topological_analysis
            })

        return result

    # Selección de estrategia de carga basada en topología
    load_strategy = _select_load_strategy(file_format, topological_analysis)

    # Filtrar kwargs según formato
    csv_params = {
        "sep",
        "encoding",
        "auto_detect",
        "max_retries",
        "header",
        "names",
        "index_col",
        "usecols",
        "dtype",
        "skiprows",
        "nrows",
        "na_values",
    }
    excel_params = {
        "sheet_name",
        "password",
        "header",
        "names",
        "index_col",
        "usecols",
        "dtype",
        "skiprows",
        "nrows",
        "na_values",
    }
    pdf_params = {"page_range", "pages", "table_settings", "max_pages", "password"}

    try:
        # Cargar usando estrategia seleccionada
        if file_format == FileFormat.CSV:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in csv_params}

            # Ajustar parámetros basados en análisis topológico
            if "byte_entropy" in topological_analysis:
                if topological_analysis["byte_entropy"] < 0.2:
                    filtered_kwargs.setdefault("encoding", "utf-16")

            result = load_from_csv(validated_path, **filtered_kwargs)

        elif file_format == FileFormat.EXCEL:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in excel_params}
            result = load_from_xlsx(validated_path, **filtered_kwargs)

        elif file_format == FileFormat.PDF:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in pdf_params}
            result = load_from_pdf(validated_path, **filtered_kwargs)

        else:
            error_msg = f"Estrategia no implementada para formato {file_format.value}"
            # Manejar formato desconocido de manera similar a original
            logger.error(error_msg)
            result = LoadResult(
                status=LoadStatus.FAILED,
                data=None,
                file_metadata=metadata,
                quality_metrics=None,
                load_time_seconds=time.time() - start_time,
                error_message=error_msg,
            )
            # return result (handled below)

        # Enriquecer resultado con análisis topológico
        result.topological_analysis = topological_analysis

        # Añadir métricas de carga al análisis
        if result.quality_metrics and result.file_metadata and result.file_metadata.size_bytes > 0:
            topological_analysis["load_metrics"] = {
                "compression_ratio": (
                    result.file_metadata.size_mb / (result.quality_metrics.memory_usage_mb + 1e-10)
                ),
                "null_density": result.quality_metrics.null_percentage / 100,
                "dimensional_efficiency": (
                    result.quality_metrics.total_rows * result.quality_metrics.total_columns
                ) / (result.file_metadata.size_bytes + 1e-10)
            }

    except Exception as e:
        error_msg = f"Error en estrategia de carga {load_strategy}: {type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)

        result = LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg,
        )

        result.topological_analysis = topological_analysis

    # Logging final con perspectiva topológica
    logger.info("=" * 80)
    logger.info(f"Carga completada - Estado: {result.status.value}")
    logger.info(f"Estrategia: {load_strategy}")
    logger.info(f"Análisis topológico: {len(topological_analysis)} métricas")
    logger.info(f"Tiempo total: {result.load_time_seconds:.3f} segundos")

    if hasattr(result, 'topological_analysis') and result.topological_analysis:
        for key, value in list(result.topological_analysis.items())[:3]:
            logger.debug(f"  {key}: {value}")

    logger.info("=" * 80)

    if telemetry_context:
        status_str = (
            "success" if result.status == LoadStatus.SUCCESS
            else "warning" if result.status == LoadStatus.PARTIAL_SUCCESS
            else "failure"
        )

        telemetry_context.end_step("load_data", status_str, {
            **result.to_dict(),
            "topological_analysis": topological_analysis,
            "load_strategy": load_strategy
        })

    return result


# ============================================================================
# FUNCIONES DE CARGA JERÁRQUICA
# ============================================================================
def load_data_with_hierarchy(
    path: str,
    level: HierarchyLevel,
    preserve_topology: bool = True,
    **kwargs,
) -> LoadResult:
    """
    Carga datos con preservación de isomorfismos topológicos entre niveles.

    Args:
        path: Ruta al archivo
        level: Nivel en la jerarquía
        preserve_topology: Si True, mantiene relaciones topológicas
        **kwargs: Argumentos adicionales para load_data

    Returns:
        LoadResult con estructura jerárquica y análisis de homología
    """
    # 1. Carga estándar con análisis topológico
    result = load_data(path, **kwargs)

    if result.status != LoadStatus.SUCCESS:
        return result

    # 2. Análisis de homología entre niveles
    homology_analysis = {
        "source_level": level.name,
        "dimensionality": result.quality_metrics.total_columns if result.quality_metrics else 0,
        "cardinality": result.quality_metrics.total_rows if result.quality_metrics else 0,
        "topological_invariants": {}
    }

    # 3. Preservación de estructura topológica
    if preserve_topology and result.data is not None:
        if isinstance(result.data, pd.DataFrame):
            df = result.data

            # Calcular invariantes topológicos
            homology_analysis["topological_invariants"] = {
                "connected_components": _count_connected_components(df),
                "cycles": _detect_data_cycles(df),
                "boundary_matrix_rank": _calculate_boundary_rank(df)
            }

            # Inyectar metadatos topológicos
            if not hasattr(df, 'attrs'):
                df.attrs = {}

            df.attrs.update({
                "hierarchy_level": level.value,
                "level_name": level.name,
                "topological_invariants": homology_analysis["topological_invariants"],
                "is_foundation": (level == HierarchyLevel.LOGISTICS),
                "parent_level": max(level.value - 1, 0) if level.value > 0 else None
            })

            # Preservar relaciones de adjacencia (para grafos de datos)
            if level.value > 0:
                df.attrs["child_relations"] = _extract_child_relations(df, level)

    # 4. Enriquecer resultado con análisis de homología
    result.homology_analysis = homology_analysis

    # 5. Validación de coherencia dimensional
    if result.quality_metrics:
        expected_dimensions = {
            HierarchyLevel.ROOT: (1, 50),        # Raíz: pocas dimensiones clave
            HierarchyLevel.STRATEGY: (3, 100),   # Estrategia: dimensiones moderadas
            HierarchyLevel.TACTIC: (10, 500),    # Táctica: alta dimensionalidad
            HierarchyLevel.LOGISTICS: (5, 200)   # Logística: dimensiones específicas
        }

        min_dim, max_dim = expected_dimensions.get(level, (1, 1000))
        actual_dim = result.quality_metrics.total_columns

        if actual_dim < min_dim:
            result.add_warning(
                f"Dimensionalidad baja para nivel {level.name}: "
                f"{actual_dim} columnas (esperado: {min_dim}+)"
            )
        elif actual_dim > max_dim:
            result.add_warning(
                f"Dimensionalidad alta para nivel {level.name}: "
                f"{actual_dim} columnas (esperado: ≤{max_dim})"
            )

    logger.info(
        f"Carga jerárquica completada - Nivel: {level.name}, "
        f"Invariantes: {homology_analysis['topological_invariants']}"
    )

    return result


def _count_connected_components(df: pd.DataFrame) -> int:
    """
    Calcula componentes conexos en el espacio de datos.
    Usa clustering para identificar grupos desconectados.
    """
    try:
        if len(df) < 2 or len(df.columns) < 2:
            return 1

        # Seleccionar columnas numéricas para análisis
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return 1

        # Muestreo para eficiencia
        sample_size = min(1000, len(df))
        sample = df[numeric_cols].sample(sample_size).fillna(0)

        # Intentar importar sklearn de forma segura
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(sample)
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(scaled)

            # Contar clusters no ruido
            n_components = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

            return max(n_components, 1)
        except ImportError:
            # Fallback simple basado en correlación si sklearn no está disponible
            corr = sample.corr().abs()
            # Umbral de conexión
            adj = (corr > 0.1).astype(int).values
            # BFS simple para contar componentes
            n = len(adj)
            visited = [False] * n
            count = 0
            for i in range(n):
                if not visited[i]:
                    count += 1
                    queue = [i]
                    visited[i] = True
                    while queue:
                        u = queue.pop(0)
                        for v in range(n):
                            if adj[u][v] and not visited[v]:
                                visited[v] = True
                                queue.append(v)
            return count

    except Exception:
        return 1


def _detect_data_cycles(df: pd.DataFrame) -> int:
    """
    Detecta ciclos en relaciones de datos (para grafos implícitos).
    """
    try:
        if len(df.columns) < 3:
            return 0

        # Buscar relaciones circulares en columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 3:
            return 0

        # Matriz de correlación
        corr_matrix = df[numeric_cols].corr().abs()

        # Detectar triángulos de alta correlación (ciclos de longitud 3)
        cycles = 0
        n = len(numeric_cols)

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if (corr_matrix.iloc[i, j] > 0.8 and
                        corr_matrix.iloc[j, k] > 0.8 and
                        corr_matrix.iloc[k, i] > 0.8):
                        cycles += 1

        return cycles

    except Exception:
        return 0


def _calculate_boundary_rank(df: pd.DataFrame) -> int:
    """
    Calcula el rango de la matriz de borde (concepto de homología).
    """
    try:
        # Matriz binaria de presencia/ausencia (simplificada)
        binary_matrix = df.notnull().astype(int).values

        if binary_matrix.size == 0:
            return 0

        # Rango de la matriz (dimensión del espacio columna)
        rank = np.linalg.matrix_rank(binary_matrix)

        return rank

    except Exception:
        return 0


def _extract_child_relations(df: pd.DataFrame, level: HierarchyLevel) -> Dict:
    """
    Extrae relaciones padre-hijo basadas en estructura de datos.
    """
    relations = {
        "potential_children": 0,
        "foreign_key_candidates": [],
        "hierarchical_columns": []
    }

    try:
        # Buscar columnas que puedan ser claves foráneas
        for col in df.columns:
            col_str = str(col).lower()

            # Heurísticas para identificar relaciones
            if any(term in col_str for term in ['id', 'code', 'key', 'ref', 'parent']):
                relations["foreign_key_candidates"].append(col)

            if any(term in col_str for term in ['level', 'hierarchy', 'tier', 'grade']):
                relations["hierarchical_columns"].append(col)

        # Estimar número de hijos potenciales
        if relations["foreign_key_candidates"]:
            for col in relations["foreign_key_candidates"][:2]:
                unique_values = df[col].nunique()
                if unique_values > 1:
                    relations["potential_children"] = max(
                        relations["potential_children"],
                        unique_values
                    )

    except Exception:
        pass

    return relations


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

        result.update(
            {
                "exists": metadata.exists,
                "size_mb": round(metadata.size_mb, 4),
                "size_bytes": metadata.size_bytes,
                "format": metadata.format.value,
                "readable": metadata.readable,
                "is_symlink": metadata.is_symlink,
                "supported": metadata.format != FileFormat.UNKNOWN,
                "modified_time": metadata.modified_time.isoformat()
                if metadata.modified_time
                else None,
                "error": metadata.error,
            }
        )

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
                        f.name
                        for f in path_obj.parent.iterdir()
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
            errors.append(
                f"Archivo muy pequeño: {metadata.size_bytes} bytes (mínimo: {min_size_bytes})"
            )
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
            with open(path_obj, "rb") as f:
                header = f.read(1024)  # Leer primeros 1KB

                # Verificar si parece ser binario corrupto
                null_count = header.count(b"\x00")
                if null_count > len(header) * 0.5:
                    warnings.append(
                        "Archivo parece contener muchos bytes nulos (posible corrupción)"
                    )

        except PermissionError:
            errors.append(f"Sin permisos de lectura: {path_obj}")
            return False, errors, warnings
        except IOError as e:
            errors.append(f"Error de I/O al leer: {e}")
            return False, errors, warnings

    # Verificaciones específicas por formato
    if metadata.format == FileFormat.PDF:
        try:
            with open(path_obj, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF"):
                    warnings.append("Archivo no comienza con firma PDF estándar")
        except Exception:
            pass

    elif metadata.format == FileFormat.EXCEL:
        try:
            with open(path_obj, "rb") as f:
                header = f.read(4)
                # ZIP signature para xlsx, OLE signature para xls
                if not (header.startswith(b"PK") or header.startswith(b"\xd0\xcf\x11\xe0")):
                    warnings.append("Archivo no tiene firma Excel esperada")
        except Exception:
            pass

    return True, errors, warnings
