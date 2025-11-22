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
import numpy as np
import pdfplumber

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
DEFAULT_ENCODINGS = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']

# Delimitadores comunes para CSV
COMMON_DELIMITERS = [';', ',', '\t', '|']

# Extensiones soportadas por tipo
SUPPORTED_CSV_EXTENSIONS = {'.csv', '.txt', '.tsv'}
SUPPORTED_EXCEL_EXTENSIONS = {'.xlsx', '.xls', '.xlsm', '.xlsb'}
SUPPORTED_PDF_EXTENSIONS = {'.pdf'}

ALL_SUPPORTED_EXTENSIONS = (
    SUPPORTED_CSV_EXTENSIONS | 
    SUPPORTED_EXCEL_EXTENSIONS | 
    SUPPORTED_PDF_EXTENSIONS
)

# Configuración de validación
MAX_DUPLICATE_COLUMNS_ALLOWED = 0  # No permitir columnas duplicadas
MAX_NULL_PERCENTAGE_WARNING = 50  # Advertir si más del 50% de datos son nulos
MIN_VALID_ROWS_PERCENTAGE = 1  # Al menos 1% de filas deben ser válidas


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
    
    @classmethod
    def from_path(cls, path: Path) -> 'FileMetadata':
        """Crea metadatos desde una ruta de archivo"""
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        size_mb = size_bytes / (1024 * 1024)
        readable = path.is_file() and exists
        modified_time = None
        
        if exists:
            try:
                modified_time = datetime.fromtimestamp(path.stat().st_mtime)
            except Exception as e:
                logger.warning(f"No se pudo obtener tiempo de modificación: {e}")
        
        # Determinar formato
        extension = path.suffix.lower()
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
            modified_time=modified_time
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
            "warnings": self.warnings
        }


@dataclass
class LoadResult:
    """Resultado de una operación de carga"""
    status: LoadStatus
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame], None]
    file_metadata: FileMetadata
    quality_metrics: Optional[DataQualityMetrics]
    load_time_seconds: float
    encoding_used: Optional[str] = None
    delimiter_used: Optional[str] = None
    sheets_loaded: Optional[List[str]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario"""
        return {
            "status": self.status.value,
            "file_path": str(self.file_metadata.path),
            "file_size_mb": self.file_metadata.size_mb,
            "format": self.file_metadata.format.value,
            "load_time_seconds": round(self.load_time_seconds, 3),
            "encoding_used": self.encoding_used,
            "delimiter_used": self.delimiter_used,
            "sheets_loaded": self.sheets_loaded,
            "quality_metrics": self.quality_metrics.to_dict() if self.quality_metrics else None,
            "error_message": self.error_message,
            "has_data": self.data is not None
        }


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
        Path: Ruta normalizada
        
    Raises:
        ValueError: Si la ruta es inválida
        FileNotFoundError: Si el archivo no existe
    """
    if not path:
        raise ValueError("La ruta del archivo no puede estar vacía")
    
    try:
        path = Path(path).resolve()
    except Exception as e:
        raise ValueError(f"Ruta inválida '{path}': {e}")
    
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    
    if not path.is_file():
        raise ValueError(f"La ruta no apunta a un archivo: {path}")
    
    # Validar permisos de lectura
    if not path.stat().st_mode & 0o444:  # Check read permission
        raise PermissionError(f"No hay permisos de lectura para: {path}")
    
    return path


def _validate_file_size(metadata: FileMetadata) -> None:
    """
    Valida que el tamaño del archivo sea razonable.
    
    Args:
        metadata: Metadatos del archivo
        
    Raises:
        ValueError: Si el archivo es demasiado grande
    """
    if metadata.size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"Archivo demasiado grande: {metadata.size_mb:.2f} MB. "
            f"Máximo permitido: {MAX_FILE_SIZE_MB} MB"
        )
    
    if metadata.size_bytes == 0:
        logger.warning(f"El archivo está vacío: {metadata.path}")


def _analyze_dataframe_quality(df: pd.DataFrame) -> DataQualityMetrics:
    """
    Analiza la calidad de un DataFrame y genera métricas.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        DataQualityMetrics: Métricas de calidad
    """
    metrics = DataQualityMetrics()
    
    if df.empty:
        metrics.add_warning("DataFrame está vacío")
        return metrics
    
    # Métricas básicas
    metrics.total_rows = len(df)
    metrics.total_columns = len(df.columns)
    
    # Análisis de nulos
    null_count = df.isnull().sum().sum()
    total_cells = metrics.total_rows * metrics.total_columns
    metrics.null_cells = int(null_count)
    metrics.null_percentage = (null_count / total_cells * 100) if total_cells > 0 else 0
    
    # Columnas duplicadas
    duplicates = df.columns[df.columns.duplicated()].tolist()
    if duplicates:
        metrics.duplicate_columns = duplicates
        metrics.add_warning(f"Columnas duplicadas encontradas: {duplicates}")
    
    # Columnas con todos nulos
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        metrics.columns_with_all_nulls = all_null_cols
        metrics.add_warning(f"Columnas con todos valores nulos: {all_null_cols}")
    
    # Columnas vacías (sin nombre)
    empty_cols = [col for col in df.columns if not str(col).strip() or str(col).lower() == 'nan']
    if empty_cols:
        metrics.empty_columns = empty_cols
        metrics.add_warning(f"Columnas sin nombre encontradas: {len(empty_cols)}")
    
    # Tipos de datos
    metrics.data_types = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    
    # Uso de memoria
    try:
        memory_bytes = df.memory_usage(deep=True).sum()
        metrics.memory_usage_mb = memory_bytes / (1024 * 1024)
    except Exception as e:
        logger.warning(f"No se pudo calcular uso de memoria: {e}")
    
    # Advertencias de tamaño
    if metrics.total_rows > MAX_ROWS_WARNING:
        metrics.add_warning(
            f"DataFrame muy grande: {metrics.total_rows:,} filas "
            f"(umbral: {MAX_ROWS_WARNING:,})"
        )
    
    if metrics.total_columns > MAX_COLS_WARNING:
        metrics.add_warning(
            f"Muchas columnas: {metrics.total_columns} "
            f"(umbral: {MAX_COLS_WARNING})"
        )
    
    if metrics.total_rows < MIN_ROWS_WARNING:
        metrics.add_warning(f"Muy pocas filas: {metrics.total_rows}")
    
    # Advertencia de nulos
    if metrics.null_percentage > MAX_NULL_PERCENTAGE_WARNING:
        metrics.add_warning(
            f"Alto porcentaje de valores nulos: {metrics.null_percentage:.1f}%"
        )
    
    return metrics


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza un DataFrame.
    
    Args:
        df: DataFrame a limpiar
        
    Returns:
        DataFrame limpio
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Limpiar nombres de columnas
    if df_clean.columns.tolist():
        # Convertir a string y limpiar espacios
        df_clean.columns = [
            str(col).strip() if col is not None else f"Column_{i}"
            for i, col in enumerate(df_clean.columns)
        ]
        
        # Manejar columnas duplicadas
        duplicated_mask = df_clean.columns.duplicated()
        if duplicated_mask.any():
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
            logger.warning(f"Se renombraron columnas duplicadas: {sum(duplicated_mask)} duplicados")
    
    # Eliminar filas completamente vacías
    rows_before = len(df_clean)
    df_clean = df_clean.dropna(how='all')
    rows_removed = rows_before - len(df_clean)
    
    if rows_removed > 0:
        logger.info(f"Se eliminaron {rows_removed} filas completamente vacías")
    
    return df_clean


def _detect_csv_delimiter(
    path: Path,
    sample_size: int = 5,
    encoding: str = 'utf-8'
) -> Optional[str]:
    """
    Intenta detectar el delimitador de un archivo CSV.
    
    Args:
        path: Ruta al archivo CSV
        sample_size: Número de líneas a samplear
        encoding: Encoding a usar
        
    Returns:
        Delimitador detectado o None
    """
    try:
        with open(path, 'r', encoding=encoding) as f:
            lines = [f.readline() for _ in range(sample_size)]
        
        # Contar ocurrencias de cada delimitador
        delimiter_counts = {delim: 0 for delim in COMMON_DELIMITERS}
        
        for line in lines:
            for delim in COMMON_DELIMITERS:
                delimiter_counts[delim] += line.count(delim)
        
        # Encontrar el delimitador más común
        if delimiter_counts:
            detected = max(delimiter_counts.items(), key=lambda x: x[1])
            if detected[1] > 0:
                logger.debug(f"Delimitador detectado: '{detected[0]}' (ocurrencias: {detected[1]})")
                return detected[0]
        
    except Exception as e:
        logger.debug(f"No se pudo detectar delimitador: {e}")
    
    return None


# ============================================================================
# FUNCIONES DE CARGA PRINCIPALES
# ============================================================================
def load_from_csv(
    path: PathType,
    sep: Optional[str] = None,
    encoding: Optional[str] = None,
    auto_detect: bool = True,
    **kwargs
) -> LoadResult:
    """
    Carga un archivo CSV con detección automática de encoding y delimitador.
    
    Args:
        path: Ruta al archivo CSV
        sep: Delimitador (None para auto-detección)
        encoding: Encoding (None para intentar múltiples)
        auto_detect: Si True, intenta detectar delimitador y encoding
        **kwargs: Argumentos adicionales para pd.read_csv
        
    Returns:
        LoadResult con el DataFrame y métricas
    """
    start_time = time.time()
    
    # Validar archivo
    path = _validate_file_path(path)
    metadata = FileMetadata.from_path(path)
    _validate_file_size(metadata)
    
    logger.info(f"Cargando CSV: {path} ({metadata.size_mb:.2f} MB)")
    
    # Determinar encodings a probar
    encodings_to_try = [encoding] if encoding else DEFAULT_ENCODINGS
    
    # Determinar delimitadores a probar
    if sep:
        delimiters_to_try = [sep]
    elif auto_detect:
        detected_delim = _detect_csv_delimiter(path)
        delimiters_to_try = [detected_delim] if detected_delim else COMMON_DELIMITERS
    else:
        delimiters_to_try = [';']  # Default
    
    # Intentar cargar con diferentes combinaciones
    last_error = None
    
    for enc in encodings_to_try:
        for delim in delimiters_to_try:
            try:
                logger.debug(f"Intentando cargar CSV con encoding='{enc}', sep='{delim}'")
                
                df = pd.read_csv(
                    path,
                    sep=delim,
                    encoding=enc,
                    **kwargs
                )
                
                # Limpiar y analizar
                df = _clean_dataframe(df)
                
                if df.empty:
                    logger.warning(f"CSV cargado pero está vacío: {path}")
                    quality_metrics = DataQualityMetrics()
                    quality_metrics.add_warning("Archivo CSV vacío")
                    
                    return LoadResult(
                        status=LoadStatus.EMPTY,
                        data=df,
                        file_metadata=metadata,
                        quality_metrics=quality_metrics,
                        load_time_seconds=time.time() - start_time,
                        encoding_used=enc,
                        delimiter_used=delim
                    )
                
                quality_metrics = _analyze_dataframe_quality(df)
                
                logger.info(
                    f"CSV cargado exitosamente: {quality_metrics.total_rows:,} filas, "
                    f"{quality_metrics.total_columns} columnas "
                    f"(encoding: {enc}, delimitador: '{delim}')"
                )
                
                return LoadResult(
                    status=LoadStatus.SUCCESS,
                    data=df,
                    file_metadata=metadata,
                    quality_metrics=quality_metrics,
                    load_time_seconds=time.time() - start_time,
                    encoding_used=enc,
                    delimiter_used=delim
                )
                
            except UnicodeDecodeError as e:
                last_error = e
                logger.debug(f"Error de encoding con '{enc}': {e}")
                continue
            except pd.errors.EmptyDataError as e:
                logger.warning(f"Archivo CSV vacío o mal formado: {path}")
                return LoadResult(
                    status=LoadStatus.EMPTY,
                    data=pd.DataFrame(),
                    file_metadata=metadata,
                    quality_metrics=None,
                    load_time_seconds=time.time() - start_time,
                    error_message=str(e)
                )
            except Exception as e:
                last_error = e
                logger.debug(f"Error con encoding '{enc}' y delimitador '{delim}': {e}")
                continue
    
    # Si llegamos aquí, ninguna combinación funcionó
    error_msg = f"No se pudo cargar el CSV después de intentar {len(encodings_to_try)} encodings y {len(delimiters_to_try)} delimitadores. Último error: {last_error}"
    logger.error(error_msg)
    
    return LoadResult(
        status=LoadStatus.FAILED,
        data=None,
        file_metadata=metadata,
        quality_metrics=None,
        load_time_seconds=time.time() - start_time,
        error_message=error_msg
    )


def load_from_xlsx(
    path: PathType,
    sheet_name: Optional[Union[str, int, List[str]]] = 0,
    **kwargs
) -> LoadResult:
    """
    Carga un archivo Excel con manejo robusto de errores.
    
    Args:
        path: Ruta al archivo Excel
        sheet_name: Nombre/índice de hoja(s) o None para todas
        **kwargs: Argumentos adicionales para pd.read_excel
        
    Returns:
        LoadResult con DataFrame(s) y métricas
    """
    start_time = time.time()
    
    # Validar archivo
    path = _validate_file_path(path)
    metadata = FileMetadata.from_path(path)
    _validate_file_size(metadata)
    
    sheet_info = sheet_name if sheet_name is not None else "todas las hojas"
    logger.info(f"Cargando Excel: {path} ({metadata.size_mb:.2f} MB), hojas: {sheet_info}")
    
    try:
        # Cargar datos
        data = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
        
        # Procesar según el tipo de dato retornado
        if isinstance(data, dict):
            # Múltiples hojas
            sheets_loaded = list(data.keys())
            logger.info(f"Se cargaron {len(sheets_loaded)} hojas: {sheets_loaded}")
            
            # Limpiar cada hoja
            cleaned_data = {}
            total_rows = 0
            total_cols = 0
            all_warnings = []
            
            for sheet, df in data.items():
                df_clean = _clean_dataframe(df)
                cleaned_data[sheet] = df_clean
                
                if not df_clean.empty:
                    metrics = _analyze_dataframe_quality(df_clean)
                    total_rows += metrics.total_rows
                    total_cols += metrics.total_columns
                    all_warnings.extend([f"[{sheet}] {w}" for w in metrics.warnings])
            
            # Crear métricas combinadas
            quality_metrics = DataQualityMetrics()
            quality_metrics.total_rows = total_rows
            quality_metrics.total_columns = total_cols
            quality_metrics.warnings = all_warnings
            
            if not any(not df.empty for df in cleaned_data.values()):
                logger.warning("Todas las hojas están vacías")
                return LoadResult(
                    status=LoadStatus.EMPTY,
                    data=cleaned_data,
                    file_metadata=metadata,
                    quality_metrics=quality_metrics,
                    load_time_seconds=time.time() - start_time,
                    sheets_loaded=sheets_loaded
                )
            
            return LoadResult(
                status=LoadStatus.SUCCESS,
                data=cleaned_data,
                file_metadata=metadata,
                quality_metrics=quality_metrics,
                load_time_seconds=time.time() - start_time,
                sheets_loaded=sheets_loaded
            )
        
        else:
            # Una sola hoja
            df = _clean_dataframe(data)
            
            if df.empty:
                logger.warning(f"Hoja '{sheet_name}' está vacía")
                quality_metrics = DataQualityMetrics()
                quality_metrics.add_warning(f"Hoja '{sheet_name}' vacía")
                
                return LoadResult(
                    status=LoadStatus.EMPTY,
                    data=df,
                    file_metadata=metadata,
                    quality_metrics=quality_metrics,
                    load_time_seconds=time.time() - start_time,
                    sheets_loaded=[str(sheet_name)]
                )
            
            quality_metrics = _analyze_dataframe_quality(df)
            
            logger.info(
                f"Excel cargado exitosamente: {quality_metrics.total_rows:,} filas, "
                f"{quality_metrics.total_columns} columnas"
            )
            
            return LoadResult(
                status=LoadStatus.SUCCESS,
                data=df,
                file_metadata=metadata,
                quality_metrics=quality_metrics,
                load_time_seconds=time.time() - start_time,
                sheets_loaded=[str(sheet_name)]
            )
    
    except ValueError as e:
        error_msg = f"Error de valor al leer Excel: {e}"
        if "worksheet" in str(e).lower() or "sheet" in str(e).lower():
            error_msg = f"Hoja '{sheet_name}' no encontrada en {path}"
        
        logger.error(error_msg)
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg
        )
    
    except Exception as e:
        error_msg = f"Error inesperado al leer Excel {path}: {e}"
        logger.error(error_msg, exc_info=True)
        
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg
        )


def load_from_pdf(
    path: PathType,
    page_range: Optional[range] = None,
    table_settings: Optional[Dict[str, Any]] = None,
    **kwargs
) -> LoadResult:
    """
    Extrae tablas de un PDF con manejo robusto de errores.
    
    Args:
        path: Ruta al archivo PDF
        page_range: Rango de páginas (ej: range(0, 5))
        table_settings: Configuración para extracción de tablas
        **kwargs: Argumentos para pdfplumber.open
        
    Returns:
        LoadResult con DataFrame y métricas
    """
    start_time = time.time()
    
    # Validar archivo
    path = _validate_file_path(path)
    metadata = FileMetadata.from_path(path)
    _validate_file_size(metadata)
    
    logger.info(f"Extrayendo tablas del PDF: {path} ({metadata.size_mb:.2f} MB)")
    
    table_settings = table_settings or {}
    tables_data = []
    pages_processed = 0
    pages_with_tables = 0
    total_tables_found = 0
    
    try:
        with pdfplumber.open(path, **kwargs) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF tiene {total_pages} páginas")
            
            # Determinar páginas a procesar
            if page_range is not None:
                start_page = max(0, page_range.start)
                end_page = min(total_pages, page_range.stop)
                pages_to_process = pdf.pages[start_page:end_page]
                logger.info(f"Procesando páginas {start_page + 1} a {end_page}")
            else:
                pages_to_process = pdf.pages
            
            # Procesar cada página
            for i, page in enumerate(pages_to_process):
                page_num = (page_range.start if page_range else 0) + i + 1
                pages_processed += 1
                
                try:
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
                            # Crear DataFrame
                            if len(table) > 1:
                                headers = table[0] if table[0] else [f"Col_{j}" for j in range(len(table[0] or []))]
                                df_table = pd.DataFrame(table[1:], columns=headers)
                            else:
                                df_table = pd.DataFrame(table)
                            
                            if not df_table.empty:
                                tables_data.append(df_table)
                                logger.debug(
                                    f"Página {page_num}, tabla {table_idx + 1}: "
                                    f"{df_table.shape[0]} filas × {df_table.shape[1]} columnas"
                                )
                        
                        except Exception as e:
                            logger.warning(
                                f"Error al procesar tabla {table_idx + 1} en página {page_num}: {e}"
                            )
                            continue
                
                except Exception as e:
                    logger.warning(f"Error al procesar página {page_num}: {e}")
                    continue
        
        # Procesar resultados
        if not tables_data:
            logger.warning(
                f"No se encontraron tablas válidas en el PDF. "
                f"Páginas procesadas: {pages_processed}"
            )
            
            quality_metrics = DataQualityMetrics()
            quality_metrics.add_warning(
                f"No se encontraron tablas en {pages_processed} páginas procesadas"
            )
            
            return LoadResult(
                status=LoadStatus.EMPTY,
                data=pd.DataFrame(),
                file_metadata=metadata,
                quality_metrics=quality_metrics,
                load_time_seconds=time.time() - start_time
            )
        
        # Combinar todas las tablas
        try:
            combined_df = pd.concat(tables_data, ignore_index=True)
            combined_df = _clean_dataframe(combined_df)
            
            quality_metrics = _analyze_dataframe_quality(combined_df)
            quality_metrics.add_warning(
                f"Datos extraídos de {total_tables_found} tablas en "
                f"{pages_with_tables} de {pages_processed} páginas"
            )
            
            logger.info(
                f"PDF procesado exitosamente: {total_tables_found} tablas combinadas, "
                f"{quality_metrics.total_rows:,} filas, {quality_metrics.total_columns} columnas"
            )
            
            return LoadResult(
                status=LoadStatus.SUCCESS,
                data=combined_df,
                file_metadata=metadata,
                quality_metrics=quality_metrics,
                load_time_seconds=time.time() - start_time
            )
        
        except Exception as e:
            error_msg = f"Error al combinar tablas del PDF: {e}"
            logger.error(error_msg, exc_info=True)
            
            return LoadResult(
                status=LoadStatus.FAILED,
                data=None,
                file_metadata=metadata,
                quality_metrics=None,
                load_time_seconds=time.time() - start_time,
                error_message=error_msg
            )
    
    except Exception as e:
        error_msg = f"Error al abrir o procesar PDF {path}: {e}"
        logger.error(error_msg, exc_info=True)
        
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg
        )


# ============================================================================
# FUNCIÓN FACTORY PRINCIPAL
# ============================================================================
def load_data(
    path: PathType,
    format_hint: Optional[str] = None,
    **kwargs
) -> LoadResult:
    """
    Función factory que carga datos según la extensión del archivo.
    Incluye validaciones exhaustivas y métricas detalladas.
    
    Args:
        path: Ruta al archivo
        format_hint: Sugerencia de formato (sobrescribe detección automática)
        **kwargs: Argumentos específicos del formato
        
    Returns:
        LoadResult con datos, métricas y metadatos
    
    Ejemplos:
        >>> result = load_data("datos.csv")
        >>> if result.status == LoadStatus.SUCCESS:
        >>>     df = result.data
        >>>     print(result.quality_metrics.to_dict())
        
        >>> result = load_data("libro.xlsx", sheet_name=None)
        >>> if result.status == LoadStatus.SUCCESS:
        >>>     for sheet, df in result.data.items():
        >>>         print(f"{sheet}: {len(df)} filas")
    """
    start_time = time.time()
    
    logger.info("="*80)
    logger.info(f"Iniciando carga de datos desde: {path}")
    logger.info("="*80)
    
    try:
        # Validar y preparar ruta
        path = _validate_file_path(path)
        metadata = FileMetadata.from_path(path)
        
        # Determinar formato
        if format_hint:
            format_hint_upper = format_hint.upper()
            if format_hint_upper == "CSV":
                file_format = FileFormat.CSV
            elif format_hint_upper in ["EXCEL", "XLSX", "XLS"]:
                file_format = FileFormat.EXCEL
            elif format_hint_upper == "PDF":
                file_format = FileFormat.PDF
            else:
                logger.warning(
                    f"Sugerencia de formato '{format_hint}' no reconocida. "
                    f"Usando detección automática."
                )
                file_format = metadata.format
        else:
            file_format = metadata.format
        
        # Validar formato soportado
        if file_format == FileFormat.UNKNOWN:
            extension = path.suffix.lower()
            error_msg = (
                f"Formato de archivo no soportado: {extension}. "
                f"Formatos soportados: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}"
            )
            logger.error(error_msg)
            
            return LoadResult(
                status=LoadStatus.FAILED,
                data=None,
                file_metadata=metadata,
                quality_metrics=None,
                load_time_seconds=time.time() - start_time,
                error_message=error_msg
            )
        
        logger.info(
            f"Formato detectado: {file_format.value}, "
            f"Tamaño: {metadata.size_mb:.2f} MB"
        )
        
        # Cargar según formato
        if file_format == FileFormat.CSV:
            result = load_from_csv(path, **kwargs)
        elif file_format == FileFormat.EXCEL:
            result = load_from_xlsx(path, **kwargs)
        elif file_format == FileFormat.PDF:
            result = load_from_pdf(path, **kwargs)
        else:
            # Este caso no debería ocurrir, pero por seguridad
            error_msg = f"Formato {file_format} no implementado"
            logger.error(error_msg)
            return LoadResult(
                status=LoadStatus.FAILED,
                data=None,
                file_metadata=metadata,
                quality_metrics=None,
                load_time_seconds=time.time() - start_time,
                error_message=error_msg
            )
        
        # Logging final
        logger.info("="*80)
        logger.info(f"Carga completada - Estado: {result.status.value}")
        logger.info(f"Tiempo total: {result.load_time_seconds:.3f} segundos")
        
        if result.quality_metrics:
            logger.info(
                f"Filas: {result.quality_metrics.total_rows:,}, "
                f"Columnas: {result.quality_metrics.total_columns}"
            )
            if result.quality_metrics.warnings:
                logger.warning(
                    f"Advertencias ({len(result.quality_metrics.warnings)}): "
                    f"{result.quality_metrics.warnings}"
                )
        
        logger.info("="*80)
        
        return result
    
    except FileNotFoundError as e:
        error_msg = str(e)
        logger.error(error_msg)
        
        # Crear metadata básica para archivo no encontrado
        try:
            path_obj = Path(path)
            metadata = FileMetadata(
                path=path_obj,
                size_bytes=0,
                size_mb=0,
                format=FileFormat.UNKNOWN,
                exists=False,
                readable=False
            )
        except Exception:
            metadata = None
        
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=metadata,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg
        )
    
    except Exception as e:
        error_msg = f"Error inesperado al cargar {path}: {e}"
        logger.error(error_msg, exc_info=True)
        
        return LoadResult(
            status=LoadStatus.FAILED,
            data=None,
            file_metadata=None,
            quality_metrics=None,
            load_time_seconds=time.time() - start_time,
            error_message=error_msg
        )


# ============================================================================
# FUNCIONES DE UTILIDAD ADICIONALES
# ============================================================================
def get_file_info(path: PathType) -> Dict[str, Any]:
    """
    Obtiene información de un archivo sin cargarlo.
    
    Args:
        path: Ruta al archivo
        
    Returns:
        Diccionario con metadatos del archivo
    """
    try:
        path = Path(path).resolve()
        metadata = FileMetadata.from_path(path)
        
        return {
            "path": str(metadata.path),
            "exists": metadata.exists,
            "size_mb": metadata.size_mb,
            "format": metadata.format.value,
            "readable": metadata.readable,
            "modified_time": metadata.modified_time.isoformat() if metadata.modified_time else None,
            "supported": metadata.format != FileFormat.UNKNOWN
        }
    except Exception as e:
        logger.error(f"Error al obtener información del archivo: {e}")
        return {
            "path": str(path),
            "error": str(e)
        }


def validate_file_before_load(path: PathType) -> Tuple[bool, Optional[str]]:
    """
    Valida un archivo antes de intentar cargarlo.
    
    Args:
        path: Ruta al archivo
        
    Returns:
        Tuple[bool, Optional[str]]: (es_valido, mensaje_error)
    """
    try:
        path = _validate_file_path(path)
        metadata = FileMetadata.from_path(path)
        
        if metadata.format == FileFormat.UNKNOWN:
            return False, f"Formato no soportado: {path.suffix}"
        
        if metadata.size_mb > MAX_FILE_SIZE_MB:
            return False, f"Archivo demasiado grande: {metadata.size_mb:.2f} MB (máx: {MAX_FILE_SIZE_MB} MB)"
        
        if metadata.size_bytes == 0:
            return False, "Archivo vacío"
        
        return True, None
    
    except Exception as e:
        return False, str(e)