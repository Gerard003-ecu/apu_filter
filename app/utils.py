"""
Módulo de utilidades para procesamiento de datos de APU.

Este módulo proporciona funciones robustas para normalización de texto,
conversión de números, validación de datos y manejo de archivos.
"""

import os
import re
from typing import Optional, Union, Dict, List, Tuple, Any
from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from unidecode import unidecode
import logging

# Configuración del logger
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# Unidades estándar soportadas (como frozenset para mejor rendimiento)
STANDARD_UNITS = frozenset({
    # Longitud
    'M', 'M2', 'M3', 'ML', 'KM', 'CM', 'MM',
    # Tiempo
    'HORA', 'HR', 'DIA', 'SEMANA', 'MES', 'AÑO', 'JOR',
    # Peso
    'KG', 'TON', 'LB', 'GR',
    # Volumen líquido
    'L', 'LT', 'GAL', 'ML',
    # Unidades
    'UND', 'UN', 'PAR', 'JUEGO', 'KIT',
    # Transporte
    'VIAJE', 'VIAJES', 'KM',
    # Otros
    'SERVICIO', '%'
})

# Mapeo de unidades equivalentes (optimizado como dict constante)
UNIT_MAPPING = {
    'DIAS': 'DIA', 'DÍAS': 'DIA', 'JORNAL': 'JOR', 'JORNALES': 'JOR',
    'HORAS': 'HR', 'HORA': 'HR', 'UNIDAD': 'UND', 'UNIDADES': 'UND',
    'UN': 'UND', 'METRO': 'M', 'METROS': 'M', 'MTS': 'M',
    'METRO2': 'M2', 'M2': 'M2', 'MT2': 'M2', 'METRO CUADRADO': 'M2',
    'METRO3': 'M3', 'M3': 'M3', 'MT3': 'M3', 'METRO CUBICO': 'M3',
    'KILOGRAMO': 'KG', 'KILOGRAMOS': 'KG', 'KILOS': 'KG',
    'TONELADA': 'TON', 'TONELADAS': 'TON',
    'GALON': 'GAL', 'GALONES': 'GAL', 'GLN': 'GAL',
    'LITRO': 'L', 'LITROS': 'L', 'LT': 'L',
    'VIAJES': 'VIAJE', 'VJE': 'VIAJE'
}

# Configuraciones por defecto
DEFAULT_ENCODING_ATTEMPTS = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
DEFAULT_CSV_SEPARATORS = [',', ';', '\t', '|']
NUMERIC_VALIDATION_LIMITS = {
    'min': 0,
    'max': 1e12,
    'allow_zero': True
}

# Patrones regex compilados (más eficiente)
NUMERIC_PATTERN = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
APU_INVALID_CHARS_PATTERN = re.compile(r'[^\w.\-]')
WHITESPACE_PATTERN = re.compile(r'\s+')

# ============================================================================
# FUNCIONES DE NORMALIZACIÓN DE TEXTO
# ============================================================================

@lru_cache(maxsize=1024)
def normalize_text(text: str, preserve_special_chars: bool = False) -> str:
    """
    Normaliza un texto de forma consistente y robusta con cache.

    Args:
        text: Texto a normalizar
        preserve_special_chars: Si True, preserva algunos caracteres especiales útiles

    Returns:
        Texto normalizado

    Raises:
        TypeError: Si text no puede convertirse a string
    """
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception as e:
            raise TypeError(f"No se puede convertir a string: {e}")

    # Validación de entrada vacía
    if not text:
        return ""

    # Convertir a minúsculas y remover espacios extra
    text = text.lower().strip()
    
    # Remover acentos y caracteres especiales
    text = unidecode(text)
    
    # Definir patrones de caracteres permitidos
    if preserve_special_chars:
        # Preservar caracteres útiles para descripciones técnicas
        pattern = r"[^a-z0-9\s#\-_/\.@]"
    else:
        # Solo caracteres básicos para comparaciones
        pattern = r"[^a-z0-9\s]"
    
    # Remover caracteres no permitidos
    text = re.sub(pattern, "", text)
    
    # Normalizar espacios (múltiples espacios a uno solo)
    text = WHITESPACE_PATTERN.sub(" ", text)
    
    return text.strip()


def normalize_text_series(
    text_series: pd.Series, 
    preserve_special_chars: bool = False,
    chunk_size: int = 10000
) -> pd.Series:
    """
    Normaliza una serie de texto de forma vectorizada y eficiente.
    
    Incluye procesamiento por chunks para series grandes.

    Args:
        text_series: Serie de pandas con texto a normalizar
        preserve_special_chars: Si True, preserva algunos caracteres especiales útiles
        chunk_size: Tamaño del chunk para procesamiento de series grandes

    Returns:
        Serie de texto normalizada
    """
    if text_series is None or text_series.empty:
        return text_series

    # Asegurar que todos los elementos sean strings
    text_series = text_series.astype(str)
    
    # Para series grandes, procesar por chunks
    if len(text_series) > chunk_size:
        result_chunks = []
        for i in range(0, len(text_series), chunk_size):
            chunk = text_series.iloc[i:i+chunk_size]
            normalized_chunk = chunk.apply(
                lambda x: _safe_normalize(x, preserve_special_chars)
            )
            result_chunks.append(normalized_chunk)
        return pd.concat(result_chunks)
    
    # Para series pequeñas, procesar directamente
    return text_series.apply(lambda x: _safe_normalize(x, preserve_special_chars))


def _safe_normalize(text: str, preserve_special_chars: bool) -> str:
    """Función auxiliar para normalización segura."""
    try:
        return normalize_text(text, preserve_special_chars)
    except Exception as e:
        logger.warning(f"Error normalizando texto '{text}': {e}")
        return str(text)

# ============================================================================
# FUNCIONES DE CONVERSIÓN NUMÉRICA
# ============================================================================

def parse_number(
    s: Optional[Union[str, float, int]], 
    decimal_separator: str = "auto",
    default_value: float = 0.0
) -> float:
    """
    Convierte una cadena a número de punto flotante de forma robusta.

    Args:
        s: Valor a convertir (string, float, int)
        decimal_separator: "auto", "comma" o "dot"
        default_value: Valor por defecto si la conversión falla

    Returns:
        Número convertido o default_value si falla
    """
    if s is None:
        return default_value

    # Si ya es numérico, retornar directamente
    if isinstance(s, (int, float)):
        if pd.isna(s) or np.isnan(s):
            return default_value
        return float(s)

    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return default_value

    # Limpiar string inicial
    s_cleaned = s.strip()
    if not s_cleaned:
        return default_value

    # Remover símbolos comunes de moneda y porcentaje
    for char in ['$', '€', '£', '¥', '%', ' ']:
        s_cleaned = s_cleaned.replace(char, '')

    if not s_cleaned:
        return default_value

    # Detección automática del separador decimal
    if decimal_separator == "auto":
        decimal_separator = _detect_decimal_separator(s_cleaned)

    # Limpiar según el separador decimal detectado
    s_cleaned = _normalize_numeric_string(s_cleaned, decimal_separator)

    # Validar y convertir
    if NUMERIC_PATTERN.match(s_cleaned):
        try:
            result = float(s_cleaned)
            # Validar que el resultado no sea infinito
            if np.isinf(result):
                return default_value
            return result
        except (ValueError, TypeError, OverflowError):
            return default_value
    
    return default_value


def _detect_decimal_separator(s: str) -> str:
    """Detecta el separador decimal en un string numérico."""
    comma_count = s.count(',')
    dot_count = s.count('.')
    
    if comma_count > 0 and dot_count > 0:
        # Ambos presentes: asumir que la coma es decimal si está después del punto
        if comma_count == 1 and s.rfind(',') > s.rfind('.'):
            return "comma"
        return "dot"
    elif comma_count == 1 and dot_count == 0:
        # Solo una coma: probablemente decimal
        # Verificar si hay 3 dígitos después (sería miles)
        comma_pos = s.find(',')
        digits_after = len(s[comma_pos+1:])
        return "dot" if digits_after == 3 else "comma"
    
    return "dot"


def _normalize_numeric_string(s: str, decimal_separator: str) -> str:
    """Normaliza un string numérico según el separador decimal."""
    if decimal_separator == "comma":
        # Coma como decimal, punto como miles
        s = s.replace(".", "")  # Eliminar separadores de miles
        s = s.replace(",", ".")  # Convertir coma decimal a punto
    else:
        # Punto como decimal, coma como miles
        s = s.replace(",", "")  # Eliminar separadores de miles
    
    # Manejar casos edge como "1.234.567" (múltiples puntos)
    if s.count('.') > 1:
        parts = s.split('.')
        integer_part = ''.join(parts[:-1])
        decimal_part = parts[-1]
        s = f"{integer_part}.{decimal_part}"
    
    return s

# ============================================================================
# FUNCIONES DE VALIDACIÓN Y LIMPIEZA DE CÓDIGOS APU
# ============================================================================

@lru_cache(maxsize=512)
def clean_apu_code(code: str, validate_format: bool = True) -> str:
    """
    Limpia y valida un código de APU de forma robusta con cache.

    Args:
        code: Código de APU a limpiar
        validate_format: Si True, valida el formato básico

    Returns:
        Código de APU limpio y validado

    Raises:
        ValueError: Si el código es inválido y validate_format=True
        TypeError: Si code no puede convertirse a string
    """
    if not isinstance(code, str):
        try:
            code = str(code)
        except Exception as e:
            raise TypeError(f"No se puede convertir código APU a string: {e}")

    original_code = code
    code = code.strip().upper()

    # Remover caracteres no permitidos (mantener letras, números, puntos, guiones)
    code = APU_INVALID_CHARS_PATTERN.sub('', code)
    
    # Remover puntos y guiones al final
    code = code.rstrip('.-')

    # Validaciones opcionales de formato
    if validate_format:
        if not code:
            raise ValueError(f"Código APU no puede estar vacío: '{original_code}'")

        if len(code) < 2:
            raise ValueError(f"Código APU demasiado corto: '{original_code}'")

        # Verificar que tenga al menos un número
        if not any(char.isdigit() for char in code):
            logger.warning(f"Código APU sin números: '{original_code}' -> '{code}'")
    
    return code

# ============================================================================
# FUNCIONES DE NORMALIZACIÓN DE UNIDADES
# ============================================================================

@lru_cache(maxsize=256)
def normalize_unit(unit: str) -> str:
    """
    Normaliza y valida una unidad de medida con cache.

    Args:
        unit: Unidad a normalizar

    Returns:
        Unidad normalizada o 'UND' si no es válida
    """
    if not unit or not isinstance(unit, str):
        return 'UND'

    unit = unit.upper().strip()
    
    # Verificar en mapeo primero (más común)
    if unit in UNIT_MAPPING:
        return UNIT_MAPPING[unit]
    
    # Si es una unidad estándar, retornarla
    if unit in STANDARD_UNITS:
        return unit
    
    # Intentar limpiar y verificar
    clean_unit = re.sub(r'[^A-Z0-9]', '', unit)
    if clean_unit in STANDARD_UNITS:
        return clean_unit
    
    # Log solo para unidades no triviales
    if unit not in ('', 'UND') and len(unit) > 1:
        logger.debug(f"Unidad no reconocida: '{unit}' -> usando 'UND'")
    
    return 'UND'

# ============================================================================
# FUNCIONES DE LECTURA DE ARCHIVOS
# ============================================================================

def safe_read_dataframe(
    path: Union[str, Path],
    header: int = 0,
    encoding: str = "auto",
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Lee un archivo en DataFrame de forma robusta con detección automática.

    Args:
        path: Ruta al archivo
        header: Fila a usar como encabezado
        encoding: Codificación a usar ("auto" para detección)
        nrows: Número de filas a leer (None para todas)
        usecols: Lista de columnas a leer (None para todas)

    Returns:
        DataFrame leído o DataFrame vacío si falla
    """
    path = Path(path) if not isinstance(path, Path) else path
    
    if not path.exists():
        logger.error(f"Archivo no encontrado: {path}")
        return pd.DataFrame()

    try:
        # Detección automática de encoding
        if encoding == "auto":
            encoding = _detect_file_encoding(path)

        # Leer según extensión
        file_extension = path.suffix.lower()
        
        if file_extension == ".csv":
            return _read_csv_robust(path, encoding, header, nrows, usecols)
        elif file_extension in [".xls", ".xlsx"]:
            return _read_excel_robust(path, header, nrows, usecols)
        else:
            logger.error(f"Formato no soportado: {file_extension}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error leyendo archivo {path}: {e}")
        return pd.DataFrame()


def _detect_file_encoding(path: Path) -> str:
    """Detecta la codificación de un archivo de texto."""
    for enc in DEFAULT_ENCODING_ATTEMPTS:
        try:
            with open(path, 'r', encoding=enc) as f:
                f.read(1024)  # Leer solo una muestra
            return enc
        except UnicodeDecodeError:
            continue
    return 'latin1'  # Fallback


def _read_csv_robust(
    path: Path,
    encoding: str,
    header: int,
    nrows: Optional[int],
    usecols: Optional[List[str]]
) -> pd.DataFrame:
    """Lee un archivo CSV de forma robusta."""
    # Detectar separador
    separator = _detect_csv_separator(path, encoding)
    
    # Configurar parámetros de lectura
    read_params = {
        'filepath_or_buffer': path,
        'encoding': encoding,
        'sep': separator,
        'engine': 'python',
        'header': header,
        'on_bad_lines': 'skip',
        'low_memory': False  # Evita warnings de tipos mixtos
    }
    
    if nrows is not None:
        read_params['nrows'] = nrows
    if usecols is not None:
        read_params['usecols'] = usecols
    
    return pd.read_csv(**read_params)


def _read_excel_robust(
    path: Path,
    header: int,
    nrows: Optional[int],
    usecols: Optional[List[str]]
) -> pd.DataFrame:
    """Lee un archivo Excel de forma robusta."""
    read_params = {
        'io': path,
        'header': header
    }
    
    if nrows is not None:
        read_params['nrows'] = nrows
    if usecols is not None:
        read_params['usecols'] = usecols
    
    return pd.read_excel(**read_params)


def _detect_csv_separator(path: Path, encoding: str) -> str:
    """Detecta el separador de un archivo CSV."""
    try:
        with open(path, 'r', encoding=encoding) as f:
            sample = f.read(4096)
        
        best_sep = ','
        best_count = 0
        
        for sep in DEFAULT_CSV_SEPARATORS:
            # Contar ocurrencias considerando saltos de línea
            lines = sample.split('\n')[:5]  # Primeras 5 líneas
            if len(lines) > 1:
                counts = [line.count(sep) for line in lines if line]
                if counts and min(counts) > 0:
                    avg_count = sum(counts) / len(counts)
                    if avg_count > best_count:
                        best_count = avg_count
                        best_sep = sep
        
        return best_sep
    except Exception:
        return ','

# ============================================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================================

def validate_numeric_value(
    value: float,
    field_name: str = "valor",
    min_value: float = None,
    max_value: float = None,
    allow_zero: bool = True,
    allow_negative: bool = False,
    allow_inf: bool = False
) -> Tuple[bool, str]:
    """
    Valida un valor numérico según criterios configurables.

    Args:
        value: Valor a validar
        field_name: Nombre del campo para mensajes de error
        min_value: Valor mínimo permitido (None para sin límite)
        max_value: Valor máximo permitido (None para sin límite)
        allow_zero: Si permite valor cero
        allow_negative: Si permite valores negativos
        allow_inf: Si permite valores infinitos

    Returns:
        Tuple (es_válido, mensaje_error)
    """
    # Validar tipo
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False, f"{field_name} debe ser numérico"

    # Validar nulos
    if pd.isna(value):
        return False, f"{field_name} no puede ser nulo"

    # Validar infinitos
    if np.isinf(value):
        if not allow_inf:
            return False, f"{field_name} no puede ser infinito"
        return True, ""

    # Validar cero
    if not allow_zero and value == 0:
        return False, f"{field_name} no puede ser cero"

    # Validar negativos
    if not allow_negative and value < 0:
        return False, f"{field_name} no puede ser negativo"

    # Validar rango mínimo
    if min_value is not None and value < min_value:
        return False, f"{field_name} no puede ser menor que {min_value}"

    # Validar rango máximo
    if max_value is not None and value > max_value:
        return False, f"{field_name} no puede ser mayor que {max_value}"

    return True, ""


def validate_series(
    series: pd.Series,
    return_mask: bool = True,
    **kwargs
) -> Union[pd.Series, pd.DataFrame]:
    """
    Aplica validación numérica a una serie completa.

    Args:
        series: Serie a validar
        return_mask: Si True, retorna máscara booleana. Si False, retorna DataFrame con detalles
        **kwargs: Argumentos para validate_numeric_value

    Returns:
        Serie booleana o DataFrame con validación y mensajes
    """
    if series.empty:
        return series if return_mask else pd.DataFrame()

    if return_mask:
        # Retornar solo máscara booleana
        return series.apply(
            lambda x: validate_numeric_value(x, **kwargs)[0]
        )
    else:
        # Retornar DataFrame con detalles
        validation_results = series.apply(
            lambda x: validate_numeric_value(x, **kwargs)
        )
        
        return pd.DataFrame({
            'value': series,
            'is_valid': validation_results.apply(lambda x: x[0]),
            'error_message': validation_results.apply(lambda x: x[1])
        })

# ============================================================================
# FUNCIONES DE ANÁLISIS Y DETECCIÓN
# ============================================================================

def create_apu_signature(
    apu_data: Dict[str, Any],
    key_fields: Optional[List[str]] = None
) -> str:
    """
    Crea una firma única para un APU basada en sus datos clave.

    Args:
        apu_data: Diccionario con datos del APU
        key_fields: Campos a usar para la firma (None para usar default)

    Returns:
        Firma única del APU
    """
    if key_fields is None:
        key_fields = ['CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU']
    
    signature_parts = []
    
    for field in key_fields:
        value = apu_data.get(field, '')
        if value:
            # Normalizar el valor para la firma
            if isinstance(value, (int, float)):
                normalized = str(value)
            else:
                normalized = normalize_text(str(value))
            
            if normalized:  # Solo añadir si no está vacío
                signature_parts.append(normalized)
    
    return '|'.join(signature_parts) if signature_parts else 'empty_signature'


def detect_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 1.5,
    return_bounds: bool = False
) -> Union[pd.Series, Tuple[pd.Series, Dict[str, float]]]:
    """
    Detecta valores atípicos en una serie numérica con métodos configurables.

    Args:
        series: Serie numérica a analizar
        method: Método de detección ("iqr", "zscore", "modified_zscore")
        threshold: Umbral para detección (1.5 para IQR, 3 para z-score)
        return_bounds: Si True, retorna también los límites utilizados

    Returns:
        Serie booleana indicando outliers, opcionalmente con límites

    Raises:
        ValueError: Si el método no es soportado
    """
    # Validar entrada
    if series.empty:
        result = pd.Series(dtype=bool)
        bounds = {}
        return (result, bounds) if return_bounds else result

    # Remover valores nulos para el cálculo
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        result = pd.Series([False] * len(series), index=series.index)
        bounds = {}
        return (result, bounds) if return_bounds else result

    outliers = None
    bounds = {}

    if method == "iqr":
        outliers, bounds = _detect_outliers_iqr(series, clean_series, threshold)
    elif method == "zscore":
        outliers, bounds = _detect_outliers_zscore(series, clean_series, threshold)
    elif method == "modified_zscore":
        outliers, bounds = _detect_outliers_modified_zscore(series, clean_series, threshold)
    else:
        raise ValueError(f"Método no soportado: {method}")

    return (outliers, bounds) if return_bounds else outliers


def _detect_outliers_iqr(
    series: pd.Series,
    clean_series: pd.Series,
    threshold: float
) -> Tuple[pd.Series, Dict[str, float]]:
    """Detección de outliers usando IQR."""
    Q1 = clean_series.quantile(0.25)
    Q3 = clean_series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (series < lower_bound) | (series > upper_bound)
    bounds = {
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    return outliers, bounds


def _detect_outliers_zscore(
    series: pd.Series,
    clean_series: pd.Series,
    threshold: float
) -> Tuple[pd.Series, Dict[str, float]]:
    """Detección de outliers usando z-score."""
    mean = clean_series.mean()
    std = clean_series.std()
    
    if std == 0:  # Evitar división por cero
        outliers = pd.Series([False] * len(series), index=series.index)
        bounds = {'mean': mean, 'std': 0, 'threshold': threshold}
    else:
        z_scores = np.abs((series - mean) / std)
        outliers = z_scores > threshold
        bounds = {
            'mean': mean, 'std': std,
            'threshold': threshold,
            'lower_bound': mean - threshold * std,
            'upper_bound': mean + threshold * std
        }
    
    return outliers, bounds


def _detect_outliers_modified_zscore(
    series: pd.Series,
    clean_series: pd.Series,
    threshold: float
) -> Tuple[pd.Series, Dict[str, float]]:
    """Detección de outliers usando Modified Z-score (más robusto)."""
    median = clean_series.median()
    mad = np.median(np.abs(clean_series - median))
    
    if mad == 0:
        # Usar MAD mínimo para evitar división por cero
        mad = 1.4826 * clean_series.std()
        if mad == 0:
            outliers = pd.Series([False] * len(series), index=series.index)
            bounds = {'median': median, 'mad': 0, 'threshold': threshold}
            return outliers, bounds
    
    modified_z_scores = 0.6745 * (series - median) / mad
    outliers = np.abs(modified_z_scores) > threshold
    
    bounds = {
        'median': median,
        'mad': mad,
        'threshold': threshold,
        'lower_bound': median - threshold * mad / 0.6745,
        'upper_bound': median + threshold * mad / 0.6745
    }
    
    return outliers, bounds

# ============================================================================
# FUNCIONES DE MANIPULACIÓN DE DATAFRAMES
# ============================================================================

def find_and_rename_columns(
    df: pd.DataFrame,
    column_map: Dict[str, List[str]],
    case_sensitive: bool = False
) -> pd.DataFrame:
    """
    Busca y renombra columnas en un DataFrame con búsqueda flexible.

    Args:
        df: El DataFrame en el que se buscarán y renombrarán las columnas
        column_map: Diccionario que mapea nombres estándar a posibles nombres
        case_sensitive: Si la búsqueda debe ser sensible a mayúsculas

    Returns:
        DataFrame con las columnas renombradas

    Raises:
        ValueError: Si hay conflictos en el mapeo
    """
    if df.empty:
        return df

    renamed_cols = {}
    used_cols = set()
    
    for standard_name, possible_names in column_map.items():
        found = False
        for col in df.columns:
            if col in used_cols:
                continue
                
            col_compare = col if case_sensitive else str(col).lower()
            
            for p_name in possible_names:
                p_name_compare = p_name if case_sensitive else str(p_name).lower()
                
                # Búsqueda flexible: coincidencia exacta o parcial
                if (p_name_compare == col_compare or
                    p_name_compare in col_compare or
                    col_compare in p_name_compare):
                    
                    if standard_name in renamed_cols.values():
                        logger.warning(
                            f"Columna '{standard_name}' ya mapeada. "
                            f"Ignorando '{col}'"
                        )
                        continue
                    
                    renamed_cols[col] = standard_name
                    used_cols.add(col)
                    found = True
                    break
            
            if found:
                break
    
    # Log columnas no mapeadas si hay muchas
    unmapped = set(df.columns) - used_cols
    if len(unmapped) > 0 and len(unmapped) <= 5:
        logger.debug(f"Columnas no mapeadas: {unmapped}")
    
    return df.rename(columns=renamed_cols)

# ============================================================================
# FUNCIONES DE SERIALIZACIÓN
# ============================================================================

def sanitize_for_json(data: Any, max_depth: int = 100) -> Any:
    """
    Convierte tipos de datos no serializables a tipos nativos de Python.

    Args:
        data: La estructura de datos a sanear
        max_depth: Profundidad máxima de recursión

    Returns:
        La estructura de datos saneada

    Raises:
        RecursionError: Si se excede la profundidad máxima
    """
    if max_depth <= 0:
        raise RecursionError("Profundidad máxima de recursión alcanzada")

    # Manejar diccionarios
    if isinstance(data, dict):
        return {
            k: sanitize_for_json(v, max_depth - 1)
            for k, v in data.items()
        }
    
    # Manejar listas y tuplas
    if isinstance(data, (list, tuple)):
        return [
            sanitize_for_json(v, max_depth - 1)
            for v in data
        ]
    
    # Manejar Series de pandas
    if isinstance(data, pd.Series):
        return sanitize_for_json(data.to_list(), max_depth - 1)
    
    # Manejar DataFrames de pandas
    if isinstance(data, pd.DataFrame):
        return sanitize_for_json(data.to_dict('records'), max_depth - 1)
    
    # Conversión de tipos de NumPy a Python nativo
    if isinstance(data, (np.integer, np.int32, np.int64)):
        return int(data)
    
    if isinstance(data, (np.floating, np.float32, np.float64)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    
    if isinstance(data, np.ndarray):
        return sanitize_for_json(data.tolist(), max_depth - 1)
    
    if isinstance(data, (np.bool_, bool)):
        return bool(data)
    
    # Manejar pd.NA, pd.NaT y otros nulos de Pandas
    if pd.isna(data):
        return None
    
    # Manejar fechas
    if hasattr(data, 'isoformat'):
        return data.isoformat()
    
    # Para otros tipos, intentar conversión a string
    if hasattr(data, '__dict__'):
        return sanitize_for_json(data.__dict__, max_depth - 1)
    
    return data

# ============================================================================
# FUNCIONES ADICIONALES DE UTILIDAD
# ============================================================================

def calculate_statistics(series: pd.Series) -> Dict[str, float]:
    """
    Calcula estadísticas descriptivas robustas para una serie numérica.

    Args:
        series: Serie numérica

    Returns:
        Diccionario con estadísticas
    """
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {
            'count': 0, 'mean': None, 'std': None,
            'min': None, 'max': None, 'median': None
        }
    
    return {
        'count': len(clean_series),
        'mean': float(clean_series.mean()),
        'std': float(clean_series.std()),
        'min': float(clean_series.min()),
        'max': float(clean_series.max()),
        'median': float(clean_series.median()),
        'q1': float(clean_series.quantile(0.25)),
        'q3': float(clean_series.quantile(0.75)),
        'null_count': len(series) - len(clean_series),
        'null_percentage': (len(series) - len(clean_series)) / len(series) * 100
    }


def batch_process_dataframe(
    df: pd.DataFrame,
    process_func: callable,
    batch_size: int = 1000,
    **kwargs
) -> pd.DataFrame:
    """
    Procesa un DataFrame en lotes para optimizar memoria.

    Args:
        df: DataFrame a procesar
        process_func: Función de procesamiento
        batch_size: Tamaño del lote
        **kwargs: Argumentos adicionales para process_func

    Returns:
        DataFrame procesado
    """
    if len(df) <= batch_size:
        return process_func(df, **kwargs)
    
    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        processed = process_func(batch, **kwargs)
        results.append(processed)
    
    return pd.concat(results, ignore_index=True)

# ============================================================================
# LISTA DE EXPORTACIÓN
# ============================================================================

__all__ = [
    # Funciones principales
    'normalize_text',
    'normalize_text_series',
    'parse_number',
    'clean_apu_code',
    'normalize_unit',
    'safe_read_dataframe',
    'validate_numeric_value',
    'validate_series',
    'create_apu_signature',
    'detect_outliers',
    'find_and_rename_columns',
    'sanitize_for_json',
    # Funciones adicionales
    'calculate_statistics',
    'batch_process_dataframe',
    # Constantes
    'STANDARD_UNITS',
    'UNIT_MAPPING'
]