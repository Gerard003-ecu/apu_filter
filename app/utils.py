import os
import re
from typing import Optional, Union

import numpy as np
import pandas as pd
from unidecode import unidecode
import logging

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normaliza un texto de forma consistente, manejando None y la cadena 'none'."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if text.lower() == 'none':
        return ""

    normalized = unidecode(text.lower().strip())
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def normalize_text_series(text_series: pd.Series) -> pd.Series:
    """Aplica la normalización de texto a una serie de Pandas."""
    if text_series is None or text_series.empty:
        return pd.Series(dtype='str')
    return text_series.astype(str).apply(normalize_text)

def parse_number(s: Optional[Union[str, float, int]]) -> float:
    """Convierte una cadena a número de punto flotante de forma robusta."""
    if s is None:
        return 0.0
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        s = str(s)

    s_cleaned = s.strip()
    # Lista expandida de símbolos a eliminar (de Propuesta 3)
    symbols_to_remove = ['$', '€', '£', '¥', '₹', ' ', '%', 'USD', 'EUR', 'COP']
    for symbol in symbols_to_remove:
        s_cleaned = s_cleaned.replace(symbol, "")

    if not s_cleaned:
        return 0.0

    has_comma = ',' in s_cleaned
    has_dot = '.' in s_cleaned

    if has_comma and has_dot:
        # El último separador es el decimal (lógica de Propuesta 2)
        s_cleaned = s_cleaned.replace('.', '') if s_cleaned.rfind('.') < s_cleaned.rfind(',') else s_cleaned.replace(',', '')
        s_cleaned = s_cleaned.replace(',', '.')
    elif has_comma:
        s_cleaned = s_cleaned.replace(',', '.')

    try:
        return float(s_cleaned)
    except (ValueError, TypeError):
        return 0.0

def clean_apu_code(code: str, validate_format: bool = True) -> str:
    """
    Limpia y valida un código de APU de forma robusta.

    Args:
        code: Código de APU a limpiar
        validate_format: Si True, valida el formato básico

    Returns:
        Código de APU limpio y validado

    Raises:
        ValueError: Si el código es inválido y validate_format=True
    """
    if not isinstance(code, str):
        code = str(code)

    original_code = code
    code = code.strip().upper()

    # Remover caracteres no permitidos (mantener letras, números, puntos, guiones)
    code = re.sub(r'[^\w.\-]', '', code)

    # Remover puntos y guiones al final
    code = code.rstrip('.-')

    # Validaciones opcionales de formato
    if validate_format:
        if not code:
            raise ValueError("Código APU no puede estar vacío")

        if len(code) < 2:
            raise ValueError(f"Código APU demasiado corto: {original_code}")

        # Verificar que tenga al menos un número (para evitar códigos solo con letras)
        if not any(char.isdigit() for char in code):
            logger.warning(f"Código APU sin números: {original_code} -> {code}")

    return code

STANDARD_UNITS = { 'M', 'M2', 'M3', 'ML', 'KM', 'CM', 'MM', 'HR', 'DIA', 'JOR', 'KG', 'TON', 'LB', 'GR', 'L', 'LT', 'GAL', 'UND', 'UN', 'PAR', 'JUEGO', 'KIT', 'VIAJE', '%' }

def normalize_unit(unit: str) -> str:
    """Normaliza y valida una unidad de medida con lógica de precedencia."""
    if not unit or not isinstance(unit, str):
        return 'UND'

    unit_upper = unidecode(unit.upper().strip())

    # 1. Comprobar si ya es una unidad estándar
    if unit_upper in STANDARD_UNITS:
        return unit_upper

    # 2. Usar el mapeo expandido
    unit_mapping = {
        'DIAS': 'DIA',
        'DÍA': 'DIA',
        'DÍAS': 'DIA',
        'JORNAL': 'JOR',
        'JORNALES': 'JOR',
        'HORAS': 'HR',
        'HORA': 'HR',
        'UNIDAD': 'UND',
        'UNIDADES': 'UND',
        'UN': 'UND',
        'METRO': 'M',
        'METROS': 'M',
        'MTS': 'M',
        'METRO2': 'M2',
        'MT2': 'M2',
        'METRO CUADRADO': 'M2',
        'METRO3': 'M3',
        'MT3': 'M3',
        'METRO CUBICO': 'M3',
        'KILOGRAMO': 'KG',
        'KILOS': 'KG',
        'TONELADA': 'TON',
        'GALON': 'GAL',
        'LITRO': 'L',
        'LITROS': 'L',
        'LT': 'L',
        'VIAJES': 'VIAJE',
        'VJE': 'VIAJE'
    }
    # Normalizar espacios para el mapeo
    unit_norm_space = re.sub(r'\s+', ' ', unit_upper)
    if unit_norm_space in unit_mapping:
        return unit_mapping[unit_norm_space]

    # 3. Limpieza final como último recurso
    unit_clean = re.sub(r'[^A-Z0-9]', '', unit_upper)
    if unit_clean in unit_mapping:
        return unit_mapping[unit_clean]
    if unit_clean in STANDARD_UNITS:
        return unit_clean

    logger.debug(f"Unidad no reconocida: '{unit}' -> usando 'UND'")
    return 'UND'

def safe_read_dataframe(path: str, header: int = 0, encoding: str = "auto") -> pd.DataFrame:
    """
    Lee un archivo en DataFrame de forma robusta con detección automática.

    Args:
        path: Ruta al archivo
        header: Fila a usar como encabezado
        encoding: Codificación a usar ("auto" para detección)

    Returns:
        DataFrame leído o DataFrame vacío si falla
    """
    if not path or not os.path.exists(path):
        logger.error(f"Archivo no encontrado: {path}")
        return pd.DataFrame()

    try:
        # Detección automática de encoding
        if encoding == "auto":
            encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings_to_try:
                try:
                    with open(path, 'r', encoding=enc) as f:
                        f.read(1024)  # Leer solo una muestra
                    encoding = enc
                    break
                except UnicodeDecodeError:
                    continue
            else:
                encoding = 'latin1'  # Fallback

        # Leer según extensión
        if path.endswith(".csv"):
            # Intentar detectar separador
            with open(path, 'r', encoding=encoding) as f:
                sample = f.read(4096)

            separators = [',', ';', '\t', '|']
            best_sep = ','
            best_count = 0

            for sep in separators:
                count = sample.count(sep)
                if count > best_count:
                    best_count = count
                    best_sep = sep

            return pd.read_csv(
                path,
                encoding=encoding,
                sep=best_sep,
                engine="python",
                header=header,
                on_bad_lines='skip'
            )

        elif path.endswith((".xls", ".xlsx")):
            return pd.read_excel(path, header=header)
        else:
            logger.error(f"Formato no soportado: {path}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error leyendo archivo {path}: {e}")
        return pd.DataFrame()

def sanitize_for_json(data: any) -> any:
    """Convierte tipos de datos no serializables, incluyendo arrays de NumPy."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    if isinstance(data, np.ndarray):
        return [sanitize_for_json(item) for item in data.tolist()]
    if isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    if isinstance(data, (np.float64, np.float32, np.float16)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    if isinstance(data, np.bool_):
        return bool(data) # De Propuesta 3
    if pd.isna(data):
        return None
    return data

def validate_numeric_value(value, field_name, min_v, max_v, allow_zero):
    """Valida un valor numérico con el orden de comprobación corregido."""
    if not isinstance(value, (int, float, np.number)):
        return False, f"{field_name} debe ser numérico"
    if pd.isna(value):
        return False, f"{field_name} no puede ser nulo"

    # Orden corregido
    if np.isinf(value):
        return False, f"{field_name} no puede ser infinito"
    if not allow_zero and value == 0:
        return False, f"{field_name} no puede ser cero"
    if value < min_v:
        return False, f"{field_name} no puede ser menor que {min_v}"
    if value > max_v:
        return False, f"{field_name} no puede ser mayor que {max_v}"
    return True, ""

def validate_series(series: pd.Series, **kwargs) -> pd.Series:
    """
    Aplica validación numérica a una serie completa.

    Args:
        series: Serie a validar
        **kwargs: Argumentos para validate_numeric_value

    Returns:
        Serie booleana indicando qué valores son válidos
    """
    def validate_single_value(x):
        try:
            return validate_numeric_value(x, **kwargs)[0]
        except:
            return False

    return series.apply(validate_single_value)

def create_apu_signature(apu_data: dict) -> str:
    """
    Crea una firma única para un APU basada en sus datos clave.

    Args:
        apu_data: Diccionario con datos del APU

    Returns:
        Firma única del APU
    """
    key_fields = ['CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU']
    signature_parts = []

    for field in key_fields:
        value = apu_data.get(field, '')
        if value:
            normalized = normalize_text(str(value))
            signature_parts.append(normalized)

    return '|'.join(signature_parts)

def detect_outliers(series: pd.Series, method: str = "iqr") -> pd.Series:
    """
    Detecta valores atípicos en una serie numérica.

    Args:
        series: Serie numérica a analizar
        method: Método de detección ("iqr" o "zscore")

    Returns:
        Serie booleana indicando outliers
    """
    # Remover valores nulos para el cálculo
    clean_series = series.dropna()

    if len(clean_series) == 0:
        return pd.Series([False] * len(series), index=series.index)

    if method == "iqr":
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)

    elif method == "zscore":
        mean = clean_series.mean()
        std = clean_series.std()
        if std == 0:  # Evitar división por cero
            return pd.Series([False] * len(series), index=series.index)
        z_scores = np.abs((series - mean) / std)
        return z_scores > 3

    else:
        raise ValueError(f"Método no soportado: {method}")

def find_and_rename_columns(
    df: pd.DataFrame, column_map: dict
) -> pd.DataFrame:
    """Busca y renombra columnas en un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame en el que se buscarán y renombrarán las columnas.
        column_map (dict): Un diccionario que mapea los nombres de columna
                           estándar a una lista de posibles nombres.

    Returns:
        pd.DataFrame: El DataFrame con las columnas renombradas.
    """
    renamed_cols = {}
    for standard_name, possible_names in column_map.items():
        for col in df.columns:
            if any(str(p_name).lower() in str(col).lower() for p_name in possible_names):
                renamed_cols[col] = standard_name
                break
    return df.rename(columns=renamed_cols)

def sanitize_for_json(data: any) -> any:
    """Convierte tipos de datos no serializables, incluyendo arrays de NumPy."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    if isinstance(data, np.ndarray):
        return [sanitize_for_json(item) for item in data.tolist()]
    if isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    if isinstance(data, (np.float64, np.float32, np.float16)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    if isinstance(data, np.bool_):
        return bool(data) # De Propuesta 3
    if pd.isna(data):
        return None
    return data

# Asegurar que estas funciones estén disponibles
__all__ = [
    'normalize_text',
    'normalize_text_series',
    'parse_number',
    'clean_apu_code',
    'normalize_unit',
    'safe_read_dataframe',
    'validate_numeric_value',
    'validate_series',  # AÑADIDA
    'create_apu_signature',  # AÑADIDA
    'detect_outliers',  # AÑADIDA
    'find_and_rename_columns',
    'sanitize_for_json'
]
