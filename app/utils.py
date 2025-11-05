import os
import re
from typing import Optional, Union

import numpy as np
import pandas as pd
from unidecode import unidecode
import logging

logger = logging.getLogger(__name__)

def normalize_text(text: str, preserve_special_chars: bool = False) -> str:
    """
    Normaliza un texto de forma consistente y robusta.

    Args:
        text: Texto a normalizar
        preserve_special_chars: Si True, preserva algunos caracteres especiales útiles

    Returns:
        Texto normalizado
    """
    if not isinstance(text, str):
        text = str(text)

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
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def normalize_text_series(text_series: pd.Series, preserve_special_chars: bool = False) -> pd.Series:
    """
    Normaliza una serie de texto de forma vectorizada y eficiente.

    Args:
        text_series: Serie de pandas con texto a normalizar
        preserve_special_chars: Si True, preserva algunos caracteres especiales útiles

    Returns:
        Serie de texto normalizada
    """
    if text_series is None or text_series.empty:
        return text_series

    # Asegurar que todos los elementos sean strings
    text_series = text_series.astype(str)

    # Aplicar normalización de forma vectorizada cuando sea posible
    def safe_normalize(x):
        try:
            return normalize_text(x, preserve_special_chars)
        except Exception:
            return str(x)

    return text_series.apply(safe_normalize)

def parse_number(s: Optional[Union[str, float, int]], decimal_separator: str = "auto") -> float:
    """
    Convierte una cadena a número de punto flotante de forma robusta.

    Args:
        s: Valor a convertir (string, float, int)
        decimal_separator: "auto", "comma" o "dot"

    Returns:
        Número convertido o 0.0 si falla
    """
    if s is None:
        return 0.0

    # Si ya es numérico, retornar directamente
    if isinstance(s, (int, float)):
        return float(s)

    if not isinstance(s, str):
        s = str(s)

    s_cleaned = s.strip().replace("$", "").replace(" ", "").replace("%", "")

    if not s_cleaned:
        return 0.0

    # Detección automática del separador decimal
    if decimal_separator == "auto":
        comma_count = s_cleaned.count(',')
        dot_count = s_cleaned.count('.')

        if comma_count > 0 and dot_count > 0:
            # Ambos presentes: asumir que la coma es decimal y el punto de miles
            if comma_count == 1 and s_cleaned.rfind(',') > s_cleaned.rfind('.'):
                decimal_separator = "comma"
            else:
                decimal_separator = "dot"
        elif comma_count == 1 and dot_count == 0:
            decimal_separator = "comma"
        else:
            decimal_separator = "dot"

    # Limpiar según el separador decimal detectado
    if decimal_separator == "comma":
        # Coma como decimal, punto como miles
        s_cleaned = s_cleaned.replace(".", "")  # Eliminar separadores de miles
        s_cleaned = s_cleaned.replace(",", ".")  # Convertir coma decimal a punto
    else:
        # Punto como decimal, coma como miles
        s_cleaned = s_cleaned.replace(",", "")  # Eliminar separadores de miles
        # El punto ya está correcto para float()

    # Manejar casos edge como "1.234.567" (múltiples puntos)
    if s_cleaned.count('.') > 1:
        # Conservar solo el último punto como decimal
        parts = s_cleaned.split('.')
        integer_part = ''.join(parts[:-1])
        decimal_part = parts[-1]
        s_cleaned = f"{integer_part}.{decimal_part}"

    # Validar que el string resultante sea un número válido
    if re.match(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', s_cleaned):
        try:
            return float(s_cleaned)
        except (ValueError, TypeError):
            return 0.0
    else:
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

# Unidades estándar soportadas
STANDARD_UNITS = {
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
}

def normalize_unit(unit: str) -> str:
    """
    Normaliza y valida una unidad de medida.

    Args:
        unit: Unidad a normalizar

    Returns:
        Unidad normalizada o 'UND' si no es válida
    """
    if not unit or not isinstance(unit, str):
        return 'UND'

    unit = unit.upper().strip()

    # Mapeo de unidades equivalentes
    unit_mapping = {
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

    # Aplicar mapeo
    if unit in unit_mapping:
        return unit_mapping[unit]

    # Si no está en el mapeo pero es una unidad estándar, retornarla
    if unit in STANDARD_UNITS:
        return unit

    # Si no es estándar, intentar limpiar y verificar
    clean_unit = re.sub(r'[^A-Z0-9]', '', unit)
    if clean_unit in STANDARD_UNITS:
        return clean_unit

    # Loggear unidades no reconocidas
    if unit not in ('', 'UND'):
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

def validate_numeric_value(value: float, field_name: str = "valor",
                          min_value: float = 0, max_value: float = 1e12,
                          allow_zero: bool = True) -> tuple[bool, str]:
    """
    Valida un valor numérico según criterios configurables.

    Args:
        value: Valor a validar
        field_name: Nombre del campo para mensajes de error
        min_value: Valor mínimo permitido
        max_value: Valor máximo permitido
        allow_zero: Si permite valor cero

    Returns:
        Tuple (es_válido, mensaje_error)
    """
    if not isinstance(value, (int, float)):
        return False, f"{field_name} debe ser numérico"

    if pd.isna(value):
        return False, f"{field_name} no puede ser nulo"

    if not allow_zero and value == 0:
        return False, f"{field_name} no puede ser cero"

    if value < min_value:
        return False, f"{field_name} no puede ser menor que {min_value}"

    if value > max_value:
        return False, f"{field_name} no puede ser mayor que {max_value}"

    if np.isinf(value):
        return False, f"{field_name} no puede ser infinito"

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
    """Convierte tipos de datos no serializables a tipos nativos de Python.

    Recorre recursivamente una estructura de datos (dict, list) y convierte
    tipos no serializables para JSON a tipos nativos de Python.

    Args:
        data (any): La estructura de datos a sanear.

    Returns:
        any: La estructura de datos saneada.
    """
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    # Conversión de tipos de NumPy a Python nativo
    if isinstance(data, (np.int64, np.int32)):
        return int(data)
    if isinstance(data, (np.float64, np.float32)):
        # Manejar NaN específicamente aquí
        return None if np.isnan(data) else float(data)
    # Manejar pd.NA y otros nulos de Pandas
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
