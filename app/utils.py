from typing import Optional
import re
from unidecode import unidecode
import numpy as np
import pandas as pd


def clean_apu_code(code: str) -> str:
    """Limpia y estandariza un código de APU.

    Convierte comas a puntos y elimina caracteres no numéricos,
    permitiendo un único punto como separador decimal.

    Args:
        code (str): El código de APU a limpiar.

    Returns:
        str: El código de APU estandarizado.
    """
    if not isinstance(code, str):
        code = str(code)
    # Reemplazar comas por puntos y eliminar caracteres no deseados
    cleaned_code = re.sub(r'[^\d.]', '', code.replace(',', '.'))
    # Asegurarse de que no termine con un punto
    return cleaned_code.strip().rstrip('.')

def normalize_text_series(text_series: pd.Series) -> pd.Series:
    """Normaliza un texto para la búsqueda.

    Convierte a minúsculas, elimina tildes, reemplaza caracteres no
    alfanuméricos por espacios y simplifica los espacios en blanco.

    Args:
        text_series (pd.Series): La serie de texto a normalizar.

    Returns:
        pd.Series: La serie de texto normalizada.
    """
    if text_series is None or text_series.empty:
        return text_series

    # Asegurarse de que toda la data sea string
    text_series = text_series.astype(str)

    # Convertir a minúsculas
    text_series = text_series.str.lower()
    # Quitar tildes
    text_series = (
        text_series.str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    # Reemplazar caracteres no alfanuméricos por espacios
    text_series = text_series.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    # Simplificar espacios múltiples a uno solo
    text_series = text_series.str.replace(r"\s+", " ", regex=True).str.strip()
    return text_series


def safe_read_dataframe(path: str, header: int = 0) -> pd.DataFrame:
    """Lee un archivo (CSV o Excel) en un DataFrame de Pandas de forma segura.

    Maneja los errores de archivo no encontrado o de formato.

    Args:
        path (str): La ruta al archivo.
        header (int): El índice de la fila a utilizar como encabezado.

    Returns:
        pd.DataFrame: El DataFrame leído o un DataFrame vacío si ocurre un error.
    """
    try:
        if path.endswith(".csv"):
            return pd.read_csv(
                path, encoding="latin1", sep=None, engine="python", header=header
            )
        elif path.endswith((".xls", ".xlsx")):
            return pd.read_excel(path, header=header)
        else:
            return None
    except FileNotFoundError:
        return None
    except Exception:
        return None


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

def parse_number(s: Optional[str]) -> float:
    """Convierte una cadena a un número de punto flotante de forma segura.

    Args:
        s (Optional[str]): La cadena a convertir.

    Returns:
        float: El número convertido o 0.0 si la conversión falla.
    """
    if not s or not isinstance(s, str):
        return 0.0
    s_cleaned = s.strip().replace("$", "").replace(" ", "")
    if not s_cleaned:
        return 0.0

    # Heurística para manejar formatos numéricos inconsistentes
    if "," in s_cleaned:
        # Si la coma existe, es el separador decimal. Los puntos son de miles.
        s_cleaned = s_cleaned.replace(".", "")
        s_cleaned = s_cleaned.replace(",", ".")
    elif s_cleaned.count('.') > 1:
        # Múltiples puntos sin comas: son separadores de miles.
        s_cleaned = s_cleaned.replace(".", "")
    elif s_cleaned.count('.') == 1:
        # Un solo punto: puede ser decimal o de miles (ej. 80.000).
        integer_part, fractional_part = s_cleaned.split('.')
        if len(fractional_part) == 3 and integer_part != "0":
            # Probablemente es un separador de miles, como en "80.000"
            s_cleaned = integer_part + fractional_part
        # De lo contrario, se asume que es un punto decimal (ej. 0.125, 123.45)

    try:
        return float(s_cleaned)
    except (ValueError, TypeError):
        return 0.0

def normalize_text(text: str) -> str:
    """Normaliza un único string de texto.

    Args:
        text (str): El texto a normalizar.

    Returns:
        str: El texto normalizado.
    """
    if not isinstance(text, str):
        text = str(text)

    normalized = text.lower().strip()
    normalized = unidecode(normalized)
    normalized = re.sub(r"[^a-z0-9\s#\-]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized
