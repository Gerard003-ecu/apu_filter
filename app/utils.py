import re

import pandas as pd


def clean_apu_code(code: str) -> str:
    """Limpia un código de APU para que sea consistente para el merge."""
    if not isinstance(code, str):
        code = str(code)
    return re.sub(r'[^\d,]', '', code.replace('.', ',')).strip().rstrip(',')

def normalize_text(text_series):
    """
    Normaliza un texto para búsqueda: convierte a minúsculas, elimina tildes,
    reemplaza caracteres no alfanuméricos por espacios y simplifica espacios.
    """
    if text_series is None or text_series.empty:
        return text_series

    # Asegurarse de que toda la data sea string
    text_series = text_series.astype(str)

    # Convertir a minúsculas
    text_series = text_series.str.lower()
    # Quitar tildes
    text_series = text_series.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    # Reemplazar caracteres no alfanuméricos por espacios
    text_series = text_series.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    # Simplificar espacios múltiples a uno solo
    text_series = text_series.str.replace(r"\s+", " ", regex=True).str.strip()
    return text_series


def safe_read_dataframe(path, header=0):
    """
    Lee un archivo (CSV o Excel) en un DataFrame de Pandas de forma segura,
    manejando errores de archivo no encontrado o de formato.
    """
    try:
        if path.endswith(".csv"):
            return pd.read_csv(path, encoding="latin1", sep=None, engine="python", header=header)
        elif path.endswith((".xls", ".xlsx")):
            return pd.read_excel(path, header=header)
        else:
            return None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def find_and_rename_columns(df, column_map):
    """
    Busca y renombra columnas en un DataFrame basado en un mapa de posibles nombres.
    """
    renamed_cols = {}
    for standard_name, possible_names in column_map.items():
        for col in df.columns:
            if any(str(p_name).lower() in str(col).lower() for p_name in possible_names):
                renamed_cols[col] = standard_name
                break
    return df.rename(columns=renamed_cols)
