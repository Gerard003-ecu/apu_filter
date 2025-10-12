import re


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
