import re

def clean_apu_code(code: str) -> str:
    """Limpia un código de APU para que sea consistente para el merge."""
    if not isinstance(code, str):
        code = str(code)
    # Estandariza a comas, elimina todo lo que no sea número o coma, y quita espacios/comas al final.
    return re.sub(r'[^\d,]', '', code.replace('.', ',')).strip().rstrip(',')