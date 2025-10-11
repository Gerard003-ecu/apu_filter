import re


def clean_apu_code(code: str) -> str:
    """Limpia un código de APU para que sea consistente para el merge."""
    if not isinstance(code, str):
        code = str(code)
    return re.sub(r'[^\d,]', '', code.replace('.', ',')).strip().rstrip(',')
