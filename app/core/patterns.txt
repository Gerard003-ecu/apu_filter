import re
from typing import Pattern


class RegexPatterns:
    """Patrones regex pre-compilados para alto rendimiento."""

    # Limpieza básica
    PREFIX_CLEANER: Pattern = re.compile(
        r"^(?:M\.O\.|EQUIPO|TRANSPORTE|HERRAMIENTA|MATERIAL|INSUMO)[\s.:-]*", re.IGNORECASE
    )

    SPECIAL_CHARS: Pattern = re.compile(r"[^a-zA-Z0-9\s]")
    WHITESPACE: Pattern = re.compile(r"\s+")

    # Clasificación por palabras clave
    MATERIALES: Pattern = re.compile(
        r"\b(CEMENTO|ACERO|LADRILLO|ARENA|GRAVA|CONCRETO|TUBERIA|PVC|ALAMBRE)\b",
        re.IGNORECASE,
    )

    MANO_DE_OBRA: Pattern = re.compile(
        r"\b(CUADRILLA|OFICIAL|AYUDANTE|PEON|CAPATAZ|OPERADOR|CONDUCTOR|ALBAÑIL)\b",
        re.IGNORECASE,
    )

    EQUIPO: Pattern = re.compile(
        r"\b(HERRAMIENTA|MENOR|ANDAMIO|EQUIPO|MEZCLADORA|VIBRADOR|COMPACTADOR|RETROEXCAVADORA|VOLQUETA)\b",
        re.IGNORECASE,
    )

    TRANSPORTE: Pattern = re.compile(
        r"\b(TRANSPORTE|ACARREO|CARGUE|DESCARGUE|MOVILIZACION)\b", re.IGNORECASE
    )
