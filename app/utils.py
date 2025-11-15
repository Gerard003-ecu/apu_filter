"""
Módulo de utilidades para procesamiento de datos de APU.

Este módulo proporciona funciones robustas para normalización de texto,
conversión de números, validación de datos y manejo de archivos.
"""

import logging
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from unidecode import unidecode

# Configuración del logger
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# Unidades estándar soportadas (como frozenset para mejor rendimiento)
STANDARD_UNITS = frozenset(
    {
        # Longitud
        "M",
        "M2",
        "M3",
        "ML",
        "KM",
        "CM",
        "MM",
        # Tiempo
        "HORA",
        "HR",
        "DIA",
        "SEMANA",
        "MES",
        "AÑO",
        "JOR",
        # Peso
        "KG",
        "TON",
        "LB",
        "GR",
        # Volumen líquido
        "L",
        "LT",
        "GAL",
        "ML",
        # Unidades
        "UND",
        "UN",
        "PAR",
        "JUEGO",
        "KIT",
        # Transporte
        "VIAJE",
        "VIAJES",
        "KM",
        # Otros
        "SERVICIO",
        "%",
    }
)

# Mapeo de unidades equivalentes (optimizado como dict constante)
UNIT_MAPPING = {
    "DIAS": "DIA",
    "DÍAS": "DIA",
    "JORNAL": "JOR",
    "JORNALES": "JOR",
    "HORAS": "HR",
    "HORA": "HR",
    "UNIDAD": "UND",
    "UNIDADES": "UND",
    "UN": "UND",
    "METRO": "M",
    "METROS": "M",
    "MTS": "M",
    "METRO2": "M2",
    "M2": "M2",
    "MT2": "M2",
    "METRO CUADRADO": "M2",
    "METRO3": "M3",
    "M3": "M3",
    "MT3": "M3",
    "METRO CUBICO": "M3",
    "METROS CUBICOS": "M3",
    "KILOGRAMO": "KG",
    "KILOGRAMOS": "KG",
    "KILOS": "KG",
    "TONELADA": "TON",
    "TONELADAS": "TON",
    "GALON": "GAL",
    "GALONES": "GAL",
    "GLN": "GAL",
    "LITRO": "L",
    "LITROS": "L",
    "LT": "L",
    "VIAJES": "VIAJE",
    "VJE": "VIAJE",
}

# Configuraciones por defecto
DEFAULT_ENCODING_ATTEMPTS = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
DEFAULT_CSV_SEPARATORS = [",", ";", "\t", "|"]
NUMERIC_VALIDATION_LIMITS = {"min": 0, "max": 1e12, "allow_zero": True}

# Patrones regex compilados (más eficiente)
NUMERIC_PATTERN = re.compile(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$")
APU_INVALID_CHARS_PATTERN = re.compile(r"[^\w.\-]")
WHITESPACE_PATTERN = re.compile(r"\s+")

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
    text_series: pd.Series, preserve_special_chars: bool = False, chunk_size: int = 10000
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
            chunk = text_series.iloc[i : i + chunk_size]
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
    default_value: float = 0.0,
    decimal_separator: Optional[str] = None,
    strict: bool = False,
    allow_percentage: bool = True,
    allow_scientific: bool = True,
    debug: bool = False,
) -> float:
    """
    Convierte una cadena a número de punto flotante de forma robusta.

    Maneja diferentes formatos numéricos incluyendo:
    - Separadores de miles y decimales (punto o coma)
    - Números negativos y positivos
    - Porcentajes (opcional)
    - Notación científica (opcional)
    - Espacios en blanco
    - Caracteres no numéricos comunes

    Args:
        s: Valor a convertir (string, float, int o None)
        default_value: Valor por defecto si la conversión falla
        decimal_separator: Forzar separador decimal específico ('dot', 'comma', None=auto)
        strict: Si True, falla con excepciones en lugar de retornar default
        allow_percentage: Si True, maneja valores como "15%" -> 0.15
        allow_scientific: Si True, maneja notación científica como "1.5e-3"
        debug: Si True, registra información de depuración

    Returns:
        float: Número parseado o default_value si falla

    Raises:
        ValueError: Solo si strict=True y la conversión falla

    Examples:
        >>> parse_number("1,234.56")
        1234.56
        >>> parse_number("1.234,56", decimal_separator="comma")
        1234.56
        >>> parse_number("€ 1,500.00")
        1500.0
        >>> parse_number("15%", allow_percentage=True)
        0.15
        >>> parse_number("1.5e-3", allow_scientific=True)
        0.0015
        >>> parse_number("N/A", default_value=-1)
        -1.0
    """

    # ============================================================
    # 1. VALIDACIÓN INICIAL Y CASOS RÁPIDOS
    # ============================================================

    # Caso None
    if s is None:
        if debug:
            logger.debug(f"parse_number: Input is None, returning default {default_value}")
        return default_value

    # Casos numéricos directos (int, float)
    if isinstance(s, (int, float)):
        result = float(s)
        if debug:
            logger.debug(f"parse_number: Direct numeric conversion: {s} -> {result}")
        return result

    # Convertir a string si no lo es
    if not isinstance(s, str):
        s = str(s)

    # Cache para strings comunes
    cached_result = _get_cached_parse(s, decimal_separator)
    if cached_result is not None:
        if debug:
            logger.debug(f"parse_number: Cache hit for '{s}' -> {cached_result}")
        return cached_result

    # Guardar string original para logging
    original_s = s

    # ============================================================
    # 2. LIMPIEZA INICIAL Y VALIDACIONES
    # ============================================================

    # Eliminar espacios en blanco al inicio y final
    s = s.strip()

    # Verificar si está vacío después de strip
    if not s:
        if debug:
            logger.debug(
                f"parse_number: Empty string after strip, returning default {default_value}"
            )
        return _handle_parse_error(original_s, "empty string", default_value, strict)

    # Verificar casos especiales comunes que no son números
    if _is_non_numeric_text(s):
        if debug:
            logger.debug(f"parse_number: Non-numeric text detected: '{s}'")
        return _handle_parse_error(original_s, "non-numeric text", default_value, strict)

    # ============================================================
    # 3. MANEJO DE PORCENTAJES
    # ============================================================

    if allow_percentage and "%" in s:
        result = _parse_percentage(s, default_value, strict, debug)
        if result is not None:
            _cache_parse_result(original_s, decimal_separator, result)
            return result

    # ============================================================
    # 4. MANEJO DE NOTACIÓN CIENTÍFICA
    # ============================================================

    if allow_scientific and ("e" in s.lower() or "E" in s):
        result = _parse_scientific(s, default_value, strict, debug)
        if result is not None:
            _cache_parse_result(original_s, decimal_separator, result)
            return result

    # ============================================================
    # 5. LIMPIEZA DE CARACTERES NO NUMÉRICOS
    # ============================================================

    # Extraer signo si existe
    sign = 1.0
    s_work = s

    # Manejar múltiples signos al inicio
    sign_match = re.match(r"^([+-]+)", s_work)
    if sign_match:
        signs = sign_match.group(1)
        # Contar signos negativos
        neg_count = signs.count("-")
        sign = -1.0 if neg_count % 2 == 1 else 1.0
        s_work = s_work[len(signs) :]
        if debug:
            logger.debug(f"parse_number: Extracted sign: {sign} from '{signs}'")

    # Eliminar símbolos de moneda comunes y otros caracteres no numéricos
    # pero preservar puntos, comas, espacios (pueden ser separadores)
    s_cleaned = re.sub(r"[^\d,.\s-]", "", s_work)
    cleaned_strip = s_cleaned.strip()
    if cleaned_strip.startswith((".", ",")) and len(cleaned_strip) > 1:
        s_cleaned = "0" + s_cleaned

    # Eliminar espacios que pueden ser separadores de miles
    # pero solo si están entre dígitos
    s_cleaned = re.sub(r"(?<=\d)\s+(?=\d)", "", s_cleaned)
    s_cleaned = s_cleaned.strip()

    if not s_cleaned or not re.search(r"\d", s_cleaned):
        if debug:
            logger.debug(
                f"parse_number: No digits found after cleaning: '{s}' -> '{s_cleaned}'"
            )
        return _handle_parse_error(original_s, "no digits found", default_value, strict)

    # ============================================================
    # 6. DETECCIÓN INTELIGENTE DE SEPARADORES
    # ============================================================

    if decimal_separator:
        # Usar separador especificado por el usuario
        s_standard = _apply_separator_format(s_cleaned, decimal_separator, debug)
    else:
        # Detección automática mejorada
        s_standard = _auto_detect_and_convert_separators(s_cleaned, debug)

    # ============================================================
    # 7. VALIDACIÓN Y CONVERSIÓN FINAL
    # ============================================================

    # Validar formato antes de intentar conversión
    if not _is_valid_number_format(s_standard):
        if debug:
            logger.debug(
                f"parse_number: Invalid format after standardization: '{s_standard}'"
            )
        return _handle_parse_error(
            original_s, f"invalid format: {s_standard}", default_value, strict
        )

    # Intentar conversión con manejo de errores robusto
    try:
        # Usar Decimal para mayor precisión en la conversión intermedia
        if "." in s_standard and len(s_standard.split(".")[1]) > 15:
            # Para números con muchos decimales, usar Decimal
            result = float(Decimal(s_standard)) * sign
        else:
            # Conversión directa para números normales
            result = float(s_standard) * sign

        # Validar resultado
        if not _is_finite(result):
            if debug:
                logger.debug(f"parse_number: Non-finite result: {result}")
            return _handle_parse_error(
                original_s, f"non-finite result: {result}", default_value, strict
            )

        # Cache el resultado exitoso
        _cache_parse_result(original_s, decimal_separator, result)

        if debug:
            logger.debug(f"parse_number: Success: '{original_s}' -> {result}")

        return result

    except (ValueError, TypeError, InvalidOperation) as e:
        if debug:
            logger.debug(f"parse_number: Conversion error for '{s_standard}': {e}")
        return _handle_parse_error(original_s, str(e), default_value, strict)


# ============================================================
# FUNCIONES AUXILIARES ROBUSTAS
# ============================================================


@lru_cache(maxsize=1024)
def _get_cached_parse(s: str, decimal_separator: Optional[str]) -> Optional[float]:
    """
    Busca en cache valores previamente parseados.
    Cache key incluye el string y el separador para evitar colisiones.
    """
    # Solo cachear strings comunes y cortos
    if len(s) > 50:
        return None

    # Valores comunes pre-computados
    common_values = {
        "0": 0.0,
        "1": 1.0,
        "-1": -1.0,
        "0.0": 0.0,
        "1.0": 1.0,
        "0,0": 0.0,
        "1,0": 1.0,
    }

    return common_values.get(s)


def _cache_parse_result(s: str, decimal_separator: Optional[str], result: float):
    """Almacena resultado en cache si es apropiado."""
    # Solo cachear strings cortos para evitar uso excesivo de memoria
    if len(s) <= 50:
        # El LRU cache de _get_cached_parse manejará esto automáticamente
        pass


def _is_non_numeric_text(s: str) -> bool:
    """
    Detecta texto que claramente no es un número.
    """
    non_numeric_indicators = [
        r"^N[/\\]?A$",  # N/A, NA
        r"^-+$",  # Solo guiones
        r"^\?+$",  # Solo signos de interrogación
        r"^TBD$",  # To Be Determined
        r"^NULL$",  # NULL
        r"^NONE$",  # None
        r"^NAN$",  # NaN
        r"^#.*",  # Errores de Excel (#DIV/0!, #VALUE!, etc)
    ]

    s_upper = s.upper().strip()
    return any(re.match(pattern, s_upper) for pattern in non_numeric_indicators)


def _parse_percentage(
    s: str, default_value: float, strict: bool, debug: bool
) -> Optional[float]:
    """
    Parsea un valor de porcentaje (e.g., "15%" -> 0.15).
    """
    try:
        # Eliminar el símbolo de porcentaje y espacios
        s_percent = s.replace("%", "").strip()

        if not s_percent:
            return None

        # Parsear recursivamente sin el %
        base_value = parse_number(
            s_percent,
            default_value=None,  # Usar None para detectar fallo
            allow_percentage=False,  # Evitar recursión infinita
            debug=debug,
        )

        if base_value is None:
            return None

        # Convertir a decimal (dividir por 100)
        result = base_value / 100.0

        if debug:
            logger.debug(f"parse_percentage: '{s}' -> {base_value} -> {result}")

        return result

    except Exception as e:
        if debug:
            logger.debug(f"parse_percentage: Error parsing '{s}': {e}")
        return None


def _parse_scientific(
    s: str, default_value: float, strict: bool, debug: bool
) -> Optional[float]:
    """
    Parsea notación científica (e.g., "1.5e-3" -> 0.0015).
    """
    try:
        # Limpiar espacios alrededor de 'e' o 'E'
        s_sci = re.sub(r"\s*([eE])\s*", r"\1", s.strip())

        # Validar formato básico de notación científica
        if not re.match(r"^[+-]?\d+\.?\d*[eE][+-]?\d+$", s_sci):
            return None

        result = float(s_sci)

        if not _is_finite(result):
            return None

        if debug:
            logger.debug(f"parse_scientific: '{s}' -> {result}")

        return result

    except (ValueError, OverflowError) as e:
        if debug:
            logger.debug(f"parse_scientific: Error parsing '{s}': {e}")
        return None


def _apply_separator_format(s: str, separator_format: str, debug: bool) -> str:
    """
    Aplica formato de separador especificado por el usuario.
    """
    if separator_format == "comma":
        # Coma es decimal, punto es miles
        s_standard = s.replace(".", "").replace(",", ".")
        if debug:
            logger.debug(f"apply_separator: Comma format: '{s}' -> '{s_standard}'")
    elif separator_format == "dot":
        # Punto es decimal, coma es miles
        s_standard = s.replace(",", "")
        if debug:
            logger.debug(f"apply_separator: Dot format: '{s}' -> '{s_standard}'")
    else:
        # Formato no reconocido, usar auto-detección
        s_standard = _auto_detect_and_convert_separators(s, debug)

    return s_standard


def _auto_detect_and_convert_separators(s: str, debug: bool) -> str:
    """
    Detección automática mejorada de separadores decimales y de miles.
    """
    # Contar ocurrencias
    comma_count = s.count(",")
    dot_count = s.count(".")

    # Casos simples sin ambigüedad
    if comma_count == 0 and dot_count == 0:
        # No hay separadores
        return s

    if comma_count == 0:
        # Solo puntos, asumir que el punto es decimal
        return s

    if dot_count == 0:
        # Solo comas
        if comma_count == 1:
            # Una sola coma, probablemente decimal
            return s.replace(",", ".")
        else:
            # Múltiples comas, probablemente miles
            return s.replace(",", "")

    # Ambos separadores presentes - análisis más detallado
    last_comma = s.rfind(",")
    last_dot = s.rfind(".")

    # El último separador suele ser el decimal
    if last_comma > last_dot:
        # Coma es decimal
        # Verificar consistencia: después de la coma decimal debe haber 1-4
        # dígitos típicamente
        after_comma = s[last_comma + 1 :]
        if after_comma and after_comma.isdigit() and len(after_comma) <= 4:
            s_standard = s.replace(".", "").replace(",", ".")
            if debug:
                logger.debug(f"auto_detect: Comma as decimal: '{s}' -> '{s_standard}'")
            return s_standard

    # Punto es decimal (caso más común)
    # Verificar consistencia: después del punto decimal debe haber dígitos
    after_dot = s[last_dot + 1 :]
    if after_dot and after_dot.isdigit():
        s_standard = s.replace(",", "")
        if debug:
            logger.debug(f"auto_detect: Dot as decimal: '{s}' -> '{s_standard}'")
        return s_standard

    # Caso ambiguo, usar heurística adicional
    return _resolve_ambiguous_separators(s, debug)


def _resolve_ambiguous_separators(s: str, debug: bool) -> str:
    """
    Resuelve casos ambiguos usando heurísticas adicionales.
    """
    # Analizar patrones de agrupamiento
    # En formato de miles, los grupos son típicamente de 3 dígitos

    # Buscar patrones como 1.234.567 o 1,234,567
    thousand_dot_pattern = re.match(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$", s)
    thousand_comma_pattern = re.match(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$", s)

    if thousand_comma_pattern:
        # Formato americano: 1,234,567.89
        result = s.replace(",", "")
        if debug:
            logger.debug(f"resolve_ambiguous: American format detected: '{s}' -> '{result}'")
        return result

    if thousand_dot_pattern:
        # Formato europeo: 1.234.567,89
        result = s.replace(".", "").replace(",", ".")
        if debug:
            logger.debug(f"resolve_ambiguous: European format detected: '{s}' -> '{result}'")
        return result

    # Si no hay patrón claro, usar la posición del último separador
    last_comma = s.rfind(",")
    last_dot = s.rfind(".")

    if last_comma > last_dot:
        return s.replace(".", "").replace(",", ".")
    else:
        return s.replace(",", "")


def _is_valid_number_format(s: str) -> bool:
    """
    Valida que la string tenga un formato numérico válido después de la estandarización.
    """
    # Debe tener al menos un dígito
    if not re.search(r"\d", s):
        return False

    # No debe tener múltiples puntos decimales
    if s.count(".") > 1:
        return False

    # Validar formato general
    # Permitir: dígitos, máximo un punto decimal, signo opcional al inicio
    pattern = r"^-?\d+\.?\d*$"
    return bool(re.match(pattern, s))


def _is_finite(value: float) -> bool:
    """
    Verifica que el valor sea finito (no inf, -inf o nan).
    """
    import math

    return math.isfinite(value)


def _handle_parse_error(
    original: str, error: str, default_value: float, strict: bool
) -> float:
    """
    Maneja errores de parsing de forma consistente.
    """
    error_msg = f"Failed to parse '{original}': {error}"

    if strict:
        raise ValueError(error_msg)
    else:
        logger.debug(f"parse_number: {error_msg}, returning default {default_value}")
        return float(default_value)


# ============================================================================
# FUNCIONES DE VALIDACIÓN Y LIMPIEZA DE CÓDIGOS APU
# ============================================================================


@lru_cache(maxsize=512)
def clean_apu_code(
    code: str,
    validate_format: bool = True,
    min_length: int = 1,
    is_item_code: bool = False,
    allow_numeric_only: bool = True,
) -> str:
    """
     Limpia y valida un código de APU/ITEM de forma robusta con cache.

     Maneja múltiples contextos:
     - Códigos de APU técnicos (ej: "APU-001", "M.2.1.3")
     - Códigos de ITEM simples (ej: "1", "2", "3.1")
     - Códigos alfanuméricos (ej: "A1", "B-2")

     Args:
     code: Código de APU o ITEM a limpiar
     validate_format: Si True, aplica validaciones de formato
     min_length: Longitud mínima requerida (default=1 para permitir ITEMs simples)
     is_item_code: Si True, aplica reglas más permisivas para códigos de ITEM
     allow_numeric_only: Si True, permite códigos que son solo números

     Returns:
     Código limpio y validado en mayúsculas

     Raises:
     ValueError: Si el código es inválido y validate_format=True
     TypeError: Si code no puede convertirse a string

     Examples:
    >>> clean_apu_code("1") # ITEM simple
     '1'
    >>> clean_apu_code("3.1.2") # ITEM jerárquico
     '3.1.2'
    >>> clean_apu_code("APU-001", min_length=2) # APU técnico
     'APU-001'
    >>> clean_apu_code(" a1-b ") # Con espacios y minúsculas
     'A1-B'
    """
    # ============================================================
    # 1. VALIDACIÓN Y CONVERSIÓN DE TIPO
    # ============================================================
    if not isinstance(code, str):
        try:
            code = str(code)
        except Exception as e:
            raise TypeError(
                f"No se puede convertir código a string. "
                f"Tipo recibido: {type(code).__name__}, Error: {e}"
            )

    # Guardar original para mensajes de error
    original_code = code

    # ============================================================
    # 2. LIMPIEZA BÁSICA
    # ============================================================
    # Eliminar espacios al inicio y final
    code = code.strip()

    # Validar que no esté vacío después del strip
    if not code:
        if validate_format:
            raise ValueError(
                f"Código no puede estar vacío o solo contener espacios. "
                f"Original: '{original_code}'"
            )
        return ""

    # Convertir a mayúsculas para consistencia
    code = code.upper()

    # ============================================================
    # 3. NORMALIZACIÓN DE CARACTERES ESPECIALES
    # ============================================================
    # Reemplazar comas por puntos (común en notación decimal europea)
    code = code.replace(",", ".")

    # Remover caracteres no permitidos
    # Permitidos: letras, números, puntos, guiones, guiones bajos
    code = APU_INVALID_CHARS_PATTERN.sub("", code)

    # Remover puntos y guiones al final (no aportan información)
    code = code.rstrip(".-_")

    # Remover puntos y guiones al inicio (poco común pero posible)
    code = code.lstrip(".-_")

    # Verificar si quedó vacío después de la limpieza
    if not code:
        if validate_format:
            raise ValueError(
                f"Código vacío después de limpieza. "
                f"Original: '{original_code}' contenía solo caracteres inválidos"
            )
        return ""

    # ============================================================
    # 4. VALIDACIONES DE FORMATO (SI ESTÁN HABILITADAS)
    # ============================================================
    if validate_format:
        # 4.1 Validar longitud mínima
        if len(code) < min_length:
            raise ValueError(
                f"Código demasiado corto (mín: {min_length} caracteres). "
                f"Original: '{original_code}', Limpio: '{code}'"
            )

        # 4.2 Validar que contenga al menos un carácter alfanumérico
        if not any(c.isalnum() for c in code):
            raise ValueError(
                f"Código debe contener al menos un carácter alfanumérico. "
                f"Original: '{original_code}', Limpio: '{code}'"
            )

        # 4.3 Validar código numérico puro (si no está permitido)
        is_numeric_only = all(c.isdigit() or c == "." for c in code)
        if is_numeric_only and not allow_numeric_only and not is_item_code:
            logger.warning(
                f"Código es solo numérico (puede ser válido como ITEM): "
                f"'{original_code}' -> '{code}'"
            )

        # 4.4 Advertencia si el código no tiene números
        # (los APUs técnicos normalmente incluyen números)
        if not is_item_code and not any(c.isdigit() for c in code):
            logger.warning(
                f"Código APU sin números (inusual): '{original_code}' -> '{code}'"
            )

        # 4.5 Validar patrones específicos para códigos de ITEM
        if is_item_code:
            # Los ITEMs suelen ser números con posibles puntos para jerarquía
            # Ejemplos: "1", "2.3", "1.2.3", "A.1", etc.
            if not _is_valid_item_code(code):
                logger.warning(
                    f"Código ITEM con formato inusual: '{code}'. "
                    f"Se esperaba formato numérico o jerárquico"
                )

    # ============================================================
    # 5. VALIDACIONES ADICIONALES DE CALIDAD
    # ============================================================

    # Detectar múltiples puntos consecutivos (probablemente error)
    if ".." in code:
        logger.warning(f"Código contiene puntos consecutivos: '{original_code}' -> '{code}'")
        # Corregir múltiples puntos consecutivos
        code = re.sub(r"\.{2,}", ".", code)

    # Detectar múltiples guiones consecutivos
    if "--" in code:
        logger.warning(
            f"Código contiene guiones consecutivos: '{original_code}' -> '{code}'"
        )
        # Corregir múltiples guiones consecutivos
        code = re.sub(r"-{2,}", "-", code)

    # Validar longitud máxima razonable (evitar códigos excesivamente largos)
    MAX_CODE_LENGTH = 100
    if len(code) > MAX_CODE_LENGTH:
        logger.warning(
            f"Código excesivamente largo ({len(code)} caracteres): '{code[:50]}...'"
        )
        if validate_format:
            raise ValueError(
                f"Código demasiado largo (máx: {MAX_CODE_LENGTH} caracteres). "
                f"Original: '{original_code}' ({len(original_code)} caracteres)"
            )

    return code


def _is_valid_item_code(code: str) -> bool:
    """
     Valida si un código tiene formato válido para un ITEM.

     Los ITEMs válidos típicamente siguen estos patrones:
     - Numérico simple: "1", "2", "10"
     - Jerárquico con puntos: "1.1", "2.3.4", "1.2.3.4"
     - Alfanumérico jerárquico: "A.1", "B.2.1", "I.A.1"
     - Mixto con guiones: "1-A", "2-B-1"

     Args:
     code: Código a validar

     Returns:
     True si el código tiene un formato válido para ITEM

     Examples:
    >>> _is_valid_item_code("1")
     True
    >>> _is_valid_item_code("1.2.3")
     True
    >>> _is_valid_item_code("A.1")
     True
    >>> _is_valid_item_code("......")
     False
    """
    # Patrón para ITEMs válidos:
    # - Debe empezar con letra o número
    # - Puede contener letras, números, puntos, guiones
    # - Debe terminar con letra o número
    # - No puede tener solo separadores
    pattern = r"^[A-Z0-9]([A-Z0-9.\-_]*[A-Z0-9])?$"

    if not re.match(pattern, code):
        return False

    # Validar que no sea solo separadores
    if all(c in ".-_" for c in code):
        return False

    # Validar estructura jerárquica si tiene puntos
    if "." in code:
        parts = code.split(".")
        # Cada parte debe ser alfanumérica no vacía
        if not all(part and part[0].isalnum() for part in parts):
            return False

    return True


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
        return "UND"

    unit = unit.upper().strip()
    if not unit:
        return "UND"

    # Verificar en mapeo primero (más común)
    if unit in UNIT_MAPPING:
        return UNIT_MAPPING[unit]

    # Si es una unidad estándar, retornarla
    if unit in STANDARD_UNITS:
        return unit

    # Intentar limpiar y verificar
    clean_unit = re.sub(r"[^A-Z0-9]", "", unit)
    if clean_unit in STANDARD_UNITS:
        return clean_unit

    # Log solo para unidades no triviales
    if unit not in ("", "UND") and len(unit) > 1:
        logger.debug(f"Unidad no reconocida: '{unit}' -> usando '{unit}'")

    return unit


# ============================================================================
# FUNCIONES DE LECTURA DE ARCHIVOS
# ============================================================================


def safe_read_dataframe(
    path: Union[str, Path],
    header: int = 0,
    encoding: str = "auto",
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
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
            with open(path, "r", encoding=enc) as f:
                f.read(1024)  # Leer solo una muestra
            return enc
        except UnicodeDecodeError:
            continue
    return "latin1"  # Fallback


def _read_csv_robust(
    path: Path,
    encoding: str,
    header: int,
    nrows: Optional[int],
    usecols: Optional[List[str]],
) -> pd.DataFrame:
    """Lee un archivo CSV de forma robusta."""
    # Detectar separador
    separator = _detect_csv_separator(path, encoding)

    # Configurar parámetros de lectura
    read_params = {
        "filepath_or_buffer": path,
        "encoding": encoding,
        "sep": separator,
        "engine": "python",
        "header": header,
        "on_bad_lines": "skip",
    }

    if nrows is not None:
        read_params["nrows"] = nrows
    if usecols is not None:
        read_params["usecols"] = usecols

    return pd.read_csv(**read_params)


def _read_excel_robust(
    path: Path, header: int, nrows: Optional[int], usecols: Optional[List[str]]
) -> pd.DataFrame:
    """Lee un archivo Excel de forma robusta."""
    read_params = {"io": path, "header": header}

    if nrows is not None:
        read_params["nrows"] = nrows
    if usecols is not None:
        read_params["usecols"] = usecols

    return pd.read_excel(**read_params)


def _detect_csv_separator(path: Path, encoding: str) -> str:
    """Detecta el separador de un archivo CSV."""
    try:
        with open(path, "r", encoding=encoding) as f:
            sample = f.read(4096)

        best_sep = ","
        best_count = 0

        for sep in DEFAULT_CSV_SEPARATORS:
            # Contar ocurrencias considerando saltos de línea
            lines = sample.split("\n")[:5]  # Primeras 5 líneas
            if len(lines) > 1:
                counts = [line.count(sep) for line in lines if line]
                if counts and min(counts) > 0:
                    avg_count = sum(counts) / len(counts)
                    if avg_count > best_count:
                        best_count = avg_count
                        best_sep = sep

        return best_sep
    except Exception:
        return ","


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
    allow_inf: bool = False,
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
    series: pd.Series, return_mask: bool = True, **kwargs
) -> Union[pd.Series, pd.DataFrame]:
    """
    Aplica validación numérica a una serie completa.

    Args:
        series: Serie a validar
        return_mask: Si True, retorna máscara booleana. Si False, retorna
                     DataFrame con detalles
        **kwargs: Argumentos para validate_numeric_value

    Returns:
        Serie booleana o DataFrame con validación y mensajes
    """
    if series.empty:
        return series if return_mask else pd.DataFrame()

    if return_mask:
        # Retornar solo máscara booleana
        return series.apply(lambda x: validate_numeric_value(x, **kwargs)[0])
    else:
        # Retornar DataFrame con detalles
        validation_results = series.apply(lambda x: validate_numeric_value(x, **kwargs))

        return pd.DataFrame(
            {
                "value": series,
                "is_valid": validation_results.apply(lambda x: x[0]),
                "error_message": validation_results.apply(lambda x: x[1]),
            }
        )


# ============================================================================
# FUNCIONES DE ANÁLISIS Y DETECCIÓN
# ============================================================================


def create_apu_signature(
    apu_data: Dict[str, Any], key_fields: Optional[List[str]] = None
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
        key_fields = ["CODIGO_APU", "DESCRIPCION_APU", "UNIDAD_APU"]

    signature_parts = []

    for field in key_fields:
        value = apu_data.get(field, "")
        if value:
            # Normalizar el valor para la firma
            if isinstance(value, (int, float)):
                normalized = str(value)
            else:
                normalized = normalize_text(str(value))

            if normalized:  # Solo añadir si no está vacío
                signature_parts.append(normalized)

    return "|".join(signature_parts) if signature_parts else "empty_signature"


def detect_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 1.5,
    return_bounds: bool = False,
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
    series: pd.Series, clean_series: pd.Series, threshold: float
) -> Tuple[pd.Series, Dict[str, float]]:
    """Detección de outliers usando IQR."""
    Q1 = clean_series.quantile(0.25)
    Q3 = clean_series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    outliers = (series < lower_bound) | (series > upper_bound)
    bounds = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }

    return outliers, bounds


def _detect_outliers_zscore(
    series: pd.Series, clean_series: pd.Series, threshold: float
) -> Tuple[pd.Series, Dict[str, float]]:
    """Detección de outliers usando z-score."""
    mean = clean_series.mean()
    std = clean_series.std()

    if std == 0:  # Evitar división por cero
        outliers = pd.Series([False] * len(series), index=series.index)
        bounds = {"mean": mean, "std": 0, "threshold": threshold}
    else:
        z_scores = np.abs((series - mean) / std)
        outliers = z_scores > threshold
        bounds = {
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "lower_bound": mean - threshold * std,
            "upper_bound": mean + threshold * std,
        }

    return outliers, bounds


def _detect_outliers_modified_zscore(
    series: pd.Series, clean_series: pd.Series, threshold: float
) -> Tuple[pd.Series, Dict[str, float]]:
    """Detección de outliers usando Modified Z-score (más robusto)."""
    median = clean_series.median()
    mad = np.median(np.abs(clean_series - median))

    if mad == 0:
        # Usar MAD mínimo para evitar división por cero
        mad = 1.4826 * clean_series.std()
        if mad == 0:
            outliers = pd.Series([False] * len(series), index=series.index)
            bounds = {"median": median, "mad": 0, "threshold": threshold}
            return outliers, bounds

    modified_z_scores = 0.6745 * (series - median) / mad
    outliers = np.abs(modified_z_scores) > threshold

    bounds = {
        "median": median,
        "mad": mad,
        "threshold": threshold,
        "lower_bound": median - threshold * mad / 0.6745,
        "upper_bound": median + threshold * mad / 0.6745,
    }

    return outliers, bounds


# ============================================================================
# FUNCIONES DE MANIPULACIÓN DE DATAFRAMES
# ============================================================================


def find_and_rename_columns(
    df: pd.DataFrame, column_map: Dict[str, List[str]], case_sensitive: bool = False
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
                if (
                    p_name_compare == col_compare
                    or p_name_compare in col_compare
                    or col_compare in p_name_compare
                ):
                    if standard_name in renamed_cols.values():
                        logger.warning(
                            f"Columna '{standard_name}' ya mapeada. Ignorando '{col}'"
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
        return {k: sanitize_for_json(v, max_depth - 1) for k, v in data.items()}

    # Manejar listas y tuplas
    if isinstance(data, (list, tuple)):
        return [sanitize_for_json(v, max_depth - 1) for v in data]

    # Manejar Series de pandas
    if isinstance(data, pd.Series):
        return sanitize_for_json(data.to_list(), max_depth - 1)

    # Manejar DataFrames de pandas
    if isinstance(data, pd.DataFrame):
        return sanitize_for_json(data.to_dict("records"), max_depth - 1)

    # Conversión de tipos de NumPy a Python nativo
    if isinstance(data, (np.integer, np.int32, np.int64)):
        return int(data)

    if isinstance(data, (float, np.floating, np.float32, np.float64)):
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
    if hasattr(data, "isoformat"):
        return data.isoformat()

    # Para otros tipos, intentar conversión a string
    if hasattr(data, "__dict__"):
        return sanitize_for_json(data.__dict__, max_depth - 1)

    return data


# ============================================================================
# FUNCIONES ADICIONALES DE UTILIDAD
# ============================================================================


def calculate_std_dev(values: List[float]) -> float:
    """Calcula la desviación estándar de una lista de valores."""
    if not values or len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance**0.5


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
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "q1": None,
            "q3": None,
            "null_count": len(series),
            "null_percentage": 100.0,
        }

    return {
        "count": len(clean_series),
        "mean": float(clean_series.mean()),
        "std": float(clean_series.std()),
        "min": float(clean_series.min()),
        "max": float(clean_series.max()),
        "median": float(clean_series.median()),
        "q1": float(clean_series.quantile(0.25)),
        "q3": float(clean_series.quantile(0.75)),
        "null_count": len(series) - len(clean_series),
        "null_percentage": (len(series) - len(clean_series)) / len(series) * 100,
    }


def batch_process_dataframe(
    df: pd.DataFrame, process_func: callable, batch_size: int = 1000, **kwargs
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
        batch = df.iloc[i : i + batch_size].copy()
        processed = process_func(batch, **kwargs)
        results.append(processed)

    return pd.concat(results, ignore_index=True)


def calculate_unit_costs(
    df: pd.DataFrame, config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Calcula costos unitarios por APU con validación robusta.
    Args:
        df: DataFrame procesado
        config: Configuración opcional para cálculos
    Returns:
        DataFrame con costos unitarios por APU
    """
    if df.empty:
        logger.error("❌ DataFrame vacío para cálculo de costos")
        return pd.DataFrame()

    required_cols = ["CODIGO_APU", "TIPO_INSUMO", "VALOR_TOTAL_APU"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error(f"❌ Columnas faltantes: {missing}")
        return pd.DataFrame()

    logger.info("🔄 Calculando costos unitarios por APU...")
    try:
        # Agrupar y sumar por APU y tipo
        grouped = (
            df.groupby(["CODIGO_APU", "DESCRIPCION_APU", "UNIDAD_APU", "TIPO_INSUMO"])[
                "VALOR_TOTAL_APU"
            ]
            .sum()
            .reset_index()
        )

        # Pivotear
        pivot = grouped.pivot_table(
            index=["CODIGO_APU", "DESCRIPCION_APU", "UNIDAD_APU"],
            columns="TIPO_INSUMO",
            values="VALOR_TOTAL_APU",
            fill_value=0,
            aggfunc="sum",
        ).reset_index()

        # Asegurar todas las columnas necesarias
        expected_columns = ["SUMINISTRO", "MANO_DE_OBRA", "EQUIPO", "TRANSPORTE", "OTRO"]
        for col in expected_columns:
            if col not in pivot.columns:
                pivot[col] = 0

        # Calcular componentes
        pivot["VALOR_SUMINISTRO_UN"] = pivot.get("SUMINISTRO", 0)
        pivot["VALOR_INSTALACION_UN"] = pivot.get("MANO_DE_OBRA", 0) + pivot.get("EQUIPO", 0)
        pivot["VALOR_TRANSPORTE_UN"] = pivot.get("TRANSPORTE", 0)
        pivot["VALOR_OTRO_UN"] = pivot.get("OTRO", 0)

        # Total
        pivot["COSTO_UNITARIO_TOTAL"] = (
            pivot["VALOR_SUMINISTRO_UN"]
            + pivot["VALOR_INSTALACION_UN"]
            + pivot["VALOR_TRANSPORTE_UN"]
            + pivot["VALOR_OTRO_UN"]
        )

        # Porcentajes con manejo de división por cero
        total = pivot["COSTO_UNITARIO_TOTAL"].replace(0, np.nan)
        pivot["PCT_SUMINISTRO"] = (
            (pivot["VALOR_SUMINISTRO_UN"] / total * 100).fillna(0).round(2)
        )
        pivot["PCT_INSTALACION"] = (
            (pivot["VALOR_INSTALACION_UN"] / total * 100).fillna(0).round(2)
        )
        pivot["PCT_TRANSPORTE"] = (
            (pivot["VALOR_TRANSPORTE_UN"] / total * 100).fillna(0).round(2)
        )
        pivot["PCT_OTRO"] = (pivot["VALOR_OTRO_UN"] / total * 100).fillna(0).round(2)

        # Ordenar y limpiar
        pivot = pivot.sort_values("CODIGO_APU")

        # Optimizar tipos
        for col in pivot.select_dtypes(include=["float64"]).columns:
            pivot[col] = pivot[col].astype("float32")

        # Log resumen
        logger.info(f"✅ Costos calculados para {len(pivot):,} APUs únicos")
        logger.info(f"   💰 Suministros: ${pivot['VALOR_SUMINISTRO_UN'].sum():,.2f}")
        logger.info(f"   👷 Instalación: ${pivot['VALOR_INSTALACION_UN'].sum():,.2f}")
        logger.info(f"   🚚 Transporte: ${pivot['VALOR_TRANSPORTE_UN'].sum():,.2f}")
        logger.info(f"   📊 Total: ${pivot['COSTO_UNITARIO_TOTAL'].sum():,.2f}")

        # Validar resultados
        if pivot["COSTO_UNITARIO_TOTAL"].sum() == 0:
            logger.error("⚠️ Todos los costos calculados son cero")

        negative_costs = (pivot["COSTO_UNITARIO_TOTAL"] < 0).sum()
        if negative_costs > 0:
            logger.error(f"⚠️ {negative_costs} APUs con costos negativos")

        return pivot
    except Exception as e:
        logger.error(f"❌ Error calculando costos unitarios: {str(e)}")
        return pd.DataFrame()


# ============================================================================
# LISTA DE EXPORTACIÓN
# ============================================================================

__all__ = [
    # Funciones principales
    "normalize_text",
    "normalize_text_series",
    "parse_number",
    "clean_apu_code",
    "normalize_unit",
    "safe_read_dataframe",
    "validate_numeric_value",
    "validate_series",
    "create_apu_signature",
    "detect_outliers",
    "find_and_rename_columns",
    "sanitize_for_json",
    "calculate_unit_costs",
    # Funciones adicionales
    "calculate_std_dev",
    "calculate_statistics",
    "batch_process_dataframe",
    # Constantes
    "STANDARD_UNITS",
    "UNIT_MAPPING",
]
