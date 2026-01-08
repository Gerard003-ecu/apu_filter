"""
Procesador APU con arquitectura modular de especialistas y teoría categórica (V2).

Este módulo implementa un sistema avanzado para el procesamiento de datos de
Análisis de Precios Unitarios (APU). Utiliza una arquitectura modular donde
componentes "especialistas", cada uno con una responsabilidad única, colaboran
para interpretar y estructurar líneas de texto con formatos variables.

La versión V2 introduce conceptos de Teoría de Categorías:
- Mónadas (OptionMonad) para manejo seguro de valores y errores.
- Functores y Homomorfismos para transformaciones estructurales.
- Validación de Invariantes Algebraicos y Topológicos.

El `APUProcessor` principal mantiene la compatibilidad con la interfaz esperada
por el `LoadDataStep` del pipeline.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Generic, TypeVar, Callable

import pandas as pd
from lark import Token, Transformer, v_args
from lark.exceptions import LarkError

from .schemas import Equipo, InsumoProcesado, ManoDeObra, Otro, Suministro, Transporte
from .utils import parse_number

logger = logging.getLogger(__name__)


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ENUMS Y DATACLASSES
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


@dataclass
class ParsingStats:
    """Estadísticas detalladas del proceso de parsing."""

    total_lines: int = 0
    successful_parses: int = 0
    lark_parse_errors: int = 0
    lark_unexpected_input: int = 0
    lark_unexpected_chars: int = 0
    transformer_errors: int = 0
    empty_results: int = 0
    fallback_attempts: int = 0
    fallback_successes: int = 0
    cache_hits: int = 0
    failed_lines: List[Dict[str, Any]] = field(default_factory=list)


class TipoInsumo(Enum):
    """Enumeración de tipos de insumo válidos."""

    MANO_DE_OBRA = "MANO_DE_OBRA"
    EQUIPO = "EQUIPO"
    TRANSPORTE = "TRANSPORTE"
    SUMINISTRO = "SUMINISTRO"
    OTRO = "OTRO"


class FormatoLinea(Enum):
    """Enumeración de formatos de línea detectados."""

    MO_COMPLETA = "MO_COMPLETA"
    INSUMO_BASICO = "INSUMO_BASICO"
    DESCONOCIDO = "DESCONOCIDO"


@dataclass
class ValidationThresholds:
    """Umbrales de validación para diferentes tipos de insumos."""

    min_jornal: float = 50000
    max_jornal: float = 10000000
    min_rendimiento: float = 0.001
    max_rendimiento: float = 1000
    max_rendimiento_tipico: float = 100
    min_cantidad: float = 0.001
    max_cantidad: float = 1000000
    min_precio: float = 0.01
    max_precio: float = 1e9


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# MÓNADAS Y ESTRUCTURAS CATEGÓRICAS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

T = TypeVar('T')

class OptionMonad(Generic[T]):
    """
    Mónada Option/Maybe para manejo seguro de valores.
    Permite encadenar operaciones sin excepciones explícitas.
    """

    def __init__(self, value: Optional[T] = None, error: str = ""):
        self._value = value
        self._error = error

    def is_valid(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> T:
        if self._value is None:
            raise ValueError(f"Acceso a valor inválido: {self._error}")
        return self._value

    @property
    def error(self) -> str:
        return self._error

    def bind(self, f: Callable[[T], 'OptionMonad']) -> 'OptionMonad':
        """Operación bind de la mónada (>>=)."""
        if not self.is_valid():
            return self
        try:
            return f(self.value)
        except Exception as e:
            return OptionMonad(error=f"Bind error: {e}")

    def map(self, f: Callable[[T], T]) -> 'OptionMonad':
        """Operación map del functor."""
        if not self.is_valid():
            return self
        try:
            return OptionMonad(f(self.value))
        except Exception as e:
            return OptionMonad(error=f"Map error: {e}")


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# GRAMÁTICA LARK
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

APU_GRAMMAR = r"""
    ?start: line

    // Una línea DEBE tener al menos un campo con contenido significativo.
    // Se define explícitamente la estructura para evitar ambigüedades.
    line: field (SEP field)*

    // Un campo puede estar vacío (para manejar ';;') o contener un valor.
    // MEJORA: Hacemos explícito que un campo vacío es válido estructuralmente.
    field: FIELD_VALUE -> field_with_value
         |              -> field_empty

    // MEJORA: Patrón más restrictivo que excluye caracteres de control
    // y limita la longitud máxima para evitar backtracking excesivo.
    FIELD_VALUE: /[^;\r\n\x00-\x1f]{1,2000}/

    // MEJORA: Separador más estricto, solo punto y coma con espacios opcionales.
    SEP: /[ \t]*;[ \t]*/

    // Terminales ignorados explícitamente.
    %import common.WS_INLINE
    %ignore WS_INLINE
"""


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# COMPONENTES ESPECIALISTAS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class PatternMatcher:
    """
    Especialista en detección de patrones y clasificación de líneas de texto.

    Esta clase encapsula la lógica para identificar si una línea de texto
    corresponde a un encabezado, un resumen, una categoría o si contiene
    ciertos tipos de contenido (numérico, porcentajes), basándose en un
    conjunto de palabras clave y expresiones regulares predefinidas.
    """

    # Palabras clave de encabezado de tabla
    HEADER_KEYWORDS = [
        "DESCRIPCION", "DESCRIPCIÓN", "DESC", "UND", "UNID", "UNIDAD",
        "CANT", "CANTIDAD", "PRECIO", "VALOR", "TOTAL", "DESP",
        "DESPERDICIO", "REND", "RENDIMIENTO", "JORNAL", "ITEM",
        "CODIGO", "CÓDIGO",
    ]

    # Palabras clave de resumen/totalización
    SUMMARY_KEYWORDS = [
        "SUBTOTAL", "TOTAL", "RESUMEN", "SUMA", "TOTALES",
        "ACUMULADO", "GRAN TOTAL", "COSTO DIRECTO",
    ]

    # Categorías típicas (exactas)
    CATEGORY_PATTERNS = [
        r"^MATERIALES?$", r"^MANO\s+DE\s+OBRA$", r"^EQUIPO$",
        r"^TRANSPORTE$", r"^OTROS?$", r"^SERVICIOS?$",
        r"^HERRAMIENTAS?$", r"^SUMINISTROS?$",
    ]

    def __init__(self):
        """Inicializa el PatternMatcher y pre-compila los patrones regex."""
        self._pattern_cache: Dict[str, re.Pattern] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compila todos los patrones regex para optimizar el rendimiento."""
        summary_pattern = "|".join(self.SUMMARY_KEYWORDS)
        self._pattern_cache["summary"] = re.compile(summary_pattern, re.IGNORECASE)

        category_pattern = "|".join(self.CATEGORY_PATTERNS)
        self._pattern_cache["category"] = re.compile(category_pattern, re.IGNORECASE)

        self._pattern_cache["numeric"] = re.compile(r"[\d,.]")
        self._pattern_cache["text"] = re.compile(r"[a-zA-Z]{3,}")
        self._pattern_cache["percentage"] = re.compile(r"\d+\s*%")

    def count_header_keywords(self, text: str) -> int:
        """Cuenta cuántas palabras clave de encabezado están presentes en el texto."""
        text_upper = text.upper()
        return sum(1 for keyword in self.HEADER_KEYWORDS if keyword in text_upper)

    def is_likely_header(self, text: str, field_count: int) -> bool:
        """Determina si una línea es probablemente un encabezado de tabla."""
        keyword_count = self.count_header_keywords(text)

        if field_count <= 2 and keyword_count >= 3:
            return True

        words = text.upper().split()
        if words and len(words) > 2:
            header_word_ratio = sum(1 for w in words if w in self.HEADER_KEYWORDS) / len(words)
            if header_word_ratio > 0.6:
                return True

        return False

    def is_likely_summary(self, text: str, field_count: int) -> bool:
        """Determina si una línea es probablemente un subtotal o resumen."""
        if field_count <= 2 and self._pattern_cache["summary"].search(text):
            return True

        text_stripped = text.strip()
        for keyword in self.SUMMARY_KEYWORDS:
            if text_stripped.upper().startswith(keyword):
                return True

        return False

    def is_likely_category(self, text: str, field_count: int) -> bool:
        """Determina si una línea es probablemente una línea de categoría."""
        if field_count <= 2:
            return bool(self._pattern_cache["category"].match(text.strip()))
        return False

    def is_likely_chapter_header(self, text: str) -> bool:
        """Determina si una línea es probablemente un encabezado de capítulo."""
        text_upper = text.strip().upper()
        if text_upper.startswith(("CAPITULO", "CAPÍTULO", "TITULO", "TÍTULO", "NIVEL")):
            return True

        if (text_upper.isupper() and not self.has_numeric_content(text)
                and len(text) < 100
                and not self.is_likely_category(text, 1)
                and not self.is_likely_header(text, 1)):
            return True

        return False

    def has_numeric_content(self, text: str) -> bool:
        """Verifica si el texto contiene cualquier carácter numérico."""
        return bool(self._pattern_cache["numeric"].search(text))

    def has_percentage(self, text: str) -> bool:
        """Verifica si el texto contiene un símbolo de porcentaje."""
        return bool(self._pattern_cache["percentage"].search(text))


class UnitsValidator:
    """
    Especialista en la validación y normalización de unidades de medida.

    Esta clase centraliza el conocimiento sobre las unidades de medida
    aceptadas, proporcionando métodos para verificar la validez de una unidad
    y para convertirla a un formato canónico estandarizado.
    """

    VALID_UNITS: Set[str] = {
        "UND", "UN", "UNID", "UNIDAD", "UNIDADES",
        "M", "MT", "MTS", "MTR", "MTRS", "METRO", "METROS", "ML",
        "KM", "M2", "MT2", "MTS2", "MTRS2", "METROSCUAD", "METROSCUADRADOS",
        "M3", "MT3", "MTS3", "MTRS3", "METROSCUB", "METROSCUBICOS",
        "HR", "HRS", "HORA", "HORAS", "MIN", "MINUTO", "MINUTOS",
        "DIA", "DIAS", "SEM", "SEMANA", "SEMANAS", "MES", "MESES",
        "JOR", "JORN", "JORNAL", "JORNALES",
        "G", "GR", "GRAMO", "GRAMOS", "KG", "KGS", "KILO", "KILOS",
        "KILOGRAMO", "KILOGRAMOS", "TON", "TONS", "TONELADA", "TONELADAS",
        "LB", "LIBRA", "LIBRAS", "GAL", "GLN", "GALON", "GALONES",
        "LT", "LTS", "LITRO", "LITROS", "ML", "MILILITRO", "MILILITROS",
        "VIAJE", "VIAJES", "VJE", "VJ", "BULTO", "BULTOS",
        "SACO", "SACOS", "PAQ", "PAQUETE", "PAQUETES",
        "GLOBAL", "GLB", "GB",
    }

    @classmethod
    @lru_cache(maxsize=256)
    def normalize_unit(cls, unit: str) -> str:
        """
        Normaliza una unidad a su forma canónica (ej. "Metro" -> "M").
        """
        if not unit:
            return "UND"

        unit_clean = re.sub(r"[^A-Z0-9]", "", unit.upper().strip())

        unit_mappings = {
            "UNID": "UND", "UN": "UND", "UNIDAD": "UND",
            "MT": "M", "MTS": "M", "MTR": "M", "MTRS": "M",
            "JORN": "JOR", "JORNAL": "JOR", "JORNALES": "JOR",
        }

        return unit_mappings.get(
            unit_clean, unit_clean if unit_clean in cls.VALID_UNITS else "UND"
        )

    @classmethod
    def is_valid(cls, unit: str) -> bool:
        """Verifica si una cadena de texto representa una unidad válida."""
        if not unit:
            return False
        unit_clean = re.sub(r"[^A-Z0-9]", "", unit.upper().strip())
        return unit_clean in cls.VALID_UNITS or len(unit_clean) <= 4


class NumericFieldExtractor:
    """
    Especialista en la extracción e identificación de campos numéricos.

    Esta clase es responsable de parsear valores numéricos de cadenas de texto,
    manejando diferentes separadores decimales. Su función más importante es
    la identificación inteligente de valores de "rendimiento" y "jornal".
    """

    def __init__(
        self,
        config: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
        thresholds: Optional[ValidationThresholds] = None,
    ):
        """Inicializa el extractor."""
        self.config = config or {}
        self.profile = profile or {}
        self.pattern_matcher = PatternMatcher()
        self.thresholds = thresholds or ValidationThresholds()
        number_format = self.profile.get("number_format", {})
        self.decimal_separator = number_format.get("decimal_separator")

    def extract_all_numeric_values(
        self, fields: List[str], skip_first: bool = True
    ) -> List[float]:
        """Extrae todos los valores numéricos válidos de una lista de campos."""
        start_idx = 1 if skip_first else 0
        numeric_values = []

        for f in fields[start_idx:]:
            if not f:
                continue

            value = self.parse_number_safe(f)
            if value is not None and value >= 0:
                numeric_values.append(value)

        return numeric_values

    def parse_number_safe(self, value: str) -> Optional[float]:
        """Parsea un número de forma segura, utilizando el separador decimal configurado."""
        if not value or not isinstance(value, str):
            return None
        try:
            return parse_number(value, decimal_separator=self.decimal_separator)
        except (ValueError, TypeError, AttributeError):
            return None

    def identify_mo_values(
        self, numeric_values: List[float]
    ) -> Optional[Tuple[float, float]]:
        """Identifica rendimiento y jornal de una lista de valores numéricos."""
        if len(numeric_values) < 2:
            return None

        jornal_candidates = [
            v for v in numeric_values
            if self.thresholds.min_jornal <= v <= self.thresholds.max_jornal
        ]

        rendimiento_candidates = [
            v for v in numeric_values
            if (self.thresholds.min_rendimiento <= v <= self.thresholds.max_rendimiento_tipico
                and v not in jornal_candidates)
        ]

        if jornal_candidates and rendimiento_candidates:
            jornal = max(jornal_candidates)
            rendimiento = min(rendimiento_candidates)
            return rendimiento, jornal

        if len(numeric_values) >= 2:
            sorted_values = sorted(numeric_values, reverse=True)
            for val in sorted_values:
                if val >= self.thresholds.min_jornal:
                    jornal = val
                    for other_val in numeric_values:
                        if (other_val != jornal
                                and other_val <= self.thresholds.max_rendimiento_tipico):
                            return other_val, jornal
                    break

        return None

    def extract_insumo_values(self, fields: List[str], start_from: int = 2) -> List[float]:
        """Extrae valores numéricos para insumos básicos (no Mano de Obra)."""
        valores = []
        for i in range(start_from, len(fields)):
            if fields[i] and "%" not in fields[i]:
                val = self.parse_number_safe(fields[i])
                if val is not None and val >= 0:
                    valores.append(val)
        return valores


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# TRANSFORMER ORQUESTADOR (V2 CON MÓNADAS Y FUNCTORES COMPLETOS)
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


@v_args(inline=False)
class APUTransformer(Transformer):
    """
    Orquestador que coordina a los especialistas para transformar una línea.
    V2: Implementa lógica categórica con mónadas y functores.
    ROBUSTECIDO: Incluye validación algebraica y normalización categórica.
    """

    _SEP_TOKEN_TYPE = "SEP"
    _FIELD_VALUE_TOKEN_TYPE = "FIELD_VALUE"
    _MIN_FIELDS_FOR_VALID_LINE = 3
    _MAX_DESCRIPTION_LENGTH = 500

    def __init__(
        self,
        apu_context: Dict[str, Any],
        config: Dict[str, Any],
        profile: Dict[str, Any],
        keyword_cache: Any,
    ):
        """Inicializa el Transformer con validación de parámetros."""
        if apu_context is None:
            logger.warning("apu_context es None, usando diccionario vacío")
        if config is None:
            logger.warning("config es None, usando diccionario vacío")

        self.apu_context = apu_context if isinstance(apu_context, dict) else {}
        self.config = config if isinstance(config, dict) else {}
        self.profile = profile if isinstance(profile, dict) else {}
        self.keyword_cache = keyword_cache

        try:
            self.pattern_matcher = PatternMatcher()
            self.units_validator = UnitsValidator()
            # Restauración crítica: Cargar umbrales desde configuración
            self.thresholds = self._load_validation_thresholds()
            self.numeric_extractor = NumericFieldExtractor(
                self.config, self.profile, self.thresholds
            )
        except Exception as e:
            logger.error(f"Error inicializando especialistas: {e}")
            raise RuntimeError(f"Fallo en inicialización de APUTransformer: {e}") from e

        super().__init__()

    def _load_validation_thresholds(self) -> ValidationThresholds:
        """Carga los umbrales de validación desde la configuración."""
        mo_config = self.config.get("validation_thresholds", {}).get("MANO_DE_OBRA", {})
        return ValidationThresholds(
            min_jornal=mo_config.get("min_jornal", 50000),
            max_jornal=mo_config.get("max_jornal", 10000000),
            min_rendimiento=mo_config.get("min_rendimiento", 0.001),
            max_rendimiento=mo_config.get("max_rendimiento", 1000),
            max_rendimiento_tipico=mo_config.get("max_rendimiento_tipico", 100),
        )

    def _extract_value(self, item: Any) -> str:
        """Extrae el valor string de un token o string de forma segura."""
        if item is None:
            return ""
        if isinstance(item, Token):
            raw_value = item.value
            return str(raw_value).strip() if raw_value is not None else ""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, (list, tuple)):
            if not item:
                return ""
            for sub_item in item:
                extracted = self._extract_value(sub_item)
                if extracted:
                    return extracted
            return ""
        try:
            return str(item).strip()
        except Exception as e:
            logger.warning(f"No se pudo convertir a string: {type(item).__name__}, error: {e}")
            return ""

    def field(self, args: List[Any]) -> str:
        """Procesa un campo individual parseado por Lark."""
        if not args:
            return ""
        if len(args) > 1:
            logger.debug(f"field() recibió {len(args)} elementos, concatenando")
            parts = [self._extract_value(arg) for arg in args]
            result = " ".join(filter(None, parts))
        else:
            result = self._extract_value(args[0])

        if len(result) > self._MAX_DESCRIPTION_LENGTH:
            result = result[: self._MAX_DESCRIPTION_LENGTH]

        return result

    def field_with_value(self, args: List[Any]) -> str:
        """Procesa un campo que tiene valor explícito."""
        return self.field(args)

    def field_empty(self, args: List[Any]) -> str:
        """Procesa un campo vacío explícito."""
        return ""

    # --- MÉTODOS CATEGÓRICOS V2 ---

    def _extract_fields_as_monad(self, args: List[Any]) -> OptionMonad[List[str]]:
        """
        Extrae campos usando estructura de mónada Option (Maybe).
        Implementa functor de extracción y validación de homogeneidad.
        """
        if not args:
            return OptionMonad(error="Args vacío")

        fields = []
        structural_errors = []

        for i, arg in enumerate(args):
            # Morfismo de token a valor con validación algebraica
            if isinstance(arg, Token):
                if arg.type == self._SEP_TOKEN_TYPE:
                    continue
                value = self._extract_value(arg)
                if value is not None:
                    # Verificar homogeneidad algebraica si ya hay campos
                    if not self._validate_algebraic_homogeneity(value, len(fields), fields):
                        structural_errors.append(f"Token rompe homogeneidad: {value[:30]}")
                    else:
                        fields.append(value)

            elif isinstance(arg, list):
                # Producto categórico: procesar lista recursivamente
                for sub_arg in arg:
                    if isinstance(sub_arg, Token) and sub_arg.type == self._SEP_TOKEN_TYPE:
                        continue
                    value = self._extract_value(sub_arg)
                    if value is not None:
                         fields.append(value)
            else:
                value = self._extract_value(arg)
                if value is not None:
                     fields.append(value)

        # Si hay errores estructurales, reportar pero permitir continuación si es recuperable
        if structural_errors:
            logger.debug(f"Advertencias estructurales: {structural_errors}")

        # Limpieza categórica: normalizar estructura
        if not fields:
             return OptionMonad(error="Todos los campos son morfismos nulos")

        # Filtrar campos vacíos finales
        clean_fields = self._filter_trailing_empty(fields)

        if not clean_fields:
             return OptionMonad(error="Campos vacíos después de limpieza")

        # Aplicar functor de normalización categórica
        normalized_fields = self._apply_categorical_normalization(clean_fields)

        return OptionMonad(normalized_fields)

    def _validate_algebraic_homogeneity(self, value: str, position: int, context: List[str]) -> bool:
        """
        Valida homogeneidad algebraica: cada campo debe pertenecer al mismo anillo.
        Un campo debe ser compatible con sus vecinos en el anillo de campos.
        """
        if not value:
            return True

        field_type = self._classify_field_algebraic_type(value)

        # Generador del anillo (primer campo) siempre válido
        if position == 0:
            return True

        # Verificar compatibilidad con campo anterior
        if context:
            prev_field = context[-1]
            prev_type = self._classify_field_algebraic_type(prev_field)

            # Reglas de composición algebraica (Definición explicita de morfismos válidos)
            composition_rules = {
                ('ALPHA', 'ALPHA'): True,    # Palabras pueden seguir palabras
                ('ALPHA', 'NUMERIC'): True,   # Números pueden seguir palabras (cantidad)
                ('NUMERIC', 'ALPHA'): True,   # Palabras pueden seguir números (unidad)
                ('NUMERIC', 'NUMERIC'): True, # Números pueden seguir números (precio, total)
                ('ALPHA', 'MIXED_NUMERIC'): True, # Mixto puede seguir palabras
                ('MIXED_NUMERIC', 'ALPHA'): True, # Palabras pueden seguir mixto
                ('OTHER', 'ALPHA'): True,
                ('ALPHA', 'OTHER'): True
            }

            # Permitir transiciones genéricas si no están prohibidas explícitamente
            return composition_rules.get((prev_type, field_type), True)

        return True

    def _classify_field_algebraic_type(self, field: str) -> str:
        """Clasifica un campo en tipos algebraicos básicos."""
        if not field:
            return 'EMPTY'

        # Heurística simple
        if self._looks_numeric(field):
            return 'NUMERIC'

        alpha_chars = sum(c.isalpha() for c in field)
        digit_chars = sum(c.isdigit() for c in field)

        if alpha_chars > digit_chars:
            return 'ALPHA'
        elif digit_chars > alpha_chars:
            return 'MIXED_NUMERIC'

        return 'OTHER'

    def _apply_categorical_normalization(self, fields: List[str]) -> List[str]:
        """Aplica normalización categórica para preservar estructura algebraica."""
        normalized = []
        for i, field in enumerate(fields):
            cleaned = field.strip()

            # Invariante posicional 0 (Descripción): Normalizar espacios
            if i == 0:
                cleaned = ' '.join(cleaned.split())

            # Invariante posicional 1 (Unidad): Normalizar si parece unidad
            elif i == 1:
                cleaned = self.units_validator.normalize_unit(cleaned)

            # Resto: Normalizar numéricos
            else:
                if self._looks_numeric(cleaned):
                    cleaned = self._normalize_numeric_representation(cleaned)

            if len(cleaned) > self._MAX_DESCRIPTION_LENGTH:
                 cleaned = cleaned[:self._MAX_DESCRIPTION_LENGTH]

            normalized.append(cleaned)

        return normalized

    def _looks_numeric(self, field: str) -> bool:
        """Determina si un campo parece ser numérico categóricamente."""
        if not field:
            return False
        # Eliminar caracteres comunes
        cleaned = field.replace(',', '.').replace('$', '').replace('%', '').strip()
        if not cleaned: return False

        # Patrón simple
        return bool(re.match(r'^-?\d+(\.\d+)?$', cleaned))

    def _normalize_numeric_representation(self, field: str) -> str:
        """Normaliza representación numérica a forma canónica (punto decimal)."""
        decimal_sep = self.profile.get('number_format', {}).get('decimal_separator', '.')

        # Lógica heurística para 1.234,56 vs 1,234.56
        if ',' in field and '.' in field:
            if field.rfind('.') > field.rfind(','): # 1,234.56
                field = field.replace(',', '')
            else: # 1.234,56
                field = field.replace('.', '').replace(',', '.')
        elif ',' in field:
            if decimal_sep == '.': # Asumir coma es decimal si config dice punto? O al revés?
                # Si solo hay coma y esperamos punto, probablemente es decimal (1,5)
                # O miles (1,000). Ambigüo.
                # Si el profile dice que el separador es coma, no tocamos.
                # Si el profile dice punto, reemplazamos coma por punto.
                if re.match(r'^\d+,\d+$', field):
                    field = field.replace(',', '.')

        return field

    def _filter_trailing_empty(self, tokens: List[str]) -> List[str]:
        """Elimina campos vacíos al final de una lista de campos."""
        if not tokens:
            return []
        last_non_empty_idx = -1
        for i in range(len(tokens) - 1, -1, -1):
            token = tokens[i]
            if isinstance(token, str) and token.strip():
                last_non_empty_idx = i
                break
        if last_non_empty_idx < 0:
            return []
        return tokens[: last_non_empty_idx + 1]

    def line(self, args: List[Any]) -> Optional[InsumoProcesado]:
        """
        Procesa una línea usando teoría de mónadas.
        V2: Reemplaza lógica imperativa por composición funcional.
        """
        # Functor: Mapear estructura cruda a campos semánticos
        fields_monad = self._extract_fields_as_monad(args)

        if not fields_monad.is_valid():
            logger.debug(f"Línea monádica inválida: {fields_monad.error}")
            return None

        fields = fields_monad.value

        # Composición algebraica de validaciones
        validation_chain = (
            self._validate_minimal_cardinality(fields)
            .bind(lambda f: self._validate_description_epicenter(f))
            # Opcional: .bind(lambda f: self._validate_structural_integrity(f))
        )

        if not validation_chain.is_valid():
            logger.debug(f"Línea falló validación algebraica: {validation_chain.error}")
            return None

        clean_fields = validation_chain.value

        # Detección del grupo fundamental del formato
        formato = self._detect_format_categorical(clean_fields)

        if formato == FormatoLinea.DESCONOCIDO:
            logger.debug(f"Línea: grupo fundamental desconocido para {clean_fields[0][:50]}...")
            return None

        # Homomorfismo: Mapear formato a constructor
        return self._dispatch_builder(formato, clean_fields)

    def _validate_minimal_cardinality(self, fields: List[str]) -> OptionMonad[List[str]]:
        """Valida cardinalidad mínima (teoría de conjuntos)."""
        if len(fields) < self._MIN_FIELDS_FOR_VALID_LINE:
            return OptionMonad(error=f"Cardinalidad insuficiente: {len(fields)}")
        return OptionMonad(fields)

    def _validate_description_epicenter(self, fields: List[str]) -> OptionMonad[List[str]]:
        """Valida que el campo de descripción sea epicéntrico (no vacío)."""
        if not fields or not fields[0] or not fields[0].strip():
             return OptionMonad(error="Descripción vacía")
        return OptionMonad(fields)

    def _detect_format_categorical(self, fields: List[str]) -> FormatoLinea:
        """Detección de formato V2 usando lógica de especialistas existente."""
        return self._detect_format(fields)

    def _detect_format(self, fields: List[str]) -> FormatoLinea:
        """Detecta el formato de la línea usando los especialistas."""
        if not fields or not fields[0]:
            return FormatoLinea.DESCONOCIDO

        descripcion = fields[0].strip()
        num_fields = len(fields)

        if self._is_noise_line(descripcion, num_fields):
            return FormatoLinea.DESCONOCIDO

        tipo_probable = self._classify_insumo(descripcion)

        if tipo_probable == TipoInsumo.MANO_DE_OBRA and num_fields >= 5:
            if self._validate_mo_format(fields):
                return FormatoLinea.MO_COMPLETA

        if num_fields >= 3:
             numeric_values = self.numeric_extractor.extract_all_numeric_values(fields)
             if len(numeric_values) >= 1:
                 return FormatoLinea.INSUMO_BASICO

        return FormatoLinea.DESCONOCIDO

    def _is_noise_line(self, descripcion: str, num_fields: int) -> bool:
        """Detecta si una línea es ruido (encabezado, resumen, etc.)."""
        if self.pattern_matcher.is_likely_summary(descripcion, num_fields):
            return True
        if self.pattern_matcher.is_likely_header(descripcion, num_fields):
            return True
        if self.pattern_matcher.is_likely_category(descripcion, num_fields):
            return True
        return False

    def _validate_mo_format(self, fields: List[str]) -> bool:
        """Valida el formato de Mano de Obra."""
        if len(fields) < 5:
            return False
        numeric_values = self.numeric_extractor.extract_all_numeric_values(fields)
        mo_values = self.numeric_extractor.identify_mo_values(numeric_values)
        return mo_values is not None

    def _dispatch_builder(self, formato: FormatoLinea, tokens: List[str]) -> Optional[InsumoProcesado]:
        """Llama al método constructor adecuado según el formato detectado."""
        builder_map = {
            FormatoLinea.MO_COMPLETA: self._build_mo_completa,
            FormatoLinea.INSUMO_BASICO: self._build_insumo_basico,
        }
        builder = builder_map.get(formato)
        if builder is None:
            return None

        try:
            return builder(tokens)
        except Exception as e:
            logger.error(f"Error construyendo {formato}: {e}")
            return None

    def _build_mo_completa(self, tokens: List[str]) -> Optional[ManoDeObra]:
        """Construye un objeto `ManoDeObra` a partir de una línea de formato completo."""
        try:
            descripcion = tokens[0]
            unidad = self.units_validator.normalize_unit(tokens[1]) if len(tokens) > 1 else "JOR"
            numeric_values = self.numeric_extractor.extract_all_numeric_values(tokens)
            mo_values = self.numeric_extractor.identify_mo_values(numeric_values)

            if not mo_values:
                return None

            rendimiento, jornal = mo_values
            cantidad = 1.0 / rendimiento if rendimiento > 0 else 0
            valor_total = cantidad * jornal

            if cantidad <= 0 or valor_total <= 0:
                return None

            context = self.apu_context.copy()
            context.pop("cantidad_apu", None)
            context.pop("precio_unitario_apu", None)
            return ManoDeObra(
                descripcion_insumo=descripcion,
                unidad_insumo=unidad,
                cantidad=round(cantidad, 6),
                precio_unitario=round(jornal, 2),
                valor_total=round(valor_total, 2),
                rendimiento=round(rendimiento, 6),
                formato_origen="MO_COMPLETA",
                tipo_insumo="MANO_DE_OBRA",
                **context,
            )
        except Exception as e:
            logger.error(f"Error construyendo MO_COMPLETA: {e}")
            return None

    def _build_insumo_basico(self, tokens: List[str]) -> Optional[InsumoProcesado]:
        """Construye un objeto de insumo a partir de una línea de formato básico."""
        try:
            if len(tokens) < 3:
                return None

            descripcion = tokens[0]
            unidad = self.units_validator.normalize_unit(tokens[1]) if len(tokens) > 1 else "UND"
            tipo_insumo = self._classify_insumo(descripcion)

            if unidad == "%" or tipo_insumo == TipoInsumo.OTRO:
                return self._build_insumo_porcentual_o_indirecto(tokens, tipo_insumo, unidad)

            valores = self.numeric_extractor.extract_insumo_values(tokens)
            if len(valores) < 2:
                return None

            cantidad, precio, total = 0.0, 0.0, 0.0
            if len(valores) == 4:
                cantidad, _, precio, total = valores
            elif len(valores) == 3:
                cantidad, precio, total = valores
            elif len(valores) == 2:
                cantidad, precio = valores
                total = cantidad * precio
            else:
                return None

            if cantidad > 0 and total > 0:
                precio = total / cantidad

            if total <= 0 or cantidad <= 0:
                return None

            InsumoClass = self._get_insumo_class(tipo_insumo)
            context = self.apu_context.copy()
            context.pop("cantidad_apu", None)
            context.pop("precio_unitario_apu", None)

            return InsumoClass(
                descripcion_insumo=descripcion,
                unidad_insumo=unidad,
                cantidad=round(cantidad, 6),
                precio_unitario=round(precio, 2),
                valor_total=round(total, 2),
                rendimiento=round(cantidad, 6),
                formato_origen="INSUMO_BASICO",
                tipo_insumo=tipo_insumo.value,
                **context,
            )
        except Exception as e:
            logger.error(f"Error construyendo INSUMO_BASICO: {e}")
            return None

    def _build_insumo_porcentual_o_indirecto(
        self, tokens: List[str], tipo_insumo: TipoInsumo, unidad: str
    ) -> Optional[InsumoProcesado]:
        """Construye un insumo para líneas porcentuales o indirectas."""
        descripcion = tokens[0]
        valores = self.numeric_extractor.extract_all_numeric_values(tokens, skip_first=False)

        if not valores:
            return None

        total = valores[-1]
        if total <= 0:
            return None

        cantidad = 1.0
        precio = total

        InsumoClass = self._get_insumo_class(tipo_insumo)
        context = self.apu_context.copy()
        context.pop("cantidad_apu", None)
        context.pop("precio_unitario_apu", None)

        return InsumoClass(
            descripcion_insumo=descripcion,
            unidad_insumo=unidad,
            cantidad=round(cantidad, 6),
            precio_unitario=round(precio, 2),
            valor_total=round(total, 2),
            rendimiento=0.0,
            formato_origen="INSUMO_INDIRECTO",
            tipo_insumo=tipo_insumo.value,
            **context,
        )

    @lru_cache(maxsize=2048)
    def _classify_insumo(self, descripcion: str) -> TipoInsumo:
        """Clasifica el tipo de insumo basándose en palabras clave."""
        if not descripcion:
            return TipoInsumo.OTRO
        desc_upper = descripcion.upper()
        rules = self.config.get("apu_processor_rules", {})
        special_cases = rules.get("special_cases", {})
        mo_keywords = rules.get("mo_keywords", [])
        equipo_keywords = rules.get("equipo_keywords", [])
        otro_keywords = rules.get("otro_keywords", [])

        for case, tipo_str in special_cases.items():
            if case in desc_upper:
                return TipoInsumo(tipo_str)

        if any(kw in desc_upper for kw in mo_keywords):
            return TipoInsumo.MANO_DE_OBRA
        if any(kw in desc_upper for kw in equipo_keywords):
            return TipoInsumo.EQUIPO
        if any(kw in desc_upper for kw in otro_keywords):
            return TipoInsumo.OTRO

        return TipoInsumo.SUMINISTRO

    def _get_insumo_class(self, tipo_insumo: TipoInsumo):
        """Obtiene la clase de `schemas` correspondiente a un `TipoInsumo`."""
        class_mapping = {
            TipoInsumo.MANO_DE_OBRA: ManoDeObra,
            TipoInsumo.EQUIPO: Equipo,
            TipoInsumo.TRANSPORTE: Transporte,
            TipoInsumo.SUMINISTRO: Suministro,
            TipoInsumo.OTRO: Otro,
        }
        return class_mapping.get(tipo_insumo, Suministro)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# PROCESADOR PRINCIPAL - COMPATIBLE CON LoadDataStep
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class APUProcessor:
    """Procesador de APUs - Métodos de parsing robustecidos."""

    def __init__(
        self,
        config,
        profile: Optional[Dict[str, Any]] = None,
        parse_cache: Optional[Dict[str, Any]] = None,
    ):
        """Inicializa el procesador con cache opcional de parsing."""
        self.config = config
        self.profile = profile or {}
        self.parser = self._initialize_parser()
        self.keyword_cache = {}
        self.current_chapter = "GENERAL"
        self.parse_cache = parse_cache or {}
        self.global_stats = {
            "total_apus": 0,
            "total_insumos": 0,
            "format_detected": None,
        }
        self.parsing_stats = ParsingStats()
        self.debug_mode = self.config.get("debug_mode", False)
        self.raw_records = []
        if self.parse_cache:
            logger.info(f"✓ APUProcessor inicializado con cache de {len(self.parse_cache)} líneas")

    def _detect_record_format(self, records: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Detecta automáticamente el formato de los registros de entrada."""
        if not records:
            return ("unknown", "No hay registros para analizar")
        first_record = records[0]
        if "lines" in first_record:
            return ("grouped", "Formato agrupado (legacy)")
        if "insumo_line" in first_record and "apu_code" in first_record:
            return ("flat", "Formato plano (nuevo)")
        return ("unknown", "Formato no reconocido")

    def _group_flat_records(self, flat_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Agrupa registros planos por APU."""
        logger.info(f"Agrupando {len(flat_records)} registros planos por APU...")
        grouped = defaultdict(lambda: {"lines": [], "_lark_trees": [], "metadata": {}})

        for record in flat_records:
            apu_code = record.get("apu_code", "UNKNOWN")
            insumo_line = record.get("insumo_line", "")
            if insumo_line:
                grouped[apu_code]["lines"].append(insumo_line)
                lark_tree = record.get("_lark_tree")
                grouped[apu_code]["_lark_trees"].append(lark_tree)

            if not grouped[apu_code]["metadata"]:
                grouped[apu_code]["metadata"] = {
                    "apu_code": apu_code,
                    "apu_desc": record.get("apu_desc", ""),
                    "apu_unit": record.get("apu_unit", ""),
                    "category": record.get("category", "INDEFINIDO"),
                    "source_line": record.get("source_line", 0),
                }

        result = []
        for apu_code, data in grouped.items():
            record = {
                "codigo_apu": apu_code,
                "descripcion_apu": data["metadata"].get("apu_desc", ""),
                "unidad_apu": data["metadata"].get("apu_unit", ""),
                "lines": data["lines"],
                "_lark_trees": data["_lark_trees"],
                "category": data["metadata"].get("category", "INDEFINIDO"),
                "source_line": data["metadata"].get("source_line", 0),
            }
            result.append(record)
        return result

    def process_all(self) -> pd.DataFrame:
        """Procesa todos los registros de APU crudos y devuelve un DataFrame."""
        if not self.raw_records:
            return pd.DataFrame()

        format_type, format_desc = self._detect_record_format(self.raw_records)
        self.global_stats["format_detected"] = format_type

        if format_type == "flat":
            processed_records = self._group_flat_records(self.raw_records)
        elif format_type == "grouped":
            processed_records = self.raw_records
        else:
            logger.error("❌ Formato no reconocido.")
            return pd.DataFrame()

        all_results = []
        self.global_stats["total_apus"] = len(processed_records)

        for i, record in enumerate(processed_records):
            try:
                apu_context = self._extract_apu_context(record)
                if "lines" in record and record["lines"]:
                    apu_cache = self._prepare_apu_cache(record)
                    insumos = self._process_apu_lines(record["lines"], apu_context, apu_cache)
                    if insumos:
                        all_results.extend(insumos)
            except Exception as e:
                logger.error(f"Error procesando APU {i}: {e}")
                continue

        self.global_stats["total_insumos"] = len(all_results)
        self._log_global_stats()

        return self._convert_to_dataframe(all_results) if all_results else pd.DataFrame()

    def _prepare_apu_cache(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara el cache de parsing específico para un APU."""
        apu_cache = {}
        if "_lark_trees" in record and record["_lark_trees"]:
            lines = record.get("lines", [])
            trees = record["_lark_trees"]
            for line, tree in zip(lines, trees):
                if tree is not None:
                    apu_cache[line.strip()] = tree
        return {**self.parse_cache, **apu_cache}

    def _extract_apu_context(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae el contexto relevante de un registro de APU."""
        return {
            "codigo_apu": record.get("codigo_apu") or record.get("apu_code", ""),
            "descripcion_apu": (record.get("descripcion_apu") or record.get("apu_desc", "")),
            "unidad_apu": record.get("unidad_apu") or record.get("apu_unit", ""),
            "cantidad_apu": record.get("cantidad_apu", 1.0),
            "precio_unitario_apu": record.get("precio_unitario_apu", 0.0),
            "categoria": record.get("category", "INDEFINIDO"),
        }

    def _process_apu_lines(
        self,
        lines: List[str],
        apu_context: Dict[str, Any],
        line_cache: Optional[Dict[str, Any]] = None,
    ) -> List["InsumoProcesado"]:
        """Procesa líneas de APU con reutilización de cache de parsing."""
        if not lines:
            return []
        if self.parser is None:
            return []

        results = []
        stats = ParsingStats()
        active_cache = self._validate_and_merge_cache(line_cache)
        transformer = APUTransformer(apu_context, self.config, self.profile, self.keyword_cache)

        for line_num, line in enumerate(lines, start=1):
            if not self._is_valid_line(line):
                continue

            stats.total_lines += 1
            line_clean = line.strip()

            if transformer.pattern_matcher.is_likely_chapter_header(line_clean):
                self.current_chapter = line_clean
                continue

            apu_context["capitulo"] = self.current_chapter
            tree = None
            insumo = None

            try:
                cache_key = self._compute_cache_key(line_clean)
                if cache_key in active_cache and self._is_valid_tree(active_cache[cache_key]):
                    tree = active_cache[cache_key]
                    stats.cache_hits += 1

                if tree is None:
                    tree = self._parse_line_safe(line_clean, line_num, stats)

                if tree is not None:
                    insumo = self._transform_tree_safe(tree, transformer, line_clean, line_num, stats)

                if insumo is not None:
                    if self._validate_insumo(insumo):
                        insumo.line_number = line_num
                        results.append(insumo)
                    else:
                        stats.empty_results += 1
                else:
                    pass

            except Exception as e:
                self._handle_unexpected_error(e, line_num, line_clean, apu_context.get("codigo_apu"), stats)

        self._merge_stats(stats)
        return results

    def _validate_and_merge_cache(self, line_cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Valida y combina caches de parsing."""
        combined = {}
        if self.parse_cache: combined.update(self.parse_cache)
        if line_cache: combined.update(line_cache)
        return combined

    def _is_valid_line(self, line: Any) -> bool:
        """Verifica si una línea es válida para procesamiento."""
        if not isinstance(line, str) or not line.strip(): return False
        return len(line.strip()) >= 3

    def _compute_cache_key(self, line: str) -> str:
        """Computa una clave de cache para una línea."""
        return " ".join(line.split())

    def _is_valid_tree(self, tree: Any) -> bool:
        """Verifica que un árbol Lark del cache es válido."""
        return tree is not None and hasattr(tree, "data") and hasattr(tree, "children")

    def _parse_line_safe(self, line: str, line_num: int, stats: ParsingStats) -> Optional[Any]:
        """Parsea una línea de forma segura con manejo específico de errores."""
        from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF, UnexpectedInput
        try:
            return self.parser.parse(line)
        except UnexpectedCharacters as uc:
            stats.lark_unexpected_chars += 1
            logger.debug(f"Carácter inesperado línea {line_num}: {uc}")
            return None
        except UnexpectedToken as ut:
            stats.lark_parse_errors += 1
            logger.debug(f"Token inesperado línea {line_num}: {ut}")
            return None
        except UnexpectedEOF as ueof:
            stats.lark_parse_errors += 1
            logger.debug(f"EOF inesperado línea {line_num}: {ueof}")
            return None
        except UnexpectedInput as ui:
            stats.lark_unexpected_input += 1
            logger.debug(f"Input inesperado línea {line_num}: {ui}")
            return None
        except Exception as e:
            stats.lark_parse_errors += 1
            logger.error(f"Error general Lark línea {line_num}: {e}")
            return None

    def _transform_tree_safe(
        self, tree: Any, transformer: APUTransformer, line: str, line_num: int, stats: ParsingStats
    ) -> Optional[InsumoProcesado]:
        """Transforma un árbol Lark de forma segura."""
        try:
            result = transformer.transform(tree)

            if isinstance(result, list):
                for item in result:
                    if item:
                        stats.successful_parses += 1
                        return item
                stats.empty_results += 1
                return None

            if result is not None:
                stats.successful_parses += 1
                return result

            stats.empty_results += 1
            return None
        except Exception as e:
            stats.transformer_errors += 1
            logger.error(f"Error transformación línea {line_num}: {e}")
            return None

    def _validate_insumo(self, insumo: InsumoProcesado) -> bool:
        """Valida que un insumo tiene los campos mínimos requeridos."""
        if insumo is None: return False
        if not insumo.descripcion_insumo or not insumo.descripcion_insumo.strip(): return False
        return True

    def _handle_unexpected_error(self, error, line_num, line, apu_code, stats):
        """Maneja errores inesperados de forma centralizada."""
        logger.error(f"Error inesperado línea {line_num}: {error}")
        stats.failed_lines.append({"line": line_num, "error": str(error)})

    def _merge_stats(self, apu_stats: ParsingStats):
        """Combina estadísticas de un APU con las globales."""
        self.parsing_stats.total_lines += apu_stats.total_lines
        self.parsing_stats.successful_parses += apu_stats.successful_parses
        self.parsing_stats.lark_parse_errors += apu_stats.lark_parse_errors
        self.parsing_stats.transformer_errors += apu_stats.transformer_errors
        self.parsing_stats.empty_results += apu_stats.empty_results
        self.parsing_stats.cache_hits += apu_stats.cache_hits
        self.parsing_stats.failed_lines.extend(apu_stats.failed_lines)

    def _log_global_stats(self):
        """Registra estadísticas globales del procesamiento."""
        logger.info(f"Stats: {self.parsing_stats}")

    def _initialize_parser(self) -> Optional["Lark"]:
        """Inicializa el parser Lark con validación exhaustiva."""
        try:
            from lark import Lark
            return Lark(APU_GRAMMAR, start="line", parser="lalr", maybe_placeholders=False)
        except Exception as e:
            logger.error(f"Error inicializando Lark: {e}")
            return None

    def _convert_to_dataframe(self, insumos: List[InsumoProcesado]) -> pd.DataFrame:
        """Convierte una lista de objetos `InsumoProcesado` a un DataFrame."""
        records = []
        for insumo in insumos:
            record = {
                "CODIGO_APU": getattr(insumo, "codigo_apu", ""),
                "DESCRIPCION_APU": getattr(insumo, "descripcion_apu", ""),
                "UNIDAD_APU": getattr(insumo, "unidad_apu", ""),
                "DESCRIPCION_INSUMO": getattr(insumo, "descripcion_insumo", ""),
                "UNIDAD_INSUMO": getattr(insumo, "unidad_insumo", ""),
                "CANTIDAD_APU": getattr(insumo, "cantidad", 0.0),
                "PRECIO_UNIT_APU": getattr(insumo, "precio_unitario", 0.0),
                "VALOR_TOTAL_APU": getattr(insumo, "valor_total", 0.0),
                "RENDIMIENTO": getattr(insumo, "rendimiento", 0.0),
                "TIPO_INSUMO": getattr(insumo, "tipo_insumo", "OTRO"),
                "FORMATO_ORIGEN": getattr(insumo, "formato_origen", ""),
                "CATEGORIA": getattr(insumo, "categoria", ""),
                "NORMALIZED_DESC": getattr(insumo, "normalized_desc", ""),
            }
            records.append(record)
        return pd.DataFrame(records)
