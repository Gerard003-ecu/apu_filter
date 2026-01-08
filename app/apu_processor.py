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
# MÓNADAS Y ESTRUCTURAS CATEGÓRICAS (REFINADO)
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

T = TypeVar('T')
U = TypeVar('U')

class OptionMonad(Generic[T]):
    """
    Mónada Option/Maybe para manejo seguro de valores.
    REFINADO: Tipado correcto para functor y composición monádica.
    """

    __slots__ = ('_value', '_error')

    def __init__(self, value: Optional[T] = None, error: str = ""):
        self._value = value
        self._error = error

    @classmethod
    def pure(cls, value: T) -> 'OptionMonad[T]':
        """Inyección monádica (unit/return): T -> M[T]."""
        return cls(value=value)

    @classmethod
    def fail(cls, error: str) -> 'OptionMonad[T]':
        """Constructor de fallo explícito."""
        return cls(value=None, error=error)

    def is_valid(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> T:
        if self._value is None:
            raise ValueError(f"Acceso a valor inválido: {self._error}")
        return self._value

    def get_or_else(self, default: T) -> T:
        """Extracción segura con valor por defecto."""
        return self._value if self._value is not None else default

    @property
    def error(self) -> str:
        return self._error

    def bind(self, f: Callable[[T], 'OptionMonad[U]']) -> 'OptionMonad[U]':
        """
        Operación bind de la mónada (>>=).
        REFINADO: Preserva cadena de errores para trazabilidad.
        """
        if not self.is_valid():
            return OptionMonad.fail(self._error)
        try:
            result = f(self._value)
            if not isinstance(result, OptionMonad):
                return OptionMonad.fail(f"Bind requiere retorno OptionMonad, recibido: {type(result)}")
            return result
        except Exception as e:
            return OptionMonad.fail(f"Bind error [{self._error}] -> {e}")

    def map(self, f: Callable[[T], U]) -> 'OptionMonad[U]':
        """
        Operación map del functor: (T -> U) -> M[T] -> M[U].
        REFINADO: Tipado correcto que permite transformación de tipos.
        """
        if not self.is_valid():
            return OptionMonad.fail(self._error)
        try:
            return OptionMonad.pure(f(self._value))
        except Exception as e:
            return OptionMonad.fail(f"Map error: {e}")

    def flat_map(self, f: Callable[[T], 'OptionMonad[U]']) -> 'OptionMonad[U]':
        """Alias semántico para bind (convención Scala/funcional)."""
        return self.bind(f)

    def filter(self, predicate: Callable[[T], bool], error_msg: str = "Filtro fallido") -> 'OptionMonad[T]':
        """Filtrado monádico con predicado."""
        if not self.is_valid():
            return self
        try:
            if predicate(self._value):
                return self
            return OptionMonad.fail(error_msg)
        except Exception as e:
            return OptionMonad.fail(f"Error en predicado: {e}")

    def __repr__(self) -> str:
        if self.is_valid():
            return f"Some({self._value!r})"
        return f"None({self._error})"


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
        """
        Identifica rendimiento y jornal de una lista de valores numéricos.
        REFINADO: Validación de invariantes y ordenamiento coherente.
        """
        if len(numeric_values) < 2:
            return None

        # Filtrar valores inválidos (negativos o cero)
        valid_values = [v for v in numeric_values if v > 0]

        if len(valid_values) < 2:
            return None

        # Identificar candidatos a jornal (valores altos)
        jornal_candidates = [
            v for v in valid_values
            if self.thresholds.min_jornal <= v <= self.thresholds.max_jornal
        ]

        # Identificar candidatos a rendimiento (valores bajos)
        rendimiento_candidates = [
            v for v in valid_values
            if self.thresholds.min_rendimiento <= v <= self.thresholds.max_rendimiento_tipico
        ]

        # Caso ideal: hay candidatos claros para ambos
        if jornal_candidates and rendimiento_candidates:
            # Tomar el jornal más alto y el rendimiento más bajo
            jornal = max(jornal_candidates)
            # Filtrar rendimientos que sean menores al jornal (invariante algebraico)
            valid_rendimientos = [r for r in rendimiento_candidates if r < jornal]

            if valid_rendimientos:
                rendimiento = min(valid_rendimientos)
                return (rendimiento, jornal)

        # Caso alternativo: inferir por orden de magnitud
        if len(valid_values) >= 2:
            sorted_values = sorted(valid_values)

            # El valor más grande podría ser jornal
            potential_jornal = sorted_values[-1]

            # Buscar un valor que pueda ser rendimiento
            for val in sorted_values[:-1]:
                ratio = potential_jornal / val if val > 0 else float('inf')

                # Heurística: el jornal debería ser significativamente mayor que el rendimiento
                # Un jornal típico es >1000 veces el rendimiento
                if ratio >= 100 and val <= self.thresholds.max_rendimiento:
                    # Validar que el potencial jornal está en rango
                    if self.thresholds.min_jornal <= potential_jornal <= self.thresholds.max_jornal:
                        return (val, potential_jornal)

        # Fallback: usar los dos valores extremos si son coherentes
        if len(valid_values) >= 2:
            min_val = min(valid_values)
            max_val = max(valid_values)

            if (min_val <= self.thresholds.max_rendimiento_tipico and
                self.thresholds.min_jornal <= max_val <= self.thresholds.max_jornal and
                max_val > min_val * 10):  # El jornal debe ser al menos 10x el rendimiento
                return (min_val, max_val)

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
        """
        Carga los umbrales de validación desde la configuración.
        REFINADO: Manejo robusto de configuración ausente o malformada.
        """
        defaults = ValidationThresholds()

        if not self.config:
            logger.debug("Config vacía, usando umbrales por defecto")
            return defaults

        try:
            validation_config = self.config.get("validation_thresholds", {})
            if not isinstance(validation_config, dict):
                logger.warning("validation_thresholds no es un diccionario, usando defaults")
                return defaults

            mo_config = validation_config.get("MANO_DE_OBRA", {})
            if not isinstance(mo_config, dict):
                mo_config = {}

            return ValidationThresholds(
                min_jornal=self._safe_float(mo_config.get("min_jornal"), defaults.min_jornal),
                max_jornal=self._safe_float(mo_config.get("max_jornal"), defaults.max_jornal),
                min_rendimiento=self._safe_float(mo_config.get("min_rendimiento"), defaults.min_rendimiento),
                max_rendimiento=self._safe_float(mo_config.get("max_rendimiento"), defaults.max_rendimiento),
                max_rendimiento_tipico=self._safe_float(
                    mo_config.get("max_rendimiento_tipico"), defaults.max_rendimiento_tipico
                ),
                min_cantidad=self._safe_float(mo_config.get("min_cantidad"), defaults.min_cantidad),
                max_cantidad=self._safe_float(mo_config.get("max_cantidad"), defaults.max_cantidad),
                min_precio=self._safe_float(mo_config.get("min_precio"), defaults.min_precio),
                max_precio=self._safe_float(mo_config.get("max_precio"), defaults.max_precio),
            )
        except Exception as e:
            logger.error(f"Error cargando umbrales de validación: {e}, usando defaults")
            return defaults

    def _safe_float(self, value: Any, default: float) -> float:
        """Conversión segura a float con valor por defecto."""
        if value is None:
            return default
        try:
            result = float(value)
            if result <= 0 and default > 0:
                # No permitir valores no positivos para umbrales que deben ser positivos
                return default
            return result
        except (ValueError, TypeError):
            return default

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
        REFINADO: Validación consistente y manejo robusto de estructuras anidadas.
        """
        if not args:
            return OptionMonad.fail("Args vacío - imposible extraer campos")

        fields: List[str] = []
        structural_warnings: List[str] = []

        def extract_and_validate(item: Any, position: int, accumulated: List[str]) -> Optional[str]:
            """Extractor con validación de homogeneidad integrada."""
            if isinstance(item, Token):
                if item.type == self._SEP_TOKEN_TYPE:
                    return None
                value = self._extract_value(item)
            elif isinstance(item, str):
                value = item.strip()
            else:
                value = self._extract_value(item)

            if value is None:
                return None

            value = value.strip()
            if not value:
                return ""  # Campo vacío explícito, válido estructuralmente

            # Validar homogeneidad solo para campos no vacíos
            if not self._validate_algebraic_homogeneity(value, position, accumulated):
                structural_warnings.append(
                    f"Campo {position} rompe homogeneidad: '{value[:30]}...'"
                )
                # Permitir continuación pero registrar advertencia

            return value

        # Procesamiento con tracking de posición
        position = 0
        for arg in args:
            if isinstance(arg, Token):
                extracted = extract_and_validate(arg, position, fields)
                if extracted is not None:
                    fields.append(extracted)
                    position += 1

            elif isinstance(arg, (list, tuple)):
                # Aplanar estructuras anidadas preservando orden
                for sub_arg in arg:
                    if isinstance(sub_arg, Token) and sub_arg.type == self._SEP_TOKEN_TYPE:
                        continue
                    extracted = extract_and_validate(sub_arg, position, fields)
                    if extracted is not None:
                        fields.append(extracted)
                        position += 1
            else:
                extracted = extract_and_validate(arg, position, fields)
                if extracted is not None:
                    fields.append(extracted)
                    position += 1

        # Log de advertencias estructurales (no bloquean, solo informan)
        if structural_warnings:
            logger.debug(f"Advertencias de homogeneidad: {structural_warnings[:3]}")

        # Validación de resultado
        if not fields:
            return OptionMonad.fail("Extracción produjo lista vacía")

        # Limpieza de campos vacíos trailing
        clean_fields = self._filter_trailing_empty(fields)

        if not clean_fields:
            return OptionMonad.fail("Todos los campos son vacíos después de limpieza")

        # Verificar que al menos hay contenido significativo
        non_empty_count = sum(1 for f in clean_fields if f.strip())
        if non_empty_count == 0:
            return OptionMonad.fail("Sin contenido significativo en campos")

        # Aplicar normalización categórica
        normalized_fields = self._apply_categorical_normalization(clean_fields)

        return OptionMonad.pure(normalized_fields)

    def _validate_algebraic_homogeneity(self, value: str, position: int, context: List[str]) -> bool:
        """
        Valida homogeneidad algebraica usando teoría de anillos.
        REFINADO: Cobertura completa de transiciones y manejo de tipos especiales.

        El espacio de campos forma un monoide bajo concatenación, donde cada
        transición entre tipos debe preservar la estructura semántica del APU.
        """
        if not value or not value.strip():
            return True  # Elemento neutro del monoide

        field_type = self._classify_field_algebraic_type(value)

        # Generador del anillo (posición 0) siempre válido - axioma de identidad
        if position == 0:
            return True

        if not context:
            return True

        prev_field = context[-1]
        prev_type = self._classify_field_algebraic_type(prev_field)

        # Matriz de transiciones válidas (morfismos del anillo de campos)
        # Definición explícita de todas las composiciones permitidas
        valid_transitions: Dict[Tuple[str, str], bool] = {
            # Transiciones desde ALPHA
            ('ALPHA', 'ALPHA'): True,        # Descripción continuada
            ('ALPHA', 'NUMERIC'): True,      # Descripción -> valor numérico
            ('ALPHA', 'MIXED_NUMERIC'): True,# Descripción -> código mixto
            ('ALPHA', 'EMPTY'): True,        # Campo opcional vacío
            ('ALPHA', 'OTHER'): True,        # Caracteres especiales permitidos

            # Transiciones desde NUMERIC
            ('NUMERIC', 'ALPHA'): True,      # Valor -> unidad/texto
            ('NUMERIC', 'NUMERIC'): True,    # Secuencia de valores
            ('NUMERIC', 'MIXED_NUMERIC'): True,
            ('NUMERIC', 'EMPTY'): True,
            ('NUMERIC', 'OTHER'): False,     # Números no deben preceder símbolos aislados

            # Transiciones desde MIXED_NUMERIC
            ('MIXED_NUMERIC', 'ALPHA'): True,
            ('MIXED_NUMERIC', 'NUMERIC'): True,
            ('MIXED_NUMERIC', 'MIXED_NUMERIC'): True,
            ('MIXED_NUMERIC', 'EMPTY'): True,
            ('MIXED_NUMERIC', 'OTHER'): False,

            # Transiciones desde EMPTY
            ('EMPTY', 'ALPHA'): True,
            ('EMPTY', 'NUMERIC'): True,
            ('EMPTY', 'MIXED_NUMERIC'): True,
            ('EMPTY', 'EMPTY'): True,        # Campos vacíos consecutivos (sparse data)
            ('EMPTY', 'OTHER'): True,

            # Transiciones desde OTHER
            ('OTHER', 'ALPHA'): True,
            ('OTHER', 'NUMERIC'): True,
            ('OTHER', 'MIXED_NUMERIC'): True,
            ('OTHER', 'EMPTY'): True,
            ('OTHER', 'OTHER'): False,       # Evitar ruido acumulativo
        }

        transition_key = (prev_type, field_type)
        return valid_transitions.get(transition_key, False)

    def _classify_field_algebraic_type(self, field: str) -> str:
        """
        Clasifica un campo en tipos algebraicos del anillo de campos.
        REFINADO: Manejo de casos especiales y porcentajes.
        """
        if not field:
            return 'EMPTY'

        field_stripped = field.strip()
        if not field_stripped:
            return 'EMPTY'

        # Detectar porcentajes como tipo especial de NUMERIC
        if re.match(r'^-?\d+([.,]\d+)?\s*%$', field_stripped):
            return 'NUMERIC'

        # Detectar moneda como NUMERIC
        if re.match(r'^[$€£¥]?\s*-?\d+([.,]\d+)?$', field_stripped):
            return 'NUMERIC'

        # Numérico puro (incluye separadores de miles)
        if self._looks_numeric(field_stripped):
            return 'NUMERIC'

        # Conteo de caracteres para clasificación
        alpha_chars = sum(1 for c in field_stripped if c.isalpha())
        digit_chars = sum(1 for c in field_stripped if c.isdigit())
        total_alnum = alpha_chars + digit_chars

        if total_alnum == 0:
            return 'OTHER'

        alpha_ratio = alpha_chars / total_alnum if total_alnum > 0 else 0
        digit_ratio = digit_chars / total_alnum if total_alnum > 0 else 0

        if alpha_ratio >= 0.7:
            return 'ALPHA'
        elif digit_ratio >= 0.7:
            return 'MIXED_NUMERIC'
        elif digit_chars > 0 and alpha_chars > 0:
            return 'MIXED_NUMERIC'

        return 'OTHER'

    def _apply_categorical_normalization(self, fields: List[str]) -> List[str]:
        """
        Aplica normalización categórica preservando invariantes algebraicos.
        REFINADO: Normalización posicional consistente.
        """
        if not fields:
            return fields

        normalized: List[str] = []

        for i, field in enumerate(fields):
            if field is None:
                normalized.append("")
                continue

            cleaned = field.strip()

            # Posición 0 (Descripción): Normalizar espacios múltiples
            if i == 0:
                cleaned = ' '.join(cleaned.split())
                # Capitalización consistente para descripción
                if cleaned and cleaned.isupper() and len(cleaned) > 20:
                    # Mantener mayúsculas para descripciones cortas (códigos)
                    # pero normalizar descripciones largas
                    cleaned = cleaned.title()

            # Posición 1 (Unidad): Normalizar a forma canónica
            elif i == 1:
                if cleaned and not self._looks_numeric(cleaned):
                    cleaned = self.units_validator.normalize_unit(cleaned)

            # Posiciones numéricas: Normalizar representación
            else:
                if cleaned and self._looks_numeric(cleaned):
                    cleaned = self._normalize_numeric_representation(cleaned)

            # Aplicar límite de longitud
            if len(cleaned) > self._MAX_DESCRIPTION_LENGTH:
                cleaned = cleaned[:self._MAX_DESCRIPTION_LENGTH]

            normalized.append(cleaned)

        return normalized

    def _looks_numeric(self, field: str) -> bool:
        """
        Determina si un campo es numéricamente interpretable.
        REFINADO: Soporte para formatos internacionales y moneda.
        """
        if not field:
            return False

        # Normalizar: remover símbolos monetarios y espacios
        cleaned = re.sub(r'[$€£¥\s]', '', field.strip())
        if not cleaned:
            return False

        # Patrón para números con separadores de miles y decimales
        # Soporta: 1234.56, 1,234.56, 1.234,56, 1234,56
        patterns = [
            r'^-?\d+$',                           # Entero simple
            r'^-?\d+[.,]\d+$',                    # Decimal simple
            r'^-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?$',  # Con separadores de miles
        ]

        return any(re.match(p, cleaned) for p in patterns)

    def _normalize_numeric_representation(self, field: str) -> str:
        """
        Normaliza representación numérica a forma canónica.
        REFINADO: Desambiguación robusta de separadores basada en heurísticas.
        """
        if not field:
            return field

        original = field.strip()

        # Remover símbolos monetarios pero preservar signo
        cleaned = re.sub(r'[$€£¥]', '', original).strip()

        if not cleaned:
            return original

        # Obtener separador decimal del profile
        configured_decimal = self.profile.get('number_format', {}).get('decimal_separator', '.')

        has_comma = ',' in cleaned
        has_period = '.' in cleaned

        # Caso 1: Solo un tipo de separador
        if has_comma and not has_period:
            # Determinar si la coma es decimal o miles
            # Si hay un solo grupo después de la coma con 1-2 dígitos, probablemente es decimal
            match = re.match(r'^-?(\d+),(\d+)$', cleaned)
            if match:
                decimal_part = match.group(2)
                if len(decimal_part) <= 2:
                    # Probablemente decimal: 1,5 o 1,50
                    return cleaned.replace(',', '.')
                elif len(decimal_part) == 3:
                    # Ambiguo: 1,234 podría ser mil o 1.234
                    # Usar configuración del profile
                    if configured_decimal == ',':
                        return cleaned.replace(',', '.')
                    else:
                        # Asumir miles
                        return cleaned.replace(',', '')
                else:
                    # Múltiples dígitos después de coma, probablemente decimal
                    return cleaned.replace(',', '.')

            # Múltiples comas: definitivamente separador de miles (1,234,567)
            if cleaned.count(',') > 1:
                return cleaned.replace(',', '')

        elif has_period and not has_comma:
            # Similar lógica para puntos
            match = re.match(r'^-?(\d+)\.(\d+)$', cleaned)
            if match:
                decimal_part = match.group(2)
                if len(decimal_part) <= 2:
                    return cleaned  # Ya está en formato correcto
                elif len(decimal_part) == 3:
                    if configured_decimal == '.':
                        return cleaned
                    else:
                        return cleaned.replace('.', '')

            if cleaned.count('.') > 1:
                return cleaned.replace('.', '')

        # Caso 2: Ambos separadores presentes
        elif has_comma and has_period:
            last_comma = cleaned.rfind(',')
            last_period = cleaned.rfind('.')

            if last_period > last_comma:
                # Formato: 1,234.56 (coma miles, punto decimal)
                return cleaned.replace(',', '')
            else:
                # Formato: 1.234,56 (punto miles, coma decimal)
                return cleaned.replace('.', '').replace(',', '.')

        return cleaned

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
        Procesa una línea usando composición monádica.
        REFINADO: Cadena de transformaciones con manejo explícito de errores.
        """
        # Paso 1: Extracción monádica de campos
        fields_monad = self._extract_fields_as_monad(args)

        if not fields_monad.is_valid():
            logger.debug(f"Extracción fallida: {fields_monad.error}")
            return None

        # Paso 2: Cadena de validaciones algebraicas
        # Composición: validate_cardinality >>= validate_description >>= validate_structure
        validation_result = (
            self._validate_minimal_cardinality(fields_monad.value)
            .bind(self._validate_description_epicenter)
            .bind(self._validate_structural_integrity)
        )

        if not validation_result.is_valid():
            logger.debug(f"Validación fallida: {validation_result.error}")
            return None

        clean_fields = validation_result.value

        # Paso 3: Clasificación con detección de ruido
        descripcion = clean_fields[0].strip() if clean_fields else ""
        num_fields = len(clean_fields)

        if self._is_noise_line(descripcion, num_fields):
            logger.debug(f"Línea clasificada como ruido: {descripcion[:40]}...")
            return None

        # Paso 4: Detección del formato
        formato = self._detect_format(clean_fields)

        if formato == FormatoLinea.DESCONOCIDO:
            logger.debug(f"Formato desconocido para: {descripcion[:40]}...")
            return None

        # Paso 5: Construcción del insumo mediante dispatch
        result = self._dispatch_builder(formato, clean_fields)

        if result is None:
            logger.debug(f"Constructor retornó None para formato {formato}")

        return result

    def _validate_minimal_cardinality(self, fields: List[str]) -> OptionMonad[List[str]]:
        """Valida cardinalidad mínima (teoría de conjuntos)."""
        if len(fields) < self._MIN_FIELDS_FOR_VALID_LINE:
            return OptionMonad.fail(f"Cardinalidad insuficiente: {len(fields)}")
        return OptionMonad.pure(fields)

    def _validate_description_epicenter(self, fields: List[str]) -> OptionMonad[List[str]]:
        """Valida que el campo de descripción sea epicéntrico (no vacío)."""
        if not fields or not fields[0] or not fields[0].strip():
             return OptionMonad.fail("Descripción vacía")
        return OptionMonad.pure(fields)

    def _validate_structural_integrity(self, fields: List[str]) -> OptionMonad[List[str]]:
        """
        Valida integridad estructural usando teoría de grafos.
        REFINADO: Construcción correcta del grafo y verificación de conectividad.
        """
        if not fields:
            return OptionMonad.fail("Lista de campos vacía")

        n = len(fields)

        # Para líneas muy cortas, la conectividad es trivial
        if n <= 2:
            return OptionMonad.pure(fields)

        graph = self._build_field_dependency_graph(fields)

        if not self._is_graph_connected(graph, n):
            # Intentar recuperación: verificar si la desconexión es por campos vacíos
            non_empty_indices = [i for i, f in enumerate(fields) if f.strip()]
            if len(non_empty_indices) >= 2:
                # Reconstruir grafo solo con campos no vacíos
                subgraph = self._build_induced_subgraph(graph, non_empty_indices)
                if self._is_graph_connected(subgraph, len(non_empty_indices)):
                    return OptionMonad.pure(fields)

            return OptionMonad.fail(
                f"Grafo de campos no conexo: {len(graph)} nodos con aristas de {n} totales"
            )

        return OptionMonad.pure(fields)

    def _build_field_dependency_graph(self, fields: List[str]) -> Dict[int, Set[int]]:
        """
        Construye grafo de dependencias entre campos.
        REFINADO: Todos los nodos incluidos, aristas bidireccionales.
        """
        n = len(fields)
        # Inicializar todos los nodos (incluso aislados)
        graph: Dict[int, Set[int]] = {i: set() for i in range(n)}

        for i in range(n):
            # Dependencias posicionales (adyacencia lineal)
            if i > 0:
                graph[i].add(i - 1)
                graph[i - 1].add(i)

            # Dependencias semánticas (campos relacionados)
            for j in range(i + 1, n):
                if self._fields_are_semantically_related(fields[i], fields[j]):
                    graph[i].add(j)
                    graph[j].add(i)

        return graph

    def _build_induced_subgraph(
        self, graph: Dict[int, Set[int]], nodes: List[int]
    ) -> Dict[int, Set[int]]:
        """Construye subgrafo inducido por un conjunto de nodos."""
        node_set = set(nodes)
        # Reindexar nodos a 0..len(nodes)-1
        old_to_new = {old: new for new, old in enumerate(nodes)}

        subgraph: Dict[int, Set[int]] = {i: set() for i in range(len(nodes))}

        for old_node in nodes:
            new_node = old_to_new[old_node]
            for neighbor in graph.get(old_node, set()):
                if neighbor in node_set:
                    subgraph[new_node].add(old_to_new[neighbor])

        return subgraph

    def _is_graph_connected(self, graph: Dict[int, Set[int]], expected_nodes: int) -> bool:
        """
        Verifica si un grafo es conexo usando BFS.
        REFINADO: Manejo correcto del número esperado de nodos.
        """
        if expected_nodes == 0:
            return True

        if expected_nodes == 1:
            return len(graph) >= 1

        if not graph:
            return expected_nodes == 0

        # BFS desde el primer nodo
        start_node = next(iter(graph))
        visited: Set[int] = set()
        queue: List[int] = [start_node]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            # Añadir vecinos no visitados
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

        # Conexo si visitamos todos los nodos esperados
        return len(visited) == expected_nodes

    def _fields_are_semantically_related(self, field1: str, field2: str) -> bool:
        """
        Determina relación semántica entre campos.
        REFINADO: Criterios más robustos de similitud.
        """
        if not field1 or not field2:
            return False

        f1_clean = field1.strip().upper()
        f2_clean = field2.strip().upper()

        # Evitar falsos positivos con campos muy cortos
        if len(f1_clean) < 3 or len(f2_clean) < 3:
            return False

        # Contención directa (un campo es subcadena del otro)
        if len(f1_clean) >= 4 and f1_clean in f2_clean:
            return True
        if len(f2_clean) >= 4 and f2_clean in f1_clean:
            return True

        # Palabras compartidas significativas
        words1 = set(w for w in f1_clean.split() if len(w) >= 3)
        words2 = set(w for w in f2_clean.split() if len(w) >= 3)

        if words1 and words2:
            intersection = words1 & words2
            if len(intersection) >= 1:
                return True

        return False

    def _has_negative_cycles(self, graph: Dict[int, Set[int]]) -> bool:
        """Verifica existencia de ciclos (helper implementado por completitud)."""
        # En un grafo no dirigido (simétrico), ciclos son triviales.
        # Para cumplir el requerimiento, implementamos detección básica de ciclos DFS.
        visited = set()
        rec_stack = set()

        def dfs(u, p):
            visited.add(u)
            rec_stack.add(u)
            for v in graph.get(u, set()):
                if v == p: continue
                if v in rec_stack: return True
                if v not in visited:
                    if dfs(v, u): return True
            rec_stack.remove(u)
            return False

        for node in graph:
            if node not in visited:
                if dfs(node, -1): return True
        return False

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
