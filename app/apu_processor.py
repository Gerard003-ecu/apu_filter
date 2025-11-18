"""
Procesador APU con arquitectura modular de especialistas.

Este módulo implementa un sistema avanzado para el procesamiento de datos de
Análisis de Precios Unitarios (APU). Utiliza una arquitectura modular donde
componentes "especialistas", cada uno con una responsabilidad única, colaboran
para interpretar y estructurar líneas de texto con formatos variables.

El `APUProcessor` principal mantiene la compatibilidad con la interfaz esperada
por el `LoadDataStep` del pipeline, orquestando internamente a estos especialistas
para lograr un procesamiento robusto y flexible.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

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
# GRAMÁTICA LARK
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

APU_GRAMMAR = r"""
    ?start: line

    // CAMBIO CLAVE: Se eliminó la opcionalidad externa.
    // Una línea ahora DEBE tener al menos un 'field'.
    line: field (SEP field)*

    field: FIELD_VALUE?  // El campo en sí puede estar vacío (ej. 'dato1;;dato3')

    FIELD_VALUE: /[^;\r\n]+/ // El contenido del campo (si existe)
    SEP: /\s*;\s*/          // Separador flexible

    NEWLINE: /[\r\n]+/

    %import common.WS
    %ignore WS
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
        "DESCRIPCION",
        "DESCRIPCIÓN",
        "DESC",
        "UND",
        "UNID",
        "UNIDAD",
        "CANT",
        "CANTIDAD",
        "PRECIO",
        "VALOR",
        "TOTAL",
        "DESP",
        "DESPERDICIO",
        "REND",
        "RENDIMIENTO",
        "JORNAL",
        "ITEM",
        "CODIGO",
        "CÓDIGO",
    ]

    # Palabras clave de resumen/totalización
    SUMMARY_KEYWORDS = [
        "SUBTOTAL",
        "TOTAL",
        "RESUMEN",
        "SUMA",
        "TOTALES",
        "ACUMULADO",
        "GRAN TOTAL",
        "COSTO DIRECTO",
    ]

    # Categorías típicas (exactas)
    CATEGORY_PATTERNS = [
        r"^MATERIALES?$",
        r"^MANO\s+DE\s+OBRA$",
        r"^EQUIPO$",
        r"^TRANSPORTE$",
        r"^OTROS?$",
        r"^SERVICIOS?$",
        r"^HERRAMIENTAS?$",
        r"^SUMINISTROS?$",
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
        """
        Cuenta cuántas palabras clave de encabezado están presentes en el texto.

        Args:
            text: El texto a analizar.

        Returns:
            El número de palabras clave de encabezado encontradas.
        """
        text_upper = text.upper()
        return sum(1 for keyword in self.HEADER_KEYWORDS if keyword in text_upper)

    def is_likely_header(self, text: str, field_count: int) -> bool:
        """Determina si una línea es probablemente un encabezado de tabla."""
        keyword_count = self.count_header_keywords(text)

        if field_count <= 2 and keyword_count >= 3:
            return True

        words = text.upper().split()
        if words and len(words) > 2:
            header_word_ratio = sum(1 for w in words if w in self.HEADER_KEYWORDS) / len(
                words
            )
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
        "UND",
        "UN",
        "UNID",
        "UNIDAD",
        "UNIDADES",
        "M",
        "MT",
        "MTS",
        "MTR",
        "MTRS",
        "METRO",
        "METROS",
        "ML",
        "KM",
        "M2",
        "MT2",
        "MTS2",
        "MTRS2",
        "METROSCUAD",
        "METROSCUADRADOS",
        "M3",
        "MT3",
        "MTS3",
        "MTRS3",
        "METROSCUB",
        "METROSCUBICOS",
        "HR",
        "HRS",
        "HORA",
        "HORAS",
        "MIN",
        "MINUTO",
        "MINUTOS",
        "DIA",
        "DIAS",
        "SEM",
        "SEMANA",
        "SEMANAS",
        "MES",
        "MESES",
        "JOR",
        "JORN",
        "JORNAL",
        "JORNALES",
        "G",
        "GR",
        "GRAMO",
        "GRAMOS",
        "KG",
        "KGS",
        "KILO",
        "KILOS",
        "KILOGRAMO",
        "KILOGRAMOS",
        "TON",
        "TONS",
        "TONELADA",
        "TONELADAS",
        "LB",
        "LIBRA",
        "LIBRAS",
        "GAL",
        "GLN",
        "GALON",
        "GALONES",
        "LT",
        "LTS",
        "LITRO",
        "LITROS",
        "ML",
        "MILILITRO",
        "MILILITROS",
        "VIAJE",
        "VIAJES",
        "VJE",
        "VJ",
        "BULTO",
        "BULTOS",
        "SACO",
        "SACOS",
        "PAQ",
        "PAQUETE",
        "PAQUETES",
        "GLOBAL",
        "GLB",
        "GB",
    }

    @classmethod
    @lru_cache(maxsize=256)
    def normalize_unit(cls, unit: str) -> str:
        """
        Normaliza una unidad a su forma canónica (ej. "Metro" -> "M").

        Args:
            unit: La cadena de texto de la unidad a normalizar.

        Returns:
            La unidad normalizada. Devuelve "UND" si la unidad es vacía o
            no reconocida.
        """
        if not unit:
            return "UND"

        unit_clean = re.sub(r"[^A-Z0-9]", "", unit.upper().strip())

        unit_mappings = {
            "UNID": "UND",
            "UN": "UND",
            "UNIDAD": "UND",
            "MT": "M",
            "MTS": "M",
            "MTR": "M",
            "MTRS": "M",
            "JORN": "JOR",
            "JORNAL": "JOR",
            "JORNALES": "JOR",
            # Agregar más mapeos según sea necesario
        }

        return unit_mappings.get(
            unit_clean, unit_clean if unit_clean in cls.VALID_UNITS else "UND"
        )

    @classmethod
    def is_valid(cls, unit: str) -> bool:
        """
        Verifica si una cadena de texto representa una unidad válida.

        Args:
            unit: La unidad a validar.

        Returns:
            True si la unidad es reconocida, False en caso contrario.
        """
        if not unit:
            return False
        unit_clean = re.sub(r"[^A-Z0-9]", "", unit.upper().strip())
        return unit_clean in cls.VALID_UNITS or len(unit_clean) <= 4


class NumericFieldExtractor:
    """
    Especialista en la extracción e identificación de campos numéricos.

    Esta clase es responsable de parsear valores numéricos de cadenas de texto,
    manejando diferentes separadores decimales. Su función más importante es
    la identificación inteligente de valores de "rendimiento" y "jornal"
    para insumos de Mano de Obra, utilizando heurísticas basadas en umbrales
    y magnitud relativa.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
        thresholds: Optional[ValidationThresholds] = None,
    ):
        """
        Inicializa el extractor.

        Args:
            config: El diccionario de configuración global.
            profile: El perfil de configuración específico del archivo.
            thresholds: Un objeto `ValidationThresholds` con los umbrales.
        """
        self.config = config or {}
        self.profile = profile or {}
        self.pattern_matcher = PatternMatcher()
        self.thresholds = thresholds or ValidationThresholds()
        # CAMBIO: Leer el separador decimal desde el profile
        number_format = self.profile.get("number_format", {})
        self.decimal_separator = number_format.get("decimal_separator")

    def extract_all_numeric_values(
        self, fields: List[str], skip_first: bool = True
    ) -> List[float]:
        """
        Extrae todos los valores numéricos válidos de una lista de campos.

        Args:
            fields: La lista de cadenas de texto (campos) a procesar.
            skip_first: Si es True, ignora el primer campo (usualmente la
                        descripción).

        Returns:
            Una lista de los valores numéricos encontrados.
        """
        start_idx = 1 if skip_first else 0
        numeric_values = []

        for field in fields[start_idx:]:
            if not field:
                continue

            value = self.parse_number_safe(field)
            if value is not None and value >= 0:
                numeric_values.append(value)

        return numeric_values

    def parse_number_safe(self, value: str) -> Optional[float]:
        """
        Parsea un número de forma segura, utilizando el separador decimal
        configurado.

        Args:
            value: La cadena de texto que contiene el número.

        Returns:
            El número como flotante, o None si el parseo falla.
        """
        if not value or not isinstance(value, str):
            return None
        try:
            # CAMBIO: Pasar el separador decimal a la función de parseo
            return parse_number(value, decimal_separator=self.decimal_separator)
        except (ValueError, TypeError, AttributeError):
            return None

    def identify_mo_values(
        self, numeric_values: List[float]
    ) -> Optional[Tuple[float, float]]:
        """
        Identifica rendimiento y jornal de una lista de valores numéricos.

        Utiliza heurísticas basadas en rangos típicos y magnitud para
        distinguir entre el valor del jornal (generalmente un número grande)
        y el rendimiento (un número más pequeño).

        Args:
            numeric_values: Lista de valores numéricos extraídos de una línea
                            de Mano de Obra.

        Returns:
            Una tupla (rendimiento, jornal) si se identifican ambos, o None.
        """
        if len(numeric_values) < 2:
            return None

        # Heurística 1: Buscar por rangos típicos
        jornal_candidates = [
            v
            for v in numeric_values
            if self.thresholds.min_jornal <= v <= self.thresholds.max_jornal
        ]

        rendimiento_candidates = [
            v
            for v in numeric_values
            if (
                self.thresholds.min_rendimiento
                <= v
                <= self.thresholds.max_rendimiento_tipico
                and v not in jornal_candidates
            )
        ]

        if jornal_candidates and rendimiento_candidates:
            # Tomar el jornal más grande y el rendimiento más pequeño
            jornal = max(jornal_candidates)
            rendimiento = min(rendimiento_candidates)
            return rendimiento, jornal

        # Heurística 2: Si no encontramos con rangos, usar posición relativa
        if len(numeric_values) >= 2:
            sorted_values = sorted(numeric_values, reverse=True)

            # El valor más grande que sea >= min_jornal es probablemente el jornal
            for val in sorted_values:
                if val >= self.thresholds.min_jornal:
                    jornal = val
                    # Buscar rendimiento entre los valores restantes
                    for other_val in numeric_values:
                        if (
                            other_val != jornal
                            and other_val <= self.thresholds.max_rendimiento_tipico
                        ):
                            return other_val, jornal
                    break

        return None

    def extract_insumo_values(self, fields: List[str], start_from: int = 2) -> List[float]:
        """
        Extrae valores numéricos para insumos básicos (no Mano de Obra).

        Args:
            fields: Lista de campos de la línea.
            start_from: Índice desde el cual empezar a buscar valores.

        Returns:
            Lista de valores numéricos (cantidad, precio, total).
        """
        valores = []
        for i in range(start_from, len(fields)):
            if fields[i] and "%" not in fields[i]:  # Ignorar desperdicio
                val = self.parse_number_safe(fields[i])
                if val is not None and val >= 0:
                    valores.append(val)
        return valores


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# TRANSFORMER ORQUESTADOR
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


@v_args(inline=False)
class APUTransformer(Transformer):
    """
    Orquestador que coordina a los especialistas para transformar una línea.

    Esta clase actúa como un `Transformer` para la librería Lark. Recibe el
    árbol de parseo de una línea, extrae los campos y utiliza a los
    especialistas (`PatternMatcher`, `NumericFieldExtractor`, etc.) para
    detectar el formato de la línea y despacharla al método constructor
    apropiado (`_build_mo_completa`, `_build_insumo_basico`).
    """

    def __init__(
        self,
        apu_context: Dict[str, Any],
        config: Dict[str, Any],
        profile: Dict[str, Any],
        keyword_cache: Any,
    ):
        """
        Inicializa el Transformer.

        Args:
            apu_context: Diccionario con el contexto del APU actual (código,
                         descripción, etc.).
            config: Diccionario de configuración de la aplicación.
            profile: Perfil de configuración específico para el archivo.
            keyword_cache: Cache de palabras clave (actualmente no usado).
        """
        self.apu_context = apu_context or {}
        self.config = config or {}
        self.profile = profile or {}
        self.keyword_cache = keyword_cache

        # Inicializar especialistas
        self.pattern_matcher = PatternMatcher()
        self.units_validator = UnitsValidator()
        self.thresholds = self._load_validation_thresholds()
        # CAMBIO: Pasar el profile al NumericFieldExtractor
        self.numeric_extractor = NumericFieldExtractor(
            self.config, self.profile, self.thresholds
        )
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

    def _extract_value(self, item) -> str:
        """Extrae el valor de string de un token o string de forma segura."""
        if item is None:
            return ""
        if isinstance(item, Token):
            return str(item.value).strip() if item.value else ""
        if isinstance(item, (str, bytes)):
            value = item.decode("utf-8") if isinstance(item, bytes) else item
            return value.strip()
        try:
            return str(item).strip()
        except Exception:
            return ""

    def line(self, args):
        """
        Procesa una línea parseada por Lark.

        Args:
            args: Argumentos proporcionados por Lark (campos de la línea).

        Returns:
            Un objeto `InsumoProcesado` si la línea es válida y procesable,
            o None en caso contrario.
        """
        fields = []
        # CORRECCIÓN: Filtrar los tokens SEP ('\;') que Lark incluye en `args`.
        # Lark pasa una lista plana [field, SEP, field, SEP, ...].
        # Un SEP es un Token, un field no.
        filtered_args = [arg for arg in args if not isinstance(arg, Token) or arg.type != 'SEP']

        for arg in filtered_args:
            if isinstance(arg, list):
                fields.extend([self._extract_value(f) for f in arg])
            else:
                fields.append(self._extract_value(arg))

        clean_fields = self._filter_trailing_empty(fields)

        if not clean_fields or not clean_fields[0]:
            return None

        formato = self._detect_format(clean_fields)

        if formato == FormatoLinea.DESCONOCIDO:
            return None

        return self._dispatch_builder(formato, clean_fields)

    def field(self, args):
        """Procesa un campo individual parseado por Lark."""
        if not args:
            return ""
        return self._extract_value(args[0]) if args else ""

    def _filter_trailing_empty(self, tokens: List[str]) -> List[str]:
        """Elimina campos vacíos al final de una lista de campos."""
        if not tokens:
            return []

        last_idx = -1
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i]:
                last_idx = i
                break

        return tokens[: last_idx + 1] if last_idx >= 0 else []

    def _detect_format(self, fields: List[str]) -> FormatoLinea:
        """
        Detecta el formato de la línea usando los especialistas.

        Coordina al `PatternMatcher` para filtrar ruido (resúmenes,
        encabezados) y al `NumericFieldExtractor` para determinar si
        la línea tiene la estructura de un insumo de Mano de Obra o
        de un insumo básico.

        Args:
            fields: La lista de campos de la línea.

        Returns:
            El `FormatoLinea` detectado.
        """
        if not fields or not fields[0]:
            return FormatoLinea.DESCONOCIDO

        descripcion = fields[0].strip()
        num_fields = len(fields)

        # Usar PatternMatcher para filtrar ruido contextualmente
        if self._is_noise_line(descripcion, num_fields):
            return FormatoLinea.DESCONOCIDO

        if num_fields < 3:
            return FormatoLinea.DESCONOCIDO

        # Clasificar tipo de insumo
        tipo_probable = self._classify_insumo(descripcion)

        # Detectar MO_COMPLETA si es mano de obra y tiene formato válido
        if tipo_probable == TipoInsumo.MANO_DE_OBRA and num_fields >= 5:
            if self._validate_mo_format(fields):
                logger.debug(f"MO_COMPLETA detectado: {descripcion[:30]}...")
                return FormatoLinea.MO_COMPLETA

        # Detectar INSUMO_BASICO si tiene suficientes campos numéricos
        if num_fields >= 4:
            numeric_values = self.numeric_extractor.extract_all_numeric_values(fields)
            if len(numeric_values) >= 2:
                logger.debug(f"INSUMO_BASICO detectado: {descripcion[:30]}...")
                return FormatoLinea.INSUMO_BASICO

        return FormatoLinea.DESCONOCIDO

    def _is_noise_line(self, descripcion: str, num_fields: int) -> bool:
        """Detecta si una línea es ruido (encabezado, resumen, etc.)."""
        if self.pattern_matcher.is_likely_summary(descripcion, num_fields):
            logger.debug(f"Línea de resumen ignorada: {descripcion[:30]}...")
            return True

        if self.pattern_matcher.is_likely_header(descripcion, num_fields):
            logger.debug(f"Línea de encabezado ignorada: {descripcion[:30]}...")
            return True

        if self.pattern_matcher.is_likely_category(descripcion, num_fields):
            logger.debug(f"Línea de categoría ignorada: {descripcion[:30]}...")
            return True

        return False

    def _validate_mo_format(self, fields: List[str]) -> bool:
        """Valida el formato de Mano de Obra usando el NumericFieldExtractor."""
        if len(fields) < 5:
            return False

        numeric_values = self.numeric_extractor.extract_all_numeric_values(fields)
        mo_values = self.numeric_extractor.identify_mo_values(numeric_values)

        return mo_values is not None

    def _dispatch_builder(
        self, formato: FormatoLinea, tokens: List[str]
    ) -> Optional[InsumoProcesado]:
        """
        Llama al método constructor adecuado según el formato detectado.

        Args:
            formato: El `FormatoLinea` detectado.
            tokens: La lista de campos de la línea.

        Returns:
            Un objeto `InsumoProcesado` o None si la construcción falla.
        """
        try:
            if formato == FormatoLinea.MO_COMPLETA:
                return self._build_mo_completa(tokens)
            elif formato == FormatoLinea.INSUMO_BASICO:
                return self._build_insumo_basico(tokens)
            return None
        except Exception as e:
            logger.error(f"Error construyendo {formato.value}: {e}")
            return None

    def _build_mo_completa(self, tokens: List[str]) -> Optional[ManoDeObra]:
        """
        Construye un objeto `ManoDeObra` a partir de una línea de formato completo.

        Utiliza el `NumericFieldExtractor` para encontrar el rendimiento y el
        jornal, y luego calcula los demás valores.

        Args:
            tokens: Lista de campos de la línea.

        Returns:
            Un objeto `ManoDeObra` o None.
        """
        try:
            descripcion = tokens[0]
            unidad = (
                self.units_validator.normalize_unit(tokens[1]) if len(tokens) > 1 else "JOR"
            )

            # Usar NumericFieldExtractor para identificar valores
            numeric_values = self.numeric_extractor.extract_all_numeric_values(tokens)
            mo_values = self.numeric_extractor.identify_mo_values(numeric_values)

            if not mo_values:
                logger.debug("No se pudieron identificar jornal y rendimiento")
                return None

            rendimiento, jornal = mo_values

            # Cálculos
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
        """
        Construye un objeto de insumo a partir de una línea de formato básico.

        Clasifica el tipo de insumo basándose en la descripción y luego extrae
        los valores numéricos (cantidad, precio, total).

        Args:
            tokens: Lista de campos de la línea.

        Returns:
            Un objeto `InsumoProcesado` (o una de sus subclases) o None.
        """
        try:
            if len(tokens) < 4:
                return None

            descripcion = tokens[0]
            unidad = (
                self.units_validator.normalize_unit(tokens[1]) if len(tokens) > 1 else "UND"
            )

            # Usar NumericFieldExtractor para valores
            valores = self.numeric_extractor.extract_insumo_values(tokens)

            if len(valores) < 2:
                return None

            # Interpretar valores
            cantidad = valores[0] if len(valores) > 0 else 1.0
            precio = valores[1] if len(valores) > 1 else 0.0
            total = valores[2] if len(valores) > 2 else cantidad * precio

            # Corregir si es necesario
            if total == 0 and cantidad > 0 and precio > 0:
                total = cantidad * precio
            elif precio == 0 and cantidad > 0 and total > 0:
                precio = total / cantidad

            if total <= 0:
                return None

            tipo_insumo = self._classify_insumo(descripcion)
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

    @lru_cache(maxsize=2048)
    def _classify_insumo(self, descripcion: str) -> TipoInsumo:
        """
        Clasifica el tipo de insumo basándose en palabras clave en la descripción.

        Args:
            descripcion: La descripción del insumo.

        Returns:
            El `TipoInsumo` más probable.
        """
        if not descripcion:
            return TipoInsumo.OTRO

        desc_upper = descripcion.upper()

        # CAMBIO: Leer reglas desde la config
        rules = self.config.get("apu_processor_rules", {})
        special_cases = rules.get("special_cases", {})
        mo_keywords = rules.get("mo_keywords", [])
        equipo_keywords = rules.get("equipo_keywords", [])

        for case, tipo_str in special_cases.items():
            if case in desc_upper:
                return TipoInsumo(tipo_str)

        if any(kw in desc_upper for kw in mo_keywords):
            return TipoInsumo.MANO_DE_OBRA
        if any(kw in desc_upper for kw in equipo_keywords):
            return TipoInsumo.EQUIPO

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
    """
    Procesador de APUs con soporte para múltiples formatos de entrada.

    Soporta dos formatos:
    1. Formato agrupado (legacy): [{"codigo_apu": "X", "lines": [...]}]
    2. Formato plano (nuevo): [{"apu_code": "X", "insumo_line": "...", "_lark_tree": ...}]
    """

    def __init__(
        self,
        config,
        profile: Optional[Dict[str, Any]] = None,
        parse_cache: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa el procesador con cache opcional de parsing.

        Args:
            config: Configuración del sistema.
            profile: Perfil de parsing.
            parse_cache: Cache de árboles Lark pre-parseados.
        """
        self.config = config
        self.profile = profile or {}
        self.parser = self._initialize_parser()
        self.keyword_cache = {}

        # Cache de parsing (optimización)
        self.parse_cache = parse_cache or {}

        # Estadísticas globales
        self.global_stats = {
            "total_apus": 0,
            "total_insumos": 0,
            "format_detected": None,
        }

        self.parsing_stats = ParsingStats()
        self.debug_mode = self.config.get("debug_mode", False)

        # Registros crudos (se establecerán externamente)
        self.raw_records = []

        if self.parse_cache:
            logger.info(
                f"✓ APUProcessor inicializado con cache de {len(self.parse_cache)} "
                f"líneas pre-parseadas"
            )

    def _detect_record_format(
        self,
        records: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """
        Detecta automáticamente el formato de los registros de entrada.

        Args:
            records: Lista de registros a analizar.

        Returns:
            Tupla (formato, descripción) donde formato es "grouped" o "flat".
        """
        if not records:
            return ("unknown", "No hay registros para analizar")

        first_record = records[0]

        # Formato agrupado (legacy): tiene clave "lines"
        if "lines" in first_record:
            return (
                "grouped",
                "Formato agrupado (legacy): cada registro es un APU con lista de líneas"
            )

        # Formato plano (nuevo): tiene claves "insumo_line" y "apu_code"
        if "insumo_line" in first_record and "apu_code" in first_record:
            return (
                "flat",
                "Formato plano (nuevo): cada registro es un insumo individual"
            )

        # Formato desconocido
        logger.warning(
            f"Formato de registro desconocido. Claves encontradas: {first_record.keys()}"
        )
        return ("unknown", "Formato no reconocido")

    def _group_flat_records(
        self,
        flat_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Agrupa registros planos por APU.

        Convierte el formato plano (nuevo) al formato agrupado que el resto
        del procesador puede manejar, pero preservando optimizaciones como
        el árbol Lark pre-parseado.

        Args:
            flat_records: Lista de registros en formato plano.

        Returns:
            Lista de registros en formato agrupado.
        """
        logger.info(f"Agrupando {len(flat_records)} registros planos por APU...")

        # Agrupar por apu_code
        grouped = defaultdict(lambda: {
            "lines": [],
            "_lark_trees": [],  # Preservar árboles pre-parseados
            "metadata": {}
        })

        for record in flat_records:
            apu_code = record.get("apu_code", "UNKNOWN")

            # Agregar línea de insumo
            insumo_line = record.get("insumo_line", "")
            if insumo_line:
                grouped[apu_code]["lines"].append(insumo_line)

                # Preservar árbol Lark si existe
                lark_tree = record.get("_lark_tree")
                grouped[apu_code]["_lark_trees"].append(lark_tree)

            # Preservar metadata del APU (solo la primera vez)
            if not grouped[apu_code]["metadata"]:
                grouped[apu_code]["metadata"] = {
                    "apu_code": apu_code,
                    "apu_desc": record.get("apu_desc", ""),
                    "apu_unit": record.get("apu_unit", ""),
                    "category": record.get("category", "INDEFINIDO"),
                    "source_line": record.get("source_line", 0),
                }

        # Convertir a lista de registros agrupados
        result = []
        for apu_code, data in grouped.items():
            record = {
                "codigo_apu": apu_code,  # Usar nombre legacy para compatibilidad
                "descripcion_apu": data["metadata"].get("apu_desc", ""),
                "unidad_apu": data["metadata"].get("apu_unit", ""),
                "lines": data["lines"],
                "_lark_trees": data["_lark_trees"],  # Nueva clave para optimización
                "category": data["metadata"].get("category", "INDEFINIDO"),
                "source_line": data["metadata"].get("source_line", 0),
            }
            result.append(record)

        logger.info(f"✓ Agrupados en {len(result)} APUs distintos")

        return result

    def process_all(self) -> pd.DataFrame:
        """
        Procesa todos los registros de APU crudos y devuelve un DataFrame.

        Este método ahora es ADAPTATIVO:
        - Detecta automáticamente el formato de entrada
        - Convierte formato plano a agrupado si es necesario
        - Reutiliza árboles Lark pre-parseados cuando están disponibles
        - Mantiene compatibilidad con formato legacy

        Returns:
            DataFrame con todos los insumos procesados y estructurados.
        """
        if not self.raw_records:
            logger.warning("No hay registros crudos para procesar")
            return pd.DataFrame()

        logger.info(f"Iniciando procesamiento de {len(self.raw_records)} registros")

        # 🔥 PASO 1: Detectar formato de entrada
        format_type, format_desc = self._detect_record_format(self.raw_records)
        self.global_stats["format_detected"] = format_type

        logger.info(f"📋 Formato detectado: {format_desc}")

        # 🔥 PASO 2: Normalizar a formato agrupado si es necesario
        if format_type == "flat":
            processed_records = self._group_flat_records(self.raw_records)
        elif format_type == "grouped":
            processed_records = self.raw_records
            logger.info("✓ Formato ya está agrupado, no se requiere conversión")
        else:
            logger.error(
                "❌ Formato de entrada no reconocido. "
                "No se puede procesar sin formato conocido."
            )
            return pd.DataFrame()

        # 🔥 PASO 3: Procesar cada APU
        all_results = []
        self.global_stats["total_apus"] = len(processed_records)

        for i, record in enumerate(processed_records):
            try:
                apu_context = self._extract_apu_context(record)

                if "lines" in record and record["lines"]:
                    # Preparar cache específico para este APU
                    apu_cache = self._prepare_apu_cache(record)

                    insumos = self._process_apu_lines(
                        record["lines"],
                        apu_context,
                        apu_cache
                    )

                    if insumos:
                        all_results.extend(insumos)
                else:
                    logger.debug(
                        f"APU {apu_context.get('codigo_apu')} no tiene líneas para procesar"
                    )

                # Log de progreso
                if (i + 1) % 50 == 0:
                    logger.info(
                        f"Progreso: {i + 1}/{len(processed_records)} APUs procesados "
                        f"({len(all_results)} insumos extraídos hasta ahora)"
                    )

            except Exception as e:
                logger.error(
                    f"Error procesando APU {i} "
                    f"[{record.get('codigo_apu', 'UNKNOWN')}]: {e}"
                )
                if self.debug_mode:
                    import traceback
                    logger.debug(f"Traceback:\n{traceback.format_exc()}")
                continue

        # 🔥 PASO 4: Log de resultados finales
        self.global_stats["total_insumos"] = len(all_results)
        self._log_global_stats()

        # 🔥 PASO 5: Convertir a DataFrame
        if all_results:
            return self._convert_to_dataframe(all_results)
        else:
            logger.warning("⚠️  No se encontraron insumos válidos en ningún APU")
            return pd.DataFrame()

    def _prepare_apu_cache(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepara el cache de parsing específico para un APU.

        Si el registro tiene árboles Lark pre-parseados (_lark_trees),
        crea un mapeo línea -> árbol para ese APU específico.

        Args:
            record: Registro del APU con posibles árboles pre-parseados.

        Returns:
            Diccionario de cache línea -> árbol para este APU.
        """
        apu_cache = {}

        # Si el registro tiene árboles pre-parseados, mapearlos
        if "_lark_trees" in record and record["_lark_trees"]:
            lines = record.get("lines", [])
            trees = record["_lark_trees"]

            # Crear mapeo línea -> árbol
            for line, tree in zip(lines, trees):
                if tree is not None:
                    apu_cache[line.strip()] = tree

            if apu_cache:
                logger.debug(
                    f"✓ Cache específico de APU preparado: {len(apu_cache)} árboles"
                )

        # Combinar con cache global
        combined_cache = {**self.parse_cache, **apu_cache}

        return combined_cache

    def _extract_apu_context(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae el contexto relevante de un registro de APU.

        Soporta tanto nombres de claves legacy como nuevos.

        Args:
            record: Registro de APU (formato agrupado).

        Returns:
            Diccionario con contexto del APU normalizado.
        """
        # Intentar claves nuevas primero, luego legacy
        return {
            "codigo_apu": record.get("codigo_apu") or record.get("apu_code", ""),
            "descripcion_apu": (
                record.get("descripcion_apu") or
                record.get("apu_desc", "")
            ),
            "unidad_apu": record.get("unidad_apu") or record.get("apu_unit", ""),
            "cantidad_apu": record.get("cantidad_apu", 1.0),
            "precio_unitario_apu": record.get("precio_unitario_apu", 0.0),
            "categoria": record.get("category", "INDEFINIDO"),
        }

    def _process_apu_lines(
        self,
        lines: List[str],
        apu_context: Dict[str, Any],
        line_cache: Optional[Dict[str, Any]] = None
    ) -> List['InsumoProcesado']:
        """
        Procesa líneas de APU con reutilización de cache de parsing.

        Args:
            lines: Lista de líneas a procesar.
            apu_context: Contexto del APU.
            line_cache: Cache de árboles Lark para estas líneas específicas.

        Returns:
            Lista de insumos procesados.
        """
        if not lines:
            return []

        results = []
        stats = ParsingStats()

        # Usar cache combinado (específico del APU + global)
        active_cache = line_cache if line_cache is not None else self.parse_cache

        apu_code = apu_context.get("codigo_apu", "UNKNOWN")

        logger.debug(
            f"Procesando {len(lines)} líneas para APU: {apu_code} "
            f"(cache: {len(active_cache)} entradas)"
        )

        for line_num, line in enumerate(lines, start=1):
            if not line or not line.strip():
                continue

            stats.total_lines += 1
            line_clean = line.strip()
            insumo = None

            try:
                if self.parser:
                    # 🔥 OPTIMIZACIÓN: Usar cache si está disponible
                    tree = None
                    used_cache = False

                    if line_clean in active_cache:
                        tree = active_cache[line_clean]
                        used_cache = True
                        stats.cache_hits += 1
                        logger.debug(
                            f"  ⚡ Línea {line_num}: Usando árbol Lark del cache"
                        )

                    if tree is None:
                        # Parsear normalmente
                        try:
                            tree = self.parser.parse(line_clean)
                        except LarkError as lark_error:
                            # Si falla aquí con validación unificada, es inesperado
                            logger.warning(
                                f"  ⚠️  Línea {line_num}: Falló Lark pero pasó validación previa\n"
                                f"      Error: {lark_error}\n"
                                f"      Línea: {line_clean[:100]}"
                            )
                            stats.lark_parse_errors += 1
                            continue

                    # Transformar árbol a insumo
                    try:
                        transformer = APUTransformer(
                            apu_context,
                            self.config,
                            self.profile,
                            self.keyword_cache
                        )
                        insumo = transformer.transform(tree)

                        if isinstance(insumo, list):
                            if insumo:
                                insumo = insumo[0]
                                stats.successful_parses += 1
                            else:
                                stats.empty_results += 1
                                logger.debug(
                                    f"  ⚠️  Línea {line_num}: Transformer devolvió lista vacía"
                                )
                                insumo = None
                        else:
                            stats.successful_parses += 1

                    except Exception as transform_error:
                        stats.transformer_errors += 1
                        logger.error(
                            f"  ✗ Línea {line_num}: Error en transformer\n"
                            f"    Error: {type(transform_error).__name__}: {transform_error}\n"
                            f"    Línea: {line_clean[:100]}"
                        )

                        if self.debug_mode:
                            import traceback
                            logger.debug(f"Traceback:\n{traceback.format_exc()}")

                        continue

                # Agregar resultado si es válido
                if insumo:
                    insumo.line_number = line_num
                    results.append(insumo)
                else:
                    stats.failed_lines.append({
                        "line_number": line_num,
                        "content": line_clean,
                        "apu_code": apu_code
                    })

            except Exception as unexpected_error:
                logger.error(
                    f"  🚨 Línea {line_num}: Error inesperado\n"
                    f"    Tipo: {type(unexpected_error).__name__}\n"
                    f"    Error: {unexpected_error}\n"
                    f"    Línea: {line_clean}"
                )

                if self.debug_mode:
                    import traceback
                    logger.debug(f"Traceback completo:\n{traceback.format_exc()}")

                stats.failed_lines.append({
                    "line_number": line_num,
                    "content": line_clean,
                    "error": str(unexpected_error),
                    "apu_code": apu_code
                })
                continue

        # Log de estadísticas del APU
        self._log_parsing_stats(apu_code, stats)

        # Actualizar estadísticas globales
        self._merge_stats(stats)

        return results

    def _merge_stats(self, apu_stats: ParsingStats):
        """Combina estadísticas de un APU con las globales."""
        self.parsing_stats.total_lines += apu_stats.total_lines
        self.parsing_stats.successful_parses += apu_stats.successful_parses
        self.parsing_stats.lark_parse_errors += apu_stats.lark_parse_errors
        self.parsing_stats.transformer_errors += apu_stats.transformer_errors
        self.parsing_stats.empty_results += apu_stats.empty_results
        self.parsing_stats.cache_hits += apu_stats.cache_hits
        self.parsing_stats.failed_lines.extend(apu_stats.failed_lines)

    def _log_parsing_stats(self, apu_code: str, stats: ParsingStats):
        """
        Registra estadísticas detalladas del parsing de un APU.

        Args:
            apu_code: Código del APU procesado.
            stats: Estadísticas del procesamiento.
        """
        if stats.total_lines == 0:
            return

        success_rate = (
            (stats.successful_parses / stats.total_lines * 100)
            if stats.total_lines > 0 else 0
        )
        cache_rate = (
            (stats.cache_hits / stats.total_lines * 100)
            if stats.total_lines > 0 else 0
        )

        # Solo mostrar detalles si hay problemas o en modo debug
        if success_rate < 100 or self.debug_mode:
            logger.info("-" * 70)
            logger.info(f"📈 APU: {apu_code}")
            logger.info(f"   Líneas procesadas:  {stats.total_lines}")
            logger.info(f"   ✓ Exitosos:         {stats.successful_parses} ({success_rate:.1f}%)")
            logger.info(f"   ⚡ Cache hits:       {stats.cache_hits} ({cache_rate:.1f}%)")

            if stats.lark_parse_errors > 0:
                logger.info(f"   ✗ Errores Lark:     {stats.lark_parse_errors}")
            if stats.transformer_errors > 0:
                logger.info(f"   ✗ Errores Trans.:   {stats.transformer_errors}")
            if stats.empty_results > 0:
                logger.info(f"   ⚠️  Resultados vacíos: {stats.empty_results}")

            logger.info("-" * 70)

    def _log_global_stats(self):
        """Registra estadísticas globales del procesamiento."""
        logger.info("=" * 80)
        logger.info("📊 RESUMEN GLOBAL DE PROCESAMIENTO")
        logger.info("=" * 80)
        logger.info(f"Formato detectado:           {self.global_stats['format_detected']}")
        logger.info(f"Total APUs procesados:       {self.global_stats['total_apus']}")
        logger.info(f"Total insumos extraídos:     {self.global_stats['total_insumos']}")
        logger.info(f"Total líneas procesadas:     {self.parsing_stats.total_lines}")
        logger.info("")
        logger.info("Resultados de parsing:")
        logger.info(f"  ✓ Exitosos:                {self.parsing_stats.successful_parses}")
        logger.info(f"  ⚡ Cache hits:              {self.parsing_stats.cache_hits}")
        logger.info(f"  ✗ Errores Lark:            {self.parsing_stats.lark_parse_errors}")
        logger.info(f"  ✗ Errores Transformer:     {self.parsing_stats.transformer_errors}")
        logger.info(f"  ⚠️  Resultados vacíos:      {self.parsing_stats.empty_results}")
        logger.info("")

        if self.parsing_stats.total_lines > 0:
            success_rate = (
                self.parsing_stats.successful_parses /
                self.parsing_stats.total_lines * 100
            )
            cache_efficiency = (
                self.parsing_stats.cache_hits /
                self.parsing_stats.total_lines * 100
            )

            logger.info(f"Tasa de éxito:               {success_rate:.2f}%")
            logger.info(f"Eficiencia de cache:         {cache_efficiency:.2f}%")

        logger.info("=" * 80)

        # Alertas
        if self.global_stats['total_insumos'] == 0:
            logger.error(
                "🚨 CRÍTICO: 0 insumos extraídos.\n"
                "   Posibles causas:\n"
                "   1. Formato de datos incompatible con gramática\n"
                "   2. Errores en el transformer\n"
                "   3. Configuración de perfil incorrecta\n"
                "   → Revise los logs detallados arriba"
            )
        elif success_rate < 50:
            logger.warning(
                f"⚠️  Tasa de éxito baja ({success_rate:.1f}%).\n"
                f"   Considere revisar la gramática o el formato de datos."
            )

    def _initialize_parser(self):
        """Inicializa el parser Lark (implementar según tu código existente)."""
        # Placeholder - implementar según tu lógica
        try:
            from lark import Lark

            return Lark(
                APU_GRAMMAR,
                start='line',
                parser='lalr',
                maybe_placeholders=False,
                cache=True,
            )
        except Exception as e:
            logger.error(f"Error inicializando parser Lark: {e}")
            return None

    def _convert_to_dataframe(self, insumos: List[InsumoProcesado]) -> pd.DataFrame:
        """
        Convierte una lista de objetos `InsumoProcesado` a un DataFrame.
        """
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
                "NORMALIZED_DESC": getattr(insumo, "normalized_desc", "")
            }
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"✓ DataFrame creado: {len(df)} filas, {len(df.columns)} columnas")
        return df
