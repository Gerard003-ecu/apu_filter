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
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from lark import Lark, Token, Transformer, v_args
from lark.exceptions import LarkError

from .schemas import Equipo, InsumoProcesado, ManoDeObra, Otro, Suministro, Transporte
from .utils import parse_number

logger = logging.getLogger(__name__)


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ENUMS Y DATACLASSES
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


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
        self, config: Optional[Dict[str, Any]] = None, thresholds: Optional[ValidationThresholds] = None
    ):
        """
        Inicializa el extractor.

        Args:
            config: El diccionario de configuración.
            thresholds: Un objeto `ValidationThresholds` con los umbrales
                        para la identificación de valores.
        """
        self.config = config or {}
        self.pattern_matcher = PatternMatcher()
        self.thresholds = thresholds or ValidationThresholds()
        # AÑADIR: Leer el separador decimal de la configuración
        number_format = self.config.get("number_format", {})
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
        self, apu_context: Dict[str, Any], config: Dict[str, Any], keyword_cache: Any
    ):
        """
        Inicializa el Transformer.

        Args:
            apu_context: Diccionario con el contexto del APU actual (código,
                         descripción, etc.).
            config: Diccionario de configuración de la aplicación.
            keyword_cache: Cache de palabras clave (actualmente no usado).
        """
        self.apu_context = apu_context or {}
        self.config = config or {}
        self.keyword_cache = keyword_cache

        # Inicializar especialistas
        self.pattern_matcher = PatternMatcher()
        self.units_validator = UnitsValidator()
        self.thresholds = self._load_validation_thresholds()
        # CAMBIO: Pasar la config al NumericFieldExtractor
        self.numeric_extractor = NumericFieldExtractor(self.config, self.thresholds)
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
                categoria="MANO_DE_OBRA",
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
                categoria=tipo_insumo.value,
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
    Procesador principal que mantiene compatibilidad con `LoadDataStep`.

    Esta clase es el punto de entrada para el procesamiento de APUs. Recibe
    los registros crudos, inicializa el parser Lark con el `APUTransformer`
    y orquesta el procesamiento de todas las líneas de todos los APUs,
    delegando la lógica compleja a la arquitectura de especialistas.
    """

    def __init__(self, raw_records: List[Dict[str, Any]], config: Dict[str, Any] = None):
        """
        Inicializa el procesador de APU.

        Args:
            raw_records: Lista de registros crudos, cada uno representando un
                         APU con sus líneas de detalle.
            config: Diccionario de configuración de la aplicación.
        """
        self.raw_records = raw_records or []
        self.config = config or {}
        self.keyword_cache = None  # Cargar si es necesario

        # Crear parser una vez
        try:
            self.parser = Lark(APU_GRAMMAR, parser="lalr", transformer=None, debug=False)
            logger.info("Parser Lark inicializado con arquitectura de especialistas")
        except LarkError as e:
            logger.error(f"Error creando parser: {e}")
            self.parser = None

        # Inicializar especialistas para procesamiento fallback
        self.pattern_matcher = PatternMatcher()
        self.units_validator = UnitsValidator()
        self.thresholds = ValidationThresholds()
        self.numeric_extractor = NumericFieldExtractor(self.config, self.thresholds)

    def process_all(self) -> pd.DataFrame:
        """
        Procesa todos los registros de APU crudos y devuelve un DataFrame.

        Itera sobre cada registro de APU, extrae el contexto y luego procesa
        cada una de sus líneas de detalle utilizando el `APUTransformer`.
        Finalmente, consolida todos los insumos procesados en un único
        DataFrame de pandas.

        Returns:
            Un DataFrame con todos los insumos procesados y estructurados.
        """
        logger.info(
            f"Iniciando procesamiento de {len(self.raw_records)} APUs con especialistas"
        )

        all_results = []

        for i, record in enumerate(self.raw_records):
            try:
                apu_context = self._extract_apu_context(record)

                if "lines" in record and record["lines"]:
                    insumos = self._process_apu_lines(record["lines"], apu_context)
                    if insumos:
                        all_results.extend(insumos)

                if (i + 1) % 100 == 0:
                    logger.info(f"Procesados {i + 1}/{len(self.raw_records)} APUs")

            except Exception as e:
                logger.error(f"Error procesando APU {i}: {e}")
                continue

        logger.info(f"Procesamiento completado: {len(all_results)} insumos extraídos")

        # Convertir a DataFrame
        if all_results:
            return self._convert_to_dataframe(all_results)
        else:
            logger.warning("No se encontraron insumos válidos")
            return pd.DataFrame()

    def _extract_apu_context(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae el contexto relevante de un registro de APU."""
        return {
            "codigo_apu": record.get("codigo_apu", ""),
            "descripcion_apu": record.get("descripcion_apu", ""),
            "unidad_apu": record.get("unidad_apu", ""),
            "cantidad_apu": record.get("cantidad_apu", 1.0),
            "precio_unitario_apu": record.get("precio_unitario_apu", 0.0),
        }

    def _process_apu_lines(
        self, lines: List[str], apu_context: Dict[str, Any]
    ) -> List[InsumoProcesado]:
        """
        Procesa las líneas de detalle de un único APU.

        Args:
            lines: Lista de cadenas de texto, cada una es una línea del APU.
            apu_context: El contexto del APU al que pertenecen las líneas.

        Returns:
            Una lista de objetos `InsumoProcesado`.
        """
        results = []

        for line_num, line in enumerate(lines):
            if not line or not line.strip():
                continue

            try:
                if self.parser:
                    # Usar parser con transformer orquestador
                    tree = self.parser.parse(line.strip())
                    transformer = APUTransformer(
                        apu_context, self.config, self.keyword_cache
                    )
                    insumo = transformer.transform(tree)

                    if isinstance(insumo, list):
                        insumo = insumo[0] if insumo else None
                else:
                    # Fallback directo con especialistas si Lark falla
                    insumo = self._process_with_specialists(line, apu_context)

                if insumo:
                    insumo.line_number = line_num
                    results.append(insumo)

            except Exception as e:
                logger.debug(f"Error procesando línea {line_num}: {e}")
                continue

        return results

    def _process_with_specialists(
        self, line: str, apu_context: Dict[str, Any]
    ) -> Optional[InsumoProcesado]:
        """
        Procesamiento de fallback usando especialistas directamente sin Lark.

        Este método sirve como una alternativa si el parser Lark no está
        disponible. Realiza una división simple por ';' y aplica la lógica
        de los especialistas.

        Args:
            line: La línea de texto a procesar.
            apu_context: El contexto del APU.

        Returns:
            Un objeto `InsumoProcesado` o None.
        """
        fields = [f.strip() for f in line.split(";")]

        # Filtrar trailing empty
        while fields and not fields[-1]:
            fields.pop()

        if len(fields) < 4:
            return None

        descripcion = fields[0]

        # Usar PatternMatcher para detectar ruido
        if self.pattern_matcher.is_likely_summary(descripcion, len(fields)):
            return None

        # Extraer valores con NumericFieldExtractor
        valores = self.numeric_extractor.extract_insumo_values(fields)

        if len(valores) < 2:
            return None

        # Construir insumo
        unidad = self.units_validator.normalize_unit(fields[1]) if len(fields) > 1 else "UND"
        cantidad = valores[0] if valores else 1.0
        precio = valores[1] if len(valores) > 1 else 0.0
        total = valores[2] if len(valores) > 2 else cantidad * precio

        return Otro(
            descripcion_insumo=descripcion,
            unidad_insumo=unidad,
            cantidad=round(cantidad, 6),
            precio_unitario=round(precio, 2),
            valor_total=round(total, 2),
            rendimiento=round(cantidad, 6),
            formato_origen="SPECIALIST",
            tipo_insumo="OTRO",
            **apu_context,
        )

    def _convert_to_dataframe(self, insumos: List[InsumoProcesado]) -> pd.DataFrame:
        """
        Convierte una lista de objetos `InsumoProcesado` a un DataFrame.

        Args:
            insumos: La lista de insumos.

        Returns:
            Un DataFrame de pandas con los datos estructurados.
        """
        records = []
        for insumo in insumos:
            record = {
                "codigo_apu": getattr(insumo, "codigo_apu", ""),
                "descripcion_apu": getattr(insumo, "descripcion_apu", ""),
                "unidad_apu": getattr(insumo, "unidad_apu", ""),
                "descripcion_insumo": getattr(insumo, "descripcion_insumo", ""),
                "unidad_insumo": getattr(insumo, "unidad_insumo", ""),
                "cantidad": getattr(insumo, "cantidad", 0.0),
                "precio_unitario": getattr(insumo, "precio_unitario", 0.0),
                "valor_total": getattr(insumo, "valor_total", 0.0),
                "rendimiento": getattr(insumo, "rendimiento", 0.0),
                "tipo_insumo": getattr(insumo, "tipo_insumo", "OTRO"),
                "formato_origen": getattr(insumo, "formato_origen", ""),
            }
            records.append(record)

        return pd.DataFrame(records)
