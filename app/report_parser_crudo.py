"""
Módulo para el parseo crudo de reportes de Análisis de Precios Unitarios (APU).

Este módulo proporciona una clase `ReportParserCrudo` que implementa una máquina
de estados robusta para procesar, línea por línea, archivos de APU con un
formato semi-estructurado. Su objetivo principal es identificar y extraer los
registros de insumos asociados a cada APU, manteniendo el contexto del APU
al que pertenecen.
"""

import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from lark import Lark

from .utils import clean_apu_code

logger = logging.getLogger(__name__)


@dataclass
class LineValidationResult:
    """Resultado detallado de la validación de una línea."""

    is_valid: bool
    reason: str = ""
    fields_count: int = 0
    has_numeric_fields: bool = False
    validation_layer: str = ""  # "basic", "lark", "both"
    lark_tree: Any = None  # Árbol de parsing si fue exitoso


@dataclass
class ValidationStats:
    """Estadísticas detalladas de validación."""

    total_evaluated: int = 0
    passed_basic: int = 0
    passed_lark: int = 0
    passed_both: int = 0

    failed_basic_fields: int = 0
    failed_basic_numeric: int = 0
    failed_basic_subtotal: int = 0
    failed_basic_junk: int = 0

    failed_lark_parse: int = 0
    failed_lark_unexpected_input: int = 0
    failed_lark_unexpected_chars: int = 0

    cached_parses: int = 0

    failed_samples: List[Dict[str, Any]] = field(default_factory=list)


class ParserError(Exception):
    """Excepción base para errores ocurridos durante el parseo."""

    pass


class FileReadError(ParserError):
    """Indica un error al leer el archivo de entrada."""

    pass


class ParseStrategyError(ParserError):
    """Indica un error en la lógica de la estrategia de parseo."""

    pass


@dataclass
class APUContext:
    """
    Almacena el contexto de un APU mientras se procesan sus líneas.

    Attributes:
        apu_code: El código (ITEM) del APU.
        apu_desc: La descripción del APU.
        apu_unit: La unidad de medida del APU.
        source_line: El número de línea donde se detectó el APU.
    """

    apu_code: str
    apu_desc: str
    apu_unit: str
    source_line: int
    default_unit: str = "UND"

    def __post_init__(self):
        """Realiza validación y normalización después de la inicialización."""
        self.apu_code = self.apu_code.strip() if self.apu_code else ""
        self.apu_desc = self.apu_desc.strip() if self.apu_desc else ""
        self.apu_unit = self.apu_unit.strip().upper() if self.apu_unit else self.default_unit
        if not self.apu_code:
            raise ValueError("El código del APU no puede estar vacío.")

    @property
    def is_valid(self) -> bool:
        """Comprueba si el contexto del APU es válido."""
        return bool(self.apu_code and len(self.apu_code) >= 2)


@dataclass
class ParserContext:
    """
    Mantiene el estado mutable del parseo (La Pirámide en construcción).
    Actúa como la 'Memoria de Corto Plazo' del sistema.
    """

    current_apu: Optional[APUContext] = None  # El 'Padre' actual (Nivel 2)
    current_category: str = "INDEFINIDO"
    current_line_number: int = 0
    raw_records: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Telemetría interna
    stats: Counter = field(default_factory=Counter)

    def has_active_parent(self) -> bool:
        """Valida la lógica piramidal: ¿Existe un nodo padre activo?"""
        return self.current_apu is not None


class LineHandler(ABC):
    """
    Unidad de Trabajo Discreta.
    Patrón: Chain of Responsibility.
    """

    def __init__(self, parent_parser):
        self.parent = parent_parser  # Acceso a utilidades (Lark, Regex)

    @abstractmethod
    def can_handle(self, line: str, next_line: Optional[str] = None) -> bool:
        """Determina si este handler es responsable de la línea."""
        pass

    @abstractmethod
    def handle(
        self, line: str, context: ParserContext, next_line: Optional[str] = None
    ) -> bool:
        """
        Procesa la línea y actualiza el contexto (mutación de estado).
        Aquí se aplica la lógica de negocio.
        Returns: True si debe avanzar una línea extra (por encabezados multilínea), False si no.
        """
        pass


class JunkHandler(LineHandler):
    """Detecta y descarta basura, separadores o líneas decorativas."""

    def can_handle(self, line: str, next_line: Optional[str] = None) -> bool:
        return self.parent._is_junk_line(line.upper())

    def handle(
        self, line: str, context: ParserContext, next_line: Optional[str] = None
    ) -> bool:
        context.stats["junk_lines_skipped"] += 1
        return False


class HeaderHandler(LineHandler):
    """Detecta encabezados de APU (Nivel 2)."""

    def can_handle(self, line: str, next_line: Optional[str] = None) -> bool:
        line_upper = line.upper()
        is_header_line = "UNIDAD:" in line_upper
        is_item_line_next = (
            next_line is not None and "ITEM:" in next_line.upper()
        )
        return is_header_line and is_item_line_next

    def handle(
        self, line: str, context: ParserContext, next_line: Optional[str] = None
    ) -> bool:
        header_line = line
        item_line = next_line.strip() if next_line else ""

        try:
            apu_context_result = self.parent._extract_apu_header(
                header_line, item_line, context.current_line_number
            )

            if apu_context_result is not None:
                context.current_apu = apu_context_result
                context.current_category = "INDEFINIDO"
                context.stats["apus_detected"] += 1

                logger.info(
                    f"✓ APU detectado [línea {context.current_line_number}]: "
                    f"{context.current_apu.apu_code} - "
                    f"{context.current_apu.apu_desc[:50]}"
                )
            else:
                logger.warning(
                    f"Encabezado APU inválido en línea {context.current_line_number}"
                )
        except Exception as e:
            logger.warning(
                f"✗ Fallo al parsear encabezado de APU en línea {context.current_line_number}: {e}"
            )
            context.current_apu = None

        return True  # Consume la siguiente línea (ITEM)


class CategoryHandler(LineHandler):
    """Detecta cambios de categoría."""

    def can_handle(self, line: str, next_line: Optional[str] = None) -> bool:
        return self.parent._detect_category(line.upper()) is not None

    def handle(
        self, line: str, context: ParserContext, next_line: Optional[str] = None
    ) -> bool:
        new_category = self.parent._detect_category(line.upper())
        if new_category:
            context.current_category = new_category
            context.stats[f"category_{new_category}"] += 1
            logger.debug(f"  → Categoría: {new_category}")
        return False


class InsumoHandler(LineHandler):
    """Detecta y procesa líneas de insumos (Nivel 3)."""

    def can_handle(self, line: str, next_line: Optional[str] = None) -> bool:
        # Validación ligera preliminar: debe tener al menos un separador y algún número
        return ";" in line and any(c.isdigit() for c in line)

    def handle(
        self, line: str, context: ParserContext, next_line: Optional[str] = None
    ) -> bool:
        # 1. VALIDACIÓN PIRAMIDAL (Lógica Estructural)
        if not context.has_active_parent():
            # ERROR CRÍTICO DE NEGOCIO: Recurso Huérfano
            logger.warning(
                f"⚠️ Recurso Huérfano detectado en línea {context.current_line_number}. Ignorando."
            )
            context.stats["orphans_discarded"] += 1
            return False

        fields = [f.strip() for f in line.split(";")]
        validation_result = self.parent._validate_insumo_line(line, fields)

        if validation_result.is_valid:
            record = self.parent._build_insumo_record(
                context.current_apu,
                context.current_category,
                line,
                context.current_line_number,
                validation_result,
            )
            context.raw_records.append(record)
            context.stats["insumos_extracted"] += 1

            if self.parent.debug_mode:
                logger.debug(
                    f"  ✓ Insumo válido [línea {context.current_line_number}] "
                    f"[{validation_result.validation_layer}]: "
                    f"{fields[0][:40]}... ({validation_result.fields_count} campos)"
                )
        else:
            context.stats["lines_ignored_in_context"] += 1
            if self.parent.debug_mode:
                logger.debug(
                    f"  ✗ Rechazada [línea {context.current_line_number}]: {validation_result.reason}"
                )
        return False


class ReportParserCrudo:
    """
    Parser robusto tipo máquina de estados para archivos APU semi-estructurados.

    ROBUSTECIDO: Constantes centralizadas, límites de recursos, manejo defensivo.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # CONSTANTES DE CLASE
    # ═══════════════════════════════════════════════════════════════════════════

    # Límites de recursos
    _MAX_CACHE_SIZE: int = 50000
    _MAX_FAILED_SAMPLES: int = 20
    _MAX_LINE_LENGTH: int = 5000
    _MIN_FIELDS_FOR_INSUMO: int = 5
    _MIN_LINE_LENGTH: int = 3

    # Configuración de validación
    _CACHE_KEY_MAX_LENGTH: int = 2000

    CATEGORY_KEYWORDS = {
        "MATERIALES": {"MATERIALES", "MATERIAL", "MAT.", "INSUMOS"},
        "MANO DE OBRA": {"MANO DE OBRA", "MANO OBRA", "M.O.", "MO", "PERSONAL", "OBRERO"},
        "EQUIPO": {"EQUIPO", "EQUIPOS", "MAQUINARIA", "MAQ."},
        "TRANSPORTE": {"TRANSPORTE", "TRANSPORTES", "TRANS.", "ACARREO"},
        "HERRAMIENTA": {"HERRAMIENTA", "HERRAMIENTAS", "HERR.", "UTILES"},
        "OTROS": {"OTROS", "OTRO", "VARIOS", "ADICIONALES"},
    }

    JUNK_KEYWORDS = frozenset(
        {  # ROBUSTECIDO: frozenset para inmutabilidad y rendimiento
            "SUBTOTAL",
            "COSTO DIRECTO",
            "DESCRIPCION",
            "IMPUESTOS",
            "POLIZAS",
            "TOTAL",
            "IVA",
            "AIU",
        }
    )

    # Patrones pre-compilados para rendimiento
    _NUMERIC_PATTERN = re.compile(r"\d+[.,]\d+|\d+")
    _DECORATIVE_PATTERN = re.compile(r"^[=\-_\s*]+$")
    _UNIT_PATTERN = re.compile(r"UNIDAD:\s*(\S+)", re.IGNORECASE)
    _ITEM_PATTERN = re.compile(r"ITEM:\s*([\S,]+)", re.IGNORECASE)

    def __init__(
        self,
        file_path: Union[str, Path],
        profile: dict,
        config: Optional[Dict] = None,
    ):
        """
        Inicializa el parser con validación exhaustiva de parámetros.
        """
        # ROBUSTECIDO: Conversión segura de file_path
        if file_path is None:
            raise ValueError("file_path no puede ser None")
        self.file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

        # ROBUSTECIDO: Validación de tipos para profile y config
        if profile is not None and not isinstance(profile, dict):
            logger.warning(f"profile no es dict ({type(profile).__name__}), usando vacío")
            profile = {}
        if config is not None and not isinstance(config, dict):
            logger.warning(f"config no es dict ({type(config).__name__}), usando vacío")
            config = {}

        self.profile = profile or {}
        self.config = config or {}

        # Validar archivo antes de continuar
        self._validate_file_path()

        # ROBUSTECIDO: Inicialización segura del parser Lark
        self.lark_parser: Optional[Lark] = None
        self._parse_cache: Dict[str, Tuple[bool, Any]] = {}
        self.validation_stats = ValidationStats()

        try:
            from .apu_processor import APU_GRAMMAR

            self.lark_parser = self._initialize_lark_parser(APU_GRAMMAR)
        except ImportError as ie:
            logger.error(
                f"No se pudo importar APU_GRAMMAR: {ie}\n"
                f"  El parser funcionará sin validación Lark"
            )
        except Exception as e:
            logger.error(
                f"Error inicializando parser Lark: {e}\n"
                f"  El parser funcionará sin validación Lark"
            )

        # Estado del parser
        self.raw_records: List[Dict[str, Any]] = []
        self.stats: Counter = Counter()
        self._parsed: bool = False

        # ROBUSTECIDO: Modo debug desde config
        self.debug_mode = self.config.get("debug_mode", False)

        logger.debug(
            f"ReportParserCrudo inicializado:\n"
            f"  Archivo: {self.file_path.name}\n"
            f"  Lark parser: {'✓' if self.lark_parser else '✗'}\n"
            f"  Debug mode: {self.debug_mode}"
        )

    def _initialize_handlers(self) -> List[LineHandler]:
        """Fabrica la cadena de responsabilidad en orden de prioridad."""
        return [
            JunkHandler(self),  # 1. Descartar basura obvia
            HeaderHandler(self),  # 2. Detectar cambios de estructura (Nuevos APUs)
            CategoryHandler(self),  # 3. Detectar cambios de categoría
            InsumoHandler(self),  # 4. Procesar datos (Hojas del árbol)
        ]

    def _initialize_lark_parser(self, grammar: Optional[str] = None) -> Optional[Lark]:
        """
        Inicializa el parser Lark con la MISMA gramática que usa APUProcessor.
        """
        try:
            from lark import Lark
            from lark.exceptions import ConfigurationError, GrammarError
        except ImportError as ie:
            logger.error(f"No se pudo importar Lark: {ie}\n  Ejecute: pip install lark")
            return None

        # ROBUSTECIDO: Obtener gramática si no se proporcionó
        if grammar is None:
            try:
                from .apu_processor import APU_GRAMMAR

                grammar = APU_GRAMMAR
            except ImportError:
                logger.error("No se pudo importar APU_GRAMMAR desde apu_processor")
                return None

        # ROBUSTECIDO: Validar que la gramática no está vacía
        if not grammar or not isinstance(grammar, str) or not grammar.strip():
            logger.error("La gramática proporcionada está vacía o no es válida")
            return None

        try:
            # ROBUSTECIDO: Configuración idéntica a APUProcessor para coherencia
            parser_config = {
                "start": "line",
                "parser": "lalr",
                "maybe_placeholders": False,
                "propagate_positions": False,
                "cache": True,
            }

            parser = Lark(grammar, **parser_config)
            return parser

        except GrammarError as ge:
            logger.error(
                f"Error de gramática Lark:\n"
                f"  Mensaje: {ge}\n"
                f"  Revise que APU_GRAMMAR sea válida"
            )
            return None

        except ConfigurationError as ce:
            logger.error(f"Error de configuración Lark: {ce}")
            return None

        except Exception as e:
            logger.error(
                f"Error inesperado inicializando parser Lark: {e}"
            )
            return None

    def _validate_with_lark(
        self, line: str, use_cache: bool = True
    ) -> Tuple[bool, Optional[Any], str]:
        """
        Valida una línea usando el parser Lark.
        """
        # ROBUSTECIDO: Verificar disponibilidad del parser
        if self.lark_parser is None:
            return (True, None, "Lark no disponible - validación omitida")

        # ROBUSTECIDO: Validar entrada
        if not line or not isinstance(line, str):
            return (False, None, "Línea vacía o tipo inválido")

        line_clean = line.strip()

        # ROBUSTECIDO: Validar longitud antes de procesar
        if len(line_clean) > self._MAX_LINE_LENGTH:
            return (False, None, f"Línea demasiado larga: {len(line_clean)} caracteres")

        if len(line_clean) < self._MIN_LINE_LENGTH:
            return (False, None, f"Línea demasiado corta: {len(line_clean)} caracteres")

        # ROBUSTECIDO: Normalizar clave de cache para mejor hit rate
        cache_key = self._compute_cache_key(line_clean)

        # Verificar cache con validación
        if use_cache and cache_key in self._parse_cache:
            self.validation_stats.cached_parses += 1
            cached_result = self._parse_cache[cache_key]

            if isinstance(cached_result, tuple) and len(cached_result) == 2:
                is_valid, tree = cached_result
                return (is_valid, tree, "" if is_valid else "Cached failure")
            else:
                del self._parse_cache[cache_key]

        # ROBUSTECIDO: Importar excepciones específicas de Lark
        from lark.exceptions import (
            LarkError,
            UnexpectedCharacters,
            UnexpectedEOF,
            UnexpectedInput,
            UnexpectedToken,
        )

        try:
            tree = self.lark_parser.parse(line_clean)

            if not self._is_valid_tree(tree):
                if use_cache:
                    self._cache_result(cache_key, False, None)
                return (False, None, "Árbol de parsing inválido")

            # Cache de éxito
            if use_cache:
                self._cache_result(cache_key, True, tree)

            return (True, tree, "")

        except UnexpectedCharacters as uc:
            self.validation_stats.failed_lark_unexpected_chars += 1
            error_msg = (
                f"Carácter inesperado en columna {uc.column}: "
                f"'{line_clean[max(0, uc.column - 5) : uc.column + 5]}'"
            )
            if use_cache:
                self._cache_result(cache_key, False, None)
            return (False, None, f"Lark UnexpectedCharacters: {error_msg}")

        except UnexpectedToken as ut:
            self.validation_stats.failed_lark_parse += 1
            error_msg = f"Token inesperado '{ut.token}', esperado: {ut.expected}"
            if use_cache:
                self._cache_result(cache_key, False, None)
            return (False, None, f"Lark UnexpectedToken: {error_msg}")

        except UnexpectedEOF as ueof:
            self.validation_stats.failed_lark_parse += 1
            error_msg = f"Fin de entrada inesperado, esperado: {ueof.expected}"
            if use_cache:
                self._cache_result(cache_key, False, None)
            return (False, None, f"Lark UnexpectedEOF: {error_msg}")

        except UnexpectedInput as ui:
            self.validation_stats.failed_lark_unexpected_input += 1
            if use_cache:
                self._cache_result(cache_key, False, None)
            return (False, None, f"Lark UnexpectedInput: {ui}")

        except LarkError as le:
            self.validation_stats.failed_lark_parse += 1
            if use_cache:
                self._cache_result(cache_key, False, None)
            return (False, None, f"Lark Error genérico: {le}")

        except Exception as e:
            self.validation_stats.failed_lark_parse += 1
            logger.error(f"Error inesperado en validación Lark: {e}")
            if use_cache:
                self._cache_result(cache_key, False, None)
            return (False, None, f"Error inesperado: {type(e).__name__}: {e}")

    def _compute_cache_key(self, line: str) -> str:
        """
        Computa una clave de cache normalizada para una línea.
        """
        # Normalizar espacios múltiples
        normalized = " ".join(line.split())

        # Limitar longitud de clave
        if len(normalized) > self._CACHE_KEY_MAX_LENGTH:
            import hashlib

            hash_suffix = hashlib.md5(normalized.encode()).hexdigest()[:16]
            normalized = (
                normalized[: self._CACHE_KEY_MAX_LENGTH - 20] + f"...[{hash_suffix}]"
            )

        return normalized

    def _cache_result(self, key: str, is_valid: bool, tree: Any) -> None:
        """
        Almacena un resultado en cache con control de tamaño.
        """
        if len(self._parse_cache) >= self._MAX_CACHE_SIZE:
            keys_to_remove = list(self._parse_cache.keys())[: self._MAX_CACHE_SIZE // 10]
            for k in keys_to_remove:
                del self._parse_cache[k]

        self._parse_cache[key] = (is_valid, tree)

    def _is_valid_tree(self, tree: Any) -> bool:
        """
        Verifica que un árbol Lark es válido y usable.
        """
        if tree is None:
            return False

        try:
            if not hasattr(tree, "data"):
                return False
            if not hasattr(tree, "children"):
                return False
            if not isinstance(tree.data, str):
                return False
            return True
        except Exception:
            return False

    def _validate_basic_structure(self, line: str, fields: List[str]) -> Tuple[bool, str]:
        """
        Validación básica PRE-Lark para filtrado rápido.
        """
        if not line or not isinstance(line, str):
            self.validation_stats.failed_basic_fields += 1
            return (False, "Línea vacía o tipo inválido")

        if not fields or not isinstance(fields, list):
            self.validation_stats.failed_basic_fields += 1
            return (False, "Campos vacíos o tipo inválido")

        if len(fields) < self._MIN_FIELDS_FOR_INSUMO:
            self.validation_stats.failed_basic_fields += 1
            return (
                False,
                f"Insuficientes campos: {len(fields)} < {self._MIN_FIELDS_FOR_INSUMO}",
            )

        first_field = fields[0] if fields else ""
        if not first_field or not first_field.strip():
            self.validation_stats.failed_basic_fields += 1
            return (False, "Campo de descripción vacío")

        if len(first_field.strip()) < 2:
            self.validation_stats.failed_basic_fields += 1
            return (False, f"Descripción demasiado corta: '{first_field}'")

        line_upper = line.upper()
        subtotal_keywords = frozenset(
            {
                "SUBTOTAL",
                "TOTAL",
                "SUMA",
                "SUMATORIA",
                "COSTO DIRECTO",
                "COSTO TOTAL",
                "PRECIO TOTAL",
                "VALOR TOTAL",
                "GRAN TOTAL",
            }
        )

        for keyword in subtotal_keywords:
            if keyword in line_upper:
                self.validation_stats.failed_basic_subtotal += 1
                return (False, f"Línea de subtotal/total: contiene '{keyword}'")

        if self._is_junk_line(line_upper):
            self.validation_stats.failed_basic_junk += 1
            return (False, "Línea decorativa/separador")

        has_numeric = False
        for f in fields[1:]:
            if f and self._NUMERIC_PATTERN.search(f.strip()):
                has_numeric = True
                break

        if not has_numeric:
            self.validation_stats.failed_basic_numeric += 1
            return (False, "Sin campos numéricos detectables")

        for i, f in enumerate(fields):
            if len(f) > 500:
                self.validation_stats.failed_basic_fields += 1
                return (False, f"Campo {i} excesivamente largo: {len(f)} caracteres")

        self.validation_stats.passed_basic += 1
        return (True, "")

    def _validate_insumo_line(self, line: str, fields: List[str]) -> LineValidationResult:
        """
        Validación UNIFICADA de una línea candidata a insumo.
        """
        self.validation_stats.total_evaluated += 1

        if not line or not isinstance(line, str):
            return LineValidationResult(
                is_valid=False,
                reason="Línea vacía o tipo inválido",
                fields_count=0,
                validation_layer="input_validation",
            )

        if not fields or not isinstance(fields, list):
            return LineValidationResult(
                is_valid=False,
                reason="Campos vacíos o tipo inválido",
                fields_count=0,
                validation_layer="input_validation",
            )

        basic_valid, basic_reason = self._validate_basic_structure(line, fields)

        if not basic_valid:
            return LineValidationResult(
                is_valid=False,
                reason=f"Básica: {basic_reason}",
                fields_count=len(fields),
                has_numeric_fields=False,
                validation_layer="basic_failed",
            )

        lark_valid, lark_tree, lark_reason = self._validate_with_lark(line)

        if lark_valid:
            self.validation_stats.passed_lark += 1
            self.validation_stats.passed_both += 1

            return LineValidationResult(
                is_valid=True,
                reason="Validación completa exitosa",
                fields_count=len(fields),
                has_numeric_fields=True,
                validation_layer="both",
                lark_tree=lark_tree,
            )
        else:
            self._record_failed_sample(line, fields, lark_reason)

            return LineValidationResult(
                is_valid=False,
                reason=f"Lark: {lark_reason}",
                fields_count=len(fields),
                has_numeric_fields=True,
                validation_layer="lark_failed",
            )

    def _record_failed_sample(self, line: str, fields: List[str], reason: str) -> None:
        """
        Registra una muestra de línea fallida para análisis posterior.
        """
        max_samples = self.config.get("max_failed_samples", self._MAX_FAILED_SAMPLES)

        if len(self.validation_stats.failed_samples) >= max_samples:
            return

        safe_line = line[:200] if isinstance(line, str) else str(line)[:200]
        safe_fields = []
        empty_positions = []

        if isinstance(fields, list):
            for i, f in enumerate(fields):
                if isinstance(f, str):
                    safe_fields.append(f[:100] if len(f) > 100 else f)
                    if not f.strip():
                        empty_positions.append(i)
                else:
                    safe_fields.append(str(f)[:100])

        safe_reason = reason[:300] if isinstance(reason, str) else str(reason)[:300]

        sample = {
            "line": safe_line,
            "fields": safe_fields,
            "fields_count": len(fields) if isinstance(fields, list) else 0,
            "reason": safe_reason,
            "has_empty_fields": bool(empty_positions),
            "empty_field_positions": empty_positions,
            "line_length": len(line) if isinstance(line, str) else 0,
            "first_field_preview": safe_fields[0][:50] if safe_fields else "",
        }

        self.validation_stats.failed_samples.append(sample)

    def _log_validation_summary(self):
        """Registra un resumen detallado de la validación."""
        total = self.validation_stats.total_evaluated
        valid = self.stats.get("insumos_extracted", 0)

        logger.info("=" * 80)
        logger.info("📊 RESUMEN DE VALIDACIÓN CON LARK")
        logger.info("=" * 80)
        logger.info(f"Total líneas evaluadas: {total}")
        if total > 0:
            valid_percent = f"({valid / total * 100:.1f}%)"
            logger.info(f"✓ Insumos válidos (ambas capas): {valid} {valid_percent}")
        else:
            logger.info("✓ Insumos válidos (ambas capas): 0 (0.0%)")

        logger.info(f"  - Pasaron validación básica: {self.validation_stats.passed_basic}")
        logger.info(f"  - Pasaron validación Lark: {self.validation_stats.passed_lark}")
        logger.info(f"  - Cache hits: {self.validation_stats.cached_parses}")
        logger.info("")
        logger.info("Rechazos por validación básica:")
        logger.info(
            f"  - Campos insuficientes/vacíos: {self.validation_stats.failed_basic_fields}"
        )
        logger.info(f"  - Sin datos numéricos: {self.validation_stats.failed_basic_numeric}")
        logger.info(f"  - Subtotales: {self.validation_stats.failed_basic_subtotal}")
        logger.info(f"  - Líneas decorativas: {self.validation_stats.failed_basic_junk}")
        logger.info("")
        logger.info("Rechazos por validación Lark:")
        logger.info(f"  - Parse error genérico: {self.validation_stats.failed_lark_parse}")
        logger.info(
            f"  - Unexpected input: {self.validation_stats.failed_lark_unexpected_input}"
        )
        logger.info(
            "  - Unexpected characters: "
            f"{self.validation_stats.failed_lark_unexpected_chars}"
        )
        logger.info("=" * 80)

        # Mostrar muestras de fallos
        if self.validation_stats.failed_samples:
            logger.info("")
            logger.info("🔍 MUESTRAS DE LÍNEAS RECHAZADAS POR LARK:")
            logger.info("-" * 80)

            for idx, sample in enumerate(self.validation_stats.failed_samples, 1):
                logger.info(f"\nMuestra #{idx}:")
                logger.info(f"  Razón: {sample['reason']}")
                logger.info(f"  Campos: {sample['fields_count']}")
                logger.info(f"  Campos vacíos: {sample['has_empty_fields']}")
                if sample["has_empty_fields"]:
                    logger.info(f"  Posiciones vacías: {sample['empty_field_positions']}")
                logger.info(f"  Contenido: {sample['line']}")
                logger.info(f"  Campos: {sample['fields']}")

            logger.info("-" * 80)

        if valid == 0 and total > 0:
            logger.error("🚨 CRÍTICO: 0 insumos válidos con validación Lark.")
        elif total > 0 and valid < total * 0.5:
            logger.warning(
                f"⚠️  Tasa de validación baja: {valid / total * 100:.1f}%"
            )

    def get_parse_cache(self) -> Dict[str, Any]:
        """
        Retorna el cache de parsing para reutilización en APUProcessor.
        """
        valid_cache = {}
        invalid_count = 0

        for line, cached_value in self._parse_cache.items():
            if not isinstance(cached_value, tuple) or len(cached_value) != 2:
                invalid_count += 1
                continue

            is_valid, tree = cached_value

            if not is_valid or tree is None:
                continue

            if not self._is_valid_tree(tree):
                invalid_count += 1
                continue

            normalized_key = self._compute_cache_key(line)
            valid_cache[normalized_key] = tree

        if invalid_count > 0:
            logger.debug(f"Cache: {invalid_count} entradas inválidas filtradas")

        logger.info(f"Cache de parsing exportado: {len(valid_cache)} árboles válidos")

        return valid_cache

    def _validate_file_path(self) -> None:
        """Valida que la ruta del archivo sea un archivo válido y no vacío."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"La ruta no es un archivo: {self.file_path}")
        if self.file_path.stat().st_size == 0:
            raise ValueError(f"El archivo está vacío: {self.file_path}")

    def parse_to_raw(self) -> List[Dict[str, Any]]:
        """
        Punto de entrada principal para parsear el archivo.
        """
        if self._parsed:
            return self.raw_records

        logger.info(f"Iniciando parseo línea por línea de: {self.file_path.name}")

        try:
            content = self._read_file_safely()
            lines = content.split("\n")
            self.stats["total_lines"] = len(lines)

            # Inicializar handlers y contexto
            handlers = self._initialize_handlers()
            context = ParserContext()

            logger.info(f"🚀 Iniciando procesamiento de {len(lines)} líneas con Lógica Piramidal.")

            i = 0
            while i < len(lines):
                line = lines[i]
                context.current_line_number = i + 1
                line = line.strip()

                if not line:
                    i += 1
                    continue

                next_line = lines[i + 1].strip() if i + 1 < len(lines) else None
                handled = False

                for handler in handlers:
                    if handler.can_handle(line, next_line):
                        should_advance_extra = handler.handle(line, context, next_line)
                        if should_advance_extra:
                            i += 1  # Saltar la siguiente línea también (ej. ITEM)
                        handled = True
                        break

                if not handled:
                    logger.debug(f"Línea {i+1} no reconocida por ningún handler.")

                i += 1

            # Actualizar estado del objeto principal
            self.stats.update(context.stats)
            self.raw_records = context.raw_records
            self._parsed = True

            logger.info(
                f"Parseo completo. Extraídos {self.stats['insumos_extracted']} "
                "registros crudos."
            )

            self._log_validation_summary()

        except Exception as e:
            logger.error(f"Error crítico de parseo: {e}", exc_info=True)
            raise ParseStrategyError(
                f"Falló el parseo con estrategia Chain of Responsibility: {e}"
            ) from e

        return self.raw_records

    def _read_file_safely(self) -> str:
        """
        Lee el contenido del archivo intentando múltiples codificaciones.
        """
        default_encodings = self.config.get(
            "encodings", ["utf-8", "latin1", "cp1252", "iso-8859-1"]
        )
        encodings_to_try = [self.profile.get("encoding")] + default_encodings

        for encoding in filter(None, encodings_to_try):
            try:
                with open(self.file_path, "r", encoding=encoding, errors="strict") as f:
                    content = f.read()
                self.stats["encoding_used"] = encoding
                logger.info(f"Archivo leído exitosamente con codificación: {encoding}")
                return content
            except (UnicodeDecodeError, TypeError, LookupError):
                continue
        raise FileReadError(
            f"No se pudo leer el archivo {self.file_path} con ninguna de las "
            "codificaciones especificadas."
        )

    def _detect_category(self, line_upper: str) -> Optional[str]:
        """
        Detecta si una línea representa una categoría de insumos.
        """
        if len(line_upper) > 50 or sum(c.isdigit() for c in line_upper) > 3:
            return None
        for canonical, variations in self.CATEGORY_KEYWORDS.items():
            for variation in variations:
                pattern = (
                    r"\b" + re.escape(variation) + r"\b"
                    if "." not in variation
                    else re.escape(variation)
                )
                if re.search(pattern, line_upper):
                    return canonical
        return None

    def _is_junk_line(self, line_upper: str) -> bool:
        """
        Determina si una línea debe ser ignorada por ser "ruido".
        """
        if not line_upper or not isinstance(line_upper, str):
            return True

        stripped = line_upper.strip()

        if len(stripped) < self._MIN_LINE_LENGTH:
            return True

        for keyword in self.JUNK_KEYWORDS:
            if keyword in line_upper:
                return True

        if self._DECORATIVE_PATTERN.search(stripped):
            return True

        return False

    def _extract_apu_header(
        self, header_line: str, item_line: str, line_number: int
    ) -> Optional[APUContext]:
        """
        Extrae información del encabezado APU de forma segura.
        """
        try:
            parts = header_line.split(";")
            apu_desc = parts[0].strip() if parts else ""

            unit_match = self._UNIT_PATTERN.search(header_line)
            default_unit = self.config.get("default_unit", "UND")
            apu_unit = unit_match.group(1).strip() if unit_match else default_unit

            item_match = self._ITEM_PATTERN.search(item_line)
            if item_match:
                apu_code_raw = item_match.group(1)
            else:
                apu_code_raw = f"UNKNOWN_APU_{line_number}"

            apu_code = clean_apu_code(apu_code_raw)

            if not apu_code or len(apu_code) < 2:
                logger.warning(f"Código APU inválido extraído: '{apu_code}'")
                return None

            return APUContext(
                apu_code=apu_code,
                apu_desc=apu_desc,
                apu_unit=apu_unit,
                source_line=line_number,
            )

        except ValueError as ve:
            logger.debug(f"Validación de APUContext falló: {ve}")
            return None
        except Exception as e:
            logger.warning(f"Error extrayendo encabezado APU: {e}")
            return None

    def _build_insumo_record(
        self,
        context: APUContext,
        category: str,
        line: str,
        line_number: int,
        validation_result: LineValidationResult,
    ) -> Dict[str, Any]:
        """
        Construye un registro de insumo de forma estructurada.
        """
        return {
            "apu_code": context.apu_code,
            "apu_desc": context.apu_desc,
            "apu_unit": context.apu_unit,
            "category": category,
            "insumo_line": line,
            "source_line": line_number,
            "fields_count": validation_result.fields_count,
            "validation_layer": validation_result.validation_layer,
            "_lark_tree": validation_result.lark_tree,
        }
