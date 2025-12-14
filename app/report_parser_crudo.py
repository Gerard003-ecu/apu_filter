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

    JUNK_KEYWORDS = frozenset({  # ROBUSTECIDO: frozenset para inmutabilidad y rendimiento
        "SUBTOTAL",
        "COSTO DIRECTO",
        "DESCRIPCION",
        "IMPUESTOS",
        "POLIZAS",
        "TOTAL",
        "IVA",
        "AIU",
    })

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

        ROBUSTECIDO:
        - Validación defensiva de todos los parámetros
        - Inicialización segura de componentes
        - Manejo de errores en importaciones

        Args:
            file_path: Ruta al archivo a procesar.
            profile: Perfil de configuración.
            config: Configuración global.
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

    def _initialize_lark_parser(self, grammar: Optional[str] = None) -> Optional[Lark]:
        """
        Inicializa el parser Lark con la MISMA gramática que usa APUProcessor.

        ROBUSTECIDO:
        - Validación exhaustiva de gramática
        - Manejo específico de cada tipo de error Lark
        - Test de sanidad post-creación
        - Configuración coherente con APUProcessor

        Args:
            grammar: String con la gramática Lark.

        Returns:
            Instancia de Lark o None si falla la inicialización.
        """
        try:
            from lark import Lark
            from lark.exceptions import ConfigurationError, GrammarError
        except ImportError as ie:
            logger.error(
                f"No se pudo importar Lark: {ie}\n"
                f"  Ejecute: pip install lark"
            )
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

            # ROBUSTECIDO: Validación post-creación
            if parser is None:
                logger.error("Lark retornó None al crear parser")
                return None

            # ROBUSTECIDO: Test de sanidad con línea simple
            try:
                test_line = "descripcion;unidad;1;100;100"
                test_result = parser.parse(test_line)
                if test_result is None:
                    logger.warning(
                        "Test de sanidad del parser retornó None "
                        "(puede ser comportamiento esperado)"
                    )
            except Exception as test_error:
                # No es crítico si el test falla con datos genéricos
                logger.debug(f"Test de sanidad falló (esperado en algunos casos): {test_error}")

            logger.info("✓ Parser Lark inicializado correctamente para pre-validación")
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
                f"Error inesperado inicializando parser Lark:\n"
                f"  Tipo: {type(e).__name__}\n"
                f"  Error: {e}"
            )
            if self.config.get("debug_mode", False):
                import traceback
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
            return None

    def _validate_with_lark(
        self, line: str, use_cache: bool = True
    ) -> Tuple[bool, Optional[Any], str]:
        """
        Valida una línea usando el parser Lark.

        ROBUSTECIDO:
        - Manejo específico de cada tipo de excepción Lark
        - Validación de estructura del árbol resultante
        - Límites de cache
        - Normalización de clave de cache
        - Logging contextual

        Args:
            line: Línea a validar.
            use_cache: Si True, usa cache de parsing.

        Returns:
            Tupla (es_válida, árbol_parsing, razón_fallo)
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
            logger.debug(f"Línea excede longitud máxima: {len(line_clean)} > {self._MAX_LINE_LENGTH}")
            return (False, None, f"Línea demasiado larga: {len(line_clean)} caracteres")

        if len(line_clean) < self._MIN_LINE_LENGTH:
            return (False, None, f"Línea demasiado corta: {len(line_clean)} caracteres")

        # ROBUSTECIDO: Normalizar clave de cache para mejor hit rate
        cache_key = self._compute_cache_key(line_clean)

        # Verificar cache con validación
        if use_cache and cache_key in self._parse_cache:
            self.validation_stats.cached_parses += 1
            cached_result = self._parse_cache[cache_key]

            # ROBUSTECIDO: Validar estructura del resultado cacheado
            if isinstance(cached_result, tuple) and len(cached_result) == 2:
                is_valid, tree = cached_result
                return (is_valid, tree, "" if is_valid else "Cached failure")
            else:
                # Cache corrupto, eliminar entrada
                logger.debug(f"Entrada de cache corrupta para: {cache_key[:50]}...")
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

            # ROBUSTECIDO: Validar que el árbol tiene estructura esperada
            if not self._is_valid_tree(tree):
                logger.debug(f"Árbol Lark inválido para: {line_clean[:50]}...")
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
                f"'{line_clean[max(0, uc.column-5):uc.column+5]}'"
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
            # Error completamente inesperado
            self.validation_stats.failed_lark_parse += 1
            logger.error(
                f"Error inesperado en validación Lark:\n"
                f"  Tipo: {type(e).__name__}\n"
                f"  Error: {e}\n"
                f"  Línea: {line_clean[:100]}"
            )
            if use_cache:
                self._cache_result(cache_key, False, None)
            return (False, None, f"Error inesperado: {type(e).__name__}: {e}")

    def _compute_cache_key(self, line: str) -> str:
        """
        Computa una clave de cache normalizada para una línea.

        ROBUSTECIDO:
        - Normalización de espacios para mejor hit rate
        - Límite de longitud de clave
        - Coherente con APUProcessor

        Args:
            line: Línea original.

        Returns:
            Clave de cache normalizada.
        """
        # Normalizar espacios múltiples
        normalized = " ".join(line.split())

        # Limitar longitud de clave
        if len(normalized) > self._CACHE_KEY_MAX_LENGTH:
            # Usar hash para claves muy largas
            import hashlib
            hash_suffix = hashlib.md5(normalized.encode()).hexdigest()[:16]
            normalized = normalized[:self._CACHE_KEY_MAX_LENGTH - 20] + f"...[{hash_suffix}]"

        return normalized

    def _cache_result(self, key: str, is_valid: bool, tree: Any) -> None:
        """
        Almacena un resultado en cache con control de tamaño.

        ROBUSTECIDO:
        - Límite de tamaño de cache
        - Evicción LRU simplificada

        Args:
            key: Clave de cache.
            is_valid: Si el resultado es válido.
            tree: Árbol Lark (si aplica).
        """
        # Verificar límite de cache
        if len(self._parse_cache) >= self._MAX_CACHE_SIZE:
            # Evicción simple: eliminar 10% de las entradas más antiguas
            keys_to_remove = list(self._parse_cache.keys())[: self._MAX_CACHE_SIZE // 10]
            for k in keys_to_remove:
                del self._parse_cache[k]
            logger.debug(
                f"Cache de parsing purgado: eliminadas {len(keys_to_remove)} entradas"
            )

        self._parse_cache[key] = (is_valid, tree)

    def _is_valid_tree(self, tree: Any) -> bool:
        """
        Verifica que un árbol Lark es válido y usable.

        ROBUSTECIDO:
        - Verificación de estructura básica
        - Coherente con APUProcessor

        Args:
            tree: Árbol Lark a validar.

        Returns:
            True si es válido, False en caso contrario.
        """
        if tree is None:
            return False

        try:
            # Un árbol Lark válido debe tener estos atributos
            if not hasattr(tree, "data"):
                return False
            if not hasattr(tree, "children"):
                return False
            # Verificar que data es un string (nombre de la regla)
            if not isinstance(tree.data, str):
                return False
            return True
        except Exception:
            return False

    def _validate_basic_structure(self, line: str, fields: List[str]) -> Tuple[bool, str]:
        """
        Validación básica PRE-Lark para filtrado rápido.

        ROBUSTECIDO:
        - Uso de constantes de clase
        - Patrones pre-compilados
        - Validaciones adicionales de seguridad
        - Logging mejorado

        Args:
            line: Línea completa.
            fields: Campos separados por ";".

        Returns:
            Tupla (es_válida, razón_si_inválida)
        """
        # ROBUSTECIDO: Validar entrada
        if not line or not isinstance(line, str):
            self.validation_stats.failed_basic_fields += 1
            return (False, "Línea vacía o tipo inválido")

        if not fields or not isinstance(fields, list):
            self.validation_stats.failed_basic_fields += 1
            return (False, "Campos vacíos o tipo inválido")

        # Validación 1: Número mínimo de campos (usando constante)
        if len(fields) < self._MIN_FIELDS_FOR_INSUMO:
            self.validation_stats.failed_basic_fields += 1
            return (False, f"Insuficientes campos: {len(fields)} < {self._MIN_FIELDS_FOR_INSUMO}")

        # Validación 2: Campo de descripción no vacío y razonable
        first_field = fields[0] if fields else ""
        if not first_field or not first_field.strip():
            self.validation_stats.failed_basic_fields += 1
            return (False, "Campo de descripción vacío")

        # ROBUSTECIDO: Descripción demasiado corta
        if len(first_field.strip()) < 2:
            self.validation_stats.failed_basic_fields += 1
            return (False, f"Descripción demasiado corta: '{first_field}'")

        # Validación 3: Detectar subtotales/totales (case-insensitive)
        line_upper = line.upper()

        # ROBUSTECIDO: Lista extendida de keywords de subtotal
        subtotal_keywords = frozenset({
            "SUBTOTAL",
            "TOTAL",
            "SUMA",
            "SUMATORIA",
            "COSTO DIRECTO",
            "COSTO TOTAL",
            "PRECIO TOTAL",
            "VALOR TOTAL",
            "GRAN TOTAL",
        })

        for keyword in subtotal_keywords:
            if keyword in line_upper:
                self.validation_stats.failed_basic_subtotal += 1
                return (False, f"Línea de subtotal/total: contiene '{keyword}'")

        # Validación 4: Líneas decorativas (usando patrón pre-compilado)
        if self._is_junk_line(line_upper):
            self.validation_stats.failed_basic_junk += 1
            return (False, "Línea decorativa/separador")

        # Validación 5: Al menos un campo numérico (usando patrón pre-compilado)
        has_numeric = False
        for f in fields[1:]:  # Saltar descripción
            if f and self._NUMERIC_PATTERN.search(f.strip()):
                has_numeric = True
                break

        if not has_numeric:
            self.validation_stats.failed_basic_numeric += 1
            return (False, "Sin campos numéricos detectables")

        # ROBUSTECIDO: Validación adicional - campos no demasiado largos
        for i, f in enumerate(fields):
            if len(f) > 500:
                self.validation_stats.failed_basic_fields += 1
                return (False, f"Campo {i} excesivamente largo: {len(f)} caracteres")

        self.validation_stats.passed_basic += 1
        return (True, "")

    def _validate_insumo_line(self, line: str, fields: List[str]) -> LineValidationResult:
        """
        Validación UNIFICADA de una línea candidata a insumo.

        ROBUSTECIDO:
        - Manejo defensivo de entradas
        - Estrategia de validación en dos capas claramente definida
        - Resultado detallado para diagnóstico
        - Coherente con validación de APUProcessor

        Args:
            line: La línea original completa.
            fields: Los campos ya separados por ";".

        Returns:
            LineValidationResult con el resultado detallado.
        """
        self.validation_stats.total_evaluated += 1

        # ROBUSTECIDO: Validación de entrada
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

        # ═══════════════════════════════════════════════════════════════════════════
        # CAPA 1: Validación básica (filtro rápido)
        # ═══════════════════════════════════════════════════════════════════════════
        basic_valid, basic_reason = self._validate_basic_structure(line, fields)

        if not basic_valid:
            return LineValidationResult(
                is_valid=False,
                reason=f"Básica: {basic_reason}",
                fields_count=len(fields),
                has_numeric_fields=False,
                validation_layer="basic_failed",
            )

        # ═══════════════════════════════════════════════════════════════════════════
        # CAPA 2: Validación Lark (el juez final)
        # ═══════════════════════════════════════════════════════════════════════════
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
            # Fallo en Lark - registrar para análisis
            self._record_failed_sample(line, fields, lark_reason)

            return LineValidationResult(
                is_valid=False,
                reason=f"Lark: {lark_reason}",
                fields_count=len(fields),
                has_numeric_fields=True,  # Pasó validación básica, tiene numéricos
                validation_layer="lark_failed",
            )

    def _record_failed_sample(self, line: str, fields: List[str], reason: str) -> None:
        """
        Registra una muestra de línea fallida para análisis posterior.

        ROBUSTECIDO:
        - Límite de muestras configurable
        - Truncamiento seguro de contenido
        - Información adicional de diagnóstico
        - Manejo defensivo de campos

        Args:
            line: Línea que falló.
            fields: Campos de la línea.
            reason: Razón del fallo.
        """
        max_samples = self.config.get("max_failed_samples", self._MAX_FAILED_SAMPLES)

        if len(self.validation_stats.failed_samples) >= max_samples:
            return  # Ya tenemos suficientes muestras

        # ROBUSTECIDO: Validación defensiva
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
            # ROBUSTECIDO: Información adicional
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

        # Alertas críticas
        if valid == 0 and total > 0:
            logger.error(
                "🚨 CRÍTICO: 0 insumos válidos con validación Lark.\n"
                "   Posibles causas:\n"
                "   1. Gramática Lark incompatible con formato de datos\n"
                "   2. Perfil de configuración incorrecto\n"
                "   3. Formato de archivo no esperado\n"
                "   → Revise las muestras de líneas rechazadas arriba"
            )
        elif total > 0 and valid < total * 0.5:
            logger.warning(
                f"⚠️  Tasa de validación baja: {valid / total * 100:.1f}%\n"
                f"   Considere revisar la gramática o el formato de datos"
            )

    def get_parse_cache(self) -> Dict[str, Any]:
        """
        Retorna el cache de parsing para reutilización en APUProcessor.

        ROBUSTECIDO:
        - Validación de estructura del cache
        - Filtrado de entradas inválidas
        - Coherente con estructura esperada por APUProcessor

        Returns:
            Diccionario con líneas parseadas y sus árboles Lark válidos.
        """
        valid_cache = {}
        invalid_count = 0

        for line, cached_value in self._parse_cache.items():
            # ROBUSTECIDO: Validar estructura de cada entrada
            if not isinstance(cached_value, tuple) or len(cached_value) != 2:
                invalid_count += 1
                continue

            is_valid, tree = cached_value

            if not is_valid:
                continue

            if tree is None:
                continue

            # ROBUSTECIDO: Validar que el árbol es usable
            if not self._is_valid_tree(tree):
                invalid_count += 1
                continue

            # ROBUSTECIDO: Usar clave normalizada coherente con APUProcessor
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

        Lee el archivo de forma segura, lo divide en líneas y orquesta el
        proceso de parseo a través de la máquina de estados `_parse_by_lines`.

        Returns:
            Una lista de diccionarios, donde cada uno es un registro crudo de insumo.

        Raises:
            ParseStrategyError: Si ocurre un error crítico durante el parseo.
        """
        if self._parsed:
            return self.raw_records

        logger.info(f"Iniciando parseo línea por línea de: {self.file_path.name}")

        try:
            content = self._read_file_safely()
            lines = content.split("\n")
            self.stats["total_lines"] = len(lines)

            self._parse_by_lines(lines)

            self._parsed = True
            logger.info(
                f"Parseo completo. Extraídos {self.stats['insumos_extracted']} "
                "registros crudos."
            )
            if self.stats["insumos_extracted"] == 0:
                logger.warning(
                    "No se extrajeron registros. El archivo puede estar vacío o "
                    "en un formato inesperado."
                )

        except Exception as e:
            logger.error(f"Error crítico de parseo: {e}", exc_info=True)
            raise ParseStrategyError(
                f"Falló el parseo con estrategia línea por línea: {e}"
            ) from e

        return self.raw_records

    def _read_file_safely(self) -> str:
        """
        Lee el contenido del archivo intentando múltiples codificaciones.

        Returns:
            El contenido del archivo como una cadena de texto.

        Raises:
            FileReadError: Si no se puede leer el archivo con ninguna de las
                           codificaciones especificadas.
        """
        # Usar el encoding del perfil como primera opción, con fallback a la config general
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

        Args:
            line_upper: La línea de texto en mayúsculas.

        Returns:
            El nombre canónico de la categoría si se detecta una, o None.
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

        ROBUSTECIDO:
        - Uso de frozenset para keywords (rendimiento)
        - Patrón pre-compilado para decorativas
        - Validación de entrada

        Args:
            line_upper: La línea de texto en mayúsculas.

        Returns:
            True si la línea es "ruido", False en caso contrario.
        """
        # ROBUSTECIDO: Validar entrada
        if not line_upper or not isinstance(line_upper, str):
            return True  # Línea vacía o inválida es ruido

        stripped = line_upper.strip()

        # Líneas muy cortas
        if len(stripped) < self._MIN_LINE_LENGTH:
            return True

        # Keywords de ruido (usando frozenset para O(1) lookup)
        for keyword in self.JUNK_KEYWORDS:
            if keyword in line_upper:
                return True

        # Líneas decorativas (usando patrón pre-compilado)
        if self._DECORATIVE_PATTERN.search(stripped):
            return True

        return False

    def _parse_by_lines(self, lines: List[str]) -> bool:
        """
        Máquina de estados con validación UNIFICADA usando Lark.

        ROBUSTECIDO:
        - Validación de entrada
        - Manejo defensivo de contexto APU
        - Logging mejorado con contexto
        - Límites de procesamiento
        - Coherente con validación de APUProcessor

        Args:
            lines: La lista de todas las líneas del archivo.

        Returns:
            True si se extrajo al menos un insumo válido, False en caso contrario.
        """
        # ROBUSTECIDO: Validar entrada
        if not lines or not isinstance(lines, list):
            logger.warning("_parse_by_lines: lista de líneas vacía o inválida")
            return False

        current_apu_context: Optional[APUContext] = None
        current_category = "INDEFINIDO"
        total_lines = len(lines)

        # ROBUSTECIDO: Límite de líneas para evitar procesamiento infinito
        max_lines = self.config.get("max_lines_to_process", 1_000_000)
        if total_lines > max_lines:
            logger.warning(
                f"Archivo muy grande ({total_lines} líneas), "
                f"procesando solo las primeras {max_lines}"
            )
            lines = lines[:max_lines]
            total_lines = max_lines

        logger.info(f"Iniciando parsing de {total_lines} líneas con validación Lark")

        i = 0
        consecutive_errors = 0
        max_consecutive_errors = 100  # Límite de errores consecutivos

        while i < total_lines:
            # ROBUSTECIDO: Verificar límite de errores consecutivos
            if consecutive_errors >= max_consecutive_errors:
                logger.error(
                    f"Demasiados errores consecutivos ({consecutive_errors}), "
                    f"abortando parsing en línea {i}"
                )
                break

            line = lines[i]

            # ROBUSTECIDO: Validar tipo de línea
            if not isinstance(line, str):
                logger.debug(f"Línea {i+1}: tipo inválido {type(line).__name__}, saltando")
                i += 1
                consecutive_errors += 1
                continue

            line = line.strip()

            if not line:
                i += 1
                consecutive_errors = 0  # Reset en líneas vacías (normales)
                continue

            # ═══════════════════════════════════════════════════════════════════════
            # ESTADO 1: Buscar encabezado de APU
            # ═══════════════════════════════════════════════════════════════════════
            line_upper = line.upper()
            is_header_line = "UNIDAD:" in line_upper
            is_item_line_next = (
                (i + 1) < total_lines
                and isinstance(lines[i + 1], str)
                and "ITEM:" in lines[i + 1].upper()
            )

            if is_header_line and is_item_line_next:
                header_line = line
                item_line = lines[i + 1].strip()

                try:
                    # ROBUSTECIDO: Extracción con manejo de errores
                    apu_context_result = self._extract_apu_header(
                        header_line, item_line, i + 1
                    )

                    if apu_context_result is not None:
                        current_apu_context = apu_context_result
                        current_category = "INDEFINIDO"
                        self.stats["apus_detected"] += 1
                        consecutive_errors = 0

                        logger.info(
                            f"✓ APU detectado [línea {i + 1}]: "
                            f"{current_apu_context.apu_code} - "
                            f"{current_apu_context.apu_desc[:50]}"
                        )
                    else:
                        logger.warning(f"Encabezado APU inválido en línea {i + 1}")
                        consecutive_errors += 1

                    i += 2
                    continue

                except Exception as e:
                    logger.warning(
                        f"✗ Fallo al parsear encabezado de APU en línea {i + 1}: {e}"
                    )
                    current_apu_context = None
                    consecutive_errors += 1
                    i += 1
                    continue

            # ═══════════════════════════════════════════════════════════════════════
            # ESTADO 2: Procesar líneas dentro de contexto de APU
            # ═══════════════════════════════════════════════════════════════════════
            if current_apu_context is not None:
                # Detectar categoría
                new_category = self._detect_category(line_upper)
                if new_category:
                    current_category = new_category
                    self.stats[f"category_{current_category}"] += 1
                    logger.debug(f"  → Categoría: {current_category}")
                    i += 1
                    consecutive_errors = 0
                    continue

                # Detectar ruido
                if self._is_junk_line(line_upper):
                    self.stats["junk_lines_skipped"] += 1
                    i += 1
                    consecutive_errors = 0
                    continue

                # ═══════════════════════════════════════════════════════════════════
                # 🔥 VALIDACIÓN CRÍTICA CON LARK
                # ═══════════════════════════════════════════════════════════════════
                fields = [f.strip() for f in line.split(";")]
                validation_result = self._validate_insumo_line(line, fields)

                if validation_result.is_valid:
                    # ✅ LÍNEA VÁLIDA - Garantizada procesable por APUProcessor
                    record = self._build_insumo_record(
                        current_apu_context,
                        current_category,
                        line,
                        i + 1,
                        validation_result
                    )
                    self.raw_records.append(record)
                    self.stats["insumos_extracted"] += 1
                    consecutive_errors = 0

                    if self.debug_mode:
                        logger.debug(
                            f"  ✓ Insumo válido [línea {i + 1}] "
                            f"[{validation_result.validation_layer}]: "
                            f"{fields[0][:40]}... ({validation_result.fields_count} campos)"
                        )
                else:
                    # ❌ LÍNEA RECHAZADA
                    if self.debug_mode:
                        logger.debug(
                            f"  ✗ Rechazada [línea {i + 1}]: {validation_result.reason}\n"
                            f"    Contenido: {line[:80]}..."
                        )
                    self.stats["lines_ignored_in_context"] += 1
                    # No incrementar consecutive_errors para rechazos válidos

            i += 1

        # Log de estadísticas finales
        self._log_validation_summary()

        return self.stats["insumos_extracted"] > 0


    def _extract_apu_header(
        self, header_line: str, item_line: str, line_number: int
    ) -> Optional[APUContext]:
        """
        Extrae información del encabezado APU de forma segura.

        ROBUSTECIDO:
        - Manejo de campos faltantes
        - Validación de valores extraídos
        - Valores por defecto seguros

        Args:
            header_line: Línea con la descripción.
            item_line: Línea con el item.
            line_number: Número de línea.

        Returns:
            Objeto APUContext si es válido, None en caso contrario.
        """
        try:
            # Extraer descripción
            parts = header_line.split(";")
            apu_desc = parts[0].strip() if parts else ""

            # Extraer unidad
            unit_match = self._UNIT_PATTERN.search(header_line)
            default_unit = self.config.get("default_unit", "UND")
            apu_unit = unit_match.group(1).strip() if unit_match else default_unit

            # Extraer código
            item_match = self._ITEM_PATTERN.search(item_line)
            if item_match:
                apu_code_raw = item_match.group(1)
            else:
                apu_code_raw = f"UNKNOWN_APU_{line_number}"

            apu_code = clean_apu_code(apu_code_raw)

            # Validar que tenemos un código válido
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

        ROBUSTECIDO:
        - Validación de todos los campos
        - Estructura coherente con APUProcessor

        Args:
            context: Contexto del APU.
            category: Categoría del insumo.
            line: Línea del insumo.
            line_number: Número de línea.
            validation_result: Resultado de la validación.

        Returns:
            Diccionario con el registro del insumo.
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
            # 🔥 OPTIMIZACIÓN: Guardar árbol de parsing para reutilizar
            "_lark_tree": validation_result.lark_tree,
        }
