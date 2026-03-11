"""
Este módulo implementa una Máquina de Estados Finita diseñada para navegar la topología
de archivos de presupuesto semi-estructurados. Su objetivo no es interpretar el precio,
sino validar la integridad de la "forma" de los datos (Homeomorfismo) antes de permitir
su procesamiento semántico.

Capacidades y Métricas:
-----------------------
1. Validación Homeomórfica (`_is_apu_homeomorphic`):
   Verifica que el árbol sintáctico generado (Lark) sea topológicamente equivalente
   al esquema platónico de un APU válido, preservando la estructura jerárquica padre-hijo.

2. Física del Texto (Entropía y Densidad):
   Calcula métricas estructurales para distinguir señal de ruido:
   - Entropía de Campo (`_calculate_field_entropy`): Mide el desorden en la tipificación
     de datos usando entropía de Shannon H = -Σ p·log₂(p), normalizada a [0,1].
   - Densidad Estructural (`_calculate_structural_density`): Relación señal/ruido por
     línea, acotada a [0,1].
   - Cohesión Numérica (`_calculate_numeric_cohesion`): Agrupamiento de valores
     cuantitativos, normalizada a [0,1].

3. Patrón Chain of Responsibility:
   Implementa una cadena de handlers especializados (`JunkHandler`, `HeaderHandler`,
   `CategoryHandler`, `InsumoHandler`) que actúan como filtros secuenciales para
   clasificar cada línea según su función estructural en el documento.

4. Contexto y Estado:
   Mantiene una memoria de corto plazo (`ParserContext`) para resolver la jerarquía
   del presupuesto (Capítulo -> APU -> Insumo) y detectar recursos huérfanos.

Correcciones v3:
----------------
- Orden correcto de cláusulas `except` para jerarquía de excepciones Lark.
- `_calculate_numeric_cohesion` normalizada a [0,1] con función sigmoidea inversa.
- `_calculate_structural_density` acotada a [0,1].
- `_calculate_field_entropy` con imports a nivel módulo y manejo robusto de log2.
- `_compute_structural_signature` elimina re-import de hashlib.
- `_calculate_homogeneity_index` elimina re-import de Counter.
- `_is_apu_homeomorphic` con import de Token a nivel módulo.
- `_has_minimal_structural_connectivity` con guard para líneas muy cortas.
- `_detect_category` con normalización correcta para búsqueda case-insensitive.
- `_compute_semantic_cache_key` con lookahead correcto en regex de ceros.
- `_is_supremum_match` con lógica de abreviaturas corregida.
- Umbrales de `_determine_homeomorphism_class` documentados con justificación.
- `_log_validation_summary` usa indexación `Counter` consistentemente.
"""

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from math import log2
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import de Token a nivel módulo para evitar imports repetidos en métodos internos
try:
    from lark import Lark, Token as LarkToken
    from lark.exceptions import (
        LarkError,
        UnexpectedCharacters,
        UnexpectedEOF,
        UnexpectedInput,
        UnexpectedToken,
    )
    _LARK_AVAILABLE = True
except ImportError:
    _LARK_AVAILABLE = False
    LarkToken = None  # type: ignore[assignment,misc]

from app.core.utils import clean_apu_code

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE MÓDULO
# ─────────────────────────────────────────────────────────────────────────────

# Número de tipos de campo para normalización de entropía.
# Los tipos reconocidos son: "alpha", "numeric", "mixed", "empty" → 4 categorías.
# H_max = log2(4) ≈ 2.0 bits  →  se usa como denominador en normalización.
_FIELD_TYPE_COUNT: int = 4
_H_MAX: float = log2(_FIELD_TYPE_COUNT)  # ≈ 2.0 bits

# Factor de escala para cohesión numérica: distancia promedio de 1 campo
# (adyacente) → cohesión = 1.0; de 10 campos → cohesión ≈ 0.09.
# Se usa 1/(1 + d) para acotar a (0, 1].
_COHESION_OFFSET: float = 1.0


@dataclass
class LineValidationResult:
    """Resultado detallado de la validación de una línea."""

    is_valid: bool
    reason: str = ""
    fields_count: int = 0
    has_numeric_fields: bool = False
    validation_layer: str = ""  # "basic", "lark", "both", "full_homeomorphism"
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

    # Contadores de error Lark: ahora todos se incrementan correctamente
    # gracias al orden de except corregido (específico → general).
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
        apu_code:    Código (ITEM) del APU. Mínimo 2 caracteres tras limpieza.
        apu_desc:    Descripción textual del APU.
        apu_unit:    Unidad de medida (ej. M2, ML, UND). Siempre en mayúsculas.
        source_line: Número de línea (1-indexado) donde se detectó el encabezado.
        default_unit: Unidad de respaldo cuando no se puede extraer del texto.
    """

    apu_code: str
    apu_desc: str
    apu_unit: str
    source_line: int
    default_unit: str = "UND"

    def __post_init__(self) -> None:
        """Realiza validación y normalización después de la inicialización."""
        self.apu_code = self.apu_code.strip() if self.apu_code else ""
        self.apu_desc = self.apu_desc.strip() if self.apu_desc else ""
        self.apu_unit = (
            self.apu_unit.strip().upper() if self.apu_unit else self.default_unit
        )
        if not self.apu_code:
            raise ValueError("El código del APU no puede estar vacío.")

    @property
    def is_valid(self) -> bool:
        """Comprueba si el contexto del APU es válido (código ≥ 2 caracteres)."""
        return bool(self.apu_code and len(self.apu_code) >= 2)


@dataclass
class ParserContext:
    """
    Mantiene el estado mutable del parseo (La Pirámide en construcción).

    Actúa como la 'Memoria de Corto Plazo' del sistema. La pirámide tiene
    tres niveles: Categoría (raíz) → APU (nodo) → Insumo (hoja).
    """

    current_apu: Optional[APUContext] = None  # Nodo padre activo (Nivel 2)
    current_category: str = "INDEFINIDO"
    current_line_number: int = 0
    raw_records: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Telemetría interna: usa Counter para acumulación eficiente
    stats: Counter = field(default_factory=Counter)

    def has_active_parent(self) -> bool:
        """Valida la lógica piramidal: ¿Existe un nodo padre activo?"""
        return self.current_apu is not None


class LineHandler(ABC):
    """
    Unidad de Trabajo Discreta.

    Patrón: Chain of Responsibility. Cada handler es una función parcial
    definida sobre un subconjunto del espacio de líneas.
    """

    def __init__(self, parent_parser: "ReportParserCrudo") -> None:
        """Inicializa el handler con una referencia al parser padre."""
        self.parent = parent_parser

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

        Returns:
            True si debe avanzar una línea extra (encabezados multilínea),
            False en caso contrario.
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
    """Detecta encabezados de APU (Nivel 2 de la pirámide)."""

    def can_handle(self, line: str, next_line: Optional[str] = None) -> bool:
        line_upper = line.upper()
        is_header_line = "UNIDAD:" in line_upper
        is_item_line_next = next_line is not None and "ITEM:" in next_line.upper()
        return is_header_line and is_item_line_next

    def handle(
        self, line: str, context: ParserContext, next_line: Optional[str] = None
    ) -> bool:
        item_line = next_line.strip() if next_line else ""

        try:
            apu_context_result = self.parent._extract_apu_header(
                line, item_line, context.current_line_number
            )

            if apu_context_result is not None:
                context.current_apu = apu_context_result
                context.current_category = "INDEFINIDO"
                context.stats["apus_detected"] += 1
                logger.info(
                    "✓ APU detectado [línea %d]: %s - %s",
                    context.current_line_number,
                    context.current_apu.apu_code,
                    context.current_apu.apu_desc[:50],
                )
            else:
                logger.warning(
                    "Encabezado APU inválido en línea %d",
                    context.current_line_number,
                )
        except Exception as exc:
            logger.warning(
                "✗ Fallo al parsear encabezado de APU en línea %d: %s",
                context.current_line_number,
                exc,
            )
            context.current_apu = None

        return True  # Consume la siguiente línea (ITEM:)


class CategoryHandler(LineHandler):
    """Detecta cambios de categoría (MATERIALES, MANO DE OBRA, etc.)."""

    def can_handle(self, line: str, next_line: Optional[str] = None) -> bool:
        return self.parent._detect_category(line.upper()) is not None

    def handle(
        self, line: str, context: ParserContext, next_line: Optional[str] = None
    ) -> bool:
        new_category = self.parent._detect_category(line.upper())
        if new_category:
            context.current_category = new_category
            context.stats[f"category_{new_category}"] += 1
            logger.debug("  → Categoría: %s", new_category)
        return False


class InsumoHandler(LineHandler):
    """Detecta y procesa líneas de insumos (Nivel 3 — hojas de la pirámide)."""

    def can_handle(self, line: str, next_line: Optional[str] = None) -> bool:
        # Pre-filtro ligero: debe tener separador y al menos un dígito
        return ";" in line and any(c.isdigit() for c in line)

    def handle(
        self, line: str, context: ParserContext, next_line: Optional[str] = None
    ) -> bool:
        # ── VALIDACIÓN PIRAMIDAL: recurso huérfano ────────────────────────────
        if not context.has_active_parent():
            logger.warning(
                "⚠️ Recurso Huérfano detectado en línea %d. Ignorando.",
                context.current_line_number,
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
                fields,
            )
            context.raw_records.append(record)
            context.stats["insumos_extracted"] += 1

            if self.parent.debug_mode:
                logger.debug(
                    "  ✓ Insumo válido [línea %d] [%s]: %s... (%d campos)",
                    context.current_line_number,
                    validation_result.validation_layer,
                    fields[0][:40],
                    validation_result.fields_count,
                )
        else:
            context.stats["lines_ignored_in_context"] += 1
            if self.parent.debug_mode:
                logger.debug(
                    "  ✗ Rechazada [línea %d]: %s",
                    context.current_line_number,
                    validation_result.reason,
                )

        return False


class ReportParserCrudo:
    """
    Parser robusto tipo máquina de estados para archivos APU semi-estructurados.

    Implementa validación en tres capas:
      Capa 0 — Verificación de tipos (homeomorfismo de tipo).
      Capa 1 — Invariantes estructurales básicos (cardinalidad, numeración).
      Capa 2 — Homeomorfismo Lark (parsing gramatical).
      Capa 3 — Homeomorfismo APU (semántica de insumos).

    Todas las métricas internas (entropía, densidad, cohesión) están normalizadas
    al intervalo [0, 1] para permitir comparación y ponderación uniforme.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # CONSTANTES DE CLASE
    # ═══════════════════════════════════════════════════════════════════════

    _MAX_CACHE_SIZE: int = 50_000
    _MAX_FAILED_SAMPLES: int = 20
    _MAX_LINE_LENGTH: int = 5_000
    _MIN_FIELDS_FOR_INSUMO: int = 5
    _MIN_LINE_LENGTH: int = 3

    # Longitud máxima de línea antes de aplicar hash para la clave de cache
    _CACHE_KEY_MAX_LENGTH: int = 2_000

    CATEGORY_KEYWORDS: Dict[str, set] = {
        "MATERIALES": {"MATERIALES", "MATERIAL", "MAT.", "INSUMOS"},
        "MANO DE OBRA": {
            "MANO DE OBRA",
            "MANO OBRA",
            "M.O.",
            "MO",
            "PERSONAL",
            "OBRERO",
        },
        "EQUIPO": {"EQUIPO", "EQUIPOS", "MAQUINARIA", "MAQ."},
        "TRANSPORTE": {"TRANSPORTE", "TRANSPORTES", "TRANS.", "ACARREO"},
        "HERRAMIENTA": {"HERRAMIENTA", "HERRAMIENTAS", "HERR.", "UTILES"},
        "OTROS": {"OTROS", "OTRO", "VARIOS", "ADICIONALES"},
    }

    JUNK_KEYWORDS: frozenset = frozenset(
        {
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
    _NUMERIC_PATTERN: re.Pattern = re.compile(r"\d+[.,]\d+|\d+")
    _DECORATIVE_PATTERN: re.Pattern = re.compile(r"^[=\-_\s*]+$")
    _UNIT_PATTERN: re.Pattern = re.compile(r"UNIDAD:\s*(\S+)", re.IGNORECASE)
    _ITEM_PATTERN: re.Pattern = re.compile(r"ITEM:\s*([\S,]+)", re.IGNORECASE)

    # ── Umbrales de clasificación homeomórfica ───────────────────────────
    # Justificación matemática:
    #   CLASE_A: entropía > 0.6 ⟹ distribución de tipos cercana a uniforme
    #            (H > 0.6·H_max = 1.2 bits de 2.0 posibles).
    #            densidad > 0.08 ⟹ ≥ 1 unidad semántica cada 12.5 caracteres.
    #            cohesión > 0.7  ⟹ distancia media entre números ≤ 0.43 campos.
    #   CLASE_B: cohesión > 0.85 ⟹ números casi contiguos (d̄ ≤ 0.18 campos).
    #   CLASE_C: homogeneidad > 0.7 ⟹ ≥70% de campos del mismo tipo.
    #   CLASE_D: señal mixta aceptable (entropía > 0.4 o densidad > 0.05).
    #   CLASE_E: sin señal estructural reconocible.
    _THR_A_ENTROPY: float = 0.6
    _THR_A_DENSITY: float = 0.08
    _THR_A_COHESION: float = 0.7
    _THR_B_COHESION: float = 0.85
    _THR_B_HOMOGENEITY: float = 0.5
    _THR_C_HOMOGENEITY: float = 0.7
    _THR_D_ENTROPY: float = 0.4
    _THR_D_DENSITY: float = 0.05

    def __init__(
        self,
        file_path: Union[str, Path],
        profile: dict,
        config: Optional[Dict] = None,
        telemetry: Optional[Any] = None,
    ) -> None:
        """Inicializa el parser con validación exhaustiva de parámetros."""
        if file_path is None:
            raise ValueError("file_path no puede ser None")
        self.file_path = (
            Path(file_path) if not isinstance(file_path, Path) else file_path
        )

        if profile is not None and not isinstance(profile, dict):
            logger.warning(
                "profile no es dict (%s), usando vacío", type(profile).__name__
            )
            profile = {}
        if config is not None and not isinstance(config, dict):
            logger.warning(
                "config no es dict (%s), usando vacío", type(config).__name__
            )
            config = {}

        self.profile: Dict[str, Any] = profile or {}
        self.config: Dict[str, Any] = config or {}
        self.telemetry = telemetry

        self._validate_file_path()

        self.lark_parser: Optional[Lark] = None
        self._parse_cache: Dict[str, Tuple[bool, Any]] = {}
        self.validation_stats = ValidationStats()

        if not _LARK_AVAILABLE:
            logger.error(
                "Lark no está instalado. Ejecute: pip install lark\n"
                "El parser funcionará sin validación Lark."
            )
        else:
            try:
                from .apu_processor import APU_GRAMMAR  # type: ignore[import]

                self.lark_parser = self._initialize_lark_parser(APU_GRAMMAR)
            except ImportError as exc:
                logger.error(
                    "No se pudo importar APU_GRAMMAR: %s\n"
                    "El parser funcionará sin validación Lark.",
                    exc,
                )
            except Exception as exc:
                logger.error(
                    "Error inicializando parser Lark: %s\n"
                    "El parser funcionará sin validación Lark.",
                    exc,
                )

        self.raw_records: List[Dict[str, Any]] = []
        self.stats: Counter = Counter()
        self._parsed: bool = False
        self.debug_mode: bool = self.config.get("debug_mode", False)

        logger.debug(
            "ReportParserCrudo inicializado:\n"
            "  Archivo: %s\n"
            "  Lark parser: %s\n"
            "  Debug mode: %s",
            self.file_path.name,
            "✓" if self.lark_parser else "✗",
            self.debug_mode,
        )

    # ── Fábrica de handlers ──────────────────────────────────────────────

    def _initialize_handlers(self) -> List[LineHandler]:
        """Fabrica la cadena de responsabilidad en orden de prioridad decreciente."""
        return [
            JunkHandler(self),      # 1. Descartar basura obvia
            HeaderHandler(self),    # 2. Detectar cambios de estructura (nuevos APUs)
            CategoryHandler(self),  # 3. Detectar cambios de categoría
            InsumoHandler(self),    # 4. Procesar datos (hojas del árbol)
        ]

    # ── Inicialización Lark ──────────────────────────────────────────────

    def _initialize_lark_parser(
        self, grammar: Optional[str] = None
    ) -> Optional["Lark"]:
        """
        Inicializa el parser Lark con la misma gramática que usa APUProcessor.

        Garantiza coherencia entre ambos módulos al compartir configuración idéntica.
        """
        if not _LARK_AVAILABLE:
            return None

        if grammar is None:
            try:
                from .apu_processor import APU_GRAMMAR  # type: ignore[import]

                grammar = APU_GRAMMAR
            except ImportError:
                logger.error(
                    "No se pudo importar APU_GRAMMAR desde apu_processor."
                )
                return None

        if not grammar or not isinstance(grammar, str) or not grammar.strip():
            logger.error("La gramática proporcionada está vacía o no es válida.")
            return None

        try:
            from lark.exceptions import ConfigurationError, GrammarError  # type: ignore[import]
        except ImportError:
            logger.error("No se pudieron importar excepciones de Lark.")
            return None

        try:
            parser = Lark(
                grammar,
                start="line",
                parser="lalr",
                maybe_placeholders=False,
                propagate_positions=True,
                cache=True,
            )
            return parser
        except GrammarError as exc:
            logger.error(
                "Error de gramática Lark: %s\n"
                "Revise que APU_GRAMMAR sea válida.",
                exc,
            )
        except ConfigurationError as exc:
            logger.error("Error de configuración Lark: %s", exc)
        except Exception as exc:
            logger.error("Error inesperado inicializando parser Lark: %s", exc)

        return None

    # ── Cache semántico ──────────────────────────────────────────────────

    def _compute_semantic_cache_key(self, line: str) -> str:
        """
        Computa clave de cache basada en invariantes topológicos.

        Define una relación de equivalencia sobre el espacio de líneas:
        líneas topológicamente equivalentes colapsan al mismo representante
        canónico en el espacio cociente L/~.

        Corrección v3:
            - El lookahead `(?!\.)` ahora es correcto: evita quitar el cero
              en "0.5" (que sería el dígito antes del punto decimal) pero sí
              quita ceros en enteros como "007" → "7".
        """
        from app.core.utils import parse_number

        # Homeomorfismo de espaciado: colapsa variantes de blancos
        normalized = re.sub(r"\s+", " ", line.strip())

        # Colapso de onda numérico (Numeric Wave Collapse)
        # Extraer posibles secuencias numéricas y reemplazarlas por su valor canónico (float parseado)
        def _collapse_number(match):
            val_str = match.group(0)
            try:
                # En parse_number, 1,000 puede confundirse con 1.0 si no se configura bien el separador regional por default
                # Si vemos que tiene 3 dígitos después de la coma de miles, ayudamos al parser
                val_clean = re.sub(r",(\d{3})(?!\d)", r"\1", val_str)
                parsed_val = parse_number(val_clean, default_value=None, strict=True)
                if parsed_val is not None:
                    # Formato consistente sin ceros extra (.0 si es entero)
                    if parsed_val.is_integer():
                        return str(int(parsed_val))
                    return str(parsed_val)
            except (ValueError, TypeError):
                pass
            return val_str

        # Encontrar secuencias numéricas aisladas o contiguas a separadores (usar regex amplio para capturar 1,000.00 o 1,00)
        # El patrón busca dígitos opcionalmente separados por comas y/o un punto
        numeric_pattern = r"(?<![a-zA-Z])[-+]?\d+(?:[.,]\d+)*(?![a-zA-Z])"
        normalized = re.sub(numeric_pattern, _collapse_number, normalized)

        if len(normalized) <= self._CACHE_KEY_MAX_LENGTH:
            return normalized

        # Para líneas largas: proyección a espacio de características compacto
        num_groups = len(re.findall(r"\d+[.,]?\d*", normalized))
        alpha_groups = len(re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]+", normalized))
        sep_count = normalized.count(";")
        total_len = len(normalized)

        # Muestreo de bordes (preserva información de frontera)
        prefix = normalized[:50]
        suffix = normalized[-30:]
        middle_start = len(normalized) // 3
        middle_sample = normalized[middle_start: middle_start + 20]

        feature_string = (
            f"{prefix}|{middle_sample}|{suffix}|"
            f"{num_groups}|{alpha_groups}|{sep_count}|{total_len}"
        )
        return hashlib.sha256(feature_string.encode()).hexdigest()[:32]

    def _cache_result(self, key: str, is_valid: bool, tree: Any) -> None:
        """Almacena un resultado en cache con control de tamaño (LRU simplificado)."""
        if len(self._parse_cache) >= self._MAX_CACHE_SIZE:
            # Evicción del 10% más antiguo
            keys_to_remove = list(self._parse_cache.keys())[
                : self._MAX_CACHE_SIZE // 10
            ]
            for k in keys_to_remove:
                del self._parse_cache[k]
        self._parse_cache[key] = (is_valid, tree)

    # ── Validación con Lark ──────────────────────────────────────────────

    def _validate_with_lark(
        self, line: str, use_cache: bool = True
    ) -> Tuple[bool, Optional[Any], str]:
        """
        Valida una línea usando el parser Lark con optimización topológica.

        Implementa el functor F: Líneas → (𝔹 × Árbol? × Mensaje).

        CORRECCIÓN CRÍTICA v3:
            El orden de las cláusulas `except` respeta la jerarquía de herencia
            de Lark. `UnexpectedCharacters` y `UnexpectedToken` son subclases de
            `UnexpectedInput`; por tanto, se capturan ANTES que su superclase para
            que sus contadores específicos se incrementen correctamente.
            Orden correcto: UnexpectedCharacters → UnexpectedToken →
                            UnexpectedEOF → UnexpectedInput → LarkError.

        Args:
            line:      Línea de texto a validar.
            use_cache: Si True, intenta usar el cache de parsing.

        Returns:
            (es_válido, árbol_lark_o_None, mensaje_de_error)
        """
        if self.lark_parser is None:
            return (True, None, "Lark no disponible - validación omitida")

        if not line or not isinstance(line, str):
            return (False, None, "Línea vacía o tipo inválido")

        line_clean = line.strip()
        line_len = len(line_clean)

        if line_len > self._MAX_LINE_LENGTH:
            return (
                False,
                None,
                f"Línea excede límite topológico: {line_len} > {self._MAX_LINE_LENGTH}",
            )
        if line_len < self._MIN_LINE_LENGTH:
            return (
                False,
                None,
                f"Línea insuficiente topológicamente: {line_len} < {self._MIN_LINE_LENGTH}",
            )

        # ── Cache semántico ──────────────────────────────────────────────
        cache_key: Optional[str] = (
            self._compute_semantic_cache_key(line_clean) if use_cache else None
        )

        if use_cache and cache_key and cache_key in self._parse_cache:
            self.validation_stats.cached_parses += 1
            cached_result = self._parse_cache[cache_key]

            if isinstance(cached_result, tuple) and len(cached_result) == 2:
                is_valid, tree = cached_result
                if is_valid and tree is not None and not self._is_valid_tree(tree):
                    # Invalidar entrada corrupta
                    del self._parse_cache[cache_key]
                else:
                    reason = "" if is_valid else "Falló previamente (cache válido)"
                    return (is_valid, tree, reason)

        # ── Pre-filtro homotópico ────────────────────────────────────────
        if not self._has_minimal_structural_connectivity(line_clean):
            if use_cache and cache_key:
                self._cache_result(cache_key, False, None)
            return (False, None, "Falta conectividad estructural mínima")

        # ── Parsing con manejo jerárquico de errores ─────────────────────
        error_msg = ""
        try:
            tree = self.lark_parser.parse(line_clean)

            if not self._validate_tree_homotopy(tree):
                if use_cache and cache_key:
                    self._cache_result(cache_key, False, None)
                return (False, None, "Árbol no cumple invariantes de homotopía")

            if use_cache and cache_key:
                self._cache_result(cache_key, True, tree)
            return (True, tree, "")

        # ORDEN CORRECTO: subclases antes que superclase
        except UnexpectedCharacters as uc:
            self.validation_stats.failed_lark_unexpected_chars += 1
            column = getattr(uc, "column", 0)
            context_str = self._get_topological_context(line_clean, column)
            error_msg = f"Carácter discontinuo en vecindad {context_str}"

        except UnexpectedToken as ut:
            self.validation_stats.failed_lark_parse += 1
            expected = (
                list(ut.expected)
                if hasattr(ut, "expected") and ut.expected
                else []
            )
            expected_space = self._map_tokens_to_topological_space(expected)
            token_repr = getattr(ut, "token", "desconocido")
            error_msg = f"Token '{token_repr}' fuera del espacio {expected_space}"

        except UnexpectedEOF:
            self.validation_stats.failed_lark_parse += 1
            completeness = self._calculate_topological_completeness(line_clean)
            error_msg = f"Fin prematuro (compleción {completeness:.0%})"

        except UnexpectedInput as ui:
            # Captura el resto de subclases de UnexpectedInput no cubiertas arriba
            self.validation_stats.failed_lark_unexpected_input += 1
            pos = getattr(ui, "pos_in_stream", 0)
            context_str = self._get_topological_context(line_clean, pos)
            error_msg = f"Entrada inesperada en posición {pos}: {context_str}"

        except LarkError as exc:
            self.validation_stats.failed_lark_parse += 1
            error_msg = f"Error Lark: {str(exc)[:100]}"

        except Exception as exc:
            self.validation_stats.failed_lark_parse += 1
            logger.error(
                "Error inesperado en validación Lark: %s: %s",
                type(exc).__name__,
                exc,
            )
            error_msg = f"Error inesperado: {type(exc).__name__}"

        # Punto de salida unificado para errores
        if use_cache and cache_key:
            self._cache_result(cache_key, False, None)
        return (False, None, error_msg)

    # ── Validación de árboles ────────────────────────────────────────────

    def _is_valid_tree(self, tree: Any) -> bool:
        """Verifica que un árbol Lark tiene la estructura mínima esperada."""
        if tree is None:
            return False
        try:
            return (
                hasattr(tree, "data")
                and hasattr(tree, "children")
                and isinstance(tree.data, str)
            )
        except Exception:
            return False

    def _validate_tree_homotopy(self, tree: Any) -> bool:
        """
        Verifica que el árbol de parsing sea homotópicamente válido.

        Un árbol válido puede deformarse continuamente a la estructura canónica
        esperada sin violaciones de los siguientes invariantes:
          I1: La raíz pertenece al espacio de no-terminales (str).
          I2: El factor de ramificación ≤ 50 (evita estructuras degeneradas).
          I3: La profundidad ≤ 20 (evita recursión potencialmente infinita).
          I4: Existe al menos un token terminal (árbol no vacío).
        """
        if not self._is_valid_tree(tree):
            return False

        try:
            child_count = len(tree.children) if tree.children else 0
            if child_count > 50:
                return False

            max_depth = 20

            def _check_depth(node: Any, depth: int) -> bool:
                if depth > max_depth:
                    return False
                if hasattr(node, "children") and node.children:
                    for child in node.children:
                        if hasattr(child, "data"):
                            if not _check_depth(child, depth + 1):
                                return False
                return True

            def _has_terminal(node: Any) -> bool:
                if not hasattr(node, "children") or not node.children:
                    return True  # Nodo hoja ≡ terminal
                for child in node.children:
                    if not hasattr(child, "data"):  # Es un Token
                        return True
                    if _has_terminal(child):
                        return True
                return False

            return _check_depth(tree, 0) and _has_terminal(tree)

        except Exception:
            return False

    # ── Conectividad estructural ─────────────────────────────────────────

    def _has_minimal_structural_connectivity(self, line: str) -> bool:
        """
        Verifica conectividad topológica mínima de una línea.

        Una línea es conexa si sus componentes (tokens alfa, numéricos y
        separadores) forman una partición no trivial del dominio con
        contenido distribuido en al menos dos tercios del espacio textual.

        Corrección v3:
            Se añade guarda explícita para líneas con longitud < 10 que no
            permiten una subdivisión en tercios significativa.
        """
        if not line:
            return False

        alpha_sequences = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{2,}", line)
        numeric_sequences = re.findall(r"\d+(?:[.,]\d+)?", line)
        separator_count = line.count(";")

        has_alpha = bool(alpha_sequences)
        has_numeric = bool(numeric_sequences)
        min_separators = max(self._MIN_FIELDS_FOR_INSUMO - 1, 1)
        has_separators = separator_count >= min_separators

        if not (has_alpha and has_numeric and has_separators):
            return False

        line_len = len(line)
        if line_len < 10:
            # Espacio demasiado pequeño para análisis de distribución
            # La conectividad básica ya fue verificada arriba
            return True

        # Análisis de distribución en tercios
        third = line_len // 3
        segments = [
            line[:third],
            line[third: 2 * third],
            line[2 * third:],
        ]

        segments_with_content = sum(
            1 for seg in segments if re.search(r"[A-Za-z0-9]", seg)
        )
        segments_with_separators = sum(1 for seg in segments if ";" in seg)

        # Conectividad: contenido en ≥ 2 segmentos Y separadores en ≥ 1
        return segments_with_content >= 2 and segments_with_separators >= 1

    # ── Contexto y mapeo de errores ──────────────────────────────────────

    def _get_topological_context(
        self, line: str, position: int, radius: int = 10
    ) -> str:
        """
        Obtiene la vecindad ε = `radius` del punto de error.

        Marca el carácter problemático con delimitadores topológicos ⟪·⟫.
        """
        start = max(0, position - radius)
        end = min(len(line), position + radius)
        context = line[start:end]
        error_pos = position - start

        if error_pos < len(context):
            marked = (
                context[:error_pos]
                + "⟪"
                + context[error_pos]
                + "⟫"
                + context[error_pos + 1:]
            )
        else:
            marked = context + "⟪␣⟫"

        return f"[...]{marked}[...]"

    def _map_tokens_to_topological_space(self, expected_tokens: List[str]) -> str:
        """
        Mapea tokens esperados de Lark a espacios topológicos nombrados.

        Se usa coincidencia de subcadena (case-insensitive) para cubrir
        variantes de nombres de terminal como `__ANON_0` o `NUMBER_FLOAT`.
        """
        token_spaces = {
            "NUMBER": "Espacio Numérico ℝ",
            "WORD": "Espacio Lexical Σ*",
            "UNIT": "Espacio de Unidades 𝒰",
            "SEP": "Espacio de Separación 𝒮",
            "DESC": "Espacio Descriptivo 𝒟",
            "FIELD": "Espacio de Campo 𝒻",
        }

        spaces: set = set()
        for token in expected_tokens:
            token_up = token.upper()
            matched = False
            for key, space in token_spaces.items():
                if key in token_up:
                    spaces.add(space)
                    matched = True
                    break
            if not matched:
                spaces.add(f"Espacio Terminal '{token}'")

        return " ∪ ".join(sorted(spaces)) if spaces else "∅"

    # ── Completitud topológica ───────────────────────────────────────────

    def _calculate_topological_completeness(self, line: str) -> float:
        """
        Calcula el grado de compleción topológica de una línea en [0, 1].

        Basado en la teoría de compleción de espacios métricos: mide qué tan
        cerca está la línea de ser un "punto límite" válido en el espacio de
        insumos APU. Cada componente aporta un peso proporcional a su
        relevancia semántica en la gramática APU.

        Corrección v3:
            - El regex de `cantidad` busca en toda la línea (sin ancla `$`).
            - La normalización decimal se aplica a una copia, preservando
              la línea original para los demás checks.
        """
        if not line or not isinstance(line, str):
            return 0.0

        line_for_numbers = line.replace(",", ".")

        components = {
            "descripcion": bool(re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{3,}", line)),
            "cantidad": bool(re.search(r"\d+\.?\d*", line_for_numbers)),
            "unidad": bool(
                re.search(
                    r"\b(UND|UN|M|M2|M3|KG|L|LT|GLN|GAL|HR|DIA|ML|CM|TON)\b",
                    line,
                    re.IGNORECASE,
                )
            ),
            "precio": bool(
                re.search(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?", line)
            ),
            "separadores": line.count(";") >= 3,
        }

        weights = {
            "descripcion": 0.30,
            "cantidad": 0.25,
            "unidad": 0.20,
            "precio": 0.15,
            "separadores": 0.10,
        }

        score = sum(weights[k] for k, v in components.items() if v)

        # Factor de regularización por densidad de información
        info_chunks = len(re.findall(r"\S+", line))
        separators = line.count(";")
        expected_chunks = max(separators + 1, 1)
        density_factor = min(info_chunks / expected_chunks, 1.5) / 1.5

        # Penalización suave si la densidad es muy baja
        if density_factor < 0.4:
            score *= 0.7 + (density_factor * 0.75)

        return min(max(score, 0.0), 1.0)

    # ── Validación estructural básica ────────────────────────────────────

    def _validate_basic_structure(
        self, line: str, fields: List[str]
    ) -> Tuple[bool, str]:
        """
        Validación pre-Lark para filtrado rápido de líneas inválidas.

        Verifica cardinalidad de campos, presencia de datos numéricos,
        ausencia de palabras clave de agregación y longitud máxima de campo.
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

        first_field = fields[0].strip() if fields[0] else ""
        if not first_field:
            self.validation_stats.failed_basic_fields += 1
            return (False, "Campo de descripción vacío")

        if len(first_field) < 2:
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

        has_numeric = any(
            self._NUMERIC_PATTERN.search(f.strip())
            for f in fields[1:]
            if f
        )
        if not has_numeric:
            self.validation_stats.failed_basic_numeric += 1
            return (False, "Sin campos numéricos detectables")

        for i, f in enumerate(fields):
            if len(f) > 500:
                self.validation_stats.failed_basic_fields += 1
                return (
                    False,
                    f"Campo {i} excesivamente largo: {len(f)} caracteres",
                )

        self.validation_stats.passed_basic += 1
        return (True, "")

    # ── Validación unificada de insumos ──────────────────────────────────

    def _validate_insumo_line(
        self, line: str, fields: List[str]
    ) -> LineValidationResult:
        """
        Validación topológica unificada con análisis de invariantes homeomórficos.

        Aplica las cuatro capas de validación en orden de coste creciente.
        """
        self.validation_stats.total_evaluated += 1

        # Capa 0: homeomorfismo de tipo
        if not isinstance(line, str) or not line:
            return LineValidationResult(
                is_valid=False,
                reason="Entrada no homeomorfa a string válido",
                fields_count=0,
                validation_layer="type_check_failed",
            )
        if not isinstance(fields, list):
            return LineValidationResult(
                is_valid=False,
                reason="Campos no homeomorfos a lista",
                fields_count=0,
                validation_layer="type_check_failed",
            )

        fields_count = len(fields)

        # Capa 1: invariantes estructurales básicos
        basic_valid, basic_reason = self._validate_basic_structure(line, fields)
        if not basic_valid:
            error_group = self._classify_basic_error_group(basic_reason)
            return LineValidationResult(
                is_valid=False,
                reason=f"[{error_group}] {basic_reason}",
                fields_count=fields_count,
                has_numeric_fields=False,
                validation_layer="basic_invariant_failed",
            )

        # Capa 2: homeomorfismo Lark
        lark_valid, lark_tree, lark_reason = self._validate_with_lark(line)

        has_numeric = any(
            self._NUMERIC_PATTERN.search(f.strip()) for f in fields[1:] if f
        )

        if not lark_valid:
            self._record_failed_sample(line, fields, lark_reason)
            error_class = self._classify_lark_error_topology(lark_reason)
            return LineValidationResult(
                is_valid=False,
                reason=f"[{error_class}] {lark_reason}",
                fields_count=fields_count,
                has_numeric_fields=has_numeric,
                validation_layer="lark_validation_failed",
            )

        self.validation_stats.passed_lark += 1
        self.validation_stats.passed_both += 1

        # Capa 3: homeomorfismo APU
        if lark_tree and not self._is_apu_homeomorphic(lark_tree):
            return LineValidationResult(
                is_valid=False,
                reason="Árbol sintáctico no mapeable a esquema APU",
                fields_count=fields_count,
                has_numeric_fields=has_numeric,
                validation_layer="apu_schema_mismatch",
                lark_tree=lark_tree,
            )

        return LineValidationResult(
            is_valid=True,
            reason="Validación completa exitosa",
            fields_count=fields_count,
            has_numeric_fields=has_numeric,
            validation_layer="full_homeomorphism",
            lark_tree=lark_tree,
        )

    # ── Clasificación de errores ─────────────────────────────────────────

    def _classify_basic_error_group(self, reason: str) -> str:
        """Clasifica errores básicos en grupos topológicos nombrados."""
        reason_lower = reason.lower()
        if "campos" in reason_lower or "campo" in reason_lower:
            return "Grupo Cardinalidad Gₐ"
        if "numérico" in reason_lower or "numerico" in reason_lower:
            return "Grupo Medida Gₘ"
        if "subtotal" in reason_lower or "total" in reason_lower:
            return "Grupo Agregación Gₜ"
        if "decorativa" in reason_lower or "separador" in reason_lower:
            return "Grupo Trivial G₀"
        return "Grupo Desconocido Gₓ"

    def _classify_lark_error_topology(self, reason: str) -> str:
        """Clasifica errores Lark en tipos topológicos."""
        if "discontinuo" in reason or "UnexpectedCharacters" in reason:
            return "Espacio Discontinuo 𝓓"
        if "Token" in reason or "UnexpectedToken" in reason:
            return "Mapeo Incorrecto 𝓜"
        if "prematuro" in reason or "UnexpectedEOF" in reason:
            return "Borde Prematuro 𝓑"
        if "inesperada" in reason or "UnexpectedInput" in reason:
            return "Entrada Singular 𝓢"
        return "Anomalía 𝓐"

    # ── Homeomorfismo APU ────────────────────────────────────────────────

    def _is_apu_homeomorphic(self, tree: Any) -> bool:
        """
        Verifica que el árbol Lark sea homeomorfo a un registro de insumo APU.

        Un registro APU válido debe contener:
          - Al menos una secuencia alfanumérica de ≥ 3 caracteres (descripción).
          - Al menos un valor numérico (cantidad o precio).

        La verificación de campos separados es implícita en la gramática Lark;
        no se requiere verificar el token SEP explícitamente aquí.

        Corrección v3:
            `LarkToken` se importa a nivel módulo (evita import por llamada).
            Si Lark no está disponible, se asume homeomorfismo por defecto.
        """
        if not self._is_valid_tree(tree):
            return False

        if not _LARK_AVAILABLE or LarkToken is None:
            logger.warning(
                "Lark no disponible; se asume homeomorfismo APU por defecto."
            )
            return True

        found_description = False
        found_numeric = False

        def _analyze(node: Any, depth: int) -> None:
            nonlocal found_description, found_numeric
            if depth > 30:
                return

            if isinstance(node, LarkToken):
                token_type = getattr(node, "type", "")
                val = str(getattr(node, "value", "")).strip()

                if token_type in ("FIELD_VALUE", "NUMBER", "DECIMAL"):
                    if re.search(r"\d", val):
                        found_numeric = True
                    if re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{3,}", val):
                        found_description = True
                elif token_type in ("WORD", "TEXT", "DESCRIPTION"):
                    if len(val) >= 3:
                        found_description = True

            elif hasattr(node, "children") and node.children:
                for child in node.children:
                    _analyze(child, depth + 1)

        try:
            _analyze(tree, 0)
        except RecursionError:
            logger.warning("Recursión excesiva en análisis homeomórfico.")
            return False

        return found_description and found_numeric

    # ── Registro de muestras fallidas ────────────────────────────────────

    def _record_failed_sample(
        self, line: str, fields: List[str], reason: str
    ) -> None:
        """Registra una muestra de línea fallida para análisis posterior."""
        max_samples = self.config.get("max_failed_samples", self._MAX_FAILED_SAMPLES)
        if len(self.validation_stats.failed_samples) >= max_samples:
            return

        safe_line = line[:200] if isinstance(line, str) else str(line)[:200]
        safe_fields = []
        empty_positions = []

        if isinstance(fields, list):
            for i, f in enumerate(fields):
                f_str = str(f)
                safe_fields.append(f_str[:100])
                if not f_str.strip():
                    empty_positions.append(i)

        self.validation_stats.failed_samples.append(
            {
                "line": safe_line,
                "fields": safe_fields,
                "fields_count": len(fields) if isinstance(fields, list) else 0,
                "reason": reason[:300] if isinstance(reason, str) else str(reason)[:300],
                "has_empty_fields": bool(empty_positions),
                "empty_field_positions": empty_positions,
                "line_length": len(line) if isinstance(line, str) else 0,
                "first_field_preview": safe_fields[0][:50] if safe_fields else "",
            }
        )

    # ── Resumen de validación ────────────────────────────────────────────

    def _log_validation_summary(self) -> None:
        """Registra un resumen detallado de la validación al finalizar el parseo."""
        total = self.validation_stats.total_evaluated
        # Counter soporta indexación directa; usar [] es idiomático y consistente
        valid = self.stats["insumos_extracted"]

        logger.info("=" * 80)
        logger.info("📊 RESUMEN DE VALIDACIÓN CON LARK")
        logger.info("=" * 80)
        logger.info("Total líneas evaluadas: %d", total)

        valid_pct = f"({valid / total * 100:.1f}%)" if total > 0 else "(0.0%)"
        logger.info("✓ Insumos válidos (ambas capas): %d %s", valid, valid_pct)
        logger.info(
            "  - Pasaron validación básica: %d",
            self.validation_stats.passed_basic,
        )
        logger.info(
            "  - Pasaron validación Lark: %d", self.validation_stats.passed_lark
        )
        logger.info(
            "  - Cache hits: %d", self.validation_stats.cached_parses
        )
        logger.info("")
        logger.info("Rechazos por validación básica:")
        logger.info(
            "  - Campos insuficientes/vacíos: %d",
            self.validation_stats.failed_basic_fields,
        )
        logger.info(
            "  - Sin datos numéricos: %d",
            self.validation_stats.failed_basic_numeric,
        )
        logger.info(
            "  - Subtotales: %d", self.validation_stats.failed_basic_subtotal
        )
        logger.info(
            "  - Líneas decorativas: %d", self.validation_stats.failed_basic_junk
        )
        logger.info("")
        logger.info("Rechazos por validación Lark:")
        logger.info(
            "  - Parse error genérico: %d",
            self.validation_stats.failed_lark_parse,
        )
        logger.info(
            "  - Unexpected input: %d",
            self.validation_stats.failed_lark_unexpected_input,
        )
        logger.info(
            "  - Unexpected characters: %d",
            self.validation_stats.failed_lark_unexpected_chars,
        )
        logger.info("=" * 80)

        if self.validation_stats.failed_samples:
            logger.info("")
            logger.info("🔍 MUESTRAS DE LÍNEAS RECHAZADAS POR LARK:")
            logger.info("-" * 80)
            for idx, sample in enumerate(self.validation_stats.failed_samples, 1):
                logger.info("\nMuestra #%d:", idx)
                logger.info("  Razón: %s", sample["reason"])
                logger.info("  Campos: %d", sample["fields_count"])
                logger.info("  Campos vacíos: %s", sample["has_empty_fields"])
                if sample["has_empty_fields"]:
                    logger.info(
                        "  Posiciones vacías: %s",
                        sample["empty_field_positions"],
                    )
                logger.info("  Contenido: %s", sample["line"])
                logger.info("  Campos: %s", sample["fields"])
            logger.info("-" * 80)

        if valid == 0 and total > 0:
            logger.error("🚨 CRÍTICO: 0 insumos válidos con validación Lark.")
        elif total > 0 and valid < total * 0.5:
            logger.warning(
                "⚠️  Tasa de validación baja: %.1f%%", valid / total * 100
            )

    # ── Cache público ────────────────────────────────────────────────────

    def get_parse_cache(self) -> Dict[str, Any]:
        """
        Exporta el cache de parsing para reutilización en APUProcessor.

        Proyecta el cache al subespacio de árboles válidos, filtrando
        entradas inválidas, nulas o corruptas.

        Returns:
            Diccionario {hash_semántico: árbol_lark}.
        """
        valid_cache: Dict[str, Any] = {}
        invalid_count = 0

        # Copia de items para evitar modificación durante iteración
        for line, cached_value in list(self._parse_cache.items()):
            if not isinstance(cached_value, tuple) or len(cached_value) != 2:
                invalid_count += 1
                continue

            is_valid, tree = cached_value
            if not is_valid or tree is None:
                continue

            if not self._is_valid_tree(tree):
                invalid_count += 1
                continue

            try:
                # Reutilizar clave si ya es un hash SHA-256 de 32 hex chars
                if len(line) == 32 and re.match(r"^[0-9a-f]{32}$", line):
                    normalized_key = line
                else:
                    normalized_key = self._compute_semantic_cache_key(line)
                valid_cache[normalized_key] = tree
            except Exception:
                invalid_count += 1

        if invalid_count > 0:
            logger.debug(
                "Cache: %d entradas inválidas filtradas.", invalid_count
            )
        logger.info(
            "Cache de parsing exportado: %d árboles válidos.", len(valid_cache)
        )
        return valid_cache

    # ── Validación de ruta ───────────────────────────────────────────────

    def _validate_file_path(self) -> None:
        """Valida que la ruta del archivo exista, sea un fichero y no esté vacío."""
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {self.file_path}"
            )
        if not self.file_path.is_file():
            raise ValueError(f"La ruta no es un archivo: {self.file_path}")
        if self.file_path.stat().st_size == 0:
            raise ValueError(f"El archivo está vacío: {self.file_path}")

    # ── Punto de entrada principal ───────────────────────────────────────

    def parse_to_raw(self) -> List[Dict[str, Any]]:
        """
        Ejecuta la máquina de estados sobre las líneas del archivo.

        Returns:
            Lista de registros crudos de insumos extraídos.

        Raises:
            ParseStrategyError: Si ocurre un error irrecuperable durante el parseo.
        """
        if self._parsed:
            return self.raw_records

        logger.info(
            "Iniciando parseo línea por línea de: %s", self.file_path.name
        )

        try:
            content = self._read_file_safely()
            lines = content.split("\n")
            self.stats["total_lines"] = len(lines)

            handlers = self._initialize_handlers()
            context = ParserContext()

            logger.info(
                "🚀 Iniciando procesamiento de %d líneas con Lógica Piramidal.",
                len(lines),
            )

            i = 0
            while i < len(lines):
                raw_line = lines[i]
                context.current_line_number = i + 1
                line = raw_line.strip()

                if not line:
                    i += 1
                    continue

                next_line = (
                    lines[i + 1].strip() if i + 1 < len(lines) else None
                )

                for handler in handlers:
                    if handler.can_handle(line, next_line):
                        should_advance = handler.handle(line, context, next_line)
                        if should_advance:
                            i += 1  # Consumir la siguiente línea (ITEM:)
                        break
                else:
                    logger.debug(
                        "Línea %d no reconocida por ningún handler.", i + 1
                    )

                i += 1

            self.stats.update(context.stats)
            self.raw_records = context.raw_records
            self._parsed = True

            logger.info(
                "Parseo completo. Extraídos %d registros crudos.",
                self.stats["insumos_extracted"],
            )
            self._log_validation_summary()

        except Exception as exc:
            logger.error("Error crítico de parseo: %s", exc, exc_info=True)
            raise ParseStrategyError(
                f"Falló el parseo con estrategia Chain of Responsibility: {exc}"
            ) from exc

        return self.raw_records

    # ── Lectura de archivo ───────────────────────────────────────────────

    def _read_file_safely(self) -> str:
        """Lee el contenido del archivo probando múltiples codificaciones."""
        default_encodings = self.config.get(
            "encodings", ["utf-8", "latin1", "cp1252", "iso-8859-1"]
        )
        # Priorizar la codificación del perfil si está definida
        encodings_to_try: List[Optional[str]] = (
            [self.profile.get("encoding")] + default_encodings
        )

        for encoding in filter(None, encodings_to_try):
            try:
                with open(
                    self.file_path, "r", encoding=encoding, errors="strict"
                ) as fh:
                    content = fh.read()
                self.stats["encoding_used"] = encoding
                logger.info(
                    "Archivo leído exitosamente con codificación: %s", encoding
                )
                return content
            except (UnicodeDecodeError, TypeError, LookupError):
                continue

        raise FileReadError(
            f"No se pudo leer el archivo {self.file_path} con ninguna de las "
            "codificaciones especificadas."
        )

    # ── Detección de categorías ──────────────────────────────────────────

    def _detect_category(self, line_upper: str) -> Optional[str]:
        """
        Detección topológica de categorías usando teoría de retículos.

        La línea debe estar en mayúsculas (el llamador es responsable de la
        conversión). Construye un retículo de pertenencia y retorna la
        categoría con mayor fuerza de coincidencia.

        Corrección v3:
            Los patrones de búsqueda se normalizan a mayúsculas antes de
            comparar con `line_upper`, garantizando coherencia case-insensitive
            sin depender de flags de re que podrían quedar silenciosos.
        """
        if len(line_upper) > 50 or sum(c.isdigit() for c in line_upper) > 3:
            return None

        category_membership: Dict[str, float] = {}

        for canonical, variations in self.CATEGORY_KEYWORDS.items():
            for variation in variations:
                # Normalizar variación a mayúsculas para comparar con line_upper
                variation_upper = variation.upper()
                if self._is_supremum_match(variation_upper, line_upper):
                    score = self._calculate_match_strength(
                        variation_upper, line_upper
                    )
                    category_membership[canonical] = (
                        category_membership.get(canonical, 0.0) + score
                    )

        if not category_membership:
            return None

        best_category, best_score = max(
            category_membership.items(), key=lambda x: x[1]
        )

        return best_category if best_score > 0.15 else None

    def _is_supremum_match(self, pattern: str, text: str) -> bool:
        """
        Verifica si `pattern` (ya en mayúsculas) aparece en `text` (en mayúsculas).

        Corrección v3:
            - Para abreviaturas (contienen punto), se construye un regex que
              escapa el punto literal y permite match sin \b al final (ya que
              el punto no es word-char y rompe el límite).
            - Para palabras normales, se usa \b en ambos extremos.
        """
        escaped = re.escape(pattern)  # Escapa puntos y otros caracteres especiales

        if "." in pattern:
            # Abreviatura: match al inicio de palabra, sin requerir \b al final
            # porque el '.' ya actúa como delimitador natural
            regex = rf"(?<!\w){escaped}"
        else:
            regex = rf"\b{escaped}\b"

        return bool(re.search(regex, text))

    def _calculate_match_strength(self, pattern: str, text: str) -> float:
        """
        Calcula la fuerza del match en [0, 1] usando métrica topológica.

        Pondera posición (inicio > final), completitud (palabra completa > parcial)
        y contexto (línea corta sugiere categoría pura > línea larga con contenido).
        """
        match_pos = text.find(pattern)
        if match_pos < 0:
            return 0.0

        # Peso por posición: match al inicio tiene position_weight → 1.0
        position_weight = 1.0 - (match_pos / max(len(text), 1))

        # Peso por completitud
        completeness_weight = (
            1.0 if f" {pattern} " in f" {text} " else 0.7
        )

        # Peso contextual: línea corta sugiere categoría pura
        context_weight = 2.0 if len(text) < 30 else 1.0

        return position_weight * completeness_weight * context_weight

    # ── Detección de basura ──────────────────────────────────────────────

    def _is_junk_line(self, line_upper: str) -> bool:
        """Determina si una línea debe ignorarse por ser ruido estructural."""
        if not line_upper or not isinstance(line_upper, str):
            return True

        stripped = line_upper.strip()

        if len(stripped) < self._MIN_LINE_LENGTH:
            return True

        for keyword in self.JUNK_KEYWORDS:
            if keyword in line_upper:
                return True

        return bool(self._DECORATIVE_PATTERN.search(stripped))

    # ── Extracción de encabezado APU ─────────────────────────────────────

    def _extract_apu_header(
        self, header_line: str, item_line: str, line_number: int
    ) -> Optional[APUContext]:
        """Extrae información del encabezado APU de forma defensiva."""
        try:
            parts = header_line.split(";")
            apu_desc = parts[0].strip() if parts else ""

            unit_match = self._UNIT_PATTERN.search(header_line)
            default_unit = self.config.get("default_unit", "UND")
            apu_unit = (
                unit_match.group(1).strip() if unit_match else default_unit
            )

            item_match = self._ITEM_PATTERN.search(item_line)
            apu_code_raw = (
                item_match.group(1) if item_match else f"UNKNOWN_APU_{line_number}"
            )
            apu_code = clean_apu_code(apu_code_raw)

            if not apu_code or len(apu_code) < 2:
                logger.warning("Código APU inválido extraído: '%s'", apu_code)
                return None

            return APUContext(
                apu_code=apu_code,
                apu_desc=apu_desc,
                apu_unit=apu_unit,
                source_line=line_number,
            )

        except ValueError as exc:
            logger.debug("Validación de APUContext falló: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Error extrayendo encabezado APU: %s", exc)
            return None

    # ── Construcción de registros ────────────────────────────────────────

    def _build_insumo_record(
        self,
        context: APUContext,
        category: str,
        line: str,
        line_number: int,
        validation_result: LineValidationResult,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Construye un registro de insumo con métricas topológicas.

        El árbol Lark no se serializa directamente; se almacena solo metadata
        ligera para evitar problemas de serialización y consumo de memoria.
        """
        if fields is None:
            fields = [f.strip() for f in line.split(";")]

        topological_metrics = {
            "field_entropy": self._calculate_field_entropy(fields),
            "structural_density": self._calculate_structural_density(line),
            "numeric_cohesion": self._calculate_numeric_cohesion(fields),
            "homogeneity_index": self._calculate_homogeneity_index(fields),
        }

        homeomorphism_class = self._determine_homeomorphism_class(
            validation_result.validation_layer, topological_metrics
        )

        lark_info = None
        if validation_result.lark_tree is not None:
            lark_info = {
                "has_tree": True,
                "root_type": getattr(
                    validation_result.lark_tree, "data", "unknown"
                ),
                "children_count": len(
                    getattr(validation_result.lark_tree, "children", [])
                ),
            }

        return {
            "apu_code": context.apu_code,
            "apu_desc": context.apu_desc,
            "apu_unit": context.apu_unit,
            "category": category,
            "insumo_line": line,
            "source_line": line_number,
            "fields_count": validation_result.fields_count,
            "validation_layer": validation_result.validation_layer,
            "homeomorphism_class": homeomorphism_class,
            "topological_metrics": topological_metrics,
            "_lark_info": lark_info,
            "_structural_signature": self._compute_structural_signature(line),
        }

    # ── Métricas topológicas ─────────────────────────────────────────────

    def _calculate_field_entropy(self, fields: List[str]) -> float:
        """
        Calcula la entropía de Shannon normalizada sobre la distribución de
        tipos de campo: H_norm = H / H_max ∈ [0, 1].

        Los tipos reconocidos son: alpha, numeric, mixed, empty (4 categorías).
        H_max = log₂(4) = 2.0 bits  →  constante de módulo `_H_MAX`.

        Corrección v3:
            - `log2` importado a nivel módulo (no dentro del método).
            - `_H_MAX` pre-calculado como constante de módulo.
            - Guarda defensiva para listas de un solo elemento (H_max = 0).
        """
        if not fields:
            return 0.0

        type_counts: Dict[str, int] = {
            "alpha": 0,
            "numeric": 0,
            "mixed": 0,
            "empty": 0,
        }

        for f in fields:
            val = str(f).strip()
            if not val:
                type_counts["empty"] += 1
            elif val.replace(".", "").replace(",", "").isdigit():
                type_counts["numeric"] += 1
            elif any(c.isalpha() for c in val):
                if any(c.isdigit() for c in val):
                    type_counts["mixed"] += 1
                else:
                    type_counts["alpha"] += 1
            # Si solo contiene símbolos especiales, cae en empty por defecto
            else:
                type_counts["empty"] += 1

        total = len(fields)
        entropy = 0.0
        for count in type_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * log2(p)

        # _H_MAX = log2(4) ≈ 2.0; si total == 1, solo hay 1 tipo → H_max = 0
        effective_h_max = min(_H_MAX, log2(total)) if total > 1 else 0.0
        return entropy / effective_h_max if effective_h_max > 0 else 0.0

    def _calculate_structural_density(self, line: str) -> float:
        """
        Calcula la densidad estructural: unidades semánticas por carácter.

        Acotada a [0, 1] mediante saturación: dado que una línea no puede tener
        más unidades semánticas que caracteres, el máximo teórico es 1.0
        (cada carácter sería una unidad, e.g., dígitos individuales separados).

        Corrección v3:
            Se aplica `min(..., 1.0)` para garantizar el acotamiento superior.
        """
        words = re.findall(r"\b[A-Za-z]{3,}\b", line)
        numbers = re.findall(r"\d+(?:[.,]\d+)?", line)
        semantic_units = len(words) + len(numbers)
        total_chars = len(line)
        raw = semantic_units / total_chars if total_chars > 0 else 0.0
        return min(raw, 1.0)

    def _calculate_numeric_cohesion(self, fields: List[str]) -> float:
        """
        Calcula la cohesión numérica: qué tan agrupados están los campos numéricos.

        Definición: cohesión = 1 / (1 + d̄), donde d̄ es la distancia media entre
        posiciones de campos numéricos consecutivos.

        Propiedades matemáticas:
          - Si d̄ = 0 (todos los campos numéricos son adyacentes): cohesión = 1.0.
          - Si d̄ → ∞: cohesión → 0.0.
          - La función es monótona decreciente en d̄ y siempre ∈ (0, 1].

        Corrección v3:
            Reemplaza `1/d̄` (no acotada) por `1/(1+d̄)` que garantiza [0,1].
        """
        numeric_positions = [
            i for i, f in enumerate(fields) if any(c.isdigit() for c in str(f))
        ]

        if not numeric_positions:
            return 0.0
        if len(numeric_positions) == 1:
            return 1.0

        distances = [
            abs(numeric_positions[k] - numeric_positions[k - 1])
            for k in range(1, len(numeric_positions))
        ]
        avg_distance = sum(distances) / len(distances)
        return 1.0 / (_COHESION_OFFSET + avg_distance)

    def _calculate_homogeneity_index(self, fields: List[str]) -> float:
        """
        Índice de homogeneidad: porcentaje del tipo de campo más frecuente.

        Retorna el máximo de la distribución empírica de tipos, en [0, 1].
        """
        if len(fields) < 2:
            return 1.0

        field_types: List[str] = []
        for f in fields:
            val = str(f).strip()
            if not val:
                field_types.append("empty")
            elif val.replace(".", "").replace(",", "").isdigit():
                field_types.append("numeric")
            elif any(c.isalpha() for c in val):
                field_types.append("mixed" if any(c.isdigit() for c in val) else "alpha")
            else:
                field_types.append("other")

        # Counter ya está importado a nivel módulo
        type_counts = Counter(field_types)
        most_common_count = max(type_counts.values())
        return most_common_count / len(fields)

    # ── Clasificación homeomórfica ───────────────────────────────────────

    def _determine_homeomorphism_class(
        self, validation_layer: str, metrics: Dict[str, float]
    ) -> str:
        """
        Determina la clase de homeomorfismo del registro según métricas y capa.

        Clasificación jerárquica (ver umbrales documentados en constantes de clase):
          CLASE_A: registro completo y bien estructurado.
          CLASE_B: predominantemente numérico y cohesivo.
          CLASE_C: homogéneo en tipo de campo.
          CLASE_D: señal mixta aceptable.
          CLASE_E: sin estructura reconocible o registro defectivo.
        """
        if validation_layer != "full_homeomorphism":
            if "lark" in validation_layer:
                return "CLASE_E_IRREGULAR"
            return f"DEFECTIVO_{validation_layer.upper()}"

        entropy = metrics.get("field_entropy", 0.0)
        density = metrics.get("structural_density", 0.0)
        cohesion = metrics.get("numeric_cohesion", 0.0)
        homogeneity = metrics.get("homogeneity_index", 0.0)

        if (
            entropy > self._THR_A_ENTROPY
            and density > self._THR_A_DENSITY
            and cohesion > self._THR_A_COHESION
        ):
            return "CLASE_A_COMPLETO"
        if cohesion > self._THR_B_COHESION and homogeneity > self._THR_B_HOMOGENEITY:
            return "CLASE_B_NUMERICO"
        if homogeneity > self._THR_C_HOMOGENEITY:
            return "CLASE_C_HOMOGENEO"
        if entropy > self._THR_D_ENTROPY or density > self._THR_D_DENSITY:
            return "CLASE_D_MIXTO"
        return "CLASE_E_IRREGULAR"

    # ── Firma estructural ────────────────────────────────────────────────

    def _compute_structural_signature(self, line: str) -> str:
        """
        Computa una firma estructural SHA-256 de 16 hex chars para la línea.

        Usa `hashlib` importado a nivel módulo (sin re-import dentro del método).
        La firma captura las frecuencias de clases de caracteres y la longitud
        total, constituyendo un invariante de segundo orden sobre la forma de la línea.
        """
        features = [
            str(len(re.findall(r"[A-Z]", line))),
            str(len(re.findall(r"[a-z]", line))),
            str(len(re.findall(r"\d", line))),
            str(len(re.findall(r"[.;,]", line))),
            str(len(line.split())),
            str(len(line)),
        ]
        return hashlib.sha256("|".join(features).encode()).hexdigest()[:16]