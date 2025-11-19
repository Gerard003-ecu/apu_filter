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

    Esta clase procesa un archivo línea por línea, identificando bloques que
    pertenecen a un APU específico. Utiliza un enfoque de máquina de estados
    simple:
    1. Busca un encabezado de APU (líneas con "UNIDAD:" y "ITEM:").
    2. Una vez en un contexto de APU, procesa las líneas subsecuentes como
       posibles insumos, categorías o líneas de "ruido" a ignorar.
    3. Repite el proceso hasta el final del archivo.

    El resultado es una lista de registros "crudos", donde cada registro
    contiene la línea del insumo y el contexto del APU al que pertenece.
    """

    CATEGORY_KEYWORDS = {
        "MATERIALES": {"MATERIALES", "MATERIAL", "MAT.", "INSUMOS"},
        "MANO DE OBRA": {"MANO DE OBRA", "MANO OBRA", "M.O.", "MO", "PERSONAL", "OBRERO"},
        "EQUIPO": {"EQUIPO", "EQUIPOS", "MAQUINARIA", "MAQ."},
        "TRANSPORTE": {"TRANSPORTE", "TRANSPORTES", "TRANS.", "ACARREO"},
        "HERRAMIENTA": {"HERRAMIENTA", "HERRAMIENTAS", "HERR.", "UTILES"},
        "OTROS": {"OTROS", "OTRO", "VARIOS", "ADICIONALES"},
    }

    JUNK_KEYWORDS = {
        "SUBTOTAL",
        "COSTO DIRECTO",
        "DESCRIPCION",
        "IMPUESTOS",
        "POLIZAS",
        "TOTAL",
        "IVA",
        "AIU",
    }

    def __init__(
        self,
        file_path: Union[str, Path],
        profile: dict,
        config: Optional[Dict] = None,
    ):
        """
        Inicializa el parser.

        Args:
            file_path: La ruta al archivo a ser parseado.
            config: Un diccionario de configuración opcional.
        """
        self.file_path = Path(file_path)
        self.profile = profile or {}
        self.config = config or {}
        self._validate_file_path()

        # --- INICIO DE LA MODIFICACIÓN ---
        from .apu_processor import APU_GRAMMAR  # Importar la gramática
        self.lark_parser = self._initialize_lark_parser(APU_GRAMMAR)
        self._parse_cache: Dict[str, Tuple[bool, Any]] = {}
        self.validation_stats = ValidationStats()
        # --- FIN DE LA MODIFICACIÓN ---

        self.raw_records: List[Dict[str, Any]] = []
        self.stats: Counter = Counter()
        self._parsed: bool = False

    def _initialize_lark_parser(self, grammar: Optional[str] = None) -> Optional[Lark]:
        """
        Inicializa el parser Lark con la MISMA gramática que usa APUProcessor.

        Args:
            grammar: String con la gramática Lark. Si es None, se carga desde archivo.

        Returns:
            Instancia de Lark o None si falla la inicialización.
        """
        try:
            if grammar is None:
                # Cargar desde el mismo lugar que APUProcessor
                from .apu_processor import APU_GRAMMAR
                grammar = APU_GRAMMAR

            parser = Lark(
                grammar,
                start='line',
                parser='lalr',
                # Usar las mismas opciones que APUProcessor
                maybe_placeholders=False,
                cache=True,
            )

            logger.info("✓ Parser Lark inicializado correctamente para pre-validación")
            return parser

        except Exception as e:
            logger.error(
                f"✗ Error al inicializar parser Lark: {e}\n"
                f"  Continuando SIN validación Lark (modo permisivo forzado)"
            )
            return None

    def _validate_with_lark(
        self,
        line: str,
        use_cache: bool = True
    ) -> tuple[bool, Optional[Any], str]:
        """
        Valida una línea usando el parser Lark.

        Esta es la validación CRÍTICA que garantiza que solo pasamos líneas
        que APUProcessor podrá procesar exitosamente.

        Args:
            line: Línea a validar.
            use_cache: Si True, usa cache de parsing.

        Returns:
            Tupla (es_válida, árbol_parsing, razón_fallo)
        """
        if not self.lark_parser:
            return (True, None, "Lark no disponible - validación omitida")

        # Verificar cache
        if use_cache and line in self._parse_cache:
            self.validation_stats.cached_parses += 1
            is_valid, tree = self._parse_cache[line]
            return (is_valid, tree, "" if is_valid else "Cached failure")

        line_clean = line.strip()

        try:
            tree = self.lark_parser.parse(line_clean)

            # Cache de éxito
            if use_cache:
                self._parse_cache[line] = (True, tree)

            return (True, tree, "")

        except Exception as e:
            # Cache de fallo
            if use_cache:
                self._parse_cache[line] = (False, None)

            error_type = type(e).__name__
            error_msg = str(e)

            # Clasificar tipo de error
            if "UnexpectedInput" in error_type:
                self.validation_stats.failed_lark_unexpected_input += 1
            elif "UnexpectedCharacters" in error_type:
                self.validation_stats.failed_lark_unexpected_chars += 1
            else:
                self.validation_stats.failed_lark_parse += 1

            return (False, None, f"Lark {error_type}: {error_msg}")

    def _validate_basic_structure(
        self,
        line: str,
        fields: List[str]
    ) -> tuple[bool, str]:
        """
        Validación básica PRE-Lark para filtrado rápido.

        Esta validación es RÁPIDA y elimina casos obvios antes de invocar Lark.

        Args:
            line: Línea completa.
            fields: Campos separados por ";".

        Returns:
            Tupla (es_válida, razón_si_inválida)
        """
        # Validación 1: Número mínimo de campos
        if len(fields) < 5:
            self.validation_stats.failed_basic_fields += 1
            return (False, f"Insuficientes campos: {len(fields)} < 5")

        # Validación 2: Campo de descripción no vacío
        if not fields[0] or not fields[0].strip():
            self.validation_stats.failed_basic_fields += 1
            return (False, "Campo de descripción vacío")

        # Validación 3: Detectar subtotales/totales
        line_upper = line.upper()
        subtotal_keywords = [
            "SUBTOTAL", "TOTAL", "SUMA", "SUMATORIA",
            "COSTO DIRECTO", "COSTO TOTAL", "PRECIO TOTAL"
        ]

        if any(keyword in line_upper for keyword in subtotal_keywords):
            self.validation_stats.failed_basic_subtotal += 1
            return (False, "Línea de subtotal/total")

        # Validación 4: Líneas decorativas
        if self._is_junk_line(line_upper):
            self.validation_stats.failed_basic_junk += 1
            return (False, "Línea decorativa/separador")

        # Validación 5: Al menos un campo numérico
        has_numeric = False
        numeric_pattern = re.compile(r'\d+[.,]\d+|\d+')

        for f in fields[1:]:  # Saltar descripción
            if numeric_pattern.search(f.strip()):
                has_numeric = True
                break

        if not has_numeric:
            self.validation_stats.failed_basic_numeric += 1
            return (False, "Sin campos numéricos detectables")

        self.validation_stats.passed_basic += 1
        return (True, "")

    def _validate_insumo_line(
        self,
        line: str,
        fields: List[str]
    ) -> LineValidationResult:
        """
        Validación UNIFICADA de una línea candidata a insumo.

        Estrategia de validación en dos capas:
        1. Validación básica (rápida, filtro de casos obvios)
        2. Validación Lark (estricta, garantiza compatibilidad)

        Args:
            line: La línea original completa.
            fields: Los campos ya separados por ";".

        Returns:
            LineValidationResult con el resultado detallado.
        """
        self.validation_stats.total_evaluated += 1

        # CAPA 1: Validación básica (filtro rápido)
        basic_valid, basic_reason = self._validate_basic_structure(line, fields)

        if not basic_valid:
            return LineValidationResult(
                is_valid=False,
                reason=f"Básica: {basic_reason}",
                fields_count=len(fields),
                validation_layer="basic"
            )

        # CAPA 2: Validación Lark (el juez final)
        lark_valid, lark_tree, lark_reason = self._validate_with_lark(line)

        if lark_valid:
            self.validation_stats.passed_lark += 1
            if basic_valid:
                self.validation_stats.passed_both += 1

            return LineValidationResult(
                is_valid=True,
                reason="Validación completa exitosa",
                fields_count=len(fields),
                has_numeric_fields=True,
                validation_layer="both",
                lark_tree=lark_tree
            )
        else:
            # Fallo en Lark
            self._record_failed_sample(line, fields, lark_reason)

            return LineValidationResult(
                is_valid=False,
                reason=f"Lark: {lark_reason}",
                fields_count=len(fields),
                has_numeric_fields=True,
                validation_layer="lark_failed"
            )

    def _record_failed_sample(
        self,
        line: str,
        fields: List[str],
        reason: str
    ):
        """
        Registra una muestra de línea fallida para análisis posterior.

        Args:
            line: Línea que falló.
            fields: Campos de la línea.
            reason: Razón del fallo.
        """
        max_samples = self.config.get("max_failed_samples", 10)
        if len(self.validation_stats.failed_samples) < max_samples:
            self.validation_stats.failed_samples.append({
                "line": line[:200],
                "fields": fields,
                "fields_count": len(fields),
                "reason": reason,
                "has_empty_fields": any(not f.strip() for f in fields),
                "empty_field_positions": [
                    i for i, f in enumerate(fields) if not f.strip()
                ],
            })



    def _log_validation_summary(self):
        """Registra un resumen detallado de la validación."""
        total = self.validation_stats.total_evaluated
        valid = self.stats.get("insumos_extracted", 0)

        logger.info("=" * 80)
        logger.info("📊 RESUMEN DE VALIDACIÓN CON LARK")
        logger.info("=" * 80)
        logger.info(f"Total líneas evaluadas: {total}")
        if total > 0:
            valid_percent = f"({valid/total*100:.1f}%)"
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
        logger.info(
            f"  - Sin datos numéricos: {self.validation_stats.failed_basic_numeric}"
        )
        logger.info(f"  - Subtotales: {self.validation_stats.failed_basic_subtotal}")
        logger.info(f"  - Líneas decorativas: {self.validation_stats.failed_basic_junk}")
        logger.info("")
        logger.info("Rechazos por validación Lark:")
        logger.info(
            f"  - Parse error genérico: {self.validation_stats.failed_lark_parse}"
        )
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
                if sample['has_empty_fields']:
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
                f"⚠️  Tasa de validación baja: {valid/total*100:.1f}%\n"
                f"   Considere revisar la gramática o el formato de datos"
            )

    def get_parse_cache(self) -> Dict[str, Any]:
        """
        Retorna el cache de parsing para reutilización en APUProcessor.

        Returns:
            Diccionario con líneas parseadas y sus árboles Lark.
        """
        return {
            line: tree
            for line, (is_valid, tree) in self._parse_cache.items()
            if is_valid and tree is not None
        }

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

        Se considera "ruido" a líneas vacías, subtotales, totales, o líneas
        puramente decorativas (ej. '-----').

        Args:
            line_upper: La línea de texto en mayúsculas.

        Returns:
            True si la línea es "ruido", False en caso contrario.
        """
        if len(line_upper.strip()) < 3:
            return True
        for keyword in self.JUNK_KEYWORDS:
            if keyword in line_upper:
                return True
        # Lines with decorative characters
        if re.search(r"^[=\-_\s*]+$", line_upper):
            return True
        return False

    def _parse_by_lines(self, lines: List[str]) -> bool:
        """
        Máquina de estados con validación UNIFICADA usando Lark.

        Cambio crítico: Ahora usa el MISMO parser que APUProcessor para
        garantizar que solo se extraen líneas que serán procesables.

        Args:
            lines: La lista de todas las líneas del archivo.

        Returns:
            True si se extrajo al menos un insumo válido, False en caso contrario.
        """
        current_apu_context: Optional[APUContext] = None
        current_category = "INDEFINIDO"
        i = 0

        logger.info(f"Iniciando parsing de {len(lines)} líneas con validación Lark")

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Estado 1: Buscar encabezado de APU
            is_header_line = "UNIDAD:" in line.upper()
            is_item_line_next = (i + 1) < len(lines) and "ITEM:" in lines[i + 1].upper()

            if is_header_line and is_item_line_next:
                header_line = line
                item_line = lines[i + 1].strip()

                try:
                    apu_desc = header_line.split(";")[0].strip()
                    unit_match = re.search(r"UNIDAD:\s*(\S+)", header_line, re.IGNORECASE)
                    apu_unit = (
                        unit_match.group(1) if unit_match else self.config.default_unit
                    )

                    item_match = re.search(r"ITEM:\s*([\S,]+)", item_line, re.IGNORECASE)
                    apu_code_raw = (
                        item_match.group(1) if item_match else f"UNKNOWN_APU_{i + 1}"
                    )
                    apu_code = clean_apu_code(apu_code_raw)

                    current_apu_context = APUContext(
                        apu_code=apu_code,
                        apu_desc=apu_desc,
                        apu_unit=apu_unit,
                        source_line=i + 1,
                    )
                    current_category = "INDEFINIDO"
                    self.stats["apus_detected"] += 1

                    logger.info(
                        f"✓ APU detectado [línea {i + 1}]: {apu_code} - {apu_desc[:50]}"
                    )

                    i += 2
                    continue

                except Exception as e:
                    logger.warning(
                        f"✗ Fallo al parsear encabezado de APU en línea {i + 1}: {e}"
                    )
                    current_apu_context = None
                    i += 1
                    continue

            # Estado 2: Procesar líneas dentro de contexto de APU
            if current_apu_context:
                line_upper = line.upper()

                # Detectar categoría
                new_category = self._detect_category(line_upper)
                if new_category:
                    current_category = new_category
                    self.stats[f"category_{current_category}"] += 1
                    logger.debug(f"  → Categoría: {current_category}")
                    i += 1
                    continue

                # Detectar ruido
                if self._is_junk_line(line_upper):
                    self.stats["junk_lines_skipped"] += 1
                    i += 1
                    continue

                # 🔥 VALIDACIÓN CRÍTICA CON LARK
                fields = [f.strip() for f in line.split(";")]
                validation_result = self._validate_insumo_line(line, fields)

                if validation_result.is_valid:
                    # ✅ LÍNEA VÁLIDA - Garantizada procesable por APUProcessor
                    record = {
                        "apu_code": current_apu_context.apu_code,
                        "apu_desc": current_apu_context.apu_desc,
                        "apu_unit": current_apu_context.apu_unit,
                        "category": current_category,
                        "insumo_line": line,
                        "source_line": i + 1,
                        "fields_count": validation_result.fields_count,
                        "validation_layer": validation_result.validation_layer,
                        # 🔥 OPTIMIZACIÓN: Guardar árbol de parsing para reutilizar
                        "_lark_tree": validation_result.lark_tree,
                    }
                    self.raw_records.append(record)
                    self.stats["insumos_extracted"] += 1

                    logger.debug((
                        f"  ✓ Insumo válido [línea {i + 1}] "
                        f"[{validation_result.validation_layer}]: "
                        f"{fields[0][:40]}... ({validation_result.fields_count} campos)"
                    ))
                else:
                    # ❌ LÍNEA RECHAZADA
                    logger.debug(
                        f"  ✗ Rechazada [línea {i + 1}]: {validation_result.reason}\n"
                        f"    Contenido: {line[:80]}..."
                    )
                    self.stats["lines_ignored_in_context"] += 1

            i += 1

        # Log de estadísticas finales
        self._log_validation_summary()

        return self.stats["insumos_extracted"] > 0

