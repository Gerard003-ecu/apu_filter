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
from typing import Any, Dict, List, Optional, Union

from .utils import clean_apu_code

logger = logging.getLogger(__name__)


@dataclass
class LineValidationResult:
    """Resultado de la validación de una línea."""

    is_valid: bool
    reason: str = ""
    fields_count: int = 0
    has_numeric_fields: bool = False


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
class ParserConfig:
    """
    Configuración simplificada para el parser.

    Attributes:
        encodings: Lista de codificaciones a intentar al leer el archivo.
        default_unit: Unidad por defecto a asignar si no se puede extraer.
        max_lines_to_process: Límite de líneas a procesar para evitar sobrecargas.
    """

    encodings: List[str] = field(
        default_factory=lambda: ["utf-8", "latin1", "cp1252", "iso-8859-1"]
    )
    default_unit: str = "UND"
    max_lines_to_process: int = 100000


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
        config: Optional[ParserConfig] = None,
    ):
        """
        Inicializa el parser.

        Args:
            file_path: La ruta al archivo a ser parseado.
            config: Un objeto `ParserConfig` opcional con la configuración.
        """
        self.file_path = Path(file_path)
        self.profile = profile or {}
        self.config = config or ParserConfig()
        self.numeric_pattern = self._build_numeric_pattern()
        self.validation_stats = {
            "total_lines_evaluated": 0,
            "valid_insumos": 0,
            "rejected_insufficient_fields": 0,
            "rejected_no_numeric_data": 0,
            "rejected_empty_key_field": 0,
            "rejected_subtotal_line": 0,
        }
        self._validate_file_path()

        self.raw_records: List[Dict[str, Any]] = []
        self.stats: Counter = Counter()
        self._parsed: bool = False

    def _build_numeric_pattern(self) -> re.Pattern:
        """Construye el patrón regex para validar números según el perfil."""
        number_format = self.profile.get("number_format", {})
        decimal_separator = number_format.get("decimal_separator")

        if decimal_separator == "comma":
            decimal_char = ","
            thousands_char = r"\."
        elif decimal_separator == "dot":
            decimal_char = r"\."
            thousands_char = ","
        else:
            # Si no se especifica, permitir ambos formatos
            decimal_char = r"[,.]"
            thousands_char = r"[.,]"

        # Patrón mejorado que es más flexible
        pattern = (
            r"^\s*[-+]?"  # Signo opcional
            r"(\d{1,3}(" + thousands_char + r"\d{3})*|\d+)"  # Parte entera con o sin separadores de miles
            r"(" + decimal_char + r"\d+)?"  # Parte decimal opcional
            r"\s*$"
        )
        return re.compile(pattern)

    def _validate_insumo_line(self, line: str, fields: List[str]) -> LineValidationResult:
        """Validación estricta de una línea candidata a insumo ANTES de enviarla a Lark."""
        # 1. Número mínimo de campos
        if len(fields) < 5:
            return LineValidationResult(
                is_valid=False, reason=f"Insuficientes campos: {len(fields)} < 5"
            )

        # 2. Descripción no vacía
        if not fields[0] or not fields[0].strip():
            return LineValidationResult(is_valid=False, reason="Campo de descripción vacío")

        # 3. Detectar líneas de subtotal/total
        if any(keyword in line.upper() for keyword in self.JUNK_KEYWORDS):
            return LineValidationResult(
                is_valid=False, reason="Línea de subtotal/junk detectada"
            )

        # 4. Al menos 2 campos numéricos válidos
        numeric_fields_found = 0
        for field in fields[1:]:  # Saltar descripción
            if field and self.numeric_pattern.match(field.strip()):
                numeric_fields_found += 1

        if numeric_fields_found < 2:
            return LineValidationResult(
                is_valid=False,
                reason=f"Campos numéricos insuficientes: {numeric_fields_found} < 2",
            )

        return LineValidationResult(is_valid=True)

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
        # CAMBIO: Usar el encoding del perfil como primera opción
        encodings_to_try = [self.profile.get("encoding")] + self.config.encodings

        for encoding in filter(
            None, encodings_to_try
        ):  # filter(None, ...) para saltar si el perfil no tiene encoding
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
        Máquina de estados que procesa el archivo línea por línea.

        Itera sobre cada línea y, dependiendo del estado actual (si se está
        dentro de un contexto de APU o no), decide cómo procesarla.

        Args:
            lines: La lista de todas las líneas del archivo.

        Returns:
            True si se extrajo al menos un insumo, False en caso contrario.
        """
        current_apu_context: Optional[APUContext] = None
        current_category = "INDEFINIDO"
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Estado 1: Buscar un encabezado de APU.
            # Un encabezado se define por una línea "UNIDAD:" seguida de "ITEM:".
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
                    current_category = "INDEFINIDO"  # Reiniciar categoría para nuevo APU
                    self.stats["apus_detected"] += 1
                    logger.debug(
                        f"Nuevo contexto de APU encontrado en línea {i + 1}: {apu_code}"
                    )
                    i += 2  # Saltar las dos líneas del encabezado
                    continue
                except Exception as e:
                    logger.warning(
                        f"Fallo al parsear encabezado de APU en línea {i + 1}: {e}"
                    )
                    current_apu_context = None
                    i += 1
                    continue

            # Estado 2: Procesar líneas dentro de un contexto de APU.
            if current_apu_context:
                line_upper = line.upper()

                # Comprobar si es una nueva categoría
                new_category = self._detect_category(line_upper)
                if new_category:
                    current_category = new_category
                    self.stats[f"category_{current_category}"] += 1
                    i += 1
                    continue

                # Comprobar si es una línea de "ruido"
                if self._is_junk_line(line_upper):
                    self.stats["junk_lines_skipped"] += 1
                    i += 1
                    continue

                # --- INICIO DE LA MODIFICACIÓN ---
                # Asumir que es una línea de insumo y VALIDARLA ESTRICTAMENTE
                fields = [f.strip() for f in line.split(";")]
                self.validation_stats["total_lines_evaluated"] += 1

                validation_result = self._validate_insumo_line(line, fields)

                if validation_result.is_valid:
                    # ✅ Línea VÁLIDA - Agregar a registros
                    record = {
                        "apu_code": current_apu_context.apu_code,
                        "apu_desc": current_apu_context.apu_desc,
                        "apu_unit": current_apu_context.apu_unit,
                        "category": current_category,
                        "insumo_line": line,
                        "source_line": i + 1,
                    }
                    self.raw_records.append(record)
                    self.stats["insumos_extracted"] += 1
                    self.validation_stats["valid_insumos"] += 1
                    logger.debug(f" ✓ Insumo válido [línea {i + 1}]: {fields[0][:50]}...")
                else:
                    # ❌ Línea RECHAZADA - Registrar y continuar
                    if "Insuficientes campos" in validation_result.reason:
                        self.validation_stats["rejected_insufficient_fields"] += 1
                    elif "numéricos insuficientes" in validation_result.reason:
                        self.validation_stats["rejected_no_numeric_data"] += 1
                    elif "descripción vacío" in validation_result.reason:
                        self.validation_stats["rejected_empty_key_field"] += 1
                    elif "subtotal" in validation_result.reason:
                        self.validation_stats["rejected_subtotal_line"] += 1
                    logger.debug(
                        f" ✗ Línea rechazada [línea {i + 1}]: {validation_result.reason} -> Contenido: {line[:80]}..."
                    )
                    self.stats["lines_ignored_in_context"] += 1
                # --- FIN DE LA MODIFICACIÓN ---

            i += 1
        self._log_validation_summary()  # Añadir esta llamada al final
        return self.stats["insumos_extracted"] > 0

    def _log_validation_summary(self):
        """Registra un resumen detallado de la validación."""
        total_eval = self.validation_stats["total_lines_evaluated"]
        valid = self.validation_stats["valid_insumos"]

        if total_eval == 0:
            logger.warning("⚠️  No se evaluaron líneas para validación")
            return

        logger.info("=" * 70)
        logger.info("📊 RESUMEN DE VALIDACIÓN DE LÍNEAS")
        logger.info("=" * 70)
        logger.info(f"Total líneas evaluadas:        {total_eval}")
        if total_eval > 0:
            logger.info(
                f"✓ Insumos válidos:             {valid} ({valid/total_eval*100:.1f}%)"
            )
        else:
            logger.info("✓ Insumos válidos:             0 (0.0%)")
        logger.info(
            f"✗ Rechazados - Campos insuf.:  {self.validation_stats['rejected_insufficient_fields']}"
        )
        logger.info(
            f"✗ Rechazados - Sin numéricos:  {self.validation_stats['rejected_no_numeric_data']}"
        )
        logger.info(
            f"✗ Rechazados - Desc. vacía:    {self.validation_stats['rejected_empty_key_field']}"
        )
        logger.info(
            f"✗ Rechazados - Subtotales:     {self.validation_stats['rejected_subtotal_line']}"
        )
        logger.info("=" * 70)

        if valid == 0 and total_eval > 0:
            logger.error(
                "🚨 CRÍTICO: 0 insumos válidos encontrados. "
                "Revise el formato del archivo o el perfil de configuración."
            )
