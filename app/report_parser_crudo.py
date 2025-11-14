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
        self.profile = profile  # Guardar el perfil
        self.config = config or ParserConfig()
        self._validate_file_path()

        self.raw_records: List[Dict[str, Any]] = []
        self.stats: Counter = Counter()
        self._parsed: bool = False

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

                # Asumir que es una línea de insumo
                fields = [f.strip() for f in line.split(";")]
                if len(fields) >= 5 and fields[0]:
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
                else:
                    self.stats["lines_ignored_in_context"] += 1

            i += 1

        return self.stats["insumos_extracted"] > 0
