# app/report_parser_crudo.py

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .utils import clean_apu_code

logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Base exception for parser errors."""
    pass

class FileReadError(ParserError):
    """Error reading the file."""
    pass

class ParseStrategyError(ParserError):
    """Error in the parsing strategy."""
    pass


@dataclass
class ParserConfig:
    """Simplified configuration for the parser."""
    encodings: List[str] = field(default_factory=lambda: ['utf-8', 'latin1', 'cp1252', 'iso-8859-1'])
    default_unit: str = "UND"
    max_lines_to_process: int = 100000

@dataclass
class APUContext:
    """Context of an APU."""
    apu_code: str
    apu_desc: str
    apu_unit: str
    source_line: int

    def __post_init__(self):
        """Validation and normalization."""
        self.apu_code = self.apu_code.strip() if self.apu_code else ""
        self.apu_desc = self.apu_desc.strip() if self.apu_desc else ""
        self.apu_unit = self.apu_unit.strip().upper() if self.apu_unit else self.default_unit
        if not self.apu_code:
            raise ValueError("APU code cannot be empty")

    @property
    def is_valid(self) -> bool:
        """Checks if the context is valid."""
        return bool(self.apu_code and len(self.apu_code) >= 2)


class ReportParserCrudo:
    """
    A robust, line-by-line state machine parser for semi-structured APU files.
    This version focuses on a single, well-defined strategy.
    """
    CATEGORY_KEYWORDS = {
        'MATERIALES': {'MATERIALES', 'MATERIAL', 'MAT.', 'INSUMOS'},
        'MANO DE OBRA': {'MANO DE OBRA', 'MANO OBRA', 'M.O.', 'MO', 'PERSONAL', 'OBRERO'},
        'EQUIPO': {'EQUIPO', 'EQUIPOS', 'MAQUINARIA', 'MAQ.'},
        'TRANSPORTE': {'TRANSPORTE', 'TRANSPORTES', 'TRANS.', 'ACARREO'},
        'HERRAMIENTA': {'HERRAMIENTA', 'HERRAMIENTAS', 'HERR.', 'UTILES'},
        'OTROS': {'OTROS', 'OTRO', 'VARIOS', 'ADICIONALES'},
    }

    JUNK_KEYWORDS = {
        "SUBTOTAL", "COSTO DIRECTO", "DESCRIPCION", "IMPUESTOS",
        "POLIZAS", "TOTAL", "IVA", "AIU"
    }

    def __init__(self, file_path: Union[str, Path], config: Optional[ParserConfig] = None):
        self.file_path = Path(file_path)
        self.config = config or ParserConfig()
        self._validate_file_path()

        self.raw_records: List[Dict[str, Any]] = []
        self.stats: Counter = Counter()
        self._parsed: bool = False

    def _validate_file_path(self) -> None:
        """Validates the file path."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")
        if self.file_path.stat().st_size == 0:
            raise ValueError(f"File is empty: {self.file_path}")

    def parse_to_raw(self) -> List[Dict[str, Any]]:
        """Main entry point for parsing the file."""
        if self._parsed:
            return self.raw_records

        logger.info(f"Starting line-by-line parsing of: {self.file_path.name}")

        try:
            content = self._read_file_safely()
            lines = content.split('\n')
            self.stats['total_lines'] = len(lines)

            self._parse_by_lines(lines)

            self._parsed = True
            logger.info(f"Parsing complete. Extracted {self.stats['insumos_extracted']} raw records.")
            if self.stats['insumos_extracted'] == 0:
                 logger.warning("No records were extracted. The file might be empty or in an unexpected format.")

        except Exception as e:
            logger.error(f"Critical parsing error: {e}", exc_info=True)
            raise ParseStrategyError(f"Failed to parse file with line-by-line strategy: {e}") from e

        return self.raw_records

    def _read_file_safely(self) -> str:
        """Reads the file content trying multiple encodings."""
        for encoding in self.config.encodings:
            try:
                with open(self.file_path, 'r', encoding=encoding, errors='strict') as f:
                    content = f.read()
                self.stats['encoding_used'] = encoding
                logger.info(f"File read successfully with encoding: {encoding}")
                return content
            except (UnicodeDecodeError, TypeError, LookupError):
                continue
        raise FileReadError(f"Could not read file {self.file_path} with any of the specified encodings.")

    def _detect_category(self, line_upper: str) -> Optional[str]:
        """Detects if a line represents a category."""
        if len(line_upper) > 50 or sum(c.isdigit() for c in line_upper) > 3:
            return None
        for canonical, variations in self.CATEGORY_KEYWORDS.items():
            for variation in variations:
                pattern = r'\b' + re.escape(variation) + r'\b' if '.' not in variation else re.escape(variation)
                if re.search(pattern, line_upper):
                    return canonical
        return None

    def _is_junk_line(self, line_upper: str) -> bool:
        """Determines if a line should be ignored."""
        if len(line_upper.strip()) < 3:
            return True
        for keyword in self.JUNK_KEYWORDS:
            if keyword in line_upper:
                return True
        # Lines with decorative characters
        if re.search(r'^[=\-_\s*]+$', line_upper):
            return True
        return False

    def _parse_by_lines(self, lines: List[str]) -> bool:
        """
        State machine to parse the file line by line.
        """
        current_apu_context: Optional[APUContext] = None
        current_category = "INDEFINIDO"
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # State 1: Look for APU Header
            # The header is defined by a "UNIDAD:" line followed by an "ITEM:" line.
            is_header_line = "UNIDAD:" in line.upper()
            is_item_line_next = (i + 1) < len(lines) and "ITEM:" in lines[i + 1].upper()

            if is_header_line and is_item_line_next:
                header_line = line
                item_line = lines[i + 1].strip()

                try:
                    apu_desc = header_line.split(';')[0].strip()
                    unit_match = re.search(r'UNIDAD:\s*(\S+)', header_line, re.IGNORECASE)
                    apu_unit = unit_match.group(1) if unit_match else self.config.default_unit

                    item_match = re.search(r'ITEM:\s*([\S,]+)', item_line, re.IGNORECASE)
                    apu_code_raw = item_match.group(1) if item_match else f"UNKNOWN_APU_{i+1}"
                    apu_code = clean_apu_code(apu_code_raw)

                    current_apu_context = APUContext(
                        apu_code=apu_code,
                        apu_desc=apu_desc,
                        apu_unit=apu_unit,
                        source_line=i + 1
                    )
                    current_category = "INDEFINIDO" # Reset category for new APU
                    self.stats['apus_detected'] += 1
                    logger.debug(f"Found new APU context at line {i+1}: {apu_code}")
                    i += 2  # Skip both header and item lines
                    continue
                except Exception as e:
                    logger.warning(f"Failed to parse APU header at line {i+1}: {e}")
                    current_apu_context = None
                    i += 1
                    continue

            # State 2: Process lines within an APU context
            if current_apu_context:
                line_upper = line.upper()

                # Check for new category
                new_category = self._detect_category(line_upper)
                if new_category:
                    current_category = new_category
                    self.stats[f'category_{current_category}'] += 1
                    i += 1
                    continue

                # Check for junk lines
                if self._is_junk_line(line_upper):
                    self.stats['junk_lines_skipped'] += 1
                    i += 1
                    continue

                # Assume it's an insumo line
                fields = [f.strip() for f in line.split(';')]
                if len(fields) >= 5 and fields[0]:
                    record = {
                        'apu_code': current_apu_context.apu_code,
                        'apu_desc': current_apu_context.apu_desc,
                        'apu_unit': current_apu_context.apu_unit,
                        'category': current_category,
                        'insumo_line': line,
                        'source_line': i + 1,
                    }
                    self.raw_records.append(record)
                    self.stats['insumos_extracted'] += 1
                else:
                    self.stats['lines_ignored_in_context'] += 1

            i += 1

        return self.stats['insumos_extracted'] > 0
