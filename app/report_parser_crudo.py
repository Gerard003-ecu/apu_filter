import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from .utils import clean_apu_code

logger = logging.getLogger(__name__)


class ParserState(Enum):
    IDLE = "IDLE"
    AWAITING_DESCRIPTION = "AWAITING_DESCRIPTION"
    SKIPPING_HEADERS = "SKIPPING_HEADERS"
    PROCESSING_APU = "PROCESSING_APU"


class ReportParserCrudo:
    """
    Etapa 1: Extrae datos crudos del archivo sin aplicar lógica de negocio.
    Salida: lista de diccionarios con solo strings.
    """

    PATTERNS = {
        "item_code": re.compile(r"ITEM:\s*([^;]+)", re.IGNORECASE),
        "generic_data": re.compile(
            r"^(?P<descripcion>[^;]+);"
            r"(?P<col2>[^;]*);"
            r"(?P<col3>[^;]*);"
            r"(?P<col4>[^;]*);"
            r"(?P<col5>[^;]*);"
            r"(?P<col6>[^;]*)"
        ),
    }

    CATEGORY_KEYWORDS = {"MATERIALES", "MANO DE OBRA", "EQUIPO", "OTROS", "TRANSPORTE"}

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_records: List[Dict[str, str]] = []
        self.state = ParserState.IDLE
        self.context = {
            "apu_code": "",
            "apu_desc": "",
            "apu_unit": "",
            "category": "INDEFINIDO",
        }

    def parse_to_raw(self) -> List[Dict[str, str]]:
        """Punto de entrada: devuelve solo datos crudos como strings."""
        logger.info(f"🔍 Iniciando extracción cruda desde: {self.file_path}")
        try:
            with open(self.file_path, "r", encoding="latin1") as f:
                for line_num, line in enumerate(f, 1):
                    self._process_line(line, line_num)
        except Exception as e:
            logger.error(f"❌ Error al leer {self.file_path}: {e}")
            return []
        logger.info(f"✅ Extracción cruda completada: {len(self.raw_records)} registros")
        return self.raw_records

    def _process_line(self, line: str, line_num: int):
        line = line.strip()
        if not line:
            return

        # Detectar nuevo ITEM en cualquier estado
        if self._try_start_new_apu(line, line_num):
            return

        if self.state == ParserState.IDLE:
            return
        elif self.state == ParserState.AWAITING_DESCRIPTION:
            if self._is_valid_apu_description(line):
                self._capture_apu_description(line)
                self.state = ParserState.PROCESSING_APU
            else:
                # No inferir, solo esperar
                return
        elif self.state == ParserState.PROCESSING_APU:
            if self._try_detect_category_change(line):
                return
            if self._has_data_structure(line):
                self._add_raw_record(insumo_line=line)

    def _try_start_new_apu(self, line: str, line_num: int) -> bool:
        match = self.PATTERNS["item_code"].search(line.upper())
        if not match:
            return False

        raw_code = match.group(1).strip()
        if not raw_code:
            return False

        # Extraer descripción inline si existe (solo como string)
        inline_desc = self._extract_inline_description(line)

        # Extraer unidad inline si existe (solo como string)
        inline_unit = self._extract_inline_unit(line)

        self.context = {
            "apu_code": raw_code,
            "apu_desc": inline_desc,
            "apu_unit": inline_unit or "UND",
            "category": "INDEFINIDO",
        }

        if inline_desc:
            self.state = ParserState.PROCESSING_APU
        else:
            self.state = ParserState.AWAITING_DESCRIPTION

        return True

    def _extract_inline_description(self, line: str) -> str:
        # Extrae solo texto, sin validación ni limpieza profunda
        # Usa patrones simples y devuelve string crudo
        patterns = [
            r'ITEM:\s*[^;]+;\s*(?:DESCRIPCION|DESCRIPCIÓN)\s*:\s*([^;]+)',
            r'ITEM:\s*[^;]+;\s*([^;]+)',
        ]
        for pat in patterns:
            match = re.search(pat, line, re.IGNORECASE)
            if match:
                desc = match.group(1).strip()
                if len(desc) > 5 and not desc.upper().startswith(("UNIDAD", "DESCRIP")):
                    return desc
        return ""

    def _extract_inline_unit(self, line: str) -> str:
        # Solo extrae si hay "UNIDAD:" explícito
        match = re.search(r"UNIDAD\s*:\s*([^;,\s]+)", line, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _is_valid_apu_description(self, line: str) -> bool:
        # Solo verifica que no sea encabezado o basura obvia
        first_part = line.split(";")[0].strip()
        if len(first_part) < 5:
            return False
        if first_part.upper() in {
            "ITEM", "DESCRIPCION", "DESCRIPCIÓN", "UNIDAD", "CANTIDAD", "VALOR TOTAL"
        }:
            return False
        if re.match(r"^[\d.,$%]+$", first_part):
            return False
        return True

    def _capture_apu_description(self, line: str):
        desc = line.split(";")[0].strip()
        self.context["apu_desc"] = desc

    def _try_detect_category_change(self, line: str) -> bool:
        first_part = line.split(";")[0].strip().upper()
        if first_part in self.CATEGORY_KEYWORDS and not self._has_data_structure(line):
            self.context["category"] = first_part
            return True
        return False

    def _has_data_structure(self, line: str) -> bool:
        return line.count(";") >= 2

    def _add_raw_record(self, **kwargs):
        cleaned_code = clean_apu_code(self.context["apu_code"])
        record = {
            "apu_code": cleaned_code,
            "apu_desc": self.context["apu_desc"],
            "apu_unit": self.context["apu_unit"],
            "category": self.context["category"],
            "insumo_line": kwargs.get("insumo_line", ""),
        }
        self.raw_records.append(record)
