import logging
import re
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ReportParser:
    """
    Parsea un archivo de reporte de APUs en formato de texto (tipo SAGUT)
    utilizando una máquina de estados y expresiones regulares para manejar
    formatos inconsistentes y delimitadores variables.
    """

    PATTERNS = {
        "item_code": re.compile(r"ITEM:\s*([\d\s,.]*)$"),
        "category": re.compile(r"^(MATERIALES|MANO DE OBRA|EQUIPO|OTROS)"),
        # CSV (semicolon-delimited) patterns
        "mano_de_obra_compleja_csv": re.compile(
            r"^(?P<descripcion>.+?)\s*;"
            r"(?P<jornal_base>[\d.,]+);"
            r"(?P<prestaciones>\S+);"
            r"(?P<jornal_total>[\d.,]+);"
            r"(?P<rendimiento>[\d.,]+);"
            r"(?P<valor_total>[\d.,]+)$"
        ),
        "insumo_full_csv": re.compile(
            r"^(?P<descripcion>[^;]+);"
            r"(?P<unidad>[A-Z0-9/%]+);"
            r"(?P<cantidad>[\d.,\s]+);"
            r"(?P<desperdicio>[\d.,\s-]+);"
            r"(?P<precio_unit>[\d.,\s]+);"
            r"(?P<valor_total>[\d.,\s]+)$"
        ),
        "insumo_simple_csv": re.compile(
            r"^(?P<descripcion>[^;]+);"
            r"(?P<unidad>[A-Z0-9/%]+);"
            r"(?P<cantidad>[\d.,\s]+);"
            r";"
            r"(?P<precio_unit>[\d.,\s]+);"
            r"(?P<valor_total>[\d.,\s]+)$"
        ),
        "herramienta_menor_csv": re.compile(
            r"^(?P<descripcion>EQUIPO Y HERRAMIENTA\s*|HERRAMIENTA MENOR\s*);"
            r"(?P<unidad>UND|%);"
            r"(?P<base_calculo>[\d.,\s]+);"
            r";"
            r"(?P<porcentaje>[\d.,%]+);"
            r"(?P<valor_total>[\d.,\s]+)$"
        ),
        # TXT (space-delimited) patterns
        "mano_de_obra_compleja_txt": re.compile(
            r"^(?P<descripcion>.+?)\s+"
            r"(?P<jornal_base>[\d.,]+)\s+"
            r"(?P<prestaciones>\S+)\s+"  # Puede ser % o número
            r"(?P<jornal_total>[\d.,]+)\s+"
            r"(?P<rendimiento>[\d.,]+)\s+"
            r"(?P<valor_total>[\d.,]+)$"
        ),
        "insumo_full_txt": re.compile(
            r"^(?P<descripcion>.+?)\s+"
            r"(?P<unidad>[A-Z0-9%]{2,10})\s+"
            r"(?P<cantidad>[\d.,]+)\s+"
            r"(?P<desperdicio>[\d.,]+|-)\s+"
            r"(?P<precio_unit>.+?)\s{2,}"
            r"(?P<valor_total>[\d\s.,]+)$"
        ),
        "insumo_simple_txt": re.compile(
            r"^(?P<descripcion>.+?)\s+"
            r"(?P<unidad>[A-Z0-9%]{2,10})\s+"
            r"(?P<cantidad>[\d.,]+)\s+"
            r"(?P<precio_unit>.+?)\s{2,}"
            r"(?P<valor_total>[\d\s.,]+)$"
        ),
        "herramienta_menor_txt": re.compile(
            r"^(?P<descripcion>EQUIPO Y HERRAMIENTA|HERRAMIENTA MENOR)\s+"
            r"(?P<unidad>UND|%)\s+"
            r"(?P<base_calculo>[\d.,]+)\s+"
            r"(?P<porcentaje>[\d.,%]+)\s+"
            r"(?P<valor_total>[\d.,]+)$"
        ),
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.apus_data = []
        self._current_apu_code: Optional[str] = None
        self._current_apu_desc: str = ""
        self._current_category: str = "INDEFINIDO"

    def parse(self) -> pd.DataFrame:
        logger.info(f"Iniciando el parsing del archivo: {self.file_path}")
        try:
            with open(self.file_path, "r", encoding="latin1") as f:
                for line in f:
                    self._process_line(line)
        except Exception as e:
            logger.error(f"Error al parsear {self.file_path}: {e}", exc_info=True)
            return pd.DataFrame()

        if not self.apus_data:
            logger.warning("No se extrajeron datos, devolviendo DataFrame vacío.")
            return pd.DataFrame()

        df = pd.DataFrame(self.apus_data)
        df["CODIGO_APU"] = df["CODIGO_APU"].str.strip()
        df["NORMALIZED_DESC"] = self._normalize_text(df["DESCRIPCION_INSUMO"])
        return df

    def _process_line(self, line: str):
        line = line.strip()
        if not line:
            return

        match_item = self.PATTERNS["item_code"].search(line.upper())
        if match_item:
            self._start_new_apu(match_item.group(1))
            return

        match_category = self.PATTERNS["category"].match(line)
        if match_category:
            self._current_category = match_category.group(1)
            return

        if self._current_apu_code:
            # Try CSV patterns first
            match_herramienta = self.PATTERNS["herramienta_menor_csv"].match(line)
            if match_herramienta:
                self._parse_herramienta_menor(match_herramienta.groupdict())
                return

            if self._current_category == "MANO DE OBRA":
                match_mano_de_obra_compleja = self.PATTERNS["mano_de_obra_compleja_csv"].match(line)
                if match_mano_de_obra_compleja:
                    self._parse_mano_de_obra_compleja(match_mano_de_obra_compleja.groupdict())
                    return

            match_insumo_full = self.PATTERNS["insumo_full_csv"].match(line)
            if match_insumo_full:
                self._parse_insumo(match_insumo_full.groupdict())
                return

            match_insumo_simple = self.PATTERNS["insumo_simple_csv"].match(line)
            if match_insumo_simple:
                data = match_insumo_simple.groupdict()
                data["desperdicio"] = ""
                self._parse_insumo(data)
                return

            # Then try TXT patterns
            match_herramienta = self.PATTERNS["herramienta_menor_txt"].match(line)
            if match_herramienta:
                self._parse_herramienta_menor(match_herramienta.groupdict())
                return

            if self._current_category == "MANO DE OBRA":
                match_mano_de_obra_compleja = self.PATTERNS["mano_de_obra_compleja_txt"].match(line)
                if match_mano_de_obra_compleja:
                    self._parse_mano_de_obra_compleja(match_mano_de_obra_compleja.groupdict())
                    return

            match_insumo_full = self.PATTERNS["insumo_full_txt"].match(line)
            if match_insumo_full:
                self._parse_insumo(match_insumo_full.groupdict())
                return

            match_insumo_simple = self.PATTERNS["insumo_simple_txt"].match(line)
            if match_insumo_simple:
                data = match_insumo_simple.groupdict()
                data["desperdicio"] = ""
                self._parse_insumo(data)
                return

        self._current_apu_desc = line

    def _start_new_apu(self, raw_code: str):
        self._current_apu_code = re.sub(r"[\s,.]+$", "", raw_code.replace(" ", ","))
        self._current_category = "INDEFINIDO"

    def _parse_insumo(self, data: Dict[str, str]):
        try:
            self.apus_data.append(
                {
                    "CODIGO_APU": self._current_apu_code,
                    "DESCRIPCION_APU": self._current_apu_desc,
                    "DESCRIPCION_INSUMO": data["descripcion"].strip(),
                    "UNIDAD": data["unidad"],
                    "CANTIDAD_APU": self._to_numeric_safe(data["cantidad"]),
                    "PRECIO_UNIT_APU": self._to_numeric_safe(data["precio_unit"]),
                    "VALOR_TOTAL_APU": self._to_numeric_safe(data["valor_total"]),
                    "CATEGORIA": self._current_category,
                    "DESPERDICIO": self._to_numeric_safe(data.get("desperdicio")),
                }
            )
        except Exception as e:
            logger.warning(f"No se pudo parsear la línea de insumo: '{data}'. Error: {e}")

    def _parse_herramienta_menor(self, data: Dict[str, str]):
        try:
            valor_total = self._to_numeric_safe(data["valor_total"])
            self.apus_data.append(
                {
                    "CODIGO_APU": self._current_apu_code,
                    "DESCRIPCION_APU": self._current_apu_desc,
                    "DESCRIPCION_INSUMO": data["descripcion"].strip(),
                    "UNIDAD": data["unidad"],
                    "CANTIDAD_APU": 1,
                    "PRECIO_UNIT_APU": valor_total,
                    "VALOR_TOTAL_APU": valor_total,
                    "CATEGORIA": self._current_category,
                }
            )
        except Exception as e:
            logger.warning(
                f"No se pudo parsear la línea de herramienta: '{data}'. Error: {e}"
            )

    def _parse_mano_de_obra_compleja(self, data: Dict[str, str]):
        try:
            rendimiento = self._to_numeric_safe(data["rendimiento"])
            jornal_total = self._to_numeric_safe(data["jornal_total"])
            cantidad = (1 / rendimiento) if rendimiento > 0 else 0
            precio_unitario = jornal_total
            self.apus_data.append(
                {
                    "CODIGO_APU": self._current_apu_code,
                    "DESCRIPCION_APU": self._current_apu_desc,
                    "DESCRIPCION_INSUMO": data["descripcion"].strip(),
                    "UNIDAD": "JOR",
                    "CANTIDAD_APU": cantidad,
                    "PRECIO_UNIT_APU": precio_unitario,
                    "VALOR_TOTAL_APU": self._to_numeric_safe(data["valor_total"]),
                    "CATEGORIA": self._current_category,
                }
            )
        except Exception as e:
            logger.warning(
                f"No se pudo parsear la línea de mano de obra compleja: '{data}'. Error: {e}"
            )

    def _to_numeric_safe(self, s: Optional[str]) -> float:
        if not s:
            return 0.0
        # Eliminar espacios, luego puntos (miles), luego cambiar coma por punto (decimal)
        s_cleaned = s.replace(" ", "").replace(".", "").replace(",", ".").strip()
        # Manejar el caso de una cadena vacía después de la limpieza
        if not s_cleaned or s_cleaned == "-":
            return 0.0
        try:
            return float(s_cleaned)
        except (ValueError, TypeError):
             return 0.0

    def _normalize_text(self, series: pd.Series) -> pd.Series:
        from unidecode import unidecode

        normalized = series.astype(str).str.lower().str.strip()
        normalized = normalized.apply(unidecode)
        normalized = normalized.str.replace(r"[^a-z0-9\s#\-]", "", regex=True)
        normalized = normalized.str.replace(r"\s+", " ", regex=True)
        return normalized