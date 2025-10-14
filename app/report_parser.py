# En app/report_parser.py

import logging
import re
from typing import Dict, Optional

import pandas as pd

from .utils import clean_apu_code

logger = logging.getLogger(__name__)

class ReportParser:
    """
    Parsea un archivo de reporte de APUs en formato de texto (tipo SAGUT)
    utilizando una máquina de estados pura para manejar el formato de reporte.
    VERSIÓN FINAL Y ROBUSTA.
    """
    PATTERNS = {
        "item_code": re.compile(r"ITEM:\s*([^;]+)", re.IGNORECASE),
        "insumo_full": re.compile(
            r"^(?P<descripcion>[^;]+);"
            r"(?P<unidad>[^;]*);"
            r"(?P<cantidad>[^;]*);"
            r"(?P<desperdicio>[^;]*);"
            r"(?P<precio_unit>[^;]*);"
            r"(?P<valor_total>[^;]*)",
            re.IGNORECASE
        ),
        "mano_de_obra": re.compile(
            r"^(?P<descripcion>(M\.O\.|SISO|INGENIERO).+?);"
            r"(?P<jornal_base>[\d.,\s]+);"
            r"(?P<prestaciones>[\d%.,\s]+);"
            r"(?P<jornal_total>[\d.,\s]+);"
            r"(?P<rendimiento>[\d.,\s]+);"
            r"(?P<valor_total>[\d.,\s]+)"
        ),
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.apus_data = []
        self.context = {
            "apu_code": None,
            "apu_desc": "",
            "apu_unit": "",
            "category": "INDEFINIDO",
        }
        self.potential_apu_desc = ""
        # Métricas
        self.stats = {
            "total_lines": 0,
            "processed_lines": 0,
            "items_found": 0,
            "insumos_parsed": 0,
            "mo_parsed": 0,
            "garbage_lines": 0
        }

    def parse(self) -> pd.DataFrame:
        logger.info(f"Iniciando el parsing del archivo: {self.file_path}")
        try:
            with open(self.file_path, "r", encoding="latin1") as f:
                for line in f:
                    self.stats["total_lines"] += 1
                    self._process_line(line)
        except Exception as e:
            logger.error(f"Error al parsear {self.file_path}: {e}", exc_info=True)
            return pd.DataFrame()

        logger.info(f"Métricas de parsing para {self.file_path}: {self.stats}")
        if not self.apus_data:
            logger.warning("No se extrajeron datos de APU, devolviendo DataFrame vacío.")
            return pd.DataFrame()

        df = pd.DataFrame(self.apus_data)
        df.rename(columns={
            "DESCRIPCION": "DESCRIPCION_INSUMO",
            "CANTIDAD": "CANTIDAD_APU",
            "VR_UNITARIO": "PRECIO_UNIT_APU",
            "VR_TOTAL": "VALOR_TOTAL_APU",
            "UNIDAD": "UNIDAD_INSUMO"
        }, inplace=True)
        df["NORMALIZED_DESC"] = self._normalize_text(df["DESCRIPCION_INSUMO"])
        return df

    def _process_line(self, line: str):
        """
        Versión final refinada del proceso principal.
        """
        line = line.strip()
        if not line:
            return

        self.stats["processed_lines"] += 1

        if self._is_garbage_line(line):
            self.stats["garbage_lines"] += 1
            return

        upper_line = line.upper()

        match_item = self.PATTERNS["item_code"].search(upper_line)
        if match_item:
            raw_code = match_item.group(1).strip()
            unit_match = re.search(r"UNIDAD:\s*([A-Z0-9/%]+)", upper_line)
            unit = unit_match.group(1) if unit_match else "INDEFINIDO"

            logger.debug(f"ITEM detectado: código='{raw_code}', unidad='{unit}'")
            self._start_new_apu(raw_code, unit)
            return

        category_keywords = {"MATERIALES", "MANO DE OBRA", "EQUIPO", "OTROS"}
        first_part = line.split(';', 1)[0].strip().upper()

        for category in category_keywords:
            if category in first_part:
                self.context["category"] = category
                logger.debug(f"Categoría establecida: {category}")
                return

        if not self.context["apu_code"]:
            if self._is_potential_description(line):
                self.potential_apu_desc = line.split(';', 1)[0].strip()
            return

        processed = self._try_parse_data_line(line)
        if not processed and self._is_potential_description(line):
            self.potential_apu_desc = line.split(';', 1)[0].strip()

    def _try_parse_data_line(self, line: str) -> bool:
        """
        Intenta parsear una línea como datos, retorna True si tuvo éxito.
        """
        match_mo = self.PATTERNS["mano_de_obra"].match(line)
        if match_mo:
            data = match_mo.groupdict()
            if self._is_valid_mo_data(data):
                self._parse_mano_de_obra(data)
                return True

        match_insumo = self.PATTERNS["insumo_full"].match(line)
        if match_insumo:
            data = match_insumo.groupdict()
            if self._is_valid_insumo_data(data):
                self._parse_insumo(data)
                return True

        if self._has_data_structure(line):
            logger.warning(f"Línea con estructura no parseada: {line[:100]}")

        return False

    def _calculate_mo_quantity(self, valor_total: float, jornal_total: float, rendimiento: float) -> float:
        """
        Versión mejorada con manejo de casos edge.
        """
        if jornal_total <= 0:
            return 0.0

        if valor_total > 0:
            cantidad_base = valor_total / jornal_total
        else:
            cantidad_base = 0

        if rendimiento > 0:
            if 0 < rendimiento < 0.2:
                logger.debug(f"Rendimiento muy bajo ({rendimiento}), interpretando como horas")
                cantidad_base *= 8
            elif rendimiento > 10:
                logger.debug(f"Rendimiento inusualmente alto: {rendimiento}")

        return cantidad_base

    def _should_add_insumo(self, descripcion: str, cantidad: float, valor_total: float) -> bool:
        """
        Versión más estricta con validaciones adicionales.
        """
        if not descripcion or len(descripcion.strip()) < 2:
            return False

        invalid_descriptions = {"", "-", "N/A", "NO APLICA", "S/D"}
        if descripcion.upper() in invalid_descriptions:
            return False

        if cantidad <= 0 and valor_total <= 0:
            return False

        if 0 < cantidad < 1e-10 or 0 < valor_total < 0.001:
            logger.debug(f"Valores muy pequeños descartados: cantidad={cantidad}, total={valor_total}")
            return False

        return True

    def _is_garbage_line(self, line: str) -> bool:
        """Filtra líneas que no aportan información."""
        upper_line = line.upper()
        garbage_keywords = [
            "FORMATO DE ANÁLISIS", "COSTOS DIRECTOS", "COSTO TOTAL",
            "PRESUPUESTO OFICIAL", "CONSTRUCTOR:", "REPRESENTANTE LEGAL:",
            "NIT:", "CIUDAD:", "FECHA:", "PROPONENTE:", "SUBTOTAL"
        ]
        return any(keyword in upper_line for keyword in garbage_keywords)

    def _is_potential_description(self, line: str) -> bool:
        """Determina si una línea podría ser una descripción de APU."""
        return line.count(';') < 2 and not line.replace('.', '', 1).isdigit()

    def _is_valid_mo_data(self, data: Dict[str, str]) -> bool:
        """Valida que los datos de mano de obra extraídos sean coherentes."""
        return all(data.get(k) for k in ["valor_total", "jornal_total", "rendimiento"])

    def _is_valid_insumo_data(self, data: Dict[str, str]) -> bool:
        """Valida que los datos de insumo extraídos sean coherentes."""
        return all(data.get(k) for k in ["descripcion", "valor_total"])

    def _has_data_structure(self, line: str) -> bool:
        """Comprueba si la línea parece tener datos por la cantidad de separadores."""
        return line.count(';') >= 5

    def _start_new_apu(self, raw_code: str, unit: Optional[str]):
        cleaned_code = clean_apu_code(raw_code)
        if not cleaned_code:
            self.context["apu_code"] = None
            return

        self.context["apu_code"] = cleaned_code
        self.context["apu_desc"] = self.potential_apu_desc
        self.context["apu_unit"] = unit.strip() if unit else "INDEFINIDO"
        self.context["category"] = "INDEFINIDO"
        self.potential_apu_desc = ""
        self.stats["items_found"] += 1

    def _parse_insumo(self, data: Dict[str, str]):
        descripcion = data["descripcion"].strip()
        cantidad = self._to_numeric_safe(data["cantidad"])
        valor_total = self._to_numeric_safe(data["valor_total"])
        precio_unit = self._to_numeric_safe(data["precio_unit"])

        if not self._should_add_insumo(descripcion, cantidad, valor_total):
            return

        if cantidad == 0 and valor_total > 0 and precio_unit > 0:
            cantidad = valor_total / precio_unit

        self.apus_data.append({
            "CODIGO_APU": self.context["apu_code"],
            "DESCRIPCION_APU": self.context["apu_desc"],
            "UNIDAD_APU": self.context["apu_unit"],
            "DESCRIPCION": descripcion,
            "UNIDAD": data["unidad"].strip(),
            "CANTIDAD": cantidad,
            "VR_UNITARIO": precio_unit,
            "VR_TOTAL": valor_total,
            "CATEGORIA": self.context["category"],
            "RENDIMIENTO": 0.0,
        })
        self.stats["insumos_parsed"] += 1

    def _parse_mano_de_obra(self, data: Dict[str, str]):
        valor_total = self._to_numeric_safe(data["valor_total"])
        jornal_total = self._to_numeric_safe(data["jornal_total"])
        rendimiento = self._to_numeric_safe(data["rendimiento"])
        descripcion = data["descripcion"].strip()

        cantidad = self._calculate_mo_quantity(valor_total, jornal_total, rendimiento)

        if not self._should_add_insumo(descripcion, cantidad, valor_total):
            return

        self.apus_data.append({
            "CODIGO_APU": self.context["apu_code"],
            "DESCRIPCION_APU": self.context["apu_desc"],
            "UNIDAD_APU": self.context["apu_unit"],
            "DESCRIPCION": descripcion,
            "UNIDAD": "JOR",
            "CANTIDAD": cantidad,
            "VR_UNITARIO": jornal_total,
            "VR_TOTAL": valor_total,
            "CATEGORIA": "MANO DE OBRA",
            "RENDIMIENTO": rendimiento,
        })
        self.stats["mo_parsed"] += 1

    def _to_numeric_safe(self, s: Optional[str]) -> float:
        if not s:
            return 0.0
        s_cleaned = s.replace(" ", "").replace(".", "").replace(",", ".").strip()
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
