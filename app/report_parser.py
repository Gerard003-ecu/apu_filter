# En app/report_parser.py
# REEMPLAZA LA CLASE ReportParser COMPLETA CON ESTO:

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
        "item_code": re.compile(r"ITEM:\s*([^;]+)(?:;UNIDAD:\s*([A-Z0-9/%]+))?.*$"),
        "insumo_full": re.compile(
            r"^(?P<descripcion>[^;]+);"
            r"(?P<unidad>[A-Z0-9/%]+);"
            r"(?P<cantidad>[\d.,\s]+);"
            r"(?P<desperdicio>[\d.,\s-]*);" # Hacer desperdicio opcional
            r"(?P<precio_unit>[\d.,\s]+);"
            r"(?P<valor_total>[\d.,\s]+)" # No anclar al final
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
            "apu_desc": None,
            "apu_unit": "INDEFINIDO", # Nuevo campo para la unidad del APU
            "category": "INDEFINIDO",
        }
        self.potential_apu_desc = ""

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
            logger.warning("No se extrajeron datos de APU, devolviendo DataFrame vacío.")
            return pd.DataFrame()

        df = pd.DataFrame(self.apus_data)
        # Estandarizar nombres de columnas para consistencia
        df.rename(columns={
            "DESCRIPCION": "DESCRIPCION_INSUMO",
            "CANTIDAD": "CANTIDAD_APU",
            "VR_UNITARIO": "PRECIO_UNIT_APU",
            "VR_TOTAL": "VALOR_TOTAL_APU",
            "UNIDAD": "UNIDAD_INSUMO"
        }, inplace=True)

        df["NORMALIZED_DESC"] = self._normalize_text(df["DESCRIPCION_INSUMO"])

        # Log del DataFrame parseado para depuración
        # Muestra las primeras 5 filas del dataframe parseado
        # logger.debug(f"DataFrame parseado para pruebas:\n{df.head().to_string()}")

        return df

    def _process_line(self, line: str):
        line = line.strip()
        if not line:
            return

        # --- Lógica de Máquina de Estados ---

        # 1. ¿Es una línea de ITEM? (Máxima prioridad)
        match_item = self.PATTERNS["item_code"].search(line.upper())
        if match_item:
            raw_code = match_item.group(1).strip()
            unit = match_item.group(2) # Puede ser None si no se encuentra
            self._start_new_apu(raw_code, unit)
            return

        # 2. ¿Es un encabezado de CATEGORÍA?
        category_keywords = {"MATERIALES", "MANO DE OBRA", "EQUIPO", "OTROS"}
        # Limpiamos la línea para la comparación
        first_part = line.split(';')[0].strip().upper()
        if first_part in category_keywords:
            self.context["category"] = first_part
            return

        # 3. ¿Es una línea de DATOS? (Solo si ya tenemos un APU activo)
        if self.context["apu_code"]:
            match_mo = self.PATTERNS["mano_de_obra"].match(line)
            if match_mo:
                self._parse_mano_de_obra(match_mo.groupdict())
                return

            match_insumo = self.PATTERNS["insumo_full"].match(line)
            if match_insumo:
                self._parse_insumo(match_insumo.groupdict())
                return

        # 4. Si no es nada de lo anterior, es una potencial DESCRIPCIÓN de APU
        # La guardamos para usarla cuando encontremos el próximo "ITEM:"
        first_part = line.split(';')[0].strip()
        is_subtotal = "SUBTOTAL" in first_part.upper()
        is_costo_directo = "COSTO DIRECTO" in first_part.upper()
        if first_part and not is_subtotal and not is_costo_directo:
            self.potential_apu_desc = first_part

    def _start_new_apu(self, raw_code: str, unit: Optional[str]):
        """
        Limpia y establece el código del APU para el nuevo bloque.
        Si el código no es válido, se establece a None para detener la asignación de insumos.
        """
        cleaned_code = clean_apu_code(raw_code)

        # Si el código está vacío después de la limpieza, es un APU de plantilla sin ITEM.
        # Invalida el APU actual para evitar la contaminación de datos.
        if not cleaned_code:
            self.context["apu_code"] = None
        else:
            self.context["apu_code"] = cleaned_code
            self.context["apu_desc"] = self.potential_apu_desc
            self.context["apu_unit"] = unit.strip() if unit else "INDEFINIDO"

        self.context["category"] = "INDEFINIDO"

    def _parse_insumo(self, data: Dict[str, str]):
        valor_total = self._to_numeric_safe(data["valor_total"])
        precio_unit = self._to_numeric_safe(data["precio_unit"])
        cantidad = self._to_numeric_safe(data["cantidad"])

        # Calcular cantidad implícita si es cero pero los costos no lo son
        if cantidad == 0 and valor_total > 0 and precio_unit > 0:
            cantidad = valor_total / precio_unit
            logger.debug(f"Cantidad implícita calculada para '{data['descripcion']}': {cantidad}")

        self.apus_data.append({
            "CODIGO_APU": self.context["apu_code"],
            "DESCRIPCION_APU": self.context["apu_desc"],
            "UNIDAD_APU": self.context["apu_unit"],
            "DESCRIPCION": data["descripcion"].strip(),
            "UNIDAD": data["unidad"],
            "CANTIDAD": cantidad,
            "VR_UNITARIO": precio_unit,
            "VR_TOTAL": valor_total,
            "CATEGORIA": self.context["category"],
            "RENDIMIENTO": 0.0,  # Rendimiento no aplica para insumos generales
        })

    def _parse_mano_de_obra(self, data: Dict[str, str]):
        valor_total = self._to_numeric_safe(data["valor_total"])
        jornal_total = self._to_numeric_safe(data["jornal_total"])
        cantidad = valor_total / jornal_total if jornal_total > 0 else 0

        self.apus_data.append(
            {
                "CODIGO_APU": self.context["apu_code"],
                "DESCRIPCION_APU": self.context["apu_desc"],
                "UNIDAD_APU": self.context["apu_unit"],
                "DESCRIPCION": data["descripcion"].strip(),
                "UNIDAD": "JOR",
                "CANTIDAD": cantidad,
                "VR_UNITARIO": jornal_total,
                "VR_TOTAL": valor_total,
                "CATEGORIA": "MANO DE OBRA",
                "RENDIMIENTO": self._to_numeric_safe(data["rendimiento"]),
            }
        )

    def _to_numeric_safe(self, s: Optional[str]) -> float:
        # ... (función de limpieza numérica como la tienes)
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
        # ... (función de normalización como la tienes)
        from unidecode import unidecode
        normalized = series.astype(str).str.lower().str.strip()
        normalized = normalized.apply(unidecode)
        normalized = normalized.str.replace(r"[^a-z0-9\s#\-]", "", regex=True)
        normalized = normalized.str.replace(r"\s+", " ", regex=True)
        return normalized
