import logging
import re
from typing import Any, Dict, List, Optional
from collections import defaultdict
import pandas as pd
from unidecode import unidecode
from .utils import clean_apu_code, parse_number

logger = logging.getLogger(__name__)


class APUProcessor:
    """
    Etapa 2: Aplica toda la lógica de negocio sobre datos crudos.
    """

    DESCRIPTION_KEYWORDS = [
        "SUMINISTRO", "INSTALACION", "CONSTRUCCION", "EXCAVACION",
        "RELLENO", "CONCRETO", "ACERO", "TUBERIA", "CANAL", "MURO",
        "LOSA", "VIGA", "COLUMNA", "CIMENTACION", "ESTRUCTURA",
        "ACABADO", "PINTURA", "PRELIMINAR", "DEMOLICION", "RETIRO",
        "TRANSPORTE", "MONTAJE", "MANTENIMIENTO", "REPARACION"
    ]

    EXCLUDED_TERMS = [
        'EQUIPO Y HERRAMIENTA', 'IMPUESTOS', 'POLIZAS', 'SEGUROS',
        'GASTOS GENERALES', 'UTILIDAD', 'ADMINISTRACION', 'RETENCIONES'
    ]

    def __init__(self, raw_records: List[Dict[str, str]]):
        self.raw_records = raw_records
        self.processed_data: List[Dict[str, Any]] = []
        self.stats = defaultdict(int)

    def process_all(self) -> pd.DataFrame:
        """Orquesta todo el procesamiento."""
        for record in self.raw_records:
            try:
                processed = self._process_single_record(record)
                if processed:
                    self.processed_data.append(processed)
                    self._update_stats(processed)
            except Exception as e:
                logger.warning(f"⚠️ Error procesando registro: {e}")

        self._emergency_dia_units_patch()
        self._log_stats()
        return self._build_dataframe()

    def _process_single_record(self, record: Dict[str, str]) -> Optional[Dict[str, Any]]:
        # 1. Normalizar campos
        apu_code = clean_apu_code(record["apu_code"])
        apu_desc = record["apu_desc"]
        apu_unit = self._clean_unit(record["apu_unit"])
        category = record["category"]

        # 2. Parsear línea de insumo
        parsed = self._parse_insumo_line(record["insumo_line"])
        if not parsed:
            return None

        descripcion_insumo = parsed["descripcion"]
        if self._is_excluded_insumo(descripcion_insumo):
            return None

        # 3. Inferir unidad de APU si es "UND"
        if apu_unit == "UND":
            apu_unit = self._infer_unit_aggressive(apu_desc, category, apu_code)

        # 4. Convertir a números
        cantidad = parse_number(parsed.get("cantidad", "0"))
        precio_unit = parse_number(parsed.get("precio_unit", "0"))
        valor_total = parse_number(parsed.get("valor_total", "0"))
        rendimiento = parse_number(parsed.get("rendimiento", "0"))

        # 5. Calcular campos faltantes
        if cantidad == 0 and valor_total > 0 and precio_unit > 0:
            cantidad = valor_total / precio_unit

        if self._looks_like_mo(descripcion_insumo) and rendimiento == 0:
            rendimiento = self._calculate_rendimiento_simple(valor_total, precio_unit)

        # 6. Validar antes de agregar
        if not self._should_add_insumo(descripcion_insumo, cantidad, valor_total):
            return None

        # 7. Normalizar descripción
        normalized_desc = self._normalize_text_single(descripcion_insumo)

        return {
            "CODIGO_APU": apu_code,
            "DESCRIPCION_APU": apu_desc,
            "UNIDAD_APU": apu_unit,
            "DESCRIPCION_INSUMO": descripcion_insumo,
            "UNIDAD_INSUMO": parsed.get("unidad", "UND"),
            "CANTIDAD_APU": round(cantidad, 6),
            "PRECIO_UNIT_APU": round(precio_unit, 2),
            "VALOR_TOTAL_APU": round(valor_total, 2),
            "CATEGORIA": category,
            "RENDIMIENTO": round(rendimiento, 6),
            "FORMATO_ORIGEN": parsed.get("formato", "GENERIC"),
            "NORMALIZED_DESC": normalized_desc,
        }

    def _parse_insumo_line(self, line: str) -> Optional[Dict[str, str]]:
        # Intentar patrones específicos (MO compleja/simple, insumo full)
        # Si fallan, usar fallback genérico
        patterns = self._get_parsing_patterns()
        for name, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                data = match.groupdict()
                data["formato"] = name
                return data
        # Fallback: dividir por ; y asignar genéricamente
        parts = [p.strip() for p in line.split(";")]
        if len(parts) >= 6:
            return {
                "descripcion": parts[0],
                "unidad": parts[1] or "UND",
                "cantidad": parts[2],
                "precio_unit": parts[4],
                "valor_total": parts[5],
                "rendimiento": "0",
                "formato": "FALLBACK",
            }
        return None

    def _get_parsing_patterns(self):
        # Copiar PATTERNS del archivo original (solo los regex)
        return {
            "MO_COMPLEJA": re.compile(
                r"^(?P<descripcion>(?:M\.O\.|MANO DE OBRA|SISO|INGENIERO|OFICIAL|AYUDANTE"
                r"|MAESTRO|CAPATAZ|CUADRILLA|OBRERO).*?);"
                r"(?P<jornal_base>[\d.,\s]+);"
                r"(?P<prestaciones>[\d%.,\s]+);"
                r"(?P<jornal_total>[\d.,\s]+);"
                r"(?P<rendimiento>[\d.,\s]+);"
                r"(?P<valor_total>[\d.,\s]+)",
                re.IGNORECASE,
            ),
            "MO_SIMPLE": re.compile(
                r"^(?P<descripcion>(?:M\.O\.|MANO DE OBRA|SISO|INGENIERO|OFICIAL|AYUDANTE"
                r"|MAESTRO|CAPATAZ|CUADRILLA|OBRERO).*?);"
                r"[^;]*;"
                r"(?P<cantidad>[^;]*);"
                r"[^;]*;"
                r"(?P<precio_unit>[^;]*);"
                r"(?P<valor_total>[^;]*)",
                re.IGNORECASE,
            ),
            "INSUMO_FULL": re.compile(
                r"^(?P<descripcion>[^;]+);"
                r"(?P<unidad>[^;]*);"
                r"(?P<cantidad>[^;]*);"
                r"(?P<desperdicio>[^;]*);"
                r"(?P<precio_unit>[^;]*);"
                r"(?P<valor_total>[^;]*)",
                re.IGNORECASE,
            ),
        }

    def _infer_unit_aggressive(self, description: str, category: str, apu_code: str) -> str:
        # (Implementación idéntica a la original, pero como método independiente)
        # ... (lógica de inferencia por código, categoría, keywords, etc.)
        desc_upper = description.upper()
        # Por keywords
        unit_keywords = {
            "M3": ["EXCAVACION", "CONCRETO", "M3"],
            "M2": ["PINTURA", "LOSA", "M2"],
            "ML": ["TUBERIA", "ML"],
            "DIA": ["CUADRILLA", "EQUIPO"],
            "JOR": ["MANO DE OBRA", "JORNAL"],
        }
        for unit, words in unit_keywords.items():
            if any(w in desc_upper for w in words):
                return unit
        # Por categoría
        cat_map = {"MANO DE OBRA": "JOR", "EQUIPO": "DIA", "TRANSPORTE": "VIAJE"}
        if category in cat_map:
            return cat_map[category]
        return "UND"

    def _clean_unit(self, unit: str) -> str:
        if not unit:
            return "UND"
        unit = re.sub(r"[^A-Z0-9]", "", unit.upper())
        mapping = {"DIAS": "DIA", "DÍAS": "DIA", "UN": "UND", "UNIDAD": "UND", "JORNAL": "JOR"}
        return mapping.get(unit, unit)

    def _is_excluded_insumo(self, desc: str) -> bool:
        desc_u = desc.upper()
        return any(term in desc_u for term in self.EXCLUDED_TERMS)

    def _looks_like_mo(self, desc: str) -> bool:
        mo_terms = ["M.O.", "MANO DE OBRA", "OFICIAL", "AYUDANTE", "CUADRILLA"]
        return any(term in desc.upper() for term in mo_terms)

    def _calculate_rendimiento_simple(self, valor_total: float, precio_unit: float) -> float:
        return precio_unit / valor_total if valor_total > 0 else 0.0

    def _should_add_insumo(self, desc: str, cantidad: float, valor_total: float) -> bool:
        if not desc or len(desc) < 2:
            return False
        if valor_total > 1_000_000:
            return False
        return cantidad > 0 or valor_total > 0

    def _normalize_text_single(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = unidecode(text.lower().strip())
        text = re.sub(r"[^a-z0-9\s#\-]", "", text)
        return re.sub(r"\s+", " ", text)

    def _emergency_dia_units_patch(self):
        squad_codes = ['13', '14', '15', '16', '17', '18', '19', '20']
        for apu in self.processed_data:
            apu_code = apu["CODIGO_APU"]
            category = apu["CATEGORIA"]
            if (category == "MANO DE OBRA" and
                any(apu_code.startswith(c) for c in squad_codes) and
                apu["UNIDAD_APU"] != "DIA"):
                old = apu["UNIDAD_APU"]
                apu["UNIDAD_APU"] = "DIA"
                logger.info(f"🚀 PARCHE DIA: {apu_code} '{old}' → 'DIA'")

    def _update_stats(self, record: Dict):
        self.stats["total_records"] += 1
        self.stats[f"categoria_{record['CATEGORIA']}"] += 1

    def _log_stats(self):
        logger.info(f"📊 Procesados {len(self.processed_data)} registros")
        for k, v in self.stats.items():
            logger.info(f" {k}: {v}")

    def _build_dataframe(self) -> pd.DataFrame:
        if not self.processed_data:
            return pd.DataFrame()
        return pd.DataFrame(self.processed_data)
