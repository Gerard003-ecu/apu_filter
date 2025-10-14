import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from unidecode import unidecode

from .utils import clean_apu_code

logger = logging.getLogger(__name__)


class ReportParser:
    PATTERNS = {
        "item_code": re.compile(r"ITEM:\s*([^;]+)", re.IGNORECASE),
        "insumo_full": re.compile(
            r"^(?P<descripcion>[^;]+);"
            r"(?P<unidad>[^;]*);"
            r"(?P<cantidad>[^;]*);"
            r"(?P<desperdicio>[^;]*);"
            r"(?P<precio_unit>[^;]*);"
            r"(?P<valor_total>[^;]*)",
            re.IGNORECASE,
        ),
        "mano_de_obra_compleja": re.compile(
            r"^(?P<descripcion>(?:M\.O\.|MANO DE OBRA|SISO|INGENIERO|OFICIAL|AYUDANTE"
            r"|MAESTRO|CAPATAZ|CUADRILLA|OBRERO).*?);"
            r"(?P<jornal_base>[\d.,\s]+);"
            r"(?P<prestaciones>[\d%.,\s]+);"
            r"(?P<jornal_total>[\d.,\s]+);"
            r"(?P<rendimiento>[\d.,\s]+);"
            r"(?P<valor_total>[\d.,\s]+)",
            re.IGNORECASE,
        ),
        "mano_de_obra_simple": re.compile(
            r"^(?P<descripcion>(?:M\.O\.|MANO DE OBRA|SISO|INGENIERO|OFICIAL|AYUDANTE"
            r"|MAESTRO|CAPATAZ|CUADRILLA|OBRERO).*?);"
            r"[^;]*;"
            r"(?P<cantidad>[^;]*);"
            r"[^;]*;"
            r"(?P<precio_unit>[^;]*);"
            r"(?P<valor_total>[^;]*)",
            re.IGNORECASE,
        ),
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
        self.apus_data: List[Dict[str, Any]] = []
        self.context = {
            "apu_code": None, "apu_desc": "", "apu_unit": "", "category": "INDEFINIDO",
        }
        self.potential_apu_desc = ""
        self.stats = {
            "total_lines": 0, "processed_lines": 0, "items_found": 0,
            "insumos_parsed": 0, "mo_compleja_parsed": 0, "mo_simple_parsed": 0,
            "garbage_lines": 0, "unparsed_data_lines": 0, "fallback_parsed": 0
        }

    def parse(self) -> pd.DataFrame:
        logger.info(f"🔍 Iniciando parsing del archivo: {self.file_path}")
        logger.info(f"📊 Contexto inicial: {self.context}")

        try:
            with open(self.file_path, "r", encoding="latin1") as f:
                for line_num, line in enumerate(f, 1):
                    self.stats["total_lines"] += 1
                    logger.debug(f"--- Línea {line_num} ---")
                    self._process_line(line, line_num)
                    logger.debug(f"Contexto actual: {self.context}")

        except Exception as e:
            logger.error(f"❌ Error al parsear {self.file_path}: {e}", exc_info=True)
            return pd.DataFrame()

        self._log_parsing_stats()
        if not self.apus_data:
            logger.warning("⚠️ No se extrajeron datos de APU, devolviendo DataFrame vacío.")
            return pd.DataFrame()

        df = self._build_dataframe()
        logger.info(f"✅ DataFrame generado con {len(df)} registros")
        return df

    def _is_potential_description(self, line: str) -> bool:
        """
        Determina si una línea podría ser una descripción de APU.
        Debe tener texto significativo y no parecer una línea de datos o basura.
        """
        if self._has_data_structure(line) or self._is_garbage_line(line):
            return False
        # Una descripción debe tener al menos una palabra de 3 o más letras
        return re.search(r"[a-zA-Z]{3,}", line) is not None

    def _process_line(self, line: str, line_num: int):
        """
        VERSIÓN REPARADA: Garantiza que las categorías se detecten ANTES de los insumos
        """
        line = line.strip()
        if not line:
            self.context["apu_code"] = None
            self.potential_apu_desc = ""
            logger.debug("🔄 Contexto de APU reseteado por línea vacía.")
            return

        self.stats["processed_lines"] += 1

        # 1. Filtrado de basura (siempre primero)
        if self._is_garbage_line(line):
            self.stats["garbage_lines"] += 1
            return

        upper_line = line.upper()

        # 2. ✨ NUEVO FILTRO ESPECÍFICO: IMPUESTOS Y RETENCIONES o POLIZAS
        if "IMPUESTOS Y RETENCIONES" in upper_line or "POLIZAS" in upper_line:
            self.stats["garbage_lines"] += 1
            logger.debug(
                f"🚫 Línea ignorada por contener IMPUESTOS/POLIZAS: {line[:50]}..."
            )
            return

        # 3. Detección de ITEM (máxima prioridad)
        match_item = self.PATTERNS["item_code"].search(upper_line)
        if match_item:
            raw_code = match_item.group(1).strip()
            unit_match = re.search(r"UNIDAD:\s*([A-Z0-9/%]+)", upper_line)
            unit = unit_match.group(1) if unit_match else "INDEFINIDO"

            logger.debug(
                f"🆕 ITEM detectado (L{line_num}): código='{raw_code}', unidad='{unit}'"
            )
            self._start_new_apu(raw_code, unit)
            return

        # 4. Lógica dependiente del estado
        if not self.context["apu_code"]:
            # ESTADO INACTIVO: Solo buscar descripción
            if self._is_potential_description(line):
                self.potential_apu_desc = line.split(';', 1)[0].strip()
            return

        # --- ESTADO ACTIVO: APU en progreso ---
        # ORDEN CRÍTICO REPARADO:

        # A. PRIMERO: Intentar detectar categoría (¡ESTO FALTA!)
        category_detected = self._try_detect_category_change(line, upper_line)
        if category_detected:
            return  # ¡Categoría detectada! No procesar como dato

        # B. SEGUNDO: Intentar parsear como dato
        if self._try_parse_as_data_line(line, line_num):
            return

        # C. TERCERO: Considerar como descripción potencial
        if self._is_potential_description(line):
            self.potential_apu_desc = line.split(';', 1)[0].strip()

    def _try_parse_as_data_line(self, line: str, line_num: int) -> bool:
        """Versión con logs de diagnóstico"""
        has_data_structure = self._has_data_structure(line)
        current_category = self.context["category"]

        logger.debug(
            f"🔍 Analizando línea {line_num}: cat='{current_category}', "
            f"datos={has_data_structure}"
        )
        logger.debug(f"   Contenido: {line[:80]}...")

        # ... resto de la lógica de parsing ...
        match_mo_compleja = self.PATTERNS["mano_de_obra_compleja"].match(line)
        if match_mo_compleja:
            self._parse_mano_de_obra_compleja(match_mo_compleja.groupdict())
            return True

        match_mo_simple = self.PATTERNS["mano_de_obra_simple"].match(line)
        if match_mo_simple:
            self._parse_mano_de_obra_simple(match_mo_simple.groupdict())
            return True

        match_insumo = self.PATTERNS["insumo_full"].match(line)
        if match_insumo:
            self._parse_insumo(match_insumo.groupdict())
            return True

        if self._has_data_structure(line):
            if self._try_fallback_parsing(line, line_num):
                return True

        processed = (
            match_mo_compleja
            or match_mo_simple
            or match_insumo
            or self._has_data_structure(line)
        )
        if not processed:
            logger.debug("   ❌ No se pudo parsear como dato")
        return processed

    def _try_detect_category_change(self, line: str, upper_line: str) -> bool:
        """
        Detecta si una línea es un cambio de categoría.
        Una línea de categoría contiene una palabra clave de categoría y no es una
        línea de datos.
        """
        line_clean = line.strip()
        if not line_clean:
            return False

        # Extraer la primera parte de la línea, que podría ser la categoría
        first_part = line_clean.split(';')[0].strip().upper()

        category_mappings = {
            "MATERIALES": [
                "MATERIALES", "MATERIALES Y SUMINISTROS", "MATERIAL",
                "MATERIALES Y ACCESORIOS", "SUMINISTROS"
            ],
            "MANO DE OBRA": [
                "MANO DE OBRA", "MANO DE OBRA DIRECTA", "M.O.",
                "MO", "MANO DE OBRA Y SUPERVISIÓN"
            ],
            "EQUIPO": [
                "EQUIPO", "EQUIPOS", "EQUIPOS Y HERRAMIENTAS",
                "HERRAMIENTAS", "MAQUINARIA"
            ],
            "TRANSPORTE": ["TRANSPORTE", "TRANSPORTES", "FLETE"],
            "OTROS": ["OTROS", "OTROS GASTOS", "GASTOS GENERALES", "INDIRECTOS"]
        }

        found_category = None
        # Buscar si la primera parte coincide con alguna palabra clave de categoría
        for category, keywords in category_mappings.items():
            if first_part in keywords:
                found_category = category
                break

        if not found_category:
            return False

        # Tenemos una coincidencia potencial. Ahora, hay que asegurarse de que no es
        # una línea de datos.
        # Una línea de datos real (insumo, mano de obra) tendrá valores numéricos para
        # cantidad, precio, valor, etc.
        # Una línea de categoría como "MATERIALES;;;;" no los tendrá.

        # Usamos la expresión regular de insumo para ver si coincide
        match = self.PATTERNS["insumo_full"].match(line_clean)
        if match:
            data = match.groupdict()
            # Si hay valores numéricos significativos, es una línea de datos, no un
            # encabezado de categoría.
            if (self._to_numeric_safe(data["cantidad"]) > 0 or
                self._to_numeric_safe(data["precio_unit"]) > 0 or
                self._to_numeric_safe(data["valor_total"]) > 0):
                # Es una línea de datos que casualmente empieza con una palabra de categoría
                # ej: "EQUIPO DE SEGURIDAD;UND;1;..."
                logger.debug(
                    f"Línea '{line_clean[:30]}...' parece categoría pero tiene "
                    f"datos, se ignora."
                )
                return False

        # Si llegamos aquí, es porque la línea empieza con una palabra clave de categoría y
        # no parece tener datos numéricos. Es un cambio de categoría.
        return self._update_category(found_category, line_clean)

    def _update_category(self, new_category: str, line: str) -> bool:
        """Actualiza la categoría y registra el cambio."""
        old_category = self.context["category"]
        if old_category != new_category:
            self.context["category"] = new_category
            logger.debug(
                f"📂 Categoría cambiada: {old_category} -> {new_category} "
                f"(línea: '{line}')"
            )
        return True

    def _try_fallback_parsing(self, line: str, line_num: int) -> bool:
        match = self.PATTERNS["generic_data"].match(line)
        if not match:
            return False
        data = match.groupdict()
        desc = data["descripcion"].strip()
        if self._looks_like_mo(desc):
            return False
        vals = [
            self._to_numeric_safe(v) for v in data.values() if isinstance(v, str)
        ]
        vals = [v for v in vals if v > 0]
        if len(vals) >= 2:
            vals.sort(reverse=True)
            valor_total, precio_unit = vals[0], vals[1]
            cantidad = valor_total / precio_unit if precio_unit > 0 else 0
            if self._should_add_insumo(desc, cantidad, valor_total):
                self._add_apu_data(
                    descripcion=desc,
                    unidad="UND",
                    cantidad=cantidad,
                    precio_unit=precio_unit,
                    valor_total=valor_total,
                    rendimiento=0.0,
                    formato="FALLBACK",
                    categoria=self.context["category"],
                )
                self.stats["fallback_parsed"] += 1
                logger.debug(f"🔄 Fallback exitoso (L{line_num}): {desc[:50]}...")
                return True
        return False

    def _start_new_apu(self, raw_code: str, unit: str):
        cleaned_code = clean_apu_code(raw_code)
        if not cleaned_code:
            logger.warning(f"⚠️ Código APU no válido: '{raw_code}'")
            self.context["apu_code"] = None
            return
        self.context = {
            "apu_code": cleaned_code,
            "apu_desc": self.potential_apu_desc,
            "apu_unit": unit.strip(),
            "category": "INDEFINIDO",
        }
        self.potential_apu_desc = ""
        self.stats["items_found"] += 1
        logger.debug(
            f"✅ Nuevo APU iniciado: {cleaned_code} - {self.context['apu_desc']}"
        )

    def _parse_insumo(self, data: Dict[str, str]):
        desc = data["descripcion"].strip()
        cantidad = self._to_numeric_safe(data["cantidad"])
        valor_total = self._to_numeric_safe(data["valor_total"])
        precio_unit = self._to_numeric_safe(data["precio_unit"])
        if cantidad == 0 and valor_total > 0 and precio_unit > 0:
            cantidad = valor_total / precio_unit
        if self._should_add_insumo(desc, cantidad, valor_total):
            self._add_apu_data(
                descripcion=desc,
                unidad=data["unidad"].strip(),
                cantidad=cantidad,
                precio_unit=precio_unit,
                valor_total=valor_total,
                rendimiento=0.0,
                formato="INSUMO_GENERAL",
                categoria=self.context["category"],
            )
            self.stats["insumos_parsed"] += 1
            logger.debug(f"✅ Insumo agregado: {desc[:50]}...")

    def _parse_mano_de_obra_compleja(self, data: Dict[str, str]):
        desc = data["descripcion"].strip()
        valor_total = self._to_numeric_safe(data["valor_total"])
        jornal_total = self._to_numeric_safe(data["jornal_total"])
        rendimiento = self._to_numeric_safe(data["rendimiento"])
        cantidad = self._calculate_mo_quantity(valor_total, jornal_total)
        if self._should_add_insumo(desc, cantidad, valor_total):
            self._add_apu_data(
                descripcion=desc,
                unidad="JOR",
                cantidad=cantidad,
                precio_unit=jornal_total,
                valor_total=valor_total,
                rendimiento=rendimiento,
                formato="MO_COMPLEJA",
                categoria="MANO DE OBRA",
            )
            self.stats["mo_compleja_parsed"] += 1
            logger.debug(f"✅ MO Compleja agregada: {desc[:50]}...")

    def _parse_mano_de_obra_simple(self, data: Dict[str, str]):
        desc = data["descripcion"].strip()
        cantidad = self._to_numeric_safe(data["cantidad"])
        precio_unit = self._to_numeric_safe(data["precio_unit"])
        valor_total = self._to_numeric_safe(data["valor_total"])
        if cantidad == 0 and valor_total > 0 and precio_unit > 0:
            cantidad = valor_total / precio_unit
        rendimiento = self._calculate_rendimiento_simple(valor_total, precio_unit)
        if self._should_add_insumo(desc, cantidad, valor_total):
            self._add_apu_data(
                descripcion=desc,
                unidad="JOR",
                cantidad=cantidad,
                precio_unit=precio_unit,
                valor_total=valor_total,
                rendimiento=rendimiento,
                formato="MO_SIMPLE",
                categoria="MANO DE OBRA",
            )
            self.stats["mo_simple_parsed"] += 1
            logger.debug(f"✅ MO Simple agregada: {desc[:50]}...")

    def _add_apu_data(self, **kwargs):
        base_data = {
            "CODIGO_APU": self.context["apu_code"],
            "DESCRIPCION_APU": self.context["apu_desc"],
            "UNIDAD_APU": self.context["apu_unit"],
        }

        descripcion_insumo = kwargs["descripcion"]

        record = {
            "DESCRIPCION_INSUMO": descripcion_insumo,
            "UNIDAD_INSUMO": kwargs["unidad"],
            "CANTIDAD_APU": round(kwargs["cantidad"], 6),
            "PRECIO_UNIT_APU": round(kwargs["precio_unit"], 2),
            "VALOR_TOTAL_APU": round(kwargs["valor_total"], 2),
            "CATEGORIA": kwargs["categoria"],
            "RENDIMIENTO": round(kwargs["rendimiento"], 6),
            "FORMATO_ORIGEN": kwargs["formato"],
            "NORMALIZED_DESC": self._normalize_text_single(descripcion_insumo),
        }
        self.apus_data.append({**base_data, **record})

    def _to_numeric_safe(self, s: Optional[str]) -> float:
        if not s or not isinstance(s, str):
            return 0.0
        s_cleaned = s.strip().replace("$", "").replace(" ", "")
        if not s_cleaned:
            return 0.0

        # Heurística para manejar formatos numéricos inconsistentes
        if "," in s_cleaned:
            # Si la coma existe, es el separador decimal. Los puntos son de miles.
            s_cleaned = s_cleaned.replace(".", "")
            s_cleaned = s_cleaned.replace(",", ".")
        elif s_cleaned.count('.') > 1:
            # Múltiples puntos sin comas: son separadores de miles.
            s_cleaned = s_cleaned.replace(".", "")
        elif s_cleaned.count('.') == 1:
            # Un solo punto: puede ser decimal o de miles (ej. 80.000).
            integer_part, fractional_part = s_cleaned.split('.')
            if len(fractional_part) == 3 and integer_part != "0":
                # Probablemente es un separador de miles, como en "80.000"
                s_cleaned = integer_part + fractional_part
            # De lo contrario, se asume que es un punto decimal (ej. 0.125, 123.45)

        try:
            return float(s_cleaned)
        except (ValueError, TypeError):
            return 0.0

    def _should_add_insumo(
        self, desc: str, cantidad: float, valor_total: float
    ) -> bool:
        if not desc or len(desc.strip()) < 2:
            return False
        return not (cantidad <= 0 and valor_total <= 0)

    def _looks_like_mo(self, line: str) -> bool:
        mo_keywords = [
            "M.O.",
            "MANO DE OBRA",
            "SISO",
            "INGENIERO",
            "OFICIAL",
            "OBRERO",
            "AYUDANTE",
            "MAESTRO",
            "TOPOGRAFO",
            "CAPATAZ",
            "CUADRILLA",
        ]
        return any(keyword in line.upper() for keyword in mo_keywords)

    def _calculate_mo_quantity(self, valor_total: float, jornal_total: float) -> float:
        if jornal_total <= 0:
            return 0.0
        return valor_total / jornal_total

    def _calculate_rendimiento_simple(
        self, valor_total: float, precio_unit: float
    ) -> float:
        if valor_total <= 0 or precio_unit <= 0:
            return 0.0
        return precio_unit / valor_total

    def _is_garbage_line(self, line: str) -> bool:
        upper_line = line.upper()
        return any(
            kw in upper_line
            for kw in [
                "FORMATO DE ANÁLISIS",
                "COSTOS DIRECTOS",
                "PRESUPUESTO OFICIAL",
                "REPRESENTANTE LEGAL",
                "SUBTOTAL",
                "PÁGINA",
                "===",
            ]
        )

    def _has_data_structure(self, line: str) -> bool:
        return line.count(";") >= 2

    def _build_dataframe(self) -> pd.DataFrame:
        if not self.apus_data:
            return pd.DataFrame()
        return pd.DataFrame(self.apus_data)

    def _normalize_text_single(self, text: str) -> str:
        """Normaliza un único string de texto."""
        if not isinstance(text, str):
            text = str(text)

        normalized = text.lower().strip()
        normalized = unidecode(normalized)
        normalized = re.sub(r"[^a-z0-9\s#\-]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _log_parsing_stats(self):
        logger.info("📊 MÉTRICAS FINALES DE PARSING:")
        for key, value in self.stats.items():
            logger.info(f"   {key}: {value}")
        total_parsed = sum(
            [
                self.stats[k]
                for k in [
                    "insumos_parsed",
                    "mo_compleja_parsed",
                    "mo_simple_parsed",
                    "fallback_parsed",
                ]
            ]
        )
        if self.stats["processed_lines"] > 0:
            success_rate = total_parsed / self.stats["processed_lines"] * 100
            logger.info(f"   TASA_ÉXITO_PARSE: {success_rate:.1f}%")
