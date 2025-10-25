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

    def _process_line(self, line: str, line_num: int):
        """
        VERSIÓN CORREGIDA: Captura descripciones ANTES del ITEM y las preserva correctamente.
        """
        line = line.strip()
        if not line:
            # NO resetear potential_apu_desc aquí, solo el contexto actual
            self.context["apu_code"] = None
            logger.debug(
                "🔄 Contexto de APU reseteado por línea"
                " vacía (descripción potencial preservada)."
                )
            return

        self.stats["processed_lines"] += 1

        # 0. ⚡ FILTRO PRIORITARIO: Descartar líneas de metadatos
        if self._is_metadata_line(line):
            self.stats["garbage_lines"] += 1
            logger.debug(f"⏭️  Línea de metadato descartada en L{line_num}")
            return

        # 1. Filtrado de basura general
        if self._is_garbage_line(line):
            self.stats["garbage_lines"] += 1
            return

        upper_line = line.upper()

        # 2. 🔥 CAPTURA DE DESCRIPCIÓN POTENCIAL (ANTES DE BUSCAR ITEM)
        # Si no estamos en un APU activo Y la línea parece una descripción
        if not self.context["apu_code"] and not self.PATTERNS["item_code"].search(upper_line):
            if self._is_potential_description(line):
                # Capturar toda la línea como descripción potencial
                self.potential_apu_desc = line.split(';')[0].strip()
                logger.debug(
                    f"📝 Descripción potencial capturada"
                    f" (L{line_num}): '{self.potential_apu_desc[:50]}...'")
                return

        # 3. Detección de ITEM (usa la descripción capturada previamente)
        match_item = self.PATTERNS["item_code"].search(upper_line)
        if match_item:
            raw_code = match_item.group(1).strip()
            unit_match = re.search(r"UNIDAD:\s*([A-Z0-9/%]+)", upper_line)
            unit = unit_match.group(1) if unit_match else "INDEFINIDO"

            logger.debug(
                f"🆕 ITEM detectado (L{line_num}): código='{raw_code}', unidad='{unit}', "
                f"descripción pendiente='{self.potential_apu_desc[:30]}...'"
            )
            self._start_new_apu(raw_code, unit)
            return

        # 4. Lógica dependiente del estado
        if not self.context["apu_code"]:
            # ESTADO INACTIVO: Ya capturamos descripción arriba, no hacer nada más
            return

        # --- ESTADO ACTIVO: APU en progreso ---

        # A. PRIMERO: Intentar detectar categoría
        category_detected = self._try_detect_category_change(line, upper_line)
        if category_detected:
            return

        # B. SEGUNDO: Intentar parsear como dato
        if self._try_parse_as_data_line(line, line_num):
            return

        # C. TERCERO: Si estamos en un APU pero la línea no es dato ni categoría,
        #    podría ser la descripción del SIGUIENTE APU
        if self._is_potential_description(line):
            # Guardar para el próximo APU
            self.potential_apu_desc = line.split(';')[0].strip()
            logger.debug(
                f"📝 Nueva descripción potencial capturada"
                f" (L{line_num}): '{self.potential_apu_desc[:50]}...'"
                )

    def _start_new_apu(self, raw_code: str, unit: str):
        """
        Usa la descripción capturada previamente y la preserva correctamente.
        """
        cleaned_code = clean_apu_code(raw_code)

        if not cleaned_code:
            logger.warning(
                f"⚠️ Código APU no válido: '{raw_code}' - Reseteando contexto completo"
            )
            # NO resetear potential_apu_desc aquí, podría ser para el siguiente APU válido
            self.context = {
                "apu_code": None,
                "apu_desc": "",
                "apu_unit": "",
                "category": "INDEFINIDO",
            }
            return

        # 🔥 IMPORTANTE: Usar la descripción capturada previamente
        descripcion_apu = self.potential_apu_desc if self.potential_apu_desc else ""

        # APU válido: establecer nuevo contexto con la descripción capturada
        self.context = {
            "apu_code": cleaned_code,
            "apu_desc": descripcion_apu,  # 🔥 Aquí usamos la descripción capturada
            "apu_unit": unit.strip(),
            "category": "INDEFINIDO",
        }

        logger.info(
            f"✅ Nuevo APU iniciado: {cleaned_code} - '{descripcion_apu[:50]}...' ({unit})"
        )

        # Resetear SOLO después de usar la descripción
        self.potential_apu_desc = ""
        self.stats["items_found"] += 1

    def _is_potential_description(self, line: str) -> bool:
        """
        VERSIÓN MEJORADA: Determina si una línea podría ser una descripción de APU.
        """
        # Eliminar espacios y verificar que no esté vacía
        line_clean = line.strip()
        if not line_clean:
            return False

        # No debe ser una línea de datos (con múltiples punto y comas)
        if self._has_data_structure(line):
            return False

        # No debe ser basura o metadatos
        if self._is_garbage_line(line) or self._is_metadata_line(line):
            return False

        # No debe parecer una categoría sola
        upper_line = line_clean.upper()
        if upper_line in self.CATEGORY_KEYWORDS:
            return False

        # CRITERIOS POSITIVOS para una descripción:
        # 1. Debe tener al menos 5 caracteres
        if len(line_clean) < 5:
            return False

        # 2. Debe contener al menos una palabra significativa (3+ letras)
        if not re.search(r"[a-zA-Z]{3,}", line_clean):
            return False

        # 3. No debe empezar con números solos (podría ser un código mal formateado)
        if re.match(r"^\d+\.?\d*$", line_clean):
            return False

        # 4. Típicamente las descripciones de APU contienen palabras como:
        description_keywords = [
            "SUMINISTRO", "INSTALACION", "CONSTRUCCION", "EXCAVACION",
            "RELLENO", "CONCRETO", "ACERO", "TUBERIA", "CANAL", "MURO",
            "LOSA", "VIGA", "COLUMNA", "CIMENTACION", "ESTRUCTURA",
            "ACABADO", "PINTURA", "PRELIMINAR", "DEMOLICION", "RETIRO",
            "TRANSPORTE", "MONTAJE", "MANTENIMIENTO", "REPARACION"
        ]

        # Si contiene alguna palabra clave típica de descripción, es muy probable que lo sea
        for keyword in description_keywords:
            if keyword in upper_line:
                logger.debug(
                    f"✅ Descripción detectada por palabra clave (L{line_num}): '{keyword}': {line_clean[:50]}..."
                    )
                return True

        # Si no tiene palabras clave pero tiene suficiente texto alfabético,
        # podría ser una descripción
        alpha_chars = sum(1 for c in line_clean if c.isalpha())
        if alpha_chars >= 10:  # Al menos 10 letras
            return True

        return False

    def _log_parsing_stats(self):
        """VERSIÓN MEJORADA: Incluye estadísticas de descripciones capturadas"""
        logger.info("📊 MÉTRICAS FINALES DE PARSING:")
        for key, value in self.stats.items():
            logger.info(f"   {key}: {value}")

        # Contar APUs con descripción
        apus_con_desc = sum(
            1 for apu in self.apus_data if apu.get("DESCRIPCION_APU")
            )
        total_apus = len(
            set(apu["CODIGO_APU"] for apu in self.apus_data if apu.get("CODIGO_APU"))
            )

        logger.info(f"   APUs con descripción: {apus_con_desc}/{total_apus}")

        # Mostrar muestra de APUs con sus descripciones
        if self.apus_data:
            unique_apus = {}
            for apu in self.apus_data:
                codigo = apu.get("CODIGO_APU")
                if codigo and codigo not in unique_apus:
                    unique_apus[codigo] = apu.get("DESCRIPCION_APU", "")

            logger.info("📝 Muestra de APUs con sus descripciones:")
            for codigo, desc in list(unique_apus.items())[:5]:
                desc_preview = desc[:50] + "..." if len(desc) > 50 else desc
                logger.info(f"   {codigo}: '{desc_preview}'")

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

    def _is_metadata_line(self, line: str) -> bool:
        """
        Detecta si una línea contiene metadatos que deben ser ignorados.

        Estas líneas suelen aparecer como encabezados o subtotales que no son insumos reales
        y pueden causar errores de parsing si se procesan como datos.

        Args:
            line: Línea de texto a evaluar

        Returns:
            True si la línea contiene palabras clave de metadatos, False en caso contrario
        """
        if not line:
            return False

        # Palabras clave que identifican líneas de metadatos no procesables
        metadata_keywords = [
            'EQUIPO Y HERRAMIENTA',
            'EQUIPOS Y HERRAMIENTA',
            'EQUIPO Y HERRAMIENTAS',
            'EQUIPOS Y HERRAMIENTAS',
            'IMPUESTOS Y RETENCIONES',
            'IMPUESTOS',
            'POLIZAS',
            'PÓLIZAS',  # Versión con acento
        ]

        upper_line = line.upper()

        # Verificar si alguna palabra clave está presente en la línea
        for keyword in metadata_keywords:
            if keyword in upper_line:
                logger.debug(
                    f"🚫 Línea de metadato detectada ('{keyword}'): {line[:60]}..."
                )
                return True

        return False

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
