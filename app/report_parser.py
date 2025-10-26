import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from unidecode import unidecode

from .utils import clean_apu_code

logger = logging.getLogger(__name__)


class ParserState(Enum):
    IDLE = "IDLE"
    AWAITING_DESCRIPTION = "AWAITING_DESCRIPTION"
    PROCESSING_APU = "PROCESSING_APU"


class ReportParser:
    """Analizador de informes de Análisis de Precios Unitarios (APU).

    Esta clase se encarga de leer un archivo de texto con un formato no
    estándar, identificar y extraer los datos de los APU, y convertirlos
    en un DataFrame de Pandas estructurado.

    Attributes:
        file_path (str): La ruta al archivo a analizar.
        apus_data (List[Dict[str, Any]]): Una lista de diccionarios con
                                          los datos de los APU extraídos.
        context (Dict[str, Any]): El contexto actual del análisis.
        potential_apu_desc (str): Una descripción de APU potencial.
        stats (Dict[str, int]): Estadísticas del proceso de análisis.
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
    DESCRIPTION_KEYWORDS = [
            "SUMINISTRO", "INSTALACION", "CONSTRUCCION", "EXCAVACION",
            "RELLENO", "CONCRETO", "ACERO", "TUBERIA", "CANAL", "MURO",
            "LOSA", "VIGA", "COLUMNA", "CIMENTACION", "ESTRUCTURA",
            "ACABADO", "PINTURA", "PRELIMINAR", "DEMOLICION", "RETIRO",
            "TRANSPORTE", "MONTAJE", "MANTENIMIENTO", "REPARACION"
        ]

    def __init__(self, file_path: str):
        """Inicializa el ReportParser.

        Args:
            file_path (str): La ruta al archivo a analizar.
        """
        self.file_path = file_path
        self.apus_data: List[Dict[str, Any]] = []
        self.state = ParserState.IDLE
        self.context = {
            "apu_code": None, "apu_desc": "", "apu_unit": "", "category": "INDEFINIDO",
        }
        self.potential_apu_desc = ""
        self.stats = {
            "total_lines": 0, "processed_lines": 0, "items_found": 0,
            "insumos_parsed": 0, "mo_compleja_parsed": 0, "mo_simple_parsed": 0,
            "garbage_lines": 0, "unparsed_data_lines": 0, "fallback_parsed": 0,
            "state_transitions": 0
        }

    def parse(self) -> pd.DataFrame:
        """Analiza el archivo con DEBUG EXTENDIDO de unidades."""
        logger.info(f"🔍 Iniciando parsing DEL ARCHIVO: {self.file_path}")

        # DEBUG CRÍTICO: Mostrar TODAS las líneas con ITEM
        try:
            with open(self.file_path, "r", encoding="latin1") as f:
                logger.info("🚨 DEBUG CRÍTICO - TODAS LAS LÍNEAS CON 'ITEM':")
                for line_num, line in enumerate(f, 1):
                    if "ITEM" in line.upper():
                        logger.info(f" Línea {line_num}: {line.strip()}")
        except Exception as e:
            logger.error(f"❌ Error en debug: {e}")

        # Continuar con el parsing normal
        try:
            with open(self.file_path, "r", encoding="latin1") as f:
                for line_num, line in enumerate(f, 1):
                    self.stats["total_lines"] += 1
                    self._process_line(line, line_num)
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
        """Procesa una línea con máquina de estados simplificada y robusta."""
        line = line.strip()

        # REGLA 1: Ignorar líneas vacías, basura y metadatos
        if not line:
            self._transition_to_idle("línea en blanco")
            return

        if self._is_garbage_line(line) or self._is_metadata_line(line):
            self.stats["garbage_lines"] += 1
            return

        self.stats["processed_lines"] += 1

        # REGLA 2: Detectar nuevo ITEM (SIEMPRE disponible, en cualquier estado)
        if self._try_start_new_apu(line, line_num):
            return

        # REGLA 3: Procesamiento según estado actual
        if self.state == ParserState.IDLE:
            # En IDLE, solo nos interesan los nuevos ITEMs
            return

        elif self.state == ParserState.AWAITING_DESCRIPTION:
            # ESTADO CRÍTICO: La primera línea después de ITEM debe ser la descripción
            self._capture_apu_description(line, line_num)

        elif self.state == ParserState.PROCESSING_APU:
            # Procesar categorías e insumos del APU actual
            self._process_apu_data(line, line_num)

    def _capture_apu_description(self, line: str, line_num: int):
        """Captura la descripción del APU - SIMPLE Y DIRECTA."""
        # LÓGICA SIMPLIFICADA: La primera línea no vacía después de ITEM es la descripción
        description = line.split(';')[0].strip()

        # Validaciones básicas
        if not description or len(description) < 3:
            logger.warning(
                "⚠️ Descripción muy corta o vacía en APU %s", self.context['apu_code']
            )
            description = "DESCRIPCIÓN NO DISPONIBLE"

        # ASIGNAR DESCRIPCIÓN SIEMPRE
        self.context["apu_desc"] = description
        self._transition_to(ParserState.PROCESSING_APU, "descripción capturada")

        logger.info(
            "✅ Descripción asignada a %s: '%s...'",
            self.context['apu_code'], description[:60]
        )

        # IMPORTANTE: La línea podría contener datos después de ';'
        # Si tiene estructura de datos, procesarla también
        if self._has_data_structure(line):
            remaining_data = ';'.join(line.split(';')[1:])
            if remaining_data.strip():
                self._try_parse_as_data_line(remaining_data, line_num)

    def _process_apu_data(self, line: str, line_num: int):
        """Procesa datos dentro de un APU activo."""
        # Prioridad 1: Detectar cambio de categoría
        if self._try_detect_category_change(line):
            return

        # Prioridad 2: Intentar parsear como dato
        if self._try_parse_as_data_line(line, line_num):
            return

        # Prioridad 3: Línea no reconocida (registrar pero continuar)
        logger.debug(
            "⚠️ Línea %d no reconocida en APU %s: %s...",
            line_num, self.context['apu_code'], line[:60]
        )
        self.stats["unparsed_data_lines"] += 1

    def _try_start_new_apu(self, line: str, line_num: int) -> bool:
        """Inicia un nuevo APU con extracción ULTRA-AGRESIVA de unidad."""
        match_item = self.PATTERNS["item_code"].search(line.upper())
        if not match_item:
            return False

        raw_code = match_item.group(1).strip()

        # 🚨 EXTRACCIÓN ULTRA-AGRESIVA
        unit = self._extract_unit_emergency(line, line_num)

        cleaned_code = clean_apu_code(raw_code)
        if not cleaned_code:
            logger.warning(f"⚠️ Código de APU inválido: '{raw_code}'")
            return False

        self.context = {
            "apu_code": cleaned_code,
            "apu_desc": "",
            "apu_unit": unit,
            "category": "INDEFINIDO",
        }

        self.stats["items_found"] += 1
        self._transition_to(ParserState.AWAITING_DESCRIPTION, f"nuevo APU: {cleaned_code}")

        logger.info(f"🔄 Nuevo APU iniciado: {cleaned_code} | Unidad: {unit}")
        return True

    def _extract_unit_emergency(self, line: str, line_num: int) -> str:
        """
        Extracción de unidad de EMERGENCIA - Enfoque brutalmente simple.
        """
        line_upper = line.upper().strip()

        logger.info(f"🚨 EMERGENCY UNIT EXTRACTION Línea {line_num}: '{line_upper}'")

        # ESTRATEGIA 1: Buscar UNIDAD: o UNIDAD (prioriza con ':')
        text_after_unidad = ""
        if "UNIDAD:" in line_upper:
            parts = line_upper.split("UNIDAD:", 1)
            if len(parts) > 1:
                text_after_unidad = parts[1].strip()
        elif "UNIDAD" in line_upper:
            parts = line_upper.split("UNIDAD", 1)
            if len(parts) > 1:
                text_after_unidad = parts[1].strip()

        if text_after_unidad:
            unit = self._extract_unit_from_text(text_after_unidad)
            if unit:
                logger.info(f"🎯 UNIDAD encontrado -> '{unit}'")
                return unit

        # ESTRATEGIA 2: Buscar unidades directamente en toda la línea
        direct_units = self._find_units_bruteforce(line_upper)
        if direct_units:
            logger.info(f"🎯 Unidad directa -> '{direct_units}'")
            return direct_units

        # ESTRATEGIA 3: Si todo falla, asumir UND (fallback de emergencia)
        logger.warning(f"🚨 ASIGNANDO 'UND' COMO FALLBACK de emergencia")
        return "UND"

    def _extract_unit_from_text(self, text: str) -> str:
        """Extrae unidad de un texto usando múltiples estrategias."""
        # Intentar extraer primera palabra
        words = text.split()
        if words:
            first_word = words[0].strip(';,.')
            unit = self._clean_unit_brutal(first_word)
            if self._is_valid_unit(unit):
                return unit

        # Buscar unidades en el texto completo
        return self._find_units_bruteforce(text)

    def _find_units_bruteforce(self, text: str) -> str:
        """Búsqueda brutal de unidades en el texto."""
        # Unidades CRÍTICAS (con variaciones), ordenadas por longitud
        critical_units = sorted(
            ['DIA', 'DIAS', 'DÍAS', 'JOR', 'JORNAL', 'M2', 'M3', 'UND', 'UNIDAD', 'HORA', 'HORAS', 'LOTE', 'SERVICIO'],
            key=len, reverse=True
        )
        for unit in critical_units:
            if re.search(r'\b' + re.escape(unit) + r'\b', text):
                return self._clean_unit_brutal(unit)

        # Unidades secundarias
        secondary_units = sorted(['ML', 'KM', 'CM', 'KG', 'TON', 'L', 'GAL', 'M'], key=len, reverse=True)
        for unit in secondary_units:
            if re.search(r'\b' + re.escape(unit) + r'\b', text):
                return unit

        return ""

    def _clean_unit_brutal(self, unit: str) -> str:
        """Limpieza brutal de unidad."""
        if not unit:
            return ""

        # Remover caracteres no deseados
        unit = re.sub(r'[^A-Z0-9/%]', '', unit.upper())

        # Mapeo directo
        unit_map = {
            'M2': 'M2', 'M3': 'M3', 'ML': 'ML', 'M': 'M',
            'DIA': 'DIA', 'DIAS': 'DIA', 'DÍAS': 'DIA',
            'JOR': 'JOR', 'JORNAL': 'JOR',
            'HORA': 'HORA', 'HR': 'HORA', 'HORAS': 'HORA',
            'UND': 'UND', 'UN': 'UND', 'UNIDAD': 'UND',
            'LOTE': 'LOTE', 'SERVICIO': 'SERVICIO'
        }

        return unit_map.get(unit, unit)

    def _clean_unit(self, unit: str) -> str:
        """Limpia y normaliza la unidad extraída - VERSIÓN SIMPLIFICADA."""
        if not unit:
            return ""

        # Limpiar espacios y caracteres extraños
        unit = re.sub(r'[^\w]', '', unit.strip())

        # Normalizar unidades comunes (MAYÚSCULAS)
        unit = unit.upper()

        # Mapeo de normalización
        unit_mappings = {
            'M2': 'M2', 'M3': 'M3', 'ML': 'ML', 'M': 'M',
            'DIA': 'DIA', 'DIAS': 'DIA', 'DÍAS': 'DIA',
            'JOR': 'JOR', 'JORNAL': 'JOR',
            'HORA': 'HORA', 'HR': 'HORA', 'HORAS': 'HORA',
            'UND': 'UND', 'UN': 'UND', 'UNIT': 'UND', 'UNIDAD': 'UND',
            'SERVICIO': 'SERVICIO', 'SERV': 'SERVICIO',
            'LOTE': 'LOTE', 'LOT': 'LOTE',
            'KG': 'KG', 'GR': 'GR', 'TON': 'TON',
            'L': 'L', 'ML': 'ML', 'GAL': 'GAL'
        }

        return unit_mappings.get(unit, unit)

    def _is_valid_unit(self, unit: str) -> bool:
        """Verifica si la unidad es válida (VERSIÓN PERMISIVA)."""
        if not unit or unit == "INDEFINIDO":
            return False

        # Cualquier unidad no vacía es válida en modo emergencia
        valid_units = {
            'M2', 'M3', 'ML', 'M', 'CM', 'MM', 'KM',
            'UND', 'UNIDAD', 'U', 'UNIT',
            'JOR', 'DIA', 'HORA', 'MES', 'SEMANA',
            'LOTE', 'SERVICIO', 'KG', 'GR', 'TON', 'LB',
            'L', 'ML', 'GAL', 'PT', '%'
        }

        return unit in valid_units

    def _fallback_unit_detection(self, line: str, line_num: int) -> str:
        """
        Detección de unidad como fallback - VERSIÓN CORREGIDA Y MEJORADA.
        Ahora normaliza las unidades encontradas.
        """
        logger.debug(f"🔄 Usando fallback para unidad en línea {line_num}")

        # Lista extendida de unidades para buscar, incluyendo variaciones comunes.
        # Ordenadas por longitud para priorizar coincidencias más largas (ej. 'JORNAL' sobre 'JOR').
        known_units_variations = sorted([
            'M2', 'M3', 'ML', 'M', 'UND', 'UN', 'UNIT', 'UNIDAD', 'SERVICIO', 'SERV',
            'JOR', 'JORNAL', 'DIA', 'DIAS', 'DÍAS', 'HORA', 'HR', 'HORAS',
            'LOTE', 'LOT', 'KG', 'GR', 'TON', 'L', 'GAL'
        ], key=len, reverse=True)

        for unit_variation in known_units_variations:
            # Buscar la variación como palabra completa en la línea
            if re.search(r'\b' + re.escape(unit_variation) + r'\b', line):
                # Si se encuentra, normalizarla usando el método centralizado
                normalized_unit = self._clean_unit(unit_variation)
                logger.debug(f"🔄 Unidad '{normalized_unit}' (detectada como '{unit_variation}') por fallback")
                return normalized_unit

        # Estrategia adicional: buscar después del último ';'
        parts = line.split(';')
        if len(parts) > 1:
            last_part = parts[-1].strip()
            # Normalizar la última parte y verificar si es una unidad válida
            normalized_last_part = self._clean_unit(last_part)
            if self._is_valid_unit(normalized_last_part):
                logger.debug(f"🔄 Unidad '{normalized_last_part}' detectada en último segmento")
                return normalized_last_part

        logger.warning(f"⚠️ No se pudo detectar unidad en línea {line_num}")
        return "INDEFINIDO"

    def _debug_unit_extraction(self, line: str, line_num: int):
        """Método temporal para debuggear la extracción de unidades."""
        if "ITEM:" in line.upper() and "UNIDAD" in line.upper():
            logger.info(f"🔍 DEBUG Línea {line_num}: {line.strip()}")

            # Probar cada patrón individualmente
            patterns = [
                r"UNIDAD\s*:\s*([^;,\s]+)",
                r"UNIDAD\s+([^;,\s]+)",
                r"UNIDAD:\s*(\w+)"
            ]

            for i, pattern in enumerate(patterns):
                match = re.search(pattern, line.upper())
                if match:
                    logger.info(f" Patrón {i}: ✅ COINCIDENCIA -> '{match.group(1)}'")
                else:
                    logger.info(f" Patrón {i}: ❌ NO COINCIDENCIA")

    def _extract_unit_ultra_robust(self, line: str, line_num: int) -> str:
        """
        Extrae la unidad de medida con enfoque ULTRA-ROBUSTO.
        Si no encuentra UNIDAD:, busca directamente unidades conocidas.
        """
        line_upper = line.upper().strip()

        # DEBUG CRÍTICO
        logger.info(f"🔍 ANALIZANDO UNIDAD en línea {line_num}: '{line_upper}'")

        # ESTRATEGIA 1: Buscar patrón UNIDAD: explícito
        unidad_patterns = [
            r"UNIDAD\s*:\s*([^;]+)",  # UNIDAD: valor (hasta ;)
            r"UNIDAD\s*:\s*(\S+)",  # UNIDAD: valor (una palabra)
            r"UNIDAD\s+([^;]+)",  # UNIDAD valor (sin :)
            r"UNIDAD\s+(\S+)",  # UNIDAD valor (una palabra sin :)
        ]

        for pattern in unidad_patterns:
            match = re.search(pattern, line_upper)
            if match:
                raw_unit = match.group(1).strip()
                logger.info(f"🎯 Patrón UNIDAD coincidió: '{raw_unit}'")

                # Extraer la primera palabra después de UNIDAD:
                unit = self._extract_first_word(raw_unit)
                unit = self._clean_unit_aggressive(unit)

                if self._is_valid_unit(unit):
                    logger.info(f"✅ Unidad '{unit}' extraída de patrón UNIDAD")
                    return unit

        # ESTRATEGIA 2: Buscar directamente unidades conocidas en toda la línea
        direct_units = self._find_direct_units(line_upper)
        if direct_units:
            logger.info(f"🎯 Unidad directa encontrada: '{direct_units}'")
            return direct_units

        # ESTRATEGIA 3: Buscar después del código del ITEM
        unit_after_item = self._extract_unit_after_item(line_upper, line_num)
        if unit_after_item:
            return unit_after_item

        logger.warning(f"🚨 NO SE PUDO EXTRAER UNIDAD en línea {line_num}")
        return "INDEFINIDO"

    def _extract_first_word(self, text: str) -> str:
        """Extrae la primera palabra de un texto."""
        words = text.split()
        return words[0] if words else ""

    def _clean_unit_aggressive(self, unit: str) -> str:
        """Limpia unidad de forma agresiva."""
        if not unit:
            return ""

        # Remover caracteres no alfabéticos (excepto / y %)
        unit = re.sub(r"[^A-Z0-9/%]", "", unit.upper())

        # Mapeo directo y simple
        unit_map = {
            "M2": "M2",
            "M3": "M3",
            "ML": "ML",
            "M": "M",
            "DIA": "DIA",
            "DIAS": "DIA",
            "DÍAS": "DIA",
            "JOR": "JOR",
            "JORNAL": "JOR",
            "HORA": "HORA",
            "HR": "HORA",
            "UND": "UND",
            "UN": "UND",
            "UNIDAD": "UND",
            "LOTE": "LOTE",
            "SERVICIO": "SERVICIO",
            "KG": "KG",
            "TON": "TON",
        }

        return unit_map.get(unit, unit)

    def _find_direct_units(self, line: str) -> str:
        """Busca unidades directamente en la línea."""
        # Unidades prioritarias para el estimador
        priority_units = ["DIA", "JOR", "M2", "M3", "UND", "HORA"]

        for unit in priority_units:
            # Buscar como palabra completa
            if re.search(r"\b" + unit + r"\b", line):
                return unit

        # Unidades secundarias
        secondary_units = ["LOTE", "SERVICIO", "KG", "TON", "ML", "M"]
        for unit in secondary_units:
            if re.search(r"\b" + unit + r"\b", line):
                return unit

        return ""

    def _extract_unit_after_item(self, line: str, line_num: int) -> str:
        """Intenta extraer unidad después del código del ITEM."""
        # Buscar patrones comunes: ITEM: CODIGO; UNIDAD: VALOR
        patterns = [
            r"ITEM[^;]*;\s*UNIDAD[^:]*:\s*([^;]+)",  # ITEM...; UNIDAD:...
            r"ITEM[^;]*;\s*(\w+)\s*;",  # ITEM...; UNIDAD;
            r"ITEM[^;]*;\s*([^;]+)$",  # ITEM...; algo_al_final
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                potential_unit = match.group(1).strip()
                unit = self._clean_unit_aggressive(potential_unit)
                if self._is_valid_unit(unit):
                    logger.info(f"🎯 Unidad '{unit}' extraída después de ITEM")
                    return unit

        return ""

    def _is_potential_description(self, line: str, line_num: int) -> bool:
        """Determina si una línea es una descripción válida de APU."""
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 5:
            return False

        # CRITERIOS DE EXCLUSIÓN
        # 1. No debe tener estructura de datos
        if self._is_structured_data_line(line):
            return False

        # 2. No debe ser basura o metadatos
        if self._is_garbage_line(line) or self._is_metadata_line(line):
            return False

        # 3. No debe ser solo una categoría
        first_part = line_clean.split(';')[0].strip().upper()
        if first_part in self.CATEGORY_KEYWORDS:
            return False

        # 4. No debe ser solo números
        if re.match(r"^[\d.,\s]+$", first_part):
            return False

        # CRITERIOS DE INCLUSIÓN
        # 1. Debe tener contenido alfabético significativo
        if not re.search(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]{3,}", first_part):
            return False

        # 2. BONUS: Si contiene palabras clave de construcción
        for keyword in self.DESCRIPTION_KEYWORDS:
            if keyword in first_part:
                return True

        # 3. Contenido alfabético suficiente
        alpha_count = sum(1 for c in first_part if c.isalpha())
        return alpha_count >= 10

    def _log_parsing_stats(self):
        """Registrar estadísticas con diagnóstico EXTENDIDO de unidades."""
        logger.info("=" * 60)
        logger.info("📊 MÉTRICAS FINALES DE PARSING - DIAGNÓSTICO EXTENDIDO")
        logger.info("=" * 60)

        for key, value in self.stats.items():
            logger.info(f" {key:.<35} {value}")

        # ANÁLISIS PROFUNDO DE UNIDADES
        unique_apus = {}
        for apu in self.apus_data:
            codigo = apu.get("CODIGO_APU")
            if codigo and codigo not in unique_apus:
                unique_apus[codigo] = {
                    "desc": apu.get("DESCRIPCION_APU", ""),
                    "unit": apu.get("UNIDAD_APU", "INDEFINIDO"),
                }

        # Contadores específicos
        apus_con_unidad_valida = 0
        unidades_encontradas = []

        for codigo, data in unique_apus.items():
            if data["unit"] != "INDEFINIDO":
                apus_con_unidad_valida += 1
                unidades_encontradas.append(data["unit"])

        total_apus = len(unique_apus)

        logger.info(f" APUs con descripción válida:......... {total_apus}/{total_apus}")
        logger.info(
            f" APUs con unidad válida:............ {apus_con_unidad_valida}/{total_apus}"
        )

        # DISTRIBUCIÓN DETALLADA
        from collections import Counter

        unit_counter = Counter(unidades_encontradas)

        logger.info("\n📏 DISTRIBUCIÓN DETALLADA DE UNIDADES:")
        for unit, count in unit_counter.most_common():
            logger.info(f" {unit:.<20} {count}")

        # UNIDADES CRÍTICAS PARA EL ESTIMADOR
        unidades_criticas = {"DIA", "JOR", "M2", "M3"}
        for unidad in unidades_criticas:
            count = sum(1 for u in unidades_encontradas if u == unidad)
            logger.info(f" {unidad + ' (CRÍTICA)':.<20} {count}")

        # MUESTRA DE APUs CON UNIDADES VÁLIDAS (si las hay)
        if apus_con_unidad_valida > 0:
            logger.info("\n✅ APUs CON UNIDADES VÁLIDAS:")
            valid_apus = [
                (codigo, data)
                for codigo, data in unique_apus.items()
                if data["unit"] != "INDEFINIDO"
            ]
            for i, (codigo, data) in enumerate(valid_apus[:10], 1):
                logger.info(
                    f" {i}. {codigo}: '{data['unit']}' - '{data['desc'][:50]}...'"
                )
        else:
            logger.info("\n❌ NINGÚN APU CON UNIDAD VÁLIDA ENCONTRADO")

        logger.info("=" * 60)

    def _try_parse_as_data_line(self, line: str, line_num: int) -> bool:
        """Intenta analizar una línea como una línea de datos.

        Args:
            line (str): La línea a analizar.
            line_num (int): El número de la línea en el archivo.

        Returns:
            bool: True si la línea se analizó como datos, False en caso contrario.
        """
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

    def _try_detect_category_change(self, line: str) -> bool:
        """Detecta cambios de categoría evitando falsos positivos."""
        line_clean = line.strip()
        if not line_clean:
            return False

        first_part = line_clean.split(';')[0].strip().upper()

        # Mapeo robusto de categorías
        category_mappings = {
            "MATERIALES": ["MATERIALES", "MATERIALES Y SUMINISTROS", "MATERIAL"],
            "MANO DE OBRA": ["MANO DE OBRA", "MANO DE OBRA DIRECTA", "M.O.", "MO"],
            "EQUIPO": ["EQUIPO", "EQUIPOS", "EQUIPOS Y HERRAMIENTAS", "MAQUINARIA"],
            "TRANSPORTE": ["TRANSPORTE", "TRANSPORTES", "FLETE"],
            "OTROS": ["OTROS", "OTROS GASTOS", "GASTOS GENERALES"]
        }

        # Buscar categoría
        found_category = None
        for category, keywords in category_mappings.items():
            if first_part in keywords:
                found_category = category
                break

        if not found_category:
            return False

        # 🔴 VALIDACIÓN CRÍTICA: No debe ser línea de datos
        if self._is_structured_data_line(line):
            logger.debug(f"❌ '{first_part}' parece categoría pero tiene datos")
            return False

        # Cambiar categoría solo si es diferente
        if self.context["category"] != found_category:
            old_category = self.context["category"]
            self.context["category"] = found_category
            logger.info(f"📂 Categoría cambiada: {old_category} → {found_category}")
            return True

        return False

    def _is_structured_data_line(self, line: str) -> bool:
        """Verifica si una línea tiene estructura de datos real."""
        if line.count(';') < 2:
            return False

        parts = line.split(';')
        numeric_values = 0

        # Verificar valores numéricos en campos de datos
        for part in parts[1:]: # Ignorar descripción
            value = self._to_numeric_safe(part)
            if value > 0:
                numeric_values += 1

        # Requiere al menos 2 valores numéricos significativos
        return numeric_values >= 2

    def _transition_to(self, new_state: ParserState, reason: str):
        """Realiza transición controlada entre estados."""
        old_state = self.state
        if old_state != new_state:
            self.state = new_state
            self.stats["state_transitions"] += 1
            logger.debug(f"🔄 {old_state.value} → {new_state.value} ({reason})")

    def _transition_to_idle(self, reason: str):
        """Resetea a IDLE (ahora más simple)."""
        if self.state != ParserState.IDLE:
            logger.debug(
                "🔄 Reset a IDLE (%s) | APU: %s",
                reason, self.context.get('apu_code', 'N/A')
            )
            self.state = ParserState.IDLE
            self.context["apu_code"] = None
            self.stats["state_transitions"] += 1

    def _try_fallback_parsing(self, line: str, line_num: int) -> bool:
        """Intenta un análisis genérico como último recurso.

        Args:
            line (str): La línea a analizar.
            line_num (int): El número de la línea en el archivo.

        Returns:
            bool: True si el análisis fue exitoso, False en caso contrario.
        """
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
        """Analiza y agrega un insumo general a los datos del APU.

        Args:
            data (Dict[str, str]): Los datos del insumo extraídos.
        """
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
        """Analiza y agrega datos de mano de obra compleja al APU.

        Args:
            data (Dict[str, str]): Los datos de mano de obra extraídos.
        """
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
        """Analiza y agrega datos de mano de obra simple al APU.

        Args:
            data (Dict[str, str]): Los datos de mano de obra extraídos.
        """
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
        """Agrega un registro de datos de APU a la lista de datos.

        Args:
            **kwargs: Los datos del registro de APU a agregar.
        """
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
        """Convierte una cadena a un número de punto flotante de forma segura.

        Args:
            s (Optional[str]): La cadena a convertir.

        Returns:
            float: El número convertido o 0.0 si la conversión falla.
        """
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
        """Determina si se debe agregar un insumo a los datos del APU.

        Args:
            desc (str): La descripción del insumo.
            cantidad (float): La cantidad del insumo.
            valor_total (float): El valor total del insumo.

        Returns:
            bool: True si se debe agregar el insumo, False en caso contrario.
        """
        if not desc or len(desc.strip()) < 2:
            return False
        return not (cantidad <= 0 and valor_total <= 0)

    def _looks_like_mo(self, line: str) -> bool:
        """Comprueba si una línea parece ser de mano de obra.

        Args:
            line (str): La línea a comprobar.

        Returns:
            bool: True si la línea parece ser de mano de obra, False en caso contrario.
        """
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
        """Calcula la cantidad de mano de obra.

        Args:
            valor_total (float): El valor total.
            jornal_total (float): El jornal total.

        Returns:
            float: La cantidad de mano de obra.
        """
        if jornal_total <= 0:
            return 0.0
        return valor_total / jornal_total

    def _calculate_rendimiento_simple(
        self, valor_total: float, precio_unit: float
    ) -> float:
        """Calcula el rendimiento para mano de obra simple.

        Args:
            valor_total (float): El valor total.
            precio_unit (float): El precio unitario.

        Returns:
            float: El rendimiento calculado.
        """
        if valor_total <= 0 or precio_unit <= 0:
            return 0.0
        return precio_unit / valor_total

    def _is_garbage_line(self, line: str) -> bool:
        """Comprueba si una línea es basura y debe ser ignorada.

        Args:
            line (str): La línea a comprobar.

        Returns:
            bool: True si la línea es basura, False en caso contrario.
        """
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
        """Comprueba si una línea tiene una estructura de datos similar a un CSV.

        Args:
            line (str): La línea a comprobar.

        Returns:
            bool: True si la línea tiene una estructura de datos, False en caso contrario.
        """
        return line.count(";") >= 2

    def _build_dataframe(self) -> pd.DataFrame:
        """Construye un DataFrame a partir de los datos de APU analizados.

        Returns:
            pd.DataFrame: El DataFrame construido.
        """
        if not self.apus_data:
            return pd.DataFrame()
        return pd.DataFrame(self.apus_data)

    def _normalize_text_single(self, text: str) -> str:
        """Normaliza un único string de texto.

        Args:
            text (str): El texto a normalizar.

        Returns:
            str: El texto normalizado.
        """
        if not isinstance(text, str):
            text = str(text)

        normalized = text.lower().strip()
        normalized = unidecode(normalized)
        normalized = re.sub(r"[^a-z0-9\s#\-]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized
