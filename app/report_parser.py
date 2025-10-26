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

    def _emergency_dia_units_patch(self):
        """
        Parche de emergencia: fuerza unidades DIA en APUs que son claramente cuadrillas
        pero no fueron inferidas correctamente.
        """
        # Patrones de códigos que típicamente son cuadrillas
        squad_codes = ['13', '14', '15', '16', '17', '18', '19', '20']

        for apu in self.apus_data:
            current_unit = apu.get("UNIDAD_APU", "")
            apu_code = apu.get("CODIGO_APU", "")
            category = apu.get("CATEGORIA", "")

            # Condiciones para forzar DIA
            force_dia_conditions = [
                # Si es mano de obra y tiene código de cuadrilla
                category == "MANO DE OBRA" and any(
                    apu_code.startswith(code) for code in squad_codes
                ),
                # Si el código contiene patrones de cuadrilla
                re.match(r'1[3-9]', apu_code) and category == "MANO DE OBRA",
                # Si la descripción sugiere cuadrilla aunque sea genérica
                any(keyword in apu.get("DESCRIPCION_APU", "").upper()
                    for keyword in ['DESCRIPCION', 'APU', 'ITEM']),
            ]

            if any(force_dia_conditions) and current_unit != "DIA":
                old_unit = current_unit
                apu["UNIDAD_APU"] = "DIA"
                logger.info(f"🚀 PARCHE DIA: {apu_code} '{old_unit}' → 'DIA'")

    def parse(self) -> pd.DataFrame:
        """Analiza el archivo y aplica parches de emergencia."""
        logger.info(f"🔍 Iniciando parsing con inferencia agresiva: {self.file_path}")

        try:
            with open(self.file_path, "r", encoding="latin1") as f:
                for line_num, line in enumerate(f, 1):
                    self.stats["total_lines"] += 1
                    self._process_line(line, line_num)
        except Exception as e:
            logger.error(f"❌ Error al parsear {self.file_path}: {e}", exc_info=True)
            return pd.DataFrame()

        # 🚀 APLICAR PARCHES DE EMERGENCIA
        self._emergency_dia_units_patch()

        self._log_parsing_stats()
        if not self.apus_data:
            logger.warning("⚠️ No se extrajeron datos de APU, devolviendo DataFrame vacío.")
            return pd.DataFrame()

        df = self._build_dataframe()
        logger.info(f"✅ DataFrame generado con {len(df)} registros")
        return df

    def _process_line(self, line: str, line_num: int):
        """Procesa una línea con máquina de estados corregida para captura de descripción."""
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
            # ESTADO CRÍTICO CORREGIDO: Solo capturar líneas que parecen descripciones reales
            if self._is_valid_apu_description(line):
                self._capture_apu_description(line, line_num)
            else:
                # Si no es una descripción válida, continuar esperando
                logger.debug(
                    f"⏳ Esperando descripción válida para APU {self.context['apu_code']}: "
                    f"'{line[:60]}...'"
                )
                return

        elif self.state == ParserState.PROCESSING_APU:
            # Procesar categorías e insumos del APU actual
            self._process_apu_data(line, line_num)

    def _is_valid_apu_description(self, line: str) -> bool:
        """Determina si una línea es una descripción válida de APU (NO encabezados)."""
        line_clean = line.strip()

        # CRITERIOS DE EXCLUSIÓN (lo que NO es una descripción)
        exclusion_patterns = [
            r'^DESCRIPCION$', r'^DESCRIPCIÓN$',  # Encabezados de tabla
            r'^ITEM$', r'^UNIDAD$', r'^CANTIDAD$',  # Otros encabezados
            r'^CODIGO$', r'^CÓDIGO$',
            r'^MATERIALES$', r'^MANO DE OBRA$', r'^EQUIPO$',  # Categorías
            r'^VALOR TOTAL$', r'^PRECIO UNIT$',
            r'^;+DESCRIPCION;+', r'^;+DESCRIPCIÓN;+'  # Encabezados con separadores
        ]

        for pattern in exclusion_patterns:
            if re.match(pattern, line_clean.upper()):
                return False

        # CRITERIOS DE INCLUSIÓN (lo que SÍ es una descripción)
        inclusion_criteria = [
            len(line_clean) >= 5,  # Longitud mínima reducida
            not line_clean.upper().startswith(';'),  # No empieza con separador
            bool(re.search(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]', line_clean)),  # Contiene texto
            not re.match(r'^[\d\s.,;]+$', line_clean)  # No es solo números/puntuación
        ]

        return all(inclusion_criteria)

    def _capture_apu_description(self, line: str, line_num: int):
        """Captura la descripción del APU con validación mejorada."""
        # Tomar solo la primera parte antes de cualquier ';' como descripción
        description = line.split(';')[0].strip()

        # Validación adicional contra encabezados
        if description.upper() in [
            'DESCRIPCION', 'DESCRIPCIÓN', 'ITEM', 'UNIDAD', 'CANTIDAD'
        ]:
            logger.warning(
                f"⚠️ Se rechazó encabezado como descripción para APU "
                f"{self.context['apu_code']}"
            )
            description = "DESCRIPCIÓN NO ESPECIFICADA"

        # Validaciones básicas de calidad
        if not description or len(description) < 5:
            logger.warning(
                f"⚠️ Descripción muy corta o vacía en APU {self.context['apu_code']}"
            )
            description = "DESCRIPCIÓN NO DISPONIBLE"

        # ASIGNAR DESCRIPCIÓN
        self.context["apu_desc"] = description

        # 🎯 INFERIR UNIDAD SI ES NECESARIO
        if self.context["apu_unit"] == "UND" and not self.context.get("unit_was_explicit"):
            inferred_unit = self._infer_unit_from_context(
                description, self.context["category"]
            )
            self.context["apu_unit"] = inferred_unit
            logger.info(
                f"🎯 Unidad inferida '{inferred_unit}' para APU {self.context['apu_code']}"
            )

        self._transition_to(ParserState.PROCESSING_APU, "descripción válida capturada")

        logger.info(
            f"✅ Descripción APU {self.context['apu_code']}: '{description[:70]}...'"
        )

        # Si la línea contiene datos después de ';', procesarlos también
        if self._has_data_structure(line):
            remaining_data = ';'.join(line.split(';')[1:])
            if remaining_data.strip():
                self._try_parse_as_data_line(remaining_data, line_num)

    def _infer_unit_aggressive(self, description: str, category: str, apu_code: str) -> str:
        """
        Inferencia ULTRA-AGRESIVA de unidades usando múltiples estrategias.
        """
        desc_upper = description.upper()

        # ESTRATEGIA 1: Por código de APU (patrones comunes)
        code_unit = self._infer_unit_from_code(apu_code)
        if code_unit:
            logger.info(f"🎯 Unidad '{code_unit}' inferida desde código APU: {apu_code}")
            return code_unit

        # ESTRATEGIA 2: Por categoría (prioridad absoluta)
        category_units = {
            "MANO DE OBRA": "JOR",
            "EQUIPO": "DIA",
            "TRANSPORTE": "VIAJE",
            "MATERIALES": "M2",  # Asumir M2 para materiales por defecto
            "OTROS": "UND",
        }
        if category in category_units:
            unit = category_units[category]
            logger.info(f"🎯 Unidad '{unit}' inferida desde categoría: {category}")
            return unit

        # ESTRATEGIA 3: Análisis de palabras clave expandido
        unit_keywords = {
            'DIA': ['CUADRILLA', 'EQUIPO', 'BRIGADA', 'GRUPO', 'OPERARIO', 'OBRERO',
                    'OFICIAL', 'AYUDANTE', 'MAESTRO', 'CAPATAZ', 'PEON', 'PEÓN',
                    'JORNADA', 'DIARIO', 'DIA', 'DÍAS', 'TURNO', 'GUARDIÁ', 'VIGILAN'],
            'JOR': ['MANO DE OBRA', 'M.O.', 'MO ', 'SALARIO', 'JORNAL', 'SUELDO',
                    'TRABAJADOR', 'EMPLEADO', 'OPERADOR', 'TECNICO', 'TÉCNICO'],
            'M2': ['M2', 'M²', 'METRO CUADRADO', 'METRO CUADRAD', 'SUPERFICIE',
                   'AREA', 'ÁREA', 'LOSA', 'PISO', 'PARED', 'MURO', 'FACHADA',
                   'TECHO', 'CUBIERTA', 'LAMINA', 'LÁMINA', 'PLACA', 'PANEL',
                   'AZULEJO', 'CERAMICA', 'CERÁMICA', 'PINTURA', 'ACABADO'],
            'M3': ['M3', 'M³', 'METRO CUBICO', 'METRO CÚBICO', 'VOLUMEN',
                   'EXCAVACION', 'EXCAVACIÓN', 'RELLENO', 'CONCRETO', 'HORMIGON',
                   'ZAPATA', 'CIMIENTO', 'COLUMNA', 'VIGA', 'MORTERO', 'GRANITO',
                   'ARENA', 'GRAVA', 'PIEDRA', 'AGREGAD', 'TIERRA', 'DEMOLICION'],
            'ML': ['ML', 'METRO LINEAL', 'LINEAL', 'LONGITUD', 'TUBERIA', 'TUBERÍA',
                   'CANAL', 'CONDUCCION', 'CONDUCCIÓN', 'LINEA', 'LÍNEA', 'TUBO',
                   'PERFIL', 'ANGULO', 'ÁNGULO', 'VARILLA', 'CABLE', 'ALAMBRE',
                   'CAÑO', 'CAÑERIA', 'CAÑERÍA', 'DUCTO', 'TRAYECTORIA']
        }

        for unit, keywords in unit_keywords.items():
            for keyword in keywords:
                if keyword in desc_upper:
                    logger.info(f"🎯 Unidad '{unit}' inferida por keyword: '{keyword}'")
                    return unit

        # ESTRATEGIA 4: Por tipo de trabajo basado en descripción parcial
        work_patterns = [
            (['CIMIENTO', 'ZAPATA', 'CONCRETO', 'HORMIGON'], 'M3'),
            (['LOSA', 'VIGA', 'COLUMNA', 'ESTRUCTURA'], 'M3'),
            (['PISO', 'PARED', 'MURO', 'TECHO', 'CUBIERTA'], 'M2'),
            (['PINTURA', 'ACABADO', 'ENCHAPE', 'REVOQUE'], 'M2'),
            (['TUBERIA', 'TUBERÍA', 'CAÑERIA', 'CAÑERÍA'], 'ML'),
            (['CABLE', 'ELECTRIC', 'ILUMINAC', 'ALAMBRE'], 'ML'),
            (['EXCAVAC', 'RELLENO', 'DEMOLICION', 'MOVIMIENTO'], 'M3')
        ]

        for keywords, unit in work_patterns:
            if any(keyword in desc_upper for keyword in keywords):
                logger.info(f"🎯 Unidad '{unit}' inferida por patrón de trabajo: {keywords}")
                return unit

        # ESTRATEGIA 5: Por código numérico (último recurso)
        return self._infer_unit_from_code_fallback(apu_code, category)

    def _infer_unit_from_code(self, apu_code: str) -> str:
        """Infiere unidad basándose en patrones del código APU."""
        code_patterns = [
            (r'^\d+[AB]?$', 'M2'),  # Códigos simples probablemente son M2
            (r'\d+[,-]\d+[ABC]?$', 'M2'),  # Códigos con coma/guion
            (r'.*[ABCD]$', 'M2'),  # Códigos que terminan con letra
            (r'.*[XYZ]$', 'ML'),  # Códigos especiales para lineales
            (r'.*[PQ]$', 'M3'),  # Códigos especiales para volumétricos
        ]

        for pattern, unit in code_patterns:
            if re.match(pattern, apu_code):
                return unit

        return ""

    def _infer_unit_from_code_fallback(self, apu_code: str, category: str) -> str:
        """Inferencia de último recurso basada en código y categoría."""
        # Mapeo por categoría con códigos específicos
        if category == "MATERIALES":
            # Para materiales, usar M2 como predeterminado (más común en construcción)
            return "M2"
        elif category == "MANO DE OBRA":
            return "JOR"
        elif category == "EQUIPO":
            return "DIA"
        else:
            return "UND"

    def _infer_unit_from_context(self, description: str, category: str) -> str:
        """Usa inferencia ultra-agresiva como estrategia principal."""
        # Obtener código APU del contexto actual
        apu_code = self.context.get("apu_code", "")

        return self._infer_unit_aggressive(description, category, apu_code)

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
        unit_was_explicit = bool(unit)

        cleaned_code = clean_apu_code(raw_code)
        if not cleaned_code:
            logger.warning(f"⚠️ Código de APU inválido: '{raw_code}'")
            return False

        self.context = {
            "apu_code": cleaned_code,
            "apu_desc": "",
            "apu_unit": unit or "UND",
            "category": "INDEFINIDO",
            "unit_was_explicit": unit_was_explicit,
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

        # ESTRATEGIA 3: Si todo falla, devolver cadena vacía
        logger.warning("🚨 No se encontró unidad explícita, se inferirá más tarde")
        return ""

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
        critical_units = sorted([
            'DIA', 'DIAS', 'DÍAS', 'JOR', 'JORNAL', 'M2', 'M3', 'UND',
            'UNIDAD', 'HORA', 'HORAS', 'LOTE', 'SERVICIO'
        ], key=len, reverse=True)
        for unit in critical_units:
            if re.search(r'\b' + re.escape(unit) + r'\b', text):
                return self._clean_unit_brutal(unit)

        # Unidades secundarias
        secondary_units = sorted(
            ['ML', 'KM', 'CM', 'KG', 'TON', 'L', 'GAL', 'M'],
            key=len,
            reverse=True
        )
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
            'L': 'L', 'GAL': 'GAL'
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
        # Ordenadas por longitud para priorizar coincidencias más largas
        # (ej. 'JORNAL' sobre 'JOR').
        known_units_variations = sorted([
            'M2', 'M3', 'ML', 'M', 'UND', 'UN', 'UNIT', 'UNIDAD', 'SERVICIO',
            'SERV', 'JOR', 'JORNAL', 'DIA', 'DIAS', 'DÍAS', 'HORA', 'HR',
            'HORAS', 'LOTE', 'LOT', 'KG', 'GR', 'TON', 'L', 'GAL'
        ], key=len, reverse=True)

        for unit_variation in known_units_variations:
            # Buscar la variación como palabra completa en la línea
            if re.search(r'\b' + re.escape(unit_variation) + r'\b', line):
                # Si se encuentra, normalizarla usando el método centralizado
                normalized_unit = self._clean_unit(unit_variation)
                logger.debug(
                    f"🔄 Unidad '{normalized_unit}' (detectada como "
                    f"'{unit_variation}') por fallback"
                )
                return normalized_unit

        # Estrategia adicional: buscar después del último ';'
        parts = line.split(';')
        if len(parts) > 1:
            last_part = parts[-1].strip()
            # Normalizar la última parte y verificar si es una unidad válida
            normalized_last_part = self._clean_unit(last_part)
            if self._is_valid_unit(normalized_last_part):
                logger.debug(
                    f"🔄 Unidad '{normalized_last_part}' detectada en último segmento"
                )
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
        """Registrar estadísticas con foco en unidades críticas."""
        logger.info("=" * 60)
        logger.info("📊 MÉTRICAS FINALES - INFERENCIA AGRESIVA")
        logger.info("=" * 60)

        # ... (código de estadísticas básicas igual) ...

        # ANÁLISIS DE UNIDADES CRÍTICAS
        critical_units = {'DIA', 'JOR', 'M2', 'M3', 'ML'}
        unit_counts = {unit: 0 for unit in critical_units}

        for apu in self.apus_data:
            unit = apu.get("UNIDAD_APU", "")
            if unit in critical_units:
                unit_counts[unit] += 1

        logger.info("\n🎯 UNIDADES CRÍTICAS PARA EL ESTIMADOR:")
        for unit in critical_units:
            count = unit_counts[unit]
            if count >= 3:
                status = "✅ SUFICIENTE"
            elif count > 0:
                status = "⚠️ INSUFICIENTE"
            else:
                status = "❌ FALTANTE"
            logger.info(f" {unit:.<20} {count:.<3} {status}")

        # RECOMENDACIONES ESPECÍFICAS
        logger.info("\n💡 RECOMENDACIONES:")
        if unit_counts['DIA'] < 5:
            logger.info(" • Se necesitan más APUs con UNIDAD=DIA para cuadrillas")
        if unit_counts['M2'] < 3:
            logger.info(" • Se necesitan más APUs con UNIDAD=M2 para suministros")
        if unit_counts['M3'] < 2:
            logger.info(
                " • Se necesitan más APUs con UNIDAD=M3 para materiales volumétricos"
            )

        logger.info("=" * 60)

    def _try_parse_as_data_line(self, line: str, line_num: int) -> bool:
        """Intenta analizar una línea como una línea de datos con validación mejorada."""
        # 🔴 FILTRO CRÍTICO: Rechazar metadatos antes de cualquier procesamiento
        if self._is_metadata_line(line):
            logger.debug(f"🚫 Rechazando metadato en L{line_num}: {line[:60]}...")
            return False

        has_data_structure = self._has_data_structure(line)
        current_category = self.context["category"]

        logger.debug(f"🔍 Analizando línea {line_num}: cat='{current_category}', datos={has_data_structure}")
        logger.debug(f" Contenido: {line[:80]}...")

        # Intentar patrones específicos primero
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
            # 🔴 VALIDACIÓN ADICIONAL: Verificar que no sea un metadato disfrazado
            descripcion = match_insumo.groupdict().get("descripcion", "").strip().upper()
            if any(meta in descripcion for meta in ['EQUIPO Y HERRAMIENTA', 'IMPUESTOS', 'POLIZAS', 'SEGUROS']):
                logger.warning(f"🚫 Rechazando insumo con metadato: {descripcion}")
                return False

            self._parse_insumo(match_insumo.groupdict())
            return True

        # Fallback parsing con validación mejorada
        if self._has_data_structure(line):
            if self._try_fallback_parsing(line, line_num):
                return True

        logger.debug(" ❌ No se pudo parsear como dato válido")
        return False

    def _try_detect_category_change(self, line: str) -> bool:
        """Detecta cambios de categoría evitando falsos positivos."""
        line_clean = line.strip()
        if not line_clean:
            return False

        first_part = line_clean.split(';')[0].strip().upper()

        category_mappings = {
            "MATERIALES": ["MATERIALES", "MATERIALES Y SUMINISTROS", "MATERIAL"],
            "MANO DE OBRA": ["MANO DE OBRA", "MANO DE OBRA DIRECTA", "M.O.", "MO"],
            "EQUIPO": ["EQUIPO", "EQUIPOS", "EQUIPOS Y HERRAMIENTAS", "MAQUINARIA"],
            "TRANSPORTE": ["TRANSPORTE", "TRANSPORTES", "FLETE"],
            "OTROS": ["OTROS", "OTROS GASTOS", "GASTOS GENERALES"]
        }

        found_category = None
        for category, keywords in category_mappings.items():
            if first_part in keywords:
                found_category = category
                break

        if not found_category:
            return False

        # --- INICIO DE LA CORRECCIÓN ---
        # Usar la función que sí existe: _has_data_structure
        if self._has_data_structure(line):
            logger.debug(f"❌ '{first_part}' parece categoría pero tiene estructura de datos, ignorando.")
            return False
        # --- FIN DE LA CORRECCIÓN ---

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
        """Intenta un análisis genérico como último recurso con validación mejorada."""
        match = self.PATTERNS["generic_data"].match(line)
        if not match:
            return False

        data = match.groupdict()
        desc = data["descripcion"].strip()

        # 🔴 FILTRO CRÍTICO: Rechazar metadatos en fallback
        desc_upper = desc.upper()
        metadata_indicators = [
            'EQUIPO Y HERRAMIENTA', 'IMPUESTOS', 'POLIZAS', 'SEGUROS',
            'GASTOS GENERALES', 'UTILIDAD', 'ADMINISTRACION'
        ]

        if any(meta in desc_upper for meta in metadata_indicators):
            logger.warning(f"🚫 Rechazando fallback con metadato: {desc}")
            return False

        # Validar que tenga contenido real
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

            # 🔴 VALIDACIÓN DE MONTOS RAZONABLES
            if valor_total > 1000000: # Rechazar valores excesivos
                logger.warning(f"🚫 Rechazando valor total excesivo en fallback: {valor_total}")
                return False

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

    def _should_add_insumo(self, desc: str, cantidad: float, valor_total: float) -> bool:
        """Determina si se debe agregar un insumo con criterios más estrictos."""
        if not desc or len(desc.strip()) < 2:
            return False

        # 🔴 CRITERIOS DE EXCLUSIÓN MÁS ESTRICTOS
        desc_upper = desc.upper()

        # Excluir metadatos conocidos
        excluded_terms = [
            'EQUIPO Y HERRAMIENTA', 'IMPUESTOS', 'POLIZAS', 'SEGUROS',
            'GASTOS GENERALES', 'UTILIDAD', 'ADMINISTRACION', 'RETENCIONES'
        ]

        if any(term in desc_upper for term in excluded_terms):
            return False

        # Excluir valores excesivamente altos
        if valor_total > 1000000: # Ajustar según el contexto del proyecto
            logger.warning(f"🚫 Rechazando insumo con valor total excesivo: {valor_total}")
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
        """Detecta si una línea contiene metadatos que deben ser ignorados - VERSIÓN COMPLETA."""
        if not line:
            return True

        upper_line = line.upper().strip()

        # METADATOS CRÍTICOS QUE DEBEN SER IGNORADOS
        critical_metadata = [
            # Costos indirectos y generales
            'EQUIPO Y HERRAMIENTA', 'EQUIPOS Y HERRAMIENTA',
            'IMPUESTOS Y RETENCIONES', 'IMPUESTOS',
            'POLIZAS', 'PÓLIZAS', 'SEGUROS',
            'GASTOS GENERALES', 'COSTOS INDIRECTOS',
            'ADMINISTRACION', 'ADMINISTRACIÓN',
            'UTILIDAD', 'UTILIDADES',

            # Encabezados de tabla
            'DESCRIPCION', 'DESCRIPCIÓN', 'UNIDAD', 'CANTIDAD',
            'PRECIO UNIT', 'PRECIO UNITARIO', 'VALOR TOTAL',
            'ITEM', 'CODIGO', 'CÓDIGO', 'RENDIMIENTO',

            # Subtotales y totales
            'SUBTOTAL', 'TOTAL', 'SUMA',

            # Categorías (solo como encabezados, no como datos)
            'MATERIALES', 'MANO DE OBRA', 'EQUIPO', 'TRANSPORTE', 'OTROS'
        ]
        category_keywords = ['MATERIALES', 'MANO DE OBRA', 'EQUIPO', 'TRANSPORTE', 'OTROS']


        # Verificar coincidencia exacta o casi exacta
        for keyword in critical_metadata:
            # Coincidencia exacta
            if upper_line == keyword:
                return True

            # Coincidencia con separadores (ej: "DESCRIPCION;UNIDAD;CANTIDAD...")
            if re.match(r'^;*\s*' + re.escape(keyword) + r'\s*;*$', upper_line):
                return True

            # Coincidencia como primera parte de la línea
            if upper_line.startswith(keyword + ';') or upper_line.endswith(';' + keyword):
                if keyword in category_keywords and self._has_data_structure(line):
                    continue
                return True

        # Líneas que son puramente separadores o formato
        if re.match(r'^[;=\-\s]+$', upper_line):
            return True

        # Líneas que contienen porcentajes de costos indirectos
        if re.search(r'\d+%', upper_line) and any(kw in upper_line for kw in ['IMPUESTO', 'POLIZA', 'UTILIDAD']):
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
