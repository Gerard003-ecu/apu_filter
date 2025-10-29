import logging
import re
from enum import Enum
from typing import Dict, List
from collections import Counter

from .utils import clean_apu_code

logger = logging.getLogger(__name__)


class ParserState(Enum):
    """
    Define los estados posibles de la máquina de estados del parser.

    Attributes:
        IDLE: Estado inicial o esperando un nuevo APU.
        AWAITING_DESCRIPTION: Se encontró un código de APU y espera la línea de descripción.
        SKIPPING_HEADERS: Ignorando líneas de cabecera después de la descripción.
        PROCESSING_APU: Procesando activamente las líneas de insumos de un APU.
    """
    IDLE = "IDLE"
    AWAITING_DESCRIPTION = "AWAITING_DESCRIPTION"
    SKIPPING_HEADERS = "SKIPPING_HEADERS"
    PROCESSING_APU = "PROCESSING_APU"


class ReportParserCrudo:
    """
    Extrae datos crudos de un archivo de reporte de APU sin procesar.

    Esta clase funciona como una máquina de estados para parsear un archivo de texto
    plano y extraer la información de los Análisis de Precios Unitarios (APU).
    La salida es una lista de diccionarios donde cada valor es una cadena de texto,
    sin aplicar conversiones de tipo ni lógica de negocio compleja.

    Attributes:
        file_path (str): La ruta al archivo a parsear.
        raw_records (List[Dict[str, str]]): La lista de registros crudos extraídos.
        state (ParserState): El estado actual de la máquina de estados.
        context (Dict[str, str]): Almacena información del APU actual (código,
                                   descripción, unidad, categoría).
        stats (Dict): Estadísticas del proceso de parseo.
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

    # Mejorado: Incluir variaciones comunes
    CATEGORY_KEYWORDS = {
        "MATERIALES", 
        "MANO DE OBRA", 
        "EQUIPO", 
        "EQUIPOS",
        "OTROS", 
        "TRANSPORTE",
        "TRANSPORTES",
        "HERRAMIENTA",
        "HERRAMIENTAS",
        "SUBCONTRATOS"
    }

    def __init__(self, file_path: str):
        """
        Inicializa el parser con la ruta del archivo.

        Args:
            file_path: La ruta al archivo de reporte de APU.
        """
        self.file_path = file_path
        self.raw_records: List[Dict[str, str]] = []
        self.state = ParserState.IDLE
        self.context = {
            "apu_code": "",
            "apu_desc": "",
            "apu_unit": "",
            "category": "INDEFINIDO",
        }
        # Nuevo: Estadísticas para debugging
        self.stats = {
            "total_lines": 0,
            "apu_count": 0,
            "category_changes": Counter(),
            "skipped_lines": 0,
            "lines_by_state": Counter(),
        }

    def parse_to_raw(self) -> List[Dict[str, str]]:
        """
        Punto de entrada principal para iniciar el proceso de parseo.

        Lee el archivo línea por línea y procesa cada una según la máquina de
        estados.

        Returns:
            Una lista de diccionarios, donde cada diccionario representa un
            registro de insumo crudo extraído del archivo.
        """
        logger.info(f"🔍 Iniciando extracción cruda desde: {self.file_path}")
        try:
            with open(self.file_path, "r", encoding="latin1") as f:
                for line_num, line in enumerate(f, 1):
                    self.stats["total_lines"] += 1
                    self._process_line(line, line_num)
        except Exception as e:
            logger.error(f"❌ Error al leer {self.file_path}: {e}")
            return []
        
        self._log_statistics()
        return self.raw_records

    def _log_statistics(self):
        """Registra estadísticas del proceso de parseo para debugging."""
        logger.info(f"✅ Extracción cruda completada: {len(self.raw_records)} registros")
        logger.info(f"📊 Estadísticas del parseo:")
        logger.info(f"   - Total de líneas procesadas: {self.stats['total_lines']}")
        logger.info(f"   - APUs detectados: {self.stats['apu_count']}")
        logger.info(f"   - Líneas omitidas: {self.stats['skipped_lines']}")
        logger.info(f"   - Cambios de categoría detectados:")
        
        for category, count in self.stats['category_changes'].items():
            logger.info(f"      * {category}: {count} veces")
        
        # Advertencia si no se detectaron categorías
        if sum(self.stats['category_changes'].values()) == 0:
            logger.warning("⚠️  NO SE DETECTARON CAMBIOS DE CATEGORÍA - Revisar formato del archivo")
        
        # Advertencia si todos los insumos están como INDEFINIDO
        indefinidos = sum(1 for r in self.raw_records if r.get('category') == 'INDEFINIDO')
        if indefinidos == len(self.raw_records) and len(self.raw_records) > 0:
            logger.warning(f"⚠️  TODOS los {indefinidos} insumos están marcados como INDEFINIDO")

    def _process_line(self, line: str, line_num: int):
        """
        Procesa una única línea del archivo basado en el estado actual del parser.

        Args:
            line: La línea de texto a procesar.
            line_num: El número de línea actual, para propósitos de logging.
        """
        line = line.strip()
        if not line:
            return

        self.stats["lines_by_state"][self.state.value] += 1

        # Detectar nuevo ITEM en cualquier estado
        if self._try_start_new_apu(line, line_num):
            return

        if self.state == ParserState.IDLE:
            self.stats["skipped_lines"] += 1
            return
        elif self.state == ParserState.AWAITING_DESCRIPTION:
            if self._is_valid_apu_description(line):
                self._capture_apu_description(line)
                self.state = ParserState.PROCESSING_APU
            else:
                self.stats["skipped_lines"] += 1
                return
        elif self.state == ParserState.PROCESSING_APU:
            # CRÍTICO: Detectar categoría ANTES de verificar estructura de datos
            if self._try_detect_category_change(line):
                logger.debug(f"Línea {line_num}: Categoría detectada -> {self.context['category']}")
                return
            if self._has_data_structure(line):
                self._add_raw_record(insumo_line=line)
            else:
                logger.debug(f"Línea {line_num}: Sin estructura de datos -> '{line[:50]}...'")
                self.stats["skipped_lines"] += 1

    def _try_start_new_apu(self, line: str, line_num: int) -> bool:
        """
        Intenta detectar el inicio de un nuevo APU en la línea.

        Si encuentra un "ITEM:", reinicia el contexto y cambia el estado
        del parser.

        Args:
            line: La línea de texto a analizar.
            line_num: El número de línea actual.

        Returns:
            True si se encontró e inició un nuevo APU, False en caso contrario.
        """
        match = self.PATTERNS["item_code"].search(line.upper())
        if not match:
            return False

        raw_code = match.group(1).strip()
        if not raw_code:
            return False

        # Extraer descripción inline si existe
        inline_desc = self._extract_inline_description(line)

        # Extraer unidad inline si existe
        inline_unit = self._extract_inline_unit(line)

        self.context = {
            "apu_code": raw_code,
            "apu_desc": inline_desc,
            "apu_unit": inline_unit or "UND",
            "category": "INDEFINIDO",  # Reiniciar categoría
        }

        self.stats["apu_count"] += 1
        logger.debug(f"Línea {line_num}: Nuevo APU detectado -> {raw_code}")

        if inline_desc:
            self.state = ParserState.PROCESSING_APU
        else:
            self.state = ParserState.AWAITING_DESCRIPTION

        return True

    def _extract_inline_description(self, line: str) -> str:
        """
        Extrae la descripción de un APU si está en la misma línea que el "ITEM:".

        Args:
            line: La línea de texto.

        Returns:
            La descripción encontrada o una cadena vacía.
        """
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
        """
        Extrae la unidad de un APU si está explícitamente en la misma línea ("UNIDAD:").

        Args:
            line: La línea de texto.

        Returns:
            La unidad encontrada o una cadena vacía.
        """
        match = re.search(r"UNIDAD\s*:\s*([^;,\s]+)", line, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _is_valid_apu_description(self, line: str) -> bool:
        """
        Determina si una línea parece ser una descripción válida de APU.

        Se usa cuando el parser está en estado AWAITING_DESCRIPTION.

        Args:
            line: La línea de texto.

        Returns:
            True si la línea parece una descripción válida.
        """
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
        """
        Captura la descripción del APU desde la línea y la guarda en el contexto.

        Args:
            line: La línea que contiene la descripción.
        """
        desc = line.split(";")[0].strip()
        self.context["apu_desc"] = desc
        logger.debug(f"Descripción APU capturada: {desc[:50]}...")

    def _try_detect_category_change(self, line: str) -> bool:
        """
        Detecta si la línea es un cambio de categoría (e.g., "MANO DE OBRA").
        
        MEJORADO: Más flexible y robusto en la detección.

        Args:
            line: La línea de texto.

        Returns:
            True si la línea es un cambio de categoría, False en caso contrario.
        """
        # Normalizar la línea para comparación
        line_upper = line.upper().strip()
        
        # Obtener la primera parte (antes del primer ;)
        first_part = line_upper.split(";")[0].strip()
        
        # Método 1: Coincidencia exacta en la primera parte
        if first_part in self.CATEGORY_KEYWORDS:
            self.context["category"] = first_part
            self.stats["category_changes"][first_part] += 1
            logger.debug(f"✓ Categoría detectada (exacta): {first_part}")
            return True
        
        # Método 2: Buscar palabra clave en cualquier parte de la primera sección
        # (útil si hay caracteres especiales o espacios extra)
        for keyword in self.CATEGORY_KEYWORDS:
            if keyword in first_part:
                # Verificar que no sea parte de una descripción más larga con datos numéricos
                # (evitar falsos positivos)
                if not self._looks_like_insumo_line(line):
                    self.context["category"] = keyword
                    self.stats["category_changes"][keyword] += 1
                    logger.debug(f"✓ Categoría detectada (contenida): {keyword} en '{first_part}'")
                    return True
        
        # Método 3: Buscar en toda la línea si es corta (probablemente un encabezado)
        if len(line_upper) < 50 and ";" not in line_upper[10:]:  # Pocos puntos y comas
            for keyword in self.CATEGORY_KEYWORDS:
                if keyword in line_upper:
                    self.context["category"] = keyword
                    self.stats["category_changes"][keyword] += 1
                    logger.debug(f"✓ Categoría detectada (línea corta): {keyword}")
                    return True
        
        return False

    def _looks_like_insumo_line(self, line: str) -> bool:
        """
        Determina si una línea parece ser un insumo con datos numéricos.
        
        Ayuda a evitar falsos positivos en la detección de categorías.

        Args:
            line: La línea de texto.

        Returns:
            True si la línea parece contener datos de insumo.
        """
        parts = line.split(";")
        if len(parts) < 3:
            return False
        
        # Verificar si hay al menos 2 campos que parezcan numéricos
        numeric_fields = 0
        for part in parts[1:]:  # Saltar descripción
            part_clean = part.strip().replace(",", "").replace(".", "").replace("$", "")
            if part_clean and part_clean.replace("-", "").isdigit():
                numeric_fields += 1
        
        return numeric_fields >= 2

    def _has_data_structure(self, line: str) -> bool:
        """
        Verifica si una línea tiene la estructura de un insumo (contiene ';').

        Args:
            line: La línea de texto.

        Returns:
            True si la línea parece ser un insumo.
        """
        # Mejorado: Verificar que tenga suficientes columnas Y datos numéricos
        if line.count(";") < 2:
            return False
        
        # Verificar que no sea solo una línea de encabezado
        first_part = line.split(";")[0].strip().upper()
        if first_part in {"DESCRIPCION", "CÓDIGO", "CODIGO", "UNIDAD", "CANTIDAD"}:
            return False
        
        return True

    def _add_raw_record(self, **kwargs):
        """
        Crea un nuevo registro crudo y lo añade a la lista de resultados.

        Utiliza la información del contexto actual del APU.

        Args:
            **kwargs: Argumentos clave-valor, se espera 'insumo_line'.
        """
        cleaned_code = clean_apu_code(self.context["apu_code"])
        record = {
            "apu_code": cleaned_code,
            "apu_desc": self.context["apu_desc"],
            "apu_unit": self.context["apu_unit"],
            "category": self.context["category"],
            "insumo_line": kwargs.get("insumo_line", ""),
        }
        self.raw_records.append(record)
        
        # Debug logging cada ciertos registros
        if len(self.raw_records) % 100 == 0:
            logger.debug(f"Registros procesados: {len(self.raw_records)}, "
                        f"Categoría actual: {self.context['category']}")
