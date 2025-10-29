import logging
import re
from enum import Enum
from typing import Dict, List

from .utils import clean_apu_code

logger = logging.getLogger(__name__)


class ParserState(Enum):
    """
    Define los estados posibles de la máquina de estados del parser.

    Attributes:
        IDLE: Estado inicial o esperando un nuevo APU.
        AWAITING_DESCRIPTION: Se encontró un código de APU y se espera la línea de descripción.
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
                    self._process_line(line, line_num)
        except Exception as e:
            logger.error(f"❌ Error al leer {self.file_path}: {e}")
            return []
        logger.info(f"✅ Extracción cruda completada: {len(self.raw_records)} registros")
        return self.raw_records

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

    def _try_detect_category_change(self, line: str) -> bool:
        """
        Detecta si la línea es un cambio de categoría (e.g., "MANO DE OBRA").

        Args:
            line: La línea de texto.

        Returns:
            True si la línea es un cambio de categoría, False en caso contrario.
        """
        first_part = line.split(";")[0].strip().upper()
        if first_part in self.CATEGORY_KEYWORDS and not self._has_data_structure(line):
            self.context["category"] = first_part
            return True
        return False

    def _has_data_structure(self, line: str) -> bool:
        """
        Verifica si una línea tiene la estructura de un insumo (contiene ';').

        Args:
            line: La línea de texto.

        Returns:
            True si la línea parece ser un insumo.
        """
        return line.count(";") >= 2

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
