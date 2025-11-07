import logging
import re
from collections import Counter
from enum import Enum
from typing import Dict, List

from .utils import clean_apu_code

logger = logging.getLogger(__name__)


class ParserState(Enum):
    """
    Define los estados posibles de la máquina de estados del parser.

    Attributes:
        IDLE: Estado inicial o esperando un nuevo APU.
        AWAITING_DESCRIPTION: Se encontró un código de APU y espera la línea de descripción.
        PROCESSING_APU: Procesando activamente las líneas de insumos de un APU.
    """
    IDLE = "IDLE"
    AWAITING_DESCRIPTION = "AWAITING_DESCRIPTION"
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
        _apu_start_line (int): Línea donde se inició el APU actual (para diagnóstico).
    """

    PATTERNS = {
        "item_code": re.compile(r"ITEM:\s*([^;]+)", re.IGNORECASE),
        "inline_desc": re.compile(r"ITEM:\s*[^;]+;\s*(?:DESCRIPCION|DESCRIPCIÓN)\s*:\s*([^;]+)", re.IGNORECASE),
        "inline_unit": re.compile(r"UNIDAD\s*:\s*([^;,\s]+)", re.IGNORECASE),
        "header_keywords": re.compile(r"^(?:ITEM|DESCRIPCION|DESCRIPCIÓN|UNIDAD|CANTIDAD|VALOR\s+TOTAL|CODIGO|CÓDIGO)$", re.IGNORECASE),
    }

    # Mejorado: Incluir variaciones comunes y normalizadas
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
        "SUBCONTRATOS",
        "SERVICIOS",
        "INSUMOS",
    }

    # Límite de caracteres para considerar una línea como categoría (evita falsos positivos)
    MAX_CATEGORY_LINE_LENGTH = 80

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
            "apu_unit": "UND",  # Valor por defecto
            "category": "INDEFINIDO",
        }
        self._apu_start_line = 0  # Para rastrear origen de APU actual

        # Estadísticas mejoradas
        self.stats = {
            "total_lines": 0,
            "apu_count": 0,
            "category_changes": Counter(),
            "skipped_lines": 0,
            "lines_by_state": Counter(),
            "invalid_apu_codes": 0,
            "insumos_sin_categoria": 0,
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
            logger.error(f"❌ Error al leer {self.file_path}: {e}", exc_info=True)
            return []

        self._log_statistics()
        return self.raw_records

    def _log_statistics(self):
        """Registra estadísticas del proceso de parseo para debugging."""
        logger.info(f"✅ Extracción cruda completada: {len(self.raw_records)} registros")
        logger.info("📊 Estadísticas del parseo:")
        logger.info(f"   - Total de líneas procesadas: {self.stats['total_lines']}")
        logger.info(f"   - APUs detectados: {self.stats['apu_count']}")
        logger.info(f"   - Líneas omitidas: {self.stats['skipped_lines']}")
        logger.info(f"   - Códigos APU inválidos: {self.stats['invalid_apu_codes']}")
        logger.info(f"   - Insumos sin categoría asignada: {self.stats['insumos_sin_categoria']}")

        if self.stats['category_changes']:
            logger.info("   - Cambios de categoría detectados:")
            for category, count in self.stats['category_changes'].most_common():
                logger.info(f"      * {category}: {count} veces")
        else:
            logger.warning(
                "⚠️ NO SE DETECTARON CAMBIOS DE CATEGORÍA - Revisar formato del archivo"
            )

        if len(self.raw_records) > 0 and all(
            r.get('category') == 'INDEFINIDO' for r in self.raw_records
        ):
            logger.warning(
                f"⚠️ TODOS los {len(self.raw_records)} insumos están marcados como INDEFINIDO"
            )

    def _process_line(self, line: str, line_num: int):
        """
        Procesa una única línea del archivo basado en el estado actual del parser.

        Args:
            line: La línea de texto a procesar.
            line_num: El número de línea actual, para propósitos de logging.
        """
        line_clean = line.strip()
        if not line_clean:
            return

        self.stats["lines_by_state"][self.state.value] += 1

        # Siempre intentar detectar nuevo APU primero (prioridad máxima)
        if self._try_start_new_apu(line_clean, line_num):
            return

        # Si estamos en IDLE, ignoramos (ya se hizo en try_start_new_apu)
        if self.state == ParserState.IDLE:
            self.stats["skipped_lines"] += 1
            return

        # Estado AWAITING_DESCRIPTION: esperamos descripción. Si no llega en 5 líneas, forzamos.
        if self.state == ParserState.AWAITING_DESCRIPTION:
            if self._is_valid_apu_description(line_clean):
                self._capture_apu_description(line_clean)
                self.state = ParserState.PROCESSING_APU
            else:
                # Si pasan 5 líneas sin descripción, asumimos que no hay y pasamos a processing con descripción vacía
                if line_num > self._apu_start_line + 5:
                    logger.warning(
                        f"Línea {line_num}: APU {self.context['apu_code']} sin descripción tras 5 líneas. Forzando a PROCESSING_APU."
                    )
                    self.state = ParserState.PROCESSING_APU
                else:
                    self.stats["skipped_lines"] += 1
                    return

        # Estado PROCESSING_APU: procesar insumos o cambios de categoría
        elif self.state == ParserState.PROCESSING_APU:
            # 1. Intentar detectar cambio de categoría (prioridad alta)
            if self._try_detect_category_change(line_clean, line_num):
                return

            # 2. Validar si es línea de insumo
            if self._has_valid_insumo_structure(line_clean):
                self._add_raw_record(insumo_line=line_clean)
            else:
                # Posible encabezado secundario, comentario, o línea corrupta
                log_line = line_clean[:50]
                logger.debug(f"Línea {line_num}: No es insumo válido -> '{log_line}...' (estado: {self.state.value})")
                self.stats["skipped_lines"] += 1

    def _try_start_new_apu(self, line: str, line_num: int) -> bool:
        """
        Intenta detectar el inicio de un nuevo APU en la línea.

        Si encuentra un "ITEM:", reinicia el contexto, cierra el APU anterior si existía,
        y cambia el estado del parser.

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
        if not raw_code or len(raw_code) < 3 or not re.search(r'[A-Za-z0-9]', raw_code):
            logger.warning(f"Línea {line_num}: Código APU inválido o demasiado corto: '{raw_code}'")
            self.stats["invalid_apu_codes"] += 1
            return False

        # Si ya había un APU activo, cerrarlo implícitamente (evita fugas de contexto)
        if self.state != ParserState.IDLE:
            if self.state == ParserState.AWAITING_DESCRIPTION:
                logger.warning(f"Línea {line_num}: Nuevo APU iniciado sin descripción para APU anterior: {self.context['apu_code']}")
            # Registrar que se cerró un APU implícitamente
            if self.context["apu_code"] and self.context["apu_desc"]:
                pass  # Ya se registró el insumo, no hay que hacer nada más

        # Extraer descripción inline
        inline_desc_match = self.PATTERNS["inline_desc"].search(line)
        inline_desc = inline_desc_match.group(1).strip() if inline_desc_match else ""

        # Extraer unidad inline
        inline_unit_match = self.PATTERNS["inline_unit"].search(line)
        inline_unit = inline_unit_match.group(1).strip() if inline_unit_match else "UND"

        # Limpiar código APU inmediatamente
        cleaned_code = clean_apu_code(raw_code)

        # Reiniciar contexto
        self.context = {
            "apu_code": cleaned_code,
            "apu_desc": inline_desc,
            "apu_unit": inline_unit,
            "category": "INDEFINIDO",
        }
        self._apu_start_line = line_num
        self.state = ParserState.AWAITING_DESCRIPTION if not inline_desc else ParserState.PROCESSING_APU

        self.stats["apu_count"] += 1
        logger.debug(f"Línea {line_num}: Nuevo APU detectado -> {cleaned_code} (desc: '{inline_desc[:30]}...')")

        return True

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

        # Evitar encabezados
        if self.PATTERNS["header_keywords"].fullmatch(first_part.upper()):
            return False

        # Evitar líneas puramente numéricas o de separadores
        if re.match(r"^[.,\d\s$%\-]+$", first_part):
            return False

        # Requerir longitud mínima
        if len(first_part) < 5:
            return False

        # Evitar que sea solo un código o número
        if re.match(r"^[A-Z0-9]{1,6}$", first_part):
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
        logger.debug(f"Línea {self._apu_start_line}: Descripción APU capturada: '{desc[:50]}...'")

    def _try_detect_category_change(self, line: str, line_num: int) -> bool:
        """
        Detecta si la línea es un cambio de categoría (e.g., "MANO DE OBRA").

        Solo se considera categoría si:
        - La línea es corta (< 80 chars)
        - No contiene más de un ';'
        - La primera parte coincide con una palabra clave

        Args:
            line: La línea de texto.
            line_num: Número de línea para logging.

        Returns:
            True si la línea es un cambio de categoría, False en caso contrario.
        """
        line_clean = line.strip()
        line_upper = line_clean.upper()

        # Si es muy larga, no es categoría
        if len(line_clean) > self.MAX_CATEGORY_LINE_LENGTH:
            return False

        # Si tiene más de un ;, probablemente es un insumo
        if line_clean.count(";") > 1:
            return False

        # Obtener primera parte (antes del primer ;)
        first_part = line_upper.split(";")[0].strip()

        # Si está vacía, ignorar
        if not first_part:
            return False

        # Método 1: Coincidencia exacta
        if first_part in self.CATEGORY_KEYWORDS:
            old_category = self.context["category"]
            self.context["category"] = first_part
            if old_category != first_part:
                self.stats["category_changes"][first_part] += 1
                logger.debug(f"Línea {line_num}: Categoría cambiada a '{first_part}' (desde '{old_category}')")
            return True

        # Método 2: Contiene categoría como substring (solo si no hay ; ni datos numéricos)
        for keyword in self.CATEGORY_KEYWORDS:
            if keyword in first_part:
                # Evitar falsos positivos: si contiene números o símbolos típicos de insumos, no es categoría
                if self._looks_like_insumo_line(line_clean):
                    continue
                old_category = self.context["category"]
                self.context["category"] = keyword
                if old_category != keyword:
                    self.stats["category_changes"][keyword] += 1
                    logger.debug(f"Línea {line_num}: Categoría detectada por substring: '{keyword}' en '{first_part}'")
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

        numeric_count = 0
        for part in parts[1:]:  # Saltar descripción
            part_clean = part.strip().replace(",", "").replace(".", "").replace("$", "").replace("-", "")
            if part_clean and part_clean.isdigit():
                numeric_count += 1
                if numeric_count >= 2:  # Al menos 2 campos numéricos
                    return True
        return False

    def _has_valid_insumo_structure(self, line: str) -> bool:
        """
        Verifica si una línea tiene estructura válida de insumo.

        Requisitos:
        - Al menos 3 columnas separadas por ;
        - La primera columna no es un encabezado conocido
        - Al menos una columna posterior contiene datos numéricos o alfanuméricos significativos
        - No es una línea vacía o solo con separadores

        Args:
            line: La línea de texto.

        Returns:
            True si la línea es un insumo válido.
        """
        parts = line.split(";")
        if len(parts) < 3:
            return False

        # Primera parte: descripción
        first_part = parts[0].strip()
        if not first_part:
            return False

        # Evitar encabezados
        if self.PATTERNS["header_keywords"].fullmatch(first_part.upper()):
            return False

        # Al menos una columna posterior debe tener contenido no vacío y no solo símbolos
        has_valid_field = False
        for part in parts[1:]:
            part_clean = part.strip()
            if part_clean and not re.fullmatch(r"^[.,\s$%\-]+$", part_clean):
                has_valid_field = True
                break

        return has_valid_field

    def _add_raw_record(self, **kwargs):
        """
        Crea un nuevo registro crudo y lo añade a la lista de resultados.

        Utiliza la información del contexto actual del APU.
        Limpia la línea de insumo antes de guardar.

        Args:
            **kwargs: Argumentos clave-valor, se espera 'insumo_line'.
        """
        insumo_line = kwargs.get("insumo_line", "").strip()
        if not insumo_line:
            logger.warning(f"Intento de agregar registro sin línea de insumo. Contexto: {self.context}")
            return

        # Limpiar insumo_line: eliminar espacios extra, tabuladores, saltos
        insumo_line_clean = " ".join(insumo_line.split())

        # Verificar si la categoría es INDEFINIDO y registrar estadística
        if self.context["category"] == "INDEFINIDO":
            self.stats["insumos_sin_categoria"] += 1

        record = {
            "apu_code": self.context["apu_code"],
            "apu_desc": self.context["apu_desc"],
            "apu_unit": self.context["apu_unit"],
            "category": self.context["category"],
            "insumo_line": insumo_line_clean,
        }
        self.raw_records.append(record)

        # Logging cada 100 registros
        if len(self.raw_records) % 100 == 0:
            logger.debug(
                f"Registros procesados: {len(self.raw_records)} | "
                f"Categoría actual: {self.context['category']} | "
                f"APU: {self.context['apu_code']}"
            )