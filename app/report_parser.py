import re
import pandas as pd
import logging
import csv
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_delimiter(file_path: str) -> str:
    """
    Detecta el delimitador (',' o ';') de un archivo CSV usando csv.Sniffer.
    """
    try:
        with open(file_path, "r", encoding="latin1") as f:
            sample = f.read(2048)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=";,")
            return dialect.delimiter
    except (csv.Error, FileNotFoundError):
        # Fallback a ; si el sniffer falla o el archivo no existe
        return ";"

class ReportParser:
    """
    Parsea un informe de APU (Análisis de Precios Unitarios) desde un archivo
    de texto con formato CSV, extrayendo los datos de insumos y
    organizándolos en una estructura de datos.
    """
    PATTERNS = {
        'item_code': re.compile(r'ITEM:\s*([\d,\.]*)'),
    }
    CATEGORY_KEYWORDS = {
        "MATERIALES": "MATERIALES",
        "MANO DE OBRA": "MANO DE OBRA",
        "EQUIPO Y HERRAMIENTA": "EQUIPO Y HERRAMIENTA",
        "EQUIPO": "EQUIPO Y HERRAMIENTA",
        "OTROS": "OTROS"
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._all_data: List[Dict] = []
        self._current_apu_code: Optional[str] = None
        self._current_apu_desc: str = ""
        self._potential_apu_desc: str = ""
        self._current_category: str = "INDEFINIDO"

    def _to_numeric_safe(self, s: str) -> float:
        if isinstance(s, (int, float)):
            return float(s)
        if isinstance(s, str):
            s_cleaned = s.replace(".", "").replace(",", ".").strip()
            num = pd.to_numeric(s_cleaned, errors="coerce")
            return float(num) if pd.notna(num) else 0.0
        return 0.0

    def parse(self) -> pd.DataFrame:
        logging.info(f"Iniciando el análisis del archivo: {self.file_path}")
        self._all_data = []
        try:
            delimiter = detect_delimiter(self.file_path)
            with open(self.file_path, 'r', encoding='latin1') as f:
                reader = csv.reader(f, delimiter=delimiter, quotechar='"')
                for parts in reader:
                    self._process_line(parts)
        except FileNotFoundError:
            logging.error(f"El archivo no se encontró en la ruta: {self.file_path}")
            return pd.DataFrame()

        logging.info(f"Análisis completado. Se encontraron {len(self._all_data)} registros.")

        if not self._all_data:
            return pd.DataFrame()

        df = pd.DataFrame(self._all_data)

        # Añadir columna normalizada para compatibilidad con el resto del pipeline
        if 'descripcion' in df.columns:
            df['NORMALIZED_DESC'] = df['descripcion'].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)

        return df

    def _process_line(self, parts: List[str]):
        if not any(p.strip() for p in parts):
            return

        parts_stripped = [p.strip() for p in parts]

        # 1. Buscar código de ITEM
        for part in parts_stripped:
            match = self.PATTERNS['item_code'].search(part.upper())
            if match and match.group(1).strip():
                self._start_new_apu(match.group(1).strip().rstrip(".,"))
                return

        # 2. Buscar categoría
        line_content_for_category = "".join(parts_stripped)
        if line_content_for_category in self.CATEGORY_KEYWORDS:
            self._current_category = self.CATEGORY_KEYWORDS[line_content_for_category]
            return

        # 3. Buscar Insumo (lógica adaptada de process_apus_csv_v2)
        is_insumo = False
        if self._current_apu_code and len(parts_stripped) >= 6 and parts_stripped[0]:
            if "DESCRIPCION" not in parts_stripped[0].upper() and "SUBTOTAL" not in parts_stripped[0].upper():
                # Comprobar si hay valores numéricos en las columnas esperadas
                try:
                    cantidad_val = self._to_numeric_safe(parts_stripped[2])
                    precio_val = self._to_numeric_safe(parts_stripped[4])
                    valor_val = self._to_numeric_safe(parts_stripped[5])
                    if cantidad_val > 0 or precio_val > 0 or valor_val > 0:
                        is_insumo = True
                except IndexError:
                    is_insumo = False

        if is_insumo:
            self._parse_insumo(parts_stripped)
            return

        # 4. Si no es nada de lo anterior, podría ser la descripción del APU
        if parts_stripped and parts_stripped[0]:
            self._potential_apu_desc = parts_stripped[0]

    def _start_new_apu(self, raw_code: str):
        self._current_apu_code = raw_code
        self._current_apu_desc = self._potential_apu_desc
        self._current_category = "INDEFINIDO"
        self._potential_apu_desc = ""
        logging.info(f"Nuevo APU encontrado: {self._current_apu_code} - {self._current_apu_desc}")

    def _parse_insumo(self, parts: List[str]):
        # Lógica adaptada de la función `parse_data_line` original
        description = parts[0]

        # Lógica para Mano de Obra
        is_mano_de_obra = self._current_category == "MANO DE OBRA" or description.upper().startswith("M.O.")

        cantidad, precio_unit, valor_total = 0.0, 0.0, 0.0

        try:
            if is_mano_de_obra and len(parts) >= 6:
                valor_total = self._to_numeric_safe(parts[5])
                precio_unitario_jornal = self._to_numeric_safe(parts[3])
                rendimiento = self._to_numeric_safe(parts[4])

                if rendimiento != 0:
                    cantidad = 1 / rendimiento
                precio_unit = precio_unitario_jornal
            else:
                cantidad = self._to_numeric_safe(parts[2])
                precio_unit = self._to_numeric_safe(parts[4])
                valor_total = self._to_numeric_safe(parts[5])

                if valor_total == 0 and cantidad > 0 and precio_unit > 0:
                    valor_total = cantidad * precio_unit
        except IndexError:
            logging.warning(f"Línea de insumo mal formada omitida: {parts}")
            return

        if valor_total > 0:
            self._all_data.append({
                'apu_code': self._current_apu_code,
                'apu_desc': self._current_apu_desc,
                'categoria': self._current_category,
                'descripcion': description,
                'unidad': parts[1],
                'cantidad': cantidad,
                'precio_unitario': precio_unit,
                'precio_total': valor_total
            })
        else:
            logging.debug(f"Línea de insumo sin valor total omitida: {parts}")
