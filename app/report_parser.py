import re
import pandas as pd
import logging
from typing import List, Dict, Optional

# Configuración básica del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReportParser:
    """
    Analiza un informe de APU (Análisis de Precios Unitarios) desde un archivo de texto,
    extrayendo los datos de insumos y organizándolos en una estructura de datos.
    """
    PATTERNS = {
        'item_code': re.compile(r'^ITEM:\s*([\d,\.]+)'),
        'category': re.compile(r'^(MATERIALES|MANO DE OBRA|EQUIPO Y HERRAMIENTA)'),
        'insumo': re.compile(r'^(?P<descripcion>.+?)\s+(?P<unidad>[A-Z\d/]+)\s+(?P<cantidad>[\d,\.]+)\s+(?P<precio>[\d,\.]+)\s+(?P<total>[\d,\.]+)$'),
        'herramienta_menor': re.compile(r'^(EQUIPO Y HERRAMIENTA)\s+\(MANO DE OBRA\)\s+(\d+)%\s+([\d,\.]+)')
    }

    def __init__(self, file_path: str):
        """
        Inicializa el parser con la ruta al archivo a procesar.

        Args:
            file_path (str): Ruta al archivo de texto del informe de APU.
        """
        self.file_path = file_path
        self._all_data: List[Dict] = []
        self._current_apu_code: Optional[str] = None
        self._current_apu_desc: Optional[str] = None
        self._current_category: Optional[str] = None
        self._line_buffer: str = ""

    def parse(self) -> pd.DataFrame:
        """
        Punto de entrada principal para analizar el archivo.

        Lee el archivo línea por línea, procesa cada una y devuelve
        un DataFrame con los datos extraídos.

        Returns:
            pd.DataFrame: Un DataFrame con los datos de insumos de todas las APU.
                          Columnas: 'apu_code', 'apu_desc', 'categoria', 'descripcion',
                                    'unidad', 'cantidad', 'precio_unitario', 'precio_total'.
        """
        logging.info(f"Iniciando el análisis del archivo: {self.file_path}")
        self._all_data = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self._process_line(line.strip())
        except FileNotFoundError:
            logging.error(f"El archivo no se encontró en la ruta: {self.file_path}")
            return pd.DataFrame(self._all_data)

        logging.info(f"Análisis completado. Se encontraron {len(self._all_data)} registros.")

        if not self._all_data:
            return pd.DataFrame()

        df = pd.DataFrame(self._all_data)
        df['descripcion'] = self._normalize_text(df['descripcion'])
        return df

    def _process_line(self, line: str):
        """
        Procesa una sola línea del archivo, aplicando las expresiones regulares
        en un orden lógico para determinar el tipo de contenido.
        """
        if not line:
            return

        # 1. Buscar código de ITEM
        match = self.PATTERNS['item_code'].match(line)
        if match:
            self._start_new_apu(match.group(1))
            # La descripción del APU puede estar en la misma línea o en la siguiente
            desc_part = line[match.end():].strip()
            if desc_part:
                self._current_apu_desc = desc_part
            return

        # Si aún no tenemos un APU, no podemos procesar más
        if self._current_apu_code is None:
            return

        # Si no tenemos descripción de APU, la línea actual es la descripción
        if self._current_apu_desc is None:
            self._current_apu_desc = line
            return

        # 2. Buscar categoría
        match = self.PATTERNS['category'].match(line)
        if match:
            self._current_category = match.group(1)
            return

        # 3. Buscar Herramienta Menor (caso especial)
        match = self.PATTERNS['herramienta_menor'].match(line)
        if match:
            self._parse_herramienta_menor(match.groups())
            return

        # 4. Buscar Insumo normal
        # Concatenar con el buffer por si la descripción viene en varias líneas
        full_line = self._line_buffer + " " + line if self._line_buffer else line
        match = self.PATTERNS['insumo'].match(full_line)
        if match:
            self._parse_insumo(match.groupdict())
            self._line_buffer = "" # Limpiar buffer tras éxito
        else:
            # Si no coincide, podría ser una descripción de insumo multi-línea
            self._line_buffer = full_line


    def _start_new_apu(self, raw_code: str):
        """
        Reinicia el estado interno para comenzar a procesar una nueva APU.
        """
        self._current_apu_code = raw_code.replace(',', '.')
        self._current_apu_desc = None
        self._current_category = None
        self._line_buffer = ""
        logging.info(f"Nuevo APU encontrado: {self._current_apu_code}")

    def _parse_insumo(self, data: Dict):
        """
        Extrae y almacena los datos de una línea de insumo estándar.
        """
        if not all([self._current_apu_code, self._current_apu_desc, self._current_category]):
            return

        self._all_data.append({
            'apu_code': self._current_apu_code,
            'apu_desc': self._current_apu_desc,
            'categoria': self._current_category,
            'descripcion': data['descripcion'].strip(),
            'unidad': data['unidad'],
            'cantidad': self._to_numeric_safe(data['cantidad']),
            'precio_unitario': self._to_numeric_safe(data['precio']),
            'precio_total': self._to_numeric_safe(data['total'])
        })

    def _parse_herramienta_menor(self, data: tuple):
        """
        Procesa y almacena los datos del caso especial de "Herramienta Menor"
        o "Equipo y Herramienta" calculado como un porcentaje.
        """
        if not all([self._current_apu_code, self._current_apu_desc]):
            return

        descripcion = f"{data[0]} ({data[1]}%)"
        self._all_data.append({
            'apu_code': self._current_apu_code,
            'apu_desc': self._current_apu_desc,
            'categoria': 'EQUIPO Y HERRAMIENTA',
            'descripcion': descripcion,
            'unidad': '%',
            'cantidad': self._to_numeric_safe(data[1]),
            'precio_unitario': 0, # No aplica
            'precio_total': self._to_numeric_safe(data[2])
        })

    def _to_numeric_safe(self, s: str) -> float:
        """
        Convierte una cadena a un número de punto flotante, manejando
        comas decimales y errores de conversión.
        """
        try:
            return float(s.replace(',', ''))
        except (ValueError, TypeError):
            return 0.0

    def _normalize_text(self, series: pd.Series) -> pd.Series:
        """
        Normaliza una serie de texto: convierte a minúsculas, elimina espacios
        extra y otros caracteres no deseados.
        """
        return series.str.lower().str.strip().replace(r'\s+', ' ', regex=True)

if __name__ == '__main__':
    # Ejemplo de uso:
    # Reemplazar 'ruta/a/tu/archivo.txt' con la ruta real del archivo a procesar
    # parser = ReportParser('ruta/a/tu/archivo.txt')
    # df_resultados = parser.parse()
    # if not df_resultados.empty:
    #     print(df_resultados.head())
    #     # Opcional: Guardar en un CSV
    #     df_resultados.to_csv('apus_parseados.csv', index=False)
    pass
