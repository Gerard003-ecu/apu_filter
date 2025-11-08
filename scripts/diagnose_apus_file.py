# scripts/diagnose_apus_file.py

import re
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Any
import logging

# Configuración global de logging (ajustable desde fuera si se desea)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class APUFileDiagnostic:
    """
    Herramienta de diagnóstico para analizar la estructura de un archivo de APUs.
    Detecta patrones, separadores, codificación, bloques y estructura general.
    """

    # Constantes para evitar magic strings
    ENCODINGS_TO_TRY = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    CATEGORY_KEYWORDS = ['MATERIALES', 'MANO DE OBRA', 'EQUIPO', 'OTROS']
    MAX_SAMPLE_LINES = 20
    MAX_PATTERN_ANALYSIS_LINES = 100
    MAX_REPORT_SAMPLE_LINES = 10
    MAX_REPORT_PATTERNS = 5

    def __init__(self, file_path: str):
        """
        Inicializa el diagnóstico con la ruta del archivo.
        No realiza procesamiento hasta llamar a `diagnose()`.
        """
        self.file_path = Path(file_path).resolve()
        self._reset_state()

    def _reset_state(self):
        """Restablece todos los estados internos para garantizar limpieza entre ejecuciones."""
        self.stats = Counter()
        self.patterns_found: List[Dict[str, Any]] = []
        self.sample_lines: List[Dict[str, Any]] = []

    def diagnose(self) -> Dict[str, Any]:
        """
        Ejecuta el diagnóstico completo del archivo.

        Returns:
            Diccionario con estadísticas, patrones y muestras, o dict vacío si falla.
        """
        self._reset_state()

        if not self.file_path.exists():
            logger.error(f"❌ Archivo no encontrado: {self.file_path}")
            return {}

        if not self.file_path.is_file():
            logger.error(f"❌ Ruta no es un archivo: {self.file_path}")
            return {}

        logger.info(f"🔍 Analizando archivo: {self.file_path}")
        try:
            file_size = self.file_path.stat().st_size
            logger.info(f"📦 Tamaño: {file_size:,} bytes")
            self.stats['file_size'] = file_size
        except OSError as e:
            logger.error(f"❌ No se pudo obtener el tamaño del archivo: {e}")
            return {}

        content = self._read_with_fallback_encoding()
        if not content:
            return {}

        lines = content.splitlines()
        self.stats['total_lines'] = len(lines)

        # Análisis línea por línea
        self._analyze_lines(lines)

        # Análisis estructural
        self._analyze_structure(content)

        # Detección de patrones clave
        self._detect_patterns(lines)

        # Generar reporte (en logging)
        self._generate_diagnostic_report()

        return {
            'stats': dict(self.stats),
            'patterns': self.patterns_found,
            'samples': self.sample_lines
        }

    def _read_with_fallback_encoding(self) -> Optional[str]:
        """Intenta leer el archivo con múltiples encodings. Devuelve None si falla todo."""
        for encoding in self.ENCODINGS_TO_TRY:
            try:
                content = self.file_path.read_text(encoding=encoding, errors='replace')
                logger.info(f"✅ Archivo leído con encoding: {encoding}")
                self.stats['encoding'] = encoding
                return content
            except (UnicodeError, OSError, ValueError) as e:
                logger.debug(f"❌ Fallo al leer con encoding '{encoding}': {type(e).__name__}: {e}")
                continue

        logger.error("❌ No se pudo leer el archivo con ninguno de los encodings soportados.")
        return None

    def _analyze_lines(self, lines: List[str]):
        """Analiza cada línea para detectar separadores, palabras clave y características."""
        item_pattern = re.compile(r'ITEM', re.IGNORECASE)
        unidad_pattern = re.compile(r'UNIDAD', re.IGNORECASE)
        descripcion_pattern = re.compile(r'DESCRIPCION|DESCRIPCIÓN', re.IGNORECASE)

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            line_len = len(stripped)

            if not stripped:
                self.stats['empty_lines'] += 1
                continue

            self.stats['non_empty_lines'] += 1

            # Guardar muestras (solo primeras MAX_SAMPLE_LINES no vacías)
            if len(self.sample_lines) < self.MAX_SAMPLE_LINES:
                self.sample_lines.append({
                    'line_num': line_num,
                    'content': stripped[:200],
                    'length': line_len
                })

            # Separadores
            if ';' in stripped:
                self.stats['lines_with_semicolon'] += 1
                semicolon_count = stripped.count(';')
                self.stats['max_semicolons'] = max(self.stats.get('max_semicolons', 0), semicolon_count)

            if '\t' in line:
                self.stats['lines_with_tabs'] += 1

            if re.search(r'\s{2,}', stripped):
                self.stats['lines_with_multiple_spaces'] += 1

            # Palabras clave
            if item_pattern.search(stripped):
                self.stats['lines_with_ITEM'] += 1
                if line_num <= 50:
                    self.patterns_found.append({
                        'type': 'ITEM',
                        'line_num': line_num,
                        'content': stripped[:150]
                    })

            if unidad_pattern.search(stripped):
                self.stats['lines_with_UNIDAD'] += 1

            if descripcion_pattern.search(stripped):
                self.stats['lines_with_DESCRIPCION'] += 1

            # Categorías
            for category in self.CATEGORY_KEYWORDS:
                if category in stripped:
                    self.stats[f'category_{category}'] += 1

    def _analyze_structure(self, content: str):
        """Analiza la estructura del archivo por bloques lógicos."""
        # Bloques por doble salto de línea
        blocks = re.split(r'\n\s*\n', content)
        self.stats['blocks_by_double_newline'] = len([b for b in blocks if b.strip()])

        # Bloques por guiones (---)
        blocks_dashes = re.split(r'\n-{3,}\n', content)
        self.stats['blocks_by_dashes'] = len([b for b in blocks_dashes if b.strip()])

        # Bloques por igual (===)
        blocks_equals = re.split(r'\n={3,}\n', content)
        self.stats['blocks_by_equals'] = len([b for b in blocks_equals if b.strip()])

    def _detect_patterns(self, lines: List[str]):
        """Detecta patrones estructurales en las primeras líneas del archivo."""
        pattern_item = re.compile(r'ITEM\s*[:\s]\s*([^\s;,]+)', re.IGNORECASE)
        pattern_unit = re.compile(r'UNIDAD\s*[:\s]\s*([^\s;,]+)', re.IGNORECASE)
        pattern_numeric_row = re.compile(r'(?:[\d.,]+\s+){2,}[\d.,]+')  # Al menos 3 números

        lines_to_analyze = lines[:self.MAX_PATTERN_ANALYSIS_LINES]

        for line_num, line in enumerate(lines_to_analyze, 1):
            stripped = line.strip()

            # ITEM: código
            match = pattern_item.search(stripped)
            if match:
                self.patterns_found.append({
                    'type': 'ITEM_CODE',
                    'line_num': line_num,
                    'value': match.group(1),
                    'full_line': stripped[:150]
                })

            # UNIDAD: valor
            match = pattern_unit.search(stripped)
            if match:
                self.patterns_found.append({
                    'type': 'UNIT',
                    'line_num': line_num,
                    'value': match.group(1),
                    'full_line': stripped[:150]
                })

            # Fila numérica (posible fila de datos)
            if pattern_numeric_row.search(stripped):
                self.stats['numeric_rows'] += 1

    def _generate_diagnostic_report(self):
        """Genera un reporte formateado usando logging."""
        report = [
            "\n" + "=" * 80,
            "📊 REPORTE DE DIAGNÓSTICO DEL ARCHIVO APU",
            "=" * 80,
            "\n📈 ESTADÍSTICAS GENERALES:"
        ]

        report.append(f"  Total de líneas: {self.stats.get('total_lines', 0):,}")
        report.append(f"  Líneas vacías: {self.stats.get('empty_lines', 0):,}")
        report.append(f"  Líneas con contenido: {self.stats.get('non_empty_lines', 0):,}")
        report.append(f"  Encoding detectado: {self.stats.get('encoding', 'desconocido')}")

        report += [
            "\n🔍 SEPARADORES DETECTADOS:",
            f"  Líneas con punto y coma (;): {self.stats.get('lines_with_semicolon', 0):,}",
            f"  Máximo de ';' por línea: {self.stats.get('max_semicolons', 0)}",
            f"  Líneas con tabulaciones: {self.stats.get('lines_with_tabs', 0):,}",
            f"  Líneas con espacios múltiples: {self.stats.get('lines_with_multiple_spaces', 0):,}",
        ]

        report += [
            "\n🏗️ ESTRUCTURA DEL ARCHIVO:",
            f"  Bloques (por doble salto): {self.stats.get('blocks_by_double_newline', 0)}",
            f"  Bloques (por guiones): {self.stats.get('blocks_by_dashes', 0)}",
            f"  Bloques (por signos igual): {self.stats.get('blocks_by_equals', 0)}",
        ]

        report += [
            "\n🔑 PALABRAS CLAVE ENCONTRADAS:",
            f"  Líneas con 'ITEM': {self.stats.get('lines_with_ITEM', 0):,}",
            f"  Líneas con 'UNIDAD': {self.stats.get('lines_with_UNIDAD', 0):,}",
            f"  Líneas con 'DESCRIPCION': {self.stats.get('lines_with_DESCRIPCION', 0):,}",
            f"  Filas numéricas: {self.stats.get('numeric_rows', 0):,}",
        ]

        # Categorías
        categories_found = [
            f"  {cat}: {self.stats.get(f'category_{cat}', 0)} veces"
            for cat in self.CATEGORY_KEYWORDS
            if self.stats.get(f'category_{cat}', 0) > 0
        ]
        if categories_found:
            report += ["\n📦 CATEGORÍAS DETECTADAS:"] + categories_found
        else:
            report += ["\n📦 CATEGORÍAS DETECTADAS: Ninguna identificada claramente"]

        # Muestra de líneas
        report += ["\n📝 MUESTRA DE PRIMERAS LÍNEAS:"]
        for sample in self.sample_lines[:self.MAX_REPORT_SAMPLE_LINES]:
            report.append(f"  Línea {sample['line_num']:4d} ({sample['length']:3d} chars): {sample['content']}")

        # Patrones clave
        report += ["\n🎯 PATRONES CLAVE DETECTADOS:"]

        item_codes = [p for p in self.patterns_found if p['type'] == 'ITEM_CODE']
        if item_codes:
            report.append(f"\n  ✓ Códigos ITEM encontrados: {len(item_codes)}")
            for p in item_codes[:self.MAX_REPORT_PATTERNS]:
                report.append(f"    Línea {p['line_num']}: {p['value']}")
                report.append(f"      → {p['full_line']}")

        units = [p for p in self.patterns_found if p['type'] == 'UNIT']
        if units:
            report.append(f"\n  ✓ Unidades encontradas: {len(units)}")
            for p in units[:self.MAX_REPORT_PATTERNS]:
                report.append(f"    Línea {p['line_num']}: {p['value']}")

        # Recomendaciones inteligentes
        report += ["\n💡 RECOMENDACIONES:"]
        non_empty = self.stats.get('non_empty_lines', 1)

        if self.stats.get('lines_with_semicolon', 0) > non_empty * 0.5:
            report.append("  → El archivo usa PUNTO Y COMA (;) como separador principal")
            report.append("  → Considerar parsing con split(';') o csv con delimitador ';'")

        if self.stats.get('blocks_by_double_newline', 0) > 10:
            report.append("  → El archivo tiene múltiples bloques separados por líneas vacías")
            report.append("  → Considerar parsing por secciones o bloques lógicos")

        if self.stats.get('lines_with_ITEM', 0) > 0:
            report.append(f"  → Se detectaron {self.stats['lines_with_ITEM']} líneas con 'ITEM'")
            report.append("  → Posiblemente 'ITEM' marca el inicio de una entrada APU")
        else:
            report.append("  ⚠️ NO se detectaron líneas con 'ITEM' - verificar formato esperado")

        if self.stats.get('numeric_rows', 0) > 5:
            report.append("  → Hay varias filas numéricas consecutivas")
            report.append("  → Podrían representar datos tabulares (precios, cantidades)")

        report.append("=" * 80 + "\n")

        # Imprimir todo el reporte
        for line in report:
            logger.info(line)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Uso: python diagnose_apus_file.py <ruta_archivo>")
        logger.error("Ejemplo: python diagnose_apus_file.py data/apus.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    diagnostic = APUFileDiagnostic(file_path)
    result = diagnostic.diagnose()

    if not result:
        logger.error("❌ Diagnóstico fallido. Revisa los mensajes anteriores.")
        sys.exit(1)