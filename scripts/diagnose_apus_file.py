# scripts/diagnose_apus_file.py

import re
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class APUFileDiagnostic:
    """
    Herramienta de diagnóstico para analizar la estructura de un archivo de APUs.
    """
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.stats = Counter()
        self.patterns_found = []
        self.sample_lines = []
        
    def diagnose(self) -> dict:
        """Ejecuta diagnóstico completo del archivo."""
        
        if not self.file_path.exists():
            logger.error(f"❌ Archivo no encontrado: {self.file_path}")
            return {}
        
        logger.info(f"🔍 Analizando: {self.file_path}")
        logger.info(f"📦 Tamaño: {self.file_path.stat().st_size:,} bytes")
        
        # Intentar diferentes encodings
        content = self._read_with_fallback()
        if not content:
            return {}
        
        lines = content.split('\n')
        self.stats['total_lines'] = len(lines)
        
        # Análisis línea por línea
        self._analyze_lines(lines)
        
        # Análisis de estructura
        self._analyze_structure(content)
        
        # Buscar patrones clave
        self._detect_patterns(lines)
        
        # Mostrar resultados
        self._print_report()
        
        return {
            'stats': dict(self.stats),
            'patterns': self.patterns_found,
            'samples': self.sample_lines
        }
    
    def _read_with_fallback(self) -> str:
        """Intenta leer con múltiples encodings."""
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                content = self.file_path.read_text(encoding=encoding)
                logger.info(f"✅ Archivo leído con encoding: {encoding}")
                self.stats['encoding'] = encoding
                return content
            except Exception as e:
                logger.debug(f"Falló encoding {encoding}: {e}")
        
        logger.error("❌ No se pudo leer el archivo con ningún encoding")
        return ""
    
    def _analyze_lines(self, lines: list):
        """Analiza características de las líneas."""
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if not stripped:
                self.stats['empty_lines'] += 1
                continue
            
            self.stats['non_empty_lines'] += 1
            
            # Guardar muestras
            if len(self.sample_lines) < 20:
                self.sample_lines.append({
                    'line_num': i,
                    'content': stripped[:200],
                    'length': len(stripped)
                })
            
            # Detectar separadores
            if ';' in stripped:
                self.stats['lines_with_semicolon'] += 1
                self.stats['max_semicolons'] = max(
                    self.stats.get('max_semicolons', 0),
                    stripped.count(';')
                )
            
            if '\t' in line:
                self.stats['lines_with_tabs'] += 1
            
            if re.search(r'\s{2,}', stripped):
                self.stats['lines_with_multiple_spaces'] += 1
            
            # Detectar palabras clave
            upper = stripped.upper()
            if 'ITEM' in upper:
                self.stats['lines_with_ITEM'] += 1
                if i <= 50:  # Guardar primeros 50
                    self.patterns_found.append({
                        'type': 'ITEM',
                        'line_num': i,
                        'content': stripped[:150]
                    })
            
            if 'UNIDAD' in upper:
                self.stats['lines_with_UNIDAD'] += 1
            
            if 'DESCRIPCION' in upper or 'DESCRIPCIÓN' in upper:
                self.stats['lines_with_DESCRIPCION'] += 1
            
            # Detectar categorías
            for category in ['MATERIALES', 'MANO DE OBRA', 'EQUIPO', 'OTROS']:
                if category in upper:
                    self.stats[f'category_{category}'] += 1
    
    def _analyze_structure(self, content: str):
        """Analiza la estructura del archivo."""
        
        # Buscar bloques separados por líneas vacías
        blocks = re.split(r'\n\s*\n', content)
        self.stats['blocks_by_double_newline'] = len([b for b in blocks if b.strip()])
        
        # Buscar bloques separados por líneas de guiones
        blocks_by_dashes = re.split(r'\n-{3,}\n', content)
        self.stats['blocks_by_dashes'] = len([b for b in blocks_by_dashes if b.strip()])
        
        # Buscar bloques separados por líneas de igual
        blocks_by_equals = re.split(r'\n={3,}\n', content)
        self.stats['blocks_by_equals'] = len([b for b in blocks_by_equals if b.strip()])
    
    def _detect_patterns(self, lines: list):
        """Detecta patrones comunes en el archivo."""
        
        # Patrón: ITEM: xxx
        pattern_item = re.compile(r'ITEM\s*:\s*([^\s;,]+)', re.IGNORECASE)
        
        # Patrón: UNIDAD: xxx
        pattern_unit = re.compile(r'UNIDAD\s*:\s*([^\s;,]+)', re.IGNORECASE)
        
        # Patrón: Línea con múltiples campos numéricos
        pattern_numeric_row = re.compile(r'[\d.,]+\s+[\d.,]+\s+[\d.,]+')
        
        for i, line in enumerate(lines[:100], 1):  # Primeras 100 líneas
            
            item_match = pattern_item.search(line)
            if item_match:
                self.patterns_found.append({
                    'type': 'ITEM_CODE',
                    'line_num': i,
                    'value': item_match.group(1),
                    'full_line': line.strip()[:150]
                })
            
            unit_match = pattern_unit.search(line)
            if unit_match:
                self.patterns_found.append({
                    'type': 'UNIT',
                    'line_num': i,
                    'value': unit_match.group(1),
                    'full_line': line.strip()[:150]
                })
            
            if pattern_numeric_row.search(line):
                self.stats['numeric_rows'] += 1
    
    def _print_report(self):
        """Imprime reporte de diagnóstico."""
        
        print("\n" + "=" * 80)
        print("📊 REPORTE DE DIAGNÓSTICO DEL ARCHIVO APU")
        print("=" * 80)
        
        print("\n📈 ESTADÍSTICAS GENERALES:")
        print(f"  Total de líneas: {self.stats['total_lines']:,}")
        print(f"  Líneas vacías: {self.stats.get('empty_lines', 0):,}")
        print(f"  Líneas con contenido: {self.stats.get('non_empty_lines', 0):,}")
        print(f"  Encoding detectado: {self.stats.get('encoding', 'desconocido')}")
        
        print("\n🔍 SEPARADORES DETECTADOS:")
        print(f"  Líneas con punto y coma (;): {self.stats.get('lines_with_semicolon', 0):,}")
        print(f"  Máximo de ';' por línea: {self.stats.get('max_semicolons', 0)}")
        print(f"  Líneas con tabulaciones: {self.stats.get('lines_with_tabs', 0):,}")
        print(f"  Líneas con espacios múltiples: {self.stats.get('lines_with_multiple_spaces', 0):,}")
        
        print("\n🏗️ ESTRUCTURA DEL ARCHIVO:")
        print(f"  Bloques (por doble salto): {self.stats.get('blocks_by_double_newline', 0)}")
        print(f"  Bloques (por guiones): {self.stats.get('blocks_by_dashes', 0)}")
        print(f"  Bloques (por signos igual): {self.stats.get('blocks_by_equals', 0)}")
        
        print("\n🔑 PALABRAS CLAVE ENCONTRADAS:")
        print(f"  Líneas con 'ITEM': {self.stats.get('lines_with_ITEM', 0):,}")
        print(f"  Líneas con 'UNIDAD': {self.stats.get('lines_with_UNIDAD', 0):,}")
        print(f"  Líneas con 'DESCRIPCION': {self.stats.get('lines_with_DESCRIPCION', 0):,}")
        print(f"  Filas numéricas: {self.stats.get('numeric_rows', 0):,}")
        
        print("\n📦 CATEGORÍAS DETECTADAS:")
        for category in ['MATERIALES', 'MANO DE OBRA', 'EQUIPO', 'OTROS']:
            count = self.stats.get(f'category_{category}', 0)
            if count > 0:
                print(f"  {category}: {count} veces")
        
        print("\n📝 MUESTRA DE PRIMERAS LÍNEAS:")
        for sample in self.sample_lines[:10]:
            print(f"  Línea {sample['line_num']:4d} ({sample['length']:3d} chars): {sample['content']}")
        
        print("\n🎯 PATRONES CLAVE DETECTADOS:")
        item_patterns = [p for p in self.patterns_found if p['type'] == 'ITEM_CODE']
        if item_patterns:
            print(f"\n  ✓ Códigos ITEM encontrados: {len(item_patterns)}")
            for pattern in item_patterns[:5]:
                print(f"    Línea {pattern['line_num']}: {pattern['value']}")
                print(f"      → {pattern['full_line']}")
        
        unit_patterns = [p for p in self.patterns_found if p['type'] == 'UNIT']
        if unit_patterns:
            print(f"\n  ✓ Unidades encontradas: {len(unit_patterns)}")
            for pattern in unit_patterns[:5]:
                print(f"    Línea {pattern['line_num']}: {pattern['value']}")
        
        print("\n" + "=" * 80)
        print("💡 RECOMENDACIONES:")
        
        # Análisis inteligente
        if self.stats.get('lines_with_semicolon', 0) > self.stats.get('non_empty_lines', 1) * 0.5:
            print("  → El archivo usa PUNTO Y COMA (;) como separador principal")
            print("  → Usar estrategia de parsing por columnas con split(';')")
        
        if self.stats.get('blocks_by_double_newline', 0) > 10:
            print("  → El archivo tiene bloques separados por líneas vacías")
            print("  → Usar estrategia de parsing por bloques")
        
        if self.stats.get('lines_with_ITEM', 0) > 0:
            print(f"  → Se detectaron {self.stats['lines_with_ITEM']} líneas con 'ITEM'")
            print("  → Usar 'ITEM:' como delimitador de inicio de APU")
        else:
            print("  ⚠️ NO se detectaron líneas con 'ITEM' - verificar formato")
        
        print("=" * 80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python diagnose_apus_file.py <ruta_archivo>")
        print("Ejemplo: python diagnose_apus_file.py data/apus.csv")
        sys.exit(1)
    
    diagnostic = APUFileDiagnostic(sys.argv[1])
    diagnostic.diagnose()