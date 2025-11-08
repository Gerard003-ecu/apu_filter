# app/report_parser_crudo.py

import hashlib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .utils import clean_apu_code

logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Excepción base para errores del parser."""
    pass


class FileReadError(ParserError):
    """Error al leer el archivo."""
    pass


class ParseStrategyError(ParserError):
    """Error en la estrategia de parsing."""
    pass


@dataclass
class ParserConfig:
    """
    Configuración adaptable del parser con validación mejorada.
    
    Attributes:
        encodings: Lista de encodings a intentar en orden de prioridad
        strategy: Estrategia de parsing ('blocks', 'lines', 'hybrid', 'auto')
        field_separator: Separador de campos (auto-detectado si es None)
        block_separator: Expresión regular para separar bloques
        min_apu_code_length: Longitud mínima del código APU
        min_description_length: Longitud mínima de descripción
        default_unit: Unidad por defecto cuando no se encuentra
        debug_mode: Activar modo debug con información adicional
        max_debug_samples: Máximo de muestras debug a recolectar
        max_lines_to_process: Límite de líneas para evitar memoria excesiva
        confidence_threshold: Umbral de confianza para detección automática
    """
    encodings: List[str] = field(default_factory=lambda: ['utf-8', 'latin1', 'cp1252', 'iso-8859-1'])
    strategy: str = 'auto'
    field_separator: Optional[str] = None
    block_separator: str = r'\n\s*\n'
    min_apu_code_length: int = 2
    min_description_length: int = 3
    default_unit: str = "UND"
    debug_mode: bool = False
    max_debug_samples: int = 10
    max_lines_to_process: int = 100000  # Límite de seguridad
    confidence_threshold: float = 0.7  # Umbral para decisiones automáticas

    def __post_init__(self):
        """Validación post-inicialización."""
        if not self.encodings:
            raise ValueError("Debe especificar al menos un encoding")

        valid_strategies = [s.value for s in ParsingStrategy]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Estrategia inválida: {self.strategy}. Usar: {valid_strategies}")

        if self.min_apu_code_length < 1:
            raise ValueError("min_apu_code_length debe ser >= 1")

        if self.min_description_length < 1:
            raise ValueError("min_description_length debe ser >= 1")

        if self.max_lines_to_process < 100:
            raise ValueError("max_lines_to_process debe ser >= 100")

        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold debe estar entre 0 y 1")


class ParsingStrategy(Enum):
    """Estrategias de parsing disponibles."""
    BLOCKS = "blocks"
    LINES = "lines"
    HYBRID = "hybrid"
    AUTO = "auto"


@dataclass
class APUContext:
    """Contexto de un APU con validación."""
    apu_code: str
    apu_desc: str
    apu_unit: str
    confidence: float = 1.0  # Nivel de confianza en la extracción
    source_line: int = 0  # Línea donde se encontró

    def __post_init__(self):
        """Validación y normalización."""
        self.apu_code = self.apu_code.strip() if self.apu_code else ""
        self.apu_desc = self.apu_desc.strip() if self.apu_desc else ""
        self.apu_unit = self.apu_unit.strip().upper() if self.apu_unit else "UND"

        if not self.apu_code:
            raise ValueError("El código APU no puede estar vacío")

    @property
    def is_valid(self) -> bool:
        """Verifica si el contexto es válido."""
        return bool(self.apu_code and len(self.apu_code) >= 2)

    def to_dict(self) -> Dict[str, str]:
        """Convierte a diccionario para compatibilidad."""
        return {
            'apu_code': self.apu_code,
            'apu_desc': self.apu_desc,
            'apu_unit': self.apu_unit
        }


class PatternMatcher:
    """Gestor centralizado de patrones regex con cache."""

    def __init__(self):
        self._patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Compila y cachea todos los patrones."""
        patterns_def = {
            'item_flexible': r'(?:ITEM|CÓDIGO|CODIGO|COD\.?)\s*[:\-#]?\s*([A-Z0-9][A-Z0-9\-._/]*)',
            'unit_flexible': r'(?:UNIDAD|UND\.?|U\.?)\s*[:\-]?\s*([A-Z]{1,10})',
            'header_full': r'(?P<desc>.+?)\s+(?:UNIDAD|UND\.?)\s*[:\-]?\s*(?P<unit>\S+)\s+(?:ITEM|CÓDIGO|COD\.?)\s*[:\-]?\s*(?P<item>\S+)',
            'header_item_first': r'(?:ITEM|CÓDIGO|COD\.?)\s*[:\-]?\s*(?P<item>\S+).*?(?:DESCRIPCION|DESCRIPCIÓN|DESC\.?)\s*[:\-]?\s*(?P<desc>.+?)(?:\s+(?:UNIDAD|UND\.?)\s*[:\-]?\s*(?P<unit>\S+))?',
            'numeric_row': r'[\d.,]+\s+[\d.,]+\s+[\d.,]+',
            'table_header': r'^\s*(?:CODIGO|CÓDIGO|COD\.?|DESCRIPCION|DESCRIPCIÓN|DESC\.?|UNIDAD|UND\.?|CANTIDAD|CANT\.?|V\.?\s*UNITARIO|V\.?\s*TOTAL|PRECIO)\s*',
            'currency': r'[$€¥£]\s*[\d.,]+|[\d.,]+\s*(?:USD|EUR|COP|MXN)',
            'percentage': r'\d+\.?\d*\s*%',
        }

        for name, pattern in patterns_def.items():
            self._patterns[name] = re.compile(pattern, re.IGNORECASE)

    @lru_cache(maxsize=256)
    def match(self, pattern_name: str, text: str) -> Optional[re.Match]:
        """Busca coincidencia con cache LRU."""
        if pattern_name not in self._patterns:
            raise ValueError(f"Patrón no definido: {pattern_name}")
        return self._patterns[pattern_name].search(text)

    def get_pattern(self, pattern_name: str) -> re.Pattern:
        """Obtiene un patrón compilado."""
        if pattern_name not in self._patterns:
            raise ValueError(f"Patrón no definido: {pattern_name}")
        return self._patterns[pattern_name]


class ReportParserCrudo:
    """
    Parser robusto y adaptable para archivos de APU con múltiples formatos.
    
    Características mejoradas:
    - Auto-detección inteligente de formato con niveles de confianza
    - Múltiples estrategias de parsing con fallback automático
    - Diagnóstico detallado y trazabilidad completa
    - Gestión de memoria optimizada para archivos grandes
    - Cache de patrones para mejor rendimiento
    - Manejo robusto de errores con recuperación
    """

    # Categorías con variaciones expandidas
    CATEGORY_KEYWORDS = {
        'MATERIALES': {'MATERIALES', 'MATERIAL', 'MAT.', 'INSUMOS'},
        'MANO DE OBRA': {'MANO DE OBRA', 'MANO OBRA', 'M.O.', 'MO', 'PERSONAL', 'OBRERO'},
        'EQUIPO': {'EQUIPO', 'EQUIPOS', 'MAQUINARIA', 'MAQ.'},
        'TRANSPORTE': {'TRANSPORTE', 'TRANSPORTES', 'TRANS.', 'ACARREO'},
        'HERRAMIENTA': {'HERRAMIENTA', 'HERRAMIENTAS', 'HERR.', 'UTILES'},
        'OTROS': {'OTROS', 'OTRO', 'VARIOS', 'ADICIONALES'},
    }

    # Líneas que deben ignorarse (compiladas para eficiencia)
    JUNK_PATTERNS = [
        re.compile(r'^\s*SUBTOTAL', re.IGNORECASE),
        re.compile(r'^\s*COSTO\s+DIRECTO', re.IGNORECASE),
        re.compile(r'^\s*TOTAL(?:\s+|$)', re.IGNORECASE),
        re.compile(r'^\s*IMPUESTO', re.IGNORECASE),
        re.compile(r'^\s*IVA\s*', re.IGNORECASE),
        re.compile(r'^\s*[-=_]{3,}\s*$'),
        re.compile(r'^\s*[•·→←↓↑]+\s*$'),  # Bullets y flechas
        re.compile(r'^\s*Página\s+\d+', re.IGNORECASE),
        re.compile(r'^\s*Page\s+\d+', re.IGNORECASE),
    ]

    # Posibles separadores comunes con pesos
    COMMON_SEPARATORS = [
        (';', 'punto y coma', 1.0),
        ('\t', 'tabulación', 0.95),
        ('|', 'pipe', 0.9),
        (r'\s{2,}', 'múltiples espacios', 0.7),
        (',', 'coma', 0.6),
    ]

    def __init__(self, file_path: Union[str, Path], config: Optional[ParserConfig] = None):
        """
        Inicializa el parser con validación mejorada.
        
        Args:
            file_path: Ruta al archivo a parsear
            config: Configuración opcional del parser
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si la ruta no es válida
        """
        self.file_path = Path(file_path)
        self.config = config or ParserConfig()
        self.pattern_matcher = PatternMatcher()

        # Validación exhaustiva
        self._validate_file_path()

        # Estado interno mejorado
        self.raw_records: List[Dict[str, Any]] = []
        self.stats: Counter = Counter()
        self.debug_samples: List[Dict] = []
        self.errors: List[Dict[str, Any]] = []
        self._detected_separator: Optional[str] = None
        self._separator_is_regex: bool = False
        self._content_hash: Optional[str] = None
        self._parsed: bool = False

        # Cache para optimización
        self._category_cache: Dict[str, Optional[str]] = {}
        self._junk_cache: Set[str] = set()

    def _validate_file_path(self) -> None:
        """Valida exhaustivamente la ruta del archivo."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"La ruta no corresponde a un archivo: {self.file_path}")

        if self.file_path.stat().st_size == 0:
            raise ValueError(f"El archivo está vacío: {self.file_path}")

        if self.file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
            logger.warning(f"⚠️ Archivo muy grande ({self.file_path.stat().st_size:,} bytes). "
                          "El procesamiento podría ser lento.")

    def parse_to_raw(self) -> List[Dict[str, Any]]:
        """
        Punto de entrada principal con manejo robusto de errores.
        
        Returns:
            Lista de registros crudos extraídos
            
        Raises:
            FileReadError: Si no se puede leer el archivo
            ParseStrategyError: Si ninguna estrategia funciona
        """
        if self._parsed:
            logger.info("ℹ️ Archivo ya parseado, devolviendo resultados en cache")
            return self.raw_records

        logger.info(f"🔍 Iniciando parsing de: {self.file_path.name}")
        self._log_file_info()

        try:
            # Leer contenido con gestión de memoria
            content = self._read_file_safely()
            if not content or not content.strip():
                raise FileReadError("Contenido vacío o no legible")

            # Generar hash para cache/comparación
            self._content_hash = self._generate_content_hash(content)
            self.stats['content_hash'] = self._content_hash[:8]

            # Validar límites
            lines = content.split('\n')
            if len(lines) > self.config.max_lines_to_process:
                logger.warning(f"⚠️ Archivo con {len(lines):,} líneas excede el límite "
                              f"({self.config.max_lines_to_process:,}). Truncando...")
                lines = lines[:self.config.max_lines_to_process]
                content = '\n'.join(lines)

            self.stats['total_lines'] = len(lines)

            # Auto-detectar y aplicar estrategia
            success = self._execute_parsing_strategy(content)

            if not success and self.config.strategy != 'hybrid':
                logger.warning("⚠️ Estrategia principal falló. Intentando estrategia de respaldo...")
                success = self._fallback_parsing(content)

            # Validar y limpiar resultados
            self._validate_and_clean_results()

            # Generar reportes
            self._log_statistics()
            if self.config.debug_mode:
                self._log_debug_samples()

            self._parsed = True

            if not success:
                logger.error("❌ No se pudieron extraer insumos válidos")
                if self.errors:
                    self._log_errors()

            return self.raw_records

        except Exception as e:
            logger.error(f"❌ Error crítico en parsing: {e}")
            self._add_error('critical', str(e))
            raise ParseStrategyError(f"Fallo en parsing: {e}") from e

    def _log_file_info(self) -> None:
        """Registra información detallada del archivo."""
        try:
            stat = self.file_path.stat()
            logger.info(f"📦 Tamaño: {stat.st_size:,} bytes")
            logger.info(f"📅 Modificado: {stat.st_mtime}")
            self.stats['file_size_bytes'] = stat.st_size
        except OSError as e:
            logger.error(f"❌ Error al obtener información del archivo: {e}")
            self.stats['file_size_bytes'] = -1

    def _read_file_safely(self) -> str:
        """
        Lee el archivo con manejo robusto de encodings y errores.
        
        Returns:
            Contenido del archivo
            
        Raises:
            FileReadError: Si no se puede leer con ningún encoding
        """
        errors_found = []

        for encoding in self.config.encodings:
            try:
                with open(self.file_path, 'r', encoding=encoding, errors='strict') as f:
                    content = f.read()

                logger.info(f"✅ Archivo leído exitosamente con encoding: {encoding}")
                self.stats['encoding_used'] = encoding
                return content

            except UnicodeDecodeError as e:
                errors_found.append(f"{encoding}: {e}")
                # Intentar con reemplazo de errores
                try:
                    with open(self.file_path, 'r', encoding=encoding, errors='replace') as f:
                        content = f.read()

                    logger.warning(f"⚠️ Archivo leído con encoding {encoding} (con reemplazos)")
                    self.stats['encoding_used'] = f"{encoding}_with_replacements"
                    self.stats['encoding_errors'] = True
                    return content

                except Exception:
                    continue

            except Exception as e:
                errors_found.append(f"{encoding}: {e}")
                continue

        error_msg = f"No se pudo leer el archivo con ningún encoding. Errores: {errors_found}"
        logger.error(f"❌ {error_msg}")
        raise FileReadError(error_msg)

    def _generate_content_hash(self, content: str) -> str:
        """Genera un hash del contenido para identificación única."""
        return hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest()

    def _execute_parsing_strategy(self, content: str) -> bool:
        """
        Ejecuta la estrategia de parsing configurada o auto-detectada.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            True si se extrajeron insumos, False en caso contrario
        """
        # Determinar estrategia
        strategy = self._determine_strategy(content)
        logger.info(f"📋 Estrategia seleccionada: {strategy.value.upper()}")

        # Ejecutar según estrategia
        if strategy == ParsingStrategy.BLOCKS:
            return self._parse_by_blocks(content)
        elif strategy == ParsingStrategy.LINES:
            return self._parse_by_lines(content)
        elif strategy == ParsingStrategy.HYBRID:
            # Intenta bloques primero, luego líneas
            success = self._parse_by_blocks(content)
            if not success or self.stats['insumos_extracted'] < 5:
                logger.info("🔄 Cambiando a estrategia de líneas...")
                # Limpiar resultados parciales si son muy pocos
                if self.stats['insumos_extracted'] < 5:
                    self.raw_records.clear()
                    self.stats['insumos_extracted'] = 0
                success = self._parse_by_lines(content) or success
            return success
        else:
            logger.error(f"Estrategia no implementada: {strategy}")
            return False

    def _determine_strategy(self, content: str) -> ParsingStrategy:
        """
        Determina la mejor estrategia basada en análisis heurístico del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Estrategia recomendada
        """
        if self.config.strategy != 'auto':
            try:
                return ParsingStrategy(self.config.strategy)
            except ValueError:
                logger.warning(f"⚠️ Estrategia inválida '{self.config.strategy}', usando AUTO")

        # Análisis heurístico
        confidence_scores = defaultdict(float)

        # Detectar separador si no está configurado
        if self.config.field_separator:
            self._detected_separator = self.config.field_separator
            self._separator_is_regex = bool(re.match(r'\\', self.config.field_separator))
        else:
            sep_confidence = self._auto_detect_separator(content)
            confidence_scores['separator'] = sep_confidence

        # Analizar estructura de bloques
        blocks = re.split(self.config.block_separator, content)
        valid_blocks = [b for b in blocks if b.strip() and len(b.strip()) > 50]

        if len(valid_blocks) >= 3:
            # Verificar si los bloques parecen APUs
            apu_blocks = 0
            for block in valid_blocks[:10]:  # Muestra
                if self._quick_apu_check(block):
                    apu_blocks += 1

            block_confidence = apu_blocks / min(10, len(valid_blocks))
            confidence_scores['blocks'] = block_confidence

            if block_confidence >= self.config.confidence_threshold:
                logger.info(f"📊 Confianza en bloques: {block_confidence:.2f}")
                return ParsingStrategy.BLOCKS

        # Analizar estructura de líneas
        lines = content.split('\n')[:500]  # Muestra
        apu_lines = sum(1 for line in lines if self._quick_apu_check(line))
        line_confidence = apu_lines / len(lines) if lines else 0
        confidence_scores['lines'] = line_confidence

        logger.info(f"📊 Scores de confianza: {dict(confidence_scores)}")

        # Decidir estrategia
        if confidence_scores['blocks'] > confidence_scores['lines']:
            return ParsingStrategy.BLOCKS
        elif confidence_scores['lines'] >= 0.1:
            return ParsingStrategy.LINES
        else:
            # Si no hay confianza clara, usar híbrido
            return ParsingStrategy.HYBRID

    def _quick_apu_check(self, text: str) -> bool:
        """
        Verifica rápidamente si un texto podría contener información de APU.
        
        Args:
            text: Texto a verificar
            
        Returns:
            True si parece contener APU, False en caso contrario
        """
        if len(text) < 10:
            return False

        # Cache para textos ya verificados
        text_hash = hash(text[:100])  # Solo primeros 100 chars para el hash

        # Verificar patrones de APU
        has_item = bool(self.pattern_matcher.match('item_flexible', text))
        has_numbers = bool(self.pattern_matcher.match('numeric_row', text))

        return has_item or (has_numbers and len(text) > 50)

    def _auto_detect_separator(self, content: str) -> float:
        """
        Detecta automáticamente el separador de campos con nivel de confianza.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Nivel de confianza en la detección (0-1)
        """
        lines = [l for l in content.split('\n')[:500] if l.strip()]  # Muestra mejorada
        if not lines:
            return 0.0

        scores = {}
        max_score = 0

        for sep, desc, weight in self.COMMON_SEPARATORS:
            count = 0
            consistency = []

            for line in lines:
                if len(line.strip()) < 10:
                    continue

                # Dividir línea
                if sep.startswith('\\'):  # Es regex
                    parts = re.split(sep, line)
                else:
                    parts = line.split(sep)

                parts = [p.strip() for p in parts if p.strip()]

                if len(parts) >= 3:
                    count += 1
                    consistency.append(len(parts))

            if count > 0:
                # Calcular score con consistencia
                avg_parts = sum(consistency) / len(consistency) if consistency else 0
                std_dev = self._calculate_std_dev(consistency)
                consistency_factor = 1.0 if std_dev < 2 else 0.5

                score = (count / len(lines)) * weight * consistency_factor
                scores[sep] = score
                max_score = max(max_score, score)

        if scores:
            best_sep = max(scores, key=scores.get)
            confidence = scores[best_sep] / max_score if max_score > 0 else 0

            self._detected_separator = best_sep
            self._separator_is_regex = best_sep.startswith('\\')
            self.stats['detected_separator'] = best_sep
            self.stats['separator_confidence'] = round(confidence, 2)

            logger.info(f"📊 Separador detectado: '{best_sep}' (confianza: {confidence:.2f})")
            return confidence

        logger.warning("⚠️ No se pudo detectar un separador claro")
        return 0.0

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calcula la desviación estándar de una lista de valores."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _parse_by_blocks(self, content: str) -> bool:
        """
        Estrategia mejorada de parsing por bloques con recuperación de errores.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            True si se extrajeron insumos, False en caso contrario
        """
        try:
            blocks = re.split(self.config.block_separator, content)
            valid_blocks = [b.strip() for b in blocks if b.strip() and len(b.strip()) > 20]

            if len(valid_blocks) < 2:
                logger.info("⚠️ Insuficientes bloques válidos para esta estrategia")
                return False

            logger.info(f"  Procesando {len(valid_blocks)} bloques...")
            self.stats['total_blocks'] = len(blocks)
            self.stats['valid_blocks'] = len(valid_blocks)

            for block_num, block in enumerate(valid_blocks, 1):
                try:
                    self._process_block_safe(block, block_num)
                except Exception as e:
                    logger.debug(f"Error procesando bloque {block_num}: {e}")
                    self._add_error('block_processing', f"Bloque {block_num}: {e}")
                    continue

            return self.stats['insumos_extracted'] > 0

        except Exception as e:
            logger.error(f"❌ Error en estrategia de bloques: {e}")
            self._add_error('blocks_strategy', str(e))
            return False

    def _process_block_safe(self, block: str, block_num: int) -> None:
        """
        Procesa un bloque con manejo robusto de errores.
        
        Args:
            block: Contenido del bloque
            block_num: Número del bloque para trazabilidad
        """
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            self.stats['empty_blocks'] += 1
            return

        # Extraer contexto APU con validación
        try:
            apu_context = self._extract_apu_context_enhanced(lines, block_num)
        except Exception as e:
            logger.debug(f"No se pudo extraer APU del bloque {block_num}: {e}")
            apu_context = None

        if not apu_context or not apu_context.is_valid:
            self.stats['blocks_without_valid_apu'] += 1
            if self.config.debug_mode:
                self._add_debug_sample('block_no_apu', block_num, lines[:3])
            return

        self.stats['valid_apu_blocks'] += 1
        logger.debug(f"Bloque #{block_num}: APU={apu_context.apu_code} (confianza: {apu_context.confidence:.2f})")

        # Procesar líneas del bloque
        current_category = "INDEFINIDO"
        items_in_block = 0

        for line_num, line in enumerate(lines, 1):
            # Detectar cambio de categoría
            new_category = self._detect_category_cached(line.upper())
            if new_category:
                current_category = new_category
                self.stats[f'category_{current_category}'] += 1
                continue

            # Filtrar líneas no válidas
            if self._is_junk_line_cached(line):
                self.stats['junk_lines'] += 1
                continue

            # Procesar insumo
            if self._is_valid_insumo_enhanced(line):
                record = self._create_record(apu_context, current_category, line, block_num, line_num)
                self.raw_records.append(record)
                self.stats['insumos_extracted'] += 1
                items_in_block += 1

                if self.config.debug_mode and len(self.debug_samples) < self.config.max_debug_samples:
                    self._add_debug_sample('insumo_extracted', block_num, [line],
                                          {'category': current_category, 'line_num': line_num})

        if items_in_block > 0:
            self.stats['blocks_with_items'] += 1

    def _parse_by_lines(self, content: str) -> bool:
        """
        Estrategia mejorada de parsing línea por línea.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            True si se extrajeron insumos, False en caso contrario
        """
        try:
            lines = content.split('\n')
            logger.info(f"  Procesando {len(lines)} líneas...")

            current_apu: Optional[APUContext] = None
            current_category = "INDEFINIDO"
            consecutive_empty = 0
            max_consecutive_empty = 5  # Reset APU después de 5 líneas vacías

            for line_num, line in enumerate(lines, 1):
                line_clean = line.strip()

                # Contar líneas vacías consecutivas
                if not line_clean:
                    consecutive_empty += 1
                    if consecutive_empty > max_consecutive_empty and current_apu:
                        logger.debug(f"Reset APU después de {consecutive_empty} líneas vacías")
                        current_apu = None
                        current_category = "INDEFINIDO"
                    continue

                consecutive_empty = 0

                # Detectar nuevo APU
                try:
                    new_apu = self._extract_apu_context_enhanced([line_clean], line_num)
                    if new_apu and new_apu.is_valid:
                        current_apu = new_apu
                        current_category = "INDEFINIDO"
                        self.stats['apus_detected'] += 1
                        logger.debug(f"Línea {line_num}: Nuevo APU={current_apu.apu_code}")
                        continue
                except Exception as e:
                    logger.debug(f"Error extrayendo APU en línea {line_num}: {e}")

                # Sin APU activo, buscar uno
                if not current_apu:
                    self.stats['lines_without_apu'] += 1
                    continue

                # Cambiar categoría
                new_category = self._detect_category_cached(line_clean.upper())
                if new_category:
                    current_category = new_category
                    self.stats[f'category_{current_category}'] += 1
                    continue

                # Procesar posible insumo
                if not self._is_junk_line_cached(line_clean) and self._is_valid_insumo_enhanced(line_clean):
                    record = self._create_record(current_apu, current_category, line_clean, 0, line_num)
                    self.raw_records.append(record)
                    self.stats['insumos_extracted'] += 1

            return self.stats['insumos_extracted'] > 0

        except Exception as e:
            logger.error(f"❌ Error en estrategia de líneas: {e}")
            self._add_error('lines_strategy', str(e))
            return False

    def _fallback_parsing(self, content: str) -> bool:
        """
        Estrategia de respaldo cuando las principales fallan.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            True si se extrajeron insumos, False en caso contrario
        """
        logger.info("🔄 Aplicando estrategia de respaldo...")

        # Intentar con separador diferente
        original_sep = self._detected_separator
        for sep, desc, weight in self.COMMON_SEPARATORS:
            if sep != original_sep:
                logger.info(f"  Probando con separador: {desc}")
                self._detected_separator = sep
                self._separator_is_regex = sep.startswith('\\')

                # Limpiar resultados previos
                self.raw_records.clear()
                old_count = self.stats.get('insumos_extracted', 0)
                self.stats['insumos_extracted'] = 0

                # Intentar parsing
                if self._parse_by_lines(content):
                    logger.info(f"✅ Respaldo exitoso con separador: {desc}")
                    return True

                # Si no funcionó, restaurar
                self.stats['insumos_extracted'] = old_count

        self._detected_separator = original_sep
        return False

    def _extract_apu_context_enhanced(self, lines: List[str], source_line: int = 0) -> Optional[APUContext]:
        """
        Extrae contexto APU con múltiples estrategias y nivel de confianza.
        
        Args:
            lines: Líneas donde buscar el APU
            source_line: Número de línea fuente para trazabilidad
            
        Returns:
            Contexto APU si se encuentra, None en caso contrario
        """
        if not lines:
            return None

        # Estrategia 1: Patrones completos (alta confianza)
        for i, line in enumerate(lines[:5]):
            # Patrón header completo
            match = self.pattern_matcher.match('header_full', line)
            if match:
                try:
                    return APUContext(
                        apu_code=clean_apu_code(match.group('item')),
                        apu_desc=match.group('desc').strip(),
                        apu_unit=match.group('unit').strip(),
                        confidence=0.95,
                        source_line=source_line + i
                    )
                except (ValueError, AttributeError):
                    continue

            # Patrón con item primero
            match = self.pattern_matcher.match('header_item_first', line)
            if match:
                try:
                    return APUContext(
                        apu_code=clean_apu_code(match.group('item')),
                        apu_desc=match.group('desc') or f"APU {match.group('item')}",
                        apu_unit=match.group('unit') or self.config.default_unit,
                        confidence=0.90,
                        source_line=source_line + i
                    )
                except (ValueError, AttributeError):
                    continue

        # Estrategia 2: Búsqueda por campos separados (confianza media)
        item_code = unit = desc = None
        item_line = unit_line = desc_line = 0

        for i, line in enumerate(lines[:7]):
            # Buscar código
            if not item_code:
                match = self.pattern_matcher.match('item_flexible', line)
                if match:
                    candidate = clean_apu_code(match.group(1))
                    if len(candidate) >= self.config.min_apu_code_length:
                        item_code = candidate
                        item_line = source_line + i

            # Buscar unidad
            if not unit:
                match = self.pattern_matcher.match('unit_flexible', line)
                if match:
                    unit = match.group(1).strip()
                    unit_line = source_line + i

            # Buscar descripción
            if not desc and len(line) >= self.config.min_description_length:
                # Evitar líneas que son solo números o categorías
                if not self._is_junk_line_cached(line) and not self._detect_category_cached(line.upper()):
                    # Verificar que no sea una línea de datos
                    if not self.pattern_matcher.match('numeric_row', line):
                        desc = line.strip()[:200]  # Limitar longitud
                        desc_line = source_line + i

        # Construir contexto si tenemos al menos el código
        if item_code:
            confidence = 0.7
            if desc:
                confidence += 0.15
            if unit:
                confidence += 0.15

            try:
                return APUContext(
                    apu_code=item_code,
                    apu_desc=desc or f"APU {item_code}",
                    apu_unit=unit or self.config.default_unit,
                    confidence=min(confidence, 1.0),
                    source_line=item_line
                )
            except ValueError:
                pass

        return None

    def _detect_category_cached(self, line_upper: str) -> Optional[str]:
        """
        Detecta categoría con cache para mejorar rendimiento.
        
        Args:
            line_upper: Línea en mayúsculas
            
        Returns:
            Categoría detectada o None
        """
        # Verificar cache
        if line_upper in self._category_cache:
            return self._category_cache[line_upper]

        result = self._detect_category(line_upper)

        # Guardar en cache (limitar tamaño)
        if len(self._category_cache) < 1000:
            self._category_cache[line_upper] = result

        return result

    def _detect_category(self, line_upper: str) -> Optional[str]:
        """
        Detecta si una línea representa una categoría.
        
        Args:
            line_upper: Línea en mayúsculas
            
        Returns:
            Categoría detectada o None
        """
        # Validaciones rápidas
        if len(line_upper) > 100 or len(line_upper) < 2:
            return None

        # Evitar líneas con muchos números (probablemente datos)
        if sum(c.isdigit() for c in line_upper) > len(line_upper) * 0.3:
            return None

        # Evitar líneas con separadores (probablemente insumo)
        if self._detected_separator and not self._separator_is_regex:
            if self._detected_separator in line_upper and line_upper.count(self._detected_separator) > 1:
                return None

        # Buscar categorías conocidas
        for canonical, variations in self.CATEGORY_KEYWORDS.items():
            for variation in variations:
                # Buscar coincidencia exacta o al inicio/fin de línea
                pattern = rf'(^|\s){re.escape(variation)}(\s|$|:)'
                if re.search(pattern, line_upper):
                    return canonical

        return None

    def _is_junk_line_cached(self, line: str) -> bool:
        """
        Verifica si es línea basura con cache.
        
        Args:
            line: Línea a verificar
            
        Returns:
            True si es línea basura, False en caso contrario
        """
        # Cache para líneas cortas
        if len(line) < 100:
            if line in self._junk_cache:
                return True

        result = self._is_junk_line(line)

        if result and len(line) < 100 and len(self._junk_cache) < 1000:
            self._junk_cache.add(line)

        return result

    def _is_junk_line(self, line: str) -> bool:
        """
        Determina si una línea debe ignorarse.
        
        Args:
            line: Línea a verificar
            
        Returns:
            True si debe ignorarse, False en caso contrario
        """
        line_clean = line.strip()

        # Líneas muy cortas
        if len(line_clean) < 3:
            return True

        # Verificar contra patrones de basura
        for pattern in self.JUNK_PATTERNS:
            if pattern.search(line_clean):
                return True

        # Verificar si es header de tabla
        if self.pattern_matcher.match('table_header', line_clean):
            return True

        # Líneas que son solo símbolos repetidos
        if re.match(r'^[^\w\s]{3,}$', line_clean):
            return True

        return False

    def _is_valid_insumo_enhanced(self, line: str) -> bool:
        """
        Validación mejorada de insumos con múltiples criterios.
        
        Args:
            line: Línea a validar
            
        Returns:
            True si es un insumo válido, False en caso contrario
        """
        if not line or len(line.strip()) < 5:
            return False

        line_clean = line.strip()

        # Rechazar si es muy largo (probablemente párrafo)
        if len(line_clean) > 500:
            return False

        # Verificar si tiene estructura de datos tabulares
        # Criterio 1: Tiene números en formato de tabla
        if self.pattern_matcher.match('numeric_row', line_clean):
            return True

        # Criterio 2: Tiene estructura de moneda
        if self.pattern_matcher.match('currency', line_clean):
            parts = self._split_line(line_clean)
            if len(parts) >= 2:
                return True

        # Criterio 3: Usar separador detectado
        parts = self._split_line(line_clean)

        # Debe tener al menos 2 partes significativas
        if len(parts) < 2:
            return False

        # Al menos 2 partes con contenido significativo
        significant = [p for p in parts if len(p.strip()) > 1 and not p.strip().isspace()]

        # Verificar que no sean todas numéricas o todas texto
        has_number = any(any(c.isdigit() for c in p) for p in significant)
        has_text = any(any(c.isalpha() for c in p) for p in significant)

        return len(significant) >= 2 and (has_number or len(significant) >= 3)

    def _split_line(self, line: str) -> List[str]:
        """
        Divide una línea usando el separador detectado o configurado.
        
        Args:
            line: Línea a dividir
            
        Returns:
            Lista de partes
        """
        sep = self._detected_separator or ';'

        if self._separator_is_regex or sep.startswith('\\'):
            parts = re.split(sep, line)
        else:
            parts = line.split(sep)

        return [p.strip() for p in parts if p.strip()]

    def _create_record(self, apu_context: APUContext, category: str, line: str,
                      block_num: int, line_num: int) -> Dict[str, Any]:
        """
        Crea un registro completo con toda la información necesaria.
        
        Args:
            apu_context: Contexto del APU
            category: Categoría del insumo
            line: Línea del insumo
            block_num: Número de bloque (0 si no aplica)
            line_num: Número de línea
            
        Returns:
            Diccionario con el registro completo
        """
        return {
            'apu_code': apu_context.apu_code,
            'apu_desc': apu_context.apu_desc,
            'apu_unit': apu_context.apu_unit,
            'category': category,
            'insumo_line': line,
            'confidence': apu_context.confidence,
            'source_block': block_num if block_num > 0 else None,
            'source_line': line_num,
            'separator_used': self._detected_separator,
        }

    def _validate_and_clean_results(self) -> None:
        """Valida y limpia los resultados finales."""
        if not self.raw_records:
            return

        initial_count = len(self.raw_records)

        # Eliminar duplicados exactos
        seen = set()
        unique_records = []

        for record in self.raw_records:
            # Crear clave única
            key = (record['apu_code'], record['category'], record['insumo_line'])
            if key not in seen:
                seen.add(key)
                unique_records.append(record)

        self.raw_records = unique_records
        duplicates_removed = initial_count - len(self.raw_records)

        if duplicates_removed > 0:
            logger.info(f"  Eliminados {duplicates_removed} registros duplicados")
            self.stats['duplicates_removed'] = duplicates_removed

        # Validar consistencia de APUs
        apu_stats = defaultdict(lambda: {'count': 0, 'categories': set()})
        for record in self.raw_records:
            apu_code = record['apu_code']
            apu_stats[apu_code]['count'] += 1
            apu_stats[apu_code]['categories'].add(record['category'])

        self.stats['unique_apus'] = len(apu_stats)
        self.stats['avg_items_per_apu'] = sum(s['count'] for s in apu_stats.values()) / len(apu_stats) if apu_stats else 0

    def _add_error(self, error_type: str, message: str) -> None:
        """Registra un error para análisis posterior."""
        self.errors.append({
            'type': error_type,
            'message': message,
            'timestamp': logger.name
        })

    def _add_debug_sample(self, sample_type: str, block_num: int,
                         lines: List[str], extra: Optional[Dict] = None) -> None:
        """Agrega una muestra de debug si está habilitado."""
        if not self.config.debug_mode:
            return

        if len(self.debug_samples) >= self.config.max_debug_samples:
            return

        sample = {
            'type': sample_type,
            'block_num': block_num,
            'lines': lines[:3],  # Máximo 3 líneas
        }

        if extra:
            sample.update(extra)

        self.debug_samples.append(sample)

    def _log_statistics(self) -> None:
        """Registra estadísticas detalladas del parsing."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 RESUMEN DE PARSING")
        logger.info("=" * 80)

        # Información del archivo
        logger.info(f"📁 Archivo: {self.file_path.name}")
        logger.info(f"🔤 Encoding: {self.stats.get('encoding_used', 'desconocido')}")

        if self.stats.get('encoding_errors'):
            logger.warning("  ⚠️ Se encontraron errores de encoding (caracteres reemplazados)")

        logger.info(f"📏 Separador: {self.stats.get('detected_separator', 'no detectado')}")

        if 'separator_confidence' in self.stats:
            logger.info(f"  Confianza: {self.stats['separator_confidence']}")

        # Estadísticas de contenido
        logger.info("\n📈 ESTADÍSTICAS DE CONTENIDO:")
        logger.info(f"  Total líneas: {self.stats.get('total_lines', 0):,}")
        logger.info(f"  Líneas vacías: {self.stats.get('empty_lines', 0):,}")

        # Estadísticas por estrategia
        if 'total_blocks' in self.stats:
            logger.info("\n📦 ESTADÍSTICAS DE BLOQUES:")
            logger.info(f"  Total bloques: {self.stats['total_blocks']}")
            logger.info(f"  Bloques válidos: {self.stats.get('valid_blocks', 0)}")
            logger.info(f"  Bloques con APU: {self.stats.get('valid_apu_blocks', 0)}")
            logger.info(f"  Bloques con items: {self.stats.get('blocks_with_items', 0)}")

        if 'apus_detected' in self.stats:
            logger.info(f"\n📝 APUs detectados (modo línea): {self.stats['apus_detected']}")

        # Resultados principales
        logger.info("\n✅ RESULTADOS:")
        extracted = self.stats.get('insumos_extracted', 0)
        logger.info(f"  Insumos extraídos: {extracted:,}")

        if 'unique_apus' in self.stats:
            logger.info(f"  APUs únicos: {self.stats['unique_apus']}")
            logger.info(f"  Promedio items/APU: {self.stats.get('avg_items_per_apu', 0):.1f}")

        if 'duplicates_removed' in self.stats:
            logger.info(f"  Duplicados eliminados: {self.stats['duplicates_removed']}")

        # Categorías
        categories = [k for k in self.stats.keys() if k.startswith('category_')]
        if categories:
            logger.info("\n📂 CATEGORÍAS DETECTADAS:")
            total_categorized = sum(self.stats[cat] for cat in categories)
            for cat in sorted(categories):
                name = cat.replace('category_', '')
                count = self.stats[cat]
                percentage = (count / total_categorized * 100) if total_categorized > 0 else 0
                logger.info(f"  {name}: {count} ({percentage:.1f}%)")

        # Información de calidad
        logger.info("\n🔍 CALIDAD DE DATOS:")
        logger.info(f"  Líneas de ruido ignoradas: {self.stats.get('junk_lines', 0):,}")

        if 'lines_without_apu' in self.stats:
            logger.info(f"  Líneas sin APU: {self.stats.get('lines_without_apu', 0):,}")

        # Hash del contenido
        if 'content_hash' in self.stats:
            logger.info(f"\n🔐 Hash del contenido: {self.stats['content_hash']}")

        logger.info("=" * 80)

        # Alertas y recomendaciones
        if extracted == 0:
            logger.error("\n❌ ¡ADVERTENCIA! No se extrajeron insumos.")
            logger.error("📋 Posibles causas:")
            logger.error("  • Formato de archivo no reconocido")
            logger.error("  • Separador de campos incorrecto")
            logger.error("  • Estructura de APU no estándar")
            logger.error("\n💡 Recomendaciones:")
            logger.error("  • Verificar el formato del archivo")
            logger.error("  • Especificar manualmente el separador en la configuración")
            logger.error("  • Usar modo debug para más detalles")
        elif extracted < 10:
            logger.warning("\n⚠️ Se extrajeron muy pocos insumos. Verificar el formato del archivo.")

    def _log_debug_samples(self) -> None:
        """Muestra muestras detalladas de debug."""
        if not self.debug_samples:
            return

        logger.info("\n" + "=" * 80)
        logger.info("🐛 MUESTRAS DE DEBUG")
        logger.info("=" * 80)

        # Agrupar por tipo
        samples_by_type = defaultdict(list)
        for sample in self.debug_samples:
            samples_by_type[sample['type']].append(sample)

        for sample_type, samples in samples_by_type.items():
            logger.info(f"\n📌 Tipo: {sample_type.upper().replace('_', ' ')}")
            logger.info(f"  Total: {len(samples)} muestras")

            for i, sample in enumerate(samples[:3], 1):  # Máximo 3 por tipo
                logger.info(f"\n  Muestra {i}:")
                for key, value in sample.items():
                    if key != 'type':
                        if isinstance(value, list):
                            logger.info(f"    {key}:")
                            for item in value[:3]:  # Máximo 3 items
                                logger.info(f"      - {item[:100]}")  # Truncar a 100 chars
                        else:
                            logger.info(f"    {key}: {value}")

        logger.info("=" * 80)

    def _log_errors(self) -> None:
        """Registra los errores encontrados durante el parsing."""
        if not self.errors:
            return

        logger.error("\n" + "=" * 80)
        logger.error("❌ ERRORES ENCONTRADOS")
        logger.error("=" * 80)

        # Agrupar por tipo
        errors_by_type = defaultdict(list)
        for error in self.errors:
            errors_by_type[error['type']].append(error['message'])

        for error_type, messages in errors_by_type.items():
            logger.error(f"\n{error_type.upper().replace('_', ' ')}:")
            unique_messages = list(set(messages))[:5]  # Máximo 5 únicos
            for msg in unique_messages:
                logger.error(f"  • {msg[:200]}")  # Truncar mensajes largos

            if len(messages) > 5:
                logger.error(f"  ... y {len(messages) - 5} más")

        logger.error("=" * 80)

    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de estadísticas para uso externo.
        
        Returns:
            Diccionario con estadísticas clave
        """
        return {
            'file_path': str(self.file_path),
            'file_size': self.stats.get('file_size_bytes', 0),
            'encoding': self.stats.get('encoding_used', 'unknown'),
            'separator': self.stats.get('detected_separator', 'none'),
            'total_lines': self.stats.get('total_lines', 0),
            'items_extracted': self.stats.get('insumos_extracted', 0),
            'unique_apus': self.stats.get('unique_apus', 0),
            'errors_count': len(self.errors),
            'parsed_successfully': self._parsed and self.stats.get('insumos_extracted', 0) > 0
        }
