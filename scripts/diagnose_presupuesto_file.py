# scripts/diagnose_presupuesto_file.py
"""
Herramienta de diagnóstico avanzada para analizar archivos de presupuesto.

Este módulo proporciona capacidades robustas de análisis de estructura,
detección automática de encoding, identificación de separadores y
generación de reportes detallados con recomendaciones.
"""
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logging.warning("chardet no disponible. La detección automática de encoding será limitada.")

# Configuración robusta del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("PresupuestoDiagnostic")


class DiagnosticError(Exception):
    """Excepción base para errores de diagnóstico."""
    pass


class FileReadError(DiagnosticError):
    """Error al leer el archivo."""
    pass


class EncodingDetectionError(DiagnosticError):
    """Error al detectar el encoding del archivo."""
    pass


class ConfidenceLevel(Enum):
    """Niveles de confianza para las detecciones."""
    HIGH = "alta"
    MEDIUM = "media"
    LOW = "baja"
    NONE = "ninguna"


@dataclass
class HeaderCandidate:
    """Representa un candidato a encabezado detectado."""
    line_num: int
    content: str
    matches: List[str]
    match_count: int
    column_count: int
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    def __post_init__(self):
        """Valida la consistencia de los datos."""
        if self.line_num < 1:
            raise ValueError("line_num debe ser >= 1")
        if self.match_count != len(self.matches):
            raise ValueError("match_count debe coincidir con len(matches)")
        if self.column_count < 1:
            raise ValueError("column_count debe ser >= 1")


@dataclass
class SampleLine:
    """Representa una línea de muestra del archivo."""
    line_num: int
    content: str
    column_count: int
    
    def __post_init__(self):
        """Valida los datos de la línea."""
        if self.line_num < 1:
            raise ValueError("line_num debe ser >= 1")
        if self.column_count < 0:
            raise ValueError("column_count debe ser >= 0")


@dataclass
class ColumnStatistics:
    """Estadísticas sobre columnas con un número específico."""
    count: int = 0
    samples: List[str] = field(default_factory=list)
    percentage: float = 0.0
    
    def add_sample(self, sample: str, max_samples: int = 3) -> None:
        """Agrega una muestra si no se ha alcanzado el límite."""
        if len(self.samples) < max_samples:
            self.samples.append(sample)
        self.count += 1


class PresupuestoFileDiagnostic:
    """
    Herramienta de diagnóstico para analizar la estructura de un archivo de Presupuesto.
    
    Esta clase proporciona métodos para:
    - Detectar automáticamente el encoding del archivo con múltiples estrategias
    - Identificar el separador de columnas mediante análisis estadístico robusto
    - Localizar la fila de encabezado con coincidencias ponderadas de palabras clave
    - Analizar la distribución y consistencia de columnas en los datos
    - Generar reportes detallados con recomendaciones accionables
    
    Attributes:
        file_path (Path): Ruta absoluta al archivo a diagnosticar
        ENCODINGS_TO_TRY (List[str]): Encodings a probar en orden de preferencia
        HEADER_KEYWORDS (List[str]): Palabras clave para identificar encabezados
        MIN_HEADER_KEYWORD_MATCHES (int): Coincidencias mínimas para validar encabezado
        MAX_SAMPLE_LINES (int): Límite de líneas de muestra a almacenar
        MAX_REPORT_SAMPLE_LINES (int): Límite de líneas a mostrar en reporte
        MAX_LINES_TO_ANALYZE (int): Límite de líneas para archivos grandes
    """
    
    # Configuración de encodings a probar
    ENCODINGS_TO_TRY = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16']
    
    # Palabras clave para detección de encabezados (normalizadas)
    HEADER_KEYWORDS = [
        'ITEM', 'DESCRIPCION', 'CANT', 'CANTIDAD', 'UNIDAD', 'UND',
        'VR UNITARIO', 'VALOR UNITARIO', 'PRECIO', 'TOTAL', 'IMPORTE', 
        'PU', 'P U', 'SUBTOTAL', 'PARCIAL', 'COSTO'
    ]
    
    # Configuración de límites y umbrales
    MIN_HEADER_KEYWORD_MATCHES = 2
    MAX_SAMPLE_LINES = 20
    MAX_REPORT_SAMPLE_LINES = 15
    MAX_LINES_TO_ANALYZE = 1000
    CHARDET_SAMPLE_SIZE = 50000  # Bytes a leer para detección de encoding
    CHARDET_MIN_CONFIDENCE = 0.7  # Confianza mínima para aceptar detección automática
    SEPARATOR_DETECTION_SAMPLE_LINES = 100  # Líneas para analizar separador
    COLUMN_CONSISTENCY_THRESHOLD = 0.85  # Umbral para considerar columnas consistentes
    MAX_SAMPLES_PER_COLUMN_COUNT = 3  # Muestras a guardar por cada conteo de columnas

    def __init__(self, file_path: Union[str, Path]):
        """
        Inicializa el diagnosticador con la ruta del archivo.
        
        Args:
            file_path (Union[str, Path]): Ruta al archivo de presupuesto a analizar
            
        Raises:
            ValueError: Si la ruta es inválida o el archivo no existe
            PermissionError: Si no hay permisos de lectura
        """
        self.file_path = Path(file_path).resolve()
        
        # Validaciones exhaustivas del archivo
        if not self.file_path.exists():
            raise ValueError(f"El archivo no existe: {self.file_path}")
        
        if not self.file_path.is_file():
            raise ValueError(f"La ruta no apunta a un archivo: {self.file_path}")
        
        if self.file_path.stat().st_size == 0:
            raise ValueError(f"El archivo está vacío: {self.file_path}")
        
        # Verificar permisos de lectura
        if not self._check_read_permissions():
            raise PermissionError(f"No hay permisos de lectura para: {self.file_path}")
        
        self._reset_state()
        logger.info(f"Inicializado diagnosticador para: {self.file_path}")

    def _check_read_permissions(self) -> bool:
        """
        Verifica que el archivo tenga permisos de lectura.
        
        Returns:
            bool: True si tiene permisos de lectura, False en caso contrario
        """
        try:
            with self.file_path.open('r', encoding='utf-8', errors='ignore') as f:
                f.read(1)
            return True
        except PermissionError:
            return False
        except Exception as e:
            logger.warning(f"Error al verificar permisos: {e}")
            return True  # Permitir continuar en caso de error inesperado

    def _reset_state(self) -> None:
        """Reinicia el estado interno para un nuevo diagnóstico."""
        self.stats: Counter = Counter()
        self.sample_lines: List[SampleLine] = []
        self.header_candidate: Optional[HeaderCandidate] = None
        self.column_analysis: Dict[int, ColumnStatistics] = defaultdict(ColumnStatistics)
        self.data_start_line: Optional[int] = None
        self._encoding: Optional[str] = None
        self._separator: Optional[str] = None
        self._content_cache: Optional[str] = None

    def diagnose(self) -> Dict[str, Any]:
        """
        Ejecuta el diagnóstico completo del archivo.
        
        Returns:
            Dict[str, Any]: Diccionario con estadísticas y hallazgos del diagnóstico
            
        Raises:
            FileReadError: Si no se puede leer el archivo
            DiagnosticError: Para errores durante el diagnóstico
        """
        try:
            self._reset_state()
            logger.info(f"🔍 Iniciando diagnóstico del archivo: {self.file_path}")
            
            # Obtener información básica del archivo
            file_size = self.file_path.stat().st_size
            self.stats['file_size_bytes'] = file_size
            self.stats['file_size_human'] = self._human_readable_size(file_size)
            
            # Leer contenido con detección robusta de encoding
            content = self._read_with_fallback_encoding()
            if not content:
                raise FileReadError(
                    "No se pudo leer el contenido del archivo con ningún encoding soportado"
                )
            
            # Validar que el contenido no esté vacío después de leerlo
            if not content.strip():
                raise FileReadError("El archivo no contiene datos válidos (solo espacios/saltos)")
            
            # Procesar líneas
            lines = content.splitlines()
            total_lines = len(lines)
            self.stats['total_lines'] = total_lines
            logger.info(f"Archivo contiene {total_lines} líneas.")
            
            # Validar cantidad mínima de líneas
            if total_lines < 2:
                logger.warning("⚠️ El archivo tiene muy pocas líneas para un análisis significativo.")
            
            # Limitar análisis para archivos muy grandes
            lines_to_analyze = self._get_lines_to_analyze(lines)
            
            # Ejecutar análisis
            self._detect_separator_from_lines(lines_to_analyze)
            self._analyze_structure_single_pass(lines_to_analyze)
            self._calculate_column_statistics()
            self._determine_confidence_levels()
            self._generate_diagnostic_report()
            
            # Preparar resultado
            result = self._build_result_dict()
            logger.info("✅ Diagnóstico completado exitosamente.")
            return result
            
        except FileReadError:
            raise
        except Exception as e:
            logger.exception(f"❌ Error inesperado durante el diagnóstico: {str(e)}")
            raise DiagnosticError(f"Fallo en el diagnóstico: {str(e)}") from e

    def _get_lines_to_analyze(self, lines: List[str]) -> List[str]:
        """
        Determina qué líneas analizar según el tamaño del archivo.
        
        Args:
            lines (List[str]): Todas las líneas del archivo
            
        Returns:
            List[str]: Líneas a analizar (limitadas si es necesario)
        """
        total_lines = len(lines)
        
        if total_lines > self.MAX_LINES_TO_ANALYZE:
            logger.warning(
                f"⚠️ Archivo grande ({total_lines} líneas). "
                f"Analizando las primeras {self.MAX_LINES_TO_ANALYZE} líneas."
            )
            self.stats['truncated_analysis'] = True
            self.stats['lines_analyzed'] = self.MAX_LINES_TO_ANALYZE
            return lines[:self.MAX_LINES_TO_ANALYZE]
        else:
            self.stats['truncated_analysis'] = False
            self.stats['lines_analyzed'] = total_lines
            return lines

    def _human_readable_size(self, size_bytes: int) -> str:
        """
        Convierte bytes a formato legible por humanos.
        
        Args:
            size_bytes (int): Tamaño en bytes
            
        Returns:
            str: Tamaño formateado (ej: "1.5 MB")
        """
        if size_bytes == 0:
            return "0 B"
        
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(size_bytes)
        unit_index = 0
        
        while size >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1
        
        # Formato con decimales apropiados
        if size < 10:
            return f"{size:.2f} {units[unit_index]}"
        elif size < 100:
            return f"{size:.1f} {units[unit_index]}"
        else:
            return f"{size:.0f} {units[unit_index]}"

    def _read_with_fallback_encoding(self) -> Optional[str]:
        """
        Intenta leer el archivo con múltiples encodings.
        
        Estrategia:
        1. Probar encodings predefinidos en orden
        2. Si falla, usar chardet para detección automática
        3. Como último recurso, leer con errores ignorados
        
        Returns:
            Optional[str]: Contenido del archivo o None si falla completamente
        """
        # Estrategia 1: Encodings predefinidos
        for encoding in self.ENCODINGS_TO_TRY:
            try:
                with self.file_path.open('r', encoding=encoding, errors='strict') as f:
                    content = f.read()
                
                self._encoding = encoding
                self.stats['encoding'] = encoding
                self.stats['encoding_method'] = 'predefined'
                logger.info(f"✅ Archivo leído con encoding: {encoding}")
                return content
                
            except (UnicodeDecodeError, LookupError):
                continue
            except Exception as e:
                logger.debug(f"Error inesperado con encoding {encoding}: {e}")
                continue
        
        # Estrategia 2: Detección automática con chardet
        if CHARDET_AVAILABLE:
            content = self._read_with_chardet()
            if content:
                return content
        
        # Estrategia 3: Último recurso - leer con errores reemplazados
        logger.warning("⚠️ Usando estrategia de último recurso: lectura con errores reemplazados")
        try:
            with self.file_path.open('r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            self._encoding = 'utf-8'
            self.stats['encoding'] = 'utf-8 (con errores)'
            self.stats['encoding_method'] = 'fallback_with_errors'
            logger.warning("⚠️ Archivo leído con reemplazo de caracteres inválidos")
            return content
            
        except Exception as e:
            logger.error(f"❌ Error fatal al leer archivo: {e}")
            return None

    def _read_with_chardet(self) -> Optional[str]:
        """
        Intenta leer el archivo usando chardet para detectar el encoding.
        
        Returns:
            Optional[str]: Contenido del archivo o None si falla
        """
        try:
            logger.info("🔍 Intentando detección automática de encoding con chardet...")
            
            # Leer muestra para detección
            sample_size = min(self.CHARDET_SAMPLE_SIZE, self.stats['file_size_bytes'])
            with self.file_path.open('rb') as f:
                raw_data = f.read(sample_size)
            
            detection = chardet.detect(raw_data)
            confidence = detection.get('confidence', 0)
            detected_encoding = detection.get('encoding')
            
            logger.info(
                f"Chardet detectó: {detected_encoding} "
                f"(confianza: {confidence:.2%})"
            )
            
            # Validar confianza mínima
            if confidence < self.CHARDET_MIN_CONFIDENCE:
                logger.warning(
                    f"⚠️ Confianza insuficiente en detección automática "
                    f"({confidence:.2%} < {self.CHARDET_MIN_CONFIDENCE:.2%})"
                )
                return None
            
            if not detected_encoding:
                logger.warning("⚠️ Chardet no pudo determinar un encoding")
                return None
            
            # Intentar leer con el encoding detectado
            try:
                with self.file_path.open('r', encoding=detected_encoding) as f:
                    content = f.read()
                
                self._encoding = detected_encoding
                self.stats['encoding'] = detected_encoding
                self.stats['encoding_method'] = 'chardet'
                self.stats['encoding_confidence'] = f"{confidence:.2%}"
                logger.info(f"✅ Archivo leído con encoding detectado: {detected_encoding}")
                return content
                
            except (UnicodeDecodeError, LookupError) as e:
                logger.error(f"❌ Falló lectura con encoding detectado ({detected_encoding}): {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error durante detección automática de encoding: {e}")
            return None

    def _detect_separator_from_lines(self, lines: List[str]) -> None:
        """
        Detecta el separador de columnas analizando líneas no vacías.
        
        Args:
            lines (List[str]): Líneas a analizar
        """
        # Filtrar líneas útiles (no vacías, no comentarios)
        useful_lines = [
            line.strip() 
            for line in lines[:self.SEPARATOR_DETECTION_SAMPLE_LINES]
            if line.strip() and not self._is_comment_line(line.strip())
        ]
        
        if not useful_lines:
            logger.warning("⚠️ No hay líneas útiles para detectar separador. Usando ';'")
            self._separator = ';'
            self.stats['detected_separator'] = ';'
            self.stats['separator_confidence'] = ConfidenceLevel.NONE.value
            return
        
        # Analizar frecuencia de separadores comunes
        separators = {
            ';': 'punto y coma',
            ',': 'coma',
            '\t': 'tabulación',
            '|': 'pipe',
            '^': 'circunflejo'
        }
        
        separator_stats = {}
        
        for sep, name in separators.items():
            counts = [line.count(sep) for line in useful_lines]
            # Calcular estadísticas robustas
            if counts:
                avg_count = sum(counts) / len(counts)
                max_count = max(counts)
                min_count = min(counts)
                # Varianza para medir consistencia
                variance = sum((c - avg_count) ** 2 for c in counts) / len(counts)
                consistency = 1.0 / (1.0 + variance) if variance > 0 else 1.0
                
                separator_stats[sep] = {
                    'name': name,
                    'avg': avg_count,
                    'max': max_count,
                    'min': min_count,
                    'variance': variance,
                    'consistency': consistency,
                    'score': avg_count * consistency  # Score ponderado
                }
        
        # Seleccionar mejor candidato
        if separator_stats:
            # Ordenar por score (frecuencia * consistencia)
            best_sep = max(separator_stats.items(), key=lambda x: x[1]['score'])
            separator = best_sep[0]
            stats = best_sep[1]
            
            # Determinar nivel de confianza
            if stats['avg'] >= 3 and stats['consistency'] > 0.7:
                confidence = ConfidenceLevel.HIGH
            elif stats['avg'] >= 2 and stats['consistency'] > 0.5:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
            
            self._separator = separator
            self.stats['detected_separator'] = separator
            self.stats['separator_confidence'] = confidence.value
            self.stats['separator_avg_count'] = f"{stats['avg']:.1f}"
            
            logger.info(
                f"✅ Separador detectado: '{separator}' ({stats['name']}) - "
                f"Confianza: {confidence.value} "
                f"(promedio: {stats['avg']:.1f}, consistencia: {stats['consistency']:.2f})"
            )
        else:
            # Fallback
            logger.warning("⚠️ No se pudo detectar separador. Usando ';' por defecto")
            self._separator = ';'
            self.stats['detected_separator'] = ';'
            self.stats['separator_confidence'] = ConfidenceLevel.LOW.value

    def _is_comment_line(self, line: str) -> bool:
        """
        Determina si una línea es un comentario.
        
        Args:
            line (str): Línea a evaluar (ya debe estar stripped)
            
        Returns:
            bool: True si es comentario, False en caso contrario
        """
        comment_markers = ('#', '//', '*', '--', '/*', '\'', 'REM')
        return any(line.startswith(marker) for marker in comment_markers)

    def _analyze_structure_single_pass(self, lines: List[str]) -> None:
        """
        Analiza la estructura del archivo en una sola pasada (optimizado).
        
        Args:
            lines (List[str]): Líneas a analizar
        """
        if not self._separator:
            logger.error("❌ Separador no detectado. No se puede analizar estructura.")
            return
        
        separator = self._separator
        potential_headers: List[HeaderCandidate] = []
        header_line_num = None
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Clasificar línea
            if not stripped:
                self.stats['empty_lines'] += 1
                continue
            
            self.stats['non_empty_lines'] += 1
            
            if self._is_comment_line(stripped):
                self.stats['comment_lines'] += 1
                continue
            
            # Si aún no encontramos el encabezado, buscar candidatos
            if header_line_num is None:
                header_match_info = self._evaluate_header_candidate(stripped, line_num, separator)
                if header_match_info:
                    potential_headers.append(header_match_info)
                    # Si tenemos un candidato muy fuerte, dejamos de buscar
                    if header_match_info.match_count >= 4:
                        header_line_num = line_num
                        self.header_candidate = header_match_info
                        logger.info(
                            f"✅ Encabezado fuerte detectado en línea {line_num} "
                            f"({header_match_info.match_count} coincidencias)"
                        )
            else:
                # Ya encontramos el encabezado, procesar como datos
                self._process_data_line(stripped, line_num, separator)
        
        # Si no se encontró un encabezado fuerte, seleccionar el mejor candidato
        if header_line_num is None and potential_headers:
            # Ordenar por número de coincidencias y número de columnas
            best_candidate = max(
                potential_headers,
                key=lambda x: (x.match_count, x.column_count, -x.line_num)
            )
            self.header_candidate = best_candidate
            header_line_num = best_candidate.line_num
            logger.info(
                f"✅ Mejor candidato a encabezado: línea {header_line_num} "
                f"({best_candidate.match_count} coincidencias)"
            )
            
            # Re-procesar líneas posteriores al encabezado como datos
            for line_num, line in enumerate(lines, 1):
                if line_num <= header_line_num:
                    continue
                stripped = line.strip()
                if stripped and not self._is_comment_line(stripped):
                    self._process_data_line(stripped, line_num, separator)
        
        # Registrar resultados
        if self.header_candidate:
            self.stats['header_found_at_line'] = self.header_candidate.line_num
            self.stats['header_column_count'] = self.header_candidate.column_count
        else:
            logger.warning("⚠️ No se detectó ningún encabezado válido")
            self.stats['header_found_at_line'] = None

    def _evaluate_header_candidate(
        self, 
        line: str, 
        line_num: int, 
        separator: str
    ) -> Optional[HeaderCandidate]:
        """
        Evalúa si una línea puede ser el encabezado.
        
        Args:
            line (str): Línea a evaluar
            line_num (int): Número de línea
            separator (str): Separador de columnas
            
        Returns:
            Optional[HeaderCandidate]: Información del candidato si cumple criterios, None en caso contrario
        """
        normalized = self._normalize_header_text(line)
        matches = [kw for kw in self.HEADER_KEYWORDS if kw in normalized]
        match_count = len(matches)
        
        if match_count < self.MIN_HEADER_KEYWORD_MATCHES:
            return None
        
        columns = [col.strip() for col in line.split(separator)]
        column_count = len(columns)
        
        # Determinar confianza preliminar
        if match_count >= 5:
            confidence = ConfidenceLevel.HIGH
        elif match_count >= 3:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW
        
        try:
            return HeaderCandidate(
                line_num=line_num,
                content=line,
                matches=matches,
                match_count=match_count,
                column_count=column_count,
                confidence=confidence
            )
        except ValueError as e:
            logger.warning(f"Error al crear HeaderCandidate: {e}")
            return None

    def _process_data_line(self, line: str, line_num: int, separator: str) -> None:
        """
        Procesa una línea de datos (posterior al encabezado).
        
        Args:
            line (str): Línea a procesar
            line_num (int): Número de línea
            separator (str): Separador de columnas
        """
        # Ignorar líneas de totales o resúmenes
        if line.strip().upper().startswith('TOTAL'):
            self.stats['summary_lines_ignored'] += 1
            return

        columns = [col.strip() for col in line.split(separator)]
        num_cols = len(columns)
        
        # Actualizar estadísticas de columnas
        self.stats[f'lines_with_{num_cols}_columns'] += 1
        
        # Agregar a análisis de columnas
        self.column_analysis[num_cols].add_sample(
            line, 
            max_samples=self.MAX_SAMPLES_PER_COLUMN_COUNT
        )
        
        # Guardar muestra si no hemos alcanzado el límite
        if len(self.sample_lines) < self.MAX_SAMPLE_LINES:
            try:
                sample = SampleLine(
                    line_num=line_num,
                    content=line,
                    column_count=num_cols
                )
                self.sample_lines.append(sample)
            except ValueError as e:
                logger.warning(f"Error al crear SampleLine: {e}")
        
        # Establecer línea de inicio de datos (primera línea de datos)
        if self.data_start_line is None:
            self.data_start_line = line_num

    def _normalize_header_text(self, text: str) -> str:
        """
        Normaliza texto para detección de encabezados.
        
        Args:
            text (str): Texto a normalizar
            
        Returns:
            str: Texto normalizado (mayúsculas, sin acentos, sin puntuación)
        """
        # Convertir a mayúsculas
        normalized = text.upper()
        
        # Eliminar acentos comunes
        accent_map = {
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'À': 'A', 'È': 'E', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U',
            'Â': 'A', 'Ê': 'E', 'Î': 'I', 'Ô': 'O', 'Û': 'U',
            'Ä': 'A', 'Ë': 'E', 'Ï': 'I', 'Ö': 'O', 'Ü': 'U',
            'Ñ': 'N', 'Ç': 'C'
        }
        for accented, plain in accent_map.items():
            normalized = normalized.replace(accented, plain)
        
        # Eliminar caracteres especiales comunes en encabezados
        chars_to_remove = '.,$%:"/\\\'*#@!¡¿?()[]{}«»""''<>~`´¨^'
        for char in chars_to_remove:
            normalized = normalized.replace(char, '')
        
        # Normalizar espacios múltiples a uno solo
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()

    def _calculate_column_statistics(self) -> None:
        """Calcula estadísticas porcentuales para cada conteo de columnas."""
        total_data_lines = sum(stats.count for stats in self.column_analysis.values())
        
        if total_data_lines == 0:
            logger.warning("⚠️ No se encontraron líneas de datos para análisis")
            return
        
        for num_cols, stats in self.column_analysis.items():
            stats.percentage = (stats.count / total_data_lines) * 100
        
        # Identificar conteo de columnas dominante
        if self.column_analysis:
            dominant = max(self.column_analysis.items(), key=lambda x: x[1].count)
            self.stats['dominant_column_count'] = dominant[0]
            self.stats['column_consistency'] = dominant[1].percentage / 100

    def _determine_confidence_levels(self) -> None:
        """Determina y actualiza los niveles de confianza de las detecciones."""
        # Confianza en la consistencia de columnas
        consistency = self.stats.get('column_consistency', 0)
        
        if consistency >= 0.95:
            self.stats['column_consistency_level'] = ConfidenceLevel.HIGH.value
        elif consistency >= self.COLUMN_CONSISTENCY_THRESHOLD:
            self.stats['column_consistency_level'] = ConfidenceLevel.MEDIUM.value
        elif consistency >= 0.7:
            self.stats['column_consistency_level'] = ConfidenceLevel.LOW.value
        else:
            self.stats['column_consistency_level'] = ConfidenceLevel.NONE.value
        
        # Actualizar confianza del encabezado basado en consistencia
        if self.header_candidate:
            if (self.header_candidate.column_count == self.stats.get('dominant_column_count') 
                and consistency >= 0.9):
                self.header_candidate.confidence = ConfidenceLevel.HIGH

    def _build_result_dict(self) -> Dict[str, Any]:
        """
        Construye el diccionario de resultados del diagnóstico.
        
        Returns:
            Dict[str, Any]: Diccionario con todos los resultados
        """
        result = {
            'success': True,
            'file_path': str(self.file_path),
            'stats': dict(self.stats),
            'encoding': self._encoding,
            'separator': self._separator,
            'file_size': self.stats['file_size_human'],
        }
        
        if self.header_candidate:
            result['header_candidate'] = {
                'line_num': self.header_candidate.line_num,
                'content': self.header_candidate.content,
                'matches': self.header_candidate.matches,
                'match_count': self.header_candidate.match_count,
                'column_count': self.header_candidate.column_count,
                'confidence': self.header_candidate.confidence.value
            }
            result['data_start_line'] = self.data_start_line
        else:
            result['header_candidate'] = None
            result['data_start_line'] = None
        
        # Información de columnas
        if self.column_analysis:
            result['column_distribution'] = {
                num_cols: {
                    'count': stats.count,
                    'percentage': f"{stats.percentage:.1f}%"
                }
                for num_cols, stats in sorted(self.column_analysis.items())
            }
        
        return result

    def _generate_diagnostic_report(self) -> None:
        """Genera un reporte formateado detallado con hallazgos y recomendaciones."""
        report_lines = [
            "\n" + "=" * 90,
            "📊 REPORTE DE DIAGNÓSTICO DEL ARCHIVO DE PRESUPUESTO".center(90),
            "=" * 90
        ]
        
        # Sección 1: Información básica del archivo
        report_lines.extend([
            "\n📁 INFORMACIÓN BÁSICA DEL ARCHIVO:",
            f"  📂 Ruta: {self.file_path}",
            f"  💾 Tamaño: {self.stats.get('file_size_human', 'desconocido')}",
            f"  📏 Total de líneas: {self.stats.get('total_lines', 0):,}",
            f"  📝 Líneas analizadas: {self.stats.get('lines_analyzed', 0):,}"
        ])
        
        if self.stats.get('truncated_analysis'):
            report_lines.append("  ⚠️  Análisis limitado debido al tamaño del archivo")
        
        # Sección 2: Detalles de encoding
        report_lines.extend([
            f"\n🔤 ENCODING Y FORMATO:",
            f"  Encoding detectado: {self._encoding or 'desconocido'}",
            f"  Método de detección: {self.stats.get('encoding_method', 'desconocido')}"
        ])
        
        if 'encoding_confidence' in self.stats:
            report_lines.append(f"  Confianza: {self.stats['encoding_confidence']}")
        
        # Sección 3: Separador de columnas
        sep_display = repr(self._separator) if self._separator else 'desconocido'
        report_lines.extend([
            f"  Separador detectado: {sep_display}",
            f"  Confianza del separador: {self.stats.get('separator_confidence', 'desconocida')}"
        ])
        
        # Sección 4: Estadísticas generales
        report_lines.extend([
            f"\n📈 ESTADÍSTICAS GENERALES:",
            f"  ✓ Líneas no vacías: {self.stats.get('non_empty_lines', 0):,}",
            f"  ∅ Líneas vacías: {self.stats.get('empty_lines', 0):,}",
            f"  # Líneas de comentario: {self.stats.get('comment_lines', 0):,}"
        ])
        
        # Sección 5: Resultados de detección de encabezado
        if self.header_candidate:
            report_lines.extend([
                f"\n✅ ENCABEZADO DETECTADO (Línea {self.header_candidate.line_num}):",
                f"  Contenido: {self.header_candidate.content[:100]}{'...' if len(self.header_candidate.content) > 100 else ''}",
                f"  Columnas detectadas: {self.header_candidate.column_count}",
                f"  Palabras clave encontradas ({self.header_candidate.match_count}):",
                f"    {', '.join(self.header_candidate.matches)}",
                f"  Nivel de confianza: {self.header_candidate.confidence.value.upper()}"
            ])
            
            if self.data_start_line:
                report_lines.append(f"  Datos comienzan en línea: {self.data_start_line}")
        else:
            report_lines.extend([
                "\n⚠️  NO SE DETECTÓ UNA FILA DE ENCABEZADO CLARA",
                "  Posibles causas:",
                "    • El archivo no contiene un encabezado estándar",
                "    • Las palabras clave del encabezado son diferentes a las esperadas",
                "    • El formato del archivo no es compatible"
            ])
        
        # Sección 6: Análisis de distribución de columnas
        if self.column_analysis:
            report_lines.append("\n📊 ANÁLISIS DE DISTRIBUCIÓN DE COLUMNAS:")
            
            total_data_lines = sum(stats.count for stats in self.column_analysis.values())
            dominant_count = self.stats.get('dominant_column_count')
            
            for num_cols in sorted(self.column_analysis.keys()):
                stats = self.column_analysis[num_cols]
                is_dominant = (num_cols == dominant_count)
                marker = "✓" if is_dominant else "⚠"
                status = "DOMINANTE" if is_dominant else "MINORITARIA"
                
                report_lines.append(
                    f"  {marker} {num_cols} columna(s): {stats.count:,} líneas "
                    f"({stats.percentage:.1f}%) [{status}]"
                )
                
                # Mostrar ejemplos de líneas inconsistentes
                if not is_dominant and stats.samples:
                    example = stats.samples[0]
                    truncated = example[:80] + "..." if len(example) > 80 else example
                    report_lines.append(f"      Ejemplo: {truncated}")
        
        # Sección 7: Muestra de líneas de datos
        if self.sample_lines:
            report_lines.append(f"\n📝 MUESTRA DE LÍNEAS DE DATOS:")
            
            for sample in self.sample_lines[:self.MAX_REPORT_SAMPLE_LINES]:
                truncated = sample.content[:85] + "..." if len(sample.content) > 85 else sample.content
                report_lines.append(
                    f"  Línea {sample.line_num:>5} ({sample.column_count} cols): {truncated}"
                )
            
            if len(self.sample_lines) > self.MAX_REPORT_SAMPLE_LINES:
                remaining = len(self.sample_lines) - self.MAX_REPORT_SAMPLE_LINES
                report_lines.append(f"  ... y {remaining} línea(s) más")
        
        # Sección 8: Recomendaciones
        report_lines.append("\n💡 RECOMENDACIONES PARA PROCESAMIENTO:")
        
        if self.header_candidate:
            # Parámetro header para pandas (0-indexed)
            pandas_header = self.header_candidate.line_num - 1
            report_lines.append(f"  ✓ Usar header={pandas_header} al leer con pandas")
            
            # Advertencia si el encabezado no está en línea 0
            if pandas_header > 5:
                report_lines.append(
                    f"    ⚠ El encabezado está en línea {self.header_candidate.line_num}. "
                    f"Considere skiprows=[0-{pandas_header-1}]"
                )
        else:
            report_lines.extend([
                "  ⚠ No se detectó encabezado automáticamente",
                "    • Revisar manualmente las primeras líneas del archivo",
                "    • Especificar header=None y proporcionar names=[...] al leer"
            ])
        
        # Recomendaciones de lectura
        if self._separator and self._encoding:
            report_lines.append(
                f"  ✓ Parámetros de lectura: sep={repr(self._separator)}, "
                f"encoding='{self._encoding}'"
            )
        
        # Advertencias sobre consistencia
        consistency = self.stats.get('column_consistency', 0)
        if consistency < self.COLUMN_CONSISTENCY_THRESHOLD:
            report_lines.extend([
                f"  ⚠ ADVERTENCIA: Inconsistencia en columnas detectada "
                f"(consistencia: {consistency:.1%})",
                "    • Verificar que el separador detectado sea correcto",
                "    • Revisar si hay líneas de resumen o totales mezcladas con datos",
                "    • Considerar filtrar líneas por número de columnas durante el procesamiento"
            ])
        else:
            report_lines.append(
                f"  ✓ Consistencia de columnas: {consistency:.1%} "
                f"({self.stats.get('column_consistency_level', 'desconocida')})"
            )
        
        # Código de ejemplo para pandas
        if self.header_candidate and self._separator and self._encoding:
            report_lines.extend([
                "\n🐍 EJEMPLO DE CÓDIGO PANDAS:",
                "```python",
                "import pandas as pd",
                "",
                f"df = pd.read_csv(",
                f"    '{self.file_path.name}',",
                f"    sep={repr(self._separator)},",
                f"    encoding='{self._encoding}',",
                f"    header={self.header_candidate.line_num - 1}"
            ])
            
            # Agregar parámetros opcionales según el análisis
            if consistency < 0.95:
                dominant_cols = self.stats.get('dominant_column_count')
                if dominant_cols:
                    report_lines.append(f"    # on_bad_lines='warn'  # Advertir sobre líneas inconsistentes")
            
            report_lines.extend([
                ")",
                "```"
            ])
        
        # Sección final
        report_lines.extend([
            "\n" + "=" * 90,
            "✅ FIN DEL REPORTE DE DIAGNÓSTICO".center(90),
            "=" * 90 + "\n"
        ])
        
        # Imprimir reporte
        for line in report_lines:
            logger.info(line)


def main() -> int:
    """
    Función principal para ejecución desde línea de comandos.
    
    Returns:
        int: Código de salida (0 = éxito, 1 = error)
    """
    if len(sys.argv) < 2:
        logger.error("❌ Error: Debe proporcionar la ruta al archivo de presupuesto")
        print("\n" + "=" * 70)
        print("USO DEL SCRIPT DE DIAGNÓSTICO".center(70))
        print("=" * 70)
        print("\nSintaxis:")
        print("  python diagnose_presupuesto_file.py <ruta_al_archivo>")
        print("\nEjemplos:")
        print("  python diagnose_presupuesto_file.py presupuesto.csv")
        print("  python diagnose_presupuesto_file.py /ruta/completa/datos.txt")
        print("\nDescripción:")
        print("  Analiza la estructura de un archivo de presupuesto y genera")
        print("  un reporte detallado con recomendaciones para su procesamiento.")
        print("=" * 70 + "\n")
        return 1
    
    file_path = sys.argv[1]
    
    try:
        logger.info("=" * 80)
        logger.info(f"🚀 INICIANDO DIAGNÓSTICO".center(80))
        logger.info(f"Archivo: {file_path}".center(80))
        logger.info("=" * 80)
        
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()
        
        if not result or not result.get('success'):
            logger.error("❌ El diagnóstico no pudo completarse exitosamente")
            return 1
        
        logger.info("\n" + "🎉 DIAGNÓSTICO COMPLETADO EXITOSAMENTE 🎉".center(80))
        return 0
        
    except ValueError as ve:
        logger.error(f"❌ Error de validación: {ve}")
        return 1
    except PermissionError as pe:
        logger.error(f"❌ Error de permisos: {pe}")
        return 1
    except FileReadError as fre:
        logger.error(f"❌ Error al leer archivo: {fre}")
        return 1
    except DiagnosticError as de:
        logger.error(f"❌ Error de diagnóstico: {de}")
        return 1
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Diagnóstico interrumpido por el usuario")
        return 130
    except Exception as e:
        logger.exception(f"❌ Error inesperado: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())