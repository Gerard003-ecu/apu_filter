# scripts/diagnose_insumos_file.py
"""
Herramienta de diagnóstico avanzada para analizar archivos de insumos.

Este módulo proporciona capacidades robustas de análisis de estructura jerárquica
(grupos y tablas), detección automática de encoding, identificación de separadores
y generación de reportes detallados con recomendaciones.
"""
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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
logger = logging.getLogger("InsumosDiagnostic")


class DiagnosticError(Exception):
    """Excepción base para errores de diagnóstico."""
    pass


class FileReadError(DiagnosticError):
    """Error al leer el archivo."""
    pass


class EncodingDetectionError(DiagnosticError):
    """Error al detectar el encoding del archivo."""
    pass


class StructureError(DiagnosticError):
    """Error en la estructura del archivo."""
    pass


class ConfidenceLevel(Enum):
    """Niveles de confianza para las detecciones."""
    HIGH = "alta"
    MEDIUM = "media"
    LOW = "baja"
    NONE = "ninguna"


@dataclass
class GroupInfo:
    """Representa información de un grupo detectado en el archivo."""
    name: str
    line_num: int
    header_line: Optional[int] = None
    data_lines: int = 0
    column_counts: Counter = field(default_factory=Counter)
    samples: List[str] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    def __post_init__(self):
        """Valida la consistencia de los datos."""
        if self.line_num < 1:
            raise ValueError("line_num debe ser >= 1")
        if not self.name or not self.name.strip():
            raise ValueError("name no puede estar vacío")
        if self.data_lines < 0:
            raise ValueError("data_lines debe ser >= 0")
        if self.header_line is not None and self.header_line < self.line_num:
            raise ValueError("header_line debe ser posterior a line_num del grupo")
    
    @property
    def dominant_column_count(self) -> Optional[int]:
        """Retorna el número de columnas más frecuente en este grupo."""
        if not self.column_counts:
            return None
        return self.column_counts.most_common(1)[0][0]
    
    @property
    def column_consistency(self) -> float:
        """Calcula la consistencia de columnas en este grupo."""
        if self.data_lines == 0:
            return 0.0
        dominant = self.dominant_column_count
        if dominant is None:
            return 0.0
        return self.column_counts[dominant] / self.data_lines
    
    @property
    def has_header(self) -> bool:
        """Indica si el grupo tiene encabezado detectado."""
        return self.header_line is not None
    
    @property
    def has_data(self) -> bool:
        """Indica si el grupo tiene líneas de datos."""
        return self.data_lines > 0


@dataclass
class SampleLine:
    """Representa una línea de muestra del archivo."""
    line_num: int
    content: str
    group_name: Optional[str] = None
    column_count: int = 0
    
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


@dataclass
class HeaderCandidate:
    """Representa un candidato a encabezado de tabla."""
    line_num: int
    content: str
    matches: List[str]
    match_count: int
    column_count: int
    group_name: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    def __post_init__(self):
        """Valida la consistencia de los datos."""
        if self.line_num < 1:
            raise ValueError("line_num debe ser >= 1")
        if self.match_count != len(self.matches):
            raise ValueError("match_count debe coincidir con len(matches)")
        if self.column_count < 1:
            raise ValueError("column_count debe ser >= 1")


class InsumosFileDiagnostic:
    """
    Herramienta de diagnóstico para analizar la estructura de un archivo de Insumos.
    
    Esta clase proporciona métodos para:
    - Detectar automáticamente el encoding del archivo con múltiples estrategias
    - Identificar el separador de columnas mediante análisis estadístico robusto
    - Analizar la estructura jerárquica de grupos y sus tablas asociadas
    - Detectar inconsistencias en el formato de los datos por grupo
    - Validar la integridad estructural (grupos con encabezados y datos)
    - Generar reportes detallados con recomendaciones accionables
    
    Estructura esperada del archivo:
        G;<NOMBRE_GRUPO>
        <ENCABEZADO_TABLA>
        <DATOS>
        <DATOS>
        ...
        G;<SIGUIENTE_GRUPO>
        ...
    
    Attributes:
        file_path (Path): Ruta absoluta al archivo a diagnosticar
        ENCODINGS_TO_TRY (List[str]): Encodings a probar en orden de preferencia
        GROUP_PREFIXES (List[str]): Prefijos que identifican líneas de grupo
        HEADER_KEYWORDS (List[str]): Palabras clave para identificar encabezados
        MIN_HEADER_KEYWORD_MATCHES (int): Coincidencias mínimas para validar encabezado
        MAX_SAMPLE_LINES (int): Límite de líneas de muestra a almacenar
        MAX_REPORT_SAMPLE_LINES (int): Límite de líneas a mostrar en reporte
        MAX_LINES_TO_ANALYZE (int): Límite de líneas para archivos grandes
        MAX_GROUPS_TO_REPORT (int): Límite de grupos a reportar en detalle
    """
    
    # Configuración de encodings a probar
    ENCODINGS_TO_TRY = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16']
    
    # Prefijos para identificar líneas de grupo (normalizados)
    GROUP_PREFIXES = ['G;', 'G,', 'G\t', 'G|', 'GRUPO;', 'GRUPO,', 'GRUPO\t', 'GRUPO|']
    
    # Palabras clave para detección de encabezados de tablas
    HEADER_KEYWORDS = [
        'CODIGO', 'COD', 'DESCRIPCION', 'DESC', 'UND', 'UNIDAD',
        'VALOR', 'VR UNIT', 'VR UNITARIO', 'PRECIO', 'COSTO', 'TOTAL',
        'CANTIDAD', 'CANT', 'PARCIAL', 'SUBTOTAL'
    ]
    
    # Configuración de límites y umbrales
    MIN_HEADER_KEYWORD_MATCHES = 2
    MAX_SAMPLE_LINES = 20
    MAX_REPORT_SAMPLE_LINES = 15
    MAX_LINES_TO_ANALYZE = 2000
    MAX_GROUPS_TO_REPORT = 10
    MAX_SAMPLES_PER_GROUP = 3
    MAX_SAMPLES_PER_COLUMN_COUNT = 2
    CHARDET_SAMPLE_SIZE = 50000
    CHARDET_MIN_CONFIDENCE = 0.7
    SEPARATOR_DETECTION_SAMPLE_LINES = 100
    COLUMN_CONSISTENCY_THRESHOLD = 0.85
    MIN_DATA_LINES_FOR_VALID_GROUP = 1  # Mínimo de líneas de datos para considerar grupo válido

    def __init__(self, file_path: Union[str, Path]):
        """
        Inicializa el diagnosticador con la ruta del archivo.
        
        Args:
            file_path (Union[str, Path]): Ruta al archivo de insumos a analizar
            
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
        self.groups_found: List[GroupInfo] = []
        self.current_group: Optional[GroupInfo] = None
        self.column_analysis: Dict[int, ColumnStatistics] = defaultdict(ColumnStatistics)
        self._encoding: Optional[str] = None
        self._separator: Optional[str] = None
        self._group_names_seen: Set[str] = set()  # Para detectar duplicados

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
            logger.info(f"🔍 Iniciando diagnóstico del archivo de insumos: {self.file_path}")
            
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
            if total_lines < 3:
                logger.warning("⚠️ El archivo tiene muy pocas líneas para análisis significativo.")
            
            # Limitar análisis para archivos muy grandes
            lines_to_analyze = self._get_lines_to_analyze(lines)
            
            # Ejecutar análisis
            self._detect_separator_from_lines(lines_to_analyze)
            self._analyze_structure_hierarchical(lines_to_analyze)
            self._calculate_statistics()
            self._validate_groups_integrity()
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
        comment_markers = ('#', '//', '*', '--', '/*', '\'', 'REM', '%')
        return any(line.startswith(marker) for marker in comment_markers)

    def _normalize_text(self, text: str) -> str:
        """
        Normaliza texto eliminando comillas, espacios extra y convirtiendo a mayúsculas.
        
        Args:
            text (str): Texto a normalizar
            
        Returns:
            str: Texto normalizado
        """
        # Eliminar comillas y caracteres especiales
        normalized = text.upper()
        
        # Eliminar comillas, paréntesis y otros caracteres
        chars_to_remove = '"\'()[]{}«»""''<>'
        for char in chars_to_remove:
            normalized = normalized.replace(char, '')
        
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
        
        # Normalizar espacios múltiples a uno solo
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()

    def _is_group_line(self, line: str) -> Tuple[bool, Optional[str]]:
        """
        Determina si una línea es el inicio de un grupo.
        
        Args:
            line (str): Línea normalizada a evaluar
            
        Returns:
            Tuple[bool, Optional[str]]: (es_grupo, nombre_grupo)
        """
        if not self._separator:
            return False, None
        
        # Verificar si la línea comienza con algún prefijo de grupo
        for prefix in self.GROUP_PREFIXES:
            # Normalizar el prefijo para comparación
            normalized_prefix = prefix.replace(';', self._separator).replace(',', self._separator).replace('\t', self._separator).replace('|', self._separator)
            
            if line.startswith(normalized_prefix):
                # Extraer el nombre del grupo
                parts = line.split(self._separator, 1)
                if len(parts) >= 2:
                    group_name = parts[1].strip()
                    if group_name:  # Asegurar que el nombre no esté vacío
                        return True, group_name
        
        return False, None

    def _analyze_structure_hierarchical(self, lines: List[str]) -> None:
        """
        Analiza la estructura jerárquica del archivo (grupos y sus tablas).
        
        Args:
            lines (List[str]): Líneas a analizar
        """
        if not self._separator:
            logger.error("❌ Separador no detectado. No se puede analizar estructura.")
            return
        
        separator = self._separator
        
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
            
            # Normalizar línea para análisis
            normalized = self._normalize_text(stripped)
            
            # 1. Detectar inicio de nuevo grupo
            is_group, group_name = self._is_group_line(normalized)
            
            if is_group and group_name:
                # Finalizar grupo anterior si existe
                if self.current_group:
                    self._finalize_current_group()
                
                # Verificar duplicados
                if group_name in self._group_names_seen:
                    logger.warning(f"⚠️ Grupo duplicado detectado: '{group_name}' en línea {line_num}")
                    self.stats['duplicate_groups'] += 1
                    # Hacer único el nombre agregando sufijo
                    original_name = group_name
                    counter = 2
                    while group_name in self._group_names_seen:
                        group_name = f"{original_name} ({counter})"
                        counter += 1
                
                # Crear nuevo grupo
                try:
                    self.current_group = GroupInfo(
                        name=group_name,
                        line_num=line_num
                    )
                    self.groups_found.append(self.current_group)
                    self._group_names_seen.add(group_name)
                    self.stats['groups_detected'] += 1
                    logger.debug(f"🔍 Grupo detectado: '{group_name}' en línea {line_num}")
                except ValueError as e:
                    logger.error(f"Error al crear grupo: {e}")
                    self.current_group = None
                
                continue
            
            # 2. Si estamos dentro de un grupo, buscar encabezado o procesar datos
            if self.current_group:
                # Buscar encabezado si aún no se ha encontrado
                if not self.current_group.has_header:
                    header_info = self._evaluate_header_candidate(
                        stripped, 
                        line_num, 
                        separator,
                        self.current_group.name
                    )
                    
                    if header_info:
                        self.current_group.header_line = line_num
                        self.stats['header_lines_detected'] += 1
                        logger.debug(
                            f"📋 Encabezado detectado para grupo '{self.current_group.name}' "
                            f"en línea {line_num}: {', '.join(header_info.matches)}"
                        )
                        continue
                
                # Procesar como línea de datos si ya tenemos encabezado
                elif self.current_group.has_header:
                    self._process_data_line_for_group(stripped, line_num, separator)

        # Finalizar último grupo si existe
        if self.current_group:
            self._finalize_current_group()

    def _finalize_current_group(self) -> None:
        """Finaliza el procesamiento del grupo actual y actualiza sus estadísticas."""
        if not self.current_group:
            return
        
        # No hacer nada más, las estadísticas ya se actualizaron durante el procesamiento
        logger.debug(
            f"Finalizando grupo '{self.current_group.name}': "
            f"{self.current_group.data_lines} líneas de datos, "
            f"consistencia: {self.current_group.column_consistency:.1%}"
        )

    def _evaluate_header_candidate(
        self,
        line: str,
        line_num: int,
        separator: str,
        group_name: Optional[str] = None
    ) -> Optional[HeaderCandidate]:
        """
        Evalúa si una línea puede ser el encabezado de una tabla.
        
        Args:
            line (str): Línea a evaluar
            line_num (int): Número de línea
            separator (str): Separador de columnas
            group_name (Optional[str]): Nombre del grupo al que pertenece
            
        Returns:
            Optional[HeaderCandidate]: Información del candidato si cumple criterios, None en caso contrario
        """
        normalized = self._normalize_text(line)
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
                group_name=group_name,
                confidence=confidence
            )
        except ValueError as e:
            logger.warning(f"Error al crear HeaderCandidate: {e}")
            return None

    def _process_data_line_for_group(self, line: str, line_num: int, separator: str) -> None:
        """
        Procesa una línea de datos dentro de un grupo.
        
        Args:
            line (str): Línea a procesar
            line_num (int): Número de línea
            separator (str): Separador de columnas
        """
        if not self.current_group:
            return
        
        # Verificar que no sea otra línea de grupo
        normalized = self._normalize_text(line)
        is_group, _ = self._is_group_line(normalized)
        if is_group:
            return  # Esta línea será procesada en la siguiente iteración
        
        columns = [col.strip() for col in line.split(separator)]
        num_cols = len(columns)
        
        # Actualizar estadísticas del grupo
        self.current_group.data_lines += 1
        self.current_group.column_counts[num_cols] += 1
        
        # Guardar muestra del grupo
        if len(self.current_group.samples) < self.MAX_SAMPLES_PER_GROUP:
            self.current_group.samples.append(line)
        
        # Actualizar estadísticas globales de columnas
        self.column_analysis[num_cols].add_sample(line, self.MAX_SAMPLES_PER_COLUMN_COUNT)
        
        # Guardar muestra global
        if len(self.sample_lines) < self.MAX_SAMPLE_LINES:
            try:
                sample = SampleLine(
                    line_num=line_num,
                    content=line,
                    group_name=self.current_group.name,
                    column_count=num_cols
                )
                self.sample_lines.append(sample)
            except ValueError as e:
                logger.warning(f"Error al crear SampleLine: {e}")

    def _calculate_statistics(self) -> None:
        """Calcula estadísticas finales del análisis."""
        # Calcular estadísticas globales de columnas
        total_data_lines = sum(stats.count for stats in self.column_analysis.values())
        
        if total_data_lines > 0:
            for num_cols, stats in self.column_analysis.items():
                stats.percentage = (stats.count / total_data_lines) * 100
            
            # Identificar conteo de columnas dominante
            dominant = max(self.column_analysis.items(), key=lambda x: x[1].count)
            self.stats['dominant_column_count'] = dominant[0]
            self.stats['column_consistency'] = dominant[1].percentage / 100
        
        # Calcular estadísticas de grupos
        if self.groups_found:
            groups_with_headers = sum(1 for g in self.groups_found if g.has_header)
            groups_with_data = sum(1 for g in self.groups_found if g.has_data)
            
            self.stats['groups_with_headers'] = groups_with_headers
            self.stats['groups_with_data'] = groups_with_data
            self.stats['groups_complete'] = sum(
                1 for g in self.groups_found 
                if g.has_header and g.has_data
            )

    def _validate_groups_integrity(self) -> None:
        """Valida la integridad de los grupos detectados."""
        issues = []
        
        for group in self.groups_found:
            # Verificar que el grupo tenga encabezado
            if not group.has_header:
                issues.append(f"Grupo '{group.name}' sin encabezado detectado")
                group.confidence = ConfidenceLevel.LOW
            
            # Verificar que el grupo tenga datos
            if not group.has_data:
                issues.append(f"Grupo '{group.name}' sin líneas de datos")
                group.confidence = ConfidenceLevel.LOW
            
            # Verificar consistencia de columnas
            if group.has_data and group.column_consistency < self.COLUMN_CONSISTENCY_THRESHOLD:
                issues.append(
                    f"Grupo '{group.name}' con columnas inconsistentes "
                    f"({group.column_consistency:.1%})"
                )
                if group.confidence == ConfidenceLevel.HIGH:
                    group.confidence = ConfidenceLevel.MEDIUM
        
        if issues:
            self.stats['integrity_issues'] = len(issues)
            logger.warning(f"⚠️ Se detectaron {len(issues)} problemas de integridad:")
            for issue in issues[:10]:  # Limitar a 10 para no saturar el log
                logger.warning(f"  - {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... y {len(issues) - 10} problemas más")

    def _determine_confidence_levels(self) -> None:
        """Determina y actualiza los niveles de confianza de las detecciones."""
        # Confianza general basada en múltiples factores
        factors = []
        
        # Factor 1: Presencia de grupos
        if self.stats.get('groups_detected', 0) > 0:
            factors.append(1.0)
        else:
            factors.append(0.0)
        
        # Factor 2: Grupos completos (con encabezado y datos)
        total_groups = self.stats.get('groups_detected', 0)
        complete_groups = self.stats.get('groups_complete', 0)
        if total_groups > 0:
            factors.append(complete_groups / total_groups)
        
        # Factor 3: Consistencia de columnas
        consistency = self.stats.get('column_consistency', 0)
        factors.append(consistency)
        
        # Calcular confianza general
        if factors:
            overall_confidence = sum(factors) / len(factors)
            
            if overall_confidence >= 0.9:
                self.stats['overall_confidence'] = ConfidenceLevel.HIGH.value
            elif overall_confidence >= 0.7:
                self.stats['overall_confidence'] = ConfidenceLevel.MEDIUM.value
            elif overall_confidence >= 0.5:
                self.stats['overall_confidence'] = ConfidenceLevel.LOW.value
            else:
                self.stats['overall_confidence'] = ConfidenceLevel.NONE.value
            
            self.stats['overall_confidence_score'] = f"{overall_confidence:.1%}"

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
        
        # Información de grupos
        if self.groups_found:
            result['groups_found'] = [
                {
                    'name': group.name,
                    'line_num': group.line_num,
                    'header_line': group.header_line,
                    'data_lines': group.data_lines,
                    'dominant_columns': group.dominant_column_count,
                    'column_consistency': f"{group.column_consistency:.1%}",
                    'confidence': group.confidence.value,
                    'has_header': group.has_header,
                    'has_data': group.has_data
                }
                for group in self.groups_found
            ]
        else:
            result['groups_found'] = []
        
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
            "📊 REPORTE DE DIAGNÓSTICO DEL ARCHIVO DE INSUMOS".center(90),
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
        
        # Sección 5: Análisis de grupos (PRINCIPAL)
        total_groups = self.stats.get('groups_detected', 0)
        report_lines.append(f"\n📦 ANÁLISIS DE GRUPOS:")
        report_lines.append(f"  Total de grupos detectados: {total_groups}")
        
        if total_groups > 0:
            groups_with_headers = self.stats.get('groups_with_headers', 0)
            groups_with_data = self.stats.get('groups_with_data', 0)
            groups_complete = self.stats.get('groups_complete', 0)
            
            report_lines.extend([
                f"  Grupos con encabezado: {groups_with_headers} ({groups_with_headers/total_groups:.1%})",
                f"  Grupos con datos: {groups_with_data} ({groups_with_data/total_groups:.1%})",
                f"  Grupos completos: {groups_complete} ({groups_complete/total_groups:.1%})"
            ])
            
            if self.stats.get('duplicate_groups', 0) > 0:
                report_lines.append(
                    f"  ⚠️ Grupos duplicados detectados: {self.stats['duplicate_groups']}"
                )
            
            # Detalles de grupos individuales
            report_lines.append(f"\n  📋 DETALLE DE GRUPOS:")
            
            for i, group in enumerate(self.groups_found[:self.MAX_GROUPS_TO_REPORT], 1):
                # Indicadores de estado
                header_status = "✓" if group.has_header else "✗"
                data_status = "✓" if group.has_data else "✗"
                
                # Indicador de consistencia
                if group.has_data:
                    consistency = group.column_consistency
                    if consistency >= 0.95:
                        consistency_marker = "✓"
                    elif consistency >= self.COLUMN_CONSISTENCY_THRESHOLD:
                        consistency_marker = "~"
                    else:
                        consistency_marker = "⚠"
                else:
                    consistency_marker = "-"
                    consistency = 0.0
                
                report_lines.extend([
                    f"\n  {i}. {group.name}",
                    f"     Línea de grupo: {group.line_num}",
                    f"     {header_status} Encabezado: {'Línea ' + str(group.header_line) if group.header_line else 'No detectado'}",
                    f"     {data_status} Líneas de datos: {group.data_lines}"
                ])
                
                if group.has_data:
                    dominant_cols = group.dominant_column_count
                    report_lines.extend([
                        f"     Columnas dominantes: {dominant_cols}",
                        f"     {consistency_marker} Consistencia: {consistency:.1%}",
                        f"     Confianza del grupo: {group.confidence.value.upper()}"
                    ])
                    
                    # Mostrar distribución de columnas si hay inconsistencias
                    if len(group.column_counts) > 1:
                        report_lines.append(f"     Distribución de columnas:")
                        for cols, count in group.column_counts.most_common():
                            pct = count / group.data_lines * 100
                            marker = "→" if cols == dominant_cols else " "
                            report_lines.append(f"       {marker} {cols} cols: {count} líneas ({pct:.1f}%)")
            
            if len(self.groups_found) > self.MAX_GROUPS_TO_REPORT:
                remaining = len(self.groups_found) - self.MAX_GROUPS_TO_REPORT
                report_lines.append(f"\n  ... y {remaining} grupo(s) más")
        else:
            report_lines.extend([
                "  ❌ NO SE DETECTARON GRUPOS EN EL ARCHIVO",
                "  Esto puede indicar:",
                "    • El archivo no sigue el formato esperado",
                "    • El separador detectado es incorrecto",
                "    • Los prefijos de grupo son diferentes a los esperados"
            ])
        
        # Sección 6: Análisis global de columnas
        if self.column_analysis:
            report_lines.append(f"\n📊 ANÁLISIS GLOBAL DE COLUMNAS:")
            
            total_data_lines = sum(stats.count for stats in self.column_analysis.values())
            dominant_count = self.stats.get('dominant_column_count')
            
            report_lines.append(f"  Total de líneas de datos: {total_data_lines:,}")
            
            if dominant_count:
                report_lines.append(f"  Número de columnas principal: {dominant_count}")
                consistency = self.stats.get('column_consistency', 0)
                report_lines.append(f"  Consistencia global: {consistency:.1%}")
            
            report_lines.append(f"\n  Distribución por número de columnas:")
            for num_cols in sorted(self.column_analysis.keys()):
                stats = self.column_analysis[num_cols]
                is_dominant = (num_cols == dominant_count)
                marker = "→" if is_dominant else " "
                
                report_lines.append(
                    f"  {marker} {num_cols} columna(s): {stats.count:,} líneas ({stats.percentage:.1f}%)"
                )
                
                # Mostrar ejemplos de líneas inconsistentes
                if not is_dominant and stats.samples:
                    example = stats.samples[0]
                    truncated = example[:75] + "..." if len(example) > 75 else example
                    report_lines.append(f"      Ejemplo: {truncated}")
        
        # Sección 7: Muestra de líneas de datos
        if self.sample_lines:
            report_lines.append(f"\n📝 MUESTRA DE LÍNEAS DE DATOS:")
            
            for sample in self.sample_lines[:self.MAX_REPORT_SAMPLE_LINES]:
                group_info = f"[{sample.group_name}]" if sample.group_name else "[Sin grupo]"
                truncated = sample.content[:70] + "..." if len(sample.content) > 70 else sample.content
                report_lines.append(
                    f"  Línea {sample.line_num:>5} {group_info:30} ({sample.column_count} cols): {truncated}"
                )
            
            if len(self.sample_lines) > self.MAX_REPORT_SAMPLE_LINES:
                remaining = len(self.sample_lines) - self.MAX_REPORT_SAMPLE_LINES
                report_lines.append(f"  ... y {remaining} línea(s) más")
        
        # Sección 8: Problemas detectados
        if self.stats.get('integrity_issues', 0) > 0:
            report_lines.extend([
                f"\n⚠️  PROBLEMAS DE INTEGRIDAD DETECTADOS:",
                f"  Total de problemas: {self.stats['integrity_issues']}"
            ])
            
            # Listar grupos problemáticos
            problematic_groups = [
                g for g in self.groups_found 
                if not g.has_header or not g.has_data or g.column_consistency < 0.85
            ]
            
            if problematic_groups:
                report_lines.append("\n  Grupos con problemas:")
                for group in problematic_groups[:5]:
                    issues = []
                    if not group.has_header:
                        issues.append("sin encabezado")
                    if not group.has_data:
                        issues.append("sin datos")
                    if group.has_data and group.column_consistency < 0.85:
                        issues.append(f"columnas inconsistentes ({group.column_consistency:.1%})")
                    
                    report_lines.append(f"    • {group.name}: {', '.join(issues)}")
        
        # Sección 9: Recomendaciones
        report_lines.append(f"\n💡 RECOMENDACIONES PARA PROCESAMIENTO:")
        
        # Recomendaciones basadas en el análisis
        if total_groups == 0:
            report_lines.extend([
                "  ❌ CRÍTICO: No se detectaron grupos",
                "    • Verificar manualmente la estructura del archivo",
                "    • Validar que las líneas de grupo sigan el formato: G;<NOMBRE>",
                f"    • Confirmar que el separador sea '{self._separator}'",
                "    • Revisar los primeros caracteres de cada línea"
            ])
        else:
            complete_ratio = self.stats.get('groups_complete', 0) / total_groups
            
            if complete_ratio >= 0.9:
                report_lines.append("  ✅ Estructura del archivo es consistente y procesable")
            elif complete_ratio >= 0.7:
                report_lines.append("  ⚠️ Estructura mayormente correcta con algunos problemas menores")
            else:
                report_lines.append("  ❌ Estructura del archivo tiene problemas significativos")
            
            # Recomendaciones específicas de procesamiento
            report_lines.extend([
                "\n  📋 Parámetros de lectura recomendados:",
                f"    • sep={repr(self._separator)}",
                f"    • encoding='{self._encoding}'"
            ])
            
            # Estrategia de procesamiento
            report_lines.extend([
                "\n  🔧 Estrategia de procesamiento:",
                "    1. Identificar líneas que comiencen con 'G;' como inicios de grupo",
                "    2. La siguiente línea después del grupo es el encabezado de la tabla",
                "    3. Las líneas subsiguientes son datos hasta el próximo grupo"
            ])
            
            if self.stats.get('column_consistency', 0) < self.COLUMN_CONSISTENCY_THRESHOLD:
                report_lines.extend([
                    "\n  ⚠️ Inconsistencia de columnas detectada:",
                    "    • Implementar validación de número de columnas por línea",
                    "    • Considerar usar on_bad_lines='warn' en pandas",
                    "    • Revisar y limpiar líneas con columnas inconsistentes"
                ])
        
        # Código de ejemplo
        if total_groups > 0 and self._separator and self._encoding:
            report_lines.extend([
                "\n🐍 EJEMPLO DE CÓDIGO PYTHON PARA PROCESAMIENTO:",
                "```python",
                "import pandas as pd",
                "from pathlib import Path",
                "",
                f"file_path = Path('{self.file_path.name}')",
                f"separator = {repr(self._separator)}",
                f"encoding = '{self._encoding}'",
                "",
                "# Procesar archivo por grupos",
                "groups_data = {}",
                "current_group = None",
                "group_lines = []",
                "",
                "with file_path.open('r', encoding=encoding) as f:",
                "    for line in f:",
                "        line = line.strip()",
                "        if not line:",
                "            continue",
                "        ",
                "        # Detectar inicio de grupo",
                "        if line.startswith('G' + separator):",
                "            # Procesar grupo anterior si existe",
                "            if current_group and len(group_lines) > 1:",
                "                # Primera línea es encabezado, resto son datos",
                "                from io import StringIO",
                "                df = pd.read_csv(",
                "                    StringIO('\\n'.join(group_lines)),",
                "                    sep=separator",
                "                )",
                "                groups_data[current_group] = df",
                "            ",
                "            # Iniciar nuevo grupo",
                "            current_group = line.split(separator)[1]",
                "            group_lines = []",
                "        else:",
                "            group_lines.append(line)",
                "",
                "# Procesar último grupo",
                "if current_group and len(group_lines) > 1:",
                "    from io import StringIO",
                "    df = pd.read_csv(",
                "        StringIO('\\n'.join(group_lines)),",
                "        sep=separator",
                "    )",
                "    groups_data[current_group] = df",
                "",
                "# Ahora groups_data contiene un DataFrame por cada grupo",
                f"print(f'Grupos procesados: {{len(groups_data)}}')",
                "```"
            ])
        
        # Confianza general
        if 'overall_confidence' in self.stats:
            report_lines.extend([
                f"\n🎯 CONFIANZA GENERAL DEL DIAGNÓSTICO:",
                f"  Nivel: {self.stats['overall_confidence'].upper()}",
                f"  Score: {self.stats.get('overall_confidence_score', 'N/A')}"
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
        logger.error("❌ Error: Debe proporcionar la ruta al archivo de insumos")
        print("\n" + "=" * 70)
        print("USO DEL SCRIPT DE DIAGNÓSTICO DE INSUMOS".center(70))
        print("=" * 70)
        print("\nSintaxis:")
        print("  python diagnose_insumos_file.py <ruta_al_archivo>")
        print("\nEjemplos:")
        print("  python diagnose_insumos_file.py insumos.csv")
        print("  python diagnose_insumos_file.py /ruta/completa/insumos.txt")
        print("\nDescripción:")
        print("  Analiza la estructura jerárquica de un archivo de insumos")
        print("  (grupos y tablas) y genera un reporte detallado con")
        print("  recomendaciones para su procesamiento.")
        print("\nEstructura esperada del archivo:")
        print("  G;<NOMBRE_GRUPO>")
        print("  <ENCABEZADO>")
        print("  <DATOS>")
        print("  <DATOS>")
        print("  ...")
        print("=" * 70 + "\n")
        return 1
    
    file_path = sys.argv[1]
    
    try:
        logger.info("=" * 80)
        logger.info(f"🚀 INICIANDO DIAGNÓSTICO DE INSUMOS".center(80))
        logger.info(f"Archivo: {file_path}".center(80))
        logger.info("=" * 80)
        
        diagnostic = InsumosFileDiagnostic(file_path)
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