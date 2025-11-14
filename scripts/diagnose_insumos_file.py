# scripts/diagnose_insumos_file.py

"""
Herramienta avanzada de diagnóstico para archivos de insumos jerárquicos.

Detecta automáticamente:
- Encoding con chardet
- Dialecto CSV con csv.Sniffer
- Estructura de grupos (G;NOMBRE)
- Encabezados de tablas
- Distribución de columnas por grupo
- Inconsistencias estructurales

Autor: Ingeniero Senior
Versión: 2.0
"""

import csv
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Detección automática de encoding
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logging.warning(
        "⚠️ chardet no disponible. Instalar con: pip install chardet"
    )

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPCIONES PERSONALIZADAS
# ============================================================================

class DiagnosticError(Exception):
    """Excepción base para errores de diagnóstico."""
    pass


class FileReadError(DiagnosticError):
    """Error al leer el archivo."""
    pass


class EncodingDetectionError(DiagnosticError):
    """Error al detectar el encoding."""
    pass


# ============================================================================
# ENUMS
# ============================================================================

class ConfidenceLevel(Enum):
    """Niveles de confianza para detecciones."""
    HIGH = "alta"
    MEDIUM = "media"
    LOW = "baja"
    NONE = "ninguna"


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class GroupInfo:
    """Información de un grupo detectado."""
    name: str
    line_num: int
    original_name: str = ""  # Nombre sin normalizar
    header_line: Optional[int] = None
    header_content: str = ""
    data_lines: int = 0
    column_counts: Counter = field(default_factory=Counter)
    samples: List[str] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM

    def __post_init__(self):
        """Inicialización post-creación."""
        if not self.original_name:
            self.original_name = self.name

    @property
    def dominant_column_count(self) -> Optional[int]:
        """Número de columnas más frecuente."""
        if not self.column_counts:
            return None
        return self.column_counts.most_common(1)[0][0]

    @property
    def column_consistency(self) -> float:
        """Porcentaje de filas con el número de columnas dominante."""
        if self.data_lines == 0:
            return 0.0
        dominant = self.dominant_column_count
        if dominant is None:
            return 0.0
        return self.column_counts[dominant] / self.data_lines

    @property
    def has_header(self) -> bool:
        """Indica si tiene encabezado detectado."""
        return self.header_line is not None

    @property
    def has_data(self) -> bool:
        """Indica si tiene datos."""
        return self.data_lines > 0

    @property
    def is_complete(self) -> bool:
        """Indica si está completo (header + data)."""
        return self.has_header and self.has_data


@dataclass
class FileStatistics:
    """Estadísticas del archivo analizado."""
    # Básicas
    file_size: int = 0
    total_lines: int = 0
    lines_analyzed: int = 0
    empty_lines: int = 0
    non_empty_lines: int = 0
    comment_lines: int = 0

    # Encoding y formato
    encoding: str = "unknown"
    encoding_confidence: float = 0.0
    encoding_method: str = "unknown"

    # CSV
    csv_delimiter: Optional[str] = None
    csv_quotechar: Optional[str] = None
    csv_dialect_detected: bool = False

    # Grupos
    groups_detected: int = 0
    groups_with_headers: int = 0
    groups_with_data: int = 0
    groups_complete: int = 0
    duplicate_groups: int = 0

    # Columnas
    dominant_column_count: Optional[int] = None
    column_consistency: float = 0.0
    column_distribution: Counter = field(default_factory=Counter)

    # Integridad
    integrity_issues: int = 0
    overall_confidence: str = "ninguna"
    overall_confidence_score: float = 0.0

    # Flags
    truncated_analysis: bool = False


@dataclass
class DiagnosticResult:
    """Resultado completo del diagnóstico."""
    success: bool
    file_path: str
    stats: FileStatistics
    groups: List[GroupInfo] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class InsumosFileDiagnostic:
    """
    Diagnosticador avanzado para archivos de insumos jerárquicos.

    Estructura esperada:
        G;<NOMBRE_GRUPO>
        <ENCABEZADO>
        <DATO1>
        <DATO2>
        G;<OTRO_GRUPO>
        ...

    Uso:
        diagnostic = InsumosFileDiagnostic("insumos.csv")
        result = diagnostic.diagnose()
        if result.success:
            print(result.stats)
    """

    # Configuración
    FALLBACK_ENCODINGS = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    GROUP_MARKERS = ['G', 'GRUPO']  # Prefijos válidos para grupos

    # Palabras clave para encabezados
    HEADER_KEYWORDS = [
        'CODIGO', 'COD', 'DESCRIPCION', 'DESC', 'UND', 'UNIDAD',
        'VALOR', 'VR', 'UNITARIO', 'PRECIO', 'COSTO', 'TOTAL',
        'CANTIDAD', 'CANT', 'PARCIAL', 'SUBTOTAL'
    ]

    # Límites
    CHARDET_SAMPLE_SIZE = 50_000
    CSV_SNIFFER_SAMPLE_SIZE = 8_192
    MAX_LINES_TO_ANALYZE = 2_000
    MAX_SAMPLES_PER_GROUP = 3
    MIN_HEADER_KEYWORD_MATCHES = 2
    COLUMN_CONSISTENCY_THRESHOLD = 0.85
    MAX_GROUPS_IN_REPORT = 10

    # Patrones regex compilados
    PATTERN_COMMENT = re.compile(r'^\s*[#\/\*\-\'%]')
    PATTERN_MULTIPLE_SPACES = re.compile(r'\s{2,}')

    def __init__(self, file_path: Union[str, Path]):
        """
        Inicializa el diagnóstico.

        Args:
            file_path: Ruta al archivo de insumos

        Raises:
            ValueError: Si el archivo no existe o está vacío
        """
        self.file_path = Path(file_path).resolve()

        # Validaciones
        if not self.file_path.exists():
            raise ValueError(f"Archivo no encontrado: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"La ruta no es un archivo: {self.file_path}")

        file_size = self.file_path.stat().st_size
        if file_size == 0:
            raise ValueError("El archivo está vacío")

        # Estado interno
        self.stats = FileStatistics(file_size=file_size)
        self.groups: List[GroupInfo] = []
        self.recommendations: List[str] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []

        # Estado temporal durante análisis
        self._current_group: Optional[GroupInfo] = None
        self._group_names_seen: Set[str] = set()
        self._encoding: Optional[str] = None
        self._separator: Optional[str] = None

        logger.info(f"📂 Archivo: {self.file_path.name} ({self._human_size(file_size)})")

    # ========================================================================
    # MÉTODO PRINCIPAL
    # ========================================================================

    def diagnose(self) -> DiagnosticResult:
        """
        Ejecuta diagnóstico completo.

        Returns:
            DiagnosticResult con toda la información
        """
        logger.info("🔍 Iniciando diagnóstico de insumos...")

        try:
            # 1. Detectar encoding
            encoding_info = self._detect_encoding()
            if not encoding_info:
                return self._build_failure("No se pudo detectar encoding")

            self._encoding = encoding_info[0]
            self.stats.encoding = encoding_info[0]
            self.stats.encoding_confidence = encoding_info[1]

            # 2. Leer contenido
            content = self._read_file(self._encoding)
            if not content:
                return self._build_failure("No se pudo leer el archivo")

            lines = content.splitlines()
            self.stats.total_lines = len(lines)

            # 3. Limitar para archivos grandes
            lines_to_analyze = self._limit_lines(lines)

            # 4. Detectar dialecto CSV
            self._detect_csv_dialect(content)

            # 5. Análisis estructural jerárquico
            self._analyze_hierarchical_structure(lines_to_analyze)

            # 6. Análisis de distribución de columnas
            self._analyze_column_distribution()

            # 7. Validar integridad
            self._validate_integrity()

            # 8. Generar recomendaciones
            self._generate_recommendations()

            # 9. Calcular confianza general
            self._calculate_overall_confidence()

            # 10. Reporte en logging
            self._log_report()

            logger.info("✅ Diagnóstico completado exitosamente")

            return DiagnosticResult(
                success=True,
                file_path=str(self.file_path),
                stats=self.stats,
                groups=self.groups,
                recommendations=self.recommendations,
                errors=self.errors,
                warnings=self.warnings
            )

        except Exception as e:
            logger.error(f"❌ Error durante diagnóstico: {e}", exc_info=True)
            return self._build_failure(str(e))

    # ========================================================================
    # DETECCIÓN DE ENCODING
    # ========================================================================

    def _detect_encoding(self) -> Optional[Tuple[str, float]]:
        """
        Detecta encoding con chardet o fallback.

        Returns:
            (encoding, confianza) o None
        """
        if CHARDET_AVAILABLE:
            result = self._detect_encoding_with_chardet()
            if result:
                self.stats.encoding_method = "chardet"
                return result

        logger.warning("⚠️ Usando detección manual de encoding")
        result = self._detect_encoding_fallback()
        if result:
            self.stats.encoding_method = "manual"
        return result

    def _detect_encoding_with_chardet(self) -> Optional[Tuple[str, float]]:
        """Detecta encoding con chardet."""
        try:
            with open(self.file_path, 'rb') as f:
                sample = f.read(self.CHARDET_SAMPLE_SIZE)

            detection = chardet.detect(sample)
            encoding = detection.get('encoding')
            confidence = detection.get('confidence', 0.0)

            if not encoding:
                return None

            logger.info(
                f"✅ Encoding: {encoding} (confianza: {confidence:.1%})"
            )

            if confidence < 0.7:
                self.warnings.append(
                    f"Baja confianza en encoding ({confidence:.1%})"
                )

            return (encoding, confidence)

        except Exception as e:
            logger.error(f"Error en chardet: {e}")
            return None

    def _detect_encoding_fallback(self) -> Optional[Tuple[str, float]]:
        """Fallback manual para encoding."""
        for encoding in self.FALLBACK_ENCODINGS:
            try:
                with open(self.file_path, 'r', encoding=encoding) as f:
                    f.read(1024)
                logger.info(f"✅ Encoding (manual): {encoding}")
                return (encoding, 0.5)
            except (UnicodeDecodeError, LookupError):
                continue

        self.errors.append("No se pudo determinar encoding")
        return None

    def _read_file(self, encoding: str) -> Optional[str]:
        """Lee el archivo completo."""
        try:
            return self.file_path.read_text(
                encoding=encoding,
                errors='replace'
            )
        except Exception as e:
            self.errors.append(f"Error al leer archivo: {e}")
            return None

    # ========================================================================
    # DETECCIÓN DE DIALECTO CSV
    # ========================================================================

    def _detect_csv_dialect(self, content: str) -> None:
        """
        Detecta dialecto CSV con csv.Sniffer.

        Args:
            content: Contenido del archivo
        """
        try:
            sample = content[:self.CSV_SNIFFER_SAMPLE_SIZE]
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)

            self._separator = dialect.delimiter
            self.stats.csv_delimiter = dialect.delimiter
            self.stats.csv_quotechar = dialect.quotechar
            self.stats.csv_dialect_detected = True

            logger.info(
                f"✅ Dialecto CSV detectado: delimitador='{dialect.delimiter}'"
            )

        except csv.Error:
            logger.warning("⚠️ csv.Sniffer falló, usando detección manual")
            self._detect_separator_fallback(content.splitlines()[:100])
        except Exception as e:
            logger.warning(f"⚠️ Error en detección de dialecto: {e}")
            self._detect_separator_fallback(content.splitlines()[:100])

    def _detect_separator_fallback(self, lines: List[str]) -> None:
        """Fallback manual para detectar separador."""
        separators = [';', ',', '\t', '|']
        scores = Counter()

        clean_lines = [
            line.strip() for line in lines
            if line.strip() and not self._is_comment(line.strip())
        ]

        for line in clean_lines:
            for sep in separators:
                count = line.count(sep)
                if count > 0:
                    scores[sep] += count

        if scores:
            best = scores.most_common(1)[0][0]
            self._separator = best
            self.stats.csv_delimiter = best
            logger.info(f"✅ Separador (fallback): '{best}'")
        else:
            self._separator = ';'
            self.stats.csv_delimiter = ';'
            logger.warning("⚠️ Usando separador por defecto: ';'")

    # ========================================================================
    # ANÁLISIS ESTRUCTURAL JERÁRQUICO
    # ========================================================================

    def _analyze_hierarchical_structure(self, lines: List[str]) -> None:
        """
        Analiza estructura jerárquica de grupos.

        Args:
            lines: Líneas a analizar
        """
        if not self._separator:
            logger.error("❌ No hay separador detectado")
            return

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Clasificar línea
            if not stripped:
                self.stats.empty_lines += 1
                continue

            self.stats.non_empty_lines += 1

            if self._is_comment(stripped):
                self.stats.comment_lines += 1
                continue

            # Detectar grupo
            is_group, group_name = self._is_group_line(stripped)

            if is_group and group_name:
                self._finalize_current_group()
                self._start_new_group(group_name, line_num, stripped)
                continue

            # Procesar dentro de grupo
            if self._current_group:
                self._process_line_in_group(stripped, line_num)

        # Finalizar último grupo
        self._finalize_current_group()

    def _is_comment(self, line: str) -> bool:
        """Determina si es comentario."""
        return bool(self.PATTERN_COMMENT.match(line))

    def _is_group_line(self, line: str) -> Tuple[bool, Optional[str]]:
        """
        Determina si una línea es el inicio de un grupo usando una expresión
        regular flexible.

        Args:
        line (str): Línea a evaluar (puede tener espacios iniciales y no estar normalizada)

        Returns:
        Tuple[bool, Optional[str]]: (es_grupo, nombre_grupo)
        """
        if not self._separator:
            return False, None

        # Patrón de Regex:
        # ^\s* -> Permite espacios en blanco al inicio (opcional).
        # (G\d*) -> Busca una 'G' seguida de cero o más dígitos (G, G1, G22, etc.).
        # \s* -> Permite espacios después del identificador.
        # {re.escape(self._separator)} -> Usa el separador detectado de forma segura.
        # (.*) -> Captura el resto de la línea como el nombre del grupo.
        pattern = re.compile(
            rf"^\s*(G\d*)\s*{re.escape(self._separator)}(.*)",
            re.IGNORECASE
        )

        match = pattern.match(line)

        if match:
            # El grupo 2 de la captura es el nombre del grupo.
            group_name_raw = match.group(2).strip()

            # El nombre del grupo puede estar seguido de más separadores y datos.
            # Lo limpiamos quedándonos solo con la primera parte.
            group_name = group_name_raw.split(self._separator)[0].strip()

            if group_name:
                return True, group_name

        return False, None

    def _start_new_group(
        self,
        group_name: str,
        line_num: int,
        original_line: str
    ) -> None:
        """Inicia un nuevo grupo."""
        # Normalizar nombre
        normalized_name = self._normalize_group_name(group_name)

        # Manejar duplicados
        if normalized_name in self._group_names_seen:
            self.warnings.append(
                f"Grupo duplicado: '{normalized_name}' en línea {line_num}"
            )
            self.stats.duplicate_groups += 1

            # Hacer nombre único
            counter = 2
            base_name = normalized_name
            while normalized_name in self._group_names_seen:
                normalized_name = f"{base_name} ({counter})"
                counter += 1

        # Crear grupo
        self._current_group = GroupInfo(
            name=normalized_name,
            original_name=group_name,
            line_num=line_num
        )

        self.groups.append(self._current_group)
        self._group_names_seen.add(normalized_name)
        self.stats.groups_detected += 1

        logger.debug(f"📦 Grupo: '{normalized_name}' (línea {line_num})")

    def _normalize_group_name(self, name: str) -> str:
        """Normaliza nombre de grupo para comparación."""
        # Convertir a mayúsculas
        normalized = name.upper().strip()

        # Eliminar acentos comunes
        replacements = {
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'Ñ': 'N', 'Ü': 'U'
        }
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        # Normalizar espacios
        normalized = self.PATTERN_MULTIPLE_SPACES.sub(' ', normalized)

        return normalized.strip()

    def _finalize_current_group(self) -> None:
        """Finaliza procesamiento del grupo actual."""
        if not self._current_group:
            return

        # Actualizar estadísticas
        if self._current_group.has_header:
            self.stats.groups_with_headers += 1

        if self._current_group.has_data:
            self.stats.groups_with_data += 1

        if self._current_group.is_complete:
            self.stats.groups_complete += 1

        logger.debug(
            f"   └─ Datos: {self._current_group.data_lines} líneas"
        )

    def _process_line_in_group(self, line: str, line_num: int) -> None:
        """Procesa línea dentro de un grupo."""
        if not self._current_group or not self._separator:
            return

        # Si no tiene header, intentar detectar
        if not self._current_group.has_header:
            if self._is_header_line(line):
                self._current_group.header_line = line_num
                self._current_group.header_content = line
                logger.debug(f"   📋 Header en línea {line_num}")
                return

        # Procesar como dato (usar csv.reader para robustez)
        self._process_data_line(line, line_num)

    def _is_header_line(self, line: str) -> bool:
        """Determina si es encabezado."""
        normalized = line.upper()

        # Contar keywords
        matches = sum(
            1 for keyword in self.HEADER_KEYWORDS
            if keyword in normalized
        )

        return matches >= self.MIN_HEADER_KEYWORD_MATCHES

    def _process_data_line(self, line: str, line_num: int) -> None:
        """
        Procesa línea de datos usando csv.reader para robustez.

        Args:
            line: Línea a procesar
            line_num: Número de línea
        """
        if not self._current_group or not self._separator:
            return

        try:
            # Usar csv.reader para manejar quotes, escapes, etc.
            reader = csv.reader(
                [line],
                delimiter=self._separator,
                quotechar=self.stats.csv_quotechar or '"'
            )

            row = next(reader)
            num_cols = len(row)

            # Actualizar grupo
            self._current_group.data_lines += 1
            self._current_group.column_counts[num_cols] += 1

            # Guardar muestra
            if len(self._current_group.samples) < self.MAX_SAMPLES_PER_GROUP:
                self._current_group.samples.append(line)

        except Exception as e:
            logger.debug(f"Error procesando línea {line_num}: {e}")
            # Fallback: split manual
            parts = line.split(self._separator)
            num_cols = len(parts)
            self._current_group.data_lines += 1
            self._current_group.column_counts[num_cols] += 1

    # ========================================================================
    # ANÁLISIS DE COLUMNAS
    # ========================================================================

    def _analyze_column_distribution(self) -> None:
        """Analiza distribución global de columnas."""
        # Agregar todas las columnas de todos los grupos
        total_counts = Counter()

        for group in self.groups:
            total_counts.update(group.column_counts)

        self.stats.column_distribution = total_counts

        if total_counts:
            # Columnas dominantes
            most_common = total_counts.most_common(1)[0]
            self.stats.dominant_column_count = most_common[0]

            # Consistencia global
            total_lines = sum(total_counts.values())
            if total_lines > 0:
                self.stats.column_consistency = most_common[1] / total_lines

    # ========================================================================
    # VALIDACIÓN E INTEGRIDAD
    # ========================================================================

    def _validate_integrity(self) -> None:
        """Valida integridad de grupos."""
        issues = []

        for group in self.groups:
            # Sin header
            if not group.has_header:
                issues.append(f"Grupo '{group.name}' sin encabezado")
                group.confidence = ConfidenceLevel.LOW

            # Sin datos
            if not group.has_data:
                issues.append(f"Grupo '{group.name}' sin datos")
                group.confidence = ConfidenceLevel.LOW

            # Columnas inconsistentes
            if group.has_data:
                consistency = group.column_consistency
                if consistency < self.COLUMN_CONSISTENCY_THRESHOLD:
                    issues.append(
                        f"Grupo '{group.name}' columnas inconsistentes "
                        f"({consistency:.1%})"
                    )
                    if group.confidence == ConfidenceLevel.HIGH:
                        group.confidence = ConfidenceLevel.MEDIUM
                elif consistency >= 0.95:
                    group.confidence = ConfidenceLevel.HIGH

        self.stats.integrity_issues = len(issues)
        self.warnings.extend(issues)

        if issues:
            logger.warning(f"⚠️ {len(issues)} problemas de integridad")

    # ========================================================================
    # RECOMENDACIONES
    # ========================================================================

    def _generate_recommendations(self) -> None:
        """Genera recomendaciones inteligentes."""

        # Sin grupos detectados
        if self.stats.groups_detected == 0:
            self.recommendations.extend([
                "❌ CRÍTICO: No se detectaron grupos",
                "Verificar que el formato sea: G;<NOMBRE>",
                f"Confirmar separador: '{self._separator}'",
                "Revisar primeras líneas del archivo manualmente"
            ])
            return

        # Encoding
        if self.stats.encoding_confidence < 0.7:
            self.recommendations.append(
                f"⚠️ Verificar encoding ({self.stats.encoding_confidence:.1%} confianza)"
            )

        # Lectura recomendada
        if self._separator and self._encoding:
            self.recommendations.append(
                f"✅ Leer con: sep='{self._separator}', encoding='{self._encoding}'"
            )

        # Grupos incompletos
        complete_ratio = (
            self.stats.groups_complete / self.stats.groups_detected
            if self.stats.groups_detected > 0 else 0
        )

        if complete_ratio < 0.5:
            self.recommendations.append(
                f"⚠️ Solo {complete_ratio:.0%} de grupos están completos"
            )
        elif complete_ratio >= 0.9:
            self.recommendations.append(
                f"✅ {complete_ratio:.0%} de grupos están completos"
            )

        # Columnas irregulares
        if self.stats.column_consistency < self.COLUMN_CONSISTENCY_THRESHOLD:
            self.recommendations.extend([
                (
                    "⚠️ Columnas irregulares (consistencia: "
                    f"{self.stats.column_consistency:.1%})"
                ),
                "Usar on_bad_lines='warn' en pandas",
                "Validar número de columnas por línea",
            ])

        # Estrategia de procesamiento
        if self.stats.groups_complete > 0:
            self.recommendations.extend([
                "📋 Estrategia sugerida:",
                "1. Identificar líneas que empiecen con 'G;'",
                "2. Siguiente línea = encabezado",
                "3. Líneas subsiguientes = datos",
                "4. Procesar cada grupo como DataFrame independiente"
            ])

    def _calculate_overall_confidence(self) -> None:
        """Calcula confianza general."""
        factors = []

        # Factor 1: Grupos detectados
        if self.stats.groups_detected > 0:
            factors.append(1.0)
        else:
            factors.append(0.0)

        # Factor 2: Grupos completos
        if self.stats.groups_detected > 0:
            complete_ratio = (
                self.stats.groups_complete / self.stats.groups_detected
            )
            factors.append(complete_ratio)

        # Factor 3: Consistencia de columnas
        factors.append(self.stats.column_consistency)

        # Factor 4: Confianza de encoding
        if self.stats.encoding_confidence > 0:
            factors.append(self.stats.encoding_confidence)

        # Calcular promedio
        if factors:
            score = sum(factors) / len(factors)
            self.stats.overall_confidence_score = score

            if score >= 0.9:
                self.stats.overall_confidence = ConfidenceLevel.HIGH.value
            elif score >= 0.7:
                self.stats.overall_confidence = ConfidenceLevel.MEDIUM.value
            elif score >= 0.5:
                self.stats.overall_confidence = ConfidenceLevel.LOW.value
            else:
                self.stats.overall_confidence = ConfidenceLevel.NONE.value

    # ========================================================================
    # REPORTES Y UTILIDADES
    # ========================================================================

    def _log_report(self) -> None:
        """Genera reporte en logging."""
        lines = [
            "\n" + "=" * 90,
            "📊 REPORTE - DIAGNÓSTICO DE INSUMOS JERÁRQUICOS".center(90),
            "=" * 90,

            "\n📁 ARCHIVO:",
            f"  Ruta: {self.file_path}",
            f"  Tamaño: {self._human_size(self.stats.file_size)}",
            f"  Líneas: {self.stats.total_lines:,}",

            "\n🔤 FORMATO:",
            f"  Encoding: {self.stats.encoding} ({self.stats.encoding_confidence:.1%})",
            f"  Separador: '{self.stats.csv_delimiter}'",
            f"  Dialecto detectado: {'Sí' if self.stats.csv_dialect_detected else 'No'}",

            "\n📦 GRUPOS:",
            f"  Total detectados: {self.stats.groups_detected}",
            f"  Con encabezado: {self.stats.groups_with_headers}",
            f"  Con datos: {self.stats.groups_with_data}",
            f"  Completos: {self.stats.groups_complete}",
        ]

        if self.stats.duplicate_groups > 0:
            lines.append(f"  ⚠️ Duplicados: {self.stats.duplicate_groups}")

        # Detalle de grupos
        if self.groups:
            lines.append("\n📋 DETALLE DE GRUPOS:")
            for i, group in enumerate(self.groups[:self.MAX_GROUPS_IN_REPORT], 1):
                status = "✅" if group.is_complete else "⚠️"
                lines.extend([
                    f"\n  {i}. {status} {group.name}",
                    f"     Línea: {group.line_num}",
                    (
                        "     Header: "
                        f"{'Línea ' + str(group.header_line) if group.has_header else 'No'}"
                    ),
                    f"     Datos: {group.data_lines} líneas",
                ])

                if group.has_data:
                    lines.append(
                        f"     Columnas: {group.dominant_column_count} "
                        f"({group.column_consistency:.1%} consistencia)"
                    )

            if len(self.groups) > self.MAX_GROUPS_IN_REPORT:
                lines.append(
                    f"\n  ... y {len(self.groups) - self.MAX_GROUPS_IN_REPORT} más"
                )

        # Columnas
        if self.stats.column_distribution:
            lines.append("\n📊 DISTRIBUCIÓN DE COLUMNAS:")
            for cols, count in self.stats.column_distribution.most_common(5):
                marker = "→" if cols == self.stats.dominant_column_count else " "
                lines.append(f"  {marker} {cols} columnas: {count} líneas")

        # Recomendaciones
        if self.recommendations:
            lines.append("\n💡 RECOMENDACIONES:")
            for rec in self.recommendations:
                lines.append(f"  {rec}")

        # Confianza
        lines.extend([
            "\n🎯 CONFIANZA GENERAL:",
            f"  Nivel: {self.stats.overall_confidence.upper()}",
            f"  Score: {self.stats.overall_confidence_score:.1%}",
            "\n" + "=" * 90 + "\n"
        ])

        for line in lines:
            logger.info(line)

    def _limit_lines(self, lines: List[str]) -> List[str]:
        """Limita líneas para archivos grandes."""
        if len(lines) > self.MAX_LINES_TO_ANALYZE:
            logger.warning(
                f"⚠️ Archivo grande. Analizando primeras "
                f"{self.MAX_LINES_TO_ANALYZE:,} líneas"
            )
            self.stats.truncated_analysis = True
            self.stats.lines_analyzed = self.MAX_LINES_TO_ANALYZE
            return lines[:self.MAX_LINES_TO_ANALYZE]

        self.stats.lines_analyzed = len(lines)
        return lines

    def _human_size(self, size_bytes: int) -> str:
        """Convierte bytes a formato legible."""
        if size_bytes == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB"]
        size = float(size_bytes)
        unit_idx = 0

        while size >= 1024.0 and unit_idx < len(units) - 1:
            size /= 1024.0
            unit_idx += 1

        if size < 10:
            return f"{size:.2f} {units[unit_idx]}"
        return f"{size:.1f} {units[unit_idx]}"

    def _build_failure(self, reason: str) -> DiagnosticResult:
        """Construye resultado de fallo."""
        self.errors.append(reason)
        logger.error(f"❌ {reason}")

        return DiagnosticResult(
            success=False,
            file_path=str(self.file_path),
            stats=self.stats,
            errors=self.errors,
            warnings=self.warnings
        )

    def to_dict(self) -> Dict[str, Any]:
        """Exporta resultado como diccionario."""
        return {
            'file_path': str(self.file_path),
            'stats': asdict(self.stats),
            'groups': [
                {
                    'name': g.name,
                    'original_name': g.original_name,
                    'line_num': g.line_num,
                    'header_line': g.header_line,
                    'data_lines': g.data_lines,
                    'dominant_columns': g.dominant_column_count,
                    'column_consistency': f"{g.column_consistency:.1%}",
                    'confidence': g.confidence.value,
                    'is_complete': g.is_complete
                }
                for g in self.groups
            ],
            'recommendations': self.recommendations,
            'warnings': self.warnings,
            'errors': self.errors
        }


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    """Función principal CLI."""
    if len(sys.argv) < 2:
        print("❌ Uso: python diagnose_insumos_file.py <archivo>")
        print("📖 Ejemplo: python diagnose_insumos_file.py insumos.csv")
        print("\n💡 Instalar: pip install chardet")
        return 1

    file_path = sys.argv[1]

    try:
        diagnostic = InsumosFileDiagnostic(file_path)
        result = diagnostic.diagnose()

        if not result.success:
            logger.error("❌ Diagnóstico falló")
            for error in result.errors:
                logger.error(f"  - {error}")
            return 1

        # Exportar JSON si se solicita
        if len(sys.argv) > 2 and sys.argv[2] == '--json':
            import json
            output = diagnostic.to_dict()
            print(json.dumps(output, indent=2, ensure_ascii=False))

        return 0

    except ValueError as e:
        logger.error(f"❌ Error de validación: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Interrumpido por usuario")
        return 130
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
