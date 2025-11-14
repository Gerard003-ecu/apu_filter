# scripts/diagnose_presupuesto_file.py

"""
Herramienta avanzada de diagnóstico para archivos de presupuesto.

Detecta automáticamente:
- Encoding con chardet
- Dialecto CSV con csv.Sniffer
- Fila de encabezado por keywords
- Distribución de columnas
- Inconsistencias estructurales

Autor: Ingeniero Senior
Versión: 2.0
"""

import csv
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Detección automática de encoding
try:
    import chardet

    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logging.warning("⚠️ chardet no disponible. Instalar con: pip install chardet")

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
class HeaderCandidate:
    """Candidato a encabezado detectado."""

    line_num: int
    content: str
    matches: List[str]
    column_count: int
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM

    @property
    def match_count(self) -> int:
        """Número de keywords coincidentes."""
        return len(self.matches)

    @property
    def score(self) -> float:
        """Score ponderado para comparación."""
        # Más matches = mejor, líneas tempranas = mejor
        return self.match_count * 10 - (self.line_num * 0.1)


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

    # Encabezado
    header_line: Optional[int] = None
    header_column_count: Optional[int] = None
    data_start_line: Optional[int] = None

    # Columnas
    dominant_column_count: Optional[int] = None
    column_consistency: float = 0.0
    column_distribution: Counter = field(default_factory=Counter)

    # Resumen
    summary_lines_ignored: int = 0

    # Confianza
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
    header_candidate: Optional[HeaderCandidate] = None
    recommendations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    samples: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================


class PresupuestoFileDiagnostic:
    """
    Diagnosticador avanzado para archivos de presupuesto.

    Estructura esperada:
        <METADATA_OPCIONAL>
        <ENCABEZADO>
        <DATO1>
        <DATO2>
        ...
        <TOTAL_OPCIONAL>

    Uso:
        diagnostic = PresupuestoFileDiagnostic("presupuesto.csv")
        result = diagnostic.diagnose()
        if result.success:
            print(f"Header en línea: {result.header_candidate.line_num}")
    """

    # Configuración
    FALLBACK_ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]

    # Keywords para encabezados
    HEADER_KEYWORDS = [
        "ITEM",
        "DESCRIPCION",
        "CANT",
        "CANTIDAD",
        "UNIDAD",
        "UND",
        "VR UNITARIO",
        "VALOR UNITARIO",
        "PRECIO",
        "TOTAL",
        "IMPORTE",
        "PU",
        "P U",
        "SUBTOTAL",
        "PARCIAL",
        "COSTO",
        "VR",
        "VALOR",
    ]

    # Límites
    CHARDET_SAMPLE_SIZE = 50_000
    CSV_SNIFFER_SAMPLE_SIZE = 8_192
    MAX_LINES_TO_ANALYZE = 1_000
    MAX_SAMPLES = 20
    MIN_HEADER_KEYWORD_MATCHES = 2
    COLUMN_CONSISTENCY_THRESHOLD = 0.85
    MAX_HEADER_SEARCH_LINES = 50  # Buscar header en primeras N líneas

    # Patrones regex compilados
    PATTERN_COMMENT = re.compile(r"^\s*[#\/\*\-\'%]")
    PATTERN_TOTAL_LINE = re.compile(r"^\s*TOTAL", re.IGNORECASE)
    PATTERN_MULTIPLE_SPACES = re.compile(r"\s{2,}")

    def __init__(self, file_path: Union[str, Path]):
        """
        Inicializa el diagnóstico.

        Args:
            file_path: Ruta al archivo de presupuesto

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
        self.header_candidate: Optional[HeaderCandidate] = None
        self.recommendations: List[str] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.samples: List[Dict[str, Any]] = []

        # Estado temporal
        self._encoding: Optional[str] = None
        self._separator: Optional[str] = None
        self._column_stats: Dict[int, Counter] = defaultdict(Counter)

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
        logger.info("🔍 Iniciando diagnóstico de presupuesto...")

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

            # 5. Análisis estructural
            self._analyze_structure(lines_to_analyze)

            # 6. Análisis de columnas
            self._analyze_column_distribution()

            # 7. Validar consistencia
            self._validate_structure()

            # 8. Generar recomendaciones
            self._generate_recommendations()

            # 9. Calcular confianza general
            self._calculate_overall_confidence()

            # 10. Reporte
            self._log_report()

            logger.info("✅ Diagnóstico completado exitosamente")

            return DiagnosticResult(
                success=True,
                file_path=str(self.file_path),
                stats=self.stats,
                header_candidate=self.header_candidate,
                recommendations=self.recommendations,
                errors=self.errors,
                warnings=self.warnings,
                samples=self.samples,
            )

        except Exception as e:
            logger.error(f"❌ Error durante diagnóstico: {e}", exc_info=True)
            return self._build_failure(str(e))

    # ========================================================================
    # DETECCIÓN DE ENCODING
    # ========================================================================

    def _detect_encoding(self) -> Optional[Tuple[str, float]]:
        """Detecta encoding con chardet o fallback."""
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
            with open(self.file_path, "rb") as f:
                sample = f.read(self.CHARDET_SAMPLE_SIZE)

            detection = chardet.detect(sample)
            encoding = detection.get("encoding")
            confidence = detection.get("confidence", 0.0)

            if not encoding:
                return None

            logger.info(f"✅ Encoding: {encoding} (confianza: {confidence:.1%})")

            if confidence < 0.7:
                self.warnings.append(f"Baja confianza en encoding ({confidence:.1%})")

            return (encoding, confidence)

        except Exception as e:
            logger.error(f"Error en chardet: {e}")
            return None

    def _detect_encoding_fallback(self) -> Optional[Tuple[str, float]]:
        """Fallback manual para encoding."""
        for encoding in self.FALLBACK_ENCODINGS:
            try:
                with open(self.file_path, "r", encoding=encoding) as f:
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
            return self.file_path.read_text(encoding=encoding, errors="replace")
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
            sample = content[: self.CSV_SNIFFER_SAMPLE_SIZE]
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)

            self._separator = dialect.delimiter
            self.stats.csv_delimiter = dialect.delimiter
            self.stats.csv_quotechar = dialect.quotechar
            self.stats.csv_dialect_detected = True

            logger.info(f"✅ Dialecto CSV: delimitador='{dialect.delimiter}'")

        except csv.Error:
            logger.warning("⚠️ csv.Sniffer falló, usando detección manual")
            self._detect_separator_fallback(content.splitlines()[:100])
        except Exception as e:
            logger.warning(f"⚠️ Error en detección de dialecto: {e}")
            self._detect_separator_fallback(content.splitlines()[:100])

    def _detect_separator_fallback(self, lines: List[str]) -> None:
        """Fallback manual para detectar separador."""
        separators = [";", ",", "\t", "|"]
        scores = Counter()

        clean_lines = [
            line.strip()
            for line in lines
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
            self._separator = ";"
            self.stats.csv_delimiter = ";"
            logger.warning("⚠️ Usando separador por defecto: ';'")

    # ========================================================================
    # ANÁLISIS ESTRUCTURAL
    # ========================================================================

    def _analyze_structure(self, lines: List[str]) -> None:
        """
        Analiza estructura del archivo.

        Args:
            lines: Líneas a analizar
        """
        if not self._separator:
            logger.error("❌ No hay separador detectado")
            return

        # Fase 1: Buscar encabezado
        header_candidates = self._find_header_candidates(lines)

        if header_candidates:
            # Seleccionar mejor candidato
            self.header_candidate = max(header_candidates, key=lambda x: x.score)
            self.stats.header_line = self.header_candidate.line_num
            self.stats.header_column_count = self.header_candidate.column_count

            logger.info(
                f"✅ Header detectado: línea {self.header_candidate.line_num} "
                f"({self.header_candidate.match_count} keywords)"
            )

            # Fase 2: Procesar datos después del header
            self._process_data_lines(lines, self.header_candidate.line_num)
        else:
            logger.warning("⚠️ No se detectó encabezado")
            self.warnings.append("No se pudo detectar fila de encabezado")

            # Procesar todas las líneas como datos potenciales
            self._process_data_lines(lines, 0)

    def _find_header_candidates(self, lines: List[str]) -> List[HeaderCandidate]:
        """
        Busca candidatos a encabezado en las primeras líneas.

        Returns:
            Lista de candidatos ordenados por score
        """
        candidates = []
        search_limit = min(len(lines), self.MAX_HEADER_SEARCH_LINES)

        for line_num, line in enumerate(lines[:search_limit], 1):
            stripped = line.strip()

            # Ignorar vacías y comentarios
            if not stripped or self._is_comment(stripped):
                continue

            # Evaluar como candidato
            candidate = self._evaluate_header_candidate(stripped, line_num)
            if candidate:
                candidates.append(candidate)

                # Si encontramos uno muy bueno, podemos parar
                if candidate.match_count >= 5:
                    logger.debug(
                        f"Header fuerte en línea {line_num}: {candidate.match_count} matches"
                    )
                    break

        return candidates

    def _evaluate_header_candidate(
        self, line: str, line_num: int
    ) -> Optional[HeaderCandidate]:
        """
        Evalúa si una línea puede ser encabezado.

        Returns:
            HeaderCandidate si cumple criterios, None en caso contrario
        """
        # Normalizar y buscar keywords
        normalized = self._normalize_text(line)
        matches = [kw for kw in self.HEADER_KEYWORDS if kw in normalized]

        if len(matches) < self.MIN_HEADER_KEYWORD_MATCHES:
            return None

        # Contar columnas usando csv.reader
        try:
            reader = csv.reader(
                [line], delimiter=self._separator, quotechar=self.stats.csv_quotechar or '"'
            )
            row = next(reader)
            column_count = len(row)
        except Exception:
            # Fallback
            column_count = len(line.split(self._separator))

        # Determinar confianza
        if len(matches) >= 5:
            confidence = ConfidenceLevel.HIGH
        elif len(matches) >= 3:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        return HeaderCandidate(
            line_num=line_num,
            content=line,
            matches=matches,
            column_count=column_count,
            confidence=confidence,
        )

    def _process_data_lines(self, lines: List[str], header_line: int) -> None:
        """
        Procesa líneas de datos después del encabezado.

        Args:
            lines: Todas las líneas
            header_line: Número de línea del header (0 si no hay)
        """
        data_started = False

        for line_num, line in enumerate(lines, 1):
            # Saltar hasta después del header
            if line_num <= header_line:
                continue

            stripped = line.strip()

            # Clasificar línea
            if not stripped:
                self.stats.empty_lines += 1
                continue

            self.stats.non_empty_lines += 1

            if self._is_comment(stripped):
                self.stats.comment_lines += 1
                continue

            # Detectar líneas de total/resumen
            if self.PATTERN_TOTAL_LINE.match(stripped):
                self.stats.summary_lines_ignored += 1
                continue

            # Procesar como dato
            self._process_data_line(stripped, line_num)

            if not data_started:
                self.stats.data_start_line = line_num
                data_started = True

    def _process_data_line(self, line: str, line_num: int) -> None:
        """
        Procesa una línea de datos usando csv.reader.

        Args:
            line: Línea a procesar
            line_num: Número de línea
        """
        try:
            # Usar csv.reader para robustez
            reader = csv.reader(
                [line], delimiter=self._separator, quotechar=self.stats.csv_quotechar or '"'
            )

            row = next(reader)
            num_cols = len(row)

            # Actualizar estadísticas
            self._column_stats[num_cols]["count"] += 1

            # Guardar muestra
            if len(self.samples) < self.MAX_SAMPLES:
                self.samples.append(
                    {"line_num": line_num, "content": line, "column_count": num_cols}
                )

        except Exception as e:
            logger.debug(f"Error procesando línea {line_num}: {e}")
            # Fallback: split manual
            parts = line.split(self._separator)
            num_cols = len(parts)
            self._column_stats[num_cols]["count"] += 1

    def _is_comment(self, line: str) -> bool:
        """Determina si es comentario."""
        return bool(self.PATTERN_COMMENT.match(line))

    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparación."""
        # Mayúsculas
        normalized = text.upper()

        # Eliminar acentos
        replacements = {"Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ñ": "N", "Ü": "U"}
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        # Normalizar espacios
        normalized = self.PATTERN_MULTIPLE_SPACES.sub(" ", normalized)

        return normalized.strip()

    # ========================================================================
    # ANÁLISIS DE COLUMNAS
    # ========================================================================

    def _analyze_column_distribution(self) -> None:
        """Analiza distribución de columnas."""
        # Convertir a Counter
        total_counts = Counter()
        for num_cols, stats in self._column_stats.items():
            total_counts[num_cols] = stats["count"]

        self.stats.column_distribution = total_counts

        if total_counts:
            # Columnas dominantes
            most_common = total_counts.most_common(1)[0]
            self.stats.dominant_column_count = most_common[0]

            # Consistencia
            total_lines = sum(total_counts.values())
            if total_lines > 0:
                self.stats.column_consistency = most_common[1] / total_lines

    # ========================================================================
    # VALIDACIÓN
    # ========================================================================

    def _validate_structure(self) -> None:
        """Valida estructura del archivo."""

        # Sin header
        if not self.header_candidate:
            self.warnings.append("No se detectó fila de encabezado")

        # Sin datos
        if not self.stats.data_start_line:
            self.warnings.append("No se detectaron líneas de datos")

        # Columnas inconsistentes
        if self.stats.column_consistency < self.COLUMN_CONSISTENCY_THRESHOLD:
            self.warnings.append(
                f"Columnas inconsistentes ({self.stats.column_consistency:.1%})"
            )

        # Header y datos con diferente número de columnas
        if (
            self.header_candidate
            and self.stats.dominant_column_count
            and self.header_candidate.column_count != self.stats.dominant_column_count
        ):
            self.warnings.append(
                f"Header tiene {self.header_candidate.column_count} columnas "
                f"pero datos tienen {self.stats.dominant_column_count}"
            )

    # ========================================================================
    # RECOMENDACIONES
    # ========================================================================

    def _generate_recommendations(self) -> None:
        """Genera recomendaciones inteligentes."""

        # Encoding
        if self.stats.encoding_confidence < 0.7:
            self.recommendations.append(
                f"⚠️ Verificar encoding ({self.stats.encoding_confidence:.1%} confianza)"
            )

        # Lectura con pandas
        if self.header_candidate and self._separator and self._encoding:
            pandas_header = self.header_candidate.line_num - 1
            self.recommendations.append(
                f"✅ Leer con pandas: header={pandas_header}, "
                f"sep='{self._separator}', encoding='{self._encoding}'"
            )

            # Advertencia si header no está al inicio
            if pandas_header > 5:
                self.recommendations.append(
                    f"⚠️ Header en línea {self.header_candidate.line_num}. "
                    f"Considerar skiprows para omitir metadata inicial"
                )
        elif not self.header_candidate:
            self.recommendations.append(
                "⚠️ Sin header detectado. Usar header=None y names=[...] en pandas"
            )

        # Consistencia de columnas
        consistency = self.stats.column_consistency
        if consistency < self.COLUMN_CONSISTENCY_THRESHOLD:
            self.recommendations.extend(
                [
                    f"⚠️ Columnas irregulares (consistencia: {consistency:.1%})",
                    "Usar on_bad_lines='warn' en pandas",
                    "Validar separador detectado",
                ]
            )
        elif consistency >= 0.95:
            self.recommendations.append(f"✅ Estructura consistente ({consistency:.1%})")

        # Líneas de resumen
        if self.stats.summary_lines_ignored > 0:
            self.recommendations.append(
                f"ℹ️ Se detectaron {self.stats.summary_lines_ignored} líneas "
                "de total/resumen. Considerar filtrarlas después de leer."
            )

    def _calculate_overall_confidence(self) -> None:
        """Calcula confianza general."""
        factors = []

        # Factor 1: Header detectado
        if self.header_candidate:
            if self.header_candidate.confidence == ConfidenceLevel.HIGH:
                factors.append(1.0)
            elif self.header_candidate.confidence == ConfidenceLevel.MEDIUM:
                factors.append(0.7)
            else:
                factors.append(0.5)
        else:
            factors.append(0.0)

        # Factor 2: Consistencia de columnas
        factors.append(self.stats.column_consistency)

        # Factor 3: Confianza de encoding
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
            "📊 REPORTE - DIAGNÓSTICO DE PRESUPUESTO".center(90),
            "=" * 90,
            "\n📁 ARCHIVO:",
            f"  Ruta: {self.file_path}",
            f"  Tamaño: {self._human_size(self.stats.file_size)}",
            f"  Líneas: {self.stats.total_lines:,}",
            "\n🔤 FORMATO:",
            f"  Encoding: {self.stats.encoding} ({self.stats.encoding_confidence:.1%})",
            f"  Separador: '{self.stats.csv_delimiter}'",
            f"  Dialecto detectado: {'Sí' if self.stats.csv_dialect_detected else 'No'}",
        ]

        # Encabezado
        if self.header_candidate:
            lines.extend(
                [
                    "\n✅ ENCABEZADO DETECTADO:",
                    f"  Línea: {self.header_candidate.line_num}",
                    f"  Columnas: {self.header_candidate.column_count}",
                    (
                        f"  Keywords ({self.header_candidate.match_count}): "
                        f"{', '.join(self.header_candidate.matches)}"
                    ),
                    f"  Confianza: {self.header_candidate.confidence.value.upper()}",
                    f"  Contenido: {self.header_candidate.content[:80]}...",
                ]
            )

            if self.stats.data_start_line:
                lines.append(f"  Datos desde línea: {self.stats.data_start_line}")
        else:
            lines.extend(
                [
                    "\n⚠️ NO SE DETECTÓ ENCABEZADO",
                    "  Posibles causas:",
                    "  - Formato no estándar",
                    "  - Keywords diferentes a las esperadas",
                    "  - Archivo sin encabezado",
                ]
            )

        # Columnas
        if self.stats.column_distribution:
            lines.append("\n📊 DISTRIBUCIÓN DE COLUMNAS:")
            for cols, count in self.stats.column_distribution.most_common(5):
                marker = "→" if cols == self.stats.dominant_column_count else " "
                pct = (count / sum(self.stats.column_distribution.values())) * 100
                lines.append(f"  {marker} {cols} columnas: {count} líneas ({pct:.1f}%)")

            lines.append(f"\n  Consistencia: {self.stats.column_consistency:.1%}")

        # Muestras
        if self.samples:
            lines.append("\n📝 MUESTRAS:")
            for sample in self.samples[:10]:
                content = (
                    sample["content"][:70] + "..."
                    if len(sample["content"]) > 70
                    else sample["content"]
                )
                lines.append(
                    f"  L{sample['line_num']:>4} ({sample['column_count']} cols): {content}"
                )

        # Recomendaciones
        if self.recommendations:
            lines.append("\n💡 RECOMENDACIONES:")
            for rec in self.recommendations:
                lines.append(f"  {rec}")

        # Confianza
        lines.extend(
            [
                "\n🎯 CONFIANZA GENERAL:",
                f"  Nivel: {self.stats.overall_confidence.upper()}",
                f"  Score: {self.stats.overall_confidence_score:.1%}",
                "\n" + "=" * 90 + "\n",
            ]
        )

        for line in lines:
            logger.info(line)

    def _limit_lines(self, lines: List[str]) -> List[str]:
        """Limita líneas para archivos grandes."""
        if len(lines) > self.MAX_LINES_TO_ANALYZE:
            logger.warning(
                f"⚠️ Archivo grande. Analizando primeras {self.MAX_LINES_TO_ANALYZE:,} líneas"
            )
            self.stats.truncated_analysis = True
            self.stats.lines_analyzed = self.MAX_LINES_TO_ANALYZE
            return lines[: self.MAX_LINES_TO_ANALYZE]

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
            warnings=self.warnings,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Exporta resultado como diccionario."""
        return {
            "file_path": str(self.file_path),
            "stats": asdict(self.stats),
            "header": (
                {
                    "line_num": self.header_candidate.line_num,
                    "content": self.header_candidate.content,
                    "matches": self.header_candidate.matches,
                    "column_count": self.header_candidate.column_count,
                    "confidence": self.header_candidate.confidence.value,
                }
                if self.header_candidate
                else None
            ),
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "errors": self.errors,
            "samples": self.samples[:10],
        }


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:
    """Función principal CLI."""
    if len(sys.argv) < 2:
        print("❌ Uso: python diagnose_presupuesto_file.py <archivo>")
        print("📖 Ejemplo: python diagnose_presupuesto_file.py presupuesto.csv")
        print("\n💡 Instalar: pip install chardet")
        return 1

    file_path = sys.argv[1]

    try:
        diagnostic = PresupuestoFileDiagnostic(file_path)
        result = diagnostic.diagnose()

        if not result.success:
            logger.error("❌ Diagnóstico falló")
            for error in result.errors:
                logger.error(f"  - {error}")
            return 1

        # Exportar JSON si se solicita
        if len(sys.argv) > 2 and sys.argv[2] == "--json":
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
