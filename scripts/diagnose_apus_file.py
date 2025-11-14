# scripts/diagnose_apus_file.py

"""
Herramienta avanzada de diagnóstico para archivos APU.
Detecta automáticamente encoding, dialecto CSV, estructura y patrones.

Autor: Ingeniero Senior
Versión: 2.0
"""

import csv
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Dependencia externa para detección de encoding
try:
    import chardet

    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logging.warning("⚠️ chardet no disponible. Instalar con: pip install chardet")

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class FileStats:
    """Estadísticas del archivo analizado."""

    file_size: int = 0
    encoding: str = "unknown"
    encoding_confidence: float = 0.0
    total_lines: int = 0
    empty_lines: int = 0
    non_empty_lines: int = 0

    # Análisis CSV
    csv_delimiter: Optional[str] = None
    csv_quotechar: Optional[str] = None
    csv_doublequote: bool = False
    csv_skipinitialspace: bool = False

    # Columnas
    column_counts: Counter = field(default_factory=Counter)
    most_common_column_count: Optional[int] = None
    column_count_variance: float = 0.0

    # Palabras clave
    lines_with_item: int = 0
    lines_with_unidad: int = 0
    lines_with_descripcion: int = 0
    numeric_rows: int = 0

    # Separadores (legacy)
    lines_with_semicolon: int = 0
    lines_with_tabs: int = 0
    max_semicolons: int = 0

    # Estructura
    blocks_by_double_newline: int = 0
    blocks_by_dashes: int = 0
    blocks_by_equals: int = 0

    # Categorías
    categories: Counter = field(default_factory=Counter)


@dataclass
class Pattern:
    """Patrón detectado en el archivo."""

    type: str
    line_num: int
    value: Optional[str] = None
    content: str = ""
    confidence: float = 1.0


@dataclass
class DiagnosticResult:
    """Resultado completo del diagnóstico."""

    success: bool
    stats: FileStats
    patterns: List[Pattern] = field(default_factory=list)
    sample_lines: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class APUFileDiagnostic:
    """
    Herramienta de diagnóstico avanzada para analizar archivos APU.

    Características:
    - Detección automática de encoding con chardet
    - Inferencia de dialecto CSV con csv.Sniffer
    - Análisis de columnas irregulares
    - Detección de patrones estructurales
    - Recomendaciones inteligentes

    Uso:
        diagnostic = APUFileDiagnostic("archivo.csv")
        result = diagnostic.diagnose()
        if result.success:
            print(result.stats)
    """

    # Constantes de configuración
    FALLBACK_ENCODINGS = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
    CATEGORY_KEYWORDS = ["MATERIALES", "MANO DE OBRA", "EQUIPO", "OTROS"]

    # Límites de procesamiento
    ENCODING_SAMPLE_SIZE = 100_000  # bytes para detección de encoding
    CSV_SNIFFER_SAMPLE_SIZE = 8_192  # bytes para csv.Sniffer
    MAX_LINES_COLUMN_ANALYSIS = 500
    MAX_SAMPLE_LINES = 20
    MAX_PATTERN_DETECTION_LINES = 100

    # Patrones regex compilados (eficiencia)
    PATTERN_ITEM = re.compile(r"ITEM\s*[:\s]\s*([\d,\.]+)", re.IGNORECASE)
    PATTERN_UNIDAD = re.compile(r"UNIDAD\s*[:\s]\s*([^\s;,]+)", re.IGNORECASE)
    PATTERN_DESCRIPCION = re.compile(r"DESCRIPCION|DESCRIPCIÓN", re.IGNORECASE)
    PATTERN_NUMERIC_ROW = re.compile(r"(?:[\d.,]+\s+){2,}[\d.,]+")
    PATTERN_MULTIPLE_SPACES = re.compile(r"\s{2,}")

    def __init__(self, file_path: str):
        """
        Inicializa el diagnóstico.

        Args:
            file_path: Ruta al archivo a analizar

        Raises:
            ValueError: Si la ruta no es válida
        """
        self.file_path = Path(file_path).resolve()
        self.stats = FileStats()
        self.patterns: List[Pattern] = []
        self.sample_lines: List[Dict[str, Any]] = []
        self.recommendations: List[str] = []
        self.errors: List[str] = []

    def diagnose(self) -> DiagnosticResult:
        """
        Ejecuta diagnóstico completo del archivo.

        Returns:
            DiagnosticResult con toda la información recopilada
        """
        logger.info(f"🔍 Iniciando diagnóstico: {self.file_path}")

        # Validaciones iniciales
        if not self._validate_file():
            return self._build_failure_result()

        # 1. Detectar encoding (con chardet si disponible)
        encoding_info = self._detect_encoding()
        if not encoding_info:
            return self._build_failure_result()

        self.stats.encoding = encoding_info[0]
        self.stats.encoding_confidence = encoding_info[1]

        # 2. Leer contenido con encoding correcto
        content = self._read_file(self.stats.encoding)
        if content is None:
            return self._build_failure_result()

        lines = content.splitlines()
        self.stats.total_lines = len(lines)

        # 3. Análisis básico de líneas
        self._analyze_basic_line_stats(lines)

        # 4. Detección de dialecto CSV
        self._detect_csv_dialect(content)

        # 5. Análisis de columnas (irregularidades)
        if self.stats.csv_delimiter:
            self._analyze_column_distribution(lines)

        # 6. Análisis estructural (bloques)
        self._analyze_block_structure(content)

        # 7. Detección de patrones clave
        self._detect_key_patterns(lines)

        # 8. Análisis de palabras clave
        self._analyze_keywords(lines)

        # 9. Generar recomendaciones
        self._generate_recommendations()

        # 10. Logging del reporte
        self._log_diagnostic_report()

        return self._build_success_result()

    def _validate_file(self) -> bool:
        """Valida que el archivo existe y es accesible."""
        if not self.file_path.exists():
            error = f"Archivo no encontrado: {self.file_path}"
            logger.error(f"❌ {error}")
            self.errors.append(error)
            return False

        if not self.file_path.is_file():
            error = f"Ruta no es un archivo: {self.file_path}"
            logger.error(f"❌ {error}")
            self.errors.append(error)
            return False

        try:
            self.stats.file_size = self.file_path.stat().st_size
            logger.info(f"📦 Tamaño: {self.stats.file_size:,} bytes")

            if self.stats.file_size == 0:
                error = "El archivo está vacío"
                logger.error(f"❌ {error}")
                self.errors.append(error)
                return False

        except OSError as e:
            error = f"Error al acceder al archivo: {e}"
            logger.error(f"❌ {error}")
            self.errors.append(error)
            return False

        return True

    def _detect_encoding(self) -> Optional[Tuple[str, float]]:
        """
        Detecta el encoding del archivo usando chardet o fallback manual.

        Returns:
            Tupla (encoding, confianza) o None si falla
        """
        if CHARDET_AVAILABLE:
            return self._detect_encoding_with_chardet()
        else:
            logger.warning("⚠️ Usando detección manual de encoding (menos precisa)")
            return self._detect_encoding_fallback()

    def _detect_encoding_with_chardet(self) -> Optional[Tuple[str, float]]:
        """
        Usa chardet para detectar encoding automáticamente.

        Returns:
            Tupla (encoding, confianza) o None
        """
        try:
            # Leer muestra de bytes
            with open(self.file_path, "rb") as f:
                raw_data = f.read(self.ENCODING_SAMPLE_SIZE)

            if not raw_data:
                self.errors.append("Archivo vacío")
                return None

            # Detectar encoding
            detection = chardet.detect(raw_data)
            encoding = detection.get("encoding")
            confidence = detection.get("confidence", 0.0)

            if not encoding:
                logger.warning("⚠️ chardet no pudo determinar encoding")
                return self._detect_encoding_fallback()

            logger.info(f"✅ Encoding detectado: {encoding} (confianza: {confidence:.1%})")

            # Si la confianza es muy baja, advertir
            if confidence < 0.7:
                logger.warning(
                    f"⚠️ Baja confianza en encoding ({confidence:.1%}). Verificar resultados."
                )

            return (encoding, confidence)

        except Exception as e:
            logger.error(f"❌ Error en detección con chardet: {e}")
            return self._detect_encoding_fallback()

    def _detect_encoding_fallback(self) -> Optional[Tuple[str, float]]:
        """
        Fallback: prueba encodings comunes manualmente.

        Returns:
            Tupla (encoding, 0.5) o None
        """
        for encoding in self.FALLBACK_ENCODINGS:
            try:
                with open(self.file_path, "r", encoding=encoding) as f:
                    # Intentar leer una muestra
                    f.read(1024)

                logger.info(f"✅ Encoding funcionó: {encoding}")
                return (encoding, 0.5)  # Confianza media

            except (UnicodeDecodeError, LookupError):
                continue

        error = "No se pudo determinar encoding con ningún método"
        logger.error(f"❌ {error}")
        self.errors.append(error)
        return None

    def _read_file(self, encoding: str) -> Optional[str]:
        """
        Lee el archivo completo con el encoding especificado.

        Args:
            encoding: Encoding a usar

        Returns:
            Contenido del archivo o None si falla
        """
        try:
            # Para archivos muy grandes, considerar lectura por chunks
            if self.stats.file_size > 50_000_000:  # > 50MB
                logger.warning(
                    f"⚠️ Archivo grande ({self.stats.file_size:,} bytes). "
                    "El análisis puede tardar."
                )

            content = self.file_path.read_text(
                encoding=encoding,
                errors="replace",  # Reemplazar caracteres inválidos
            )

            return content

        except Exception as e:
            error = f"Error al leer archivo con encoding {encoding}: {e}"
            logger.error(f"❌ {error}")
            self.errors.append(error)
            return None

    def _detect_csv_dialect(self, content: str) -> None:
        """
        Usa csv.Sniffer para detectar automáticamente el dialecto CSV.

        Args:
            content: Contenido del archivo
        """
        try:
            # Extraer muestra para el sniffer
            sample = content[: self.CSV_SNIFFER_SAMPLE_SIZE]

            # Intentar detectar dialecto
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)

            # Guardar propiedades del dialecto
            self.stats.csv_delimiter = dialect.delimiter
            self.stats.csv_quotechar = dialect.quotechar
            self.stats.csv_doublequote = dialect.doublequote
            self.stats.csv_skipinitialspace = dialect.skipinitialspace

            logger.info(
                f"✅ Dialecto CSV detectado:\n"
                f"   - Delimitador: '{self.stats.csv_delimiter}'\n"
                f"   - Quote char: '{self.stats.csv_quotechar}'\n"
                f"   - Double quote: {self.stats.csv_doublequote}"
            )

            # Validar que tiene sentido
            if self.stats.csv_delimiter in ["\n", "\r"]:
                logger.warning(
                    "⚠️ Delimitador detectado parece inválido. "
                    "Puede no ser un CSV bien formado."
                )
                self.stats.csv_delimiter = None

        except csv.Error as e:
            logger.warning(
                f"⚠️ No se pudo detectar dialecto CSV: {e}. "
                "Puede no ser un archivo CSV estándar."
            )
            # Intentar detección manual básica como fallback
            self._detect_delimiter_fallback(content.splitlines()[:100])

        except Exception as e:
            logger.warning(f"⚠️ Error inesperado en csv.Sniffer: {e}")
            self._detect_delimiter_fallback(content.splitlines()[:100])

    def _detect_delimiter_fallback(self, lines: List[str]) -> None:
        """
        Fallback manual para detección de delimitador.

        Args:
            lines: Primeras líneas del archivo
        """
        potential_delimiters = [";", ",", "\t", "|", ":"]
        delimiter_scores = Counter()

        clean_lines = [line.strip() for line in lines if line.strip()]

        for line in clean_lines:
            for delim in potential_delimiters:
                count = line.count(delim)
                if count > 0:
                    delimiter_scores[delim] += 1

        if delimiter_scores:
            best_delim = delimiter_scores.most_common(1)[0][0]
            self.stats.csv_delimiter = best_delim
            logger.info(f"✅ Delimitador detectado (fallback): '{best_delim}'")

    def _analyze_column_distribution(self, lines: List[str]) -> None:
        """
        Analiza la distribución de columnas usando csv.reader.
        Detecta filas irregulares.

        Args:
            lines: Líneas del archivo
        """
        if not self.stats.csv_delimiter:
            return

        try:
            # Analizar primeras N líneas
            lines_to_analyze = lines[: self.MAX_LINES_COLUMN_ANALYSIS]

            # Usar csv.reader con el delimitador detectado
            reader = csv.reader(
                lines_to_analyze,
                delimiter=self.stats.csv_delimiter,
                quotechar=self.stats.csv_quotechar or '"',
            )

            column_counts = Counter()

            for row in reader:
                num_columns = len(row)
                column_counts[num_columns] += 1

            self.stats.column_counts = column_counts

            if column_counts:
                # Número de columnas más común
                most_common = column_counts.most_common(1)[0]
                self.stats.most_common_column_count = most_common[0]

                # Calcular varianza (dispersión)
                total_rows = sum(column_counts.values())
                expected = self.stats.most_common_column_count

                variance = (
                    sum(
                        count * ((cols - expected) ** 2)
                        for cols, count in column_counts.items()
                    )
                    / total_rows
                    if total_rows > 0
                    else 0
                )

                self.stats.column_count_variance = variance

                logger.info(
                    f"📊 Análisis de columnas:\n"
                    f"   - Columnas más común: {self.stats.most_common_column_count}\n"
                    f"   - Distribución: {dict(column_counts.most_common(5))}\n"
                    f"   - Varianza: {variance:.2f}"
                )

                # Advertir si hay mucha irregularidad
                if len(column_counts) > 5:
                    logger.warning(
                        f"⚠️ Detectadas {len(column_counts)} distribuciones "
                        "diferentes de columnas. El archivo puede tener "
                        "formato inconsistente."
                    )

        except Exception as e:
            logger.warning(f"⚠️ Error en análisis de columnas: {e}")

    def _analyze_basic_line_stats(self, lines: List[str]) -> None:
        """
        Análisis básico de estadísticas de líneas.

        Args:
            lines: Todas las líneas del archivo
        """
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            if not stripped:
                self.stats.empty_lines += 1
                continue

            self.stats.non_empty_lines += 1

            # Guardar muestras
            if len(self.sample_lines) < self.MAX_SAMPLE_LINES:
                self.sample_lines.append(
                    {
                        "line_num": line_num,
                        "content": stripped[:200],
                        "length": len(stripped),
                    }
                )

            # Estadísticas de separadores (legacy)
            if ";" in stripped:
                self.stats.lines_with_semicolon += 1
                semicolon_count = stripped.count(";")
                self.stats.max_semicolons = max(self.stats.max_semicolons, semicolon_count)

            if "\t" in line:
                self.stats.lines_with_tabs += 1

    def _analyze_block_structure(self, content: str) -> None:
        """
        Analiza estructura de bloques en el archivo.

        Args:
            content: Contenido completo del archivo
        """
        # Bloques por doble salto
        blocks_double = [b for b in re.split(r"\n\s*\n", content) if b.strip()]
        self.stats.blocks_by_double_newline = len(blocks_double)

        # Bloques por guiones
        blocks_dashes = [b for b in re.split(r"\n-{3,}\n", content) if b.strip()]
        self.stats.blocks_by_dashes = len(blocks_dashes)

        # Bloques por signos igual
        blocks_equals = [b for b in re.split(r"\n={3,}\n", content) if b.strip()]
        self.stats.blocks_by_equals = len(blocks_equals)

    def _detect_key_patterns(self, lines: List[str]) -> None:
        """
        Detecta patrones estructurales clave (ITEM, UNIDAD, etc.).

        Args:
            lines: Líneas del archivo
        """
        lines_to_analyze = lines[: self.MAX_PATTERN_DETECTION_LINES]

        for line_num, line in enumerate(lines_to_analyze, 1):
            stripped = line.strip()

            # Patrón ITEM
            match = self.PATTERN_ITEM.search(stripped)
            if match:
                self.patterns.append(
                    Pattern(
                        type="ITEM_CODE",
                        line_num=line_num,
                        value=match.group(1),
                        content=stripped[:150],
                    )
                )

            # Patrón UNIDAD
            match = self.PATTERN_UNIDAD.search(stripped)
            if match:
                self.patterns.append(
                    Pattern(
                        type="UNIT",
                        line_num=line_num,
                        value=match.group(1),
                        content=stripped[:150],
                    )
                )

            # Filas numéricas
            if self.PATTERN_NUMERIC_ROW.search(stripped):
                self.stats.numeric_rows += 1

    def _analyze_keywords(self, lines: List[str]) -> None:
        """
        Analiza presencia de palabras clave importantes.

        Args:
            lines: Líneas del archivo
        """
        for line in lines:
            stripped = line.strip()

            if self.PATTERN_ITEM.search(stripped):
                self.stats.lines_with_item += 1

            if self.PATTERN_UNIDAD.search(stripped):
                self.stats.lines_with_unidad += 1

            if self.PATTERN_DESCRIPCION.search(stripped):
                self.stats.lines_with_descripcion += 1

            # Categorías
            for category in self.CATEGORY_KEYWORDS:
                if category in stripped.upper():
                    self.stats.categories[category] += 1

    def _generate_recommendations(self) -> None:
        """Genera recomendaciones inteligentes basadas en el análisis."""

        # Recomendación de encoding
        if self.stats.encoding_confidence < 0.7:
            self.recommendations.append(
                f"⚠️ Baja confianza en encoding ({self.stats.encoding_confidence:.1%}). "
                f"Verificar manualmente si hay caracteres extraños."
            )

        # Recomendación de lectura CSV
        if self.stats.csv_delimiter:
            self.recommendations.append(
                f"✅ Usar pandas.read_csv() con:\n"
                f"   sep='{self.stats.csv_delimiter}', "
                f"encoding='{self.stats.encoding}'"
            )

        # Columnas irregulares
        if self.stats.column_count_variance > 2.0:
            self.recommendations.append(
                f"⚠️ Alta variabilidad en número de columnas (varianza: "
                f"{self.stats.column_count_variance:.2f}). "
                f"Considerar:\n"
                f"   - Usar on_bad_lines='warn' en pandas\n"
                f"   - Parsing personalizado por bloques"
            )

        # Estructura de bloques
        if self.stats.blocks_by_double_newline > 10:
            self.recommendations.append(
                f"📦 Archivo con estructura de bloques "
                f"({self.stats.blocks_by_double_newline} bloques). "
                "Considerar parsing basado en bloques."
            )

        # Patrones ITEM
        item_patterns = [p for p in self.patterns if p.type == "ITEM_CODE"]
        if item_patterns:
            self.recommendations.append(
                f"✅ Detectados {len(item_patterns)} códigos ITEM. "
                f"Probablemente 'ITEM' marca inicio de cada APU."
            )
        elif self.stats.lines_with_item == 0:
            self.recommendations.append(
                "⚠️ NO se detectaron líneas con 'ITEM'. "
                "Verificar si el formato es el esperado."
            )

        # Datos numéricos
        if self.stats.numeric_rows > 20:
            self.recommendations.append(
                f"📊 Detectadas {self.stats.numeric_rows} filas con datos numéricos. "
                f"Posiblemente representa tabla de precios/cantidades."
            )

    def _log_diagnostic_report(self) -> None:
        """Genera y registra el reporte completo en logging."""

        report_lines = [
            "\n" + "=" * 80,
            "📊 REPORTE DE DIAGNÓSTICO AVANZADO - ARCHIVO APU",
            "=" * 80,
            "\n🔍 DETECCIÓN AUTOMÁTICA DE FORMATO:",
            f"  Encoding: {self.stats.encoding} "
            f"(confianza: {self.stats.encoding_confidence:.1%})",
            f"  Delimitador CSV: '{self.stats.csv_delimiter or 'N/A'}'",
            f"  Quote char: '{self.stats.csv_quotechar or 'N/A'}'",
            "\n📈 ESTADÍSTICAS GENERALES:",
            f"  Tamaño: {self.stats.file_size:,} bytes",
            f"  Total líneas: {self.stats.total_lines:,}",
            f"  Líneas vacías: {self.stats.empty_lines:,}",
            f"  Líneas con contenido: {self.stats.non_empty_lines:,}",
        ]

        # Análisis de columnas
        if self.stats.column_counts:
            report_lines.extend(
                [
                    "\n📊 ANÁLISIS DE COLUMNAS:",
                    f"  Número más común: {self.stats.most_common_column_count}",
                    f"  Distribución: {dict(self.stats.column_counts.most_common(5))}",
                    f"  Varianza: {self.stats.column_count_variance:.2f}",
                    "  "
                    + (
                        "✅ Estructura regular"
                        if self.stats.column_count_variance < 1.0
                        else "⚠️ Estructura irregular"
                    ),
                ]
            )

        # Estructura
        report_lines.extend(
            [
                "\n🏗️ ESTRUCTURA:",
                f"  Bloques (líneas vacías): {self.stats.blocks_by_double_newline}",
                f"  Bloques (guiones): {self.stats.blocks_by_dashes}",
                f"  Bloques (iguales): {self.stats.blocks_by_equals}",
            ]
        )

        # Palabras clave
        report_lines.extend(
            [
                "\n🔑 PALABRAS CLAVE:",
                f"  'ITEM': {self.stats.lines_with_item:,} líneas",
                f"  'UNIDAD': {self.stats.lines_with_unidad:,} líneas",
                f"  'DESCRIPCION': {self.stats.lines_with_descripcion:,} líneas",
                f"  Filas numéricas: {self.stats.numeric_rows:,}",
            ]
        )

        # Categorías
        if self.stats.categories:
            report_lines.append("\n📦 CATEGORÍAS:")
            for cat, count in self.stats.categories.most_common():
                report_lines.append(f"  {cat}: {count} veces")

        # Patrones clave
        item_codes = [p for p in self.patterns if p.type == "ITEM_CODE"]
        if item_codes:
            report_lines.append(f"\n🎯 CÓDIGOS ITEM ({len(item_codes)} encontrados):")
            for p in item_codes[:5]:
                report_lines.append(f"  Línea {p.line_num}: {p.value}")

        # Muestra
        report_lines.append("\n📝 MUESTRA DE LÍNEAS:")
        for sample in self.sample_lines[:10]:
            report_lines.append(
                f"  L{sample['line_num']:4d} ({sample['length']:3d} chars): "
                f"{sample['content']}"
            )

        # Recomendaciones
        if self.recommendations:
            report_lines.append("\n💡 RECOMENDACIONES:")
            for rec in self.recommendations:
                for line in rec.split("\n"):
                    report_lines.append(f"  {line}")

        report_lines.append("=" * 80 + "\n")

        # Imprimir
        for line in report_lines:
            logger.info(line)

    def _build_success_result(self) -> DiagnosticResult:
        """Construye resultado exitoso."""
        return DiagnosticResult(
            success=True,
            stats=self.stats,
            patterns=self.patterns,
            sample_lines=self.sample_lines,
            recommendations=self.recommendations,
            errors=self.errors,
        )

    def _build_failure_result(self) -> DiagnosticResult:
        """Construye resultado de fallo."""
        return DiagnosticResult(success=False, stats=self.stats, errors=self.errors)


def main():
    """Función principal para ejecución standalone."""
    if len(sys.argv) < 2:
        print("❌ Uso: python diagnose_apus_file.py <ruta_archivo>")
        print("📖 Ejemplo: python diagnose_apus_file.py data/apus.csv")
        print("\n💡 Instalar dependencias: pip install chardet")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        diagnostic = APUFileDiagnostic(file_path)
        result = diagnostic.diagnose()

        if not result.success:
            logger.error("❌ Diagnóstico fallido:")
            for error in result.errors:
                logger.error(f"  - {error}")
            sys.exit(1)

        logger.info("✅ Diagnóstico completado exitosamente")

        # Exportar resultado como JSON si se requiere
        if len(sys.argv) > 2 and sys.argv[2] == "--json":
            import json
            from dataclasses import asdict

            output = {
                "success": result.success,
                "stats": asdict(result.stats),
                "recommendations": result.recommendations,
                "errors": result.errors,
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Proceso interrumpido por el usuario")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
