# scripts/clean_csv.py
"""
Filtro de líneas para archivos CSV que preserva el formato original.

Este módulo proporciona funcionalidad para limpiar archivos CSV eliminando
líneas problemáticas mientras preserva exactamente el formato original de
las líneas válidas.
"""

import codecs
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

# Configuración de logging con handler específico para evitar afectar otros módulos
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SkipReason(Enum):
    """Razones por las que una fila puede ser saltada."""

    EMPTY = "Línea vacía"
    COMMENT = "Comentario"
    INCONSISTENT_DELIMITERS = "Número inconsistente de delimitadores"
    WHITESPACE_ONLY = "Solo espacios en blanco"


@dataclass
class CleaningStats:
    """Estadísticas del proceso de limpieza."""

    rows_written: int = 0
    rows_skipped: int = 0
    skip_reasons: Dict[SkipReason, int] = field(
        default_factory=lambda: {reason: 0 for reason in SkipReason}
    )

    def record_skip(self, reason: SkipReason) -> None:
        """Registra una fila saltada con su razón."""
        if not isinstance(reason, SkipReason):
            raise TypeError(f"reason debe ser SkipReason, no {type(reason).__name__}")
        self.rows_skipped += 1
        self.skip_reasons[reason] += 1

    def record_written(self) -> None:
        """Registra una fila escrita exitosamente."""
        self.rows_written += 1

    @property
    def total_processed(self) -> int:
        """Retorna el total de filas procesadas (escritas + saltadas)."""
        return self.rows_written + self.rows_skipped

    def reset(self) -> None:
        """Reinicia todas las estadísticas a cero."""
        self.rows_written = 0
        self.rows_skipped = 0
        self.skip_reasons = {reason: 0 for reason in SkipReason}

    def to_dict(self) -> Dict[str, Any]:
        """Convierte las estadísticas a un diccionario serializable."""
        return {
            "rows_written": self.rows_written,
            "rows_skipped": self.rows_skipped,
            "total_processed": self.total_processed,
            "skip_reasons": {
                reason.name: count for reason, count in self.skip_reasons.items()
            },
        }


class CSVCleaner:
    """
    Filtro de líneas para archivos CSV que preserva el formato original.

    Este limpiador NO reformatea el CSV, solo actúa como filtro:
    - Lee líneas del archivo original
    - Decide si cada línea es válida
    - Escribe las líneas válidas EXACTAMENTE como las encontró

    Esto evita problemas de re-formatting que pueden romper parsers
    posteriores que esperan un formato específico.
    """

    # Delimitadores válidos comunes
    VALID_DELIMITERS = frozenset({";", ",", "\t", "|"})

    # Tamaño máximo de archivo (100MB por defecto)
    MAX_FILE_SIZE = 100 * 1024 * 1024

    # Prefijos de comentarios reconocidos
    COMMENT_PREFIXES = ("#",)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        delimiter: str = ";",
        encoding: str = "utf-8",
        overwrite: bool = False,
        strict_mode: bool = True,
        verbose: bool = False,
        use_atomic_write: bool = True,
    ):
        """
        Inicializa el limpiador de CSV.

        Args:
            input_path: Ruta al archivo CSV de entrada
            output_path: Ruta al archivo CSV de salida
            delimiter: Delimitador del CSV (por defecto ';')
            encoding: Codificación del archivo (por defecto 'utf-8')
            overwrite: Si True, sobrescribe archivo de salida existente
            strict_mode: Si True, valida estrictamente el número de delimitadores
            verbose: Si True, muestra información detallada de depuración
            use_atomic_write: Si True, usa escritura atómica (archivo temporal + rename)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.delimiter = delimiter
        self.encoding = encoding
        self.overwrite = overwrite
        self.strict_mode = strict_mode
        self.verbose = verbose
        self.use_atomic_write = use_atomic_write
        self.stats = CleaningStats()
        self.expected_delimiter_count: Optional[int] = None
        self._temp_file_path: Optional[Path] = None

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    def _validate_encoding(self) -> None:
        """
        Valida que el encoding especificado sea válido.

        Raises:
            ValueError: Si el encoding no es reconocido
        """
        try:
            codecs.lookup(self.encoding)
        except LookupError:
            raise ValueError(
                f"Encoding no reconocido: '{self.encoding}'. "
                f"Ejemplos válidos: utf-8, latin-1, cp1252, iso-8859-1"
            )

    def _validate_inputs(self) -> None:
        """
        Valida los parámetros de entrada antes de procesar.

        Raises:
            ValueError: Si alguna validación falla
            FileNotFoundError: Si el archivo de entrada no existe
            PermissionError: Si no hay permisos suficientes
        """
        # Validar encoding primero
        self._validate_encoding()

        # Validar archivo de entrada existe
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"El archivo de entrada no existe: {self.input_path}"
            )

        if not self.input_path.is_file():
            raise ValueError(
                f"La ruta de entrada no es un archivo: {self.input_path}"
            )

        # Validar permisos de lectura
        if not os.access(self.input_path, os.R_OK):
            raise PermissionError(
                f"Sin permisos de lectura para: {self.input_path}"
            )

        # Validar tamaño del archivo
        try:
            file_size = self.input_path.stat().st_size
        except OSError as e:
            raise IOError(f"No se pudo obtener información del archivo: {e}")

        if file_size == 0:
            raise ValueError(f"El archivo de entrada está vacío: {self.input_path}")

        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"El archivo excede el tamaño máximo permitido "
                f"({self.MAX_FILE_SIZE / 1024 / 1024:.2f} MB): "
                f"{file_size / 1024 / 1024:.2f} MB"
            )

        # Validar que entrada y salida no sean el mismo archivo
        try:
            input_resolved = self.input_path.resolve()
            output_resolved = self.output_path.resolve()
            if input_resolved == output_resolved:
                raise ValueError(
                    "El archivo de entrada y salida no pueden ser el mismo"
                )
        except OSError as e:
            raise ValueError(f"Error resolviendo rutas de archivos: {e}")

        # Validar archivo de salida
        if self.output_path.exists() and not self.overwrite:
            raise ValueError(
                f"El archivo de salida ya existe: {self.output_path}. "
                f"Use overwrite=True para sobrescribir."
            )

        # Validar directorio de salida
        output_dir = self.output_path.parent
        if not output_dir.exists():
            raise ValueError(f"El directorio de salida no existe: {output_dir}")

        if not output_dir.is_dir():
            raise ValueError(
                f"La ruta padre de salida no es un directorio: {output_dir}"
            )

        # Validar permisos de escritura en directorio de salida
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(
                f"Sin permisos de escritura en directorio: {output_dir}"
            )

        # Validar delimitador
        if not self.delimiter:
            raise ValueError("El delimitador no puede estar vacío")

        if len(self.delimiter) != 1:
            raise ValueError(
                f"El delimitador debe ser un solo carácter, "
                f"recibido: '{self.delimiter}' (longitud: {len(self.delimiter)})"
            )

        if self.delimiter not in self.VALID_DELIMITERS:
            logger.warning(
                f"Delimitador inusual detectado: '{repr(self.delimiter)}'. "
                f"Delimitadores comunes: {set(self.VALID_DELIMITERS)}"
            )

        logger.debug("✅ Validaciones de entrada completadas exitosamente")

    def _count_delimiters(self, line: str) -> int:
        """
        Cuenta el número de delimitadores en una línea.

        Nota: Este método hace un conteo simple sin considerar campos
        entrecomillados, ya que el objetivo es filtrar líneas obviamente
        malformadas, no parsear el CSV completamente.

        Args:
            line: Línea a analizar (sin salto de línea)

        Returns:
            Número de delimitadores encontrados
        """
        if not line:
            return 0
        return line.count(self.delimiter)

    def _is_empty_line(self, line: str) -> bool:
        """
        Determina si una línea está vacía o contiene solo espacios.

        Args:
            line: Línea a evaluar (sin salto de línea)

        Returns:
            True si la línea está vacía o solo contiene espacios
        """
        return not line or not line.strip()

    def _is_comment_line(self, line: str) -> bool:
        """
        Determina si una línea es un comentario.

        Args:
            line: Línea a evaluar (sin salto de línea)

        Returns:
            True si la línea es un comentario
        """
        stripped = line.strip()
        if not stripped:
            return False
        return any(stripped.startswith(prefix) for prefix in self.COMMENT_PREFIXES)

    def _is_all_whitespace_fields(self, line: str) -> bool:
        """
        Determina si una línea contiene solo campos vacíos o con espacios.
        Por ejemplo: ";;;" o "  ;  ;  "

        Precondición: La línea NO está vacía (ya verificado por _is_empty_line)

        Args:
            line: Línea a evaluar (sin salto de línea, no vacía)

        Returns:
            True si todos los campos están vacíos o solo contienen espacios
        """
        # Si no contiene el delimitador, verificar si es solo espacios
        if self.delimiter not in line:
            return not line.strip()

        # Dividir por el delimitador y verificar si todos los campos están vacíos
        fields = line.split(self.delimiter)
        return all(not f.strip() for f in fields)

    def _should_skip_line(self, line: str, line_num: int) -> Optional[SkipReason]:
        """
        Determina si una línea debe ser saltada y por qué razón.

        El orden de verificación está optimizado para casos comunes:
        1. Líneas vacías (muy común)
        2. Comentarios (común en algunos archivos)
        3. Solo espacios en blanco
        4. Delimitadores inconsistentes (requiere conteo)

        Args:
            line: Línea a evaluar (sin el salto de línea final)
            line_num: Número de línea (para logging)

        Returns:
            SkipReason si debe saltarse, None si es válida
        """
        # 1. Verificar líneas completamente vacías (caso más común y rápido)
        if not line:
            if self.verbose:
                logger.debug(f"Línea {line_num}: Línea vacía")
            return SkipReason.EMPTY

        # 2. Verificar si es solo espacios en blanco (sin delimitadores)
        stripped = line.strip()
        if not stripped:
            if self.verbose:
                logger.debug(f"Línea {line_num}: Solo espacios")
            return SkipReason.EMPTY

        # 3. Verificar líneas de comentario
        if any(stripped.startswith(prefix) for prefix in self.COMMENT_PREFIXES):
            if self.verbose:
                logger.debug(f"Línea {line_num}: Comentario detectado")
            return SkipReason.COMMENT

        # 4. Verificar líneas con solo espacios en blanco en todos los campos
        if self._is_all_whitespace_fields(line):
            if self.verbose:
                logger.debug(f"Línea {line_num}: Solo espacios en blanco en campos")
            return SkipReason.WHITESPACE_ONLY

        # 5. Validar consistencia de delimitadores (si está en modo estricto)
        if self.strict_mode and self.expected_delimiter_count is not None:
            delimiter_count = self._count_delimiters(line)
            if delimiter_count != self.expected_delimiter_count:
                if self.verbose:
                    logger.debug(
                        f"Línea {line_num}: Esperados {self.expected_delimiter_count} "
                        f"delimitadores, encontrados {delimiter_count}"
                    )
                return SkipReason.INCONSISTENT_DELIMITERS

        return None

    def _process_header(self, header_line: str) -> None:
        """
        Procesa la línea de encabezado y establece la configuración esperada.

        Args:
            header_line: Línea de encabezado (sin salto de línea)

        Raises:
            ValueError: Si el encabezado es inválido
        """
        # Validar que el header no esté completamente vacío
        if not header_line or not header_line.strip():
            raise ValueError("El encabezado del CSV está vacío")

        if self._is_all_whitespace_fields(header_line):
            raise ValueError(
                "El encabezado del CSV contiene solo espacios en blanco"
            )

        # Contar delimitadores en el header para validación futura
        self.expected_delimiter_count = self._count_delimiters(header_line)

        # Advertir si el header no tiene delimitadores (posible error de configuración)
        if self.expected_delimiter_count == 0:
            logger.warning(
                f"⚠️  El encabezado no contiene el delimitador '{repr(self.delimiter)}'. "
                f"¿Es correcto el delimitador especificado?"
            )
            logger.warning(f"   Contenido del header: {header_line[:100]}...")

        # Extraer nombres de columnas para logging
        column_names = header_line.split(self.delimiter)
        num_columns = len(column_names)

        logger.info(f"✅ Encabezado detectado con {num_columns} columnas")

        if self.verbose:
            logger.debug(
                f"Delimitadores en encabezado: {self.expected_delimiter_count}"
            )
            # Mostrar solo las primeras columnas si hay muchas
            if num_columns > 10:
                logger.debug(f"Primeras 10 columnas: {column_names[:10]}")
            else:
                logger.debug(f"Columnas: {column_names}")

        # Advertir sobre encabezados duplicados
        column_names_stripped = [col.strip() for col in column_names]
        unique_columns = set(column_names_stripped)

        if len(column_names_stripped) != len(unique_columns):
            seen = set()
            duplicates = set()
            for col in column_names_stripped:
                if col in seen:
                    duplicates.add(col)
                seen.add(col)
            logger.warning(f"⚠️  Encabezados duplicados detectados: {duplicates}")

        # Advertir sobre columnas vacías
        empty_columns = [i for i, col in enumerate(column_names_stripped) if not col]
        if empty_columns:
            logger.warning(
                f"⚠️  Columnas con nombre vacío en posiciones: {empty_columns}"
            )

    def _create_temp_file(self) -> TextIO:
        """
        Crea un archivo temporal para escritura atómica.

        Returns:
            Handle del archivo temporal abierto para escritura
        """
        output_dir = self.output_path.parent
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=".csv_clean_",
            dir=output_dir,
        )
        os.close(fd)  # Cerrar el descriptor, abriremos con open()
        self._temp_file_path = Path(temp_path)

        return open(
            self._temp_file_path,
            "w",
            encoding=self.encoding,
            newline="",
        )

    def _finalize_output(self) -> None:
        """
        Finaliza la escritura moviendo el archivo temporal al destino final.

        Raises:
            IOError: Si no se puede mover el archivo
        """
        if self._temp_file_path and self._temp_file_path.exists():
            try:
                # En sistemas POSIX, rename es atómico
                # En Windows, puede fallar si el destino existe
                if self.output_path.exists():
                    self.output_path.unlink()
                shutil.move(str(self._temp_file_path), str(self.output_path))
                self._temp_file_path = None
            except OSError as e:
                raise IOError(f"Error finalizando archivo de salida: {e}")

    def _cleanup_temp_file(self) -> None:
        """Elimina el archivo temporal si existe."""
        if self._temp_file_path and self._temp_file_path.exists():
            try:
                self._temp_file_path.unlink()
            except OSError:
                logger.warning(
                    f"No se pudo eliminar archivo temporal: {self._temp_file_path}"
                )
            finally:
                self._temp_file_path = None

    def clean(self) -> CleaningStats:
        """
        Ejecuta el proceso de limpieza del CSV.

        IMPORTANTE: Este método NO reformatea el CSV. Lee líneas del archivo
        original y escribe las líneas válidas EXACTAMENTE como las encontró,
        preservando comillas, espacios, y cualquier otro formato.

        Returns:
            CleaningStats con las estadísticas del proceso

        Raises:
            ValueError: Si hay errores de validación
            FileNotFoundError: Si el archivo de entrada no existe
            PermissionError: Si no hay permisos suficientes
            IOError: Si hay errores de lectura/escritura
        """
        # Reiniciar estadísticas
        self.stats.reset()
        self.expected_delimiter_count = None

        # Validar parámetros de entrada
        self._validate_inputs()

        logger.info(f"🧹 Iniciando limpieza: {self.input_path} -> {self.output_path}")
        logger.info(f"   Delimitador: '{repr(self.delimiter)}'")
        logger.info(f"   Encoding: {self.encoding}")
        logger.info(f"   Modo estricto: {self.strict_mode}")
        logger.info(f"   Escritura atómica: {self.use_atomic_write}")
        logger.info("   Modo filtro: PRESERVA FORMATO ORIGINAL")

        outfile = None
        try:
            # Abrir archivo de entrada
            with open(
                self.input_path,
                "r",
                encoding=self.encoding,
                errors="replace",
            ) as infile:

                # Preparar archivo de salida
                if self.use_atomic_write:
                    outfile = self._create_temp_file()
                else:
                    outfile = open(
                        self.output_path,
                        "w",
                        encoding=self.encoding,
                        newline="",
                    )

                # Procesar encabezado
                header_line = infile.readline()

                if not header_line:
                    raise ValueError(
                        "El archivo CSV está vacío o no contiene encabezado"
                    )

                # Remover salto de línea para validación
                header_clean = header_line.rstrip("\r\n")
                self._process_header(header_clean)

                # Escribir encabezado EXACTAMENTE como se leyó
                outfile.write(header_line)

                # Procesar resto de líneas
                line_num = 1  # Header es línea 1

                for raw_line in infile:
                    line_num += 1

                    # Remover salto de línea solo para validación
                    line_clean = raw_line.rstrip("\r\n")

                    skip_reason = self._should_skip_line(line_clean, line_num)

                    if skip_reason:
                        self.stats.record_skip(skip_reason)
                        continue

                    # Escribir la línea EXACTAMENTE como se leyó
                    outfile.write(raw_line)
                    self.stats.record_written()

            # Cerrar archivo de salida antes de moverlo
            if outfile:
                outfile.close()
                outfile = None

            # Mover archivo temporal al destino final
            if self.use_atomic_write:
                self._finalize_output()

            self._print_summary()
            return self.stats

        except UnicodeDecodeError as e:
            raise ValueError(
                f"Error de codificación leyendo archivo. "
                f"Encoding actual: '{self.encoding}'. "
                f"Intente con: latin-1, cp1252, iso-8859-1. Detalle: {e}"
            )
        except PermissionError as e:
            raise PermissionError(f"Permiso denegado: {e}")
        except OSError as e:
            raise IOError(f"Error de I/O: {e}")
        finally:
            # Asegurar cierre del archivo de salida
            if outfile and not outfile.closed:
                outfile.close()

            # Limpiar archivo temporal en caso de error
            if self._temp_file_path:
                self._cleanup_temp_file()

    def _print_summary(self) -> None:
        """Imprime un resumen detallado del proceso de limpieza."""
        total = self.stats.total_processed
        logger.info("🎉 Limpieza completada exitosamente")
        logger.info(f"   📊 Total filas procesadas: {total}")
        logger.info(f"   ✅ Filas escritas: {self.stats.rows_written}")
        logger.info(f"   ⏭️  Filas saltadas: {self.stats.rows_skipped}")

        if self.stats.rows_skipped > 0:
            logger.info("   📋 Detalle de filas saltadas:")
            for reason, count in self.stats.skip_reasons.items():
                if count > 0:
                    percentage = (count / self.stats.rows_skipped) * 100
                    logger.info(f"      - {reason.value}: {count} ({percentage:.1f}%)")

        # Advertencia si no se escribió ninguna fila
        if self.stats.rows_written == 0:
            logger.warning(
                "⚠️  No se escribió ninguna fila de datos. "
                "Verifique el archivo de entrada y el delimitador configurado."
            )

        # Advertencia si se saltaron muchas filas
        if total > 0:
            skip_ratio = self.stats.rows_skipped / total
            if skip_ratio > 0.5:
                logger.warning(
                    f"⚠️  Se saltó más del 50% de las filas ({skip_ratio:.1%}). "
                    f"Verifique la configuración del delimitador."
                )


def main():
    """Función principal para ejecución desde línea de comandos."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Filtro de líneas para archivos CSV (preserva formato original)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s input.csv output.csv
  %(prog)s input.csv output.csv -d ","
  %(prog)s input.csv output.csv --overwrite --verbose
  %(prog)s input.csv output.csv --no-strict
  %(prog)s input.csv output.csv --no-atomic

IMPORTANTE:
  Este limpiador NO reformatea el CSV. Lee líneas del archivo original
  y escribe las líneas válidas EXACTAMENTE como las encontró, preservando
  comillas, espacios, y cualquier otro formato. Solo actúa como filtro
  de líneas problemáticas.

Códigos de salida:
  0 - Éxito
  1 - Error de validación o procesamiento
  2 - Error de argumentos
  130 - Interrumpido por usuario (Ctrl+C)
        """,
    )

    parser.add_argument("input_file", help="Archivo CSV de entrada")
    parser.add_argument("output_file", help="Archivo CSV de salida")
    parser.add_argument(
        "-d",
        "--delimiter",
        default=";",
        help='Delimitador del CSV (por defecto: ";")',
    )
    parser.add_argument(
        "-e",
        "--encoding",
        default="utf-8",
        help="Codificación del archivo (por defecto: utf-8)",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Sobrescribir archivo de salida si existe",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="No validar estrictamente el número de delimitadores",
    )
    parser.add_argument(
        "--no-atomic",
        action="store_true",
        help="No usar escritura atómica (archivo temporal + rename)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Modo verbose para debugging",
    )

    args = parser.parse_args()

    try:
        cleaner = CSVCleaner(
            input_path=args.input_file,
            output_path=args.output_file,
            delimiter=args.delimiter,
            encoding=args.encoding,
            overwrite=args.overwrite,
            strict_mode=not args.no_strict,
            verbose=args.verbose,
            use_atomic_write=not args.no_atomic,
        )

        stats = cleaner.clean()

        # Código de salida basado en resultados
        if stats.rows_written == 0:
            logger.warning("Saliendo con código 1: no se escribieron filas")
            sys.exit(1)

        sys.exit(0)

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"❌ Error de validación: {e}")
        sys.exit(1)
    except PermissionError as e:
        logger.error(f"❌ Error de permisos: {e}")
        sys.exit(1)
    except IOError as e:
        logger.error(f"❌ Error de I/O: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Error inesperado: {type(e).__name__}: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()