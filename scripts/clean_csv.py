# scripts/clean_csv.py
import csv
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class SkipReason(Enum):
    """Razones por las que una fila puede ser saltada."""
    EMPTY = "Línea vacía"
    COMMENT = "Comentario"
    INCONSISTENT_COLUMNS = "Número inconsistente de columnas"
    MALFORMED = "Formato inválido"


@dataclass
class CleaningStats:
    """Estadísticas del proceso de limpieza."""
    rows_written: int = 0
    rows_skipped: int = 0
    skip_reasons: Dict[SkipReason, int] = None
    
    def __post_init__(self):
        if self.skip_reasons is None:
            self.skip_reasons = {reason: 0 for reason in SkipReason}
    
    def record_skip(self, reason: SkipReason):
        """Registra una fila saltada con su razón."""
        self.rows_skipped += 1
        self.skip_reasons[reason] += 1
    
    def record_written(self):
        """Registra una fila escrita exitosamente."""
        self.rows_written += 1


class CSVCleaner:
    """
    Limpiador robusto de archivos CSV con validaciones exhaustivas.
    """
    
    # Delimitadores válidos comunes
    VALID_DELIMITERS = {';', ',', '\t', '|'}
    
    # Tamaño máximo de archivo (100MB por defecto)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        delimiter: str = ';',
        encoding: str = 'utf-8',
        overwrite: bool = False,
        strict_mode: bool = True,
        verbose: bool = False
    ):
        """
        Inicializa el limpiador de CSV.
        
        Args:
            input_path: Ruta al archivo CSV de entrada
            output_path: Ruta al archivo CSV de salida
            delimiter: Delimitador del CSV (por defecto ';')
            encoding: Codificación del archivo (por defecto 'utf-8')
            overwrite: Si True, sobrescribe archivo de salida existente
            strict_mode: Si True, valida estrictamente el número de columnas
            verbose: Si True, muestra información detallada de depuración
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.delimiter = delimiter
        self.encoding = encoding
        self.overwrite = overwrite
        self.strict_mode = strict_mode
        self.verbose = verbose
        self.stats = CleaningStats()
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
    
    def _validate_inputs(self) -> None:
        """
        Valida los parámetros de entrada antes de procesar.
        
        Raises:
            ValueError: Si alguna validación falla
            FileNotFoundError: Si el archivo de entrada no existe
        """
        # Validar archivo de entrada
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"El archivo de entrada no existe: {self.input_path}"
            )
        
        if not self.input_path.is_file():
            raise ValueError(
                f"La ruta de entrada no es un archivo: {self.input_path}"
            )
        
        # Validar tamaño del archivo
        file_size = self.input_path.stat().st_size
        if file_size == 0:
            raise ValueError(
                f"El archivo de entrada está vacío: {self.input_path}"
            )
        
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"El archivo excede el tamaño máximo permitido "
                f"({self.MAX_FILE_SIZE / 1024 / 1024:.2f} MB): {file_size / 1024 / 1024:.2f} MB"
            )
        
        # Validar que entrada y salida no sean el mismo archivo
        if self.input_path.resolve() == self.output_path.resolve():
            raise ValueError(
                "El archivo de entrada y salida no pueden ser el mismo"
            )

        # Validar archivo de salida
        if self.output_path.exists() and not self.overwrite:
            raise ValueError(
                f"El archivo de salida ya existe: {self.output_path}. "
                f"Use overwrite=True para sobrescribir."
            )
        
        # Validar que se pueda escribir en el directorio de salida
        output_dir = self.output_path.parent
        if not output_dir.exists():
            raise ValueError(
                f"El directorio de salida no existe: {output_dir}"
            )
        
        if not output_dir.is_dir():
            raise ValueError(
                f"La ruta padre de salida no es un directorio: {output_dir}"
            )
        
        # Validar delimitador
        if not self.delimiter:
            raise ValueError("El delimitador no puede estar vacío")
        
        if len(self.delimiter) != 1:
            raise ValueError(
                f"El delimitador debe ser un solo carácter: '{self.delimiter}'"
            )
        
        if self.delimiter not in self.VALID_DELIMITERS:
            logger.warning(
                f"Delimitador inusual detectado: '{self.delimiter}'. "
                f"Delimitadores comunes: {self.VALID_DELIMITERS}"
            )
        
        logger.debug("✅ Validaciones de entrada completadas exitosamente")
    
    def _read_header(self, reader: csv.reader) -> Optional[List[str]]:
        """
        Lee y valida el encabezado del CSV.
        
        Args:
            reader: Reader de CSV
            
        Returns:
            Lista con los campos del encabezado o None si no hay
            
        Raises:
            ValueError: Si el encabezado es inválido
        """
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(
                "El archivo CSV está vacío o solo contiene el encabezado"
            )
        
        # Validar que el header no esté completamente vacío
        if not header or not any(field.strip() for field in header):
            raise ValueError(
                "El encabezado del CSV está vacío o contiene solo espacios"
            )
        
        # Validar que no haya nombres de columna duplicados
        header_stripped = [h.strip() for h in header]
        if len(header_stripped) != len(set(header_stripped)):
            duplicates = [
                h for h in header_stripped 
                if header_stripped.count(h) > 1
            ]
            logger.warning(
                f"⚠️  Encabezados duplicados detectados: {set(duplicates)}"
            )
        
        logger.info(f"✅ Encabezado detectado con {len(header)} columnas")
        if self.verbose:
            logger.debug(f"Columnas: {header}")
        
        return header
    
    def _should_skip_row(
        self, 
        row: List[str], 
        num_columns: int, 
        line_num: int
    ) -> Optional[SkipReason]:
        """
        Determina si una fila debe ser saltada y por qué razón.
        
        Args:
            row: Fila a evaluar
            num_columns: Número esperado de columnas
            line_num: Número de línea (para logging)
            
        Returns:
            SkipReason si debe saltarse, None si es válida
        """
        # Validar que row no sea None o vacío
        if not row:
            if self.verbose:
                logger.debug(f"Línea {line_num}: Fila vacía (None o lista vacía)")
            return SkipReason.EMPTY
        
        # 1. Ignorar líneas completamente en blanco
        if not any(field.strip() for field in row):
            if self.verbose:
                logger.debug(f"Línea {line_num}: Todos los campos están vacíos")
            return SkipReason.EMPTY
        
        # 2. Ignorar líneas de comentario (comienzan con '#')
        first_field = row[0].strip() if row else ""
        if first_field.startswith('#'):
            if self.verbose:
                logger.debug(f"Línea {line_num}: Comentario detectado")
            return SkipReason.COMMENT
        
        # 3. Validar consistencia de columnas (si está en modo estricto)
        if self.strict_mode and len(row) != num_columns:
            if self.verbose:
                logger.debug(
                    f"Línea {line_num}: Esperadas {num_columns} columnas, "
                    f"encontradas {len(row)}"
                )
            return SkipReason.INCONSISTENT_COLUMNS
        
        return None
    
    def clean(self) -> CleaningStats:
        """
        Ejecuta el proceso de limpieza del CSV.
        
        Returns:
            CleaningStats con las estadísticas del proceso
            
        Raises:
            ValueError: Si hay errores de validación
            IOError: Si hay errores de lectura/escritura
        """
        # Validar parámetros de entrada
        self._validate_inputs()
        
        logger.info(f"🧹 Iniciando limpieza: {self.input_path} -> {self.output_path}")
        logger.info(f"   Delimitador: '{self.delimiter}'")
        logger.info(f"   Encoding: {self.encoding}")
        logger.info(f"   Modo estricto: {self.strict_mode}")
        
        try:
            with open(
                self.input_path, 
                'r', 
                encoding=self.encoding, 
                errors='replace'
            ) as infile, \
                 open(
                     self.output_path, 
                     'w', 
                     encoding=self.encoding, 
                     newline=''
                 ) as outfile:
                
                reader = csv.reader(infile, delimiter=self.delimiter)
                writer = csv.writer(outfile, delimiter=self.delimiter)
                
                # Leer y validar encabezado
                header = self._read_header(reader)
                writer.writerow(header)
                num_columns = len(header)
                
                # Procesar filas
                for line_num, row in enumerate(reader, start=2):
                    skip_reason = self._should_skip_row(row, num_columns, line_num)
                    
                    if skip_reason:
                        self.stats.record_skip(skip_reason)
                        continue
                    
                    try:
                        writer.writerow(row)
                        self.stats.record_written()
                    except Exception as e:
                        logger.error(
                            f"Error al escribir línea {line_num}: {e}"
                        )
                        self.stats.record_skip(SkipReason.MALFORMED)
            
            self._print_summary()
            return self.stats
            
        except PermissionError as e:
            raise IOError(
                f"Permiso denegado al acceder a los archivos: {e}"
            )
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Error de codificación. Intente con un encoding diferente: {e}"
            )
        except csv.Error as e:
            raise ValueError(
                f"Error al parsear el CSV: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error inesperado durante la limpieza: {e}"
            )
    
    def _print_summary(self) -> None:
        """Imprime un resumen detallado del proceso de limpieza."""
        logger.info("🎉 Limpieza completada exitosamente")
        logger.info(f"   ✅ Filas escritas: {self.stats.rows_written}")
        logger.info(f"   ⏭️  Filas saltadas: {self.stats.rows_skipped}")
        
        if self.stats.rows_skipped > 0:
            logger.info("   📊 Detalle de filas saltadas:")
            for reason, count in self.stats.skip_reasons.items():
                if count > 0:
                    logger.info(f"      - {reason.value}: {count}")
        
        # Advertencia si no se escribió ninguna fila
        if self.stats.rows_written == 0:
            logger.warning(
                "⚠️  No se escribió ninguna fila de datos. "
                "Verifique el archivo de entrada."
            )


def main():
    """Función principal para ejecución desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Limpia archivos CSV eliminando filas problemáticas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s input.csv output.csv
  %(prog)s input.csv output.csv -d ","
  %(prog)s input.csv output.csv --overwrite --verbose
  %(prog)s input.csv output.csv --no-strict
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Archivo CSV de entrada'
    )
    parser.add_argument(
        'output_file',
        help='Archivo CSV de salida'
    )
    parser.add_argument(
        '-d', '--delimiter',
        default=';',
        help='Delimitador del CSV (por defecto: ";")'
    )
    parser.add_argument(
        '-e', '--encoding',
        default='utf-8',
        help='Codificación del archivo (por defecto: utf-8)'
    )
    parser.add_argument(
        '-o', '--overwrite',
        action='store_true',
        help='Sobrescribir archivo de salida si existe'
    )
    parser.add_argument(
        '--no-strict',
        action='store_true',
        help='No validar estrictamente el número de columnas'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Modo verbose para debugging'
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
            verbose=args.verbose
        )
        
        cleaner.clean()
        sys.exit(0)
        
    except (ValueError, FileNotFoundError, IOError) as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()