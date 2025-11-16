# scripts/clean_csv.py
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, List, TextIO
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
    INCONSISTENT_DELIMITERS = "Número inconsistente de delimitadores"
    WHITESPACE_ONLY = "Solo espacios en blanco"


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
    Filtro de líneas para archivos CSV que preserva el formato original.
    
    Este limpiador NO reformatea el CSV, solo actúa como filtro:
    - Lee líneas del archivo original
    - Decide si cada línea es válida
    - Escribe las líneas válidas EXACTAMENTE como las encontró
    
    Esto evita problemas de re-formatting que pueden romper parsers 
    posteriores que esperan un formato específico.
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
            strict_mode: Si True, valida estrictamente el número de delimitadores
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
        self.expected_delimiter_count: Optional[int] = None
        
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
        
        # Validar que entrada y salida no sean el mismo archivo
        if self.input_path.resolve() == self.output_path.resolve():
            raise ValueError(
                "El archivo de entrada y salida no pueden ser el mismo"
            )
        
        logger.debug("✅ Validaciones de entrada completadas exitosamente")
    
    def _count_delimiters(self, line: str) -> int:
        """
        Cuenta el número de delimitadores en una línea.
        
        Args:
            line: Línea a analizar
            
        Returns:
            Número de delimitadores encontrados
        """
        return line.count(self.delimiter)
    
    def _is_empty_line(self, line: str) -> bool:
        """
        Determina si una línea está vacía o contiene solo espacios.
        
        Args:
            line: Línea a evaluar
            
        Returns:
            True si la línea está vacía o solo contiene espacios
        """
        return not line.strip()
    
    def _is_comment_line(self, line: str) -> bool:
        """
        Determina si una línea es un comentario (comienza con #).
        
        Args:
            line: Línea a evaluar
            
        Returns:
            True si la línea es un comentario
        """
        return line.strip().startswith('#')
    
    def _is_all_whitespace_fields(self, line: str) -> bool:
        """
        Determina si una línea contiene solo campos vacíos o con espacios.
        Por ejemplo: ";;;" o "  ;  ;  "
        
        Args:
            line: Línea a evaluar
            
        Returns:
            True si todos los campos están vacíos o solo contienen espacios
        """
        if not line.strip():
            return True
        
        # Dividir por el delimitador y verificar si todos los campos están vacíos
        fields = line.split(self.delimiter)
        return all(not field.strip() for field in fields)
    
    def _should_skip_line(self, line: str, line_num: int) -> Optional[SkipReason]:
        """
        Determina si una línea debe ser saltada y por qué razón.
        
        Args:
            line: Línea a evaluar (sin el salto de línea final)
            line_num: Número de línea (para logging)
            
        Returns:
            SkipReason si debe saltarse, None si es válida
        """
        # 1. Ignorar líneas completamente vacías
        if self._is_empty_line(line):
            if self.verbose:
                logger.debug(f"Línea {line_num}: Línea vacía")
            return SkipReason.EMPTY
        
        # 2. Ignorar líneas de comentario (comienzan con '#')
        if self._is_comment_line(line):
            if self.verbose:
                logger.debug(f"Línea {line_num}: Comentario detectado")
            return SkipReason.COMMENT
        
        # 3. Ignorar líneas con solo espacios en blanco en todos los campos
        if self._is_all_whitespace_fields(line):
            if self.verbose:
                logger.debug(f"Línea {line_num}: Solo espacios en blanco")
            return SkipReason.WHITESPACE_ONLY
        
        # 4. Validar consistencia de delimitadores (si está en modo estricto)
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
        if self._is_empty_line(header_line):
            raise ValueError(
                "El encabezado del CSV está vacío"
            )
        
        if self._is_all_whitespace_fields(header_line):
            raise ValueError(
                "El encabezado del CSV contiene solo espacios en blanco"
            )
        
        # Contar delimitadores en el header para validación futura
        self.expected_delimiter_count = self._count_delimiters(header_line)
        
        # Extraer nombres de columnas para logging
        column_names = header_line.split(self.delimiter)
        num_columns = len(column_names)
        
        logger.info(f"✅ Encabezado detectado con {num_columns} columnas")
        
        if self.verbose:
            logger.debug(f"Delimitadores en encabezado: {self.expected_delimiter_count}")
            logger.debug(f"Columnas: {column_names}")
        
        # Advertir sobre encabezados duplicados
        column_names_stripped = [col.strip() for col in column_names]
        if len(column_names_stripped) != len(set(column_names_stripped)):
            duplicates = [
                col for col in column_names_stripped 
                if column_names_stripped.count(col) > 1
            ]
            logger.warning(
                f"⚠️  Encabezados duplicados detectados: {set(duplicates)}"
            )
    
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
            IOError: Si hay errores de lectura/escritura
        """
        # Validar parámetros de entrada
        self._validate_inputs()
        
        logger.info(f"🧹 Iniciando limpieza: {self.input_path} -> {self.output_path}")
        logger.info(f"   Delimitador: '{self.delimiter}'")
        logger.info(f"   Encoding: {self.encoding}")
        logger.info(f"   Modo estricto: {self.strict_mode}")
        logger.info(f"   Modo filtro: PRESERVA FORMATO ORIGINAL")
        
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
                     newline=''  # Importante: preservar los saltos de línea originales
                 ) as outfile:
                
                # Procesar encabezado
                header_line = infile.readline()
                
                if not header_line:
                    raise ValueError(
                        "El archivo CSV está vacío o no contiene encabezado"
                    )
                
                # Remover salto de línea para validación, pero guardarlo para escritura
                line_ending = self._detect_line_ending(header_line)
                header_clean = header_line.rstrip('\r\n')
                
                self._process_header(header_clean)
                
                # Escribir encabezado EXACTAMENTE como se leyó
                outfile.write(header_line)
                
                # Procesar resto de líneas
                line_num = 2  # Empezamos en 2 porque la línea 1 es el header
                
                for raw_line in infile:
                    # Remover salto de línea solo para validación
                    line_clean = raw_line.rstrip('\r\n')
                    
                    # Si la línea está completamente vacía (EOF), saltar
                    if not raw_line:
                        continue
                    
                    skip_reason = self._should_skip_line(line_clean, line_num)
                    
                    if skip_reason:
                        self.stats.record_skip(skip_reason)
                        line_num += 1
                        continue
                    
                    # Escribir la línea EXACTAMENTE como se leyó
                    # (incluyendo su salto de línea original)
                    outfile.write(raw_line)
                    self.stats.record_written()
                    line_num += 1
            
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
        except Exception as e:
            raise RuntimeError(
                f"Error inesperado durante la limpieza: {e}"
            )
    
    def _detect_line_ending(self, line: str) -> str:
        """
        Detecta el tipo de salto de línea usado.
        
        Args:
            line: Línea a analizar
            
        Returns:
            '\r\n' para Windows, '\n' para Unix
        """
        if line.endswith('\r\n'):
            return '\r\n'
        elif line.endswith('\n'):
            return '\n'
        elif line.endswith('\r'):
            return '\r'
        return '\n'  # Default
    
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
        description='Filtro de líneas para archivos CSV (preserva formato original)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s input.csv output.csv
  %(prog)s input.csv output.csv -d ","
  %(prog)s input.csv output.csv --overwrite --verbose
  %(prog)s input.csv output.csv --no-strict

IMPORTANTE:
  Este limpiador NO reformatea el CSV. Lee líneas del archivo original
  y escribe las líneas válidas EXACTAMENTE como las encontró, preservando
  comillas, espacios, y cualquier otro formato. Solo actúa como filtro
  de líneas problemáticas.
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
        help='No validar estrictamente el número de delimitadores'
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