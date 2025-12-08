import logging
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

from lark import Lark
from lark.exceptions import LarkError, UnexpectedEOF, UnexpectedCharacters

logger = logging.getLogger(__name__)

def diagnose_grammar_mismatches(
    csv_file: str,
    grammar: str,
    output_file: str = "grammar_diagnosis.txt",
    max_sample_lines: int = 20,
    skip_header_patterns: Optional[List[str]] = None,
    encoding: str = "utf-8"
) -> List[Dict[str, Any]]:
    """
    Diagnóstica líneas que fallan parsing Lark para ajustar gramática.

    Args:
        csv_file: Archivo a analizar.
        grammar: Gramática Lark.
        output_file: Archivo de salida.
        max_sample_lines: Número máximo de líneas fallidas a mostrar en el reporte.
        skip_header_patterns: Patrones para identificar líneas de encabezado a saltar.
        encoding: Codificación del archivo CSV.

    Returns:
        Lista de diccionarios con información de las líneas fallidas.

    Raises:
        FileNotFoundError: Si el archivo CSV no existe.
        ValueError: Si la gramática es inválida o vacía.
        RuntimeError: Si hay problemas al crear el parser.
    """
    # Validación de parámetros
    if not csv_file or not isinstance(csv_file, str):
        raise ValueError("csv_file debe ser una cadena no vacía")

    if not grammar or not isinstance(grammar, str):
        raise ValueError("grammar debe ser una cadena no vacía")

    if not output_file or not isinstance(output_file, str):
        raise ValueError("output_file debe ser una cadena no vacía")

    if not isinstance(max_sample_lines, int) or max_sample_lines <= 0:
        raise ValueError("max_sample_lines debe ser un entero positivo")

    if skip_header_patterns is not None and not isinstance(skip_header_patterns, list):
        raise ValueError("skip_header_patterns debe ser una lista de cadenas o None")

    if not isinstance(encoding, str):
        raise ValueError("encoding debe ser una cadena")

    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"El archivo CSV no existe: {csv_file}")

    if not csv_path.is_file():
        raise FileNotFoundError(f"La ruta no es un archivo: {csv_file}")

    # Validar gramática creando un parser temporal
    try:
        temp_parser = Lark(grammar, start="line", parser="lalr")
        del temp_parser
    except Exception as e:
        raise ValueError(f"Gramática Lark inválida: {str(e)}")

    # Configurar patrones de encabezado por defecto
    if skip_header_patterns is None:
        skip_header_patterns = ["UNIDAD:", "ITEM:"]

    # Crear el parser
    try:
        parser = Lark(grammar, start="line", parser="lalr")
    except Exception as e:
        raise RuntimeError(f"Error al crear el parser Lark: {str(e)}")

    # Leer archivo CSV
    try:
        with open(csv_file, "r", encoding=encoding, newline='') as f:
            raw_lines = f.readlines()
    except UnicodeDecodeError:
        # Intentar con codificación alternativa
        try:
            with open(csv_file, "r", encoding="latin-1", newline='') as f:
                raw_lines = f.readlines()
            logger.warning(f"Archivo {csv_file} leído con codificación latin-1 en lugar de {encoding}")
        except UnicodeDecodeError:
            raise UnicodeDecodeError(f"No se pudo leer el archivo {csv_file} con las codificaciones UTF-8 o Latin-1")
    except IOError as e:
        raise IOError(f"Error al leer el archivo CSV: {str(e)}")

    # Filtrar y limpiar líneas
    processed_lines = []
    for idx, line in enumerate(raw_lines, 1):
        stripped_line = line.rstrip('\n\r')  # Eliminar solo saltos de línea, mantener espacios en blanco dentro de la línea
        if stripped_line.strip():  # Solo incluir líneas no vacías
            processed_lines.append((idx, stripped_line))
    
    failed_lines = []
    analyzed_count = 0

    for original_idx, line in processed_lines:
        # Verificar si es línea de encabezado
        is_header = any(pattern.upper() in line.upper() for pattern in skip_header_patterns)
        if is_header:
            continue

        analyzed_count += 1

        # Intentar parsing
        try:
            parse_result = parser.parse(line)
            # Verificar si el parseo fue exitoso (opcional: validar estructura mínima)
            if parse_result is None:
                logger.warning(f"Línea {original_idx} parseada pero resultado es None")
        except (LarkError, UnexpectedEOF, UnexpectedCharacters) as e:
            fields = line.split(";")

            # Identificar campos vacíos y con solo espacios
            empty_positions = []
            whitespace_only_positions = []
            for i, field in enumerate(fields):
                if not field.strip():
                    if not field:  # Campo completamente vacío
                        empty_positions.append(i)
                    else:  # Campo con solo espacios
                        whitespace_only_positions.append(i)

            failed_lines.append({
                "line_num": original_idx,
                "line": line,
                "error": str(e),
                "error_type": type(e).__name__,
                "fields": fields,
                "fields_count": len(fields),
                "empty_field_positions": empty_positions,
                "whitespace_only_positions": whitespace_only_positions,
                "line_length": len(line),
                "field_lengths": [len(f) for f in fields],
                "has_trailing_semicolon": line.endswith(';'),
                "has_leading_semicolon": line.startswith(';'),
            })
        except Exception as e:
            logger.error(f"Error inesperado al parsear línea {original_idx}: {str(e)}")
            # Añadir como línea fallida con error genérico
            fields = line.split(";")
            empty_positions = [i for i, f in enumerate(fields) if not f.strip()]

            failed_lines.append({
                "line_num": original_idx,
                "line": line,
                "error": f"Error inesperado: {str(e)}",
                "error_type": "UnexpectedError",
                "fields": fields,
                "fields_count": len(fields),
                "empty_field_positions": empty_positions,
                "whitespace_only_positions": [],
                "line_length": len(line),
                "field_lengths": [len(f) for f in fields],
                "has_trailing_semicolon": line.endswith(';'),
                "has_leading_semicolon": line.startswith(';'),
            })

    # Generar reporte detallado
    _generate_detailed_report(
        output_file=output_file,
        total_lines=len(processed_lines),
        analyzed_lines=analyzed_count,
        failed_lines=failed_lines,
        max_sample_lines=max_sample_lines,
        encoding_used=encoding
    )

    logger.info(f"✓ Diagnóstico completado. Líneas analizadas: {analyzed_count}, Fallidas: {len(failed_lines)}")
    logger.info(f"✓ Reporte guardado en: {output_file}")

    return failed_lines


def _generate_detailed_report(
    output_file: str,
    total_lines: int,
    analyzed_lines: int,
    failed_lines: List[Dict[str, Any]],
    max_sample_lines: int,
    encoding_used: str
) -> None:
    """
    Genera un reporte detallado de las líneas fallidas.

    Args:
        output_file: Ruta del archivo de salida.
        total_lines: Total de líneas en el archivo.
        analyzed_lines: Número de líneas analizadas (excluyendo encabezados).
        failed_lines: Lista de líneas que fallaron.
        max_sample_lines: Máximo número de líneas de ejemplo a mostrar.
        encoding_used: Codificación usada para leer el archivo.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 120 + "\n")
            f.write("DIAGNÓSTICO DE INCOMPATIBILIDAD GRAMÁTICA-DATOS\n")
            f.write("=" * 120 + "\n\n")

            f.write(f"Archivo analizado: {os.path.basename(output_file.replace('grammar_diagnosis.txt', '')) if 'grammar_diagnosis.txt' in output_file else 'Desconocido'}\n")
            f.write(f"Codificación usada: {encoding_used}\n")
            f.write(f"Total líneas en archivo: {total_lines}\n")
            f.write(f"Líneas analizadas (sin encabezados): {analyzed_lines}\n")
            f.write(f"Líneas que fallan Lark: {len(failed_lines)}\n")

            if analyzed_lines > 0:
                failure_rate = len(failed_lines) / analyzed_lines * 100
                f.write(f"Tasa de fallo: {failure_rate:.2f}%\n\n")
            else:
                f.write("Tasa de fallo: 0% (no se analizaron líneas)\n\n")

            # Análisis estadístico
            if failed_lines:
                _write_statistical_analysis(f, failed_lines)

                f.write("\n" + "MUESTRAS DE LÍNEAS FALLIDAS:\n")
                f.write("-" * 120 + "\n")

                # Mostrar muestras de líneas fallidas
                sample_lines = failed_lines[:max_sample_lines]
                for fl in sample_lines:
                    f.write(f"\nLínea {fl['line_num']}:\n")
                    f.write(f"  Error ({fl['error_type']}): {fl['error']}\n")
                    f.write(f"  Campos: {fl['fields_count']} | Longitud línea: {fl['line_length']}\n")
                    f.write(f"  Contenido: {repr(fl['line'])}\n")

                    if fl['fields_count'] > 0:
                        f.write(f"  Campos (longitudes): {list(zip(range(fl['fields_count']), fl['field_lengths']))}\n")

                    # Mostrar campos vacíos y con solo espacios
                    if fl['empty_field_positions']:
                        f.write(f"  ⚠️  Campos completamente vacíos en posiciones: {fl['empty_field_positions']}\n")
                    if fl['whitespace_only_positions']:
                        f.write(f"  ⚠️  Campos con solo espacios en posiciones: {fl['whitespace_only_positions']}\n")

                    # Mostrar características especiales
                    special_features = []
                    if fl['has_trailing_semicolon']:
                        special_features.append("termina en ;")
                    if fl['has_leading_semicolon']:
                        special_features.append("comienza con ;")
                    if special_features:
                        f.write(f"  💡 Características especiales: {', '.join(special_features)}\n")
            else:
                f.write("✓ No se encontraron líneas fallidas - la gramática es compatible con todos los datos analizados.\n")

    except IOError as e:
        logger.error(f"Error al escribir el archivo de reporte: {str(e)}")
        raise


def _write_statistical_analysis(f, failed_lines: List[Dict[str, Any]]) -> None:
    """
    Escribe análisis estadístico de las líneas fallidas.

    Args:
        f: Archivo de salida.
        failed_lines: Lista de líneas fallidas.
    """
    # Contar tipos de error
    error_type_counts = {}
    for fl in failed_lines:
        err_type = fl['error_type']
        error_type_counts[err_type] = error_type_counts.get(err_type, 0) + 1

    # Contar campos vacíos
    total_empty_fields = sum(len(fl['empty_field_positions']) for fl in failed_lines)
    total_whitespace_fields = sum(len(fl['whitespace_only_positions']) for fl in failed_lines)

    # Contar patrones de fallo comunes
    trailing_semicolon_count = sum(1 for fl in failed_lines if fl['has_trailing_semicolon'])
    leading_semicolon_count = sum(1 for fl in failed_lines if fl['has_leading_semicolon'])

    # Distribución de cantidad de campos
    field_count_distribution = {}
    for fl in failed_lines:
        count = fl['fields_count']
        field_count_distribution[count] = field_count_distribution.get(count, 0) + 1

    # Escribir análisis
    f.write("ANÁLISIS ESTADÍSTICO:\n")
    f.write("-" * 60 + "\n")
    f.write(f"Total de líneas fallidas: {len(failed_lines)}\n")

    if error_type_counts:
        f.write(f"Tipos de error:\n")
        for err_type, count in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {err_type}: {count} ({count/len(failed_lines)*100:.1f}%)\n")

    f.write(f"Campos vacíos totales: {total_empty_fields}\n")
    f.write(f"Campos con solo espacios: {total_whitespace_fields}\n")
    f.write(f"Líneas que terminan en ';': {trailing_semicolon_count}\n")
    f.write(f"Líneas que comienzan con ';': {leading_semicolon_count}\n")

    if field_count_distribution:
        f.write(f"Distribución de cantidad de campos:\n")
        for count, freq in sorted(field_count_distribution.items()):
            f.write(f"  - {count} campos: {freq} líneas\n")

    f.write("\n")


if __name__ == "__main__":
    # Asegurar que el directorio raíz del proyecto esté en el PYTHONPATH
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_grammar.py <path_to_apu_file> [output_file]")
        sys.exit(1)

    file_to_diagnose = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "grammar_diagnosis.txt"

    try:
        from app.apu_processor import APU_GRAMMAR
    except ImportError:
        logger.error("No se pudo importar APU_GRAMMAR desde app.apu_processor")
        # Definir una gramática de ejemplo para pruebas
        APU_GRAMMAR = r"""
        start: line
        line: field (";" field)*
        field: /[^\n;]*/
        %import common.WS
        """
        logger.info("Usando gramática de ejemplo para pruebas")

    try:
        failed_lines = diagnose_grammar_mismatches(
            csv_file=file_to_diagnose,
            grammar=APU_GRAMMAR,
            output_file=output_file
        )
        logger.info(f"Diagnóstico completado. Se encontraron {len(failed_lines)} líneas fallidas.")
    except Exception as e:
        logger.error(f"Error durante el diagnóstico: {str(e)}")
        sys.exit(1)