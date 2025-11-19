import logging

from lark import Lark
from lark.exceptions import LarkError

logger = logging.getLogger(__name__)

def diagnose_grammar_mismatches(
    csv_file: str,
    grammar: str,
    output_file: str = "grammar_diagnosis.txt"
):
    """
    Diagnóstica líneas que fallan parsing Lark para ajustar gramática.

    Args:
        csv_file: Archivo a analizar.
        grammar: Gramática Lark.
        output_file: Archivo de salida.
    """

    parser = Lark(grammar, start='line', parser='lalr')

    with open(csv_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    failed_lines = []

    for idx, line in enumerate(lines, 1):
        # Saltar encabezados
        if "UNIDAD:" in line or "ITEM:" in line:
            continue

        # Intentar parsing
        try:
            parser.parse(line)
        except LarkError as e:
            fields = line.split(";")
            empty_positions = [i for i, f in enumerate(fields) if not f.strip()]

            failed_lines.append({
                "line_num": idx,
                "line": line,
                "error": str(e),
                "fields": fields,
                "fields_count": len(fields),
                "empty_field_positions": empty_positions,
            })

    # Generar reporte
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("DIAGNÓSTICO DE INCOMPATIBILIDAD GRAMÁTICA-DATOS\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Total líneas analizadas: {len(lines)}\n")
        f.write(f"Líneas que fallan Lark: {len(failed_lines)}\n")
        if len(lines) > 0:
            f.write(f"Tasa de fallo: {len(failed_lines)/len(lines)*100:.2f}%\n\n")

        # Análisis de patrones
        empty_field_count = sum(
            1 for fl in failed_lines
            if any(not f.strip() for f in fl["fields"])
        )

        f.write(f"Líneas con campos vacíos: {empty_field_count}\n\n")

        f.write("MUESTRAS DE LÍNEAS FALLIDAS:\n")
        f.write("-" * 100 + "\n")

        for fl in failed_lines[:20]:  # Primeras 20
            f.write(f"\nLínea {fl['line_num']}:\n")
            f.write(f"  Error: {fl['error']}\n")
            f.write(f"  Campos: {fl['fields_count']}\n")
            f.write(f"  Contenido: {fl['line']}\n")
            f.write(f"  Campos: {fl['fields']}\n")

            # Detectar campos vacíos
            empty_positions = [
                i for i, f in enumerate(fl["fields"])
                if not f.strip()
            ]
            if empty_positions:
                f.write(f"  ⚠️  Campos vacíos en posiciones: {empty_positions}\n")

    logger.info(f"✓ Diagnóstico guardado en: {output_file}")
    return failed_lines

if __name__ == '__main__':
    import os
    import sys
    # Asegurar que el directorio raíz del proyecto esté en el PYTHONPATH
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from app.apu_processor import APU_GRAMMAR

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_grammar.py <path_to_apu_file>")
        sys.exit(1)

    file_to_diagnose = sys.argv[1]
    diagnose_grammar_mismatches(file_to_diagnose, APU_GRAMMAR)
