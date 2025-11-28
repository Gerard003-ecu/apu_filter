"""
Interfaz de herramientas para la Matriz de Interacción Central (MIC).
Actúa como adaptador entre la API REST y los scripts de mantenimiento.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

from scripts.clean_csv import CSVCleaner
from scripts.diagnose_apus_file import APUFileDiagnostic
from scripts.diagnose_insumos_file import InsumosFileDiagnostic
from scripts.diagnose_presupuesto_file import PresupuestoFileDiagnostic

logger = logging.getLogger(__name__)


def diagnose_file(file_path: Union[str, Path], file_type: str) -> Dict[str, Any]:
    """
    Ejecuta el diagnóstico apropiado según el tipo de archivo.

    Args:
        file_path: Ruta al archivo a diagnosticar
        file_type: Tipo de archivo ('apus', 'insumos', 'presupuesto')

    Returns:
        Diccionario con resultados del diagnóstico
    """
    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    try:
        if file_type == "apus":
            diagnostic = APUFileDiagnostic(str(path))
            result = diagnostic.diagnose()
            # APUFileDiagnostic refactorizado ahora tiene to_dict()
            return diagnostic.to_dict()

        elif file_type == "insumos":
            diagnostic = InsumosFileDiagnostic(str(path))
            result = diagnostic.diagnose()
            return diagnostic.to_dict()

        elif file_type == "presupuesto":
            diagnostic = PresupuestoFileDiagnostic(str(path))
            result = diagnostic.diagnose()
            return diagnostic.to_dict()

        else:
            return {
                "success": False,
                "error": f"Unknown file type: {file_type}. Expected: apus, insumos, presupuesto",
            }

    except Exception as e:
        logger.error(f"Error executing diagnosis for {file_type}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def clean_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    delimiter: str = ";",
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Ejecuta la limpieza de un archivo CSV.

    Args:
        input_path: Ruta al archivo sucio
        output_path: Ruta donde guardar el archivo limpio (opcional)
        delimiter: Delimitador esperado
        encoding: Encoding esperado

    Returns:
        Diccionario con estadísticas de limpieza y ruta de salida
    """
    input_p = Path(input_path)
    if not input_p.exists():
        return {"success": False, "error": f"Input file not found: {input_path}"}

    # Si no se da output, crear uno temporal o sufijado
    if not output_path:
        output_p = input_p.with_name(f"{input_p.stem}_clean{input_p.suffix}")
    else:
        output_p = Path(output_path)

    try:
        cleaner = CSVCleaner(
            input_path=str(input_p),
            output_path=str(output_p),
            delimiter=delimiter,
            encoding=encoding,
            overwrite=True,
        )

        stats = cleaner.clean()

        result = stats.to_dict()
        result["success"] = True
        result["output_path"] = str(output_p)
        return result

    except Exception as e:
        logger.error(f"Error executing cleaner: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_telemetry_status(telemetry_context=None) -> Dict[str, Any]:
    """
    Obtiene el estado actual del sistema (Vector de Estado).
    """
    # Si recibimos un contexto activo, lo usamos
    if telemetry_context:
        return telemetry_context.get_business_report()

    # Si no, devolvemos un estado genérico o global si existiera
    # En esta arquitectura, la telemetría suele estar ligada a un request,
    # pero podríamos devolver métricas globales de la app si estuvieran disponibles.
    return {
        "status": "IDLE",
        "message": "No active processing context",
        "system_health": "UNKNOWN",
    }
