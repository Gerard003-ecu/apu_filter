"""
app/adapters/mic_vectors.py

Adaptadores de Vectores de Capacidad para la MIC.
Transforman 'Intenciones' (Diccionarios) en llamadas a los Motores Físicos y Tácticos.

Cada función es un morfismo: Dict → Componente Nuclear → Dict
"""
import logging
from typing import Dict, Any, List

from app.flux_condenser import DataFluxCondenser, CondenserConfig
from app.report_parser_crudo import ReportParserCrudo
from app.apu_processor import APUProcessor
from app.schemas import Stratum

logger = logging.getLogger(__name__)


# ==============================================================================
# VECTOR FÍSICO 1: ESTABILIZACIÓN DE FLUJO (FluxCondenser)
# ==============================================================================
def vector_stabilize_flux(file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vector de Nivel PHYSICS.
    Invoca al DataFluxCondenser para estabilizar la ingesta de un archivo.
    """
    try:
        condenser_conf = CondenserConfig(
            system_capacitance=config.get("system_capacitance", 5000.0),
            system_inductance=config.get("system_inductance", 2.0),
            base_resistance=config.get("base_resistance", 10.0)
        )

        condenser = DataFluxCondenser(
            config=config,
            profile=config.get("file_profile", {}),
            condenser_config=condenser_conf
        )

        df_stabilized = condenser.stabilize(file_path)
        physics_report = condenser.get_physics_report()

        return {
            "success": True,
            "data": df_stabilized.to_dict("records"),
            "physics_metrics": physics_report,
            "stratum": Stratum.PHYSICS
        }

    except Exception as e:
        logger.error(f"Fallo en vector 'stabilize_flux': {e}")
        return {"success": False, "error": str(e)}


# ==============================================================================
# VECTOR FÍSICO 2: PARSING TOPOLÓGICO (ReportParserCrudo)
# ==============================================================================
def vector_parse_raw_structure(file_path: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vector de Nivel PHYSICS.
    Utiliza ReportParserCrudo para validar la topología del archivo.
    Extrae registros crudos y cache de parsing para inyección en el estrato táctico.
    """
    try:
        parser = ReportParserCrudo(file_path, profile=profile)
        raw_records = parser.parse_to_raw()
        cache = parser.get_parse_cache()

        return {
            "success": True,
            "raw_records": raw_records,
            "parse_cache": cache,
            "validation_stats": parser.validation_stats.__dict__,
            "stratum": Stratum.PHYSICS
        }
    except Exception as e:
        logger.error(f"Fallo en vector 'parse_raw_structure': {e}")
        return {"success": False, "error": str(e)}


# ==============================================================================
# VECTOR TÁCTICO: ESTRUCTURACIÓN LÓGICA (APUProcessor)
# ==============================================================================
def vector_structure_logic(
    raw_records: List[Dict],
    parse_cache: Dict,
    config: Dict
) -> Dict[str, Any]:
    """
    Vector de Nivel TACTICS.
    Transforma registros crudos en estructuras de costos validadas.
    Recibe datos pre-procesados por PHYSICS (inversión de dependencia).
    """
    try:
        processor = APUProcessor(config=config, parse_cache=parse_cache)
        processor.raw_records = raw_records

        df_processed = processor.process_all()

        return {
            "success": True,
            "processed_data": df_processed.to_dict("records"),
            "quality_report": processor.get_quality_report(),
            "stratum": Stratum.TACTICS
        }
    except Exception as e:
        logger.error(f"Fallo en vector 'structure_logic': {e}")
        return {"success": False, "error": str(e)}
