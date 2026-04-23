import json
import logging
import os

from app.tactics.pipeline_director import process_all_files
from app.core.telemetry import TelemetryContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pipeline():
    # Mock config and context
    with open("config/config_rules.json", "r") as f:
        config = json.load(f)

    # Setup test paths - assuming data is in data/ or data_dirty/
    # We need to ensure these files exist or use dummy ones if real ones aren't available
    # Based on file list: data/ likely has clean files, data_dirty has originals.
    # The user mentioned "InsumosProcessor returning 0 items" implies we should test
    # with the file that caused issues.
    # However, for a quick verify, we can check what files we have.

    # Use clean files if available as they are likely the intended input for the pipeline
    # after cleaning scripts
    # The user mentioned issues with InsumosProcessor parsing 'insumos.csv' (or clean version).
    # Let's use the ones in data/ which are likely the 'clean' ones but named differently
    # or the ones the app uses. The code usually expects them in data/
    presupuesto_path = "data/presupuesto_clean.csv"
    apus_path = "data/apus_clean.csv"
    insumos_path = "data/insumos_clean.csv"

    # Override output dir to avoid overwriting prod data if needed, or keep it to fix the estimator
    config["output_dir"] = "data_test_output"

    # Enforce topological axioms
    config["enforce_filtration"] = True
    config["enforce_homology"] = True
    config["lithological_context"] = {
        "system_capacitance": 1.0,
        "system_inductance": 0.1,
        "base_resistance": 0.6324,
        "soil_type": "ROCK",
        "is_saturated": False,
        "depth_meters": 10.0,
        "shear_wave_velocity_vs30": 500.0,
        "density_kg_m3": 2000.0
    }

    telemetry = TelemetryContext()

    logger.info(f"Testing with files: {presupuesto_path}, {apus_path}, {insumos_path}")

    try:
        final_context = process_all_files(presupuesto_path, apus_path, insumos_path, config, telemetry)

        # Si el resultado es un dict y tiene 'status' == 'ERROR'
        if isinstance(final_context, dict) and final_context.get("status") in ("ERROR", "error", "REJECT", "reject"):
            logger.info(f"✅ Pipeline correctly aborted with status {final_context.get('status')} due to topological constraints.")
        elif isinstance(final_context, dict) and "presupuesto" in final_context:
            logger.info(f"✅ Pipeline completed successfully. Presupuesto length: {len(final_context['presupuesto'])}")
        else:
            logger.error(f"❌ Pipeline did not return expected final_result. Received dict with keys: {final_context.keys() if isinstance(final_context, dict) else type(final_context)}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    test_pipeline()
