import sys
import os
import logging
import json
import pandas as pd
from app.pipeline_director import PipelineDirector
from app.telemetry import TelemetryContext
from app.flux_condenser import ProcessingStats

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
    # The user mentioned "InsumosProcessor returning 0 items" implies we should test with the file that caused issues.
    # However, for a quick verify, we can check what files we have.

    # Use clean files if available as they are likely the intended input for the pipeline after cleaning scripts
    # The user mentioned issues with InsumosProcessor parsing 'insumos.csv' (or clean version).
    # Let's use the ones in data/ which are likely the 'clean' ones but named differently or the ones the app uses.
    # The code usually expects them in data/
    presupuesto_path = "data/presupuesto_clean.csv"
    apus_path = "data/apus_clean.csv"
    insumos_path = "data/insumos_clean.csv"

    # Override output dir to avoid overwriting prod data if needed, or keep it to fix the estimator
    config["output_dir"] = "data_test_output"

    telemetry = TelemetryContext()
    director = PipelineDirector(config, telemetry)

    initial_context = {
        "presupuesto_path": presupuesto_path,
        "apus_path": apus_path,
        "insumos_path": insumos_path,
    }

    logger.info(f"Testing with files: {initial_context}")

    try:
        final_context = director.execute(initial_context)

        # Verification 1: Insumos
        df_insumos = final_context.get("df_insumos")
        if df_insumos is not None and not df_insumos.empty:
            logger.info(f"✅ Insumos loaded: {len(df_insumos)} rows")
            print(df_insumos.head())
        else:
            logger.error("❌ Insumos DataFrame is empty!")

        # Verification 2: APU Classification
        df_final = final_context.get("df_final")
        if df_final is not None and not df_final.empty:
            if "tipo_apu" in df_final.columns:
                counts = df_final["tipo_apu"].value_counts()
                logger.info(f"✅ APU Types Distribution:\n{counts}")
                if len(counts) > 1:
                     logger.info("✅ Classification logic seems to be working (more than one type found).")
                else:
                     logger.warning("⚠️ Only one APU type found. Check if this is expected for this dataset.")
            else:
                logger.error("❌ 'tipo_apu' column missing in final dataframe")
        else:
             logger.error("❌ Final DataFrame is empty!")

        # Verification 3: Output files
        output_dir = config["output_dir"]
        processed_file = os.path.join(output_dir, config.get("processed_apus_file", "processed_apus.json"))

        # We need to manually save it as the director only returns context,
        # process_all_files handles saving usually.
        # But wait, the Director itself doesn't save, the wrapper function does.
        # I should check if I need to call process_all_files or just verify the context.
        # The Step 'BuildOutputStep' creates 'final_result' in context.

        final_result = final_context.get("final_result")
        if final_result:
             processed_apus = final_result.get("processed_apus")
             if processed_apus:
                 logger.info(f"✅ processed_apus in result: {len(processed_apus)} items")
             else:
                 logger.error("❌ processed_apus missing or empty in result")
        else:
             logger.error("❌ final_result missing in context")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    test_pipeline()
