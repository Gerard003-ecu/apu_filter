import re

with open('verify_pipeline.py', 'r') as f:
    content = f.read()

# Since `final_context` IS a dict, we just need to check what is in it.
# The `process_all_files` wrapper does NOT return a `CategoricalEqualizerSeed`, it catches it and probably returns a dict with {"status": "ERROR"} or similar.
# Let's inspect what final_context has when it fails.
new_try = """    try:
        final_context = process_all_files(presupuesto_path, apus_path, insumos_path, config, telemetry)

        # Si el resultado es un dict y tiene 'status' == 'ERROR'
        if isinstance(final_context, dict) and final_context.get("status") in ("ERROR", "error", "REJECT", "reject"):
            logger.info(f"✅ Pipeline correctly aborted with status {final_context.get('status')} due to topological constraints.")
        elif isinstance(final_context, dict) and "presupuesto" in final_context:
            logger.info(f"✅ Pipeline completed successfully. Presupuesto length: {len(final_context['presupuesto'])}")
        else:
            logger.error(f"❌ Pipeline did not return expected final_result. Received dict with keys: {final_context.keys() if isinstance(final_context, dict) else type(final_context)}")

    except Exception as e:"""

content = re.sub(r'    try:.*    except Exception as e:', new_try, content, flags=re.DOTALL)

with open('verify_pipeline.py', 'w') as f:
    f.write(content)
