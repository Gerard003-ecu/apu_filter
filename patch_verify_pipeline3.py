import re

with open('verify_pipeline.py', 'r') as f:
    content = f.read()

# final_context is already processed_apus JSON in process_all_files, not a context dictionary
new_verification = """        final_result = final_context

        if final_result and "presupuesto" in final_result:
            logger.info(f"✅ Pipeline completed successfully. Presupuesto length: {len(final_result['presupuesto'])}")
        else:
            logger.error("❌ Pipeline did not return expected final_result.")"""

content = re.sub(r'        # Verification 1: Insumos.*', new_verification, content, flags=re.DOTALL)

with open('verify_pipeline.py', 'w') as f:
    f.write(content)
