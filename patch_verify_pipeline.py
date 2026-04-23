import re

with open('verify_pipeline.py', 'r') as f:
    content = f.read()

new_imports = """from app.tactics.pipeline_director import process_all_files
from app.core.telemetry import TelemetryContext"""

content = content.replace("from app.tactics.pipeline_director import PipelineDirector\nfrom app.core.telemetry import TelemetryContext", new_imports)

old_try = """    telemetry = TelemetryContext()
    director = PipelineDirector(config, telemetry)

    initial_context = {
        "presupuesto_path": presupuesto_path,
        "apus_path": apus_path,
        "insumos_path": insumos_path,
    }

    logger.info(f"Testing with files: {initial_context}")

    try:
        final_context = director.execute(initial_context)"""

new_try = """    # Enforce topological axioms
    config["enforce_filtration"] = True
    config["enforce_homology"] = True

    telemetry = TelemetryContext()

    logger.info(f"Testing with files: {presupuesto_path}, {apus_path}, {insumos_path}")

    try:
        final_context = process_all_files(presupuesto_path, apus_path, insumos_path, config, telemetry)"""

content = content.replace(old_try, new_try)

with open('verify_pipeline.py', 'w') as f:
    f.write(content)
