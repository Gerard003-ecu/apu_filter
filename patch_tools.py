import re

file_path = "app/adapters/tools_interface.py"
with open(file_path, "r") as f:
    content = f.read()

# Since I just replaced the wrong text earlier or couldn't find the old block (as I just searched for `def register_core_vectors`),
# let's just make sure it's placed nicely.
improbability_drive_registration = """
    # Motor de Improbabilidad (Fat-Tail Risk)
    try:
        from app.core.immune_system.improbability_drive import ImprobabilityDriveService
        improbability_drive = ImprobabilityDriveService(mic)

        def calculate_fat_tail_risk_handler(**kwargs):
            \"\"\"
            Handler of the Improbability Tensor.
            Ensures that the result is passed through the MIC explicitly mapping failures to fast fail.
            \"\"\"
            result_dict = improbability_drive._morphism_handler(**kwargs)

            if not result_dict.get("success", False):
                # Ensure MIC triggers a Fast-Fail for CategoricalEqualizerSeed
                # Return the error in the schema expected by VectorResult
                return {
                    "status": "error",
                    "error_message": result_dict.get("error_message", "Unknown error"),
                    "error_type": result_dict.get("error_type", "CalculationError"),
                    "details": result_dict
                }

            return result_dict

        mic.register_vector("calculate_fat_tail_risk", Stratum.STRATEGY, calculate_fat_tail_risk_handler)
        logger.info("✅ Motor de Improbabilidad (Estrato STRATEGY) registrado en la MIC")
    except Exception as e:
        logger.warning("⚠️ Motor de Improbabilidad no disponible: %s", e)
"""

content = re.sub(
    r"    # STRATEGY\n    mic\.register_vector\(\n        \"lateral_thinking_pivot\", Stratum\.STRATEGY, vector_lateral_pivot\n    \)",
    "    # STRATEGY\n    mic.register_vector(\n        \"lateral_thinking_pivot\", Stratum.STRATEGY, vector_lateral_pivot\n    )" + improbability_drive_registration,
    content
)

with open(file_path, "w") as f:
    f.write(content)
