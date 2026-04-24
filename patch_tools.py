import re

file_path = "app/adapters/tools_interface.py"
with open(file_path, "r") as f:
    content = f.read()

improbability_drive_registration = """    try:
        from app.core.immune_system.improbability_drive import ImprobabilityDriveService
        improbability_drive = ImprobabilityDriveService(mic)
        improbability_drive.register_in_mic()
        logger.info("✅ Motor de Improbabilidad (Estrato Ω) registrado en la MIC")
    except Exception as e:
        logger.warning("⚠️ Motor de Improbabilidad no disponible: %s", e)

"""

content = re.sub(
    r"    try:\n        from app\.wisdom\.semantic_dictionary import SemanticDictionaryService",
    improbability_drive_registration + "    try:\n        from app.wisdom.semantic_dictionary import SemanticDictionaryService",
    content
)

with open(file_path, "w") as f:
    f.write(content)
