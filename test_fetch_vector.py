import sys
from app.wisdom.semantic_dictionary import SemanticDictionaryService
from app.adapters.tools_interface import MICRegistry

mic = MICRegistry()
mic.bypass_stratum_checks = True # if this exists
svc = SemanticDictionaryService()
svc.register_in_mic(mic)

print("\nVector execution:")
res = mic.project_intent(
    "fetch_narrative",
    {"domain": "FINAL_VERDICTS", "classification": "SYNERGY_RISK"},
    {"force_override": True}
)
print(res)
