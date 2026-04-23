import json

with open('config/config_rules.json', 'r') as f:
    data = json.load(f)

for rule in data["apu_classification_rules"]["rules"]:
    if rule["type"] == "CONSTRUCCION_MIXTO":
        rule["condition"] = "(porcentaje_materiales >= 40.0 and porcentaje_materiales < 60.0) or (porcentaje_mo_eq >= 40.0 and porcentaje_mo_eq < 60.0)"
    if rule["type"] == "OBRA_COMPLETA":
        rule["condition"] = "porcentaje_materiales >= 0 and porcentaje_mo_eq >= 0"

with open('config/config_rules.json', 'w') as f:
    json.dump(data, f, indent=2)
