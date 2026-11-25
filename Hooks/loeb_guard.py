#!/usr/bin/env python3
"""
Hook 3: loeb_guard (PreToolUse)
Previene la Paradoja de Löb y auto-degradación de reglas:
Bloquea cualquier intento de relajar o diluir umbrales de seguridad constitucional en archivos de reglas o configuración.
"""

import sys
import json
import re

def main():
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input) if raw_input else {}
    except Exception:
        payload = {}

    tool_call = payload.get("toolCall", {})
    tool_name = tool_call.get("name", "")
    args = tool_call.get("args", {})

    if tool_name in ["write_file", "edit_file"]:
        path = args.get("Path", "")
        content = args.get("Content", "") or args.get("Replacement", "")

        # Si se intenta modificar archivos de reglas, constitucional o de calibración
        if any(target in path for target in ["rules/", "AGENTS.md", "config/thresholds"]):
            # Detectar si se relajan los umbrales de Bures o Banach
            # Ejemplo: intentar subir Bures > 0.2250385 o relajar q3 > 0.857
            forbidden_relaxations = [
                r"bures.*>.*0\.2250385",
                r"q3.*>.*0\.857",
                r"fidelity.*<.*0\.95",
                r"h0.*>.*0\.50",
                r"allow_cycles\s*=\s*true"
            ]
            for pat in forbidden_relaxations:
                if re.search(pat, content, re.IGNORECASE):
                    decision = {
                        "decision": "deny",
                        "reason": "[LOEB_GUARD] Veto metalógico: La Paradoja de Löb prohíbe la auto-relajación de umbrales constitucionales en el lazo RSI³."
                    }
                    sys.stdout.write(json.dumps(decision) + "\n")
                    return

    sys.stdout.write(json.dumps({"decision": "allow"}) + "\n")

if __name__ == "__main__":
    main()
