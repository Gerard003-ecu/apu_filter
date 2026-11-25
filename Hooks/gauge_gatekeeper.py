#!/usr/bin/env python3
"""
Hook 1: gauge_gatekeeper (PreToolUse)
Intercepta la ejecución de comandos de shell o llamadas al bridge de hardware.
Garantiza el protocolo PASSIVE_SILENT (Back-Action == 0.0 dB) y previene disparos espurios de hardware Crowbar.
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
    last_user_input = payload.get("lastUserInput", "").lower()

    # 1. Auditoría de rol testigo pasivo (PASSIVE_SILENT)
    is_witness_role = "witness" in last_user_input or "pasivo" in last_user_input or "qnd" in last_user_input

    if tool_name == "run_command":
        cmd = args.get("CommandLine", "")
        # Si se pretende mutar memoria o ejecutar escrituras destructivas durante atestación pasiva
        if is_witness_role:
            destructive_patterns = [r"\brm\b", r"\bmv\b", r">", r"\bdrop\b", r"\bkill\b", r"0x3FF44008"]
            for pat in destructive_patterns:
                if re.search(pat, cmd, re.IGNORECASE):
                    decision = {
                        "decision": "deny",
                        "reason": f"[GAUGE_GATEKEEPER] Veto constitucional: El invariante PASSIVE_SILENT prohibe mutacion de estado (Back-Action != 0 dB) con el comando: {cmd}"
                    }
                    sys.stdout.write(json.dumps(decision) + "\n")
                    return

    # 2. Protección de disparo de hardware perimetral (ESP32 Crowbar)
    if "trigger_esp32_crowbar" in tool_name or ("run_command" in tool_name and "0x3FF44008" in args.get("CommandLine", "")):
        conf_code = args.get("confirmation_code", "")
        if conf_code != "SOVEREIGN_VETO_EXECUTE_0x03" and "SOVEREIGN_VETO_EXECUTE_0x03" not in args.get("CommandLine", ""):
            decision = {
                "decision": "deny",
                "reason": "[GAUGE_GATEKEEPER] Veto de hardware: No se autoriza el disparo de Crowbar BT151 sin el código SOVEREIGN_VETO_EXECUTE_0x03."
            }
            sys.stdout.write(json.dumps(decision) + "\n")
            return

    # Si pasa todas las aduanas inmunológicas
    sys.stdout.write(json.dumps({"decision": "allow"}) + "\n")

if __name__ == "__main__":
    main()
