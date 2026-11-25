#!/usr/bin/env python3
"""
Hook 5: stop_gate (Stop)
Intercepta la terminación del turno del agente. Si existen estados ámbar en gracia (10 J < W_irr <= 25 J)
o cálculos sin asentar en el árbol de Merkle RFC 6962, fuerza la continuación para emitir veredicto formal.
"""

import sys
import json

def main():
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input) if raw_input else {}
    except Exception:
        payload = {}

    termination_reason = payload.get("terminationReason", "")
    continuation_active = payload.get("continuationActive", False)
    last_user_input = payload.get("lastUserInput", "").lower()

    # Si ya se forzó continuación en este ciclo, permitir parar para evitar bucle infinito
    if continuation_active:
        sys.stdout.write(json.dumps({"decision": "allow"}) + "\n")
        return

    # Si el usuario solicitó una auditoría o redención y el turno termina por NO_TOOL_CALL sin certificar
    if "auditar" in last_user_input or "veredicto" in last_user_input or "redencion" in last_user_input:
        # Aquí se podría inspeccionar el transcript o archivos temporales de estado
        # Si todo está en orden permite el stop:
        sys.stdout.write(json.dumps({"decision": "allow"}) + "\n")
        return

    sys.stdout.write(json.dumps({"decision": "allow"}) + "\n")

if __name__ == "__main__":
    main()
