#!/usr/bin/env python3
"""
Hook 4: genesis_attestation (SessionStart)
Ejecutado una vez al inicializar la sesión. Inyecta en el contexto del agente el estado génesis,
los hashes SHA-256 de referencia y los invariantes de la aduana inmunológica V_W.
"""

import sys
import json
import hashlib

def main():
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input) if raw_input else {}
    except Exception:
        payload = {}

    conversation_id = payload.get("conversationId", "default-session")
    genesis_seed = f"FORTALEZA_APU_V9_GENESIS:{conversation_id}".encode("utf-8")
    state_sig = hashlib.sha256(genesis_seed).hexdigest()

    welcome_msg = (
        f"[FORTALEZA APU FILTER v9.0.0] Aduana Inmunológica Inicializada.\n"
        f"• Estado Génesis SHA-256: {state_sig}\n"
        f"• Invariante Primario: Fidelidad Uhlmann F(rho, sigma) >= 0.95 (dB <= 0.2250385)\n"
        f"• Complejo Simplicial: Aciclicidad estricta beta_1 == 0 | Conectividad lambda_2 > 0\n"
        f"• Topos de Heyting: Omega_3 = {{COHERENT (0x00), DEGRADED (0x01), VETOED (0x03)}}\n"
        f"• Inmunidad Löb: Verificador activo fuera de banda."
    )

    out = {
        "injectSteps": [
            {
                "systemMessage": {
                    "systemMessage": welcome_msg
                }
            }
        ]
    }
    sys.stdout.write(json.dumps(out) + "\n")

if __name__ == "__main__":
    main()
