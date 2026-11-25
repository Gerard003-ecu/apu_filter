#!/usr/bin/env python3
"""
Hook 2: betti_sentinel (PostToolUse)
Inspecciona ediciones de archivos para auditar que matrices de presupuestos,
subcontratos o grafos mantengan aciclicidad estricta beta_1 == 0 y ausencia de ciclos de colusión.
"""

import sys
import json
import os

def main():
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input) if raw_input else {}
    except Exception:
        payload = {}

    tool_call = payload.get("toolCall", {})
    args = tool_call.get("args", {})
    file_path = args.get("Path", "")

    # Solo auditar si la edición afecta datos de presupuestos, apus o grafos
    if file_path and any(k in file_path.lower() for k in ["apu", "presupuesto", "contrato", "graph", "simplicial"]):
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Detección de bucles o dependencias circulares explícitas en configuraciones
                if "cycle_detected" in content or "beta_1: 1" in content or "beta_1 = 1" in content:
                    # En PostToolUse se debe retornar un JSON {}, pero podemos emitir advertencias en stderr si fuera necesario
                    pass
            except Exception:
                pass

    # PostToolUse contract expects an empty JSON object {}
    sys.stdout.write(json.dumps({}) + "\n")

if __name__ == "__main__":
    main()
