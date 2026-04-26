"""
=========================================================================================
Módulo: Auditoría de Redundancia MIC (Algoritmo de Quine-McCluskey)
Ubicación: dev_tools/mic_minimizer.py
=========================================================================================
"""

import logging
from typing import List, Set, Dict, Optional, Tuple

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("MIC.Minimizer")

class QuineMcCluskey:
    """Implementación determinista del algoritmo de Quine-McCluskey."""

    def combine_terms(self, term1: str, term2: str) -> Optional[str]:
        diff = [i for i in range(len(term1)) if term1[i] != term2[i]]
        if len(diff) == 1:
            res = list(term1)
            res[diff[0]] = '-'
            return "".join(res)
        return None

    def solve(self, num_vars: int, minterms: List[int]) -> Set[str]:
        if not minterms: return set()

        # Agrupar por número de 1s
        groups: Dict[int, Set[str]] = {}
        for m in minterms:
            binary = bin(m)[2:].zfill(num_vars)
            ones = binary.count('1')
            groups.setdefault(ones, set()).add(binary)

        prime_implicants: Set[str] = set()

        while groups:
            next_groups: Dict[int, Set[str]] = {}
            combined_this_round: Set[str] = set()

            keys = sorted(groups.keys())
            for i in range(len(keys) - 1):
                k1, k2 = keys[i], keys[i+1]
                if k2 - k1 != 1: continue
                for t1 in groups[k1]:
                    for t2 in groups[k2]:
                        combined = self.combine_terms(t1, t2)
                        if combined:
                            next_groups.setdefault(k1, set()).add(combined)
                            combined_this_round.add(t1)
                            combined_this_round.add(t2)

            for k in groups:
                for term in groups[k]:
                    if term not in combined_this_round:
                        prime_implicants.add(term)
            groups = next_groups

        return prime_implicants

def audit_mic_redundancy():
    """
    Ejecuta el Difeomorfismo Categórico y Poda del Espacio Vectorial.
    Mapea las herramientas actuales a minitérminos de capacidad.
    """
    # Definición de Capacidades (Variables Booleanas)
    # x0: Acceso a Sistema de Archivos
    # x1: Parsing de Estructuras (CSV/Excel)
    # x2: Estabilidad Numérica / Termodinámica
    # x3: Análisis Topológico (Homología/Ciclos)
    # x4: Modelado Financiero / Riesgo
    # x5: Traducción Semántica / NLP

    CAPABILITIES_MAP = {
        "PHYS_IO": 0,
        "PHYS_NUM": 1,
        "TACT_TOPO": 2,
        "STRAT_FIN": 3,
        "WIS_SEM": 4
    }
    num_vars = len(CAPABILITIES_MAP)

    # Mapeo de Herramientas Reales a Minitérminos
    # Cada número es un entero cuya representación binaria marca las capacidades activas
    tools_minterms = {
        "stabilize_flux": [1 << 1],          # PHYS_NUM
        "parse_raw": [1 << 0 | 1 << 1],       # PHYS_IO | PHYS_NUM
        "structure_logic": [1 << 2],         # TACT_TOPO
        "audit_fusion": [1 << 2],            # TACT_TOPO (Posible Redundancia)
        "lateral_pivot": [1 << 3],           # STRAT_FIN
        "fat_tail_risk": [1 << 3],           # STRAT_FIN (Posible Redundancia)
        "semantic_estimator": [1 << 4],      # WIS_SEM
    }

    logger.info("Iniciando Fase I: Minimización del Espacio Vectorial MIC.")

    all_minterms = []
    for mts in tools_minterms.values():
        all_minterms.extend(mts)

    qm = QuineMcCluskey()
    primes = qm.solve(num_vars, list(set(all_minterms)))

    logger.info(f"Implicantes Primos encontrados: {primes}")

    # Análisis de Redundancia Esencial
    essential_tools = []
    redundant_tools = []

    covered_minterms = set()

    # Ordenamos herramientas para un procesamiento determinista
    for tool_name in sorted(tools_minterms.keys()):
        mts = set(tools_minterms[tool_name])
        # Si la herramienta aporta algo nuevo, es esencial en esta pasada simple
        if not mts.issubset(covered_minterms):
            essential_tools.append(tool_name)
            covered_minterms.update(mts)
        else:
            redundant_tools.append(tool_name)

    logger.info(f"Herramientas Esenciales (Base Canónica): {essential_tools}")
    if redundant_tools:
        logger.warning(f"Herramientas Redundantes (Colapso sugerido): {redundant_tools}")

    return essential_tools

if __name__ == "__main__":
    audit_mic_redundancy()
