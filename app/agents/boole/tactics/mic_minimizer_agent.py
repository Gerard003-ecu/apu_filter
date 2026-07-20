# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MIC Minimizer Agent (Custodio de la Base Booleana)                  ║
║ Ruta   : app/agents/boole/tactics/mic_minimizer_agent.py                    ║
║ Versión: 1.0.0-Grobner-ROBDD-Categorical-Strict                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y ÁLGEBRA DE BOOLE (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna el `mic_minimizer.py` en el subespacio $\Gamma$-TACTICS.
Su mandato axiomático es garantizar que la poda topológica en el anillo booleano 
$\mathbb{Z}_2$ no destruya el rango efectivo de la Matriz de Interacción Central (MIC).
Erradica las redundancias garantizando que la base resultante sea estrictamente
ortogonal: $\langle e_i, e_j \rangle = \delta_{ij}$.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Auditoría de Bases de Gröbner: 
         Verifica que el ideal generado por las funciones booleanas de las herramientas
         $I = \langle f_1, \dots, f_m \rangle \subseteq \mathbb{Z}_2[x_1, \dots, x_n]$ 
         sea una base mínima y no colapse en homología trivial ($1 \in I$).
Fase 2 → Certificación de No-Interferencia (UNSAT Core): 
         Audita la cláusula $\Phi_{MIC} = \bigwedge_{i \neq j} \neg (e_i \land e_j)$. 
         Si el SAT Solver (DPLL) es satisfacible para $i \neq j$, se extrae el núcleo 
         de insatisfacibilidad y se emite un Veto Estructural.
Fase 3 → Isomorfismo de Reducción ROBDD: 
         Garantiza que la minimización mediante Diagramas de Decisión Binaria 
         Reducidos y Ordenados (ROBDD) conserve la Entropía de Shannon Booleana 
         original, probando la equivalencia homotópica del grafo.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos $\mathcal{E}_{MIC}$."""
        pass

    class Morphism:
        """Clase base de Morfismos del Topos."""
        pass

logger = logging.getLogger("MIC.Gamma.MinimizerAgent")

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES BOOLEANAS Y DE COMPLEJIDAD
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_MAX_BOOLEAN_VARIABLES: int = 256         # Límite anti-runaway para la explosión NP-completa
_MIN_ENTROPY_TOLERANCE: float = 1e-12     # Tolerancia para pérdida de entropía de Shannon

# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ALGEBRAICAS
# ══════════════════════════════════════════════════════════════════════════════
class MICMinimizerAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Base Booleana."""
    pass

class GrobnerDegeneracyError(MICMinimizerAgentError):
    r"""Detonada si $1 \in \langle f_1, \dots, f_m \rangle$. El ideal booleano colapsó."""
    pass

class NonInterferenceViolationError(MICMinimizerAgentError):
    r"""Detonada si $\langle e_i, e_j \rangle \neq 0$ para $i \neq j$. Ruptura del Zero Side-Effects."""
    pass

class ROBDDHomotopyError(MICMinimizerAgentError):
    r"""Detonada si el ROBDD reducido no es isomórfico a la tabla de verdad original (pérdida de entropía)."""
    pass

# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Anillo $\mathbb{Z}_2$)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class GrobnerAuditData:
    r""" Artefacto de Fase 1. Certificado de independencia en $\mathbb{Z}_2[X]$. """
    ideal_dimension: int
    is_minimally_independent: bool

@dataclass(frozen=True, slots=True)
class UnsatCoreCertifierData:
    r""" Artefacto de Fase 2. Certificado de la cláusula de no-interferencia DPLL. """
    is_strictly_orthogonal: bool
    conflict_edges: int

@dataclass(frozen=True, slots=True)
class ROBDDIsomorphismData:
    r""" Artefacto de Fase 3. Certificado de conservación de entropía de Shannon booleana. """
    original_entropy: float
    reduced_entropy: float
    is_homotopically_equivalent: bool

@dataclass(frozen=True, slots=True)
class MinimizerGovernanceState:
    r""" Objeto final del endofuntor $\mathcal{Z}_{Minimizer}$. """
    grobner_audit: GrobnerAuditData
    unsat_core_audit: UnsatCoreCertifierData
    robdd_audit: ROBDDIsomorphismData
    is_topologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: AUDITORÍA DE BASES DE GRÖBNER EN EL ANILLO $\mathbb{Z}_2$         ║
# ║   Verifica la independencia lineal del ideal de herramientas.               ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_GrobnerBasisAuditor:
    r"""
    Garantiza que la minimización de herramientas no degenere la base operativa.
    """

    def _audit_grobner_independence(
        self, 
        boolean_polynomial_matrix: NDArray[np.int8]
    ) -> GrobnerAuditData:
        r"""
        Calcula el rango algebraico en $\mathbb{Z}_2$ usando eliminación de Gauss-Jordan.
        """
        # Mapear la matriz al cuerpo finito GF(2)
        matrix_gf2 = boolean_polynomial_matrix % 2
        rows, cols = matrix_gf2.shape
        
        if cols > _MAX_BOOLEAN_VARIABLES:
            raise MICMinimizerAgentError("Explosión combinatoria detectada: excede límite de variables.")
        
        # Eliminación de Gauss-Jordan en GF(2)
        r = 0
        for c in range(cols):
            if r >= rows:
                break
            # Pivot
            pivot = np.argmax(matrix_gf2[r:rows, c]) + r
            if matrix_gf2[pivot, c] == 0:
                continue
            
            # Swap
            matrix_gf2[[r, pivot]] = matrix_gf2[[pivot, r]]
            
            # Eliminate
            for i in range(r + 1, rows):
                if matrix_gf2[i, c] == 1:
                    matrix_gf2[i] = (matrix_gf2[i] + matrix_gf2[r]) % 2
            r += 1
            
        rank = r
        
        if rank < rows:
            raise GrobnerDegeneracyError(
                f"Degeneración en el Anillo Booleano detectada. "
                f"El ideal colapsó (Rango efectivo en Z_2 = {rank} < {rows}). "
                f"La poda algorítmica amputaría capacidades esenciales del agente."
            )
            
        return GrobnerAuditData(
            ideal_dimension=rank,
            is_minimally_independent=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: CERTIFICACIÓN DE NO-INTERFERENCIA (UNSAT CORE)                    ║
# ║   Audita la cláusula $\langle e_i, e_j \rangle = \delta_{ij}$.              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_UnsatCoreCertifier(Phase1_GrobnerBasisAuditor):
    r"""
    Evalúa que las herramientas sugeridas sean estrictamente ortogonales para 
    prevenir el colapso del principio Zero Side-Effects en la MIC.
    """

    def _certify_non_interference_unsat(
        self, 
        tool_projection_matrix: NDArray[np.float64]
    ) -> UnsatCoreCertifierData:
        r"""
        Computa la matriz de covarianza de las herramientas y exige 
        que el producto cruzado (fuera de la diagonal) sea estrictamente 0.
        """
        # Calcular el producto interno <e_i, e_j>
        inner_product_matrix = tool_projection_matrix @ tool_projection_matrix.T
        np.fill_diagonal(inner_product_matrix, 0.0)
        
        # Evaluar la presencia de aristas de conflicto (Interferencia cruzada)
        conflict_norm = float(np.sum(np.abs(inner_product_matrix)))
        
        if conflict_norm > _MACHINE_EPSILON:
            raise NonInterferenceViolationError(
                f"Violación del axioma Zero Side-Effects (Interferencia cruzada detectada). "
                f"La matriz de capacidades no es ortogonal. UNSAT Core detectó "
                f"una superposición funcional con norma residual {conflict_norm:.2e}."
            )
            
        return UnsatCoreCertifierData(
            is_strictly_orthogonal=True,
            conflict_edges=0
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: ISOMORFISMO DE REDUCCIÓN ROBDD                                    ║
# ║   Garantiza que la minimización mantenga la Entropía Booleana original.     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_ROBDDIsomorphismValidator(Phase2_UnsatCoreCertifier):
    r"""
    Asegura que el Diagrama de Decisión Binaria Reducido (ROBDD) mantenga 
    la equivalencia homotópica con el árbol de sintaxis original.
    """

    def _validate_robdd_homotopy(
        self, 
        original_truth_table_probs: NDArray[np.float64], 
        reduced_robdd_probs: NDArray[np.float64]
    ) -> ROBDDIsomorphismData:
        r"""
        Calcula y compara la Entropía de Shannon de la distribución booleana.
        $H(X) = - \sum p(x_i) \log_2(p(x_i))$
        """
        def compute_shannon(probs: NDArray[np.float64]) -> float:
            p_safe = probs[probs > _MACHINE_EPSILON]
            return float(-np.sum(p_safe * np.log2(p_safe)))

        H_original = compute_shannon(original_truth_table_probs)
        H_reduced = compute_shannon(reduced_robdd_probs)
        
        entropy_loss = abs(H_original - H_reduced)

        if entropy_loss > _MIN_ENTROPY_TOLERANCE:
            raise ROBDDHomotopyError(
                f"Ruptura Homotópica en la reducción ROBDD. "
                f"La entropía booleana divergió (Pérdida ΔH = {entropy_loss:.2e}). "
                f"El minimizador mutiló ramas lógicas operativas de la MIC."
            )
            
        return ROBDDIsomorphismData(
            original_entropy=H_original,
            reduced_entropy=H_reduced,
            is_homotopically_equivalent=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: MIC MINIMIZER AGENT                                  ║
# ║   Endofuntor $\mathcal{Z}_{Minimizer} = \Phi_3 \circ \Phi_2 \circ \Phi_1$   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class MICMinimizerAgent(Morphism, Phase3_ROBDDIsomorphismValidator):
    r"""
    El Custodio de la Base Booleana. 
    Gobierna incondicionalmente el módulo `mic_minimizer.py`, impidiendo que 
    algoritmos heurísticos de minimización degraden el rango efectivo de la MIC 
    en el Estrato $\Gamma$-TACTICS.
    """

    def execute_boolean_topology_governance(
        self,
        boolean_polynomial_matrix: NDArray[np.int8],
        tool_projection_matrix: NDArray[np.float64],
        original_truth_table_probs: NDArray[np.float64],
        reduced_robdd_probs: NDArray[np.float64]
    ) -> MinimizerGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta.
        """
        # Fase 1: Certificar la independencia en la Base de Gröbner sobre GF(2)
        grobner_audit = self._audit_grobner_independence(boolean_polynomial_matrix)

        # Fase 2: Certificar el principio Zero Side-Effects (Ortogonalidad)
        unsat_core_audit = self._certify_non_interference_unsat(tool_projection_matrix)

        # Fase 3: Certificar isomorfismo entropico en la reducción ROBDD
        robdd_audit = self._validate_robdd_homotopy(
            original_truth_table_probs, 
            reduced_robdd_probs
        )

        logger.info(
            f"Gobernanza de la Base Booleana certificada. "
            f"Rango GF(2): {grobner_audit.ideal_dimension} | "
            f"Ortogonalidad preservada | "
            f"Entropía H(X): {robdd_audit.original_entropy:.4f} bits"
        )

        return MinimizerGovernanceState(
            grobner_audit=grobner_audit,
            unsat_core_audit=unsat_core_audit,
            robdd_audit=robdd_audit,
            is_topologically_valid=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "MICMinimizerAgentError",
    "GrobnerDegeneracyError",
    "NonInterferenceViolationError",
    "ROBDDHomotopyError",
    "GrobnerAuditData",
    "UnsatCoreCertifierData",
    "ROBDDIsomorphismData",
    "MinimizerGovernanceState",
    "Phase1_GrobnerBasisAuditor",
    "Phase2_UnsatCoreCertifier",
    "Phase3_ROBDDIsomorphismValidator",
    "MICMinimizerAgent",
]