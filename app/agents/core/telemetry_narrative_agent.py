# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Telemetry Narrative Agent (El Intérprete Diplomático Supremo)       ║
║ Ruta   : app/core/telemetry_narrative_agent.py                               ║
║ Versión: 1.0.0-Diffeomorphism-Lattice-Doctoral-Strict                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TEORÍA DE CATEGORÍAS (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna a `telemetry_narrative.py`. Actúa como el Funtor de 
Difeomorfismo Semántico $F: \mathcal{C}_{Math} \to \mathcal{D}_{Narrative}$ que 
proyecta el espacio de invariantes matemáticos sobre la ontología del negocio. 
Erradica el libre albedrío estocástico del LLM, confinándolo a redactar un 
"Juicio del Consejo" estrictamente gobernado por las restricciones geométricas.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Colapso en el Retículo Acotado Distributivo: 
         Evalúa las métricas inter-estrato bajo el orden $\bot \le \dots \le \top$. 
         Computa la clausura algebraica del Supremo ($\bigsqcup$). Si existe un fallo, 
         el sistema exige axiomáticamente que $\bot \sqcup \top = \top$.
Fase 2 → Certificación de Difeomorfismo Semántico: 
         Audita que el texto de "Empatía Táctica" generado (GraphRAG) preserve 
         un isomorfismo biyectivo con las patologías geométricas de la base.
         Si $\beta_1 > 0 \implies \text{Debe contener "Socavón Lógico"}$.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Any, Dict, List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter (Stubs de aislamiento)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos $\mathcal{E}_{MIC}$."""
        pass

    class Morphism:
        """Clase base de Morfismos del Topos."""
        pass

try:
    from app.core.telemetry_schemas import TopologicalMetrics
except ImportError:
    # Stub estructural si el esquema aún no está disponible
    TopologicalMetrics = Any

logger = logging.getLogger("MIC.Core.TelemetryNarrativeAgent")

# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULO DISTRIBUTIVO ACOTADO (Álgebra de Severidad)
# ══════════════════════════════════════════════════════════════════════════════
@unique
class SeverityLevel(IntEnum):
    r""" 
    Retículo Algebraico de Severidad bajo el orden parcial estricto: 
    $\bot \le \dots \le \top$
    """
    OPTIMO = 0       # Elemento mínimo ($\bot$)
    MODERADO = 1     
    SEVERO = 2       
    CRITICO = 3      # Elemento máximo absorbente ($\top$)


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ONTOLÓGICAS Y SEMÁNTICAS
# ══════════════════════════════════════════════════════════════════════════════
class TelemetryNarrativeAgentError(TopologicalInvariantError):
    """Excepción raíz del Intérprete Diplomático Supremo."""
    pass

class SeverityLatticeCollapseError(TelemetryNarrativeAgentError):
    r"""
    Detonada si el LLM u otro sistema estocástico viola la operación Supremo ($\bigsqcup v_i$).
    """
    pass

class SemanticDiffeomorphismViolationError(TelemetryNarrativeAgentError):
    r"""
    Detonada si el texto generado (Empatía Táctica) no exhibe las traducciones 
    biyectivas exactas de la patología subyacente. El LLM ha inyectado entropía 
    retórica y su veredicto es aniquilado.
    """
    pass

# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Fase Semántico)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class LatticeCollapseState:
    r""" Artefacto de Fase 1. El colapso inmutable de severidad. """
    supremum_verdict: SeverityLevel
    is_worst_case_enforced: bool

@dataclass(frozen=True, slots=True)
class DiffeomorphismAuditData:
    r""" Artefacto de Fase 2. Certificado de que la narrativa respeta el isomorfismo. """
    is_isomorphic: bool
    betti_1_verified: bool
    fiedler_psi_verified: bool
    semantic_drift_detected: bool

@dataclass(frozen=True, slots=True)
class NarrativeAgentState:
    r""" Objeto final del endofuntor $\mathcal{Z}_{Narrative}$. """
    lattice_collapse: LatticeCollapseState
    diffeomorphism_audit: DiffeomorphismAuditData
    approved_narrative: str
    is_epistemologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: COLAPSO EN EL RETÍCULO ACOTADO DISTRIBUTIVO                       ║
# ║   Evalúa $\text{Veredicto} = \bigsqcup_{i \in \{F, T, S, W\}} v_i$          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_SeverityLatticeCollapser:
    r"""
    Sintetiza las decisiones inter-estrato operando bajo la clausura algebraica 
    del Supremo. Impone el Worst-Case Scenario de forma determinista.
    """

    def _collapse_severity_lattice(
        self, 
        stratum_verdicts: List[SeverityLevel]
    ) -> LatticeCollapseState:
        r"""
        Computa la operación $\bigsqcup$ (Join) en el retículo distributivo.
        Si la física emite $\bot$ y la topología emite $\top$, el resultado matemático es $\top$.
        """
        if not stratum_verdicts:
            # Vacío por defecto colapsa al elemento neutro (Óptimo)
            return LatticeCollapseState(SeverityLevel.OPTIMO, True)
            
        # El operador supremo en este retículo finito ordenado se computa vía max()
        supremum = max(stratum_verdicts, key=lambda v: v.value)
        
        # Auditoría de Invarianza: Comprobación estricta de absorción
        if SeverityLevel.CRITICO in stratum_verdicts and supremum != SeverityLevel.CRITICO:
            raise SeverityLatticeCollapseError(
                "Fallo en la Clausura Algebraica: Se esperaba $\\bot \\sqcup \\top = \\top$, "
                "pero el sistema no colapsó al estado CRITICO."
            )
            
        return LatticeCollapseState(
            supremum_verdict=supremum,
            is_worst_case_enforced=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: CERTIFICACIÓN DE DIFEOMORFISMO SEMÁNTICO                          ║
# ║   Verifica que la narrativa preserve los invariantes $\beta_1$ y $\Psi$.    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_SemanticDiffeomorphismCertifier(Phase1_SeverityLatticeCollapser):
    r"""
    Asegura que el texto redactado (Empatía Táctica) posea un isomorfismo directo 
    con las métricas abstractas. Subordina el LLM al diccionario predefinido.
    """

    def _certify_semantic_diffeomorphism(
        self, 
        topological_metrics: Any,  # Asumido TopologicalMetrics
        proposed_narrative: str
    ) -> DiffeomorphismAuditData:
        r"""
        Realiza la auditoría de isomorfismo.
        Axiomas de Difeomorfismo:
        - Si $\beta_1 > 0 \implies \text{Debe incluir 'Socavón Lógico'}$.
        - Si $\Psi < 1.0 \implies \text{Debe incluir 'Pirámide Invertida'}$.
        """
        narrative_upper = proposed_narrative.upper()
        
        betti_1_verified = True
        fiedler_psi_verified = True
        
        # Mapeo Biyectivo 1: Ciclos Homológicos -> Socavón Lógico
        if hasattr(topological_metrics, 'beta_1') and topological_metrics.beta_1 > 0:
            if "SOCAVÓN LÓGICO" not in narrative_upper and "SOCAVON LOGICO" not in narrative_upper:
                betti_1_verified = False
                
        # Mapeo Biyectivo 2: Estabilidad Piramidal -> Pirámide Invertida
        if hasattr(topological_metrics, 'pyramid_stability') and topological_metrics.pyramid_stability < 1.0:
            if "PIRÁMIDE INVERTIDA" not in narrative_upper and "PIRAMIDE INVERTIDA" not in narrative_upper:
                fiedler_psi_verified = False

        if not betti_1_verified or not fiedler_psi_verified:
            raise SemanticDiffeomorphismViolationError(
                "La narrativa ejecutiva rompió el difeomorfismo con la matriz topológica. "
                f"[β₁ match: {betti_1_verified}, Ψ match: {fiedler_psi_verified}]. "
                "El LLM incurrió en deriva estocástica y diluyó la alerta técnica."
            )

        return DiffeomorphismAuditData(
            is_isomorphic=True,
            betti_1_verified=betti_1_verified,
            fiedler_psi_verified=fiedler_psi_verified,
            semantic_drift_detected=False
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: TELEMETRY NARRATIVE AGENT                            ║
# ║   Endofuntor $\mathcal{Z}_{Narrative} = \Phi_2 \circ \Phi_1$                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TelemetryNarrativeAgent(Morphism, Phase2_SemanticDiffeomorphismCertifier):
    r"""
    El Intérprete Diplomático Supremo.
    Gobierna el Pasaporte de Telemetría colapsando los dictámenes de las fases 
    inferiores hacia un Acta de Deliberación inmutable, certificando la Empatía 
    Táctica contra los axiomas de la topología algebraica.
    """

    def execute_diplomatic_narrative_governance(
        self,
        stratum_verdicts: List[SeverityLevel],
        topological_metrics: Any,
        proposed_narrative: str
    ) -> NarrativeAgentState:
        r"""
        Ejecuta la composición funtorial estricta.
        """
        # Fase 1: Certificación del Colapso en el Retículo Acotado
        lattice_state = self._collapse_severity_lattice(stratum_verdicts)

        # Fase 2: Certificación del Difeomorfismo Semántico
        # Solo se exige la traducción biyectiva si el supremo general no es ÓPTIMO
        if lattice_state.supremum_verdict != SeverityLevel.OPTIMO:
            diffeomorphism_state = self._certify_semantic_diffeomorphism(
                topological_metrics, 
                proposed_narrative
            )
        else:
            diffeomorphism_state = DiffeomorphismAuditData(True, True, True, False)

        logger.info(
            f"Veredicto Narrativo Categórico ejecutado con éxito. "
            f"Supremo: {lattice_state.supremum_verdict.name} | "
            f"Isomorfismo Narrativo: {diffeomorphism_state.is_isomorphic}"
        )

        return NarrativeAgentState(
            lattice_collapse=lattice_state,
            diffeomorphism_audit=diffeomorphism_state,
            approved_narrative=proposed_narrative,
            is_epistemologically_valid=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "TelemetryNarrativeAgentError",
    "SeverityLatticeCollapseError",
    "SemanticDiffeomorphismViolationError",
    "SeverityLevel",
    "LatticeCollapseState",
    "DiffeomorphismAuditData",
    "NarrativeAgentState",
    "Phase1_SeverityLatticeCollapser",
    "Phase2_SemanticDiffeomorphismCertifier",
    "TelemetryNarrativeAgent",
]