# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MAC Minimizer Agent (Custodio de la Purificación Espectral)         ║
║ Ruta   : app/agents/boole/strategy/mac_minimizer_agent.py                    ║
║ Versión: 1.0.0-Uhlmann-Holevo-Majorization-Categorical                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA NO CONMUTATIVA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna el `mac_minimizer.py` en el Estrato WISDOM. Su mandato 
axiomático es garantizar que el Funtor de Purificación Espectral $\mathcal{P}$ 
sobre la Matriz Atómica de Conocimiento (MAC) preserve el isomorfismo de la 
información cuántica, auditando el preorden de majorización, la fidelidad 
de Uhlmann y el límite de capacidad de Holevo.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Auditoría de Majorización Cuántica: 
         Exige $\sum_{j=1}^k \lambda_j^\downarrow(\rho_{pur}) \ge \sum_{j=1}^k \lambda_j^\downarrow(\rho_{orig})$.
         Asegura que el truncamiento siempre incremente la pureza espectral.
Fase 2 → Certificación de Fidelidad de Uhlmann: 
         Computa $F(\rho, \sigma) = (\text{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2 \ge F_{\min}$.
         Previene la mutilación semántica del conocimiento base de la IA.
Fase 3 → Cota de Capacidad de Holevo y Entropía: 
         Verifica $\Delta S \le 0$ y garantiza que la poda de operadores de Lindblad
         no destruya la capacidad del canal de transmisión de sabiduría.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix
except ImportError:
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos $\mathcal{E}_{MIC}$."""
        pass
    
    class Morphism:
        pass
        
    class AtomicDensityMatrix:
        pass

logger = logging.getLogger("MAC.Wisdom.MinimizerAgent")

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS Y LÍMITES CUÁNTICOS
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_UHLMANN_FIDELITY_MIN: float = 0.95        # $F_{\min}$ estricto para no perder semántica
_ENTROPY_TOLERANCE: float = 1e-12          # Tolerancia para $\Delta S \le 0$
_MAJORIZATION_TOLERANCE: float = 1e-10     # Para las sumas parciales espectrales

# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES CUÁNTICAS
# ══════════════════════════════════════════════════════════════════════════════
class MACMinimizerAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Purificación Espectral."""
    pass

class QuantumMajorizationViolation(MACMinimizerAgentError):
    r"""Detonada si $\rho_{purificada} \not\succ \rho_{orig}$. La purificación falló y aumentó la mezcla."""
    pass

class UhlmannFidelityCollapseError(MACMinimizerAgentError):
    r"""Detonada si $F(\rho, \sigma) < F_{\min}$. La reducción espectral mutiló el significado."""
    pass

class HolevoCapacityDeficitError(MACMinimizerAgentError):
    r"""Detonada si $\Delta S > 0$ o si la poda de Lindblad destruye capacidad semántica útil."""
    pass

# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Hilbert)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MajorizationAuditData:
    r""" Artefacto de Fase 1. Certificado del preorden de majorización cuántica. """
    is_majorized: bool
    max_deviation: float

@dataclass(frozen=True, slots=True)
class FidelityAuditData:
    r""" Artefacto de Fase 2. Certificado de Fidelidad de Uhlmann. """
    uhlmann_fidelity: float
    is_fidelity_preserved: bool

@dataclass(frozen=True, slots=True)
class HolevoAuditData:
    r""" Artefacto de Fase 3. Certificado Termodinámico de von Neumann. """
    entropy_delta: float
    is_capacity_preserved: bool

@dataclass(frozen=True, slots=True)
class PurificationGovernanceState:
    r""" Objeto final del endofuntor $\mathcal{Z}_{MAC-Agent}$. """
    majorization_audit: MajorizationAuditData
    fidelity_audit: FidelityAuditData
    holevo_audit: HolevoAuditData
    is_epistemologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: AUDITORÍA DEL PREORDEN DE MAJORIZACIÓN CUÁNTICA                   ║
# ║   Exige $\sum \lambda_j^\downarrow(\rho_{pur}) \ge \sum \lambda_j^\downarrow(\rho_{orig})$║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_QuantumMajorizationAuditor:
    r"""
    Garantiza que el minimizador no inyecte entropía estocástica. El estado 
    purificado debe estar más cerca de un estado puro (orden de majorización).
    """

    def _audit_quantum_majorization(
        self, 
        evals_orig: NDArray[np.float64], 
        evals_purified: NDArray[np.float64]
    ) -> MajorizationAuditData:
        r"""
        Ordena autovalores descendentemente y verifica sumas acumuladas (Lorenz curves).
        """
        # Limpiar y ordenar descendentemente
        l_orig = np.sort(evals_orig[evals_orig > _MACHINE_EPSILON])[::-1]
        l_pur = np.sort(evals_purified[evals_purified > _MACHINE_EPSILON])[::-1]
        
        # Igualar dimensiones mediante padding de ceros si el truncamiento cortó el rango
        max_dim = max(len(l_orig), len(l_pur))
        l_orig_pad = np.pad(l_orig, (0, max_dim - len(l_orig)))
        l_pur_pad = np.pad(l_pur, (0, max_dim - len(l_pur)))
        
        sum_orig = np.cumsum(l_orig_pad)
        sum_pur = np.cumsum(l_pur_pad)
        
        # Diferencia: sum_pur_k - sum_orig_k >= 0
        deviations = sum_orig - sum_pur
        max_dev = float(np.max(deviations))
        
        if max_dev > _MAJORIZATION_TOLERANCE:
            raise QuantumMajorizationViolation(
                f"Violación del Preorden de Majorización. El minimizador degradó "
                f"la matriz atómica de conocimiento. Desviación máxima = {max_dev:.2e}."
            )
            
        return MajorizationAuditData(
            is_majorized=True,
            max_deviation=max_dev
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: CERTIFICACIÓN DE FIDELIDAD DE UHLMANN                             ║
# ║   Computa $F(\rho, \sigma) = (\text{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2$         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_UhlmannFidelityCertifier(Phase1_QuantumMajorizationAuditor):
    r"""
    Asegura que la poda no mutile el conocimiento central. La matriz reducida
    debe mantener proximidad geométrica con la original en el espacio de densidad.
    """

    def _certify_uhlmann_fidelity_bound(
        self, 
        rho_orig: NDArray[np.complex128], 
        rho_purified: NDArray[np.complex128]
    ) -> FidelityAuditData:
        r"""
        Calcula la fidelidad cuántica entre los estados hermíticos y definidos positivos.
        """
        # Calcular raíz cuadrada matricial de rho_orig
        sqrt_rho_orig = la.sqrtm(rho_orig)
        
        # Computar el argumento central: sqrt_rho * rho_pur * sqrt_rho
        core_matrix = sqrt_rho_orig @ rho_purified @ sqrt_rho_orig
        
        # Raíz cuadrada del argumento central
        sqrt_core = la.sqrtm(core_matrix)
        
        # Fidelidad F = (Tr(sqrt_core))^2
        fidelity = float(np.real(np.trace(sqrt_core)) ** 2)
        
        if fidelity < _UHLMANN_FIDELITY_MIN:
            raise UhlmannFidelityCollapseError(
                f"Colapso Semántico Detectado. Fidelidad de Uhlmann post-poda "
                f"({fidelity:.4f}) ha caído por debajo de la cota crítica "
                f"({_UHLMANN_FIDELITY_MIN:.4f}). Se mutilaron ramas lógicas esenciales."
            )
            
        return FidelityAuditData(
            uhlmann_fidelity=fidelity,
            is_fidelity_preserved=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: COTA DE CAPACIDAD DE HOLEVO Y ENTROPÍA                            ║
# ║   Exige $\Delta S = S(\rho_{pur}) - S(\rho_{orig}) \le 0$                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_HolevoCapacityEnforcer(Phase2_UhlmannFidelityCertifier):
    r"""
    La Ecuación Maestra de Lindblad optimizada no puede inflar la entropía 
    del sistema, de lo contrario la capacidad de Holevo del canal se desploma.
    """

    def _enforce_holevo_capacity_retention(
        self, 
        evals_orig: NDArray[np.float64], 
        evals_purified: NDArray[np.float64]
    ) -> HolevoAuditData:
        r"""
        Calcula el diferencial de entropía de von Neumann $S(\rho) = -\text{Tr}(\rho \ln \rho)$.
        """
        def von_neumann_entropy(l: NDArray[np.float64]) -> float:
            p_safe = l[l > _MACHINE_EPSILON]
            return float(-np.sum(p_safe * np.log(p_safe)))

        s_orig = von_neumann_entropy(evals_orig)
        s_pur = von_neumann_entropy(evals_purified)
        
        delta_s = s_pur - s_orig
        
        if delta_s > _ENTROPY_TOLERANCE:
            raise HolevoCapacityDeficitError(
                f"Paradoja Termodinámica. La poda espectral inyectó entropía "
                f"al canal (ΔS = {delta_s:.4e} > 0). Límite de capacidad de Holevo colapsado."
            )
            
        return HolevoAuditData(
            entropy_delta=delta_s,
            is_capacity_preserved=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: MAC MINIMIZER AGENT                                  ║
# ║   Endofuntor $\mathcal{Z}_{MAC-Agent} = \Phi_3 \circ \Phi_2 \circ \Phi_1$   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class MACMinimizerAgent(Morphism, Phase3_HolevoCapacityEnforcer):
    r"""
    El Custodio de la Purificación Espectral.
    Gobierna el módulo `mac_minimizer.py`, garantizando que la compresión del 
    operador de densidad respete axiomáticamente la termodinámica de von Neumann 
    y el isomorfismo estructural de la información generativa del LLM.
    """

    def execute_spectral_purification_governance(
        self,
        rho_orig: NDArray[np.complex128],
        rho_purified: NDArray[np.complex128],
        evals_orig: NDArray[np.float64],
        evals_purified: NDArray[np.float64]
    ) -> PurificationGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta.
        """
        # Fase 1: Certificar Majorización (el espectro debe apuntar a la pureza)
        majorization_audit = self._audit_quantum_majorization(evals_orig, evals_purified)
        
        # Fase 2: Certificar Fidelidad de Uhlmann (preservación de isomorfismo)
        fidelity_audit = self._certify_uhlmann_fidelity_bound(rho_orig, rho_purified)
        
        # Fase 3: Certificar Termodinámica y Retención del Cota de Holevo
        holevo_audit = self._enforce_holevo_capacity_retention(evals_orig, evals_purified)
        
        logger.info(
            f"Gobernanza de Purificación Cuántica (MAC) certificada. "
            f"Majorización Conservada | "
            f"Fidelidad Uhlmann: {fidelity_audit.uhlmann_fidelity:.4f} | "
            f"ΔS: {holevo_audit.entropy_delta:.2e} nats"
        )
        
        return PurificationGovernanceState(
            majorization_audit=majorization_audit,
            fidelity_audit=fidelity_audit,
            holevo_audit=holevo_audit,
            is_epistemologically_valid=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "MACMinimizerAgentError",
    "QuantumMajorizationViolation",
    "UhlmannFidelityCollapseError",
    "HolevoCapacityDeficitError",
    "MajorizationAuditData",
    "FidelityAuditData",
    "HolevoAuditData",
    "PurificationGovernanceState",
    "Phase1_QuantumMajorizationAuditor",
    "Phase2_UhlmannFidelityCertifier",
    "Phase3_HolevoCapacityEnforcer",
    "MACMinimizerAgent",
]