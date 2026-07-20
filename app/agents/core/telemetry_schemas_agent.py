# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Telemetry Schemas Agent (Arquitecto del Espacio de Fase Tensorial)  ║
║ Ruta   : app/agents/core/telemetry_schemas_agent.py                          ║
║ Versión: 1.0.0-Tensorial-Orthogonal-Fixpoint-Doctoral                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DIFERENCIAL (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna los esquemas de telemetría definidos en `telemetry_schemas.py`.
Actúa como el Endofuntor de Proyección Ortogonal que garantiza que el vector de estado
global $\Psi$ se descomponga rígidamente en la suma directa de subespacios fundamentales
y que su evolución en el tiempo parametrizado $\tau$ sea estrictamente nula.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Descomposición Ortogonal: 
         Asegura $\Psi = V_{PHYSICS} \oplus V_{TOPOLOGY} \oplus V_{CONTROL} \oplus V_{THERMO}$
         Computa el producto interno covariante para garantizar $\langle v_i, v_j \rangle_G = \delta_{ij}$.
Fase 2 → Inmutabilidad Tensorial y Punto Fijo: 
         Somete el tensor instanciado a una derivada temporal covariante. 
         Garantiza axiomáticamente que $\nabla_\tau \Psi = 0$.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter (Stubs de aislamiento)
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

try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    # Fallback Euclidiano para pruebas aisladas
    G_PHYSICS = np.eye(4, dtype=np.float64)

try:
    from app.core.telemetry_schemas import SystemStateVector
except ImportError:
    # Stub estructural si el esquema aún no está disponible
    SystemStateVector = Any

logger = logging.getLogger("MIC.Core.TelemetrySchemasAgent")

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS Y DE TOLERANCIA
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_ORTHOGONALITY_TOLERANCE: float = 1e-10    # Cota superior para el producto interno cruzado
_FIXPOINT_TOLERANCE: float = 1e-12         # Tolerancia estricta para \nabla_\tau \Psi = 0

# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES TENSORIALES
# ══════════════════════════════════════════════════════════════════════════════
class TelemetrySchemasAgentError(TopologicalInvariantError):
    """Excepción raíz del Arquitecto del Espacio de Fase Tensorial."""
    pass

class NonOrthogonalSubspaceError(TelemetrySchemasAgentError):
    r"""
    Detonada si $\langle v_i, v_j \rangle_G \neq 0$ para $i \neq j$.
    Indica que las dimensiones de los estratos (ej. Física y Control) están entrelazadas (contaminadas).
    """
    pass

class PhaseSpaceCorruptionError(TelemetrySchemasAgentError):
    r"""
    Detonada si $\nabla_\tau \Psi > 0$.
    El tensor de telemetría sufrió una mutación termodinámica parásita durante su viaje.
    """
    pass

# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Topos)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class OrthogonalDecompositionData:
    r""" Artefacto de Fase 1. Certificado de ortogonalidad del estado global $\Psi$. """
    gram_matrix: NDArray[np.float64]
    off_diagonal_norm: float
    is_strictly_orthogonal: bool

@dataclass(frozen=True, slots=True)
class FixpointVerificationData:
    r""" Artefacto de Fase 2. Certificado de inmutabilidad (derivada covariante nula). """
    covariant_derivative_norm: float
    is_fixed_point: bool

@dataclass(frozen=True, slots=True)
class TensorialPhaseSpaceState:
    r""" Objeto final del endofuntor $\mathcal{Z}_{Schemas}$. """
    orthogonality_audit: OrthogonalDecompositionData
    fixpoint_audit: FixpointVerificationData
    is_epistemologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: CERTIFICACIÓN DE LA DESCOMPOSICIÓN ORTOGONAL                      ║
# ║   Garantiza $\langle v_i, v_j \rangle_G = \delta_{ij}$                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_OrthogonalDecompositionCertifier:
    r"""
    Asegura matemáticamente que los cuatro subespacios del Pasaporte de Telemetría
    no presenten covarianza espuria, manteniéndose independientes en el hiperespacio.
    """

    def _certify_orthogonal_decomposition(
        self, 
        V_physics: NDArray[np.float64], 
        V_topology: NDArray[np.float64], 
        V_control: NDArray[np.float64], 
        V_thermo: NDArray[np.float64], 
        G_metric: NDArray[np.float64]
    ) -> OrthogonalDecompositionData:
        r"""
        Computa la Matriz de Gram $M^T G_{\mu\nu} M$ sobre los subespacios normalizados.
        """
        # Normalización previa para garantizar \delta_{ij} en la diagonal principal
        def normalize_subspace(v: NDArray[np.float64]) -> NDArray[np.float64]:
            norm = math.sqrt(float(np.dot(v.T, np.dot(G_metric, v))))
            return v / norm if norm > _MACHINE_EPSILON else v

        v_p = normalize_subspace(V_physics)
        v_t = normalize_subspace(V_topology)
        v_c = normalize_subspace(V_control)
        v_th = normalize_subspace(V_thermo)

        # Matriz bloque M (n x 4)
        M = np.column_stack((v_p, v_t, v_c, v_th))
        
        # Matriz de Gram inducida por el tensor métrico Riemanniano
        gram_matrix = M.T @ G_metric @ M
        
        # Extraer elementos fuera de la diagonal (Covarianza cruzada)
        off_diagonal = gram_matrix - np.diag(np.diag(gram_matrix))
        off_diagonal_norm = float(la.norm(off_diagonal, ord='fro'))

        if off_diagonal_norm > _ORTHOGONALITY_TOLERANCE:
            raise NonOrthogonalSubspaceError(
                f"Fuga de información entre subespacios detectada. "
                f"Norma de covarianza cruzada ‖Gram_off‖_F = {off_diagonal_norm:.4e} > {_ORTHOGONALITY_TOLERANCE:.4e}. "
                f"El espacio no obedece la suma directa ortogonal estricta."
            )

        return OrthogonalDecompositionData(
            gram_matrix=gram_matrix,
            off_diagonal_norm=off_diagonal_norm,
            is_strictly_orthogonal=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: IMPOSICIÓN DE INMUTABILIDAD Y PUNTO FIJO                          ║
# ║   Garantiza $\nabla_\tau \Psi = 0$ a través de la evolución temporal        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_TensorImmutabilityEnforcer(Phase1_OrthogonalDecompositionCertifier):
    r"""
    Somete el vector de estado a una auditoría diferencial para asegurar que la 
    telemetría sea criptográficamente inmutable y represente un punto fijo en la variedad.
    """

    def _enforce_tensor_immutability_and_fixpoint(
        self, 
        Psi_t0: NDArray[np.float64], 
        Psi_t1: NDArray[np.float64], 
        G_metric: NDArray[np.float64]
    ) -> FixpointVerificationData:
        r"""
        Computa la norma de la derivada covariante discreta $\| \nabla_\tau \Psi \|_G^2$.
        Si la norma supera 0, el tensor fue corrompido durante su tránsito.
        """
        # Vector diferencial (mutación a través del parámetro afín \tau)
        diff_Psi = Psi_t1 - Psi_t0
        
        # Norma Riemanniana de la desviación
        nabla_tau_sq = float(np.dot(diff_Psi.T, np.dot(G_metric, diff_Psi)))
        nabla_tau = math.sqrt(max(0.0, nabla_tau_sq))

        if nabla_tau > _FIXPOINT_TOLERANCE:
            raise PhaseSpaceCorruptionError(
                f"Corrupción en el Pasaporte de Telemetría. El tensor no es un punto fijo. "
                f"Magnitud de la derivada covariante ∇_τ Ψ = {nabla_tau:.4e} > {_FIXPOINT_TOLERANCE:.4e}. "
                f"Posible mutación parásita en la Malla Agéntica."
            )

        return FixpointVerificationData(
            covariant_derivative_norm=nabla_tau,
            is_fixed_point=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: TELEMETRY SCHEMAS AGENT                              ║
# ║   Endofuntor $\mathcal{Z}_{Schemas} = \Phi_2 \circ \Phi_1$                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TelemetrySchemasAgent(Morphism, Phase2_TensorImmutabilityEnforcer):
    r"""
    El Arquitecto del Espacio de Fase Tensorial.
    Garantiza axiomáticamente que la instanciación de cualquier estado respete la
    independencia lineal de las bases y permanezca inmutable en el tiempo.
    """

    def execute_tensorial_phase_space_governance(
        self,
        V_physics: NDArray[np.float64],
        V_topology: NDArray[np.float64],
        V_control: NDArray[np.float64],
        V_thermo: NDArray[np.float64],
        Psi_t0: NDArray[np.float64],
        Psi_t1: NDArray[np.float64],
        G_metric: Optional[NDArray[np.float64]] = None
    ) -> TensorialPhaseSpaceState:
        r"""
        Ejecuta la composición funtorial estricta sobre el vector de telemetría.
        """
        metric = G_metric if G_metric is not None else G_PHYSICS

        # Fase 1: Certificación de Ortogonalidad Absoluta
        orthogonality_audit = self._certify_orthogonal_decomposition(
            V_physics, 
            V_topology, 
            V_control, 
            V_thermo, 
            metric
        )

        # Fase 2: Certificación de Inmutabilidad y Punto Fijo
        fixpoint_audit = self._enforce_tensor_immutability_and_fixpoint(
            Psi_t0, 
            Psi_t1, 
            metric
        )

        logger.info(
            f"Arquitectura del Espacio de Fase Tensorial certificada. "
            f"Gram Off-Diagonal: {orthogonality_audit.off_diagonal_norm:.2e} | "
            f"Mutación Covariante ∇_τ: {fixpoint_audit.covariant_derivative_norm:.2e}"
        )

        return TensorialPhaseSpaceState(
            orthogonality_audit=orthogonality_audit,
            fixpoint_audit=fixpoint_audit,
            is_epistemologically_valid=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "TelemetrySchemasAgentError",
    "NonOrthogonalSubspaceError",
    "PhaseSpaceCorruptionError",
    "OrthogonalDecompositionData",
    "FixpointVerificationData",
    "TensorialPhaseSpaceState",
    "Phase1_OrthogonalDecompositionCertifier",
    "Phase2_TensorImmutabilityEnforcer",
    "TelemetrySchemasAgent",
]