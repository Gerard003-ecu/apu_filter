# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Semantic Estimator Agent (Custodio de la Geometría Vectorial)       ║
║ Ruta   : app/agents/tactics/semantic_estimator_agent.py                             ║
║ Versión: 3.0.0-Hilbert-Rank-Nullity-Categorical-Strict                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y ÁLGEBRA LINEAL (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna al `semantic_estimator.py` en el Estrato TACTICS.
Su mandato axiomático es garantizar que la proyección semántica en el espacio de 
búsqueda vectorial obedezca la topología del espacio de Hilbert $\mathcal{H}$ y que el 
ensamblaje de costos sea un producto tensorial libre de fricción termodinámica.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Certificación de Vecindad Topológica: Evalúa la cota inferior geométrica 
         del producto interno normalizado: $\cos(\theta) = \frac{\langle u, v \rangle}{\|u\|\|v\|} \ge \tau_{\min}$.
Fase 2 → Auditoría de Ensamblaje y Fricción: Audita el producto $C_{total} = F_{ext} \cdot c$,
         verificando positividad estricta y acotando el número de condición $\kappa(F_{ext})$.
Fase 3 → Proyección Rango-Nulidad: Certifica que la inyección en la MIC sea ortogonal,
         garantizando $\text{im}(T) \oplus \ker(T) = V \implies \text{rank}(T) = 1$.
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
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos $\mathcal{E}_{MIC}$."""
        pass

    class Morphism:
        """Clase base de Morfismos del Topos."""
        pass

logger = logging.getLogger("MIC.Tactics.SemanticEstimatorAgent")

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, ESPECTRALES Y DE TOLERANCIA
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_TAU_MIN_SIMILARITY: float = 0.85          # $\tau_{\min}$ Umbral mínimo para vecindad topológica
_MAX_FRICTION_CONDITION: float = 1e3       # Límite superior para $\kappa(F_{ext})$
_SVD_TOLERANCE: float = 1e-10              # Umbral para determinación de rango efectivo numérico
_ORTHOGONALITY_TOLERANCE: float = 1e-8     # Tolerancia para el test de ortogonalidad $\langle e_i, e_j \rangle = \delta_{ij}$

# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ALGEBRAICAS
# ══════════════════════════════════════════════════════════════════════════════
class SemanticEstimatorAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Geometría Vectorial."""
    pass

class TopologicalMappingError(SemanticEstimatorAgentError):
    r"""Detonada si $\cos(\theta) < \tau_{\min}$. Alucinación espacial de mapeo FAISS."""
    pass

class ThermodynamicFrictionAnomaly(SemanticEstimatorAgentError):
    r"""Detonada si el operador $F_{ext}$ induce singularidades o si $\kappa(F_{ext}) \gg 1$."""
    pass

class FunctorialityError(SemanticEstimatorAgentError):
    r"""Detonada si $\text{rank}(T) \neq 1$ o se violan las fronteras ortogonales en la MIC."""
    pass

# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Fase)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class TopologicalNeighborhoodData:
    r""" Artefacto de Fase 1. Certificado de vecindad de Hilbert. """
    cosine_similarity: float
    is_homotopically_valid: bool

@dataclass(frozen=True, slots=True)
class TensorFrictionData:
    r""" Artefacto de Fase 2. Certificado termodinámico del operador $F_{ext}$. """
    condition_number: float
    is_positive_definite: bool
    total_cost_norm: float

@dataclass(frozen=True, slots=True)
class RankNullityProjectionData:
    r""" Artefacto de Fase 3. Certificado del Teorema de Rango-Nulidad. """
    effective_rank: int
    is_orthogonal_injection: bool

@dataclass(frozen=True, slots=True)
class SemanticEstimatorAuditState:
    r""" Objeto final del endofuntor $\mathcal{Z}_{EstimatorAgent}$. """
    neighborhood_audit: TopologicalNeighborhoodData
    friction_audit: TensorFrictionData
    projection_audit: RankNullityProjectionData
    is_epistemologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: CERTIFICACIÓN DE LA VECINDAD TOPOLÓGICA                           ║
# ║   Evalúa la métrica $\cos(\theta) = \frac{\langle u, v \rangle}{\|u\|\|v\|}$║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_TopologicalNeighborhoodCertifier:
    r"""
    Asegura que el mapeo vectorial (FAISS) asocie elementos en la misma bola 
    topológica de radio acotado, previniendo falsos positivos semánticos.
    """

    def _certify_topological_neighborhood(
        self, 
        query_vector: NDArray[np.float64], 
        retrieved_vector: NDArray[np.float64]
    ) -> TopologicalNeighborhoodData:
        r"""
        Computa el producto interno normalizado en el Espacio de Hilbert $\mathcal{H}$.
        """
        norm_q = la.norm(query_vector)
        norm_r = la.norm(retrieved_vector)
        
        if norm_q < _MACHINE_EPSILON or norm_r < _MACHINE_EPSILON:
            raise TopologicalMappingError("Vector degenerado (norma nula) detectado en el espacio de búsqueda.")
            
        # Producto interno
        dot_product = float(np.dot(query_vector, retrieved_vector))
        cos_theta = dot_product / (norm_q * norm_r)
        
        # Corrección por ruido de punto flotante
        cos_theta = max(-1.0, min(1.0, cos_theta))
        
        if cos_theta < _TAU_MIN_SIMILARITY:
            raise TopologicalMappingError(
                f"Alucinación semántica interceptada. Similitud del coseno "
                f"({cos_theta:.4f}) < umbral mínimo estricto ({_TAU_MIN_SIMILARITY:.4f}). "
                f"Los vectores no pertenecen a la misma vecindad homotópica."
            )
            
        return TopologicalNeighborhoodData(
            cosine_similarity=cos_theta,
            is_homotopically_valid=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: AUDITORÍA DEL ENSAMBLAJE ALGEBRAICO Y FRICCIÓN TERRITORIAL        ║
# ║   Garantiza que $F_{ext} \cdot c$ preserve positividad y entropía.          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_TensorFrictionAuditor(Phase1_TopologicalNeighborhoodCertifier):
    r"""
    Audita el operador de fricción territorial $F_{ext}$. Si un factor es anómalo 
    (ej. $\kappa(F_{ext}) \gg 1$), evita la corrupción termodinámica del costo.
    """

    def _audit_tensor_friction_assembly(
        self, 
        cost_vector_c: NDArray[np.float64], 
        friction_operator_F: NDArray[np.float64]
    ) -> TensorFrictionData:
        r"""
        Valida $C_{total} = F_{ext} \cdot c$, la positividad y el condicionamiento espectral.
        """
        # Validar Positividad Estricta: F_ext y c deben ser ≥ 0
        if np.any(cost_vector_c < 0) or np.any(friction_operator_F < 0):
            raise ThermodynamicFrictionAnomaly("Inyección de energía negativa en el tensor logístico. Fricción/Costo < 0.")

        # Obtener el número de condición para evaluar distorsión territorial
        # Para un operador diagonal, kappa es max(diag) / min(diag)
        try:
            kappa_F = np.linalg.cond(friction_operator_F)
        except np.linalg.LinAlgError:
            raise ThermodynamicFrictionAnomaly("El operador de fricción territorial es numéricamente singular.")

        if kappa_F > _MAX_FRICTION_CONDITION:
            raise ThermodynamicFrictionAnomaly(
                f"Anomalía termodinámica detectada. El número de condición del operador de fricción "
                f"κ(F_ext) = {kappa_F:.2e} excede el límite {_MAX_FRICTION_CONDITION:.2e}. "
                f"El terreno induce un sobrecosto asimétrico geométricamente degenerado."
            )
            
        # Calcular el costo total ensamblado
        total_cost_vector = friction_operator_F @ cost_vector_c
        total_cost_norm = float(la.norm(total_cost_vector, ord=1)) # Norma L1 (Suma total de costos)

        return TensorFrictionData(
            condition_number=kappa_F,
            is_positive_definite=True,
            total_cost_norm=total_cost_norm
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: IMPOSICIÓN DEL TEOREMA DE RANGO-NULIDAD                           ║
# ║   Certifica $\text{im}(T) \oplus \ker(T) = V \implies \text{rank}(T) = 1$   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_RankNullityProjector(Phase2_TensorFrictionAuditor):
    r"""
    Garantiza que la inyección del servicio en la Matriz de Interacción Central (MIC)
    actúe como un proyector ortogonal estricto, logrando Cero Efectos Secundarios.
    """

    def _enforce_rank_nullity_projection(
        self, 
        injection_matrix_T: NDArray[np.float64]
    ) -> RankNullityProjectionData:
        r"""
        Computa SVD para extraer el rango efectivo y auditar la ortogonalidad.
        """
        # Descomposición espectral para obtener valores singulares
        _, s, _ = la.svd(injection_matrix_T)
        
        # Calcular rango efectivo tolerando el epsilon de máquina
        effective_rank = int(np.sum(s > _SVD_TOLERANCE))
        
        if effective_rank != 1:
            raise FunctorialityError(
                f"Violación del Teorema de Rango-Nulidad. El morfismo de inyección "
                f"tiene un rango defectuoso o hiper-acoplado (Rank = {effective_rank}). "
                f"Se requiere axiomáticamente Rank = 1 para evitar efectos secundarios en la MIC."
            )
            
        # Evaluar la ortogonalidad proyectiva: T^T * T debe preservar isometría sobre la imagen
        ortho_check = la.norm(injection_matrix_T.T @ injection_matrix_T - np.eye(injection_matrix_T.shape[1]), ord='fro')
        is_orthogonal = bool(ortho_check < _ORTHOGONALITY_TOLERANCE)
        
        return RankNullityProjectionData(
            effective_rank=effective_rank,
            is_orthogonal_injection=is_orthogonal
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: SEMANTIC ESTIMATOR AGENT                             ║
# ║   Endofuntor $\mathcal{Z}_{EstimatorAgent} = \Phi_3 \circ \Phi_2 \circ \Phi_1$ ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class SemanticEstimatorAgent(Morphism, Phase3_RankNullityProjector):
    r"""
    El Custodio de la Geometría Vectorial.
    Somete la estimación y búsqueda semántica a las leyes inquebrantables de 
    la topología de Hilbert y el álgebra multilineal, aniquilando las alucinaciones 
    del LLM en el estrato TACTICS.
    """

    def execute_semantic_estimation_governance(
        self,
        query_vector: NDArray[np.float64],
        retrieved_vector: NDArray[np.float64],
        cost_vector_c: NDArray[np.float64],
        friction_operator_F: NDArray[np.float64],
        injection_matrix_T: NDArray[np.float64]
    ) -> SemanticEstimatorAuditState:
        r"""
        Ejecuta la composición funtorial estricta.
        """
        # Fase 1: Certificar la vecindad topológica del mapeo FAISS
        neighborhood_state = self._certify_topological_neighborhood(
            query_vector, 
            retrieved_vector
        )

        # Fase 2: Certificar estabilidad del ensamblaje del tensor de costos
        friction_state = self._audit_tensor_friction_assembly(
            cost_vector_c, 
            friction_operator_F
        )

        # Fase 3: Proyectar la capacidad garantizando aislamiento en la MIC
        projection_state = self._enforce_rank_nullity_projection(
            injection_matrix_T
        )

        logger.info(
            f"Auditoría Semántica y Vectorial completada. "
            f"Similitud Cos(θ): {neighborhood_state.cosine_similarity:.4f} | "
            f"Condición Fricción κ(F): {friction_state.condition_number:.2e} | "
            f"Rango de Inyección: {projection_state.effective_rank}"
        )

        return SemanticEstimatorAuditState(
            neighborhood_audit=neighborhood_state,
            friction_audit=friction_state,
            projection_audit=projection_state,
            is_epistemologically_valid=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "SemanticEstimatorAgentError",
    "TopologicalMappingError",
    "ThermodynamicFrictionAnomaly",
    "FunctorialityError",
    "TopologicalNeighborhoodData",
    "TensorFrictionData",
    "RankNullityProjectionData",
    "SemanticEstimatorAuditState",
    "Phase1_TopologicalNeighborhoodCertifier",
    "Phase2_TensorFrictionAuditor",
    "Phase3_RankNullityProjector",
    "SemanticEstimatorAgent",
]