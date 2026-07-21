# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Telemetry Schemas Agent (Arquitecto del Espacio de Fase Tensorial)  ║
║ Ruta   : app/agents/core/telemetry_schemas_agent.py                          ║
║ Versión: 2.0.0-Tensorial-Orthogonal-Fixpoint-Doctoral-Strict                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DIFERENCIAL (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna los esquemas de telemetría definidos en
`telemetry_schemas.py`.

Actúa como el Endofuntor de Proyección Ortogonal que garantiza que el vector de
estado global Ψ se descomponga rígidamente en la suma directa de subespacios
fundamentales y que su evolución en el tiempo parametrizado τ sea estrictamente
nula.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Certificación de Variedad Riemanniana y Dominio Vectorial:
    Valida G ≻ 0, simetría, condición espectral, dimensión y finitud de los
    vectores de subespacio.

Fase 2 → Descomposición Ortogonal:
    Asegura Ψ = V_PHYSICS ⊕ V_TOPOLOGY ⊕ V_CONTROL ⊕ V_THERMO.
    Computa el producto interno covariante para garantizar:

        <v_i, v_j>_G = δ_ij.

Fase 3 → Inmutabilidad Tensorial y Punto Fijo:
    Somete el tensor instanciado a una derivada temporal covariante.
    Garantiza axiomáticamente que:

        ∇_τ Ψ = 0.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter (Stubs de aislamiento)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import (
        Morphism,
        CategoricalState,
        TopologicalInvariantError,
    )
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos MIC."""
        pass

    class Morphism:
        """Clase base de Morfismos del Topos."""
        pass

    CategoricalState = Any


try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    # Fallback Euclidiano para pruebas aisladas.
    G_PHYSICS = np.eye(4, dtype=np.float64)


try:
    from app.core.telemetry_schemas import SystemStateVector
except ImportError:
    # Stub estructural si el esquema aún no está disponible.
    SystemStateVector = Any


logger = logging.getLogger("MIC.Core.TelemetrySchemasAgent")

if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS Y DE TOLERANCIA
# ═══════════════════════════════════════════════════════════════════════════════

_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# Cota superior para covarianza cruzada fuera de la diagonal.
_ORTHOGONALITY_TOLERANCE: Final[float] = 1e-10

# Tolerancia estricta para ∇_τ Ψ = 0.
_FIXPOINT_TOLERANCE: Final[float] = 1e-12

# Tolerancia para simetría métrica.
_METRIC_SYMMETRY_TOLERANCE: Final[float] = 1e-12

# Número de condición máximo admisible para métricas y matrices de Gram.
_MAX_METRIC_CONDITION_NUMBER: Final[float] = 1e12

# Norma mínima para vectores de subespacio no degenerados.
_SUBSPACE_NORM_TOLERANCE: Final[float] = 1e-12


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES TENSORIALES
# ═══════════════════════════════════════════════════════════════════════════════

class TelemetrySchemasAgentError(TopologicalInvariantError):
    """Excepción raíz del Arquitecto del Espacio de Fase Tensorial."""
    pass


class DomainIntegrityViolationError(TelemetrySchemasAgentError):
    """Detonada cuando un vector, matriz o dimensión viola su contrato de dominio."""
    pass


class MetricManifoldDegeneracyError(TelemetrySchemasAgentError):
    r"""
    Detonada si la métrica G no define una variedad Riemanniana válida:
    no es cuadrada, no es finita, no es simétrica o no es definida positiva.
    """
    pass


class NonOrthogonalSubspaceError(TelemetrySchemasAgentError):
    r"""
    Detonada si <v_i, v_j>_G ≠ 0 para i ≠ j.

    Indica que las dimensiones de los estratos (por ejemplo, Física y Control)
    están entrelazadas o contaminadas por covarianza espuria.
    """
    pass


class PhaseSpaceCorruptionError(TelemetrySchemasAgentError):
    r"""
    Detonada si ∇_τ Ψ > 0.

    El tensor de telemetría sufrió una mutación termodinámica parásita durante
    su viaje entre instantes parametrizados.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Topos)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MetricManifoldData:
    r"""Artefacto de Fase 1. Certificado de la variedad Riemanniana."""
    dimension: int
    metric_condition_number: float
    symmetry_deviation: float
    min_eigenvalue: float
    is_positive_definite: bool


@dataclass(frozen=True, slots=True)
class OrthogonalDecompositionData:
    r"""Artefacto de Fase 2. Certificado de ortogonalidad del estado global Ψ."""
    gram_matrix: NDArray[np.float64]
    off_diagonal_norm: float
    is_strictly_orthogonal: bool
    diagonal_deviation: float = 0.0
    gram_condition_number: float = 1.0
    orthogonality_tolerance: float = _ORTHOGONALITY_TOLERANCE


@dataclass(frozen=True, slots=True)
class FixpointVerificationData:
    r"""Artefacto de Fase 3. Certificado de inmutabilidad y punto fijo."""
    covariant_derivative_norm: float
    is_fixed_point: bool
    fixpoint_tolerance: float = _FIXPOINT_TOLERANCE
    state_dimension: int = 0


@dataclass(frozen=True, slots=True)
class Phase1MetricHandoff:
    r"""
    Handoff formal de Fase 1 → Fase 2.

    Este objeto es la continuación material de la certificación Riemanniana y
    el prefijo obligatorio de la descomposición ortogonal.
    """
    metric_audit: MetricManifoldData
    G_certified: NDArray[np.float64]
    V_physics_certified: NDArray[np.float64]
    V_topology_certified: NDArray[np.float64]
    V_control_certified: NDArray[np.float64]
    V_thermo_certified: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Phase2OrthogonalityHandoff:
    r"""
    Handoff formal de Fase 2 → Fase 3.

    Este objeto transporta la base ortonormal certificada y la auditoría de
    ortogonalidad como prefijo obligatorio del verificador de punto fijo.
    """
    phase1_handoff: Phase1MetricHandoff
    orthogonality_audit: OrthogonalDecompositionData
    basis_matrix: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TensorialPhaseSpaceState:
    r"""Objeto final del endofuntor Z_Schemas."""
    orthogonality_audit: OrthogonalDecompositionData
    fixpoint_audit: FixpointVerificationData
    is_epistemologically_valid: bool
    metric_audit: Optional[MetricManifoldData] = None


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: CERTIFICACIÓN DE VARIEDAD RIEMANNIANA Y DOMINIO VECTORIAL         ║
# ║   Valida G ≻ 0, simetría, condición espectral y vectores de subespacio.     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase1_RiemannianMetricCertifier:
    r"""
    Certifica que el tensor métrico G define una variedad Riemanniana válida y
    que los vectores de subespacio pertenecen al dominio material correcto.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1.1. Tolerancia adaptativa
    # ─────────────────────────────────────────────────────────────────────────
    def _adaptive_tolerance(
        self,
        base_tolerance: float,
        reference: Any,
    ) -> float:
        r"""
        Construye una tolerancia numéricamente consciente:

            tol = max(tol_base, κ · ε_máquina · tamaño · escala)
        """
        if isinstance(reference, np.ndarray):
            size = max(1, int(reference.size))
            if reference.size == 0:
                scale = 1.0
            else:
                scale = max(
                    1.0,
                    float(la.norm(reference.ravel(), ord=np.inf)),
                )
        else:
            size = 1
            try:
                scale = max(1.0, abs(float(reference)))
            except (TypeError, ValueError):
                scale = 1.0

        return max(
            float(base_tolerance),
            10.0 * _MACHINE_EPSILON * size * scale,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2. Coerción de vectores finitos
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_vector(
        self,
        name: str,
        vector: Any,
        expected_dim: Optional[int] = None,
    ) -> NDArray[np.float64]:
        r"""
        Materializa un vector float64 unidimensional, verifica finitud absoluta
        y, si se indica, impone dimensión exacta.
        """
        try:
            arr = np.asarray(vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"El vector ontológico '{name}' no puede materializarse como "
                f"NDArray[np.float64]."
            ) from exc

        if arr.ndim == 0:
            arr = arr.reshape(1)
        else:
            arr = arr.reshape(-1)

        if arr.size == 0:
            raise DomainIntegrityViolationError(
                f"El vector ontológico '{name}' está vacío."
            )

        if not np.all(np.isfinite(arr)):
            raise DomainIntegrityViolationError(
                f"El vector ontológico '{name}' contiene componentes no finitas."
            )

        if expected_dim is not None and arr.size != int(expected_dim):
            raise DomainIntegrityViolationError(
                f"El vector ontológico '{name}' debe tener dimensión "
                f"{expected_dim}, pero posee {arr.size} componentes."
            )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3. Coerción de matrices cuadradas finitas
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_square_matrix(
        self,
        name: str,
        matrix: Any,
    ) -> NDArray[np.float64]:
        r"""
        Materializa una matriz float64 cuadrada y verifica finitud absoluta.
        """
        try:
            arr = np.asarray(matrix, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"La matriz ontológica '{name}' no puede materializarse como "
                f"NDArray[np.float64]."
            ) from exc

        if arr.ndim != 2:
            raise DomainIntegrityViolationError(
                f"La matriz ontológica '{name}' debe ser bidimensional."
            )

        if arr.shape[0] != arr.shape[1]:
            raise DomainIntegrityViolationError(
                f"La matriz ontológica '{name}' debe ser cuadrada."
            )

        if arr.shape[0] == 0:
            raise DomainIntegrityViolationError(
                f"La matriz ontológica '{name}' está vacía."
            )

        if not np.all(np.isfinite(arr)):
            raise DomainIntegrityViolationError(
                f"La matriz ontológica '{name}' contiene componentes no finitas."
            )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4. Certificación de métrica Riemanniana
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_riemannian_metric(
        self,
        G_metric: Optional[NDArray[np.float64]],
        expected_dim: Optional[int] = None,
    ) -> Tuple[NDArray[np.float64], MetricManifoldData]:
        r"""
        Certifica que G_metric sea una métrica Riemanniana válida:

            G = Gᵀ, G ≻ 0, κ(G) < κ_max.

        Si G_metric es None, se emplea la métrica física por defecto G_PHYSICS.
        """
        G_source = G_PHYSICS if G_metric is None else G_metric
        G = self._coerce_finite_square_matrix("G_metric", G_source)

        n = int(G.shape[0])

        if expected_dim is not None and n != int(expected_dim):
            raise DomainIntegrityViolationError(
                f"La métrica G_metric posee dimensión {n}, pero se esperaba "
                f"{expected_dim}."
            )

        # Simetría estricta de la métrica.
        symmetry_deviation = float(la.norm(G - G.T, ord="fro"))
        symmetry_tolerance = self._adaptive_tolerance(
            _METRIC_SYMMETRY_TOLERANCE,
            G,
        )

        if symmetry_deviation > symmetry_tolerance:
            raise MetricManifoldDegeneracyError(
                f"Métrica Riemanniana no simétrica. "
                f"||G - Gᵀ||_F = {symmetry_deviation:.6e} > "
                f"{symmetry_tolerance:.6e}."
            )

        # Proyección simétrica para eliminar ruido antisimétrico infinitesimal.
        G = 0.5 * (G + G.T)

        # Espectro real por simetría.
        eigenvalues = la.eigvalsh(G)
        min_eigenvalue = float(np.min(eigenvalues))
        max_eigenvalue = float(np.max(eigenvalues))

        if max_eigenvalue <= 0.0:
            raise MetricManifoldDegeneracyError(
                "Métrica Riemanniana degenerada: autovalor máximo no positivo."
            )

        # Cota de positividad relativa para evitar singularidades numéricas.
        relative_eigenvalue_tolerance = (
            10.0 * _MACHINE_EPSILON * n * max_eigenvalue
        )

        if min_eigenvalue <= relative_eigenvalue_tolerance:
            raise MetricManifoldDegeneracyError(
                f"Métrica Riemanniana no definida positiva. "
                f"λ_min={min_eigenvalue:.6e} <= "
                f"{relative_eigenvalue_tolerance:.6e}."
            )

        metric_condition_number = float(max_eigenvalue / min_eigenvalue)

        if (
            not np.isfinite(metric_condition_number)
            or metric_condition_number > _MAX_METRIC_CONDITION_NUMBER
        ):
            raise MetricManifoldDegeneracyError(
                f"Métrica Riemanniana mal condicionada. "
                f"κ(G)={metric_condition_number:.6e} > "
                f"{_MAX_METRIC_CONDITION_NUMBER:.6e}."
            )

        metric_audit = MetricManifoldData(
            dimension=n,
            metric_condition_number=metric_condition_number,
            symmetry_deviation=symmetry_deviation,
            min_eigenvalue=min_eigenvalue,
            is_positive_definite=True,
        )

        return G, metric_audit

    # ─────────────────────────────────────────────────────────────────────────
    # 1.5. Certificación de vectores de subespacio
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_subspace_vector(
        self,
        name: str,
        vector: Any,
        dimension: int,
    ) -> NDArray[np.float64]:
        r"""
        Certifica que un vector de subespacio sea finito, dimensionalmente
        compatible y no degenerado.
        """
        v = self._coerce_finite_vector(name, vector, expected_dim=dimension)

        euclidean_norm = float(la.norm(v, ord=2))
        norm_tolerance = max(
            _SUBSPACE_NORM_TOLERANCE,
            10.0 * _MACHINE_EPSILON * dimension * max(1.0, euclidean_norm),
        )

        if euclidean_norm <= norm_tolerance:
            raise DomainIntegrityViolationError(
                f"Vector de subespacio '{name}' es nulo o numéricamente "
                f"degenerado. ||{name}||₂={euclidean_norm:.6e} <= "
                f"{norm_tolerance:.6e}."
            )

        return v

    # ─────────────────────────────────────────────────────────────────────────
    # 1.6. ÚLTIMO MÉTODO DE FASE 1: HANDOFF FORMAL HACIA FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _phase1_certify_and_handoff_to_phase2(
        self,
        V_physics: NDArray[np.float64],
        V_topology: NDArray[np.float64],
        V_control: NDArray[np.float64],
        V_thermo: NDArray[np.float64],
        G_metric: Optional[NDArray[np.float64]] = None,
    ) -> Phase1MetricHandoff:
        r"""
        Último método de la Fase 1.

        Su definición formal es la continuación directa de la Fase 2:
        entrega la métrica certificada y los vectores de subespacio saneados
        como prefijo obligatorio de la descomposición ortogonal.
        """
        G_certified, metric_audit = self._certify_riemannian_metric(G_metric)
        dimension = metric_audit.dimension

        V_physics_certified = self._certify_subspace_vector(
            "V_physics",
            V_physics,
            dimension,
        )
        V_topology_certified = self._certify_subspace_vector(
            "V_topology",
            V_topology,
            dimension,
        )
        V_control_certified = self._certify_subspace_vector(
            "V_control",
            V_control,
            dimension,
        )
        V_thermo_certified = self._certify_subspace_vector(
            "V_thermo",
            V_thermo,
            dimension,
        )

        logger.debug(
            "Fase 1 completada. dim=%d | κ(G)=%.6e | λ_min=%.6e.",
            metric_audit.dimension,
            metric_audit.metric_condition_number,
            metric_audit.min_eigenvalue,
        )

        return Phase1MetricHandoff(
            metric_audit=metric_audit,
            G_certified=G_certified,
            V_physics_certified=V_physics_certified,
            V_topology_certified=V_topology_certified,
            V_control_certified=V_control_certified,
            V_thermo_certified=V_thermo_certified,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: CERTIFICACIÓN DE LA DESCOMPOSICIÓN ORTOGONAL                      ║
# ║   Garantiza <v_i, v_j>_G = δ_ij.                                           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase2_OrthogonalDecompositionCertifier(Phase1_RiemannianMetricCertifier):
    r"""
    Asegura matemáticamente que los cuatro subespacios del Pasaporte de
    Telemetría no presenten covarianza espuria, manteniéndose independientes en
    el hiperespacio Riemanniano.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1. Producto interno covariante
    # ─────────────────────────────────────────────────────────────────────────
    def _metric_inner_product(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> float:
        r"""
        Computa el producto interno covariante:

            <u, v>_G = uᵀ G v.
        """
        value = float(np.dot(u, G_metric @ v))

        if not np.isfinite(value):
            raise DomainIntegrityViolationError(
                "Producto interno covariante no finito."
            )

        return value

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2. Norma Riemanniana de subespacio
    # ─────────────────────────────────────────────────────────────────────────
    def _metric_subspace_norm(
        self,
        name: str,
        v: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> float:
        r"""
        Computa la norma Riemanniana:

            ||v||_G = sqrt(vᵀ G v).

        Exige positividad estricta para evitar subespacios degenerados.
        """
        quadratic_form = self._metric_inner_product(v, v, G_metric)

        if quadratic_form <= _SUBSPACE_NORM_TOLERANCE:
            raise DomainIntegrityViolationError(
                f"Vector de subespacio '{name}' posee norma Riemanniana "
                f"degenerada. ||{name}||_G²={quadratic_form:.6e} <= "
                f"{_SUBSPACE_NORM_TOLERANCE:.6e}."
            )

        return math.sqrt(quadratic_form)

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3. Normalización Riemanniana de subespacio
    # ─────────────────────────────────────────────────────────────────────────
    def _normalize_subspace_vector(
        self,
        name: str,
        v: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Normaliza un vector respecto a la métrica G:

            v̂ = v / ||v||_G.
        """
        norm = self._metric_subspace_norm(name, v, G_metric)
        normalized = v / norm

        if not np.all(np.isfinite(normalized)):
            raise DomainIntegrityViolationError(
                f"La normalización Riemanniana de '{name}' produjo componentes "
                f"no finitas."
            )

        return normalized

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4. Certificación interna de descomposición ortogonal
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_orthogonal_decomposition_from_certified_vectors(
        self,
        V_physics: NDArray[np.float64],
        V_topology: NDArray[np.float64],
        V_control: NDArray[np.float64],
        V_thermo: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> Tuple[OrthogonalDecompositionData, NDArray[np.float64]]:
        r"""
        Computa la Matriz de Gram Mᵀ G M sobre los subespacios normalizados y
        certifica:

            diag(Gram) ≈ 1,
            ||Gram_off||_F ≈ 0,
            κ(Gram) < κ_max.
        """
        v_p = self._normalize_subspace_vector("V_physics", V_physics, G_metric)
        v_t = self._normalize_subspace_vector("V_topology", V_topology, G_metric)
        v_c = self._normalize_subspace_vector("V_control", V_control, G_metric)
        v_th = self._normalize_subspace_vector("V_thermo", V_thermo, G_metric)

        # Matriz bloque M (n x 4).
        M = np.column_stack((v_p, v_t, v_c, v_th))

        if not np.all(np.isfinite(M)):
            raise DomainIntegrityViolationError(
                "La matriz de bases ortonormales M contiene componentes no "
                "finitas."
            )

        # Matriz de Gram inducida por el tensor métrico Riemanniano.
        gram_matrix = M.T @ G_metric @ M

        if not np.all(np.isfinite(gram_matrix)):
            raise DomainIntegrityViolationError(
                "La matriz de Gram contiene componentes no finitas."
            )

        # Simetrización de ruido numérico.
        gram_matrix = 0.5 * (gram_matrix + gram_matrix.T)

        diagonal_values = np.diag(gram_matrix)
        off_diagonal = gram_matrix - np.diag(diagonal_values)

        off_diagonal_norm = float(la.norm(off_diagonal, ord="fro"))
        diagonal_deviation = float(
            la.norm(diagonal_values - np.ones(4, dtype=np.float64), ord=np.inf)
        )

        orthogonality_tolerance = self._adaptive_tolerance(
            _ORTHOGONALITY_TOLERANCE,
            gram_matrix,
        )

        if diagonal_deviation > orthogonality_tolerance:
            raise NonOrthogonalSubspaceError(
                f"La diagonal de la matriz de Gram se aparta de la identidad. "
                f"||diag(Gram)-1||_∞ = {diagonal_deviation:.6e} > "
                f"{orthogonality_tolerance:.6e}."
            )

        if off_diagonal_norm > orthogonality_tolerance:
            raise NonOrthogonalSubspaceError(
                f"Fuga de información entre subespacios detectada. "
                f"Norma de covarianza cruzada ||Gram_off||_F = "
                f"{off_diagonal_norm:.6e} > "
                f"{orthogonality_tolerance:.6e}. "
                f"El espacio no obedece la suma directa ortogonal estricta."
            )

        # Auditoría espectral de la matriz de Gram.
        gram_eigenvalues = la.eigvalsh(gram_matrix)
        gram_min_eigenvalue = float(np.min(gram_eigenvalues))
        gram_max_eigenvalue = float(np.max(gram_eigenvalues))

        if gram_max_eigenvalue <= 0.0:
            raise NonOrthogonalSubspaceError(
                "Matriz de Gram degenerada: autovalor máximo no positivo."
            )

        gram_relative_eigenvalue_tolerance = (
            10.0 * _MACHINE_EPSILON * gram_matrix.shape[0] * gram_max_eigenvalue
        )

        if gram_min_eigenvalue <= gram_relative_eigenvalue_tolerance:
            raise NonOrthogonalSubspaceError(
                f"Matriz de Gram casi singular. "
                f"λ_min={gram_min_eigenvalue:.6e} <= "
                f"{gram_relative_eigenvalue_tolerance:.6e}."
            )

        gram_condition_number = float(
            gram_max_eigenvalue / gram_min_eigenvalue
        )

        if (
            not np.isfinite(gram_condition_number)
            or gram_condition_number > _MAX_METRIC_CONDITION_NUMBER
        ):
            raise NonOrthogonalSubspaceError(
                f"Matriz de Gram mal condicionada. "
                f"κ(Gram)={gram_condition_number:.6e} > "
                f"{_MAX_METRIC_CONDITION_NUMBER:.6e}."
            )

        audit = OrthogonalDecompositionData(
            gram_matrix=gram_matrix,
            off_diagonal_norm=off_diagonal_norm,
            is_strictly_orthogonal=True,
            diagonal_deviation=diagonal_deviation,
            gram_condition_number=gram_condition_number,
            orthogonality_tolerance=orthogonality_tolerance,
        )

        return audit, M

    # ─────────────────────────────────────────────────────────────────────────
    # 2.5. Wrapper público / retrocompatible de ortogonalidad
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_orthogonal_decomposition(
        self,
        V_physics: NDArray[np.float64],
        V_topology: NDArray[np.float64],
        V_control: NDArray[np.float64],
        V_thermo: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> OrthogonalDecompositionData:
        r"""
        Computa la Matriz de Gram Mᵀ G_{μν} M sobre los subespacios
        normalizados. Conserva la signatura original de Fase 1/2.
        """
        G_certified, _ = self._certify_riemannian_metric(G_metric)
        dimension = int(G_certified.shape[0])

        V_physics_certified = self._certify_subspace_vector(
            "V_physics",
            V_physics,
            dimension,
        )
        V_topology_certified = self._certify_subspace_vector(
            "V_topology",
            V_topology,
            dimension,
        )
        V_control_certified = self._certify_subspace_vector(
            "V_control",
            V_control,
            dimension,
        )
        V_thermo_certified = self._certify_subspace_vector(
            "V_thermo",
            V_thermo,
            dimension,
        )

        audit, _ = self._certify_orthogonal_decomposition_from_certified_vectors(
            V_physics_certified,
            V_topology_certified,
            V_control_certified,
            V_thermo_certified,
            G_certified,
        )

        return audit

    # ─────────────────────────────────────────────────────────────────────────
    # 2.6. ÚLTIMO MÉTODO DE FASE 2: HANDOFF FORMAL HACIA FASE 3
    # ─────────────────────────────────────────────────────────────────────────
    def _phase2_certify_and_handoff_to_phase3(
        self,
        phase1_handoff: Phase1MetricHandoff,
    ) -> Phase2OrthogonalityHandoff:
        r"""
        Último método de la Fase 2.

        Su definición formal es la continuación directa de la Fase 3:
        entrega la auditoría de ortogonalidad y la base ortonormal como prefijo
        obligatorio del verificador de inmutabilidad tensorial.
        """
        if not isinstance(phase1_handoff, Phase1MetricHandoff):
            raise DomainIntegrityViolationError(
                "Fase 2 exige un Phase1MetricHandoff como prefijo formal."
            )

        audit, basis_matrix = (
            self._certify_orthogonal_decomposition_from_certified_vectors(
                phase1_handoff.V_physics_certified,
                phase1_handoff.V_topology_certified,
                phase1_handoff.V_control_certified,
                phase1_handoff.V_thermo_certified,
                phase1_handoff.G_certified,
            )
        )

        logger.debug(
            "Fase 2 completada. ||Gram_off||_F=%.6e | diag_dev=%.6e | "
            "κ(Gram)=%.6e.",
            audit.off_diagonal_norm,
            audit.diagonal_deviation,
            audit.gram_condition_number,
        )

        return Phase2OrthogonalityHandoff(
            phase1_handoff=phase1_handoff,
            orthogonality_audit=audit,
            basis_matrix=basis_matrix,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: IMPOSICIÓN DE INMUTABILIDAD Y PUNTO FIJO                          ║
# ║   Garantiza ∇_τ Ψ = 0 a través de la evolución temporal.                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase3_TensorImmutabilityEnforcer(
    Phase2_OrthogonalDecompositionCertifier
):
    r"""
    Somete el vector de estado a una auditoría diferencial para asegurar que la
    telemetría sea criptográficamente inmutable y represente un punto fijo en la
    variedad.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1. Coerción de vectores de estado
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_state_vector(
        self,
        name: str,
        state_vector: Any,
        dimension: int,
    ) -> NDArray[np.float64]:
        r"""
        Coerciona un vector de estado global Ψ, exigiendo finitud y dimensión
        compatible con la métrica certificada.
        """
        return self._coerce_finite_vector(
            name,
            state_vector,
            expected_dim=dimension,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2. Norma Riemanniana de estado
    # ─────────────────────────────────────────────────────────────────────────
    def _riemannian_state_norm(
        self,
        name: str,
        state_vector: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> float:
        r"""
        Computa ||Ψ||_G = sqrt(Ψᵀ G Ψ), con saneamiento de ruido negativo
        infinitesimal.
        """
        quadratic_form = float(np.dot(state_vector, G_metric @ state_vector))

        if not np.isfinite(quadratic_form):
            raise PhaseSpaceCorruptionError(
                f"La forma cuadrática Riemanniana de '{name}' no es finita."
            )

        scale = max(1.0, abs(quadratic_form))
        quadratic_tolerance = max(
            _FIXPOINT_TOLERANCE,
            10.0 * _MACHINE_EPSILON * state_vector.size * scale,
        )

        if quadratic_form < -quadratic_tolerance:
            raise PhaseSpaceCorruptionError(
                f"Forma cuadrática Riemanniana negativa para '{name}': "
                f"{quadratic_form:.6e} < -{quadratic_tolerance:.6e}."
            )

        if quadratic_form < 0.0:
            quadratic_form = 0.0

        return math.sqrt(quadratic_form)

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3. Implementación interna de punto fijo
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_tensor_immutability_and_fixpoint_internal(
        self,
        Psi_t0: NDArray[np.float64],
        Psi_t1: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> FixpointVerificationData:
        r"""
        Computa la norma de la derivada covariante discreta:

            ||∇_τ Ψ||_G = ||Ψ(t₁) - Ψ(t₀)||_G.

        Si la norma supera la tolerancia adaptativa, el tensor fue corrompido
        durante su tránsito.
        """
        if Psi_t0.shape != Psi_t1.shape:
            raise DomainIntegrityViolationError(
                f"Los vectores de estado Ψ_t0 y Ψ_t1 poseen dimensiones "
                f"incompatibles: {Psi_t0.shape} != {Psi_t1.shape}."
            )

        diff_Psi = Psi_t1 - Psi_t0

        if not np.all(np.isfinite(diff_Psi)):
            raise PhaseSpaceCorruptionError(
                "La diferencia covariante Ψ_t1 - Ψ_t0 contiene componentes no "
                "finitas."
            )

        norm_t0 = self._riemannian_state_norm("Psi_t0", Psi_t0, G_metric)
        norm_t1 = self._riemannian_state_norm("Psi_t1", Psi_t1, G_metric)
        nabla_tau = self._riemannian_state_norm("diff_Psi", diff_Psi, G_metric)

        scale = max(1.0, norm_t0, norm_t1, nabla_tau)
        dimension = int(Psi_t0.size)

        fixpoint_tolerance = max(
            _FIXPOINT_TOLERANCE,
            10.0 * _MACHINE_EPSILON * dimension * scale,
        )

        if nabla_tau > fixpoint_tolerance:
            raise PhaseSpaceCorruptionError(
                f"Corrupción en el Pasaporte de Telemetría. El tensor no es un "
                f"punto fijo. Magnitud de la derivada covariante ∇_τ Ψ = "
                f"{nabla_tau:.6e} > {fixpoint_tolerance:.6e}. "
                f"Posible mutación parásita en la Malla Agéntica."
            )

        return FixpointVerificationData(
            covariant_derivative_norm=nabla_tau,
            is_fixed_point=True,
            fixpoint_tolerance=fixpoint_tolerance,
            state_dimension=dimension,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.4. Wrapper público / retrocompatible de inmutabilidad
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_tensor_immutability_and_fixpoint(
        self,
        Psi_t0: NDArray[np.float64],
        Psi_t1: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> FixpointVerificationData:
        r"""
        Computa la norma de la derivada covariante discreta ||∇_τ Ψ||_G².
        Conserva la signatura original de Fase 2.
        """
        G_certified, metric_audit = self._certify_riemannian_metric(G_metric)
        dimension = metric_audit.dimension

        Psi_t0_certified = self._coerce_state_vector(
            "Psi_t0",
            Psi_t0,
            dimension,
        )
        Psi_t1_certified = self._coerce_state_vector(
            "Psi_t1",
            Psi_t1,
            dimension,
        )

        return self._enforce_tensor_immutability_and_fixpoint_internal(
            Psi_t0_certified,
            Psi_t1_certified,
            G_certified,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.5. ÚLTIMO MÉTODO DE FASE 3: FINALIZACIÓN FUNTORIAL
    # ─────────────────────────────────────────────────────────────────────────
    def _phase3_finalize_from_phase2_handoff(
        self,
        phase2_handoff: Phase2OrthogonalityHandoff,
        Psi_t0: NDArray[np.float64],
        Psi_t1: NDArray[np.float64],
    ) -> TensorialPhaseSpaceState:
        r"""
        Último método de la Fase 3.

        Compone los certificados de Fase 1, Fase 2 y Fase 3 en el objeto
        terminal TensorialPhaseSpaceState.
        """
        if not isinstance(phase2_handoff, Phase2OrthogonalityHandoff):
            raise DomainIntegrityViolationError(
                "Fase 3 exige un Phase2OrthogonalityHandoff como prefijo "
                "formal."
            )

        G_certified = phase2_handoff.phase1_handoff.G_certified
        dimension = phase2_handoff.phase1_handoff.metric_audit.dimension

        Psi_t0_certified = self._coerce_state_vector(
            "Psi_t0",
            Psi_t0,
            dimension,
        )
        Psi_t1_certified = self._coerce_state_vector(
            "Psi_t1",
            Psi_t1,
            dimension,
        )

        fixpoint_audit = self._enforce_tensor_immutability_and_fixpoint_internal(
            Psi_t0_certified,
            Psi_t1_certified,
            G_certified,
        )

        state = TensorialPhaseSpaceState(
            orthogonality_audit=phase2_handoff.orthogonality_audit,
            fixpoint_audit=fixpoint_audit,
            is_epistemologically_valid=True,
            metric_audit=phase2_handoff.phase1_handoff.metric_audit,
        )

        logger.info(
            "Arquitectura del Espacio de Fase Tensorial certificada. "
            "Gram Off-Diagonal=%.6e | Mutación Covariante ∇_τ=%.6e | "
            "κ(G)=%.6e.",
            state.orthogonality_audit.off_diagonal_norm,
            state.fixpoint_audit.covariant_derivative_norm,
            phase2_handoff.phase1_handoff.metric_audit.metric_condition_number,
        )

        return state


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: TELEMETRY SCHEMAS AGENT                              ║
# ║   Endofuntor Z_Schemas = Φ₃ ∘ Φ₂ ∘ Φ₁                                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TelemetrySchemasAgent(Morphism, Phase3_TensorImmutabilityEnforcer):
    r"""
    El Arquitecto del Espacio de Fase Tensorial.

    Garantiza axiomáticamente que la instanciación de cualquier estado respete
    la independencia lineal de las bases y permanezca inmutable en el tiempo.
    """

    def execute_tensorial_phase_space_governance(
        self,
        V_physics: NDArray[np.float64],
        V_topology: NDArray[np.float64],
        V_control: NDArray[np.float64],
        V_thermo: NDArray[np.float64],
        Psi_t0: NDArray[np.float64],
        Psi_t1: NDArray[np.float64],
        G_metric: Optional[NDArray[np.float64]] = None,
    ) -> TensorialPhaseSpaceState:
        r"""
        Ejecuta la composición funtorial estricta sobre el vector de
        telemetría.
        """
        phase1_handoff = self._phase1_certify_and_handoff_to_phase2(
            V_physics=V_physics,
            V_topology=V_topology,
            V_control=V_control,
            V_thermo=V_thermo,
            G_metric=G_metric,
        )

        phase2_handoff = self._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
        )

        return self._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
            Psi_t0=Psi_t0,
            Psi_t1=Psi_t1,
        )

    def __call__(
        self,
        V_physics: NDArray[np.float64],
        V_topology: NDArray[np.float64],
        V_control: NDArray[np.float64],
        V_thermo: NDArray[np.float64],
        Psi_t0: NDArray[np.float64],
        Psi_t1: NDArray[np.float64],
        G_metric: Optional[NDArray[np.float64]] = None,
    ) -> TensorialPhaseSpaceState:
        r"""Alias invocable del endofuntor de gobierno tensorial."""
        return self.execute_tensorial_phase_space_governance(
            V_physics=V_physics,
            V_topology=V_topology,
            V_control=V_control,
            V_thermo=V_thermo,
            Psi_t0=Psi_t0,
            Psi_t1=Psi_t1,
            G_metric=G_metric,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "TelemetrySchemasAgentError",
    "DomainIntegrityViolationError",
    "MetricManifoldDegeneracyError",
    "NonOrthogonalSubspaceError",
    "PhaseSpaceCorruptionError",
    "MetricManifoldData",
    "OrthogonalDecompositionData",
    "FixpointVerificationData",
    "Phase1MetricHandoff",
    "Phase2OrthogonalityHandoff",
    "TensorialPhaseSpaceState",
    "Phase1_RiemannianMetricCertifier",
    "Phase2_OrthogonalDecompositionCertifier",
    "Phase3_TensorImmutabilityEnforcer",
    "TelemetrySchemasAgent",
]