# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Semantic Estimator Agent (Custodio de la Geometría Vectorial)       ║
║ Ruta   : app/agents/tactics/semantic_estimator_agent.py                      ║
║ Versión: 4.0.0-Hilbert-Rank-Nullity-Categorical-Strict-Nested                ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y ÁLGEBRA LINEAL (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna al `semantic_estimator.py` en el Estrato TACTICS.

Su mandato axiomático es garantizar que la proyección semántica en el espacio de
búsqueda vectorial obedezca la topología del espacio de Hilbert $\mathcal{H}$ y que el
ensamblaje de costos sea un producto tensorial libre de fricción termodinámica.

ARQUITECTURA DE 3 FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Certificación de Vecindad Topológica:
         Φ₁(u, v) = cos(θ)
         con:
             cos(θ) = ⟨u, v⟩ / (||u|| ||v||) ≥ τ_min.

Fase 2 → Auditoría de Ensamblaje y Fricción:
         Φ₂(Φ₁(...)) = (κ(F_ext), C_total)
         verificando positividad estricta y acotando:
             κ(F_ext) ≤ κ_max.

Fase 3 → Proyección Rango-Nulidad:
         Φ₃(Φ₂(Φ₁(...))) = rank(T)
         certificando:
             rank(T) = 1
         y que T actúe como una isometría parcial ortogonal sobre su imagen.

COMPOSICIÓN:
────────────
El último método de la Fase 1 emite un puente formal que es consumido por el
primer método de la Fase 2. El último método de la Fase 2 emite un puente que
es consumido por el primer método de la Fase 3.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Tuple

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

    class CategoricalState:
        """Clase base de Estados Categóricos."""
        pass


logger = logging.getLogger("MIC.Tactics.SemanticEstimatorAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. TIPOS Y CONSTANTES MATEMÁTICAS, ESPECTRALES Y DE TOLERANCIA
# ═══════════════════════════════════════════════════════════════════════════════
VectorF64 = NDArray[np.float64]
MatrixF64 = NDArray[np.float64]
OperatorF64 = NDArray[np.float64]

_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)

# Fase 1: Vecindad topológica de Hilbert.
_TAU_MIN_SIMILARITY: float = 0.85

# Fase 2: Fricción territorial y ensamblaje de costos.
_MAX_FRICTION_CONDITION: float = 1e3
_POSITIVE_FLOOR: float = 1e-12
_NEGATIVE_TOLERANCE: float = 1e-12

# Fase 3: Rango-nulidad e inyección ortogonal.
_SVD_ABSOLUTE_TOLERANCE: float = 1e-10
_ORTHOGONALITY_TOLERANCE: float = 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ALGEBRAICAS
# ═══════════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Fase)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True, eq=False)
class TopologicalNeighborhoodData:
    r"""
    Artefacto de Fase 1.
    Certificado de vecindad de Hilbert.
    """
    cosine_similarity: float
    query_norm: float
    retrieved_norm: float
    is_homotopically_valid: bool


@dataclass(frozen=True, slots=True, eq=False)
class TensorFrictionData:
    r"""
    Artefacto de Fase 2.
    Certificado termodinámico del operador $F_{ext}$.
    """
    condition_number: float
    spectral_min: float
    spectral_max: float
    total_cost_norm: float
    total_cost_vector: VectorF64
    is_positive_definite: bool


@dataclass(frozen=True, slots=True, eq=False)
class RankNullityProjectionData:
    r"""
    Artefacto de Fase 3.
    Certificado del Teorema de Rango-Nulidad y de inyección ortogonal.
    """
    matrix_shape: Tuple[int, int]
    effective_rank: int
    kernel_dimension: int
    largest_singular_value: float
    rank_tolerance: float
    orthogonality_deviation: float
    is_orthogonal_injection: bool


@dataclass(frozen=True, slots=True, eq=False)
class Phase1TopologicalBridge:
    r"""
    Puente funtorial Φ₁ → Φ₂.

    Este objeto es emitido por el último método de la Fase 1 y constituye
    la entrada formal del primer método de la Fase 2.
    """
    neighborhood_audit: TopologicalNeighborhoodData
    query_vector: VectorF64
    retrieved_vector: VectorF64
    cost_vector_c: VectorF64
    friction_operator_F: OperatorF64
    injection_matrix_T: MatrixF64


@dataclass(frozen=True, slots=True, eq=False)
class Phase2FrictionBridge:
    r"""
    Puente funtorial Φ₂ → Φ₃.

    Este objeto es emitido por el último método de la Fase 2 y constituye
    la entrada formal del primer método de la Fase 3.
    """
    phase1_bridge: Phase1TopologicalBridge
    friction_audit: TensorFrictionData


@dataclass(frozen=True, slots=True, eq=False)
class SemanticEstimatorAuditState:
    r"""
    Objeto final del endofuntor $\mathcal{Z}_{EstimatorAgent}$.
    """
    neighborhood_audit: TopologicalNeighborhoodData
    friction_audit: TensorFrictionData
    projection_audit: RankNullityProjectionData
    is_epistemologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: CERTIFICACIÓN DE LA VECINDAD TOPOLÓGICA                           ║
# ║                                                                             ║
# ║   Φ₁(u, v) = cos(θ) = ⟨u, v⟩ / (||u|| ||v||)                                ║
# ║                                                                             ║
# ║   1. Valida finiteza y dimensionalidad de vectores.                         ║
# ║   2. Calcula similitud coseno con normalización numéricamente segura.       ║
# ║   3. Exige cos(θ) ≥ τ_min.                                                  ║
# ║   4. Emite el puente formal hacia la Fase 2.                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_TopologicalNeighborhoodCertifier:
    r"""
    Fase 1 del endofuntor.

    Asegura que el mapeo vectorial (FAISS) asocie elementos en la misma bola
    topológica de radio acotado, previniendo falsos positivos semánticos.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Validación numérica elemental
    # ─────────────────────────────────────────────────────────────────────────
    def _as_finite_float(self, name: str, value: float) -> float:
        """
        Convierte un valor a float64 y exige que sea finito.
        """
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SemanticEstimatorAgentError(
                f"{name} no puede convertirse a un escalar float64."
            ) from exc

        if arr.ndim != 0:
            raise SemanticEstimatorAgentError(
                f"{name} debe ser un escalar, no un arreglo de dimensión {arr.ndim}."
            )

        scalar = float(arr)
        if not math.isfinite(scalar):
            raise SemanticEstimatorAgentError(
                f"{name} debe ser finito. Se recibió {scalar!r}."
            )

        return scalar

    def _as_finite_vector(self, name: str, value: VectorF64) -> VectorF64:
        """
        Valida que el objeto sea un vector 1-D no vacío y con componentes finitas.
        """
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SemanticEstimatorAgentError(
                f"{name} no puede convertirse a un vector float64."
            ) from exc

        if arr.ndim != 1:
            raise SemanticEstimatorAgentError(
                f"{name} debe ser un vector 1-D. Dimensión recibida: {arr.ndim}."
            )

        if arr.size == 0:
            raise SemanticEstimatorAgentError(
                f"{name} no puede ser el vector vacío."
            )

        if not np.all(np.isfinite(arr)):
            raise SemanticEstimatorAgentError(
                f"{name} contiene componentes NaN o infinitas."
            )

        return arr

    def _as_finite_matrix(self, name: str, value: MatrixF64) -> MatrixF64:
        """
        Valida que el objeto sea una matriz 2-D no vacía y finita.
        """
        try:
            mat = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SemanticEstimatorAgentError(
                f"{name} no puede convertirse a una matriz float64."
            ) from exc

        if mat.ndim != 2:
            raise SemanticEstimatorAgentError(
                f"{name} debe ser una matriz 2-D. Dimensión recibida: {mat.ndim}."
            )

        if mat.size == 0 or mat.shape[0] == 0 or mat.shape[1] == 0:
            raise SemanticEstimatorAgentError(
                f"{name} no puede ser una matriz vacía."
            )

        if not np.all(np.isfinite(mat)):
            raise SemanticEstimatorAgentError(
                f"{name} contiene entradas NaN o infinitas."
            )

        return mat

    def _as_finite_friction_operator(
        self,
        name: str,
        value: OperatorF64,
        dimension: int
    ) -> OperatorF64:
        """
        Valida un operador de fricción como:
        - Vector 1-D de factores diagonales, o
        - Matriz 2-D cuadrada compatible con el vector de costos.
        """
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SemanticEstimatorAgentError(
                f"{name} no puede convertirse a un operador float64."
            ) from exc

        if arr.ndim == 0:
            if dimension != 1:
                raise SemanticEstimatorAgentError(
                    f"{name} escalar sólo es admisible para dimensión 1."
                )
            arr = arr.reshape(1)

        if arr.ndim == 1:
            if arr.size != dimension:
                raise SemanticEstimatorAgentError(
                    f"{name} como operador diagonal debe tener tamaño {dimension}. "
                    f"Se recibió {arr.size}."
                )
        elif arr.ndim == 2:
            if arr.shape != (dimension, dimension):
                raise SemanticEstimatorAgentError(
                    f"{name} como matriz debe tener shape ({dimension}, {dimension}). "
                    f"Se recibió {arr.shape}."
                )
        else:
            raise SemanticEstimatorAgentError(
                f"{name} debe ser un vector 1-D o una matriz 2-D. "
                f"Dimensión recibida: {arr.ndim}."
            )

        if not np.all(np.isfinite(arr)):
            raise SemanticEstimatorAgentError(
                f"{name} contiene entradas NaN o infinitas."
            )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # Normas numéricamente seguras
    # ─────────────────────────────────────────────────────────────────────────
    def _safe_l2_norm(self, vector: VectorF64) -> float:
        """
        Calcula ||v||₂ con reescalado para evitar overflow/underflow.
        """
        if vector.size == 0:
            return 0.0

        scale = float(np.max(np.abs(vector)))
        if scale == 0.0:
            return 0.0

        if not math.isfinite(scale):
            return math.inf

        scaled = vector / scale
        ss = float(np.vdot(scaled, scaled).real)

        if not math.isfinite(ss):
            return math.inf

        norm = scale * math.sqrt(ss)
        return float(norm) if math.isfinite(norm) else math.inf

    def _safe_fro_norm(self, matrix: MatrixF64) -> float:
        """
        Calcula ||M||_F con reescalado para evitar overflow/underflow.
        """
        if matrix.size == 0:
            return 0.0

        scale = float(np.max(np.abs(matrix)))
        if scale == 0.0:
            return 0.0

        if not math.isfinite(scale):
            return math.inf

        scaled = matrix / scale
        ss = float(np.sum(np.abs(scaled) ** 2))

        if not math.isfinite(ss):
            return math.inf

        norm = scale * math.sqrt(ss)
        return float(norm) if math.isfinite(norm) else math.inf

    def _safe_l1_norm(self, vector: VectorF64) -> float:
        """
        Calcula ||v||₁ con reescalado para evitar overflow.
        """
        if vector.size == 0:
            return 0.0

        scale = float(np.max(np.abs(vector)))
        if scale == 0.0:
            return 0.0

        if not math.isfinite(scale):
            return math.inf

        scaled = vector / scale
        ss = float(np.sum(np.abs(scaled)))

        if not math.isfinite(ss):
            return math.inf

        norm = scale * ss
        return float(norm) if math.isfinite(norm) else math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # Similitud coseno robusta
    # ─────────────────────────────────────────────────────────────────────────
    def _safe_cosine_similarity(
        self,
        u: VectorF64,
        v: VectorF64
    ) -> Tuple[float, float, float]:
        r"""
        Calcula:
            $\cos(\theta) = \frac{\langle u, v \rangle}{\|u\|\|v\|}$

        con normalización previa para estabilidad numérica.
        """
        norm_u = self._safe_l2_norm(u)
        norm_v = self._safe_l2_norm(v)

        if not math.isfinite(norm_u) or not math.isfinite(norm_v):
            raise TopologicalMappingError(
                "Norma no finita en los vectores del espacio de búsqueda."
            )

        if norm_u <= _MACHINE_EPSILON or norm_v <= _MACHINE_EPSILON:
            raise TopologicalMappingError(
                "Vector degenerado (norma nula o subnormal) detectado en el espacio de búsqueda."
            )

        u_unit = u / norm_u
        v_unit = v / norm_v

        cos_theta = float(np.vdot(u_unit, v_unit).real)

        if not math.isfinite(cos_theta):
            raise TopologicalMappingError(
                "Similitud coseno no finita."
            )

        # Corrección por ruido de punto flotante.
        cos_theta = max(-1.0, min(1.0, cos_theta))

        return cos_theta, norm_u, norm_v

    # ─────────────────────────────────────────────────────────────────────────
    # Certificación de vecindad topológica
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_topological_neighborhood(
        self,
        query_vector: VectorF64,
        retrieved_vector: VectorF64
    ) -> TopologicalNeighborhoodData:
        r"""
        Computa el producto interno normalizado en el Espacio de Hilbert $\mathcal{H}$.

        Condición de vecindad:
            $\cos(\theta) \geq \tau_{\min}$.
        """
        q = self._as_finite_vector("query_vector", query_vector)
        r = self._as_finite_vector("retrieved_vector", retrieved_vector)

        if q.shape != r.shape:
            raise SemanticEstimatorAgentError(
                f"Vectores incompatibles: query_vector={q.shape}, "
                f"retrieved_vector={r.shape}."
            )

        cos_theta, norm_q, norm_r = self._safe_cosine_similarity(q, r)

        if cos_theta < _TAU_MIN_SIMILARITY:
            raise TopologicalMappingError(
                "Alucinación semántica interceptada. Similitud del coseno "
                f"({cos_theta:.6f}) < umbral mínimo estricto ({_TAU_MIN_SIMILARITY:.6f}). "
                "Los vectores no pertenecen a la misma vecindad homotópica."
            )

        return TopologicalNeighborhoodData(
            cosine_similarity=cos_theta,
            query_norm=norm_q,
            retrieved_norm=norm_r,
            is_homotopically_valid=True
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ÚLTIMO MÉTODO DE FASE 1
    # Puente formal hacia FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _complete_phase1_topological_certification(
        self,
        query_vector: VectorF64,
        retrieved_vector: VectorF64,
        cost_vector_c: VectorF64,
        friction_operator_F: OperatorF64,
        injection_matrix_T: MatrixF64
    ) -> Phase1TopologicalBridge:
        r"""
        Último método de la Fase 1.

        Ejecuta:
            1. Validación de entradas.
            2. Certificación de vecindad topológica.
            3. Emisión del puente funtorial hacia la Fase 2.

        Este retorno es la continuación formal de la Fase 1 y el argumento
        inicial obligatorio del primer método de la Fase 2.
        """
        q = self._as_finite_vector("query_vector", query_vector)
        r = self._as_finite_vector("retrieved_vector", retrieved_vector)

        if q.shape != r.shape:
            raise SemanticEstimatorAgentError(
                f"Vectores incompatibles: query_vector={q.shape}, "
                f"retrieved_vector={r.shape}."
            )

        c = self._as_finite_vector("cost_vector_c", cost_vector_c)

        F = self._as_finite_friction_operator(
            "friction_operator_F",
            friction_operator_F,
            dimension=c.size
        )

        T = self._as_finite_matrix("injection_matrix_T", injection_matrix_T)

        neighborhood_audit = self._certify_topological_neighborhood(q, r)

        return Phase1TopologicalBridge(
            neighborhood_audit=neighborhood_audit,
            query_vector=q,
            retrieved_vector=r,
            cost_vector_c=c,
            friction_operator_F=F,
            injection_matrix_T=T
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: AUDITORÍA DEL ENSAMBLAJE ALGEBRAICO Y FRICCIÓN TERRITORIAL        ║
# ║                                                                             ║
# ║   Φ₂(Φ₁(...)) = (κ(F_ext), C_total)                                         ║
# ║                                                                             ║
# ║   1. Consume el puente emitido por la Fase 1.                               ║
# ║   2. Valida positividad estricta del operador de fricción.                  ║
# ║   3. Acota el número de condición espectral.                                ║
# ║   4. Computa C_total = F_ext · c.                                           ║
# ║   5. Emite el puente formal hacia la Fase 3.                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_TensorFrictionAuditor(Phase1_TopologicalNeighborhoodCertifier):
    r"""
    Fase 2 del endofuntor.

    Audita el operador de fricción territorial $F_{ext}$. Si un factor es anómalo
    (ej. $\kappa(F_{ext}) \gg 1$), evita la corrupción termodinámica del costo.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMER MÉTODO DE FASE 2
    # Inicio formal a partir del puente de Fase 1
    # ─────────────────────────────────────────────────────────────────────────
    def _begin_phase2_from_phase1_bridge(
        self,
        phase1_bridge: Phase1TopologicalBridge
    ) -> Phase2FrictionBridge:
        r"""
        Primer método de la Fase 2.

        Consume el `Phase1TopologicalBridge` emitido por el último método de la
        Fase 1 y ejecuta la auditoría de fricción territorial.
        """
        if not isinstance(phase1_bridge, Phase1TopologicalBridge):
            raise SemanticEstimatorAgentError(
                "La Fase 2 requiere un Phase1TopologicalBridge emitido por la Fase 1."
            )

        friction_audit = self._audit_tensor_friction_assembly(
            cost_vector_c=phase1_bridge.cost_vector_c,
            friction_operator_F=phase1_bridge.friction_operator_F
        )

        return Phase2FrictionBridge(
            phase1_bridge=phase1_bridge,
            friction_audit=friction_audit
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Auditoría de ensamblaje algebraico y fricción
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_tensor_friction_assembly(
        self,
        cost_vector_c: VectorF64,
        friction_operator_F: OperatorF64
    ) -> TensorFrictionData:
        r"""
        Valida:
            $C_{total} = F_{ext} \cdot c$,
        la positividad estricta y el condicionamiento espectral.

        Para operadores diagonales (vector 1-D):
            $\kappa(F) = \frac{\max_i F_i}{\min_i F_i}$.

        Para operadores matriciales (matriz 2-D):
            Se exige que la parte simétrica sea definida positiva y:
            $\kappa(F) = \frac{\lambda_{\max}}{\lambda_{\min}}$.
        """
        c = self._as_finite_vector("cost_vector_c", cost_vector_c)
        F = self._as_finite_friction_operator(
            "friction_operator_F",
            friction_operator_F,
            dimension=c.size
        )

        # ── Saneamiento del vector de costos ─────────────────────────────────
        c_clean = c.copy()

        small_negative_c = (c_clean < 0.0) & (c_clean >= -_NEGATIVE_TOLERANCE)
        c_clean[small_negative_c] = 0.0

        if np.any(c_clean < 0.0):
            raise ThermodynamicFrictionAnomaly(
                "Inyección de energía negativa en el vector de costos. "
                "cost_vector_c contiene componentes negativas no infinitesimales."
            )

        # ── Auditoría del operador de fricción ───────────────────────────────
        if F.ndim == 1:
            diag = F.copy()

            small_negative_diag = (diag < 0.0) & (diag >= -_NEGATIVE_TOLERANCE)
            diag[small_negative_diag] = 0.0

            if np.any(diag < 0.0):
                raise ThermodynamicFrictionAnomaly(
                    "Inyección de energía negativa en el operador de fricción diagonal. "
                    "friction_operator_F contiene factores negativos no infinitesimales."
                )

            spectral_min = float(np.min(diag))
            spectral_max = float(np.max(diag))

            if spectral_min <= _POSITIVE_FLOOR:
                raise ThermodynamicFrictionAnomaly(
                    "Operador de fricción diagonal singular o no positivo. "
                    f"min(diag)={spectral_min:.6e} <= piso positivo {_POSITIVE_FLOOR:.6e}."
                )

            condition_number = float(spectral_max / spectral_min)
            total_cost_vector = diag * c_clean

        else:
            F_clean = F.copy()

            small_negative_F = (F_clean < 0.0) & (F_clean >= -_NEGATIVE_TOLERANCE)
            F_clean[small_negative_F] = 0.0

            if np.any(F_clean < 0.0):
                raise ThermodynamicFrictionAnomaly(
                    "Inyección de energía negativa en el operador de fricción matricial. "
                    "friction_operator_F contiene entradas negativas no infinitesimales."
                )

            # El operador territorial se modela como auto-adjunto.
            F_sym = 0.5 * (F_clean + F_clean.T)

            fro_original = self._safe_fro_norm(F_clean)
            fro_sym = self._safe_fro_norm(F_sym)
            fro_asym = self._safe_fro_norm(F_clean - F_sym)

            if math.isfinite(fro_original) and math.isfinite(fro_asym):
                if fro_asym > 1e-8 * max(1.0, fro_original):
                    logger.warning(
                        "Operador de fricción con asimetría relevante. "
                        f"||F-F^T||_F={fro_asym:.3e}. Se impone parte simétrica."
                    )
            elif not math.isfinite(fro_asym):
                logger.warning(
                    "No fue posible certificar finiteza de la asimetría del operador de fricción. "
                    "Se procede bajo simetrización forzada."
                )

            try:
                eigenvalues = la.eigvalsh(F_sym, check_finite=False)
            except la.LinAlgError as exc:
                raise ThermodynamicFrictionAnomaly(
                    "El operador de fricción territorial es numéricamente singular o no diagonalizable."
                ) from exc

            eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

            if not np.all(np.isfinite(eigenvalues)):
                raise ThermodynamicFrictionAnomaly(
                    "El espectro del operador de fricción contiene valores no finitos."
                )

            spectral_min = float(np.min(eigenvalues))
            spectral_max = float(np.max(eigenvalues))

            if spectral_min <= _POSITIVE_FLOOR:
                raise ThermodynamicFrictionAnomaly(
                    "Operador de fricción no definido positivo. "
                    f"lambda_min={spectral_min:.6e} <= piso positivo {_POSITIVE_FLOOR:.6e}."
                )

            condition_number = float(spectral_max / spectral_min)
            total_cost_vector = F_sym @ c_clean

        if not math.isfinite(condition_number):
            raise ThermodynamicFrictionAnomaly(
                "Número de condición del operador de fricción no finito."
            )

        if condition_number > _MAX_FRICTION_CONDITION:
            raise ThermodynamicFrictionAnomaly(
                "Anomalía termodinámica detectada. El número de condición del operador de fricción "
                f"κ(F_ext)={condition_number:.6e} excede el límite {_MAX_FRICTION_CONDITION:.6e}. "
                "El terreno induce un sobrecosto asimétrico geométricamente degenerado."
            )

        if not np.all(np.isfinite(total_cost_vector)):
            raise ThermodynamicFrictionAnomaly(
                "El costo total ensamblado contiene componentes no finitas."
            )

        total_clean = np.asarray(total_cost_vector, dtype=np.float64).copy()

        small_negative_total = (total_clean < 0.0) & (total_clean >= -_NEGATIVE_TOLERANCE)
        total_clean[small_negative_total] = 0.0

        if np.any(total_clean < 0.0):
            raise ThermodynamicFrictionAnomaly(
                "Costo total ensamblado con componentes negativas no infinitesimales."
            )

        total_cost_norm = self._safe_l1_norm(total_clean)

        if not math.isfinite(total_cost_norm):
            raise ThermodynamicFrictionAnomaly(
                "Norma L1 del costo total no finita."
            )

        return TensorFrictionData(
            condition_number=condition_number,
            spectral_min=spectral_min,
            spectral_max=spectral_max,
            total_cost_norm=total_cost_norm,
            total_cost_vector=total_clean,
            is_positive_definite=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: IMPOSICIÓN DEL TEOREMA DE RANGO-NULIDAD                           ║
# ║                                                                             ║
# ║   Φ₃(Φ₂(Φ₁(...))) = rank(T)                                                 ║
# ║                                                                             ║
# ║   1. Consume el puente emitido por la Fase 2.                               ║
# ║   2. Computa SVD robusto de la matriz de inyección T.                       ║
# ║   3. Exige rank(T) = 1.                                                     ║
# ║   4. Certifica que T sea una isometría parcial ortogonal.                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_RankNullityProjector(Phase2_TensorFrictionAuditor):
    r"""
    Fase 3 del endofuntor.

    Garantiza que la inyección del servicio en la Matriz de Interacción Central (MIC)
    acte como un proyector ortogonal estricto / isometría parcial de rango 1,
    logrando Cero Efectos Secundarios.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMER MÉTODO DE FASE 3
    # Inicio formal a partir del puente de Fase 2
    # ─────────────────────────────────────────────────────────────────────────
    def _begin_phase3_from_phase2_bridge(
        self,
        phase2_bridge: Phase2FrictionBridge
    ) -> RankNullityProjectionData:
        r"""
        Primer método de la Fase 3.

        Consume el `Phase2FrictionBridge` emitido por la Fase 2 y ejecuta la
        imposición de rango-nulidad sobre la matriz de inyección.
        """
        if not isinstance(phase2_bridge, Phase2FrictionBridge):
            raise SemanticEstimatorAgentError(
                "La Fase 3 requiere un Phase2FrictionBridge emitido por la Fase 2."
            )

        return self._enforce_rank_nullity_projection(
            injection_matrix_T=phase2_bridge.phase1_bridge.injection_matrix_T
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Imposición de rango-nulidad e inyección ortogonal
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_rank_nullity_projection(
        self,
        injection_matrix_T: MatrixF64
    ) -> RankNullityProjectionData:
        r"""
        Computa SVD para extraer el rango efectivo y auditar la ortogonalidad.

        Condición axiomática:
            $\operatorname{rank}(T) = 1$.

        Condición de inyección ortogonal:
            $T$ debe ser una isometría parcial de rango 1, es decir,
            su único valor singular no nulo debe satisfacer:
                $\sigma_1 = 1$,
            y los proyectores inducidos:
                $P_{row} = T^T T$,
                $P_{col} = T T^T$,
            deben ser idempotentes y simétricos numéricamente.
        """
        T = self._as_finite_matrix("injection_matrix_T", injection_matrix_T)
        m, n = T.shape

        try:
            _, singular_values, _ = la.svd(
                T,
                full_matrices=False,
                check_finite=False
            )
        except la.LinAlgError as exc:
            raise FunctorialityError(
                "Fallo en la descomposición SVD de la matriz de inyección T."
            ) from exc

        if singular_values.size == 0:
            raise FunctorialityError(
                "La matriz de inyección no produjo valores singulares."
            )

        s = np.asarray(singular_values, dtype=np.float64)

        if not np.all(np.isfinite(s)):
            raise FunctorialityError(
                "Los valores singulares de T contienen NaN o infinitos."
            )

        sigma_max = float(s[0])

        rank_tolerance = max(
            _SVD_ABSOLUTE_TOLERANCE,
            float(max(m, n)) * _MACHINE_EPSILON * max(sigma_max, 1.0)
        )

        effective_rank = int(np.sum(s > rank_tolerance))
        kernel_dimension = int(n - effective_rank)

        if effective_rank != 1:
            raise FunctorialityError(
                "Violación del Teorema de Rango-Nulidad. El morfismo de inyección "
                f"tiene un rango defectuoso o hiper-acoplado (Rank={effective_rank}). "
                "Se requiere axiomáticamente Rank=1 para evitar efectos secundarios en la MIC."
            )

        if not math.isfinite(sigma_max) or sigma_max <= rank_tolerance:
            raise FunctorialityError(
                "Valor singular dominante no finito o numéricamente nulo."
            )

        # Normalización por el valor singular dominante para auditar la
        # isometría parcial sin contaminación de escala.
        T_unit = T / sigma_max

        try:
            P_row = T_unit.T @ T_unit
            P_col = T_unit @ T_unit.T
        except Exception as exc:
            raise FunctorialityError(
                "Fallo al construir los proyectores inducidos por la inyección."
            ) from exc

        if not np.all(np.isfinite(P_row)) or not np.all(np.isfinite(P_col)):
            raise FunctorialityError(
                "Los proyectores inducidos por la inyección contienen entradas no finitas."
            )

        row_sym_deviation = self._safe_fro_norm(P_row - P_row.T)
        row_idempotence_deviation = self._safe_fro_norm(P_row @ P_row - P_row)

        col_sym_deviation = self._safe_fro_norm(P_col - P_col.T)
        col_idempotence_deviation = self._safe_fro_norm(P_col @ P_col - P_col)

        sigma_deviation = abs(sigma_max - 1.0)

        orthogonality_deviation = max(
            sigma_deviation,
            row_sym_deviation,
            row_idempotence_deviation,
            col_sym_deviation,
            col_idempotence_deviation
        )

        if not math.isfinite(orthogonality_deviation):
            raise FunctorialityError(
                "Desviación de ortogonalidad no finita."
            )

        t_unit_fro = self._safe_fro_norm(T_unit)

        if not math.isfinite(t_unit_fro):
            raise FunctorialityError(
                "Norma de Frobenius de la inyección normalizada no finita."
            )

        ortho_tolerance = max(
            _ORTHOGONALITY_TOLERANCE,
            100.0 * _MACHINE_EPSILON * max(1.0, t_unit_fro * t_unit_fro)
        )

        is_orthogonal = orthogonality_deviation <= ortho_tolerance

        if not is_orthogonal:
            raise FunctorialityError(
                "La matriz de inyección no es una isometría parcial ortogonal. "
                f"Desviación={orthogonality_deviation:.6e} > tolerancia={ortho_tolerance:.6e}. "
                "Se viola el aislamiento ortogonal en la MIC."
            )

        return RankNullityProjectionData(
            matrix_shape=(m, n),
            effective_rank=effective_rank,
            kernel_dimension=kernel_dimension,
            largest_singular_value=sigma_max,
            rank_tolerance=rank_tolerance,
            orthogonality_deviation=orthogonality_deviation,
            is_orthogonal_injection=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: SEMANTIC ESTIMATOR AGENT                             ║
# ║                                                                             ║
# ║   Endofuntor:                                                               ║
# ║       Z_EstimatorAgent = Φ₃ ∘ Φ₂ ∘ Φ₁                                       ║
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
        query_vector: VectorF64,
        retrieved_vector: VectorF64,
        cost_vector_c: VectorF64,
        friction_operator_F: OperatorF64,
        injection_matrix_T: MatrixF64
    ) -> SemanticEstimatorAuditState:
        r"""
        Ejecuta la composición funtorial estricta en 3 fases anidadas.

        Flujo:
        ----
        1. Fase 1:
           `_complete_phase1_topological_certification`
           → `Phase1TopologicalBridge`

        2. Fase 2:
           `_begin_phase2_from_phase1_bridge`
           → `Phase2FrictionBridge`

        3. Fase 3:
           `_begin_phase3_from_phase2_bridge`
           → `RankNullityProjectionData`

        4. Ensamblaje final:
           `SemanticEstimatorAuditState`
        """
        # ── Fase 1: Certificar la vecindad topológica del mapeo FAISS ────────
        phase1_bridge = self._complete_phase1_topological_certification(
            query_vector=query_vector,
            retrieved_vector=retrieved_vector,
            cost_vector_c=cost_vector_c,
            friction_operator_F=friction_operator_F,
            injection_matrix_T=injection_matrix_T
        )

        # ── Fase 2: Certificar estabilidad del ensamblaje del tensor de costos ─
        phase2_bridge = self._begin_phase2_from_phase1_bridge(
            phase1_bridge=phase1_bridge
        )

        # ── Fase 3: Proyectar la capacidad garantizando aislamiento en la MIC ─
        projection_audit = self._begin_phase3_from_phase2_bridge(
            phase2_bridge=phase2_bridge
        )

        # ── Ensamblaje del objeto final ──────────────────────────────────────
        final_state = SemanticEstimatorAuditState(
            neighborhood_audit=phase1_bridge.neighborhood_audit,
            friction_audit=phase2_bridge.friction_audit,
            projection_audit=projection_audit,
            is_epistemologically_valid=True
        )

        logger.info(
            "Auditoría Semántica y Vectorial completada. "
            f"Cos(θ): {final_state.neighborhood_audit.cosine_similarity:.6f} | "
            f"κ(F): {final_state.friction_audit.condition_number:.6e} | "
            f"Rank(T): {final_state.projection_audit.effective_rank} | "
            f"Ortho_dev: {final_state.projection_audit.orthogonality_deviation:.6e}"
        )

        return final_state


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "SemanticEstimatorAgentError",
    "TopologicalMappingError",
    "ThermodynamicFrictionAnomaly",
    "FunctorialityError",
    "TopologicalNeighborhoodData",
    "TensorFrictionData",
    "RankNullityProjectionData",
    "Phase1TopologicalBridge",
    "Phase2FrictionBridge",
    "SemanticEstimatorAuditState",
    "Phase1_TopologicalNeighborhoodCertifier",
    "Phase2_TensorFrictionAuditor",
    "Phase3_RankNullityProjector",
    "SemanticEstimatorAgent",
]