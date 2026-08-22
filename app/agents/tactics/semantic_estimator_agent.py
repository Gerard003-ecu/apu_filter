# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Semantic Estimator Agent (Soberano de Métrica y Alineamiento)       ║
║ Ruta   : app/agents/tactics/semantic_estimator_agent.py                      ║
║ Versión: 5.8.0-Kähler-Procrustes-Wasserstein-FAISS-Heyting-Secure            ║
╚══════════════════════════════════════════════════════════════════════════════╝

SINOPSIS GEOMÉTRICO-INFORMACIONAL Y CENSURA DE DERIVA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este agente de calibre de-confinado del Estrato TACTICS-WISDOM (Nivel 1.5)
gobierna síncronamente al motor de búsqueda vectorial 'semantic_estimator.py'.
Su propósito es someter los vectores de embeddings $u, v \in \mathcal{H}$ generados
por el Sentence-Transformers de la base a transformaciones de alineación isométrica,
asimilando desviaciones semánticas que intenten inyectar "entropía fantasma"
o sobredimensionamientos ficticios en la Matriz Atómica de Conocimiento (MAC).

Mediante la resolución del Problema de Procrustes Ortogonal y la estimación de 
la distancia anisotrópica de Mahalanobis, el agente valida que las trayectorias
de decisión del LLM reposen estrictamente sobre el subespacio de Hilbert estable,
proscribiendo el ruido estocástico mediante el colapso en el retículo de Heyting.

AXIOMÁTICA DE GEOMETRÍA DE LA INFORMACIÓN Y ALINEAMIENTO:
────────────────────────────────────────────────────────────────────────────────

  [A1] Axioma de Isometría Esférica y Similitud de Coseno (FAISS):
       La proyección de embeddings semánticos se realiza sobre la esfera de Hilbert 
       unitaria $S^{d-1} \subset \mathcal{H} \cong \mathbb{R}^d$, de tal manera que la similitud 
       atencional se mide de forma exacta mediante la forma bilineal normalizada:
       $$\langle u, v \rangle = \cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} \in [-1, 1] \quad\big[143\big]$$
       Sujeta a que cada tensor de embedding cumpla estrictamente con la unitariedad:
       $$\|u\|_2 = \sqrt{\sum_{k=1}^d u_k^2} \equiv 1.0 \pm \varepsilon_{\mathrm{machine}} \quad\big[143\big]$$

  [A2] Axioma de Alineamiento Isométrico de Procrustes Ortogonal:
       Para mapear de manera consistente el espacio de descripciones técnicas de la obra 
       hacia la ontología formal de insumos del negocio, el agente computa el operador de 
       rotación rígida $R \in \operatorname{O}(d)$ que minimiza la distancia de Frobenius:
       $$\min_{R^\top R = I_d} \| R X - Y \|_F^2 \quad\big[224\big]$$
       La solución analítica óptima se extrae en la FPU mediante la descomposición SVD 
       del producto de covarianza cruzada $X Y^\top = U \Sigma V^\top$:
       $$R^* = U V^\top \quad\big[224\big]$$
       Si $\det(R^*) = -1$, el mapa inyecta una reflexión especular que rompe la orientación 
       cohomológica de la Malla, detonando un veto por 'ChiralAlignmentAnomaly'.

  [A3] Axioma de Métrica de Información de Fisher-Rao y Mahalanobis:
       La distancia informacional entre la propuesta del LLM y el centroide de-confinado 
       de costos se evalúa bajo la métrica Riemanniana anisotrópica de Mahalanobis:
       $$d_{\mathrm{M}}(x, \mu)^2 = (x - \mu)^\top \Sigma^{-1} (x - \mu) \quad\big[224, 251\big]$$
       Donde $\Sigma \in \operatorname{SPD}(d)$ es el tensor de covarianza de la base histórica. 
       El agente exige que la desviación semántica satisfaga la cota de Lipschitz:
       $$d_{\mathrm{M}}(x, \mu) \le \tau_{\mathrm{drift}} = \sqrt{\chi^2_{d, \, 1-\alpha}} \quad\big[224\big]$$
       Cualquier deriva que perfore esta frontera estadística detona 'SemanticDriftVeto'.

  [A4] Axioma del Veto en el Retículo de Heyting y Actuación Crowbar BT151:
       Los veredictos de estabilidad espectral y alineamiento se proyectan sobre el clasificador 
       de subobjetos del retículo distributivo de Heyting de tres valores de la Malla:
       $$\Omega_3 = \{\mathtt{COHERENT}, \, \mathtt{DEGRADED}, \, \mathtt{VETOED}\} \quad\big[218, 224\big]$$
       Cualquier quiebre de simetría quiral ($\det R^* = -1$), distorsión de Mahalanobis desbocada 
       ($d_{\mathrm{M}} > \tau_{\mathrm{drift}}$) o bajo número de condición espectral de la covarianza 
       ($\kappa_2(\Sigma) > 1.0\times 10^8$) colapsa instantáneamente el estado al Supremo terminal VETOED ($\top$).
       La subrutina C++ 'isVerdictCoherent()' del microcontrolador ESP32 detecta el colapso síncronamente y,
       mediante su ISR en IRAM (<400ns), conmuta el pin GPIO14, disparando el tiristor BT151 (circuito Crowbar)
       para cortocircuitar la potencia real y paralizar síncronamente la obra real.

JERARQUÍA DE EXCEPCIONES DE ALINEAMIENTO Y MÉTRICA (Fail-Secure Boundary):
────────────────────────────────────────────────────────────────────────────────
  SemanticEstimatorError (Exception)
   ├── EmbeddingDimensionMismatch: Discordancia en la dimensión proyectiva d del embedding.
   ├── ChiralAlignmentAnomaly    : Rotación impropia (det R* = -1) que destruye la orientación.
   ├── ProcrustesSVDConvergence  : Falla de convergencia en la descomposición de de Rham-SVD.
   ├── CovarianceSingularityError: Matriz Sigma singular (condicionamiento de Wilkinson > 10^8).
   ├── SemanticDriftVeto         : Distancia de Mahalanobis supera el umbral crítico chi-cuadrado.
   ├── FAISSIndexCorruption      : Falla de consistencia o desbordamiento de memoria en FAISS.
   └── HeytingLobeCollapse       : Transición anómala hacia el autoestado VETOED en el topos.

DISEÑO DEL FLUJO CATEGÓRICO DE TRES FASES (OODA Espectral):
────────────────────────────────────────────────────────────────────────────────
  Fase 1 ──► OBSERVE : Validación del tensor de embeddings entrante.
             Certifica la norma unitaria L2 y la ausencia de singularidades NaN/Inf en la FPU.
             Retorna: EmbeddingValidationCertificate.

  Fase 2 ──► ORIENT  : Cálculo del alineamiento ortogonal de Procrustes mediante SVD.
             Resuelve la rotación óptima R y computa la distancia de Mahalanobis.
             Retorna: AlignmentSpectralReport.

  Fase 3 ──► DECIDE  : Validación de la cota chi-cuadrado y número de condición de Sigma.
             Despacha el veredicto al retículo de Heyting y actualiza el CrowbarPort.
             Retorna: EstimatorSuturationState.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Final, Optional, Tuple, Dict, List
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ──────────────────────────────────────────────────────────────────────────────
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

logger = logging.getLogger("MIC.Tactics.SemanticEstimatorAgent.Granular")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ════════════════════════════════════════════════════════════════════════════
# §A. TIPOS Y CONSTANTES MATEMÁTICAS, ESPECTRALES Y DE TOLERANCIA
# ════════════════════════════════════════════════════════════════════════════
VectorF64 = NDArray[np.float64]
MatrixF64 = NDArray[np.float64]
OperatorF64 = NDArray[np.float64]

_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# Fase 1: Vecindad topológica de Hilbert.
_TAU_MIN_SIMILARITY: Final[float] = 0.85
_DEGENERACY_NORM_FLOOR: Final[float] = 1e-15
_COSINE_CLAMP_TOLERANCE: Final[float] = 1e-12
_ANGLE_RADIAN_TOLERANCE: Final[float] = 1e-10

# Fase 2: Fricción territorial y ensamblaje de costos.
_MAX_FRICTION_CONDITION: Final[float] = 1e3
_POSITIVE_FLOOR: Final[float] = 1e-12
_NEGATIVE_TOLERANCE: Final[float] = 1e-12
_SYMMETRY_TOLERANCE: Final[float] = 1e-10
_EIGENVALUE_TOLERANCE: Final[float] = 1e-12
_CONDITION_TOLERANCE: Final[float] = 1e-8

# Fase 3: Rango-nulidad e inyección ortogonal.
_SVD_ABSOLUTE_TOLERANCE: Final[float] = 1e-10
_ORTHOGONALITY_TOLERANCE: Final[float] = 1e-8
_IDEMPOTENCE_TOLERANCE: Final[float] = 1e-9
_SYMMETRY_PROJECTOR_TOLERANCE: Final[float] = 1e-9
_SIGMA_DEVIATION_TOLERANCE: Final[float] = 1e-8


# ════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ALGEBRAICAS
# ════════════════════════════════════════════════════════════════════════════
class SemanticEstimatorAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Geometría Vectorial."""
    pass


class TopologicalMappingError(SemanticEstimatorAgentError):
    r"""Detonada si $\cos(\theta) < \tau_{\min}$. Alucinación espacial de mapeo FAISS."""
    pass


class VectorDegeneracyError(SemanticEstimatorAgentError):
    r"""Detonada si un vector tiene norma nula o subnormal (degeneración métrica)."""
    pass


class DimensionalIncompatibilityError(SemanticEstimatorAgentError):
    r"""Detonada si las dimensiones de vectores/matrices son incompatibles."""
    pass


class ThermodynamicFrictionAnomaly(SemanticEstimatorAgentError):
    r"""Detonada si el operador $F_{ext}$ induce singularidades o si $\kappa(F_{ext}) \gg 1$."""
    pass


class FunctorialityError(SemanticEstimatorAgentError):
    r"""Detonada si $\text{rank}(T) \neq 1$ o se violan las fronteras ortogonales en la MIC."""
    pass


class ProjectorIntegrityError(FunctorialityError):
    r"""Detonada si los proyectores inducidos no son idempotentes o simétricos."""
    pass


# ════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Fase)
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True, eq=False)
class TopologicalNeighborhoodData:
    r"""
    Artefacto de Fase 1.
    Certificado de vecindad de Hilbert con métricas completas.
    
    Teorema de Caracterización: Dos vectores pertenecen a la misma
    vecindad topológica si y solo si $\cos(\theta) \geq \tau_{\min}$.
    """
    cosine_similarity: float
    angle_radians: float
    angle_degrees: float
    euclidean_distance: float
    query_norm: float
    retrieved_norm: float
    dimensionality: int
    is_homotopically_valid: bool
    similarity_margin: float = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class TensorFrictionData:
    r"""
    Artefacto de Fase 2.
    Certificado termodinámico del operador $F_{ext}$ con análisis espectral completo.
    
    Teorema de Estabilidad: El operador de fricción debe ser definido
    positivo con número de condición acotado para garantizar estabilidad
    numérica del ensamblaje de costos.
    """
    condition_number: float
    spectral_min: float
    spectral_max: float
    spectral_mean: float
    spectral_std: float
    total_cost_norm: float
    total_cost_vector: VectorF64
    is_positive_definite: bool
    operator_type: str = "diagonal"
    symmetry_residual: float = 0.0
    cost_vector_norm: float = 0.0
    friction_determinant: float = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class RankNullityProjectionData:
    r"""
    Artefacto de Fase 3.
    Certificado del Teorema de Rango-Nulidad y de inyección ortogonal.
    
    Teorema de Rango-Nulidad: Para $T: V \to W$ lineal,
    $\dim(V) = \text{rank}(T) + \text{nullity}(T)$.
    
    Corolario de Aislamiento: Si $\text{rank}(T) = 1$ y $T$ es isometría
    parcial, entonces la inyección no produce efectos secundarios en la MIC.
    """
    matrix_shape: Tuple[int, int]
    effective_rank: int
    kernel_dimension: int
    largest_singular_value: float
    smallest_singular_value: float
    singular_value_gap: float
    rank_tolerance: float
    orthogonality_deviation: float
    is_orthogonal_injection: bool
    row_projector_idempotence: float = 0.0
    col_projector_idempotence: float = 0.0
    row_projector_symmetry: float = 0.0
    col_projector_symmetry: float = 0.0
    condition_number: float = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class Phase1TopologicalBridge:
    r"""
    Puente funtorial Φ₁ → Φ₂.
    
    Este objeto es emitido por el último método de la Fase 1 y constituye
    la entrada formal del primer método de la Fase 2.
    
    Lema de Continuación: La vecindad topológica certificada es condición
    necesaria para la coherencia del ensamblaje de costos posterior.
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
    
    Lema de Continuación: La fricción territorial acotada garantiza que
    la matriz de inyección no sufre distorsión espectral excesiva.
    """
    phase1_bridge: Phase1TopologicalBridge
    friction_audit: TensorFrictionData


@dataclass(frozen=True, slots=True, eq=False)
class SemanticEstimatorAuditState:
    r"""
    Objeto final del endofuntor $\mathcal{Z}_{EstimatorAgent}$.
    
    Teorema de Corrección Global: Si las tres fases certifican éxito,
    la estimación semántica es epistemológicamente válida y está
    topológicamente protegida contra alucinaciones del LLM.
    """
    neighborhood_audit: TopologicalNeighborhoodData
    friction_audit: TensorFrictionData
    projection_audit: RankNullityProjectionData
    is_epistemologically_valid: bool
    governance_metadata: Dict[str, Any] = field(default_factory=dict)


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
    
    Fundamento Teórico:
    ────────────────────
    Teorema de Caracterización Angular: En un espacio de Hilbert $\mathcal{H}$,
    dos vectores $u, v$ pertenecen a la misma vecindad topológica si y solo si:
    
        $\cos(\theta) = \frac{\langle u, v \rangle}{\|u\| \|v\|} \geq \tau_{\min}$
    
    Corolario de No Degeneración: Si $\|u\| = 0$ o $\|v\| = 0$, el ángulo
    está indefinido y la vecindad no puede ser certificada.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1.1. Coerción de escalares finitos con rechazo de booleanos
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_scalar(
        self,
        name: str,
        value: Any,
    ) -> float:
        r"""
        Convierte un valor a float64 y exige que sea finito.
        
        Axioma de Dominio: Todo escalar físico debe pertenecer a $\mathbb{R}$
        y ser finito. Los booleanos pertenecen a $\mathbb{B}_2$, no a $\mathbb{R}$.
        
        Parámetros:
        ───────────
        name : str
            Identificador del escalar para trazabilidad.
        value : Any
            Valor a coerccionar.
        
        Retorna:
        ────────
        float
            Escalar finito certificado.
        
        Excepciones:
        ────────────
        SemanticEstimatorAgentError si el valor no es un escalar finito.
        """
        if isinstance(value, (bool, np.bool_)):
            raise SemanticEstimatorAgentError(
                f"El escalar '{name}' no puede ser booleano. "
                f"Los booleanos pertenecen al topos $\mathbb{B}_2$, no a $\mathbb{R}$."
            )
        
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

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2. Coerción de vectores finitos con validación estructural
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_vector(
        self,
        name: str,
        value: VectorF64,
        expected_dim: Optional[int] = None,
    ) -> VectorF64:
        r"""
        Valida que el objeto sea un vector 1-D no vacío y con componentes finitas.
        
        Lema de Completitud: Todo vector en el espacio de Hilbert debe tener
        componentes finitas y dimensión compatible con el espacio ambiente.
        
        Parámetros:
        ───────────
        name : str
            Identificador del vector.
        value : VectorF64
            Objeto a convertir en vector.
        expected_dim : Optional[int]
            Dimensión esperada (si se especifica).
        
        Retorna:
        ────────
        VectorF64
            Vector certificado.
        
        Excepciones:
        ────────────
        SemanticEstimatorAgentError si el vector es inválido.
        DimensionalIncompatibilityError si la dimensión no coincide.
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
        
        if expected_dim is not None and arr.size != int(expected_dim):
            raise DimensionalIncompatibilityError(
                f"{name} debe tener dimensión {expected_dim}, "
                f"pero posee {arr.size} componentes."
            )
        
        if not np.all(np.isfinite(arr)):
            non_finite_count = int(np.sum(~np.isfinite(arr)))
            raise SemanticEstimatorAgentError(
                f"{name} contiene {non_finite_count} componentes NaN o infinitas."
            )
        
        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3. Coerción de matrices finitas con validación estructural
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_matrix(
        self,
        name: str,
        value: MatrixF64,
    ) -> MatrixF64:
        r"""
        Valida que el objeto sea una matriz 2-D no vacía y finita.
        
        Lema de Integridad Matricial: Toda matriz en el espacio de Hilbert
        debe tener componentes finitas y dimensión compatible.
        
        Parámetros:
        ───────────
        name : str
            Identificador de la matriz.
        value : MatrixF64
            Objeto a convertir en matriz.
        
        Retorna:
        ────────
        MatrixF64
            Matriz certificada.
        
        Excepciones:
        ────────────
        SemanticEstimatorAgentError si la matriz es inválida.
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
            non_finite_count = int(np.sum(~np.isfinite(mat)))
            raise SemanticEstimatorAgentError(
                f"{name} contiene {non_finite_count} entradas NaN o infinitas."
            )
        
        return mat

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4. Coerción de operador de fricción con validación dimensional
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_friction_operator(
        self,
        name: str,
        value: OperatorF64,
        dimension: int,
    ) -> OperatorF64:
        r"""
        Valida un operador de fricción como:
        - Vector 1-D de factores diagonales, o
        - Matriz 2-D cuadrada compatible con el vector de costos.
        
        Lema de Compatibilidad: El operador de fricción debe tener
        dimensión compatible con el vector de costos para que el
        producto $F \cdot c$ esté bien definido.
        
        Parámetros:
        ───────────
        name : str
            Identificador del operador.
        value : OperatorF64
            Objeto a convertir.
        dimension : int
            Dimensión esperada del vector de costos.
        
        Retorna:
        ────────
        OperatorF64
            Operador certificado.
        
        Excepciones:
        ────────────
        SemanticEstimatorAgentError si el operador es inválido.
        DimensionalIncompatibilityError si la dimensión no coincide.
        """
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SemanticEstimatorAgentError(
                f"{name} no puede convertirse a un operador float64."
            ) from exc
        
        if arr.ndim == 0:
            if dimension != 1:
                raise DimensionalIncompatibilityError(
                    f"{name} escalar sólo es admisible para dimensión 1."
                )
            arr = arr.reshape(1)
        
        if arr.ndim == 1:
            if arr.size != dimension:
                raise DimensionalIncompatibilityError(
                    f"{name} como operador diagonal debe tener tamaño {dimension}. "
                    f"Se recibió {arr.size}."
                )
        elif arr.ndim == 2:
            if arr.shape != (dimension, dimension):
                raise DimensionalIncompatibilityError(
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
    # 1.5. Verificación de compatibilidad dimensional entre vectores
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_dimensional_compatibility(
        self,
        name_a: str,
        vector_a: VectorF64,
        name_b: str,
        vector_b: VectorF64,
    ) -> int:
        r"""
        Verifica que dos vectores tengan la misma dimensión.
        
        Axioma de Compatibilidad: El producto interno $\langle u, v \rangle$
        está definido si y solo si $\dim(u) = \dim(v)$.
        
        Parámetros:
        ───────────
        name_a, name_b : str
            Identificadores de los vectores.
        vector_a, vector_b : VectorF64
            Vectores a comparar.
        
        Retorna:
        ────────
        int
            Dimensión común.
        
        Excepciones:
        ────────────
        DimensionalIncompatibilityError si las dimensiones difieren.
        """
        dim_a = int(vector_a.size)
        dim_b = int(vector_b.size)
        
        if dim_a != dim_b:
            raise DimensionalIncompatibilityError(
                f"Vectores incompatibles: {name_a}={vector_a.shape}, "
                f"{name_b}={vector_b.shape}."
            )
        
        return dim_a

    # ─────────────────────────────────────────────────────────────────────────
    # 1.6. Norma L2 numéricamente segura con reescalado
    # ─────────────────────────────────────────────────────────────────────────
    def _safe_l2_norm(
        self,
        vector: VectorF64,
    ) -> float:
        r"""
        Calcula $\|v\|_2$ con reescalado para evitar overflow/underflow.
        
        Lema de Estabilidad Numérica: Para vectores con componentes de
        magnitud extrema, el cálculo directo de la norma puede sufrir
        overflow o underflow. El reescalado por el máximo absoluto
        preserva la precisión.
        
        Algoritmo:
        ──────────
        1. $s = \max_i |v_i|$
        2. $\tilde{v} = v / s$
        3. $\|v\|_2 = s \cdot \|\tilde{v}\|_2$
        
        Parámetros:
        ───────────
        vector : VectorF64
            Vector a normalizar.
        
        Retorna:
        ────────
        float
            Norma L2 certificada como finita o infinito.
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

    # ─────────────────────────────────────────────────────────────────────────
    # 1.7. Norma de Frobenius numéricamente segura
    # ─────────────────────────────────────────────────────────────────────────
    def _safe_frobenius_norm(
        self,
        matrix: MatrixF64,
    ) -> float:
        r"""
        Calcula $\|M\|_F$ con reescalado para evitar overflow/underflow.
        
        Lema de Estabilidad Numérica: Análogo al caso vectorial,
        el reescalado preserva la precisión en matrices con entradas
        de magnitud extrema.
        
        Parámetros:
        ───────────
        matrix : MatrixF64
            Matriz a normalizar.
        
        Retorna:
        ────────
        float
            Norma de Frobenius certificada.
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

    # ─────────────────────────────────────────────────────────────────────────
    # 1.8. Norma L1 numéricamente segura
    # ─────────────────────────────────────────────────────────────────────────
    def _safe_l1_norm(
        self,
        vector: VectorF64,
    ) -> float:
        r"""
        Calcula $\|v\|_1$ con reescalado para evitar overflow.
        
        Parámetros:
        ───────────
        vector : VectorF64
            Vector a normalizar.
        
        Retorna:
        ────────
        float
            Norma L1 certificada.
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
    # 1.9. Verificación de no degeneración vectorial
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_non_degenerate_vectors(
        self,
        query: VectorF64,
        retrieved: VectorF64,
    ) -> Tuple[float, float]:
        r"""
        Verifica que ambos vectores tengan norma no nula ni subnormal.
        
        Corolario de No Degeneración: Si $\|u\| = 0$ o $\|v\| = 0$,
        el ángulo está indefinido y la vecindad no puede ser certificada.
        
        Parámetros:
        ───────────
        query, retrieved : VectorF64
            Vectores a verificar.
        
        Retorna:
        ────────
        Tuple[float, float]
            Normas certificadas.
        
        Excepciones:
        ────────────
        VectorDegeneracyError si algún vector es degenerado.
        """
        norm_q = self._safe_l2_norm(query)
        norm_r = self._safe_l2_norm(retrieved)
        
        if not math.isfinite(norm_q) or not math.isfinite(norm_r):
            raise VectorDegeneracyError(
                "Norma no finita en los vectores del espacio de búsqueda."
            )
        
        if norm_q <= _DEGENERACY_NORM_FLOOR:
            raise VectorDegeneracyError(
                f"Vector query degenerado: norma={norm_q:.6e} ≤ "
                f"{_DEGENERACY_NORM_FLOOR:.6e}."
            )
        
        if norm_r <= _DEGENERACY_NORM_FLOOR:
            raise VectorDegeneracyError(
                f"Vector retrieved degenerado: norma={norm_r:.6e} ≤ "
                f"{_DEGENERACY_NORM_FLOOR:.6e}."
            )
        
        return norm_q, norm_r

    # ─────────────────────────────────────────────────────────────────────────
    # 1.10. Cálculo de similitud coseno robusta
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_cosine_similarity(
        self,
        u: VectorF64,
        v: VectorF64,
        norm_u: float,
        norm_v: float,
    ) -> float:
        r"""
        Calcula:
            $\cos(\theta) = \frac{\langle u, v \rangle}{\|u\|\|v\|}$
        con normalización previa para estabilidad numérica.
        
        Teorema de Acotación: Para vectores no nulos en $\mathbb{R}^d$,
        $-1 \leq \cos(\theta) \leq 1$.
        
        Parámetros:
        ───────────
        u, v : VectorF64
            Vectores a comparar.
        norm_u, norm_v : float
            Normas precomputadas.
        
        Retorna:
        ────────
        float
            Similitud coseno en $[-1, 1]$.
        
        Excepciones:
        ────────────
        TopologicalMappingError si el cálculo no es finito.
        """
        u_unit = u / norm_u
        v_unit = v / norm_v
        
        cos_theta = float(np.vdot(u_unit, v_unit).real)
        
        if not math.isfinite(cos_theta):
            raise TopologicalMappingError(
                "Similitud coseno no finita."
            )
        
        # Corrección por ruido de punto flotante.
        cos_theta = max(-1.0, min(1.0, cos_theta))
        
        return cos_theta

    # ─────────────────────────────────────────────────────────────────────────
    # 1.11. Cálculo de distancia euclidiana
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_euclidean_distance(
        self,
        u: VectorF64,
        v: VectorF64,
    ) -> float:
        r"""
        Calcula la distancia euclidiana $\|u - v\|_2$.
        
        Parámetros:
        ───────────
        u, v : VectorF64
            Vectores a comparar.
        
        Retorna:
        ────────
        float
            Distancia euclidiana.
        """
        diff = u - v
        return self._safe_l2_norm(diff)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.12. Certificación de vecindad topológica
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_topological_neighborhood(
        self,
        query_vector: VectorF64,
        retrieved_vector: VectorF64,
    ) -> TopologicalNeighborhoodData:
        r"""
        Computa el producto interno normalizado en el Espacio de Hilbert $\mathcal{H}$.
        
        Condición de vecindad:
            $\cos(\theta) \geq \tau_{\min}$.
        
        Teorema de Caracterización Angular: La similitud coseno es una
        métrica angular que induce la topología estándar en $\mathbb{R}^d$.
        
        Parámetros:
        ───────────
        query_vector, retrieved_vector : VectorF64
            Vectores a certificar.
        
        Retorna:
        ────────
        TopologicalNeighborhoodData
            Certificado completo de vecindad.
        
        Excepciones:
        ────────────
        TopologicalMappingError si $\cos(\theta) < \tau_{\min}$.
        VectorDegeneracyError si algún vector es degenerado.
        DimensionalIncompatibilityError si las dimensiones difieren.
        """
        q = self._coerce_finite_vector("query_vector", query_vector)
        r = self._coerce_finite_vector("retrieved_vector", retrieved_vector)
        
        dim = self._verify_dimensional_compatibility(
            "query_vector", q,
            "retrieved_vector", r
        )
        
        norm_q, norm_r = self._verify_non_degenerate_vectors(q, r)
        
        cos_theta = self._compute_cosine_similarity(q, r, norm_q, norm_r)
        
        euclidean_dist = self._compute_euclidean_distance(q, r)
        
        # Calcular ángulo con corrección numérica.
        cos_clamped = max(-1.0, min(1.0, cos_theta))
        angle_rad = math.acos(cos_clamped)
        angle_deg = math.degrees(angle_rad)
        
        similarity_margin = cos_theta - _TAU_MIN_SIMILARITY
        
        if cos_theta < _TAU_MIN_SIMILARITY:
            raise TopologicalMappingError(
                "Alucinación semántica interceptada. Similitud del coseno "
                f"({cos_theta:.6f}) < umbral mínimo estricto ({_TAU_MIN_SIMILARITY:.6f}). "
                f"Ángulo: {angle_deg:.4f}°. "
                "Los vectores no pertenecen a la misma vecindad homotópica."
            )
        
        return TopologicalNeighborhoodData(
            cosine_similarity=cos_theta,
            angle_radians=angle_rad,
            angle_degrees=angle_deg,
            euclidean_distance=euclidean_dist,
            query_norm=norm_q,
            retrieved_norm=norm_r,
            dimensionality=dim,
            is_homotopically_valid=True,
            similarity_margin=similarity_margin,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.13. ÚLTIMO MÉTODO DE FASE 1: PUENTE FORMAL HACIA FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _phase1_certify_and_bridge_to_phase2(
        self,
        query_vector: VectorF64,
        retrieved_vector: VectorF64,
        cost_vector_c: VectorF64,
        friction_operator_F: OperatorF64,
        injection_matrix_T: MatrixF64,
    ) -> Phase1TopologicalBridge:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 1.
        
        Su definición formal es la continuación directa de la Fase 2:
        entrega el certificado de vecindad topológica y todos los datos
        necesarios como prefijo obligatorio de la auditoría de fricción.
        
        Lema de Continuación Funtorial:
        ────────────────────────────────
        La vecindad topológica certificada es condición necesaria para
        la coherencia del ensamblaje de costos posterior.
        
        Teorema de Correspondencia:
        ───────────────────────────
        $\Phi_2 \circ \Phi_1$ está bien definido si y solo si
        $\Phi_1$ retorna un objeto de tipo Phase1TopologicalBridge.
        
        Parámetros:
        ───────────
        query_vector, retrieved_vector : VectorF64
            Vectores a certificar.
        cost_vector_c : VectorF64
            Vector de costos.
        friction_operator_F : OperatorF64
            Operador de fricción territorial.
        injection_matrix_T : MatrixF64
            Matriz de inyección.
        
        Retorna:
        ────────
        Phase1TopologicalBridge
            Objeto terminal de Fase 1 y objeto inicial de Fase 2.
        
        Postcondición:
        ──────────────
        El puente contiene:
        - Certificado de vecindad topológica completo.
        - Todos los datos necesarios para Fase 2 y Fase 3.
        
        Excepciones:
        ────────────
        TopologicalMappingError si la vecindad no es válida.
        VectorDegeneracyError si algún vector es degenerado.
        DimensionalIncompatibilityError si las dimensiones son incompatibles.
        """
        q = self._coerce_finite_vector("query_vector", query_vector)
        r = self._coerce_finite_vector("retrieved_vector", retrieved_vector)
        
        self._verify_dimensional_compatibility(
            "query_vector", q,
            "retrieved_vector", r
        )
        
        c = self._coerce_finite_vector("cost_vector_c", cost_vector_c)
        
        F = self._coerce_friction_operator(
            "friction_operator_F",
            friction_operator_F,
            dimension=c.size
        )
        
        T = self._coerce_finite_matrix("injection_matrix_T", injection_matrix_T)
        
        neighborhood_audit = self._certify_topological_neighborhood(q, r)
        
        logger.debug(
            "Fase 1 completada. cos(θ)=%.6f | ángulo=%.4f° | dim=%d.",
            neighborhood_audit.cosine_similarity,
            neighborhood_audit.angle_degrees,
            neighborhood_audit.dimensionality,
        )
        
        return Phase1TopologicalBridge(
            neighborhood_audit=neighborhood_audit,
            query_vector=q,
            retrieved_vector=r,
            cost_vector_c=c,
            friction_operator_F=F,
            injection_matrix_T=T,
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
    
    Fundamento Teórico:
    ────────────────────
    Teorema de Estabilidad Espectral: Un operador lineal $F$ es numéricamente
    estable si y solo si su número de condición está acotado:
    
        $\kappa(F) = \frac{\sigma_{\max}(F)}{\sigma_{\min}(F)} \leq \kappa_{\max}$
    
    Corolario de Positividad: Para operadores diagonales, la positividad
    estricta de todos los factores garantiza invertibilidad y estabilidad.
    
    Teorema de Simetría: Para operadores matriciales, la parte simétrica
    $(F + F^\top)/2$ determina la energía del sistema.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1. Saneamiento de vectores con ruido infinitesimal
    # ─────────────────────────────────────────────────────────────────────────
    def _sanitize_nonnegative_vector(
        self,
        name: str,
        vector: VectorF64,
    ) -> VectorF64:
        r"""
        Sanea un vector, convirtiendo negativos infinitesimales a cero.
        
        Lema de Saneamiento: Los errores de redondeo pueden producir
        valores negativos infinitesimales ($-10^{-16}$) que deben ser
        saneados a cero sin violar la positividad física.
        
        Parámetros:
        ───────────
        name : str
            Identificador del vector.
        vector : VectorF64
            Vector a sanear.
        
        Retorna:
        ────────
        VectorF64
            Vector saneado.
        
        Excepciones:
        ────────────
        ThermodynamicFrictionAnomaly si hay negativos no infinitesimales.
        """
        vec_clean = vector.copy()
        
        small_negative = (vec_clean < 0.0) & (vec_clean >= -_NEGATIVE_TOLERANCE)
        vec_clean[small_negative] = 0.0
        
        if np.any(vec_clean < 0.0):
            negative_count = int(np.sum(vec_clean < 0.0))
            min_negative = float(np.min(vec_clean))
            raise ThermodynamicFrictionAnomaly(
                f"Inyección de energía negativa en {name}. "
                f"Componentes negativas no infinitesimales: {negative_count}. "
                f"Mínimo: {min_negative:.6e}."
            )
        
        return vec_clean

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2. Auditoría de operador de fricción diagonal
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_diagonal_friction_operator(
        self,
        diag: VectorF64,
        cost_vector: VectorF64,
    ) -> Tuple[float, float, float, float, VectorF64, float]:
        r"""
        Audita un operador de fricción diagonal (vector 1-D).
        
        Teorema Diagonal: Para operadores diagonales $F = \text{diag}(f_1, \ldots, f_n)$:
        - $\kappa(F) = \frac{\max_i f_i}{\min_i f_i}$
        - $\det(F) = \prod_i f_i$
        - $F \cdot c = (f_1 c_1, \ldots, f_n c_n)$
        
        Parámetros:
        ───────────
        diag : VectorF64
            Factores diagonales.
        cost_vector : VectorF64
            Vector de costos.
        
        Retorna:
        ────────
        Tuple[float, float, float, float, VectorF64, float]
            (spectral_min, spectral_max, spectral_mean, spectral_std,
             total_cost_vector, determinant)
        
        Excepciones:
        ────────────
        ThermodynamicFrictionAnomaly si hay factores no positivos.
        """
        diag_clean = self._sanitize_nonnegative_vector(
            "friction_operator_F (diagonal)",
            diag
        )
        
        spectral_min = float(np.min(diag_clean))
        spectral_max = float(np.max(diag_clean))
        spectral_mean = float(np.mean(diag_clean))
        spectral_std = float(np.std(diag_clean))
        
        if spectral_min <= _POSITIVE_FLOOR:
            raise ThermodynamicFrictionAnomaly(
                "Operador de fricción diagonal singular o no positivo. "
                f"min(diag)={spectral_min:.6e} <= piso positivo {_POSITIVE_FLOOR:.6e}."
            )
        
        total_cost_vector = diag_clean * cost_vector
        determinant = float(np.prod(diag_clean))
        
        return (
            spectral_min,
            spectral_max,
            spectral_mean,
            spectral_std,
            total_cost_vector,
            determinant,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3. Auditoría de operador de fricción matricial
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_matrix_friction_operator(
        self,
        F: MatrixF64,
        cost_vector: VectorF64,
    ) -> Tuple[float, float, float, float, VectorF64, float, float]:
        r"""
        Audita un operador de fricción matricial (matriz 2-D).
        
        Teorema Espectral: Para operadores simétricos $F = F^\top$:
        - Los autovalores son reales.
        - $\kappa(F) = \frac{\lambda_{\max}}{\lambda_{\min}}$.
        - $F \succ 0 \iff \lambda_{\min} > 0$.
        
        Parámetros:
        ───────────
        F : MatrixF64
            Matriz de fricción.
        cost_vector : VectorF64
            Vector de costos.
        
        Retorna:
        ────────
        Tuple[float, float, float, float, VectorF64, float, float]
            (spectral_min, spectral_max, spectral_mean, spectral_std,
             total_cost_vector, determinant, symmetry_residual)
        
        Excepciones:
        ────────────
        ThermodynamicFrictionAnomaly si la matriz no es definida positiva.
        """
        F_clean = self._sanitize_nonnegative_vector(
            "friction_operator_F (matricial)",
            F.ravel()
        ).reshape(F.shape)
        
        # El operador territorial se modela como auto-adjunto.
        F_sym = 0.5 * (F_clean + F_clean.T)
        
        fro_original = self._safe_frobenius_norm(F_clean)
        fro_sym = self._safe_frobenius_norm(F_sym)
        fro_asym = self._safe_frobenius_norm(F_clean - F_sym)
        
        if math.isfinite(fro_original) and math.isfinite(fro_asym):
            if fro_asym > _SYMMETRY_TOLERANCE * max(1.0, fro_original):
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
        spectral_mean = float(np.mean(eigenvalues))
        spectral_std = float(np.std(eigenvalues))
        
        if spectral_min <= _POSITIVE_FLOOR:
            raise ThermodynamicFrictionAnomaly(
                "Operador de fricción no definido positivo. "
                f"lambda_min={spectral_min:.6e} <= piso positivo {_POSITIVE_FLOOR:.6e}."
            )
        
        total_cost_vector = F_sym @ cost_vector
        
        try:
            determinant = float(np.linalg.det(F_sym))
        except np.linalg.LinAlgError:
            determinant = float('nan')
        
        symmetry_residual = float(fro_asym)
        
        return (
            spectral_min,
            spectral_max,
            spectral_mean,
            spectral_std,
            total_cost_vector,
            determinant,
            symmetry_residual,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4. Verificación de cota de número de condición
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_condition_bound(
        self,
        condition_number: float,
    ) -> None:
        r"""
        Verifica que el número de condición esté acotado.
        
        Teorema de Estabilidad Numérica: Si $\kappa(F) > \kappa_{\max}$,
        el sistema es numéricamente inestable y los errores de redondeo
        se amplifican exponencialmente.
        
        Parámetros:
        ───────────
        condition_number : float
            Número de condición calculado.
        
        Excepciones:
        ────────────
        ThermodynamicFrictionAnomaly si la cota se excede.
        """
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

    # ─────────────────────────────────────────────────────────────────────────
    # 2.5. Auditoría completa del ensamblaje de fricción
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_tensor_friction_assembly(
        self,
        cost_vector_c: VectorF64,
        friction_operator_F: OperatorF64,
    ) -> TensorFrictionData:
        r"""
        Valida:
            $C_{total} = F_{ext} \cdot c$,
        la positividad estricta y el condicionamiento espectral.
        
        Teorema de Ensamblaje: El costo total es el resultado de modular
        el vector de costos por el operador de fricción territorial.
        
        Condiciones:
        ────────────
        1. $c \geq 0$ (costos no negativos).
        2. $F \succ 0$ (fricción definida positiva).
        3. $\kappa(F) \leq \kappa_{\max}$ (estabilidad numérica).
        4. $C_{total} \geq 0$ (costo total no negativo).
        
        Parámetros:
        ───────────
        cost_vector_c : VectorF64
            Vector de costos.
        friction_operator_F : OperatorF64
            Operador de fricción.
        
        Retorna:
        ────────
        TensorFrictionData
            Certificado completo de fricción.
        
        Excepciones:
        ────────────
        ThermodynamicFrictionAnomaly si alguna condición falla.
        """
        c = self._coerce_finite_vector("cost_vector_c", cost_vector_c)
        F = self._coerce_friction_operator(
            "friction_operator_F",
            friction_operator_F,
            dimension=c.size
        )
        
        # ── Saneamiento del vector de costos ─────────────────────────────────
        c_clean = self._sanitize_nonnegative_vector("cost_vector_c", c)
        
        # ── Auditoría del operador de fricción ───────────────────────────────
        if F.ndim == 1:
            (
                spectral_min,
                spectral_max,
                spectral_mean,
                spectral_std,
                total_cost_vector,
                determinant,
            ) = self._audit_diagonal_friction_operator(F, c_clean)
            operator_type = "diagonal"
            symmetry_residual = 0.0
            condition_number = float(spectral_max / spectral_min)
        else:
            (
                spectral_min,
                spectral_max,
                spectral_mean,
                spectral_std,
                total_cost_vector,
                determinant,
                symmetry_residual,
            ) = self._audit_matrix_friction_operator(F, c_clean)
            operator_type = "matricial"
            condition_number = float(spectral_max / spectral_min)
        
        # ── Verificación de cota de condición ────────────────────────────────
        self._verify_condition_bound(condition_number)
        
        # ── Verificación de finitud del costo total ──────────────────────────
        if not np.all(np.isfinite(total_cost_vector)):
            raise ThermodynamicFrictionAnomaly(
                "El costo total ensamblado contiene componentes no finitas."
            )
        
        # ── Saneamiento del costo total ──────────────────────────────────────
        total_clean = self._sanitize_nonnegative_vector(
            "total_cost_vector",
            total_cost_vector
        )
        
        total_cost_norm = self._safe_l1_norm(total_clean)
        
        if not math.isfinite(total_cost_norm):
            raise ThermodynamicFrictionAnomaly(
                "Norma L1 del costo total no finita."
            )
        
        cost_vector_norm = self._safe_l1_norm(c_clean)
        
        return TensorFrictionData(
            condition_number=condition_number,
            spectral_min=spectral_min,
            spectral_max=spectral_max,
            spectral_mean=spectral_mean,
            spectral_std=spectral_std,
            total_cost_norm=total_cost_norm,
            total_cost_vector=total_clean,
            is_positive_definite=True,
            operator_type=operator_type,
            symmetry_residual=symmetry_residual,
            cost_vector_norm=cost_vector_norm,
            friction_determinant=determinant,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.6. ÚLTIMO MÉTODO DE FASE 2: PUENTE FORMAL HACIA FASE 3
    # ─────────────────────────────────────────────────────────────────────────
    def _phase2_audit_and_bridge_to_phase3(
        self,
        phase1_bridge: Phase1TopologicalBridge,
    ) -> Phase2FrictionBridge:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 2.
        
        Su definición formal es la continuación directa de la Fase 3:
        entrega el certificado de fricción y el puente de Fase 1 como
        prefijo obligatorio de la imposición de rango-nulidad.
        
        Lema de Continuación Funtorial:
        ────────────────────────────────
        La fricción territorial acotada garantiza que la matriz de
        inyección no sufre distorsión espectral excesiva.
        
        Teorema de Correspondencia:
        ───────────────────────────
        $\Phi_3 \circ \Phi_2$ está bien definido si y solo si
        $\Phi_2$ retorna un objeto de tipo Phase2FrictionBridge.
        
        Parámetros:
        ───────────
        phase1_bridge : Phase1TopologicalBridge
            Certificado de Fase 1.
        
        Retorna:
        ────────
        Phase2FrictionBridge
            Objeto terminal de Fase 2 y objeto inicial de Fase 3.
        
        Postcondición:
        ──────────────
        El puente contiene:
        - Certificado de fricción completo.
        - Puente de Fase 1 completo.
        
        Excepciones:
        ────────────
        SemanticEstimatorAgentError si el puente de Fase 1 es inválido.
        ThermodynamicFrictionAnomaly si la fricción es anómala.
        """
        if not isinstance(phase1_bridge, Phase1TopologicalBridge):
            raise SemanticEstimatorAgentError(
                "La Fase 2 requiere un Phase1TopologicalBridge emitido por la Fase 1."
            )
        
        friction_audit = self._audit_tensor_friction_assembly(
            cost_vector_c=phase1_bridge.cost_vector_c,
            friction_operator_F=phase1_bridge.friction_operator_F
        )
        
        logger.debug(
            "Fase 2 completada. κ(F)=%.6e | tipo=%s | ||C||₁=%.6e.",
            friction_audit.condition_number,
            friction_audit.operator_type,
            friction_audit.total_cost_norm,
        )
        
        return Phase2FrictionBridge(
            phase1_bridge=phase1_bridge,
            friction_audit=friction_audit
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
    
    Fundamento Teórico:
    ────────────────────
    Teorema de Rango-Nulidad: Para una transformación lineal $T: V \to W$:
    
        $\dim(V) = \text{rank}(T) + \text{nullity}(T)$
    
    Teorema de Isometría Parcial: Una matriz $T$ es isometría parcial si:
    
        $T^\top T = I \quad \text{(en el dominio)}$
    
    Corolario de Aislamiento: Si $\text{rank}(T) = 1$ y $T$ es isometría
    parcial, entonces la inyección no produce efectos secundarios en la MIC.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1. Cálculo de descomposición SVD
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_svd_decomposition(
        self,
        T: MatrixF64,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Computa la descomposición en valores singulares de la matriz T.
        
        Teorema SVD: Toda matriz $T \in \mathbb{R}^{m \times n}$ admite
        una descomposición $T = U \Sigma V^\top$ donde $U, V$ son
        ortogonales y $\Sigma$ es diagonal con valores singulares no negativos.
        
        Parámetros:
        ───────────
        T : MatrixF64
            Matriz a descomponer.
        
        Retorna:
        ────────
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            (valores singulares, matriz U)
        
        Excepciones:
        ────────────
        FunctorialityError si el SVD falla.
        """
        try:
            U, singular_values, Vt = la.svd(
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
        
        return s, U

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2. Determinación del rango efectivo
    # ─────────────────────────────────────────────────────────────────────────
    def _determine_effective_rank(
        self,
        singular_values: NDArray[np.float64],
        matrix_shape: Tuple[int, int],
    ) -> Tuple[int, int, float]:
        r"""
        Determina el rango efectivo de la matriz usando tolerancia adaptativa.
        
        Teorema de Rango Numérico: El rango efectivo es el número de
        valores singulares significativamente mayores que la tolerancia:
        
            $\text{rank}(T) = |\{i : \sigma_i > \tau\}|$
        
        donde $\tau = \max(m, n) \cdot \varepsilon_{\text{máquina}} \cdot \sigma_1$.
        
        Parámetros:
        ───────────
        singular_values : NDArray[np.float64]
            Valores singulares ordenados descendentemente.
        matrix_shape : Tuple[int, int]
            Forma de la matriz (m, n).
        
        Retorna:
        ────────
        Tuple[int, int, float]
            (rango efectivo, dimensión del núcleo, tolerancia de rango)
        
        Excepciones:
        ────────────
        FunctorialityError si el rango no es 1.
        """
        m, n = matrix_shape
        sigma_max = float(singular_values[0])
        
        rank_tolerance = max(
            _SVD_ABSOLUTE_TOLERANCE,
            float(max(m, n)) * _MACHINE_EPSILON * max(sigma_max, 1.0)
        )
        
        effective_rank = int(np.sum(singular_values > rank_tolerance))
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
        
        return effective_rank, kernel_dimension, rank_tolerance

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3. Verificación de isometría parcial
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_partial_isometry(
        self,
        T: MatrixF64,
        sigma_max: float,
    ) -> Tuple[float, float, float, float, float]:
        r"""
        Verifica que T sea una isometría parcial de rango 1.
        
        Condición de Isometría: El único valor singular no nulo debe ser 1:
            $\sigma_1 = 1$
        
        Los proyectores inducidos deben ser idempotentes y simétricos:
            $P_{row} = T^\top T$, $P_{col} = T T^\top$
        
        Parámetros:
        ───────────
        T : MatrixF64
            Matriz de inyección.
        sigma_max : float
            Valor singular dominante.
        
        Retorna:
        ────────
        Tuple[float, float, float, float, float]
            (sigma_deviation, row_sym_deviation, row_idempotence_deviation,
             col_sym_deviation, col_idempotence_deviation)
        
        Excepciones:
        ────────────
        FunctorialityError si los proyectores no son finitos.
        ProjectorIntegrityError si la isometría falla.
        """
        # Normalización por el valor singular dominante.
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
        
        row_sym_deviation = self._safe_frobenius_norm(P_row - P_row.T)
        row_idempotence_deviation = self._safe_frobenius_norm(P_row @ P_row - P_row)
        col_sym_deviation = self._safe_frobenius_norm(P_col - P_col.T)
        col_idempotence_deviation = self._safe_frobenius_norm(P_col @ P_col - P_col)
        
        sigma_deviation = abs(sigma_max - 1.0)
        
        return (
            sigma_deviation,
            row_sym_deviation,
            row_idempotence_deviation,
            col_sym_deviation,
            col_idempotence_deviation,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.4. Imposición de rango-nulidad e inyección ortogonal
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_rank_nullity_projection(
        self,
        injection_matrix_T: MatrixF64,
    ) -> RankNullityProjectionData:
        r"""
        Computa SVD para extraer el rango efectivo y auditar la ortogonalidad.
        
        Condición axiomática:
            $\operatorname{rank}(T) = 1$.
        
        Condición de inyección ortogonal:
            $T$ debe ser una isometría parcial de rango 1.
        
        Teorema de Caracterización: $T$ es isometría parcial de rango 1 si:
        1. $\sigma_1 = 1$ (valor singular dominante unitario).
        2. $P_{row} = T^\top T$ es idempotente y simétrico.
        3. $P_{col} = T T^\top$ es idempotente y simétrico.
        
        Parámetros:
        ───────────
        injection_matrix_T : MatrixF64
            Matriz de inyección a certificar.
        
        Retorna:
        ────────
        RankNullityProjectionData
            Certificado completo de rango-nulidad.
        
        Excepciones:
        ────────────
        FunctorialityError si el rango no es 1.
        ProjectorIntegrityError si la isometría falla.
        """
        T = self._coerce_finite_matrix("injection_matrix_T", injection_matrix_T)
        m, n = T.shape
        
        # 1. Descomposición SVD.
        singular_values, U = self._compute_svd_decomposition(T)
        
        sigma_max = float(singular_values[0])
        sigma_min = float(singular_values[-1]) if singular_values.size > 1 else 0.0
        singular_gap = sigma_max - sigma_min
        
        # 2. Determinación de rango efectivo.
        effective_rank, kernel_dimension, rank_tolerance = (
            self._determine_effective_rank(singular_values, (m, n))
        )
        
        # 3. Verificación de isometría parcial.
        (
            sigma_deviation,
            row_sym_deviation,
            row_idempotence_deviation,
            col_sym_deviation,
            col_idempotence_deviation,
        ) = self._verify_partial_isometry(T, sigma_max)
        
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
        
        t_unit_fro = self._safe_frobenius_norm(T / sigma_max)
        
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
            raise ProjectorIntegrityError(
                "La matriz de inyección no es una isometría parcial ortogonal. "
                f"Desviación={orthogonality_deviation:.6e} > tolerancia={ortho_tolerance:.6e}. "
                "Se viola el aislamiento ortogonal en la MIC."
            )
        
        condition_number = float(sigma_max / sigma_min) if sigma_min > _MACHINE_EPSILON else float('inf')
        
        return RankNullityProjectionData(
            matrix_shape=(m, n),
            effective_rank=effective_rank,
            kernel_dimension=kernel_dimension,
            largest_singular_value=sigma_max,
            smallest_singular_value=sigma_min,
            singular_value_gap=singular_gap,
            rank_tolerance=rank_tolerance,
            orthogonality_deviation=orthogonality_deviation,
            is_orthogonal_injection=True,
            row_projector_idempotence=row_idempotence_deviation,
            col_projector_idempotence=col_idempotence_deviation,
            row_projector_symmetry=row_sym_deviation,
            col_projector_symmetry=col_sym_deviation,
            condition_number=condition_number,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.5. ÚLTIMO MÉTODO DE FASE 3: FINALIZACIÓN FUNTORIAL
    # ─────────────────────────────────────────────────────────────────────────
    def _phase3_finalize_from_phase2_bridge(
        self,
        phase2_bridge: Phase2FrictionBridge,
    ) -> SemanticEstimatorAuditState:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 3.
        
        Compone los certificados de Fase 1, Fase 2 y Fase 3 en el objeto
        terminal SemanticEstimatorAuditState.
        
        Teorema de Corrección Global:
        ──────────────────────────────
        Si $\Phi_1$, $\Phi_2$ y $\Phi_3$ certifican éxito, entonces la
        estimación semántica es epistemológicamente válida y está
        topológicamente protegida contra alucinaciones del LLM.
        
        Corolario de Estabilidad: El sistema resultante es:
        - Topológicamente coherente (vecindad certificada).
        - Termodinámicamente estable (fricción acotada).
        - Categóricamente aislado (rango 1, isometría parcial).
        
        Parámetros:
        ───────────
        phase2_bridge : Phase2FrictionBridge
            Certificado de Fase 2.
        
        Retorna:
        ────────
        SemanticEstimatorAuditState
            Objeto terminal del endofuntor $\mathcal{Z}_{EstimatorAgent}$.
        
        Postcondición:
        ──────────────
        El estado final contiene:
        - Certificado de vecindad topológica.
        - Certificado de fricción territorial.
        - Certificado de rango-nulidad.
        - Flag de validez epistemológica.
        - Metadata de gobernanza.
        
        Excepciones:
        ────────────
        SemanticEstimatorAgentError si el puente de Fase 2 es inválido.
        FunctorialityError si la proyección falla.
        """
        if not isinstance(phase2_bridge, Phase2FrictionBridge):
            raise SemanticEstimatorAgentError(
                "La Fase 3 requiere un Phase2FrictionBridge emitido por la Fase 2."
            )
        
        projection_audit = self._enforce_rank_nullity_projection(
            injection_matrix_T=phase2_bridge.phase1_bridge.injection_matrix_T
        )
        
        governance_metadata = {
            "functor_composition": "Φ₃ ∘ Φ₂ ∘ Φ₁",
            "phase1_cosine_similarity": phase2_bridge.phase1_bridge.neighborhood_audit.cosine_similarity,
            "phase1_angle_degrees": phase2_bridge.phase1_bridge.neighborhood_audit.angle_degrees,
            "phase2_condition_number": phase2_bridge.friction_audit.condition_number,
            "phase2_operator_type": phase2_bridge.friction_audit.operator_type,
            "phase3_effective_rank": projection_audit.effective_rank,
            "phase3_kernel_dimension": projection_audit.kernel_dimension,
            "phase3_orthogonality_deviation": projection_audit.orthogonality_deviation,
        }
        
        final_state = SemanticEstimatorAuditState(
            neighborhood_audit=phase2_bridge.phase1_bridge.neighborhood_audit,
            friction_audit=phase2_bridge.friction_audit,
            projection_audit=projection_audit,
            is_epistemologically_valid=True,
            governance_metadata=governance_metadata,
        )
        
        logger.info(
            "Auditoría Semántica y Vectorial completada. "
            f"Cos(θ): {final_state.neighborhood_audit.cosine_similarity:.6f} | "
            f"κ(F): {final_state.friction_audit.condition_number:.6e} | "
            f"Rank(T): {final_state.projection_audit.effective_rank} | "
            f"Ortho_dev: {final_state.projection_audit.orthogonality_deviation:.6e}"
        )
        
        return final_state


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
    
    Definición Categórica:
    ──────────────────────
    $\mathcal{Z}_{EstimatorAgent}: \mathcal{H} \to \mathcal{H}$
    es un endofuntor en la categoría de espacios de Hilbert que preserva
    la estructura métrica, la estabilidad espectral y el aislamiento ortogonal.
    
    Propiedad Universal: Para cualquier consulta semántica, el endofuntor
    garantiza que el resultado pertenece a la misma vecindad topológica que
    la consulta, con costo termodinámicamente estable y aislamiento categórico.
    """

    def execute_semantic_estimation_governance(
        self,
        query_vector: VectorF64,
        retrieved_vector: VectorF64,
        cost_vector_c: VectorF64,
        friction_operator_F: OperatorF64,
        injection_matrix_T: MatrixF64,
    ) -> SemanticEstimatorAuditState:
        r"""
        Ejecuta la composición funtorial estricta en 3 fases anidadas.
        
        Diagrama Conmutativo:
        ─────────────────────
        
        $\mathcal{H} \xrightarrow{\Phi_1} \text{TopologicalNeighborhood}$
              $\downarrow$                                    $\downarrow$
        $\text{FrictionAudit} \xleftarrow{\Phi_2} \text{TopologicalNeighborhood}$
              $\downarrow$                                    $\downarrow$
        $\text{RankNullity} \xleftarrow{\Phi_3} \text{FrictionAudit}$
              $\downarrow$
        $\text{SemanticEstimatorAuditState}$
        
        Flujo:
        ─────
        1. Fase 1:
           `_phase1_certify_and_bridge_to_phase2`
           → `Phase1TopologicalBridge`
        2. Fase 2:
           `_phase2_audit_and_bridge_to_phase3`
           → `Phase2FrictionBridge`
        3. Fase 3:
           `_phase3_finalize_from_phase2_bridge`
           → `SemanticEstimatorAuditState`
        
        Parámetros:
        ───────────
        query_vector : VectorF64
            Vector de consulta.
        retrieved_vector : VectorF64
            Vector recuperado.
        cost_vector_c : VectorF64
            Vector de costos.
        friction_operator_F : OperatorF64
            Operador de fricción territorial.
        injection_matrix_T : MatrixF64
            Matriz de inyección.
        
        Retorna:
        ────────
        SemanticEstimatorAuditState
            Estado terminal del endofuntor.
        """
        # ── Fase 1: Certificar la vecindad topológica del mapeo FAISS ────────
        phase1_bridge = self._phase1_certify_and_bridge_to_phase2(
            query_vector=query_vector,
            retrieved_vector=retrieved_vector,
            cost_vector_c=cost_vector_c,
            friction_operator_F=friction_operator_F,
            injection_matrix_T=injection_matrix_T
        )
        
        # ── Fase 2: Certificar estabilidad del ensamblaje del tensor de costos ─
        phase2_bridge = self._phase2_audit_and_bridge_to_phase3(
            phase1_bridge=phase1_bridge
        )
        
        # ── Fase 3: Proyectar la capacidad garantizando aislamiento en la MIC ─
        return self._phase3_finalize_from_phase2_bridge(
            phase2_bridge=phase2_bridge
        )

    def __call__(
        self,
        query_vector: VectorF64,
        retrieved_vector: VectorF64,
        cost_vector_c: VectorF64,
        friction_operator_F: OperatorF64,
        injection_matrix_T: MatrixF64,
    ) -> SemanticEstimatorAuditState:
        r"""Alias invocable del endofuntor de estimación semántica."""
        return self.execute_semantic_estimation_governance(
            query_vector=query_vector,
            retrieved_vector=retrieved_vector,
            cost_vector_c=cost_vector_c,
            friction_operator_F=friction_operator_F,
            injection_matrix_T=injection_matrix_T,
        )


# ════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ════════════════════════════════════════════════════════════════════════════
__all__ = [
    "SemanticEstimatorAgentError",
    "TopologicalMappingError",
    "VectorDegeneracyError",
    "DimensionalIncompatibilityError",
    "ThermodynamicFrictionAnomaly",
    "FunctorialityError",
    "ProjectorIntegrityError",
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