# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Semantic Dictionary Agent (Custodio de la Ontología y Difeomorfismo)║
║ Ruta   : app/agents/wisdom/semantic_dictionary_agent.py                             ║
║ Versión: 3.0.0-Categorical-Diffeomorphism-Thermodynamic-Strict-Nested        ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DIFERENCIAL (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna el `semantic_dictionary.py` en el Estrato Ω (WISDOM).

Su mandato axiomático es garantizar que el Funtor de Proyección Semántica
$F: Top \to Narr$ preserve la invarianza homotópica, aniquilando las
alucinaciones estocásticas de la Inteligencia Artificial mediante:

1. Auditoría de inmersión difeomórfica del pushforward $F_*$.
2. Certificación de retracto de deformación por cota de Lipschitz.
3. Gobernanza termodinámica del caché semántico como operador densidad.

ARQUITECTURA DE 3 FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Inmersión Difeomórfica:
         Φ₁(F_*) = (rank(F_*), dim ker(F_*), κ(F_*))
         Se exige dim ker(F_*) = 0.

Fase 2 → Retracto de Deformación Lipschitz:
         Φ₂(Φ₁(...)) = L_medido
         Se exige |F(x) - F(y)|_Narr ≤ L_max |x - y|_Top.

Fase 3 → Gobernanza Termodinámica:
         Φ₃(Φ₂(Φ₁(...))) = S(ρ_C), purga térmica y estabilidad.
         Se modela el caché como:
             ρ_C = Σ_i p_i |v_i⟩⟨v_i|
         y se computa:
             S(ρ_C) = -k_B Tr(ρ_C ln ρ_C).

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
from typing import List, Tuple

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


try:
    from app.wisdom.semantic_dictionary import BOLTZMANN_CONSTANT
except ImportError:
    BOLTZMANN_CONSTANT = 1.0


logger = logging.getLogger("MAC.Wisdom.SemanticDictionaryAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. TIPOS Y CONSTANTES FÍSICO-MATEMÁTICAS
# ═══════════════════════════════════════════════════════════════════════════════
VectorF64 = NDArray[np.float64]
MatrixF64 = NDArray[np.float64]
DensityMatrix = NDArray[np.complex128]

_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)

# Fase 1: SVD e inmersión difeomórfica.
_SVD_ABSOLUTE_TOLERANCE: float = 1e-10
_JACOBIAN_CONDITION_MAX: float = 1e8

# Fase 2: Retracto de deformación Lipschitz.
_LIPSCHITZ_MAX_RATIO: float = 5.0
_LIPSCHITZ_ZERO_TOLERANCE: float = 1e-12

# Fase 3: Gobernanza termodinámica.
_ENTROPY_CRITICAL_THRESHOLD: float = 2.0
_ORTHOGONALITY_TOLERANCE: float = 0.15
_ENTROPY_EIGENVALUE_FLOOR: float = _MACHINE_EPSILON
_DENSITY_TRACE_FLOOR: float = _MACHINE_EPSILON


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ONTOLÓGICAS
# ═══════════════════════════════════════════════════════════════════════════════
class SemanticDictionaryAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Ontología."""
    pass


class SemanticDegeneracyError(SemanticDictionaryAgentError):
    r"""Detonada si $\dim(\ker(F_*)) > 0$ (colapso de dimensiones semánticas)."""
    pass


class LipschitzRetractionViolation(SemanticDictionaryAgentError):
    r"""Detonada si $|F(x) - F(y)| > L_{\max} |x - y|$ (alucinación detectada)."""
    pass


class ThermodynamicEvictionAnomaly(SemanticDictionaryAgentError):
    r"""Detonada si la entropía $S(\rho_C)$ no puede ser mitigada por la purga."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Topos)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True, eq=False)
class DiffeomorphicImmersionData:
    r"""
    Artefacto de Fase 1.
    Certificación de inmersión del pushforward $F_*$.
    """
    jacobian_shape: Tuple[int, int]
    numerical_rank: int
    kernel_dimension: int
    smallest_singular_value: float
    condition_number: float
    rank_tolerance: float
    is_isomorphic: bool


@dataclass(frozen=True, slots=True, eq=False)
class LipschitzRetractionData:
    r"""
    Artefacto de Fase 2.
    Certificación del retracto topológico del LLM.
    """
    topological_distance: float
    narrative_distance: float
    measured_lipschitz_constant: float
    is_deformation_bounded: bool


@dataclass(frozen=True, slots=True, eq=False)
class ThermodynamicEvictionState:
    r"""
    Artefacto de Fase 3.
    Estado térmico post-purga del caché semántico.
    """
    von_neumann_entropy_pre: float
    von_neumann_entropy_post: float
    evicted_vectors_count: int
    cache_cardinality_pre: int
    cache_cardinality_post: int
    is_thermally_stable: bool


@dataclass(frozen=True, slots=True, eq=False)
class Phase1ImmersionBridge:
    r"""
    Puente funtorial Φ₁ → Φ₂.

    Este objeto es emitido por el último método de la Fase 1 y constituye
    la entrada formal del primer método de la Fase 2.
    """
    immersion_audit: DiffeomorphicImmersionData
    jacobian_F_star: MatrixF64
    distance_topological: float
    distance_narrative: float
    cache_states: Tuple[VectorF64, ...]
    decision_trajectory: VectorF64


@dataclass(frozen=True, slots=True, eq=False)
class Phase2LipschitzBridge:
    r"""
    Puente funtorial Φ₂ → Φ₃.

    Este objeto es emitido por el último método de la Fase 2 y constituye
    la entrada formal del primer método de la Fase 3.
    """
    phase1_bridge: Phase1ImmersionBridge
    lipschitz_audit: LipschitzRetractionData


@dataclass(frozen=True, slots=True, eq=False)
class SemanticDictionaryAuditState:
    r"""
    Objeto final del endofuntor $\mathcal{Z}_{DictAgent}$.
    """
    immersion_audit: DiffeomorphicImmersionData
    lipschitz_audit: LipschitzRetractionData
    thermodynamic_state: ThermodynamicEvictionState
    is_epistemologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: AUDITORÍA DE INMERSIÓN DIFEOMÓRFICA                               ║
# ║                                                                             ║
# ║   Φ₁(F_*) = (rank(F_*), dim ker(F_*), κ(F_*))                               ║
# ║                                                                             ║
# ║   1. Valida finiteza y dimensionalidad del Jacobiano.                       ║
# ║   2. Computa SVD robusto del pushforward $F_*$.                             ║
# ║   3. Determina rango numérico y núcleo.                                     ║
# ║   4. Exige dim ker(F_*) = 0.                                                ║
# ║   5. Emite el puente formal hacia la Fase 2.                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_DiffeomorphicImmersionAuditor:
    r"""
    Fase 1 del endofuntor.

    Garantiza que el Funtor de Proyección $F: Top \to Narr$ no colapse
    distintos riesgos topológicos en la misma frase ambigua.
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
            raise SemanticDictionaryAgentError(
                f"{name} no puede convertirse a un escalar float64."
            ) from exc

        if arr.ndim != 0:
            raise SemanticDictionaryAgentError(
                f"{name} debe ser un escalar, no un arreglo de dimensión {arr.ndim}."
            )

        scalar = float(arr)
        if not math.isfinite(scalar):
            raise SemanticDictionaryAgentError(
                f"{name} debe ser finito. Se recibió {scalar!r}."
            )

        return scalar

    def _as_nonnegative_float(self, name: str, value: float) -> float:
        """
        Valida un escalar finito no negativo.
        Tolera negatividad numérica infinitesimal.
        """
        scalar = self._as_finite_float(name, value)

        if scalar < 0.0:
            if scalar >= -100.0 * _MACHINE_EPSILON:
                return 0.0
            raise SemanticDictionaryAgentError(
                f"{name} debe ser no negativo. Se recibió {scalar!r}."
            )

        return scalar

    def _as_finite_vector(self, name: str, value: VectorF64) -> VectorF64:
        """
        Valida que el objeto sea un vector 1-D no vacío y con componentes finitas.
        """
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SemanticDictionaryAgentError(
                f"{name} no puede convertirse a un vector float64."
            ) from exc

        if arr.ndim != 1:
            raise SemanticDictionaryAgentError(
                f"{name} debe ser un vector 1-D. Dimensión recibida: {arr.ndim}."
            )

        if arr.size == 0:
            raise SemanticDictionaryAgentError(
                f"{name} no puede ser el vector vacío."
            )

        if not np.all(np.isfinite(arr)):
            raise SemanticDictionaryAgentError(
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
            raise SemanticDictionaryAgentError(
                f"{name} no puede convertirse a una matriz float64."
            ) from exc

        if mat.ndim != 2:
            raise SemanticDictionaryAgentError(
                f"{name} debe ser una matriz 2-D. Dimensión recibida: {mat.ndim}."
            )

        if mat.size == 0 or mat.shape[0] == 0 or mat.shape[1] == 0:
            raise SemanticDictionaryAgentError(
                f"{name} no puede ser una matriz vacía."
            )

        if not np.all(np.isfinite(mat)):
            raise SemanticDictionaryAgentError(
                f"{name} contiene entradas NaN o infinitas."
            )

        return mat

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

    # ─────────────────────────────────────────────────────────────────────────
    # Validación de caché semántico
    # ─────────────────────────────────────────────────────────────────────────
    def _validate_cache_states(
        self,
        cache_states,
        dimension: int
    ) -> Tuple[VectorF64, ...]:
        """
        Valida una colección de estados de caché.

        Acepta:
        - None → caché vacío.
        - Lista/tupla de vectores.
        - Matriz 2-D donde cada fila es un estado.
        - Vector 1-D interpretado como un único estado.
        """
        if cache_states is None:
            return tuple()

        if isinstance(cache_states, np.ndarray):
            if cache_states.ndim == 1:
                raw_states = [cache_states]
            elif cache_states.ndim == 2:
                raw_states = [cache_states[i, :] for i in range(cache_states.shape[0])]
            else:
                raise SemanticDictionaryAgentError(
                    "cache_states debe ser None, vector 1-D, matriz 2-D o secuencia de vectores."
                )
        else:
            try:
                raw_states = list(cache_states)
            except TypeError as exc:
                raise SemanticDictionaryAgentError(
                    "cache_states no es iterable ni convertible a secuencia de vectores."
                ) from exc

        validated_states: List[VectorF64] = []

        for idx, state in enumerate(raw_states):
            state_name = f"cache_states[{idx}]"
            vector = self._as_finite_vector(state_name, state)

            if vector.size != dimension:
                raise SemanticDictionaryAgentError(
                    f"{state_name} tiene dimensión {vector.size}, "
                    f"incompatible con la dimensión esperada {dimension}."
                )

            validated_states.append(vector)

        return tuple(validated_states)

    # ─────────────────────────────────────────────────────────────────────────
    # Auditoría de inmersión difeomórfica
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_diffeomorphic_immersion(
        self,
        jacobian_F_star: MatrixF64
    ) -> DiffeomorphicImmersionData:
        r"""
        Computa el SVD del Jacobiano (pushforward $F_*$) para extraer el núcleo.

        Condición de inmersión:
            $\dim(\ker(F_*)) = n - \operatorname{rank}(F_*) = 0$.

        Además se impone condicionamiento espectral acotado:
            $\kappa(F_*) \leq \kappa_{\max}$.
        """
        J = self._as_finite_matrix("jacobian_F_star", jacobian_F_star)
        m, n = J.shape

        try:
            _, singular_values, _ = la.svd(
                J,
                full_matrices=False,
                check_finite=False
            )
        except la.LinAlgError as exc:
            raise SemanticDegeneracyError(
                "Fallo en la descomposición SVD del pushforward F_*."
            ) from exc

        if singular_values.size == 0:
            raise SemanticDegeneracyError(
                "El Jacobiano no produjo valores singulares."
            )

        s = np.asarray(singular_values, dtype=np.float64)

        if not np.all(np.isfinite(s)):
            raise SemanticDegeneracyError(
                "Los valores singulares de F_* contienen NaN o infinitos."
            )

        s_max = float(s[0])
        s_min = float(s[-1])

        # Tolerancia adaptativa de rango numérico.
        rank_tolerance = max(
            _SVD_ABSOLUTE_TOLERANCE,
            float(max(m, n)) * _MACHINE_EPSILON * max(s_max, 1.0)
        )

        numerical_rank = int(np.sum(s > rank_tolerance))
        kernel_dimension = int(n - numerical_rank)

        condition_number = float(s_max / s_min) if s_min > 0.0 else math.inf

        if kernel_dimension > 0:
            raise SemanticDegeneracyError(
                "Degeneración de Gauge detectada. "
                f"El núcleo de la proyección tiene dimensión {kernel_dimension} > 0. "
                "Traducciones semánticas colisionan."
            )

        if not math.isfinite(condition_number):
            raise SemanticDegeneracyError(
                "Número de condición no finito para el pushforward F_*."
            )

        if condition_number > _JACOBIAN_CONDITION_MAX:
            raise SemanticDegeneracyError(
                "Pushforward F_* numéricamente hiper-caótico. "
                f"κ(F_*)={condition_number:.6e} > κ_max={_JACOBIAN_CONDITION_MAX:.6e}."
            )

        return DiffeomorphicImmersionData(
            jacobian_shape=(m, n),
            numerical_rank=numerical_rank,
            kernel_dimension=kernel_dimension,
            smallest_singular_value=s_min,
            condition_number=condition_number,
            rank_tolerance=rank_tolerance,
            is_isomorphic=True
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ÚLTIMO MÉTODO DE FASE 1
    # Puente formal hacia FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _complete_phase1_diffeomorphic_certification(
        self,
        jacobian_F_star: MatrixF64,
        distance_topological: float,
        distance_narrative: float,
        cache_states,
        decision_trajectory: VectorF64
    ) -> Phase1ImmersionBridge:
        r"""
        Último método de la Fase 1.

        Ejecuta:
            1. Validación de entradas.
            2. Auditoría de inmersión difeomórfica.
            3. Emisión del puente funtorial hacia la Fase 2.

        Este retorno es la continuación formal de la Fase 1 y el argumento
        inicial obligatorio del primer método de la Fase 2.
        """
        J = self._as_finite_matrix("jacobian_F_star", jacobian_F_star)

        d_top = self._as_nonnegative_float(
            "distance_topological",
            distance_topological
        )
        d_narr = self._as_nonnegative_float(
            "distance_narrative",
            distance_narrative
        )

        trajectory = self._as_finite_vector(
            "decision_trajectory",
            decision_trajectory
        )

        trajectory_norm = self._safe_l2_norm(trajectory)

        if not math.isfinite(trajectory_norm):
            raise SemanticDictionaryAgentError(
                "La norma de decision_trajectory no es finita."
            )

        if trajectory_norm <= _MACHINE_EPSILON:
            raise SemanticDictionaryAgentError(
                "decision_trajectory no puede ser el vector nulo."
            )

        cache_tuple = self._validate_cache_states(
            cache_states=cache_states,
            dimension=trajectory.size
        )

        immersion_audit = self._audit_diffeomorphic_immersion(J)

        return Phase1ImmersionBridge(
            immersion_audit=immersion_audit,
            jacobian_F_star=J,
            distance_topological=d_top,
            distance_narrative=d_narr,
            cache_states=cache_tuple,
            decision_trajectory=trajectory
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: CERTIFICACIÓN DEL RETRACTO DE DEFORMACIÓN (LIPSCHITZ)             ║
# ║                                                                             ║
# ║   Φ₂(Φ₁(...)) = L_medido                                                    ║
# ║                                                                             ║
# ║   1. Consume el puente emitido por la Fase 1.                               ║
# ║   2. Calcula la constante de Lipschitz efectiva.                            ║
# ║   3. Exige L_medido ≤ L_max.                                                ║
# ║   4. Emite el puente formal hacia la Fase 3.                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_LipschitzRetractionCertifier(Phase1_DiffeomorphicImmersionAuditor):
    r"""
    Fase 2 del endofuntor.

    Aplica una cota de Lipschitz sobre las plantillas generativas, actuando
    como un retracto de deformación que aplasta las alucinaciones
    probabilísticas del LLM.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMER MÉTODO DE FASE 2
    # Inicio formal a partir del puente de Fase 1
    # ─────────────────────────────────────────────────────────────────────────
    def _begin_phase2_from_phase1_bridge(
        self,
        phase1_bridge: Phase1ImmersionBridge
    ) -> Phase2LipschitzBridge:
        r"""
        Primer método de la Fase 2.

        Consume el `Phase1ImmersionBridge` emitido por el último método de la
        Fase 1 y ejecuta la certificación de Lipschitz.
        """
        if not isinstance(phase1_bridge, Phase1ImmersionBridge):
            raise SemanticDictionaryAgentError(
                "La Fase 2 requiere un Phase1ImmersionBridge emitido por la Fase 1."
            )

        lipschitz_audit = self._certify_lipschitz_retraction(
            distance_topological=phase1_bridge.distance_topological,
            distance_narrative=phase1_bridge.distance_narrative
        )

        return Phase2LipschitzBridge(
            phase1_bridge=phase1_bridge,
            lipschitz_audit=lipschitz_audit
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Certificación del retracto de deformación
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_lipschitz_retraction(
        self,
        distance_topological: float,
        distance_narrative: float
    ) -> LipschitzRetractionData:
        r"""
        Certifica:
            $|F(x) - F(y)|_{Narr} \leq L_{\max} |x - y|_{Top}$.

        Si la distancia topológica es infinitesimal, se exige que la distancia
        narrativa sea numéricamente nula; de otro modo, la constante de
        Lipschitz efectiva diverge.
        """
        d_top = self._as_nonnegative_float(
            "distance_topological",
            distance_topological
        )
        d_narr = self._as_nonnegative_float(
            "distance_narrative",
            distance_narrative
        )

        if d_top <= _MACHINE_EPSILON:
            if d_narr <= _LIPSCHITZ_ZERO_TOLERANCE:
                measured_lipschitz = 0.0
            else:
                raise LipschitzRetractionViolation(
                    "Variación topológica infinitesimal con distancia narrativa no nula. "
                    f"d_top={d_top:.6e}, d_narr={d_narr:.6e}. "
                    "La constante de Lipschitz efectiva diverge."
                )
        else:
            measured_lipschitz = float(d_narr / d_top)

            if not math.isfinite(measured_lipschitz):
                raise LipschitzRetractionViolation(
                    "Constante de Lipschitz medida no finita."
                )

        if measured_lipschitz > _LIPSCHITZ_MAX_RATIO:
            raise LipschitzRetractionViolation(
                "Alucinación probabilística. Cota de Lipschitz violada: "
                f"L={measured_lipschitz:.6f} > L_max={_LIPSCHITZ_MAX_RATIO:.6f}. "
                "Probabilidad de validación colapsada a P=0."
            )

        return LipschitzRetractionData(
            topological_distance=d_top,
            narrative_distance=d_narr,
            measured_lipschitz_constant=measured_lipschitz,
            is_deformation_bounded=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: GOBERNANZA TERMODINÁMICA DE EVICCIÓN                              ║
# ║                                                                             ║
# ║   Φ₃(Φ₂(Φ₁(...))) = (S_pre, S_post, purga, estabilidad)                     ║
# ║                                                                             ║
# ║   1. Consume el puente emitido por la Fase 2.                               ║
# ║   2. Construye el operador densidad ρ_C del caché.                          ║
# ║   3. Computa la entropía de von Neumann.                                    ║
# ║   4. Purga estados ortogonales a la geodésica de decisión.                  ║
# ║   5. Garantiza estabilidad térmica post-purga.                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_ThermodynamicEvictor(Phase2_LipschitzRetractionCertifier):
    r"""
    Fase 3 del endofuntor.

    Modela el caché del diccionario semántico como un operador densidad
    $\rho_C$ y purga los tensores que se vuelven ortogonales a la geodésica
    actual de decisión.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMER MÉTODO DE FASE 3
    # Inicio formal a partir del puente de Fase 2
    # ─────────────────────────────────────────────────────────────────────────
    def _begin_phase3_from_phase2_bridge(
        self,
        phase2_bridge: Phase2LipschitzBridge
    ) -> ThermodynamicEvictionState:
        r"""
        Primer método de la Fase 3.

        Consume el `Phase2LipschitzBridge` emitido por la Fase 2 y ejecuta la
        gobernanza termodinámica del caché semántico.
        """
        if not isinstance(phase2_bridge, Phase2LipschitzBridge):
            raise SemanticDictionaryAgentError(
                "La Fase 3 requiere un Phase2LipschitzBridge emitido por la Fase 2."
            )

        return self._govern_thermodynamic_eviction(
            cache_states=phase2_bridge.phase1_bridge.cache_states,
            decision_trajectory=phase2_bridge.phase1_bridge.decision_trajectory
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Construcción del operador densidad
    # ─────────────────────────────────────────────────────────────────────────
    def _build_density_operator(
        self,
        states: Tuple[VectorF64, ...] | List[VectorF64],
        dimension: int
    ) -> Tuple[DensityMatrix, int]:
        r"""
        Construye:
            $\rho_C = \frac{1}{N}\sum_i |v_i\rangle\langle v_i|$,
        con cada $|v_i\rangle$ normalizado.

        Returns
        -------
        Tuple[DensityMatrix, int]
            (rho, valid_count)
        """
        if dimension <= 0:
            raise ThermodynamicEvictionAnomaly(
                "No puede construirse un operador densidad con dimensión no positiva."
            )

        rho = np.zeros((dimension, dimension), dtype=np.complex128)
        valid_count = 0

        for state in states:
            vector = np.asarray(state, dtype=np.float64)

            if vector.size != dimension:
                raise ThermodynamicEvictionAnomaly(
                    "Estado de caché con dimensión incompatible al construir ρ_C."
                )

            norm = self._safe_l2_norm(vector)

            if norm <= _MACHINE_EPSILON:
                continue

            unit = (vector / norm).astype(np.complex128)
            rho += np.outer(unit, unit.conj())
            valid_count += 1

        if valid_count > 0:
            rho /= float(valid_count)

        # Hermitización numérica explícita.
        rho = 0.5 * (rho + rho.conj().T)

        trace = float(np.trace(rho).real)

        if trace > _DENSITY_TRACE_FLOOR:
            rho /= trace
        elif valid_count > 0:
            raise ThermodynamicEvictionAnomaly(
                "El operador densidad construido tiene traza no positiva."
            )

        return rho, valid_count

    # ─────────────────────────────────────────────────────────────────────────
    # Entropía de von Neumann
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_von_neumann_entropy(self, rho: DensityMatrix) -> float:
        r"""
        Computa:
            $S(\rho) = -k_B \operatorname{Tr}(\rho \ln \rho)$.

        Implementación robusta:
        - Hermitización.
        - Normalización de traza.
        - Espectro real vía `eigvalsh`.
        - Descarte de autovalores numéricamente nulos.
        - Renormalización positiva.
        """
        rho_arr = np.asarray(rho, dtype=np.complex128)

        if rho_arr.size == 0:
            return 0.0

        if rho_arr.ndim != 2 or rho_arr.shape[0] != rho_arr.shape[1]:
            raise ThermodynamicEvictionAnomaly(
                "El operador densidad debe ser una matriz cuadrada."
            )

        if not np.all(np.isfinite(rho_arr)):
            raise ThermodynamicEvictionAnomaly(
                "El operador densidad contiene entradas no finitas."
            )

        rho_h = 0.5 * (rho_arr + rho_arr.conj().T)
        trace = float(np.trace(rho_h).real)

        if trace > _DENSITY_TRACE_FLOOR:
            rho_h = rho_h / trace
        else:
            if np.any(np.abs(rho_h) > 1e-12):
                raise ThermodynamicEvictionAnomaly(
                    "Operador densidad no nulo con traza numéricamente nula."
                )
            return 0.0

        try:
            eigenvalues = la.eigvalsh(rho_h, check_finite=False)
        except la.LinAlgError as exc:
            raise ThermodynamicEvictionAnomaly(
                "Fallo en la diagonalización del operador densidad."
            ) from exc

        eigenvalues = np.real(eigenvalues)

        if not np.all(np.isfinite(eigenvalues)):
            raise ThermodynamicEvictionAnomaly(
                "Autovalores del operador densidad no finitos."
            )

        min_eigenvalue = float(np.min(eigenvalues))

        # Se toleran negatividades infinitesimales por redondeo.
        if min_eigenvalue < -1e-8:
            raise ThermodynamicEvictionAnomaly(
                "Operador densidad no positivo semidefinido. "
                f"Autovalor mínimo: {min_eigenvalue:.6e}."
            )

        eigenvalues = np.where(
            eigenvalues > _ENTROPY_EIGENVALUE_FLOOR,
            eigenvalues,
            0.0
        )

        total_mass = float(np.sum(eigenvalues))

        if total_mass > _DENSITY_TRACE_FLOOR:
            eigenvalues = eigenvalues / total_mass
        else:
            return 0.0

        positive_eigenvalues = eigenvalues[eigenvalues > 0.0]

        if positive_eigenvalues.size == 0:
            return 0.0

        try:
            k_b = float(BOLTZMANN_CONSTANT)
        except (TypeError, ValueError):
            k_b = 1.0

        if not math.isfinite(k_b) or k_b <= 0.0:
            k_b = 1.0

        entropy = -k_b * float(
            np.sum(positive_eigenvalues * np.log(positive_eigenvalues))
        )

        if not math.isfinite(entropy):
            raise ThermodynamicEvictionAnomaly(
                "Entropía de von Neumann no finita."
            )

        if entropy < 0.0 and entropy > -1e-12:
            entropy = 0.0

        return float(entropy)

    # ─────────────────────────────────────────────────────────────────────────
    # Gobernanza termodinámica de evicción
    # ─────────────────────────────────────────────────────────────────────────
    def _govern_thermodynamic_eviction(
        self,
        cache_states,
        decision_trajectory: VectorF64
    ) -> ThermodynamicEvictionState:
        r"""
        Mide la entropía del caché y purga estados usando:
            $\cos(\theta) = \frac{|\langle u, v \rangle|}{\|u\|\|v\|}$.

        Política de estabilidad:
        - Si $S(\rho_C) > S_{crit}$, se purgan estados con
          $\cos(\theta) < \tau_{orth}$.
        - Si la purga deja vacío el caché, se retiene el estado más alineado
          como ancla Dirac para evitar el vacío ontológico.
        - Si la entropía post-purga sigue crítica, se colapsa el caché al
          estado más alineado.
        """
        trajectory = self._as_finite_vector(
            "decision_trajectory",
            decision_trajectory
        )

        dimension = trajectory.size
        trajectory_norm = self._safe_l2_norm(trajectory)

        if not math.isfinite(trajectory_norm):
            raise ThermodynamicEvictionAnomaly(
                "Norma no finita de la trayectoria de decisión."
            )

        if trajectory_norm <= _MACHINE_EPSILON:
            raise ThermodynamicEvictionAnomaly(
                "La trayectoria de decisión no puede ser el vector nulo."
            )

        trajectory_unit = (trajectory / trajectory_norm).astype(np.complex128)

        cache_tuple = self._validate_cache_states(
            cache_states=cache_states,
            dimension=dimension
        )

        cache_cardinality_pre = len(cache_tuple)

        if cache_cardinality_pre == 0:
            return ThermodynamicEvictionState(
                von_neumann_entropy_pre=0.0,
                von_neumann_entropy_post=0.0,
                evicted_vectors_count=0,
                cache_cardinality_pre=0,
                cache_cardinality_post=0,
                is_thermally_stable=True
            )

        valid_states: List[VectorF64] = []
        evicted_count = 0

        # Primera depuración: estados nulos son físicamente inválidos.
        for idx, state in enumerate(cache_tuple):
            state_name = f"cache_states[{idx}]"
            vector = self._as_finite_vector(state_name, state)

            if vector.size != dimension:
                raise ThermodynamicEvictionAnomaly(
                    f"{state_name} tiene dimensión incompatible con la trayectoria."
                )

            norm = self._safe_l2_norm(vector)

            if norm <= _MACHINE_EPSILON:
                evicted_count += 1
                continue

            valid_states.append(vector)

        zero_evictions = evicted_count

        if not valid_states:
            return ThermodynamicEvictionState(
                von_neumann_entropy_pre=0.0,
                von_neumann_entropy_post=0.0,
                evicted_vectors_count=evicted_count,
                cache_cardinality_pre=cache_cardinality_pre,
                cache_cardinality_post=0,
                is_thermally_stable=True
            )

        rho_pre, _ = self._build_density_operator(valid_states, dimension)
        entropy_pre = self._compute_von_neumann_entropy(rho_pre)

        surviving_states = list(valid_states)
        rho_post = rho_pre
        entropy_post = entropy_pre

        if entropy_pre > _ENTROPY_CRITICAL_THRESHOLD:
            alignments: List[Tuple[float, VectorF64]] = []

            for state in valid_states:
                norm = self._safe_l2_norm(state)

                if norm <= _MACHINE_EPSILON:
                    continue

                unit = (state / norm).astype(np.complex128)
                cos_theta = float(abs(np.vdot(trajectory_unit, unit)))

                if not math.isfinite(cos_theta):
                    raise ThermodynamicEvictionAnomaly(
                        "Coseno de alineación no finito durante la purga térmica."
                    )

                # Acotación numérica estricta.
                cos_theta = min(1.0, max(0.0, cos_theta))
                alignments.append((cos_theta, state))

            if alignments:
                surviving_states = [
                    state
                    for cos_theta, state in alignments
                    if cos_theta >= _ORTHOGONALITY_TOLERANCE
                ]

                best_cos, best_state = max(
                    alignments,
                    key=lambda item: item[0]
                )

                purge_evictions = len(valid_states) - len(surviving_states)

                # Retención de emergencia para evitar vacío ontológico.
                if not surviving_states:
                    surviving_states = [best_state]
                    purge_evictions = len(valid_states) - 1
                    logger.warning(
                        "Purga térmica dejó el caché vacío. "
                        "Se retiene el estado más alineado como ancla Dirac. "
                        f"max_cos_theta={best_cos:.6f}."
                    )

                evicted_count = zero_evictions + purge_evictions

                rho_post, _ = self._build_density_operator(
                    surviving_states,
                    dimension
                )
                entropy_post = self._compute_von_neumann_entropy(rho_post)

                # Si la entropía sigue crítica, colapso puro al estado más alineado.
                if entropy_post > _ENTROPY_CRITICAL_THRESHOLD:
                    surviving_states = [best_state]
                    evicted_count = zero_evictions + len(valid_states) - 1

                    rho_post, _ = self._build_density_operator(
                        surviving_states,
                        dimension
                    )
                    entropy_post = self._compute_von_neumann_entropy(rho_post)

                    if entropy_post > _ENTROPY_CRITICAL_THRESHOLD:
                        raise ThermodynamicEvictionAnomaly(
                            "La entropía del caché no pudo ser mitigada ni siquiera "
                            "colapsando al estado más alineado."
                        )

        cache_cardinality_post = len(surviving_states)
        is_thermally_stable = entropy_post <= _ENTROPY_CRITICAL_THRESHOLD

        if not is_thermally_stable:
            raise ThermodynamicEvictionAnomaly(
                "El caché semántico permanece térmicamente inestable tras la purga. "
                f"S_post={entropy_post:.6f} > S_crit={_ENTROPY_CRITICAL_THRESHOLD:.6f}."
            )

        return ThermodynamicEvictionState(
            von_neumann_entropy_pre=entropy_pre,
            von_neumann_entropy_post=entropy_post,
            evicted_vectors_count=evicted_count,
            cache_cardinality_pre=cache_cardinality_pre,
            cache_cardinality_post=cache_cardinality_post,
            is_thermally_stable=is_thermally_stable
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: SEMANTIC DICTIONARY AGENT                            ║
# ║                                                                             ║
# ║   Endofuntor:                                                               ║
# ║       Z_DictAgent = Φ₃ ∘ Φ₂ ∘ Φ₁                                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class SemanticDictionaryAgent(Morphism, Phase3_ThermodynamicEvictor):
    r"""
    El Custodio de la Ontología y Fibrado Semántico puro.

    Subordina la traducción lingüística al determinismo de las variedades
    algebraicas, vetando cualquier divergencia de Gauge estocástica inyectada
    por el LLM.
    """

    def execute_semantic_dictionary_governance(
        self,
        jacobian_F_star: MatrixF64,
        distance_topological: float,
        distance_narrative: float,
        cache_states: List[VectorF64],
        decision_trajectory: VectorF64
    ) -> SemanticDictionaryAuditState:
        r"""
        Ejecuta la composición endofuntorial rigurosa en 3 fases anidadas.

        Flujo:
        ----
        1. Fase 1:
           `_complete_phase1_diffeomorphic_certification`
           → `Phase1ImmersionBridge`

        2. Fase 2:
           `_begin_phase2_from_phase1_bridge`
           → `Phase2LipschitzBridge`

        3. Fase 3:
           `_begin_phase3_from_phase2_bridge`
           → `ThermodynamicEvictionState`

        4. Ensamblaje final:
           `SemanticDictionaryAuditState`
        """
        # ── Fase 1: Inmersión difeomórfica del pushforward ───────────────────
        phase1_bridge = self._complete_phase1_diffeomorphic_certification(
            jacobian_F_star=jacobian_F_star,
            distance_topological=distance_topological,
            distance_narrative=distance_narrative,
            cache_states=cache_states,
            decision_trajectory=decision_trajectory
        )

        # ── Fase 2: Retracto de deformación Lipschitz ────────────────────────
        phase2_bridge = self._begin_phase2_from_phase1_bridge(
            phase1_bridge=phase1_bridge
        )

        # ── Fase 3: Gobernanza termodinámica del caché ───────────────────────
        thermodynamic_state = self._begin_phase3_from_phase2_bridge(
            phase2_bridge=phase2_bridge
        )

        # ── Ensamblaje del objeto final ──────────────────────────────────────
        final_state = SemanticDictionaryAuditState(
            immersion_audit=phase1_bridge.immersion_audit,
            lipschitz_audit=phase2_bridge.lipschitz_audit,
            thermodynamic_state=thermodynamic_state,
            is_epistemologically_valid=True
        )

        entropy_delta = (
            thermodynamic_state.von_neumann_entropy_post
            - thermodynamic_state.von_neumann_entropy_pre
        )

        logger.info(
            "Gobernanza Categórica completada. "
            f"Núcleo: {final_state.immersion_audit.kernel_dimension} | "
            f"κ: {final_state.immersion_audit.condition_number:.3e} | "
            f"L: {final_state.lipschitz_audit.measured_lipschitz_constant:.3f} | "
            f"ΔS: {entropy_delta:.3f} | "
            f"Evictados: {thermodynamic_state.evicted_vectors_count} | "
            f"Caché post: {thermodynamic_state.cache_cardinality_post}"
        )

        return final_state


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "SemanticDictionaryAgentError",
    "SemanticDegeneracyError",
    "LipschitzRetractionViolation",
    "ThermodynamicEvictionAnomaly",
    "DiffeomorphicImmersionData",
    "LipschitzRetractionData",
    "ThermodynamicEvictionState",
    "Phase1ImmersionBridge",
    "Phase2LipschitzBridge",
    "SemanticDictionaryAuditState",
    "Phase1_DiffeomorphicImmersionAuditor",
    "Phase2_LipschitzRetractionCertifier",
    "Phase3_ThermodynamicEvictor",
    "SemanticDictionaryAgent",
]