# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Algebraic Tactics Agent (Operador de Anillos y Auditor Homológico)  ║
║ Ruta   : app/tactics/algebraic_tactics_agent.py                              ║
║ Versión: 2.1.0-Rigorous-Spectral-Categorical-Homological-Ring-Veto           ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA ALGEBRAICA
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra el Estrato TACTICS (Nivel 2). Actúa como Endofuntor Soberano
sobre el motor esclavo `apu_processor.py`. Su mandato axiomático es:

  (i)  garantizar la homogeneidad del tensor de costos como objeto de un anillo
       conmutativo ℛ = (ℝ, ⊕, ⊙) con absorción monádica de singularidades FPU;
  (ii) auditar la integridad homológica y espectral del 1-esqueleto inducido
       (Laplaciano combinatorio L = B₁ B₁ᵀ, números de Betti, valor de Fiedler);
  (iii) aniquilar tensores desconectados o no-anulares antes de la escalada al
       Estrato STRATEGY.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta)
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Homogeneidad del Anillo Conmutativo ℛ y saneamiento monádico:
         • Clausura aditiva (⊕), conmutatividad, asociatividad muestral.
         • Distributividad del producto Hadamard ⊙ sobre ⊕.
         • Neutros aditivo (0) y multiplicativo (1) del producto Hadamard.
         • Mónada Option: NaN/±Inf ↦ 0 (elemento absorbente seguro).
         • Espectro singular de la matriz de costos (condición, rango, norma).
         Método terminal `audit_ring_manifold` → RingHomogeneityValidation
         (dominio inicial exacto de la Fase 2).

Fase 2 → Auditoría Homológica y Espectral del 1-esqueleto:
         Continuación directa del manifold anular. Construye el grafo de
         adyacencia inducido, la matriz de incidencia orientada B₁, el
         Laplaciano L = B₁ B₁ᵀ = D − A, extrae Spec(L), β₀ = mult(λ=0),
         β₁ (ciclos independientes), valor de Fiedler λ₂ y verifica
         conectividad (Teorema de Kirchhoff–Fiedler).

Fase 3 → Proyección Ortogonal / Orquestador Supremo:
         Emisión de un CategoricalState puro al DAG del pipeline_director.py.
         Composición: CategoricalState ∘ SimplicialSkeletonAudit ∘ RingHomogeneityValidation.

AXIOMAS DE EJECUCIÓN (Formulación Rigurosa)
────────────────────────────────────────────────────────────────────────────────
§1. Anillo conmutativo ℛ = (ℝⁿ, ⊕, ⊙) sobre columnas del tensor de costos:
    (a ⊕ b)_i = a_i + b_i          (suma vectorial componente a componente)
    (a ⊙ b)_i = a_i · b_i          (producto de Hadamard)
    ∀ a,b,c ∈ ℝⁿ:
      a ⊕ b = b ⊕ a                          (conmutatividad de ⊕)
      (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)              (asociatividad de ⊕)
      a ⊙ (b ⊕ c) = (a ⊙ b) ⊕ (a ⊙ c)        (distributividad)
      a ⊕ 0 = a                              (neutro aditivo)
      a ⊙ 1 = a                              (neutro multiplicativo Hadamard)

§2. Saneamiento monádico (transformación natural η: Id ⇒ Option):
    η(x) = Nothing  si x ∉ ℝ finito (NaN, ±∞);
    η(x) = Just(x)  en caso contrario.
    La proyección Nothing ↦ 0 es el morfismo de anillos al objeto cero.

§3. Laplaciano combinatorio y Teorema de Kirchhoff–Fiedler:
    Sea G = (V,E) el grafo no dirigido inducido por el Gram de costos.
    B₁ ∈ Mat_{|V|×|E|}(ℝ)  incidencia orientada:  B₁ e_{ij} = e_i − e_j.
    L = B₁ B₁ᵀ = D − A  (simétrico, semidefinido positivo).
    Spec(L) = {0 = λ₁ ≤ λ₂ ≤ ⋯ ≤ λ_n}.
    β₀ := multiplicidad algebraica de λ = 0  =  dim(ker L).
    λ₂  = valor de Fiedler (conectividad algebraica).
    Condición de validez: β₀ ≡ 1  ∧  λ₂ > ε_Fiedler.

§4. Homología del 1-esqueleto (fórmula de Euler en dim ≤ 1):
    β₀ = |V| − rank(B₁)
    β₁ = |E| − rank(B₁)
    χ = β₀ − β₁ = |V| − |E|
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:
    class TopologicalInvariantError(Exception):
        pass

    class Morphism:
        pass

    class CategoricalState:
        def __init__(self, stratum=None, payload=None, context=None):
            self.stratum = stratum
            self.payload = payload or {}
            self.context = context or {}

    class Stratum:
        TACTICS = 2

try:
    from app.tactics.apu_processor import APUProcessor
except ImportError:
    class APUProcessor:
        def process(self, ast_data: List[Dict[str, Any]]) -> NDArray[np.float64]:
            # Stub: grafo camino conexo de 3 nodos × 2 features
            return np.array(
                [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                dtype=np.float64,
            )

logger = logging.getLogger("MIC.Tactics.AlgebraicTacticsAgent")

# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, ESPECTRALES Y TOLERANCIAS
# ══════════════════════════════════════════════════════════════════════════════
class AlgebraicConstants:
    """
    Constantes fundamentales para la Teoría de Anillos, la Homología Simplicial
    y la Teoría Espectral de Laplacianos combinatorios.
    """
    MACHINE_EPS: float = float(np.finfo(np.float64).eps)

    # Tolerancia para verificaciones axiomáticas del anillo (norma ∞)
    RING_TOLERANCE: float = 1e-9

    # Umbral de detección λ ≈ 0 en Spec(L) (Fiedler / β₀)
    FIEDLER_TOLERANCE: float = 1e-7

    # Número máximo de columnas muestreadas en tests axiomáticos (complejidad O(k³))
    RING_SAMPLE_SIZE: int = 12

    # Semilla reproducible para el muestreo de triplas del anillo
    RING_SAMPLE_SEED: int = 42

    # Condición máxima aceptable de la matriz de costos (σ_max / σ_min)
    MAX_CONDITION_NUMBER: float = 1e12

    # Dimensión mínima del tensor de costos (filas = vértices del 1-esqueleto)
    MIN_VERTICES: int = 1


# ══════════════════════════════════════════════════════════════════════════════
# §B. MÓNADA OPTION (Elemento Absorbente de Singularidades FPU)
# ══════════════════════════════════════════════════════════════════════════════
T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True, slots=True)
class OptionMonad(Generic[T]):
    r"""
    Mónada Option/Maybe sobre la categoría de tipos.

    Endofuntor T: 𝒞 → 𝒞 con transformaciones naturales:
      η : Id ⇒ T   (unit / return)
      μ : T∘T ⇒ T  (join / multiplicación)

    Absorbe singularidades de la FPU (NaN, ±Inf) hacia Nothing, que se proyecta
    al neutro aditivo 0 del anillo ℛ.
    """
    value: Optional[T]
    is_singular: bool

    # ─── Constructores ───────────────────────────────────────────────────────
    @classmethod
    def unit(cls, val: T) -> "OptionMonad[T]":
        r"""η: Id → T. Inyecta un valor; mapea no-finitos a Nothing."""
        if val is None:
            return cls(None, True)
        if isinstance(val, (float, np.floating)) and not math.isfinite(float(val)):
            return cls(None, True)
        if isinstance(val, (complex, np.complexfloating)):
            if not (math.isfinite(val.real) and math.isfinite(val.imag)):
                return cls(None, True)
        return cls(val, False)

    @classmethod
    def nothing(cls) -> "OptionMonad[T]":
        """Objeto inicial de la mónada (Nothing)."""
        return cls(None, True)

    @classmethod
    def pure(cls, val: T) -> "OptionMonad[T]":
        """Sinónimo de unit para valores ya saneados."""
        return cls(val, False)

    # ─── Operaciones monádicas ───────────────────────────────────────────────
    def bind(self, f: Callable[[T], "OptionMonad[U]"]) -> "OptionMonad[U]":
        r"""μ ∘ T(f): composición monádica estricta (>>=)."""
        if self.is_singular or self.value is None:
            return OptionMonad.nothing()
        try:
            return f(self.value)
        except Exception:
            return OptionMonad.nothing()

    def map(self, f: Callable[[T], U]) -> "OptionMonad[U]":
        r"""Acción funtorial T(f): Option a → Option b."""
        return self.bind(lambda x: OptionMonad.unit(f(x)))

    def get_or_else(self, default: T) -> T:
        """Proyección al valor subyacente o default (Nothing ↦ default)."""
        if self.is_singular or self.value is None:
            return default
        return self.value

    def __repr__(self) -> str:
        if self.is_singular:
            return "OptionMonad.Nothing"
        return f"OptionMonad.Just({self.value!r})"


# ══════════════════════════════════════════════════════════════════════════════
# §C. JERARQUÍA DE EXCEPCIONES TÁCTICAS (VETOS ESTRUCTURALES)
# ══════════════════════════════════════════════════════════════════════════════
class AlgebraicTacticsError(TopologicalInvariantError):
    """Excepción raíz del Endofuntor Táctico."""
    pass


class RingSymmetryViolation(AlgebraicTacticsError):
    r"""El operador de costo violó clausura, distributividad u homomorfismo de ℛ."""
    pass


class RingDegeneracyError(AlgebraicTacticsError):
    r"""Matriz de costos degenerada: rango nulo, condición explosiva o vacía."""
    pass


class TopologicalIslandError(AlgebraicTacticsError):
    r"""β₀ > 1: fractura logística (el grafo inducido no es conexo)."""
    pass


class HomologicalInvariantError(AlgebraicTacticsError):
    r"""Inconsistencia entre β₀ espectral y β₀ = |V| − rank(B₁)."""
    pass


class EmptyCostTensorError(AlgebraicTacticsError):
    r"""El tensor de costos incidente es el objeto inicial 0."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# §D. DTOs INMUTABLES (Contratos entre Fases Anidadas)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class RingHomogeneityValidation:
    r"""
    Producto terminal de la Fase 1 / dominio inicial de la Fase 2.
    Certificado completo de homogeneidad algebraica del anillo ℛ.
    """
    is_closed: bool
    is_commutative: bool
    is_associative: bool
    is_distributive: bool
    has_additive_identity: bool
    has_multiplicative_identity: bool
    cost_matrix: NDArray[np.float64]
    monadic_failures: int
    frobenius_norm: float
    spectral_norm: float
    condition_number: float
    matrix_rank: int
    singular_values: NDArray[np.float64]
    ring_check_log: str


@dataclass(frozen=True, slots=True)
class SimplicialSkeletonAudit:
    r"""
    Producto de la Fase 2: invariantes homológicos y espectrales del 1-esqueleto.
    """
    betti_0: int                              # dim H₀ = # componentes conexas
    betti_1: int                              # dim H₁ = # ciclos independientes
    fiedler_value: float                      # λ₂ (conectividad algebraica)
    is_connected: bool
    laplacian_spectrum: NDArray[np.float64]
    incidence_rank: int                       # rank(B₁)
    num_vertices: int
    num_edges: int
    euler_characteristic: int                 # χ = |V| − |E| = β₀ − β₁
    adjacency_density: float                  # 2|E| / (|V|(|V|−1))
    ring_validation: RingHomogeneityValidation  # enlace funtorial a Fase 1
    homology_log: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: HOMOGENEIDAD DEL ANILLO CONMUTATIVO ℛ Y SANEAMIENTO MONÁDICO
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_AlgebraicRingAuditor:
    r"""
    Subyuga al motor esclavo `apu_processor.py` y certifica que el espacio de
    costos satisface los axiomas de un anillo conmutativo
        ℛ = (ℝⁿ, ⊕, ⊙)
    bajo la suma vectorial y el producto de Hadamard, con absorción monádica
    de singularidades FPU.

    El método terminal `audit_ring_manifold` es el puente formal hacia la
    Fase 2: su codominio RingHomogeneityValidation es exactamente el dominio
    de `Phase2_TopologicalSkeletonAuditor.audit_simplicial_skeleton`.
    """

    # ─── 1.1 Saneamiento monádico del tensor ─────────────────────────────────
    @staticmethod
    def _sanitize_matrix_monadic(
        matrix: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], int]:
        r"""
        Aplica η (OptionMonad.unit) a cada entrada:
          NaN / ±Inf  ↦  Nothing  ↦  0  (neutro aditivo del anillo).
        Retorna (matriz_saneada, número_de_singularidades_absorbidas).
        """
        flat = matrix.ravel()
        sane = np.empty_like(flat, dtype=np.float64)
        failures = 0
        for i, x in enumerate(flat):
            m = OptionMonad.unit(float(x))
            if m.is_singular:
                failures += 1
                sane[i] = 0.0
            else:
                sane[i] = float(m.value)  # type: ignore[arg-type]
        return sane.reshape(matrix.shape), failures

    # ─── 1.2 Espectro singular y condición del tensor de costos ──────────────
    @staticmethod
    def _spectral_profile(
        matrix: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], int, float, float, float]:
        r"""
        Descomposición en valores singulares (SVD económica):
          σ₁ ≥ σ₂ ≥ ⋯ ≥ σ_r > 0.
        Retorna (σ, rank, ‖·‖_F, ‖·‖₂=σ₁, κ=σ₁/σ_r).
        """
        if matrix.size == 0:
            return (
                np.array([], dtype=np.float64),
                0,
                0.0,
                0.0,
                float("inf"),
            )
        # SVD de valores singulares únicamente
        try:
            sv = la.svdvals(matrix)
        except la.LinAlgError:
            sv = np.array([], dtype=np.float64)

        tol = AlgebraicConstants.RING_TOLERANCE * max(matrix.shape) * (
            float(sv[0]) if len(sv) else 1.0
        )
        rank = int(np.sum(sv > tol)) if len(sv) else 0
        fro = float(np.linalg.norm(matrix, ord="fro"))
        spectral = float(sv[0]) if len(sv) else 0.0
        if rank > 0 and sv[rank - 1] > AlgebraicConstants.MACHINE_EPS:
            cond = float(sv[0] / sv[rank - 1])
        else:
            cond = float("inf") if rank == 0 else float(sv[0] / AlgebraicConstants.MACHINE_EPS)
        return sv, rank, fro, spectral, cond

    # ─── 1.3 Muestreo determinista de columnas ───────────────────────────────
    @staticmethod
    def _sample_column_indices(n_cols: int, k: int) -> NDArray[np.int64]:
        r"""
        Selección pseudoaleatoria reproducible de hasta k índices de columna.
        Usa Generator de NumPy con semilla fija (no muta el RNG global).
        """
        if n_cols <= 0:
            return np.array([], dtype=np.int64)
        rng = np.random.default_rng(AlgebraicConstants.RING_SAMPLE_SEED)
        k_eff = min(k, n_cols)
        return rng.choice(n_cols, size=k_eff, replace=False).astype(np.int64)

    # ─── 1.4 Clausura aditiva ────────────────────────────────────────────────
    @classmethod
    def _verify_additive_closure(
        cls, cost_matrix: NDArray[np.float64]
    ) -> bool:
        r"""
        ∀ columnas u, v muestreadas: u ⊕ v es finito componente a componente.
        (En ℝ esto es tautológico tras saneamiento; se re-verifica por rigor.)
        """
        n_cols = cost_matrix.shape[1]
        if n_cols < 2:
            return True
        idxs = cls._sample_column_indices(n_cols, AlgebraicConstants.RING_SAMPLE_SIZE)
        for i, a_idx in enumerate(idxs):
            for b_idx in idxs[i + 1 :]:
                s = cost_matrix[:, a_idx] + cost_matrix[:, b_idx]
                if not np.all(np.isfinite(s)):
                    return False
        return True

    # ─── 1.5 Conmutatividad de ⊕ ─────────────────────────────────────────────
    @classmethod
    def _verify_commutativity(
        cls, cost_matrix: NDArray[np.float64]
    ) -> bool:
        r"""a ⊕ b = b ⊕ a  (norma ∞ de la diferencia < ε)."""
        n_cols = cost_matrix.shape[1]
        if n_cols < 2:
            return True
        idxs = cls._sample_column_indices(n_cols, AlgebraicConstants.RING_SAMPLE_SIZE)
        for i, a_idx in enumerate(idxs):
            for b_idx in idxs[i + 1 :]:
                a = cost_matrix[:, a_idx]
                b = cost_matrix[:, b_idx]
                if la.norm((a + b) - (b + a), ord=np.inf) >= AlgebraicConstants.RING_TOLERANCE:
                    return False
        return True

    # ─── 1.6 Asociatividad de ⊕ ──────────────────────────────────────────────
    @classmethod
    def _verify_associativity(
        cls, cost_matrix: NDArray[np.float64]
    ) -> bool:
        r"""(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)."""
        n_cols = cost_matrix.shape[1]
        if n_cols < 1:
            return True
        idxs = cls._sample_column_indices(n_cols, min(6, AlgebraicConstants.RING_SAMPLE_SIZE))
        for a_idx in idxs:
            a = cost_matrix[:, a_idx]
            for b_idx in idxs:
                b = cost_matrix[:, b_idx]
                for c_idx in idxs:
                    c = cost_matrix[:, c_idx]
                    lhs = (a + b) + c
                    rhs = a + (b + c)
                    if la.norm(lhs - rhs, ord=np.inf) >= AlgebraicConstants.RING_TOLERANCE:
                        return False
        return True

    # ─── 1.7 Distributividad de ⊙ sobre ⊕ ────────────────────────────────────
    @classmethod
    def _verify_distributivity(
        cls, cost_matrix: NDArray[np.float64]
    ) -> bool:
        r"""a ⊙ (b ⊕ c) = (a ⊙ b) ⊕ (a ⊙ c)  (Hadamard sobre suma vectorial)."""
        n_cols = cost_matrix.shape[1]
        if n_cols < 1 or cost_matrix.size == 0:
            return False
        idxs = cls._sample_column_indices(n_cols, AlgebraicConstants.RING_SAMPLE_SIZE)
        for a_idx in idxs:
            a = cost_matrix[:, a_idx]
            for b_idx in idxs:
                b = cost_matrix[:, b_idx]
                for c_idx in idxs:
                    c = cost_matrix[:, c_idx]
                    lhs = a * (b + c)
                    rhs = (a * b) + (a * c)
                    if la.norm(lhs - rhs, ord=np.inf) >= AlgebraicConstants.RING_TOLERANCE:
                        return False
        return True

    # ─── 1.8 Neutro aditivo ──────────────────────────────────────────────────
    @staticmethod
    def _verify_additive_identity(
        cost_matrix: NDArray[np.float64],
    ) -> bool:
        r"""a ⊕ 0 = a  para el vector cero externo del espacio de filas."""
        if cost_matrix.size == 0:
            return True
        zero = np.zeros(cost_matrix.shape[0], dtype=np.float64)
        n_test = min(5, cost_matrix.shape[1])
        for j in range(n_test):
            a = cost_matrix[:, j]
            if not np.allclose(
                a + zero, a, rtol=AlgebraicConstants.RING_TOLERANCE, atol=AlgebraicConstants.RING_TOLERANCE
            ):
                return False
        return True

    # ─── 1.9 Neutro multiplicativo del Hadamard ──────────────────────────────
    @staticmethod
    def _verify_multiplicative_identity(
        cost_matrix: NDArray[np.float64],
    ) -> bool:
        r"""a ⊙ 1 = a  donde 1 es el vector de unos del espacio de filas."""
        if cost_matrix.size == 0:
            return True
        one = np.ones(cost_matrix.shape[0], dtype=np.float64)
        n_test = min(5, cost_matrix.shape[1])
        for j in range(n_test):
            a = cost_matrix[:, j]
            if not np.allclose(
                a * one, a, rtol=AlgebraicConstants.RING_TOLERANCE, atol=AlgebraicConstants.RING_TOLERANCE
            ):
                return False
        return True

    # ─── 1.10 Método terminal de la Fase 1 (puente hacia Fase 2) ─────────────
    @classmethod
    def audit_ring_manifold(
        cls,
        raw_ast: List[Dict[str, Any]],
    ) -> RingHomogeneityValidation:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MÉTODO TERMINAL DE LA FASE 1 / DOMINIO INICIAL DE LA FASE 2
        ═══════════════════════════════════════════════════════════════════════

        Algoritmo:
          1. Subyugación de APUProcessor → cost_matrix.
          2. Validación dimensional (tensor 2D no vacío).
          3. Saneamiento monádico (NaN/Inf ↦ 0).
          4. Perfil espectral (SVD: rango, normas, condición).
          5. Verificación axiomática del anillo ℛ (6 axiomas).
          6. Emisión de RingHomogeneityValidation o detonación de veto.

        El objeto retornado es el argumento canónico de
        Phase2_TopologicalSkeletonAuditor.audit_simplicial_skeleton.
        """
        processor = APUProcessor()
        try:
            cost_matrix = processor.process(raw_ast)
        except Exception as e:
            raise RingSymmetryViolation(
                f"Colapso en la proyección del procesador APU: {e!s}"
            ) from e

        if not isinstance(cost_matrix, np.ndarray):
            raise RingSymmetryViolation(
                f"La matriz de costos no es ndarray; recibido {type(cost_matrix).__name__}."
            )
        if cost_matrix.ndim != 2:
            raise RingSymmetryViolation(
                f"Se exige tensor 2D (vértices × features); ndim={cost_matrix.ndim}."
            )
        if cost_matrix.size == 0 or cost_matrix.shape[0] < AlgebraicConstants.MIN_VERTICES:
            raise EmptyCostTensorError(
                "El tensor de costos es el objeto inicial 0 (variedad vacía o sin vértices)."
            )

        # Coerción a float64
        cost_matrix = np.asarray(cost_matrix, dtype=np.float64)

        # Saneamiento monádico
        sane_matrix, singularities = cls._sanitize_matrix_monadic(cost_matrix)
        if singularities > 0:
            logger.warning(
                "Mónada Option absorbió %d singularidad(es) FPU → 0.", singularities
            )

        # Perfil espectral
        sv, rank, fro, spectral, cond = cls._spectral_profile(sane_matrix)
        if rank == 0:
            raise RingDegeneracyError(
                "Matriz de costos de rango nulo tras saneamiento; "
                "el anillo colapsa al ideal cero."
            )
        if cond > AlgebraicConstants.MAX_CONDITION_NUMBER:
            logger.warning(
                "Condición κ=%.2e supera MAX_CONDITION_NUMBER=%.2e; "
                "inestabilidad numérica posible.",
                cond, AlgebraicConstants.MAX_CONDITION_NUMBER,
            )

        # Verificación axiomática compuesta
        log_parts: List[str] = []
        checks: Dict[str, bool] = {}

        checks["closure"] = cls._verify_additive_closure(sane_matrix)
        log_parts.append(f"Clausura⊕: {'OK' if checks['closure'] else 'FALLA'}")

        checks["commutativity"] = cls._verify_commutativity(sane_matrix)
        log_parts.append(f"Conmutatividad⊕: {'OK' if checks['commutativity'] else 'FALLA'}")

        checks["associativity"] = cls._verify_associativity(sane_matrix)
        log_parts.append(f"Asociatividad⊕: {'OK' if checks['associativity'] else 'FALLA'}")

        checks["distributivity"] = cls._verify_distributivity(sane_matrix)
        log_parts.append(f"Distributividad⊙/⊕: {'OK' if checks['distributivity'] else 'FALLA'}")

        checks["add_id"] = cls._verify_additive_identity(sane_matrix)
        log_parts.append(f"Neutro⊕: {'OK' if checks['add_id'] else 'FALLA'}")

        checks["mul_id"] = cls._verify_multiplicative_identity(sane_matrix)
        log_parts.append(f"Neutro⊙: {'OK' if checks['mul_id'] else 'FALLA'}")

        log_msg = "; ".join(log_parts)
        all_ok = all(checks.values())

        if not all_ok:
            failed = [k for k, v in checks.items() if not v]
            raise RingSymmetryViolation(
                f"Ruptura de simetría del anillo ℛ en axioma(s) {failed}: {log_msg}"
            )

        logger.info(
            "Homogeneidad algebraica confirmada | rank=%d | ‖·‖_F=%.4f | κ=%.2e | "
            "singularidades=%d | %s",
            rank, fro, cond, singularities, log_msg,
        )

        return RingHomogeneityValidation(
            is_closed=checks["closure"],
            is_commutative=checks["commutativity"],
            is_associative=checks["associativity"],
            is_distributive=checks["distributivity"],
            has_additive_identity=checks["add_id"],
            has_multiplicative_identity=checks["mul_id"],
            cost_matrix=sane_matrix,
            monadic_failures=singularities,
            frobenius_norm=fro,
            spectral_norm=spectral,
            condition_number=cond,
            matrix_rank=rank,
            singular_values=sv,
            ring_check_log=log_msg,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: AUDITORÍA HOMOLÓGICA Y ESPECTRAL DEL 1-ESQUELETO
#         (continuación directa del RingHomogeneityValidation de la Fase 1)
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_TopologicalSkeletonAuditor(Phase1_AlgebraicRingAuditor):
    r"""
    Continuación funtorial de la Fase 1. El dominio de sus métodos principales
    es exactamente el codominio de
        Phase1_AlgebraicRingAuditor.audit_ring_manifold
    (objeto RingHomogeneityValidation).

    Construye el grafo de adyacencia inducido por el Gram de costos, la matriz
    de incidencia orientada B₁, el Laplaciano combinatorio L = B₁ B₁ᵀ, extrae
    Spec(L), números de Betti (β₀, β₁) y el valor de Fiedler λ₂, certificando
    conectividad vía el Teorema de Kirchhoff–Fiedler.
    """

    # ─── 2.1 Adyacencia inducida por el producto de Gram ─────────────────────
    @staticmethod
    def _induce_adjacency(
        cost_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Grafo no dirigido G = (V, E) con |V| = n_filas:
          A_{ij} = 1  ⇔  ⟨c_i, c_j⟩ > ε_máquina  (i ≠ j),
        donde c_k es la k-ésima fila del tensor de costos.
        """
        gram = cost_matrix @ cost_matrix.T
        n = gram.shape[0]
        adjacency = np.zeros((n, n), dtype=np.float64)
        # Umbral relativo a la escala del Gram
        scale = max(float(np.max(np.abs(gram))), AlgebraicConstants.MACHINE_EPS)
        thr = AlgebraicConstants.MACHINE_EPS * scale
        for i in range(n):
            for j in range(i + 1, n):
                if abs(gram[i, j]) > thr:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0
        return adjacency

    # ─── 2.2 Matriz de incidencia orientada B₁ ───────────────────────────────
    @staticmethod
    def _build_oriented_incidence(
        adjacency: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], List[Tuple[int, int]]]:
        r"""
        B₁ ∈ Mat_{n×m}(ℝ): para cada arista e = {i,j} con i < j,
          columna_e = e_i − e_j   (es decir +1 en i, −1 en j).

        Retorna (B, edge_list).
        """
        n = adjacency.shape[0]
        edges: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] > AlgebraicConstants.MACHINE_EPS:
                    edges.append((i, j))
        m = len(edges)
        if m == 0:
            return np.zeros((n, 0), dtype=np.float64), edges

        B = np.zeros((n, m), dtype=np.float64)
        for col, (i, j) in enumerate(edges):
            B[i, col] = 1.0
            B[j, col] = -1.0
        return B, edges

    # ─── 2.3 Betti numbers vía rango de B₁ ───────────────────────────────────
    @staticmethod
    def _betti_from_incidence(
        n_vertices: int,
        n_edges: int,
        rank_B: int,
    ) -> Tuple[int, int]:
        r"""
        Fórmulas exactas en dimensión ≤ 1:
          β₀ = |V| − rank(B₁)
          β₁ = |E| − rank(B₁)
        """
        beta0 = max(n_vertices - rank_B, 0)
        beta1 = max(n_edges - rank_B, 0)
        return beta0, beta1

    # ─── 2.4 Espectro del Laplaciano y Fiedler ───────────────────────────────
    @staticmethod
    def _laplacian_spectrum(
        B: NDArray[np.float64],
        n: int,
    ) -> Tuple[NDArray[np.float64], float, int]:
        r"""
        L = B Bᵀ  (n×n, simétrico PSD).
        Spec(L) via eigvalsh (garantiza orden no decreciente y valores reales).
        Retorna (eigenvalues, λ₂, β₀_spectral).
        """
        if B.shape[1] == 0:
            # Sin aristas: L = 0, Spec = {0}^n
            spectrum = np.zeros(n, dtype=np.float64)
            return spectrum, 0.0, n

        L = B @ B.T
        # Simetrización numérica (elimina asimetrías O(ε) de redondeo)
        L = 0.5 * (L + L.T)
        eigenvalues = la.eigvalsh(L)
        # Clamp de valores ligeramente negativos por error de redondeo
        eigenvalues = np.maximum(eigenvalues, 0.0)

        beta0_spec = int(
            np.sum(eigenvalues <= AlgebraicConstants.FIEDLER_TOLERANCE)
        )
        fiedler = float(eigenvalues[1]) if n > 1 else 0.0
        return eigenvalues, fiedler, beta0_spec

    # ─── 2.5 Método principal de la Fase 2 (continuación de Fase 1) ──────────
    @classmethod
    def audit_simplicial_skeleton(
        cls,
        ring: RingHomogeneityValidation,
    ) -> SimplicialSkeletonAudit:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        CONTINUACIÓN DIRECTA DEL MÉTODO TERMINAL DE LA FASE 1
        ═══════════════════════════════════════════════════════════════════════

        Entrada canónica: el RingHomogeneityValidation producido por
            Phase1_AlgebraicRingAuditor.audit_ring_manifold

        Algoritmo:
          1. Guardia anular (anillo debe ser válido).
          2. Inducción del grafo de adyacencia vía Gram.
          3. Construcción de B₁ e edge_list.
          4. rank(B₁), β₀, β₁ por homología simplicial.
          5. Spec(L), λ₂ (Fiedler), β₀ espectral.
          6. Consistencia homología ↔ espectro.
          7. Veto si β₀ ≠ 1 o λ₂ ≤ ε.
          8. Emisión de SimplicialSkeletonAudit.

        El objeto retornado es el dominio de la Fase 3 (AlgebraicTacticsAgent).
        """
        # Guardia: el anillo debe haber pasado todos los axiomas
        if not (
            ring.is_closed
            and ring.is_distributive
            and ring.is_commutative
            and ring.is_associative
            and ring.has_additive_identity
            and ring.has_multiplicative_identity
        ):
            raise RingSymmetryViolation(
                "Manifold anular inválido recibido en Fase 2; "
                "imposible construir el 1-esqueleto."
            )

        cost_matrix = ring.cost_matrix
        n = cost_matrix.shape[0]

        # Caso trivial: un único vértice
        if n <= 1:
            logger.info("1-esqueleto trivial (n≤1): β₀=1, conexo por vacuidad.")
            return SimplicialSkeletonAudit(
                betti_0=1,
                betti_1=0,
                fiedler_value=float("inf"),
                is_connected=True,
                laplacian_spectrum=np.array([0.0], dtype=np.float64),
                incidence_rank=0,
                num_vertices=n,
                num_edges=0,
                euler_characteristic=n,
                adjacency_density=0.0,
                ring_validation=ring,
                homology_log="Trivial: un vértice, H₀=ℤ, H₁=0",
            )

        # Adyacencia inducida
        adjacency = cls._induce_adjacency(cost_matrix)
        B, edges = cls._build_oriented_incidence(adjacency)
        m = len(edges)

        # Rango de incidencia
        if m == 0:
            rank_B = 0
        else:
            rank_B = int(
                np.linalg.matrix_rank(B, tol=AlgebraicConstants.FIEDLER_TOLERANCE)
            )

        # Homología
        beta0_hom, beta1_hom = cls._betti_from_incidence(n, m, rank_B)

        # Espectro del Laplaciano
        spectrum, fiedler, beta0_spec = cls._laplacian_spectrum(B, n)

        # Consistencia β₀ homológico vs espectral
        log_parts: List[str] = []
        if beta0_hom != beta0_spec:
            log_parts.append(
                f"INCONSISTENCIA β₀: homología={beta0_hom}, espectro={beta0_spec}"
            )
            # Preferimos el valor espectral (más robusto numéricamente en PSD)
            # pero reportamos la divergencia
            logger.warning(
                "β₀ homológico (%d) ≠ β₀ espectral (%d); se usa max para veto.",
                beta0_hom, beta0_spec,
            )
            beta0 = max(beta0_hom, beta0_spec)
        else:
            beta0 = beta0_hom
            log_parts.append(f"β₀ consistente={beta0}")

        beta1 = beta1_hom
        chi = n - m
        # Verificación Euler: χ ≟ β₀ − β₁
        if chi != beta0 - beta1:
            log_parts.append(
                f"Euler discrepa: |V|−|E|={chi} ≠ β₀−β₁={beta0 - beta1}"
            )
        else:
            log_parts.append(f"χ={chi}=β₀−β₁")

        # Densidad de adyacencia
        max_edges = n * (n - 1) / 2.0
        density = float(m / max_edges) if max_edges > 0 else 0.0

        is_connected = (
            beta0 == 1 and fiedler > AlgebraicConstants.FIEDLER_TOLERANCE
        ) or (n <= 1)

        log_parts.append(
            f"λ₂={fiedler:.6e} | |V|={n} | |E|={m} | rank(B)={rank_B} | dens={density:.4f}"
        )
        homology_log = " | ".join(log_parts)

        if not is_connected:
            logger.error("Isla topológica: %s", homology_log)
            raise TopologicalIslandError(
                f"Obstrucción topológica: β₀={beta0} (esperado 1), "
                f"λ₂={fiedler:.6e}. Existen {beta0} componente(s) inconexa(s). "
                f"Detalle: {homology_log}"
            )

        logger.info(
            "Auditoría homológica exitosa: β₀=%d, β₁=%d, λ₂=%.6f, rank(B)=%d",
            beta0, beta1, fiedler, rank_B,
        )

        return SimplicialSkeletonAudit(
            betti_0=beta0,
            betti_1=beta1,
            fiedler_value=fiedler,
            is_connected=True,
            laplacian_spectrum=spectrum,
            incidence_rank=rank_B,
            num_vertices=n,
            num_edges=m,
            euler_characteristic=chi,
            adjacency_density=density,
            ring_validation=ring,
            homology_log=homology_log,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 / ORQUESTADOR SUPREMO: ALGEBRAIC TACTICS AGENT
#         (composición funtorial de las Fases 1 y 2)
# ══════════════════════════════════════════════════════════════════════════════
class AlgebraicTacticsAgent(Morphism, Phase2_TopologicalSkeletonAuditor):
    r"""
    Funtor de Proyección Táctica (Endofuntor sobre el estrato TACTICS).

    Composición estricta de morfismos:
        CategoricalState
            ∘  Phase2_TopologicalSkeletonAuditor.audit_simplicial_skeleton
            ∘  Phase1_AlgebraicRingAuditor.audit_ring_manifold

    Gobierna incondicionalmente a apu_processor.py, sellando los datos mediante
    Teoría de Anillos, Homología Simplicial y Teoría Espectral del Laplaciano
    antes de inyectarlos al DAG del PipelineDirector.
    """

    def __init__(
        self,
        fiedler_tolerance: float = AlgebraicConstants.FIEDLER_TOLERANCE,
        ring_tolerance: float = AlgebraicConstants.RING_TOLERANCE,
    ) -> None:
        self._fiedler_tol = float(fiedler_tolerance)
        self._ring_tol = float(ring_tolerance)

    def __call__(self, raw_ast: List[Dict[str, Any]]) -> CategoricalState:
        r"""
        Ejecuta la composición funtorial anidada de las tres fases.

        Fase 1 → RingHomogeneityValidation
        Fase 2 → SimplicialSkeletonAudit   (continuación directa)
        Fase 3 → CategoricalState          (proyección ortogonal al DAG)
        """
        if not isinstance(raw_ast, list):
            raise TypeError(
                f"Se esperaba List[Dict] para raw_ast; recibido {type(raw_ast).__name__}"
            )

        logger.info(
            "Iniciando auditoría algebraica-homológica-espectral "
            "(ε_ring=%.1e, ε_Fiedler=%.1e)…",
            self._ring_tol, self._fiedler_tol,
        )

        # ── Fase 1: Manifold anular y saneamiento monádico ───────────────────
        ring_state: RingHomogeneityValidation = self.audit_ring_manifold(raw_ast)

        # ── Fase 2: Homología simplicial + espectro de Fiedler ───────────────
        # (continuación directa del objeto RingHomogeneityValidation)
        skeleton_state: SimplicialSkeletonAudit = self.audit_simplicial_skeleton(
            ring_state
        )

        # ── Fase 3: Inyección al Estado Categórico (pureza garantizada) ──────
        return CategoricalState(
            stratum=Stratum.TACTICS,
            payload={
                "cost_matrix": ring_state.cost_matrix.tolist(),
                "fiedler_value": skeleton_state.fiedler_value,
                "betti_0": skeleton_state.betti_0,
                "betti_1": skeleton_state.betti_1,
                "incidence_rank": skeleton_state.incidence_rank,
                "num_vertices": skeleton_state.num_vertices,
                "num_edges": skeleton_state.num_edges,
                "euler_characteristic": skeleton_state.euler_characteristic,
                "adjacency_density": skeleton_state.adjacency_density,
                "laplacian_spectrum": skeleton_state.laplacian_spectrum.tolist(),
                "frobenius_norm": ring_state.frobenius_norm,
                "spectral_norm": ring_state.spectral_norm,
                "condition_number": ring_state.condition_number,
                "matrix_rank": ring_state.matrix_rank,
                "singular_values": ring_state.singular_values.tolist(),
                "ring_check_log": ring_state.ring_check_log,
                "homology_log": skeleton_state.homology_log,
            },
            context={
                "is_closed": ring_state.is_closed,
                "is_commutative": ring_state.is_commutative,
                "is_associative": ring_state.is_associative,
                "is_distributive": ring_state.is_distributive,
                "has_additive_identity": ring_state.has_additive_identity,
                "has_multiplicative_identity": ring_state.has_multiplicative_identity,
                "is_connected": skeleton_state.is_connected,
                "monadic_failures_absorbed": ring_state.monadic_failures,
                "fiedler_tolerance_applied": self._fiedler_tol,
                "ring_tolerance_applied": self._ring_tol,
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "AlgebraicConstants",
    "OptionMonad",
    "AlgebraicTacticsError",
    "RingSymmetryViolation",
    "RingDegeneracyError",
    "TopologicalIslandError",
    "HomologicalInvariantError",
    "EmptyCostTensorError",
    "RingHomogeneityValidation",
    "SimplicialSkeletonAudit",
    "Phase1_AlgebraicRingAuditor",
    "Phase2_TopologicalSkeletonAuditor",
    "AlgebraicTacticsAgent",
]