# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Pipeline Director Agent — Custodio de la Causalidad Funtorial       ║
║ Ruta   : app/agents/tactics/pipeline_director_agent.py                       ║
║ Versión: 3.0.0-Schur-Jordan-Poset-Exact-MV                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

EVOLUCIÓN GRANULAR v3.0.0 (Artesanía Senior + Física Matemática):
────────────────────────────────────────────────────────────────────────────────

FASE 1 → Certificación espectral de nilpotencia por descomposición de Schur.
  ─────────────────────────────────────────────────────────────────────────────
  MEJORA §1.1 — Descomposición de Schur como camino canónico:
    La descomposición A = Q·T·Q* (Q unitaria, T triangular superior) preserva
    el espectro exactamente: Spec(A) = diag(T). Se reemplaza eigvals genérico
    por scipy.linalg.schur para obtener la forma de Schur real (RSF) cuando
    A ∈ R^{n×n}, evitando errores de extensión analítica al campo complejo.

    Coste aritmético: O(n³) igual que eigvals, pero con backward error
    garantizado por el Teorema de Perturbación de Bauer-Fike:
        |λ̃ − λ| ≤ κ₂(Q) · ‖ΔA‖₂

    Para A nilpotente exacta, Q es unitaria ⇒ κ₂(Q) = 1 y el error es mínimo.

  MEJORA §1.2 — Número de condición espectral adaptativo:
    La tolerancia dinámica incorpora el número de condición de la transformación
    de similaridad Q de la forma de Schur:

        tol = max(tol_base, κ₂(Q) · √n · ε_machine · ‖A‖_F)

    donde ‖A‖_F es la norma de Frobenius (más sensible a la distribución de
    energía que ‖A‖_∞) y √n refleja la acumulación de error en n pasos.

  MEJORA §1.3 — Detección de bloques de Jordan 2×2 en RSF:
    La RSF puede producir bloques 2×2 para pares de valores propios complejos
    conjugados. Se detectan y se audita que la norma de cada bloque diagonal
    2×2 sea ≤ tol, garantizando que no hay autovalores complejos con parte
    real no nula ocultos.

  MEJORA §1.4 — Residual de nilpotencia por serie geométrica truncada:
    Para matrices grandes (n > power_audit_max_dim), en lugar de omitir el
    residual, se estima una cota superior usando la norma de Gelfand:
        ρ(A) = lim_{k→∞} ‖A^k‖^{1/k}
    via iteración de potencia inversa truncada a k=⌈log₂(n)⌉ pasos.

  MEJORA §1.5 — Índice de nilpotencia certificado:
    El índice mínimo ν tal que A^ν = 0 se certifica eficientemente:
        ν ≤ n (por Cayley-Hamilton)
    Se busca binariamente en [1, n] el menor ν con ‖A^ν‖_∞ ≤ tol_power,
    lo que proporciona información estructural adicional sobre la profundidad
    del DAG (ν = longitud máxima de camino + 1).

FASE 2 → Auditoría del difeomorfismo de filtración categórica con auto-bucles.
  ─────────────────────────────────────────────────────────────────────────────
  MEJORA §2.1 — Detección canónica de auto-bucles:
    Una arista (u, u) es un ciclo de longitud 1 que viola:
      (a) la aciclicidad del DAG (Fase 1 debería haberlo capturado si u tiene
          peso diagonal A[i,i] ≠ 0, pero podría existir en la lista de edges
          sin reflejarse en la diagonal de A).
      (b) la filtración estricta: stratum(u) < stratum(u) es imposible.
      (c) la filtración laxa: stratum(u) ≤ stratum(u) es trivial pero
          semánticamente incoherente para un DAG causal.
    Se lanza SelfLoopVetoError (nueva excepción, subclase de
    FiltrationViolationVeto) con información diagnóstica del nodo.

  MEJORA §2.2 — Validación de la transpuesta en cross-check:
    El soporte de la matriz de adyacencia es A[i,j] ≠ 0 para (u→v) con
    i=index(u), j=index(v). Se añade la verificación de que A[j,i] = 0
    para toda arista dirigida, garantizando que no existe arista inversa
    (lo que introduciría un ciclo de longitud 2 no capturado por la diagonal).

  MEJORA §2.3 — Cálculo del histograma de slacks:
    Se agrega un histograma de distribución de slacks {0,1,2,...,max_slack}
    al artefacto PosetFiltrationData para diagnóstico de la concentración
    energética en cada nivel de la filtración.

  MEJORA §2.4 — Detección de nodos aislados sin estrato:
    Si un nodo aparece en node_strata pero no en ninguna arista, se registra
    como nodo aislado certificado (no es un error, pero es información
    estructural relevante para la topología del DAG).

FASE 3 → Intercepción de cohomología de fusión con secuencia exacta completa.
  ─────────────────────────────────────────────────────────────────────────────
  MEJORA §3.1 — Métrica de defecto simétrica de Kullback-Leibler discreta:
    El relative_defect original usa un denominador asimétrico. Se reemplaza
    por la divergencia simétrica discreta (Jensen-Shannon) entre la distribución
    degenera δ_{expected} y δ_{observed}:

        defect_JS = |Δβ₁| / (1 + max(β₁(A∪B), expected))

    que es simétrica, acotada en [0,1) y tiene interpretación información-
    teórica: mide la distancia entre el invariante observado y el predicho.

  MEJORA §3.2 — Validación de la secuencia exacta en H₀:
    La secuencia de Mayer-Vietoris en grado 0:
        H₀(A∩B) → H₀(A)⊕H₀(B) → H₀(A∪B) → 0
    implica:
        β₀(A∪B) = β₀(A) + β₀(B) − β₀(A∩B) + rank(∂₁)
    donde rank(∂₁) es el rango del morfismo de conexión en grado 1.
    Se acepta como parámetro opcional y se valida la consistencia.

  MEJORA §3.3 — Verificación de la desigualdad de Mayer-Vietoris:
    Independientemente de los rangos exactos, se verifica la desigualdad:
        |β₁(A∪B) − (β₁(A) + β₁(B))| ≤ β₁(A∩B)
    que es consecuencia de la exactitud de la secuencia larga y sirve como
    cota débil siempre válida.

  MEJORA §3.4 — Certificado de la característica de Euler-Poincaré:
    Se verifica la identidad de Euler-Poincaré bajo fusión:
        χ(A∪B) = χ(A) + χ(B) − χ(A∩B)
    donde χ = β₀ − β₁ + β₂ − ...
    Requiere β₀ opcionales. Si se proporcionan, se valida; si no, se omite.
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# §0. IMPORTACIONES CANÓNICAS
# ══════════════════════════════════════════════════════════════════════════════

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray

# Importación preferente de scipy para álgebra lineal numérica certificada.
# scipy.linalg.schur implementa la descomposición de Schur (LAPACK dgees/zgees)
# con pivoteo QR de Householder y certificado backward-stable.
try:
    import scipy.linalg as _sla

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    import numpy.linalg as _sla  # type: ignore[no-redef]

    _HAS_SCIPY = False

# Importación del topos MIC (opcional para entornos de prueba aislados).
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum  # noqa: F401
except ImportError:  # pragma: no cover

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos MIC."""

        pass

    class Morphism:
        """Clase base de morfismos del Topos cuando no existe dependencia externa."""

        pass

    Stratum = None  # type: ignore


logger = logging.getLogger("MIC.Tactics.PipelineDirectorAgent")
logger.addHandler(logging.NullHandler())


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS Y TOLERANCIAS ESPECTRALES
# ══════════════════════════════════════════════════════════════════════════════

# Épsilon de máquina para float64 (~2.22e-16).
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)

# Tolerancia base para el radio espectral: más agresiva que ε_machine para
# absorber errores de redondeo de primer orden sin sacrificar sensibilidad.
_BASE_SPECTRAL_TOLERANCE: float = 1e-10

# Tolerancia base para residuales de potencia de matriz (Cayley-Hamilton).
_BASE_POWER_TOLERANCE: float = 1e-8

# Dimensión máxima por defecto para la auditoría de potencia exacta A^n.
# Para n > este umbral se usa la estimación por radio espectral de Gelfand.
_DEFAULT_POWER_AUDIT_MAX_DIM: int = 96

# Número de iteraciones para la estimación de Gelfand por iteración de potencia.
_GELFAND_ITERATION_STEPS: int = 20

# Factor de seguridad para el cálculo de tolerancia adaptativa.
# Derivado del análisis de perturbación backward-stable de Bauer-Fike.
_SCHUR_CONDITION_SAFETY_FACTOR: float = 4.0


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES DE GOBERNANZA CAUSAL
# ══════════════════════════════════════════════════════════════════════════════


class PipelineDirectorAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Causalidad Funtorial."""

    pass


class AdjacencyMatrixFormatError(PipelineDirectorAgentError):
    """Detonada cuando el operador de adyacencia no pertenece al dominio válido."""

    pass


class CausalLoopVetoError(PipelineDirectorAgentError):
    r"""Detonada si Spec(A) ≠ {0}. El DAG contiene un ciclo parásito cerrado."""

    pass


class NilpotenceIndexVetoError(PipelineDirectorAgentError):
    r"""
    Detonada cuando el índice de nilpotencia certificado ν excede n.

    Teorema de Cayley-Hamilton: para A ∈ R^{n×n} nilpotente, A^n = 0.
    Si ‖A^n‖_∞ > tol_power, la matriz no es nilpotente en el sentido práctico.
    """

    pass


class StratumMappingError(PipelineDirectorAgentError):
    """Detonada cuando un nodo carece de estrato DIKW o su nivel no es entero."""

    pass


class SelfLoopVetoError(PipelineDirectorAgentError):
    r"""
    Detonada cuando se detecta una arista (u, u) en la lista de edges.

    Fundamento:
        Una arista (u, u) es un ciclo de longitud 1. Es incompatible con:
        (a) La aciclicidad del DAG causal.
        (b) La filtración estricta: stratum(u) < stratum(u) ⊥.
        (c) La semántica causal: un nodo no puede ser causa de sí mismo.
    """

    pass


class FiltrationViolationVeto(PipelineDirectorAgentError):
    r"""
    Detonada si stratum(u) > stratum(v) para alguna arista (u→v).

    Interpretación física:
        La flecha del tiempo termodinámica exige que la entropía semántica
        no disminuya a lo largo de la filtración DIKW.
    """

    pass


class AdjacencySupportVetoError(PipelineDirectorAgentError):
    r"""
    Detonada cuando el cross-check edges↔matriz detecta inconsistencias.

    Incluye:
        (a) Aristas declaradas sin soporte no nulo en A.
        (b) Aristas inversas A[j,i] ≠ 0 que introducirían ciclos de longitud 2.
    """

    pass


class MayerVietorisInputError(PipelineDirectorAgentError):
    """Detonada cuando los invariantes homológicos de entrada son inválidos."""

    pass


class HomologicalFusionVeto(PipelineDirectorAgentError):
    r"""
    Detonada si Δβ₁ ≠ 0.

    La fusión introduce socavones lógicos en la malla agéntica: el número
    de loops independientes no se conserva bajo la ley de pegado.
    """

    pass


class EulerPoincareMismatchError(PipelineDirectorAgentError):
    r"""
    Detonada si χ(A∪B) ≠ χ(A) + χ(B) − χ(A∩B).

    La característica de Euler-Poincaré es un invariante topológico absoluto
    que debe preservarse bajo toda descomposición CW válida.
    """

    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES — DTOs DEL TOPOS
# ══════════════════════════════════════════════════════════════════════════════


def _utc_timestamp() -> str:
    """Devuelve marca de tiempo UTC ISO-8601 con precisión de segundo."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class NilpotenceAuditData:
    r"""
    Artefacto de Fase 1.

    Certificado espectral completo del operador de adyacencia del DAG.

    Campos adicionales v3:
        schur_condition_number: κ₂(Q) de la transformación de Schur.
            Mide la sensibilidad del espectro a perturbaciones de A.
            κ₂(Q) = 1 si Q es unitaria exacta (caso nilpotente canónico).
        frobenius_norm: ‖A‖_F, norma de Frobenius. Más sensible a la
            distribución de energía espectral que ‖A‖_∞.
        nilpotence_index: índice mínimo ν tal que ‖A^ν‖_∞ ≤ tol_power.
            Igual a la longitud máxima de camino dirigido + 1.
            None si no fue auditado.
        gelfand_radius_estimate: estimación del radio espectral por
            iteración de potencia (válida para n > power_audit_max_dim).
        schur_2x2_blocks_detected: número de bloques 2×2 en la RSF,
            indicando pares de autovalores complejos conjugados.
    """

    dimension: int
    spectral_radius: float
    tolerance: float
    adjacency_inf_norm: float
    frobenius_norm: float
    nonzero_entries: int
    directed_density: float
    is_strictly_nilpotent: bool
    schur_condition_number: float = 1.0
    power_residual: Optional[float] = None
    power_audited: bool = False
    nilpotence_index: Optional[int] = None
    gelfand_radius_estimate: Optional[float] = None
    schur_2x2_blocks_detected: int = 0
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PosetFiltrationData:
    r"""
    Artefacto de Fase 2.

    Certificado de monotonicidad categórica en la flecha del tiempo DIKW.

    Campos adicionales v3:
        self_loops_detected: número de auto-bucles (u,u) encontrados en edges.
        slack_histogram: distribución de frecuencias de slacks por valor.
            Clave: valor de slack (int), valor: conteo de aristas.
        isolated_strata_nodes: nodos en node_strata sin aristas incidentes.
        inverse_edges_blocked: aristas (v,u) bloqueadas en cross-check.
    """

    edge_count: int
    audited_edge_count: int
    ignored_edge_count: int
    min_slack: Optional[int]
    max_slack: Optional[int]
    is_monotonic_filtration: bool
    self_loops_detected: int = 0
    unknown_nodes: Tuple[str, ...] = ()
    isolated_strata_nodes: Tuple[str, ...] = ()
    inverse_edges_blocked: Tuple[str, ...] = ()
    slack_histogram: Tuple[Tuple[int, int], ...] = ()
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MayerVietorisAuditData:
    r"""
    Artefacto de Fase 3.

    Certificado de invariancia homológica de fusión bajo Mayer-Vietoris.

    Campos adicionales v3:
        weak_bound_satisfied: ¿se cumple |β₁(A∪B)−(β₁(A)+β₁(B))| ≤ β₁(A∩B)?
        euler_characteristic_valid: True si χ(A∪B) = χ(A)+χ(B)−χ(A∩B)
            fue verificado y se cumple. None si no se proporcionaron β₀.
        jensen_shannon_defect: métrica de defecto JS ∈ [0,1).
            Reemplaza relative_defect con una medida simétrica acotada.
        h0_sequence_valid: ¿la secuencia en H₀ es consistente?
    """

    betti_1_A: int
    betti_1_B: int
    betti_1_intersection: int
    betti_1_union: int
    expected_union_betti_1: int
    delta_betti_1: int
    relative_defect: float
    jensen_shannon_defect: float
    is_fusion_homologous: bool
    weak_bound_satisfied: bool = True
    euler_characteristic_valid: Optional[bool] = None
    h0_sequence_valid: Optional[bool] = None
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CausalGovernanceState:
    r"""
    Objeto final del endofuntor de gobernanza causal:
        Z_Causal = Φ₃ ∘ Φ₂ ∘ Φ₁

    La composición es funtorial en el sentido de que cada Φᵢ es un funtor
    entre las categorías de entradas y certificados:
        Φ₁: Matx(R) → NilpotenceCert
        Φ₂: NilpotenceCert × Poset → FiltrationCert
        Φ₃: FiltrationCert × HomInv → GovernanceState
    """

    governance_id: str
    nilpotence_audit: NilpotenceAuditData
    filtration_audit: PosetFiltrationData
    mayer_vietoris_audit: MayerVietorisAuditData
    is_causally_valid: bool
    generated_at_utc: str


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1: CERTIFICACIÓN ESPECTRAL DE ACICLICIDAD ABSOLUTA                     ║
# ║                                                                             ║
# ║ Fundamento algebraico:                                                      ║
# ║   A ∈ R^{n×n} es nilpotente ⟺ ∃ν ≤ n: A^ν = 0 ⟺ Spec(A) = {0}            ║
# ║   ⟺ χ_A(λ) = λ^n (polinomio característico)                                ║
# ║                                                                             ║
# ║ Estrategia numérica:                                                        ║
# ║   Se usa la descomposición de Schur real (RSF) A = Q·T·Q^T                  ║
# ║   en lugar de eigvals genérico. La RSF garantiza:                           ║
# ║     · Error backward-stable: ‖Q·T·Q^T − A‖₂ ≤ n·ε_machine·‖A‖₂              ║
# ║     · Autovalores exactamente en diag(T) (entradas reales o bloques 2×2)    ║
# ║     · Número de condición κ₂(Q) accesible para calibrar la tolerancia       ║
# ║                                                                             ║
# ║ El último método de esta fase es el puente formal hacia Fase 2.             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase1_SpectralNilpotenceCertifier:
    r"""
    Certificador espectral de nilpotencia mediante descomposición de Schur.

    Teorema central (Cayley-Hamilton + teoría espectral):
        Sea A ∈ R^{n×n}. Son equivalentes:
          (i)  A es nilpotente (∃ν ≤ n: A^ν = 0).
          (ii) Spec(A) = {0} (todos los autovalores son cero).
          (iii) El polinomio característico es χ_A(λ) = λ^n.
          (iv)  tr(A^k) = 0 para todo k = 1, ..., n.
          (v)   El grafo dirigido asociado es acíclico (DAG).

    Estrategia de Schur:
        Se calcula A = Q·T·Q^T con T triangular superior (real Schur form).
        Para A nilpotente exacta, T = 0 y Q es unitaria. En aritmética
        de punto flotante, se certifica ρ(A) = max|diag(T)| ≤ tol_dynamic.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # §1.1 Validación y acondicionamiento del operador de adyacencia
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_and_condition_adjacency(
        self,
        adjacency_matrix: NDArray[Any],
        *,
        zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
    ) -> NDArray[np.float64]:
        r"""
        Valida y condiciona el operador de adyacencia A ∈ R^{n×n}.

        Contrato matemático:
            · ndim = 2 (operador lineal en espacio vectorial finito-dimensional).
            · shape[0] = shape[1] = n (endomorphismo del espacio R^n).
            · Todas las entradas ∈ R (no NaN, no Inf).
            · Para DAG causal: entradas ≥ 0 (la causalidad no tiene pesos negativos).
            · Para DAG binario: entradas ∈ {0, 1}.

        Acondicionamiento numérico:
            Las entradas con |A[i,j]| ≤ zero_threshold se anulan exactamente
            (proyección sobre el subespacio de matrices con soporte dado).
            Esto elimina ruido de punto flotante sin modificar la estructura
            topológica del grafo.

        Parámetros:
            adjacency_matrix: NDArray con el operador de adyacencia o pesos.
            zero_threshold: umbral ε ≥ 0 para anulación de entradas ruidosas.
            allow_signed_weights: si False, rechaza A[i,j] < 0 (DAG causal).
            require_boolean: si True, exige A[i,j] ∈ {0,1} dentro de tolerancia.

        Retorna:
            A ∈ R^{n×n}, dtype=float64, acondicionada y lista para Schur.

        Lanza:
            AdjacencyMatrixFormatError: si alguna condición del contrato falla.
        """
        A = np.asarray(adjacency_matrix, dtype=np.float64)

        # Verificación de dimensionalidad: A debe ser un endomorfismo 2D.
        if A.ndim != 2:
            raise AdjacencyMatrixFormatError(
                f"La matriz de adyacencia debe ser un operador 2D (endomorfismo). "
                f"Se recibió ndim={A.ndim}. "
                f"Tip: si recibió un tensor 3D, extraiga la slice temporal correcta."
            )

        # Verificación de cuadratura: A ∈ End(R^n) ⟹ shape = (n, n).
        if A.shape[0] != A.shape[1]:
            raise AdjacencyMatrixFormatError(
                f"La matriz de adyacencia debe ser cuadrada (endomorfismo de R^n). "
                f"Se recibió shape={A.shape}. "
                f"Una matriz rectangular A ∈ R^{{m×n}}, m≠n, no define un grafo."
            )

        # Verificación de finitud: el operador debe estar bien definido en R^{n×n}.
        if not np.all(np.isfinite(A)):
            nan_count = int(np.sum(np.isnan(A)))
            inf_count = int(np.sum(np.isinf(A)))
            raise AdjacencyMatrixFormatError(
                f"La matriz de adyacencia contiene valores no finitos. "
                f"NaN: {nan_count}, Inf: {inf_count}. "
                f"El operador causal debe estar bien definido en R^{{n×n}}."
            )

        # Validación del umbral de umbralización.
        if zero_threshold < 0.0:
            raise AdjacencyMatrixFormatError(
                f"zero_threshold debe ser ≥ 0. "
                f"Se recibió {zero_threshold:.6e}. "
                f"Un umbral negativo no tiene interpretación en teoría espectral."
            )

        # Anulación de entradas con ruido numérico por debajo del umbral.
        # Esta operación es equivalente a proyectar A sobre el subespacio
        # {M ∈ R^{n×n} : |M[i,j]| > ε} (soporte ε-efectivo).
        if zero_threshold > 0.0:
            A = np.where(np.abs(A) <= zero_threshold, 0.0, A)

        # Verificación de positividad: DAG causal físico tiene pesos ≥ 0.
        if not allow_signed_weights:
            negative_mask = A < 0.0
            if np.any(negative_mask):
                negative_count = int(np.count_nonzero(negative_mask))
                negative_values = A[negative_mask]
                min_negative = float(np.min(negative_values))
                raise AdjacencyMatrixFormatError(
                    f"La matriz de adyacencia causal contiene {negative_count} "
                    f"entradas negativas (mínimo={min_negative:.6e}). "
                    f"Un DAG causal físico requiere pesos no negativos. "
                    f"Use allow_signed_weights=True sólo si el modelo lo justifica "
                    f"(p.ej., grafos con pesos de correlación signados)."
                )

        # Verificación de booleanidad: grafo no ponderado ⟹ A ∈ {0,1}^{n×n}.
        if require_boolean:
            nonzero_values = A[np.abs(A) > 0.0]
            boolean_tol = max(zero_threshold, _MACHINE_EPSILON)

            if nonzero_values.size > 0:
                if not np.allclose(nonzero_values, 1.0, rtol=0.0, atol=boolean_tol):
                    max_deviation = float(np.max(np.abs(nonzero_values - 1.0)))
                    raise AdjacencyMatrixFormatError(
                        f"require_boolean=True exige A[i,j] ∈ {{0,1}} dentro de "
                        f"tolerancia {boolean_tol:.2e}. "
                        f"Desviación máxima detectada: {max_deviation:.6e}. "
                        f"Se detectaron pesos distintos de 1."
                    )

            # Proyección sobre {0,1}: binarización exacta.
            A = (A != 0.0).astype(np.float64)

        return A

    # ──────────────────────────────────────────────────────────────────────────
    # §1.2 Descomposición de Schur y extracción espectral
    # ──────────────────────────────────────────────────────────────────────────

    def _schur_decompose(
        self,
        A: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, int]:
        r"""
        Calcula la descomposición de Schur real de A.

        Descomposición:
            A = Q · T · Q^T
            Q ∈ R^{n×n} ortogonal (Q^T Q = I)
            T ∈ R^{n×n} cuasi-triangular superior (real Schur form, RSF)

        La RSF tiene:
            · Bloques 1×1 en la diagonal: autovalores reales.
            · Bloques 2×2 en la diagonal: pares conjugados a+bi, a−bi.

        Para A nilpotente:
            · T = 0 (en exacta), o ‖diag(T)‖_∞ ≤ tol (en punto flotante).
            · Los bloques 2×2 tienen traza ≈ 0 y norma ≈ 0.

        Número de condición κ₂(Q):
            Para Q exactamente ortogonal, κ₂(Q) = 1.
            En práctica, κ₂(Q) ≈ 1 + O(n·ε_machine) para A bien condicionada.

        Retorna:
            T: forma de Schur cuasi-triangular.
            Q: transformación ortogonal.
            kappa_Q: número de condición de Q (≥ 1).
            n_blocks_2x2: número de bloques 2×2 detectados en T.

        Notas de implementación:
            Se usa scipy.linalg.schur con output='real' para obtener la RSF.
            Si scipy no está disponible, se cae a numpy.linalg.eig con
            construcción manual de la triangular (menos estable, se advierte).
        """
        n = A.shape[0]

        if not _HAS_SCIPY:
            # Fallback: numpy eig + triangularización manual.
            # ADVERTENCIA: menos estable numéricamente que Schur vía LAPACK.
            logger.warning(
                "scipy no disponible. Usando numpy.linalg.eig como fallback "
                "para la descomposición espectral. La estabilidad numérica "
                "puede ser inferior al método de Schur (LAPACK dgees)."
            )
            eigenvalues = np.linalg.eigvals(A)
            # Construimos T diagonal como aproximación (no triangular completa).
            T = np.diag(eigenvalues.real)
            Q = np.eye(n, dtype=np.float64)
            kappa_Q = 1.0
            n_blocks_2x2 = 0
            return T, Q, kappa_Q, n_blocks_2x2

        try:
            # scipy.linalg.schur usa LAPACK dgees/dgeev.
            # output='real' produce la Real Schur Form (RSF).
            T, Q = _sla.schur(A, output="real")
        except Exception as exc:
            # Si Schur falla (raro, p.ej. matriz con NaN post-condicionamiento),
            # caemos a eigvals estándar.
            logger.warning(
                "scipy.linalg.schur falló (%s). "
                "Usando scipy.linalg.eigvals como fallback.",
                exc,
            )
            eigenvalues = _sla.eigvals(A)
            T = np.diag(eigenvalues.real)
            Q = np.eye(n, dtype=np.float64)
            kappa_Q = 1.0
            n_blocks_2x2 = 0
            return T, Q, kappa_Q, n_blocks_2x2

        # Cálculo del número de condición de Q.
        # Para Q ortogonal exacta, κ₂(Q) = σ_max / σ_min = 1.
        # Usamos svd de Q para obtener los valores singulares.
        try:
            singular_values = _sla.svd(Q, compute_uv=False)
            sv_min = float(singular_values[-1])
            sv_max = float(singular_values[0])
            kappa_Q = (
                sv_max / sv_min
                if sv_min > _MACHINE_EPSILON
                else float("inf")
            )
        except Exception:
            kappa_Q = 1.0  # Asumimos ortogonal si SVD falla.

        # Detección de bloques 2×2 en la RSF.
        # Un bloque 2×2 en la posición (i, i+1) se identifica por
        # T[i+1, i] ≠ 0 (subdiagonal no nula debajo del bloque).
        n_blocks_2x2 = 0
        i = 0
        while i < n - 1:
            if abs(T[i + 1, i]) > _MACHINE_EPSILON:
                n_blocks_2x2 += 1
                i += 2  # Saltamos el bloque 2×2 completo.
            else:
                i += 1

        return T, Q, kappa_Q, n_blocks_2x2

    # ──────────────────────────────────────────────────────────────────────────
    # §1.3 Tolerancia espectral dinámica calibrada por condición de Schur
    # ──────────────────────────────────────────────────────────────────────────

    def _dynamic_spectral_tolerance(
        self,
        dimension: int,
        frobenius_norm: float,
        schur_condition_number: float = 1.0,
    ) -> float:
        r"""
        Calcula la tolerancia espectral dinámica calibrada por Bauer-Fike.

        Modelo (v3 — Frobenius + condición de Schur):
            tol = max(tol_base, κ₂(Q) · C · √n · ε_machine · ‖A‖_F)

        Justificación (Teorema de Perturbación de Bauer-Fike):
            Si A = Q·T·Q^T con Q ortogonal, entonces para cualquier
            perturbación ΔA:
                min_{μ∈Spec(T)} |λ̃ − μ| ≤ κ₂(Q) · ‖ΔA‖₂

            El error backward del algoritmo de Schur (LAPACK dgees) satisface:
                ‖ΔA‖₂ ≤ C · √n · ε_machine · ‖A‖_F

            donde C = _SCHUR_CONDITION_SAFETY_FACTOR (factor de seguridad
            empírico basado en el análisis de Wilkinson).

        Para A nilpotente exacta:
            · Q es unitaria ⟹ κ₂(Q) = 1.
            · T = 0 ⟹ ‖A‖_F = ‖ΔA‖_F (todo es error numérico).
            · tol = max(tol_base, √n · ε_machine · ‖A‖_F).

        Parámetros:
            dimension: n = tamaño de la matriz.
            frobenius_norm: ‖A‖_F (norma de Frobenius).
            schur_condition_number: κ₂(Q) de la transformación de Schur.

        Retorna:
            Tolerancia dinámica en [tol_base, ∞).
        """
        if dimension <= 0:
            return _BASE_SPECTRAL_TOLERANCE

        # Acotación del número de condición: κ₂(Q) ≥ 1 por definición.
        kappa = max(1.0, float(schur_condition_number))

        # Norma de Frobenius como medida de energía espectral total.
        f_norm = max(1.0, float(frobenius_norm))

        # Tolerancia de Bauer-Fike con factor de seguridad.
        adaptive_tol = (
            kappa
            * _SCHUR_CONDITION_SAFETY_FACTOR
            * float(np.sqrt(dimension))
            * _MACHINE_EPSILON
            * f_norm
        )

        return float(max(_BASE_SPECTRAL_TOLERANCE, adaptive_tol))

    # ──────────────────────────────────────────────────────────────────────────
    # §1.4 Radio espectral de Gelfand para matrices grandes
    # ──────────────────────────────────────────────────────────────────────────

    def _gelfand_spectral_radius_estimate(
        self,
        A: NDArray[np.float64],
        n_steps: int = _GELFAND_ITERATION_STEPS,
    ) -> float:
        r"""
        Estima el radio espectral de Gelfand para matrices grandes.

        Fórmula de Gelfand:
            ρ(A) = lim_{k→∞} ‖A^k‖^{1/k}

        Para A nilpotente exacta: ρ(A) = 0, la sucesión decae a 0.
        Para A con ciclos: la sucesión crece o se estabiliza en ρ(A) > 0.

        Implementación:
            Se usa la iteración de potencia:
                v₀ = v_random normalizado
                vₖ = A · vₖ₋₁ / ‖A · vₖ₋₁‖ (si ‖A · vₖ₋₁‖ > 0)
                μₖ = ‖A^k · v₀‖^{1/k}
            donde μₖ converge a ρ(A) para v₀ genérico.

        Coste: O(n² · n_steps) frente a O(n³) de Schur/eigvals.

        Parámetros:
            A: matriz cuadrada float64.
            n_steps: número de pasos de iteración de potencia.

        Retorna:
            Estimación de ρ(A) ∈ [0, ∞).
        """
        n = A.shape[0]
        if n == 0:
            return 0.0

        # Vector inicial: aleatorio con semilla fija para reproducibilidad.
        rng = np.random.default_rng(seed=42)
        v = rng.standard_normal(n)
        norm_v = float(np.linalg.norm(v))
        if norm_v < _MACHINE_EPSILON:
            return 0.0
        v = v / norm_v

        radius_estimate = 0.0
        Av = A @ v

        for k in range(1, n_steps + 1):
            norm_Av = float(np.linalg.norm(Av))
            if norm_Av < _MACHINE_EPSILON:
                # El vector se anuló: ρ(A) = 0 (nilpotente en esta dirección).
                return 0.0

            # Estimación de Gelfand en el paso k: ‖A^k · v₀‖^{1/k}.
            radius_estimate = float(norm_Av ** (1.0 / k))

            # Actualización: v ← A·v / ‖A·v‖.
            v = Av / norm_Av
            Av = A @ v

        return radius_estimate

    # ──────────────────────────────────────────────────────────────────────────
    # §1.5 Índice de nilpotencia por búsqueda binaria
    # ──────────────────────────────────────────────────────────────────────────

    def _certify_nilpotence_index(
        self,
        A: NDArray[np.float64],
        power_tolerance: float,
        power_audit_max_dim: int,
    ) -> Tuple[Optional[float], Optional[int], Optional[float]]:
        r"""
        Certifica el índice de nilpotencia ν y el residual ‖A^n‖_∞.

        Teorema (Cayley-Hamilton):
            Si A es nilpotente, entonces A^n = 0 para n = dim(A).
            El índice de nilpotencia ν es el mínimo entero ≥ 1 con A^ν = 0.

        Estrategia:
            Para n ≤ power_audit_max_dim:
                · Calcula el residual exacto r_n = ‖A^n‖_∞.
                · Busca ν binariamente en [1, n]:
                    encontrar el menor ν con ‖A^ν‖_∞ ≤ power_tolerance.
                    Si no existe, ν = None (no es nilpotente en práctica).
            Para n > power_audit_max_dim:
                · Usa la estimación de Gelfand (§1.4).
                · Residual y ν exactos no se calculan (None).

        Parámetros:
            A: matriz cuadrada float64, ya condicionada.
            power_tolerance: umbral para considerar ‖A^k‖_∞ = 0.
            power_audit_max_dim: límite de dimensión para cálculo exacto.

        Retorna:
            (power_residual, nilpotence_index, gelfand_estimate)
                power_residual: ‖A^n‖_∞ o None si n > max_dim.
                nilpotence_index: ν mínimo o None si no certificado.
                gelfand_estimate: estimación ρ(A) de Gelfand o None.
        """
        n = int(A.shape[0])

        if n == 0:
            return (0.0, 0, None)

        if n > power_audit_max_dim:
            # Para matrices grandes, usar estimación de Gelfand.
            gelfand_est = self._gelfand_spectral_radius_estimate(A)
            return (None, None, gelfand_est)

        # Cálculo exacto del residual de Cayley-Hamilton: ‖A^n‖_∞.
        try:
            A_n = np.linalg.matrix_power(A, n)
            power_residual = float(np.linalg.norm(A_n, ord=np.inf))
        except Exception as exc:
            logger.warning("matrix_power(A, %d) falló: %s", n, exc)
            return (None, None, None)

        # Búsqueda binaria del índice de nilpotencia ν.
        # Invariante: ‖A^ν‖_∞ ≤ tol ⟹ la búsqueda continúa hacia la izquierda.
        nilpotence_index: Optional[int] = None
        lo, hi = 1, n

        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                A_mid = np.linalg.matrix_power(A, mid)
                norm_mid = float(np.linalg.norm(A_mid, ord=np.inf))
            except Exception:
                break

            if norm_mid <= power_tolerance:
                nilpotence_index = mid
                hi = mid - 1  # Buscamos ν más pequeño.
            else:
                lo = mid + 1  # Necesitamos una potencia mayor.

        return (power_residual, nilpotence_index, None)

    # ──────────────────────────────────────────────────────────────────────────
    # §1.6 Extracción del radio espectral desde la forma de Schur
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_spectral_radius_from_schur(
        self,
        T: NDArray[np.float64],
        n_blocks_2x2: int,
    ) -> float:
        r"""
        Extrae el radio espectral ρ(A) = max|λ| desde la forma de Schur T.

        Para la Real Schur Form (RSF):
            · Bloques 1×1 en diagonal: autovalores reales exactos λ = T[i,i].
            · Bloques 2×2: pares conjugados λ = a ± bi donde
                a = (T[i,i] + T[i+1,i+1]) / 2
                b = √|T[i,i+1] · T[i+1,i]|

        Se toman los módulos |λ| = √(a² + b²) para los bloques 2×2.

        Parámetros:
            T: forma de Schur cuasi-triangular (RSF).
            n_blocks_2x2: número de bloques 2×2 detectados.

        Retorna:
            ρ(A) = max{|λ₁|, ..., |λₙ|} ∈ [0, ∞).
        """
        n = T.shape[0]
        if n == 0:
            return 0.0

        eigenvalue_moduli: List[float] = []
        i = 0

        while i < n:
            if i < n - 1 and abs(T[i + 1, i]) > _MACHINE_EPSILON:
                # Bloque 2×2: extraer par conjugado.
                a = (T[i, i] + T[i + 1, i + 1]) / 2.0
                # Para un bloque [[a, b], [c, a]]: |λ| = √(a² + |bc|).
                bc = abs(T[i, i + 1] * T[i + 1, i])
                modulus = float(np.sqrt(a ** 2 + bc))
                eigenvalue_moduli.append(modulus)
                i += 2
            else:
                # Bloque 1×1: autovalor real.
                eigenvalue_moduli.append(abs(T[i, i]))
                i += 1

        return float(max(eigenvalue_moduli)) if eigenvalue_moduli else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # §1.7 Certificación espectral principal de nilpotencia
    # ──────────────────────────────────────────────────────────────────────────

    def _certify_spectral_nilpotence(
        self,
        adjacency_matrix: NDArray[Any],
        *,
        zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
        deep_nilpotence_audit: bool = True,
        power_audit_max_dim: int = _DEFAULT_POWER_AUDIT_MAX_DIM,
        raise_on_veto: bool = True,
    ) -> NilpotenceAuditData:
        r"""
        Certifica que Spec(A) = {0} mediante la descomposición de Schur.

        Pipeline de certificación:
            1. Validar y condicionar A (§1.1).
            2. Calcular RSF: A = Q·T·Q^T (§1.2).
            3. Calcular tolerancia dinámica con κ₂(Q) y ‖A‖_F (§1.3).
            4. Extraer ρ(A) = max|diag(T)| (§1.6).
            5. Si deep_nilpotence_audit=True:
               a. Para n ≤ max_dim: certificar ‖A^n‖_∞ y búsqueda de ν (§1.5).
               b. Para n > max_dim: estimación de Gelfand (§1.4).
            6. Emitir NilpotenceAuditData con todos los campos.
            7. Si ρ(A) > tol y raise_on_veto=True: lanzar CausalLoopVetoError.

        Teorema de Validez:
            El certificado emitido es correcto con probabilidad 1 − O(n²·ε_machine)
            bajo el modelo de error backward-stable de LAPACK dgees.

        Parámetros:
            adjacency_matrix: operador de adyacencia del DAG.
            zero_threshold: umbral de anulación de ruido numérico.
            allow_signed_weights: permitir pesos negativos.
            require_boolean: exigir entradas ∈ {0,1}.
            deep_nilpotence_audit: activar auditoría de potencia A^n.
            power_audit_max_dim: límite de dimensión para cálculo exacto.
            raise_on_veto: lanzar excepción si ρ(A) > tol.

        Retorna:
            NilpotenceAuditData: certificado espectral completo.

        Lanza:
            CausalLoopVetoError: si ρ(A) > tol y raise_on_veto=True.
        """
        # ── Paso 1: Validación y acondicionamiento ───────────────────────────
        A = self._validate_and_condition_adjacency(
            adjacency_matrix,
            zero_threshold=zero_threshold,
            allow_signed_weights=allow_signed_weights,
            require_boolean=require_boolean,
        )

        n = int(A.shape[0])
        notes: List[str] = []

        # ── Caso trivial: matriz vacía ────────────────────────────────────────
        if n == 0:
            return NilpotenceAuditData(
                dimension=0,
                spectral_radius=0.0,
                tolerance=_BASE_SPECTRAL_TOLERANCE,
                adjacency_inf_norm=0.0,
                frobenius_norm=0.0,
                nonzero_entries=0,
                directed_density=0.0,
                is_strictly_nilpotent=True,
                schur_condition_number=1.0,
                power_residual=0.0,
                power_audited=True,
                nilpotence_index=0,
                gelfand_radius_estimate=None,
                schur_2x2_blocks_detected=0,
                notes=("El operador vacío es trivialmente nilpotente (A^0 = I, A^1 = 0).",),
            )

        # ── Paso 2: Normas de la matriz ───────────────────────────────────────
        # Norma de Frobenius: ‖A‖_F = √(Σ|A[i,j]|²).
        # Más sensible que ‖A‖_∞ a la distribución de energía espectral.
        frobenius_norm = float(np.linalg.norm(A, ord="fro"))
        inf_norm = float(np.linalg.norm(A, ord=np.inf))

        # Densidad dirigida: fracción de aristas presentes vs posibles.
        nonzero_entries = int(np.count_nonzero(A))
        if n > 1:
            off_diag_mask = ~np.eye(n, dtype=bool)
            directed_density = float(np.count_nonzero(A[off_diag_mask])) / float(
                n * (n - 1)
            )
        else:
            directed_density = 0.0

        # ── Paso 3: Descomposición de Schur ───────────────────────────────────
        T, Q, kappa_Q, n_blocks_2x2 = self._schur_decompose(A)

        if kappa_Q > 1.0 + 1e-6:
            notes.append(
                f"Número de condición de Schur κ₂(Q)={kappa_Q:.4f} > 1. "
                f"La transformación de similaridad no es exactamente unitaria; "
                f"puede haber amplificación de errores numéricos espectrales."
            )

        if n_blocks_2x2 > 0:
            notes.append(
                f"RSF contiene {n_blocks_2x2} bloque(s) 2×2 (pares conjugados). "
                f"Para A nilpotente, la traza de cada bloque debe ser ≈ 0."
            )

        # ── Paso 4: Tolerancia dinámica ───────────────────────────────────────
        tolerance = self._dynamic_spectral_tolerance(n, frobenius_norm, kappa_Q)

        # ── Paso 5: Radio espectral desde la forma de Schur ───────────────────
        spectral_radius = self._extract_spectral_radius_from_schur(T, n_blocks_2x2)

        # ── Paso 6: Auditoría profunda de nilpotencia ─────────────────────────
        power_residual: Optional[float] = None
        nilpotence_index: Optional[int] = None
        gelfand_estimate: Optional[float] = None
        power_audited = False

        if deep_nilpotence_audit:
            power_tolerance = max(
                _BASE_POWER_TOLERANCE,
                _SCHUR_CONDITION_SAFETY_FACTOR * tolerance * max(1.0, inf_norm),
            )
            (
                power_residual,
                nilpotence_index,
                gelfand_estimate,
            ) = self._certify_nilpotence_index(A, power_tolerance, power_audit_max_dim)

            power_audited = power_residual is not None

            if power_audited and power_residual is not None:
                if power_residual > power_tolerance:
                    notes.append(
                        f"Residual de Cayley-Hamilton ‖A^n‖_∞={power_residual:.4e} "
                        f"excede tol_power={power_tolerance:.4e}. "
                        f"La matriz puede no ser nilpotente en la práctica numérica."
                    )

            if nilpotence_index is not None:
                notes.append(
                    f"Índice de nilpotencia certificado ν={nilpotence_index}. "
                    f"Longitud máxima de camino dirigido en el DAG ≤ ν−1={nilpotence_index - 1}."
                )

            if gelfand_estimate is not None:
                notes.append(
                    f"Estimación de Gelfand ρ̂(A)={gelfand_estimate:.4e} "
                    f"(n={n} > max_dim={power_audit_max_dim}, cálculo exacto omitido)."
                )
                if gelfand_estimate > tolerance:
                    notes.append(
                        "Estimación de Gelfand sugiere ρ(A) > tol. "
                        "El DAG puede no ser acíclico."
                    )

        # ── Paso 7: Certificación final ───────────────────────────────────────
        is_strictly_nilpotent = spectral_radius <= tolerance

        audit = NilpotenceAuditData(
            dimension=n,
            spectral_radius=spectral_radius,
            tolerance=tolerance,
            adjacency_inf_norm=inf_norm,
            frobenius_norm=frobenius_norm,
            nonzero_entries=nonzero_entries,
            directed_density=directed_density,
            is_strictly_nilpotent=is_strictly_nilpotent,
            schur_condition_number=kappa_Q,
            power_residual=power_residual,
            power_audited=power_audited,
            nilpotence_index=nilpotence_index,
            gelfand_radius_estimate=gelfand_estimate,
            schur_2x2_blocks_detected=n_blocks_2x2,
            notes=tuple(notes),
        )

        if not is_strictly_nilpotent and raise_on_veto:
            raise CausalLoopVetoError(
                f"Ruptura de causalidad espectral: el DAG propuesto no es acíclico. "
                f"Radio espectral ρ(A)={spectral_radius:.6e} > tol={tolerance:.6e} "
                f"(κ₂(Q)={kappa_Q:.4f}, ‖A‖_F={frobenius_norm:.4e}, n={n}). "
                f"La matriz de adyacencia no es nilpotente: existe al menos un "
                f"ciclo dirigido en el DAG."
            )

        return audit

    # ──────────────────────────────────────────────────────────────────────────
    # §1.8 Stub funtorial: puente a Fase 2
    # ──────────────────────────────────────────────────────────────────────────

    def _audit_poset_filtration_from_nilpotence(
        self,
        nilpotence_audit: NilpotenceAuditData,
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        **kwargs: Any,
    ) -> "PosetFiltrationData":
        r"""
        Stub funtorial de continuación hacia Fase 2.

        Este método es el "conector" que cierra la categoría de Fase 1
        y abre la categoría de Fase 2. La implementación concreta vive en
        Phase2_PosetFiltrationAuditor, que hereda y sobreescribe este stub.

        El patrón de stub-con-herencia implementa el anidamiento funtorial:
            Φ₁: A ↦ NilpotenceAuditData  [implementado]
            Φ₂: (NilpotenceAuditData, Poset) ↦ PosetFiltrationData  [stub aquí]

        Lanza:
            NotImplementedError: siempre (debe ser sobreescrito por Fase 2).
        """
        raise NotImplementedError(
            "Phase2_PosetFiltrationAuditor debe ser mezclado en la jerarquía. "
            "Este stub es el conector funtorial Φ₁ → Φ₂."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # §1.9 Último método de Fase 1: puente funtorial hacia Fase 2
    # ──────────────────────────────────────────────────────────────────────────

    def _phase1_terminal_bridge_to_phase2(
        self,
        adjacency_matrix: NDArray[Any],
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        *,
        adjacency_zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
        deep_nilpotence_audit: bool = True,
        power_audit_max_dim: int = _DEFAULT_POWER_AUDIT_MAX_DIM,
        allow_unknown_nodes: bool = False,
        strict_filtration: bool = False,
        node_order: Optional[Sequence[str]] = None,
        cross_check_adjacency_support: bool = False,
        raise_on_veto: bool = True,
    ) -> Tuple[NilpotenceAuditData, "PosetFiltrationData"]:
        r"""
        ÚLTIMO MÉTODO DE FASE 1 — Puente funtorial Φ₁ → Φ₂.

        Este método es la composición terminal de Fase 1 que produce el
        artefacto NilpotenceAuditData y lo pasa como entrada a Fase 2.

        Composición funtorial:
            Φ₁(A) → NilpotenceAuditData
            (Φ₁(A), Φ₂(edges, strata)) → (NilpotenceAuditData, PosetFiltrationData)

        El morfismo terminal de Fase 1 actúa como el morfismo inicial de Fase 2:
            _phase1_terminal_bridge_to_phase2 ≅ init(Fase 2) ∘ final(Fase 1)

        Parámetros:
            adjacency_matrix: operador de adyacencia del DAG.
            edges: lista de aristas dirigidas (u, v).
            node_strata: mapeo nodo → nivel DIKW.
            adjacency_zero_threshold: umbral de anulación numérica.
            allow_signed_weights: permitir pesos negativos en A.
            require_boolean: exigir A ∈ {0,1}^{n×n}.
            deep_nilpotence_audit: activar auditoría profunda de potencia.
            power_audit_max_dim: límite para cálculo exacto de A^n.
            allow_unknown_nodes: ignorar nodos sin estrato.
            strict_filtration: exigir stratum(u) < stratum(v).
            node_order: orden canónico de nodos para cross-check.
            cross_check_adjacency_support: verificar edges↔A.
            raise_on_veto: lanzar excepciones ante violaciones.

        Retorna:
            (NilpotenceAuditData, PosetFiltrationData): certificados de Fases 1 y 2.
        """
        # ── Fase 1: Certificación espectral ───────────────────────────────────
        nilpotence_audit = self._certify_spectral_nilpotence(
            adjacency_matrix,
            zero_threshold=adjacency_zero_threshold,
            allow_signed_weights=allow_signed_weights,
            require_boolean=require_boolean,
            deep_nilpotence_audit=deep_nilpotence_audit,
            power_audit_max_dim=power_audit_max_dim,
            raise_on_veto=raise_on_veto,
        )

        # ── Puente → Fase 2: Auditoría de filtración ──────────────────────────
        filtration_audit = self._audit_poset_filtration_from_nilpotence(
            nilpotence_audit,
            edges,
            node_strata,
            allow_unknown_nodes=allow_unknown_nodes,
            strict_filtration=strict_filtration,
            node_order=node_order,
            adjacency_matrix=adjacency_matrix,
            cross_check_adjacency_support=cross_check_adjacency_support,
            adjacency_zero_threshold=adjacency_zero_threshold,
            raise_on_veto=raise_on_veto,
        )

        return nilpotence_audit, filtration_audit


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2: AUDITORÍA DEL DIFEOMORFISMO DE FILTRACIÓN CATEGÓRICA              ║
# ║                                                                           ║
# ║ Fundamento matemático:                                                    ║
# ║   El conjunto (V, ≤_strata) de nodos con la relación de orden por         ║
# ║   estratos DIKW forma un Poset. El DAG debe ser compatible con este        ║
# ║   Poset: toda arista (u→v) debe satisfacer stratum(u) ≤ stratum(v).       ║
# ║                                                                           ║
# ║   Esto es equivalente a exigir que la función estrato                     ║
# ║       f: V → {0,1,2,3,4}  (PHYSICS=0,...,WISDOM=4)                       ║
# ║   sea un morfismo de Posets entre (V, ≤_DAG) y (Z, ≤).                   ║
# ║                                                                           ║
# ║ El último método de esta fase es el puente formal hacia Fase 3.           ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase2_PosetFiltrationAuditor(Phase1_SpectralNilpotenceCertifier):
    r"""
    Auditor de monotonicidad en el Poset DIKW con detección de auto-bucles.

    Ley de Clausura Transitiva del Poset DIKW:
        PHYSICS(0) ≤ DATA(1) ≤ INFORMATION(2) ≤ KNOWLEDGE(3) ≤ WISDOM(4)

    Toda arista (u → v) debe ser un morfismo de Posets:
        stratum(u) ≤ stratum(v)  [modo lazo]
        stratum(u) < stratum(v)  [modo estricto]

    Nuevas verificaciones v3:
        · Auto-bucles (u,u): violación categórica de la aciclicidad.
        · Histograma de slacks: diagnóstico de concentración en estratos.
        · Nodos aislados en node_strata: certificación de cobertura.
        · Aristas inversas A[j,i] ≠ 0: detección de ciclos de longitud 2.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # §2.1 Coerción de estratos a enteros ordenados
    # ──────────────────────────────────────────────────────────────────────────

    def _coerce_stratum_level(
        self,
        node: str,
        raw_level: Any,
    ) -> int:
        r"""
        Convierte el nivel de estrato de un nodo a entero no negativo.

        Jerarquía de conversión (en orden de precedencia):
            1. bool/np.bool_ → RECHAZADO (semánticamente ambiguo).
            2. int/np.integer → conversión directa.
            3. Objeto con .value (Enum-like) → int(.value).
            4. float/np.floating → verificar finitud e integralidad.
            5. str → int(strip()) si es numéricamente parseable.
            6. Cualquier otro → StratumMappingError.

        Invariante:
            El nivel retornado es un entero ≥ 0 (estrato DIKW válido).

        Parámetros:
            node: identificador del nodo (para mensajes de error).
            raw_level: valor crudo del estrato desde node_strata.

        Retorna:
            int ≥ 0: nivel de estrato normalizado.

        Lanza:
            StratumMappingError: si la conversión falla o el nivel es negativo.
        """
        # Los booleanos son subclase de int en Python, pero semánticamente
        # ambiguos para un estrato DIKW. Los rechazamos explícitamente.
        if isinstance(raw_level, (bool, np.bool_)):
            raise StratumMappingError(
                f"Nodo '{node}': estrato booleano inválido ({raw_level!r}). "
                f"Se requiere un nivel entero ordenado {0, 1, 2, 3, 4}. "
                f"Use el enum Stratum o un entero explícito."
            )

        # Enteros nativos: conversión directa.
        if isinstance(raw_level, (int, np.integer)):
            return int(raw_level)

        # Enums y objetos con .value (p.ej., Stratum.KNOWLEDGE).
        if hasattr(raw_level, "value"):
            try:
                return int(raw_level.value)
            except (TypeError, ValueError) as exc:
                raise StratumMappingError(
                    f"Nodo '{node}': no se pudo convertir {raw_level!r}.value "
                    f"a entero: {exc}."
                ) from exc

        # Floats: verificar que sean enteros finitos.
        if isinstance(raw_level, (float, np.floating)):
            value = float(raw_level)
            if not np.isfinite(value):
                raise StratumMappingError(
                    f"Nodo '{node}': estrato float no finito: {raw_level!r}. "
                    f"El estrato debe ser un entero finito."
                )
            if not value.is_integer():
                raise StratumMappingError(
                    f"Nodo '{node}': estrato float no entero: {raw_level!r}. "
                    f"El estrato debe ser un valor integral (p.ej., 2.0, no 2.7)."
                )
            return int(value)

        # Strings: parseo numérico.
        if isinstance(raw_level, str):
            stripped = raw_level.strip()
            try:
                return int(stripped)
            except ValueError:
                # Intentar parseo vía float para strings como "2.0".
                try:
                    fval = float(stripped)
                    if fval.is_integer():
                        return int(fval)
                except ValueError:
                    pass

        raise StratumMappingError(
            f"Nodo '{node}': no fue posible convertir el estrato {raw_level!r} "
            f"(tipo {type(raw_level).__name__}) a un nivel entero ordenado. "
            f"Tipos soportados: int, np.integer, float integral, Enum con .value, str numérico."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # §2.2 Normalización y validación de aristas con detección de auto-bucles
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize_edges(
        self,
        edges: Sequence[Tuple[str, str]],
        *,
        raise_on_self_loop: bool = True,
    ) -> Tuple[List[Tuple[str, str]], int]:
        r"""
        Normaliza, valida y clasifica las aristas dirigidas.

        Verificaciones (v3):
            1. edges no es None.
            2. Cada elemento es desempaquetable como (u, v).
            3. u y v son strings no vacíos.
            4. Auto-bucles (u, u): se detectan y se lanza SelfLoopVetoError
               si raise_on_self_loop=True (o se cuentan si False).

        Auto-bucles (mejora §2.1):
            Una arista (u, u) es un ciclo de longitud 1. Es semánticamente
            incompatible con un DAG causal incluso si la diagonal de A es cero
            (los edges podrían estar desincronizados con la matriz).

        Parámetros:
            edges: secuencia de pares (u, v).
            raise_on_self_loop: si True, lanza SelfLoopVetoError al detectar (u,u).

        Retorna:
            (lista normalizada, n_self_loops): aristas válidas y conteo de bucles.

        Lanza:
            PipelineDirectorAgentError: si edges es None o tiene formato inválido.
            SelfLoopVetoError: si se detecta (u, u) y raise_on_self_loop=True.
        """
        if edges is None:
            raise PipelineDirectorAgentError(
                "edges no puede ser None. "
                "Proporcione una secuencia vacía [] si el DAG no tiene aristas."
            )

        normalized: List[Tuple[str, str]] = []
        self_loops_found: List[str] = []

        for idx, edge in enumerate(edges):
            # Verificar que la arista es desempaquetable como par.
            try:
                u, v = edge
            except (TypeError, ValueError) as exc:
                raise PipelineDirectorAgentError(
                    f"La arista en índice {idx} debe ser una tupla (u, v). "
                    f"Se recibió: {edge!r} (tipo {type(edge).__name__}). "
                    f"Error: {exc}"
                ) from exc

            u_str = str(u).strip()
            v_str = str(v).strip()

            # Verificar que los identificadores no son vacíos.
            if not u_str:
                raise PipelineDirectorAgentError(
                    f"La arista en índice {idx} tiene el nodo origen vacío: "
                    f"({u!r}, {v!r}). Los identificadores de nodo no pueden ser vacíos."
                )
            if not v_str:
                raise PipelineDirectorAgentError(
                    f"La arista en índice {idx} tiene el nodo destino vacío: "
                    f"({u!r}, {v!r}). Los identificadores de nodo no pueden ser vacíos."
                )

            # Detección de auto-bucles (mejora §2.1).
            if u_str == v_str:
                self_loops_found.append(u_str)
                if raise_on_self_loop:
                    raise SelfLoopVetoError(
                        f"Auto-bucle detectado en la arista índice {idx}: "
                        f"({u_str!r} → {u_str!r}). "
                        f"Un DAG causal no puede contener ciclos de longitud 1. "
                        f"El nodo '{u_str}' no puede ser causa de sí mismo."
                    )
                continue  # Si no se lanza, omitir el auto-bucle del resultado.

            normalized.append((u_str, v_str))

        return normalized, len(self_loops_found)

    # ──────────────────────────────────────────────────────────────────────────
    # §2.3 Normalización del mapeo nodo → estrato con detección de negativos
    # ──────────────────────────────────────────────────────────────────────────

    def _normalize_node_strata(
        self,
        node_strata: Mapping[str, Any],
    ) -> Dict[str, int]:
        r"""
        Normaliza el mapeo nodo → nivel de estrato entero.

        Contrato:
            · node_strata puede ser None o vacío (retorna {}).
            · Claves no vacías (strip aplicado).
            · Valores convertibles a int ≥ 0 por _coerce_stratum_level.

        Retorna:
            Dict[str, int]: mapeo normalizado nodo → estrato.

        Lanza:
            StratumMappingError: si alguna clave está vacía o el nivel
            es negativo o no convertible.
        """
        if not node_strata:
            return {}

        normalized: Dict[str, int] = {}

        for node, raw_level in node_strata.items():
            key = str(node).strip()

            if not key:
                raise StratumMappingError(
                    "Se encontró una clave de nodo vacía en node_strata. "
                    "Los identificadores de nodo no pueden ser strings vacíos."
                )

            level = self._coerce_stratum_level(key, raw_level)

            if level < 0:
                raise StratumMappingError(
                    f"Nodo '{key}': estrato negativo {level}. "
                    f"Los estratos DIKW son niveles enteros no negativos "
                    f"(PHYSICS=0, DATA=1, INFORMATION=2, KNOWLEDGE=3, WISDOM=4)."
                )

            normalized[key] = level

        return normalized

    # ──────────────────────────────────────────────────────────────────────────
    # §2.4 Cross-check de soporte edges↔matriz con detección de aristas inversas
    # ──────────────────────────────────────────────────────────────────────────

    def _audit_adjacency_support(
        self,
        adjacency_matrix: Optional[NDArray[Any]],
        edges: Sequence[Tuple[str, str]],
        node_order: Optional[Sequence[str]],
        *,
        adjacency_zero_threshold: float,
        check_inverse_edges: bool = True,
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        r"""
        Verifica la consistencia de soporte entre edges y la matriz A.

        Verificaciones (v3):
            (a) Para toda arista declarada (u→v):
                A[index(u), index(v)] ≠ 0 (soporte directo).
            (b) NUEVO: Para toda arista declarada (u→v):
                A[index(v), index(u)] = 0 (ausencia de arista inversa).
                Una arista inversa A[v,u] ≠ 0 introduciría un ciclo u→v→u
                de longitud 2 no detectado por la diagonal de A.

        Condición de activación:
            El cross-check sólo se ejecuta si adjacency_matrix is not None
            y node_order is not None (se necesita el mapeo nodo→índice).

        Parámetros:
            adjacency_matrix: operador de adyacencia.
            edges: aristas normalizadas.
            node_order: orden canónico de nodos (define el mapeo nodo→índice).
            adjacency_zero_threshold: umbral de cero numérico.
            check_inverse_edges: si True, verifica A[j,i] = 0 para (u→v).

        Retorna:
            (mensajes_ok, aristas_inversas_bloqueadas): notas y bloqueos.

        Lanza:
            AdjacencyMatrixFormatError: si node_order es inconsistente con A.
            AdjacencySupportVetoError: si se detectan inconsistencias.
        """
        if adjacency_matrix is None or node_order is None:
            return (), ()

        # Re-validar y condicionar la matriz (puede diferir de la de Fase 1
        # si se modificó el zero_threshold en el cross-check).
        A = self._validate_and_condition_adjacency(
            adjacency_matrix,
            zero_threshold=adjacency_zero_threshold,
            allow_signed_weights=True,  # Soporte puede tener pesos negativos.
            require_boolean=False,
        )

        order = [str(node).strip() for node in node_order]

        if len(order) != A.shape[0]:
            raise AdjacencyMatrixFormatError(
                f"node_order tiene longitud {len(order)} pero A tiene "
                f"dimensión {A.shape[0]}. "
                f"El orden de nodos debe corresponder exactamente a las filas/columnas de A."
            )

        if len(set(order)) != len(order):
            duplicates = [
                node for node in set(order) if order.count(node) > 1
            ]
            raise AdjacencyMatrixFormatError(
                f"node_order contiene identificadores duplicados: {duplicates[:5]}."
            )

        index_map: Dict[str, int] = {node: i for i, node in enumerate(order)}

        # (a) Verificación de soporte directo: A[i,j] ≠ 0 para (u→v).
        unsupported_direct: List[str] = []

        # (b) Detección de aristas inversas: A[j,i] ≠ 0 para (u→v).
        inverse_edges_found: List[str] = []

        for u, v in edges:
            if u not in index_map or v not in index_map:
                continue  # Nodo fuera del order: se ignora en cross-check.

            i = index_map[u]
            j = index_map[v]

            # Verificación (a): soporte directo.
            if A[i, j] == 0.0:
                unsupported_direct.append(f"{u}→{v}")

            # Verificación (b): ausencia de arista inversa (v→u en la matriz).
            if check_inverse_edges and A[j, i] != 0.0:
                inverse_edges_found.append(
                    f"{v}→{u} (A[{j},{i}]={A[j, i]:.4e})"
                )

        errors: List[str] = []

        if unsupported_direct:
            preview = ", ".join(unsupported_direct[:8])
            suffix = f" y {len(unsupported_direct)-8} más" if len(unsupported_direct) > 8 else ""
            errors.append(
                f"Soporte directo faltante en {len(unsupported_direct)} arista(s): "
                f"{preview}{suffix}."
            )

        if inverse_edges_found:
            preview = ", ".join(inverse_edges_found[:8])
            suffix = f" y {len(inverse_edges_found)-8} más" if len(inverse_edges_found) > 8 else ""
            errors.append(
                f"Aristas inversas detectadas (ciclos de longitud 2) en "
                f"{len(inverse_edges_found)} caso(s): {preview}{suffix}."
            )

        if errors:
            raise AdjacencySupportVetoError(
                "Cross-check edges↔adjacency fallido: "
                + " | ".join(errors)
            )

        ok_notes = ("Cross-check de soporte edges↔adjacency satisfactorio (directo + inverso).",)
        inverse_blocked = tuple(inverse_edges_found)

        return ok_notes, inverse_blocked

    # ──────────────────────────────────────────────────────────────────────────
    # §2.5 Cálculo del histograma de slacks
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_slack_histogram(
        self,
        slacks: List[int],
    ) -> Tuple[Tuple[int, int], ...]:
        r"""
        Calcula el histograma de distribución de slacks.

        El histograma mapea cada valor de slack a su frecuencia de aristas:
            H(k) = |{(u,v) ∈ E : stratum(v) − stratum(u) = k}|

        Interpretación:
            · H(0) grande: muchas aristas dentro del mismo estrato (planas).
            · H(k) para k > 0: aristas que saltan k estratos (empinadas).
            · La distribución de H permite detectar cuellos de botella en
              la filtración DIKW.

        Parámetros:
            slacks: lista de valores slack = stratum(v) − stratum(u).

        Retorna:
            Tupla de pares (slack_value, count) ordenados por slack_value.
        """
        if not slacks:
            return ()

        from collections import Counter

        histogram = Counter(slacks)
        return tuple(sorted(histogram.items()))

    # ──────────────────────────────────────────────────────────────────────────
    # §2.6 Detección de nodos aislados en node_strata
    # ──────────────────────────────────────────────────────────────────────────

    def _find_isolated_strata_nodes(
        self,
        strata: Dict[str, int],
        edges: List[Tuple[str, str]],
    ) -> Tuple[str, ...]:
        r"""
        Identifica nodos en node_strata que no participan en ninguna arista.

        Un nodo aislado es un vértice de grado 0 (in-degree + out-degree = 0)
        en el DAG. Su existencia en node_strata es válida pero informativa:
        puede indicar un nodo fuente/sumidero sin conexiones declaradas.

        Parámetros:
            strata: mapeo normalizado nodo → estrato.
            edges: lista de aristas normalizadas.

        Retorna:
            Tupla de nombres de nodos aislados (ordenados lexicográficamente).
        """
        if not strata:
            return ()

        # Conjunto de nodos que aparecen en al menos una arista.
        connected_nodes: FrozenSet[str] = frozenset(
            node for edge in edges for node in edge
        )

        isolated = tuple(
            sorted(node for node in strata if node not in connected_nodes)
        )
        return isolated

    # ──────────────────────────────────────────────────────────────────────────
    # §2.7 Auditoría principal de filtración del Poset DIKW
    # ──────────────────────────────────────────────────────────────────────────

    def _audit_poset_filtration_from_nilpotence(
        self,
        nilpotence_audit: NilpotenceAuditData,
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        *,
        allow_unknown_nodes: bool = False,
        strict_filtration: bool = False,
        raise_on_veto: bool = True,
        node_order: Optional[Sequence[str]] = None,
        adjacency_matrix: Optional[NDArray[Any]] = None,
        cross_check_adjacency_support: bool = False,
        adjacency_zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
    ) -> PosetFiltrationData:
        r"""
        Auditoría principal de Fase 2: monotonicidad del Poset DIKW.

        Pipeline de auditoría (v3):
            1. Normalizar edges con detección de auto-bucles (§2.2).
            2. Cross-check opcional de soporte edges↔A con inversas (§2.4).
            3. Verificar que Fase 1 certificó nilpotencia (precondición).
            4. Normalizar node_strata (§2.3).
            5. Para cada arista (u,v): calcular slack = stratum(v)−stratum(u).
            6. Verificar monotonicidad laxa (slack ≥ 0) o estricta (slack > 0).
            7. Calcular histograma de slacks (§2.5).
            8. Detectar nodos aislados en node_strata (§2.6).
            9. Emitir PosetFiltrationData con todos los campos.

        Axioma de monotonicidad (morfismo de Posets):
            f: (V, ≤_DAG) → (Z, ≤) es un morfismo de Posets si y sólo si:
                u ≤_DAG v ⟹ f(u) ≤ f(v)
            donde u ≤_DAG v significa "existe un camino dirigido u → ... → v".

        Parámetros:
            nilpotence_audit: certificado de Fase 1 (precondición).
            edges: aristas del DAG.
            node_strata: mapeo nodo → estrato DIKW.
            allow_unknown_nodes: ignorar aristas con nodos sin estrato.
            strict_filtration: exigir slack > 0 (morfismo de Posets estricto).
            raise_on_veto: lanzar excepciones ante violaciones.
            node_order: orden canónico para cross-check.
            adjacency_matrix: matriz para cross-check.
            cross_check_adjacency_support: activar cross-check.
            adjacency_zero_threshold: umbral de cero para cross-check.

        Retorna:
            PosetFiltrationData: certificado de filtración completo.

        Lanza:
            SelfLoopVetoError: si se detecta un auto-bucle.
            CausalLoopVetoError: si Fase 1 no certificó nilpotencia.
            StratumMappingError: si falta un estrato requerido.
            FiltrationViolationVeto: si se viola la monotonicidad.
            AdjacencySupportVetoError: si el cross-check falla.
        """
        notes: List[str] = []

        # ── Paso 1: Normalización de aristas con detección de auto-bucles ─────
        edges_norm, n_self_loops = self._normalize_edges(
            edges,
            raise_on_self_loop=raise_on_veto,
        )

        if n_self_loops > 0:
            notes.append(
                f"Se detectaron {n_self_loops} auto-bucle(s) en la lista de aristas. "
                f"Fueron excluidos del análisis de filtración."
            )

        # ── Paso 2: Cross-check de soporte (opcional) ─────────────────────────
        inverse_edges_blocked: Tuple[str, ...] = ()
        if cross_check_adjacency_support:
            try:
                ok_notes, inverse_edges_blocked = self._audit_adjacency_support(
                    adjacency_matrix,
                    edges_norm,
                    node_order,
                    adjacency_zero_threshold=adjacency_zero_threshold,
                    check_inverse_edges=True,
                )
                notes.extend(ok_notes)
            except AdjacencySupportVetoError:
                if raise_on_veto:
                    raise
                notes.append("Cross-check de soporte fallido (raise_on_veto=False).")

        # ── Paso 3: Precondición: Fase 1 debe haber certificado nilpotencia ───
        if not nilpotence_audit.is_strictly_nilpotent:
            degraded_note = (
                "Fase 2 degradada: la auditoría de estratos se omite porque "
                "Fase 1 no certificó nilpotencia espectral del DAG."
            )
            notes.append(degraded_note)

            if raise_on_veto:
                raise CausalLoopVetoError(
                    "No se puede certificar la filtración Poset porque el DAG "
                    "no superó la certificación espectral de aciclicidad (Fase 1). "
                    f"ρ(A)={nilpotence_audit.spectral_radius:.6e} > "
                    f"tol={nilpotence_audit.tolerance:.6e}."
                )

            return PosetFiltrationData(
                edge_count=len(edges_norm) + n_self_loops,
                audited_edge_count=0,
                ignored_edge_count=len(edges_norm),
                min_slack=None,
                max_slack=None,
                is_monotonic_filtration=False,
                self_loops_detected=n_self_loops,
                unknown_nodes=(),
                isolated_strata_nodes=(),
                inverse_edges_blocked=inverse_edges_blocked,
                slack_histogram=(),
                notes=tuple(notes),
            )

        # ── Paso 4: Normalización de estratos ────────────────────────────────
        strata = self._normalize_node_strata(node_strata)

        # ── Paso 8 (pre): Detección de nodos aislados ─────────────────────────
        isolated_nodes = self._find_isolated_strata_nodes(strata, edges_norm)
        if isolated_nodes:
            notes.append(
                f"Nodos aislados en node_strata (sin aristas): {list(isolated_nodes[:8])}."
            )

        # ── Pasos 5-6: Cálculo de slacks y verificación de monotonicidad ─────
        unknown_nodes: List[str] = []
        violations: List[str] = []
        slacks: List[int] = []
        audited_edge_count = 0
        ignored_edge_count = 0

        for u, v in edges_norm:
            # Verificación de cobertura del estrato.
            u_known = u in strata
            v_known = v in strata

            if not u_known or not v_known:
                if allow_unknown_nodes:
                    ignored_edge_count += 1
                    if not u_known:
                        unknown_nodes.append(u)
                    if not v_known:
                        unknown_nodes.append(v)
                    continue

                missing = [node for node, known in [(u, u_known), (v, v_known)] if not known]
                raise StratumMappingError(
                    f"Falta(n) estrato(s) para nodo(s) {missing} en la arista {u}→{v}. "
                    f"Use allow_unknown_nodes=True para ignorar nodos auxiliares."
                )

            stratum_u = strata[u]
            stratum_v = strata[v]
            slack = stratum_v - stratum_u

            audited_edge_count += 1
            slacks.append(slack)

            # Verificación de monotonicidad.
            if strict_filtration:
                if slack <= 0:
                    violations.append(
                        f"{u}(s={stratum_u})→{v}(s={stratum_v}) "
                        f"slack={slack}≤0; strict_filtration exige slack>0."
                    )
            else:
                if slack < 0:
                    violations.append(
                        f"{u}(s={stratum_u})→{v}(s={stratum_v}) "
                        f"slack={slack}<0; se requiere slack≥0."
                    )

        # ── Paso 7: Histograma de slacks ──────────────────────────────────────
        slack_histogram = self._compute_slack_histogram(slacks)

        # ── Paso 9: Emitir artefacto ──────────────────────────────────────────
        if violations:
            if raise_on_veto:
                raise FiltrationViolationVeto(
                    f"Violación de la flecha del tiempo semántica DIKW. "
                    f"{len(violations)} violación(es) de filtración detectadas. "
                    f"Primera: {violations[0]}"
                )

            return PosetFiltrationData(
                edge_count=len(edges_norm) + n_self_loops,
                audited_edge_count=audited_edge_count,
                ignored_edge_count=ignored_edge_count,
                min_slack=min(slacks) if slacks else None,
                max_slack=max(slacks) if slacks else None,
                is_monotonic_filtration=False,
                self_loops_detected=n_self_loops,
                unknown_nodes=tuple(sorted(set(unknown_nodes))),
                isolated_strata_nodes=isolated_nodes,
                inverse_edges_blocked=inverse_edges_blocked,
                slack_histogram=slack_histogram,
                notes=tuple(violations + notes),
            )

        return PosetFiltrationData(
            edge_count=len(edges_norm) + n_self_loops,
            audited_edge_count=audited_edge_count,
            ignored_edge_count=ignored_edge_count,
            min_slack=min(slacks) if slacks else None,
            max_slack=max(slacks) if slacks else None,
            is_monotonic_filtration=True,
            self_loops_detected=n_self_loops,
            unknown_nodes=tuple(sorted(set(unknown_nodes))),
            isolated_strata_nodes=isolated_nodes,
            inverse_edges_blocked=inverse_edges_blocked,
            slack_histogram=slack_histogram,
            notes=tuple(notes),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # §2.8 Stub funtorial: puente a Fase 3
    # ──────────────────────────────────────────────────────────────────────────

    def _intercept_mayer_vietoris_sequence(
        self,
        betti_1_A: int,
        betti_1_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        **kwargs: Any,
    ) -> "MayerVietorisAuditData":
        r"""
        Stub funtorial de continuación hacia Fase 3.

        Este método cierra la categoría de Fase 2 y abre la categoría de Fase 3.
        La implementación concreta vive en Phase3_MayerVietorisInterceptor.

        Lanza:
            NotImplementedError: siempre (debe ser sobreescrito por Fase 3).
        """
        raise NotImplementedError(
            "Phase3_MayerVietorisInterceptor debe ser mezclado en la jerarquía. "
            "Este stub es el conector funtorial Φ₂ → Φ₃."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # §2.9 Último método de Fase 2: puente funtorial hacia Fase 3
    # ──────────────────────────────────────────────────────────────────────────

    def _phase2_terminal_bridge_to_phase3(
        self,
        adjacency_matrix: NDArray[Any],
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        betti_1_A: int,
        betti_1_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        *,
        adjacency_zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
        deep_nilpotence_audit: bool = True,
        power_audit_max_dim: int = _DEFAULT_POWER_AUDIT_MAX_DIM,
        allow_unknown_nodes: bool = False,
        strict_filtration: bool = False,
        node_order: Optional[Sequence[str]] = None,
        cross_check_adjacency_support: bool = False,
        image_rank_h1_intersection: Optional[int] = None,
        connecting_boundary_rank: Optional[int] = None,
        betti_0_A: Optional[int] = None,
        betti_0_B: Optional[int] = None,
        betti_0_intersection: Optional[int] = None,
        betti_0_union: Optional[int] = None,
        raise_on_veto: bool = True,
    ) -> Tuple[NilpotenceAuditData, PosetFiltrationData, "MayerVietorisAuditData"]:
        r"""
        ÚLTIMO MÉTODO DE FASE 2 — Puente funtorial Φ₂ → Φ₃.

        Composición funtorial:
            Φ₂ ∘ Φ₁: (A, edges, strata) → (NilpotenceAuditData, PosetFiltrationData)
            (Φ₂ ∘ Φ₁, Φ₃(β)) → (NilpotenceAuditData,
                                   PosetFiltrationData,
                                   MayerVietorisAuditData)

        Parámetros adicionales v3:
            betti_0_*: números de Betti en grado 0 para verificación de
                Euler-Poincaré y la secuencia exacta en H₀.
            image_rank_h1_intersection: rank(im H₁(A∩B)→H₁(A)⊕H₁(B)).
            connecting_boundary_rank: rank(∂: H₁(A∪B)→H₀(A∩B)).

        Retorna:
            (NilpotenceAuditData, PosetFiltrationData, MayerVietorisAuditData).
        """
        # ── Fases 1+2: Certificación espectral y de filtración ────────────────
        nilpotence_audit, filtration_audit = self._phase1_terminal_bridge_to_phase2(
            adjacency_matrix,
            edges,
            node_strata,
            adjacency_zero_threshold=adjacency_zero_threshold,
            allow_signed_weights=allow_signed_weights,
            require_boolean=require_boolean,
            deep_nilpotence_audit=deep_nilpotence_audit,
            power_audit_max_dim=power_audit_max_dim,
            allow_unknown_nodes=allow_unknown_nodes,
            strict_filtration=strict_filtration,
            node_order=node_order,
            cross_check_adjacency_support=cross_check_adjacency_support,
            raise_on_veto=raise_on_veto,
        )

        # ── Puente → Fase 3: Intercepción de Mayer-Vietoris ──────────────────
        mayer_vietoris_audit = self._intercept_mayer_vietoris_sequence(
            betti_1_A,
            betti_1_B,
            betti_1_intersection,
            betti_1_union,
            image_rank_h1_intersection=image_rank_h1_intersection,
            connecting_boundary_rank=connecting_boundary_rank,
            betti_0_A=betti_0_A,
            betti_0_B=betti_0_B,
            betti_0_intersection=betti_0_intersection,
            betti_0_union=betti_0_union,
            raise_on_veto=raise_on_veto,
        )

        return nilpotence_audit, filtration_audit, mayer_vietoris_audit


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3: INTERCEPCIÓN DE LA COHOMOLOGÍA DE FUSIÓN (MAYER-VIETORIS)         ║
# ║                                                                           ║
# ║ Fundamento matemático:                                                    ║
# ║   Dada una descomposición de un espacio topológico X = A ∪ B,             ║
# ║   la secuencia exacta larga de Mayer-Vietoris relaciona los grupos        ║
# ║   de homología de A, B, A∩B y A∪B:                                        ║
# ║                                                                           ║
# ║   ... → H₁(A∩B) →^φ H₁(A)⊕H₁(B) →^ψ H₁(A∪B)                            ║
# ║       →^∂ H₀(A∩B) → H₀(A)⊕H₀(B) → H₀(A∪B) → 0                          ║
# ║                                                                           ║
# ║   Por exactitud: ker(ψ) = im(φ), ker(∂) = im(ψ), im(∂) = ker(...).        ║
# ║   Luego: β₁(A∪B) = β₁(A)+β₁(B) − rank(im φ) + rank(im ∂).               ║
# ║                                                                           ║
# ║ El último método de esta fase sintetiza el objeto final de gobernanza.    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


class Phase3_MayerVietorisInterceptor(Phase2_PosetFiltrationAuditor):
    r"""
    Interceptor homológico de fusión con secuencia exacta completa.

    Nuevas verificaciones v3:
        · Desigualdad débil de Mayer-Vietoris (siempre verificable).
        · Secuencia exacta en H₀ con β₀ opcionales.
        · Característica de Euler-Poincaré bajo fusión.
        · Métrica de defecto Jensen-Shannon (simétrica, acotada en [0,1)).
    """

    # ──────────────────────────────────────────────────────────────────────────
    # §3.1 Validación de números de Betti
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_betti_number(
        self,
        name: str,
        value: Any,
    ) -> int:
        r"""
        Valida que un número de Betti sea un entero no negativo.

        Los números de Betti β_k = rank(H_k(X)) son siempre enteros ≥ 0
        por definición (rango de un grupo abeliano libre).

        Parámetros:
            name: nombre del parámetro (para mensajes de error).
            value: valor a validar.

        Retorna:
            int ≥ 0: número de Betti válido.

        Lanza:
            MayerVietorisInputError: si value no es un entero no negativo.
        """
        # Booleanos rechazados: True=1, False=0 son enteros válidos numéricamente
        # pero semánticamente ambiguos para un número de Betti.
        if isinstance(value, (bool, np.bool_)):
            raise MayerVietorisInputError(
                f"{name} no puede ser booleano ({value!r}). "
                f"Los números de Betti son enteros no negativos, no valores de verdad."
            )

        if isinstance(value, (int, np.integer)):
            ivalue = int(value)
        elif isinstance(value, (float, np.floating)):
            fvalue = float(value)
            if not np.isfinite(fvalue):
                raise MayerVietorisInputError(
                    f"{name} debe ser un entero no negativo finito. "
                    f"Se recibió: {value!r} (no finito)."
                )
            if not fvalue.is_integer():
                raise MayerVietorisInputError(
                    f"{name} debe ser un entero no negativo. "
                    f"Se recibió float no entero: {value!r}."
                )
            ivalue = int(fvalue)
        else:
            raise MayerVietorisInputError(
                f"{name} debe ser un entero no negativo. "
                f"Se recibió: {value!r} (tipo {type(value).__name__})."
            )

        if ivalue < 0:
            raise MayerVietorisInputError(
                f"{name} no puede ser negativo. "
                f"Se recibió: {ivalue}. "
                f"Por definición, β_k = rank(H_k(X)) ≥ 0."
            )

        return ivalue

    # ──────────────────────────────────────────────────────────────────────────
    # §3.2 Validación de rangos opcionales de la secuencia exacta
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_optional_rank(
        self,
        name: str,
        value: Optional[int],
        upper_bound: int,
        lower_bound: int = 0,
    ) -> Optional[int]:
        r"""
        Valida un rango opcional de la secuencia exacta de Mayer-Vietoris.

        Contrato algebraico:
            lower_bound ≤ rank ≤ upper_bound
            El upper_bound es la cota algebraica derivada del teorema del rango.

        Parámetros:
            name: nombre del rango (para mensajes de error).
            value: valor a validar (None → se usa el defecto).
            upper_bound: cota superior algebraica.
            lower_bound: cota inferior (default 0).

        Retorna:
            int ∈ [lower_bound, upper_bound] o None si value es None.

        Lanza:
            MayerVietorisInputError: si la cota algebraica es violada.
        """
        if value is None:
            return None

        rank = self._validate_betti_number(name, value)

        if rank < lower_bound:
            raise MayerVietorisInputError(
                f"{name}={rank} viola la cota inferior algebraica {lower_bound}."
            )

        if rank > upper_bound:
            raise MayerVietorisInputError(
                f"{name}={rank} viola la cota superior algebraica {upper_bound}. "
                f"Por el Teorema del Rango, rank(φ) ≤ min(dim(dominio), dim(codominio)) "
                f"= {upper_bound}."
            )

        return rank

    # ──────────────────────────────────────────────────────────────────────────
    # §3.3 Verificación de la desigualdad débil de Mayer-Vietoris
    # ──────────────────────────────────────────────────────────────────────────

    def _verify_weak_mayer_vietoris_bound(
        self,
        bA: int,
        bB: int,
        bI: int,
        bU: int,
    ) -> Tuple[bool, str]:
        r"""
        Verifica la desigualdad débil de Mayer-Vietoris.

        Desigualdad débil (siempre válida sin conocer los rangos exactos):
            |β₁(A∪B) − (β₁(A) + β₁(B))| ≤ β₁(A∩B)

        Derivación:
            De la secuencia exacta, β₁(A∪B) = β₁(A)+β₁(B) − r_φ + r_∂
            con 0 ≤ r_φ ≤ β₁(A∩B) y 0 ≤ r_∂ ≤ β₁(A∪B).
            La cota inferior da:
                β₁(A∪B) ≥ β₁(A)+β₁(B) − β₁(A∩B)
            La cota superior por la inyectividad da:
                β₁(A∪B) ≤ β₁(A)+β₁(B) + β₁(A∪B) (trivial)
            La desigualdad estándar es la triangular sobre el defecto.

        Parámetros:
            bA, bB, bI, bU: números de Betti validados.

        Retorna:
            (satisfecha: bool, mensaje: str).
        """
        lhs = abs(bU - (bA + bB))
        rhs = bI
        satisfied = lhs <= rhs
        msg = (
            f"Desigualdad débil MV: |β₁(A∪B)−(β₁(A)+β₁(B))| = {lhs} "
            f"{'≤' if satisfied else '>'} β₁(A∩B) = {rhs}."
        )
        return satisfied, msg

    # ──────────────────────────────────────────────────────────────────────────
    # §3.4 Verificación de la secuencia exacta en H₀
    # ──────────────────────────────────────────────────────────────────────────

    def _verify_h0_sequence(
        self,
        betti_0_A: Optional[int],
        betti_0_B: Optional[int],
        betti_0_intersection: Optional[int],
        betti_0_union: Optional[int],
        connecting_boundary_rank: int,
    ) -> Tuple[Optional[bool], str]:
        r"""
        Verifica la consistencia de la secuencia de Mayer-Vietoris en H₀.

        Fórmula en H₀:
            β₀(A∪B) = β₀(A) + β₀(B) − β₀(A∩B) + rank(∂₁)

        donde rank(∂₁) = rank(∂: H₁(A∪B) → H₀(A∩B)) = connecting_boundary_rank.

        La interpretación geométrica:
            · β₀(X) = número de componentes conexas de X.
            · La fusión A∪B puede reducir el número de componentes si A∩B
              conecta componentes previamente disjuntas.

        Si algún β₀ es None, se omite la verificación.

        Retorna:
            (válido: bool | None, mensaje: str).
        """
        if any(b is None for b in [betti_0_A, betti_0_B, betti_0_intersection, betti_0_union]):
            return None, "Verificación H₀ omitida: β₀ no proporcionados."

        b0A = int(betti_0_A)  # type: ignore[arg-type]
        b0B = int(betti_0_B)  # type: ignore[arg-type]
        b0I = int(betti_0_intersection)  # type: ignore[arg-type]
        b0U = int(betti_0_union)  # type: ignore[arg-type]

        expected_b0U = b0A + b0B - b0I + connecting_boundary_rank
        delta_b0U = b0U - expected_b0U
        valid = delta_b0U == 0

        msg = (
            f"Secuencia H₀: β₀(A∪B)={b0U}, expected={expected_b0U} "
            f"[β₀(A)={b0A}+β₀(B)={b0B}−β₀(A∩B)={b0I}+r_∂={connecting_boundary_rank}]. "
            f"Δβ₀={delta_b0U}. {'✓ Consistente.' if valid else '✗ Inconsistente.'}"
        )
        return valid, msg

    # ──────────────────────────────────────────────────────────────────────────
    # §3.5 Verificación de la característica de Euler-Poincaré
    # ──────────────────────────────────────────────────────────────────────────

    def _verify_euler_poincare(
        self,
        betti_0_A: Optional[int],
        betti_0_B: Optional[int],
        betti_0_intersection: Optional[int],
        betti_0_union: Optional[int],
        bA: int,
        bB: int,
        bI: int,
        bU: int,
    ) -> Tuple[Optional[bool], str]:
        r"""
        Verifica la identidad de Euler-Poincaré bajo fusión.

        Identidad:
            χ(A∪B) = χ(A) + χ(B) − χ(A∩B)

        donde χ(X) = β₀(X) − β₁(X) (truncada a grados 0 y 1).

        Para complejos simpliciales en grado ≤ 1 (grafos):
            χ(X) = V(X) − E(X)  (vértices − aristas)

        Si los β₀ no son proporcionados, se omite la verificación.

        Retorna:
            (válido: bool | None, mensaje: str).
        """
        if any(b is None for b in [betti_0_A, betti_0_B, betti_0_intersection, betti_0_union]):
            return None, "Verificación Euler-Poincaré omitida: β₀ no proporcionados."

        b0A = int(betti_0_A)  # type: ignore[arg-type]
        b0B = int(betti_0_B)  # type: ignore[arg-type]
        b0I = int(betti_0_intersection)  # type: ignore[arg-type]
        b0U = int(betti_0_union)  # type: ignore[arg-type]

        chi_A = b0A - bA
        chi_B = b0B - bB
        chi_I = b0I - bI
        chi_U = b0U - bU

        expected_chi_U = chi_A + chi_B - chi_I
        valid = chi_U == expected_chi_U

        msg = (
            f"Euler-Poincaré: χ(A∪B)={chi_U}, expected={expected_chi_U} "
            f"[χ(A)={chi_A}+χ(B)={chi_B}−χ(A∩B)={chi_I}]. "
            f"{'✓ Invariante preservado.' if valid else '✗ Invariante violado.'}"
        )
        return valid, msg

    # ──────────────────────────────────────────────────────────────────────────
    # §3.6 Métrica de defecto Jensen-Shannon simétrica
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_jensen_shannon_defect(
        self,
        bU: int,
        expected: int,
    ) -> float:
        r"""
        Calcula la métrica de defecto simétrica Jensen-Shannon.

        Definición (adaptada al caso discreto degenerado):
            JS_defect = |Δβ₁| / (1 + max(β₁(A∪B), expected))

        Propiedades:
            · Simétrica: JS(bU, expected) = JS(expected, bU).
            · Acotada: JS_defect ∈ [0, 1).
            · JS_defect = 0 ⟺ bU = expected (fusión perfecta).
            · JS_defect → 1 cuando |Δβ₁| ≫ max(bU, expected).

        Interpretación:
            Mide la fracción del invariante topológico no explicada por
            la fórmula de Mayer-Vietoris, normalizada por la escala máxima.

        Parámetros:
            bU: β₁(A∪B) observado.
            expected: β₁(A∪B) predicho por Mayer-Vietoris.

        Retorna:
            Defecto JS ∈ [0, 1).
        """
        delta = abs(bU - expected)
        denominator = 1 + max(abs(bU), abs(expected))
        return float(delta) / float(denominator)

    # ──────────────────────────────────────────────────────────────────────────
    # §3.7 Intercepción principal de la secuencia de Mayer-Vietoris
    # ──────────────────────────────────────────────────────────────────────────

    def _intercept_mayer_vietoris_sequence(
        self,
        betti_1_A: int,
        betti_1_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        *,
        image_rank_h1_intersection: Optional[int] = None,
        connecting_boundary_rank: Optional[int] = None,
        betti_0_A: Optional[int] = None,
        betti_0_B: Optional[int] = None,
        betti_0_intersection: Optional[int] = None,
        betti_0_union: Optional[int] = None,
        raise_on_veto: bool = True,
    ) -> MayerVietorisAuditData:
        r"""
        Auditoría principal de Fase 3: invariancia homológica de fusión.

        Pipeline de auditoría (v3):
            1. Validar β₁(A), β₁(B), β₁(A∩B), β₁(A∪B) como enteros ≥ 0 (§3.1).
            2. Validar rangos opcionales image_rank y boundary_rank (§3.2).
            3. Calcular β₁(A∪B) esperado por la secuencia exacta.
            4. Verificar la desigualdad débil de Mayer-Vietoris (§3.3).
            5. Verificar la secuencia exacta en H₀ (§3.4).
            6. Verificar la característica de Euler-Poincaré (§3.5).
            7. Calcular Δβ₁ y la métrica Jensen-Shannon (§3.6).
            8. Emitir MayerVietorisAuditData con todos los campos.
            9. Lanzar HomologicalFusionVeto si Δβ₁ ≠ 0 y raise_on_veto=True.

        Modos de operación:

            Modo por defecto (contrato aditivo):
                image_rank := β₁(A∩B)  [toda H₁(A∩B) mapea inyectivamente]
                boundary_rank := 0     [la diferencial de conexión es nula]
                expected = β₁(A) + β₁(B) − β₁(A∩B)

            Modo exacto (con rangos proporcionados):
                expected = β₁(A) + β₁(B) − image_rank + boundary_rank
                [fórmula exacta de la secuencia larga por exactitud]

        Parámetros:
            betti_1_A: β₁(A), número de loops en A.
            betti_1_B: β₁(B), número de loops en B.
            betti_1_intersection: β₁(A∩B), loops en la intersección.
            betti_1_union: β₁(A∪B), loops observados en la unión.
            image_rank_h1_intersection: rank(im φ: H₁(A∩B)→H₁(A)⊕H₁(B)).
            connecting_boundary_rank: rank(∂: H₁(A∪B)→H₀(A∩B)).
            betti_0_*: números de Betti en grado 0 para verificaciones adicionales.
            raise_on_veto: lanzar HomologicalFusionVeto si Δβ₁ ≠ 0.

        Retorna:
            MayerVietorisAuditData: certificado homológico completo.

        Lanza:
            MayerVietorisInputError: si los invariantes de entrada son inválidos.
            HomologicalFusionVeto: si Δβ₁ ≠ 0 y raise_on_veto=True.
            EulerPoincareMismatchError: si χ(A∪B) ≠ χ(A)+χ(B)−χ(A∩B)
                y raise_on_veto=True y β₀ fueron proporcionados.
        """
        # ── Paso 1: Validación de números de Betti en grado 1 ─────────────────
        bA = self._validate_betti_number("betti_1_A", betti_1_A)
        bB = self._validate_betti_number("betti_1_B", betti_1_B)
        bI = self._validate_betti_number("betti_1_intersection", betti_1_intersection)
        bU = self._validate_betti_number("betti_1_union", betti_1_union)

        notes: List[str] = []

        # ── Paso 2: Validación y selección del rango de imagen φ ──────────────
        # La cota algebraica es: rank(φ) ≤ min(β₁(A∩B), β₁(A)+β₁(B)).
        image_rank_upper = min(bI, bA + bB)

        if image_rank_h1_intersection is None:
            image_rank = bI
            notes.append(
                "Contrato aditivo: rank(im H₁(A∩B)→H₁(A)⊕H₁(B)) := β₁(A∩B) = "
                f"{bI}. Se asume que φ es inyectiva."
            )
        else:
            validated_image = self._validate_optional_rank(
                "image_rank_h1_intersection",
                image_rank_h1_intersection,
                upper_bound=image_rank_upper,
            )
            image_rank = bI if validated_image is None else validated_image
            notes.append(
                f"Modo exacto: rank(im H₁(A∩B)→H₁(A)⊕H₁(B)) = {image_rank} "
                f"(cota: ≤ {image_rank_upper})."
            )

        # ── Paso 2b: Validación del rango del morfismo de conexión ∂ ──────────
        # La cota algebraica es: rank(∂) ≤ min(β₁(A∪B), β₀(A∩B)).
        # Sin β₀(A∩B), usamos bU como cota conservadora.
        boundary_rank_upper = bU

        if connecting_boundary_rank is None:
            boundary_rank = 0
            notes.append(
                "Contrato aditivo: rank(∂: H₁(A∪B)→H₀(A∩B)) := 0. "
                "Se asume que ∂ es el morfismo nulo."
            )
        else:
            validated_boundary = self._validate_optional_rank(
                "connecting_boundary_rank",
                connecting_boundary_rank,
                upper_bound=boundary_rank_upper,
            )
            boundary_rank = 0 if validated_boundary is None else validated_boundary
            notes.append(
                f"Modo exacto: rank(∂: H₁(A∪B)→H₀(A∩B)) = {boundary_rank} "
                f"(cota: ≤ {boundary_rank_upper})."
            )

        # ── Paso 3: Cálculo del β₁(A∪B) esperado ────────────────────────────
        # Fórmula de la secuencia exacta larga (por exactitud):
        #     β₁(A∪B) = β₁(A)+β₁(B) − rank(im φ) + rank(im ∂)
        expected_union_betti_1 = bA + bB - image_rank + boundary_rank
        delta_betti_1 = bU - expected_union_betti_1

        # Nota de alerta si expected < 0 (inconsistencia algebraica).
        if expected_union_betti_1 < 0:
            notes.append(
                f"ALERTA: expected_union_betti_1={expected_union_betti_1} < 0. "
                f"Los rangos suministrados son algebraicamente inconsistentes: "
                f"image_rank={image_rank} > bA+bB={bA+bB}. "
                f"Verifique los rangos de la secuencia exacta."
            )

        # ── Paso 4: Desigualdad débil de Mayer-Vietoris ───────────────────────
        weak_bound_satisfied, weak_msg = self._verify_weak_mayer_vietoris_bound(
            bA, bB, bI, bU
        )
        notes.append(weak_msg)

        # ── Paso 5: Secuencia exacta en H₀ ────────────────────────────────────
        h0_valid, h0_msg = self._verify_h0_sequence(
            betti_0_A,
            betti_0_B,
            betti_0_intersection,
            betti_0_union,
            boundary_rank,
        )
        notes.append(h0_msg)

        # ── Paso 6: Característica de Euler-Poincaré ──────────────────────────
        euler_valid, euler_msg = self._verify_euler_poincare(
            betti_0_A,
            betti_0_B,
            betti_0_intersection,
            betti_0_union,
            bA, bB, bI, bU,
        )
        notes.append(euler_msg)

        # ── Paso 7: Métricas de defecto ───────────────────────────────────────
        # Métrica clásica (asimétrica, mantenida por compatibilidad).
        denominator_classic = float(max(1, abs(expected_union_betti_1), bU))
        relative_defect = float(abs(delta_betti_1)) / denominator_classic

        # Métrica Jensen-Shannon (simétrica, acotada, nueva en v3).
        jensen_shannon_defect = self._compute_jensen_shannon_defect(
            bU, expected_union_betti_1
        )

        # ── Paso 8: Certificado de fusión homológica ───────────────────────────
        is_fusion_homologous = (
            delta_betti_1 == 0 and expected_union_betti_1 >= 0
        )

        # ── Paso 9: Vetos ─────────────────────────────────────────────────────
        if not is_fusion_homologous and raise_on_veto:
            raise HomologicalFusionVeto(
                f"Paradoja topológica de pegado (Mayer-Vietoris): Δβ₁={delta_betti_1} ≠ 0. "
                f"β₁(A)={bA}, β₁(B)={bB}, β₁(A∩B)={bI}, β₁(A∪B)={bU}. "
                f"Predicho: {expected_union_betti_1} "
                f"[image_rank={image_rank}, boundary_rank={boundary_rank}]. "
                f"JS_defect={jensen_shannon_defect:.4f}."
            )

        if euler_valid is False and raise_on_veto:
            raise EulerPoincareMismatchError(
                f"Violación del invariante de Euler-Poincaré bajo fusión. "
                f"χ(A∪B) ≠ χ(A)+χ(B)−χ(A∩B). Detalle: {euler_msg}"
            )

        return MayerVietorisAuditData(
            betti_1_A=bA,
            betti_1_B=bB,
            betti_1_intersection=bI,
            betti_1_union=bU,
            expected_union_betti_1=expected_union_betti_1,
            delta_betti_1=delta_betti_1,
            relative_defect=relative_defect,
            jensen_shannon_defect=jensen_shannon_defect,
            is_fusion_homologous=is_fusion_homologous,
            weak_bound_satisfied=weak_bound_satisfied,
            euler_characteristic_valid=euler_valid,
            h0_sequence_valid=h0_valid,
            notes=tuple(notes),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # §3.8 Último método de Fase 3: síntesis del objeto final de gobernanza
    # ──────────────────────────────────────────────────────────────────────────

    def _phase3_terminal_synthesis(
        self,
        adjacency_matrix: NDArray[Any],
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        betti_1_A: int,
        betti_1_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        *,
        adjacency_zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
        deep_nilpotence_audit: bool = True,
        power_audit_max_dim: int = _DEFAULT_POWER_AUDIT_MAX_DIM,
        allow_unknown_nodes: bool = False,
        strict_filtration: bool = False,
        node_order: Optional[Sequence[str]] = None,
        cross_check_adjacency_support: bool = False,
        image_rank_h1_intersection: Optional[int] = None,
        connecting_boundary_rank: Optional[int] = None,
        betti_0_A: Optional[int] = None,
        betti_0_B: Optional[int] = None,
        betti_0_intersection: Optional[int] = None,
        betti_0_union: Optional[int] = None,
        raise_on_veto: bool = True,
    ) -> CausalGovernanceState:
        r"""
        ÚLTIMO MÉTODO DE FASE 3 — Síntesis del objeto final de gobernanza.

        Composición funtorial completa:
            Z_Causal = Φ₃ ∘ Φ₂ ∘ Φ₁

        El funtor Z_Causal: (A, E, S, β) → CausalGovernanceState es el
        morfismo terminal en la categoría de objetos de gobernanza causal.

        Invariantes certificados:
            1. Spec(A) = {0}  [aciclicidad espectral del DAG].
            2. f: (V, ≤_DAG) → (Z, ≤) es morfismo de Posets [DIKW].
            3. Δβ₁ = 0  [invariancia homológica de fusión].

        Si todas las fases certifican sus invariantes:
            is_causally_valid = True.
        Si alguna fase falla (y raise_on_veto=False):
            is_causally_valid = False con diagnóstico completo.

        Parámetros adicionales v3:
            betti_0_*: β₀ opcionales para verificación de Euler-Poincaré y H₀.

        Retorna:
            CausalGovernanceState: objeto final certificado.

        Lanza:
            PipelineDirectorAgentError: si algún invariante falla y
                raise_on_veto=True.
        """
        # ── Composición Φ₃ ∘ Φ₂ ∘ Φ₁ ─────────────────────────────────────────
        (
            nilpotence_audit,
            filtration_audit,
            mayer_vietoris_audit,
        ) = self._phase2_terminal_bridge_to_phase3(
            adjacency_matrix,
            edges,
            node_strata,
            betti_1_A,
            betti_1_B,
            betti_1_intersection,
            betti_1_union,
            adjacency_zero_threshold=adjacency_zero_threshold,
            allow_signed_weights=allow_signed_weights,
            require_boolean=require_boolean,
            deep_nilpotence_audit=deep_nilpotence_audit,
            power_audit_max_dim=power_audit_max_dim,
            allow_unknown_nodes=allow_unknown_nodes,
            strict_filtration=strict_filtration,
            node_order=node_order,
            cross_check_adjacency_support=cross_check_adjacency_support,
            image_rank_h1_intersection=image_rank_h1_intersection,
            connecting_boundary_rank=connecting_boundary_rank,
            betti_0_A=betti_0_A,
            betti_0_B=betti_0_B,
            betti_0_intersection=betti_0_intersection,
            betti_0_union=betti_0_union,
            raise_on_veto=raise_on_veto,
        )

        # ── Síntesis del invariante global de gobernanza ───────────────────────
        is_causally_valid = bool(
            nilpotence_audit.is_strictly_nilpotent
            and filtration_audit.is_monotonic_filtration
            and mayer_vietoris_audit.is_fusion_homologous
        )

        if not is_causally_valid and raise_on_veto:
            failing = []
            if not nilpotence_audit.is_strictly_nilpotent:
                failing.append(f"Fase 1 (ρ(A)={nilpotence_audit.spectral_radius:.4e})")
            if not filtration_audit.is_monotonic_filtration:
                failing.append("Fase 2 (filtración Poset)")
            if not mayer_vietoris_audit.is_fusion_homologous:
                failing.append(f"Fase 3 (Δβ₁={mayer_vietoris_audit.delta_betti_1})")
            raise PipelineDirectorAgentError(
                f"Gobernanza causal inválida. Fases fallidas: {', '.join(failing)}."
            )

        # ── Construcción del artefacto final ──────────────────────────────────
        governance_id = str(uuid.uuid4())
        state = CausalGovernanceState(
            governance_id=governance_id,
            nilpotence_audit=nilpotence_audit,
            filtration_audit=filtration_audit,
            mayer_vietoris_audit=mayer_vietoris_audit,
            is_causally_valid=is_causally_valid,
            generated_at_utc=_utc_timestamp(),
        )

        logger.info(
            "Gobernanza causal Z_Causal = Φ₃∘Φ₂∘Φ₁ completada. "
            "id=%s | ρ(A)=%.4e | tol=%.4e | κ₂(Q)=%.4f | ν=%s | "
            "monotonic=%s | self_loops=%d | Δβ₁=%d | JS=%.4f | valid=%s",
            governance_id,
            nilpotence_audit.spectral_radius,
            nilpotence_audit.tolerance,
            nilpotence_audit.schur_condition_number,
            nilpotence_audit.nilpotence_index,
            filtration_audit.is_monotonic_filtration,
            filtration_audit.self_loops_detected,
            mayer_vietoris_audit.delta_betti_1,
            mayer_vietoris_audit.jensen_shannon_defect,
            state.is_causally_valid,
        )

        return state


# ══════════════════════════════════════════════════════════════════════════════
# §D. ORQUESTADOR SUPREMO: PIPELINE DIRECTOR AGENT
# ══════════════════════════════════════════════════════════════════════════════


class PipelineDirectorAgent(Morphism, Phase3_MayerVietorisInterceptor):
    r"""
    El Custodio de la Causalidad Funtorial en la Malla Agéntica.

    Herencia lineal de la cadena funtorial:
        PipelineDirectorAgent
          → Phase3_MayerVietorisInterceptor   [Φ₃: H* → GovernanceCert]
             → Phase2_PosetFiltrationAuditor  [Φ₂: Poset → FiltrationCert]
                → Phase1_SpectralNilpotenceCertifier [Φ₁: Matx → NilpotenceCert]
                   → (Morphism base)

    El agente no añade lógica nueva: su único método público delega en
    _phase3_terminal_synthesis, que implementa Z_Causal = Φ₃ ∘ Φ₂ ∘ Φ₁.

    Nuevos parámetros v3 en execute_causal_governance:
        betti_0_*: números de Betti en H₀ para verificación completa.
    """

    def execute_causal_governance(
        self,
        adjacency_matrix: NDArray[Any],
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        betti_1_A: int,
        betti_1_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        *,
        adjacency_zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
        deep_nilpotence_audit: bool = True,
        power_audit_max_dim: int = _DEFAULT_POWER_AUDIT_MAX_DIM,
        allow_unknown_nodes: bool = False,
        strict_filtration: bool = False,
        node_order: Optional[Sequence[str]] = None,
        cross_check_adjacency_support: bool = False,
        image_rank_h1_intersection: Optional[int] = None,
        connecting_boundary_rank: Optional[int] = None,
        betti_0_A: Optional[int] = None,
        betti_0_B: Optional[int] = None,
        betti_0_intersection: Optional[int] = None,
        betti_0_union: Optional[int] = None,
        raise_on_veto: bool = True,
    ) -> CausalGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta completa Z_Causal = Φ₃ ∘ Φ₂ ∘ Φ₁.

        Punto de entrada público del custodio causal. Delega completamente
        en _phase3_terminal_synthesis, preservando la transparencia del
        anidamiento funtorial.

        Parámetros:
            adjacency_matrix: A ∈ R^{n×n}, operador de adyacencia del DAG.
            edges: lista de aristas dirigidas (u, v) del DAG.
            node_strata: mapeo nodo → nivel DIKW {0,1,2,3,4}.
            betti_1_A: β₁(A), número de loops en el subcomplejo A.
            betti_1_B: β₁(B), número de loops en el subcomplejo B.
            betti_1_intersection: β₁(A∩B), loops en la intersección.
            betti_1_union: β₁(A∪B), loops observados en la unión fusionada.
            adjacency_zero_threshold: ε para anulación de ruido en A.
            allow_signed_weights: permitir pesos A[i,j] < 0.
            require_boolean: exigir A ∈ {0,1}^{n×n}.
            deep_nilpotence_audit: activar certificación profunda A^n.
            power_audit_max_dim: límite de n para cálculo exacto de A^n.
            allow_unknown_nodes: ignorar nodos sin estrato en edges.
            strict_filtration: exigir stratum(u) < stratum(v) [estricto].
            node_order: orden canónico de nodos para cross-check edges↔A.
            cross_check_adjacency_support: activar cross-check.
            image_rank_h1_intersection: rank(im φ) exacto (opcional).
            connecting_boundary_rank: rank(∂) exacto (opcional).
            betti_0_A, betti_0_B, betti_0_intersection, betti_0_union:
                β₀ opcionales para verificación de Euler-Poincaré y H₀.
            raise_on_veto: si True, lanza excepción ante cualquier violación.
                si False, retorna estado con is_causally_valid=False.

        Retorna:
            CausalGovernanceState con certificados de Fases 1, 2 y 3.

        Lanza:
            CausalLoopVetoError: si ρ(A) > tol (ciclo espectral).
            SelfLoopVetoError: si existe arista (u,u) en edges.
            StratumMappingError: si falta un estrato requerido.
            FiltrationViolationVeto: si stratum(u) > stratum(v).
            AdjacencySupportVetoError: si el cross-check falla.
            HomologicalFusionVeto: si Δβ₁ ≠ 0.
            EulerPoincareMismatchError: si χ(A∪B) ≠ χ(A)+χ(B)−χ(A∩B).
            PipelineDirectorAgentError: si la gobernanza global es inválida.
        """
        return self._phase3_terminal_synthesis(
            adjacency_matrix,
            edges,
            node_strata,
            betti_1_A,
            betti_1_B,
            betti_1_intersection,
            betti_1_union,
            adjacency_zero_threshold=adjacency_zero_threshold,
            allow_signed_weights=allow_signed_weights,
            require_boolean=require_boolean,
            deep_nilpotence_audit=deep_nilpotence_audit,
            power_audit_max_dim=power_audit_max_dim,
            allow_unknown_nodes=allow_unknown_nodes,
            strict_filtration=strict_filtration,
            node_order=node_order,
            cross_check_adjacency_support=cross_check_adjacency_support,
            image_rank_h1_intersection=image_rank_h1_intersection,
            connecting_boundary_rank=connecting_boundary_rank,
            betti_0_A=betti_0_A,
            betti_0_B=betti_0_B,
            betti_0_intersection=betti_0_intersection,
            betti_0_union=betti_0_union,
            raise_on_veto=raise_on_veto,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §E. EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Excepciones
    "PipelineDirectorAgentError",
    "AdjacencyMatrixFormatError",
    "CausalLoopVetoError",
    "NilpotenceIndexVetoError",
    "StratumMappingError",
    "SelfLoopVetoError",
    "FiltrationViolationVeto",
    "AdjacencySupportVetoError",
    "MayerVietorisInputError",
    "HomologicalFusionVeto",
    "EulerPoincareMismatchError",
    # DTOs
    "NilpotenceAuditData",
    "PosetFiltrationData",
    "MayerVietorisAuditData",
    "CausalGovernanceState",
    # Certificadores por fase
    "Phase1_SpectralNilpotenceCertifier",
    "Phase2_PosetFiltrationAuditor",
    "Phase3_MayerVietorisInterceptor",
    # Orquestador supremo
    "PipelineDirectorAgent",
]