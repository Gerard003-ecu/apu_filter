# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Sheaf Cohomology Orchestrator (Interferómetro de Holonomía de Gauge)║
║ Ruta   : app/boole/strategy/sheaf_cohomology_orchestrator.py                 ║
║ Versión: 4.1.0-Sheaf-Hodge-Krylov-MoorePenrose-Heyting-Strict-PhD            ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TEORÍA DE HACES CELULARES (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra el **Sistema de Propiocepción Invariante** de la Malla
agéntica, modelando el consenso global y la consistencia lógica de las reglas
de negocio mediante la teoría de **Haces Celulares (Cellular Sheaves)**.
Repudia incondicionalmente las validaciones locales o heurísticas nodo a nodo,
elevando el flujo de restricciones a una estructura cohomológica global.

El enrutador trata las dependencias, políticas y restricciones de la obra
como secciones locales de un haz celular ℱ sobre el 1-esqueleto simplicial
del presupuesto (grafo G = (V,E)). Al medir el desacuerdo local a través de
los mapas de restricción, el orquestador deriva de forma determinista la
presencia de contradicciones contractuales o dependencias circulares
parasitarias. Toda obstrucción cohomológica excedente colapsa síncronamente
el retículo distributivo de Heyting Ω₃ en RAM, gatillando el disyuntor físico
Crowbar (GPIO14) en el milisegundo cero.

AXIOMÁTICA COHOMOLÓGICA, ENERGÍA DE DIRICHLET Y CONSISTENCIA DE HODGE:
────────────────────────────────────────────────────────────────────────────────

  [A1] El Fibrado Celular y el Operador Cofrontera (δ):
       Sea G = (V, E) el grafo de restricciones de la Malla. El haz celular
       ℱ asigna un stalk F(v) ≅ ℝ^{d_v} a cada vértice y F(e) ≅ ℝ^{d_e}
       a cada arista. El operador cofrontera global
       δ: C⁰(G; ℱ) → C¹(G; ℱ) mide la discrepancia local de las secciones
       x ∈ C⁰ mediante las matrices de restricción lineales F_{v ◁ e}:
       (δx)_e = F_{v ◁ e}(x_v) − F_{u ◁ e}(x_u)

  [A2] Teorema de Rango-Nulidad e Invariantes Cohomológicos (H⁰ y H¹):
       La consistencia global del sistema se extrae analíticamente
       resolviendo los subespacios cohomológicos del complejo de cocadenas:
       H⁰(G; ℱ) ≅ ker(δ)  ∧  H¹(G; ℱ) ≅ coker(δ) = C¹ / im(δ)
       Donde dim H⁰ = dim C⁰ − rank(δ) y dim H¹ = dim C¹ − rank(δ).
       Identidad de Euler del complejo de 2 términos:
       χ(ℱ) = dim H⁰ − dim H¹ = dim C⁰ − dim C¹.
       [AXIOMA DE VETO COHOMOLÓGICO]:
       dim H¹(G; ℱ) > 0 ⟹ VETO_ABSOLUTO

  [A3] Conservación del Número de Condición (Censura del Laplaciano):
       El Laplaciano del Haz se define formalmente como L = δᵀδ ⪰ 0.
       Para evitar la amplificación cuadrática del número de condición
       espectral (κ(L) = κ(δ)²) que colapsaría la mantisa flotante de la
       FPU (IEEE-754 binary64), el ensamblaje explícito de L queda
       estrictamente PROHIBIDO como objeto de Krylov. Toda iteración
       espectral se ejecuta directamente sobre δ mediante bidiagonalización
       de Golub-Kahan. La energía de Dirichlet E(x) = ‖δx‖² = xᵀLx se
       evalúa por producto matriz-vector sin materializar L.

  [A4] Proyección de Hodge-Helmholtz y Límite Isoperimétrico de Lipschitz:
       Si H¹ = 0 pero existe frustración térmica (E(x) > ε_frustration),
       el sistema calcula la proyección armónica de Hodge x* ∈ ker(δ) vía
       LSQR. Para evitar derivas contables ficticias, la distancia de
       sanación se somete estrictamente al límite de Lipschitz fuerte
       acotado por el número de condición de-confinado:
       ‖δx* − δx‖₂ ≤ κ(δ) · ‖x* − x‖₂
       Sujeto incondicionalmente a la cota isoperimétrica de inercia:
       ‖x − x*‖₂ ≤ Δ_inertia

  [A5] Estabilidad Espectral y Cota de Wilkinson:
       La exactitud de la base del núcleo se valida síncronamente contra
       la cota de precisión de máquina de Wilkinson para el rango de δ:
       rank(δ) = # { σᵢ ∈ σ(δ) | σᵢ > SVD_TOL }
       SVD_TOL = d² · κ₂(δ) · ε_machine · σ_max(δ)
       donde d = max(dim C⁰, dim C¹), con resolución iterativa del ciclo
       κ₂ ⟷ rank ⟷ SVD_TOL hasta punto fijo (≤ 8 iteraciones).
       En régimen disperso el rango se revela por el extremo inferior
       (svds which='SM') y no por los valores singulares dominantes.

  [A6] Álgebra de Heyting Ω₃ y Colapso Terminal:
       El veredicto de gobernanza habita el retículo distributivo de
       Heyting totalmente ordenado (álgebra de Gödel–Dummett):
       Ω₃ = { COHERENT := ⊥, DEGRADED, VETOED := ⊤ }
       con join = sup = max, meet = inf = min, y pseudocomplemento
       a → b = ⊤ si a ≤ b, y a → b = b en caso contrario.
       El colapso al supremo terminal ⊤ dispara el Crowbar GPIO14.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial de de Rham-Hodge):
────────────────────────────────────────────────────────────────────────────────
La orquestación del consenso global se rige por un acoplamiento monoidal
covariante estricto, encadenando DTOs inmutables de solo lectura (F1 ⊣ F2 ⊣ F3).
El tipo de retorno del último método de Φᵢ ES el objeto inicial de Φᵢ₊₁:

  Fase 1 ──► VETO COHOMOLÓGICO Y COFRONTERA (Phase1_CohomologicalVetoCertifier)
             Ingiere la matriz de incidencia de-confinada y los mapas de
             restricción. Construye el operador cofrontera global δ, calcula
             su SVD con tolerancia de Wilkinson adaptativa y audita
             (dim H⁰, dim H¹).
             Morfismo terminal: nest_into_phase2
                 CellularSheaf  ⟶  Phase2_KrylovSpectralAuditor

  Fase 2 ──► REGULACIÓN ESPECTRAL Y KRYLOV (Phase2_KrylovSpectralAuditor)
             Hereda CohomologicalVetoData. Mide κ(δ) mediante
             bidiagonalización de Golub-Kahan-Lanczos directamente sobre δ
             sin cuadrar el operador, calcula E(x) y la constante de Poincaré.
             Morfismo terminal: nest_into_phase3
                 (Phase2, x)  ⟶  Phase3_IsoperimetricHodgeProjector

  Fase 3 ──► PROYECCIÓN DE HODGE Y VETO HEYTING (Phase3_IsoperimetricHodgeProjector)
             Hereda KrylovSpectralData. Resuelve x* mediante LSQR, verifica
             el límite isoperimétrico de Lipschitz y resuelve el veredicto
             en Ω₃. Si el residuo excede la cota o dim H¹ > 0, el retículo
             colapsa a VETOED (⊤) y conmuta GPIO14.
             Morfismo terminal: resolve_sheaf_governance
                 (Phase3, x)  ⟶  SheafGovernanceState

Funtor Maestro de Propiocepción e Invarianza Global:
  𝒵_Sheaf = Φ₃ ∘ Φ₂ ∘ Φ₁ : Sheaf × C⁰(G; ℱ) ⟶ SheafGovernanceState
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr, svds

logger = logging.getLogger("MIC.ImmuneSystem.SheafCohomology")

__version__: Final[str] = "4.1.0-Sheaf-Hodge-Krylov-MoorePenrose-Heyting-Strict-PhD"


# =============================================================================
# SECCIÓN 1: TOLERANZAS NUMÉRICAS JUSTIFICADAS (FPU IEEE-754 binary64)
# =============================================================================

_EPS_MACHINE: Final[float] = float(np.finfo(np.float64).eps)  # ≈ 2.22e-16

# ε_mach^{2/3} ≈ 3.7e-11 es la cota estándar de residuales bien condicionados;
# 1e-9 aporta margen conservador frente a acumulación de redondeo en δx.
_FRUSTRATION_TOLERANCE: Final[float] = 1e-9

_SYMMETRY_TOLERANCE_ABS: Final[float] = 1e-10
_SYMMETRY_TOLERANCE_REL: Final[float] = 1e-10

_SPECTRAL_TOLERANCE: Final[float] = 1e-9

# Umbral a partir del cual la SVD densa O(n³) (LAPACK gesdd) es admisible.
_DENSE_SPECTRAL_MAX_DIM: Final[int] = 256

# Cota de valores singulares revelados en régimen disperso (ARPACK).
_SPARSE_MAX_SINGULAR_VALUES: Final[int] = 32

_ARPACK_TOLERANCE: Final[float] = 1e-7

_SEMIPOSITIVE_TOLERANCE_ABS: Final[float] = _SPECTRAL_TOLERANCE
_SEMIPOSITIVE_TOLERANCE_REL: Final[float] = 1e-8

_HODGE_SOLVER_TOLERANCE: Final[float] = 1e-10
_HODGE_MAX_ITER: Final[int] = 10_000

_KRYLOV_MAX_ITER: Final[int] = 200
_KRYLOV_TOL: Final[float] = 1e-9

_EPSILON: Final[float] = 1e-15

_CROWBAR_GPIO_PIN: Final[int] = 14

# κ₂ a partir del cual un mapa de restricción local degrada E(x).
_RESTRICTION_KAPPA_WARN: Final[float] = 1e8

# Códigos de parada de LSQR considerados convergentes (Paige–Saunders).
_LSQR_OK_STOP: Final[frozenset[int]] = frozenset({1, 2, 3, 4, 6})


# =============================================================================
# SECCIÓN 2: EXCEPCIONES ALGEBRAICAS
# =============================================================================


class SheafCohomologyError(Exception):
    """Excepción base para fallos en el análisis cohomológico del haz.

    Jerarquía:
        SheafCohomologyError
        ├── HomologicalInconsistencyError
        ├── SheafDegeneracyError
        ├── SpectralComputationError
        ├── TopologicalBifurcationError
        ├── HeytingCollapseError
        └── IsoperimetricViolationError
    """


class HomologicalInconsistencyError(SheafCohomologyError):
    """E(x) = ‖δx‖² > ε_frustration o dim H¹ > 0.

    Semántica [A2]: x no es sección global compatible, o ℱ posee
    obstrucción topológica irreducible (coker(δ) ≠ 0).
    """


class SheafDegeneracyError(SheafCohomologyError):
    """Haz algebraicamente incoherente o degenerado.

    Ejemplos: dimensiones incompatibles, NaN/±∞, grafo sin aristas (δ = 0),
    stalks sin soporte métrico, nnz de ensamblaje incoherente.
    """


class SpectralComputationError(SheafCohomologyError):
    """Fallo del cálculo espectral (Golub–Kahan / ARPACK / LAPACK).

    Ejemplos: no convergencia, pérdida de semi-positividad numérica.
    """


class TopologicalBifurcationError(SheafCohomologyError):
    """Una inyección altera estructuralmente la variedad (Δχ ≠ 0 o Δβ₁ > 0)."""


class HeytingCollapseError(SheafCohomologyError):
    """El retículo Ω₃ colapsa al supremo terminal VETOED."""


class IsoperimetricViolationError(SheafCohomologyError):
    """‖x − x*‖₂ > Δ_inertia o violación de la cota de Lipschitz [A4]."""


# =============================================================================
# SECCIÓN 3: ESTRUCTURAS DE DATOS INMUTABLES (DTOs GEOMÉTRICOS)
# =============================================================================


@dataclass(frozen=True, slots=True)
class RestrictionMap:
    """Mapa lineal de restricción F_{v ▷ e}: F(v) → F(e).

    Formalmente, para cada incidencia v ◁ e del haz celular ℱ, este objeto
    encapsula el morfismo ℝ^{d_v} → ℝ^{d_e} que define la restricción local.
    Es el generador de los bloques del operador cofrontera δ (Axioma [A1]).

    En la categoría Banach de espacios euclídeos de dimensión finita, la
    norma de operador coincide con el radio espectral de FᵀF:
        ‖F‖_{2→2} = σ_max(F),    ‖F⁺‖_{2→2} = 1 / σ_min⁺(F).

    Atributos
    ─────────
    matrix : np.ndarray (m, n), float64, write-protected
        Matriz F_{v▷e}, m = dim F(e), n = dim F(v).
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        try:
            M = np.array(self.matrix, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise SheafDegeneracyError(
                f"El mapa de restricción no es convertible a array float64: {exc}"
            ) from exc

        if M.ndim != 2:
            raise SheafDegeneracyError(
                f"El mapa de restricción debe ser 2D; recibido ndim={M.ndim}."
            )
        if M.shape[0] == 0 or M.shape[1] == 0:
            raise SheafDegeneracyError(f"Dimensión degenerada: shape={M.shape}.")
        if not np.all(np.isfinite(M)):
            n_bad = int(np.count_nonzero(~np.isfinite(M)))
            raise SheafDegeneracyError(
                f"{n_bad} entrada(s) no finita(s) en F_{{v▷e}}."
            )

        M.setflags(write=False)
        object.__setattr__(self, "matrix", M)

    @property
    def domain_dim(self) -> int:
        """Dimensión del dominio F(v): n (columnas)."""
        return int(self.matrix.shape[1])

    @property
    def codomain_dim(self) -> int:
        """Dimensión del codominio F(e): m (filas)."""
        return int(self.matrix.shape[0])

    @property
    def operator_norm(self) -> float:
        """‖F‖_{2→2} = σ_max(F) (norma de Banach de operador)."""
        s = np.linalg.svd(self.matrix, compute_uv=False)
        return float(s[0]) if s.size else 0.0

    @property
    def condition_number(self) -> float:
        """κ₂(F_{v▷e}) = σ_max / σ_min ∈ [1, +∞].

        κ₂ = +∞ si el mapa es numéricamente rank-deficiente (σ_min < ε).
        Un κ₂ ≫ 1 amplifica el error de redondeo en E(x) = ‖δx‖².
        """
        s = np.linalg.svd(self.matrix, compute_uv=False)
        if s.size == 0:
            return float("inf")
        s_max, s_min = float(s[0]), float(s[-1])
        if s_min < _EPSILON:
            return float("inf")
        return s_max / s_min

    @property
    def moore_penrose_pinv(self) -> np.ndarray:
        """Seudoinversa de Moore–Penrose F⁺ ∈ ℝ^{n×m}.

        Satisface las cuatro ecuaciones de Penrose:
            F F⁺ F = F,  F⁺ F F⁺ = F⁺,  (F F⁺)ᵀ = F F⁺,  (F⁺ F)ᵀ = F⁺ F.
        Interviene en la proyección ortogonal sobre ker(δ) cuando se
        materializa x ↦ (I − δ⁺δ)x en régimen denso.
        """
        return np.linalg.pinv(self.matrix, rcond=None)


@dataclass(frozen=True, slots=True)
class SheafEdge:
    """Descriptor inmutable de una arista orientada del haz.

    Orientación canónica u → v. Contribución al operador cofrontera [A1]:
        (δx)_e = F_{v ▷ e} x_v − F_{u ▷ e} x_u

    Atributos
    ─────────
    edge_id        : identificador único en {edge_dims}
    u              : nodo origen
    v              : nodo destino
    restriction_u  : F_{u ▷ e} : F(u) → F(e)
    restriction_v  : F_{v ▷ e} : F(v) → F(e)
    """

    edge_id: int
    u: int
    v: int
    restriction_u: RestrictionMap
    restriction_v: RestrictionMap


# =============================================================================
# SECCIÓN 4: HAZ CELULAR (CellularSheaf) — Fibrado sobre el 1-esqueleto
# =============================================================================


class CellularSheaf:
    """Haz celular ℱ sobre el 1-esqueleto simplicial de la Malla agéntica.

    ℱ es un funtor de la categoría de celdas (vértices y aristas de G) hacia
    Vect_ℝ. A cada v ∈ V asigna el stalk F(v) ≅ ℝ^{d_v}; a cada e ∈ E asigna
    F(e) ≅ ℝ^{d_e}; y a cada incidencia u ◁ e, v ◁ e los morfismos lineales
    F_{u ▷ e}, F_{v ▷ e}.

    Espacios de cocadenas
    ─────────────────────
    C⁰(G; ℱ) = ⊕_{v ∈ V} F(v),   dim C⁰ = Σ_v d_v
    C¹(G; ℱ) = ⊕_{e ∈ E} F(e),   dim C¹ = Σ_e d_e

    El operador cofrontera δ: C⁰ → C¹ se ensambla por bloques. El Laplaciano
    L = δᵀδ hereda simetría y semi-positividad, pero NUNCA es objeto de Krylov
    (Axioma [A3]).

    Invariantes de clase
    ────────────────────
    - Nodos indexados 0, …, num_nodes − 1.
    - edge_id únicos declarados en edge_dims.
    - Grafo simple: sin lazos ni multiaristas.
    - Dimensiones de restricciones consistentes con node_dims y edge_dims.
    - Caché de δ invalidada al añadir aristas.
    """

    def __init__(
        self,
        num_nodes: int,
        node_dims: Dict[int, int],
        edge_dims: Dict[int, int],
    ) -> None:
        """Inicializa el fibrado celular base.

        Args
        ────
        num_nodes : int > 0
        node_dims : {nodo: dim F(nodo)} para nodos ∈ {0, …, num_nodes − 1}
        edge_dims : {edge_id: dim F(edge)} para aristas declaradas

        Raises
        ──────
        SheafDegeneracyError
        """
        if not isinstance(num_nodes, int) or num_nodes <= 0:
            raise SheafDegeneracyError(
                f"num_nodes debe ser entero positivo; recibido={num_nodes!r}."
            )

        self._num_nodes: Final[int] = num_nodes
        self._node_dims: Final[Dict[int, int]] = self._validate_node_dims(
            node_dims, num_nodes
        )
        self._edge_dims: Final[Dict[int, int]] = self._validate_edge_dims(edge_dims)
        self._edges: List[SheafEdge] = []
        self._added_edge_ids: set[int] = set()
        self._added_node_pairs: set[frozenset[int]] = set()

        self._node_offsets: Final[np.ndarray] = self._compute_offsets(
            self._node_dims, self._num_nodes
        )
        self._edge_offsets: Final[Dict[int, int]] = self._compute_edge_offsets_static(
            self._edge_dims
        )
        self._total_node_dim: Final[int] = int(self._node_offsets[-1])
        self._total_edge_dim: Final[int] = int(sum(self._edge_dims.values()))
        self._cached_coboundary: Optional[sp.csc_matrix] = None

    # ── 4.1 Propiedades de solo lectura ─────────────────────────────────
    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def node_dims(self) -> Dict[int, int]:
        return dict(self._node_dims)

    @property
    def edge_dims(self) -> Dict[int, int]:
        return dict(self._edge_dims)

    @property
    def edges(self) -> List[SheafEdge]:
        return list(self._edges)

    @property
    def total_node_dim(self) -> int:
        return self._total_node_dim

    @property
    def total_edge_dim(self) -> int:
        return self._total_edge_dim

    @property
    def num_edges_added(self) -> int:
        return len(self._edges)

    @property
    def num_edges_expected(self) -> int:
        return len(self._edge_dims)

    @property
    def missing_edge_ids(self) -> List[int]:
        """edge_id declarados aún no insertados, ordenados."""
        return sorted(set(self._edge_dims.keys()) - self._added_edge_ids)

    @property
    def is_fully_assembled(self) -> bool:
        return not self.missing_edge_ids

    # ── 4.2 Validación estática ─────────────────────────────────────────
    @staticmethod
    def _validate_node_dims(
        node_dims: Dict[int, int], num_nodes: int
    ) -> Dict[int, int]:
        """Cada nodo ∈ [0, n) con dimensión entera estrictamente positiva."""
        if not isinstance(node_dims, dict):
            raise SheafDegeneracyError("node_dims debe ser dict {nodo: dim}.")
        expected = set(range(num_nodes))
        actual = set(node_dims.keys())
        missing = expected - actual
        if missing:
            raise SheafDegeneracyError(
                f"Faltan dimensiones para nodos: {sorted(missing)}."
            )
        extra = actual - expected
        if extra:
            raise SheafDegeneracyError(f"Claves fuera de rango: {sorted(extra)}.")
        validated: Dict[int, int] = {}
        for i in range(num_nodes):
            dim = node_dims[i]
            if not isinstance(dim, int) or dim <= 0:
                raise SheafDegeneracyError(
                    f"dim inválida para nodo {i}: {dim!r}. Entero positivo requerido."
                )
            validated[i] = dim
        return validated

    @staticmethod
    def _validate_edge_dims(edge_dims: Dict[int, int]) -> Dict[int, int]:
        """Cada arista con dimensión entera > 0; el diccionario no es vacío."""
        if not isinstance(edge_dims, dict):
            raise SheafDegeneracyError("edge_dims debe ser dict {edge_id: dim}.")
        if not edge_dims:
            raise SheafDegeneracyError(
                "edge_dims vacío. Un haz sin aristas tiene δ = 0 y "
                "H⁰ = C⁰ trivialmente, sin información inter-agente."
            )
        validated: Dict[int, int] = {}
        for edge_id, dim in edge_dims.items():
            if not isinstance(edge_id, int) or edge_id < 0:
                raise SheafDegeneracyError(
                    f"edge_id inválido: {edge_id!r}. Entero ≥ 0 requerido."
                )
            if not isinstance(dim, int) or dim <= 0:
                raise SheafDegeneracyError(
                    f"dim inválida para arista {edge_id}: {dim!r}."
                )
            validated[edge_id] = dim
        return validated

    @staticmethod
    def _compute_offsets(dims_map: Dict[int, int], count: int) -> np.ndarray:
        """offsets[k] = Σ_{i<k} d_i, offsets[count] = dim C⁰."""
        offsets = np.zeros(count + 1, dtype=np.int64)
        for i in range(count):
            offsets[i + 1] = offsets[i] + dims_map[i]
        return offsets

    @staticmethod
    def _compute_edge_offsets_static(edge_dims: Dict[int, int]) -> Dict[int, int]:
        """Offsets acumulados en C¹ indexados por edge_id (orden determinista)."""
        offsets: Dict[int, int] = {}
        running = 0
        for edge_id in sorted(edge_dims):
            offsets[edge_id] = running
            running += edge_dims[edge_id]
        return offsets

    # ── 4.3 Construcción del haz ────────────────────────────────────────
    def add_edge(
        self,
        edge_id: int,
        u: int,
        v: int,
        F_ue: RestrictionMap,
        F_ve: RestrictionMap,
    ) -> None:
        """Añade la arista e = (u → v) con sus mapas de restricción.

        Contribución a δ (convenio u → v):
            (δx)_e = F_{v ▷ e} x_v − F_{u ▷ e} x_u

        Precondiciones (en orden):
            1. edge_id ∈ edge_dims.
            2. edge_id no duplicado.
            3. u, v ∈ [0, num_nodes).
            4. u ≠ v (sin lazos).
            5. {u, v} no agregado previamente (grafo simple).
            6. F_ue.shape == (d_e, d_u).
            7. F_ve.shape == (d_e, d_v).

        Raises
        ──────
        SheafDegeneracyError
        """
        if edge_id not in self._edge_dims:
            raise SheafDegeneracyError(
                f"Arista {edge_id} no declarada en edge_dims: "
                f"{sorted(self._edge_dims.keys())}."
            )
        if edge_id in self._added_edge_ids:
            raise SheafDegeneracyError(f"Arista {edge_id} duplicada.")

        for label, node in (("u", u), ("v", v)):
            if not (0 <= node < self._num_nodes):
                raise SheafDegeneracyError(
                    f"Nodo {label}={node} fuera de [0, {self._num_nodes})."
                )
        if u == v:
            raise SheafDegeneracyError(
                f"Lazo en arista {edge_id}: (δx)_e = F·x_u − F·x_u = 0 trivial."
            )
        node_pair = frozenset({u, v})
        if node_pair in self._added_node_pairs:
            raise SheafDegeneracyError(
                f"Ya existe arista entre {u} y {v}. Para múltiples relaciones, "
                "incremente dim F(e)."
            )

        edge_dim = self._edge_dims[edge_id]
        expected_u = (edge_dim, self._node_dims[u])
        expected_v = (edge_dim, self._node_dims[v])
        if F_ue.matrix.shape != expected_u:
            raise SheafDegeneracyError(
                f"Arista {edge_id}: F_{{u▷e}} shape {F_ue.matrix.shape} ≠ {expected_u}."
            )
        if F_ve.matrix.shape != expected_v:
            raise SheafDegeneracyError(
                f"Arista {edge_id}: F_{{v▷e}} shape {F_ve.matrix.shape} ≠ {expected_v}."
            )

        for label, rm in (("F_ue", F_ue), ("F_ve", F_ve)):
            kappa = rm.condition_number
            if kappa > _RESTRICTION_KAPPA_WARN:
                logger.warning(
                    "Arista %d, %s: κ₂=%.3e > %.3e (degradación numérica posible).",
                    edge_id, label, kappa, _RESTRICTION_KAPPA_WARN,
                )

        self._edges.append(SheafEdge(edge_id, u, v, F_ue, F_ve))
        self._added_edge_ids.add(edge_id)
        self._added_node_pairs.add(node_pair)
        self._cached_coboundary = None
        logger.debug(
            "Arista %d añadida: (%d → %d), dim F(e)=%d.",
            edge_id, u, v, edge_dim,
        )

    # ── 4.4 Ensamblaje del operador cofrontera ──────────────────────────
    def _assert_fully_assembled(self) -> None:
        missing = self.missing_edge_ids
        if missing:
            raise SheafDegeneracyError(
                f"Haz incompleto. Faltan aristas: {missing}."
            )

    def build_coboundary_operator(self) -> sp.csc_matrix:
        """Ensambla δ: C⁰ → C¹ como matriz dispersa CSC.

        Para cada arista e = (u → v) introduce bloques:
            δ_e = [ −F_{u▷e} | +F_{v▷e} ]  ∈ ℝ^{d_e × (d_u + d_v)}

        Ensamblaje vectorizado por bloques con pre-asignación de COO y
        verificación de coherencia nnz real vs estimado.

        Returns
        ───────
        sp.csc_matrix (dim C¹, dim C⁰), float64

        Raises
        ──────
        SheafDegeneracyError
        """
        if self._cached_coboundary is not None:
            return self._cached_coboundary

        self._assert_fully_assembled()

        estimated_nnz = sum(
            self._edge_dims[e.edge_id] * (self._node_dims[e.u] + self._node_dims[e.v])
            for e in self._edges
        )
        data = np.empty(estimated_nnz, dtype=np.float64)
        row_idx = np.empty(estimated_nnz, dtype=np.int64)
        col_idx = np.empty(estimated_nnz, dtype=np.int64)
        ptr = 0

        for edge in self._edges:
            edge_row_off = self._edge_offsets[edge.edge_id]
            u_col_off = int(self._node_offsets[edge.u])
            v_col_off = int(self._node_offsets[edge.v])
            F_u = edge.restriction_u.matrix
            F_v = edge.restriction_v.matrix
            d_e, d_u = F_u.shape
            _, d_v = F_v.shape

            bsz_u = d_e * d_u
            rows_u, cols_u = np.meshgrid(
                np.arange(d_e, dtype=np.int64) + edge_row_off,
                np.arange(d_u, dtype=np.int64) + u_col_off,
                indexing="ij",
            )
            data[ptr: ptr + bsz_u] = (-F_u).ravel()
            row_idx[ptr: ptr + bsz_u] = rows_u.ravel()
            col_idx[ptr: ptr + bsz_u] = cols_u.ravel()
            ptr += bsz_u

            bsz_v = d_e * d_v
            rows_v, cols_v = np.meshgrid(
                np.arange(d_e, dtype=np.int64) + edge_row_off,
                np.arange(d_v, dtype=np.int64) + v_col_off,
                indexing="ij",
            )
            data[ptr: ptr + bsz_v] = F_v.ravel()
            row_idx[ptr: ptr + bsz_v] = rows_v.ravel()
            col_idx[ptr: ptr + bsz_v] = cols_v.ravel()
            ptr += bsz_v

        if ptr != estimated_nnz:
            raise SheafDegeneracyError(
                f"Ensamblaje incoherente: nnz real {ptr} ≠ estimado {estimated_nnz}."
            )

        delta = sp.csc_matrix(
            (data, (row_idx, col_idx)),
            shape=(self._total_edge_dim, self._total_node_dim),
            dtype=np.float64,
        )
        if delta.nnz > 0 and not np.all(np.isfinite(delta.data)):
            n_bad = int(np.count_nonzero(~np.isfinite(delta.data)))
            raise SheafDegeneracyError(
                f"δ ensamblada contiene {n_bad} valor(es) no finito(s)."
            )

        self._cached_coboundary = delta
        logger.debug(
            "δ ensamblado: shape=%s, nnz=%d, densidad=%.4f%%.",
            delta.shape, delta.nnz,
            100.0 * delta.nnz / max(1, int(delta.shape[0]) * int(delta.shape[1])),
        )
        return delta

    def compute_sheaf_laplacian(self) -> sp.csc_matrix:
        """Laplaciano L = δᵀδ (uso restringido: verificación algebraica).

        Propiedades (por construcción AᵀA):
          1. L = Lᵀ (simétrica).
          2. L ⪰ 0: xᵀLx = ‖δx‖² ≥ 0.
          3. ker(L) = ker(δ) = H⁰(G; ℱ).

        PROHIBIDO como objeto de Krylov (Axioma [A3]): κ(L) = κ(δ)².
        """
        delta = self.build_coboundary_operator()
        L = (delta.T @ delta).tocsc()
        if L.nnz > 0 and not np.all(np.isfinite(L.data)):
            raise SheafDegeneracyError(
                "L = δᵀδ contiene valores no finitos (overflow en entradas extremas)."
            )
        return L

    def holder_operator_norm_bound(self) -> float:
        """Cota de Hölder ‖δ‖₂ ≤ √(‖δ‖₁ ‖δ‖∞) sin SVD.

        Identidad clásica de álgebra de Banach para operadores matriciales.
        Útil como testigo barato de σ_max cuando ARPACK no está disponible.
        """
        delta = self.build_coboundary_operator()
        if delta.nnz == 0:
            return 0.0
        abs_data = np.abs(delta.data)
        # ‖δ‖₁ = max_j Σ_i |δ_ij|,  ‖δ‖∞ = max_i Σ_j |δ_ij|
        ones_row = np.ones(delta.shape[0], dtype=np.float64)
        ones_col = np.ones(delta.shape[1], dtype=np.float64)
        col_sums = np.abs(delta).T.dot(ones_row)
        row_sums = np.abs(delta).dot(ones_col)
        norm_1 = float(np.max(col_sums)) if col_sums.size else 0.0
        norm_inf = float(np.max(row_sums)) if row_sums.size else 0.0
        del abs_data
        return float(np.sqrt(max(norm_1, 0.0) * max(norm_inf, 0.0)))


# =============================================================================
# SECCIÓN 5: ACTUADOR FÍSICO CROWBAR (GPIO14) — morfismo de colapso
# =============================================================================


def actuate_crowbar_gpio14() -> bool:
    """Conmuta el disyuntor físico Crowbar en GPIO14 (BCM).

    Si RPi.GPIO no está disponible (simulación / CI), registra la actuación
    software y retorna False sin lanzar excepción, preservando el flujo
    terminal de Φ₃ y el fast-fail de [A2].

    Returns
    ───────
    True  si el hardware fue conmutado.
    False si sólo se simuló.
    """
    try:
        import RPi.GPIO as GPIO  # type: ignore

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(_CROWBAR_GPIO_PIN, GPIO.OUT)
        GPIO.output(_CROWBAR_GPIO_PIN, GPIO.HIGH)
        logger.critical("CROWBAR ACTUADO: GPIO%d conmutado a HIGH.", _CROWBAR_GPIO_PIN)
        return True
    except Exception as exc:
        logger.critical(
            "CROWBAR SIMULADO (GPIO%d inaccesible): %s.", _CROWBAR_GPIO_PIN, exc
        )
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# ████████████████  FASE 1: VETO COHOMOLÓGICO Y COFRONTERA  ████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
# Propósito: construir δ y certificar la estructura cohomológica sin cuadrar
# el espectro. El morfismo terminal `nest_into_phase2` produce el objeto
# inicial de la FASE 2 (Phase2_KrylovSpectralAuditor), de modo que
#     last(Φ₁)  =  unit(Φ₂).
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True, eq=False)
class CohomologicalVetoData:
    """DTO inmutable: certificación cohomológica de la FASE 1.

    Precondición estricta del constructor de Phase2_KrylovSpectralAuditor.
    Encapsula δ y el espectro singular con tolerancia de Wilkinson
    resuelta iterativamente (ciclo κ₂ ⟷ rank ⟷ SVD_TOL).

    Invariante de Euler del complejo de 2 términos (testigo algebraico):
        h0_dimension − h1_dimension  ==  dim_C0 − dim_C1.
    """

    sheaf: CellularSheaf
    delta: sp.csc_matrix
    singular_values: np.ndarray
    dim_C0: int
    dim_C1: int
    delta_rank: int
    h0_dimension: int
    h1_dimension: int
    sigma_max: float
    sigma_min_positive: float
    condition_number_delta: float
    wilkinson_tolerance: float
    veto_triggered: bool
    certification_hash: str
    euler_characteristic: int
    rank_is_certified: bool


class Phase1_CohomologicalVetoCertifier:
    """FASE 1 — VETO COHOMOLÓGICO Y COFRONTERA.

    Ingesta el grafo agéntico materializado como CellularSheaf, ensambla
    δ: C⁰ → C¹ y certifica el espectro singular con tolerancia de Wilkinson
    adaptativa (Axioma [A5]):
        SVD_TOL = d² · κ₂(δ) · ε_machine · σ_max(δ)
    resolviendo el ciclo κ₂ ⟷ rank ⟷ SVD_TOL por punto fijo.

    Cadena interna de morfismos:
        certify_sheaf_topology
            → assemble_coboundary
            → extract_singular_spectrum
            → resolve_wilkinson_fixed_point
            → compute_rank_nullity_invariants
            → emit CohomologicalVetoData
            → nest_into_phase2          ★ morfismo terminal = unidad de Φ₂
    """

    _WILKINSON_MAX_ITER: Final[int] = 8
    _WILKINSON_TOL: Final[float] = 1e-12

    # ────────────────────────────────────────────────────────────────────
    # 1.1 Validación topológica del haz
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def certify_sheaf_topology(sheaf: CellularSheaf) -> None:
        """Certifica invariantes de CellularSheaf sobre API pública.

        Verifica:
            · sheaf.is_fully_assembled
            · al menos una arista (δ no trivial)

        Raises
        ──────
        SheafDegeneracyError
        """
        if not sheaf.is_fully_assembled:
            raise SheafDegeneracyError(
                f"Haz no completamente ensamblado. Faltan {len(sheaf.missing_edge_ids)} "
                f"arista(s): {sheaf.missing_edge_ids}."
            )
        if sheaf.num_edges_added == 0:
            raise SheafDegeneracyError("Haz trivial: sin aristas, δ = 0.")

    # ────────────────────────────────────────────────────────────────────
    # 1.2 Ensamblaje de δ
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def assemble_coboundary(sheaf: CellularSheaf) -> sp.csc_matrix:
        """Ensambla δ vía CellularSheaf.build_coboundary_operator.

        Encapsula la complejidad del fibrado en un único morfismo trazable.
        """
        delta = sheaf.build_coboundary_operator()
        logger.info("[FASE 1] δ ensamblado: shape=%s, nnz=%d.", delta.shape, delta.nnz)
        return delta

    # ────────────────────────────────────────────────────────────────────
    # 1.3 Espectro singular (denso exacto / disperso rango-revelador)
    # ────────────────────────────────────────────────────────────────────
    @classmethod
    def extract_singular_spectrum(
        cls,
        delta: sp.csc_matrix,
    ) -> Tuple[np.ndarray, bool]:
        """Extrae σ(δ) descendente y un flag de certificación completa.

        Estrategia (Axioma [A5], preservando [A3]: nunca se forma L = δᵀδ):
          · Si max(m, n) ≤ _DENSE_SPECTRAL_MAX_DIM: SVD densa LAPACK (completa).
          · En otro caso: híbrido ARPACK
                which='LM'  → σ_max (norma de operador),
                which='SM'  → cluster inferior (nulidad numérica).
            El rango queda certificado ssi se reveló un hueco espectral
            alrededor de la semilla de Wilkinson o se cubrió min(m,n)−1
            valores. En caso contrario `rank_is_certified=False` y dim H¹
            se interpreta como cota inferior (veto conservador).

        Returns
        ───────
        (singular_values_desc, rank_is_certified)
        """
        m, n = int(delta.shape[0]), int(delta.shape[1])
        d = max(m, n)
        p = min(m, n)

        if p <= 0:
            return np.array([], dtype=np.float64), True

        if d <= _DENSE_SPECTRAL_MAX_DIM:
            s_all = np.linalg.svd(delta.toarray(), compute_uv=False)
            return np.sort(s_all)[::-1].astype(np.float64), True

        k_hi = min(_SPARSE_MAX_SINGULAR_VALUES, max(1, p - 1))
        try:
            s_lm = svds(
                delta, k=k_hi, which="LM",
                return_singular_vectors=False, tol=_ARPACK_TOLERANCE,
            )
        except Exception as exc:
            raise SpectralComputationError(
                f"SVD LM de δ falló (d={d}, k={k_hi}): {exc}."
            ) from exc

        k_lo = min(_SPARSE_MAX_SINGULAR_VALUES, max(1, p - 1))
        try:
            s_sm = svds(
                delta, k=k_lo, which="SM",
                return_singular_vectors=False, tol=_ARPACK_TOLERANCE,
            )
        except Exception as exc:
            logger.warning(
                "SVD SM de δ no convergió (%s); se degrada a espectro LM.", exc
            )
            s_sorted = np.sort(np.asarray(s_lm, dtype=np.float64))[::-1]
            return s_sorted, False

        fused = np.unique(
            np.round(
                np.concatenate(
                    [np.asarray(s_lm, dtype=np.float64),
                     np.asarray(s_sm, dtype=np.float64)]
                ),
                decimals=15,
            )
        )
        s_sorted = np.sort(fused)[::-1].astype(np.float64)
        # Certificación completa sólo si cubrimos casi todo el rango posible
        # o el cluster SM está separado del origen (rango pleno).
        rank_is_certified = (k_hi + k_lo >= p - 1) or (
            s_sorted.size > 0 and float(np.min(np.asarray(s_sm))) > _SPECTRAL_TOLERANCE
        )
        return s_sorted, bool(rank_is_certified)

    # ────────────────────────────────────────────────────────────────────
    # 1.4 Punto fijo de Wilkinson κ₂ ⟷ rank ⟷ SVD_TOL
    # ────────────────────────────────────────────────────────────────────
    @classmethod
    def resolve_wilkinson_fixed_point(
        cls,
        singular_values_desc: np.ndarray,
        dim_C0: int,
        dim_C1: int,
    ) -> Tuple[float, int, float, float]:
        """Resuelve SVD_TOL = d² · κ₂ · ε_mach · σ_max por punto fijo.

        Semilla clásica de Wilkinson:
            SVD_TOL⁽⁰⁾ = d · ε_mach · σ_max.
        Convergencia: |tol_{k+1} − tol_k| ≤ _WILKINSON_TOL · max(1, tol_k).

        Returns
        ───────
        (wilkinson_tolerance, delta_rank, sigma_min_positive, kappa)
        """
        if singular_values_desc.size == 0:
            return 0.0, 0, 0.0, float("inf")

        d = max(dim_C0, dim_C1, 1)
        sigma_max = float(singular_values_desc[0])
        tol = d * _EPS_MACHINE * max(sigma_max, _EPSILON)
        rank = int(np.sum(singular_values_desc > tol))
        rank = min(rank, min(dim_C0, dim_C1))
        sigma_min_pos = float(singular_values_desc[rank - 1]) if rank > 0 else 0.0
        kappa = sigma_max / sigma_min_pos if sigma_min_pos > _EPSILON else float("inf")

        for _ in range(cls._WILKINSON_MAX_ITER):
            kappa_factor = kappa if np.isfinite(kappa) else 1.0
            tol_new = (d ** 2) * kappa_factor * _EPS_MACHINE * sigma_max
            if tol_new <= _EPSILON:
                tol_new = _EPSILON
            rank_new = int(np.sum(singular_values_desc > tol_new))
            rank_new = min(rank_new, min(dim_C0, dim_C1))
            sigma_min_pos_new = (
                float(singular_values_desc[rank_new - 1]) if rank_new > 0 else 0.0
            )
            kappa_new = (
                sigma_max / sigma_min_pos_new
                if sigma_min_pos_new > _EPSILON else float("inf")
            )
            if abs(tol_new - tol) <= cls._WILKINSON_TOL * max(1.0, tol):
                return float(tol_new), int(rank_new), float(sigma_min_pos_new), float(kappa_new)
            tol, rank, sigma_min_pos, kappa = (
                tol_new, rank_new, sigma_min_pos_new, kappa_new
            )
        return float(tol), int(rank), float(sigma_min_pos), float(kappa)

    # ────────────────────────────────────────────────────────────────────
    # 1.5 Rango-nulidad e invariantes de Betti del haz
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_rank_nullity_invariants(
        dim_C0: int,
        dim_C1: int,
        delta_rank: int,
    ) -> Tuple[int, int, int]:
        """Aplica el teorema rango-nulidad al complejo de 2 términos [A2].

            dim H⁰ = dim ker(δ)   = dim C⁰ − rank(δ)
            dim H¹ = dim coker(δ) = dim C¹ − rank(δ)
            χ(ℱ)   = dim H⁰ − dim H¹ = dim C⁰ − dim C¹

        Returns
        ───────
        (h0, h1, euler_characteristic)
        """
        rank = max(0, min(int(delta_rank), dim_C0, dim_C1))
        h0 = dim_C0 - rank
        h1 = dim_C1 - rank
        euler = h0 - h1
        if euler != dim_C0 - dim_C1:
            raise SheafDegeneracyError(
                f"Identidad de Euler rota: χ={euler} ≠ dim C⁰−dim C¹="
                f"{dim_C0 - dim_C1} (rank={rank})."
            )
        return int(h0), int(h1), int(euler)

    # ────────────────────────────────────────────────────────────────────
    # 1.6 Firma SHA-256 determinista
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_certification_hash(
        shape: Tuple[int, int],
        nnz: int,
        singular_values: np.ndarray,
        delta_rank: int,
        wilkinson_tolerance: float,
    ) -> str:
        """Firma SHA-256 de (shape, nnz, rank, SVD_TOL, σ) en hex."""
        h = hashlib.sha256()
        h.update(f"{int(shape[0])}x{int(shape[1])}".encode())
        h.update(f"|nnz={int(nnz)}".encode())
        h.update(f"|rank={int(delta_rank)}".encode())
        h.update(f"|tol={float(wilkinson_tolerance):.17e}".encode())
        for s in np.asarray(singular_values, dtype=np.float64).ravel().tolist():
            h.update(f"|{float(s):.17e}".encode())
        return h.hexdigest()

    # ────────────────────────────────────────────────────────────────────
    # 1.7 Emisión del DTO de veto (pre-terminal)
    # ────────────────────────────────────────────────────────────────────
    @classmethod
    def certify_cohomological_veto_axiom(
        cls,
        sheaf: CellularSheaf,
    ) -> CohomologicalVetoData:
        """Certifica [A2] y emite CohomologicalVetoData (precondición de Φ₂).

        Cadena:
            CellularSheaf
                ──(certify_sheaf_topology)──▶  ✓
                ──(assemble_coboundary)─────▶  δ
                ──(extract_singular_spectrum)▶ σ(δ), rank_is_certified
                ──(resolve_wilkinson_fixed_point)▶ SVD_TOL, rank, κ₂
                ──(compute_rank_nullity_invariants)▶ dim H⁰, dim H¹, χ
                ──(emit_axiom_[A2]_veto)────▶ veto_triggered

        El veto se *emite* aquí; el aborto de la cadena se decide en el
        orquestador o en Φ₃ (colapso Heyting), preservando composicionalidad.

        Raises
        ──────
        SheafDegeneracyError
        SpectralComputationError
        """
        cls.certify_sheaf_topology(sheaf)
        delta = cls.assemble_coboundary(sheaf)

        s_desc, rank_is_certified = cls.extract_singular_spectrum(delta)
        s_desc = np.asarray(s_desc, dtype=np.float64)
        s_desc.setflags(write=False)

        dim_C1, dim_C0 = int(delta.shape[0]), int(delta.shape[1])
        wilkinson_tol, delta_rank, sigma_min_pos, kappa = (
            cls.resolve_wilkinson_fixed_point(s_desc, dim_C0, dim_C1)
        )
        h0, h1, euler = cls.compute_rank_nullity_invariants(
            dim_C0, dim_C1, delta_rank
        )
        veto_triggered = h1 > 0
        sigma_max = float(s_desc[0]) if s_desc.size else 0.0
        cert_hash = cls.compute_certification_hash(
            (dim_C1, dim_C0), int(delta.nnz), s_desc, delta_rank, wilkinson_tol
        )

        logger.info(
            "[FASE 1 ✓] CohomologicalVetoData: dim C⁰=%d, dim C¹=%d, "
            "rank(δ)=%d, dim H⁰=%d, dim H¹=%d, χ=%d, σ_max=%.6e, σ_min⁺=%.6e, "
            "κ₂(δ)=%.3e, SVD_TOL=%.3e, certified=%s, VETO=%s.",
            dim_C0, dim_C1, delta_rank, h0, h1, euler,
            sigma_max, sigma_min_pos, kappa, wilkinson_tol,
            rank_is_certified, veto_triggered,
        )
        return CohomologicalVetoData(
            sheaf=sheaf,
            delta=delta,
            singular_values=s_desc,
            dim_C0=dim_C0,
            dim_C1=dim_C1,
            delta_rank=delta_rank,
            h0_dimension=h0,
            h1_dimension=h1,
            sigma_max=sigma_max,
            sigma_min_positive=sigma_min_pos,
            condition_number_delta=kappa,
            wilkinson_tolerance=wilkinson_tol,
            veto_triggered=veto_triggered,
            certification_hash=cert_hash,
            euler_characteristic=euler,
            rank_is_certified=rank_is_certified,
        )

    # ────────────────────────────────────────────────────────────────────
    # 1.8 ★ MORFISMO TERMINAL DE LA FASE 1 ★
    #     Tipo de retorno = objeto inicial de la FASE 2.
    #     last(Φ₁) = unit(Φ₂) = Phase2_KrylovSpectralAuditor.
    # ────────────────────────────────────────────────────────────────────
    @classmethod
    def nest_into_phase2(
        cls,
        sheaf: CellularSheaf,
    ) -> "Phase2_KrylovSpectralAuditor":
        """★ MORFISMO TERMINAL DE LA FASE 1 / UNIDAD DE LA FASE 2 ★

        Composición estricta F₁ ⊣ F₂:
            nest_into_phase2  :=  Phase2_KrylovSpectralAuditor
                                  ∘ certify_cohomological_veto_axiom.

        El auditor de Krylov nace ya alimentado con CohomologicalVetoData;
        su constructor ES la continuación formal de este método.
        """
        phase1_data = cls.certify_cohomological_veto_axiom(sheaf)
        return Phase2_KrylovSpectralAuditor(phase1_data)


# ═══════════════════════════════════════════════════════════════════════════════
# ████████████████  FASE 2: REGULACIÓN ESPECTRAL Y KRYLOV  █████████████████████
# ═══════════════════════════════════════════════════════════════════════════════
# Continuación directa de Phase1_CohomologicalVetoCertifier.nest_into_phase2.
# El constructor de Phase2_KrylovSpectralAuditor ES el inicio formal de Φ₂.
# Toda medición espectral se ejecuta sobre δ (Golub–Kahan), jamás sobre L=δᵀδ.
# El morfismo terminal `nest_into_phase3` produce el objeto inicial de Φ₃.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True, eq=False)
class KrylovSpectralData:
    """DTO inmutable: auditoría espectral de la FASE 2.

    Precondición estricta del constructor de Phase3_IsoperimetricHodgeProjector.
    """

    phase1: CohomologicalVetoData
    krylov_singular_values: np.ndarray
    krylov_dimension: int
    krylov_residual: float
    kappa_delta_krylov: float
    dirichlet_energy: float
    residual_norm: float
    poincare_constant: float
    spectral_gap_L: float
    lipschitz_bound: float
    banach_holder_bound: float


class Phase2_KrylovSpectralAuditor:
    """FASE 2 — REGULACIÓN ESPECTRAL Y KRYLOV.

    ★ INICIO FORMAL = continuación del morfismo terminal de la FASE 1 ★

    Hereda CohomologicalVetoData. Mide κ₂(δ) y el espectro de δ vía
    bidiagonalización de Golub–Kahan–Lanczos con reortogonalización de
    Gram–Schmidt modificada, aplicada DIRECTAMENTE a δ (Axioma [A3]).

    Modelo (Golub–Kahan):
        δ ≈ U B Vᵀ,  B bidiagonal superior,  UᵀU = I_k,  VᵀV = I_k.
        Recurrencias:
            β_{j+1} u_{j+1} = δ v_j − α_j u_j
            α_{j+1} v_{j+1} = δᵀ u_{j+1} − β_{j+1} v_j
        σ(B) aproxima los extremos de σ(δ) con convergencia de Kaniel–Paige.

    Cadena interna:
        __init__(CohomologicalVetoData)     ← unidad heredada de Φ₁
            → golub_kahan_lanczos_bidiagonalization
            → measure_condition_number_krylov
            → evaluate_dirichlet_energy
            → evaluate_poincare_constant
            → holder_norm_crosscheck
            → audit_krylov_spectral_stability
            → nest_into_phase3              ★ morfismo terminal = unidad de Φ₃
    """

    def __init__(self, phase1_certification: CohomologicalVetoData) -> None:
        """★ CONTINUACIÓN DE FASE 1 / INICIO DE FASE 2 ★

        Args
        ────
        phase1_certification : CohomologicalVetoData
            Salida de `certify_cohomological_veto_axiom`, inyectada por
            `nest_into_phase2`. Sin este objeto Φ₂ carece de δ certificado.
        """
        if not isinstance(phase1_certification, CohomologicalVetoData):
            raise TypeError(
                "Phase2_KrylovSpectralAuditor requiere CohomologicalVetoData "
                "como precondición (fase 1)."
            )
        self._p1: Final[CohomologicalVetoData] = phase1_certification
        self._delta: Final[sp.csc_matrix] = phase1_certification.delta
        self._sheaf: Final[CellularSheaf] = phase1_certification.sheaf

    @property
    def phase1(self) -> CohomologicalVetoData:
        """Referencia inmutable al DTO de la FASE 1."""
        return self._p1

    # ────────────────────────────────────────────────────────────────────
    # 2.1 Semilla determinista de Krylov (Rademacher vía SHA-256)
    # ────────────────────────────────────────────────────────────────────
    def _deterministic_start_vector(self, n: int) -> np.ndarray:
        """Vector de Rademacher ±1 derivado del certification_hash de Φ₁.

        Garantiza reproducibilidad bit a bit del subespacio de Krylov entre
        ejecuciones, sin comprometer la densificación espectral (el hash
        ya cifra σ(δ), de modo que la semilla no es independiente del
        operador, pero es estable).
        """
        digest = self._p1.certification_hash.encode("ascii")
        # Extiende el digest hasta cubrir n bytes.
        buf = bytearray()
        counter = 0
        while len(buf) < n:
            buf.extend(hashlib.sha256(digest + counter.to_bytes(4, "little")).digest())
            counter += 1
        signs = np.frombuffer(bytes(buf[:n]), dtype=np.uint8).astype(np.float64)
        v = np.where(signs >= 128, 1.0, -1.0)
        nrm = float(np.linalg.norm(v))
        if nrm <= _EPSILON:
            v = np.zeros(n, dtype=np.float64)
            v[0] = 1.0
            return v
        return v / nrm

    # ────────────────────────────────────────────────────────────────────
    # 2.2 Bidiagonalización Golub–Kahan–Lanczos sobre δ (sin L)
    # ────────────────────────────────────────────────────────────────────
    def golub_kahan_lanczos_bidiagonalization(
        self,
        k: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Bidiagonalización de Golub–Kahan–Lanczos de δ con MGS completo.

        Calcula hasta k valores singulares de Ritz de δ sin materializar δᵀδ.
        Ante breakdown afortunado (α_j o β_j ≈ 0) se detiene y reporta el
        subespacio invariante exacto. Si el proceso numérico falla, degrada
        a `scipy.sparse.linalg.svds` (ARPACK sobre el operador aumentado
        [0 δ; δᵀ 0], que tampoco cuadra L).

        Returns
        ───────
        (singular_values, krylov_dimension, krylov_residual)
        """
        m, n = int(self._delta.shape[0]), int(self._delta.shape[1])
        if m == 0 or n == 0:
            return np.array([], dtype=np.float64), 0, 0.0

        dim_min = min(m, n)
        if k is None:
            k = min(_SPARSE_MAX_SINGULAR_VALUES, max(1, dim_min - 1 if dim_min > 1 else 1))
        k = int(max(1, min(k, dim_min)))
        if tol is None:
            tol = _KRYLOV_TOL

        try:
            return self._golub_kahan_core(k, float(tol), m, n)
        except Exception as exc:
            logger.warning(
                "Golub–Kahan nativo falló (%s); degradación a svds LM.", exc
            )
            return self._svds_fallback(k, float(tol), m, n)

    def _golub_kahan_core(
        self,
        k: int,
        tol: float,
        m: int,
        n: int,
    ) -> Tuple[np.ndarray, int, float]:
        """Núcleo GK con reortogonalización MGS (O(k²(m+n)) + k matvecs)."""
        delta = self._delta
        V = np.zeros((n, k), dtype=np.float64)
        U = np.zeros((m, k), dtype=np.float64)
        alphas = np.zeros(k, dtype=np.float64)
        betas = np.zeros(k, dtype=np.float64)  # betas[j] = β_{j} (β₀ = 0)

        v = self._deterministic_start_vector(n)
        u = delta.dot(v)
        alpha = float(np.linalg.norm(u))
        if alpha <= tol:
            # Semilla en ker(δ) numérico: σ_max observado ≈ 0.
            return np.array([alpha], dtype=np.float64), 1, alpha

        u /= alpha
        U[:, 0] = u
        V[:, 0] = v
        alphas[0] = alpha
        effective = 1
        last_off = 0.0

        for j in range(k - 1):
            r = delta.T.dot(U[:, j]) - alphas[j] * V[:, j]
            for i in range(j + 1):
                r = r - np.dot(V[:, i], r) * V[:, i]
            beta = float(np.linalg.norm(r))
            betas[j + 1] = beta
            last_off = beta
            if beta <= tol:
                break
            v = r / beta
            V[:, j + 1] = v

            p = delta.dot(v) - beta * U[:, j]
            for i in range(j + 1):
                p = p - np.dot(U[:, i], p) * U[:, i]
            alpha = float(np.linalg.norm(p))
            alphas[j + 1] = alpha
            effective = j + 2
            if alpha <= tol:
                if alpha > _EPSILON:
                    U[:, j + 1] = p / alpha
                break
            U[:, j + 1] = p / alpha

        B = np.diag(alphas[:effective])
        if effective > 1:
            B += np.diag(betas[1:effective], 1)
        s = np.linalg.svd(B, compute_uv=False)
        s_sorted = np.sort(np.asarray(s, dtype=np.float64))[::-1]
        residual = float(last_off) if last_off > 0.0 else (
            float(abs(s_sorted[-1] - s_sorted[-2])) if s_sorted.size >= 2 else float(s_sorted[-1])
        )
        return s_sorted, int(effective), residual

    def _svds_fallback(
        self,
        k: int,
        tol: float,
        m: int,
        n: int,
    ) -> Tuple[np.ndarray, int, float]:
        """Degradación ARPACK (operador aumentado; no forma L)."""
        dim_min = min(m, n)
        k_eff = max(1, min(k, dim_min - 1)) if dim_min > 1 else 1
        try:
            s = svds(
                self._delta,
                k=k_eff,
                which="LM",
                return_singular_vectors=False,
                tol=tol,
                maxiter=_KRYLOV_MAX_ITER,
            )
        except Exception as exc:
            raise SpectralComputationError(
                f"Golub–Kahan–Lanczos/svds no convergió (k={k_eff}, tol={tol}): {exc}."
            ) from exc
        s_sorted = np.sort(np.asarray(s, dtype=np.float64))[::-1]
        residual = (
            float(abs(s_sorted[-1] - s_sorted[-2]))
            if s_sorted.size >= 2 else float(s_sorted[-1] if s_sorted.size else 0.0)
        )
        return s_sorted, int(s_sorted.size), residual

    # ────────────────────────────────────────────────────────────────────
    # 2.3 κ₂(δ) sin cuadrar el operador
    # ────────────────────────────────────────────────────────────────────
    def measure_condition_number_krylov(self) -> float:
        """κ₂(δ) = σ_max / σ_min⁺ sin materializar L = δᵀδ.

        σ_max se toma del Golub–Kahan nativo (which implícito LM).
        σ_min⁺ se sondea con svds which='SM' sobre δ (operador aumentado).
        Fallback determinista: valor certificado por la SVD de Φ₁.
        """
        m, n = int(self._delta.shape[0]), int(self._delta.shape[1])
        dim_min = min(m, n)
        if dim_min <= 1:
            return self._p1.condition_number_delta

        s_hi, _, _ = self.golub_kahan_lanczos_bidiagonalization(
            k=min(2, dim_min - 1), tol=_KRYLOV_TOL
        )
        sigma_max = float(s_hi[0]) if s_hi.size else self._p1.sigma_max

        try:
            s_lo = svds(
                self._delta,
                k=1,
                which="SM",
                return_singular_vectors=False,
                tol=_KRYLOV_TOL,
                maxiter=_KRYLOV_MAX_ITER,
            )
            sigma_min_pos = float(np.asarray(s_lo).ravel()[0]) if np.size(s_lo) else 0.0
        except Exception:
            sigma_min_pos = self._p1.sigma_min_positive

        if sigma_min_pos <= _EPSILON:
            return float("inf")
        return float(sigma_max / sigma_min_pos)

    # ────────────────────────────────────────────────────────────────────
    # 2.4 Energía de Dirichlet E(x) = ‖δx‖² (sin L)
    # ────────────────────────────────────────────────────────────────────
    def evaluate_dirichlet_energy(self, x: np.ndarray) -> Tuple[float, float]:
        """E(x) = ‖δx‖₂² y ‖δx‖₂ por un único matvec (Axioma [A3]).

        Implementación Banach: r = δx ∈ C¹,  E = ⟨r, r⟩_{C¹},  ‖r‖ = √E.
        Cualquier energía negativa por debajo de ε se recorta a 0
        (artefacto de redondeo; L ⪰ 0 impide E < 0 analíticamente).
        """
        x_ = np.asarray(x, dtype=np.float64).reshape(-1)
        residual = self._delta.dot(x_)
        energy = float(np.dot(residual, residual))
        residual_norm = float(np.linalg.norm(residual))
        if energy < 0.0 and abs(energy) <= _FRUSTRATION_TOLERANCE:
            energy = 0.0
        return energy, max(residual_norm, 0.0)

    # ────────────────────────────────────────────────────────────────────
    # 2.5 Constante de Poincaré discreta del haz
    # ────────────────────────────────────────────────────────────────────
    def evaluate_poincare_constant(self) -> float:
        """μ = λ₁⁺(L) / dim C⁰ = (σ_min⁺(δ))² / dim C⁰.

        Desigualdad de Poincaré del haz: para todo x ⊥ ker(δ),
            λ₁⁺(L) · ‖x‖₂²  ≤  ‖δx‖₂².
        μ cuantifica la densidad espectral del consenso: a mayor μ,
        más rápida es la convergencia de la proyección de Hodge a ker(δ).
        """
        dim_C0 = max(1, self._p1.dim_C0)
        gap_L = float(self._p1.sigma_min_positive) ** 2
        return gap_L / float(dim_C0)

    # ────────────────────────────────────────────────────────────────────
    # 2.6 Testigo de Hölder (álgebra de Banach) contra σ_max
    # ────────────────────────────────────────────────────────────────────
    def holder_norm_crosscheck(self) -> float:
        """√(‖δ‖₁ ‖δ‖∞) ≥ ‖δ‖₂ = σ_max. Testigo barato, nunca Krylov sobre L."""
        return self._sheaf.holder_operator_norm_bound()

    # ────────────────────────────────────────────────────────────────────
    # 2.7 Auditoría espectral (pre-terminal de Φ₂)
    # ────────────────────────────────────────────────────────────────────
    def audit_krylov_spectral_stability(self, x: np.ndarray) -> KrylovSpectralData:
        """Produce KrylovSpectralData, precondición estricta de Φ₃.

        Raises
        ──────
        SpectralComputationError
        SheafDegeneracyError
        """
        x_ = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_.shape[0] != self._p1.dim_C0:
            raise SheafDegeneracyError(
                f"x tiene longitud {x_.shape[0]}, esperada {self._p1.dim_C0}."
            )
        if not np.all(np.isfinite(x_)):
            raise SheafDegeneracyError("x contiene NaN/±∞.")

        s_krylov, krylov_dim, krylov_res = self.golub_kahan_lanczos_bidiagonalization()
        kappa_krylov = self.measure_condition_number_krylov()
        energy, residual_norm = self.evaluate_dirichlet_energy(x_)
        poincare = self.evaluate_poincare_constant()
        holder = self.holder_norm_crosscheck()

        # Cota de Lipschitz precomputada para Φ₃: factor κ(δ).
        # La distancia ‖x* − x‖ la aporta la proyección de Hodge.
        lipschitz_bound = kappa_krylov if np.isfinite(kappa_krylov) else 1e16
        spectral_gap_L = float(self._p1.sigma_min_positive ** 2)

        logger.info(
            "[FASE 2 ✓] KrylovSpectralData: E(x)=%.6e, ‖δx‖=%.6e, "
            "κ₂(δ)=%.3e, μ=%.6e, λ₁(L)=%.6e, Krylov_dim=%d, resid=%.3e, "
            "Hölder=%.3e.",
            energy, residual_norm, kappa_krylov, poincare,
            spectral_gap_L, krylov_dim, krylov_res, holder,
        )

        s_krylov_immut = np.asarray(s_krylov, dtype=np.float64).copy()
        s_krylov_immut.setflags(write=False)
        return KrylovSpectralData(
            phase1=self._p1,
            krylov_singular_values=s_krylov_immut,
            krylov_dimension=krylov_dim,
            krylov_residual=krylov_res,
            kappa_delta_krylov=kappa_krylov,
            dirichlet_energy=energy,
            residual_norm=residual_norm,
            poincare_constant=poincare,
            spectral_gap_L=spectral_gap_L,
            lipschitz_bound=lipschitz_bound,
            banach_holder_bound=holder,
        )

    # ────────────────────────────────────────────────────────────────────
    # 2.8 ★ MORFISMO TERMINAL DE LA FASE 2 ★
    #     Tipo de retorno = objeto inicial de la FASE 3.
    #     last(Φ₂) = unit(Φ₃) = Phase3_IsoperimetricHodgeProjector.
    # ────────────────────────────────────────────────────────────────────
    def nest_into_phase3(
        self,
        x: np.ndarray,
    ) -> "Phase3_IsoperimetricHodgeProjector":
        """★ MORFISMO TERMINAL DE LA FASE 2 / UNIDAD DE LA FASE 3 ★

        Composición estricta F₂ ⊣ F₃:
            nest_into_phase3(x)  :=  Phase3_IsoperimetricHodgeProjector
                                     ∘ audit_krylov_spectral_stability(x).

        El projector de Hodge nace ya alimentado con KrylovSpectralData;
        su constructor ES la continuación formal de este método.
        """
        phase2_data = self.audit_krylov_spectral_stability(x)
        return Phase3_IsoperimetricHodgeProjector(phase2_data)


# ═══════════════════════════════════════════════════════════════════════════════
# ████████████████  FASE 3: PROYECCIÓN DE HODGE Y VETO HEYTING  ████████████████
# ═══════════════════════════════════════════════════════════════════════════════
# Continuación directa de Phase2_KrylovSpectralAuditor.nest_into_phase3.
# El constructor de Phase3_IsoperimetricHodgeProjector ES el inicio formal de Φ₃.
# Morfismo terminal del módulo: resolve_sheaf_governance → SheafGovernanceState.
# ═══════════════════════════════════════════════════════════════════════════════


class HeytingTop(IntEnum):
    """Retículo distributivo de Heyting totalmente ordenado Ω₃ (Gödel–Dummett).

        Ω₃ = { COHERENT := ⊥, DEGRADED, VETOED := ⊤ }
        Orden:      COHERENT < DEGRADED < VETOED
        Join (⊔):   max
        Meet (⊓):   min
        Implicación (cadena finita):
            a → b  =  ⊤  si a ≤ b,
            a → b  =  b  si a > b.
        Negación de Heyting: ¬a = a → ⊥.

    Tabla de a → b sobre {0,1,2}:

            b\\a   0   1   2
             0     2   0   0
             1     2   2   1
             2     2   2   2
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def __le__(self, other: "HeytingTop") -> bool:
        return int(self) <= int(other)

    def join(self, other: "HeytingTop") -> "HeytingTop":
        """Supremo en Ω₃ = max."""
        return HeytingTop(max(int(self), int(other)))

    def meet(self, other: "HeytingTop") -> "HeytingTop":
        """Ínfimo en Ω₃ = min."""
        return HeytingTop(min(int(self), int(other)))

    def implication(self, other: "HeytingTop") -> "HeytingTop":
        """Implicación de Gödel: a → b = ⊤ si a ≤ b, else b."""
        if int(self) <= int(other):
            return HeytingTop.VETOED
        return other

    def negation(self) -> "HeytingTop":
        """¬a = a → ⊥."""
        return self.implication(HeytingTop.COHERENT)


@dataclass(frozen=True, slots=True, eq=False)
class SheafGovernanceState:
    """DTO terminal del funtor maestro 𝒵_Sheaf (salida del módulo)."""

    verdict: HeytingTop
    h0_dimension: int
    h1_dimension: int
    frustration_energy_before: float
    frustration_energy_after: float
    hodge_correction_norm: float
    residual_correction_norm: float
    lipschitz_lhs: float
    lipschitz_slack: float
    isoperimetric_slack: float
    kappa_delta: float
    crowbar_actuated: bool
    certification_hash: str
    euler_characteristic: int


class Phase3_IsoperimetricHodgeProjector:
    """FASE 3 — PROYECCIÓN DE HODGE Y VETO HEYTING.

    ★ INICIO FORMAL = continuación del morfismo terminal de la FASE 2 ★

    Hereda KrylovSpectralData. Sobre él ejecuta:

        1. hodge_project(x): resuelve
               min_z ‖δ z − δx‖₂    ⇒    x* = x − z ∈ ker(δ) + error,
           vía LSQR (Paige–Saunders) sin ensamblar L. Equivale a
               x* = (I − δ⁺ δ) x     (proyección ortogonal sobre ker(δ)).

        2. verify_lipschitz_isoperimetric_bound: comprueba [A4]
               ‖δx* − δx‖₂ ≤ κ(δ) · ‖x* − x‖₂
               ‖x − x*‖₂   ≤ Δ_inertia.

        3. resolve_heyting_verdict: colapsa Ω₃.

        4. actuate_crowbar_gpio14: si VETOED, conmuta GPIO14.

    Cadena interna:
        __init__(KrylovSpectralData)        ← unidad heredada de Φ₂
            → hodge_project
            → verify_lipschitz_isoperimetric_bound
            → resolve_heyting_verdict
            → actuate_crowbar_gpio14
            → resolve_sheaf_governance      ★ morfismo terminal del módulo
    """

    def __init__(self, phase2_audit: KrylovSpectralData) -> None:
        """★ CONTINUACIÓN DE FASE 2 / INICIO DE FASE 3 ★

        Args
        ────
        phase2_audit : KrylovSpectralData
            Salida de `audit_krylov_spectral_stability`, inyectada por
            `nest_into_phase3`.
        """
        if not isinstance(phase2_audit, KrylovSpectralData):
            raise TypeError(
                "Phase3_IsoperimetricHodgeProjector requiere KrylovSpectralData "
                "como precondición (fase 2)."
            )
        self._p2: Final[KrylovSpectralData] = phase2_audit
        self._delta: Final[sp.csc_matrix] = phase2_audit.phase1.delta
        self._sheaf: Final[CellularSheaf] = phase2_audit.phase1.sheaf

    @property
    def phase2(self) -> KrylovSpectralData:
        return self._p2

    # ────────────────────────────────────────────────────────────────────
    # 3.1 Proyección armónica de Hodge vía LSQR
    # ────────────────────────────────────────────────────────────────────
    def hodge_project(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Proyección armónica x* ∈ ker(δ) ∩ (x + im(δᵀ)).

        Teorema de Hodge discreto sobre un complejo de 2 términos:
            C⁰ = im(δᵀ) ⊕ ker(δ),
            x* = x − δᵀ (δδᵀ)⁺ δx = (I − δ⁺δ) x.

        Implementación: LSQR sobre δ z = δx (mínima norma), x* = x − z.
        Si ‖δx‖ ≤ √ε, x ya es numéricamente armónico y se omite el solve.

        Returns
        ───────
        (x_star, energy_before, delta_x, delta_x_star)
        """
        x_ = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_.shape[0] != self._p2.phase1.dim_C0:
            raise SheafDegeneracyError(
                f"x tiene longitud {x_.shape[0]}, esperada {self._p2.phase1.dim_C0}."
            )
        r = self._delta.dot(x_)
        residual_norm = float(np.linalg.norm(r))
        energy_before = residual_norm ** 2

        if residual_norm <= _FRUSTRATION_TOLERANCE ** 0.5:
            logger.debug(
                "hodge_project: ‖δx‖=%.6e ≤ √ε. Sin proyección.", residual_norm
            )
            x_star = x_.copy()
            return x_star, energy_before, r, self._delta.dot(x_star)

        result = lsqr(
            self._delta,
            r,
            atol=_HODGE_SOLVER_TOLERANCE,
            btol=_HODGE_SOLVER_TOLERANCE,
            iter_lim=_HODGE_MAX_ITER,
        )
        delta_x: np.ndarray = np.asarray(result[0], dtype=np.float64).reshape(-1)
        stop_reason: int = int(result[1])
        if stop_reason not in _LSQR_OK_STOP:
            raise SheafCohomologyError(
                f"LSQR no convergió en hodge_project: stop_reason={stop_reason}."
            )

        x_star = x_ - delta_x
        r_star = self._delta.dot(x_star)
        logger.info(
            "hodge_project: E(x)=%.6e → E(x*)=%.6e, ‖δx*‖=%.6e, LSQR stop=%d.",
            energy_before, float(np.dot(r_star, r_star)),
            float(np.linalg.norm(r_star)), stop_reason,
        )
        return x_star, energy_before, r, r_star

    # ────────────────────────────────────────────────────────────────────
    # 3.2 Cotas isoperimétricas y de Lipschitz [A4]
    # ────────────────────────────────────────────────────────────────────
    def verify_lipschitz_isoperimetric_bound(
        self,
        x: np.ndarray,
        x_star: np.ndarray,
        delta_x: np.ndarray,
        delta_x_star: np.ndarray,
        inertia_bound: float,
    ) -> Tuple[float, float, float, float]:
        """Verifica simultáneamente las dos cotas del Axioma [A4].

        Lipschitz de-confinada:
            ‖δx* − δx‖₂  ≤  κ(δ) · ‖x* − x‖₂.
        Isoperimétrica de inercia:
            ‖x − x*‖₂    ≤  Δ_inertia.

        Returns
        ───────
        (hodge_correction_norm, lipschitz_lhs, lipschitz_slack, isoperimetric_slack)
        slack ≥ 0  ⟺  cota cumplida.

        Raises
        ──────
        IsoperimetricViolationError
        """
        x_ = np.asarray(x, dtype=np.float64).reshape(-1)
        xs_ = np.asarray(x_star, dtype=np.float64).reshape(-1)
        dx = np.asarray(delta_x, dtype=np.float64).reshape(-1)
        dxs = np.asarray(delta_x_star, dtype=np.float64).reshape(-1)

        hodge_correction_norm = float(np.linalg.norm(x_ - xs_))
        lipschitz_lhs = float(np.linalg.norm(dxs - dx))

        kappa = float(self._p2.kappa_delta_krylov)
        if not np.isfinite(kappa):
            kappa = 1e16
        lipschitz_rhs = kappa * hodge_correction_norm
        lipschitz_slack = lipschitz_rhs - lipschitz_lhs
        isoperimetric_slack = float(inertia_bound) - hodge_correction_norm

        if lipschitz_slack < -_FRUSTRATION_TOLERANCE:
            raise IsoperimetricViolationError(
                f"Violación de Lipschitz: ‖δx*−δx‖={lipschitz_lhs:.3e} > "
                f"κ(δ)·‖x*−x‖={lipschitz_rhs:.3e}."
            )
        if isoperimetric_slack < -_FRUSTRATION_TOLERANCE:
            raise IsoperimetricViolationError(
                f"Violación isoperimétrica: ‖x−x*‖={hodge_correction_norm:.3e} > "
                f"Δ_inertia={inertia_bound:.3e}."
            )
        return hodge_correction_norm, lipschitz_lhs, lipschitz_slack, isoperimetric_slack

    # ────────────────────────────────────────────────────────────────────
    # 3.3 Resolución del veredicto en Ω₃
    # ────────────────────────────────────────────────────────────────────
    def resolve_heyting_verdict(
        self,
        energy_after: float,
        lipschitz_slack: float,
        isoperimetric_slack: float,
    ) -> HeytingTop:
        """Colapsa Ω₃ al veredicto terminal (join monótono, precedencia estricta).

          1. h1_dimension > 0                              ⟹ VETOED (⊤)
          2. lipschitz_slack < −ε  ∨  isoperim_slack < −ε  ⟹ VETOED
          3. energy_after > ε_frustration                  ⟹ DEGRADED
          4. en otro caso                                  ⟹ COHERENT (⊥)
        """
        if self._p2.phase1.h1_dimension > 0:
            return HeytingTop.VETOED
        if (
            lipschitz_slack < -_FRUSTRATION_TOLERANCE
            or isoperimetric_slack < -_FRUSTRATION_TOLERANCE
        ):
            return HeytingTop.VETOED
        if energy_after > _FRUSTRATION_TOLERANCE:
            return HeytingTop.DEGRADED
        return HeytingTop.COHERENT

    # ────────────────────────────────────────────────────────────────────
    # 3.4 Actuación del disyuntor físico
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def fire_crowbar() -> bool:
        """Delega en el actuador de módulo (GPIO14 BCM)."""
        return actuate_crowbar_gpio14()

    # ────────────────────────────────────────────────────────────────────
    # 3.5 ★ MORFISMO TERMINAL DEL MÓDULO ★
    #     Cierra el funtor maestro 𝒵_Sheaf = Φ₃ ∘ Φ₂ ∘ Φ₁.
    # ────────────────────────────────────────────────────────────────────
    def resolve_sheaf_governance(
        self,
        x: np.ndarray,
        *,
        inertia_bound: float = 1.0e3,
    ) -> SheafGovernanceState:
        """★ MORFISMO TERMINAL DEL MÓDULO ★

        Cadena funtorial de Φ₃:
            KrylovSpectralData
              ──(hodge_project)───────────────────────▶ x*, δx, δx*
              ──(verify_lipschitz_isoperimetric_bound)▶ slacks
              ──(resolve_heyting_verdict)─────────────▶ verdict ∈ Ω₃
              ──(fire_crowbar si VETOED)──────────────▶ crowbar_actuated
              ──(emit_SheafGovernanceState)───────────▶ salida de 𝒵_Sheaf
        """
        if inertia_bound < 0.0:
            raise SheafDegeneracyError(
                f"inertia_bound debe ser ≥ 0; recibido={inertia_bound!r}."
            )

        x_star, energy_before, dx, dxs = self.hodge_project(x)
        energy_after, _ = Phase2_KrylovSpectralAuditor(
            self._p2.phase1
        ).evaluate_dirichlet_energy(x_star)

        try:
            (
                hodge_correction_norm,
                lipschitz_lhs,
                lipschitz_slack,
                isoper_slack,
            ) = self.verify_lipschitz_isoperimetric_bound(
                x, x_star, dx, dxs, inertia_bound
            )
        except IsoperimetricViolationError as exc:
            logger.critical("Cota [A4] violada: %s", exc)
            hodge_correction_norm = float(
                np.linalg.norm(
                    np.asarray(x, dtype=np.float64).reshape(-1)
                    - np.asarray(x_star, dtype=np.float64).reshape(-1)
                )
            )
            lipschitz_lhs = float(
                np.linalg.norm(
                    np.asarray(dxs, dtype=np.float64) - np.asarray(dx, dtype=np.float64)
                )
            )
            lipschitz_slack, isoper_slack = -1.0, -1.0

        verdict = self.resolve_heyting_verdict(
            energy_after, lipschitz_slack, isoper_slack
        )

        crowbar = False
        if verdict == HeytingTop.VETOED:
            logger.critical(
                "COLAPSO HEYTING Ω₃ → VETOED (⊤). dim H¹=%d, energy_after=%.3e.",
                self._p2.phase1.h1_dimension, energy_after,
            )
            crowbar = self.fire_crowbar()

        final = SheafGovernanceState(
            verdict=verdict,
            h0_dimension=self._p2.phase1.h0_dimension,
            h1_dimension=self._p2.phase1.h1_dimension,
            frustration_energy_before=energy_before,
            frustration_energy_after=energy_after,
            hodge_correction_norm=hodge_correction_norm,
            residual_correction_norm=float(np.linalg.norm(dxs)),
            lipschitz_lhs=lipschitz_lhs,
            lipschitz_slack=lipschitz_slack,
            isoperimetric_slack=isoper_slack,
            kappa_delta=self._p2.kappa_delta_krylov,
            crowbar_actuated=crowbar,
            certification_hash=self._p2.phase1.certification_hash,
            euler_characteristic=self._p2.phase1.euler_characteristic,
        )
        logger.info(
            "[FASE 3 ✓] SheafGovernanceState: verdict=%s, E_before=%.3e, "
            "E_after=%.3e, Crowbar=%s.",
            verdict.name, energy_before, energy_after, crowbar,
        )
        return final


# ═══════════════════════════════════════════════════════════════════════════════
# ████████████████  PROTOCOLO INMUNOLÓGICO (OBSERVADOR EXTERNO)  ███████████████
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ThreatMetrics:
    """Métricas de amenaza devueltas por el Observador Topológico externo."""

    mahalanobis_distance: float
    is_stable: bool
    structural_alteration: int  # Δχ
    threat_level: str  # 'HEALTHY' | 'WARNING' | 'CRITICAL'
    details: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ITopologicalWatcher(Protocol):
    """Protocolo del Sistema Inmunológico (pullback categórico externo)."""

    def evaluate_manifold_deformation(
        self,
        state_tensor: np.ndarray,
        reference_chi: Optional[int] = None,
    ) -> ThreatMetrics:
        """Evalúa la deformación de la variedad dado un tensor ψ ∈ ℝ⁷."""
        ...


@dataclass(frozen=True, slots=True)
class GlobalFrustrationAssessment:
    """DTO retro (v3.x) para consumidores legacy.

    Emitido por `audit_global_state`; equivalente reducido de
    SheafGovernanceState sin colapso Heyting ni Crowbar.
    """

    frustration_energy: float
    h0_dimension: int
    h1_dimension: int
    is_coherent: bool
    spectral_gap: float
    residual_norm: float
    spectral_method: str
    delta_rank: int
    condition_number_est: float
    euler_characteristic: int


# ═══════════════════════════════════════════════════════════════════════════════
# ████████████████  ORQUESTADOR MAESTRO — FUNTOR 𝒵_Sheaf  ███████████████████████
# ═══════════════════════════════════════════════════════════════════════════════


class SheafCohomologyOrchestrator:
    """Funtor maestro 𝒵_Sheaf = Φ₃ ∘ Φ₂ ∘ Φ₁.

    Encadena las tres fases anidadas preservando la composición monoidal
    estricta. El anidamiento se realiza EXCLUSIVAMENTE a través de los
    morfismos terminales:

        Φ₁.nest_into_phase2(sheaf)  ⟶  Phase2
        Φ₂.nest_into_phase3(x)      ⟶  Phase3
        Φ₃.resolve_sheaf_governance ⟶  SheafGovernanceState

    API pública
    ───────────
    · run_full_governance(sheaf, x, inertia_bound) → SheafGovernanceState
    · audit_global_state(sheaf, x, strict_topology) → GlobalFrustrationAssessment
    · evaluate_tool_injection(base_sheaf, base_state, new_edge) → ThreatMetrics
    """

    def __init__(self, watcher: Optional[ITopologicalWatcher] = None) -> None:
        self._watcher = watcher

    # ────────────────────────────────────────────────────────────────────
    # Punto único de entrada: Φ₃ ∘ Φ₂ ∘ Φ₁  (anidamiento estricto)
    # ────────────────────────────────────────────────────────────────────
    @classmethod
    def run_full_governance(
        cls,
        sheaf: CellularSheaf,
        x: np.ndarray,
        *,
        inertia_bound: float = 1.0e3,
    ) -> SheafGovernanceState:
        """Ejecuta 𝒵_Sheaf(sheaf, x) anidando Φ₁ ⊣ Φ₂ ⊣ Φ₃.

        Fast-fail [A2]: si dim H¹ > 0 tras Φ₁, se aborta Φ₂ y Φ₃, se colapsa
        Ω₃ → VETOED y se conmuta el Crowbar sin consultar Krylov ni Hodge.
        """
        # Φ₁ → unidad de Φ₂
        phase2 = Phase1_CohomologicalVetoCertifier.nest_into_phase2(sheaf)
        p1 = phase2.phase1

        if p1.veto_triggered:
            logger.critical(
                "AXIOMA [A2] VETO ABORTIVO: dim H¹=%d > 0. "
                "Colapsando Ω₃ → VETOED (⊤) sin Φ₂/Φ₃.",
                p1.h1_dimension,
            )
            return SheafGovernanceState(
                verdict=HeytingTop.VETOED,
                h0_dimension=p1.h0_dimension,
                h1_dimension=p1.h1_dimension,
                frustration_energy_before=float("inf"),
                frustration_energy_after=float("inf"),
                hodge_correction_norm=float("inf"),
                residual_correction_norm=float("inf"),
                lipschitz_lhs=float("inf"),
                lipschitz_slack=-1.0,
                isoperimetric_slack=-1.0,
                kappa_delta=p1.condition_number_delta,
                crowbar_actuated=actuate_crowbar_gpio14(),
                certification_hash=p1.certification_hash,
                euler_characteristic=p1.euler_characteristic,
            )

        # Φ₂ → unidad de Φ₃
        phase3 = phase2.nest_into_phase3(x)
        # Φ₃ → DTO terminal
        return phase3.resolve_sheaf_governance(x, inertia_bound=inertia_bound)

    # ────────────────────────────────────────────────────────────────────
    # API de compatibilidad retro (v3.x)
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_global_state_vector(
        sheaf: CellularSheaf,
        global_state_vector: np.ndarray,
    ) -> np.ndarray:
        """Valida x ∈ C⁰: conversión, forma, dimensión y finitud."""
        try:
            x = np.asarray(global_state_vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SheafDegeneracyError(
                f"Estado global no convertible a float64: {exc}"
            ) from exc
        if x.ndim != 1:
            raise SheafDegeneracyError(
                f"Estado global debe ser vector 1D; shape={x.shape}."
            )
        if x.shape[0] != sheaf.total_node_dim:
            raise SheafDegeneracyError(
                f"Dimensión incompatible: {x.shape[0]} ≠ {sheaf.total_node_dim}."
            )
        if not np.all(np.isfinite(x)):
            raise SheafDegeneracyError("Estado global contiene NaN/∞.")
        return x

    @classmethod
    def audit_global_state(
        cls,
        sheaf: CellularSheaf,
        global_state_vector: np.ndarray,
        strict_topology: bool = True,
    ) -> GlobalFrustrationAssessment:
        """API retro: certifica coherencia global sin emitir SheafGovernanceState.

        Pipeline anidado Φ₁ ⊣ Φ₂. Para el pipeline completo con Crowbar,
        use `run_full_governance`.

        Raises
        ──────
        HomologicalInconsistencyError
        """
        phase2 = Phase1_CohomologicalVetoCertifier.nest_into_phase2(sheaf)
        p1 = phase2.phase1
        x = cls._validate_global_state_vector(sheaf, global_state_vector)
        p2 = phase2.audit_krylov_spectral_stability(x)

        is_coherent = p2.dirichlet_energy <= _FRUSTRATION_TOLERANCE
        if not is_coherent:
            raise HomologicalInconsistencyError(
                f"Fractura de consenso: E(x)={p2.dirichlet_energy:.6e} "
                f"> ε={_FRUSTRATION_TOLERANCE:.6e}."
            )
        if strict_topology and p1.h1_dimension > 0:
            raise HomologicalInconsistencyError(
                f"Paradoja de Holonomía: dim H¹={p1.h1_dimension} > 0. "
                "El LLM generó un ciclo estratégico lógicamente imposible."
            )

        return GlobalFrustrationAssessment(
            frustration_energy=p2.dirichlet_energy,
            h0_dimension=p1.h0_dimension,
            h1_dimension=p1.h1_dimension,
            is_coherent=True,
            spectral_gap=p2.spectral_gap_L,
            residual_norm=p2.residual_norm,
            spectral_method="krylov",
            delta_rank=p1.delta_rank,
            condition_number_est=p2.kappa_delta_krylov,
            euler_characteristic=p1.euler_characteristic,
        )

    # ────────────────────────────────────────────────────────────────────
    # Pullback categórico externo (ITopologicalWatcher)
    # ────────────────────────────────────────────────────────────────────
    def evaluate_tool_injection(
        self,
        base_sheaf: CellularSheaf,
        base_state: np.ndarray,
        new_edge: SheafEdge,
    ) -> ThreatMetrics:
        """Pullback categórico (Fases I–V) sobre una inyección de herramienta.

        Fase I   : simulación Mayer–Vietoris sobre G ∪ {e}; Δβ₁ determina veto.
        Fase II  : construcción del tensor de estado ψ ∈ ℝ⁷.
        Fase III : pullback vía ITopologicalWatcher.
        Fase V   : colapso de onda (fast-fail).
        """
        if self._watcher is None:
            logger.warning("Sin ITopologicalWatcher inyectado: auditoría omitida.")
            return ThreatMetrics(0.0, True, 0, "HEALTHY", {})

        import networkx as nx  # dependencia opcional, import local

        base_audit = self.audit_global_state(base_sheaf, base_state)

        G = nx.Graph()
        G.add_nodes_from(range(base_sheaf.num_nodes))
        for edge in base_sheaf.edges:
            G.add_edge(edge.u, edge.v)

        u, v = new_edge.u, new_edge.v
        has_path = nx.has_path(G, u, v) if (u in G and v in G) else False
        if has_path:
            delta_beta0, delta_beta1 = 0, 1
        else:
            delta_beta0, delta_beta1 = -1, 0

        sim_h0 = base_audit.h0_dimension + delta_beta0
        sim_h1 = base_audit.h1_dimension + delta_beta1

        if delta_beta1 > 0:
            logger.error(
                "VETO PREVENTIVO (Fase I): inyección induce Δβ₁=%d.", delta_beta1
            )
            raise TopologicalBifurcationError(
                f"Obstrucción en Fase I: ciclo homológico inducido (Δβ₁={delta_beta1})."
            )

        # ψ = [saturation, flyback, dissipated_power, β₀, β₁, entropy, exergy_loss]
        psi = np.zeros(7, dtype=np.float64)
        psi[0] = 0.05
        psi[2] = base_audit.frustration_energy
        psi[3] = float(sim_h0)
        psi[4] = float(sim_h1)
        psi[5] = 0.1

        metrics = self._watcher.evaluate_manifold_deformation(
            psi, reference_chi=base_audit.euler_characteristic
        )

        if not metrics.is_stable or metrics.threat_level == "CRITICAL":
            logger.critical(
                "VETO TOPOLÓGICO (Fase V): Δχ=%d, d_M=%.4f, status=%s.",
                metrics.structural_alteration,
                metrics.mahalanobis_distance,
                metrics.threat_level,
            )
            raise TopologicalBifurcationError(
                f"Bifurcación detectada: Δχ={metrics.structural_alteration}, "
                f"d_M={metrics.mahalanobis_distance:.4f}."
            )
        return metrics


__all__ = [
    "SheafCohomologyError",
    "HomologicalInconsistencyError",
    "SheafDegeneracyError",
    "SpectralComputationError",
    "TopologicalBifurcationError",
    "HeytingCollapseError",
    "IsoperimetricViolationError",
    "RestrictionMap",
    "SheafEdge",
    "CellularSheaf",
    "actuate_crowbar_gpio14",
    "CohomologicalVetoData",
    "Phase1_CohomologicalVetoCertifier",
    "KrylovSpectralData",
    "Phase2_KrylovSpectralAuditor",
    "HeytingTop",
    "SheafGovernanceState",
    "Phase3_IsoperimetricHodgeProjector",
    "ThreatMetrics",
    "ITopologicalWatcher",
    "SheafCohomologyOrchestrator",
    "GlobalFrustrationAssessment",
]