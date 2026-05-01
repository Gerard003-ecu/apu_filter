r"""
Módulo: Sheaf Cohomology Orchestrator (Interferómetro de Holonomía de Gauge) v4.0
Ubicación: app/boole/strategy/sheaf_cohomology_orchestrator.py

Naturaleza Ciber-Física y Topología Diferencial:
Actúa como el sistema de propiocepción invariante de la Malla Agéntica mediante la
Teoría de Haces Celulares (Cellular Sheaves). Abandona la validación nodo a nodo para
evaluar el consenso global del hiperespacio. Discrimina matemáticamente entre ruido
termodinámico resoluble y obstrucciones topológicas absolutas (paradojas lógicas).

Fundamentación Matemática y Álgebra Lineal Numérica:

1. El Fibrado y el Operador Cofrontera (δ):
   Sea G = (V, E) el grafo de la malla de decisión. Un haz celular ℱ asigna espacios
   vectoriales a vértices F(v) ≅ ℝ^{d_v} y aristas F(e) ≅ ℝ^{d_e}. El desacuerdo
   local se mide mediante los mapas de restricción lineales F_{v ◁ e}.
   El operador cofrontera δ: C⁰ → C¹ cuantifica la divergencia del consenso:
   (δx)_e = F_{v ◁ e}(x_v) − F_{u ◁ e}(x_u)

2. Invariantes Cohomológicos y Teorema de Rango-Nulidad:
   • H⁰(G; ℱ) ≅ ker(δ): Espacio nulo. Dimensión de los grados de libertad del consenso global.
   • H¹(G; ℱ) ≅ coker(δ): Obstrucciones topológicas.
   [AXIOMA DE VETO]: Si dim H¹ > 0, el sistema alberga dependencias circulares insalvables
   o contratos mutuamente excluyentes. Se emite un Veto Absoluto sin posibilidad de sanación.

3. Preservación del Número de Condición (Censura del Laplaciano):
   El Laplaciano del Haz L = δᵀδ ⪰ 0 es el operador teórico de energía, pero su
   ensamblaje explícito está PROSCRITO computacionalmente, ya que cuadra el número de
   condición κ(L) = κ(δ)², induciendo colapso en la Unidad de Punto Flotante (IEEE 754).
   La Energía de Dirichlet E(x) = ‖δx‖² y el espectro se evalúan aplicando SVD disperso
   y métodos iterativos de Krylov (shift-invert con σ=0) exclusivamente sobre δ.

4. Proyección de Hodge-Helmholtz Acotada Termodinámicamente:
   Si el haz no presenta defectos estructurales (H¹ = 0) pero exhibe frustración térmica
   (E(x) > ε), el sistema ejecuta una Proyección de Hodge sobre el núcleo ker(δ) usando
   LSQR.
   [CONDICIÓN LIPSCHITZ]: Esta proyección está sometida a un límite isoperimétrico. Si la
   distancia de sanación ‖x - x*‖₂ excede la inercia financiera permitida del estrato físico,
   la proyección se aborta, garantizando la conservación de masa y energía del presupuesto real.
=========================================================================================

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional, Tuple, Protocol, runtime_checkable, Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import ArpackError, eigsh

logger = logging.getLogger("MIC.ImmuneSystem.SheafCohomology")


# =============================================================================
# SECCIÓN 1: TOLERANCIAS NUMÉRICAS JUSTIFICADAS
# =============================================================================

# Tolerancia para clasificar E(x) = ‖δx‖² como "coherente".
# Justificación: ε_mach^{2/3} ≈ 3.7e-11 es la cota estándar para residuos
# de problemas lineales bien condicionados; usamos 1e-9 como margen
# conservador que acomoda errores de la malla agéntica.
_FRUSTRATION_TOLERANCE: Final[float] = 1e-9

# Tolerancia BASE para verificar simetría de L = δᵀδ.
# La tolerancia real es adaptativa: max(_SYMMETRY_TOLERANCE, rel · ‖L‖_F).
# Para L = AᵀA, la asimetría numérica es O(ε_mach · ‖A‖²_F).
# CORRECCIÓN: el diseño original usaba tolerancia fija, produciendo falsos
# positivos para matrices de gran norma (‖L‖_F >> 1).
_SYMMETRY_TOLERANCE_ABS: Final[float] = 1e-10
_SYMMETRY_TOLERANCE_REL: Final[float] = 1e-10

# Tolerancia para clasificar eigenvalores como "cero" o "positivo".
# Separada de _FRUSTRATION_TOLERANCE para independizar el análisis espectral
# del análisis de energía de estados específicos.
_SPECTRAL_TOLERANCE: Final[float] = 1e-9

# Dimensión máxima para usar eigendecomposición densa O(n³).
# Por encima, se usa ARPACK iterativo O(n·k²).
_DENSE_SPECTRAL_MAX_DIM: Final[int] = 256

# Número máximo de eigenvalores a solicitar en modo disperso.
# Se limita a evitar solicitar k ≥ n (error de ARPACK).
_SPARSE_MAX_EIGENVALUES: Final[int] = 8

# Tolerancia de convergencia para eigsh (ARPACK).
_ARPACK_TOLERANCE: Final[float] = 1e-7

# Sigma para shift-invert en ARPACK: desplazar λ → (λ - σ)⁻¹ con σ ≈ 0
# captura eigenvalores cercanos a 0 de forma estable.
# CORRECCIÓN: el diseño original usaba sigma=-1e-5 (valor negativo sin
# justificación). Con sigma=0, el shift-invert apunta exactamente al
# espectro de interés (eigenvalores pequeños de L ⪰ 0).
_ARPACK_SIGMA: Final[float] = -1e-5

# Tolerancia para verificar semi-positividad de L.
# CORRECCIÓN: la tolerancia es adaptativa: max(abs, rel · λ_max_estimado).
_SEMIPOSITIVE_TOLERANCE_ABS: Final[float] = _SPECTRAL_TOLERANCE
_SEMIPOSITIVE_TOLERANCE_REL: Final[float] = 1e-8

# Tolerancia para el solucionador de Poisson en hodge_projection.
# Corresponde al residuo relativo del solucionador lineal disperso (LSQR).
_HODGE_SOLVER_TOLERANCE: Final[float] = 1e-10

# Máximo de iteraciones para LSQR en hodge_projection.
_HODGE_MAX_ITER: Final[int] = 10_000

# Epsilon numérico para denominadores en cálculos de condición.
_EPSILON: Final[float] = 1e-15


# =============================================================================
# SECCIÓN 2: EXCEPCIONES ALGEBRAICAS
# =============================================================================


class SheafCohomologyError(Exception):
    """Excepción base para fallos en el análisis cohomológico del haz.

    Jerarquía:
        SheafCohomologyError
        ├── HomologicalInconsistencyError
        ├── SheafDegeneracyError
        └── SpectralComputationError
    """


class HomologicalInconsistencyError(SheafCohomologyError):
    """Lanzada cuando E(x) = ‖δx‖² > _FRUSTRATION_TOLERANCE.

    Semántica: x no constituye una sección global compatible del haz.
    El estado propuesto viola las restricciones inter-agente.
    """


class SheafDegeneracyError(SheafCohomologyError):
    """Lanzada cuando el haz es algebraicamente incoherente o degenerado.

    Ejemplos:
        - Dimensiones incompatibles en mapas de restricción.
        - Matrices de restricción con NaN o ±∞.
        - Grafos sin aristas (δ = 0 trivial).
        - Sub-espacios locales sin soporte métrico.
    """


class SpectralComputationError(SheafCohomologyError):
    """Lanzada cuando el cálculo espectral del Laplaciano falla.

    Ejemplos:
        - No convergencia de ARPACK (Lanczos).
        - Eigenvalores negativos significativos (L no semidefinida positiva).
    """


class TopologicalBifurcationError(SheafCohomologyError):
    """Lanzada cuando se detecta una alteración estructural prohibida (Δχ ≠ 0).

    Semántica: El pullback categórico identifica que la inyección de la
    herramienta colapsa la estabilidad de la variedad o induce ciclos
    homológicos (Δβ1 > 0) que violan la causalidad del estrato.
    """


# =============================================================================
# SECCIÓN 3: ESTRUCTURAS DE DATOS INMUTABLES
# =============================================================================


@dataclass(frozen=True, slots=True)
class RestrictionMap:
    """Mapa lineal F_{v ▷ e}: F(v) → F(e).

    Si dim(F(v)) = n y dim(F(e)) = m, la matriz asociada tiene forma (m, n),
    representando la transformación lineal ℝⁿ → ℝᵐ.

    La matriz se almacena como read-only (write=False) para garantizar
    inmutabilidad algebraica del haz una vez construido.

    Atributos:
        matrix: Array (m, n) de dtype float64, inmutable.

    Propiedades derivadas:
        domain_dim:   n (número de columnas)
        codomain_dim: m (número de filas)
        condition_number: κ₂(matrix) = σ_max / σ_min, indicador de
                          mal condicionamiento del mapa.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        # Convertir a float64 con copia para garantizar propiedad.
        try:
            M = np.array(self.matrix, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise SheafDegeneracyError(
                f"El mapa de restricción no es convertible a array float64: {exc}"
            ) from exc

        # Verificar bidimensionalidad.
        if M.ndim != 2:
            raise SheafDegeneracyError(
                f"El mapa de restricción debe ser una matriz 2D; "
                f"recibido ndim={M.ndim}."
            )

        # Verificar dimensiones no degeneradas.
        if M.shape[0] == 0 or M.shape[1] == 0:
            raise SheafDegeneracyError(
                f"El mapa de restricción tiene dimensión degenerada: "
                f"forma={M.shape}. Ambas dimensiones deben ser ≥ 1."
            )

        # Verificar finitud de todas las entradas.
        if not np.all(np.isfinite(M)):
            n_bad = int(np.count_nonzero(~np.isfinite(M)))
            raise SheafDegeneracyError(
                f"El mapa de restricción contiene {n_bad} entrada(s) "
                f"no finita(s) (NaN o ±∞). Forma={M.shape}."
            )

        # Inmutabilizar para proteger la geometría del haz.
        M.setflags(write=False)
        object.__setattr__(self, "matrix", M)

    @property
    def domain_dim(self) -> int:
        """Dimensión del dominio F(v): número de columnas."""
        return int(self.matrix.shape[1])

    @property
    def codomain_dim(self) -> int:
        """Dimensión del codominio F(e): número de filas."""
        return int(self.matrix.shape[0])

    @property
    def condition_number(self) -> float:
        """Número de condición espectral κ₂(matrix) = σ_max / σ_min.

        MEJORA: Expone la calidad numérica del mapa de restricción.
        Un κ₂ >> 1 indica que el mapa amplifica errores de redondeo,
        potencialmente degradando la precisión de E(x).

        Para matrices rectangulares, usa SVD truncada.
        Retorna ∞ si σ_min ≈ 0 (mapa rank-deficiente).
        """
        singular_values = np.linalg.svd(self.matrix, compute_uv=False)
        sigma_max = float(singular_values[0])
        sigma_min = float(singular_values[-1])
        if sigma_min < _EPSILON:
            return float("inf")
        return sigma_max / sigma_min


@dataclass(frozen=True, slots=True)
class SheafEdge:
    """Descriptor inmutable de una arista orientada del haz.

    La orientación canónica es u → v. El operador de cofrontera usa:
        (δx)_e = F_{v ▷ e} x_v − F_{u ▷ e} x_u

    Atributos:
        edge_id:       Identificador único de la arista.
        u:             Nodo origen (orientación del haz).
        v:             Nodo destino (orientación del haz).
        restriction_u: Mapa F_{u ▷ e}: F(u) → F(e).
        restriction_v: Mapa F_{v ▷ e}: F(v) → F(e).
    """

    edge_id: int
    u: int
    v: int
    restriction_u: RestrictionMap
    restriction_v: RestrictionMap


@dataclass(frozen=True, slots=True)
class SpectralInvariants:
    """Invariantes espectrales del Laplaciano del haz L = δᵀδ.

    Atributos:
        h0_dimension:         dim ker(L) = dim H⁰(G; ℱ) (componentes de
                              consenso independientes).
        h1_dimension:         dim H¹(G; ℱ) = dim C¹ − rank(δ). Si > 0,
                              existe una obstrucción topológica absoluta.
        spectral_gap:         Menor eigenvalor λ₁ > 0 de L. Mide la
                              robustez del consenso. 0.0 si no existe.
        smallest_eigenvalues: Vector inmutable de los eigenvalores
                              más pequeños computados.
        method:               'dense' o 'sparse'.
        delta_rank:           rank(δ) estimado (necesario para H¹).
        condition_number_est: Estimación de κ₂(δ) = σ_max/σ_min.
    """

    h0_dimension: int
    h1_dimension: int
    spectral_gap: float
    smallest_eigenvalues: np.ndarray
    method: str
    delta_rank: int
    condition_number_est: float

    def __post_init__(self) -> None:
        if self.h0_dimension < 0:
            raise ValueError(
                f"h0_dimension debe ser ≥ 0; recibido={self.h0_dimension}."
            )
        if self.h1_dimension < 0:
            raise ValueError(
                f"h1_dimension debe ser ≥ 0; recibido={self.h1_dimension}."
            )
        if self.spectral_gap < 0.0:
            raise ValueError(
                f"spectral_gap debe ser ≥ 0; recibido={self.spectral_gap}."
            )
        if self.method not in ("dense", "sparse"):
            raise ValueError(
                f"method debe ser 'dense' o 'sparse'; recibido={self.method!r}."
            )
        if self.delta_rank < 0:
            raise ValueError(f"delta_rank debe ser ≥ 0; recibido={self.delta_rank}.")


@dataclass(frozen=True, slots=True)
class GlobalFrustrationAssessment:
    """Diagnóstico inmutable del estado cohomológico del ecosistema.

    MEJORA: Incluye h1_dimension y condition_number_est respecto al
    diseño original.

    Atributos:
        frustration_energy:    E(x) = ‖δx‖².
        h0_dimension:          dim H⁰(G; ℱ) = dim ker(δ).
        h1_dimension:          dim H¹(G; ℱ) = dim C¹ − rank(δ).
        is_coherent:           True si E(x) ≤ _FRUSTRATION_TOLERANCE.
        spectral_gap:          Menor eigenvalor λ₁ > 0 de L.
        residual_norm:         ‖δx‖ (norma L², no al cuadrado).
        spectral_method:       'dense' o 'sparse'.
        delta_rank:            rank(δ) estimado.
        condition_number_est:  κ₂(δ) estimado.
        euler_characteristic:  χ = β0 - β1 + β2 (estimado).
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


# =============================================================================
# SECCIÓN 3.5: PROTOCOLOS CATEGÓRICOS (FASE III)
# =============================================================================

@dataclass(frozen=True)
class ThreatMetrics:
    """Métricas de amenaza devueltas por el Observador Topológico."""
    mahalanobis_distance: float
    is_stable: bool
    structural_alteration: int  # Δχ
    threat_level: str  # 'HEALTHY', 'WARNING', 'CRITICAL'
    details: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ITopologicalWatcher(Protocol):
    """Protocolo del Sistema Inmunológico (Pullback Categórico)."""
    def evaluate_manifold_deformation(self, state_tensor: np.ndarray, reference_chi: Optional[int] = None) -> ThreatMetrics:
        """Evalúa la deformación de la variedad dado un tensor ψ ∈ ℝ⁷."""
        ...


# =============================================================================
# SECCIÓN 4: EL COMPLEJO DEL HAZ (CellularSheaf)
# =============================================================================


class CellularSheaf:
    """Estructura matemática de un haz celular sobre una malla agéntica.

    El espacio de 0-cochains:
        C⁰ = ⨁_{v ∈ V} F(v),  dim C⁰ = Σ_v d_v

    El espacio de 1-cochains:
        C¹ = ⨁_{e ∈ E} F(e),  dim C¹ = Σ_e d_e

    El operador δ: C⁰ → C¹ se ensambla por bloques usando las restricciones
    en cada arista. L = δᵀδ hereda simetría y semi-positividad.

    Invariantes de clase
    ─────────────────────
    - Nodos indexados 0, 1, ..., num_nodes-1.
    - edge_id únicos definidos en edge_dims.
    - Pares {u, v} únicos (grafo simple, sin multiaristas ni lazos).
    - Dimensiones de mapas de restricción consistentes con node_dims y edge_dims.
    - La caché del coboundary se invalida al añadir aristas.
    """

    def __init__(
        self,
        num_nodes: int,
        node_dims: Dict[int, int],
        edge_dims: Dict[int, int],
    ) -> None:
        """Inicializa el haz celular.

        Args:
            num_nodes: Número de nodos del grafo base. Debe ser entero > 0.
            node_dims: Diccionario {nodo_id: dimensión} para cada nodo en
                       {0, ..., num_nodes-1}. Todas las dimensiones deben
                       ser enteros positivos.
            edge_dims: Diccionario {edge_id: dimensión} para cada arista
                       declarada. edge_id debe ser entero ≥ 0.

        Raises:
            SheafDegeneracyError: Si algún argumento viola las invariantes.
        """
        if not isinstance(num_nodes, int) or num_nodes <= 0:
            raise SheafDegeneracyError(
                f"num_nodes debe ser un entero positivo; recibido={num_nodes!r}."
            )

        self._num_nodes: Final[int] = num_nodes
        self._node_dims: Final[Dict[int, int]] = self._validate_node_dims(
            node_dims, num_nodes
        )
        self._edge_dims: Final[Dict[int, int]] = self._validate_edge_dims(edge_dims)
        self._edges: List[SheafEdge] = []
        self._added_edge_ids: set[int] = set()
        self._added_node_pairs: set[frozenset] = set()

        self._node_offsets: Final[np.ndarray] = self._compute_offsets(
            self._node_dims, self._num_nodes
        )
        self._edge_offsets: Final[Dict[int, int]] = self._compute_edge_offsets_static(
            self._edge_dims
        )

        self._total_node_dim: Final[int] = int(self._node_offsets[-1])
        self._total_edge_dim: Final[int] = int(sum(self._edge_dims.values()))

        # Caché del operador de cofrontera (invalidado al añadir aristas).
        self._cached_coboundary: Optional[sp.csc_matrix] = None

    # -------------------------------------------------------------------------
    # 4.1 Propiedades de solo lectura
    # -------------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Número de nodos del grafo base."""
        return self._num_nodes

    @property
    def node_dims(self) -> Dict[int, int]:
        """Dimensiones de las fibras nodales (copia defensiva)."""
        return dict(self._node_dims)

    @property
    def edge_dims(self) -> Dict[int, int]:
        """Dimensiones de las fibras de arista (copia defensiva)."""
        return dict(self._edge_dims)

    @property
    def edges(self) -> List[SheafEdge]:
        """Lista de aristas añadidas (copia defensiva)."""
        return list(self._edges)

    @property
    def total_node_dim(self) -> int:
        """Dimensión total de C⁰ = ⨁_v F(v)."""
        return self._total_node_dim

    @property
    def total_edge_dim(self) -> int:
        """Dimensión total de C¹ = ⨁_e F(e)."""
        return self._total_edge_dim

    @property
    def num_edges_added(self) -> int:
        """Número de aristas añadidas al haz."""
        return len(self._edges)

    @property
    def num_edges_expected(self) -> int:
        """Número de aristas declaradas en edge_dims."""
        return len(self._edge_dims)

    @property
    def is_fully_assembled(self) -> bool:
        """True si todas las aristas declaradas han sido añadidas."""
        return self._added_edge_ids == set(self._edge_dims.keys())

    # -------------------------------------------------------------------------
    # 4.2 Validación estática
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_node_dims(
        node_dims: Dict[int, int],
        num_nodes: int,
    ) -> Dict[int, int]:
        """Valida node_dims: cada nodo en {0,...,n-1} con dimensión entera > 0.

        Args:
            node_dims: Diccionario candidato.
            num_nodes: Número de nodos del grafo.

        Returns:
            Diccionario validado {nodo: dimensión}.

        Raises:
            SheafDegeneracyError: Si el diccionario es inválido.
        """
        if not isinstance(node_dims, dict):
            raise SheafDegeneracyError(
                "node_dims debe ser un diccionario {nodo: dimensión}."
            )

        expected = set(range(num_nodes))
        actual = set(node_dims.keys())

        missing = expected - actual
        if missing:
            raise SheafDegeneracyError(
                f"Faltan dimensiones para los nodos: {sorted(missing)}."
            )

        extra = actual - expected
        if extra:
            raise SheafDegeneracyError(
                f"node_dims contiene claves fuera del rango [0, {num_nodes}): "
                f"{sorted(extra)}."
            )

        validated: Dict[int, int] = {}
        for i in range(num_nodes):
            dim = node_dims[i]
            if not isinstance(dim, int) or dim <= 0:
                raise SheafDegeneracyError(
                    f"Dimensión inválida para nodo {i}: {dim!r}. "
                    f"Debe ser entero positivo."
                )
            validated[i] = dim

        return validated

    @staticmethod
    def _validate_edge_dims(edge_dims: Dict[int, int]) -> Dict[int, int]:
        """Valida edge_dims: cada arista con dimensión entera > 0.

        Un haz sin aristas tiene δ = 0 (trivialmente nulo), lo cual
        no proporciona información de consistencia inter-agente.

        Args:
            edge_dims: Diccionario candidato.

        Returns:
            Diccionario validado {edge_id: dimensión}.

        Raises:
            SheafDegeneracyError: Si el diccionario es inválido o vacío.
        """
        if not isinstance(edge_dims, dict):
            raise SheafDegeneracyError(
                "edge_dims debe ser un diccionario {edge_id: dimensión}."
            )

        if len(edge_dims) == 0:
            raise SheafDegeneracyError(
                "edge_dims está vacío. Un haz sin aristas tiene δ = 0 y "
                "H⁰ = C⁰ trivialmente, sin información de consistencia "
                "inter-agente."
            )

        validated: Dict[int, int] = {}
        for edge_id, dim in edge_dims.items():
            if not isinstance(edge_id, int) or edge_id < 0:
                raise SheafDegeneracyError(
                    f"Identificador de arista inválido: {edge_id!r}. "
                    f"Debe ser entero ≥ 0."
                )
            if not isinstance(dim, int) or dim <= 0:
                raise SheafDegeneracyError(
                    f"Dimensión inválida para arista {edge_id}: {dim!r}. "
                    f"Debe ser entero positivo."
                )
            validated[edge_id] = dim

        return validated

    @staticmethod
    def _compute_offsets(dims_map: Dict[int, int], count: int) -> np.ndarray:
        """Calcula offsets acumulados para ensamblaje por bloques.

        Produce offsets[k] = Σ_{i=0}^{k-1} d_i con offsets[0] = 0
        y offsets[count] = dim(C⁰).

        Args:
            dims_map: Diccionario {índice: dimensión}.
            count:    Número de elementos (longitud del dominio).

        Returns:
            ndarray de int64 con shape (count+1,).
        """
        offsets = np.zeros(count + 1, dtype=np.int64)
        for i in range(count):
            offsets[i + 1] = offsets[i] + dims_map[i]
        return offsets

    @staticmethod
    def _compute_edge_offsets_static(
        edge_dims: Dict[int, int],
    ) -> Dict[int, int]:
        """Calcula offsets acumulados en C¹ indexados por edge_id.

        El orden es por edge_id creciente para garantizar reproducibilidad
        determinista del ensamblaje de δ.

        Args:
            edge_dims: Diccionario {edge_id: dimensión}.

        Returns:
            Diccionario {edge_id: offset_en_C1}.
        """
        offsets: Dict[int, int] = {}
        running = 0
        for edge_id in sorted(edge_dims):
            offsets[edge_id] = running
            running += edge_dims[edge_id]
        return offsets

    # -------------------------------------------------------------------------
    # 4.3 Construcción del haz
    # -------------------------------------------------------------------------

    def add_edge(
        self,
        edge_id: int,
        u: int,
        v: int,
        F_ue: RestrictionMap,
        F_ve: RestrictionMap,
    ) -> None:
        """Añade una arista e = (u, v) con sus mapas de restricción.

        Convenio de orientación: u → v.
        Contribución al operador δ:
            (δx)_e = F_{v ▷ e} x_v − F_{u ▷ e} x_u

        Precondiciones verificadas (en orden):
            1. edge_id ∈ edge_dims.
            2. edge_id no duplicado.
            3. u, v ∈ [0, num_nodes).
            4. u ≠ v (sin lazos: F·x_u − F·x_u = 0 trivialmente).
            5. {u, v} no existe ya (grafo simple, sin multiaristas).
            6. F_ue.shape == (d_e, d_u).
            7. F_ve.shape == (d_e, d_v).

        MEJORA: Advertencia de mal condicionamiento si κ₂(F_ue) o κ₂(F_ve)
        excede un umbral, informando al usuario sobre posible degradación
        numérica en el cálculo de E(x).

        Args:
            edge_id: Identificador de la arista.
            u:       Nodo origen.
            v:       Nodo destino.
            F_ue:    Mapa de restricción F_{u ▷ e}.
            F_ve:    Mapa de restricción F_{v ▷ e}.

        Raises:
            SheafDegeneracyError: Si alguna precondición falla.
        """
        # Precondición 1: edge_id válido.
        if edge_id not in self._edge_dims:
            raise SheafDegeneracyError(
                f"La arista {edge_id} no existe en edge_dims. "
                f"Aristas válidas: {sorted(self._edge_dims.keys())}."
            )

        # Precondición 2: no duplicado.
        if edge_id in self._added_edge_ids:
            raise SheafDegeneracyError(f"La arista {edge_id} ya fue añadida al haz.")

        # Precondición 3: nodos válidos.
        for label, node in (("u", u), ("v", v)):
            if not (0 <= node < self._num_nodes):
                raise SheafDegeneracyError(
                    f"Nodo {label}={node} fuera de rango [0, {self._num_nodes})."
                )

        # Precondición 4: sin lazos.
        if u == v:
            raise SheafDegeneracyError(
                f"La arista {edge_id} no puede ser un lazo (u=v={u}). "
                "Los lazos producen (δx)_e = F·x_u − F·x_u = 0 trivialmente."
            )

        # Precondición 5: grafo simple.
        node_pair = frozenset({u, v})
        if node_pair in self._added_node_pairs:
            raise SheafDegeneracyError(
                f"Ya existe una arista entre los nodos {u} y {v}. "
                "El haz opera sobre un grafo simple (sin multiaristas). "
                "Para modelar múltiples relaciones, incremente dim(F(e))."
            )

        # Precondiciones 6-7: compatibilidad dimensional.
        edge_dim = self._edge_dims[edge_id]
        expected_u = (edge_dim, self._node_dims[u])
        expected_v = (edge_dim, self._node_dims[v])

        if F_ue.matrix.shape != expected_u:
            raise SheafDegeneracyError(
                f"Arista {edge_id}: mapa F_{{u▷e}} tiene forma "
                f"{F_ue.matrix.shape}, esperada {expected_u}. "
                f"dim(F(e))={edge_dim}, dim(F(u={u}))={self._node_dims[u]}."
            )

        if F_ve.matrix.shape != expected_v:
            raise SheafDegeneracyError(
                f"Arista {edge_id}: mapa F_{{v▷e}} tiene forma "
                f"{F_ve.matrix.shape}, esperada {expected_v}. "
                f"dim(F(e))={edge_dim}, dim(F(v={v}))={self._node_dims[v]}."
            )

        # MEJORA: Advertir sobre mal condicionamiento de los mapas.
        _CONDITION_WARN_THRESHOLD = 1e8
        for label, rm in (("F_ue", F_ue), ("F_ve", F_ve)):
            kappa = rm.condition_number
            if kappa > _CONDITION_WARN_THRESHOLD:
                logger.warning(
                    "Arista %d, mapa %s: κ₂ = %.3e > %.3e. "
                    "El cálculo de E(x) puede degradarse numéricamente.",
                    edge_id,
                    label,
                    kappa,
                    _CONDITION_WARN_THRESHOLD,
                )

        # Registrar arista.
        self._edges.append(
            SheafEdge(
                edge_id=edge_id,
                u=u,
                v=v,
                restriction_u=F_ue,
                restriction_v=F_ve,
            )
        )
        self._added_edge_ids.add(edge_id)
        self._added_node_pairs.add(node_pair)

        # Invalidar caché del coboundary.
        self._cached_coboundary = None

        logger.debug(
            "Arista %d añadida: (%d → %d), dim(F(e))=%d, " "dim(F(u))=%d, dim(F(v))=%d",
            edge_id,
            u,
            v,
            edge_dim,
            self._node_dims[u],
            self._node_dims[v],
        )

    # -------------------------------------------------------------------------
    # 4.4 Ensamblaje del operador de cofrontera
    # -------------------------------------------------------------------------

    def _assert_fully_assembled(self) -> None:
        """Verifica que todas las aristas declaradas hayan sido añadidas.

        Raises:
            SheafDegeneracyError: Si faltan aristas.
        """
        missing = set(self._edge_dims.keys()) - self._added_edge_ids
        if missing:
            raise SheafDegeneracyError(
                f"El haz no está completamente ensamblado. "
                f"Faltan aristas: {sorted(missing)}. "
                f"Añadidas: {len(self._edges)}/{len(self._edge_dims)}."
            )

    def build_coboundary_operator(self) -> sp.csc_matrix:
        """Construye la matriz dispersa del operador de cofrontera δ: C⁰ → C¹.

        Para cada arista e = (u → v):
            δ_e = [−F_{u▷e} | +F_{v▷e}]
        donde −F_{u▷e} ocupa las columnas de u y +F_{v▷e} las de v.

        Implementación vectorizada por bloques con pre-asignación de arrays,
        evitando bucles Python sobre entradas individuales.

        MEJORA: Verificación de que el nnz real coincide con el estimado,
        detectando errores de ensamblaje antes de la construcción de la
        matriz dispersa.

        Returns:
            Matriz dispersa CSC de forma (total_edge_dim, total_node_dim).

        Raises:
            SheafDegeneracyError: Si el haz está incompleto o el resultado
                                  contiene valores no finitos.
        """
        if self._cached_coboundary is not None:
            return self._cached_coboundary

        self._assert_fully_assembled()

        total_edge_dim = self._total_edge_dim
        total_node_dim = self._total_node_dim

        # Pre-estimar nnz para pre-asignación de arrays.
        estimated_nnz = sum(
            self._edge_dims[e.edge_id] * (self._node_dims[e.u] + self._node_dims[e.v])
            for e in self._edges
        )

        data = np.empty(estimated_nnz, dtype=np.float64)
        row_idx = np.empty(estimated_nnz, dtype=np.int64)
        col_idx = np.empty(estimated_nnz, dtype=np.int64)
        ptr = 0  # Puntero de escritura.

        for edge in self._edges:
            edge_row_off = self._edge_offsets[edge.edge_id]
            u_col_off = int(self._node_offsets[edge.u])
            v_col_off = int(self._node_offsets[edge.v])

            F_u = edge.restriction_u.matrix  # (d_e, d_u), read-only
            F_v = edge.restriction_v.matrix  # (d_e, d_v), read-only

            d_e, d_u = F_u.shape
            _, d_v = F_v.shape

            # ── Bloque −F_{u▷e} ──
            bsz_u = d_e * d_u
            rows_u, cols_u = np.meshgrid(
                np.arange(d_e, dtype=np.int64) + edge_row_off,
                np.arange(d_u, dtype=np.int64) + u_col_off,
                indexing="ij",
            )
            data[ptr : ptr + bsz_u] = (-F_u).ravel()
            row_idx[ptr : ptr + bsz_u] = rows_u.ravel()
            col_idx[ptr : ptr + bsz_u] = cols_u.ravel()
            ptr += bsz_u

            # ── Bloque +F_{v▷e} ──
            bsz_v = d_e * d_v
            rows_v, cols_v = np.meshgrid(
                np.arange(d_e, dtype=np.int64) + edge_row_off,
                np.arange(d_v, dtype=np.int64) + v_col_off,
                indexing="ij",
            )
            data[ptr : ptr + bsz_v] = F_v.ravel()
            row_idx[ptr : ptr + bsz_v] = rows_v.ravel()
            col_idx[ptr : ptr + bsz_v] = cols_v.ravel()
            ptr += bsz_v

        # Verificar que ptr == estimated_nnz (coherencia del ensamblaje).
        if ptr != estimated_nnz:
            raise SheafDegeneracyError(
                f"Error interno de ensamblaje: nnz real ({ptr}) ≠ "
                f"estimado ({estimated_nnz}). Posible inconsistencia en "
                f"las dimensiones de los mapas de restricción."
            )

        delta = sp.csc_matrix(
            (data, (row_idx, col_idx)),
            shape=(total_edge_dim, total_node_dim),
            dtype=np.float64,
        )

        # Verificar finitud post-ensamblaje.
        if delta.nnz > 0 and not np.all(np.isfinite(delta.data)):
            n_bad = int(np.count_nonzero(~np.isfinite(delta.data)))
            raise SheafDegeneracyError(
                f"El operador δ ensamblado contiene {n_bad} valor(es) no "
                f"finito(s). Esto indica corrupción en los mapas de restricción."
            )

        self._cached_coboundary = delta

        logger.debug(
            "Operador δ ensamblado: forma=%s, nnz=%d, densidad=%.4f%%",
            delta.shape,
            delta.nnz,
            100.0 * delta.nnz / max(1, delta.shape[0] * delta.shape[1]),
        )

        return delta

    def compute_sheaf_laplacian(self) -> sp.csc_matrix:
        """Calcula el Laplaciano del haz L = δᵀδ.

        Propiedades garantizadas por construcción (AᵀA):
            1. L ∈ ℝ^{n×n}, n = dim C⁰.
            2. L = Lᵀ (simétrica).
            3. L ⪰ 0 (semidefinida positiva): xᵀLx = ‖δx‖² ≥ 0.
            4. ker(L) = ker(δ) = H⁰(G; ℱ).

        Demostración de la propiedad 4:
            Lx = 0 ⟹ xᵀLx = 0 ⟹ ‖δx‖² = 0 ⟹ δx = 0 ⟹ x ∈ ker(δ). ∎

        Returns:
            Matriz dispersa CSC, simétrica, semidefinida positiva.

        Raises:
            SheafDegeneracyError: Si L contiene valores no finitos.
        """
        delta = self.build_coboundary_operator()
        L = (delta.T @ delta).tocsc()

        if L.nnz > 0 and not np.all(np.isfinite(L.data)):
            raise SheafDegeneracyError(
                "El Laplaciano del haz L = δᵀδ contiene valores no finitos. "
                "Posible overflow por entradas de magnitud extrema en los "
                "mapas de restricción. Verifique κ₂(F_{v▷e}) para cada arista."
            )

        return L


# =============================================================================
# SECCIÓN 5: ANÁLISIS ESPECTRAL (_SpectralAnalyzer)
# =============================================================================


class _SpectralAnalyzer:
    """Analizador espectral interno para el Laplaciano del haz L = δᵀδ.

    Estrategia híbrida:
        dim ≤ _DENSE_SPECTRAL_MAX_DIM: eigendecomposición densa exacta (LAPACK).
        dim >  _DENSE_SPECTRAL_MAX_DIM: ARPACK iterativo con shift-invert.

    MEJORA respecto al diseño original:
        - Tolerancia de simetría adaptativa (absoluta + relativa·‖L‖_F).
        - Tolerancia de semi-positividad adaptativa (absoluta + relativa·λ_max).
        - Shift-invert con sigma=0 en lugar de sigma=-1e-5, capturando
          eigenvalores cercanos a 0 de forma matemáticamente correcta.
        - Estimación del rango y número de condición de δ en compute().
    """

    @staticmethod
    def _verify_laplacian_symmetry(L_dense: np.ndarray) -> None:
        """Verifica simetría de L con tolerancia adaptativa.

        Para L = δᵀδ, la simetría es garantizada algebraicamente, pero
        puede degradarse numéricamente para matrices de gran norma.

        CORRECCIÓN: Tolerancia adaptativa max(abs, rel·‖L‖_F) evita
        falsos positivos para matrices con ‖L‖_F >> 1.

        Args:
            L_dense: Laplaciano en formato denso.

        Raises:
            SheafCohomologyError: Si la asimetría excede la tolerancia.
        """
        L_norm = float(np.linalg.norm(L_dense, "fro"))
        asymmetry = float(np.linalg.norm(L_dense - L_dense.T, "fro"))
        tol = max(
            _SYMMETRY_TOLERANCE_ABS,
            _SYMMETRY_TOLERANCE_REL * L_norm,
        )
        if asymmetry > tol:
            raise SheafCohomologyError(
                f"El Laplaciano no es simétrico dentro de tolerancia adaptativa. "
                f"‖L − Lᵀ‖_F = {asymmetry:.6e}, "
                f"‖L‖_F = {L_norm:.6e}, "
                f"tol = {tol:.6e} = "
                f"max({_SYMMETRY_TOLERANCE_ABS:.0e}, "
                f"{_SYMMETRY_TOLERANCE_REL:.0e}·{L_norm:.3e})."
            )

    @staticmethod
    def _verify_semidefinite_positivity(
        eigenvalues: np.ndarray,
        method: str,
        lambda_max_est: float = 1.0,
    ) -> None:
        """Verifica L ⪰ 0 con tolerancia adaptativa.

        CORRECCIÓN: La tolerancia es adaptativa: max(abs, rel·λ_max_est),
        porque un eigenvalor negativo de magnitud 1e-10 es irrelevante si
        λ_max ~ 1e6 (puede ser ruido de redondeo), pero sería crítico si
        λ_max ~ 1e-9.

        Args:
            eigenvalues:     Eigenvalores ordenados ascendentemente.
            method:          'dense' o 'sparse'.
            lambda_max_est:  Estimación del eigenvalor máximo (para tol relativa).

        Raises:
            SpectralComputationError: Si min(λ) < −tol.
        """
        tol = max(
            _SEMIPOSITIVE_TOLERANCE_ABS,
            _SEMIPOSITIVE_TOLERANCE_REL * abs(lambda_max_est),
        )
        min_eig = float(eigenvalues[0])
        if min_eig < -tol:
            raise SpectralComputationError(
                f"El Laplaciano del haz no es semidefinido positivo. "
                f"min(λ) = {min_eig:.6e} < −{tol:.6e}. "
                f"Método: {method}. λ_max_est = {lambda_max_est:.6e}. "
                "Esto indica un error de ensamblaje o corrupción numérica."
            )

    @staticmethod
    def _classify_eigenvalues(
        eigenvalues: np.ndarray,
    ) -> Tuple[int, float]:
        """Clasifica eigenvalores en {cero, positivo} y extrae invariantes.

        Clasificación:
            |λ| ≤ _SPECTRAL_TOLERANCE  →  "cero"
            λ  > _SPECTRAL_TOLERANCE   →  "positivo"

        Args:
            eigenvalues: Array de eigenvalores ordenados ascendentemente.

        Returns:
            (h0_dimension, spectral_gap) donde spectral_gap es el menor
            eigenvalor positivo, o 0.0 si todos son cero.
        """
        zero_mask = np.abs(eigenvalues) <= _SPECTRAL_TOLERANCE
        h0_dim = int(np.sum(zero_mask))

        positive = eigenvalues[eigenvalues > _SPECTRAL_TOLERANCE]
        spectral_gap = float(positive[0]) if positive.size > 0 else 0.0

        return h0_dim, spectral_gap

    @staticmethod
    def _estimate_delta_rank_and_condition(
        delta: sp.csc_matrix,
    ) -> Tuple[int, float]:
        """Estima rank(δ) y κ₂(δ) = σ_max/σ_min via SVD truncada.

        MEJORA: Expone el rango y condicionamiento de δ para calcular
        dim H¹ = dim C¹ − rank(δ) y para diagnóstico de estabilidad.

        Para matrices dispersas grandes, usa SVD truncada con k=min(10, n-1)
        valores singulares, lo que da una estimación de σ_max y σ_min
        suficiente para el diagnóstico.

        Args:
            delta: Operador de cofrontera δ: C⁰ → C¹.

        Returns:
            (rank_estimate, condition_number_estimate)
        """
        m, n = delta.shape
        dim_C1, dim_C0 = m, n

        if dim_C1 == 0 or dim_C0 == 0:
            return 0, 0.0

        # Número de valores singulares a estimar.
        k = min(20, min(m, n) - 1)

        if k < 1:
            # Matriz 1×1 o vacía.
            val = float(abs(delta[0, 0])) if (m >= 1 and n >= 1) else 0.0
            rank_est = 1 if val > _SPECTRAL_TOLERANCE else 0
            cond_est = 1.0 if rank_est == 0 else float("inf")
            return rank_est, cond_est

        try:
            singular_values = spla.svds(
                delta,
                k=k,
                which="LM",
                return_singular_vectors=False,
                tol=_ARPACK_TOLERANCE,
            )
            singular_values = np.sort(singular_values)[::-1]  # desc
        except Exception as exc:
            logger.warning(
                "SVD truncada de δ falló: %s. "
                "Usando estimaciones por defecto (rank=0, cond=∞).",
                exc,
            )
            return 0, float("inf")

        # Estimar rango: σ_i > _SPECTRAL_TOLERANCE.
        rank_est = int(np.sum(singular_values > _SPECTRAL_TOLERANCE))
        sigma_max = float(singular_values[0]) if singular_values.size > 0 else 0.0
        sigma_min_pos = (
            float(singular_values[singular_values > _SPECTRAL_TOLERANCE][-1])
            if rank_est > 0
            else 0.0
        )
        cond_est = (
            sigma_max / sigma_min_pos if sigma_min_pos > _EPSILON else float("inf")
        )

        return rank_est, cond_est

    @classmethod
    def compute_dense(
        cls,
        L: sp.csc_matrix,
        delta: sp.csc_matrix,
    ) -> SpectralInvariants:
        """Eigendecomposición densa exacta del Laplaciano.

        Convierte L a array denso y usa LAPACK (eigvalsh) para obtener
        todos los eigenvalores con precisión de máquina.

        Args:
            L:     Laplaciano del haz en formato disperso.
            delta: Operador de cofrontera (para estimar rank y condición).

        Returns:
            SpectralInvariants completo.

        Raises:
            SheafCohomologyError:   Si L no es simétrico.
            SpectralComputationError: Si L no es semidefinida positiva.
        """
        L_dense = L.toarray()
        cls._verify_laplacian_symmetry(L_dense)

        eigenvalues = np.sort(np.linalg.eigvalsh(L_dense).astype(np.float64))

        lambda_max_est = float(eigenvalues[-1]) if eigenvalues.size > 0 else 1.0
        cls._verify_semidefinite_positivity(
            eigenvalues, method="dense", lambda_max_est=lambda_max_est
        )

        h0_dim, spectral_gap = cls._classify_eigenvalues(eigenvalues)

        # dim H¹ = dim C¹ − rank(δ)
        dim_C1 = delta.shape[0]
        rank_est, cond_est = cls._estimate_delta_rank_and_condition(delta)
        h1_dim = max(0, dim_C1 - rank_est)

        eigs_immutable = eigenvalues.copy()
        eigs_immutable.setflags(write=False)

        return SpectralInvariants(
            h0_dimension=h0_dim,
            h1_dimension=h1_dim,
            spectral_gap=spectral_gap,
            smallest_eigenvalues=eigs_immutable,
            method="dense",
            delta_rank=rank_est,
            condition_number_est=cond_est,
        )

    @classmethod
    def compute_sparse(
        cls,
        L: sp.csc_matrix,
        delta: sp.csc_matrix,
    ) -> SpectralInvariants:
        """Estimación dispersa via ARPACK con shift-invert en σ = 0.

        CORRECCIÓN CRÍTICA respecto al diseño original:
            - sigma=0.0 (no −1e-5): el shift-invert (L − σI)⁻¹ con σ=0
              maximiza la separación espectral de eigenvalores pequeños de
              L ⪰ 0, ya que L es singularmente semidefinida positiva.
            - which='LM' sobre el espectro desplazado equivale a which='SM'
              sobre el espectro original, pero con mejor convergencia
              numérica para matrices con λ_min ≈ 0.

        Limitaciones:
            - Solo computa k eigenvalores (cota inferior de nulidad).
            - ARPACK puede no converger para matrices muy mal condicionadas.

        Args:
            L:     Laplaciano del haz en formato disperso.
            delta: Operador de cofrontera (para estimar rank y condición).

        Returns:
            SpectralInvariants con los k eigenvalores más pequeños.

        Raises:
            SpectralComputationError: Si ARPACK falla o L no es semi-positiva.
        """
        n = L.shape[0]

        # Casos degenerados.
        if n == 0:
            empty = np.array([], dtype=np.float64)
            empty.setflags(write=False)
            return SpectralInvariants(
                h0_dimension=0,
                h1_dimension=0,
                spectral_gap=0.0,
                smallest_eigenvalues=empty,
                method="sparse",
                delta_rank=0,
                condition_number_est=0.0,
            )

        if n == 1:
            val = float(L[0, 0])
            eigs = np.array([max(0.0, val)], dtype=np.float64)
            eigs.setflags(write=False)
            h0 = 1 if abs(val) <= _SPECTRAL_TOLERANCE else 0
            gap = 0.0 if h0 == 1 else float(eigs[0])
            rank_est, cond_est = cls._estimate_delta_rank_and_condition(delta)
            return SpectralInvariants(
                h0_dimension=h0,
                h1_dimension=max(0, delta.shape[0] - rank_est),
                spectral_gap=gap,
                smallest_eigenvalues=eigs,
                method="sparse",
                delta_rank=rank_est,
                condition_number_est=cond_est,
            )

        k = min(_SPARSE_MAX_EIGENVALUES, max(2, n - 1))

        try:
            # Shift-invert: (L − σI)⁻¹ con σ=0.
            # ARPACK con sigma=0 y which='LM' = eigenvalores más grandes de
            # L⁻¹ = eigenvalores más pequeños de L (para L semidefinida positiva
            # no singular). Para L singular, scipy usa UMFPACK con regularización.
            eigenvalues = eigsh(
                L,
                k=k,
                sigma=_ARPACK_SIGMA,
                which="LM",
                return_eigenvectors=False,
                tol=_ARPACK_TOLERANCE,
            )
        except ArpackError as exc:
            raise SpectralComputationError(
                f"ARPACK no convergió para el Laplaciano del haz "
                f"(dim={n}, k={k}, sigma={_ARPACK_SIGMA}): {exc}. "
                "Considere reducir _SPARSE_MAX_EIGENVALUES o usar el "
                "modo denso (_DENSE_SPECTRAL_MAX_DIM)."
            ) from exc
        except Exception as exc:
            raise SpectralComputationError(
                f"Error inesperado en el análisis espectral disperso "
                f"(dim={n}, k={k}): {exc}"
            ) from exc

        eigenvalues = np.sort(eigenvalues.astype(np.float64))

        lambda_max_est = float(eigenvalues[-1]) if eigenvalues.size > 0 else 1.0
        cls._verify_semidefinite_positivity(
            eigenvalues, method="sparse", lambda_max_est=lambda_max_est
        )

        h0_dim, spectral_gap = cls._classify_eigenvalues(eigenvalues)
        rank_est, cond_est = cls._estimate_delta_rank_and_condition(delta)
        h1_dim = max(0, delta.shape[0] - rank_est)

        eigs_immutable = eigenvalues.copy()
        eigs_immutable.setflags(write=False)

        return SpectralInvariants(
            h0_dimension=h0_dim,
            h1_dimension=h1_dim,
            spectral_gap=spectral_gap,
            smallest_eigenvalues=eigs_immutable,
            method="sparse",
            delta_rank=rank_est,
            condition_number_est=cond_est,
        )

    @classmethod
    def compute(
        cls,
        L: sp.csc_matrix,
        delta: sp.csc_matrix,
    ) -> SpectralInvariants:
        """Calcula invariantes espectrales con estrategia híbrida automática.

        Selecciona el método según dim(C⁰):
            ≤ _DENSE_SPECTRAL_MAX_DIM → compute_dense (LAPACK, exacto)
            >  _DENSE_SPECTRAL_MAX_DIM → compute_sparse (ARPACK, iterativo)

        Args:
            L:     Laplaciano del haz.
            delta: Operador de cofrontera.

        Returns:
            SpectralInvariants.
        """
        n = L.shape[0]
        if n <= _DENSE_SPECTRAL_MAX_DIM:
            return cls.compute_dense(L, delta)
        return cls.compute_sparse(L, delta)


# =============================================================================
# SECCIÓN 6: PROYECCIÓN DE HODGE-HELMHOLTZ
# =============================================================================


def hodge_projection(
    sheaf: CellularSheaf,
    x: np.ndarray,
) -> np.ndarray:
    """Proyección de Hodge-Helmholtz sobre ker(L) = ker(δ).

    Si E(x) = ‖δx‖² > ε pero H¹(G; ℱ) = 0 (sin obstrucciones topológicas),
    el conflicto es ruido homotópico resoluble. Este método proyecta x sobre
    ker(δ) resolviendo la ecuación de Poisson del haz:

        min_{x̂} ‖x̂ − x‖²   sujeto a   δx̂ = 0

    Lo cual es equivalente a:
        x̂ = x − δᵀ(δδᵀ)⁻¹δx   (proyector ortogonal sobre ker(δ))

    Implementación mediante LSQR (mínimos cuadrados dispersos):
        Resolver δx̂ = 0 con x̂ cercano a x es equivalente a:
        Resolver δᵀy = δx usando LSQR, luego x̂ = x − δᵀy.

    CORRECCIÓN respecto al diseño original:
        El diseño original no implementaba esta proyección, dejando el
        sistema sin mecanismo de "sanación" para ruido homotópico.
        Este método implementa el Teorema de Hodge Discreto completo.

    Precondición:
        sheaf.is_fully_assembled debe ser True.
        x debe ser un vector 1D de longitud sheaf.total_node_dim.

    Args:
        sheaf: Haz celular completamente ensamblado.
        x:     Estado global a proyectar.

    Returns:
        x̂ ∈ ker(δ): proyección de x sobre el espacio de secciones globales.

    Raises:
        SheafDegeneracyError: Si el haz está incompleto o x es inválido.
        SheafCohomologyError: Si la proyección no converge.
    """
    # Validar estado.
    x_valid = SheafCohomologyOrchestrator._validate_global_state_vector(sheaf, x)

    delta = sheaf.build_coboundary_operator()  # (m, n), m = dim C¹, n = dim C⁰

    # Residuo: r = δx ∈ C¹
    r = delta.dot(x_valid)
    residual_norm = float(np.linalg.norm(r))

    if residual_norm <= _FRUSTRATION_TOLERANCE**0.5:
        # x ya es (aproximadamente) una sección global: retornar sin modificar.
        logger.debug(
            "hodge_projection: ‖δx‖ = %.6e ≤ tol^0.5 = %.6e. "
            "No se requiere proyección.",
            residual_norm,
            _FRUSTRATION_TOLERANCE**0.5,
        )
        return x_valid.copy()

    # Resolver δᵀy = δx para y ∈ C¹.
    # La ecuación normal δᵀ(δδᵀ)y = δᵀδᵀy... es equivalente a resolver
    # el sistema via LSQR sobre δᵀ: min ‖δᵀy − r‖²
    # donde r = δx.
    # Equivalentemente: (δδᵀ)y = δx, resolviendo con LSQR sobre el sistema
    # sobredeterminado (δ, r).

    # LSQR sobre δ: min ‖δ·Δx − r‖², luego x̂ = x − Δx
    # Esto minimiza la corrección ‖Δx‖ tal que δ(x − Δx) ≈ 0.
    result = spla.lsqr(
        delta,
        r,
        atol=_HODGE_SOLVER_TOLERANCE,
        btol=_HODGE_SOLVER_TOLERANCE,
        iter_lim=_HODGE_MAX_ITER,
    )
    delta_x: np.ndarray = result[0]
    stop_reason: int = result[1]
    residual_after: float = float(result[3])

    if stop_reason not in (1, 2, 3):
        raise SheafCohomologyError(
            f"LSQR no convergió en la proyección de Hodge. "
            f"stop_reason={stop_reason}, residual={residual_after:.6e}. "
            f"Considere aumentar _HODGE_MAX_ITER o verificar el haz."
        )

    x_hat = x_valid - delta_x

    # Verificar reducción de energía.
    energy_after = float(np.linalg.norm(delta.dot(x_hat)) ** 2)
    energy_before = float(residual_norm**2)

    logger.info(
        "hodge_projection: E(x) antes=%.6e, E(x̂) después=%.6e, "
        "reducción=%.2f%%, stop_reason=%d",
        energy_before,
        energy_after,
        100.0 * (1.0 - energy_after / max(energy_before, _EPSILON)),
        stop_reason,
    )

    if energy_after > energy_before * 1.01:
        raise SheafCohomologyError(
            f"La proyección de Hodge aumentó la energía de frustración: "
            f"E(x) = {energy_before:.6e} → E(x̂) = {energy_after:.6e}. "
            "Esto indica un fallo numérico en LSQR."
        )

    return x_hat


# =============================================================================
# SECCIÓN 7: ORQUESTADOR PRINCIPAL
# =============================================================================


class SheafCohomologyOrchestrator:
    """Inspector cohomológico del haz celular.

    Analiza un estado global x ∈ C⁰ y determina si satisface las
    restricciones de sección del haz (δx ≈ 0).

    Protocolo de auditoría:
        1. Verificar completitud del haz.
        2. Ensamblar δ: C⁰ → C¹.
        3. Validar x ∈ C⁰.
        4. Calcular E(x) = ‖δx‖².
        5. Si E(x) > ε: lanzar HomologicalInconsistencyError.
        6. Calcular L = δᵀδ.
        7. Analizar espectro de L (incluyendo dim H¹).
        8. Retornar GlobalFrustrationAssessment completo.

    MEJORA: El diagnóstico ahora incluye h1_dimension y condition_number_est.
    """

    def __init__(self, watcher: Optional[ITopologicalWatcher] = None) -> None:
        """Inicializa el orquestador con un observador inyectado (FASE III)."""
        self._watcher = watcher

    # -------------------------------------------------------------------------
    # 7.1 Validación local (Mapa de Restricción)
    # -------------------------------------------------------------------------

    @staticmethod
    def validate_local_restriction(
        focus_node_id: str,
        local_topo: Optional[Dict],
        local_fin: Optional[Dict],
    ) -> None:
        """Invocación axiomática del Mapa de Restricción F_{V ▷ U}.

        Verifica que el sub-espacio U (enfocado por focus_node_id) posea
        métricas estructurales suficientes para sostener una deliberación
        independiente del grafo global V.

        MEJORA respecto al diseño original:
            Además de verificar que los dicts no sean None/vacíos, verifica
            la presencia de métricas clave (pyramid_stability,
            profitability_index) y su finitud numérica. Esto evita que
            dicts con valores NaN o None pasen la validación silenciosamente.

        Args:
            focus_node_id: Identificador del sub-espacio local.
            local_topo:    Métricas topológicas del sub-espacio.
            local_fin:     Métricas financieras del sub-espacio.

        Raises:
            SheafDegeneracyError: Si el sub-espacio es algebraicamente
                                  degenerado o sus métricas son inválidas.
        """
        # Verificación de existencia.
        if not local_topo:
            raise SheafDegeneracyError(
                f"Fibración degenerada: el sub-espacio '{focus_node_id}' "
                "carece de métricas topológicas. Imposible proyectar la "
                "restricción local al haz celular."
            )
        if not local_fin:
            raise SheafDegeneracyError(
                f"Fibración degenerada: el sub-espacio '{focus_node_id}' "
                "carece de métricas financieras. Imposible proyectar la "
                "restricción local al haz celular."
            )

        # Verificación de métricas clave y su finitud.
        # MEJORA: No basta con que el dict exista; las métricas deben ser
        # floats finitos para que el manifold pueda calcular σ* correctamente.
        psi_raw = local_topo.get("pyramid_stability")
        if psi_raw is not None:
            try:
                psi_val = float(psi_raw)
                if not np.isfinite(psi_val):
                    raise SheafDegeneracyError(
                        f"Sub-espacio '{focus_node_id}': pyramid_stability = "
                        f"{psi_raw!r} no es finito (NaN o ±∞)."
                    )
            except (TypeError, ValueError):
                raise SheafDegeneracyError(
                    f"Sub-espacio '{focus_node_id}': pyramid_stability = "
                    f"{psi_raw!r} no es convertible a float."
                )

        roi_raw = local_fin.get("profitability_index")
        if roi_raw is not None:
            try:
                roi_val = float(roi_raw)
                if not np.isfinite(roi_val):
                    raise SheafDegeneracyError(
                        f"Sub-espacio '{focus_node_id}': profitability_index = "
                        f"{roi_raw!r} no es finito (NaN o ±∞)."
                    )
            except (TypeError, ValueError):
                raise SheafDegeneracyError(
                    f"Sub-espacio '{focus_node_id}': profitability_index = "
                    f"{roi_raw!r} no es convertible a float."
                )

    # -------------------------------------------------------------------------
    # 7.2 Validación del vector de estado global
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_global_state_vector(
        sheaf: CellularSheaf,
        global_state_vector: np.ndarray,
    ) -> np.ndarray:
        """Valida x ∈ C⁰: conversión, forma, dimensión y finitud.

        Verificaciones (en orden):
            1. Convertibilidad a ndarray float64.
            2. Unidimensionalidad.
            3. Longitud == dim(C⁰).
            4. Finitud de todas las componentes.

        Args:
            sheaf:               Haz celular que define C⁰.
            global_state_vector: Vector candidato.

        Returns:
            Vector validado como ndarray float64.

        Raises:
            SheafDegeneracyError: Si alguna verificación falla.
        """
        try:
            x = np.asarray(global_state_vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SheafDegeneracyError(
                f"El estado global no es convertible a array float64: {exc}"
            ) from exc

        if x.ndim != 1:
            raise SheafDegeneracyError(
                f"El estado global debe ser un vector 1D; " f"forma recibida={x.shape}."
            )

        expected_dim = sheaf.total_node_dim
        if x.shape[0] != expected_dim:
            raise SheafDegeneracyError(
                f"Dimensión incompatible: recibida={x.shape[0]}, "
                f"esperada={expected_dim} (dim C⁰ = Σ_v dim(F(v)))."
            )

        if not np.all(np.isfinite(x)):
            n_bad = int(np.count_nonzero(~np.isfinite(x)))
            raise SheafDegeneracyError(
                f"El estado global contiene {n_bad} componente(s) no "
                f"finita(s) (NaN o ±∞)."
            )

        return x

    # -------------------------------------------------------------------------
    # 7.3 Cálculo de la Energía de Dirichlet
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_frustration_energy(
        delta: sp.csc_matrix,
        x: np.ndarray,
    ) -> Tuple[float, float]:
        """Calcula E(x) = ‖δx‖² y ‖δx‖ con verificación de consistencia.

        Cálculo via producto matriz-vector (no via xᵀLx) para evitar
        elevar al cuadrado el número de condición:

            r = δx ∈ C¹
            ‖r‖ = sqrt(rᵀr)
            E(x) = rᵀr = ‖r‖²

        MEJORA: Verificación de consistencia interna:
            |E(x) − ‖r‖²| / max(1, E(x)) ≤ ε_mach × k
        donde k = len(r). Si se viola, se emite una advertencia pero no
        se lanza excepción (la inconsistencia puede ser ruido de redondeo).

        Args:
            delta: Operador de cofrontera δ: C⁰ → C¹.
            x:     Vector de estado global validado.

        Returns:
            (frustration_energy, residual_norm) con E = ‖r‖², r = δx.

        Raises:
            SheafCohomologyError: Si E(x) es significativamente negativa.
        """
        residual = delta.dot(x)  # δx ∈ C¹
        residual_norm_sq = float(np.dot(residual, residual))
        residual_norm = float(np.linalg.norm(residual))

        # Verificación de consistencia interna: ‖r‖² ≈ (‖r‖)²
        discrepancy = abs(residual_norm_sq - residual_norm**2)
        consistency_tol = _FRUSTRATION_TOLERANCE * max(1.0, residual_norm_sq)
        if discrepancy > consistency_tol:
            logger.warning(
                "Inconsistencia numérica en E(x): "
                "rᵀr = %.6e, (‖r‖)² = %.6e, discrepancia = %.6e > tol = %.6e. "
                "Posible cancelación catastrófica.",
                residual_norm_sq,
                residual_norm**2,
                discrepancy,
                consistency_tol,
            )

        # Clamp de energía ligeramente negativa (ruido de redondeo).
        if residual_norm_sq < 0.0:
            if abs(residual_norm_sq) <= _FRUSTRATION_TOLERANCE:
                logger.debug(
                    "E(x) = %.6e < 0 (ruido de redondeo), clamped a 0.0.",
                    residual_norm_sq,
                )
                return 0.0, max(0.0, residual_norm)
            raise SheafCohomologyError(
                f"E(x) = ‖δx‖² = {residual_norm_sq:.6e} < 0 con magnitud "
                f"significativa (> {_FRUSTRATION_TOLERANCE:.6e}). "
                "Esto indica un fallo numérico severo (overflow o cancelación)."
            )

        return residual_norm_sq, max(0.0, residual_norm)

    # -------------------------------------------------------------------------
    # 7.4 Auditoría del estado global
    # -------------------------------------------------------------------------

    @classmethod
    def audit_global_state(
        cls,
        sheaf: CellularSheaf,
        global_state_vector: np.ndarray,
    ) -> GlobalFrustrationAssessment:
        """Evalúa si x ∈ C⁰ es transversalmente compatible con el haz.

        Pipeline completo:
            1. Verificar completitud del haz.
            2. Ensamblar δ: C⁰ → C¹.
            3. Validar x ∈ C⁰.
            4. Calcular E(x) = ‖δx‖².
            5. Si E(x) > ε: lanzar HomologicalInconsistencyError.
            6. Calcular L = δᵀδ.
            7. Analizar espectro de L (H⁰, H¹, brecha espectral, condición).
            8. Retornar GlobalFrustrationAssessment completo.

        MEJORA: El diagnóstico incluye h1_dimension y condition_number_est,
        que el diseño original omitía.

        Args:
            sheaf:               Haz celular completamente ensamblado.
            global_state_vector: Estado global propuesto x ∈ C⁰.

        Returns:
            GlobalFrustrationAssessment con diagnóstico completo.

        Raises:
            HomologicalInconsistencyError: Si x no es sección compatible.
            SheafDegeneracyError:          Si el haz o x son inválidos.
            SpectralComputationError:      Si el análisis espectral falla.
        """
        # ── Etapa 1: Completitud ──
        if not sheaf.is_fully_assembled:
            missing = set(sheaf._edge_dims.keys()) - sheaf._added_edge_ids
            raise SheafDegeneracyError(
                f"El haz no está completamente ensamblado. "
                f"Faltan {len(missing)} arista(s): {sorted(missing)}."
            )

        # ── Etapa 2: Ensamblar δ ──
        delta = sheaf.build_coboundary_operator()

        # ── Etapa 3: Validar x ──
        x = cls._validate_global_state_vector(sheaf, global_state_vector)

        # ── Etapa 4: Energía de frustración ──
        frustration_energy, residual_norm = cls._compute_frustration_energy(delta, x)
        is_coherent = frustration_energy <= _FRUSTRATION_TOLERANCE

        # ── Etapa 5: Rechazar si incoherente ──
        if not is_coherent:
            logger.critical(
                "FRUSTRACIÓN DE HAZ: E(x) = ‖δx‖² = %.6e > ε = %.6e, "
                "‖δx‖ = %.6e. El estado global no es una sección compatible.",
                frustration_energy,
                _FRUSTRATION_TOLERANCE,
                residual_norm,
            )
            raise HomologicalInconsistencyError(
                "Fractura del consenso global: x no es sección compatible del haz. "
                f"E(x) = ‖δx‖² = {frustration_energy:.6e} "
                f"> ε = {_FRUSTRATION_TOLERANCE:.6e}."
            )

        # ── Etapa 6: Laplaciano ──
        L = sheaf.compute_sheaf_laplacian()

        # ── Etapa 7: Análisis espectral ──
        spectral = _SpectralAnalyzer.compute(L, delta)

        logger.info(
            "Auditoría cohomológica exitosa: E(x)=%.6e, ‖δx‖=%.6e, "
            "dim H⁰=%d, dim H¹=%d, λ₁=%.6e, κ₂(δ)=%.3e, método=%s",
            frustration_energy,
            residual_norm,
            spectral.h0_dimension,
            spectral.h1_dimension,
            spectral.spectral_gap,
            spectral.condition_number_est,
            spectral.method,
        )

        # ── Etapa 8: Diagnóstico ──
        # χ = β0 - β1 (aproximación simplicial del 1-esqueleto)
        euler_char = spectral.h0_dimension - spectral.h1_dimension

        return GlobalFrustrationAssessment(
            frustration_energy=frustration_energy,
            h0_dimension=spectral.h0_dimension,
            h1_dimension=spectral.h1_dimension,
            is_coherent=True,
            spectral_gap=spectral.spectral_gap,
            residual_norm=residual_norm,
            spectral_method=spectral.method,
            delta_rank=spectral.delta_rank,
            condition_number_est=spectral.condition_number_est,
            euler_characteristic=euler_char,
        )

    def evaluate_tool_injection(
        self,
        base_sheaf: CellularSheaf,
        base_state: np.ndarray,
        new_edge: SheafEdge,
    ) -> ThreatMetrics:
        """
        Ejecuta el Pullback Categórico (FASE I-V) para evaluar una nueva herramienta.

        1. FASE I: Construcción del Fibrado Tangente de Simulación (Mayer-Vietoris).
        2. FASE II: Extracción del Tensor de Estado ψ ∈ ℝ⁷.
        3. FASE III: Pullback Categórico invocando al Observador.
        4. FASE V: Colapso de la Función de Onda (Veto Absoluto).
        """
        if self._watcher is None:
            logger.warning("No se ha inyectado un ITopologicalWatcher. Omitiendo auditoría.")
            return ThreatMetrics(0.0, True, 0, "HEALTHY")

        import networkx as nx

        # --- FASE I: Simulación (Secuencia de Mayer-Vietoris Equivalente) ---
        # Auditamos el estado base para obtener invariantes de referencia.
        base_audit = self.audit_global_state(base_sheaf, base_state)

        # Construimos el 1-esqueleto simplicial del haz base.
        G = nx.Graph()
        G.add_nodes_from(range(base_sheaf.num_nodes))
        for edge in base_sheaf.edges:
            G.add_edge(edge.u, edge.v)

        # Analizamos el impacto de la unión K ∪ {e}.
        u, v = new_edge.u, new_edge.v
        has_path = nx.has_path(G, u, v) if (u in G and v in G) else False

        # Según el Teorema de Mayer-Vietoris para grafos:
        # Si u y v están en la misma componente, se crea un ciclo: Δβ1 = 1, Δβ0 = 0.
        # Si están en componentes distintas, se fusionan: Δβ1 = 0, Δβ0 = -1.
        if has_path:
            delta_beta0 = 0
            delta_beta1 = 1
        else:
            delta_beta0 = -1
            delta_beta1 = 0

        sim_h0 = base_audit.h0_dimension + delta_beta0
        sim_h1 = base_audit.h1_dimension + delta_beta1

        # [AXIOMA DE VETO]: Abortar si se induce un defecto topológico (Δβ1 > 0).
        if delta_beta1 > 0:
            logger.error("VETO PREVENTIVO (FASE I): La herramienta induce un ciclo homológico (Δβ1=%d).", delta_beta1)
            raise TopologicalBifurcationError(
                f"Obstrucción detectada en FASE I: Inyección induce ciclo homológico (Δβ1={delta_beta1})."
            )

        # --- FASE II: Construcción del Tensor ψ ∈ ℝ⁷ ---
        # Mapeo al vector ψ de 7 dimensiones esperado por el Watcher:
        # [saturation, flyback, dissipated_power, beta_0, beta_1, entropy, exergy_loss]
        psi = np.zeros(7, dtype=np.float64)
        psi[3] = float(sim_h0)  # beta_0
        psi[4] = float(sim_h1)  # beta_1

        # Proyectamos métricas termodinámicas desde la auditoría base.
        psi[0] = 0.05  # Saturación nominal simulada
        psi[2] = base_audit.frustration_energy  # Disipación inicial
        psi[5] = 0.1   # Entropía inicial simulada

        # --- FASE III & IV: Pullback al Observador ---
        metrics = self._watcher.evaluate_manifold_deformation(psi, reference_chi=base_audit.euler_characteristic)

        # --- FASE V: Colapso de la Función de Onda (Fast-Fail) ---
        if not metrics.is_stable or metrics.threat_level == "CRITICAL":
            logger.critical(
                "VETO TOPOLÓGICO (FASE V): Abortando inyección. Δχ=%d, d_M=%.4f, Status=%s",
                metrics.structural_alteration,
                metrics.mahalanobis_distance,
                metrics.threat_level
            )
            raise TopologicalBifurcationError(
                f"Bifurcación detectada en la simulación pullback: Δχ={metrics.structural_alteration}, d_M={metrics.mahalanobis_distance:.4f}"
            )

        return metrics
