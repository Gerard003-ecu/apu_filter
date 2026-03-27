r"""
=========================================================================================
Módulo: Sheaf Cohomology Orchestrator (Interferómetro de Holonomía de Gauge)
Ubicación: app/core/immune_system/calibration/sheaf_cohomology_orchestrator.py
=========================================================================================

Naturaleza Ciber-Física:
    Validador de congruencia topológica global para la Malla Agéntica mediante la
    Teoría de Haces Celulares (Cellular Sheaves). Actúa como el sistema de 
    propiocepción de la red, distinguiendo entre ruido termodinámico resoluble y 
    paradojas lógicas insalvables (obstrucciones topológicas).

Fundamentación Matemática y Topología Algebraica:
    Sea G = (V, E) un grafo dirigido representando la malla de decisión.
    Un haz celular ℱ asigna a la topología un fibrado de espacios vectoriales:
        • F(v) ≅ ℝ^{d_v} a cada nodo v ∈ V (Espacio de estado local del agente).
        • F(e) ≅ ℝ^{d_e} a cada arista e ∈ E (Espacio del contrato de interfaz).
    
    El acoplamiento se rige por mapas de restricción lineales F_{v ◁ e}: F(v) → F(e),
    los cuales proyectan la "verdad local" de un agente hacia el contrato compartido.

Complejo de Cocadenas y Operadores:
    El sistema modela el flujo de información sobre la secuencia exacta corta:
        C⁰(G; ℱ) xrightarrow{\delta} C¹(G; ℱ)
    Donde C⁰ es el espacio de 0-cocadenas (estados globales) y C¹ es el espacio 
    de 1-cocadenas (evaluaciones de arista). 
    El operador co-frontera (Coboundary) δ evalúa la divergencia discreta de la red.

Termodinámica de la Información (Energía de Dirichlet):
    El Laplaciano del Haz se define axiomáticamente como L = δᵀδ [2]. Sin embargo, por
    rigor de análisis numérico, L JAMÁS se ensambla explícitamente para evitar elevar al 
    cuadrado el número de condición κ(L) = κ(δ)².
    La "Frustración Global" o fricción operativa se cuantifica mediante la Energía 
    de Dirichlet del estado x:
        E(x) = xᵀLx = ‖δx‖² ≥ 0.

Invariantes Cohomológicos y Veredicto Estructural:
    El módulo extrae el espectro de la variedad mediante Descomposición en Valores 
    Singulares Dispersa (Sparse SVD) sobre δ, donde λ_i(L) = σ_i(δ)², garantizando
    convergencia asintótica y evadiendo el colapso del solver de Lanczos.

    1. H⁰(G; ℱ) ≅ ker(δ) = ker(L): 
       Dimensión del núcleo. Mide los grados de libertad para un consenso absoluto.
    2. Brecha Espectral (λ₁ > 0): 
       El primer autovalor no nulo (Fiedler value del haz). Cuantifica la "rigidez" 
       del consenso frente a inyecciones de ruido estocástico.
    3. H¹(G; ℱ) ≅ coker(δ): 
       Calculado estrictamente mediante el Teorema de Rango-Nulidad: 
       dim H¹ = dim C¹ - rank(δ). 
       [AXIOMA DE VETO]: Si dim H¹ > 0, existe una "Obstrucción Topológica Absoluta".
       Las directrices de los agentes conforman una paradoja irreconciliable.

Resolución y Fagocitosis (Teorema de Hodge Discreto):
    Si E(x) > ε pero H¹ = 0, el conflicto es ruido homotópico. El sistema no aborta; 
    ejecuta una Proyección de Hodge-Helmholtz ortogonal sobre ker(L), resolviendo 
    la ecuación de Poisson del haz para inyectar a los agentes hacia la sección 
    global matemáticamente válida más cercana, sanando el vector de estado.
=========================================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackError, eigsh

logger = logging.getLogger("MIC.ImmuneSystem.SheafCohomology")

# ═══════════════════════════════════════════════════════════════════════════════
# TOLERANCIAS NUMÉRICAS CON JUSTIFICACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

# Tolerancia para clasificar energía de frustración como "coherente"
# Justificación: ε_mach^{2/3} ≈ 3.7e-11 es una cota estándar para
# residuos de problemas lineales bien condicionados; usamos 1e-9 como
# margen conservador que acomoda errores de la malla agéntica
_FRUSTRATION_TOLERANCE: Final[float] = 1e-9

# Tolerancia para verificar simetría de L = δᵀδ
# Para matrices SPD construidas como AᵀA, la asimetría numérica es O(ε·‖A‖²)
_SYMMETRY_TOLERANCE: Final[float] = 1e-12

# Tolerancia espectral para clasificar eigenvalores como "cero" o "positivo"
# Separada de _FRUSTRATION_TOLERANCE para independizar el análisis espectral
# del análisis de energía de estados específicos
_SPECTRAL_TOLERANCE: Final[float] = 1e-9

# Dimensión máxima para usar eigendecomposición densa O(n³)
# Por encima, se usa ARPACK iterativo O(n·k²) con k eigenvalores
_DENSE_SPECTRAL_MAX_DIM: Final[int] = 256

# Número máximo de eigenvalores a solicitar en modo disperso
_SPARSE_MAX_EIGENVALUES: Final[int] = 8

# Tolerancia de convergencia para eigsh (ARPACK)
_ARPACK_TOLERANCE: Final[float] = 1e-7


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES ALGEBRAICAS
# ═══════════════════════════════════════════════════════════════════════════════

class SheafCohomologyError(Exception):
    """Excepción base para fallos en el análisis cohomológico del haz."""


class HomologicalInconsistencyError(SheafCohomologyError):
    """
    Lanzada cuando un estado global propuesto no satisface las restricciones
    del haz y, por tanto, no constituye una sección global compatible.
    
    Semántica: E(x) = ‖δx‖² > _FRUSTRATION_TOLERANCE
    """


class SheafDegeneracyError(SheafCohomologyError):
    """
    Lanzada cuando las dimensiones, mapas de restricción o ensamblajes del haz
    son algebraicamente incoherentes o degenerados.
    
    Ejemplos: dimensiones incompatibles, matrices no finitas, grafos vacíos.
    """


class SpectralComputationError(SheafCohomologyError):
    """
    Lanzada cuando el cálculo espectral del Laplaciano falla de forma
    irrecuperable (convergencia de ARPACK, eigenvalores negativos severos).
    """


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class RestrictionMap:
    """
    Mapa lineal F_{v ▷ e}: F(v) → F(e).

    Si dim(F(v)) = n y dim(F(e)) = m, entonces la matriz asociada debe tener
    forma (m, n), representando una transformación lineal ℝⁿ → ℝᵐ.

    La matriz interna se almacena como read-only para garantizar inmutabilidad
    algebraica del haz una vez construido.
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
                f"El mapa de restricción debe ser una matriz 2D, "
                f"recibido ndim={M.ndim}."
            )

        if M.shape[0] == 0 or M.shape[1] == 0:
            raise SheafDegeneracyError(
                f"El mapa de restricción tiene dimensión degenerada: "
                f"forma={M.shape}."
            )

        if not np.all(np.isfinite(M)):
            non_finite_count = int(np.count_nonzero(~np.isfinite(M)))
            raise SheafDegeneracyError(
                f"El mapa de restricción contiene {non_finite_count} "
                f"entrada(s) no finita(s) (NaN o ±∞)."
            )

        # Inmutabilizar la matriz para proteger la geometría del haz
        M.setflags(write=False)
        object.__setattr__(self, "matrix", M)

    @property
    def domain_dim(self) -> int:
        """Dimensión del dominio (número de columnas)."""
        return int(self.matrix.shape[1])

    @property
    def codomain_dim(self) -> int:
        """Dimensión del codominio (número de filas)."""
        return int(self.matrix.shape[0])


@dataclass(frozen=True, slots=True)
class SheafEdge:
    """
    Descriptor inmutable de una arista orientada del haz.

    La orientación canónica es u → v. El operador de cofrontera usa
    el convenio:
        (δx)_e = F_{v ▷ e} x_v − F_{u ▷ e} x_u

    Atributos:
        edge_id: Identificador único de la arista
        u: Nodo origen (según orientación del haz)
        v: Nodo destino (según orientación del haz)
        restriction_u: Mapa F_{u ▷ e}: F(u) → F(e)
        restriction_v: Mapa F_{v ▷ e}: F(v) → F(e)
    """
    edge_id: int
    u: int
    v: int
    restriction_u: RestrictionMap
    restriction_v: RestrictionMap


@dataclass(frozen=True, slots=True)
class SpectralInvariants:
    """
    Invariantes espectrales del Laplaciano del haz L = δᵀδ.

    Atributos:
        h0_dimension: dim ker(L) = dim H⁰(G; F), el número de componentes
                      de consenso independientes
        spectral_gap: Menor eigenvalor estrictamente positivo de L (λ₁),
                      que mide la robustez del consenso. Si todos los
                      eigenvalores son cero, spectral_gap = 0.0
        smallest_eigenvalues: Vector de los eigenvalores más pequeños
                             computados (inmutable)
        method: 'dense' o 'sparse', indicando el método de cálculo
    """
    h0_dimension: int
    spectral_gap: float
    smallest_eigenvalues: np.ndarray
    method: str

    def __post_init__(self) -> None:
        if self.h0_dimension < 0:
            raise ValueError(
                f"h0_dimension debe ser no negativo, recibido: {self.h0_dimension}"
            )
        if self.spectral_gap < 0.0:
            raise ValueError(
                f"spectral_gap debe ser no negativo, recibido: {self.spectral_gap}"
            )
        if self.method not in ("dense", "sparse"):
            raise ValueError(
                f"method debe ser 'dense' o 'sparse', recibido: {self.method!r}"
            )


@dataclass(frozen=True, slots=True)
class GlobalFrustrationAssessment:
    """
    Diagnóstico inmutable del estado cohomológico del ecosistema.

    Atributos:
        frustration_energy: Energía de Dirichlet E(x) = ‖δx‖²
        h0_dimension: dim ker(L) = dim H⁰(G; F)
        is_coherent: True si E(x) ≤ _FRUSTRATION_TOLERANCE
        spectral_gap: Menor eigenvalor estrictamente positivo de L
        residual_norm: ‖δx‖ (norma L² del residuo, no al cuadrado)
        spectral_method: 'dense' o 'sparse'
    """
    frustration_energy: float
    h0_dimension: int
    is_coherent: bool
    spectral_gap: float
    residual_norm: float
    spectral_method: str


# ═══════════════════════════════════════════════════════════════════════════════
# EL COMPLEJO DEL HAZ
# ═══════════════════════════════════════════════════════════════════════════════

class CellularSheaf:
    """
    Estructura matemática de un haz celular sobre una malla agéntica.

    El espacio de 0-cochains es:
        C⁰ = ⨁_{v ∈ V} F(v)    con dim C⁰ = Σ_v d_v

    El espacio de 1-cochains es:
        C¹ = ⨁_{e ∈ E} F(e)    con dim C¹ = Σ_e d_e

    El operador δ: C⁰ → C¹ se ensambla por bloques usando las restricciones
    en cada arista. El Laplaciano L = δᵀδ hereda simetría y semi-positividad.

    Invariantes de clase:
    ─────────────────────
    - Los nodos están indexados 0, 1, ..., num_nodes-1
    - Las aristas tienen edge_id únicos definidos en edge_dims
    - Cada arista se añade exactamente una vez
    - Los pares (u, v) son únicos (grafo simple, no multigrafo)
    - Las dimensiones de los mapas de restricción son consistentes
    """

    def __init__(
        self,
        num_nodes: int,
        node_dims: Dict[int, int],
        edge_dims: Dict[int, int],
    ) -> None:
        if not isinstance(num_nodes, int) or num_nodes <= 0:
            raise SheafDegeneracyError(
                f"num_nodes debe ser un entero positivo; recibido={num_nodes!r}."
            )

        self._num_nodes: Final[int] = num_nodes
        self._node_dims: Final[Dict[int, int]] = self._validate_node_dims(
            node_dims, num_nodes
        )
        self._edge_dims: Final[Dict[int, int]] = self._validate_edge_dims(
            edge_dims
        )
        self._edges: List[SheafEdge] = []
        self._added_edge_ids: set[int] = set()
        self._added_node_pairs: set[frozenset[int]] = set()

        self._node_offsets: Final[np.ndarray] = self._compute_offsets(
            self._node_dims, self._num_nodes
        )
        self._edge_offsets: Final[Dict[int, int]] = self._compute_edge_offsets_static(
            self._edge_dims
        )

        # Cachear dimensiones totales (invariantes una vez construido)
        self._total_node_dim: Final[int] = int(self._node_offsets[-1])
        self._total_edge_dim: Final[int] = int(
            sum(self._edge_dims.values())
        )

        # Caché del operador de cofrontera (invalidado al añadir aristas)
        self._cached_coboundary: Optional[sp.csc_matrix] = None

    # ─────────────────────────────────────────────────────────────────────────
    # PROPIEDADES DE SOLO LECTURA
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        """Número de nodos del grafo base."""
        return self._num_nodes

    @property
    def node_dims(self) -> Dict[int, int]:
        """Dimensiones de los espacios de fibra nodales (copia defensiva)."""
        return dict(self._node_dims)

    @property
    def edge_dims(self) -> Dict[int, int]:
        """Dimensiones de los espacios de fibra de arista (copia defensiva)."""
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

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDACIÓN ESTÁTICA
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_node_dims(
        node_dims: Dict[int, int],
        num_nodes: int,
    ) -> Dict[int, int]:
        """
        Valida que cada nodo 0..num_nodes-1 tenga dimensión entera positiva
        y que no existan claves espurias.
        """
        if not isinstance(node_dims, dict):
            raise SheafDegeneracyError(
                "node_dims debe ser un diccionario {nodo: dimensión}."
            )

        expected_keys = set(range(num_nodes))
        actual_keys = set(node_dims.keys())

        missing = expected_keys - actual_keys
        if missing:
            raise SheafDegeneracyError(
                f"Faltan dimensiones para nodos: {sorted(missing)}."
            )

        extra = actual_keys - expected_keys
        if extra:
            raise SheafDegeneracyError(
                f"node_dims contiene claves fuera del rango nodal válido "
                f"[0, {num_nodes}): {sorted(extra)}."
            )

        validated: Dict[int, int] = {}
        for i in range(num_nodes):
            dim = node_dims[i]
            if not isinstance(dim, int) or dim <= 0:
                raise SheafDegeneracyError(
                    f"Dimensión inválida para nodo {i}: {dim!r}. "
                    "Debe ser entero positivo."
                )
            validated[i] = dim

        return validated

    @staticmethod
    def _validate_edge_dims(edge_dims: Dict[int, int]) -> Dict[int, int]:
        """
        Valida que cada arista tenga dimensión entera positiva y que el
        diccionario no esté vacío (un haz sin aristas es degenerado para
        análisis cohomológico).
        """
        if not isinstance(edge_dims, dict):
            raise SheafDegeneracyError(
                "edge_dims debe ser un diccionario {arista: dimensión}."
            )

        if len(edge_dims) == 0:
            raise SheafDegeneracyError(
                "edge_dims está vacío. Un haz sin aristas tiene δ = 0 y "
                "H⁰ = C⁰ trivialmente, lo cual no proporciona información "
                "de consistencia inter-agente."
            )

        validated: Dict[int, int] = {}
        for edge_id, dim in edge_dims.items():
            if not isinstance(edge_id, int) or edge_id < 0:
                raise SheafDegeneracyError(
                    f"Identificador de arista inválido: {edge_id!r}. "
                    "Debe ser entero no negativo."
                )
            if not isinstance(dim, int) or dim <= 0:
                raise SheafDegeneracyError(
                    f"Dimensión inválida para arista {edge_id}: {dim!r}. "
                    "Debe ser entero positivo."
                )
            validated[edge_id] = dim

        return validated

    @staticmethod
    def _compute_offsets(
        dims_map: Dict[int, int],
        count: int,
    ) -> np.ndarray:
        """
        Calcula offsets acumulados para ensamblaje por bloques.

        Si dims_map[i] = d_i para i = 0, ..., count-1, produce:
            offsets[k] = Σ_{i=0}^{k-1} d_i

        con offsets[0] = 0 y offsets[count] = dim(C⁰).
        """
        offsets = np.zeros(count + 1, dtype=np.int64)
        for i in range(count):
            offsets[i + 1] = offsets[i] + dims_map[i]
        return offsets

    @staticmethod
    def _compute_edge_offsets_static(
        edge_dims: Dict[int, int],
    ) -> Dict[int, int]:
        """
        Calcula offsets acumulados en C¹ indexados por edge_id.
        El orden es por edge_id creciente para reproducibilidad determinista.
        """
        offsets: Dict[int, int] = {}
        running = 0
        for edge_id in sorted(edge_dims):
            offsets[edge_id] = running
            running += edge_dims[edge_id]
        return offsets

    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN DEL HAZ
    # ─────────────────────────────────────────────────────────────────────────

    def add_edge(
        self,
        edge_id: int,
        u: int,
        v: int,
        F_ue: RestrictionMap,
        F_ve: RestrictionMap,
    ) -> None:
        """
        Añade una arista e = (u, v) con mapas de restricción:
            F_{u ▷ e}: F(u) → F(e)
            F_{v ▷ e}: F(v) → F(e)

        Precondiciones verificadas:
        ───────────────────────────
        1. edge_id existe en edge_dims
        2. edge_id no ha sido añadido previamente
        3. u, v son nodos válidos en [0, num_nodes)
        4. u ≠ v (sin lazos degenerados)
        5. El par {u, v} no existe ya (grafo simple)
        6. F_ue.matrix.shape == (d_e, d_u)
        7. F_ve.matrix.shape == (d_e, d_v)

        Args:
            edge_id: Identificador de la arista (debe estar en edge_dims)
            u: Nodo origen
            v: Nodo destino
            F_ue: Mapa de restricción F_{u ▷ e}
            F_ve: Mapa de restricción F_{v ▷ e}

        Raises:
            SheafDegeneracyError: Si alguna precondición falla
        """
        # Verificación 1: edge_id válido
        if edge_id not in self._edge_dims:
            raise SheafDegeneracyError(
                f"La arista {edge_id} no existe en edge_dims. "
                f"Aristas válidas: {sorted(self._edge_dims.keys())}."
            )

        # Verificación 2: edge_id no duplicado
        if edge_id in self._added_edge_ids:
            raise SheafDegeneracyError(
                f"La arista {edge_id} ya fue añadida al haz."
            )

        # Verificación 3: nodos válidos
        if not (0 <= u < self._num_nodes):
            raise SheafDegeneracyError(
                f"Nodo u={u} fuera de rango [0, {self._num_nodes})."
            )
        if not (0 <= v < self._num_nodes):
            raise SheafDegeneracyError(
                f"Nodo v={v} fuera de rango [0, {self._num_nodes})."
            )

        # Verificación 4: sin lazos
        if u == v:
            raise SheafDegeneracyError(
                f"La arista {edge_id} no puede ser un lazo: u=v={u}. "
                "Los lazos son algebraicamente degenerados para δ ya que "
                "F_{u▷e}x_u − F_{u▷e}x_u = 0 trivialmente."
            )

        # Verificación 5: par {u, v} único (grafo simple)
        node_pair = frozenset({u, v})
        if node_pair in self._added_node_pairs:
            raise SheafDegeneracyError(
                f"Ya existe una arista entre los nodos {u} y {v}. "
                "El haz opera sobre un grafo simple (sin multiaristas). "
                "Para modelar múltiples relaciones entre agentes, "
                "incremente la dimensión del espacio de fibra de arista."
            )

        # Verificación 6-7: compatibilidad dimensional
        edge_dim = self._edge_dims[edge_id]
        expected_u_shape = (edge_dim, self._node_dims[u])
        expected_v_shape = (edge_dim, self._node_dims[v])

        if F_ue.matrix.shape != expected_u_shape:
            raise SheafDegeneracyError(
                f"Incoherencia dimensional en Arista {edge_id}: mapa F_{{u▷e}} tiene forma "
                f"{F_ue.matrix.shape}, esperada {expected_u_shape}. "
                f"dim(F(e))={edge_dim}, dim(F(u={u}))={self._node_dims[u]}."
            )

        if F_ve.matrix.shape != expected_v_shape:
            raise SheafDegeneracyError(
                f"Incoherencia dimensional en Arista {edge_id}: mapa F_{{v▷e}} tiene forma "
                f"{F_ve.matrix.shape}, esperada {expected_v_shape}. "
                f"dim(F(e))={edge_dim}, dim(F(v={v}))={self._node_dims[v]}."
            )

        # Registrar arista
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

        # Invalidar caché del operador de cofrontera
        self._cached_coboundary = None

        logger.debug(
            "Arista %d añadida: (%d → %d), dim(F(e))=%d, "
            "dim(F(u))=%d, dim(F(v))=%d",
            edge_id, u, v, edge_dim,
            self._node_dims[u], self._node_dims[v],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ENSAMBLAJE DEL OPERADOR DE COFRONTERA
    # ─────────────────────────────────────────────────────────────────────────

    def _assert_fully_assembled(self) -> None:
        """
        Verifica que todas las aristas declaradas en edge_dims hayan sido
        añadidas al haz.

        Raises:
            SheafDegeneracyError: Si faltan aristas
        """
        missing = set(self._edge_dims.keys()) - self._added_edge_ids
        if missing:
            raise SheafDegeneracyError(
                f"El haz no está completamente ensamblado. "
                f"Faltan aristas: {list(sorted(missing))}. "
                f"Añadidas: {len(self._edges)}/{len(self._edge_dims)}."
            )

    def build_coboundary_operator(self) -> sp.csc_matrix:
        """
        Construye la matriz dispersa del operador de cofrontera δ: C⁰ → C¹.

        Para cada arista e = (u, v) con orientación u → v, el bloque
        correspondiente en δ es:

            δ_e = [−F_{u▷e} | +F_{v▷e}]

        donde −F_{u▷e} ocupa las columnas de u y +F_{v▷e} las columnas de v,
        en las filas de e.

        Implementación vectorizada:
        ───────────────────────────
        Para cada arista, se insertan los bloques de restricción usando
        meshgrid de índices, evitando bucles Python sobre entradas
        individuales de la matriz.

        Returns:
            Matriz dispersa CSC de forma (total_edge_dim, total_node_dim)

        Raises:
            SheafDegeneracyError: Si el haz no está completamente ensamblado
                                  o el resultado contiene valores no finitos
        """
        # Retornar caché si disponible
        if self._cached_coboundary is not None:
            return self._cached_coboundary

        self._assert_fully_assembled()

        total_edge_dim = self._total_edge_dim
        total_node_dim = self._total_node_dim

        # Pre-estimar la cantidad de no-ceros para pre-asignación
        estimated_nnz = sum(
            self._edge_dims[e.edge_id] * (self._node_dims[e.u] + self._node_dims[e.v])
            for e in self._edges
        )

        data = np.empty(estimated_nnz, dtype=np.float64)
        row_indices = np.empty(estimated_nnz, dtype=np.int64)
        col_indices = np.empty(estimated_nnz, dtype=np.int64)
        ptr = 0  # Puntero de escritura en los arrays pre-asignados

        for edge in self._edges:
            edge_row_offset = self._edge_offsets[edge.edge_id]
            u_col_offset = int(self._node_offsets[edge.u])
            v_col_offset = int(self._node_offsets[edge.v])

            F_u = edge.restriction_u.matrix  # (d_e, d_u)
            F_v = edge.restriction_v.matrix  # (d_e, d_v)

            d_e, d_u = F_u.shape
            _, d_v = F_v.shape

            # ── Bloque −F_{u▷e} (vectorizado) ──
            block_size_u = d_e * d_u
            rows_u, cols_u = np.meshgrid(
                np.arange(d_e) + edge_row_offset,
                np.arange(d_u) + u_col_offset,
                indexing="ij",
            )
            data[ptr:ptr + block_size_u] = (-F_u).ravel()
            row_indices[ptr:ptr + block_size_u] = rows_u.ravel()
            col_indices[ptr:ptr + block_size_u] = cols_u.ravel()
            ptr += block_size_u

            # ── Bloque +F_{v▷e} (vectorizado) ──
            block_size_v = d_e * d_v
            rows_v, cols_v = np.meshgrid(
                np.arange(d_e) + edge_row_offset,
                np.arange(d_v) + v_col_offset,
                indexing="ij",
            )
            data[ptr:ptr + block_size_v] = F_v.ravel()
            row_indices[ptr:ptr + block_size_v] = rows_v.ravel()
            col_indices[ptr:ptr + block_size_v] = cols_v.ravel()
            ptr += block_size_v

        # Truncar a la cantidad real de entradas (ptr puede ser < estimated_nnz
        # solo si la estimación fue incorrecta, lo cual no debería pasar)
        data = data[:ptr]
        row_indices = row_indices[:ptr]
        col_indices = col_indices[:ptr]

        delta = sp.csc_matrix(
            (data, (row_indices, col_indices)),
            shape=(total_edge_dim, total_node_dim),
            dtype=np.float64,
        )

        # Verificar finitud post-ensamblaje
        if delta.nnz > 0 and not np.all(np.isfinite(delta.data)):
            raise SheafDegeneracyError(
                "El operador de cofrontera ensamblado contiene valores no "
                "finitos. Esto indica corrupción en los mapas de restricción."
            )

        # Cachear resultado
        self._cached_coboundary = delta

        logger.debug(
            "Operador δ ensamblado: forma=%s, nnz=%d, densidad=%.4f%%",
            delta.shape, delta.nnz,
            100.0 * delta.nnz / max(1, delta.shape[0] * delta.shape[1]),
        )

        return delta

    def compute_sheaf_laplacian(self) -> sp.csc_matrix:
        """
        Calcula el Laplaciano del haz L = δᵀδ.

        Propiedades garantizadas (por construcción como AᵀA):
        ─────────────────────────────────────────────────────
        1. L ∈ ℝ^{n×n} con n = dim(C⁰)
        2. L = Lᵀ (simétrica)
        3. L ⪰ 0 (semidefinida positiva): xᵀLx = ‖δx‖² ≥ 0
        4. ker(L) = ker(δ) = H⁰(G; F)

        La propiedad 4 se demuestra así:
            Lx = 0 ⟹ xᵀLx = 0 ⟹ ‖δx‖² = 0 ⟹ δx = 0

        Returns:
            Matriz dispersa CSC simétrica semidefinida positiva

        Raises:
            SheafDegeneracyError: Si L contiene valores no finitos
        """
        delta = self.build_coboundary_operator()
        L = (delta.T @ delta).tocsc()

        if L.nnz > 0 and not np.all(np.isfinite(L.data)):
            raise SheafDegeneracyError(
                "El Laplaciano del haz contiene valores no finitos. "
                "Esto puede deberse a entradas de magnitud extrema en "
                "los mapas de restricción que causan overflow en δᵀδ."
            )

        return L


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS ESPECTRAL
# ═══════════════════════════════════════════════════════════════════════════════

class _SpectralAnalyzer:
    """
    Analizador espectral interno para el Laplaciano del haz.

    Estrategia híbrida:
    ───────────────────
    - dim ≤ _DENSE_SPECTRAL_MAX_DIM: eigendecomposición densa exacta O(n³)
    - dim > _DENSE_SPECTRAL_MAX_DIM: ARPACK iterativo O(n·k²) con k
      eigenvalores más pequeños

    La separación en clase interna permite testear el análisis espectral
    independientemente del orquestador.
    """

    @staticmethod
    def _verify_laplacian_symmetry(L_dense: np.ndarray) -> None:
        """
        Verifica simetría del Laplaciano denso.

        Para L = δᵀδ, la simetría es garantizada algebraicamente, pero
        puede degradarse numéricamente para matrices grandes con entradas
        de magnitudes dispares.

        Raises:
            SheafCohomologyError: Si la asimetría excede tolerancia
        """
        asymmetry_norm = float(np.linalg.norm(L_dense - L_dense.T, "fro"))
        L_norm = float(np.linalg.norm(L_dense, "fro"))

        # Tolerancia adaptativa: max(absoluta, relativa)
        tol = max(_SYMMETRY_TOLERANCE, _SYMMETRY_TOLERANCE * L_norm)

        if asymmetry_norm > tol:
            raise SheafCohomologyError(
                f"El Laplaciano del haz no es simétrico dentro de tolerancia. "
                f"‖L − Lᵀ‖_F = {asymmetry_norm:.6e}, "
                f"‖L‖_F = {L_norm:.6e}, "
                f"tol = {tol:.6e}."
            )

    @staticmethod
    def _verify_semidefinite_positivity(
        eigenvalues: np.ndarray,
        method: str,
    ) -> None:
        """
        Verifica que L sea semidefinida positiva (todos eigenvalores ≥ −ε).

        Un eigenvalor significativamente negativo indica un error de
        ensamblaje o corrupción numérica.

        Raises:
            SpectralComputationError: Si hay eigenvalores negativos severos
        """
        min_eigenvalue = float(eigenvalues[0])
        if min_eigenvalue < -_SPECTRAL_TOLERANCE:
            raise SpectralComputationError(
                f"El Laplaciano del haz no es semidefinido positivo. "
                f"min(λ) = {min_eigenvalue:.6e} < −{_SPECTRAL_TOLERANCE:.6e}. "
                f"Método: {method}. "
                "Esto indica un error de ensamblaje del haz o corrupción "
                "numérica severa."
            )

    @staticmethod
    def _classify_eigenvalues(
        eigenvalues: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Clasifica eigenvalores en cero/positivo para extraer invariantes.

        Un eigenvalor λ se clasifica como:
        - cero si |λ| ≤ _SPECTRAL_TOLERANCE
        - positivo si λ > _SPECTRAL_TOLERANCE

        Returns:
            (h0_dimension, spectral_gap) donde spectral_gap es el menor
            eigenvalor estrictamente positivo, o 0.0 si no existe
        """
        zero_mask = np.abs(eigenvalues) <= _SPECTRAL_TOLERANCE
        h0_dim = int(np.sum(zero_mask))

        positive_eigenvalues = eigenvalues[eigenvalues > _SPECTRAL_TOLERANCE]
        if positive_eigenvalues.size > 0:
            spectral_gap = float(positive_eigenvalues[0])
        else:
            spectral_gap = 0.0

        return h0_dim, spectral_gap

    @classmethod
    def compute_dense(cls, L: sp.csc_matrix) -> SpectralInvariants:
        """
        Eigendecomposición densa exacta del Laplaciano.

        Convierte L a array denso y usa LAPACK (eigvalsh) para obtener
        todos los eigenvalores con precisión de máquina.

        Args:
            L: Laplaciano del haz en formato disperso

        Returns:
            SpectralInvariants con todos los eigenvalores

        Raises:
            SheafCohomologyError: Si L no es simétrico
            SpectralComputationError: Si L no es semidefinida positiva
        """
        L_dense = L.toarray()

        cls._verify_laplacian_symmetry(L_dense)

        eigenvalues = np.linalg.eigvalsh(L_dense)
        eigenvalues = np.sort(eigenvalues.astype(np.float64))

        cls._verify_semidefinite_positivity(eigenvalues, method="dense")

        h0_dim, spectral_gap = cls._classify_eigenvalues(eigenvalues)

        # Inmutabilizar eigenvalores
        eigenvalues_immutable = eigenvalues.copy()
        eigenvalues_immutable.setflags(write=False)

        return SpectralInvariants(
            h0_dimension=h0_dim,
            spectral_gap=spectral_gap,
            smallest_eigenvalues=eigenvalues_immutable,
            method="dense",
        )

    @classmethod
    def compute_sparse(cls, L: sp.csc_matrix) -> SpectralInvariants:
        """
        Estimación dispersa de invariantes espectrales via ARPACK.

        Solicita los k eigenvalores más pequeños de L usando el algoritmo
        de Lanczos implícitamente reiniciado (eigsh con which='SM').

        Limitaciones:
        ─────────────
        - Solo computa k eigenvalores, no el espectro completo
        - La nulidad es una cota inferior: podrían existir más eigenvalores
          cero fuera de los k solicitados
        - ARPACK puede no converger para matrices muy mal condicionadas

        Args:
            L: Laplaciano del haz en formato disperso

        Returns:
            SpectralInvariants con los k eigenvalores más pequeños

        Raises:
            SpectralComputationError: Si ARPACK falla o L no es semi-positiva
        """
        n = L.shape[0]

        if n == 0:
            empty_eigs = np.array([], dtype=np.float64)
            empty_eigs.setflags(write=False)
            return SpectralInvariants(
                h0_dimension=0,
                spectral_gap=0.0,
                smallest_eigenvalues=empty_eigs,
                method="sparse",
            )

        if n == 1:
            val = float(L[0, 0])
            eigs = np.array([val], dtype=np.float64)
            eigs.setflags(write=False)

            if abs(val) <= _SPECTRAL_TOLERANCE:
                return SpectralInvariants(
                    h0_dimension=1,
                    spectral_gap=0.0,
                    smallest_eigenvalues=eigs,
                    method="sparse",
                )
            return SpectralInvariants(
                h0_dimension=0,
                spectral_gap=max(0.0, val),
                smallest_eigenvalues=eigs,
                method="sparse",
            )

        k = min(_SPARSE_MAX_EIGENVALUES, max(2, n - 1))

        # Check for expected dimensions properly to prevent ARPACK errors inside
        # when dealing with heterogeneous fibers. Also use shift-invert mode to trap
        # eigenvalues close to zero stably.
        try:
            eigenvalues = eigsh(
                L, k=k, sigma=-1e-5, which="LM",
                return_eigenvectors=False,
                tol=_ARPACK_TOLERANCE,
            )
        except ArpackError as exc:
            raise SpectralComputationError(
                f"ARPACK no convergió para el Laplaciano del haz "
                f"(dim={n}, k={k}): {exc}"
            ) from exc
        except Exception as exc:
            if "dimension mismatch" in str(exc).lower():
                from app.core.immune_system.topological_watcher import DimensionalMismatchError
                raise DimensionalMismatchError(
                    f"Incoherencia dimensional durante el análisis disperso de ARPACK "
                    f"(dim={n}, k={k}): {exc}"
                ) from exc
            raise

        eigenvalues = np.sort(eigenvalues.astype(np.float64))

        cls._verify_semidefinite_positivity(eigenvalues, method="sparse")

        h0_dim, spectral_gap = cls._classify_eigenvalues(eigenvalues)

        eigenvalues_immutable = eigenvalues.copy()
        eigenvalues_immutable.setflags(write=False)

        return SpectralInvariants(
            h0_dimension=h0_dim,
            spectral_gap=spectral_gap,
            smallest_eigenvalues=eigenvalues_immutable,
            method="sparse",
        )

    @classmethod
    def compute(cls, L: sp.csc_matrix) -> SpectralInvariants:
        """
        Calcula invariantes espectrales con estrategia híbrida automática.

        Args:
            L: Laplaciano del haz

        Returns:
            SpectralInvariants
        """
        n = L.shape[0]
        if n <= _DENSE_SPECTRAL_MAX_DIM:
            return cls.compute_dense(L)
        return cls.compute_sparse(L)


# ═══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR
# ═══════════════════════════════════════════════════════════════════════════════

class SheafCohomologyOrchestrator:
    """
    Inspector cohomológico del haz.

    Analiza un estado global x ∈ C⁰ y determina si satisface todas las
    restricciones locales del haz, es decir, si δx ≈ 0.

    Si E(x) = ‖δx‖² es pequeña, x es coherente (sección global compatible).
    Si E(x) excede la tolerancia, x viola las restricciones del haz.

    Protocolo de auditoría:
    ───────────────────────
    1. Verificar completitud del haz
    2. Ensamblar δ: C⁰ → C¹
    3. Validar x ∈ C⁰
    4. Calcular E(x) = ‖δx‖²
    5. Si E(x) > ε: rechazar por inconsistencia (lanzar excepción)
    6. Si E(x) ≤ ε: calcular L = δᵀδ y analizar espectro
    7. Retornar diagnóstico completo
    """

    @staticmethod
    def validate_local_restriction(
        focus_node_id: str,
        local_topo: Optional[Dict],
        local_fin: Optional[Dict]
    ) -> None:
        """
        Invocación axiomática del Mapa de Restricción (F_{V \triangleright U}).

        Verifica que el sub-espacio enfocado (U) posea una métrica estructural
        (Laplaciano local, conectividad algebraica) capaz de sostener
        una deliberación independiente del grafo global (V).

        Si el sub-grafo carece de soporte (e.g. métricas nulas o no calculables
        por degeneración), se levanta SheafDegeneracyError, forzando la
        saturación del retículo de veredicto en el Estrato Ω.

        Args:
            focus_node_id: Identificador del sub-espacio.
            local_topo: Métricas topológicas del sub-espacio.
            local_fin: Métricas financieras del sub-espacio.

        Raises:
            SheafDegeneracyError: Si el sub-espacio es algebraicamente degenerado.
        """
        if not local_topo or not local_fin:
            raise SheafDegeneracyError(
                f"Fibración degenerada: el sub-espacio '{focus_node_id}' carece de "
                "soporte estructural métrico. Imposible proyectar restricción local al haz celular."
            )

    @staticmethod
    def _validate_global_state_vector(
        sheaf: CellularSheaf,
        global_state_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Valida el vector de estado global x ∈ C⁰.

        Verificaciones:
        ───────────────
        1. Convertibilidad a ndarray float64
        2. Unidimensionalidad
        3. Longitud = dim(C⁰) = Σ_v dim(F(v))
        4. Finitud de todas las componentes

        Args:
            sheaf: Haz celular que define C⁰
            global_state_vector: Vector candidato

        Returns:
            Vector validado como ndarray float64

        Raises:
            SheafDegeneracyError: Si alguna verificación falla
        """
        try:
            x = np.asarray(global_state_vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SheafDegeneracyError(
                f"El estado global no es convertible a array float64: {exc}"
            ) from exc

        if x.ndim != 1:
            raise SheafDegeneracyError(
                f"El estado global debe ser un vector 1D; "
                f"forma recibida={x.shape}."
            )

        expected_dim = sheaf.total_node_dim
        if x.shape[0] != expected_dim:
            raise SheafDegeneracyError(
                f"Dimensión incompatible del estado global: "
                f"recibida={x.shape[0]}, esperada={expected_dim} "
                f"(dim C⁰ = Σ_v dim(F(v)))."
            )

        if not np.all(np.isfinite(x)):
            non_finite_count = int(np.count_nonzero(~np.isfinite(x)))
            raise SheafDegeneracyError(
                f"El estado global contiene {non_finite_count} "
                f"componente(s) no finita(s) (NaN o ±∞)."
            )

        return x

    @staticmethod
    def _compute_frustration_energy(
        delta: sp.csc_matrix,
        x: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calcula rigurosamente la energía de Dirichlet y la norma del residuo.

        E(x) = ‖δx‖² = (δx)ᵀ(δx)

        Usar ‖δx‖² directamente (en lugar de xᵀLx) es numéricamente más
        estable porque:
        1. Evita formar L = δᵀδ explícitamente
        2. El producto δx es un producto matriz-vector (sin acumulación
           cuadrática de errores)
        3. La norma ‖·‖² es siempre no negativa en aritmética exacta

        En aritmética de punto flotante, la energía puede ser ligeramente
        negativa por errores de redondeo. Se aplica clamp a 0 si |E| ≤ ε.

        Args:
            delta: Operador de cofrontera δ: C⁰ → C¹
            x: Vector de estado global x ∈ C⁰

        Returns:
            (frustration_energy, residual_norm) donde
            frustration_energy = ‖δx‖² y residual_norm = ‖δx‖

        Raises:
            SheafCohomologyError: Si la energía es significativamente negativa
        """
        residual = delta.dot(x)  # δx ∈ C¹
        residual_norm = float(np.linalg.norm(residual))
        energy = float(np.dot(residual, residual))  # ‖δx‖²

        # Verificar consistencia: energy ≈ residual_norm²
        if abs(energy - residual_norm ** 2) > _FRUSTRATION_TOLERANCE * max(1.0, energy):
            logger.warning(
                "Inconsistencia numérica en energía de frustración: "
                "‖δx‖² = %.6e vs (‖δx‖)² = %.6e",
                energy, residual_norm ** 2,
            )

        if energy < 0.0:
            if abs(energy) <= _FRUSTRATION_TOLERANCE:
                logger.debug(
                    "Energía de frustración ligeramente negativa (%.6e), "
                    "clamped a 0.0.",
                    energy,
                )
                return 0.0, residual_norm
            raise SheafCohomologyError(
                f"Energía de frustración negativa fuera de tolerancia: "
                f"{energy:.6e}. "
                "Esto indica un fallo numérico severo en el cálculo de "
                "‖δx‖² (posible overflow o cancelación catastrófica)."
            )

        return energy, residual_norm

    @classmethod
    def audit_global_state(
        cls,
        sheaf: CellularSheaf,
        global_state_vector: np.ndarray,
    ) -> GlobalFrustrationAssessment:
        """
        Evalúa si un estado global x ∈ C⁰ es transversalmente compatible
        con todas las restricciones del haz.

        Pipeline:
        ─────────
        1. Verificar completitud del haz
        2. Ensamblar δ: C⁰ → C¹
        3. Validar x ∈ C⁰
        4. Calcular E(x) = ‖δx‖²
        5. Si E(x) > ε: lanzar HomologicalInconsistencyError
        6. Calcular L = δᵀδ
        7. Analizar espectro de L
        8. Retornar GlobalFrustrationAssessment

        Args:
            sheaf: Haz celular completamente ensamblado
            global_state_vector: Estado global propuesto x ∈ C⁰

        Returns:
            GlobalFrustrationAssessment con diagnóstico completo

        Raises:
            HomologicalInconsistencyError:
                Si x no es una sección global compatible (E(x) > ε)
            SheafDegeneracyError:
                Si el haz o el vector de estado son inválidos
            SpectralComputationError:
                Si el análisis espectral falla
        """
        # ── Etapa 1: Verificar completitud ──
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
        frustration_energy, residual_norm = cls._compute_frustration_energy(
            delta, x
        )
        is_coherent = frustration_energy <= _FRUSTRATION_TOLERANCE

        # ── Etapa 5: Rechazar si incoherente ──
        if not is_coherent:
            logger.critical(
                "FRUSTRACIÓN DE HAZ: E(x) = ‖δx‖² = %.6e, ‖δx‖ = %.6e. "
                "El estado global propuesto no satisface las restricciones "
                "del haz.",
                frustration_energy,
                residual_norm,
            )
            raise HomologicalInconsistencyError(
                "Fractura del consenso global: el estado propuesto no "
                "constituye una sección global compatible del haz. "
                f"E(x) = ‖δx‖² = {frustration_energy:.6e} "
                f"excede tolerancia = {_FRUSTRATION_TOLERANCE:.6e}."
            )

        # ── Etapa 6: Calcular Laplaciano ──
        L = sheaf.compute_sheaf_laplacian()

        # ── Etapa 7: Análisis espectral ──
        spectral = _SpectralAnalyzer.compute(L)

        logger.info(
            "Auditoría cohomológica exitosa: E(x)=%.6e, ‖δx‖=%.6e, "
            "dim H⁰=%d, brecha espectral=%.6e, método=%s",
            frustration_energy,
            residual_norm,
            spectral.h0_dimension,
            spectral.spectral_gap,
            spectral.method,
        )

        # ── Etapa 8: Retornar diagnóstico ──
        return GlobalFrustrationAssessment(
            frustration_energy=frustration_energy,
            h0_dimension=spectral.h0_dimension,
            is_coherent=True,
            spectral_gap=spectral.spectral_gap,
            residual_norm=residual_norm,
            spectral_method=spectral.method,
        )