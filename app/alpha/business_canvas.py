"""
=========================================================================================
Módulo: Business Canvas Topology (Condición de Frontera Macroscópica — Estrato α)
Ubicación: app/alfa/business_canvas.py
=========================================================================================

Naturaleza Ciber-Física:
    Constituye el límite topológico supremo (Estrato α) del ecosistema APU_filter.
    Modela el Business Model Canvas (BMC) de la organización matriz no como una
    abstracción gráfica, sino como un 1-complejo simplicial finito K, auditando
    matemáticamente su viabilidad estructural antes de permitir la deliberación
    del Estrato Ω y el procesamiento táctico de presupuestos.

1. Complejo de Cadenas y Operador Frontera:
    Sea G = (V, E, w) el digrafo ponderado del ecosistema de negocio. 
    Su proyección al 1-esqueleto no dirigido K induce la secuencia exacta corta 
    de cadenas con coeficientes en ℝ:
        0 ⟶ C₁(K; ℝ) xrightarrow{∂₁} C₀(K; ℝ) ⟶ 0
    El operador frontera discreto ∂₁ se materializa algorítmicamente como la 
    matriz de incidencia de dimensiones |V| × |E|.

2. Invariantes Homológicos (Teorema de Rango-Nulidad):
    La viabilidad intrínseca del modelo se extrae computando el rango numérico 
    estricto del operador frontera ∂₁ mediante Descomposición en Valores Singulares 
    (SVD) para mantener estabilidad frente al ruido flotante:
        • β₀ = |V| - rank(∂₁) (Componentes Conexas: Fragmentación del valor)
        • β₁ = |E| - rank(∂₁) (Dimensión del Espacio de Ciclos: ker(∂₁))
    
    [AXIOMA DE CANIBALIZACIÓN]: Si la homología revela β₁ > 0 (clases homológicas
    no triviales en H₁), el BMC alberga bucles logísticos tóxicos irreconciliables.
    El sistema impone un veto absoluto retornando REJECTED_TOXIC_CYCLES.

3. Invariante Macroscópico de Euler-Poincaré:
    La salud sistémica se verifica evaluando la característica de Euler del lienzo:
        χ(K) = β₀ - β₁ = |V| - |E|
    Un BMC degenerado con χ ≤ 0 colapsa automáticamente la Malla Agéntica.

4. Espectro Combinatorio y Robustez de la Cadena de Valor:
    El Laplaciano Combinatorio de grado 0 se define como L₀ = ∂₁∂₁ᵀ. 
    Su espectro determina la resiliencia estructural de la matriz empresarial.
    Si el valor de Fiedler (conectividad algebraica λ₂) decae por debajo de la
    tolerancia admisible (λ₂ < MIN_FIEDLER_VALUE), la empresa sufre de "Fragilidad 
    Espectral", indicando una alta probabilidad de fractura ante perturbaciones.
=========================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import networkx as nx
import numpy as np
from numpy.typing import NDArray

# Dependencias MIC
from app.core.mic_algebra import CategoricalState, Morphism
from app.adapters.mic_vectors import VectorResultStatus, _build_result, _build_error, VectorMetrics
from app.core.schemas import Stratum

logger = logging.getLogger("MIC.Alpha.BusinessCanvas")


# =============================================================================
# TIPOS
# =============================================================================

FloatArray = NDArray[np.floating]
Edge = Tuple[str, str]
WeightedEdge = Tuple[str, str, float]


# =============================================================================
# CONSTANTES MATEMÁTICAS
# =============================================================================

# Tolerancia para comparaciones en punto flotante (machine epsilon ~ 2.2e-16)
EPSILON: float = 1e-12

# Tolerancia para determinación de rango numérico vía SVD
# Basado en: tol = max(m,n) * eps * σ_max
RANK_TOL: float = 1e-10

# Tolerancia para autovalores considerados como cero
EIGENVALUE_ZERO_TOL: float = 1e-10

# Umbral mínimo de conectividad algebraica de Fiedler
MIN_FIEDLER_VALUE: float = 0.05

# Límite para enumeración de ciclos (evitar explosión combinatoria)
MAX_CYCLE_ENUMERATION: int = 10_000


# =============================================================================
# CONFIGURACIÓN DEL BMC
# =============================================================================

BMC_NODES: Tuple[str, ...] = (
    "P_soc",   # Socios clave
    "P_rec",   # Recursos clave
    "P_act",   # Actividades clave
    "P_val",   # Propuesta de valor
    "P_can",   # Canales
    "P_rel",   # Relaciones con clientes
    "P_seg",   # Segmentos de clientes
    "P_cost",  # Estructura de costes
    "P_ing",   # Fuentes de ingresos
)

BASE_EDGES: Tuple[WeightedEdge, ...] = (
    ("P_soc", "P_act", 1.0),
    ("P_rec", "P_act", 1.0),
    ("P_act", "P_val", 1.0),
    ("P_soc", "P_cost", 1.0),
    ("P_rec", "P_cost", 1.0),
    ("P_act", "P_cost", 1.0),
    ("P_val", "P_can", 1.0),
    ("P_val", "P_rel", 1.0),
    ("P_can", "P_seg", 1.0),
    ("P_rel", "P_seg", 1.0),
    ("P_seg", "P_ing", 1.0),
)


# =============================================================================
# EXCEPCIONES JERÁRQUICAS
# =============================================================================

class BMCTopologyError(Exception):
    """Excepción base para errores topológicos del BMC."""


class TopologicalInvariantError(BMCTopologyError):
    """Violación de invariantes topológicos fundamentales."""


class SpectralAnalysisError(BMCTopologyError):
    """Error en análisis espectral del Laplaciano."""


class PayloadValidationError(BMCTopologyError):
    """Payload inválido para construcción del complejo."""


class HomologicalInconsistencyError(BMCTopologyError):
    """Inconsistencia en cálculos homológicos (violación de identidades algebraicas)."""


# =============================================================================
# ENUMERACIONES
# =============================================================================

class MergeVerdict(str, Enum):
    """Veredictos para auditoría de fusión estratégica."""
    ACCEPTED = "ACCEPTED"
    REJECTED_TOXIC_CYCLES = "REJECTED_TOXIC_CYCLES"
    REJECTED_DISCONNECTED = "REJECTED_DISCONNECTED"
    REJECTED_SPECTRAL_FRAGILITY = "REJECTED_SPECTRAL_FRAGILITY"
    REJECTED_HOMOLOGICAL_DEFECT = "REJECTED_HOMOLOGICAL_DEFECT"


class ConnectivityClass(str, Enum):
    """Clasificación de conectividad del complejo."""
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    STRONGLY_CONNECTED = "STRONGLY_CONNECTED"
    WEAKLY_CONNECTED = "WEAKLY_CONNECTED"


# =============================================================================
# FUNCIONES AUXILIARES PURAS
# =============================================================================

def canonicalize_edge(u: str, v: str) -> Edge:
    """
    Orienta canónicamente una arista usando orden lexicográfico.
    
    Garantiza representación única para aristas no dirigidas:
    {u, v} → (min(u,v), max(u,v))
    """
    return (u, v) if u < v else (v, u)


def compute_numerical_rank(matrix: FloatArray, tol: Optional[float] = None) -> int:
    """
    Calcula rango numérico usando descomposición SVD.
    
    El rango se determina contando valores singulares σᵢ tales que:
        σᵢ > tol × σ_max
    
    Args:
        matrix: Matriz a analizar
        tol: Tolerancia relativa (default: max(m,n) × ε_mach)
    
    Returns:
        Rango numérico de la matriz
    """
    if matrix.size == 0:
        return 0
    
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    
    if len(singular_values) == 0:
        return 0
    
    sigma_max = singular_values[0]
    if sigma_max < EPSILON:
        return 0
    
    if tol is None:
        m, n = matrix.shape
        tol = max(m, n) * np.finfo(float).eps
    
    threshold = tol * sigma_max
    return int(np.sum(singular_values > threshold))


def compute_null_space_basis(
    matrix: FloatArray, 
    tol: Optional[float] = None
) -> FloatArray:
    """
    Calcula base ortonormal del núcleo (kernel) de una matriz.
    
    Usando SVD: A = UΣVᵀ, entonces ker(A) = span{vᵢ : σᵢ ≈ 0}
    
    Args:
        matrix: Matriz cuyo kernel se calcula
        tol: Tolerancia para valores singulares nulos
    
    Returns:
        Matriz cuyas columnas forman base ortonormal de ker(A)
    """
    if matrix.size == 0:
        return np.array([]).reshape(0, 0)
    
    U, s, Vh = np.linalg.svd(matrix, full_matrices=True)
    m, n = matrix.shape
    
    if len(s) == 0:
        return Vh.T
    
    if tol is None:
        tol = max(m, n) * np.finfo(float).eps
    
    sigma_max = s[0] if s[0] > EPSILON else 1.0
    threshold = tol * sigma_max
    
    # Valores singulares considerados no nulos (rango numérico)
    rank = int(np.sum(s > threshold))
    null_dim = n - rank
    
    if null_dim == 0:
        return np.array([]).reshape(n, 0)
    
    # Las últimas 'null_dim' filas de Vᵀ (cuyas dimensiones son n x n por full_matrices=True)
    # corresponden a la base del kernel
    null_space = Vh[-(null_dim):, :].T
    
    return null_space


def safe_eigenvalues_symmetric(matrix: FloatArray) -> FloatArray:
    """
    Calcula autovalores de matriz simétrica con estabilidad numérica.
    
    1. Fuerza simetría exacta: A ← (A + Aᵀ)/2
    2. Usa rutina LAPACK optimizada para matrices simétricas
    3. Limpia errores numéricos cerca de cero
    
    Args:
        matrix: Matriz simétrica real
    
    Returns:
        Autovalores ordenados ascendentemente
    """
    # Asegurar simetría numérica exacta
    symmetric = (matrix + matrix.T) / 2.0
    
    # eigvalsh usa rutina optimizada para matrices Hermitianas/simétricas
    eigenvalues = np.linalg.eigvalsh(symmetric)
    
    # Limpiar errores numéricos: valores muy pequeños → 0
    eigenvalues = np.where(
        np.abs(eigenvalues) < EIGENVALUE_ZERO_TOL,
        0.0,
        eigenvalues
    )
    
    return np.sort(eigenvalues)


# =============================================================================
# DATACLASSES INMUTABLES
# =============================================================================

@dataclass(frozen=True)
class ChainComplex1D:
    """
    Complejo de cadenas del 1-esqueleto K.
    
    Representa la secuencia exacta:
        0 → C₁(K) --∂₁--> C₀(K) → 0
    
    Attributes:
        vertex_basis: Base ordenada de C₀ (|V| elementos)
        edge_basis: Base ordenada de C₁ con orientación canónica (|E| elementos)
        boundary_1: Matriz ∂₁ ∈ ℝ^{|V|×|E|}
        laplacian_0: L₀ = ∂₁∂₁ᵀ ∈ ℝ^{|V|×|V|} (Laplaciano en 0-cadenas)
        laplacian_1: L₁ = ∂₁ᵀ∂₁ ∈ ℝ^{|E|×|E|} (Laplaciano en 1-cadenas)
    
    Invariante algebraico:
        ∂₀ ∘ ∂₁ = 0 (trivialmente satisfecho pues ∂₀ ≡ 0)
    """
    vertex_basis: Tuple[str, ...]
    edge_basis: Tuple[Edge, ...]
    boundary_1: FloatArray
    laplacian_0: FloatArray
    laplacian_1: FloatArray
    
    def __post_init__(self) -> None:
        """Validación de consistencia dimensional."""
        n_v = len(self.vertex_basis)
        n_e = len(self.edge_basis)
        
        if self.boundary_1.shape != (n_v, n_e):
            raise ValueError(
                f"∂₁ debe tener forma ({n_v}, {n_e}), tiene {self.boundary_1.shape}"
            )
        if self.laplacian_0.shape != (n_v, n_v):
            raise ValueError(
                f"L₀ debe tener forma ({n_v}, {n_v}), tiene {self.laplacian_0.shape}"
            )
        if self.laplacian_1.shape != (n_e, n_e):
            raise ValueError(
                f"L₁ debe tener forma ({n_e}, {n_e}), tiene {self.laplacian_1.shape}"
            )
    
    @property
    def dimension_0(self) -> int:
        """Dimensión de C₀."""
        return len(self.vertex_basis)
    
    @property
    def dimension_1(self) -> int:
        """Dimensión de C₁."""
        return len(self.edge_basis)


@dataclass(frozen=True)
class HomologyMetrics:
    """
    Invariantes homológicos del 2-complejo simplicial K.
    
    Teorema de Euler-Poincaré:
        χ(K) = Σₖ (-1)ᵏ βₖ = β₀ - β₁ + β₂
    ... (omitted) ...
    """
    n_vertices: int
    n_edges: int
    rank_boundary_1: int
    nullity_boundary_1: int
    beta_0: int
    beta_1: int
    euler_char: int
    euler_from_betti: int
    beta_2: int = 0
    
    def __post_init__(self) -> None:
        """Validación de consistencia algebraica."""
        # Verificar Euler-Poincaré
        if self.euler_char != self.euler_from_betti:
            raise HomologicalInconsistencyError(
                f"Violación Euler-Poincaré: χ={self.euler_char} ≠ β₀-β₁={self.euler_from_betti}"
            )
        
        # Verificar rank-nullity theorem
        expected_nullity = self.n_edges - self.rank_boundary_1
        if self.nullity_boundary_1 != expected_nullity:
            raise HomologicalInconsistencyError(
                f"Violación rank-nullity: nullity={self.nullity_boundary_1} ≠ "
                f"|E|-rank={expected_nullity}"
            )


@dataclass(frozen=True)
class SpectralMetrics:
    """
    Invariantes espectrales del Laplaciano combinatorio L₀.
    
    Propiedades fundamentales:
    - L₀ es semidefinido positivo (spec(L₀) ⊆ [0, ∞))
    - λ₀ = 0 siempre (vector propio: 1 = (1,...,1)ᵀ)
    - mult(0) = β₀ (teorema de Hodge discreto)
    - λ₁ > 0 ⟺ grafo conexo
    - tr(L₀) = 2|E| = Σᵢ deg(vᵢ)
    
    Conectividad algebraica de Fiedler:
    - λ₁ mide la "robustez" de la conectividad
    - Cota de Cheeger: h(G)/2 ≤ λ₁ ≤ 2h(G)
    """
    eigenvalues: Tuple[float, ...]
    fiedler_value: float
    spectral_gap: float
    multiplicity_zero: int
    spectral_radius: float
    trace_laplacian: float
    
    def __post_init__(self) -> None:
        """Validación de propiedades espectrales."""
        if self.eigenvalues and self.eigenvalues[0] < -EIGENVALUE_ZERO_TOL:
            raise SpectralAnalysisError(
                f"L₀ no semidefinido positivo: λ_min={self.eigenvalues[0]}"
            )


@dataclass(frozen=True)
class CycleSpaceMetrics:
    """
    Métricas del espacio de ciclos ker(∂₁) ⊆ C₁(K).
    
    El espacio de ciclos tiene dimensión β₁ y representa
    los "agujeros 1-dimensionales" del complejo.
    """
    dimension: int
    cycle_basis_edges: Tuple[Tuple[Edge, ...], ...]
    directed_cycles_count: int
    is_dag: bool


@dataclass(frozen=True)
class BmcTopologyMetrics:
    """Resultado consolidado del análisis topológico del BMC."""
    # Homología
    beta_0: int
    beta_1: int
    euler_char: int
    rank_boundary_1: int
    nullity_boundary_1: int
    
    # Espectro
    fiedler_value: float
    spectral_gap: float
    spectral_radius: float
    multiplicity_zero: int
    trace_laplacian: float
    
    # Ciclos
    directed_cycle_count: int
    fundamental_cycle_count: int
    
    # Clasificaciones booleanas
    is_connected: bool
    has_cycle_space: bool
    has_directed_feedback: bool
    is_dag: bool
    is_spectrally_stable: bool
    connectivity_class: ConnectivityClass
    
    # Metadatos
    n_vertices: int
    n_edges: int


# =============================================================================
# IMPLEMENTACIÓN PRINCIPAL
# =============================================================================

class AlphaTopologyVector(Morphism):
    """
    Morfismo topológico del Estrato α.
    
    Define un funtor F: DiGraph_{BMC} → Vec_{ℝ} que asigna a cada
    configuración del BMC su estructura algebraico-topológica.
    
    Pipeline de análisis:
    ----------------------
    1. Construcción del digrafo causal G
    2. Proyección al 1-complejo no dirigido K
    3. Construcción del complejo de cadenas C_*(K)
    4. Cálculo de homología H_*(K) vía álgebra lineal
    5. Análisis espectral del Laplaciano combinatorio
    6. Detección de patologías dirigidas
    7. Validación de consistencia algebraica
    """

    def __init__(self, name: str = "alpha_business_canvas_topology") -> None:
        super().__init__(name)
        self.target_stratum = Stratum.ALPHA

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return frozenset(self.target_stratum.requires())

    @property
    def codomain(self) -> Stratum:
        return self.target_stratum

    # -------------------------------------------------------------------------
    # INTERFAZ PÚBLICA
    # -------------------------------------------------------------------------

    def __call__(
        self, 
        state_vector: CategoricalState, 
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Ejecuta análisis topológico completo del BMC.
        
        Args:
            state_vector: Estado categórico con payload de configuración
            
        Returns:
            Resultado estructurado con métricas, narrativa y status
        """
        try:
            payload = self._extract_payload(state_vector)
            metrics = self._compute_full_analysis(payload)
            
            self._enforce_topological_constraints(metrics)
            
            narrative = self._generate_narrative(metrics)
            
            return _build_result(
                success=True,
                stratum=self.target_stratum,
                status=VectorResultStatus.SUCCESS,
                metrics=VectorMetrics(),
                metrics_payload=self._metrics_to_dict(metrics),
                narrative=narrative,
            )

        except BMCTopologyError as e:
            logger.error("Error topológico en BMC: %s", str(e))
            return _build_error(
                stratum=self.target_stratum,
                status=VectorResultStatus.TOPOLOGY_ERROR,
                error=str(e),
                narrative="VETO ALFA: defecto topológico/algebraico detectado.",
            )
        except Exception as e:
            logger.exception("Error inesperado en análisis topológico.")
            return _build_error(
                stratum=self.target_stratum,
                status=VectorResultStatus.LOGIC_ERROR,
                error=f"Error interno: {str(e)}",
            )

    # -------------------------------------------------------------------------
    # EXTRACCIÓN Y VALIDACIÓN
    # -------------------------------------------------------------------------

    def _extract_payload(self, state_vector: CategoricalState) -> Mapping[str, Any]:
        """Extrae payload del estado categórico con validación de tipo."""
        payload = getattr(state_vector, "payload", None)
        if payload is None:
            return {}
        if not isinstance(payload, Mapping):
            raise PayloadValidationError(
                f"Payload debe ser Mapping, recibido: {type(payload).__name__}"
            )
        return payload

    def _validate_payload_schema(self, payload: Mapping[str, Any]) -> None:
        """Valida estructura del payload."""
        allowed_keys = frozenset({
            "disable_nodes", "remove_edges", "edge_weights", "extra_edges"
        })
        
        unknown = set(payload.keys()) - allowed_keys
        if unknown:
            logger.warning("Claves no reconocidas en payload: %s", sorted(unknown))
        
        for key in allowed_keys:
            if key in payload and not isinstance(payload[key], (list, tuple)):
                raise PayloadValidationError(
                    f"'{key}' debe ser lista, recibido: {type(payload[key]).__name__}"
                )

    def _validate_positive_weight(self, weight: float, context: str) -> None:
        """Valida que un peso sea estrictamente positivo."""
        if not isinstance(weight, (int, float)):
            raise PayloadValidationError(f"Peso no numérico en {context}")
        if weight <= 0:
            raise PayloadValidationError(
                f"Peso debe ser > 0 en {context}, recibido: {weight}"
            )

    # -------------------------------------------------------------------------
    # CONSTRUCCIÓN DEL GRAFO DIRIGIDO
    # -------------------------------------------------------------------------

    def _build_directed_business_graph(
        self, 
        payload: Mapping[str, Any]
    ) -> nx.DiGraph:
        """
        Construye el digrafo causal del BMC desde configuración base + payload.
        """
        self._validate_payload_schema(payload)
        
        G = nx.DiGraph()
        G.add_nodes_from(BMC_NODES)
        
        for source, target, weight in BASE_EDGES:
            G.add_edge(source, target, weight=weight)
        
        # Aplicar modificaciones
        self._apply_node_disabling(G, payload)
        self._apply_edge_removal(G, payload)
        self._apply_weight_modifications(G, payload)
        self._apply_extra_edges(G, payload)
        
        return G

    def _apply_node_disabling(
        self, G: nx.DiGraph, payload: Mapping[str, Any]
    ) -> None:
        """Desactiva nodos especificados (mutación in-place)."""
        disable_nodes = payload.get("disable_nodes", [])
        valid_nodes = frozenset(BMC_NODES)
        
        for node in disable_nodes:
            if node not in valid_nodes:
                raise PayloadValidationError(f"Nodo inválido: '{node}'")
        
        G.remove_nodes_from(disable_nodes)

    def _apply_edge_removal(
        self, G: nx.DiGraph, payload: Mapping[str, Any]
    ) -> None:
        """Elimina aristas especificadas."""
        for edge in payload.get("remove_edges", []):
            src, tgt = edge.get("source"), edge.get("target")
            if src is None or tgt is None:
                raise PayloadValidationError(f"Arista incompleta: {edge}")
            if G.has_edge(src, tgt):
                G.remove_edge(src, tgt)

    def _apply_weight_modifications(
        self, G: nx.DiGraph, payload: Mapping[str, Any]
    ) -> None:
        """Modifica pesos de aristas existentes."""
        current_nodes = frozenset(G.nodes())
        
        for edge in payload.get("edge_weights", []):
            src = edge.get("source")
            tgt = edge.get("target")
            weight = edge.get("weight")
            
            if None in (src, tgt, weight):
                raise PayloadValidationError(f"Especificación incompleta: {edge}")
            
            if src not in current_nodes or tgt not in current_nodes:
                raise PayloadValidationError(f"Nodos ausentes: ({src}, {tgt})")
            
            self._validate_positive_weight(weight, f"edge_weights ({src}->{tgt})")
            
            if not G.has_edge(src, tgt):
                raise PayloadValidationError(f"Arista inexistente: ({src}, {tgt})")
            
            G[src][tgt]["weight"] = float(weight)

    def _apply_extra_edges(
        self, G: nx.DiGraph, payload: Mapping[str, Any]
    ) -> None:
        """Añade aristas adicionales."""
        current_nodes = frozenset(G.nodes())
        
        for edge in payload.get("extra_edges", []):
            src = edge.get("source")
            tgt = edge.get("target")
            weight = edge.get("weight", 1.0)
            
            if src is None or tgt is None:
                raise PayloadValidationError(f"Arista incompleta: {edge}")
            
            if src not in current_nodes or tgt not in current_nodes:
                raise PayloadValidationError(f"Nodos ausentes: ({src}, {tgt})")
            
            self._validate_positive_weight(weight, f"extra_edges ({src}->{tgt})")
            G.add_edge(src, tgt, weight=float(weight))

    # -------------------------------------------------------------------------
    # PROYECCIÓN AL 1-COMPLEJO
    # -------------------------------------------------------------------------

    def _to_weighted_undirected(self, digraph: nx.DiGraph) -> nx.Graph:
        """
        Proyecta digrafo a su 1-esqueleto no dirigido.
        
        Los pesos de aristas bidireccionales se suman:
        w({u,v}) = w(u→v) + w(v→u)
        """
        H = nx.Graph()
        H.add_nodes_from(digraph.nodes(data=True))
        
        for u, v, data in digraph.edges(data=True):
            weight = float(data.get("weight", 1.0))
            if H.has_edge(u, v):
                H[u][v]["weight"] += weight
            else:
                H.add_edge(u, v, weight=weight)
        
        return H

    # -------------------------------------------------------------------------
    # COMPLEJO DE CADENAS
    # -------------------------------------------------------------------------

    def _build_chain_complex_1d(self, graph: nx.Graph) -> ChainComplex1D:
        """
        Construye el complejo de cadenas 1-dimensional.
        
        Convención de signos para arista canónica e = {u,v} con u < v:
            ∂₁(e) = [v] - [u]
        en notación de generadores de C₀.
        
        Matricialmente:
            [∂₁]_{índice(u), j} = -1
            [∂₁]_{índice(v), j} = +1
        """
        # Bases ordenadas lexicográficamente (determinismo)
        vertex_basis = tuple(sorted(graph.nodes()))
        vertex_index = {v: i for i, v in enumerate(vertex_basis)}
        
        # Aristas con orientación canónica
        canonical_edges = sorted({
            canonicalize_edge(u, v) for u, v in graph.edges()
        })
        edge_basis = tuple(canonical_edges)
        
        n_v = len(vertex_basis)
        n_e = len(edge_basis)
        
        # Construir matriz de frontera ∂₁
        boundary_1 = np.zeros((n_v, n_e), dtype=np.float64)
        
        for j, (u, v) in enumerate(edge_basis):
            boundary_1[vertex_index[u], j] = -1.0
            boundary_1[vertex_index[v], j] = +1.0
        
        # Laplacianos combinatorios
        laplacian_0 = boundary_1 @ boundary_1.T  # L₀ = ∂₁∂₁ᵀ
        laplacian_1 = boundary_1.T @ boundary_1  # L₁ = ∂₁ᵀ∂₁
        
        return ChainComplex1D(
            vertex_basis=vertex_basis,
            edge_basis=edge_basis,
            boundary_1=boundary_1,
            laplacian_0=laplacian_0,
            laplacian_1=laplacian_1,
        )

    # -------------------------------------------------------------------------
    # HOMOLOGÍA
    # -------------------------------------------------------------------------

    def _compute_homology_metrics(
        self, 
        chain_complex: ChainComplex1D
    ) -> HomologyMetrics:
        """
        Calcula invariantes homológicos del 1-complejo.
        
        Para 0 → C₁ --∂₁--> C₀ → 0:
        - H₀ = coker(∂₁) = C₀/im(∂₁)  →  β₀ = |V| - rank(∂₁)
        - H₁ = ker(∂₁)                 →  β₁ = nullity(∂₁)
        """
        d1 = chain_complex.boundary_1
        n_v, n_e = d1.shape
        
        rank_d1 = compute_numerical_rank(d1, tol=RANK_TOL)
        nullity_d1 = n_e - rank_d1
        
        beta_0 = n_v - rank_d1
        beta_1 = nullity_d1
        
        euler_char = n_v - n_e
        beta_2 = 0 # Default for 1-skeleton
        euler_from_betti = beta_0 - beta_1 + beta_2
        
        return HomologyMetrics(
            n_vertices=n_v,
            n_edges=n_e,
            rank_boundary_1=rank_d1,
            nullity_boundary_1=nullity_d1,
            beta_0=beta_0,
            beta_1=beta_1,
            euler_char=euler_char,
            euler_from_betti=euler_from_betti,
        )

    def _compute_cycle_space_basis(
        self, 
        chain_complex: ChainComplex1D
    ) -> Tuple[Tuple[Edge, ...], ...]:
        """
        Calcula base del espacio de 1-ciclos ker(∂₁).
        
        Cada vector de la base representa un ciclo fundamental.
        """
        null_space = compute_null_space_basis(chain_complex.boundary_1)
        
        if null_space.size == 0:
            return ()
        
        cycle_basis: List[Tuple[Edge, ...]] = []
        
        for col_idx in range(null_space.shape[1]):
            cycle_vector = null_space[:, col_idx]
            cycle_edges = tuple(
                chain_complex.edge_basis[j]
                for j, coef in enumerate(cycle_vector)
                if abs(coef) > EPSILON
            )
            if cycle_edges:
                cycle_basis.append(cycle_edges)
        
        return tuple(cycle_basis)

    # -------------------------------------------------------------------------
    # ANÁLISIS ESPECTRAL
    # -------------------------------------------------------------------------

    def _compute_spectral_metrics(
        self, 
        chain_complex: ChainComplex1D
    ) -> SpectralMetrics:
        """
        Calcula invariantes espectrales del Laplaciano L₀.
        
        Propiedades utilizadas:
        - L₀ simétrico semidefinido positivo
        - λ₀ = 0, mult(0) = β₀
        - λ₁ = conectividad algebraica de Fiedler
        - tr(L₀) = 2|E|
        """
        L0 = chain_complex.laplacian_0
        n = L0.shape[0]
        
        # Caso degenerado: grafo vacío
        if n == 0:
            return SpectralMetrics(
                eigenvalues=(),
                fiedler_value=0.0,
                spectral_gap=0.0,
                multiplicity_zero=0,
                spectral_radius=0.0,
                trace_laplacian=0.0,
            )
        
        eigenvalues = safe_eigenvalues_symmetric(L0)
        eigenvalues_tuple = tuple(float(x) for x in eigenvalues)
        
        trace_L0 = float(np.trace(L0))
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        
        # Contar autovalores nulos (= β₀)
        mult_zero = int(np.sum(np.abs(eigenvalues) < EIGENVALUE_ZERO_TOL))
        
        # Valor de Fiedler: segundo autovalor más pequeño
        if n <= 1:
            # Grafo con un solo vértice o vacío: trivialmente conexo, sin aristas
            fiedler = 0.0
        else:
            fiedler = float(eigenvalues[1])
        
        spectral_gap = fiedler
        
        return SpectralMetrics(
            eigenvalues=eigenvalues_tuple,
            fiedler_value=fiedler,
            spectral_gap=spectral_gap,
            multiplicity_zero=mult_zero,
            spectral_radius=spectral_radius,
            trace_laplacian=trace_L0,
        )

    # -------------------------------------------------------------------------
    # ANÁLISIS DE CICLOS DIRIGIDOS
    # -------------------------------------------------------------------------

    def _analyze_directed_cycles(self, digraph: nx.DiGraph) -> CycleSpaceMetrics:
        """
        Analiza ciclos dirigidos (patologías causales).
        
        Nota: Los ciclos dirigidos son independientes de β₁ del 1-complejo.
        Representan dependencias circulares en la estructura causal.
        """
        # Verificación rápida de DAG (O(V+E))
        is_dag = nx.is_directed_acyclic_graph(digraph)
        
        if is_dag:
            return CycleSpaceMetrics(
                dimension=0,
                cycle_basis_edges=(),
                directed_cycles_count=0,
                is_dag=True,
            )
        
        # Enumerar ciclos con límite de seguridad
        cycle_count = 0
        sample_cycles: List[Tuple[Edge, ...]] = []
        
        try:
            for cycle in nx.simple_cycles(digraph):
                cycle_count += 1
                
                if len(sample_cycles) < 10:
                    edges = tuple(
                        (cycle[i], cycle[(i + 1) % len(cycle)])
                        for i in range(len(cycle))
                    )
                    sample_cycles.append(edges)
                
                if cycle_count >= MAX_CYCLE_ENUMERATION:
                    logger.warning(
                        "Límite de enumeración alcanzado: %d ciclos", 
                        MAX_CYCLE_ENUMERATION
                    )
                    break
        except Exception as e:
            raise TopologicalInvariantError(
                f"Error enumerando ciclos dirigidos: {e}"
            )
        
        return CycleSpaceMetrics(
            dimension=cycle_count,
            cycle_basis_edges=tuple(sample_cycles),
            directed_cycles_count=cycle_count,
            is_dag=False,
        )

    # -------------------------------------------------------------------------
    # VALIDACIÓN DE CONSISTENCIA
    # -------------------------------------------------------------------------

    def _validate_internal_consistency(
        self,
        homology: HomologyMetrics,
        spectral: SpectralMetrics,
        graph: nx.Graph,
    ) -> None:
        """
        Valida consistencias algebraicas internas.
        
        Verificaciones:
        1. β₀ = componentes conexas (NetworkX)
        2. β₁ = |E| - |V| + β₀ (número ciclomático)
        3. mult(0, L₀) = β₀ (Hodge discreto)
        4. tr(L₀) ≈ 2|E|
        """
        # β₀ vs NetworkX
        if graph.number_of_nodes() > 0:
            nx_components = nx.number_connected_components(graph)
            if nx_components != homology.beta_0:
                raise HomologicalInconsistencyError(
                    f"β₀ inconsistente: calculado={homology.beta_0}, NetworkX={nx_components}"
                )
        
        # Número ciclomático
        cyclomatic = (
            graph.number_of_edges() 
            - graph.number_of_nodes() 
            + homology.beta_0
        )
        if cyclomatic != homology.beta_1:
            raise HomologicalInconsistencyError(
                f"β₁ inconsistente: calculado={homology.beta_1}, ciclomático={cyclomatic}"
            )
        
        # Hodge discreto: mult(0) = β₀
        if spectral.multiplicity_zero != homology.beta_0:
            raise HomologicalInconsistencyError(
                f"Violación Hodge: mult(0)={spectral.multiplicity_zero}, β₀={homology.beta_0}"
            )
        
        # Traza del Laplaciano
        expected_trace = 2.0 * graph.number_of_edges()
        relative_error = abs(spectral.trace_laplacian - expected_trace)
        if expected_trace > 0:
            relative_error /= expected_trace
        
        if relative_error > 1e-6:
            logger.warning(
                "Traza L₀ inconsistente: esperado=%.2f, obtenido=%.2f",
                expected_trace, spectral.trace_laplacian
            )

    def _enforce_topological_constraints(self, metrics: BmcTopologyMetrics) -> None:
        """
        Verifica restricciones topológicas críticas del BMC.
        
        Condiciones necesarias para BMC válido:
        1. β₀ = 1 (conexo)
        2. β₁ = 0 (sin ciclos estructurales)
        3. DAG (sin ciclos dirigidos)
        4. λ₁ > MIN_FIEDLER_VALUE (conectividad algebraica estricta)
        """
        if metrics.beta_0 > 1:
            raise TopologicalInvariantError(
                f"BMC fragmentado: β₀={metrics.beta_0} componentes"
            )
        
        if not metrics.is_spectrally_stable:
            raise TopologicalInvariantError(
                f"Fractura organizacional: Fiedler value (λ₂={metrics.fiedler_value:.4f}) "
                f"es menor al umbral {MIN_FIEDLER_VALUE}"
            )

        if metrics.beta_1 > 0:
            raise TopologicalInvariantError(
                f"Espacio de ciclos no trivial: β₁={metrics.beta_1}"
            )
        
        if metrics.directed_cycle_count > 0:
            raise TopologicalInvariantError(
                f"Ciclos causales detectados: {metrics.directed_cycle_count}"
            )

    # -------------------------------------------------------------------------
    # ANÁLISIS COMPLETO
    # -------------------------------------------------------------------------

    def _compute_full_analysis(
        self, 
        payload: Mapping[str, Any]
    ) -> BmcTopologyMetrics:
        """Pipeline completo de análisis topológico."""
        # Construcción
        digraph = self._build_directed_business_graph(payload)
        undirected = self._to_weighted_undirected(digraph)
        chain_complex = self._build_chain_complex_1d(undirected)
        
        # Invariantes
        homology = self._compute_homology_metrics(chain_complex)
        spectral = self._compute_spectral_metrics(chain_complex)
        cycles = self._analyze_directed_cycles(digraph)
        
        # Validación cruzada
        self._validate_internal_consistency(homology, spectral, undirected)
        
        # Clasificación de conectividad
        if homology.beta_0 == 1:
            if digraph.number_of_nodes() > 0 and nx.is_strongly_connected(digraph):
                conn_class = ConnectivityClass.STRONGLY_CONNECTED
            elif digraph.number_of_nodes() > 0 and nx.is_weakly_connected(digraph):
                conn_class = ConnectivityClass.WEAKLY_CONNECTED
            else:
                conn_class = ConnectivityClass.CONNECTED
        else:
            conn_class = ConnectivityClass.DISCONNECTED
        
        return BmcTopologyMetrics(
            beta_0=homology.beta_0,
            beta_1=homology.beta_1,
            euler_char=homology.euler_char,
            rank_boundary_1=homology.rank_boundary_1,
            nullity_boundary_1=homology.nullity_boundary_1,
            fiedler_value=spectral.fiedler_value,
            spectral_gap=spectral.spectral_gap,
            spectral_radius=spectral.spectral_radius,
            multiplicity_zero=spectral.multiplicity_zero,
            trace_laplacian=spectral.trace_laplacian,
            directed_cycle_count=cycles.directed_cycles_count,
            fundamental_cycle_count=len(cycles.cycle_basis_edges),
            is_connected=(homology.beta_0 == 1),
            has_cycle_space=(homology.beta_1 > 0),
            has_directed_feedback=(cycles.directed_cycles_count > 0),
            is_dag=cycles.is_dag,
            is_spectrally_stable=(spectral.fiedler_value >= MIN_FIEDLER_VALUE),
            connectivity_class=conn_class,
            n_vertices=homology.n_vertices,
            n_edges=homology.n_edges,
        )

    # -------------------------------------------------------------------------
    # GENERACIÓN DE RESULTADOS
    # -------------------------------------------------------------------------

    def _generate_narrative(self, metrics: BmcTopologyMetrics) -> str:
        """Genera narrativa descriptiva del análisis."""
        parts: List[str] = []
        
        if metrics.is_connected:
            parts.append("BMC algebraicamente conexo")
        else:
            parts.append(f"BMC fragmentado ({metrics.beta_0} componentes)")
        
        if metrics.beta_1 == 0:
            parts.append("homológicamente acíclico")
        else:
            parts.append(f"con {metrics.beta_1} ciclos fundamentales")
        
        if metrics.is_dag:
            parts.append("causalmente consistente (DAG)")
        else:
            parts.append(f"con {metrics.directed_cycle_count} ciclos causales")
        
        if metrics.is_spectrally_stable:
            parts.append("espectralmente estable")
        else:
            parts.append(f"espectralmente frágil (λ₁={metrics.fiedler_value:.4f})")
        
        return "; ".join(parts) + "."

    def _metrics_to_dict(self, metrics: BmcTopologyMetrics) -> Dict[str, Any]:
        """Serializa métricas a diccionario."""
        return {
            "beta_0": metrics.beta_0,
            "beta_1": metrics.beta_1,
            "euler_char": metrics.euler_char,
            "rank_boundary_1": metrics.rank_boundary_1,
            "nullity_boundary_1": metrics.nullity_boundary_1,
            "fiedler_value": metrics.fiedler_value,
            "spectral_gap": metrics.spectral_gap,
            "spectral_radius": metrics.spectral_radius,
            "multiplicity_zero": metrics.multiplicity_zero,
            "trace_laplacian": metrics.trace_laplacian,
            "directed_cycle_count": metrics.directed_cycle_count,
            "fundamental_cycle_count": metrics.fundamental_cycle_count,
            "is_connected": metrics.is_connected,
            "has_cycle_space": metrics.has_cycle_space,
            "has_directed_feedback": metrics.has_directed_feedback,
            "is_dag": metrics.is_dag,
            "is_spectrally_stable": metrics.is_spectrally_stable,
            "connectivity_class": metrics.connectivity_class.value,
            "n_vertices": metrics.n_vertices,
            "n_edges": metrics.n_edges,
        }

    # -------------------------------------------------------------------------
    # AUDITORÍA DE FUSIÓN (MAYER-VIETORIS)
    # -------------------------------------------------------------------------

    def audit_strategic_fusion(
        self,
        current_bmc: nx.DiGraph,
        new_domain: nx.DiGraph,
    ) -> MergeVerdict:
        """
        Auditoría algebraico-topológica inspirada en Mayer-Vietoris.
        
        Para 1-complejos A, B con A∩B, la secuencia de Mayer-Vietoris:
        
        ... → H₁(A∩B) → H₁(A)⊕H₁(B) → H₁(A∪B) → H₀(A∩B) → ...
        
        Permite detectar si la fusión introduce defectos topológicos.
        
        Criterios de rechazo:
        1. A∪B desconexo (β₀ > 1)
        2. Defecto homológico (β₁ crece anómalamente)
        3. Nuevos ciclos dirigidos
        4. Fragilidad espectral (λ₁ < umbral)
        """
        # Construir unión e intersección
        union_g = nx.compose(current_bmc, new_domain)
        inter_g = self._compute_edge_intersection(current_bmc, new_domain)
        
        # Proyectar a 1-complejos
        A_u = self._to_weighted_undirected(current_bmc)
        B_u = self._to_weighted_undirected(new_domain)
        U_u = self._to_weighted_undirected(union_g)
        I_u = self._to_weighted_undirected(inter_g)
        
        # Complejos de cadenas
        cc_A = self._build_chain_complex_1d(A_u)
        cc_B = self._build_chain_complex_1d(B_u)
        cc_U = self._build_chain_complex_1d(U_u)
        cc_I = self._build_chain_complex_1d(I_u)
        
        # Homología
        h_A = self._compute_homology_metrics(cc_A)
        h_B = self._compute_homology_metrics(cc_B)
        h_U = self._compute_homology_metrics(cc_U)
        h_I = self._compute_homology_metrics(cc_I)
        
        # Espectro de la unión
        s_U = self._compute_spectral_metrics(cc_U)
        
        # Ciclos dirigidos
        dc_A = self._analyze_directed_cycles(current_bmc).directed_cycles_count
        dc_B = self._analyze_directed_cycles(new_domain).directed_cycles_count
        dc_U = self._analyze_directed_cycles(union_g).directed_cycles_count
        
        # Heurística MV para H₁:
        # En secuencia exacta: β₁(U) ≤ β₁(A) + β₁(B) - β₁(A∩B) + corrección
        mv_defect = h_U.beta_1 - (h_A.beta_1 + h_B.beta_1 - h_I.beta_1)
        
        # Ciclos dirigidos nuevos
        new_directed = dc_U - dc_A - dc_B
        
        # Evaluación de criterios
        if h_U.beta_0 > 1:
            logger.error("Fusión rechazada: unión desconectada (β₀=%d)", h_U.beta_0)
            return MergeVerdict.REJECTED_DISCONNECTED
        
        if new_directed > 0:
            logger.error("Fusión rechazada: %d nuevos ciclos dirigidos", new_directed)
            return MergeVerdict.REJECTED_TOXIC_CYCLES

        if h_U.beta_1 > max(h_A.beta_1, h_B.beta_1) or mv_defect > 0:
            logger.error(
                "Fusión rechazada: defecto homológico β₁(U)=%d, MV_defect=%d",
                h_U.beta_1, mv_defect
            )
            return MergeVerdict.REJECTED_HOMOLOGICAL_DEFECT
        
        if s_U.fiedler_value < MIN_FIEDLER_VALUE:
            logger.error(
                "Fusión rechazada: fragilidad espectral (λ₁=%.4f < %.4f)",
                s_U.fiedler_value, MIN_FIEDLER_VALUE
            )
            return MergeVerdict.REJECTED_SPECTRAL_FRAGILITY
        
        return MergeVerdict.ACCEPTED

    def _compute_edge_intersection(
        self, 
        g1: nx.DiGraph, 
        g2: nx.DiGraph
    ) -> nx.DiGraph:
        """Calcula intersección de digrafos (nodos y aristas comunes)."""
        common_nodes = set(g1.nodes()) & set(g2.nodes())
        common_edges = set(g1.edges()) & set(g2.edges())
        
        I = nx.DiGraph()
        I.add_nodes_from(common_nodes)
        
        for u, v in common_edges:
            w1 = float(g1[u][v].get("weight", 1.0))
            w2 = float(g2[u][v].get("weight", 1.0))
            I.add_edge(u, v, weight=min(w1, w2))
        
        return I

    # -------------------------------------------------------------------------
    # API AVANZADA
    # -------------------------------------------------------------------------

    def build_analysis_bundle(
        self, 
        payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Genera paquete analítico completo para inspección matemática.
        
        Incluye representaciones matriciales, espectro completo y bases de ciclos.
        """
        digraph = self._build_directed_business_graph(payload)
        undirected = self._to_weighted_undirected(digraph)
        cc = self._build_chain_complex_1d(undirected)
        homology = self._compute_homology_metrics(cc)
        spectral = self._compute_spectral_metrics(cc)
        cycles = self._analyze_directed_cycles(digraph)
        cycle_basis = self._compute_cycle_space_basis(cc)
        
        return {
            "bases": {
                "vertices": list(cc.vertex_basis),
                "edges": [list(e) for e in cc.edge_basis],
            },
            "matrices": {
                "boundary_1": cc.boundary_1.tolist(),
                "laplacian_0": cc.laplacian_0.tolist(),
                "laplacian_1": cc.laplacian_1.tolist(),
            },
            "homology": {
                "n_vertices": homology.n_vertices,
                "n_edges": homology.n_edges,
                "rank_d1": homology.rank_boundary_1,
                "nullity_d1": homology.nullity_boundary_1,
                "beta_0": homology.beta_0,
                "beta_1": homology.beta_1,
                "euler_char": homology.euler_char,
            },
            "spectral": {
                "eigenvalues": list(spectral.eigenvalues),
                "fiedler": spectral.fiedler_value,
                "spectral_gap": spectral.spectral_gap,
                "spectral_radius": spectral.spectral_radius,
                "mult_zero": spectral.multiplicity_zero,
                "trace": spectral.trace_laplacian,
            },
            "cycles": {
                "directed_count": cycles.directed_cycles_count,
                "is_dag": cycles.is_dag,
                "fundamental_basis": [
                    [list(e) for e in c] for c in cycle_basis
                ],
            },
            "graph_stats": {
                "directed_nodes": digraph.number_of_nodes(),
                "directed_edges": digraph.number_of_edges(),
                "undirected_nodes": undirected.number_of_nodes(),
                "undirected_edges": undirected.number_of_edges(),
            },
        }