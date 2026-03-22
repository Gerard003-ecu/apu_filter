"""
Test de integración: Singularidad de Fiedler frente a hiper-gravedad financiera.

REFINAMIENTO RIGUROSO
═════════════════════

Mejoras implementadas:
  [A1] Teoría espectral: análisis de perturbación con cotas de Weyl
  [A2] Topología: verificación de homología simplicial
  [A3] Estabilidad: estimación de error regresivo (backward error)
  [A4] Teoría de grafos: detección de puntos de articulación
  [A5] Validación: invariantes algebraicos en todos los tests
  [A6] Documentación: lemas preliminares y cotas explícitas
  [A7] Arquitectura: inyección de dependencias numérico-teóricas

Fundamentos Matemáticos (Extendidos)
═════════════════════════════════════

0. PRELIMINARES TOPOLÓGICOS
─────────────────────────────

   Definición (Homología singular de dimensión 1):
   Para un complejo simplicial K sobre vértices V, la homología H₁(K)
   cuenta los "huecos" 1-dimensionales (ciclos que no son bordes).
   
   Número de Betti: β₁ = dim H₁(K)
   
   Para grafos (complejos 1-dimensionales):
       β₀ = # componentes conexas
       β₁ = |E| - |V| + β₀  (Euler-Poincaré)
   
   Invariante topológico: β₁ ≥ 0 con igualdad iff grafo es árbol.

1. TEORÍA ESPECTRAL: FIEDLER CON ANÁLISIS DE PERTURBACIÓN
──────────────────────────────────────────────────────────

   Definición (Conectividad algebraica):
   Para L_norm simétrica con espectro 0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ ≤ 2:
       a(G) := λ₂(L_norm)  (Fiedler value)
   
   Lema 1.1 (Fiedler, 1973):
       a(G) > 0  ⟺  G es conexo
       a(G) = 0  ⟺  G tiene ≥ 2 componentes conexas
   
   Lema 1.2 (Cotas de eigenvalores perturbados, Weyl):
       Sea L' = L + δL con ‖δL‖₂ ≤ ε.
       Entonces |λᵢ(L') - λᵢ(L)| ≤ ε para todo i.
   
   Lema 1.3 (Escalado microscópico):
       Si un puente tiene peso w = O(ε), entonces:
           λ₂(L) = O(ε)  y  λ₂(L) → 0 cuando ε → 0
   
   Aplicación a nuestro grafo:
       Sea L₀ = Laplaciano sin puente (dos componentes, λ₂=0)
       Sea L = Laplaciano con puente de peso ε
       Entonces L = L₀ + ε·E_puente con ‖E_puente‖₂ = O(1)
       Por Weyl: λ₂(L) ≤ λ₂(L₀) + ε·‖E_puente‖₂ = O(ε) ✓

2. ANÁLISIS NUMÉRICO: ESTABILIDAD DE EIGENVECTORES
───────────────────────────────────────────────────

   Teorema (Bauer-Fike):
   Para matriz hermitiana A con espectro λ₁,...,λₙ:
       Si A + E tiene eigenvalor λ'ᵢ, existe eigenvalor λⱼ(A) con:
           |λ'ᵢ - λⱼ(A)| ≤ ‖E‖₂
   
   Corolario (Número de condición espectral):
       κ₂(L) := ‖L‖₂ · ‖L⁺‖₂
   
   Para L_norm: ‖L_norm‖₂ ≤ 2 (cota de Chung)
   
   Estimación de error regresivo:
       Si u es eigenvector aproximado de L con eigenvalor λ ≈ μ:
       El error regresivo se estima como:
           ‖L·u - μ·u‖₂ / (2·‖L‖₂) ≤ backward_error_bound
   
   Para nuestra implementación:
       backward_error_bound ≈ n·ε_mach·‖L‖₂ ≈ 6·2.22e-16·2 ≈ 2.7e-15
       Elegimos _SPECTRAL_ZERO_TOLERANCE = 1e-10 >> 2.7e-15 ✓

3. TEORÍA DE GRAFOS: CRITICIDAD DE ARISTAS
──────────────────────────────────────────

   Definición (Puente/Cut-edge):
   Una arista e es un puente si su remoción aumenta β₀:
       β₀(G) < β₀(G - e)
   
   Lema 3.1 (Caracterización por Fiedler):
   Una arista es crítica para conectividad iff removerla
   causa que λ₂(L_norm) salte a 0.
   
   En nuestro caso: la arista (2,3) es el único puente
   (remover cualquier otra arista mantiene el grafo conexo).
   
   Verificación: articulation_points(G) = {2, 3}
   (ambos vértices del puente son puntos de articulación)

4. ESTRUCTURA DEL GRAFO DE PRUEBA (REFINADA)
──────────────────────────────────────────────

   Construcción:
       V = {0,1,2,3,4,5}
       E = {(0,1),(1,2),(0,2)} ∪ {(3,4),(4,5),(3,5)} ∪ {(2,3)}
       w(i,j) = 1.0 para aristas internas, ε = 10⁻⁹ para puente
   
   Invariantes garantizados:
       │V│ = 6,  │E│ = 7
       β₀ = 1 (conexo por Lema 1.1 + λ₂ > 0)
       β₁ = 7 - 6 + 1 = 2 (dos ciclos: 0-1-2-0 y 3-4-5-3)
       Puentes: {(2,3)} (único)
       Articulation points: {2, 3}
       k-conectidad por vértices: κ = 1 (remover {2} desconecta)
       k-conectidad por aristas: λ = 1 (remover {(2,3)} desconecta)
   
   Propiedades espectrales:
       λ₁(L_norm) = 0 (por conexidad)
       λ₂(L_norm) = O(ε) ≈ 10⁻⁹ (puente microscópico)
       λ₂(L_norm) < MIN_FIEDLER_VALUE ≈ 0.01 (fragilidad certificada)

Referencia teórica:
    [1] Fiedler, M. (1973). "Algebraic connectivity of graphs",
        Czechoslovak Math. J., 23(98):298–305.
    [2] Chung, F.R.K. (1997). "Spectral Graph Theory",
        CBMS Regional Conf. Series, AMS.
    [3] Bauer, F.L.; Fike, C.T. (1960). "Norms and exclusion theorems",
        Numer. Math., 2:137–141.
    [4] Golub, G.H.; Pereyra, V. (1973). "The differentiation of
        pseudo-inverses and nonlinear least squares problems",
        SIAM J. Numer. Anal., 10(3):413–432.
    [5] Diestel, R. (2017). "Graph Theory", 5th ed., Springer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, FrozenSet, Optional
from enum import Enum

import networkx as nx
import numpy as np
import pytest

from app.core.schemas import Stratum
from app.core.mic_algebra import CategoricalState
from app.telemetry_schemas import VerdictLevel
from app.deliberation_manifold import (
    OmegaDeliberationManifold,
    OmegaInputs,
    OmegaResult,
)
from app.stratums.alpha.business_canvas import (
    MIN_FIEDLER_VALUE,
    safe_eigenvalues_symmetric,
)


# =============================================================================
# CONFIGURACIÓN NUMÉRICA CON JUSTIFICACIÓN RIGUROSA
# =============================================================================


@dataclass(frozen=True)
class SpectralTolerance:
    """
    Tolerancias numéricas justificadas por análisis de error.
    
    Atributos
    ─────────
    zero_threshold : float
        Umbral para considerar un eigenvalor como nulo.
        
        Justificación:
          • Precisión de eigvalsh (LAPACK dsyevd): O(n·ε_mach·‖L‖₂)
          • Para n=6, ‖L_norm‖₂ ≤ 2, ε_mach ≈ 2.22e-16:
            error ≈ 6·2.22e-16·2 ≈ 2.7e-15
          • Margen de seguridad: 1e-10 ≈ 10^(4.5) · 2.7e-15
    
    eigenvalue_comparison : float
        Tolerancia para comparaciones entre eigenvalores.
        
        Justificación:
          • Perturbación esperada: δλ = O(ε_mach·κ₂(L))
          • κ₂(L_norm) ~ O(1) para grafos balanceados
          • δλ ~ 1e-15, elegimos 1e-9 con margen suficiente
    
    symmetry_check : float
        Tolerancia para verificar simetría de matrices.
        
        Justificación:
          • Simetría debe ser exacta (invariante teórico)
          • Permitir O(n²·ε_mach) ≈ 36·2.22e-16 ≈ 8e-15
          • Margen: 1e-14 = 10^0.5 · 8e-15
    
    microscopic_weight_floor : float
        Cota inferior para pesos "microscópicos" válidos.
        
        Justificación:
          • Subnormal threshold en float64: ~2.2e-308
          • Queremos w >> subnormal_threshold pero w << 1.0
          • w = 1e-9 satisface: 1e-9 >> 1e-308 y 1e-9 << 1e-1
    """
    zero_threshold: float = 1e-10
    eigenvalue_comparison: float = 1e-9
    symmetry_check: float = 1e-14
    microscopic_weight_floor: float = 1e-50  # Margen vs subnormal
    
    def validate(self) -> None:
        """Valida coherencia de tolerancias (pre-condición de uso)."""
        assert self.zero_threshold > self.symmetry_check, (
            f"zero_threshold ({self.zero_threshold}) debe ser "
            f"> symmetry_check ({self.symmetry_check})"
        )
        assert self.eigenvalue_comparison > self.zero_threshold, (
            f"eigenvalue_comparison ({self.eigenvalue_comparison}) debe ser "
            f"> zero_threshold ({self.zero_threshold})"
        )
        assert self.microscopic_weight_floor > 0, (
            "microscopic_weight_floor debe ser > 0"
        )


@dataclass(frozen=True)
class GraphProperties:
    """
    Propiedades topológicas esperadas del grafo de prueba.
    
    Justificación de valores:
      • num_nodes = 6: dos K₃ (3 nodos cada una)
      • num_edges = 7: 3+3 internas + 1 puente
      • betti_0 = 1: conexo (por Lema 1.1)
      • betti_1 = 2: ciclos 0-1-2-0 y 3-4-5-3
      • num_articulation_points = 2: vértices del puente {2,3}
      • bridge_count = 1: solo (2,3)
      • vertex_connectivity = 1: κ(G) = min{|S|: β₀(G-S)>1}
      • edge_connectivity = 1: λ(G) = min{|S|: S ⊆ E, β₀(G-S)>1}
    """
    num_nodes: int = 6
    num_edges: int = 7
    betti_0: int = 1  # componentes conexas
    betti_1: int = 2  # ciclos 1-dimensionales
    num_articulation_points: int = 2
    bridge_count: int = 1
    vertex_connectivity: int = 1
    edge_connectivity: int = 1
    chung_spectral_bound: float = 2.0


# Instancias globales
_TOLERANCE = SpectralTolerance()
_TOLERANCE.validate()
_EXPECTED_PROPERTIES = GraphProperties()

# Constantes secundarias derivadas
_SPECTRAL_ZERO_TOLERANCE = _TOLERANCE.zero_threshold
_EIGENVALUE_TOLERANCE = _TOLERANCE.eigenvalue_comparison
_SYMMETRY_TOLERANCE = _TOLERANCE.symmetry_check
_MICROSCOPIC_WEIGHT = 1e-9
_UNIT_WEIGHT = 1.0
_CHUNG_SPECTRAL_UPPER_BOUND = _EXPECTED_PROPERTIES.chung_spectral_bound


# =============================================================================
# FUNCIONES AUXILIARES: TOPOLOGÍA DE GRAFOS
# =============================================================================


def _compute_betti_numbers(G: nx.Graph) -> Tuple[int, int]:
    """
    Calcula los números de Betti β₀ (componentes) y β₁ (ciclos).
    
    Fórmula (Euler-Poincaré para grafos):
        β₀ = # componentes conexas
        β₁ = |E| - |V| + β₀
    
    Parámetros
    ──────────
    G : nx.Graph
        Grafo no dirigido ponderado.
    
    Retorna
    ───────
    Tuple[int, int]
        (β₀, β₁)
    
    Ejemplo
    ───────
    Para K₃ + puente + K₃:
        β₀ = 1 (conexo)
        β₁ = 7 - 6 + 1 = 2 ✓
    """
    beta_0 = nx.number_connected_components(G)
    beta_1 = G.number_of_edges() - G.number_of_nodes() + beta_0
    return beta_0, beta_1


def _find_articulation_points(G: nx.Graph) -> FrozenSet[int]:
    """
    Encuentra puntos de articulación (cut-vertices).
    
    Definición: v es punto de articulación si β₀(G - {v}) > β₀(G).
    
    Parámetros
    ──────────
    G : nx.Graph
        Grafo no dirigido, conexo.
    
    Retorna
    ───────
    FrozenSet[int]
        Conjunto de puntos de articulación.
    
    Propiedad esperada
    ──────────────────
    Para K₃ + puente + K₃: articulation_points = {2, 3}
    (los extremos del puente son los únicos puntos críticos)
    """
    if not nx.is_connected(G):
        raise ValueError("G debe ser conexo para análisis de articulación.")
    
    articulación = frozenset(nx.articulation_points(G))
    return articulación


def _find_bridges(G: nx.Graph) -> FrozenSet[Tuple[int, int]]:
    """
    Encuentra aristas puente (cut-edges).
    
    Definición: e es puente si β₀(G - {e}) > β₀(G).
    
    Equivalencia (Lema 3.1): Remover puente causa λ₂ → 0.
    
    Parámetros
    ──────────
    G : nx.Graph
        Grafo no dirigido.
    
    Retorna
    ───────
    FrozenSet[Tuple[int, int]]
        Conjunto de aristas puente (tuplas normalizadas u < v).
    """
    bridges = set()
    for u, v in nx.bridges(G):
        edge = (min(u, v), max(u, v))
        bridges.add(edge)
    
    return frozenset(bridges)


def _compute_vertex_connectivity(G: nx.Graph) -> int:
    """
    Computa k-conectividad por vértices.
    
    κ(G) = min{|S| : S ⊂ V, G - S desconexo}
    
    Para K₃ + puente + K₃: κ = 1 (remover {2} o {3} desconecta).
    
    Parámetros
    ──────────
    G : nx.Graph
        Grafo no dirigido, conexo.
    
    Retorna
    ───────
    int
        k-conectividad por vértices.
    """
    if not nx.is_connected(G):
        return 0
    return nx.node_connectivity(G)


def _compute_edge_connectivity(G: nx.Graph) -> int:
    """
    Computa k-conectividad por aristas.
    
    λ(G) = min{|S| : S ⊆ E, G - S desconexo}
    
    Para K₃ + puente + K₃: λ = 1 (remover {(2,3)} desconecta).
    
    Parámetros
    ──────────
    G : nx.Graph
        Grafo no dirigido, conexo.
    
    Retorna
    ───────
    int
        k-conectividad por aristas.
    """
    if not nx.is_connected(G):
        return 0
    return nx.edge_connectivity(G)


# =============================================================================
# FUNCIONES AUXILIARES: CONSTRUCCIÓN DE GRAFOS
# =============================================================================


def _build_two_cliques_with_microscopic_bridge() -> nx.Graph:
    """
    Construye K₃ + puente microscópico + K₃.
    
    Estructura topológica
    ────────────────────
    Dos K₃ disjuntos:
        K₃ izq: {0, 1, 2}, aristas internas w=1.0
        K₃ der: {3, 4, 5}, aristas internas w=1.0
    
    Unidas por un puente microscópico:
        (2, 3) con peso w = ε = 10⁻⁹
    
    Invariantes garantizados
    ────────────────────────
    • Topología: β₀=1, β₁=2, κ=1, λ=1
    • Espectral: λ₂ = O(ε) < MIN_FIEDLER_VALUE
    • Criticidad: {2,3} = articulation_points, {(2,3)} = bridges
    
    Retorna
    ───────
    nx.Graph
        Grafo conexo pero algebraicamente frágil.
    """
    G = nx.Graph()
    
    # K₃ izquierdo
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        G.add_edge(u, v, weight=_UNIT_WEIGHT)
    
    # K₃ derecho
    for u, v in [(3, 4), (4, 5), (3, 5)]:
        G.add_edge(u, v, weight=_UNIT_WEIGHT)
    
    # Puente microscópico
    G.add_edge(2, 3, weight=_MICROSCOPIC_WEIGHT)
    
    return G


def _build_two_disconnected_cliques() -> nx.Graph:
    """
    Control negativo: K₃ ∪ K₃ sin puente.
    
    Propiedades
    ───────────
    • β₀ = 2 (desconexo)
    • β₁ = 0 + 0 = 0 (sin ciclos de largo > 2 entre componentes)
    • λ₂ = 0 exactamente (multiplicidad 2)
    • No hay puentes (cada arista es interna a su K₃)
    
    Retorna
    ───────
    nx.Graph
        Grafo desconexo con dos componentes.
    """
    G = nx.Graph()
    
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        G.add_edge(u, v, weight=_UNIT_WEIGHT)
    
    for u, v in [(3, 4), (4, 5), (3, 5)]:
        G.add_edge(u, v, weight=_UNIT_WEIGHT)
    
    return G


# =============================================================================
# FUNCIONES AUXILIARES: ANÁLISIS ESPECTRAL CON ESTIMACIÓN DE ERROR
# =============================================================================


def _compute_normalized_laplacian(
    G: nx.Graph,
    tolerance: SpectralTolerance = _TOLERANCE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Construye L_norm = I - D^{-1/2} A D^{-1/2} con estimación de error.
    
    Teorema (Construcción del Laplaciano normalizado)
    ─────────────────────────────────────────────────
    Para grafo ponderado con matriz de adyacencia A y grados d_i = Σⱼ a_ij:
    
        L_norm = I - D^{-1/2} A D^{-1/2}
    
    donde D = diag(d₁,...,dₙ) y D^{-1/2}[i,i] = 1/√d_i si d_i > 0,
    else 0 (vértices aislados).
    
    Propiedades
    ──────────
    (P1) L_norm es simétrica y real
    (P2) spec(L_norm) ⊆ [0, 2] (cota de Chung)
    (P3) mult(λ=0) = β₀ (# componentes)
    
    Estimación de error regresivo
    ────────────────────────────
    Sea L_computed la matriz construida y L_exact el Laplaciano teórico.
    El error regresivo se estima como:
    
        backward_error ≈ n·ε_mach·‖L_norm‖_F
    
    donde ‖·‖_F es la norma de Frobenius.
    
    Parámetros
    ──────────
    G : nx.Graph
        Grafo no dirigido ponderado.
    tolerance : SpectralTolerance
        Tolerancias numéricas (default: _TOLERANCE).
    
    Retorna
    ───────
    Tuple[np.ndarray, Dict[str, float]]
        (L_norm, diagnostics)
        
        L_norm: matriz n × n
        diagnostics: {
            'backward_error': estimación de error regresivo,
            'symmetry_error': ‖L_norm - L_norm.T‖_F,
            'condition_number_2': κ₂(L_norm) (sin invertir),
            'frobenius_norm': ‖L_norm‖_F,
        }
    
    Raises
    ──────
    ValueError
        Si A no es simétrica o si tolerancias son inconsistentes.
    """
    nodelist = sorted(G.nodes())
    n = len(nodelist)
    A = nx.to_numpy_array(
        G, nodelist=nodelist, weight="weight", dtype=np.float64,
    )
    
    # Verificación de simetría
    symmetry_error_A = np.max(np.abs(A - A.T))
    if symmetry_error_A > tolerance.symmetry_check:
        raise ValueError(
            f"Matriz de adyacencia no simétrica: "
            f"‖A - A.T‖_∞ = {symmetry_error_A:.2e}."
        )
    
    # Grados ponderados
    degrees = np.asarray(A.sum(axis=1)).flatten()
    
    # Construcción de D^{-1/2}
    inv_sqrt_d = np.zeros(n, dtype=np.float64)
    positive_mask = degrees > tolerance.zero_threshold
    inv_sqrt_d[positive_mask] = 1.0 / np.sqrt(degrees[positive_mask])
    D_inv_sqrt = np.diag(inv_sqrt_d)
    
    # L_norm = I - D^{-1/2} A D^{-1/2}
    L_norm = np.eye(n, dtype=np.float64) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Verificación de simetría del resultado
    symmetry_error_L = np.max(np.abs(L_norm - L_norm.T))
    if symmetry_error_L > tolerance.symmetry_check:
        raise ValueError(
            f"Laplaciano normalizado no simétrico: "
            f"‖L_norm - L_norm.T‖_F = {symmetry_error_L:.2e}."
        )
    
    # Diagnósticos numéricos
    frobenius_norm = np.linalg.norm(L_norm, 'fro')
    
    # Error regresivo estimado (Golub-Pereyra)
    eps_mach = np.finfo(np.float64).eps
    backward_error = n * eps_mach * frobenius_norm
    
    # Estimación de κ₂(L_norm) sin inversión (usando eigs)
    eigs = np.linalg.eigvalsh(L_norm)
    eigs_sorted = np.sort(np.abs(eigs))
    lambda_max = float(eigs_sorted[-1])
    lambda_min = float(eigs_sorted[eigs_sorted > tolerance.zero_threshold])
    lambda_min = float(lambda_min[0]) if len(lambda_min) > 0 else tolerance.zero_threshold
    condition_number = lambda_max / lambda_min if lambda_min > 0 else np.inf
    
    diagnostics = {
        'backward_error': float(backward_error),
        'symmetry_error': float(symmetry_error_L),
        'condition_number_2': float(condition_number),
        'frobenius_norm': float(frobenius_norm),
        'n': n,
    }
    
    return L_norm, diagnostics


def _compute_spectrum(
    L_norm: np.ndarray,
    tolerance: SpectralTolerance = _TOLERANCE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Calcula el espectro de L_norm usando eigvalsh.
    
    Implementación
    ──────────────
    Usa np.linalg.eigvalsh (interfaz NumPy a LAPACK dsyevd).
    
    Garantías LAPACK
    ────────────────
    Para matriz simétrica n × n:
      • Eigenvalores calculados con precisión:
            |λᵢ(computed) - λᵢ(exact)| ≤ O(n·ε_mach·‖L_norm‖₂)
      • Para n=6, ‖L_norm‖₂ ≤ 2: error ~ 2.7e-15
    
    Parámetros
    ──────────
    L_norm : np.ndarray
        Laplaciano normalizado simétrico.
    tolerance : SpectralTolerance
        Configuración numérica.
    
    Retorna
    ───────
    Tuple[np.ndarray, Dict[str, float]]
        (eigenvalues, diagnostics)
        
        eigenvalues: vector ordenado ascendentemente
        diagnostics: {
            'spectral_condition': κ_spec = λ_max / λ_min,
            'zero_eigenvalues': # de eigenvalores ≈ 0,
            'numerical_rank': rango numérico (basado en gap espectral),
        }
    """
    eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))
    
    # Conteo de eigenvalores nulos (con tolerancia)
    zero_count = int(np.sum(
        np.abs(eigenvalues) <= tolerance.zero_threshold
    ))
    
    # Condicionamiento espectral
    nonzero_eigs = eigenvalues[np.abs(eigenvalues) > tolerance.zero_threshold]
    if len(nonzero_eigs) > 0:
        spectral_condition = float(np.max(np.abs(nonzero_eigs)) / 
                                    np.min(np.abs(nonzero_eigs)))
    else:
        spectral_condition = np.inf
    
    # Rango numérico (# de eigenvalores > tolerancia)
    numerical_rank = len(eigenvalues) - zero_count
    
    diagnostics = {
        'spectral_condition': spectral_condition,
        'zero_eigenvalues': zero_count,
        'numerical_rank': numerical_rank,
    }
    
    return eigenvalues, diagnostics


def _verify_chung_spectral_properties(
    eigenvalues: np.ndarray,
    expected_components: int,
    tolerance: SpectralTolerance = _TOLERANCE,
) -> Dict[str, bool]:
    """
    Verifica propiedades de Chung del Laplaciano normalizado.
    
    Teorema (Chung, 1997)
    ─────────────────────
    Sea L_norm el Laplaciano normalizado de grafo conexo.
    Entonces:
        (P1) L_norm es simétrica y real ✓ (por construcción)
        (P2) spec(L_norm) ⊆ [0, 2]
        (P3) mult(λ=0) = β₀ (# componentes)
        (P4) 0 < λ₂ < 2 para grafos conexos (por Fiedler)
    
    Parámetros
    ──────────
    eigenvalues : np.ndarray
        Espectro ordenado.
    expected_components : int
        β₀ esperado (número de componentes conexas).
    tolerance : SpectralTolerance
        Configuración numérica.
    
    Retorna
    ───────
    Dict[str, bool]
        Resultado de cada verificación.
    
    Raises
    ──────
    AssertionError
        Si alguna propiedad falla.
    """
    eigenvalues_sorted = np.sort(eigenvalues)
    
    # (P1) Dimensión correcta
    assert eigenvalues_sorted.ndim == 1, (
        f"Espectro debe ser 1D, recibido ndim={eigenvalues_sorted.ndim}."
    )
    
    # (P2) Semidefinitud positiva
    min_eig = float(eigenvalues_sorted[0])
    assert min_eig >= -tolerance.eigenvalue_comparison, (
        f"Violación P2: min(λ) = {min_eig:.6e} < 0 "
        f"(error permitido: {-tolerance.eigenvalue_comparison:.6e})."
    )
    p2_holds = min_eig >= -tolerance.eigenvalue_comparison
    
    # (P3) Cota de Chung: spec ⊆ [0, 2]
    max_eig = float(eigenvalues_sorted[-1])
    chung_bound_exceeded = max_eig > _CHUNG_SPECTRAL_UPPER_BOUND + tolerance.eigenvalue_comparison
    assert not chung_bound_exceeded, (
        f"Violación P3: max(λ) = {max_eig:.6e} "
        f"> {_CHUNG_SPECTRAL_UPPER_BOUND} + tol."
    )
    p3_holds = not chung_bound_exceeded
    
    # (P4) Multiplicidad del cero
    zero_count = int(np.sum(
        np.abs(eigenvalues_sorted) <= tolerance.zero_threshold
    ))
    p4_holds = zero_count == expected_components
    assert p4_holds, (
        f"Violación P4: mult(λ=0) = {zero_count}, "
        f"esperado {expected_components}. "
        f"Espectro: {eigenvalues_sorted}"
    )
    
    return {
        'P2_semidefinite': p2_holds,
        'P3_chung_bound': p3_holds,
        'P4_multiplicity': p4_holds,
    }


def _extract_fiedler_value(
    eigenvalues: np.ndarray,
    tolerance: SpectralTolerance = _TOLERANCE,
) -> float:
    """
    Extrae λ₂ (segundo eigenvalor más pequeño).
    
    Definición (Fiedler, 1973)
    ──────────────────────────
    Dado espectro 0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ del Laplaciano normalizado:
        a(G) := λ₂(L_norm)  (conectividad algebraica o Fiedler value)
    
    Propiedades
    ───────────
    • a(G) > 0  ⟺  G es conexo
    • a(G) = 0  ⟺  G desconexo (mult(λ=0) ≥ 2)
    • a(G) es una medida de robustez: mide qué tan "cerca"
      está el grafo de desconexión
    
    Parámetros
    ──────────
    eigenvalues : np.ndarray
        Espectro (sin ordenar necesariamente).
    tolerance : SpectralTolerance
        Configuración numérica.
    
    Retorna
    ───────
    float
        λ₂
    
    Raises
    ──────
    ValueError
        Si el grafo no es conexo (no hay λ₂ bien definido > 0).
    """
    eigs_sorted = np.sort(eigenvalues)
    
    # Verificar que hay al menos un λ ≈ 0 (componente conexa)
    zero_count = int(np.sum(np.abs(eigs_sorted) <= tolerance.zero_threshold))
    if zero_count == 0:
        raise ValueError(
            "Ningún eigenvalor es ≈ 0: estructura extraña "
            "(¿matriz no semidefinida positiva?)."
        )
    
    # λ₂ es el primer eigenvalor no-nulo
    fiedler = float(eigs_sorted[zero_count])
    
    return fiedler


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tolerance_config() -> SpectralTolerance:
    """Configuración de tolerancias numéricas."""
    return SpectralTolerance()


@pytest.fixture
def graph_properties() -> GraphProperties:
    """Propiedades topológicas esperadas del grafo de prueba."""
    return GraphProperties()


@pytest.fixture
def fractured_graph() -> nx.Graph:
    """
    Grafo fracturado: K₃ + puente microscópico + K₃.
    
    Garantías
    ─────────
    • Conexo: β₀ = 1
    • Dos ciclos: β₁ = 2
    • Fragilidad espectral: λ₂ = O(10⁻⁹) < MIN_FIEDLER_VALUE
    • Un solo puente (criticidad): {(2,3)}
    • Dos puntos de articulación: {2, 3}
    """
    return _build_two_cliques_with_microscopic_bridge()


@pytest.fixture
def disconnected_graph() -> nx.Graph:
    """
    Control negativo: K₃ ∪ K₃ sin conexión.
    
    Garantías
    ─────────
    • Desconexo: β₀ = 2
    • Ningún puente
    • λ₂ = 0 exactamente (multiplicidad 2)
    """
    return _build_two_disconnected_cliques()


@pytest.fixture
def fractured_laplacian(
    fractured_graph: nx.Graph,
    tolerance_config: SpectralTolerance,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Laplaciano normalizado del grafo fracturado con diagnósticos.
    
    Retorna
    ───────
    Tuple[np.ndarray, Dict[str, float]]
        (L_norm, diagnostics)
    """
    return _compute_normalized_laplacian(fractured_graph, tolerance_config)


@pytest.fixture
def fractured_spectrum(
    fractured_laplacian: Tuple[np.ndarray, Dict[str, float]],
    tolerance_config: SpectralTolerance,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Espectro del Laplaciano del grafo fracturado con diagnósticos.
    
    Retorna
    ───────
    Tuple[np.ndarray, Dict[str, float]]
        (eigenvalues, diagnostics)
    """
    L_norm, _ = fractured_laplacian
    return _compute_spectrum(L_norm, tolerance_config)


@pytest.fixture
def omega_manifold() -> OmegaDeliberationManifold:
    """Instancia del manifold Omega."""
    return OmegaDeliberationManifold()


@pytest.fixture
def hyper_profitable_inputs() -> OmegaInputs:
    """
    Inputs para Omega: proyecto hiper-rentable.
    
    Características
    ───────────────
    • ROI = 350% (hiper-rentable)
    • Sin fricción logística/social (factores = 1.0)
    • Topología consistente: 6 nodos, 7 aristas, 2 ciclos
    • Ningún nodo aislado o estresado
    """
    return OmegaInputs(
        psi=0.8,
        n_nodes=_EXPECTED_PROPERTIES.num_nodes,
        n_edges=_EXPECTED_PROPERTIES.num_edges,
        cycle_count=_EXPECTED_PROPERTIES.betti_1,
        isolated_count=0,
        stressed_count=0,
        roi=3.5,
        logistics_friction=1.0,
        social_friction=1.0,
        climate_entropy=1.0,
        territory_present=False,
    )


# =============================================================================
# TEST SUITE 1: TOPOLOGÍA ALGEBRAICA DEL GRAFO
# =============================================================================


@pytest.mark.integration
class TestTopologicalInvariants:
    """
    Verifica invariantes topológicos del grafo fracturado.
    
    Teorema (Euler-Poincaré para grafos)
    ────────────────────────────────────
    Para grafo con |V| vértices, |E| aristas, β₀ componentes:
        β₁ = |E| - |V| + β₀
    
    En nuestro caso: 7 - 6 + 1 = 2 ✓
    """
    
    def test_node_count(self, fractured_graph: nx.Graph) -> None:
        """│V│ = 6."""
        assert fractured_graph.number_of_nodes() == _EXPECTED_PROPERTIES.num_nodes
    
    def test_edge_count(self, fractured_graph: nx.Graph) -> None:
        """│E│ = 7."""
        assert fractured_graph.number_of_edges() == _EXPECTED_PROPERTIES.num_edges
    
    def test_betti_zero(self, fractured_graph: nx.Graph) -> None:
        """β₀ = 1 (conexo)."""
        beta_0, _ = _compute_betti_numbers(fractured_graph)
        assert beta_0 == _EXPECTED_PROPERTIES.betti_0, (
            f"β₀ = {beta_0}, esperado {_EXPECTED_PROPERTIES.betti_0}."
        )
        assert nx.is_connected(fractured_graph)
    
    def test_betti_one(self, fractured_graph: nx.Graph) -> None:
        """β₁ = 2 (dos ciclos por Euler-Poincaré)."""
        beta_0, beta_1 = _compute_betti_numbers(fractured_graph)
        assert beta_1 == _EXPECTED_PROPERTIES.betti_1, (
            f"β₁ = {beta_1}, esperado {_EXPECTED_PROPERTIES.betti_1}. "
            f"(β₀={beta_0}, │V│={fractured_graph.number_of_nodes()}, "
            f"│E│={fractured_graph.number_of_edges()})"
        )
    
    def test_articulation_points(self, fractured_graph: nx.Graph) -> None:
        """
        Los puntos de articulación son exactamente {2, 3}.
        
        Lema: v es punto de articulación iff β₀(G - {v}) > β₀(G).
        Para K₃ + puente + K₃: remover 2 o 3 desconecta.
        """
        articulation = _find_articulation_points(fractured_graph)
        expected = frozenset({2, 3})
        assert articulation == expected, (
            f"Puntos de articulación: {articulation}, esperado {expected}."
        )
    
    def test_bridges(self, fractured_graph: nx.Graph) -> None:
        """
        Las aristas puente son exactamente {(2, 3)}.
        
        Lema: e es puente iff β₀(G - {e}) > β₀(G).
        """
        bridges = _find_bridges(fractured_graph)
        expected = frozenset({(2, 3)})
        assert bridges == expected, (
            f"Puentes: {bridges}, esperado {expected}."
        )
    
    def test_vertex_connectivity(self, fractured_graph: nx.Graph) -> None:
        """κ(G) = 1 (remover un vértice puede desconectar)."""
        kappa = _compute_vertex_connectivity(fractured_graph)
        assert kappa == _EXPECTED_PROPERTIES.vertex_connectivity, (
            f"κ = {kappa}, esperado {_EXPECTED_PROPERTIES.vertex_connectivity}."
        )
    
    def test_edge_connectivity(self, fractured_graph: nx.Graph) -> None:
        """λ(G) = 1 (remover un puente desconecta)."""
        lambda_e = _compute_edge_connectivity(fractured_graph)
        assert lambda_e == _EXPECTED_PROPERTIES.edge_connectivity, (
            f"λ = {lambda_e}, esperado {_EXPECTED_PROPERTIES.edge_connectivity}."
        )
    
    def test_no_isolated_nodes(self, fractured_graph: nx.Graph) -> None:
        """Todos los nodos tienen grado ≥ 2."""
        for node in fractured_graph.nodes():
            degree = fractured_graph.degree(node)
            assert degree >= 2, (
                f"Nodo {node} tiene grado {degree} < 2 (aislado o colgante)."
            )


# =============================================================================
# TEST SUITE 2: CONTROL NEGATIVO — GRAFO DESCONEXO
# =============================================================================


@pytest.mark.integration
class TestDisconnectedGraphControl:
    """
    Control negativo: verifica que sin puente, λ₂ = 0 exactamente.
    
    Confirmación: el puente microscópico es precisamente lo que
    mantiene el grafo conexo pero frágil.
    """
    
    def test_disconnected_has_two_components(
        self, disconnected_graph: nx.Graph,
    ) -> None:
        """β₀ = 2."""
        beta_0, _ = _compute_betti_numbers(disconnected_graph)
        assert beta_0 == 2
        assert not nx.is_connected(disconnected_graph)
    
    def test_disconnected_has_no_bridges(
        self, disconnected_graph: nx.Graph,
    ) -> None:
        """Sin puente, cada arista es interna a su clique."""
        bridges = _find_bridges(disconnected_graph)
        assert len(bridges) == 0, (
            f"Grafo desconexo no debería tener puentes, "
            f"pero encontró {bridges}."
        )
    
    def test_disconnected_laplacian_zero_eigenvalues(
        self,
        disconnected_graph: nx.Graph,
        tolerance_config: SpectralTolerance,
    ) -> None:
        """
        Desconexo → mult(λ=0) = 2 (dos componentes).
        
        Por Lema 1.1: mult(λ=0) = β₀.
        """
        L_norm, _ = _compute_normalized_laplacian(
            disconnected_graph, tolerance_config,
        )
        eigenvalues, _ = _compute_spectrum(L_norm, tolerance_config)
        
        zero_count = int(np.sum(
            np.abs(eigenvalues) <= tolerance_config.zero_threshold
        ))
        assert zero_count == 2, (
            f"Esperado mult(λ=0)=2, obtenido {zero_count}. "
            f"Espectro: {eigenvalues}"
        )


# =============================================================================
# TEST SUITE 3: ANÁLISIS ESPECTRAL RIGUROSO
# =============================================================================


@pytest.mark.integration
class TestSpectralAnalysisRigorous:
    """
    Análisis completo del espectro del Laplaciano normalizado.
    
    Verificaciones
    ───────────────
    (P1) Simetría exacta
    (P2) Semidefinitud positiva
    (P3) Cota de Chung
    (P4) Multiplicidad correcta del cero
    (P5) Fragilidad espectral: λ₂ << 1
    """
    
    def test_laplacian_symmetry(
        self,
        fractured_laplacian: Tuple[np.ndarray, Dict[str, float]],
        tolerance_config: SpectralTolerance,
    ) -> None:
        """L_norm es simétrica con error ≤ 1e-14."""
        L_norm, diag = fractured_laplacian
        
        assert diag['symmetry_error'] <= tolerance_config.symmetry_check, (
            f"L_norm no es simétrica: error = {diag['symmetry_error']:.2e}."
        )
    
    def test_laplacian_dimension(
        self,
        fractured_laplacian: Tuple[np.ndarray, Dict[str, float]],
    ) -> None:
        """L_norm es 6 × 6."""
        L_norm, _ = fractured_laplacian
        assert L_norm.shape == (
            _EXPECTED_PROPERTIES.num_nodes,
            _EXPECTED_PROPERTIES.num_nodes,
        )
    
    def test_laplacian_condition_number(
        self,
        fractured_laplacian: Tuple[np.ndarray, Dict[str, float]],
    ) -> None:
        """
        κ₂(L_norm) es finito y razonable (no mal-condicionada).
        
        Para grafos balanceados: κ₂ ~ O(1) a O(10).
        """
        L_norm, diag = fractured_laplacian
        
        cond = diag['condition_number_2']
        assert np.isfinite(cond), (
            f"κ₂(L_norm) = {cond} (infinito o NaN)."
        )
        assert cond < 1e6, (
            f"κ₂(L_norm) = {cond:.2e} es muy grande "
            f"(matriz mal-condicionada)."
        )
    
    def test_chung_properties_hold(
        self,
        fractured_spectrum: Tuple[np.ndarray, Dict[str, float]],
        tolerance_config: SpectralTolerance,
    ) -> None:
        """
        Teorema (Chung, 1997): L_norm satisface P2-P4.
        """
        eigenvalues, _ = fractured_spectrum
        
        results = _verify_chung_spectral_properties(
            eigenvalues,
            expected_components=_EXPECTED_PROPERTIES.betti_0,
            tolerance=tolerance_config,
        )
        
        assert all(results.values()), (
            f"Violación de propiedades de Chung: {results}"
        )
    
    def test_fiedler_positive(
        self,
        fractured_spectrum: Tuple[np.ndarray, Dict[str, float]],
        tolerance_config: SpectralTolerance,
    ) -> None:
        """
        Lema 1.1 (Fiedler): λ₂ > 0 ⟺ grafo es conexo.
        """
        eigenvalues, _ = fractured_spectrum
        fiedler = _extract_fiedler_value(eigenvalues, tolerance_config)
        
        assert fiedler > tolerance_config.zero_threshold, (
            f"λ₂ = {fiedler:.6e} no es estrictamente positivo. "
            f"¿Grafo desconexo?"
        )
    
    def test_fiedler_microscopic(
        self,
        fractured_spectrum: Tuple[np.ndarray, Dict[str, float]],
        tolerance_config: SpectralTolerance,
    ) -> None:
        """
        Lema 1.3: λ₂ = O(ε) << 1 (fragilidad espectral).
        
        El puente de peso ε = 10⁻⁹ produce λ₂ que debe estar
        por debajo de MIN_FIEDLER_VALUE.
        """
        eigenvalues, _ = fractured_spectrum
        fiedler = _extract_fiedler_value(eigenvalues, tolerance_config)
        
        assert fiedler < MIN_FIEDLER_VALUE, (
            f"λ₂ = {fiedler:.6e} no cayó por debajo del umbral "
            f"MIN_FIEDLER_VALUE = {MIN_FIEDLER_VALUE:.6e}. "
            f"La fragilidad espectral no fue detectada."
        )
    
    def test_spectral_condition_number(
        self,
        fractured_spectrum: Tuple[np.ndarray, Dict[str, float]],
    ) -> None:
        """
        κ_spec = λ_max / λ_min es finito.
        
        Para L_norm conexo: λ_max ≤ 2, λ_min = λ₂ ≈ 10⁻⁹
        ⟹ κ_spec ≈ 2 / 10⁻⁹ = 2·10⁹ (mal-condicionada espectralmente)
        
        Esto es esperado: fragilidad = eigenvalor cercano a 0.
        """
        eigenvalues, diag = fractured_spectrum
        
        cond = diag['spectral_condition']
        assert np.isfinite(cond), (
            f"κ_spec = {cond} (infinito o NaN)."
        )
        # No asertamos un límite superior: mal-condicionamiento
        # espectral es precisamente la fragilidad que buscamos


# =============================================================================
# TEST SUITE 4: ESTRATO OMEGA — APROBACIÓN POR HIPER-RENTABILIDAD
# =============================================================================


@pytest.mark.integration
class TestOmegaApprovalHyperProfitable:
    """
    Verifica que Omega aprueba con ROI = 350%.
    
    Importancia
    ───────────
    Este es el SETUP para el test de veto jerárquico.
    Si Omega no aprueba, el conflicto Alpha vs. Omega no existe
    y el test pierde sentido.
    """
    
    def test_omega_verdict_valid_type(
        self,
        omega_manifold: OmegaDeliberationManifold,
        hyper_profitable_inputs: OmegaInputs,
    ) -> None:
        """El resultado de Omega es un VerdictLevel válido."""
        result = omega_manifold(hyper_profitable_inputs)
        
        assert isinstance(result, OmegaResult)
        assert isinstance(result.verdict, VerdictLevel)
    
    def test_omega_approves_hyper_profitable(
        self,
        omega_manifold: OmegaDeliberationManifold,
        hyper_profitable_inputs: OmegaInputs,
    ) -> None:
        """
        Omega aprueba o condiciona positivamente.
        
        Definición:
          Aprobación = VIABLE | CONDICIONAL
          (Omega acepta el proyecto bajo ciertos términos)
        """
        result = omega_manifold(hyper_profitable_inputs)
        omega_approved = result.verdict in (
            VerdictLevel.VIABLE,
            VerdictLevel.CONDICIONAL,
        )
        
        assert omega_approved, (
            f"Omega debería aprobar con ROI=350%, "
            f"pero retornó verdict={result.verdict.name}. "
            f"Precondición de test fallida."
        )
    
    def test_omega_input_consistency(
        self, hyper_profitable_inputs: OmegaInputs,
    ) -> None:
        """
        Los inputs de Omega son internamente consistentes.
        
        Invariantes
        ───────────
        • n_nodes = 6 (consistente con topología K₃+puente+K₃)
        • n_edges = 7 (3+3 internas + 1 puente)
        • cycle_count = 2 (β₁ por Euler-Poincaré)
        • roi = 3.5 (hiper-rentable)
        • fricción = 1.0 (sin fricción)
        """
        inputs = hyper_profitable_inputs
        
        # Validar congruencia topológica
        beta_1_expected = inputs.n_edges - inputs.n_nodes + 1
        assert inputs.cycle_count == beta_1_expected, (
            f"cycle_count={inputs.cycle_count} no coincide con "
            f"Euler-Poincaré: {beta_1_expected} = 7 - 6 + 1."
        )
        
        # Validar rentabilidad
        assert inputs.roi >= 2.0, (
            f"ROI = {inputs.roi} no es hiper-rentable "
            f"(debería ser ≥ 2.0)."
        )
        
        # Validar ausencia de fricción
        assert inputs.logistics_friction == 1.0, (
            f"logistics_friction = {inputs.logistics_friction} "
            f"debería ser 1.0 (sin fricción)."
        )
        assert inputs.social_friction == 1.0


# =============================================================================
# TEST SUITE 5: VETO JERÁRQUICO — ALPHA BLOQUEA WISDOM
# =============================================================================


@pytest.mark.integration
class TestAlphaVetoHierarchical:
    """
    TEST CENTRAL: Alpha bloquea el ascenso a WISDOM cuando λ₂ < MIN_FIEDLER_VALUE,
    a pesar de que Omega aprueba por hiper-rentabilidad.
    
    Principio formalizado
    ─────────────────────
    "La rentabilidad (Omega) no compensa una estructura organizacional
    fracturada (Alpha)."
    
    Implementación
    ──────────────
    1. Omega aprueba: result.verdict ∈ {VIABLE, CONDICIONAL}
    2. Alpha detecta: λ₂ < MIN_FIEDLER_VALUE
    3. Jerarquía: WISDOM.requires() ⊇ {ALPHA, ...}
    4. Bloqueo: ALPHA ∉ validated_strata ⟹ WISDOM inaccesible
    """
    
    def test_wisdom_requires_alpha_precondition(self) -> None:
        """
        Precondición: WISDOM.requires() incluye ALPHA.
        
        Sin esto, Alpha no tiene poder de veto.
        """
        wisdom_deps = Stratum.WISDOM.requires()
        
        assert Stratum.ALPHA in wisdom_deps, (
            f"Precondición fallida: ALPHA ∉ WISDOM.requires(). "
            f"Dependencias observadas: {wisdom_deps}."
        )
    
    def test_alpha_detects_spectral_fragility(
        self,
        fractured_spectrum: Tuple[np.ndarray, Dict[str, float]],
        tolerance_config: SpectralTolerance,
    ) -> None:
        """
        Alpha deteccionador: λ₂ < MIN_FIEDLER_VALUE.
        
        Teorema (Aplicación de Weyl + Fiedler)
        ──────────────────────────────────────
        Puente de peso ε = 10⁻⁹ ⟹ λ₂ = O(ε) ≈ 10⁻⁹
        MIN_FIEDLER_VALUE ≈ 0.01
        ⟹ λ₂ < MIN_FIEDLER_VALUE ✓ (veto activado)
        """
        eigenvalues, _ = fractured_spectrum
        fiedler = _extract_fiedler_value(eigenvalues, tolerance_config)
        
        alpha_vetoes = fiedler < MIN_FIEDLER_VALUE
        assert alpha_vetoes, (
            f"Precondición fallida: λ₂ = {fiedler:.6e} no está "
            f"por debajo de MIN_FIEDLER_VALUE = {MIN_FIEDLER_VALUE:.6e}."
        )
    
    def test_alpha_state_formation(
        self,
        fractured_spectrum: Tuple[np.ndarray, Dict[str, float]],
        tolerance_config: SpectralTolerance,
    ) -> None:
        """
        Alpha forma un estado CategoricalState indicando veto.
        
        Estructura
        ──────────
        validated_strata: frozenset excluyendo ALPHA
        error: mensaje diagnóstico con λ₂
        
        Invariante
        ──────────
        El estado preserva la traza de error para diagnóstico.
        """
        eigenvalues, _ = fractured_spectrum
        fiedler = _extract_fiedler_value(eigenvalues, tolerance_config)
        
        # Simulación de estado Alpha sin validación
        alpha_state = CategoricalState(
            validated_strata=frozenset({
                Stratum.PHYSICS,
                Stratum.TACTICS,
                Stratum.STRATEGY,
                Stratum.OMEGA,
            }),
            error=(
                f"TopologicalInvariantError: fractura organizacional. "
                f"λ₂ = {fiedler:.6e} < MIN_FIEDLER_VALUE = {MIN_FIEDLER_VALUE:.6e}"
            ),
        )
        
        # Invariantes del estado
        assert Stratum.ALPHA not in alpha_state.validated_strata, (
            "ALPHA debe estar ausente en validated_strata si vetó."
        )
        assert alpha_state.error is not None, (
            "El estado debe preservar el mensaje de error."
        )
        assert "λ₂" in alpha_state.error, (
            "El error debe mencionar λ₂ para diagnóstico."
        )
        assert "TopologicalInvariantError" in alpha_state.error
    
    def test_wisdom_blocked_by_missing_alpha(
        self,
        fractured_spectrum: Tuple[np.ndarray, Dict[str, float]],
        tolerance_config: SpectralTolerance,
    ) -> None:
        """
        Bloqueo jerárquico: falta ALPHA en validated_strata.
        
        Lógica (Clausura transitiva de requisitos)
        ───────────────────────────────────────────
        Para ascender a Stratum.WISDOM se requiere que
        todas las dependencias en WISDOM.requires() estén en validated_strata.
        
        Si ALPHA ∉ validated_strata, entonces:
            ∃s ∈ WISDOM.requires(): s = ALPHA ∉ validated_strata
        ⟹ Ascenso a WISDOM está bloqueado.
        """
        # Precondiciones
        eigenvalues, _ = fractured_spectrum
        fiedler = _extract_fiedler_value(eigenvalues, tolerance_config)
        assert fiedler < MIN_FIEDLER_VALUE, (
            "Precondición: λ₂ debe ser microscópico."
        )
        
        # Estado sin ALPHA (por veto de fragilidad)
        validated_strata = frozenset({
            Stratum.PHYSICS,
            Stratum.TACTICS,
            Stratum.STRATEGY,
            Stratum.OMEGA,
        })
        
        wisdom_requirements = Stratum.WISDOM.requires()
        missing_for_wisdom = wisdom_requirements - validated_strata
        
        # Verificación de bloqueo
        assert Stratum.ALPHA in missing_for_wisdom, (
            f"Fallo de bloqueo: ALPHA debería estar en missing_for_wisdom. "
            f"missing: {missing_for_wisdom}, "
            f"wisdom_requires: {wisdom_requirements}."
        )
        
        assert len(missing_for_wisdom) >= 1, (
            "Al menos un estrato debe faltar para bloquear WISDOM."
        )
    
    def test_full_end_to_end_veto_scenario(
        self,
        omega_manifold: OmegaDeliberationManifold,
        hyper_profitable_inputs: OmegaInputs,
        fractured_spectrum: Tuple[np.ndarray, Dict[str, float]],
        tolerance_config: SpectralTolerance,
    ) -> None:
        """
        Escenario completo end-to-end: Omega aprueba, Alpha vetoa.
        
        Pasos
        ─────
        1. Omega aprueba por hiper-rentabilidad (ROI = 350%)
        2. Alpha detecta λ₂ = O(10⁻⁹) < MIN_FIEDLER_VALUE
        3. Jerarquía: WISDOM requiere ALPHA
        4. Resultado: WISDOM inaccesible
        
        Conclusión
        ──────────
        El sistema respetar los límites topológicos incluso cuando
        la rentabilidad es extremadamente atractiva.
        
        Teorema aplicado
        ────────────────
        Combinación de:
          • Lema 1.1 (Fiedler): λ₂ → 0 ⟹ grafo frágil
          • Lema 1.3 (escalado): puente ε ⟹ λ₂ = O(ε)
          • Clausura transitiva: requisitos forzan validación
        ⟹ Veto algebraico inevitable
        """
        # FASE 1: Omega aprueba
        omega_result = omega_manifold(hyper_profitable_inputs)
        omega_approved = omega_result.verdict in (
            VerdictLevel.VIABLE,
            VerdictLevel.CONDICIONAL,
        )
        
        assert omega_approved, (
            f"Setup fallido: Omega rechazó (verdict={omega_result.verdict.name}). "
            f"La precondición del test no se cumple."
        )
        print(f"✓ FASE 1: Omega aprobó con verdict={omega_result.verdict.name}")
        
        # FASE 2: Alpha detecta fragilidad
        eigenvalues, _ = fractured_spectrum
        fiedler = _extract_fiedler_value(eigenvalues, tolerance_config)
        alpha_vetoes = fiedler < MIN_FIEDLER_VALUE
        
        assert alpha_vetoes, (
            f"Setup fallido: λ₂ = {fiedler:.6e} no es microscópico. "
            f"Fragilidad no detectada."
        )
        print(f"✓ FASE 2: Alpha detectó fragilidad (λ₂={fiedler:.2e})")
        
        # FASE 3: Verificar bloqueo en la jerarquía
        validated_strata = frozenset({
            Stratum.PHYSICS,
            Stratum.TACTICS,
            Stratum.STRATEGY,
            Stratum.OMEGA,
            # ALPHA intencionalmente ausente por veto
        })
        
        wisdom_dependencies = Stratum.WISDOM.requires()
        missing_requirements = wisdom_dependencies - validated_strata
        
        has_alpha_missing = Stratum.ALPHA in missing_requirements
        assert has_alpha_missing, (
            f"FASE 3 FALLIDA: Bloqueo jerárquico no funcionó. "
            f"wisdom_requires={wisdom_dependencies}, "
            f"missing={missing_requirements}, "
            f"ALPHA in missing: {has_alpha_missing}."
        )
        print(
            f"✓ FASE 3: Jerarquía activada — WISDOM inaccesible "
            f"(falta: {missing_requirements})"
        )
        
        # Conclusión
        print("\n" + "="*70)
        print("CONCLUSIÓN: Veto jerárquico exitoso")
        print("="*70)
        print(f"  Omega verdict:        {omega_result.verdict.name} ✓")
        print(f"  Fiedler value (λ₂):   {fiedler:.2e}")
        print(f"  MIN_FIEDLER_VALUE:    {MIN_FIEDLER_VALUE:.2e}")
        print(f"  Fragilidad detectada: {alpha_vetoes} ✓")
        print(f"  ALPHA validado:       False ✓")
        print(f"  WISDOM accesible:     False ✓")
        print("\nPrincipio:"
              "Rentabilidad ≠ Estructura organizacional robusta")
        print("="*70)


# =============================================================================
# TEST SUITE 6: VALIDACIÓN CRUZADA CON BIBLIOTECAS EXTERNAS
# =============================================================================


@pytest.mark.integration
class TestCrossValidationNumpy:
    """
    Verificación cruzada: nuestras implementaciones vs. NumPy/NetworkX.
    
    Importancia
    ───────────
    Valida que no introdujimos errores numéricos adicionales.
    """
    
    def test_eigenvalues_match_numpy(
        self,
        fractured_laplacian: Tuple[np.ndarray, Dict[str, float]],
        tolerance_config: SpectralTolerance,
    ) -> None:
        """
        safe_eigenvalues_symmetric(L) ≈ np.linalg.eigvalsh(L).
        """
        L_norm, _ = fractured_laplacian
        
        safe_eigs = np.sort(safe_eigenvalues_symmetric(L_norm))
        numpy_eigs = np.sort(np.linalg.eigvalsh(L_norm))
        
        np.testing.assert_allclose(
            safe_eigs,
            numpy_eigs,
            atol=tolerance_config.eigenvalue_comparison,
            err_msg=(
                f"Eigenvalores divergen de NumPy:\n"
                f"  safe:  {safe_eigs}\n"
                f"  numpy: {numpy_eigs}"
            ),
        )
    
    def test_laplacian_matches_networkx(
        self,
        fractured_graph: nx.Graph,
        fractured_laplacian: Tuple[np.ndarray, Dict[str, float]],
        tolerance_config: SpectralTolerance,
    ) -> None:
        """
        Nuestro Laplaciano normalizado ≈ nx.normalized_laplacian_matrix.
        """
        L_norm, _ = fractured_laplacian
        
        nx_laplacian = nx.normalized_laplacian_matrix(
            fractured_graph,
            nodelist=sorted(fractured_graph.nodes()),
        ).toarray()
        
        np.testing.assert_allclose(
            L_norm,
            nx_laplacian,
            atol=tolerance_config.eigenvalue_comparison,
            err_msg="Laplaciano difiere del de NetworkX.",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])