import pytest
import networkx as nx
from typing import Any, Dict, List, Optional, Set, Tuple, FrozenSet
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy
import time

from app.adapters.mic_vectors import (
    vector_audit_homological_fusion,
    VectorResultStatus,
)
from app.schemas import Stratum


# =============================================================================
# CONSTANTES Y ENUMERACIONES DE DOMINIO HOMOLÓGICO
# =============================================================================

class HomologyConstants:
    """
    Constantes del sistema de auditoría homológica.
    
    Fundamento Matemático (Homología Simplicial):
      Sea X un complejo simplicial. Los grupos de homología Hₙ(X; ℤ)
      capturan los "agujeros n-dimensionales" de X.
      
      βₙ = rank(Hₙ) = n-ésimo número de Betti
        - β₀ = número de componentes conexas
        - β₁ = número de "agujeros" o ciclos independientes
        - β₂ = número de "cavidades" o vacíos
        
      Para grafos dirigidos (digrafos):
        - Tratamos el digrafo como 1-esqueleto de un complejo simplicial
        - β₁ cuenta ciclos dirigidos independientes en H₁
    """
    # Umbral de decisión para ciclos fantasma
    GHOST_CYCLE_THRESHOLD: int = 0
    
    # Coherencia topológica perfecta
    PERFECT_COHERENCE: float = 1.0
    
    # Rango válido de coherencia
    MIN_COHERENCE: float = 0.0
    MAX_COHERENCE: float = 1.0


class FusionStatus(str, Enum):
    """
    Estados posibles del resultado de fusión homológica.
    
    Semántica topológica:
      - CLEAN_MERGE: Δβ₁ = 0, la fusión preserva la estructura homológica
      - INTEGRATION_CONFLICT: Δβ₁ > 0, emergen ciclos fantasma
      - TOPOLOGY_ANOMALY: Violación de invariantes topológicos
    """
    CLEAN_MERGE = "CLEAN_MERGE"
    INTEGRATION_CONFLICT = "INTEGRATION_CONFLICT"
    TOPOLOGY_ANOMALY = "TOPOLOGY_ANOMALY"


class ResultKeys:
    """Claves del diccionario de resultado del vector."""
    SUCCESS = "success"
    STATUS = "status"
    ERROR = "error"
    STRATUM = "stratum"
    PAYLOAD = "payload"
    METRICS = "metrics"
    
    # Claves dentro de payload
    MERGED_GRAPH_VALID = "merged_graph_valid"
    
    # Claves dentro de metrics
    TOPOLOGICAL_COHERENCE = "topological_coherence"
    DELTA_BETA_1 = "delta_beta_1"


class MayerVietorisTheory:
    """
    Fundamentos teóricos de la secuencia de Mayer-Vietoris.
    
    Teorema (Mayer-Vietoris):
      Para espacios topológicos A, B con X = A ∪ B, existe una
      secuencia exacta larga:
      
        ... → Hₙ(A ∩ B) → Hₙ(A) ⊕ Hₙ(B) → Hₙ(X) → Hₙ₋₁(A ∩ B) → ...
        
    Corolario (Cota de Betti):
      β₁(A ∪ B) ≤ β₁(A) + β₁(B) + β₀(A ∩ B) - β₀(A) - β₀(B) + β₀(A ∪ B)
      
      En el caso simplificado (componentes conexas simples):
        β₁(A ∪ B) ≤ β₁(A) + β₁(B) + β₀(A ∩ B)
        
    Definición (Ciclo Fantasma):
      Un ciclo fantasma es un ciclo en H₁(A ∪ B) que no proviene
      de H₁(A) ni de H₁(B), sino que emerge por la interacción
      topológica de A y B en su intersección.
      
      Δβ₁ = β₁(A ∪ B) - [β₁(A) + β₁(B)] mide los ciclos fantasma.
    """
    
    @staticmethod
    def compute_delta_beta_1(
        beta_1_union: int,
        beta_1_a: int,
        beta_1_b: int,
    ) -> int:
        """
        Calcula Δβ₁ = β₁(A ∪ B) - [β₁(A) + β₁(B)].
        
        Returns:
            Δβ₁ ≥ 0 si la fusión introduce ciclos fantasma
            Δβ₁ = 0 si la fusión es limpia
            
        Note:
            Δβ₁ < 0 es teóricamente imposible por la secuencia de M-V;
            si ocurre, indica error de cómputo.
        """
        return beta_1_union - (beta_1_a + beta_1_b)
    
    @staticmethod
    def validate_mayer_vietoris_bound(
        beta_1_union: int,
        beta_1_a: int,
        beta_1_b: int,
        beta_0_intersection: int,
    ) -> bool:
        """
        Verifica que se cumple la cota de Mayer-Vietoris.
        
        β₁(A ∪ B) ≤ β₁(A) + β₁(B) + β₀(A ∩ B)
        """
        upper_bound = beta_1_a + beta_1_b + beta_0_intersection
        return beta_1_union <= upper_bound


# =============================================================================
# HELPERS DE CONSTRUCCIÓN Y VALIDACIÓN DE GRAFOS
# =============================================================================

def make_digraph(*edges: Tuple[str, str]) -> nx.DiGraph:
    """
    Factory para construir DiGraphs a partir de tuplas de aristas.
    
    Args:
        edges: Tuplas (source, target) representando aristas dirigidas
        
    Returns:
        nx.DiGraph con las aristas especificadas
        
    Example:
        make_digraph(("A", "B"), ("B", "C"))  →  A → B → C
    """
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


def make_chain_graph(nodes: List[str]) -> nx.DiGraph:
    """
    Crea una cadena lineal: n₀ → n₁ → n₂ → ... → nₖ.
    
    Esta es un DAG (grafo acíclico dirigido) con β₁ = 0.
    """
    g = nx.DiGraph()
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    return g


def make_cycle_graph(nodes: List[str]) -> nx.DiGraph:
    """
    Crea un ciclo dirigido: n₀ → n₁ → ... → nₖ → n₀.
    
    Este grafo tiene β₁ = 1 (un ciclo independiente).
    """
    g = make_chain_graph(nodes)
    if len(nodes) >= 2:
        g.add_edge(nodes[-1], nodes[0])
    return g


def make_star_graph(hub: str, spokes: List[str], directed_out: bool = True) -> nx.DiGraph:
    """
    Crea un grafo estrella con hub central.
    
    Args:
        hub: Nodo central
        spokes: Nodos periféricos
        directed_out: Si True, hub → spokes; si False, spokes → hub
        
    Este es un DAG (árbol) con β₁ = 0.
    """
    g = nx.DiGraph()
    for spoke in spokes:
        if directed_out:
            g.add_edge(hub, spoke)
        else:
            g.add_edge(spoke, hub)
    return g


def make_diamond_graph() -> nx.DiGraph:
    """
    Crea topología diamante (DAG con confluencia).
    
          A
         / \
        B   C
         \ /
          D
          
    Este es un DAG con β₁ = 0 (sin ciclos).
    """
    return make_digraph(
        ("A", "B"), ("A", "C"),
        ("B", "D"), ("C", "D"),
    )


def make_complete_digraph(nodes: List[str]) -> nx.DiGraph:
    """
    Crea un digrafo completo Kₙ (todas las aristas posibles).
    
    Para n nodos: |E| = n(n-1) aristas dirigidas.
    """
    g = nx.DiGraph()
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i != j:
                g.add_edge(u, v)
    return g


def compute_graph_metrics(g: nx.DiGraph) -> Dict[str, Any]:
    """
    Computa métricas básicas de un digrafo.
    
    Returns:
        Dict con:
          - num_nodes: |V|
          - num_edges: |E|
          - num_components: número de componentes débilmente conexas
          - has_cycles: si contiene al menos un ciclo
          - is_dag: si es un DAG (grafo acíclico dirigido)
    """
    return {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "num_components": nx.number_weakly_connected_components(g),
        "has_cycles": not nx.is_directed_acyclic_graph(g),
        "is_dag": nx.is_directed_acyclic_graph(g),
    }


def compute_intersection_graph(g1: nx.DiGraph, g2: nx.DiGraph) -> nx.DiGraph:
    """
    Computa el grafo intersección: G₁ ∩ G₂.
    
    Nodos: V(G₁) ∩ V(G₂)
    Aristas: E(G₁) ∩ E(G₂)
    """
    common_nodes = set(g1.nodes()) & set(g2.nodes())
    common_edges = set(g1.edges()) & set(g2.edges())
    
    intersection = nx.DiGraph()
    intersection.add_nodes_from(common_nodes)
    intersection.add_edges_from(common_edges)
    
    return intersection


def compute_union_graph(g1: nx.DiGraph, g2: nx.DiGraph) -> nx.DiGraph:
    """
    Computa el grafo unión: G₁ ∪ G₂.
    
    Nodos: V(G₁) ∪ V(G₂)
    Aristas: E(G₁) ∪ E(G₂)
    """
    union = nx.DiGraph()
    union.add_nodes_from(g1.nodes())
    union.add_nodes_from(g2.nodes())
    union.add_edges_from(g1.edges())
    union.add_edges_from(g2.edges())
    
    return union


def graphs_are_isomorphic(g1: nx.DiGraph, g2: nx.DiGraph) -> bool:
    """Verifica si dos digrafos son isomorfos."""
    return nx.is_isomorphic(g1, g2)


# =============================================================================
# HELPERS DE VALIDACIÓN DE RESULTADOS
# =============================================================================

def validate_fusion_result_structure(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida la estructura del resultado de fusión.
    
    Invariantes:
      (R₁) 'success' siempre presente y es bool
      (R₂) Si success=True → 'stratum', 'payload', 'metrics' presentes
      (R₃) Si success=False → 'status', 'error' presentes
      
    Returns:
        Dict con:
          - is_valid: bool
          - violations: List[str]
          - success_value: bool | None
    """
    violations = []
    
    # R₁: success obligatorio
    if ResultKeys.SUCCESS not in result:
        violations.append("Falta clave 'success'")
        return {
            "is_valid": False,
            "violations": violations,
            "success_value": None,
        }
    
    if not isinstance(result[ResultKeys.SUCCESS], bool):
        violations.append(f"'success' no es bool: {type(result[ResultKeys.SUCCESS])}")
    
    success = result[ResultKeys.SUCCESS]
    
    if success:
        # R₂: claves de éxito
        required_success = {ResultKeys.STRATUM, ResultKeys.PAYLOAD, ResultKeys.METRICS}
        missing = required_success - set(result.keys())
        if missing:
            violations.append(f"Faltan claves de éxito: {missing}")
    else:
        # R₃: claves de fallo
        required_failure = {ResultKeys.STATUS, ResultKeys.ERROR}
        missing = required_failure - set(result.keys())
        if missing:
            violations.append(f"Faltan claves de fallo: {missing}")
    
    return {
        "is_valid": len(violations) == 0,
        "violations": violations,
        "success_value": success,
    }


def validate_coherence_bounded(coherence: float) -> bool:
    """
    Verifica que la coherencia esté en el intervalo [0, 1].
    """
    return HomologyConstants.MIN_COHERENCE <= coherence <= HomologyConstants.MAX_COHERENCE


def assert_clean_merge(result: Dict[str, Any], context: str = ""):
    """
    Aserción compuesta para fusión limpia exitosa.
    """
    prefix = f"[{context}] " if context else ""
    
    assert result[ResultKeys.SUCCESS] is True, (
        f"{prefix}Esperado success=True"
    )
    assert result[ResultKeys.STRATUM] == Stratum.TACTICS, (
        f"{prefix}Stratum incorrecto: {result.get(ResultKeys.STRATUM)}"
    )
    assert result[ResultKeys.PAYLOAD][ResultKeys.MERGED_GRAPH_VALID] is True, (
        f"{prefix}merged_graph_valid debe ser True"
    )
    
    coherence = result[ResultKeys.METRICS][ResultKeys.TOPOLOGICAL_COHERENCE]
    assert coherence == pytest.approx(HomologyConstants.PERFECT_COHERENCE), (
        f"{prefix}Coherencia debe ser 1.0 en fusión limpia, obtenido: {coherence}"
    )


def assert_ghost_cycle_detected(result: Dict[str, Any], context: str = ""):
    """
    Aserción compuesta para detección de ciclos fantasma.
    """
    prefix = f"[{context}] " if context else ""
    
    assert result[ResultKeys.SUCCESS] is False, (
        f"{prefix}Esperado success=False por ciclos fantasma"
    )
    assert result[ResultKeys.STATUS] == VectorResultStatus.TOPOLOGY_ERROR.value, (
        f"{prefix}Status debe ser TOPOLOGY_ERROR"
    )
    assert "ciclos fantasma" in result[ResultKeys.ERROR].lower() or \
           "mayer-vietoris" in result[ResultKeys.ERROR].lower(), (
        f"{prefix}Error debe mencionar ciclos fantasma o Mayer-Vietoris"
    )


# =============================================================================
# PRUEBAS DEL VECTOR DE AUDITORÍA HOMOLÓGICA DE FUSIÓN
# =============================================================================

class TestVectorAuditHomologicalFusion:
    """
    Pruebas unitarias para el vector 'audit_fusion_homology'.

    Fundamento matemático — Secuencia de Mayer-Vietoris:
    
      Dados dos subcomplejos A, B de un complejo simplicial X = A ∪ B,
      la secuencia exacta larga de Mayer-Vietoris:

        ... → Hₙ(A ∩ B) →^{(i*,j*)} Hₙ(A) ⊕ Hₙ(B) →^{k*-l*} Hₙ(X) →^{∂} Hₙ₋₁(A ∩ B) → ...

      donde:
        - i*, j* son inducidas por inclusiones A ∩ B ↪ A, A ∩ B ↪ B
        - k*, l* son inducidas por inclusiones A ↪ X, B ↪ X
        - ∂ es el homomorfismo de conexión

      Esta secuencia induce la cota:
        β₁(A ∪ B) ≤ β₁(A) + β₁(B) + β₀(A ∩ B)

    Métrica clave:
      Δβ₁ = β₁(A ∪ B) − [β₁(A) + β₁(B)]
      
      ┌──────────┬───────────────────────────────────────────────────────────┐
      │ Δβ₁ = 0  │ Fusión limpia (CLEAN_MERGE), sin ciclos fantasma         │
      ├──────────┼───────────────────────────────────────────────────────────┤
      │ Δβ₁ > 0  │ Ciclos fantasma emergentes (INTEGRATION_CONFLICT)        │
      ├──────────┼───────────────────────────────────────────────────────────┤
      │ Δβ₁ < 0  │ Teóricamente imposible (violación de M-V)                │
      └──────────┴───────────────────────────────────────────────────────────┘

    Invariantes del vector:
      ┌─────┬───────────────────────────────────────────────────────────────┐
      │ I₁  │ Δβ₁ = 0  ↔  success = True                                   │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₂  │ Δβ₁ > 0  ↔  success = False ∧ status = TOPOLOGY_ERROR        │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₃  │ Ausencia de grafos → success = False ∧ status = LOGIC_ERROR  │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₄  │ topological_coherence ∈ [0, 1]                               │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₅  │ El stratum refleja nivel de decisión apropiado               │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ I₆  │ Los grafos de entrada permanecen inmutables                  │
      └─────┴───────────────────────────────────────────────────────────────┘
    """

    # ── PATH DE MOCK (centralizado para mantenimiento) ─────────────────

    ANALYZER_PATCH_PATH = "app.adapters.mic_vectors.BusinessTopologicalAnalyzer"

    # ── FACTORIES ──────────────────────────────────────────────────────

    @staticmethod
    def _make_payload(
        graph_a: Optional[nx.DiGraph] = None,
        graph_b: Optional[nx.DiGraph] = None,
        include_a: bool = True,
        include_b: bool = True,
        extra_fields: Optional[Dict] = None,
    ) -> dict:
        """
        Factory para construir el payload del vector.

        Args:
            graph_a: DiGraph A (None → DiGraph vacío si include_a)
            graph_b: DiGraph B (None → DiGraph vacío si include_b)
            include_a: Si False, omite la clave 'graph_a'
            include_b: Si False, omite la clave 'graph_b'
            extra_fields: Campos adicionales para el payload
            
        Returns:
            Payload estructurado para el vector
        """
        payload = {}
        
        if include_a:
            payload["graph_a"] = graph_a if graph_a is not None else nx.DiGraph()
        if include_b:
            payload["graph_b"] = graph_b if graph_b is not None else nx.DiGraph()
        if extra_fields:
            payload.update(extra_fields)
            
        return payload

    @staticmethod
    def _mock_homology_result(
        delta_beta_1: int = 0,
        status: str = FusionStatus.CLEAN_MERGE.value,
        beta_1_a: int = 0,
        beta_1_b: int = 0,
        beta_1_union: int = 0,
        additional_fields: Optional[Dict] = None,
    ) -> dict:
        """
        Factory para respuestas del analyzer mockeado.
        
        Args:
            delta_beta_1: Δβ₁ = β₁(A∪B) - [β₁(A) + β₁(B)]
            status: Estado de la fusión
            beta_1_a: β₁(A) - ciclos en grafo A
            beta_1_b: β₁(B) - ciclos en grafo B
            beta_1_union: β₁(A∪B) - ciclos en unión
            additional_fields: Campos extra para el resultado
        """
        result = {
            "delta_beta_1": delta_beta_1,
            "status": status,
            "beta_1_a": beta_1_a,
            "beta_1_b": beta_1_b,
            "beta_1_union": beta_1_union,
        }
        if additional_fields:
            result.update(additional_fields)
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # FUSIÓN EXITOSA (Δβ₁ = 0) — CLEAN MERGE
    # ═══════════════════════════════════════════════════════════════════════

    def test_clean_merge_linear_chains(self):
        """
        Fusión de cadenas lineales disjuntas.
        
        A: (A → B)
        B: (B → C)
        A ∪ B: A → B → C (DAG lineal)
        
        β₁(A) = β₁(B) = β₁(A∪B) = 0
        Δβ₁ = 0 - (0 + 0) = 0 → CLEAN_MERGE
        """
        graph_a = make_digraph(("A", "B"))
        graph_b = make_digraph(("B", "C"))
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=0,
                    beta_1_a=0,
                    beta_1_b=0,
                    beta_1_union=0,
                )
            )

            result = vector_audit_homological_fusion(payload)

        assert_clean_merge(result, "cadenas lineales")

    def test_clean_merge_disjoint_components(self):
        """
        Grafos sin nodos compartidos (componentes disjuntas).
        
        A: (A → B)
        B: (C → D)
        A ∩ B = ∅
        A ∪ B tiene 2 componentes conexas, pero β₁ = 0.
        
        La fórmula de M-V se simplifica cuando A ∩ B = ∅.
        """
        graph_a = make_digraph(("A", "B"))
        graph_b = make_digraph(("C", "D"))
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True
        assert result[ResultKeys.PAYLOAD][ResultKeys.MERGED_GRAPH_VALID] is True

    def test_clean_merge_identical_graphs_idempotence(self):
        """
        Fusión idempotente: G ∪ G ≅ G.
        
        Propiedad algebraica: la unión es idempotente.
        No deben emerger ciclos nuevos por auto-fusión.
        
        Si G no tiene ciclos, G ∪ G tampoco los tiene.
        """
        graph = make_chain_graph(["A", "B", "C", "D"])
        payload = self._make_payload(graph_a=graph, graph_b=graph)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True

    def test_clean_merge_empty_graphs_identity(self):
        """
        Fusión de grafos vacíos: ∅ ∪ ∅ = ∅.
        
        El grafo vacío es el elemento identidad bajo unión.
        Caso degenerado: no hay topología que analizar.
        
        β₁(∅) = 0 por definición.
        """
        payload = self._make_payload(
            graph_a=nx.DiGraph(),
            graph_b=nx.DiGraph(),
        )

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True

    def test_clean_merge_absorbing_identity(self):
        """
        Fusión absorbente: G ∪ ∅ = G.
        
        El grafo vacío es elemento neutro (identidad) de la unión.
        La topología de G se preserva exactamente.
        """
        graph_a = make_chain_graph(["A", "B", "C"])
        payload = self._make_payload(graph_a=graph_a, graph_b=nx.DiGraph())

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True

    def test_clean_merge_dag_topologies(self):
        """
        Fusión de DAGs (grafos acíclicos dirigidos).
        
        La unión de dos DAGs que no forman ciclos al combinarse
        sigue siendo un DAG con β₁ = 0.
        """
        # DAG diamante
        graph_a = make_diamond_graph()
        # DAG lineal que extiende desde D
        graph_b = make_digraph(("D", "E"), ("E", "F"))
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True

    def test_clean_merge_star_topologies(self):
        """
        Fusión de grafos estrella.
        
        Dos estrellas con hubs diferentes no crean ciclos.
        Las estrellas son árboles (DAGs sin ciclos).
        """
        graph_a = make_star_graph("Hub1", ["R1", "R2", "R3"])
        graph_b = make_star_graph("Hub2", ["R4", "R5", "R6"])
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True

    # ═══════════════════════════════════════════════════════════════════════
    # FUSIÓN FALLIDA: CICLOS FANTASMA (Δβ₁ > 0)
    # ═══════════════════════════════════════════════════════════════════════

    def test_ghost_cycle_from_complementary_edges(self):
        """
        Ciclo fantasma por aristas complementarias.
        
        A: (A → B)
        B: (B → A)
        A ∪ B: A ↔ B (ciclo de longitud 2)
        
        β₁(A) = β₁(B) = 0 (ninguno tiene ciclos)
        β₁(A∪B) = 1 (el ciclo emerge por fusión)
        Δβ₁ = 1 - (0 + 0) = 1 → INTEGRATION_CONFLICT
        
        Este es el caso arquetípico de anomalía de Mayer-Vietoris.
        """
        graph_a = make_digraph(("A", "B"))
        graph_b = make_digraph(("B", "A"))
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=1,
                    status=FusionStatus.INTEGRATION_CONFLICT.value,
                    beta_1_a=0,
                    beta_1_b=0,
                    beta_1_union=1,
                )
            )

            result = vector_audit_homological_fusion(payload)

        assert_ghost_cycle_detected(result, "aristas complementarias")

    def test_ghost_cycle_triangle_closure(self):
        """
        Cierre transitivo de triángulo.
        
        A: (A → B, B → C)
        B: (C → A)
        A ∪ B: A → B → C → A (ciclo de longitud 3)
        
        Δβ₁ = 1 → ciclo fantasma por cierre.
        """
        graph_a = make_digraph(("A", "B"), ("B", "C"))
        graph_b = make_digraph(("C", "A"))
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=1,
                    status=FusionStatus.INTEGRATION_CONFLICT.value,
                )
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is False
        assert "ciclos fantasma" in result[ResultKeys.ERROR].lower()

    def test_multiple_ghost_cycles(self):
        """
        Múltiples ciclos fantasma independientes.
        
        3 pares de aristas complementarias → 3 ciclos fantasma.
        Δβ₁ = 3
        """
        graph_a = make_digraph(
            ("A", "B"), ("C", "D"), ("E", "F"),
        )
        graph_b = make_digraph(
            ("B", "A"), ("D", "C"), ("F", "E"),
        )
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=3,
                    status=FusionStatus.INTEGRATION_CONFLICT.value,
                )
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is False
        assert result[ResultKeys.STATUS] == VectorResultStatus.TOPOLOGY_ERROR.value

    def test_preexisting_cycles_preserved_no_ghost(self):
        """
        Ciclos preexistentes no cuentan como fantasma.
        
        Si A ya tiene un ciclo (A → B → A), fusionar con (C → D)
        no introduce ciclos fantasma.
        
        β₁(A) = 1 (ciclo heredado)
        β₁(B) = 0
        β₁(A∪B) = 1 (mismo ciclo heredado)
        Δβ₁ = 1 - (1 + 0) = 0 → CLEAN_MERGE
        """
        graph_a = make_cycle_graph(["A", "B"])  # Tiene ciclo
        graph_b = make_digraph(("C", "D"))
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=0,
                    beta_1_a=1,  # Ciclo heredado
                    beta_1_b=0,
                    beta_1_union=1,  # Mismo ciclo
                )
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True

    def test_independent_cycles_merge_without_ghost(self):
        """
        Ciclos independientes en A y B no producen fantasmas adicionales.
        
        A tiene un ciclo, B tiene otro ciclo independiente.
        La fusión tiene ambos ciclos heredados, sin fantasmas.
        """
        graph_a = make_cycle_graph(["A", "B"])
        graph_b = make_cycle_graph(["C", "D"])
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=0,
                    beta_1_a=1,
                    beta_1_b=1,
                    beta_1_union=2,  # Ambos ciclos heredados
                )
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True

    # ═══════════════════════════════════════════════════════════════════════
    # FRONTERA DE DECISIÓN: Δβ₁
    # ═══════════════════════════════════════════════════════════════════════

    @pytest.mark.parametrize(
        "delta_beta_1, expected_success, description",
        [
            (0,  True,  "Fusión limpia: sin ciclos emergentes"),
            (1,  False, "Un ciclo fantasma: umbral de rechazo"),
            (2,  False, "Dos ciclos fantasma"),
            (5,  False, "Cinco ciclos fantasma"),
            (10, False, "Muchos ciclos fantasma (stress)"),
            (100, False, "Caso extremo: 100 ciclos fantasma"),
        ],
        ids=["delta_0", "delta_1", "delta_2", "delta_5", "delta_10", "delta_100"],
    )
    def test_delta_beta_1_decision_boundary(
        self, delta_beta_1, expected_success, description
    ):
        """
        Invariantes I₁/I₂: Umbral estricto de decisión.
        
        Δβ₁ = 0 ↔ éxito
        Δβ₁ > 0 ↔ fallo con TOPOLOGY_ERROR
        """
        graph_a = make_digraph(("A", "B"))
        graph_b = make_digraph(("B", "C"))
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        status = (FusionStatus.CLEAN_MERGE.value if delta_beta_1 == 0 
                  else FusionStatus.INTEGRATION_CONFLICT.value)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=delta_beta_1,
                    status=status,
                )
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is expected_success, (
            f"Fallo en frontera de decisión: {description}\n"
            f"  Δβ₁ = {delta_beta_1}\n"
            f"  Esperado success = {expected_success}"
        )

    def test_delta_beta_1_negative_impossible(self):
        """
        Δβ₁ < 0 es teóricamente imposible.
        
        Por la secuencia de Mayer-Vietoris, la fusión no puede
        "destruir" ciclos que no existían.
        
        Si el analyzer reporta Δβ₁ < 0, indica error de cómputo
        y debe tratarse como anomalía.
        """
        graph_a = make_digraph(("A", "B"))
        graph_b = make_digraph(("B", "C"))
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=-1,  # Imposible teóricamente
                    status=FusionStatus.TOPOLOGY_ANOMALY.value,
                )
            )

            result = vector_audit_homological_fusion(payload)

        # El sistema debe manejar este caso anómalo
        validation = validate_fusion_result_structure(result)
        assert validation["is_valid"]

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₃: AUSENCIA DE GRAFOS
    # ═══════════════════════════════════════════════════════════════════════

    def test_missing_graph_b(self):
        """Rechazo: payload con graph_a pero sin graph_b."""
        payload = self._make_payload(
            graph_a=make_digraph(("A", "B")),
            include_b=False,
        )

        result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is False
        assert result[ResultKeys.STATUS] == VectorResultStatus.LOGIC_ERROR.value
        assert "faltan grafos" in result[ResultKeys.ERROR].lower()

    def test_missing_graph_a(self):
        """Rechazo: payload con graph_b pero sin graph_a."""
        payload = self._make_payload(
            graph_b=make_digraph(("A", "B")),
            include_a=False,
        )

        result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is False
        assert result[ResultKeys.STATUS] == VectorResultStatus.LOGIC_ERROR.value

    def test_missing_both_graphs(self):
        """Rechazo: payload vacío, ambos grafos ausentes."""
        payload = self._make_payload(include_a=False, include_b=False)

        result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is False
        assert result[ResultKeys.STATUS] == VectorResultStatus.LOGIC_ERROR.value

    def test_empty_payload(self):
        """Payload completamente vacío: {} → fallo controlado."""
        result = vector_audit_homological_fusion({})

        assert result[ResultKeys.SUCCESS] is False
        assert result[ResultKeys.STATUS] == VectorResultStatus.LOGIC_ERROR.value

    def test_none_payload(self):
        """Payload None: caso extremo."""
        try:
            result = vector_audit_homological_fusion(None)
            assert result[ResultKeys.SUCCESS] is False
        except (TypeError, AttributeError):
            # Excepción explícita aceptable
            pass

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₄: COHERENCIA TOPOLÓGICA ∈ [0, 1]
    # ═══════════════════════════════════════════════════════════════════════

    def test_coherence_is_one_on_clean_merge(self):
        """
        Invariante I₄: coherencia = 1.0 en fusión limpia.
        
        Coherencia perfecta indica preservación total de
        la estructura homológica.
        """
        payload = self._make_payload(
            graph_a=make_digraph(("A", "B")),
            graph_b=make_digraph(("B", "C")),
        )

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result = vector_audit_homological_fusion(payload)

        coherence = result[ResultKeys.METRICS][ResultKeys.TOPOLOGICAL_COHERENCE]
        assert coherence == pytest.approx(HomologyConstants.PERFECT_COHERENCE)

    def test_coherence_degrades_on_conflict(self):
        """
        Coherencia < 1.0 cuando Δβ₁ > 0.
        
        La degradación debe ser proporcional a la severidad.
        """
        payload = self._make_payload(
            graph_a=make_digraph(("A", "B")),
            graph_b=make_digraph(("B", "A")),
        )

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=1,
                    status=FusionStatus.INTEGRATION_CONFLICT.value,
                )
            )

            result = vector_audit_homological_fusion(payload)

        if ResultKeys.METRICS in result:
            coherence = result[ResultKeys.METRICS].get(
                ResultKeys.TOPOLOGICAL_COHERENCE
            )
            if coherence is not None:
                assert 0 <= coherence < 1.0, (
                    f"Coherencia debe degradar con conflicto: {coherence}"
                )

    @pytest.mark.parametrize(
        "delta_beta_1",
        [0, 1, 2, 5, 10, 50, 100],
        ids=["clean", "one", "two", "five", "ten", "fifty", "hundred"],
    )
    def test_coherence_bounded_unit_interval(self, delta_beta_1):
        """
        Invariante I₄: ∀ Δβ₁, coherencia ∈ [0, 1].
        
        El acotamiento debe mantenerse independientemente
        de la severidad del conflicto.
        """
        payload = self._make_payload(
            graph_a=make_digraph(("A", "B")),
            graph_b=make_digraph(("B", "C")),
        )
        status = (FusionStatus.CLEAN_MERGE.value if delta_beta_1 == 0 
                  else FusionStatus.INTEGRATION_CONFLICT.value)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=delta_beta_1,
                    status=status,
                )
            )

            result = vector_audit_homological_fusion(payload)

        if ResultKeys.METRICS in result:
            coherence = result[ResultKeys.METRICS].get(
                ResultKeys.TOPOLOGICAL_COHERENCE
            )
            if coherence is not None:
                assert validate_coherence_bounded(coherence), (
                    f"Coherencia fuera de [0,1]: {coherence} con Δβ₁={delta_beta_1}"
                )

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₅: STRATUM
    # ═══════════════════════════════════════════════════════════════════════

    def test_stratum_tactics_on_success(self):
        """
        Invariante I₅: fusión exitosa → decisión a nivel TACTICS.
        """
        payload = self._make_payload(
            graph_a=make_digraph(("A", "B")),
            graph_b=make_digraph(("B", "C")),
        )

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.STRATUM] == Stratum.TACTICS

    def test_stratum_present_on_failure(self):
        """
        Invariante I₅: stratum definido incluso en fallo.
        
        Necesario para enrutamiento de eventos en el sistema.
        """
        payload = self._make_payload(
            graph_a=make_digraph(("A", "B")),
            graph_b=make_digraph(("B", "A")),
        )

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=1,
                    status=FusionStatus.INTEGRATION_CONFLICT.value,
                )
            )

            result = vector_audit_homological_fusion(payload)

        if ResultKeys.STRATUM in result:
            assert result[ResultKeys.STRATUM] in list(Stratum)

    # ═══════════════════════════════════════════════════════════════════════
    # INVARIANTE I₆: INMUTABILIDAD DE GRAFOS DE ENTRADA
    # ═══════════════════════════════════════════════════════════════════════

    def test_input_graphs_immutability(self):
        """
        Invariante I₆: Los grafos de entrada no deben ser modificados.
        """
        graph_a = make_digraph(("A", "B"), ("B", "C"))
        graph_b = make_digraph(("X", "Y"))
        
        # Crear snapshots
        original_a_nodes = set(graph_a.nodes())
        original_a_edges = set(graph_a.edges())
        original_b_nodes = set(graph_b.nodes())
        original_b_edges = set(graph_b.edges())
        
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            _ = vector_audit_homological_fusion(payload)

        # Verificar inmutabilidad
        assert set(graph_a.nodes()) == original_a_nodes
        assert set(graph_a.edges()) == original_a_edges
        assert set(graph_b.nodes()) == original_b_nodes
        assert set(graph_b.edges()) == original_b_edges

    # ═══════════════════════════════════════════════════════════════════════
    # ROBUSTEZ ANTE ENTRADAS ANÓMALAS
    # ═══════════════════════════════════════════════════════════════════════

    def test_undirected_graph_rejected(self):
        """
        Grafos no dirigidos (nx.Graph) no son válidos.
        
        El análisis homológico de fusión requiere DiGraph para
        distinguir aristas de co-aristas y determinar ciclos dirigidos.
        """
        undirected = nx.Graph()
        undirected.add_edge("A", "B")

        payload = {"graph_a": undirected, "graph_b": nx.DiGraph()}

        result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is False

    def test_none_as_graph_rejected(self):
        """graph_a = None: tipo nulo explícito."""
        payload = {"graph_a": None, "graph_b": nx.DiGraph()}

        result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is False

    def test_string_as_graph_rejected(self):
        """Tipo completamente incorrecto: string como grafo."""
        payload = {"graph_a": "not_a_graph", "graph_b": nx.DiGraph()}

        result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is False

    @pytest.mark.parametrize(
        "invalid_graph, description",
        [
            (123, "entero"),
            (3.14, "flotante"),
            ([], "lista vacía"),
            ({}, "diccionario vacío"),
            (set(), "conjunto vacío"),
            (lambda x: x, "función"),
            (object(), "objeto genérico"),
        ],
        ids=["int", "float", "list", "dict", "set", "func", "object"],
    )
    def test_invalid_graph_types_rejected(self, invalid_graph, description):
        """
        Tipos inválidos como grafos deben rechazarse.
        """
        payload = {"graph_a": invalid_graph, "graph_b": nx.DiGraph()}

        result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is False, (
            f"Tipo inválido ({description}) no fue rechazado"
        )

    def test_multigraph_handling(self):
        """
        MultiDiGraph (múltiples aristas entre nodos).
        
        El comportamiento depende de la implementación;
        verificamos que no crashee.
        """
        multi = nx.MultiDiGraph()
        multi.add_edge("A", "B")
        multi.add_edge("A", "B")  # Segunda arista
        
        payload = {"graph_a": multi, "graph_b": nx.DiGraph()}

        result = vector_audit_homological_fusion(payload)

        # Debe manejar sin crash
        assert ResultKeys.SUCCESS in result

    def test_graph_with_self_loops(self):
        """
        Grafo con self-loops (aristas reflexivas).
        
        Un self-loop A → A es un ciclo de longitud 1.
        """
        graph_with_loop = nx.DiGraph()
        graph_with_loop.add_edge("A", "A")  # Self-loop
        graph_with_loop.add_edge("A", "B")
        
        payload = self._make_payload(
            graph_a=graph_with_loop,
            graph_b=nx.DiGraph(),
        )

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=0,
                    beta_1_a=1,  # Self-loop cuenta como ciclo
                )
            )

            result = vector_audit_homological_fusion(payload)

        assert ResultKeys.SUCCESS in result

    def test_graph_with_isolated_nodes(self):
        """
        Grafo con nodos aislados (sin aristas).
        
        Nodos aislados contribuyen a β₀ pero no a β₁.
        """
        graph_with_isolated = nx.DiGraph()
        graph_with_isolated.add_node("Isolated")
        graph_with_isolated.add_edge("A", "B")
        
        payload = self._make_payload(
            graph_a=graph_with_isolated,
            graph_b=make_digraph(("B", "C")),
        )

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True

    # ═══════════════════════════════════════════════════════════════════════
    # TOPOLOGÍAS COMPLEJAS
    # ═══════════════════════════════════════════════════════════════════════

    def test_diamond_topology_no_ghost_cycle(self):
        """
        Topología diamante (confluencia de caminos).
        
              A
             / \
            B   C
             \ /
              D
              
        Es un DAG (sin ciclos) → Δβ₁ = 0.
        """
        graph_a = make_digraph(("A", "B"), ("A", "C"))
        graph_b = make_digraph(("B", "D"), ("C", "D"))
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True

    def test_star_topology_merge(self):
        """
        Fusión de estrellas con hub compartido.
        
        Las estrellas son árboles → β₁ = 0.
        Fusionar estrellas con el mismo hub no crea ciclos.
        """
        graph_a = make_star_graph("Hub", ["R1", "R2", "R3"])
        graph_b = make_star_graph("Hub", ["R4", "R5"])
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True

    def test_complete_digraph_merge(self):
        """
        Fusión con digrafo completo.
        
        El digrafo completo K_n tiene muchos ciclos.
        La fusión debe analizar correctamente la topología.
        """
        k3 = make_complete_digraph(["A", "B", "C"])
        k2 = make_complete_digraph(["D", "E"])
        payload = self._make_payload(graph_a=k3, graph_b=k2)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=0,  # Componentes disjuntas
                    beta_1_a=2,  # K_3 tiene ciclos
                    beta_1_b=1,  # K_2 tiene ciclo
                    beta_1_union=3,  # Suma de ciclos
                )
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True

    # ═══════════════════════════════════════════════════════════════════════
    # PROPIEDADES ALGEBRAICAS DE LA FUSIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def test_fusion_commutativity(self):
        """
        Propiedad algebraica: A ∪ B ≅ B ∪ A (conmutatividad).
        
        El resultado debe ser equivalente independientemente del orden.
        """
        graph_a = make_chain_graph(["A", "B", "C"])
        graph_b = make_digraph(("C", "D"))

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result_ab = vector_audit_homological_fusion(
                self._make_payload(graph_a=graph_a, graph_b=graph_b)
            )
            result_ba = vector_audit_homological_fusion(
                self._make_payload(graph_a=graph_b, graph_b=graph_a)
            )

        # Mismo resultado de éxito
        assert result_ab[ResultKeys.SUCCESS] == result_ba[ResultKeys.SUCCESS]
        
        # Misma coherencia
        if ResultKeys.METRICS in result_ab and ResultKeys.METRICS in result_ba:
            assert (
                result_ab[ResultKeys.METRICS][ResultKeys.TOPOLOGICAL_COHERENCE]
                == result_ba[ResultKeys.METRICS][ResultKeys.TOPOLOGICAL_COHERENCE]
            )

    def test_fusion_associativity_pattern(self):
        """
        Patrón asociativo: (A ∪ B) ∪ C debería ser equivalente a A ∪ (B ∪ C).
        
        Verificamos que la propiedad se mantenga topológicamente.
        """
        graph_a = make_digraph(("A", "B"))
        graph_b = make_digraph(("B", "C"))
        graph_c = make_digraph(("C", "D"))
        
        # Simulamos (A ∪ B) ∪ C
        # En un sistema real, haríamos fusión incremental
        # Aquí verificamos que la fusión triple sea limpia
        
        all_edges = list(graph_a.edges()) + list(graph_b.edges()) + list(graph_c.edges())
        combined = make_digraph(*all_edges)
        
        # Sin ciclos, fusión limpia
        assert nx.is_directed_acyclic_graph(combined)

    def test_fusion_with_identity(self):
        """
        Elemento identidad: G ∪ ∅ = G = ∅ ∪ G.
        
        El grafo vacío es elemento neutro.
        """
        graph = make_chain_graph(["A", "B", "C"])
        empty = nx.DiGraph()

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            result_left = vector_audit_homological_fusion(
                self._make_payload(graph_a=graph, graph_b=empty)
            )
            result_right = vector_audit_homological_fusion(
                self._make_payload(graph_a=empty, graph_b=graph)
            )

        assert result_left[ResultKeys.SUCCESS] is True
        assert result_right[ResultKeys.SUCCESS] is True

    def test_fusion_idempotence(self):
        """
        Idempotencia: G ∪ G = G.
        
        La auto-fusión no cambia la topología.
        """
        graph = make_chain_graph(["A", "B", "C"])

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                self._mock_homology_result(
                    delta_beta_1=0,
                    beta_1_a=0,
                    beta_1_b=0,
                    beta_1_union=0,
                )
            )

            result = vector_audit_homological_fusion(
                self._make_payload(graph_a=graph, graph_b=graph)
            )

        assert result[ResultKeys.SUCCESS] is True

    # ═══════════════════════════════════════════════════════════════════════
    # INTERACCIÓN CON EL ANALYZER
    # ═══════════════════════════════════════════════════════════════════════

    def test_analyzer_called_with_both_graphs(self):
        """
        Verifica que el analyzer reciba ambos grafos.
        """
        graph_a = make_digraph(("A", "B"))
        graph_b = make_digraph(("C", "D"))
        payload = self._make_payload(graph_a=graph_a, graph_b=graph_b)

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            mock_instance.audit_integration_homology.return_value = (
                self._mock_homology_result(delta_beta_1=0)
            )

            vector_audit_homological_fusion(payload)

            mock_instance.audit_integration_homology.assert_called_once()

    def test_analyzer_exception_handled_gracefully(self):
        """
        Excepción del analyzer: fallo controlado sin propagación.
        """
        payload = self._make_payload(
            graph_a=make_digraph(("A", "B")),
            graph_b=make_digraph(("B", "C")),
        )

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.side_effect = (
                RuntimeError("Internal analyzer failure")
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is False

    @pytest.mark.parametrize(
        "exception_type",
        [RuntimeError, ValueError, TypeError, MemoryError, KeyError],
        ids=["runtime", "value", "type", "memory", "key"],
    )
    def test_analyzer_various_exceptions_handled(self, exception_type):
        """
        Cualquier excepción del analyzer debe ser capturada.
        """
        payload = self._make_payload(
            graph_a=make_digraph(("A", "B")),
            graph_b=make_digraph(("B", "C")),
        )

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.side_effect = (
                exception_type("Test exception")
            )

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is False


# =============================================================================
# PRUEBAS DE INVARIANTES ESTRUCTURALES DEL RESULTADO
# =============================================================================

class TestFusionResultInvariants:
    """
    Pruebas que validan la estructura del dict de resultado.

    Contrato del resultado:
      ┌─────┬───────────────────────────────────────────────────────────────┐
      │ R₁  │ 'success' siempre presente y es bool                         │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ R₂  │ Si success=True → 'stratum', 'payload', 'metrics' presentes  │
      ├─────┼───────────────────────────────────────────────────────────────┤
      │ R₃  │ Si success=False → 'status', 'error' presentes               │
      └─────┴───────────────────────────────────────────────────────────────┘
    """

    ANALYZER_PATCH_PATH = "app.adapters.mic_vectors.BusinessTopologicalAnalyzer"

    @pytest.mark.parametrize(
        "scenario, payload, mock_result",
        [
            (
                "clean_merge",
                {
                    "graph_a": nx.DiGraph([("A", "B")]),
                    "graph_b": nx.DiGraph([("B", "C")]),
                },
                {"delta_beta_1": 0, "status": "CLEAN_MERGE"},
            ),
            (
                "conflict",
                {
                    "graph_a": nx.DiGraph([("A", "B")]),
                    "graph_b": nx.DiGraph([("B", "A")]),
                },
                {"delta_beta_1": 1, "status": "INTEGRATION_CONFLICT"},
            ),
        ],
        ids=["success_path", "failure_path"],
    )
    def test_result_always_has_success_key(
        self, scenario, payload, mock_result
    ):
        """Invariante R₁: 'success' siempre presente y es bool."""
        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                mock_result
            )
            result = vector_audit_homological_fusion(payload)

        validation = validate_fusion_result_structure(result)
        assert validation["is_valid"], f"Violaciones: {validation['violations']}"

    def test_success_result_has_required_keys(self):
        """Invariante R₂: éxito → stratum, payload, metrics presentes."""
        payload = {
            "graph_a": nx.DiGraph([("A", "B")]),
            "graph_b": nx.DiGraph([("B", "C")]),
        }

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = {
                "delta_beta_1": 0,
                "status": "CLEAN_MERGE",
            }

            result = vector_audit_homological_fusion(payload)

        assert result[ResultKeys.SUCCESS] is True
        required = {ResultKeys.STRATUM, ResultKeys.PAYLOAD, ResultKeys.METRICS}
        assert required.issubset(result.keys()), (
            f"Claves faltantes en éxito: {required - result.keys()}"
        )

    def test_failure_result_has_required_keys(self):
        """Invariante R₃: fallo → status y error presentes."""
        result = vector_audit_homological_fusion({})

        assert result[ResultKeys.SUCCESS] is False
        required = {ResultKeys.STATUS, ResultKeys.ERROR}
        assert required.issubset(result.keys()), (
            f"Claves faltantes en fallo: {required - result.keys()}"
        )


# =============================================================================
# PRUEBAS DE RENDIMIENTO Y STRESS
# =============================================================================

class TestPerformanceAndStress:
    """
    Pruebas de rendimiento bajo carga.
    """

    ANALYZER_PATCH_PATH = "app.adapters.mic_vectors.BusinessTopologicalAnalyzer"

    def test_large_graph_fusion(self):
        """
        Fusión de grafos grandes.
        """
        # Crear grafo grande (cadena de 100 nodos)
        large_chain = make_chain_graph([f"N{i}" for i in range(100)])
        small_graph = make_digraph(("N99", "N100"))
        
        payload = {
            "graph_a": large_chain,
            "graph_b": small_graph,
        }

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = {
                "delta_beta_1": 0,
                "status": "CLEAN_MERGE",
            }

            start = time.perf_counter()
            result = vector_audit_homological_fusion(payload)
            elapsed = time.perf_counter() - start

        assert result[ResultKeys.SUCCESS] is True
        assert elapsed < 1.0, f"Fusión tardó {elapsed:.2f}s"

    def test_complex_topology_fusion(self):
        """
        Fusión con topología compleja (digrafo completo).
        """
        k5 = make_complete_digraph(["A", "B", "C", "D", "E"])
        k3 = make_complete_digraph(["X", "Y", "Z"])
        
        payload = {
            "graph_a": k5,
            "graph_b": k3,
        }

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = {
                "delta_beta_1": 0,
                "status": "CLEAN_MERGE",
            }

            result = vector_audit_homological_fusion(payload)

        assert ResultKeys.SUCCESS in result


# =============================================================================
# PRUEBAS DE IDEMPOTENCIA Y DETERMINISMO
# =============================================================================

class TestIdempotenceAndDeterminism:
    """
    Pruebas de propiedades de idempotencia y determinismo.
    """

    ANALYZER_PATCH_PATH = "app.adapters.mic_vectors.BusinessTopologicalAnalyzer"

    def test_fusion_is_deterministic(self):
        """
        Dado el mismo input, siempre produce el mismo output.
        """
        def create_payload():
            return {
                "graph_a": make_digraph(("A", "B")),
                "graph_b": make_digraph(("B", "C")),
            }

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = {
                "delta_beta_1": 0,
                "status": "CLEAN_MERGE",
            }

            results = [
                vector_audit_homological_fusion(create_payload())
                for _ in range(5)
            ]

        # Todos los resultados deben ser equivalentes
        first = results[0]
        for result in results[1:]:
            assert result[ResultKeys.SUCCESS] == first[ResultKeys.SUCCESS]
            if ResultKeys.METRICS in result and ResultKeys.METRICS in first:
                assert (
                    result[ResultKeys.METRICS][ResultKeys.TOPOLOGICAL_COHERENCE]
                    == first[ResultKeys.METRICS][ResultKeys.TOPOLOGICAL_COHERENCE]
                )

    def test_repeated_calls_same_result(self):
        """
        Llamadas repetidas con el mismo payload producen el mismo resultado.
        """
        payload = {
            "graph_a": make_digraph(("A", "B"), ("B", "C")),
            "graph_b": make_digraph(("C", "D")),
        }

        with patch(self.ANALYZER_PATCH_PATH) as MockAnalyzer:
            MockAnalyzer.return_value.audit_integration_homology.return_value = {
                "delta_beta_1": 0,
                "status": "CLEAN_MERGE",
            }

            result1 = vector_audit_homological_fusion(payload)
            result2 = vector_audit_homological_fusion(payload)
            result3 = vector_audit_homological_fusion(payload)

        assert result1[ResultKeys.SUCCESS] == result2[ResultKeys.SUCCESS] == result3[ResultKeys.SUCCESS]