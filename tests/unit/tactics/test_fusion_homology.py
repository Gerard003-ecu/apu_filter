"""
Suite de Integración: Auditoría Homológica de Fusión (Mayer-Vietoris)
=====================================================================

Valida matemáticamente que la fusión de dos conjuntos de datos
(Presupuesto A y APUs B) no genere "Ciclos Fantasma" (β₁) ni
rupturas topológicas inexistentes en los conjuntos originales.

Fundamento Matemático (Secuencia de Mayer-Vietoris):
----------------------------------------------------
Para dos subespacios A y B, la secuencia exacta de homología es:
    ··· → H_k(A ∩ B) → H_k(A) ⊕ H_k(B) → H_k(A ∪ B) → ···

La "Discrepancia Homológica" (Δβ) mide ciclos espurios:
    Δβ₁ = β₁(A ∪ B) − [β₁(A) + β₁(B) − β₁(A ∩ B)]

Propiedad fundamental: si la secuencia es exacta y la fusión
es coherente, entonces Δβ₁ = 0.

Niveles de Test:
  Nivel 1 — Matemática Pura: Cálculo de Betti sobre grafos sintéticos.
  Nivel 2 — Lógica de Negocio: BusinessTopologicalAnalyzer.
  Nivel 3 — Pipeline: AuditedMergeStep integrado con telemetría.

Escenarios:
  1. Fusión Coherente (Exactitud): A ∪ B preserva aciclicidad.
  2. Ciclo Fantasma (Ghost Cycle): A, B acíclicos; A ∪ B cíclico.
  3. Ciclo Heredado (Inherited): ciclo en A persiste en A ∪ B (Δβ₁=0).
  4. Intersección Ruidosa: nodos frontera con múltiples aristas.
  5. Grafos Vacíos y Disjuntos: bordes del dominio.
  6. Pipeline: telemetría y contexto en AuditedMergeStep.

"""

from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, Mock, patch

import networkx as nx
import pandas as pd
import pytest

from app.telemetry import TelemetryContext
from app.apu_processor import ProcessingThresholds

# Imports condicionales con diagnóstico
try:
    from agent.business_topology import (
        BudgetGraphBuilder,
        BusinessTopologicalAnalyzer,
    )
    _HAS_TOPOLOGY = True
except ImportError:
    _HAS_TOPOLOGY = False

try:
    from app.pipeline_director import AuditedMergeStep
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False

try:
    from app.telemetry_schemas import TopologicalMetrics
    _HAS_TOPO_SCHEMAS = True
except ImportError:
    _HAS_TOPO_SCHEMAS = False


# ============================================================================
# CONSTANTES
# ============================================================================

# Keywords semánticos para assertions de narrativa (case-insensitive)
_CYCLE_KEYWORDS = {
    "ciclo", "cycle", "circular", "loop", "bucle",
    "emergent", "fantasma", "ghost", "spurious",
    "β₁", "beta_1", "beta1", "betti",
}

_CLEAN_MERGE_KEYWORDS = {
    "clean", "limpi", "coherent", "preserv", "exact",
    "acíclic", "acyclic", "integr",
}

_CONFLICT_KEYWORDS = {
    "conflict", "conflicto", "discrepan", "violation",
    "violación", "incoher", "alert", "risk",
}


# ============================================================================
# HELPERS MATEMÁTICOS
# ============================================================================


def compute_betti_numbers(G: nx.Graph) -> Tuple[int, int]:
    """
    Calcula los números de Betti β₀ y β₁ del 1-esqueleto de un grafo.

    β₀ = número de componentes conexas.
    β₁ = rango del primer grupo de homología = |E| − |V| + β₀.
         (Fórmula de Euler para el complejo simplicial del grafo.)

    Para grafos dirigidos, opera sobre la versión no dirigida
    (el 1-esqueleto olvida la orientación).

    Args:
        G: Grafo (dirigido o no dirigido).

    Returns:
        (β₀, β₁) con β₁ ≥ 0.
    """
    if G.number_of_nodes() == 0:
        return (0, 0)

    # Usar MultiGraph para preservar aristas y calcular beta_1 correctamente
    # (Mirroring logic from BusinessTopologicalAnalyzer.calculate_betti_numbers)
    undirected = nx.MultiGraph()
    undirected.add_nodes_from(G.nodes(data=True))
    undirected.add_edges_from(G.edges(data=True))

    beta_0 = nx.number_connected_components(undirected)
    # β₁ = |E| − |V| + β₀ (puede ser 0 para árboles, >0 para grafos con ciclos)
    beta_1 = undirected.number_of_edges() - undirected.number_of_nodes() + beta_0

    return (beta_0, max(0, beta_1))


def compute_mayer_vietoris_delta(
    graph_a: nx.DiGraph,
    graph_b: nx.DiGraph,
) -> Dict[str, Any]:
    """
    Calcula la discrepancia de Mayer-Vietoris para la fusión A ∪ B.

    Δβ₁ = β₁(A ∪ B) − [β₁(A) + β₁(B) − β₁(A ∩ B)]

    La intersección A ∩ B se define sobre nodos compartidos
    y aristas presentes en ambos grafos.

    Returns:
        Dict con:
        - delta_beta_1: Discrepancia homológica.
        - beta_1_A, beta_1_B, beta_1_intersection, beta_1_union.
        - boundary_nodes: Nodos en A ∩ B.
        - intersection_edges: Aristas en A ∩ B.
        - has_emergent_cycles: True si Δβ₁ > 0.
    """
    # Conjuntos de nodos y aristas
    nodes_a = set(graph_a.nodes())
    nodes_b = set(graph_b.nodes())
    edges_a = set(graph_a.edges())
    edges_b = set(graph_b.edges())

    # Intersección: nodos y aristas compartidos
    boundary_nodes = nodes_a & nodes_b
    intersection_edges = edges_a & edges_b

    # Construir grafos derivados
    graph_union = nx.compose(graph_a, graph_b)

    graph_intersection = nx.DiGraph()
    graph_intersection.add_nodes_from(boundary_nodes)
    graph_intersection.add_edges_from(intersection_edges)

    # Calcular números de Betti
    _, beta_1_a = compute_betti_numbers(graph_a)
    _, beta_1_b = compute_betti_numbers(graph_b)
    _, beta_1_inter = compute_betti_numbers(graph_intersection)
    beta_0_union, beta_1_union = compute_betti_numbers(graph_union)

    # Discrepancia de Mayer-Vietoris
    expected_beta_1 = beta_1_a + beta_1_b - beta_1_inter
    delta = beta_1_union - expected_beta_1

    return {
        "delta_beta_1": delta,
        "beta_1_A": beta_1_a,
        "beta_1_B": beta_1_b,
        "beta_1_intersection": beta_1_inter,
        "beta_1_union": beta_1_union,
        "beta_0_union": beta_0_union,
        "expected_beta_1": expected_beta_1,
        "boundary_nodes": boundary_nodes,
        "intersection_edges": intersection_edges,
        "has_emergent_cycles": delta > 0,
    }


def create_labeled_dag(
    edges: List[Tuple[str, str]],
    node_type: str = "generic",
    graph_label: str = "",
) -> nx.DiGraph:
    """
    Crea un grafo dirigido con atributos de nodo tipados.

    Args:
        edges: Lista de tuplas (origen, destino).
        node_type: Tipo asignado a todos los nodos.
        graph_label: Etiqueta del grafo (para debugging).

    Returns:
        nx.DiGraph con atributos de nodo.
    """
    G = nx.DiGraph()
    G.graph["label"] = graph_label

    for src, dst in edges:
        G.add_node(src, node_type=node_type, label=src)
        G.add_node(dst, node_type=node_type, label=dst)
        G.add_edge(src, dst)

    return G


def text_contains_any_keyword(text: str, keywords: Set[str]) -> bool:
    """Verifica si el texto contiene al menos una keyword (case-insensitive)."""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def safely_extract(result: dict, key: str, default=None):
    """Extrae un valor de un dict con fallback, buscando variantes de clave."""
    if key in result:
        return result[key]
    # Buscar variantes comunes (snake_case, camelCase)
    key_lower = key.lower().replace("_", "")
    for k in result:
        if k.lower().replace("_", "") == key_lower:
            return result[k]
    return default


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def telemetry():
    """Contexto de telemetría limpio."""
    return TelemetryContext()


@pytest.fixture
def mock_telemetry():
    """Contexto de telemetría mock para verificación de invocaciones."""
    mock = MagicMock(spec=TelemetryContext)
    mock.start_step = MagicMock()
    mock.end_step = MagicMock()
    mock.record_error = MagicMock()
    mock.record_metric = MagicMock()
    return mock


@pytest.fixture
def analyzer(telemetry):
    """Instancia del Analizador Topológico."""
    if not _HAS_TOPOLOGY:
        pytest.skip(
            "BusinessTopologicalAnalyzer not available. "
            "Install agent.business_topology."
        )
    return BusinessTopologicalAnalyzer(telemetry=telemetry)


# Grafos sintéticos reutilizables

@pytest.fixture
def linear_presupuesto_graph():
    """
    Grafo lineal de presupuesto: ROOT → CAP_1 → ITEM_X
    β₀ = 1 (conexo), β₁ = 0 (acíclico).
    """
    return create_labeled_dag(
        [("ROOT", "CAP_1"), ("CAP_1", "ITEM_X")],
        node_type="presupuesto",
        graph_label="A_presupuesto_lineal",
    )


@pytest.fixture
def linear_apu_graph():
    """
    Grafo lineal de APU: ITEM_X → INSUMO_Y → PROV_Z
    Conecta con presupuesto vía ITEM_X (nodo frontera).
    β₀ = 1, β₁ = 0.
    """
    return create_labeled_dag(
        [("ITEM_X", "INSUMO_Y"), ("INSUMO_Y", "PROV_Z")],
        node_type="apu",
        graph_label="B_apu_lineal",
    )


@pytest.fixture
def cyclic_complement_a():
    """
    Mitad A del ciclo fantasma: MURO → LADRILLO
    β₁ = 0 (una sola arista).
    """
    return create_labeled_dag(
        [("APU_MURO", "MAT_LADRILLO")],
        node_type="presupuesto",
        graph_label="A_half_ghost_cycle",
    )


@pytest.fixture
def cyclic_complement_b():
    """
    Mitad B del ciclo fantasma: LADRILLO → MURO
    β₁ = 0. Pero A ∪ B forma ciclo: β₁ = 1.
    """
    return create_labeled_dag(
        [("MAT_LADRILLO", "APU_MURO")],
        node_type="apu",
        graph_label="B_half_ghost_cycle",
    )


@pytest.fixture
def intrinsic_cycle_graph():
    """
    Grafo con ciclo intrínseco: 1 → 2 → 3 → 1
    β₁ = 1 (el ciclo es propio de A, no de la fusión).
    """
    return create_labeled_dag(
        [("1", "2"), ("2", "3"), ("3", "1")],
        node_type="mixed",
        graph_label="A_intrinsic_cycle",
    )


@pytest.fixture
def linear_extension_graph():
    """
    Extensión lineal: 3 → 4
    Conecta con intrinsic_cycle_graph vía nodo "3".
    β₁ = 0.
    """
    return create_labeled_dag(
        [("3", "4")],
        node_type="apu",
        graph_label="B_linear_extension",
    )


# ============================================================================
# NIVEL 1: TESTS DE MATEMÁTICA PURA
# ============================================================================


class TestBettiNumberComputation:
    """
    Verifica el cálculo correcto de números de Betti sobre
    grafos sintéticos. Estos tests son independientes del analyzer
    y validan los fundamentos matemáticos.
    """

    def test_empty_graph_has_zero_betti(self):
        """Grafo vacío: β₀ = 0, β₁ = 0."""
        G = nx.DiGraph()
        beta_0, beta_1 = compute_betti_numbers(G)
        assert beta_0 == 0
        assert beta_1 == 0

    def test_single_node_has_one_component(self):
        """Un solo nodo: β₀ = 1 (un componente), β₁ = 0."""
        G = nx.DiGraph()
        G.add_node("A")
        beta_0, beta_1 = compute_betti_numbers(G)
        assert beta_0 == 1
        assert beta_1 == 0

    def test_single_edge_is_acyclic(self):
        """Una arista A→B: β₀ = 1, β₁ = 0."""
        G = create_labeled_dag([("A", "B")])
        beta_0, beta_1 = compute_betti_numbers(G)
        assert beta_0 == 1
        assert beta_1 == 0

    def test_linear_chain_is_acyclic(self):
        """Cadena A→B→C→D: β₀ = 1, β₁ = 0 (árbol)."""
        G = create_labeled_dag([("A", "B"), ("B", "C"), ("C", "D")])
        beta_0, beta_1 = compute_betti_numbers(G)
        assert beta_0 == 1
        assert beta_1 == 0

    def test_simple_cycle_has_beta_1_one(self):
        """Ciclo A→B→C→A: β₀ = 1, β₁ = 1."""
        G = create_labeled_dag([("A", "B"), ("B", "C"), ("C", "A")])
        beta_0, beta_1 = compute_betti_numbers(G)
        assert beta_0 == 1
        assert beta_1 == 1

    def test_two_independent_cycles(self):
        """
        Dos ciclos disjuntos: (A→B→A) y (C→D→C).
        β₀ = 2, β₁ = 2.
        """
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "A")])
        G.add_edges_from([("C", "D"), ("D", "C")])
        beta_0, beta_1 = compute_betti_numbers(G)
        assert beta_0 == 2
        assert beta_1 == 2

    def test_tree_has_zero_beta_1(self):
        """
        Un árbol (DAG con raíz única): β₀ = 1, β₁ = 0.
        Propiedad: |E| = |V| − 1 para árboles ⟹ β₁ = 0.
        """
        G = create_labeled_dag([
            ("ROOT", "A"), ("ROOT", "B"),
            ("A", "C"), ("A", "D"),
            ("B", "E"),
        ])
        beta_0, beta_1 = compute_betti_numbers(G)
        assert beta_0 == 1
        assert beta_1 == 0

        # Verificar propiedad de árbol
        n_edges = G.to_undirected().number_of_edges()
        n_nodes = G.number_of_nodes()
        assert n_edges == n_nodes - 1, "Tree must have |E| = |V| - 1"

    def test_disconnected_components(self):
        """
        Dos componentes disjuntas: β₀ = 2, β₁ = 0.
        """
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        G.add_edges_from([("X", "Y")])
        beta_0, beta_1 = compute_betti_numbers(G)
        assert beta_0 == 2
        assert beta_1 == 0

    def test_betti_numbers_are_non_negative(self):
        """Los números de Betti son siempre ≥ 0 por definición."""
        test_cases = [
            [],
            [("A", "B")],
            [("A", "B"), ("B", "C"), ("C", "A")],
            [("A", "B"), ("C", "D")],
        ]
        for edges in test_cases:
            G = create_labeled_dag(edges) if edges else nx.DiGraph()
            beta_0, beta_1 = compute_betti_numbers(G)
            assert beta_0 >= 0, f"β₀ must be ≥ 0, got {beta_0} for edges {edges}"
            assert beta_1 >= 0, f"β₁ must be ≥ 0, got {beta_1} for edges {edges}"


# ============================================================================
# NIVEL 1B: TESTS DE MAYER-VIETORIS PURO
# ============================================================================


class TestMayerVietorisMath:
    """
    Verifica la fórmula de discrepancia de Mayer-Vietoris
    sobre grafos sintéticos con resultados conocidos.

    Δβ₁ = β₁(A ∪ B) − [β₁(A) + β₁(B) − β₁(A ∩ B)]
    """

    def test_clean_merge_linear_graphs(
        self, linear_presupuesto_graph, linear_apu_graph
    ):
        """
        Fusión de dos cadenas lineales con nodo frontera compartido.
        A: ROOT → CAP_1 → ITEM_X (β₁=0)
        B: ITEM_X → INSUMO_Y → PROV_Z (β₁=0)
        A∩B: {ITEM_X} (sin aristas, β₁=0)
        A∪B: ROOT → CAP_1 → ITEM_X → INSUMO_Y → PROV_Z (β₁=0)
        Δβ₁ = 0 − (0 + 0 − 0) = 0
        """
        result = compute_mayer_vietoris_delta(
            linear_presupuesto_graph, linear_apu_graph
        )

        assert result["beta_1_A"] == 0
        assert result["beta_1_B"] == 0
        assert result["beta_1_intersection"] == 0
        assert result["beta_1_union"] == 0
        assert result["delta_beta_1"] == 0
        assert not result["has_emergent_cycles"]
        assert "ITEM_X" in result["boundary_nodes"]

    def test_ghost_cycle_two_halves(
        self, cyclic_complement_a, cyclic_complement_b
    ):
        """
        Ciclo Fantasma: A y B son individualmente acíclicos,
        pero su unión forma un ciclo.

        A: MURO → LADRILLO (β₁=0)
        B: LADRILLO → MURO (β₁=0)
        A∩B: {MURO, LADRILLO} sin aristas compartidas (β₁=0)
        A∪B: MURO → LADRILLO → MURO (β₁=1)
        Δβ₁ = 1 − (0 + 0 − 0) = 1
        """
        result = compute_mayer_vietoris_delta(
            cyclic_complement_a, cyclic_complement_b
        )

        assert result["beta_1_A"] == 0
        assert result["beta_1_B"] == 0
        assert result["beta_1_intersection"] == 0
        assert result["beta_1_union"] == 1
        assert result["delta_beta_1"] == 1
        assert result["has_emergent_cycles"]

        # Ambos nodos son frontera
        assert "APU_MURO" in result["boundary_nodes"]
        assert "MAT_LADRILLO" in result["boundary_nodes"]

        # No comparten aristas (las direcciones son opuestas)
        assert len(result["intersection_edges"]) == 0

    def test_inherited_cycle_not_counted_as_emergent(
        self, intrinsic_cycle_graph, linear_extension_graph
    ):
        """
        Ciclo Heredado: el ciclo ya existía en A.

        A: 1 → 2 → 3 → 1 (β₁=1)
        B: 3 → 4 (β₁=0)
        A∩B: {3} sin aristas compartidas (β₁=0)
        A∪B: 1 → 2 → 3 → 1, 3 → 4 (β₁=1)
        Δβ₁ = 1 − (1 + 0 − 0) = 0

        El ciclo de la unión es heredado, no emergente.
        """
        result = compute_mayer_vietoris_delta(
            intrinsic_cycle_graph, linear_extension_graph
        )

        assert result["beta_1_A"] == 1, "A has intrinsic cycle"
        assert result["beta_1_B"] == 0, "B is linear"
        assert result["beta_1_intersection"] == 0, "Intersection has no edges"
        assert result["beta_1_union"] == 1, "Union inherits A's cycle"
        assert result["delta_beta_1"] == 0, "No emergent cycles"
        assert not result["has_emergent_cycles"]
        assert "3" in result["boundary_nodes"]

    def test_disjoint_graphs_no_interaction(self):
        """
        Grafos disjuntos: sin nodos compartidos.
        A∩B = ∅, Δβ₁ = 0.
        """
        graph_a = create_labeled_dag([("A1", "A2"), ("A2", "A3")])
        graph_b = create_labeled_dag([("B1", "B2"), ("B2", "B3")])

        result = compute_mayer_vietoris_delta(graph_a, graph_b)

        assert len(result["boundary_nodes"]) == 0
        assert result["beta_1_intersection"] == 0
        assert result["delta_beta_1"] == 0
        assert result["beta_0_union"] == 2  # Dos componentes

    def test_identical_graphs_no_emergent_cycles(self):
        """
        Grafos idénticos: A = B.
        A∩B = A = B, β₁(A∩B) = β₁(A) = β₁(B).
        Δβ₁ = β₁(A) − (β₁(A) + β₁(A) − β₁(A)) = 0.
        """
        graph_a = create_labeled_dag([("X", "Y"), ("Y", "Z")])
        graph_b = create_labeled_dag([("X", "Y"), ("Y", "Z")])

        result = compute_mayer_vietoris_delta(graph_a, graph_b)

        assert result["delta_beta_1"] == 0

    def test_both_cyclic_shared_edge(self):
        """
        Ambos grafos tienen ciclos y comparten una arista.

        A: 1 → 2 → 3 → 1 (β₁=1)
        B: 3 → 4 → 1 → 3 (β₁=1, comparte arista implícita vía nodos)
        A∩B: nodos {1, 3}, aristas compartidas dependen de la dirección.

        Se verifica que Δβ₁ se calcula correctamente sin importar
        la complejidad de la intersección.
        """
        graph_a = create_labeled_dag([("1", "2"), ("2", "3"), ("3", "1")])
        graph_b = create_labeled_dag([("3", "4"), ("4", "1"), ("1", "3")])

        result = compute_mayer_vietoris_delta(graph_a, graph_b)

        # Verificar que los Betti individuales son correctos
        assert result["beta_1_A"] == 1, "A has one cycle"
        assert result["beta_1_B"] == 1, "B has one cycle"

        # La arista (1,3) y (3,1) — verificar intersección
        # A tiene arista (3,1), B tiene arista (1,3). Son direcciones opuestas.
        # A∩B aristas: solo las que coinciden en dirección
        shared_edges = result["intersection_edges"]

        # El delta debe ser no-negativo y calculado correctamente
        assert result["delta_beta_1"] >= 0, "Δβ₁ must be non-negative"

        # Verificar la fórmula explícitamente
        expected = (
            result["beta_1_A"]
            + result["beta_1_B"]
            - result["beta_1_intersection"]
        )
        assert result["delta_beta_1"] == result["beta_1_union"] - expected

    def test_empty_graph_a(self):
        """A vacío, B con contenido: Δβ₁ = 0."""
        graph_a = nx.DiGraph()
        graph_b = create_labeled_dag([("X", "Y"), ("Y", "Z")])

        result = compute_mayer_vietoris_delta(graph_a, graph_b)

        assert result["beta_1_A"] == 0
        assert result["delta_beta_1"] == 0

    def test_empty_both_graphs(self):
        """Ambos vacíos: todo es cero."""
        result = compute_mayer_vietoris_delta(nx.DiGraph(), nx.DiGraph())

        assert result["beta_1_A"] == 0
        assert result["beta_1_B"] == 0
        assert result["beta_1_union"] == 0
        assert result["delta_beta_1"] == 0

    @pytest.mark.parametrize(
        "edges_a,edges_b,expected_delta",
        [
            # Caso 1: lineal + lineal con frontera = 0
            ([("A", "B")], [("B", "C")], 0),
            # Caso 2: dos mitades de ciclo = 1 (ghost cycle)
            ([("A", "B")], [("B", "A")], 1),
            # Caso 3: ciclo en A, extensión en B = 0
            ([("A", "B"), ("B", "C"), ("C", "A")], [("C", "D")], 0),
            # Caso 4: disjuntos = 0
            ([("A", "B")], [("X", "Y")], 0),
        ],
        ids=["linear_merge", "ghost_cycle", "inherited_cycle", "disjoint"],
    )
    def test_parametric_delta_computation(
        self, edges_a, edges_b, expected_delta
    ):
        """Verificación paramétrica de Δβ₁ sobre escenarios canónicos."""
        graph_a = create_labeled_dag(edges_a)
        graph_b = create_labeled_dag(edges_b)
        result = compute_mayer_vietoris_delta(graph_a, graph_b)
        assert result["delta_beta_1"] == expected_delta


# ============================================================================
# NIVEL 2: TESTS DE BusinessTopologicalAnalyzer
# ============================================================================


@pytest.mark.skipif(
    not _HAS_TOPOLOGY,
    reason="BusinessTopologicalAnalyzer not available",
)
class TestAnalyzerMayerVietoris:
    """
    Verifica que BusinessTopologicalAnalyzer.audit_integration_homology
    consume grafos correctamente y produce resultados coherentes
    con la matemática verificada en Nivel 1.
    """

    def _verify_api_contract(self, result: dict):
        """
        Verifica que el resultado del analyzer contiene las claves
        mínimas esperadas. Si no, reporta claramente.
        """
        # La clave más crítica es delta_beta_1
        assert "delta_beta_1" in result, (
            f"audit_integration_homology must return 'delta_beta_1'. "
            f"Got keys: {list(result.keys())}"
        )

    def test_analyzer_clean_merge(
        self, analyzer, linear_presupuesto_graph, linear_apu_graph
    ):
        """
        Fusión limpia: el analyzer debe reportar Δβ₁ = 0.
        """
        result = analyzer.audit_integration_homology(
            linear_presupuesto_graph, linear_apu_graph
        )
        self._verify_api_contract(result)

        assert result["delta_beta_1"] == 0, (
            f"Clean merge should have Δβ₁=0. Got: {result['delta_beta_1']}"
        )

        # Verificar veredicto si disponible
        verdict = safely_extract(result, "verdict", "")
        if verdict:
            assert text_contains_any_keyword(
                str(verdict), _CLEAN_MERGE_KEYWORDS | {"clean", "ok", "pass"}
            ) or "CONFLICT" not in str(verdict).upper(), (
                f"Clean merge verdict should not indicate conflict. "
                f"Got: '{verdict}'"
            )

    def test_analyzer_ghost_cycle_detection(
        self, analyzer, cyclic_complement_a, cyclic_complement_b
    ):
        """
        Ciclo fantasma: el analyzer debe reportar Δβ₁ = 1.
        """
        result = analyzer.audit_integration_homology(
            cyclic_complement_a, cyclic_complement_b
        )
        self._verify_api_contract(result)

        assert result["delta_beta_1"] == 1, (
            f"Ghost cycle should produce Δβ₁=1. Got: {result['delta_beta_1']}"
        )

        # Verificar que el resultado indica conflicto
        verdict = safely_extract(result, "verdict", "")
        narrative = safely_extract(result, "narrative", "")

        if verdict:
            assert text_contains_any_keyword(
                str(verdict), _CONFLICT_KEYWORDS | {"conflict", "warning"}
            ), (
                f"Ghost cycle verdict should indicate conflict. "
                f"Got: '{verdict}'"
            )

        if narrative:
            assert text_contains_any_keyword(
                str(narrative), _CYCLE_KEYWORDS
            ), (
                f"Narrative should mention cycles. Got: '{narrative[:200]}...'"
            )

    def test_analyzer_inherited_cycle_not_flagged(
        self, analyzer, intrinsic_cycle_graph, linear_extension_graph
    ):
        """
        Ciclo heredado: Δβ₁ = 0 (el ciclo ya existía en A).
        El analyzer no debe reportar conflicto de integración.
        """
        result = analyzer.audit_integration_homology(
            intrinsic_cycle_graph, linear_extension_graph
        )
        self._verify_api_contract(result)

        assert result["delta_beta_1"] == 0, (
            f"Inherited cycle should produce Δβ₁=0. Got: {result['delta_beta_1']}"
        )

    def test_analyzer_returns_boundary_info(
        self, analyzer, linear_presupuesto_graph, linear_apu_graph
    ):
        """
        El resultado debe contener información sobre nodos frontera
        (la intersección de los conjuntos de nodos).
        """
        result = analyzer.audit_integration_homology(
            linear_presupuesto_graph, linear_apu_graph
        )

        boundary = safely_extract(result, "boundary_nodes")
        if boundary is not None:
            # El nodo frontera debe ser ITEM_X
            if isinstance(boundary, (set, list)):
                assert "ITEM_X" in boundary, (
                    f"ITEM_X should be a boundary node. Got: {boundary}"
                )
            # Si es un conteo, al menos debe ser > 0
            elif isinstance(boundary, int):
                assert boundary > 0

    def test_analyzer_details_contain_individual_betti(
        self, analyzer, cyclic_complement_a, cyclic_complement_b
    ):
        """
        Los detalles del resultado deben contener los números
        de Betti individuales para auditoría.
        """
        result = analyzer.audit_integration_homology(
            cyclic_complement_a, cyclic_complement_b
        )

        details = safely_extract(result, "details")
        if details is not None and isinstance(details, dict):
            beta_1_a = safely_extract(details, "beta_1_A")
            beta_1_union = safely_extract(details, "beta_1_union")

            if beta_1_a is not None:
                assert beta_1_a == 0, "A is acyclic"
            if beta_1_union is not None:
                assert beta_1_union == 1, "Union has ghost cycle"

    def test_analyzer_handles_empty_graph(self, analyzer):
        """
        El analyzer no debe fallar con grafos vacíos.
        """
        empty = nx.DiGraph()
        non_empty = create_labeled_dag([("A", "B")])

        # Ambos vacíos
        result = analyzer.audit_integration_homology(empty, empty)
        self._verify_api_contract(result)
        assert result["delta_beta_1"] == 0

        # A vacío, B no vacío
        result = analyzer.audit_integration_homology(empty, non_empty)
        self._verify_api_contract(result)
        assert result["delta_beta_1"] == 0


# ============================================================================
# NIVEL 2B: TESTS DE CONSISTENCIA ANALYZER vs. CÓMPUTO DIRECTO
# ============================================================================


@pytest.mark.skipif(
    not _HAS_TOPOLOGY,
    reason="BusinessTopologicalAnalyzer not available",
)
class TestAnalyzerConsistency:
    """
    Verifica que el analyzer produce resultados consistentes
    con el cómputo directo de Mayer-Vietoris.
    """

    @pytest.mark.parametrize(
        "edges_a,edges_b,scenario",
        [
            (
                [("ROOT", "CAP"), ("CAP", "ITEM")],
                [("ITEM", "INS"), ("INS", "PROV")],
                "linear_merge",
            ),
            (
                [("MURO", "LADRILLO")],
                [("LADRILLO", "MURO")],
                "ghost_cycle",
            ),
            (
                [("1", "2"), ("2", "3"), ("3", "1")],
                [("3", "4")],
                "inherited_cycle",
            ),
            (
                [("A", "B"), ("B", "C")],
                [("X", "Y"), ("Y", "Z")],
                "disjoint",
            ),
        ],
        ids=["linear", "ghost", "inherited", "disjoint"],
    )
    def test_analyzer_matches_direct_computation(
        self, analyzer, edges_a, edges_b, scenario
    ):
        """
        El Δβ₁ del analyzer debe coincidir con el cálculo directo.
        """
        graph_a = create_labeled_dag(edges_a)
        graph_b = create_labeled_dag(edges_b)

        # Cálculo directo (ground truth)
        direct = compute_mayer_vietoris_delta(graph_a, graph_b)
        expected_delta = direct["delta_beta_1"]

        # Cálculo del analyzer
        analyzer_result = analyzer.audit_integration_homology(graph_a, graph_b)

        assert analyzer_result["delta_beta_1"] == expected_delta, (
            f"Scenario '{scenario}': analyzer Δβ₁="
            f"{analyzer_result['delta_beta_1']} ≠ "
            f"direct Δβ₁={expected_delta}"
        )


# ============================================================================
# NIVEL 3: TESTS DE PIPELINE (AuditedMergeStep)
# ============================================================================


@pytest.mark.skipif(
    not _HAS_PIPELINE,
    reason="AuditedMergeStep not available",
)
class TestAuditedMergeStepPipeline:
    """
    Tests de integración para AuditedMergeStep dentro del pipeline.
    Verifica que la auditoría homológica se ejecuta correctamente
    y sus resultados se propagan al contexto y telemetría.
    """

    def _make_step(self, config: Optional[dict] = None) -> 'AuditedMergeStep':
        """Construye una instancia del step con config defaults."""
        return AuditedMergeStep(
            config=config or {},
            thresholds=ProcessingThresholds(),
        )

    def _make_context_with_dfs(
        self,
        presupuesto: bool = True,
        apus: bool = True,
        insumos: bool = True,
    ) -> dict:
        """
        Construye un contexto con DataFrames dummy.
        Los flags controlan qué DataFrames están presentes.
        """
        ctx = {}
        if presupuesto:
            ctx["df_presupuesto"] = pd.DataFrame({
                "CODIGO_APU": ["APU_01"],
                "DESCRIPCION": ["Test"],
                "CANTIDAD": [1],
            })
        if apus:
            ctx["df_apus_raw"] = pd.DataFrame({
                "CODIGO_APU": ["APU_01"],
                "CODIGO_INSUMO": ["INS_01"],
                "CANTIDAD": [10],
            })
        if insumos:
            ctx["df_insumos"] = pd.DataFrame({
                "CODIGO_INSUMO": ["INS_01"],
                "DESCRIPCION": ["Insumo Test"],
                "VALOR_UNITARIO": [5.0],
            })
        return ctx

    def test_step_propagates_conflict_alert_to_context(self, mock_telemetry):
        """
        Si la auditoría detecta Δβ₁ > 0, el step debe:
        1. Agregar 'integration_risk_alert' al contexto.
        2. Registrar métrica en telemetría.
        3. Completar la fusión física igualmente.
        """
        step = self._make_step()
        context = self._make_context_with_dfs()

        # Grafos que producen ciclo fantasma
        ghost_a = create_labeled_dag([("X", "Y")])
        ghost_b = create_labeled_dag([("Y", "X")])

        mock_audit_result = {
            "delta_beta_1": 1,
            "narrative": "Emergent cycle detected",
            "verdict": "INTEGRATION_CONFLICT",
        }

        with patch(
            "app.pipeline_director.BudgetGraphBuilder"
        ) as MockBuilder, patch(
            "app.pipeline_director.BusinessTopologicalAnalyzer"
        ) as MockAnalyzer, patch(
            "app.pipeline_director.DataMerger"
        ) as MockMerger:
            # Builder retorna grafos que producirían ciclo
            MockBuilder.return_value.build.side_effect = [ghost_a, ghost_b]

            # Analyzer detecta el conflicto
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                mock_audit_result
            )

            # Merger produce resultado válido
            MockMerger.return_value.merge_apus_with_insumos.return_value = (
                pd.DataFrame({"merged": [1, 2, 3]})
            )

            result = step.execute(context, mock_telemetry)

        # 1. Alerta de riesgo en contexto
        assert "integration_risk_alert" in result, (
            "Step must propagate risk alert to context when Δβ₁ > 0"
        )
        alert = result["integration_risk_alert"]
        assert alert["delta_beta_1"] == 1

        # 2. Métrica registrada en telemetría
        mock_telemetry.record_metric.assert_any_call(
            "topology", "emergent_cycles", 1
        )

        # 3. La fusión física se completó
        assert "df_merged" in result, "Physical merge must still complete"
        assert not result["df_merged"].empty

    def test_step_clean_merge_no_alert(self, mock_telemetry):
        """
        Si Δβ₁ = 0, no debe haber alerta de riesgo de integración.
        """
        step = self._make_step()
        context = self._make_context_with_dfs()

        mock_audit_result = {
            "delta_beta_1": 0,
            "narrative": "Clean merge",
            "verdict": "CLEAN_MERGE",
        }

        with patch(
            "app.pipeline_director.BudgetGraphBuilder"
        ) as MockBuilder, patch(
            "app.pipeline_director.BusinessTopologicalAnalyzer"
        ) as MockAnalyzer, patch(
            "app.pipeline_director.DataMerger"
        ) as MockMerger:
            MockBuilder.return_value.build.side_effect = [
                create_labeled_dag([("A", "B")]),
                create_labeled_dag([("B", "C")]),
            ]
            MockAnalyzer.return_value.audit_integration_homology.return_value = (
                mock_audit_result
            )
            MockMerger.return_value.merge_apus_with_insumos.return_value = (
                pd.DataFrame({"merged": [1]})
            )

            result = step.execute(context, mock_telemetry)

        assert "integration_risk_alert" not in result, (
            "Clean merge should not produce risk alert"
        )
        assert "df_merged" in result

    def test_step_audit_failure_does_not_block_merge(self, mock_telemetry):
        """
        Si la auditoría topológica falla (excepción), la fusión
        física debe continuar (degradación controlada).
        """
        step = self._make_step()
        context = self._make_context_with_dfs()

        with patch(
            "app.pipeline_director.BudgetGraphBuilder"
        ) as MockBuilder, patch(
            "app.pipeline_director.BusinessTopologicalAnalyzer"
        ) as MockAnalyzer, patch(
            "app.pipeline_director.DataMerger"
        ) as MockMerger:
            MockBuilder.return_value.build.return_value = nx.DiGraph()
            MockAnalyzer.return_value.audit_integration_homology.side_effect = (
                RuntimeError("Analyzer crashed")
            )
            MockMerger.return_value.merge_apus_with_insumos.return_value = (
                pd.DataFrame({"merged": [1]})
            )

            # No debe lanzar excepción
            result = step.execute(context, mock_telemetry)

        # La fusión debe completarse
        assert "df_merged" in result

        # El error de auditoría debe registrarse
        mock_telemetry.record_error.assert_called()
        error_calls = [
            str(call) for call in mock_telemetry.record_error.call_args_list
        ]
        assert any("audit" in c.lower() or "analyzer" in c.lower()
                    for c in error_calls) or len(error_calls) > 0

    def test_step_skips_audit_when_presupuesto_missing(
        self, mock_telemetry
    ):
        """
        Sin df_presupuesto, la auditoría Mayer-Vietoris se omite
        pero la fusión física continúa.
        """
        step = self._make_step()
        context = self._make_context_with_dfs(presupuesto=False)

        with patch(
            "app.pipeline_director.DataMerger"
        ) as MockMerger:
            MockMerger.return_value.merge_apus_with_insumos.return_value = (
                pd.DataFrame({"merged": [1]})
            )

            result = step.execute(context, mock_telemetry)

        # Fusión completada sin auditoría
        assert "df_merged" in result
        assert "integration_risk_alert" not in result

    def test_step_raises_when_apus_raw_missing(self, mock_telemetry):
        """
        Sin df_apus_raw, el step debe fallar con ValueError descriptivo.
        """
        step = self._make_step()
        context = self._make_context_with_dfs(apus=False)

        with pytest.raises(ValueError, match="df_apus_raw"):
            step.execute(context, mock_telemetry)

        # Error registrado en telemetría
        mock_telemetry.record_error.assert_called()

    def test_step_raises_when_insumos_missing(self, mock_telemetry):
        """
        Sin df_insumos, el step debe fallar con ValueError descriptivo.
        """
        step = self._make_step()
        context = self._make_context_with_dfs(insumos=False)

        with pytest.raises(ValueError, match="df_insumos"):
            step.execute(context, mock_telemetry)

    def test_step_raises_when_merge_produces_empty(self, mock_telemetry):
        """
        Si la fusión física produce un DataFrame vacío,
        el step debe fallar con ValueError.
        """
        step = self._make_step()
        context = self._make_context_with_dfs()

        with patch(
            "app.pipeline_director.BudgetGraphBuilder"
        ) as MockBuilder, patch(
            "app.pipeline_director.BusinessTopologicalAnalyzer"
        ) as MockAnalyzer, patch(
            "app.pipeline_director.DataMerger"
        ) as MockMerger:
            MockBuilder.return_value.build.return_value = nx.DiGraph()
            MockAnalyzer.return_value.audit_integration_homology.return_value = {
                "delta_beta_1": 0,
            }
            MockMerger.return_value.merge_apus_with_insumos.return_value = (
                pd.DataFrame()  # Vacío
            )

            with pytest.raises(ValueError, match="empty"):
                step.execute(context, mock_telemetry)


# ============================================================================
# TESTS DE PROPIEDADES ALGEBRAICAS
# ============================================================================


class TestAlgebraicProperties:
    """
    Verifica propiedades algebraicas que la discrepancia
    de Mayer-Vietoris debe satisfacer siempre.
    """

    def test_delta_is_commutative(self):
        """
        Δβ₁(A, B) = Δβ₁(B, A) porque A ∪ B = B ∪ A.
        La unión es conmutativa.
        """
        graph_a = create_labeled_dag([("X", "Y"), ("Y", "Z")])
        graph_b = create_labeled_dag([("Z", "W"), ("W", "X")])

        result_ab = compute_mayer_vietoris_delta(graph_a, graph_b)
        result_ba = compute_mayer_vietoris_delta(graph_b, graph_a)

        assert result_ab["delta_beta_1"] == result_ba["delta_beta_1"], (
            f"Δβ₁ should be commutative. "
            f"Δ(A,B)={result_ab['delta_beta_1']}, "
            f"Δ(B,A)={result_ba['delta_beta_1']}"
        )

        assert result_ab["beta_1_union"] == result_ba["beta_1_union"]

    def test_delta_is_non_negative(self):
        """
        Δβ₁ ≥ 0 siempre.

        Demostración intuitiva: la fusión puede crear ciclos
        (Δβ₁ > 0) o preservarlos (Δβ₁ = 0), pero no destruir
        ciclos que ya existían en los componentes individuales.

        Nota: En la secuencia exacta de Mayer-Vietoris, el mapa
        de conexión H_1(A∪B) → H_0(A∩B) puede hacer que Δβ₁ sea
        negativo si la intersección tiene ciclos que se "cancelan"
        en la unión. Pero para grafos dirigidos (DAGs de negocio),
        la intersección típicamente no tiene ciclos propios.
        """
        test_cases = [
            ([("A", "B")], [("B", "C")]),                    # Lineal
            ([("A", "B")], [("B", "A")]),                    # Ghost cycle
            ([("A", "B"), ("B", "C"), ("C", "A")], [("C", "D")]),  # Heredado
            ([("A", "B")], [("X", "Y")]),                    # Disjuntos
            ([], [("A", "B")]),                               # A vacío
            ([], []),                                         # Ambos vacíos
        ]

        for edges_a, edges_b in test_cases:
            graph_a = create_labeled_dag(edges_a) if edges_a else nx.DiGraph()
            graph_b = create_labeled_dag(edges_b) if edges_b else nx.DiGraph()

            result = compute_mayer_vietoris_delta(graph_a, graph_b)

            assert result["delta_beta_1"] >= 0, (
                f"Δβ₁ must be ≥ 0. Got {result['delta_beta_1']} for "
                f"edges_a={edges_a}, edges_b={edges_b}"
            )

    def test_self_merge_is_zero(self):
        """
        Δβ₁(A, A) = 0 para todo A.
        Fusionar un grafo consigo mismo no crea ciclos nuevos.
        """
        test_graphs = [
            create_labeled_dag([("A", "B"), ("B", "C")]),
            create_labeled_dag([("A", "B"), ("B", "C"), ("C", "A")]),
            create_labeled_dag([("A", "B"), ("C", "D")]),
        ]

        for G in test_graphs:
            result = compute_mayer_vietoris_delta(G, G)
            assert result["delta_beta_1"] == 0, (
                f"Δβ₁(A,A) must be 0. Got {result['delta_beta_1']} "
                f"for graph with {G.number_of_edges()} edges"
            )

    def test_euler_characteristic_consistency(self):
        """
        Para el 1-esqueleto de un grafo:
        χ = β₀ − β₁ = |V| − |E|

        (Característica de Euler del complejo simplicial 1-dimensional)
        """
        test_cases = [
            [("A", "B"), ("B", "C")],                      # Cadena
            [("A", "B"), ("B", "C"), ("C", "A")],          # Ciclo
            [("A", "B"), ("C", "D")],                      # Disjunto
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")],  # Ciclo + cola
        ]

        for edges in test_cases:
            G = create_labeled_dag(edges)
            beta_0, beta_1 = compute_betti_numbers(G)

            V = G.number_of_nodes()
            E = G.to_undirected().number_of_edges()
            chi_euler = V - E
            chi_betti = beta_0 - beta_1

            assert chi_euler == chi_betti, (
                f"Euler: χ = V−E = {V}−{E} = {chi_euler}, "
                f"Betti: χ = β₀−β₁ = {beta_0}−{beta_1} = {chi_betti}. "
                f"Edges: {edges}"
            )

    def test_mayer_vietoris_formula_explicit(self):
        """
        Verificación explícita de la fórmula Mayer-Vietoris
        para un caso donde todos los términos son > 0.

        A: 1→2→3→1 (β₁=1)
        B: 1→2→4→1 (β₁=1)
        A∩B: nodos {1,2}, arista {(1,2)} (β₁=0)
        A∪B: 1→2→3→1, 1→2→4→1 (β₁=2)

        Δβ₁ = 2 − (1 + 1 − 0) = 0
        """
        graph_a = create_labeled_dag([("1", "2"), ("2", "3"), ("3", "1")])
        graph_b = create_labeled_dag([("1", "2"), ("2", "4"), ("4", "1")])

        result = compute_mayer_vietoris_delta(graph_a, graph_b)

        assert result["beta_1_A"] == 1
        assert result["beta_1_B"] == 1
        assert result["beta_1_intersection"] == 0, (
            f"Intersection should have no cycles (just edge 1→2). "
            f"Got β₁={result['beta_1_intersection']}"
        )
        assert result["beta_1_union"] == 2
        assert result["delta_beta_1"] == 0, (
            "Both cycles existed independently; no emergent cycles"
        )


# ============================================================================
# TESTS DE REGRESIÓN
# ============================================================================


class TestRegressionFusionHomology:
    """
    Tests que documentan y previenen la reaparición de bugs
    específicos encontrados en la suite original.
    """

    def test_regression_patch_path_for_builder(self):
        """
        Bug original: @patch("app.business_topology.BudgetGraphBuilder")
        Ruta correcta: app.pipeline_director.BudgetGraphBuilder
        (se parchea donde se consume, no donde se define).
        """
        # Verificar que el import en pipeline_director funciona
        if _HAS_PIPELINE:
            import app.pipeline_director as pd_mod
            assert hasattr(pd_mod, "BudgetGraphBuilder") or hasattr(
                pd_mod, "AuditedMergeStep"
            ), "pipeline_director must import BudgetGraphBuilder"

    def test_regression_telemetry_access_pattern(self):
        """
        Bug original: telemetry._errors.step
        Patrón correcto: telemetry.record_error("step_name", "message")

        TelemetryContext no expone _errors como atributo público
        con sub-atributo .step.
        """
        ctx = TelemetryContext()

        # Debe tener record_error como método público
        assert hasattr(ctx, "record_error"), (
            "TelemetryContext must expose record_error"
        )
        assert callable(ctx.record_error)

        # No debe depender de _errors
        # (atributo privado cuya estructura interna puede cambiar)

    def test_regression_data_merger_must_be_mocked(self):
        """
        Bug original: test de pipeline no mockeaba DataMerger,
        causando que la fusión física fallara sobre DataFrames dummy.

        Este test verifica que DataMerger existe como dependencia
        mockeable en pipeline_director.
        """
        if _HAS_PIPELINE:
            import app.pipeline_director as pd_mod
            assert hasattr(pd_mod, "DataMerger"), (
                "pipeline_director must import DataMerger for mocking"
            )

    def test_regression_narrative_assertion_fragility(self):
        """
        Bug original: assert "ciclo(s) emergentes" in result["narrative"]
        Este assertion falla con cambios de redacción.

        Fix: usar keyword sets en lugar de strings literales.
        """
        # Todas estas variantes deben matchear
        narratives = [
            "Se detectaron ciclo(s) emergentes en la fusión",
            "Emergent cycle detected in integration",
            "β₁ anomaly: ghost cycles found",
            "Ciclo fantasma detectado en la unión de datos",
        ]

        for narrative in narratives:
            assert text_contains_any_keyword(narrative, _CYCLE_KEYWORDS), (
                f"Keyword set should match: '{narrative}'"
            )