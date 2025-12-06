"""
Tests para Topological State Analyzer
=====================================

Tests unitarios y de integración para SystemTopology y PersistenceHomology.

Estructura:
-----------
1. Fixtures compartidos
2. Tests de SystemTopology
   - Inicialización y configuración
   - Gestión de nodos
   - Gestión de conectividad
   - Cálculo de números de Betti
   - Detección de ciclos estructurales
   - Análisis de salud topológica
3. Tests de PersistenceHomology
   - Inicialización y configuración
   - Gestión de datos
   - Cálculo de intervalos de persistencia
   - Análisis de estados
   - Estadísticas y comparaciones
4. Tests de integración
5. Tests de utilidades
"""

import math
import pytest
from collections import deque
from typing import List, Tuple

from agent.topological_analyzer import (
    # Clases principales
    SystemTopology,
    PersistenceHomology,
    # Enums
    MetricState,
    HealthLevel,
    # Dataclasses
    BettiNumbers,
    PersistenceInterval,
    RequestLoopInfo,
    TopologicalHealth,
    PersistenceAnalysisResult,
    # Utilidades
    compute_wasserstein_distance,
    create_simple_topology,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def empty_topology() -> SystemTopology:
    """Topología sin conexiones."""
    return SystemTopology()


@pytest.fixture
def connected_topology() -> SystemTopology:
    """Topología completamente conectada (árbol)."""
    topo = SystemTopology()
    topo.update_connectivity([
        ("Agent", "Core"),
        ("Core", "Redis"),
        ("Core", "Filesystem"),
    ])
    return topo


@pytest.fixture
def cyclic_topology() -> SystemTopology:
    """Topología con un ciclo."""
    topo = SystemTopology()
    topo.update_connectivity([
        ("Agent", "Core"),
        ("Core", "Redis"),
        ("Core", "Filesystem"),
        ("Redis", "Agent"),  # Crea ciclo Agent-Core-Redis-Agent
    ])
    return topo


@pytest.fixture
def fragmented_topology() -> SystemTopology:
    """Topología fragmentada en múltiples componentes."""
    topo = SystemTopology()
    topo.update_connectivity([
        ("Agent", "Core"),
        # Redis y Filesystem quedan aislados
    ])
    return topo


@pytest.fixture
def empty_persistence() -> PersistenceHomology:
    """Analizador de persistencia vacío."""
    return PersistenceHomology(window_size=10)


@pytest.fixture
def persistence_with_data() -> PersistenceHomology:
    """Analizador con datos de ejemplo."""
    ph = PersistenceHomology(window_size=20)
    # Datos estables
    for _ in range(10):
        ph.add_reading("stable_metric", 0.3)
    # Datos con excursión
    for i in range(20):
        value = 0.9 if 8 <= i <= 14 else 0.3
        ph.add_reading("excursion_metric", value)
    return ph


# =============================================================================
# TESTS: BettiNumbers Dataclass
# =============================================================================

class TestBettiNumbers:
    """Tests para la dataclass BettiNumbers."""

    def test_creation_valid(self):
        """Creación con valores válidos."""
        betti = BettiNumbers(b0=1, b1=0, num_vertices=4, num_edges=3)
        assert betti.b0 == 1
        assert betti.b1 == 0
        assert betti.num_vertices == 4
        assert betti.num_edges == 3

    def test_creation_negative_raises(self):
        """Valores negativos deben lanzar excepción."""
        with pytest.raises(ValueError, match="no pueden ser negativos"):
            BettiNumbers(b0=-1, b1=0)

        with pytest.raises(ValueError, match="no pueden ser negativos"):
            BettiNumbers(b0=1, b1=-1)

    def test_is_connected_property(self):
        """Propiedad is_connected."""
        assert BettiNumbers(b0=1, b1=0).is_connected is True
        assert BettiNumbers(b0=2, b1=0).is_connected is False
        assert BettiNumbers(b0=4, b1=0).is_connected is False

    def test_is_acyclic_property(self):
        """Propiedad is_acyclic."""
        assert BettiNumbers(b0=1, b1=0).is_acyclic is True
        assert BettiNumbers(b0=1, b1=1).is_acyclic is False
        assert BettiNumbers(b0=1, b1=3).is_acyclic is False

    def test_is_ideal_property(self):
        """Propiedad is_ideal (conectado y acíclico)."""
        assert BettiNumbers(b0=1, b1=0).is_ideal is True
        assert BettiNumbers(b0=2, b1=0).is_ideal is False
        assert BettiNumbers(b0=1, b1=1).is_ideal is False
        assert BettiNumbers(b0=2, b1=1).is_ideal is False

    def test_euler_characteristic(self):
        """Característica de Euler χ = β₀ - β₁."""
        assert BettiNumbers(b0=1, b1=0).euler_characteristic == 1
        assert BettiNumbers(b0=1, b1=1).euler_characteristic == 0
        assert BettiNumbers(b0=3, b1=1).euler_characteristic == 2
        assert BettiNumbers(b0=1, b1=3).euler_characteristic == -2

    def test_string_representation(self):
        """Representación en string."""
        ideal = BettiNumbers(b0=1, b1=0)
        assert "✓" in str(ideal)
        assert "β₀=1" in str(ideal)

        non_ideal = BettiNumbers(b0=2, b1=1)
        assert "⚠" in str(non_ideal)

    def test_immutability(self):
        """BettiNumbers es inmutable (frozen)."""
        betti = BettiNumbers(b0=1, b1=0)
        with pytest.raises(AttributeError):
            betti.b0 = 2


# =============================================================================
# TESTS: PersistenceInterval Dataclass
# =============================================================================

class TestPersistenceInterval:
    """Tests para la dataclass PersistenceInterval."""

    def test_finite_interval(self):
        """Intervalo finito (característica muerta)."""
        interval = PersistenceInterval(birth=5, death=10, dimension=0)
        assert interval.lifespan == 5
        assert interval.is_alive is False
        assert interval.persistence == 5 / math.sqrt(2)

    def test_infinite_interval(self):
        """Intervalo infinito (característica viva)."""
        interval = PersistenceInterval(birth=5, death=-1, dimension=0)
        assert interval.lifespan == float('inf')
        assert interval.is_alive is True
        assert interval.persistence == float('inf')

    def test_amplitude(self):
        """Amplitud de la característica."""
        interval = PersistenceInterval(birth=0, death=5, dimension=0, amplitude=0.8)
        assert interval.amplitude == 0.8

    def test_string_representation(self):
        """Representación en string."""
        finite = PersistenceInterval(birth=3, death=7, dimension=0)
        assert str(finite) == "[3, 7)"

        infinite = PersistenceInterval(birth=3, death=-1, dimension=0)
        assert str(infinite) == "[3, ∞)"


# =============================================================================
# TESTS: SystemTopology - Inicialización
# =============================================================================

class TestSystemTopologyInit:
    """Tests de inicialización de SystemTopology."""

    def test_default_initialization(self):
        """Inicialización con valores default."""
        topo = SystemTopology()
        assert len(topo.nodes) == 4
        assert topo.REQUIRED_NODES <= topo.nodes
        assert len(topo.edges) == 0

    def test_custom_max_history(self):
        """Inicialización con historial personalizado."""
        topo = SystemTopology(max_history=100)
        # Verificar que acepta más requests
        for i in range(100):
            topo.record_request(f"req_{i}")
        # No deberíamos perder ninguno aún
        assert len(topo._request_history) == 100

    def test_invalid_max_history_raises(self):
        """max_history < 1 debe lanzar excepción."""
        with pytest.raises(ValueError, match="max_history debe ser >= 1"):
            SystemTopology(max_history=0)

        with pytest.raises(ValueError, match="max_history debe ser >= 1"):
            SystemTopology(max_history=-5)

    def test_custom_nodes(self):
        """Inicialización con nodos personalizados."""
        topo = SystemTopology(custom_nodes={"Gateway", "LoadBalancer"})
        assert "Gateway" in topo.nodes
        assert "LoadBalancer" in topo.nodes
        assert topo.REQUIRED_NODES <= topo.nodes

    def test_custom_topology(self):
        """Inicialización con topología esperada personalizada."""
        custom_edges = {("Gateway", "Core"), ("Core", "Cache")}
        topo = SystemTopology(
            custom_nodes={"Gateway", "Cache"},
            custom_topology=custom_edges
        )
        # Verificar que se incluyen en expected
        missing = topo.get_missing_connections()
        assert ("Gateway", "Core") in missing or ("Core", "Gateway") in missing


# =============================================================================
# TESTS: SystemTopology - Gestión de Nodos
# =============================================================================

class TestSystemTopologyNodes:
    """Tests de gestión de nodos."""

    def test_add_node_valid(self, empty_topology):
        """Agregar nodo válido."""
        assert empty_topology.add_node("NewService") is True
        assert "NewService" in empty_topology.nodes

    def test_add_node_duplicate(self, empty_topology):
        """Agregar nodo duplicado retorna False."""
        empty_topology.add_node("NewService")
        assert empty_topology.add_node("NewService") is False

    def test_add_node_invalid_types(self, empty_topology):
        """Agregar nodos inválidos retorna False."""
        assert empty_topology.add_node("") is False
        assert empty_topology.add_node("   ") is False
        assert empty_topology.add_node(None) is False
        assert empty_topology.add_node(123) is False

    def test_add_node_whitespace_stripped(self, empty_topology):
        """Espacios en blanco se eliminan."""
        assert empty_topology.add_node("  Service  ") is True
        assert "Service" in empty_topology.nodes

    def test_remove_node_dynamic(self, empty_topology):
        """Eliminar nodo dinámico."""
        empty_topology.add_node("TempService")
        assert empty_topology.remove_node("TempService") is True
        assert "TempService" not in empty_topology.nodes

    def test_remove_node_required_fails(self, empty_topology):
        """No se pueden eliminar nodos requeridos."""
        assert empty_topology.remove_node("Agent") is False
        assert empty_topology.remove_node("Core") is False
        assert "Agent" in empty_topology.nodes

    def test_remove_node_nonexistent(self, empty_topology):
        """Eliminar nodo inexistente retorna False."""
        assert empty_topology.remove_node("NonExistent") is False

    def test_has_node(self, empty_topology):
        """Verificar existencia de nodos."""
        assert empty_topology.has_node("Agent") is True
        assert empty_topology.has_node("NonExistent") is False


# =============================================================================
# TESTS: SystemTopology - Conectividad
# =============================================================================

class TestSystemTopologyConnectivity:
    """Tests de gestión de conectividad."""

    def test_update_connectivity_valid(self, empty_topology):
        """Actualizar con conexiones válidas."""
        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
        ])
        assert edges_added == 2
        assert len(warnings) == 0
        assert ("Agent", "Core") in empty_topology.edges or \
               ("Core", "Agent") in empty_topology.edges

    def test_update_connectivity_clears_previous(self, connected_topology):
        """Update limpia conexiones anteriores."""
        initial_edges = len(connected_topology.edges)
        assert initial_edges == 3

        connected_topology.update_connectivity([("Agent", "Core")])
        assert len(connected_topology.edges) == 1

    def test_update_connectivity_invalid_format(self, empty_topology):
        """Formatos inválidos generan warnings."""
        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "Core"),
            "invalid_edge",  # String, no tuple
            ("Only", ),  # Tupla de 1 elemento
            ("A", "B", "C"),  # Tupla de 3 elementos
        ])
        assert edges_added == 1  # Solo la primera es válida
        assert len(warnings) == 3

    def test_update_connectivity_invalid_node_types(self, empty_topology):
        """Nodos no-string generan warnings."""
        edges_added, warnings = empty_topology.update_connectivity([
            (123, "Core"),
            ("Agent", None),
        ])
        assert edges_added == 0
        assert len(warnings) == 2

    def test_update_connectivity_self_loop_ignored(self, empty_topology):
        """Auto-loops se ignoran."""
        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Core"),  # Self-loop
        ])
        assert edges_added == 1
        assert any("Auto-loop" in w for w in warnings)

    def test_update_connectivity_unknown_nodes(self, empty_topology):
        """Nodos desconocidos sin auto_add generan warnings."""
        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "Core"),
            ("Unknown1", "Unknown2"),
        ], validate_nodes=True, auto_add_nodes=False)
        assert edges_added == 1
        assert len(warnings) >= 1

    def test_update_connectivity_auto_add_nodes(self, empty_topology):
        """auto_add_nodes agrega nodos faltantes."""
        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "NewService"),
        ], validate_nodes=True, auto_add_nodes=True)
        assert edges_added == 1
        assert "NewService" in empty_topology.nodes

    def test_add_edge_single(self, empty_topology):
        """Agregar arista individual."""
        assert empty_topology.add_edge("Agent", "Core") is True
        assert len(empty_topology.edges) == 1

    def test_add_edge_nonexistent_node(self, empty_topology):
        """Agregar arista con nodo inexistente falla."""
        assert empty_topology.add_edge("Agent", "Unknown") is False

    def test_add_edge_self_loop(self, empty_topology):
        """Self-loop no permitido."""
        assert empty_topology.add_edge("Agent", "Agent") is False

    def test_remove_edge(self, connected_topology):
        """Eliminar arista existente."""
        initial = len(connected_topology.edges)
        assert connected_topology.remove_edge("Agent", "Core") is True
        assert len(connected_topology.edges) == initial - 1

    def test_remove_edge_nonexistent(self, empty_topology):
        """Eliminar arista inexistente retorna False."""
        assert empty_topology.remove_edge("Agent", "Core") is False


# =============================================================================
# TESTS: SystemTopology - Números de Betti
# =============================================================================

class TestSystemTopologyBettiNumbers:
    """Tests de cálculo de números de Betti."""

    def test_initial_state_no_edges(self, empty_topology):
        """Estado inicial: 4 componentes (nodos aislados)."""
        betti = empty_topology.calculate_betti_numbers()
        assert betti.b0 == 4  # 4 nodos aislados = 4 componentes
        assert betti.b1 == 0  # Sin aristas = sin ciclos
        assert betti.is_connected is False
        assert betti.is_acyclic is True

    def test_tree_topology(self, connected_topology):
        """Topología en árbol: conectado y acíclico."""
        betti = connected_topology.calculate_betti_numbers()
        assert betti.b0 == 1  # Todo conectado
        assert betti.b1 == 0  # Árbol = sin ciclos
        assert betti.is_ideal is True
        # Verificar fórmula de Euler: |E| - |V| + β₀ = β₁
        # 3 - 4 + 1 = 0 ✓
        assert betti.num_edges - betti.num_vertices + betti.b0 == betti.b1

    def test_cyclic_topology(self, cyclic_topology):
        """Topología con ciclo: β₁ > 0."""
        betti = cyclic_topology.calculate_betti_numbers()
        assert betti.b0 == 1  # Sigue conectado
        assert betti.b1 == 1  # Un ciclo
        assert betti.is_acyclic is False
        # Verificar: 4 - 4 + 1 = 1 ✓
        assert betti.num_edges - betti.num_vertices + betti.b0 == betti.b1

    def test_fragmented_topology(self, fragmented_topology):
        """Topología fragmentada: β₀ > 1."""
        betti = fragmented_topology.calculate_betti_numbers()
        # {Agent, Core} + {Redis} + {Filesystem} = 3 componentes
        assert betti.b0 == 3
        assert betti.b1 == 0
        assert betti.is_connected is False

    def test_multiple_cycles(self, empty_topology):
        """Múltiples ciclos incrementan β₁."""
        # Crear grafo con 2 ciclos: Agent-Core-Redis-Agent y Core-Redis-FS-Core
        empty_topology.add_node("FS")
        empty_topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Redis", "Agent"),  # Ciclo 1
            ("Core", "Filesystem"),
            ("Filesystem", "Redis"),  # Ciclo 2 (compartido)
        ], validate_nodes=False)

        betti = empty_topology.calculate_betti_numbers()
        # 5 aristas, 4 nodos, 1 componente => β₁ = 5 - 4 + 1 = 2
        assert betti.b1 == 2

    def test_include_isolated_false(self, fragmented_topology):
        """Excluir nodos aislados del cálculo."""
        betti_with_isolated = fragmented_topology.calculate_betti_numbers(include_isolated=True)
        betti_without_isolated = fragmented_topology.calculate_betti_numbers(include_isolated=False)

        # Con aislados: 3 componentes (Agent-Core, Redis, Filesystem)
        assert betti_with_isolated.b0 == 3
        # Sin aislados: solo los conectados (Agent-Core) = 1 componente
        assert betti_without_isolated.b0 == 1
        assert betti_without_isolated.num_vertices == 2

    def test_empty_graph(self):
        """Grafo completamente vacío."""
        topo = SystemTopology(custom_nodes=set())  # No añade nodos custom
        # Pero REQUIRED_NODES siempre se añaden
        betti = topo.calculate_betti_numbers(include_isolated=False)
        # Sin nodos conectados
        assert betti.num_vertices == 0 or betti.b0 == 0

    def test_euler_characteristic_consistency(self, cyclic_topology):
        """χ = |V| - |E| = β₀ - β₁."""
        betti = cyclic_topology.calculate_betti_numbers()
        chi_from_graph = betti.num_vertices - betti.num_edges
        chi_from_betti = betti.b0 - betti.b1
        assert chi_from_graph == chi_from_betti
        assert chi_from_graph == betti.euler_characteristic


# =============================================================================
# TESTS: SystemTopology - Ciclos y Anomalías
# =============================================================================

class TestSystemTopologyCyclesAndAnomalies:
    """Tests de detección de ciclos y anomalías."""

    def test_find_structural_cycles_none(self, connected_topology):
        """Árbol no tiene ciclos estructurales."""
        cycles = connected_topology.find_structural_cycles()
        assert len(cycles) == 0

    def test_find_structural_cycles_one(self, cyclic_topology):
        """Detectar un ciclo estructural."""
        cycles = cyclic_topology.find_structural_cycles()
        assert len(cycles) == 1
        # El ciclo debe contener Agent, Core, Redis
        cycle_set = set(cycles[0])
        assert {"Agent", "Core", "Redis"} <= cycle_set or \
               {"Agent", "Core", "Redis"} == cycle_set

    def test_record_request_valid(self, empty_topology):
        """Registrar requests válidos."""
        assert empty_topology.record_request("req_123") is True
        assert empty_topology.record_request("req_456") is True

    def test_record_request_invalid(self, empty_topology):
        """Requests inválidos retornan False."""
        assert empty_topology.record_request("") is False
        assert empty_topology.record_request("   ") is False
        assert empty_topology.record_request(None) is False
        assert empty_topology.record_request(123) is False

    def test_detect_request_loops_no_loops(self, empty_topology):
        """Sin loops si cada request es único."""
        for i in range(10):
            empty_topology.record_request(f"unique_req_{i}")

        loops = empty_topology.detect_request_loops(threshold=3)
        assert len(loops) == 0

    def test_detect_request_loops_with_loops(self, empty_topology):
        """Detectar loops cuando hay repeticiones."""
        empty_topology.record_request("normal_1")
        empty_topology.record_request("normal_2")

        # Simular reintentos
        for _ in range(5):
            empty_topology.record_request("retry_req")

        loops = empty_topology.detect_request_loops(threshold=3)
        assert len(loops) == 1
        assert loops[0].request_id == "retry_req"
        assert loops[0].count == 5

    def test_detect_request_loops_threshold(self, empty_topology):
        """El threshold controla qué se considera loop."""
        for _ in range(3):
            empty_topology.record_request("retry_a")
        for _ in range(5):
            empty_topology.record_request("retry_b")

        # Con threshold=4, solo retry_b es loop
        loops = empty_topology.detect_request_loops(threshold=4)
        assert len(loops) == 1
        assert loops[0].request_id == "retry_b"

        # Con threshold=3, ambos son loops
        loops = empty_topology.detect_request_loops(threshold=3)
        assert len(loops) == 2

    def test_get_disconnected_nodes(self, fragmented_topology):
        """Identificar nodos desconectados."""
        disconnected = fragmented_topology.get_disconnected_nodes()
        assert "Redis" in disconnected
        assert "Filesystem" in disconnected
        assert "Agent" not in disconnected
        assert "Core" not in disconnected

    def test_get_missing_connections(self, fragmented_topology):
        """Identificar conexiones faltantes."""
        missing = fragmented_topology.get_missing_connections()
        # Debe faltar Core-Redis y Core-Filesystem
        assert len(missing) >= 2

    def test_get_unexpected_connections(self, cyclic_topology):
        """Identificar conexiones no esperadas."""
        unexpected = cyclic_topology.get_unexpected_connections()
        # Redis-Agent no está en la topología esperada
        assert len(unexpected) >= 1

    def test_clear_request_history(self, empty_topology):
        """Limpiar historial de requests."""
        for i in range(10):
            empty_topology.record_request(f"req_{i}")

        empty_topology.clear_request_history()
        loops = empty_topology.detect_request_loops(threshold=1)
        assert len(loops) == 0


# =============================================================================
# TESTS: SystemTopology - Salud Topológica
# =============================================================================

class TestSystemTopologyHealth:
    """Tests del análisis de salud topológica."""

    def test_healthy_topology(self, connected_topology):
        """Topología ideal = salud óptima."""
        health = connected_topology.get_topological_health()
        assert health.level == HealthLevel.HEALTHY
        assert health.health_score >= 0.9
        assert health.is_healthy is True
        assert len(health.disconnected_nodes) == 0
        assert len(health.missing_edges) == 0

    def test_fragmented_topology_unhealthy(self, fragmented_topology):
        """Fragmentación degrada la salud."""
        health = fragmented_topology.get_topological_health()
        assert health.level in (HealthLevel.DEGRADED, HealthLevel.UNHEALTHY, HealthLevel.CRITICAL)
        assert health.health_score < 0.9
        assert len(health.disconnected_nodes) > 0
        assert "connectivity" in health.diagnostics or \
               "disconnected" in health.diagnostics

    def test_cyclic_topology_degraded(self, cyclic_topology):
        """Ciclos degradan la salud (pero menos que fragmentación)."""
        health = cyclic_topology.get_topological_health()
        # Un ciclo no es crítico, pero no es ideal
        assert health.betti.b1 > 0
        assert health.health_score < 1.0
        assert "cycles" in health.diagnostics

    def test_health_with_retry_loops(self, connected_topology):
        """Loops de reintentos afectan la salud."""
        # Simular reintentos
        for _ in range(5):
            connected_topology.record_request("failing_request")

        health = connected_topology.get_topological_health()
        # Aunque la topología es ideal, los loops afectan
        assert len(health.request_loops) > 0

    def test_health_score_bounds(self, empty_topology):
        """El score de salud está entre 0 y 1."""
        # Peor caso: todo fragmentado
        health = empty_topology.get_topological_health()
        assert 0.0 <= health.health_score <= 1.0

    def test_health_diagnostics_not_empty(self, fragmented_topology):
        """Siempre hay diagnósticos."""
        health = fragmented_topology.get_topological_health()
        assert len(health.diagnostics) > 0


# =============================================================================
# TESTS: SystemTopology - Utilidades
# =============================================================================

class TestSystemTopologyUtilities:
    """Tests de métodos de utilidad."""

    def test_cyclomatic_complexity_tree(self, connected_topology):
        """Complejidad ciclomática de un árbol."""
        cc = connected_topology.calculate_cyclomatic_complexity()
        # Árbol: β₁ = 0, β₀ = 1 => CC = 1
        assert cc == 1

    def test_cyclomatic_complexity_with_cycles(self, cyclic_topology):
        """Complejidad ciclomática con ciclos."""
        cc = cyclic_topology.calculate_cyclomatic_complexity()
        # β₁ = 1, β₀ = 1 => CC = 2
        assert cc == 2

    def test_adjacency_matrix(self, connected_topology):
        """Matriz de adyacencia correcta."""
        matrix = connected_topology.get_adjacency_matrix()
        assert matrix["Agent"]["Core"] == 1
        assert matrix["Core"]["Agent"] == 1
        assert matrix["Agent"]["Redis"] == 0

    def test_to_dict(self, connected_topology):
        """Serialización a diccionario."""
        data = connected_topology.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert "betti_numbers" in data
        assert len(data["nodes"]) == 4

    def test_repr(self, connected_topology):
        """Representación en string."""
        repr_str = repr(connected_topology)
        assert "SystemTopology" in repr_str
        assert "nodes=" in repr_str
        assert "edges=" in repr_str


# =============================================================================
# TESTS: PersistenceHomology - Inicialización
# =============================================================================

class TestPersistenceHomologyInit:
    """Tests de inicialización de PersistenceHomology."""

    def test_default_initialization(self):
        """Inicialización con valores default."""
        ph = PersistenceHomology()
        assert ph.window_size == PersistenceHomology.DEFAULT_WINDOW_SIZE
        assert len(ph.metrics) == 0

    def test_custom_window_size(self):
        """Inicialización con window_size personalizado."""
        ph = PersistenceHomology(window_size=50)
        assert ph.window_size == 50

    def test_invalid_window_size_raises(self):
        """window_size muy pequeño debe lanzar excepción."""
        with pytest.raises(ValueError, match="window_size debe ser"):
            PersistenceHomology(window_size=2)

        with pytest.raises(ValueError, match="window_size debe ser"):
            PersistenceHomology(window_size=0)


# =============================================================================
# TESTS: PersistenceHomology - Gestión de Datos
# =============================================================================

class TestPersistenceHomologyData:
    """Tests de gestión de datos."""

    def test_add_reading_valid(self, empty_persistence):
        """Agregar lecturas válidas."""
        assert empty_persistence.add_reading("cpu", 0.5) is True
        assert empty_persistence.add_reading("cpu", 0.7) is True
        assert "cpu" in empty_persistence.metrics

    def test_add_reading_creates_buffer(self, empty_persistence):
        """Primera lectura crea el buffer."""
        assert len(empty_persistence.metrics) == 0
        empty_persistence.add_reading("new_metric", 0.5)
        assert "new_metric" in empty_persistence.metrics

    def test_add_reading_invalid_name(self, empty_persistence):
        """Nombres inválidos retornan False."""
        assert empty_persistence.add_reading("", 0.5) is False
        assert empty_persistence.add_reading("   ", 0.5) is False
        assert empty_persistence.add_reading(None, 0.5) is False
        assert empty_persistence.add_reading(123, 0.5) is False

    def test_add_reading_invalid_value(self, empty_persistence):
        """Valores inválidos retornan False."""
        assert empty_persistence.add_reading("cpu", "not_a_number") is False
        assert empty_persistence.add_reading("cpu", float('nan')) is False

    def test_add_reading_infinite_capped(self, empty_persistence):
        """Valores infinitos se capean."""
        assert empty_persistence.add_reading("cpu", float('inf')) is True
        buffer = empty_persistence.get_buffer("cpu")
        assert buffer[0] < float('inf')  # Fue capeado

    def test_add_readings_batch(self, empty_persistence):
        """Agregar múltiples lecturas en batch."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        count = empty_persistence.add_readings_batch("metric", values)
        assert count == 5
        assert len(empty_persistence.get_buffer("metric")) == 5

    def test_get_buffer_copy(self, empty_persistence):
        """get_buffer retorna copia, no referencia."""
        empty_persistence.add_reading("cpu", 0.5)
        buffer1 = empty_persistence.get_buffer("cpu")
        buffer2 = empty_persistence.get_buffer("cpu")

        buffer1.append(999)
        assert buffer2[-1] != 999

    def test_get_buffer_nonexistent(self, empty_persistence):
        """Buffer inexistente retorna None."""
        assert empty_persistence.get_buffer("nonexistent") is None

    def test_window_size_enforced(self, empty_persistence):
        """El tamaño de ventana se respeta."""
        for i in range(20):  # window_size = 10
            empty_persistence.add_reading("cpu", i)

        buffer = empty_persistence.get_buffer("cpu")
        assert len(buffer) == 10
        assert buffer[0] == 10  # Los primeros 10 se perdieron

    def test_clear_metric(self, empty_persistence):
        """Limpiar métrica específica."""
        empty_persistence.add_reading("cpu", 0.5)
        empty_persistence.add_reading("mem", 0.3)

        assert empty_persistence.clear_metric("cpu") is True
        assert "cpu" not in empty_persistence.metrics
        assert "mem" in empty_persistence.metrics

    def test_clear_all(self, persistence_with_data):
        """Limpiar todas las métricas."""
        assert len(persistence_with_data.metrics) > 0
        persistence_with_data.clear_all()
        assert len(persistence_with_data.metrics) == 0


# =============================================================================
# TESTS: PersistenceHomology - Intervalos de Persistencia
# =============================================================================

class TestPersistenceHomologyIntervals:
    """Tests de cálculo de intervalos de persistencia."""

    def test_no_excursions(self, empty_persistence):
        """Sin excursiones = diagrama vacío."""
        for _ in range(10):
            empty_persistence.add_reading("cpu", 0.3)

        diagram = empty_persistence.get_persistence_diagram("cpu", threshold=0.5)
        assert len(diagram) == 0

    def test_single_finite_excursion(self, empty_persistence):
        """Una excursión finita = un intervalo cerrado."""
        values = [0.3, 0.3, 0.7, 0.8, 0.6, 0.3, 0.3]
        #               ^    ^    ^
        #             birth      death
        empty_persistence.add_readings_batch("cpu", values)

        diagram = empty_persistence.get_persistence_diagram("cpu", threshold=0.5)
        assert len(diagram) == 1
        assert diagram[0].birth == 2
        assert diagram[0].death == 5
        assert diagram[0].is_alive is False

    def test_single_active_excursion(self, empty_persistence):
        """Excursión activa = intervalo abierto (death=-1)."""
        values = [0.3, 0.3, 0.7, 0.8, 0.9]
        empty_persistence.add_readings_batch("cpu", values)

        diagram = empty_persistence.get_persistence_diagram("cpu", threshold=0.5)
        assert len(diagram) == 1
        assert diagram[0].is_alive is True
        assert diagram[0].death == -1

    def test_multiple_excursions(self, empty_persistence):
        """Múltiples excursiones = múltiples intervalos."""
        values = [0.3, 0.7, 0.3, 0.8, 0.9, 0.3, 0.6, 0.3]
        #              [1,2)     [3,   5)      [6,7)
        empty_persistence.add_readings_batch("cpu", values)

        diagram = empty_persistence.get_persistence_diagram("cpu", threshold=0.5)
        assert len(diagram) == 3

    def test_amplitude_tracking(self, empty_persistence):
        """Amplitud máxima se registra correctamente."""
        values = [0.3, 0.6, 0.9, 0.7, 0.3]  # max above threshold = 0.9 - 0.5 = 0.4
        empty_persistence.add_readings_batch("cpu", values)

        diagram = empty_persistence.get_persistence_diagram("cpu", threshold=0.5)
        assert len(diagram) == 1
        assert diagram[0].amplitude == pytest.approx(0.4, rel=0.01)

    def test_nonexistent_metric(self, empty_persistence):
        """Métrica inexistente = diagrama vacío."""
        diagram = empty_persistence.get_persistence_diagram("nonexistent", threshold=0.5)
        assert len(diagram) == 0


# =============================================================================
# TESTS: PersistenceHomology - Análisis de Estados
# =============================================================================

class TestPersistenceHomologyAnalysis:
    """Tests del análisis de persistencia."""

    def test_stable_all_below_threshold(self, empty_persistence):
        """Todo bajo umbral = STABLE."""
        for _ in range(10):
            empty_persistence.add_reading("cpu", 0.3)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)
        assert result.state == MetricState.STABLE
        assert result.feature_count == 0
        assert result.noise_count == 0

    def test_unknown_insufficient_data(self, empty_persistence):
        """Datos insuficientes = UNKNOWN."""
        empty_persistence.add_reading("cpu", 0.5)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)
        assert result.state == MetricState.UNKNOWN

    def test_noise_short_excursion(self, empty_persistence):
        """Excursión corta = NOISE."""
        # window=10, noise_ratio=0.2 => noise_limit=2
        values = [0.3] * 7 + [0.7] + [0.3] * 2  # Excursión de 1
        empty_persistence.add_readings_batch("cpu", values)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)
        # La última lectura es 0.3, así que no hay excursión activa
        # Pero hubo una excursión de duración 1, que es < 2 => NOISE
        assert result.state in (MetricState.STABLE, MetricState.NOISE)

    def test_feature_long_excursion(self, empty_persistence):
        """Excursión larga = FEATURE."""
        # window=10, noise_ratio=0.2 => noise_limit=2
        values = [0.3] * 5 + [0.7] * 5  # Excursión de 5, termina activa
        empty_persistence.add_readings_batch("cpu", values)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)
        # 5 >= 2 => FEATURE o CRITICAL (si está activa y es larga)
        assert result.state in (MetricState.FEATURE, MetricState.CRITICAL)

    def test_critical_persistent_active(self, empty_persistence):
        """Excursión activa muy larga = CRITICAL."""
        # window=10, critical_ratio=0.5 => critical_limit=5
        values = [0.3] * 2 + [0.8] * 8  # Activa por 8 puntos
        empty_persistence.add_readings_batch("cpu", values)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)
        assert result.state == MetricState.CRITICAL
        assert result.active_count > 0

    def test_custom_ratios(self, empty_persistence):
        """Ratios personalizados funcionan."""
        values = [0.3] * 5 + [0.7] * 5
        empty_persistence.add_readings_batch("cpu", values)

        # Con noise_ratio alto, más cosas son "noise"
        result = empty_persistence.analyze_persistence(
            "cpu", threshold=0.5, noise_ratio=0.6
        )
        # noise_limit = 6, excursión activa de 5 => aún no es feature
        if "noise_limit" in result.metadata:
            assert result.metadata["noise_limit"] == 6

    def test_result_metadata(self, persistence_with_data):
        """El resultado incluye metadata útil."""
        result = persistence_with_data.analyze_persistence(
            "excursion_metric", threshold=0.5
        )
        assert "window_size" in result.metadata
        assert "data_length" in result.metadata


# =============================================================================
# TESTS: PersistenceHomology - Estadísticas y Métricas
# =============================================================================

class TestPersistenceHomologyStatistics:
    """Tests de estadísticas y métricas."""

    def test_statistics_basic(self, empty_persistence):
        """Estadísticas básicas correctas."""
        values = [1, 2, 3, 4, 5]
        empty_persistence.add_readings_batch("test", values)

        stats = empty_persistence.get_statistics("test")
        assert stats["min"] == 1
        assert stats["max"] == 5
        assert stats["mean"] == 3
        assert stats["median"] == 3
        assert stats["count"] == 5

    def test_statistics_std(self, empty_persistence):
        """Desviación estándar correcta."""
        values = [2, 4, 6, 8]  # mean=5, var=5
        empty_persistence.add_readings_batch("test", values)

        stats = empty_persistence.get_statistics("test")
        assert stats["std"] == pytest.approx(2.236, rel=0.01)

    def test_statistics_median_even(self, empty_persistence):
        """Mediana con número par de elementos."""
        values = [1, 2, 3, 4]  # median = (2+3)/2 = 2.5
        empty_persistence.add_readings_batch("test", values)

        stats = empty_persistence.get_statistics("test")
        assert stats["median"] == 2.5

    def test_statistics_nonexistent(self, empty_persistence):
        """Métrica inexistente retorna None."""
        assert empty_persistence.get_statistics("nonexistent") is None

    def test_total_persistence(self, empty_persistence):
        """Persistencia total calculada correctamente."""
        values = [0.3, 0.7, 0.8, 0.3, 0.6, 0.7, 0.3]
        #              [1,      3)      [4,   6)
        # lifespans = 2, 2 => total = 4
        empty_persistence.add_readings_batch("test", values)

        intervals = empty_persistence.get_persistence_diagram("test", 0.5)
        total = empty_persistence.compute_total_persistence(intervals)
        assert total == 4

    def test_persistence_entropy(self, empty_persistence):
        """Entropía de persistencia."""
        # Dos intervalos de igual duración => máxima entropía para 2 elementos
        values = [0.3, 0.7, 0.8, 0.3, 0.7, 0.8, 0.3]
        empty_persistence.add_readings_batch("test", values)

        intervals = empty_persistence.get_persistence_diagram("test", 0.5)
        entropy = empty_persistence.compute_persistence_entropy(intervals)
        # Con 2 intervalos iguales, H = 1.0 (normalizada)
        assert entropy == pytest.approx(1.0, rel=0.01)

    def test_compare_diagrams(self, empty_persistence):
        """Comparación de diagramas de persistencia."""
        # Métrica 1: una excursión larga
        values1 = [0.3] * 5 + [0.7] * 4 + [0.3]
        empty_persistence.add_readings_batch("m1", values1)

        # Métrica 2: igual
        values2 = [0.3] * 5 + [0.7] * 4 + [0.3]
        empty_persistence.add_readings_batch("m2", values2)

        # Distancia 0 para diagramas iguales
        dist = empty_persistence.compare_diagrams("m1", "m2", 0.5)
        assert dist == pytest.approx(0, abs=0.01)

        # Métrica 3: diferente
        values3 = [0.3] * 8 + [0.7] * 2
        empty_persistence.add_readings_batch("m3", values3)

        dist13 = empty_persistence.compare_diagrams("m1", "m3", 0.5)
        assert dist13 > 0


# =============================================================================
# TESTS: Funciones de Utilidad
# =============================================================================

class TestUtilityFunctions:
    """Tests de funciones de utilidad del módulo."""

    def test_create_simple_topology(self):
        """Crear topología simple predefinida."""
        topo = create_simple_topology()
        assert len(topo.edges) == 3

        betti = topo.calculate_betti_numbers()
        assert betti.is_ideal is True

    def test_wasserstein_distance_identical(self):
        """Wasserstein de diagramas idénticos = 0."""
        intervals = [
            PersistenceInterval(birth=0, death=5, dimension=0),
            PersistenceInterval(birth=2, death=7, dimension=0),
        ]

        dist = compute_wasserstein_distance(intervals, intervals)
        assert dist == 0

    def test_wasserstein_distance_empty(self):
        """Wasserstein con diagramas vacíos."""
        dist = compute_wasserstein_distance([], [])
        assert dist == 0

    def test_wasserstein_distance_one_empty(self):
        """Wasserstein con un diagrama vacío."""
        intervals = [
            PersistenceInterval(birth=0, death=5, dimension=0),
        ]

        dist = compute_wasserstein_distance(intervals, [])
        assert dist > 0

    def test_wasserstein_distance_different(self):
        """Wasserstein de diagramas diferentes."""
        intervals1 = [PersistenceInterval(birth=0, death=10, dimension=0)]
        intervals2 = [PersistenceInterval(birth=0, death=5, dimension=0)]

        dist = compute_wasserstein_distance(intervals1, intervals2)
        assert dist == pytest.approx(5, rel=0.01)

    def test_wasserstein_invalid_p(self):
        """p < 1 debe lanzar excepción."""
        with pytest.raises(ValueError, match="p debe ser >= 1"):
            compute_wasserstein_distance([], [], p=0)


# =============================================================================
# TESTS: Integración
# =============================================================================

class TestIntegration:
    """Tests de integración entre componentes."""

    def test_topology_and_persistence_workflow(self):
        """Flujo completo de análisis topológico."""
        # 1. Crear topología
        topo = SystemTopology()

        # 2. Simular conexiones dinámicas
        topo.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem"),
        ])

        # 3. Verificar salud inicial
        health = topo.get_topological_health()
        assert health.is_healthy

        # 4. Simular problema: Redis se desconecta
        topo.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Filesystem"),
        ])

        health = topo.get_topological_health()
        assert not health.is_healthy
        assert len(health.disconnected_nodes) > 0

        # 5. Simular reintentos fallidos
        for _ in range(10):
            topo.record_request("redis_connect_retry")

        loops = topo.detect_request_loops()
        assert len(loops) > 0

    def test_persistence_degradation_detection(self):
        """Detectar degradación progresiva con persistencia."""
        ph = PersistenceHomology(window_size=30)

        # Fase 1: Sistema estable
        for _ in range(10):
            ph.add_reading("latency", 50)

        result = ph.analyze_persistence("latency", threshold=100)
        assert result.state == MetricState.STABLE

        # Fase 2: Degradación comienza
        for _ in range(5):
            ph.add_reading("latency", 80)

        result = ph.analyze_persistence("latency", threshold=100)
        assert result.state == MetricState.STABLE  # Aún bajo umbral

        # Fase 3: Problema evidente
        for _ in range(10):
            ph.add_reading("latency", 150)

        result = ph.analyze_persistence("latency", threshold=100)
        assert result.state in (MetricState.FEATURE, MetricState.CRITICAL)

        # Fase 4: Recuperación
        for _ in range(10):
            ph.add_reading("latency", 50)

        result = ph.analyze_persistence("latency", threshold=100)
        # Debería ver la característica pasada como FEATURE, no CRITICAL
        assert result.feature_count >= 1 or result.state == MetricState.STABLE

    def test_multi_metric_analysis(self):
        """Análisis de múltiples métricas correlacionadas."""
        ph = PersistenceHomology(window_size=20)

        # Simular correlación CPU/Memoria
        for i in range(20):
            cpu = 30 + (i % 5) * 10 + (20 if 10 <= i <= 15 else 0)
            mem = 40 + (i % 5) * 5 + (15 if 10 <= i <= 15 else 0)
            ph.add_reading("cpu", cpu)
            ph.add_reading("mem", mem)

        cpu_result = ph.analyze_persistence("cpu", threshold=60)
        mem_result = ph.analyze_persistence("mem", threshold=60)

        # Ambas deberían mostrar patrones similares
        assert cpu_result.state == mem_result.state or \
               abs(cpu_result.feature_count - mem_result.feature_count) <= 1


# =============================================================================
# TESTS: Casos Edge
# =============================================================================

class TestEdgeCases:
    """Tests de casos límite."""

    def test_topology_single_node(self):
        """Topología con un solo nodo conectado."""
        topo = SystemTopology()
        topo.update_connectivity([])  # Sin aristas

        betti = topo.calculate_betti_numbers()
        assert betti.b0 == 4  # Todos aislados
        assert betti.b1 == 0

    def test_complete_graph(self):
        """Grafo completo (todos conectados con todos)."""
        topo = SystemTopology()
        nodes = list(topo.REQUIRED_NODES)
        edges = [
            (nodes[i], nodes[j])
            for i in range(len(nodes))
            for j in range(i + 1, len(nodes))
        ]
        topo.update_connectivity(edges)

        betti = topo.calculate_betti_numbers()
        # 4 nodos, 6 aristas, 1 componente => β₁ = 6 - 4 + 1 = 3
        assert betti.b0 == 1
        assert betti.b1 == 3

    def test_persistence_constant_above_threshold(self):
        """Métrica constantemente sobre umbral."""
        ph = PersistenceHomology(window_size=10)

        for _ in range(10):
            ph.add_reading("high", 100)

        result = ph.analyze_persistence("high", threshold=50)
        assert result.state == MetricState.CRITICAL
        assert result.active_count == 1

    def test_persistence_oscillating(self):
        """Métrica oscilante alrededor del umbral."""
        ph = PersistenceHomology(window_size=10)

        for i in range(10):
            value = 0.6 if i % 2 == 0 else 0.4
            ph.add_reading("osc", value)

        result = ph.analyze_persistence("osc", threshold=0.5)
        # Muchas excursiones cortas = ruido
        assert result.noise_count >= result.feature_count

    def test_large_history(self):
        """Historia grande de requests."""
        topo = SystemTopology(max_history=1000)

        for i in range(1000):
            topo.record_request(f"req_{i % 100}")  # 100 IDs únicos, cada uno 10 veces

        loops = topo.detect_request_loops(threshold=5)
        # Todos deberían aparecer como loops (10 >= 5)
        assert len(loops) == 100

    def test_repr_methods(self, connected_topology, persistence_with_data):
        """Verificar que __repr__ no falla."""
        assert "SystemTopology" in repr(connected_topology)
        assert "PersistenceHomology" in repr(persistence_with_data)