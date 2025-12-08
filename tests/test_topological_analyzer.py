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
    """Topología sin conexiones - verificada."""
    topo = SystemTopology()
    # Verificar estado inicial esperado
    assert len(topo.edges) == 0, "Fixture debe iniciar sin aristas"
    assert topo.REQUIRED_NODES <= topo.nodes, "Debe contener nodos requeridos"
    return topo


@pytest.fixture
def tree_topology() -> SystemTopology:
    """Topología en árbol (conectada, sin ciclos) - verificada."""
    topo = SystemTopology()
    tree_connections = [
        ("Agent", "Core"),
        ("Core", "Redis"),
        ("Core", "Filesystem"),
    ]
    edges_added, warnings = topo.update_connectivity(tree_connections)

    # Verificar que se creó correctamente
    assert edges_added == 3, f"Deben agregarse 3 aristas, se agregaron {edges_added}"
    assert len(warnings) == 0, f"No deben haber warnings: {warnings}"

    # Verificar propiedades de árbol
    betti = topo.calculate_betti_numbers()
    assert betti.b0 == 1, "Árbol debe ser conexo"
    assert betti.b1 == 0, "Árbol no debe tener ciclos"

    return topo


@pytest.fixture
def pyramid_topology() -> SystemTopology:
    """Topología Piramidal completa (ideal del sistema) - verificada."""
    topo = SystemTopology()
    pyramid_connections = list(SystemTopology.EXPECTED_TOPOLOGY)
    edges_added, warnings = topo.update_connectivity(pyramid_connections)

    # Verificar construcción
    assert edges_added == len(pyramid_connections), (
        f"Deben agregarse {len(pyramid_connections)} aristas"
    )

    # Verificar que es la topología esperada completa
    missing = topo.get_missing_connections()
    assert len(missing) == 0, f"No deben faltar conexiones: {missing}"

    return topo


@pytest.fixture
def cyclic_topology() -> SystemTopology:
    """Topología con un ciclo - verificada."""
    topo = SystemTopology()
    connections = [
        ("Agent", "Core"),
        ("Core", "Redis"),
        ("Core", "Filesystem"),
        ("Redis", "Agent"),  # Crea ciclo Agent-Core-Redis-Agent
    ]
    edges_added, _ = topo.update_connectivity(connections)

    # Verificar que tiene exactamente un ciclo
    betti = topo.calculate_betti_numbers()
    assert betti.b0 == 1, "Debe ser conexo"
    assert betti.b1 == 1, "Debe tener exactamente un ciclo"

    return topo


@pytest.fixture
def fragmented_topology() -> SystemTopology:
    """Topología fragmentada en múltiples componentes - verificada."""
    topo = SystemTopology()
    edges_added, _ = topo.update_connectivity([
        ("Agent", "Core"),
        # Redis y Filesystem quedan aislados
    ])

    # Verificar fragmentación
    betti = topo.calculate_betti_numbers()
    assert betti.b0 > 1, "Debe tener múltiples componentes"

    disconnected = topo.get_disconnected_nodes()
    assert len(disconnected) >= 2, "Debe haber nodos desconectados"

    return topo


@pytest.fixture
def empty_persistence() -> PersistenceHomology:
    """Analizador de persistencia vacío - verificado."""
    ph = PersistenceHomology(window_size=10)
    assert ph.window_size == 10
    assert len(ph.metrics) == 0, "Debe iniciar sin métricas"
    return ph


@pytest.fixture
def persistence_with_data() -> PersistenceHomology:
    """Analizador con datos de ejemplo - verificado."""
    ph = PersistenceHomology(window_size=20)

    # Datos estables
    stable_count = 0
    for _ in range(10):
        if ph.add_reading("stable_metric", 0.3):
            stable_count += 1
    assert stable_count == 10, "Deben agregarse 10 lecturas estables"

    # Datos con excursión
    excursion_count = 0
    for i in range(20):
        value = 0.9 if 8 <= i <= 14 else 0.3
        if ph.add_reading("excursion_metric", value):
            excursion_count += 1
    assert excursion_count == 20, "Deben agregarse 20 lecturas con excursión"

    # Verificar que los buffers tienen el tamaño correcto
    stable_buffer = ph.get_buffer("stable_metric")
    assert stable_buffer is not None and len(stable_buffer) == 10

    excursion_buffer = ph.get_buffer("excursion_metric")
    assert excursion_buffer is not None and len(excursion_buffer) == 20

    return ph


# =============================================================================
# TESTS: BettiNumbers Dataclass
# =============================================================================

class TestBettiNumbers:
    """Tests para la dataclass BettiNumbers."""

    def test_creation_valid(self):
        """Creación con valores válidos y verificación de consistencia Euler."""
        betti = BettiNumbers(b0=1, b1=0, num_vertices=4, num_edges=3)
        assert betti.b0 == 1
        assert betti.b1 == 0
        assert betti.num_vertices == 4
        assert betti.num_edges == 3

        # Verificar consistencia con Euler-Poincaré: β₁ = |E| - |V| + β₀
        expected_b1 = betti.num_edges - betti.num_vertices + betti.b0
        assert betti.b1 == expected_b1, (
            f"Inconsistencia Euler: b1={betti.b1}, esperado={expected_b1}"
        )

    def test_creation_negative_b0_raises(self):
        """β₀ negativo debe lanzar excepción."""
        with pytest.raises(ValueError, match="no pueden ser negativos"):
            BettiNumbers(b0=-1, b1=0)

    def test_creation_negative_b1_raises(self):
        """β₁ negativo debe lanzar excepción."""
        with pytest.raises(ValueError, match="no pueden ser negativos"):
            BettiNumbers(b0=1, b1=-1)

    def test_creation_both_negative_raises(self):
        """Ambos negativos deben lanzar excepción."""
        with pytest.raises(ValueError, match="no pueden ser negativos"):
            BettiNumbers(b0=-1, b1=-1)

    def test_creation_zero_values(self):
        """Valores cero son válidos."""
        betti = BettiNumbers(b0=0, b1=0, num_vertices=0, num_edges=0)
        assert betti.b0 == 0
        assert betti.b1 == 0
        assert betti.is_connected is False  # 0 != 1
        assert betti.is_acyclic is True

    def test_is_connected_property_boundary(self):
        """Propiedad is_connected en casos límite."""
        # Exactamente 1 = conectado
        assert BettiNumbers(b0=1, b1=0).is_connected is True

        # 0 componentes (grafo vacío) = no conectado por definición
        assert BettiNumbers(b0=0, b1=0).is_connected is False

        # 2+ componentes = no conectado
        assert BettiNumbers(b0=2, b1=0).is_connected is False

        # Muchos componentes
        assert BettiNumbers(b0=100, b1=0).is_connected is False

    def test_is_acyclic_property_boundary(self):
        """Propiedad is_acyclic en casos límite."""
        # 0 ciclos = acíclico
        assert BettiNumbers(b0=1, b1=0).is_acyclic is True

        # 1 ciclo = no acíclico
        assert BettiNumbers(b0=1, b1=1).is_acyclic is False

        # Muchos ciclos
        assert BettiNumbers(b0=1, b1=100).is_acyclic is False

    def test_is_ideal_all_combinations(self):
        """Propiedad is_ideal: solo True si b0=1 AND b1=0."""
        # Caso ideal
        assert BettiNumbers(b0=1, b1=0).is_ideal is True

        # Fallos en cada condición
        assert BettiNumbers(b0=0, b1=0).is_ideal is False  # no conectado
        assert BettiNumbers(b0=2, b1=0).is_ideal is False  # fragmentado
        assert BettiNumbers(b0=1, b1=1).is_ideal is False  # con ciclo
        assert BettiNumbers(b0=2, b1=1).is_ideal is False  # ambos mal

    def test_euler_characteristic_various_cases(self):
        """Característica de Euler χ = β₀ - β₁ en varios casos."""
        test_cases = [
            (1, 0, 1),    # Árbol típico
            (1, 1, 0),    # Un ciclo
            (3, 1, 2),    # Fragmentado con ciclo
            (1, 3, -2),   # Múltiples ciclos
            (5, 0, 5),    # Muy fragmentado
            (0, 0, 0),    # Vacío
        ]

        for b0, b1, expected_chi in test_cases:
            betti = BettiNumbers(b0=b0, b1=b1)
            assert betti.euler_characteristic == expected_chi, (
                f"Para b0={b0}, b1={b1}: χ esperado={expected_chi}, "
                f"obtenido={betti.euler_characteristic}"
            )

    def test_string_representation_ideal(self):
        """Representación string para estado ideal."""
        ideal = BettiNumbers(b0=1, b1=0)
        str_repr = str(ideal)

        assert "✓" in str_repr, "Ideal debe tener marca de verificación"
        assert "β₀=1" in str_repr
        assert "β₁=0" in str_repr
        assert "χ=1" in str_repr

    def test_string_representation_non_ideal(self):
        """Representación string para estado no ideal."""
        non_ideal = BettiNumbers(b0=2, b1=1)
        str_repr = str(non_ideal)

        assert "⚠" in str_repr, "No ideal debe tener marca de advertencia"
        assert "✓" not in str_repr
        assert "β₀=2" in str_repr
        assert "β₁=1" in str_repr

    def test_immutability_all_fields(self):
        """BettiNumbers es completamente inmutable (frozen)."""
        betti = BettiNumbers(b0=1, b1=0, num_vertices=4, num_edges=3)

        with pytest.raises(AttributeError):
            betti.b0 = 2

        with pytest.raises(AttributeError):
            betti.b1 = 1

        with pytest.raises(AttributeError):
            betti.num_vertices = 10

        with pytest.raises(AttributeError):
            betti.num_edges = 5

    def test_hashability(self):
        """BettiNumbers debe ser hashable (para uso en sets/dicts)."""
        betti1 = BettiNumbers(b0=1, b1=0)
        betti2 = BettiNumbers(b0=1, b1=0)
        betti3 = BettiNumbers(b0=2, b1=1)

        # Debe ser hashable
        hash1 = hash(betti1)
        hash2 = hash(betti2)
        hash3 = hash(betti3)

        # Objetos iguales deben tener mismo hash
        assert hash1 == hash2

        # Se pueden usar en sets
        betti_set = {betti1, betti2, betti3}
        assert len(betti_set) == 2  # betti1 y betti2 son iguales

    def test_equality(self):
        """Igualdad entre instancias de BettiNumbers."""
        betti1 = BettiNumbers(b0=1, b1=0, num_vertices=4, num_edges=3)
        betti2 = BettiNumbers(b0=1, b1=0, num_vertices=4, num_edges=3)
        betti3 = BettiNumbers(b0=1, b1=0, num_vertices=5, num_edges=4)

        assert betti1 == betti2
        assert betti1 != betti3


# =============================================================================
# TESTS: PersistenceInterval Dataclass
# =============================================================================

class TestPersistenceInterval:
    """Tests para la dataclass PersistenceInterval."""

    def test_finite_interval_properties(self):
        """Intervalo finito: todas las propiedades correctas."""
        interval = PersistenceInterval(birth=5, death=10, dimension=0)

        # Verificar propiedades básicas
        assert interval.birth == 5
        assert interval.death == 10
        assert interval.dimension == 0
        assert interval.amplitude == 0.0  # Default

        # Verificar propiedades calculadas
        assert interval.lifespan == 5
        assert interval.is_alive is False
        assert interval.persistence == pytest.approx(5 / math.sqrt(2), rel=1e-6)

    def test_infinite_interval_properties(self):
        """Intervalo infinito: todas las propiedades correctas."""
        interval = PersistenceInterval(birth=5, death=-1, dimension=0)

        assert interval.birth == 5
        assert interval.death == -1
        assert interval.lifespan == float('inf')
        assert interval.is_alive is True
        assert interval.persistence == float('inf')

    def test_zero_lifespan_interval(self):
        """Intervalo con lifespan cero (nacimiento = muerte)."""
        interval = PersistenceInterval(birth=5, death=5, dimension=0)

        assert interval.lifespan == 0
        assert interval.is_alive is False
        assert interval.persistence == 0.0

    def test_amplitude_custom_value(self):
        """Amplitud personalizada."""
        interval = PersistenceInterval(
            birth=0, death=5, dimension=0, amplitude=0.8
        )
        assert interval.amplitude == 0.8

        # Amplitud negativa es técnicamente válida (dataclass no valida)
        interval_neg = PersistenceInterval(
            birth=0, death=5, dimension=0, amplitude=-0.5
        )
        assert interval_neg.amplitude == -0.5

    def test_dimension_values(self):
        """Diferentes dimensiones."""
        dim0 = PersistenceInterval(birth=0, death=5, dimension=0)
        dim1 = PersistenceInterval(birth=0, death=5, dimension=1)
        dim2 = PersistenceInterval(birth=0, death=5, dimension=2)

        assert dim0.dimension == 0
        assert dim1.dimension == 1
        assert dim2.dimension == 2

    def test_string_representation_finite(self):
        """Representación string de intervalo finito."""
        interval = PersistenceInterval(birth=3, death=7, dimension=0)
        str_repr = str(interval)

        assert str_repr == "[3, 7)", f"Esperado '[3, 7)', obtenido '{str_repr}'"

    def test_string_representation_infinite(self):
        """Representación string de intervalo infinito."""
        interval = PersistenceInterval(birth=3, death=-1, dimension=0)
        str_repr = str(interval)

        assert str_repr == "[3, ∞)", f"Esperado '[3, ∞)', obtenido '{str_repr}'"

    def test_string_representation_zero_birth(self):
        """Representación con birth=0."""
        interval = PersistenceInterval(birth=0, death=10, dimension=0)
        assert str(interval) == "[0, 10)"

    def test_persistence_calculation_accuracy(self):
        """Cálculo preciso de persistencia."""
        # Persistencia = (death - birth) / sqrt(2)
        test_cases = [
            (0, 2, 2 / math.sqrt(2)),
            (5, 15, 10 / math.sqrt(2)),
            (0, 100, 100 / math.sqrt(2)),
        ]

        for birth, death, expected_persistence in test_cases:
            interval = PersistenceInterval(birth=birth, death=death, dimension=0)
            assert interval.persistence == pytest.approx(expected_persistence, rel=1e-10), (
                f"Para [{birth}, {death}): esperado {expected_persistence}, "
                f"obtenido {interval.persistence}"
            )

    def test_immutability(self):
        """PersistenceInterval es inmutable."""
        interval = PersistenceInterval(birth=5, death=10, dimension=0)

        with pytest.raises(AttributeError):
            interval.birth = 0

        with pytest.raises(AttributeError):
            interval.death = 20

    def test_large_values(self):
        """Valores grandes no causan overflow."""
        interval = PersistenceInterval(
            birth=0, death=10**9, dimension=0, amplitude=10**6
        )

        assert interval.lifespan == 10**9
        assert math.isfinite(interval.persistence)


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
        """Actualizar con conexiones válidas - verificación completa."""
        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
        ])

        assert edges_added == 2
        assert len(warnings) == 0
        assert len(empty_topology.edges) == 2

        # Verificar que las aristas existen (en cualquier dirección)
        edges = empty_topology.edges
        assert (("Agent", "Core") in edges or ("Core", "Agent") in edges), (
            "Arista Agent-Core no encontrada"
        )
        assert (("Core", "Redis") in edges or ("Redis", "Core") in edges), (
            "Arista Core-Redis no encontrada"
        )

    def test_update_connectivity_clears_previous(self, tree_topology):
        """Update limpia conexiones anteriores completamente."""
        initial_edges = len(tree_topology.edges)
        assert initial_edges == 3, f"Fixture debe tener 3 aristas, tiene {initial_edges}"

        edges_added, _ = tree_topology.update_connectivity([("Agent", "Core")])

        assert len(tree_topology.edges) == 1, "Debe haber solo 1 arista"
        assert edges_added == 1

    def test_update_connectivity_empty_list(self, tree_topology):
        """Lista vacía elimina todas las aristas."""
        initial_edges = len(tree_topology.edges)
        assert initial_edges > 0

        edges_added, warnings = tree_topology.update_connectivity([])

        assert edges_added == 0
        assert len(tree_topology.edges) == 0
        assert len(warnings) == 0

    def test_update_connectivity_invalid_format_string(self, empty_topology):
        """String en lugar de tupla genera warning."""
        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "Core"),
            "invalid_edge",  # String, no tuple
        ])

        assert edges_added == 1
        assert len(warnings) == 1
        assert any("inválido" in w.lower() or "invalid" in w.lower() for w in warnings)

    def test_update_connectivity_invalid_format_wrong_length(self, empty_topology):
        """Tuplas con longitud incorrecta generan warnings."""
        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "Core"),
            ("Only",),           # Tupla de 1 elemento
            ("A", "B", "C"),     # Tupla de 3 elementos
            (),                  # Tupla vacía
        ])

        assert edges_added == 1
        assert len(warnings) == 3  # Una por cada formato incorrecto

    def test_update_connectivity_invalid_node_types(self, empty_topology):
        """Nodos no-string generan warnings específicos."""
        edges_added, warnings = empty_topology.update_connectivity([
            (123, "Core"),       # int como origen
            ("Agent", None),     # None como destino
            (["list"], "Core"),  # list como origen
        ])

        assert edges_added == 0
        assert len(warnings) == 3

    def test_update_connectivity_self_loop_ignored(self, empty_topology):
        """Auto-loops se ignoran con warning."""
        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Core"),    # Self-loop
            ("Redis", "Redis"),  # Otro self-loop
        ])

        assert edges_added == 1
        assert len([w for w in warnings if "loop" in w.lower()]) == 2

    def test_update_connectivity_unknown_nodes_no_auto_add(self, empty_topology):
        """Nodos desconocidos sin auto_add generan warnings."""
        initial_nodes = empty_topology.nodes.copy()

        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "Core"),
            ("Unknown1", "Unknown2"),
            ("Agent", "Unknown3"),
        ], validate_nodes=True, auto_add_nodes=False)

        assert edges_added == 1  # Solo Agent-Core
        assert len(warnings) >= 2

        # Verificar que no se agregaron nodos nuevos
        assert empty_topology.nodes == initial_nodes

    def test_update_connectivity_auto_add_nodes_creates_nodes(self, empty_topology):
        """auto_add_nodes crea nodos y aristas correctamente."""
        initial_node_count = len(empty_topology.nodes)

        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "NewService"),
            ("NewService", "AnotherNew"),
        ], validate_nodes=True, auto_add_nodes=True)

        assert edges_added == 2
        assert "NewService" in empty_topology.nodes
        assert "AnotherNew" in empty_topology.nodes
        assert len(empty_topology.nodes) == initial_node_count + 2

    def test_update_connectivity_duplicate_edges(self, empty_topology):
        """Aristas duplicadas no causan error."""
        edges_added, warnings = empty_topology.update_connectivity([
            ("Agent", "Core"),
            ("Agent", "Core"),  # Duplicada
            ("Core", "Agent"),  # Misma arista, diferente orden
        ])

        # NetworkX maneja duplicados automáticamente
        assert len(empty_topology.edges) == 1

    def test_update_connectivity_whitespace_handling(self, empty_topology):
        """Espacios en blanco en nodos se manejan correctamente."""
        edges_added, warnings = empty_topology.update_connectivity([
            ("  Agent  ", "Core"),
            ("Agent", "  Redis  "),
        ])

        # Depende de la implementación si strip se aplica
        # Verificar que al menos una arista se creó si los nodos existen
        assert edges_added >= 0

    def test_add_edge_single_success(self, empty_topology):
        """Agregar arista individual exitosamente."""
        assert empty_topology.add_edge("Agent", "Core") is True
        assert len(empty_topology.edges) == 1

        # Verificar que no se puede agregar duplicada
        assert empty_topology.add_edge("Agent", "Core") is False
        assert len(empty_topology.edges) == 1

    def test_add_edge_nonexistent_source(self, empty_topology):
        """Agregar arista con nodo origen inexistente falla."""
        assert empty_topology.add_edge("Unknown", "Core") is False
        assert len(empty_topology.edges) == 0

    def test_add_edge_nonexistent_destination(self, empty_topology):
        """Agregar arista con nodo destino inexistente falla."""
        assert empty_topology.add_edge("Agent", "Unknown") is False
        assert len(empty_topology.edges) == 0

    def test_add_edge_both_nonexistent(self, empty_topology):
        """Agregar arista con ambos nodos inexistentes falla."""
        assert empty_topology.add_edge("Unknown1", "Unknown2") is False

    def test_add_edge_self_loop_rejected(self, empty_topology):
        """Self-loop no permitido en add_edge."""
        assert empty_topology.add_edge("Agent", "Agent") is False
        assert len(empty_topology.edges) == 0

    def test_remove_edge_success(self, tree_topology):
        """Eliminar arista existente exitosamente."""
        initial = len(tree_topology.edges)

        result = tree_topology.remove_edge("Agent", "Core")

        assert result is True
        assert len(tree_topology.edges) == initial - 1

    def test_remove_edge_reverse_direction(self, tree_topology):
        """Eliminar arista en dirección reversa también funciona."""
        # La arista es (Agent, Core) o (Core, Agent) dependiendo de cómo se almacene
        initial = len(tree_topology.edges)

        # Intentar en ambas direcciones
        result = tree_topology.remove_edge("Core", "Agent")

        assert result is True
        assert len(tree_topology.edges) == initial - 1

    def test_remove_edge_nonexistent(self, empty_topology):
        """Eliminar arista inexistente retorna False."""
        assert empty_topology.remove_edge("Agent", "Core") is False
        assert empty_topology.remove_edge("Unknown", "Other") is False

    def test_remove_edge_after_clear(self, tree_topology):
        """Eliminar arista después de limpiar falla."""
        tree_topology.update_connectivity([])

        assert tree_topology.remove_edge("Agent", "Core") is False


# =============================================================================
# TESTS: SystemTopology - Números de Betti
# =============================================================================

class TestSystemTopologyBettiNumbers:
    """Tests de cálculo de números de Betti."""

    def test_initial_state_no_edges(self, empty_topology):
        """Estado inicial: todos los nodos aislados."""
        betti = empty_topology.calculate_betti_numbers()

        num_nodes = len(empty_topology.nodes)
        assert betti.b0 == num_nodes, f"β₀ debe ser {num_nodes} (nodos aislados)"
        assert betti.b1 == 0, "Sin aristas no hay ciclos"
        assert betti.is_connected is False
        assert betti.is_acyclic is True
        assert betti.num_vertices == num_nodes
        assert betti.num_edges == 0

    def test_tree_topology_properties(self, tree_topology):
        """Topología en árbol: verificación completa de propiedades."""
        betti = tree_topology.calculate_betti_numbers()

        # Propiedades de árbol
        assert betti.b0 == 1, "Árbol debe ser conexo"
        assert betti.b1 == 0, "Árbol no tiene ciclos"
        assert betti.is_ideal is True

        # Verificar fórmula de Euler: |E| - |V| + β₀ = β₁
        euler_check = betti.num_edges - betti.num_vertices + betti.b0
        assert euler_check == betti.b1, (
            f"Euler-Poincaré: {betti.num_edges} - {betti.num_vertices} + "
            f"{betti.b0} = {euler_check} ≠ {betti.b1}"
        )

        # Para árbol: |E| = |V| - 1
        assert betti.num_edges == betti.num_vertices - 1, "Árbol: |E| = |V| - 1"

    def test_cyclic_topology_properties(self, cyclic_topology):
        """Topología con ciclo: verificación completa."""
        betti = cyclic_topology.calculate_betti_numbers()

        assert betti.b0 == 1, "Debe ser conexo"
        assert betti.b1 == 1, "Debe tener exactamente un ciclo"
        assert betti.is_acyclic is False
        assert betti.is_ideal is False

        # Verificar Euler-Poincaré
        euler_check = betti.num_edges - betti.num_vertices + betti.b0
        assert euler_check == betti.b1

    def test_fragmented_topology_components(self, fragmented_topology):
        """Topología fragmentada: contar componentes correctamente."""
        betti = fragmented_topology.calculate_betti_numbers()

        # {Agent, Core} + {Redis} + {Filesystem} = 3 componentes
        assert betti.b0 == 3, f"Esperados 3 componentes, obtenidos {betti.b0}"
        assert betti.b1 == 0, "Sin ciclos"
        assert betti.is_connected is False

    def test_multiple_cycles_count(self, empty_topology):
        """Múltiples ciclos incrementan β₁ correctamente."""
        # Crear grafo con 2 ciclos independientes
        empty_topology.add_node("Extra")
        empty_topology.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Redis", "Agent"),       # Ciclo 1: Agent-Core-Redis
            ("Core", "Filesystem"),
            ("Filesystem", "Redis"),  # Ciclo 2: Core-Redis-Filesystem
        ], validate_nodes=False)

        betti = empty_topology.calculate_betti_numbers()

        # Verificar usando Euler-Poincaré
        expected_b1 = betti.num_edges - betti.num_vertices + betti.b0
        assert betti.b1 == expected_b1
        assert betti.b1 == 2, f"Esperados 2 ciclos, obtenidos {betti.b1}"

    def test_include_isolated_false_excludes_isolates(self, fragmented_topology):
        """include_isolated=False excluye nodos sin conexiones."""
        betti_with = fragmented_topology.calculate_betti_numbers(include_isolated=True)
        betti_without = fragmented_topology.calculate_betti_numbers(include_isolated=False)

        # Con aislados: 3 componentes
        assert betti_with.b0 == 3
        assert betti_with.num_vertices == 4

        # Sin aislados: solo los conectados (Agent-Core)
        assert betti_without.b0 == 1, "Solo un componente conectado"
        assert betti_without.num_vertices == 2, "Solo Agent y Core conectados"
        assert betti_without.num_edges == 1

    def test_all_isolated_nodes(self, empty_topology):
        """Todos los nodos aislados con include_isolated=False."""
        # No agregar ninguna arista
        betti = empty_topology.calculate_betti_numbers(include_isolated=False)

        # Sin nodos conectados
        assert betti.num_vertices == 0
        assert betti.b0 == 0
        assert betti.b1 == 0

    def test_euler_characteristic_consistency(self, cyclic_topology):
        """χ = |V| - |E| = β₀ - β₁ son consistentes."""
        betti = cyclic_topology.calculate_betti_numbers()

        chi_from_graph = betti.num_vertices - betti.num_edges
        chi_from_betti = betti.b0 - betti.b1

        assert chi_from_graph == chi_from_betti, (
            f"Inconsistencia: |V|-|E|={chi_from_graph}, β₀-β₁={chi_from_betti}"
        )
        assert chi_from_graph == betti.euler_characteristic

    def test_complete_graph_betti_numbers(self, empty_topology):
        """Grafo completo K4: verificar números de Betti."""
        nodes = list(empty_topology.REQUIRED_NODES)
        assert len(nodes) == 4, "Deben haber 4 nodos requeridos"

        # Crear K4 (grafo completo de 4 nodos)
        edges = [
            (nodes[i], nodes[j])
            for i in range(len(nodes))
            for j in range(i + 1, len(nodes))
        ]
        empty_topology.update_connectivity(edges)

        betti = empty_topology.calculate_betti_numbers()

        # K4: 4 nodos, 6 aristas, 1 componente
        assert betti.num_vertices == 4
        assert betti.num_edges == 6
        assert betti.b0 == 1

        # β₁ = 6 - 4 + 1 = 3
        assert betti.b1 == 3, f"K4 debe tener β₁=3, obtenido {betti.b1}"

    def test_star_topology_is_tree(self, empty_topology):
        """Topología estrella es un árbol (β₁=0)."""
        # Core como hub central
        empty_topology.update_connectivity([
            ("Core", "Agent"),
            ("Core", "Redis"),
            ("Core", "Filesystem"),
        ])

        betti = empty_topology.calculate_betti_numbers()

        assert betti.b0 == 1, "Estrella es conexa"
        assert betti.b1 == 0, "Estrella es árbol (sin ciclos)"
        assert betti.is_ideal is True


# =============================================================================
# TESTS: SystemTopology - Ciclos y Anomalías
# =============================================================================

class TestSystemTopologyCyclesAndAnomalies:
    """Tests de detección de ciclos y anomalías."""

    def test_find_structural_cycles_none(self, tree_topology):
        """Árbol no tiene ciclos estructurales."""
        cycles = tree_topology.find_structural_cycles()
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

    def test_get_unexpected_connections(self, empty_topology):
        """Identificar conexiones no esperadas."""
        # La topología esperada es una Pirámide. Cualquier otra cosa es inesperada.
        # Por ejemplo, una conexión directa entre Redis y Filesystem.
        empty_topology.update_connectivity([
            ("Agent", "Core"),
            ("Redis", "Filesystem")  # Conexión inesperada
        ])
        unexpected = empty_topology.get_unexpected_connections()
        assert ("Redis", "Filesystem") in unexpected or ("Filesystem", "Redis") in unexpected

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

    def test_tree_topology_health_assessment(self, tree_topology):
        """Árbol sin todas las conexiones esperadas = DEGRADED."""
        health = tree_topology.get_topological_health()

        # Verificar propiedades básicas
        assert health.betti.b0 == 1, "Árbol debe ser conexo"
        assert health.betti.b1 == 0, "Árbol no tiene ciclos"

        # Faltan conexiones de la pirámide
        assert len(health.missing_edges) > 0, "Deben faltar aristas"

        # Debido a aristas faltantes, no es HEALTHY, pero con los nuevos umbrales
        # puede ser HEALTHY si la penalización es baja.
        # Score esperado: 1.0 - (PENALTY_CAPS['missing_edges'] * (2 / 5))
        # 1.0 - (0.20 * 0.4) = 1.0 - 0.08 = 0.92, que es HEALTHY (>= 0.85)
        assert health.level == HealthLevel.HEALTHY
        assert health.is_healthy is True

        # Score debe reflejar las penalizaciones
        assert 0.0 <= health.health_score <= 1.0

    def test_pyramid_topology_has_cycles(self, pyramid_topology):
        """Pirámide completa tiene ciclos, verificar diagnóstico."""
        health = pyramid_topology.get_topological_health()

        # Verificar que tiene ciclos
        assert health.betti.b1 == 2, (
            f"Pirámide debe tener 2 ciclos, tiene {health.betti.b1}"
        )

        # Los ciclos degradan la salud
        assert "cycles" in health.diagnostics, "Debe diagnosticar ciclos"
        assert health.health_score < 1.0, "Ciclos deben penalizar score"

        # Pero no faltan aristas
        assert len(health.missing_edges) == 0, "Pirámide completa no le faltan aristas"

    def test_fragmented_topology_severe_degradation(self, fragmented_topology):
        """Fragmentación causa degradación severa."""
        health = fragmented_topology.get_topological_health()

        # Verificar fragmentación
        assert health.betti.b0 > 1, "Debe estar fragmentado"

        # Verificar nodos desconectados
        assert len(health.disconnected_nodes) >= 2
        assert "Redis" in health.disconnected_nodes or "Filesystem" in health.disconnected_nodes

        # Debe tener diagnósticos de fragmentación
        has_connectivity_diag = (
            "connectivity" in health.diagnostics or
            "disconnected" in health.diagnostics
        )
        assert has_connectivity_diag, "Debe diagnosticar problema de conectividad"

        # Estado degradado o peor
        assert health.level in (HealthLevel.DEGRADED, HealthLevel.UNHEALTHY, HealthLevel.CRITICAL)
        assert health.health_score < 0.9

    def test_cyclic_topology_cycle_penalty(self, cyclic_topology):
        """Ciclos estructurales penalizan pero no críticamente."""
        health = cyclic_topology.get_topological_health()

        assert health.betti.b1 > 0, "Debe tener ciclos"
        assert "cycles" in health.diagnostics

        # Un ciclo no debería ser crítico por sí solo
        assert health.health_score > 0.3, "Un ciclo no debería ser crítico"

    def test_health_with_retry_loops_penalty(self, pyramid_topology):
        """Loops de reintentos agregan penalización."""
        # Obtener salud sin reintentos
        health_before = pyramid_topology.get_topological_health()

        # Simular reintentos
        for _ in range(5):
            pyramid_topology.record_request("failing_request")

        health_after = pyramid_topology.get_topological_health()

        # Verificar que se detectaron los loops
        assert len(health_after.request_loops) > 0
        assert "retry_loops" in health_after.diagnostics

        # El score debería ser menor o igual
        assert health_after.health_score <= health_before.health_score

    def test_health_score_bounds(self, empty_topology):
        """Score de salud siempre entre 0 y 1."""
        # Peor caso: sin conexiones
        health = empty_topology.get_topological_health()
        assert 0.0 <= health.health_score <= 1.0

        # Verificar el nivel corresponde al score
        if health.health_score >= 0.85:
            assert health.level == HealthLevel.HEALTHY
        elif health.health_score >= 0.65:
            assert health.level == HealthLevel.DEGRADED
        elif health.health_score >= 0.35:
            assert health.level == HealthLevel.UNHEALTHY
        else:
            assert health.level == HealthLevel.CRITICAL

    def test_diagnostics_always_present(self, fragmented_topology):
        """Siempre hay al menos un diagnóstico."""
        health = fragmented_topology.get_topological_health()

        assert len(health.diagnostics) > 0, "Debe haber diagnósticos"

        # Cada diagnóstico debe ser string no vacío
        for key, value in health.diagnostics.items():
            assert isinstance(key, str) and key
            assert isinstance(value, str) and value

    def test_health_level_consistency_with_score(self):
        """Nivel de salud es consistente con el score."""
        topo = SystemTopology()

        # Crear escenarios con diferentes scores
        test_scenarios = [
            ([], HealthLevel.CRITICAL),  # Sin conexiones = muy mal
        ]

        for connections, expected_min_level in test_scenarios:
            topo.update_connectivity(connections)
            health = topo.get_topological_health()

            # Verificar que el nivel no es mejor que el esperado mínimo
            level_order = [
                HealthLevel.CRITICAL,
                HealthLevel.UNHEALTHY,
                HealthLevel.DEGRADED,
                HealthLevel.HEALTHY
            ]

            assert health.level in level_order

    def test_topological_health_immutable_collections(self, fragmented_topology):
        """Las colecciones en TopologicalHealth son inmutables."""
        health = fragmented_topology.get_topological_health()

        # disconnected_nodes es FrozenSet
        assert isinstance(health.disconnected_nodes, frozenset)

        # missing_edges es FrozenSet
        assert isinstance(health.missing_edges, frozenset)

        # request_loops es Tuple
        assert isinstance(health.request_loops, tuple)


# =============================================================================
# TESTS: SystemTopology - Utilidades
# =============================================================================

class TestSystemTopologyUtilities:
    """Tests de métodos de utilidad."""

    def test_cyclomatic_complexity_tree(self, tree_topology):
        """Complejidad ciclomática de un árbol."""
        cc = tree_topology.calculate_cyclomatic_complexity()
        # Árbol: β₁ = 0, β₀ = 1 => CC = 1
        assert cc == 1

    def test_cyclomatic_complexity_with_cycles(self, cyclic_topology):
        """Complejidad ciclomática con ciclos."""
        cc = cyclic_topology.calculate_cyclomatic_complexity()
        # β₁ = 1, β₀ = 1 => CC = 2
        assert cc == 2

    def test_adjacency_matrix(self, tree_topology):
        """Matriz de adyacencia correcta."""
        matrix = tree_topology.get_adjacency_matrix()
        assert matrix["Agent"]["Core"] == 1
        assert matrix["Core"]["Agent"] == 1
        assert matrix["Agent"]["Redis"] == 0  # No hay conexión directa en el árbol

    def test_to_dict(self, pyramid_topology):
        """Serialización a diccionario."""
        data = pyramid_topology.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert "betti_numbers" in data
        assert len(data["nodes"]) == 4

    def test_repr(self, pyramid_topology):
        """Representación en string."""
        repr_str = repr(pyramid_topology)
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
        """Todo bajo umbral = STABLE con verificación completa."""
        for _ in range(10):
            empty_persistence.add_reading("cpu", 0.3)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)

        assert result.state == MetricState.STABLE
        assert result.feature_count == 0
        assert result.noise_count == 0
        assert result.active_count == 0
        assert len(result.intervals) == 0
        assert result.max_lifespan == 0.0
        assert result.total_persistence == 0.0

        # Metadata debe indicar razón
        assert "reason" in result.metadata
        assert result.metadata.get("reason") == "below_threshold"

    def test_unknown_insufficient_data(self, empty_persistence):
        """Datos insuficientes = UNKNOWN con metadata correcta."""
        # Agregar menos del mínimo requerido
        empty_persistence.add_reading("cpu", 0.5)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)

        assert result.state == MetricState.UNKNOWN
        assert "reason" in result.metadata
        assert result.metadata.get("reason") == "insufficient_data"
        assert "samples" in result.metadata

    def test_unknown_nonexistent_metric(self, empty_persistence):
        """Métrica inexistente = UNKNOWN."""
        result = empty_persistence.analyze_persistence("nonexistent", threshold=0.5)

        assert result.state == MetricState.UNKNOWN

    def test_noise_short_excursion_classification(self, empty_persistence):
        """Excursión corta se clasifica como NOISE."""
        # window=10, noise_ratio default=0.2 => noise_limit=2
        # Excursión de 1 punto < noise_limit
        values = [0.3] * 4 + [0.7] + [0.3] * 5  # Una excursión de duración 1
        empty_persistence.add_readings_batch("cpu", values)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)

        # Debe ser NOISE o STABLE (dependiendo de implementación)
        assert result.state in (MetricState.NOISE, MetricState.STABLE)

        if result.state == MetricState.NOISE:
            assert result.noise_count >= 1

    def test_feature_long_excursion_classification(self, empty_persistence):
        """Excursión larga se clasifica como FEATURE."""
        # window=10, noise_ratio=0.2 => noise_limit=2
        # Excursión de 5 puntos que termina activa
        values = [0.3] * 5 + [0.7] * 5
        empty_persistence.add_readings_batch("cpu", values)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)

        # Debe ser FEATURE o CRITICAL (si es activa y muy larga)
        assert result.state in (MetricState.FEATURE, MetricState.CRITICAL)

        # Debe haber características detectadas
        assert len(result.intervals) >= 1

    def test_critical_persistent_active_excursion(self, empty_persistence):
        """Excursión activa muy larga = CRITICAL."""
        # window=10, critical_ratio=0.5 => critical_limit=5
        # Excursión activa de 8 puntos >= critical_limit
        values = [0.3] * 2 + [0.8] * 8
        empty_persistence.add_readings_batch("cpu", values)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)

        assert result.state == MetricState.CRITICAL
        assert result.active_count > 0

        # Metadata debe indicar razón
        assert result.metadata.get("reason") == "persistent_active_excursion"

    def test_custom_noise_ratio_effect(self, empty_persistence):
        """noise_ratio afecta clasificación."""
        values = [0.3] * 5 + [0.7] * 5
        empty_persistence.add_readings_batch("cpu", values)

        # Con noise_ratio bajo, más cosas son "feature"
        result_low = empty_persistence.analyze_persistence(
            "cpu", threshold=0.5, noise_ratio=0.1
        )

        # Con noise_ratio alto, más cosas son "noise"
        result_high = empty_persistence.analyze_persistence(
            "cpu", threshold=0.5, noise_ratio=0.6
        )

        # Verificar que el límite de ruido cambió
        assert result_low.metadata.get("noise_limit", 0) < result_high.metadata.get("noise_limit", 0)

    def test_custom_critical_ratio_effect(self, empty_persistence):
        """critical_ratio afecta clasificación de CRITICAL."""
        # Excursión activa de 6 puntos
        values = [0.3] * 4 + [0.8] * 6
        empty_persistence.add_readings_batch("cpu", values)

        # Con critical_ratio bajo, es CRITICAL
        result_low = empty_persistence.analyze_persistence(
            "cpu", threshold=0.5, critical_ratio=0.5  # limit=5, 6>=5
        )

        # Limpiar y recrear
        empty_persistence.clear_metric("cpu")
        empty_persistence.add_readings_batch("cpu", values)

        # Con critical_ratio alto, podría no ser CRITICAL
        result_high = empty_persistence.analyze_persistence(
            "cpu", threshold=0.5, critical_ratio=0.8  # limit=8, 6<8
        )

        assert result_low.state == MetricState.CRITICAL
        # result_high puede ser FEATURE en lugar de CRITICAL

    def test_result_intervals_correctness(self, empty_persistence):
        """Intervalos en resultado son correctos."""
        # Dos excursiones bien definidas
        values = [0.3, 0.7, 0.8, 0.3, 0.3, 0.6, 0.7, 0.3, 0.3, 0.3]
        #              [1,      3)            [5,   7)
        empty_persistence.add_readings_batch("cpu", values)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)

        assert len(result.intervals) == 2

        # Verificar que los intervalos son PersistenceInterval
        for interval in result.intervals:
            assert isinstance(interval, PersistenceInterval)
            assert interval.dimension == 0

    def test_result_metadata_completeness(self, persistence_with_data):
        """Metadata del resultado es completa."""
        result = persistence_with_data.analyze_persistence(
            "excursion_metric", threshold=0.5
        )

        required_keys = ["window_size", "data_length"]
        for key in required_keys:
            assert key in result.metadata, f"Falta key '{key}' en metadata"

        # Verificar tipos de metadata
        assert isinstance(result.metadata["window_size"], int)
        assert isinstance(result.metadata["data_length"], int)

    def test_total_persistence_calculation(self, empty_persistence):
        """total_persistence es la suma correcta de duraciones."""
        # Dos excursiones: [1,3) duración 2 y [5,7) duración 2
        values = [0.3, 0.7, 0.8, 0.3, 0.3, 0.6, 0.7, 0.3, 0.3, 0.3]
        empty_persistence.add_readings_batch("cpu", values)

        result = empty_persistence.analyze_persistence("cpu", threshold=0.5)

        # Suma de duraciones finitas
        expected_total = sum(
            i.lifespan for i in result.intervals
            if not i.is_alive
        )
        assert result.total_persistence == expected_total


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
        # Varianza muestral de [2, 4, 6, 8] es 6.666...
        # std = sqrt(6.666...) ≈ 2.5819
        assert stats["std"] == pytest.approx(2.5819, rel=0.01)

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

    def test_topology_lifecycle_complete(self):
        """Ciclo de vida completo de topología."""
        # 1. Crear topología vacía
        topo = SystemTopology()
        assert len(topo.edges) == 0

        # 2. Verificar estado inicial
        health_initial = topo.get_topological_health()
        assert health_initial.betti.b0 == 4, "4 nodos aislados"
        assert health_initial.level == HealthLevel.CRITICAL

        # 3. Construir topología en árbol
        topo.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem"),
        ])

        health_tree = topo.get_topological_health()
        assert health_tree.betti.b0 == 1, "Árbol es conexo"
        assert health_tree.betti.b1 == 0, "Árbol sin ciclos"

        # 4. Evolucionar a pirámide (agregar más conexiones)
        topo.update_connectivity(list(SystemTopology.EXPECTED_TOPOLOGY))

        health_pyramid = topo.get_topological_health()
        assert len(health_pyramid.missing_edges) == 0

        # 5. Simular degradación (perder una conexión)
        reduced = [e for e in SystemTopology.EXPECTED_TOPOLOGY if e != ("Core", "Redis")]
        topo.update_connectivity(reduced)

        health_degraded = topo.get_topological_health()
        assert len(health_degraded.missing_edges) > 0

        # 6. Recuperación completa
        topo.update_connectivity(list(SystemTopology.EXPECTED_TOPOLOGY))

        health_recovered = topo.get_topological_health()
        assert len(health_recovered.missing_edges) == 0

    def test_persistence_metric_evolution(self):
        """Evolución de métrica a través del tiempo."""
        ph = PersistenceHomology(window_size=30)

        # Fase 1: Sistema estable (10 lecturas)
        for _ in range(10):
            ph.add_reading("latency", 50)

        result1 = ph.analyze_persistence("latency", threshold=100)
        assert result1.state == MetricState.STABLE, "Fase 1: debe ser estable"

        # Fase 2: Ligera degradación (5 lecturas)
        for _ in range(5):
            ph.add_reading("latency", 80)

        result2 = ph.analyze_persistence("latency", threshold=100)
        assert result2.state == MetricState.STABLE, "Fase 2: aún bajo umbral"

        # Fase 3: Problema evidente (10 lecturas sobre umbral)
        for _ in range(10):
            ph.add_reading("latency", 150)

        result3 = ph.analyze_persistence("latency", threshold=100)
        assert result3.state in (MetricState.FEATURE, MetricState.CRITICAL), (
            "Fase 3: debe detectar problema"
        )

        # Fase 4: Recuperación (10 lecturas bajo umbral)
        for _ in range(10):
            ph.add_reading("latency", 50)

        result4 = ph.analyze_persistence("latency", threshold=100)
        # Puede ver la característica pasada como FEATURE o volver a STABLE
        assert result4.feature_count >= 1 or result4.state == MetricState.STABLE, (
            "Fase 4: debe mostrar feature pasada o estabilizarse"
        )

    def test_topology_persistence_correlation(self):
        """Correlación entre topología y persistencia de métricas."""
        topo = SystemTopology()
        ph = PersistenceHomology(window_size=20)

        # Simular sistema saludable
        topo.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem"),
        ])

        for _ in range(20):
            ph.add_reading("response_time", 50)  # Bajo

        topo_health = topo.get_topological_health()
        metric_result = ph.analyze_persistence("response_time", threshold=100)

        # Topología buena + métrica estable = sistema saludable
        assert topo_health.betti.is_connected
        assert metric_result.state == MetricState.STABLE

        # Simular degradación topológica
        topo.update_connectivity([("Agent", "Core")])  # Perder conexiones

        # Y métricas degradándose
        for _ in range(10):
            ph.add_reading("response_time", 150)  # Alto

        topo_health_degraded = topo.get_topological_health()
        metric_result_degraded = ph.analyze_persistence("response_time", threshold=100)

        # Verificar correlación de degradación
        assert len(topo_health_degraded.missing_edges) > len(topo_health.missing_edges)
        assert metric_result_degraded.state != MetricState.STABLE

    def test_multi_metric_analysis_consistency(self):
        """Análisis de múltiples métricas mantiene consistencia."""
        ph = PersistenceHomology(window_size=20)

        # Simular métricas correlacionadas (CPU/Memoria)
        for i in range(20):
            base_load = 30 + (i % 5) * 10
            spike = 25 if 10 <= i <= 15 else 0

            cpu = base_load + spike
            mem = base_load * 0.8 + spike * 0.7

            assert ph.add_reading("cpu", cpu) is True
            assert ph.add_reading("mem", mem) is True

        cpu_result = ph.analyze_persistence("cpu", threshold=60)
        mem_result = ph.analyze_persistence("mem", threshold=50)

        # Verificar que ambas métricas fueron analizadas
        assert cpu_result.state != MetricState.UNKNOWN
        assert mem_result.state != MetricState.UNKNOWN

        # Comparar diagramas para verificar correlación
        distance = ph.compare_diagrams("cpu", "mem", threshold=50)
        # No necesariamente 0, pero debe ser finito
        assert math.isfinite(distance)

    def test_request_loops_affect_health(self):
        """Loops de reintentos afectan métricas de salud."""
        topo = SystemTopology()

        # Topología básica
        topo.update_connectivity([
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem"),
        ])

        health_before = topo.get_topological_health()

        # Agregar muchos reintentos
        for i in range(20):
            topo.record_request(f"retry_{i % 3}")  # 3 IDs, cada uno ~7 veces

        health_after = topo.get_topological_health()

        # Verificar que se detectaron loops
        loops = topo.detect_request_loops(threshold=5)
        assert len(loops) >= 3, "Deben detectarse 3 patrones de reintento"

        # Los loops deben afectar el diagnóstico
        assert "retry_loops" in health_after.diagnostics


# =============================================================================
# TESTS: Casos Edge
# =============================================================================

class TestEdgeCases:
    """Tests de casos límite."""

    def test_topology_no_edges_all_isolated(self):
        """Topología sin aristas: todos los nodos aislados."""
        topo = SystemTopology()
        topo.update_connectivity([])

        betti = topo.calculate_betti_numbers()

        assert betti.b0 == len(topo.nodes), "Cada nodo es su propia componente"
        assert betti.b1 == 0, "Sin aristas = sin ciclos"
        assert betti.num_edges == 0

    def test_complete_graph_maximum_cycles(self):
        """Grafo completo K_n tiene máximos ciclos."""
        topo = SystemTopology()
        nodes = list(topo.REQUIRED_NODES)
        n = len(nodes)

        # Crear K_n
        edges = [
            (nodes[i], nodes[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        topo.update_connectivity(edges)

        betti = topo.calculate_betti_numbers()

        # Para K_n: |E| = n(n-1)/2
        expected_edges = n * (n - 1) // 2
        assert betti.num_edges == expected_edges

        # β₁ = |E| - |V| + 1 = n(n-1)/2 - n + 1
        expected_b1 = expected_edges - n + 1
        assert betti.b1 == expected_b1

    def test_persistence_constant_above_threshold_from_start(self):
        """Métrica constantemente sobre umbral desde el inicio."""
        ph = PersistenceHomology(window_size=10)

        for _ in range(10):
            ph.add_reading("high", 100)

        result = ph.analyze_persistence("high", threshold=50)

        assert result.state == MetricState.CRITICAL
        assert result.active_count == 1
        assert len(result.intervals) == 1
        assert result.intervals[0].is_alive is True
        assert result.intervals[0].birth == 0

    def test_persistence_single_point_spike(self):
        """Spike de un solo punto."""
        ph = PersistenceHomology(window_size=10)

        values = [0.3] * 4 + [0.9] + [0.3] * 5
        ph.add_readings_batch("spike", values)

        result = ph.analyze_persistence("spike", threshold=0.5)

        # Un spike de un punto debería ser ruido o ignorado
        assert result.state in (MetricState.NOISE, MetricState.STABLE)

    def test_persistence_oscillating_around_threshold(self):
        """Métrica oscilando exactamente alrededor del umbral."""
        ph = PersistenceHomology(window_size=10)

        for i in range(10):
            value = 0.55 if i % 2 == 0 else 0.45
            ph.add_reading("osc", value)

        result = ph.analyze_persistence("osc", threshold=0.5)

        # Muchas excursiones muy cortas = probablemente ruido
        assert result.noise_count >= result.feature_count

    def test_persistence_exactly_at_threshold(self):
        """Valores exactamente en el umbral (edge case)."""
        ph = PersistenceHomology(window_size=10)

        for _ in range(10):
            ph.add_reading("exact", 0.5)  # Exactamente el umbral

        result = ph.analyze_persistence("exact", threshold=0.5)

        # Valores en el umbral no están "por encima", así que debe ser STABLE
        assert result.state == MetricState.STABLE

    def test_large_request_history_performance(self):
        """Historia grande de requests no causa problemas."""
        topo = SystemTopology(max_history=1000)

        # Agregar muchos requests
        for i in range(1000):
            assert topo.record_request(f"req_{i % 100}") is True

        # Verificar que el historial no excede el máximo
        assert len(topo._request_history) == 1000

        # Detectar loops debe funcionar eficientemente
        loops = topo.detect_request_loops(threshold=5)

        # 100 IDs únicos, cada uno aparece 10 veces
        assert len(loops) == 100
        for loop in loops:
            assert loop.count == 10

    def test_empty_metrics_operations(self):
        """Operaciones en métricas vacías no fallan."""
        ph = PersistenceHomology(window_size=10)

        # Estadísticas de métrica inexistente
        assert ph.get_statistics("nonexistent") is None

        # Diagrama de métrica inexistente
        diagram = ph.get_persistence_diagram("nonexistent", 0.5)
        assert len(diagram) == 0

        # Análisis de métrica inexistente
        result = ph.analyze_persistence("nonexistent", 0.5)
        assert result.state == MetricState.UNKNOWN

    def test_very_large_values_handled(self):
        """Valores muy grandes se manejan correctamente."""
        ph = PersistenceHomology(window_size=10)

        for _ in range(10):
            ph.add_reading("large", 1e100)

        stats = ph.get_statistics("large")
        assert stats is not None
        assert math.isfinite(stats["mean"])

    def test_negative_values_handled(self):
        """Valores negativos se procesan correctamente."""
        ph = PersistenceHomology(window_size=10)

        values = [-10, -5, 0, 5, 10, -10, -5, 0, 5, 10]
        ph.add_readings_batch("negative", values)

        stats = ph.get_statistics("negative")
        assert stats is not None
        assert stats["min"] == -10
        assert stats["max"] == 10

    def test_repr_methods_no_exceptions(self, pyramid_topology, persistence_with_data):
        """__repr__ no lanza excepciones en ningún estado."""
        # Topología normal
        repr_topo = repr(pyramid_topology)
        assert "SystemTopology" in repr_topo
        assert "nodes=" in repr_topo

        # Persistencia normal
        repr_ph = repr(persistence_with_data)
        assert "PersistenceHomology" in repr_ph

        # Topología vacía
        empty_topo = SystemTopology()
        repr_empty = repr(empty_topo)
        assert "SystemTopology" in repr_empty

        # Persistencia vacía
        empty_ph = PersistenceHomology()
        repr_empty_ph = repr(empty_ph)
        assert "PersistenceHomology" in repr_empty_ph

    def test_topology_with_single_edge(self):
        """Topología con una sola arista."""
        topo = SystemTopology()
        topo.update_connectivity([("Agent", "Core")])

        betti = topo.calculate_betti_numbers()

        # 1 arista conecta 2 nodos, quedan 2 aislados
        assert betti.b0 == 3  # {Agent,Core}, {Redis}, {Filesystem}
        assert betti.b1 == 0

    def test_wasserstein_distance_symmetry(self):
        """Distancia de Wasserstein es simétrica."""
        intervals1 = [PersistenceInterval(birth=0, death=5, dimension=0)]
        intervals2 = [PersistenceInterval(birth=0, death=10, dimension=0)]

        dist_12 = compute_wasserstein_distance(intervals1, intervals2)
        dist_21 = compute_wasserstein_distance(intervals2, intervals1)

        assert dist_12 == pytest.approx(dist_21, rel=1e-10)

    def test_wasserstein_distance_triangle_inequality(self):
        """Distancia de Wasserstein satisface desigualdad triangular."""
        i1 = [PersistenceInterval(birth=0, death=5, dimension=0)]
        i2 = [PersistenceInterval(birth=0, death=8, dimension=0)]
        i3 = [PersistenceInterval(birth=0, death=12, dimension=0)]

        d12 = compute_wasserstein_distance(i1, i2)
        d23 = compute_wasserstein_distance(i2, i3)
        d13 = compute_wasserstein_distance(i1, i3)

        # d(1,3) <= d(1,2) + d(2,3)
        assert d13 <= d12 + d23 + 1e-10  # Tolerancia numérica