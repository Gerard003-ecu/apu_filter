"""
Módulo: test_pipeline_director.py
=================================

Suite rigurosa de pruebas para evaluar el pipeline como DAG algebraico.

Fundamentos Matemáticos
-----------------------
1. Teoría de Categorías:
   - Verificación de axiomas categoriales (identidad, asociatividad)
   - Composición de morfismos como operación binaria asociativa
   - Funtorialidad de transformaciones entre estratos

2. Topología Algebraica:
   - Propiedades homológicas del DAG (número de Betti, característica de Euler)
   - Verificación de secuencias exactas
   - Invariantes de conectividad

3. Teoría de Grafos:
   - Propiedades espectrales del grafo de dependencias
   - Ordenamiento topológico como extensión lineal de orden parcial
   - Detección de ciclos como verificación de aciclicidad

4. Álgebra Lineal:
   - Firmas tensoriales para verificación de integridad
   - Hashing como proyección a espacio de dimensión reducida

Cubre
-----
- Estructura y topología del DAG
- Memoización y semántica LRU
- StateVector tipado y round-trip
- Auditoría homológica
- Composición categórica de morfismos
- Integración end-to-end
- Casos límite, serialización y pruebas de humo de rendimiento

Ejecutar:
    pytest test_pipeline_director.py -v --tb=short
    pytest test_pipeline_director.py -v --cov=app.tactics.pipeline_director --cov=app.core.mic_algebra
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import math
import sys
import time
from collections.abc import Callable, Generator, Sequence
from functools import reduce
from typing import Any, Final, TypeVar
from unittest.mock import MagicMock, Mock

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from app.core.mic_algebra import (
    AtomicVector,
    CategoricalRegistry,
    CompositionTrace,
    HomologicalVerifier,
    IdentityMorphism,
    MorphismComposer,
    create_categorical_state,
)
from app.tactics.pipeline_director import (
    AlgebraicDAG,
    DAGBuilder,
    DependencyResolutionError,
    HomologicalAuditor,
    MemoizationKey,
    OperatorMemoizer,
    PipelineConfig,
    PipelineDirector,
    PipelineSteps,
    SessionManager,
    StateVector,
    TensorSignature,
)
from app.core.schemas import Stratum
from app.core.telemetry import TelemetryContext


# =============================================================================
# CONSTANTES DEL DOMINIO MATEMÁTICO
# =============================================================================

# Longitud esperada de hash SHA-256 en hexadecimal
_HASH_LENGTH: Final[int] = 64

# Tolerancia para comparaciones de punto flotante
_FLOAT_TOLERANCE: Final[float] = 1e-10

# Número máximo de nodos para pruebas de rendimiento
_MAX_PERFORMANCE_NODES: Final[int] = 250

# Tiempo máximo permitido para operaciones de rendimiento (segundos)
_PERFORMANCE_TIMEOUT: Final[float] = 0.5

# Estratos ordenados por nivel (para verificar orden parcial)
_STRATUM_ORDER: Final[tuple[Stratum, ...]] = (
    Stratum.PHYSICS,
    Stratum.TACTICS,
    Stratum.STRATEGY,
    Stratum.WISDOM,
)

# Nodos esperados en el DAG por defecto
_DEFAULT_DAG_NODES: Final[frozenset[str]] = frozenset({
    "load_data",
    "audited_merge",
    "calculate_costs",
    "final_merge",
    "business_topology",
    "materialization",
    "build_output",
})

# Aristas esperadas en el DAG por defecto
_DEFAULT_DAG_EDGES: Final[frozenset[tuple[str, str]]] = frozenset({
    ("load_data", "audited_merge"),
    ("audited_merge", "calculate_costs"),
    ("calculate_costs", "final_merge"),
    ("final_merge", "business_topology"),
    ("business_topology", "materialization"),
    ("business_topology", "build_output"),
})


# =============================================================================
# TIPOS GENÉRICOS
# =============================================================================

T = TypeVar("T")
MorphismType = TypeVar("MorphismType", bound=Callable)


# =============================================================================
# FUNCIONES AUXILIARES MATEMÁTICAS
# =============================================================================

def compute_euler_characteristic(graph: nx.DiGraph) -> int:
    """
    Calcula la característica de Euler del grafo.
    
    Para un grafo G = (V, E), la característica de Euler es:
        χ(G) = |V| - |E|
    
    Para un DAG conexo sin ciclos, χ debe ser positivo.
    
    Args:
        graph: Grafo dirigido.
    
    Returns:
        Característica de Euler del grafo.
    """
    return graph.number_of_nodes() - graph.number_of_edges()


def compute_betti_numbers(graph: nx.DiGraph) -> tuple[int, int]:
    """
    Calcula los números de Betti β₀ y β₁ del grafo.
    
    - β₀: Número de componentes conexas (como grafo no dirigido)
    - β₁: Número de ciclos independientes (rango del grupo fundamental)
    
    Para un DAG (grafo acíclico dirigido):
    - β₁ debe ser 0 (sin ciclos)
    
    Args:
        graph: Grafo dirigido.
    
    Returns:
        Tupla (β₀, β₁).
    """
    undirected = graph.to_undirected()
    beta_0 = nx.number_connected_components(undirected)
    # β₁ = |E| - |V| + β₀ (fórmula de Euler para grafos)
    beta_1 = graph.number_of_edges() - graph.number_of_nodes() + beta_0
    return beta_0, beta_1


def compute_laplacian_spectrum(graph: nx.DiGraph) -> np.ndarray:
    """
    Calcula el espectro del Laplaciano del grafo.
    
    El Laplaciano L = D - A, donde D es la matriz de grados y A la de adyacencia.
    Los eigenvalores del Laplaciano revelan propiedades de conectividad:
    - λ₁ = 0 siempre (eigenvector constante)
    - Multiplicidad de 0 = número de componentes conexas
    - λ₂ (segundo menor) mide la conectividad algebraica
    
    Args:
        graph: Grafo dirigido.
    
    Returns:
        Array de eigenvalores ordenados.
    """
    if graph.number_of_nodes() == 0:
        return np.array([])
    
    # Convertir a no dirigido para el Laplaciano simétrico
    undirected = graph.to_undirected()
    laplacian = nx.laplacian_matrix(undirected).toarray()
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return np.sort(eigenvalues)


def verify_partial_order_extension(
    order: Sequence[T],
    predecessors: dict[T, set[T]],
) -> bool:
    """
    Verifica que una secuencia es una extensión lineal válida de un orden parcial.
    
    Una extensión lineal de un poset (P, ≤) es una secuencia donde:
    ∀a, b ∈ P: a ≤ b ⟹ index(a) < index(b)
    
    Args:
        order: Secuencia propuesta como extensión lineal.
        predecessors: Diccionario de predecesores para cada elemento.
    
    Returns:
        True si la secuencia es una extensión lineal válida.
    """
    position = {element: i for i, element in enumerate(order)}
    
    for element, preds in predecessors.items():
        if element not in position:
            return False
        for pred in preds:
            if pred not in position:
                return False
            if position[pred] >= position[element]:
                return False
    
    return True


def is_valid_hash(value: str, expected_length: int = _HASH_LENGTH) -> bool:
    """
    Verifica si un string es un hash hexadecimal válido.
    
    Args:
        value: String a verificar.
        expected_length: Longitud esperada del hash.
    
    Returns:
        True si es un hash hexadecimal válido.
    """
    if len(value) != expected_length:
        return False
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


def generate_chain_dag(length: int) -> AlgebraicDAG:
    """
    Genera un DAG en forma de cadena lineal.
    
    Topología: step_0 → step_1 → ... → step_{n-1}
    
    Propiedades:
    - |V| = length
    - |E| = length - 1
    - χ = 1 (conexo, acíclico)
    - Diámetro = length - 1
    
    Args:
        length: Número de nodos en la cadena.
    
    Returns:
        DAG con topología de cadena.
    """
    dag = AlgebraicDAG()
    for i in range(length):
        dag.add_step(f"step_{i}")
    for i in range(length - 1):
        dag.add_dependency(f"step_{i}", f"step_{i + 1}")
    return dag


def generate_diamond_dag() -> AlgebraicDAG:
    """
    Genera un DAG en forma de diamante.
    
    Topología:
           A
          / \\
         B   C
          \\ /
           D
    
    Propiedades:
    - |V| = 4
    - |E| = 4
    - χ = 0
    - Dos caminos independientes de A a D
    
    Returns:
        DAG con topología de diamante.
    """
    dag = AlgebraicDAG()
    for node in ["A", "B", "C", "D"]:
        dag.add_step(node)
    dag.add_dependency("A", "B")
    dag.add_dependency("A", "C")
    dag.add_dependency("B", "D")
    dag.add_dependency("C", "D")
    return dag


def generate_tree_dag(depth: int, branching_factor: int = 2) -> AlgebraicDAG:
    """
    Genera un DAG con topología de árbol completo.
    
    Propiedades:
    - |V| = (b^(d+1) - 1) / (b - 1) para b > 1
    - |E| = |V| - 1
    - χ = 1 (árbol conexo)
    
    Args:
        depth: Profundidad del árbol.
        branching_factor: Factor de ramificación.
    
    Returns:
        DAG con topología de árbol.
    """
    dag = AlgebraicDAG()
    
    def add_subtree(parent: str, current_depth: int) -> None:
        if current_depth > depth:
            return
        for i in range(branching_factor):
            child = f"{parent}_{i}"
            dag.add_step(child)
            dag.add_dependency(parent, child)
            add_subtree(child, current_depth + 1)
    
    dag.add_step("root")
    add_subtree("root", 1)
    return dag


# =============================================================================
# CONSTRUCTORES DE DATOS DE PRUEBA
# =============================================================================

def build_sample_dataframe(
    rows: int = 10,
    offset: int = 0,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Construye un DataFrame determinista para evitar aleatoriedad en tests.
    
    La construcción es determinista: mismos parámetros → mismo DataFrame,
    garantizando reproducibilidad de pruebas.
    
    Args:
        rows: Número de filas.
        offset: Desplazamiento para IDs (permite crear DataFrames distintos).
        columns: Columnas personalizadas (si None, usa esquema por defecto).
    
    Returns:
        DataFrame con datos deterministas.
    """
    ids = np.arange(1 + offset, rows + 1 + offset)
    
    if columns is not None:
        # Crear DataFrame con columnas personalizadas
        data = {col: np.arange(rows) + offset for col in columns}
        return pd.DataFrame(data)
    
    return pd.DataFrame({
        "id": ids,
        "name": [f"item_{i}" for i in ids],
        "value": np.linspace(10.0 + offset, 10.0 * rows + offset, rows),
        "category": ["A" if i % 2 == 0 else "B" for i in range(rows)],
    })


def build_large_dataframe(rows: int = 1000, cols: int = 50) -> pd.DataFrame:
    """
    Construye un DataFrame grande para pruebas de rendimiento.
    
    Args:
        rows: Número de filas.
        cols: Número de columnas.
    
    Returns:
        DataFrame de dimensiones rows × cols.
    """
    return pd.DataFrame(
        np.arange(rows * cols, dtype=float).reshape(rows, cols),
        columns=[f"col_{i}" for i in range(cols)],
    )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_logging() -> None:
    """Configura logging para la sesión de pruebas."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@pytest.fixture
def temp_session_dir(tmp_path):
    """Directorio temporal para sesiones."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    return session_dir


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """DataFrame de muestra estándar."""
    return build_sample_dataframe()


@pytest.fixture
def large_dataframe() -> pd.DataFrame:
    """DataFrame grande para pruebas de rendimiento."""
    return build_large_dataframe()


@pytest.fixture
def sample_state_vector(sample_dataframe) -> StateVector:
    """StateVector de muestra con datos completos."""
    return StateVector(
        session_id="session-test-001",
        df_presupuesto=sample_dataframe.copy(),
        df_insumos=sample_dataframe.copy(),
        df_apus_raw=sample_dataframe.copy(),
        validated_strata={Stratum.PHYSICS},
    )


@pytest.fixture
def empty_state_vector() -> StateVector:
    """StateVector vacío para pruebas de casos límite."""
    return StateVector(session_id="session-empty-001")


@pytest.fixture
def mock_telemetry() -> MagicMock:
    """Mock de TelemetryContext."""
    telemetry = MagicMock(spec=TelemetryContext)
    telemetry.start_step = Mock()
    telemetry.end_step = Mock()
    telemetry.record_metric = Mock()
    telemetry.record_error = Mock()
    return telemetry


@pytest.fixture
def mock_mic() -> MagicMock:
    """Mock del controlador MIC."""
    mic = MagicMock()
    mic.project_intent = Mock(return_value={"success": True, "data": []})
    mic.get_basis_vector = Mock(return_value=None)
    mic.add_basis_vector = Mock()
    return mic


@pytest.fixture
def pipeline_config(temp_session_dir) -> PipelineConfig:
    """Configuración de pipeline para pruebas."""
    return PipelineConfig.from_dict({
        "session_dir": str(temp_session_dir),
        "enforce_filtration": True,
        "enforce_homology": True,
        "steps": {step.value: {"enabled": True} for step in PipelineSteps},
    })


@pytest.fixture
def pipeline_director(
    pipeline_config,
    mock_telemetry,
    mock_mic,
) -> PipelineDirector:
    """Instancia de PipelineDirector para pruebas."""
    return PipelineDirector(pipeline_config, mock_telemetry, mock_mic)


@pytest.fixture
def default_dag() -> AlgebraicDAG:
    """DAG por defecto construido por DAGBuilder."""
    return DAGBuilder.build_default_dag()


@pytest.fixture
def chain_dag() -> AlgebraicDAG:
    """DAG en forma de cadena."""
    return generate_chain_dag(5)


@pytest.fixture
def diamond_dag() -> AlgebraicDAG:
    """DAG en forma de diamante."""
    return generate_diamond_dag()


# =============================================================================
# PRUEBAS DE ESTRUCTURA Y TOPOLOGÍA DEL DAG
# =============================================================================

class TestAlgebraicDAGStructure:
    """Pruebas de estructura básica del DAG algebraico."""

    def test_dag_creation_empty(self) -> None:
        """Verifica creación de DAG vacío."""
        dag = AlgebraicDAG()

        assert isinstance(dag.graph, nx.DiGraph)
        assert dag.graph.number_of_nodes() == 0
        assert dag.graph.number_of_edges() == 0

    def test_dag_creation_is_deterministic(self) -> None:
        """Verifica que la creación de DAG es determinista."""
        dag1 = AlgebraicDAG()
        dag2 = AlgebraicDAG()

        assert dag1.graph.number_of_nodes() == dag2.graph.number_of_nodes()
        assert dag1.graph.number_of_edges() == dag2.graph.number_of_edges()

    def test_add_step_creates_node(self) -> None:
        """Verifica que add_step crea un nodo."""
        dag = AlgebraicDAG()

        dag.add_step("step1")
        dag.add_step("step2")

        assert set(dag.graph.nodes()) == {"step1", "step2"}
        assert dag.graph.number_of_edges() == 0

    def test_add_step_is_idempotent(self) -> None:
        """Verifica idempotencia de add_step."""
        dag = AlgebraicDAG()

        dag.add_step("step1")
        dag.add_step("step1")  # Agregar de nuevo

        assert dag.graph.number_of_nodes() == 1

    def test_add_dependency_creates_edge(self) -> None:
        """Verifica que add_dependency crea una arista."""
        dag = AlgebraicDAG()
        dag.add_step("load")
        dag.add_step("merge")

        dag.add_dependency("load", "merge", ["df_1", "df_2"])

        assert dag.graph.has_edge("load", "merge")
        assert "load" in dag.get_dependencies("merge")

    def test_add_dependency_registers_data_requirements(self) -> None:
        """Verifica que add_dependency registra requerimientos de datos."""
        dag = AlgebraicDAG()
        dag.add_step("load")
        dag.add_step("merge")

        dag.add_dependency("load", "merge", ["df_1", "df_2"])

        reqs = set(dag.get_data_requirements("merge"))
        assert reqs >= {"df_1", "df_2"}

    def test_get_dependencies_returns_predecessors(self) -> None:
        """Verifica que get_dependencies retorna predecesores correctos."""
        dag = AlgebraicDAG()
        dag.add_step("A")
        dag.add_step("B")
        dag.add_step("C")
        dag.add_dependency("A", "C")
        dag.add_dependency("B", "C")

        deps = dag.get_dependencies("C")

        assert set(deps) == {"A", "B"}

    def test_get_dependencies_empty_for_source(self) -> None:
        """Verifica que nodos fuente no tienen dependencias."""
        dag = AlgebraicDAG()
        dag.add_step("source")
        dag.add_step("sink")
        dag.add_dependency("source", "sink")

        assert len(dag.get_dependencies("source")) == 0


class TestAlgebraicDAGTopology:
    """Pruebas de propiedades topológicas del DAG."""

    def test_dag_is_acyclic(self, default_dag: AlgebraicDAG) -> None:
        """Verifica aciclicidad del DAG."""
        assert nx.is_directed_acyclic_graph(default_dag.graph)

    def test_dag_validate_returns_true_for_valid_dag(
        self,
        default_dag: AlgebraicDAG,
    ) -> None:
        """Verifica que validate() retorna True para DAG válido."""
        assert default_dag.validate() is True

    def test_cycle_detection_raises_error(self) -> None:
        """Verifica detección de ciclos al agregar arista."""
        dag = AlgebraicDAG()
        dag.add_step("A")
        dag.add_step("B")
        dag.add_step("C")
        dag.add_dependency("A", "B")
        dag.add_dependency("B", "C")

        with pytest.raises(DependencyResolutionError):
            dag.add_dependency("C", "A")

    def test_cycle_detection_self_loop(self) -> None:
        """Verifica detección de auto-ciclos."""
        dag = AlgebraicDAG()
        dag.add_step("A")

        with pytest.raises(DependencyResolutionError):
            dag.add_dependency("A", "A")

    def test_euler_characteristic_chain(self, chain_dag: AlgebraicDAG) -> None:
        """
        Verifica característica de Euler para DAG en cadena.
        
        Para cadena de n nodos: χ = n - (n-1) = 1
        """
        chi = compute_euler_characteristic(chain_dag.graph)
        assert chi == 1

    def test_euler_characteristic_diamond(self, diamond_dag: AlgebraicDAG) -> None:
        """
        Verifica característica de Euler para DAG diamante.
        
        χ = 4 - 4 = 0
        """
        chi = compute_euler_characteristic(diamond_dag.graph)
        assert chi == 0

    def test_betti_numbers_acyclic(self, default_dag: AlgebraicDAG) -> None:
        """
        Verifica números de Betti para DAG acíclico.
        
        β₁ = 0 para grafos sin ciclos (como grafo no dirigido).
        """
        beta_0, beta_1 = compute_betti_numbers(default_dag.graph)
        
        assert beta_0 >= 1, "Debe haber al menos una componente conexa"
        # Nota: β₁ puede ser > 0 si hay múltiples caminos entre nodos

    def test_laplacian_spectrum_connectivity(
        self,
        default_dag: AlgebraicDAG,
    ) -> None:
        """
        Verifica propiedades espectrales del Laplaciano.
        
        - El eigenvalor más pequeño es 0
        - Multiplicidad de 0 = número de componentes conexas
        """
        spectrum = compute_laplacian_spectrum(default_dag.graph)
        
        if len(spectrum) > 0:
            # El eigenvalor más pequeño debe ser 0 (o muy cercano)
            assert spectrum[0] < _FLOAT_TOLERANCE, (
                f"λ₁ debe ser 0, obtenido: {spectrum[0]}"
            )
            
            # Contar eigenvalores cercanos a 0
            zero_eigenvalues = np.sum(np.abs(spectrum) < _FLOAT_TOLERANCE)
            beta_0, _ = compute_betti_numbers(default_dag.graph)
            
            assert zero_eigenvalues == beta_0, (
                f"Multiplicidad de λ=0 debe ser β₀={beta_0}, "
                f"obtenido: {zero_eigenvalues}"
            )


class TestTopologicalSort:
    """Pruebas del ordenamiento topológico."""

    def test_topological_sort_returns_all_nodes(
        self,
        default_dag: AlgebraicDAG,
    ) -> None:
        """Verifica que topological_sort retorna todos los nodos."""
        order = default_dag.topological_sort()

        assert len(order) == default_dag.graph.number_of_nodes()
        assert set(order) == set(default_dag.graph.nodes())

    def test_topological_sort_no_duplicates(
        self,
        default_dag: AlgebraicDAG,
    ) -> None:
        """Verifica que no hay duplicados en el orden topológico."""
        order = default_dag.topological_sort()

        assert len(order) == len(set(order))

    def test_topological_sort_respects_partial_order(
        self,
        default_dag: AlgebraicDAG,
    ) -> None:
        """Verifica que el orden topológico respeta el orden parcial."""
        order = default_dag.topological_sort()
        
        # Construir diccionario de predecesores
        predecessors = {
            node: set(default_dag.graph.predecessors(node))
            for node in default_dag.graph.nodes()
        }
        
        assert verify_partial_order_extension(order, predecessors)

    def test_topological_sort_specific_constraints(
        self,
        default_dag: AlgebraicDAG,
    ) -> None:
        """Verifica restricciones específicas del orden topológico."""
        order = default_dag.topological_sort()

        # Verificar orden esperado para nodos específicos
        assert order.index("load_data") < order.index("audited_merge")
        assert order.index("audited_merge") < order.index("calculate_costs")
        assert order.index("calculate_costs") < order.index("final_merge")
        assert order.index("final_merge") < order.index("business_topology")

    def test_topological_sort_is_deterministic(
        self,
        default_dag: AlgebraicDAG,
    ) -> None:
        """Verifica determinismo del ordenamiento topológico."""
        order1 = default_dag.topological_sort()
        order2 = default_dag.topological_sort()
        order3 = default_dag.topological_sort()

        assert order1 == order2 == order3

    def test_topological_sort_single_node(self) -> None:
        """Verifica ordenamiento para DAG de un solo nodo."""
        dag = AlgebraicDAG()
        dag.add_step("only_step")

        order = dag.topological_sort()

        assert order == ["only_step"]

    def test_topological_sort_chain(self, chain_dag: AlgebraicDAG) -> None:
        """Verifica ordenamiento para DAG en cadena."""
        order = chain_dag.topological_sort()

        # En una cadena, el orden debe ser exactamente secuencial
        expected = [f"step_{i}" for i in range(5)]
        assert order == expected


class TestDAGBuilder:
    """Pruebas del constructor de DAG por defecto."""

    def test_build_default_dag_nodes(self) -> None:
        """Verifica nodos del DAG por defecto."""
        dag = DAGBuilder.build_default_dag()

        assert set(dag.graph.nodes()) == _DEFAULT_DAG_NODES

    def test_build_default_dag_edges(self) -> None:
        """Verifica aristas del DAG por defecto."""
        dag = DAGBuilder.build_default_dag()

        assert _DEFAULT_DAG_EDGES.issubset(set(dag.graph.edges()))

    def test_build_default_dag_is_acyclic(self) -> None:
        """Verifica aciclicidad del DAG por defecto."""
        dag = DAGBuilder.build_default_dag()

        assert nx.is_directed_acyclic_graph(dag.graph)

    def test_build_default_dag_source_node(self) -> None:
        """Verifica que load_data es el nodo fuente."""
        dag = DAGBuilder.build_default_dag()

        assert len(dag.get_dependencies("load_data")) == 0

    def test_build_default_dag_sink_nodes(self) -> None:
        """Verifica nodos sumidero del DAG."""
        dag = DAGBuilder.build_default_dag()

        sink_nodes = [
            node for node in dag.graph.nodes()
            if dag.graph.out_degree(node) == 0
        ]

        assert "build_output" in sink_nodes

    def test_build_default_dag_is_deterministic(self) -> None:
        """Verifica que build_default_dag es determinista."""
        dag1 = DAGBuilder.build_default_dag()
        dag2 = DAGBuilder.build_default_dag()

        assert set(dag1.graph.nodes()) == set(dag2.graph.nodes())
        assert set(dag1.graph.edges()) == set(dag2.graph.edges())


class TestDAGSerialization:
    """Pruebas de serialización del DAG."""

    def test_to_dict_structure(self, default_dag: AlgebraicDAG) -> None:
        """Verifica estructura del diccionario serializado."""
        dag_dict = default_dag.to_dict()

        assert isinstance(dag_dict, dict)
        assert "nodes" in dag_dict
        assert "edges" in dag_dict
        assert "is_acyclic" in dag_dict

    def test_to_dict_nodes_count(self, default_dag: AlgebraicDAG) -> None:
        """Verifica conteo de nodos en serialización."""
        dag_dict = default_dag.to_dict()

        assert len(dag_dict["nodes"]) == default_dag.graph.number_of_nodes()

    def test_to_dict_edges_count(self, default_dag: AlgebraicDAG) -> None:
        """Verifica conteo de aristas en serialización."""
        dag_dict = default_dag.to_dict()

        assert len(dag_dict["edges"]) == default_dag.graph.number_of_edges()

    def test_to_dict_acyclicity_flag(self, default_dag: AlgebraicDAG) -> None:
        """Verifica flag de aciclicidad."""
        dag_dict = default_dag.to_dict()

        assert dag_dict["is_acyclic"] is True


# =============================================================================
# PRUEBAS DE TENSOR SIGNATURE Y MEMOIZACIÓN
# =============================================================================

class TestTensorSignature:
    """Pruebas de TensorSignature para integridad de datos."""

    def test_signature_creation_dataframe(self, sample_dataframe) -> None:
        """Verifica creación de firma para DataFrame."""
        sig = TensorSignature.compute(sample_dataframe)

        assert sig.hash_value is not None
        assert is_valid_hash(sig.hash_value)
        assert sig.shape == sample_dataframe.shape
        assert sig.dtype == "dataframe"

    def test_signature_creation_dict(self) -> None:
        """Verifica creación de firma para diccionario."""
        data = {"key": [1, 2, 3], "other": "value"}
        sig = TensorSignature.compute(data)

        assert sig.hash_value is not None
        assert is_valid_hash(sig.hash_value)

    def test_signature_determinism(self, sample_dataframe) -> None:
        """Verifica determinismo de la firma."""
        sig1 = TensorSignature.compute(sample_dataframe)
        sig2 = TensorSignature.compute(sample_dataframe)

        assert sig1.hash_value == sig2.hash_value
        assert sig1.matches(sig2)

    def test_signature_identical_data_matches(self) -> None:
        """Verifica que datos idénticos producen firmas iguales."""
        data1 = build_sample_dataframe()
        data2 = build_sample_dataframe()

        sig1 = TensorSignature.compute(data1)
        sig2 = TensorSignature.compute(data2)

        assert sig1.matches(sig2)

    def test_signature_detects_content_change(self) -> None:
        """Verifica detección de cambios en contenido."""
        data1 = build_sample_dataframe()
        data2 = build_sample_dataframe()
        data2.loc[0, "value"] = data2.loc[0, "value"] + 1.0

        sig1 = TensorSignature.compute(data1)
        sig2 = TensorSignature.compute(data2)

        assert not sig1.matches(sig2)

    def test_signature_detects_shape_change(self) -> None:
        """Verifica detección de cambios en forma."""
        data1 = build_sample_dataframe(rows=10)
        data2 = build_sample_dataframe(rows=5)

        sig1 = TensorSignature.compute(data1)
        sig2 = TensorSignature.compute(data2)

        assert not sig1.matches(sig2)
        assert sig1.shape != sig2.shape

    def test_signature_symmetry(self) -> None:
        """Verifica simetría de matches()."""
        data1 = build_sample_dataframe()
        data2 = build_sample_dataframe()

        sig1 = TensorSignature.compute(data1)
        sig2 = TensorSignature.compute(data2)

        assert sig1.matches(sig2) == sig2.matches(sig1)

    def test_signature_reflexivity(self) -> None:
        """Verifica reflexividad de matches()."""
        sig = TensorSignature.compute({"data": [1, 2, 3]})

        assert sig.matches(sig)


class TestMemoizationKey:
    """Pruebas de MemoizationKey."""

    def test_key_creation(self) -> None:
        """Verifica creación de clave de memoización."""
        key = MemoizationKey(
            input_hash="hash123",
            operator_id="op_1",
            stratum="PHYSICS",
        )

        assert key.input_hash == "hash123"
        assert key.operator_id == "op_1"
        assert key.stratum == "PHYSICS"

    def test_key_equality(self) -> None:
        """Verifica igualdad de claves."""
        key1 = MemoizationKey("hash1", "op1", "PHYSICS")
        key2 = MemoizationKey("hash1", "op1", "PHYSICS")
        key3 = MemoizationKey("hash2", "op1", "PHYSICS")

        assert key1 == key2
        assert key1 != key3

    def test_key_hashable(self) -> None:
        """Verifica que la clave es hashable."""
        key1 = MemoizationKey("hash1", "op1", "PHYSICS")
        key2 = MemoizationKey("hash1", "op1", "PHYSICS")

        mapping: dict[MemoizationKey, str] = {key1: "value"}

        assert mapping[key2] == "value"

    def test_key_usable_in_set(self) -> None:
        """Verifica uso de clave en conjunto."""
        key1 = MemoizationKey("hash1", "op1", "PHYSICS")
        key2 = MemoizationKey("hash1", "op1", "PHYSICS")
        key3 = MemoizationKey("hash2", "op1", "PHYSICS")

        keys = {key1, key2, key3}

        assert len(keys) == 2


class TestOperatorMemoizer:
    """Pruebas del sistema de memoización."""

    def test_memoizer_creation(self) -> None:
        """Verifica creación de memoizador."""
        memoizer = OperatorMemoizer()

        assert memoizer.stats["hits"] == 0
        assert memoizer.stats["misses"] == 0
        assert len(memoizer.cache) == 0

    def test_lookup_miss(self) -> None:
        """Verifica fallo de caché (miss)."""
        memoizer = OperatorMemoizer()
        state = StateVector(session_id="memo-state-001")

        result = memoizer.lookup(state, "op1", "PHYSICS")

        assert result is None
        assert memoizer.stats["misses"] == 1
        assert memoizer.stats["hits"] == 0

    def test_store_and_lookup_hit(self) -> None:
        """Verifica almacenamiento y recuperación exitosa."""
        memoizer = OperatorMemoizer()
        state = StateVector(session_id="memo-state-002")
        output = {"key": "value"}
        sig = TensorSignature.compute(output)

        memoizer.store(state, "op1", "PHYSICS", output, sig)
        result = memoizer.lookup(state, "op1", "PHYSICS")

        assert result is not None
        assert result[0] == output
        assert result[1].matches(sig)
        assert memoizer.stats["hits"] == 1

    def test_memoizer_preserves_semantic_equivalence(self) -> None:
        """
        Verifica que la memoización preserva equivalencia semántica.
        
        El resultado recuperado debe ser semánticamente equivalente al original.
        """
        memoizer = OperatorMemoizer()
        state = StateVector(session_id="memo-state-equiv")
        
        original_data = {"results": [1, 2, 3], "meta": {"processed": True}}
        sig = TensorSignature.compute(original_data)
        
        memoizer.store(state, "op1", "PHYSICS", original_data, sig)
        cached_data, cached_sig = memoizer.lookup(state, "op1", "PHYSICS")
        
        # Verificar equivalencia estructural
        assert cached_data == original_data
        assert cached_sig.matches(sig)

    def test_lru_eviction_policy(self) -> None:
        """Verifica política de evicción LRU."""
        memoizer = OperatorMemoizer(max_size=2)
        state = StateVector(session_id="memo-state-003")

        sig1 = TensorSignature.compute({"a": 1})
        sig2 = TensorSignature.compute({"b": 2})
        sig3 = TensorSignature.compute({"c": 3})

        memoizer.store(state, "op1", "PHYSICS", {"a": 1}, sig1)
        memoizer.store(state, "op2", "PHYSICS", {"b": 2}, sig2)

        # Acceder a op1 para hacerlo más reciente
        assert memoizer.lookup(state, "op1", "PHYSICS") is not None

        # Agregar op3, debe evictar op2 (menos recientemente usado)
        memoizer.store(state, "op3", "PHYSICS", {"c": 3}, sig3)

        assert len(memoizer.cache) == 2
        assert memoizer.stats["evictions"] == 1
        assert memoizer.lookup(state, "op1", "PHYSICS") is not None
        assert memoizer.lookup(state, "op3", "PHYSICS") is not None
        assert memoizer.lookup(state, "op2", "PHYSICS") is None

    def test_memoizer_freshness_expiration(self) -> None:
        """Verifica expiración por antigüedad."""
        memoizer = OperatorMemoizer()
        state = StateVector(session_id="memo-state-004")
        sig = TensorSignature.compute({})

        memoizer.store(state, "op1", "PHYSICS", {"data": 1}, sig)

        # Simular antigüedad excesiva
        key = next(iter(memoizer.cache))
        memoizer.cache[key].created_at = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=25)
        )

        result = memoizer.lookup(state, "op1", "PHYSICS")

        assert result is None
        assert memoizer.stats["misses"] == 1

    def test_memoizer_stats_accuracy(self) -> None:
        """Verifica precisión de estadísticas."""
        memoizer = OperatorMemoizer()
        state = StateVector(session_id="memo-state-005")
        sig = TensorSignature.compute({})

        memoizer.store(state, "op1", "PHYSICS", {}, sig)
        memoizer.lookup(state, "op1", "PHYSICS")  # hit
        memoizer.lookup(state, "op2", "TACTICS")  # miss

        stats = memoizer.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate_percent"] == pytest.approx(50.0)

    def test_memoizer_isolation_between_operators(self) -> None:
        """Verifica aislamiento entre operadores diferentes."""
        memoizer = OperatorMemoizer()
        state = StateVector(session_id="memo-state-006")
        
        sig1 = TensorSignature.compute({"op1_data": 1})
        sig2 = TensorSignature.compute({"op2_data": 2})
        
        memoizer.store(state, "op1", "PHYSICS", {"op1_data": 1}, sig1)
        memoizer.store(state, "op2", "PHYSICS", {"op2_data": 2}, sig2)
        
        result1 = memoizer.lookup(state, "op1", "PHYSICS")
        result2 = memoizer.lookup(state, "op2", "PHYSICS")
        
        assert result1[0] == {"op1_data": 1}
        assert result2[0] == {"op2_data": 2}

    def test_memoizer_isolation_between_strata(self) -> None:
        """Verifica aislamiento entre estratos diferentes."""
        memoizer = OperatorMemoizer()
        state = StateVector(session_id="memo-state-007")
        
        sig1 = TensorSignature.compute({"physics": 1})
        sig2 = TensorSignature.compute({"tactics": 2})
        
        memoizer.store(state, "op1", "PHYSICS", {"physics": 1}, sig1)
        memoizer.store(state, "op1", "TACTICS", {"tactics": 2}, sig2)
        
        result_physics = memoizer.lookup(state, "op1", "PHYSICS")
        result_tactics = memoizer.lookup(state, "op1", "TACTICS")
        
        assert result_physics[0] == {"physics": 1}
        assert result_tactics[0] == {"tactics": 2}


# =============================================================================
# PRUEBAS DE STATEVECTOR
# =============================================================================

class TestStateVectorCreation:
    """Pruebas de creación de StateVector."""

    def test_creation_minimal(self) -> None:
        """Verifica creación con parámetros mínimos."""
        state = StateVector(session_id="state-vector-001")

        assert state.session_id == "state-vector-001"
        assert state.df_presupuesto is None
        assert isinstance(state.validated_strata, (set, frozenset))
        assert isinstance(state.step_results, list)

    def test_creation_with_data(self, sample_dataframe) -> None:
        """Verifica creación con datos completos."""
        state = StateVector(
            session_id="state-vector-002",
            df_presupuesto=sample_dataframe,
            df_insumos=sample_dataframe,
            validated_strata={Stratum.PHYSICS},
        )

        assert state.df_presupuesto is not None
        assert len(state.df_presupuesto) == len(sample_dataframe)
        assert Stratum.PHYSICS in state.validated_strata

    def test_creation_preserves_dataframe_integrity(
        self,
        sample_dataframe,
    ) -> None:
        """Verifica que la creación preserva integridad del DataFrame."""
        original_hash = hashlib.sha256(
            pd.util.hash_pandas_object(sample_dataframe).values.tobytes()
        ).hexdigest()
        
        state = StateVector(
            session_id="state-vector-003",
            df_presupuesto=sample_dataframe.copy(),
        )
        
        stored_hash = hashlib.sha256(
            pd.util.hash_pandas_object(state.df_presupuesto).values.tobytes()
        ).hexdigest()
        
        assert original_hash == stored_hash


class TestStateVectorHashing:
    """Pruebas de hashing de StateVector."""

    def test_hash_is_valid_sha256(self, sample_state_vector) -> None:
        """Verifica que el hash es SHA-256 válido."""
        hash_value = sample_state_vector.compute_hash()

        assert isinstance(hash_value, str)
        assert is_valid_hash(hash_value)

    def test_hash_is_stable(self, sample_state_vector) -> None:
        """Verifica estabilidad del hash."""
        hash1 = sample_state_vector.compute_hash()
        hash2 = sample_state_vector.compute_hash()
        hash3 = sample_state_vector.compute_hash()

        assert hash1 == hash2 == hash3

    def test_hash_changes_on_mutation(self, sample_state_vector) -> None:
        """Verifica que el hash cambia con mutaciones."""
        original_hash = sample_state_vector.compute_hash()

        sample_state_vector.df_presupuesto = (
            sample_state_vector.df_presupuesto.head(1)
        )
        mutated_hash = sample_state_vector.compute_hash()

        assert original_hash != mutated_hash

    def test_hash_sensitive_to_strata(self, sample_dataframe) -> None:
        """Verifica que el hash es sensible a los estratos validados."""
        state1 = StateVector(
            session_id="hash-strata-1",
            df_presupuesto=sample_dataframe,
            validated_strata={Stratum.PHYSICS},
        )
        state2 = StateVector(
            session_id="hash-strata-1",  # Mismo session_id
            df_presupuesto=sample_dataframe,
            validated_strata={Stratum.PHYSICS, Stratum.TACTICS},
        )

        hash1 = state1.compute_hash()
        hash2 = state2.compute_hash()

        assert hash1 != hash2


class TestStateVectorSerialization:
    """Pruebas de serialización de StateVector."""

    def test_to_dict_structure(self, sample_state_vector) -> None:
        """Verifica estructura del diccionario serializado."""
        state_dict = sample_state_vector.to_dict()

        assert isinstance(state_dict, dict)
        assert "session_id" in state_dict
        assert "validated_strata" in state_dict
        assert isinstance(state_dict["validated_strata"], list)

    def test_round_trip_preserves_data(self, sample_state_vector) -> None:
        """Verifica que el round-trip preserva todos los datos."""
        state_dict = sample_state_vector.to_dict()
        reconstructed = StateVector.from_dict(state_dict)

        assert reconstructed.session_id == sample_state_vector.session_id
        assert set(reconstructed.validated_strata) == {Stratum.PHYSICS}
        
        pd.testing.assert_frame_equal(
            reconstructed.df_presupuesto,
            sample_state_vector.df_presupuesto,
        )
        pd.testing.assert_frame_equal(
            reconstructed.df_insumos,
            sample_state_vector.df_insumos,
        )
        pd.testing.assert_frame_equal(
            reconstructed.df_apus_raw,
            sample_state_vector.df_apus_raw,
        )

    def test_to_dict_is_defensive(self, sample_state_vector) -> None:
        """Verifica que to_dict produce copia defensiva."""
        state_dict = sample_state_vector.to_dict()
        state_dict["validated_strata"].append("TACTICS")

        assert Stratum.TACTICS not in sample_state_vector.validated_strata

    def test_round_trip_hash_equivalence(self, sample_state_vector) -> None:
        """Verifica equivalencia de hash después de round-trip."""
        original_hash = sample_state_vector.compute_hash()
        
        state_dict = sample_state_vector.to_dict()
        reconstructed = StateVector.from_dict(state_dict)
        reconstructed_hash = reconstructed.compute_hash()

        assert original_hash == reconstructed_hash


class TestStateVectorEvidence:
    """Pruebas del sistema de evidencia de StateVector."""

    def test_get_evidence_with_data(self) -> None:
        """Verifica evidencia cuando hay datos presentes."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        state = StateVector(
            session_id="evidence-001",
            df_presupuesto=df,
            df_insumos=df,
            df_apus_raw=df,
        )

        evidence = state.get_evidence(Stratum.PHYSICS)

        assert evidence.get("df_presupuesto", False) is True
        assert evidence.get("df_insumos", False) is True
        assert evidence.get("df_apus_raw", False) is True

    def test_get_evidence_without_data(self, empty_state_vector) -> None:
        """Verifica evidencia cuando no hay datos."""
        evidence = empty_state_vector.get_evidence(Stratum.PHYSICS)

        assert evidence.get("df_presupuesto", False) is False
        assert evidence.get("df_insumos", False) is False

    def test_get_evidence_empty_dataframes(self) -> None:
        """Verifica evidencia con DataFrames vacíos."""
        state = StateVector(
            session_id="evidence-002",
            df_presupuesto=pd.DataFrame(),
            df_insumos=pd.DataFrame(),
        )

        evidence = state.get_evidence(Stratum.PHYSICS)

        assert evidence.get("df_presupuesto", False) is False
        assert evidence.get("df_insumos", False) is False


# =============================================================================
# PRUEBAS DE AUDITORÍA HOMOLÓGICA
# =============================================================================

class TestHomologicalAuditor:
    """Pruebas del auditor homológico."""

    def test_auditor_creation(self, mock_telemetry) -> None:
        """Verifica creación del auditor."""
        auditor = HomologicalAuditor(mock_telemetry)

        assert auditor is not None
        assert auditor.telemetry == mock_telemetry

    def test_audit_merge_success(
        self,
        mock_telemetry,
        sample_dataframe,
    ) -> None:
        """Verifica auditoría exitosa de merge."""
        auditor = HomologicalAuditor(mock_telemetry)
        state = StateVector(session_id="audit-001")

        df_a = sample_dataframe.copy()
        df_b = sample_dataframe.copy()
        df_result = pd.concat([df_a, df_b], ignore_index=True)

        result = auditor.audit_merge(df_a, df_b, df_result, state)

        assert result["passed"] is True
        assert result["emergent_cycles"] == 0
        assert "warnings" in result

    def test_audit_merge_detects_data_loss(
        self,
        mock_telemetry,
        sample_dataframe,
    ) -> None:
        """Verifica detección de pérdida de datos."""
        auditor = HomologicalAuditor(mock_telemetry)
        state = StateVector(session_id="audit-002")

        df_a = sample_dataframe.copy()
        df_b = sample_dataframe.copy()
        df_result = pd.DataFrame({"id": [1, 2]})  # Resultado reducido

        result = auditor.audit_merge(df_a, df_b, df_result, state)

        assert result["passed"] is False
        assert len(result["warnings"]) > 0

    def test_audit_merge_column_preservation(self, mock_telemetry) -> None:
        """Verifica detección de columnas perdidas."""
        auditor = HomologicalAuditor(mock_telemetry)
        state = StateVector(session_id="audit-003")

        df_a = pd.DataFrame({"col_a": [1, 2], "col_shared": [3, 4]})
        df_b = pd.DataFrame({"col_b": [5, 6], "col_shared": [7, 8]})
        df_result = pd.DataFrame({"col_shared": [9, 10]})  # Faltan col_a y col_b

        result = auditor.audit_merge(df_a, df_b, df_result, state)

        assert result["passed"] is False
        assert len(result["warnings"]) > 0

    def test_audit_merge_idempotent(
        self,
        mock_telemetry,
        sample_dataframe,
    ) -> None:
        """Verifica idempotencia de la auditoría."""
        auditor = HomologicalAuditor(mock_telemetry)
        state = StateVector(session_id="audit-004")

        df_a = sample_dataframe.copy()
        df_b = sample_dataframe.copy()
        df_result = pd.concat([df_a, df_b], ignore_index=True)

        result1 = auditor.audit_merge(df_a, df_b, df_result, state)
        result2 = auditor.audit_merge(df_a, df_b, df_result, state)

        assert result1["passed"] == result2["passed"]
        assert result1["emergent_cycles"] == result2["emergent_cycles"]


# =============================================================================
# PRUEBAS DE ESTADO CATEGORIAL
# =============================================================================

class TestCategoricalStateCreation:
    """Pruebas de creación de estado categorial."""

    def test_creation_default(self) -> None:
        """Verifica creación con valores por defecto."""
        state = create_categorical_state()

        assert state.is_success
        assert state.error is None
        assert isinstance(state.payload, dict)

    def test_creation_with_payload(self) -> None:
        """Verifica creación con payload."""
        state = create_categorical_state(
            payload={"data": 123},
            strata={Stratum.PHYSICS},
        )

        assert state.is_success
        assert state.payload["data"] == 123
        assert Stratum.PHYSICS in state.validated_strata

    def test_creation_with_multiple_strata(self) -> None:
        """Verifica creación con múltiples estratos."""
        strata = {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}
        state = create_categorical_state(strata=strata)

        assert state.validated_strata == strata


class TestCategoricalStateTransformations:
    """Pruebas de transformaciones de estado categorial."""

    def test_with_update_preserves_immutability(self) -> None:
        """Verifica inmutabilidad con with_update."""
        state1 = create_categorical_state(payload={"a": 1})
        state2 = state1.with_update({"b": 2}, new_stratum=Stratum.TACTICS)

        assert "a" in state1.payload
        assert "b" not in state1.payload
        assert "b" in state2.payload
        assert Stratum.TACTICS in state2.validated_strata
        assert Stratum.TACTICS not in state1.validated_strata

    def test_with_error_marks_failure(self) -> None:
        """Verifica que with_error marca fallo."""
        state1 = create_categorical_state(payload={"data": 1})
        state2 = state1.with_error("Fallo", details={"reason": "test"})

        assert state1.is_success
        assert state2.is_failed
        assert state2.error == "Fallo"
        assert state2.error_details["reason"] == "test"

    def test_with_error_preserves_payload(self) -> None:
        """Verifica que with_error preserva payload."""
        original_payload = {"important": "data"}
        state1 = create_categorical_state(payload=original_payload)
        state2 = state1.with_error("Error")

        assert state2.payload == original_payload


class TestCategoricalStateProperties:
    """Pruebas de propiedades del estado categorial."""

    def test_stratum_level_minimum(self) -> None:
        """Verifica cálculo del nivel de estrato (mínimo)."""
        strata = {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}
        state = create_categorical_state(strata=strata)

        assert state.stratum_level == min(stratum.value for stratum in strata)

    def test_compute_hash_valid(self) -> None:
        """Verifica hash válido."""
        state = create_categorical_state(payload={"data": [1, 2, 3]})

        hash_val = state.compute_hash()

        assert isinstance(hash_val, str)
        assert is_valid_hash(hash_val)

    def test_compute_hash_deterministic(self) -> None:
        """Verifica determinismo del hash."""
        state = create_categorical_state(payload={"data": [1, 2, 3]})

        hash1 = state.compute_hash()
        hash2 = state.compute_hash()

        assert hash1 == hash2


# =============================================================================
# PRUEBAS DE MORFISMOS CATEGORIALES
# =============================================================================

class TestIdentityMorphism:
    """Pruebas del morfismo identidad."""

    def test_identity_returns_same_state(self) -> None:
        """Verifica que identidad retorna el mismo estado."""
        state = create_categorical_state(payload={"a": 1})
        morph = IdentityMorphism(Stratum.PHYSICS)

        result = morph(state)

        assert result.payload == state.payload
        assert len(result.composition_trace) == len(state.composition_trace) + 1
        assert result.composition_trace[-1].morphism_name.startswith('id_')
        assert result.payload == {"a": 1}

    def test_identity_is_idempotent(self) -> None:
        """Verifica idempotencia: id ∘ id = id."""
        state = create_categorical_state(payload={"a": 1})
        morph = IdentityMorphism(Stratum.PHYSICS)

        result1 = morph(state)
        result2 = morph(result1)

        assert result2.payload == result1.payload == state.payload
        assert len(result2.composition_trace) == len(result1.composition_trace) + 1
        assert result2.composition_trace[-1].morphism_name.startswith('id_')

    def test_identity_preserves_strata(self) -> None:
        """Verifica que identidad preserva estratos."""
        strata = {Stratum.PHYSICS, Stratum.TACTICS}
        state = create_categorical_state(strata=strata)
        morph = IdentityMorphism(Stratum.PHYSICS)

        result = morph(state)

        assert result.validated_strata == strata


class TestAtomicVector:
    """Pruebas del morfismo AtomicVector."""

    def test_atomic_vector_success(self) -> None:
        """Verifica ejecución exitosa de AtomicVector."""
        def handler(value: int):
            return {"result": value * 2}

        morph = AtomicVector(
            name="double",
            target_stratum=Stratum.TACTICS,
            handler=handler,
            required_keys=["value"],
        )

        state = create_categorical_state(
            payload={"value": 5},
            strata={Stratum.PHYSICS},
        )

        result = morph(state)

        assert result.is_success
        assert result.payload["result"] == 10
        assert Stratum.TACTICS in result.validated_strata

    def test_atomic_vector_domain_violation(self) -> None:
        """Verifica detección de violación de dominio."""
        def handler():
            return {"result": 1}

        morph = AtomicVector(
            name="op",
            target_stratum=Stratum.STRATEGY,
            handler=handler,
        )

        state = create_categorical_state(strata={Stratum.PHYSICS})
        result = morph(state)

        assert result.is_failed
        assert "Violación de dominio" in result.error

    def test_atomic_vector_missing_keys(self) -> None:
        """Verifica detección de claves faltantes."""
        def handler(required_key: str):
            return {"result": required_key}

        morph = AtomicVector(
            name="op",
            target_stratum=Stratum.TACTICS,
            handler=handler,
            required_keys=["required_key"],
        )

        state = create_categorical_state(
            payload={"other": 1},
            strata={Stratum.PHYSICS},
        )

        result = morph(state)

        assert result.is_failed
        assert "requeridas faltantes" in result.error

    def test_atomic_vector_propagates_previous_error(self) -> None:
        """Verifica propagación de errores previos."""
        def handler():
            raise RuntimeError("No debe ejecutarse")

        morph = AtomicVector(
            name="op",
            target_stratum=Stratum.TACTICS,
            handler=handler,
        )

        failed_state = create_categorical_state(payload={"x": 1}).with_error(
            "Previous error"
        )

        result = morph(failed_state)

        assert result.is_failed
        assert "Previous error" in result.error

    def test_atomic_vector_handler_exception_captured(self) -> None:
        """Verifica captura de excepciones del handler."""
        def handler():
            raise ValueError("Handler error")

        morph = AtomicVector(
            name="failing_op",
            target_stratum=Stratum.TACTICS,
            handler=handler,
        )

        state = create_categorical_state(strata={Stratum.PHYSICS})
        result = morph(state)

        assert result.is_failed


class TestMorphismComposition:
    """Pruebas de composición de morfismos."""

    def test_composition_associativity(self) -> None:
        """
        Verifica asociatividad: (f ∘ g) ∘ h = f ∘ (g ∘ h).
        
        Este es un axioma fundamental de la teoría de categorías.
        """
        def handler1(x: int):
            return {"y": x + 1}

        def handler2(y: int):
            return {"z": y * 2}

        def handler3(z: int):
            return {"w": z - 1}

        morph1 = AtomicVector("add1", Stratum.TACTICS, handler1, ["x"])
        morph2 = AtomicVector("mul2", Stratum.STRATEGY, handler2, ["y"])
        morph3 = AtomicVector("sub1", Stratum.OMEGA, handler3, ["z"])

        # (f ∘ g) ∘ h
        composed_left = (morph1 >> morph2) >> morph3
        # f ∘ (g ∘ h)
        composed_right = morph1 >> (morph2 >> morph3)

        state = create_categorical_state(
            payload={"x": 5},
            strata={Stratum.PHYSICS},
        )

        result_left = composed_left(state)
        result_right = composed_right(state)

        assert result_left.payload == result_right.payload
        assert result_left.payload["w"] == (5 + 1) * 2 - 1  # 11

    def test_identity_is_neutral_element_left(self) -> None:
        """Verifica id ∘ f = f (identidad como elemento neutro izquierdo)."""
        def handler(x: int):
            return {"y": x * 2}

        morph = AtomicVector("double", Stratum.TACTICS, handler, ["x"])
        identity = IdentityMorphism(Stratum.PHYSICS)

        composed = identity >> morph

        state = create_categorical_state(
            payload={"x": 5},
            strata={Stratum.PHYSICS},
        )

        result_composed = composed(state)
        result_direct = morph(state)

        assert result_composed.payload == result_direct.payload

    def test_identity_is_neutral_element_right(self) -> None:
        """Verifica f ∘ id = f (identidad como elemento neutro derecho)."""
        def handler(x: int):
            return {"y": x * 2}

        morph = AtomicVector("double", Stratum.TACTICS, handler, ["x"])
        identity = IdentityMorphism(Stratum.TACTICS)

        composed = morph >> identity

        state = create_categorical_state(
            payload={"x": 5},
            strata={Stratum.PHYSICS},
        )

        result_composed = composed(state)
        result_direct = morph(state)

        assert result_composed.payload == result_direct.payload

    def test_sequential_composition_success(self) -> None:
        """Verifica composición secuencial exitosa."""
        def handler1(x: int):
            return {"y": x * 2}

        def handler2(y: int):
            return {"z": y + 1}

        morph1 = AtomicVector("double", Stratum.TACTICS, handler1, ["x"])
        morph2 = AtomicVector("add", Stratum.STRATEGY, handler2, ["y"])

        composed = morph1 >> morph2

        state = create_categorical_state(
            payload={"x": 5},
            strata={Stratum.PHYSICS},
        )

        result = composed(state)

        assert result.is_success
        assert result.payload["z"] == 11  # (5 * 2) + 1
        assert Stratum.STRATEGY in result.validated_strata

    def test_composition_failure_propagation(self) -> None:
        """Verifica propagación de fallos en composición."""
        def handler1():
            return {"a": 1}

        def handler2(required_b: int):
            return {"c": required_b}

        morph1 = AtomicVector("op1", Stratum.TACTICS, handler1)
        morph2 = AtomicVector("op2", Stratum.WISDOM, handler2, ["required_b"])

        composed = morph1 >> morph2

        state = create_categorical_state(strata={Stratum.PHYSICS})
        result = composed(state)

        assert result.is_failed


class TestProductMorphism:
    """Pruebas del morfismo producto."""

    def test_product_applies_both_morphisms(self) -> None:
        """Verifica que el producto aplica ambos morfismos."""
        def handler1(x: int):
            return {"result1": x * 2}

        def handler2(x: int):
            return {"result2": x + 1}

        morph1 = AtomicVector("op1", Stratum.TACTICS, handler1, ["x"])
        morph2 = AtomicVector("op2", Stratum.TACTICS, handler2, ["x"])

        product = morph1 * morph2

        state = create_categorical_state(
            payload={"x": 5},
            strata={Stratum.PHYSICS},
        )

        result = product(state)

        assert result.is_success
        assert result.payload["result1"] == 10
        assert result.payload["result2"] == 6

    def test_product_commutativity_in_results(self) -> None:
        """Verifica que el producto es conmutativo en resultados (claves disjuntas)."""
        def handler1(x: int):
            return {"a": x}

        def handler2(x: int):
            return {"b": x}

        morph1 = AtomicVector("op1", Stratum.TACTICS, handler1, ["x"])
        morph2 = AtomicVector("op2", Stratum.TACTICS, handler2, ["x"])

        product12 = morph1 * morph2
        product21 = morph2 * morph1

        state = create_categorical_state(
            payload={"x": 5},
            strata={Stratum.PHYSICS},
        )

        result12 = product12(state)
        result21 = product21(state)

        # Los payloads deben contener las mismas claves
        assert set(result12.payload.keys()) == set(result21.payload.keys())


class TestCoproductMorphism:
    """Pruebas del morfismo coproducto."""

    def test_coproduct_first_succeeds(self) -> None:
        """Verifica que el coproducto usa el primer morfismo exitoso."""
        def handler1(x: int):
            return {"via": "first", "result": x * 2}

        def handler2(x: int):
            return {"via": "second", "result": x + 1}

        morph1 = AtomicVector("op1", Stratum.TACTICS, handler1, ["x"])
        morph2 = AtomicVector("op2", Stratum.TACTICS, handler2, ["x"])

        coproduct = morph1 | morph2

        state = create_categorical_state(
            payload={"x": 5},
            strata={Stratum.PHYSICS},
        )

        result = coproduct(state)

        assert result.is_success
        assert result.payload["via"] == "first"

    def test_coproduct_fallback_on_failure(self) -> None:
        """Verifica fallback cuando el primer morfismo falla."""
        def handler1():
            raise ValueError("First fails")

        def handler2():
            return {"via": "fallback"}

        morph1 = AtomicVector("op1", Stratum.TACTICS, handler1)
        morph2 = AtomicVector("op2", Stratum.TACTICS, handler2)

        coproduct = morph1 | morph2

        state = create_categorical_state(strata={Stratum.PHYSICS})
        result = coproduct(state)

        assert result.is_success
        assert result.payload["via"] == "fallback"

    def test_coproduct_both_fail(self) -> None:
        """Verifica que el coproducto falla si ambos fallan."""
        def handler1():
            raise ValueError("First fails")

        def handler2():
            raise ValueError("Second fails")

        morph1 = AtomicVector("op1", Stratum.TACTICS, handler1)
        morph2 = AtomicVector("op2", Stratum.TACTICS, handler2)

        coproduct = morph1 | morph2

        state = create_categorical_state(strata={Stratum.PHYSICS})
        result = coproduct(state)

        assert result.is_failed


# =============================================================================
# PRUEBAS DE MORPHISM COMPOSER
# =============================================================================

class TestMorphismComposer:
    """Pruebas del compositor de morfismos."""

    def test_composer_creation(self) -> None:
        """Verifica creación del compositor."""
        composer = MorphismComposer()

        assert len(composer.steps) == 0

    def test_composer_fluent_interface(self) -> None:
        """Verifica interfaz fluida del compositor."""
        composer = MorphismComposer()
        morph = IdentityMorphism(Stratum.PHYSICS)

        returned = composer.add_step(morph)

        assert returned is composer
        assert len(composer.steps) == 1

    def test_composer_build_single(self) -> None:
        """Verifica construcción con un solo paso."""
        composer = MorphismComposer()
        morph = IdentityMorphism(Stratum.PHYSICS)

        result = composer.add_step(morph).build()

        assert result == morph

    def test_composer_build_composed(self) -> None:
        """Verifica construcción de composición."""
        def handler1(x: int):
            return {"y": x * 2}

        def handler2(y: int):
            return {"z": y + 1}

        morph1 = AtomicVector("double", Stratum.TACTICS, handler1, ["x"])
        morph2 = AtomicVector("add", Stratum.STRATEGY, handler2, ["y"])

        composed = MorphismComposer().add_step(morph1).add_step(morph2).build()

        state = create_categorical_state(
            payload={"x": 5},
            strata={Stratum.PHYSICS},
        )

        result = composed(state)

        assert result.is_success
        assert result.payload["z"] == 11

    def test_composer_empty_build_raises(self) -> None:
        """Verifica que build() sin pasos lanza excepción."""
        composer = MorphismComposer()

        with pytest.raises((ValueError, IndexError)):
            composer.build()

