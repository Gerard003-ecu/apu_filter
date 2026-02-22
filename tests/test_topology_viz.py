"""
Pruebas para el módulo de visualización de topología.

Cobertura:
- Dataclasses: CytoscapeNode, CytoscapeEdge, AnomalyData
- Validaciones: validate_session_data, validate_graph
- Extracción: extract_dataframes_from_session, extract_anomaly_data
- Construcción: build_node_element, build_edge_element
- Helpers: _safe_get_float, _safe_get_int, _build_node_label, etc.
- Endpoints: get_project_graph, get_topology_stats
- Integración: flujos completos
"""

import json
from unittest.mock import MagicMock, patch

import networkx as nx
import pandas as pd
import pytest
from flask import Flask

from app.topology_viz import (
    CYCLE_SEPARATOR,
    LABEL_ELLIPSIS,
    LABEL_MAX_LENGTH,
    AnomalyData,
    CytoscapeEdge,
    # Dataclasses
    CytoscapeNode,
    NodeClass,
    NodeColor,
    # Constantes
    NodeType,
    SessionKeys,
    _build_node_label,
    _determine_node_classes,
    _determine_node_color,
    _extract_ids_from_list,
    _extract_nodes_from_cycles,
    _get_node_cost,
    _get_node_type,
    _safe_get_float,
    _safe_get_int,
    analyze_graph_for_visualization,
    build_edge_element,
    # Funciones de endpoint
    build_graph_from_session,
    # Construcción
    build_node_element,
    convert_graph_to_cytoscape_elements,
    create_error_response,
    create_success_response,
    extract_anomaly_data,
    # Extracción
    extract_dataframes_from_session,
    # Blueprint
    topology_bp,
    validate_graph,
    # Validaciones
    validate_session_data,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def flask_app():
    """Aplicación Flask para testing."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key"
    app.register_blueprint(topology_bp)
    return app


@pytest.fixture
def client(flask_app):
    """Cliente de prueba Flask."""
    return flask_app.test_client()


@pytest.fixture
def valid_session_data():
    """Datos de sesión válidos para pruebas."""
    return {
        SessionKeys.PRESUPUESTO: [
            {"codigo": "CAP01", "descripcion": "Capítulo 1", "valor": 1000},
            {"codigo": "ITEM01", "descripcion": "Item 1", "valor": 500},
        ],
        SessionKeys.APUS_DETAIL: [
            {"codigo_apu": "APU01", "descripcion": "APU Test", "insumo": "INS01"},
            {"codigo_apu": "APU01", "descripcion": "APU Test", "insumo": "INS02"},
        ],
    }


@pytest.fixture
def minimal_session_data():
    """Datos de sesión mínimos válidos."""
    return {
        SessionKeys.PRESUPUESTO: [{"codigo": "TEST"}],
    }


@pytest.fixture
def empty_session_data():
    """Datos de sesión con listas vacías."""
    return {
        SessionKeys.PRESUPUESTO: [],
        SessionKeys.APUS_DETAIL: [],
    }


@pytest.fixture
def sample_graph():
    """Grafo de prueba básico."""
    G = nx.DiGraph()
    G.add_node("BUDGET", type="BUDGET", description="Presupuesto", level=0, total_cost=10000)
    G.add_node(
        "CAP01", type="CHAPTER", description="Cimentaciones", level=1, total_cost=5000
    )
    G.add_node("ITEM01", type="ITEM", description="Zapatas", level=2, total_cost=2500)
    G.add_node("APU01", type="APU", description="Concreto 3000 PSI", level=3, unit_cost=150)
    G.add_node("INS01", type="INSUMO", description="Cemento Portland", level=4, unit_cost=25)

    G.add_edge("BUDGET", "CAP01", total_cost=5000)
    G.add_edge("CAP01", "ITEM01", total_cost=2500)
    G.add_edge("ITEM01", "APU01", total_cost=150)
    G.add_edge("APU01", "INS01", total_cost=25)

    return G


@pytest.fixture
def graph_with_anomalies():
    """Grafo con nodos anómalos."""
    G = nx.DiGraph()
    # Nodos normales
    G.add_node("NORMAL01", type="APU", description="Normal")
    G.add_node("NORMAL02", type="INSUMO", description="Normal")
    G.add_edge("NORMAL01", "NORMAL02")

    # Nodo aislado (sin aristas)
    G.add_node("ISOLATED01", type="INSUMO", description="Aislado")

    # Ciclo
    G.add_node("CYCLE01", type="APU")
    G.add_node("CYCLE02", type="APU")
    G.add_edge("CYCLE01", "CYCLE02")
    G.add_edge("CYCLE02", "CYCLE01")

    return G


@pytest.fixture
def sample_anomaly_data():
    """Datos de anomalías para pruebas."""
    return AnomalyData(
        isolated_ids={"ISO01", "ISO02"},
        orphan_ids={"ORP01"},
        empty_ids={"EMPTY01"},
        nodes_in_cycles={"CYC01", "CYC02"},
    )


@pytest.fixture
def sample_analysis_result():
    """Resultado de análisis estructural simulado."""
    return {
        "details": {
            "anomalies": {
                "isolated_nodes": [{"id": "ISO01"}, {"id": "ISO02"}],
                "orphan_insumos": [{"id": "ORP01"}],
                "empty_apus": [{"id": "EMPTY01"}],
            },
            "cycles": {
                "count": 1,
                "list": ["CYC01 -> CYC02 -> CYC01"],
            },
        }
    }


# =============================================================================
# TESTS: Constantes y Enums
# =============================================================================


class TestConstants:
    """Pruebas para constantes y enums."""

    def test_node_type_values(self):
        """Verifica valores de NodeType."""
        assert NodeType.BUDGET.value == "BUDGET"
        assert NodeType.CHAPTER.value == "CHAPTER"
        assert NodeType.ITEM.value == "ITEM"
        assert NodeType.APU.value == "APU"
        assert NodeType.INSUMO.value == "INSUMO"
        assert NodeType.UNKNOWN.value == "UNKNOWN"

    def test_node_class_values(self):
        """Verifica valores de NodeClass."""
        assert NodeClass.NORMAL.value == "normal"
        assert NodeClass.CYCLE.value == "cycle"
        assert NodeClass.ISOLATED.value == "isolated"
        assert NodeClass.EMPTY.value == "empty"

    def test_node_color_values(self):
        """Verifica valores de NodeColor."""
        assert NodeColor.RED.value == "#EF4444"
        assert NodeColor.BLUE.value == "#3B82F6"
        assert NodeColor.ORANGE.value == "#F97316"
        assert NodeColor.BLACK.value == "#1E293B"
        assert NodeColor.GRAY.value == "#9CA3AF"

    def test_session_keys(self):
        """Verifica claves de sesión."""
        assert SessionKeys.PROCESSED_DATA == "processed_data"
        assert SessionKeys.PRESUPUESTO == "presupuesto"
        assert SessionKeys.APUS_DETAIL == "apus_detail"

    def test_label_constants(self):
        """Verifica constantes de etiquetas."""
        assert LABEL_MAX_LENGTH == 20
        assert LABEL_ELLIPSIS == "..."
        assert CYCLE_SEPARATOR == " -> "


# =============================================================================
# TESTS: Dataclasses
# =============================================================================


class TestCytoscapeNode:
    """Pruebas para CytoscapeNode."""

    def test_default_values(self):
        """Verifica valores por defecto."""
        node = CytoscapeNode(id="N1", label="Node 1", node_type="APU", color="#3B82F6")

        assert node.id == "N1"
        assert node.label == "Node 1"
        assert node.node_type == "APU"
        assert node.color == "#3B82F6"
        assert node.level == 0
        assert node.cost == 0.0
        assert node.classes == []

    def test_full_initialization(self):
        """Verifica inicialización completa."""
        node = CytoscapeNode(
            id="N1",
            label="Node 1",
            node_type="CHAPTER",
            color="#1E293B",
            level=2,
            cost=1500.50,
            classes=["CHAPTER", "normal"],
        )

        assert node.level == 2
        assert node.cost == 1500.50
        assert "CHAPTER" in node.classes
        assert node.color == "#1E293B"

    def test_to_dict_structure(self):
        """Verifica estructura de to_dict."""
        node = CytoscapeNode(
            id="N1",
            label="Test Node",
            node_type="APU",
            color="#3B82F6",
            level=3,
            cost=100.0,
            classes=["APU", "cycle"],
        )

        result = node.to_dict()

        assert "data" in result
        assert "classes" in result
        assert result["data"]["id"] == "N1"
        assert result["data"]["label"] == "Test Node"
        assert result["data"]["type"] == "APU"
        assert result["data"]["color"] == "#3B82F6"
        assert result["data"]["level"] == 3
        assert result["data"]["cost"] == 100.0
        assert result["classes"] == "APU cycle"

    def test_to_dict_empty_classes(self):
        """Verifica to_dict con clases vacías."""
        node = CytoscapeNode(
            id="N1", label="N1", node_type="APU", color="#3B82F6", classes=[]
        )

        result = node.to_dict()

        assert result["classes"] == ""


class TestCytoscapeEdge:
    """Pruebas para CytoscapeEdge."""

    def test_default_values(self):
        """Verifica valores por defecto."""
        edge = CytoscapeEdge(source="A", target="B")

        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.cost == 0.0

    def test_full_initialization(self):
        """Verifica inicialización completa."""
        edge = CytoscapeEdge(source="A", target="B", cost=500.25)

        assert edge.cost == 500.25

    def test_to_dict_structure(self):
        """Verifica estructura de to_dict."""
        edge = CytoscapeEdge(source="NODE1", target="NODE2", cost=1000.0)

        result = edge.to_dict()

        assert "data" in result
        assert result["data"]["source"] == "NODE1"
        assert result["data"]["target"] == "NODE2"
        assert result["data"]["cost"] == 1000.0
        assert "classes" not in result  # Aristas no tienen clases


class TestAnomalyData:
    """Pruebas para AnomalyData."""

    def test_default_values(self):
        """Verifica valores por defecto."""
        anomaly = AnomalyData()

        assert anomaly.isolated_ids == set()
        assert anomaly.orphan_ids == set()
        assert anomaly.empty_ids == set()
        assert anomaly.nodes_in_cycles == set()

    def test_sets_are_independent(self):
        """Verifica que los sets son instancias independientes."""
        anomaly1 = AnomalyData()
        anomaly2 = AnomalyData()

        anomaly1.isolated_ids.add("TEST")

        assert "TEST" not in anomaly2.isolated_ids


# =============================================================================
# TESTS: Validaciones
# =============================================================================


class TestValidateSessionData:
    """Pruebas para validate_session_data."""

    def test_valid_data(self, valid_session_data):
        """Verifica aceptación de datos válidos."""
        is_valid, error = validate_session_data(valid_session_data)

        assert is_valid is True
        assert error is None

    def test_minimal_valid_data(self, minimal_session_data):
        """Verifica aceptación de datos mínimos."""
        is_valid, error = validate_session_data(minimal_session_data)

        assert is_valid is True

    def test_none_data(self):
        """Verifica rechazo de None."""
        is_valid, error = validate_session_data(None)

        assert is_valid is False
        assert "None" in error

    def test_non_dict_data(self):
        """Verifica rechazo de no-diccionario."""
        is_valid, error = validate_session_data([1, 2, 3])

        assert is_valid is False
        assert "dict" in error.lower()

    def test_empty_dict(self):
        """Verifica rechazo de diccionario sin datos relevantes."""
        is_valid, error = validate_session_data({})

        assert is_valid is False
        assert "presupuesto" in error.lower() or "apus" in error.lower()

    def test_presupuesto_wrong_type(self):
        """Verifica rechazo cuando presupuesto no es lista."""
        data = {SessionKeys.PRESUPUESTO: "not a list"}

        is_valid, error = validate_session_data(data)

        assert is_valid is False
        assert "lista" in error

    def test_apus_wrong_type(self):
        """Verifica rechazo cuando apus_detail no es lista."""
        data = {SessionKeys.PRESUPUESTO: [], SessionKeys.APUS_DETAIL: {"not": "a list"}}

        is_valid, error = validate_session_data(data)

        assert is_valid is False
        assert "lista" in error

    def test_only_apus_is_valid(self):
        """Verifica que solo tener apus_detail es válido."""
        data = {SessionKeys.APUS_DETAIL: [{"codigo": "APU01"}]}

        is_valid, error = validate_session_data(data)

        assert is_valid is True


class TestValidateGraph:
    """Pruebas para validate_graph."""

    def test_valid_digraph(self, sample_graph):
        """Verifica aceptación de DiGraph válido."""
        is_valid, error = validate_graph(sample_graph)

        assert is_valid is True
        assert error is None

    def test_valid_graph(self):
        """Verifica aceptación de Graph no dirigido."""
        G = nx.Graph()
        G.add_node("A")

        is_valid, error = validate_graph(G)

        assert is_valid is True

    def test_valid_multigraph(self):
        """Verifica aceptación de MultiGraph."""
        G = nx.MultiGraph()

        is_valid, error = validate_graph(G)

        assert is_valid is True

    def test_none_graph(self):
        """Verifica rechazo de None."""
        is_valid, error = validate_graph(None)

        assert is_valid is False
        assert "None" in error

    def test_non_graph_type(self):
        """Verifica rechazo de tipo incorrecto."""
        is_valid, error = validate_graph({"nodes": [], "edges": []})

        assert is_valid is False
        assert "NetworkX" in error


# =============================================================================
# TESTS: Extracción
# =============================================================================


class TestExtractDataframesFromSession:
    """Pruebas para extract_dataframes_from_session."""

    def test_extracts_both_dataframes(self, valid_session_data):
        """Verifica extracción de ambos DataFrames."""
        df_pres, df_apus = extract_dataframes_from_session(valid_session_data)

        assert isinstance(df_pres, pd.DataFrame)
        assert isinstance(df_apus, pd.DataFrame)
        assert len(df_pres) == 2
        assert len(df_apus) == 2

    def test_handles_missing_presupuesto(self):
        """Verifica manejo de presupuesto faltante."""
        data = {SessionKeys.APUS_DETAIL: [{"codigo": "APU01"}]}

        df_pres, df_apus = extract_dataframes_from_session(data)

        assert df_pres.empty
        assert len(df_apus) == 1

    def test_handles_missing_apus(self):
        """Verifica manejo de apus faltante."""
        data = {SessionKeys.PRESUPUESTO: [{"codigo": "CAP01"}]}

        df_pres, df_apus = extract_dataframes_from_session(data)

        assert len(df_pres) == 1
        assert df_apus.empty

    def test_handles_empty_lists(self, empty_session_data):
        """Verifica manejo de listas vacías."""
        df_pres, df_apus = extract_dataframes_from_session(empty_session_data)

        assert df_pres.empty
        assert df_apus.empty


class TestExtractIdsFromList:
    """Pruebas para _extract_ids_from_list."""

    def test_extracts_from_dict_list(self):
        """Verifica extracción desde lista de dicts."""
        items = [{"id": "A"}, {"id": "B"}, {"id": "C"}]

        result = _extract_ids_from_list(items)

        assert result == {"A", "B", "C"}

    def test_extracts_from_string_list(self):
        """Verifica extracción desde lista de strings."""
        items = ["A", "B", "C"]

        result = _extract_ids_from_list(items)

        assert result == {"A", "B", "C"}

    def test_handles_mixed_list(self):
        """Verifica manejo de lista mixta."""
        items = [{"id": "A"}, "B", {"id": "C"}]

        result = _extract_ids_from_list(items)

        assert result == {"A", "B", "C"}

    def test_handles_non_list(self):
        """Verifica manejo de no-lista."""
        assert _extract_ids_from_list(None) == set()
        assert _extract_ids_from_list("string") == set()
        assert _extract_ids_from_list(123) == set()

    def test_handles_dicts_without_id(self):
        """Verifica manejo de dicts sin key 'id'."""
        items = [{"name": "A"}, {"id": "B"}]

        result = _extract_ids_from_list(items)

        assert result == {"B"}

    def test_converts_to_string(self):
        """Verifica conversión a string."""
        items = [{"id": 123}, {"id": "ABC"}]

        result = _extract_ids_from_list(items)

        assert "123" in result
        assert "ABC" in result


class TestExtractNodesFromCycles:
    """Pruebas para _extract_nodes_from_cycles."""

    def test_extracts_from_simple_cycle(self):
        """Verifica extracción de ciclo simple."""
        cycles = ["A -> B -> A"]

        result = _extract_nodes_from_cycles(cycles)

        assert result == {"A", "B"}

    def test_extracts_from_multiple_cycles(self):
        """Verifica extracción de múltiples ciclos."""
        cycles = ["A -> B -> A", "C -> D -> E -> C"]

        result = _extract_nodes_from_cycles(cycles)

        assert result == {"A", "B", "C", "D", "E"}

    def test_handles_non_list(self):
        """Verifica manejo de no-lista."""
        assert _extract_nodes_from_cycles(None) == set()
        assert _extract_nodes_from_cycles("string") == set()

    def test_handles_non_string_items(self):
        """Verifica manejo de items no-string."""
        cycles = ["A -> B -> A", 123, None]

        result = _extract_nodes_from_cycles(cycles)

        assert result == {"A", "B"}

    def test_handles_empty_parts(self):
        """Verifica manejo de partes vacías."""
        cycles = ["A ->  -> B -> A"]

        result = _extract_nodes_from_cycles(cycles)

        # Debería filtrar strings vacíos
        assert "" not in result

    def test_handles_single_node_string(self):
        """Verifica manejo de string con un solo nodo."""
        # Un ciclo debe tener al menos 2 nodos "A -> A"
        cycles = [f"A {CYCLE_SEPARATOR} A"]

        result = _extract_nodes_from_cycles(cycles)

        assert result == {"A"}


class TestExtractAnomalyData:
    """Pruebas para extract_anomaly_data."""

    def test_extracts_all_anomalies(self, sample_analysis_result):
        """Verifica extracción completa de anomalías."""
        result = extract_anomaly_data(sample_analysis_result)

        assert isinstance(result, AnomalyData)
        assert result.isolated_ids == {"ISO01", "ISO02"}
        assert result.orphan_ids == {"ORP01"}
        assert result.empty_ids == {"EMPTY01"}
        assert "CYC01" in result.nodes_in_cycles
        assert "CYC02" in result.nodes_in_cycles

    def test_handles_non_dict(self):
        """Verifica manejo de input no-dict."""
        result = extract_anomaly_data(None)

        assert result.isolated_ids == set()
        assert result.orphan_ids == set()

    def test_handles_missing_details(self):
        """Verifica manejo de 'details' faltante."""
        result = extract_anomaly_data({})

        assert isinstance(result, AnomalyData)

    def test_handles_missing_anomalies(self):
        """Verifica manejo de 'anomalies' faltante."""
        result = extract_anomaly_data({"details": {}})

        assert result.isolated_ids == set()

    def test_handles_missing_cycles(self):
        """Verifica manejo de 'cycles' faltante."""
        analysis = {"details": {"anomalies": {"isolated_nodes": [{"id": "A"}]}}}

        result = extract_anomaly_data(analysis)

        assert result.isolated_ids == {"A"}
        assert result.nodes_in_cycles == set()


# =============================================================================
# TESTS: Helpers de Tipos Seguros
# =============================================================================


class TestSafeGetFloat:
    """Pruebas para _safe_get_float."""

    def test_returns_float_value(self):
        """Verifica retorno de valor float."""
        assert _safe_get_float({"cost": 100.5}, "cost", 0.0) == 100.5

    def test_converts_int_to_float(self):
        """Verifica conversión de int a float."""
        assert _safe_get_float({"cost": 100}, "cost", 0.0) == 100.0

    def test_converts_string_to_float(self):
        """Verifica conversión de string numérico."""
        assert _safe_get_float({"cost": "150.75"}, "cost", 0.0) == 150.75

    def test_returns_default_for_missing_key(self):
        """Verifica retorno de default para key faltante."""
        assert _safe_get_float({}, "cost", 99.9) == 99.9

    def test_returns_default_for_none_value(self):
        """Verifica retorno de default para valor None."""
        assert _safe_get_float({"cost": None}, "cost", 0.0) == 0.0

    def test_returns_default_for_non_numeric(self):
        """Verifica retorno de default para valor no numérico."""
        assert _safe_get_float({"cost": "abc"}, "cost", 0.0) == 0.0

    def test_returns_none_default(self):
        """Verifica que puede retornar None como default."""
        assert _safe_get_float({}, "cost", None) is None


class TestSafeGetInt:
    """Pruebas para _safe_get_int."""

    def test_returns_int_value(self):
        """Verifica retorno de valor int."""
        assert _safe_get_int({"level": 5}, "level", 0) == 5

    def test_converts_float_to_int(self):
        """Verifica conversión de float a int."""
        assert _safe_get_int({"level": 3.7}, "level", 0) == 3

    def test_converts_string_to_int(self):
        """Verifica conversión de string numérico."""
        assert _safe_get_int({"level": "10"}, "level", 0) == 10

    def test_returns_default_for_missing_key(self):
        """Verifica retorno de default para key faltante."""
        assert _safe_get_int({}, "level", 99) == 99

    def test_returns_default_for_non_numeric(self):
        """Verifica retorno de default para valor no numérico."""
        assert _safe_get_int({"level": "abc"}, "level", 0) == 0


# =============================================================================
# TESTS: Construcción de Labels y Colores
# =============================================================================


class TestBuildNodeLabel:
    """Pruebas para _build_node_label."""

    def test_returns_id_only_when_no_description(self):
        """Verifica retorno de solo ID sin descripción."""
        result = _build_node_label("NODE01", {})

        assert result == "NODE01"

    def test_returns_id_only_for_none_description(self):
        """Verifica retorno de solo ID con descripción None."""
        result = _build_node_label("NODE01", {"description": None})

        assert result == "NODE01"

    def test_returns_id_only_for_empty_description(self):
        """Verifica retorno de solo ID con descripción vacía."""
        result = _build_node_label("NODE01", {"description": "   "})

        assert result == "NODE01"

    def test_includes_short_description(self):
        """Verifica inclusión de descripción corta."""
        result = _build_node_label("NODE01", {"description": "Short desc"})

        assert "NODE01" in result
        assert "Short desc" in result
        assert "\n" in result

    def test_truncates_long_description(self):
        """Verifica truncado de descripción larga."""
        long_desc = "A" * 50  # Más largo que LABEL_MAX_LENGTH
        result = _build_node_label("NODE01", {"description": long_desc})

        assert LABEL_ELLIPSIS in result
        assert len(result.split("\n")[1]) <= LABEL_MAX_LENGTH + len(LABEL_ELLIPSIS)

    def test_handles_non_string_description(self):
        """Verifica manejo de descripción no-string."""
        result = _build_node_label("NODE01", {"description": 12345})

        assert "12345" in result


class TestDetermineNodeColor:
    """Pruebas para _determine_node_color."""

    def test_red_for_cycles(self, sample_anomaly_data):
        """Verifica color rojo para ciclos."""
        color = _determine_node_color("CYC01", "APU", sample_anomaly_data)
        assert color == NodeColor.RED.value

    def test_red_for_isolated(self, sample_anomaly_data):
        """Verifica color rojo para aislados."""
        color = _determine_node_color("ISO01", "INSUMO", sample_anomaly_data)
        assert color == NodeColor.RED.value

    def test_black_for_budget(self, sample_anomaly_data):
        """Verifica color negro para nodos raíz/presupuesto."""
        color = _determine_node_color("ROOT", "BUDGET", sample_anomaly_data)
        assert color == NodeColor.BLACK.value

    def test_blue_for_apu(self, sample_anomaly_data):
        """Verifica color azul para APUs normales."""
        color = _determine_node_color("APU01", "APU", sample_anomaly_data)
        assert color == NodeColor.BLUE.value

    def test_orange_for_insumo(self, sample_anomaly_data):
        """Verifica color naranja para insumos normales."""
        color = _determine_node_color("INS01", "INSUMO", sample_anomaly_data)
        assert color == NodeColor.ORANGE.value

    def test_gray_for_unknown(self, sample_anomaly_data):
        """Verifica color gris para desconocidos."""
        color = _determine_node_color("UNKNOWN", "OTHER", sample_anomaly_data)
        assert color == NodeColor.GRAY.value


# =============================================================================
# TESTS: Construcción de Elementos
# =============================================================================


class TestGetNodeType:
    """Pruebas para _get_node_type."""

    def test_returns_existing_type(self):
        """Verifica retorno de tipo existente."""
        assert _get_node_type({"type": "APU"}) == "APU"

    def test_uppercases_type(self):
        """Verifica conversión a mayúsculas."""
        assert _get_node_type({"type": "chapter"}) == "CHAPTER"

    def test_returns_unknown_for_missing(self):
        """Verifica retorno de UNKNOWN para tipo faltante."""
        assert _get_node_type({}) == NodeType.UNKNOWN.value

    def test_returns_unknown_for_empty(self):
        """Verifica retorno de UNKNOWN para tipo vacío."""
        assert _get_node_type({"type": ""}) == NodeType.UNKNOWN.value

    def test_returns_unknown_for_non_string(self):
        """Verifica retorno de UNKNOWN para tipo no-string."""
        assert _get_node_type({"type": 123}) == NodeType.UNKNOWN.value


class TestDetermineNodeClasses:
    """Pruebas para _determine_node_classes."""

    def test_includes_node_type(self, sample_anomaly_data):
        """Verifica inclusión del tipo de nodo."""
        classes = _determine_node_classes("NORMAL", "APU", sample_anomaly_data)

        assert "APU" in classes

    def test_cycle_priority(self, sample_anomaly_data):
        """Verifica prioridad de clase cycle."""
        classes = _determine_node_classes("CYC01", "APU", sample_anomaly_data)

        assert NodeClass.CYCLE.value in classes
        assert NodeClass.NORMAL.value not in classes

    def test_isolated_class(self, sample_anomaly_data):
        """Verifica clase isolated."""
        classes = _determine_node_classes("ISO01", "INSUMO", sample_anomaly_data)

        assert NodeClass.ISOLATED.value in classes

    def test_orphan_gets_isolated_class(self, sample_anomaly_data):
        """Verifica que huérfanos obtienen clase isolated."""
        classes = _determine_node_classes("ORP01", "INSUMO", sample_anomaly_data)

        assert NodeClass.ISOLATED.value in classes

    def test_empty_class(self, sample_anomaly_data):
        """Verifica clase empty."""
        classes = _determine_node_classes("EMPTY01", "APU", sample_anomaly_data)

        assert NodeClass.EMPTY.value in classes

    def test_normal_class(self, sample_anomaly_data):
        """Verifica clase normal para nodo sin anomalías."""
        classes = _determine_node_classes("NORMAL_NODE", "APU", sample_anomaly_data)

        assert NodeClass.NORMAL.value in classes


class TestGetNodeCost:
    """Pruebas para _get_node_cost."""

    def test_prefers_total_cost(self):
        """Verifica preferencia por total_cost."""
        attrs = {"total_cost": 1000, "unit_cost": 100}

        assert _get_node_cost(attrs) == 1000

    def test_fallback_to_unit_cost(self):
        """Verifica fallback a unit_cost."""
        attrs = {"unit_cost": 100}

        assert _get_node_cost(attrs) == 100

    def test_returns_zero_when_no_cost(self):
        """Verifica retorno de 0 sin costos."""
        assert _get_node_cost({}) == 0.0

    def test_handles_none_total_cost(self):
        """Verifica manejo de total_cost None."""
        attrs = {"total_cost": None, "unit_cost": 50}

        assert _get_node_cost(attrs) == 50


class TestBuildNodeElement:
    """Pruebas para build_node_element."""

    def test_builds_complete_node(self, sample_anomaly_data):
        """Verifica construcción de nodo completo."""
        attrs = {
            "type": "APU",
            "description": "Concreto 3000 PSI",
            "level": 3,
            "total_cost": 500,
        }

        node = build_node_element("APU01", attrs, sample_anomaly_data)

        assert isinstance(node, CytoscapeNode)
        assert node.id == "APU01"
        assert node.node_type == "APU"
        assert node.level == 3
        assert node.cost == 500
        assert "APU" in node.classes
        assert node.color == NodeColor.BLUE.value

    def test_handles_empty_attrs(self):
        """Verifica manejo de atributos vacíos."""
        node = build_node_element("NODE01", {}, AnomalyData())

        assert node.id == "NODE01"
        assert node.node_type == NodeType.UNKNOWN.value

    def test_converts_id_to_string(self):
        """Verifica conversión de ID a string."""
        node = build_node_element(12345, {"type": "APU"}, AnomalyData())

        assert node.id == "12345"


class TestBuildEdgeElement:
    """Pruebas para build_edge_element."""

    def test_builds_complete_edge(self, sample_anomaly_data):
        """Verifica construcción de arista completa."""
        attrs = {"total_cost": 250}
        anomaly_data = AnomalyData()

        edge = build_edge_element("A", "B", attrs, anomaly_data)

        assert isinstance(edge, CytoscapeEdge)
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.cost == 250

    def test_handles_empty_attrs(self, sample_anomaly_data):
        """Verifica manejo de atributos vacíos."""
        edge = build_edge_element("A", "B", {}, AnomalyData())

        assert edge.cost == 0.0

    def test_converts_to_string(self, sample_anomaly_data):
        """Verifica conversión a string."""
        edge = build_edge_element(123, 456, {}, AnomalyData())

        assert edge.source == "123"
        assert edge.target == "456"


# =============================================================================
# TESTS: Funciones de Endpoint
# =============================================================================


class TestBuildGraphFromSession:
    """Pruebas para build_graph_from_session."""

    @patch("app.topology_viz.BudgetGraphBuilder")
    def test_builds_graph_successfully(self, mock_builder_class, valid_session_data):
        """Verifica construcción exitosa del grafo."""
        mock_graph = nx.DiGraph()
        mock_graph.add_node("TEST")
        mock_builder = MagicMock()
        mock_builder.build.return_value = mock_graph
        mock_builder_class.return_value = mock_builder

        result = build_graph_from_session(valid_session_data)

        assert isinstance(result, nx.DiGraph)
        mock_builder.build.assert_called_once()

    @patch("app.topology_viz.BudgetGraphBuilder")
    def test_raises_on_empty_data(self, mock_builder_class, empty_session_data):
        """Verifica error con datos vacíos."""
        with pytest.raises(ValueError, match="No hay datos suficientes"):
            build_graph_from_session(empty_session_data)

    @patch("app.topology_viz.BudgetGraphBuilder")
    def test_raises_on_invalid_graph(self, mock_builder_class, valid_session_data):
        """Verifica error cuando builder retorna grafo inválido."""
        mock_builder = MagicMock()
        mock_builder.build.return_value = None
        mock_builder_class.return_value = mock_builder

        with pytest.raises(ValueError, match="inválido"):
            build_graph_from_session(valid_session_data)


class TestAnalyzeGraphForVisualization:
    """Pruebas para analyze_graph_for_visualization."""

    @patch("app.topology_viz.BusinessTopologicalAnalyzer")
    def test_returns_anomaly_data(
        self, mock_analyzer_class, sample_graph, sample_analysis_result
    ):
        """Verifica retorno de AnomalyData."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_structural_integrity.return_value = sample_analysis_result
        mock_analyzer_class.return_value = mock_analyzer

        result = analyze_graph_for_visualization(sample_graph)

        assert isinstance(result, AnomalyData)

    @patch("app.topology_viz.BusinessTopologicalAnalyzer")
    def test_handles_analyzer_exception(self, mock_analyzer_class, sample_graph):
        """Verifica manejo de excepción del analizador."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_structural_integrity.side_effect = Exception("Analysis failed")
        mock_analyzer_class.return_value = mock_analyzer

        result = analyze_graph_for_visualization(sample_graph)

        # Debería retornar AnomalyData vacío, no lanzar excepción
        assert isinstance(result, AnomalyData)
        assert result.isolated_ids == set()


class TestConvertGraphToCytoscapeElements:
    """Pruebas para convert_graph_to_cytoscape_elements."""

    def test_converts_all_elements(self, sample_graph):
        """Verifica conversión de todos los elementos."""
        anomaly_data = AnomalyData()

        elements = convert_graph_to_cytoscape_elements(sample_graph, anomaly_data)

        node_count = sample_graph.number_of_nodes()
        edge_count = sample_graph.number_of_edges()

        assert len(elements) == node_count + edge_count

    def test_nodes_have_correct_structure(self, sample_graph):
        """Verifica estructura correcta de nodos."""
        elements = convert_graph_to_cytoscape_elements(sample_graph, AnomalyData())

        # Filtrar solo nodos (tienen 'classes')
        nodes = [e for e in elements if "classes" in e]

        for node in nodes:
            assert "data" in node
            assert "id" in node["data"]
            assert "label" in node["data"]
            assert "type" in node["data"]
            assert "color" in node["data"]

    def test_edges_have_correct_structure(self, sample_graph):
        """Verifica estructura correcta de aristas."""
        elements = convert_graph_to_cytoscape_elements(sample_graph, AnomalyData())

        # Filtrar solo aristas (no tienen 'classes')
        edges = [e for e in elements if "classes" not in e]

        for edge in edges:
            assert "data" in edge
            assert "source" in edge["data"]
            assert "target" in edge["data"]

    def test_handles_empty_graph(self):
        """Verifica manejo de grafo vacío."""
        G = nx.DiGraph()

        elements = convert_graph_to_cytoscape_elements(G, AnomalyData())

        assert elements == []


# =============================================================================
# TESTS: Respuestas
# =============================================================================


class TestCreateErrorResponse:
    """Pruebas para create_error_response."""

    def test_response_structure(self, flask_app):
        """Verifica estructura de respuesta de error."""
        with flask_app.app_context():
            response, status = create_error_response("Test error", 400)
            data = response.get_json()

            assert status == 400
            assert data["error"] == "Test error"
            assert data["elements"] == []
            assert data["success"] is False


class TestCreateSuccessResponse:
    """Pruebas para create_success_response."""

    def test_response_structure(self, flask_app):
        """Verifica estructura de respuesta exitosa."""
        elements = [{"data": {"id": "A"}}, {"data": {"id": "B"}}]

        with flask_app.app_context():
            response, status = create_success_response(elements)
            data = response.get_json()

            assert status == 200
            assert data["elements"] == elements
            assert data["count"] == 2
            assert data["success"] is True


# =============================================================================
# TESTS: Endpoints
# =============================================================================


class TestGetProjectGraphEndpoint:
    """Pruebas para el endpoint get_project_graph (renombrado de get_topology_data)."""

    def test_returns_404_without_session(self, client):
        """Verifica 404 sin datos en sesión."""
        response = client.get("/api/visualization/project-graph")

        assert response.status_code == 404
        data = response.get_json()
        assert "error" in data
        assert data["success"] is False

    def test_returns_400_for_invalid_session_data(self, client, flask_app):
        """Verifica 400 para datos de sesión inválidos."""
        with client.session_transaction() as sess:
            sess[SessionKeys.PROCESSED_DATA] = "invalid_data"

        response = client.get("/api/visualization/project-graph")

        assert response.status_code == 400

    @patch("app.topology_viz.build_graph_from_session")
    @patch("app.topology_viz.analyze_graph_for_visualization")
    def test_returns_success_with_valid_data(
        self, mock_analyze, mock_build, client, sample_graph, valid_session_data
    ):
        """Verifica respuesta exitosa con datos válidos."""
        mock_build.return_value = sample_graph
        mock_analyze.return_value = AnomalyData()

        with client.session_transaction() as sess:
            sess[SessionKeys.PROCESSED_DATA] = valid_session_data

        response = client.get("/api/visualization/project-graph")

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "elements" in data
        assert "count" in data

    @patch("app.topology_viz.build_graph_from_session")
    def test_returns_empty_elements_for_empty_graph(
        self, mock_build, client, valid_session_data
    ):
        """Verifica respuesta vacía para grafo vacío."""
        mock_build.return_value = nx.DiGraph()  # Grafo vacío

        with client.session_transaction() as sess:
            sess[SessionKeys.PROCESSED_DATA] = valid_session_data

        response = client.get("/api/visualization/project-graph")

        assert response.status_code == 200
        data = response.get_json()
        assert data["elements"] == []
        assert data["count"] == 0

    @patch("app.topology_viz.build_graph_from_session")
    def test_handles_value_error(self, mock_build, client, valid_session_data):
        """Verifica manejo de ValueError."""
        mock_build.side_effect = ValueError("Construction failed")

        with client.session_transaction() as sess:
            sess[SessionKeys.PROCESSED_DATA] = valid_session_data

        response = client.get("/api/visualization/project-graph")

        assert response.status_code == 400
        data = response.get_json()
        assert "Construction failed" in data["error"]

    @patch("app.topology_viz.build_graph_from_session")
    def test_handles_unexpected_exception(self, mock_build, client, valid_session_data):
        """Verifica manejo de excepción inesperada."""
        mock_build.side_effect = RuntimeError("Unexpected error")

        with client.session_transaction() as sess:
            sess[SessionKeys.PROCESSED_DATA] = valid_session_data

        response = client.get("/api/visualization/project-graph")

        assert response.status_code == 500
        data = response.get_json()
        assert "RuntimeError" in data["error"]


class TestGetTopologyStatsEndpoint:
    """Pruebas para el endpoint get_topology_stats."""

    def test_returns_404_without_session(self, client):
        """Verifica 404 sin datos en sesión."""
        response = client.get("/api/visualization/topology/stats")

        assert response.status_code == 404

    @patch("app.topology_viz.build_graph_from_session")
    @patch("app.topology_viz.analyze_graph_for_visualization")
    def test_returns_stats(
        self,
        mock_analyze,
        mock_build,
        client,
        sample_graph,
        sample_anomaly_data,
        valid_session_data,
    ):
        """Verifica retorno de estadísticas."""
        mock_build.return_value = sample_graph
        mock_analyze.return_value = sample_anomaly_data

        with client.session_transaction() as sess:
            sess[SessionKeys.PROCESSED_DATA] = valid_session_data

        response = client.get("/api/visualization/topology/stats")

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "nodes" in data
        assert "edges" in data
        assert "isolated_nodes" in data
        assert "orphan_nodes" in data
        assert "empty_apus" in data
        assert "nodes_in_cycles" in data


# =============================================================================
# TESTS: Integración
# =============================================================================


class TestIntegration:
    """Pruebas de integración end-to-end."""

    @patch("app.topology_viz.BudgetGraphBuilder")
    @patch("app.topology_viz.BusinessTopologicalAnalyzer")
    def test_full_workflow(
        self,
        mock_analyzer_class,
        mock_builder_class,
        client,
        sample_graph,
        sample_analysis_result,
        valid_session_data,
    ):
        """Prueba flujo completo desde sesión hasta respuesta."""
        # Configurar mocks
        mock_builder = MagicMock()
        mock_builder.build.return_value = sample_graph
        mock_builder_class.return_value = mock_builder

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_structural_integrity.return_value = sample_analysis_result
        mock_analyzer_class.return_value = mock_analyzer

        # Configurar sesión
        with client.session_transaction() as sess:
            sess[SessionKeys.PROCESSED_DATA] = valid_session_data

        # Ejecutar request
        response = client.get("/api/visualization/project-graph")

        # Verificar
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert len(data["elements"]) > 0

        # Verificar que se llamaron los métodos correctos
        mock_builder.build.assert_called_once()
        mock_analyzer.analyze_structural_integrity.assert_called_once()

    def test_dataclass_serialization_chain(self):
        """Prueba cadena de serialización de dataclasses."""
        # Crear nodo
        node = CytoscapeNode(
            id="TEST",
            label="Test Node",
            node_type="APU",
            color="#3B82F6",
            level=1,
            cost=100.0,
            classes=["APU", "normal"],
        )

        # Crear arista
        edge = CytoscapeEdge(source="A", target="B", cost=50.0)

        # Convertir a dict
        node_dict = node.to_dict()
        edge_dict = edge.to_dict()

        # Serializar a JSON (simula lo que hace jsonify)
        json_str = json.dumps([node_dict, edge_dict])

        # Deserializar
        parsed = json.loads(json_str)

        assert len(parsed) == 2
        assert parsed[0]["data"]["id"] == "TEST"
        assert parsed[0]["data"]["color"] == "#3B82F6"
        assert parsed[1]["data"]["source"] == "A"
