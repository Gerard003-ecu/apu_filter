"""
Suite de pruebas rigurosa para ``topology_viz.py``.

Estructura:
- Fixtures reutilizables para grafos, anomalías y configuraciones.
- Pruebas unitarias por función/método.
- Pruebas de integración para el pipeline de conversión.
- Pruebas de propiedades (idempotencia, determinismo, inmutabilidad).

Convenciones:
- Cada clase ``Test*`` agrupa pruebas de una unidad lógica.
- Los nombres siguen ``test_<condición>_<resultado_esperado>``.
- Se usa ``pytest.mark.parametrize`` para variantes sin duplicación.
- Las dependencias externas (``BudgetGraphBuilder``, ``BusinessTopologicalAnalyzer``,
  ``Flask session``) se mockean para aislamiento.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, patch

import networkx as nx
import pandas as pd
import pytest

from topology_viz import (
    ALLOWED_STRATUM_FILTERS,
    LABEL_ELLIPSIS,
    LABEL_MAX_LENGTH,
    STRATUM_VISIBLE_LEVELS,
    AnomalyData,
    CytoscapeEdge,
    CytoscapeNode,
    NodeClass,
    NodeColor,
    NodeType,
    StressConfig,
    ValidationOutcome,
    _build_edge_tooltip,
    _build_fallback_edge,
    _build_fallback_node,
    _build_node_label,
    _build_node_tooltip,
    _deduplicate_preserve_order,
    _determine_node_classes,
    _determine_node_color,
    _extract_dataframe_from_list,
    _extract_ids_from_list,
    _extract_nodes_from_cycles,
    _get_edge_score,
    _get_node_cost,
    _get_node_score,
    _get_node_type,
    _get_visible_levels,
    _identify_stressed_nodes,
    _normalize_edge_tuple,
    _normalize_identifier,
    _parse_single_cycle,
    _parse_stratum_param,
    _safe_finite_float,
    _safe_int_from_any,
    _safe_nonnegative_finite_float,
    _truncate_text,
    _try_normalize_edge,
    analyze_graph_for_visualization,
    build_edge_element,
    build_graph_from_session,
    build_node_element,
    convert_graph_to_cytoscape_elements,
    create_error_response,
    create_success_response,
    extract_anomaly_data,
    extract_dataframes_from_session,
    validate_graph,
    validate_session_data,
    validate_stratum_filter,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def empty_anomaly_data() -> AnomalyData:
    """AnomalyData sin anomalías."""
    return AnomalyData()


@pytest.fixture()
def rich_anomaly_data() -> AnomalyData:
    """AnomalyData con múltiples tipos de anomalía."""
    return AnomalyData(
        isolated_ids=frozenset({"ISO-1", "ISO-2"}),
        orphan_ids=frozenset({"ORP-1"}),
        empty_ids=frozenset({"EMP-1"}),
        nodes_in_cycles=frozenset({"CYC-A", "CYC-B", "CYC-C"}),
        stressed_ids=frozenset({"STR-1"}),
        hot_ids=frozenset({"HOT-1"}),
        anomalous_nodes=frozenset({"ANO-1", "CYC-A"}),
        node_scores={"CYC-A": 0.95, "STR-1": 0.7, "NORMAL-1": 0.1},
        anomalous_edges=frozenset({("CYC-A", "CYC-B"), ("ANO-1", "X")}),
        edge_scores={("CYC-A", "CYC-B"): 0.85, ("STR-1", "HOT-1"): 0.5},
    )


@pytest.fixture()
def simple_digraph() -> nx.DiGraph:
    """DiGraph simple con nodos tipados y niveles."""
    g = nx.DiGraph()
    g.add_node("B1", type="BUDGET", level=0, description="Presupuesto X")
    g.add_node("C1", type="CHAPTER", level=1, description="Capítulo Uno")
    g.add_node("I1", type="ITEM", level=1, description="Item Primero")
    g.add_node(
        "APU-1",
        type="APU",
        level=2,
        description="APU Principal",
        total_cost=1000.0,
    )
    g.add_node(
        "INS-1",
        type="INSUMO",
        level=3,
        description="Cemento Portland Tipo I",
        unit_cost=50.0,
    )
    g.add_node(
        "INS-2",
        type="INSUMO",
        level=3,
        description="Arena lavada",
        unit_cost=30.0,
    )

    g.add_edge("B1", "C1")
    g.add_edge("C1", "I1")
    g.add_edge("I1", "APU-1")
    g.add_edge("APU-1", "INS-1", total_cost=500.0)
    g.add_edge("APU-1", "INS-2", total_cost=300.0)

    return g


@pytest.fixture()
def stress_test_graph() -> nx.DiGraph:
    """DiGraph con un insumo altamente conectado para pruebas de estrés."""
    g = nx.DiGraph()

    for i in range(15):
        g.add_node(f"APU-{i}", type="APU", level=2)

    g.add_node("INS-POPULAR", type="INSUMO", level=3)
    g.add_node("INS-RARE", type="INSUMO", level=3)

    # INS-POPULAR conectado a 10 de 15 APUs → ratio 0.667 > 0.30
    for i in range(10):
        g.add_edge(f"APU-{i}", "INS-POPULAR")

    # INS-RARE conectado a 1 APU
    g.add_edge("APU-0", "INS-RARE")

    return g


@pytest.fixture()
def default_stress_config() -> StressConfig:
    return StressConfig()


# ======================================================================
# Tests: StressConfig
# ======================================================================


class TestStressConfig:
    """Pruebas para la configuración de estrés topológico."""

    def test_default_values(self) -> None:
        cfg = StressConfig()
        assert cfg.large_graph_threshold == 10
        assert cfg.ratio_large == 0.30
        assert cfg.ratio_small == 0.50
        assert cfg.min_absolute_connections == 2

    def test_frozen_immutability(self) -> None:
        cfg = StressConfig()
        with pytest.raises(AttributeError):
            cfg.ratio_large = 0.5  # type: ignore[misc]

    @pytest.mark.parametrize(
        "field, bad_value",
        [
            ("large_graph_threshold", 0),
            ("large_graph_threshold", -1),
            ("ratio_large", 0.0),
            ("ratio_large", 1.5),
            ("ratio_large", -0.1),
            ("ratio_small", 0.0),
            ("ratio_small", 2.0),
            ("min_absolute_connections", 0),
            ("min_absolute_connections", -5),
        ],
    )
    def test_invalid_values_raise(self, field: str, bad_value: Any) -> None:
        with pytest.raises(ValueError):
            StressConfig(**{field: bad_value})

    def test_get_threshold_ratio_large_graph(self) -> None:
        cfg = StressConfig(large_graph_threshold=10, ratio_large=0.3, ratio_small=0.5)
        assert cfg.get_threshold_ratio(11) == 0.3
        assert cfg.get_threshold_ratio(100) == 0.3

    def test_get_threshold_ratio_small_graph(self) -> None:
        cfg = StressConfig(large_graph_threshold=10, ratio_large=0.3, ratio_small=0.5)
        assert cfg.get_threshold_ratio(10) == 0.5
        assert cfg.get_threshold_ratio(5) == 0.5

    def test_boundary_threshold(self) -> None:
        cfg = StressConfig(large_graph_threshold=10)
        assert cfg.get_threshold_ratio(10) == cfg.ratio_small
        assert cfg.get_threshold_ratio(11) == cfg.ratio_large


# ======================================================================
# Tests: AnomalyData
# ======================================================================


class TestAnomalyData:
    """Pruebas para el contenedor de anomalías."""

    def test_default_empty(self) -> None:
        ad = AnomalyData()
        assert len(ad.isolated_ids) == 0
        assert len(ad.node_scores) == 0
        assert len(ad.anomalous_edges) == 0

    def test_frozen_immutability(self) -> None:
        ad = AnomalyData()
        with pytest.raises(AttributeError):
            ad.isolated_ids = frozenset({"X"})  # type: ignore[misc]

    def test_is_node_anomalous_with_anomaly(
        self, rich_anomaly_data: AnomalyData
    ) -> None:
        assert rich_anomaly_data.is_node_anomalous("CYC-A") is True
        assert rich_anomaly_data.is_node_anomalous("ISO-1") is True
        assert rich_anomaly_data.is_node_anomalous("ORP-1") is True
        assert rich_anomaly_data.is_node_anomalous("EMP-1") is True
        assert rich_anomaly_data.is_node_anomalous("STR-1") is True
        assert rich_anomaly_data.is_node_anomalous("HOT-1") is True
        assert rich_anomaly_data.is_node_anomalous("ANO-1") is True

    def test_is_node_anomalous_normal_node(
        self, rich_anomaly_data: AnomalyData
    ) -> None:
        assert rich_anomaly_data.is_node_anomalous("NORMAL-1") is False
        assert rich_anomaly_data.is_node_anomalous("UNKNOWN-X") is False

    def test_is_node_anomalous_empty_data(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        assert empty_anomaly_data.is_node_anomalous("ANY") is False


# ======================================================================
# Tests: CytoscapeNode
# ======================================================================


class TestCytoscapeNode:
    """Pruebas para serialización de nodos Cytoscape."""

    def test_to_dict_structure(self) -> None:
        node = CytoscapeNode(
            id="N1",
            label="Nodo 1",
            node_type="APU",
            color="#3B82F6",
            level=2,
            cost=100.0,
            weight=100.0,
            score=0.5,
            is_evidence=True,
            tooltip="Info",
            classes=("APU", "normal"),
        )
        d = node.to_dict()
        assert "data" in d
        assert "classes" in d
        assert d["data"]["id"] == "N1"
        assert d["data"]["type"] == "APU"
        assert d["data"]["is_evidence"] is True
        assert d["classes"] == "APU normal"

    def test_empty_classes_produces_empty_string(self) -> None:
        node = CytoscapeNode(
            id="X", label="X", node_type="UNKNOWN", color="#000"
        )
        assert node.to_dict()["classes"] == ""

    def test_frozen(self) -> None:
        node = CytoscapeNode(
            id="X", label="X", node_type="APU", color="#000"
        )
        with pytest.raises(AttributeError):
            node.id = "Y"  # type: ignore[misc]


# ======================================================================
# Tests: CytoscapeEdge
# ======================================================================


class TestCytoscapeEdge:
    """Pruebas para serialización de aristas Cytoscape."""

    def test_to_dict_structure(self) -> None:
        edge = CytoscapeEdge(
            source="A",
            target="B",
            cost=50.0,
            score=0.8,
            is_evidence=True,
            tooltip="Edge info",
        )
        d = edge.to_dict()
        assert d["data"]["source"] == "A"
        assert d["data"]["target"] == "B"
        assert d["data"]["cost"] == 50.0
        assert d["data"]["is_evidence"] is True

    def test_defaults(self) -> None:
        edge = CytoscapeEdge(source="A", target="B")
        assert edge.cost == 0.0
        assert edge.is_evidence is False

    def test_frozen(self) -> None:
        edge = CytoscapeEdge(source="A", target="B")
        with pytest.raises(AttributeError):
            edge.source = "C"  # type: ignore[misc]


# ======================================================================
# Tests: ValidationOutcome
# ======================================================================


class TestValidationOutcome:
    """Pruebas para resultados de validación."""

    def test_ok_outcome(self) -> None:
        v = ValidationOutcome(True, None)
        assert v.ok is True
        assert v.message is None

    def test_error_outcome(self) -> None:
        v = ValidationOutcome(False, "Error message")
        assert v.ok is False
        assert v.message == "Error message"

    def test_frozen(self) -> None:
        v = ValidationOutcome(True)
        with pytest.raises(AttributeError):
            v.ok = False  # type: ignore[misc]


# ======================================================================
# Tests: _normalize_identifier
# ======================================================================


class TestNormalizeIdentifier:
    """Pruebas para normalización de identificadores."""

    @pytest.mark.parametrize(
        "value, default, expected",
        [
            (None, "unknown", "unknown"),
            ("", "unknown", "unknown"),
            ("   ", "fallback", "fallback"),
            ("APU-001", "x", "APU-001"),
            ("  spaced  ", "x", "spaced"),
            (123, "x", "123"),
            (45.6, "x", "45.6"),
            (True, "x", "True"),
            (None, "", ""),
        ],
    )
    def test_normalization(
        self, value: Any, default: str, expected: str
    ) -> None:
        assert _normalize_identifier(value, default) == expected

    def test_idempotence(self) -> None:
        values = ["hello", "  spaced  ", None, 42]
        for v in values:
            first = _normalize_identifier(v)
            second = _normalize_identifier(first)
            assert first == second


# ======================================================================
# Tests: _normalize_edge_tuple
# ======================================================================


class TestNormalizeEdgeTuple:
    """Pruebas para normalización de tuplas de arista."""

    def test_basic(self) -> None:
        assert _normalize_edge_tuple("A", "B") == ("A", "B")

    def test_with_spaces(self) -> None:
        assert _normalize_edge_tuple("  A  ", "  B  ") == ("A", "B")

    def test_with_none(self) -> None:
        result = _normalize_edge_tuple(None, "B")
        assert result == ("unknown", "B")

    def test_both_none(self) -> None:
        result = _normalize_edge_tuple(None, None)
        assert result == ("unknown", "unknown")


# ======================================================================
# Tests: _safe_finite_float
# ======================================================================


class TestSafeFiniteFloat:
    """Pruebas para conversión segura a float finito."""

    @pytest.mark.parametrize(
        "value, default, expected",
        [
            (42, 0.0, 42.0),
            (3.14, 0.0, 3.14),
            ("10.5", 0.0, 10.5),
            ("-5.0", 0.0, -5.0),
            (None, 0.0, 0.0),
            ("abc", 99.0, 99.0),
            (float("inf"), 0.0, 0.0),
            (float("-inf"), 0.0, 0.0),
            (float("nan"), 0.0, 0.0),
            ([], 0.0, 0.0),
        ],
    )
    def test_conversion(
        self, value: Any, default: float, expected: float
    ) -> None:
        assert _safe_finite_float(value, default) == pytest.approx(expected)


# ======================================================================
# Tests: _safe_nonnegative_finite_float
# ======================================================================


class TestSafeNonnegativeFiniteFloat:
    """Pruebas para conversión a float finito no negativo."""

    @pytest.mark.parametrize(
        "value, default, expected",
        [
            (42, 0.0, 42.0),
            (0.0, 0.0, 0.0),
            (-5.0, 0.0, 0.0),
            ("-10", 0.0, 0.0),
            (float("inf"), 0.0, 0.0),
            (None, 5.0, 5.0),
            ("abc", 1.0, 1.0),
        ],
    )
    def test_conversion(
        self, value: Any, default: float, expected: float
    ) -> None:
        assert _safe_nonnegative_finite_float(value, default) == pytest.approx(
            expected
        )


# ======================================================================
# Tests: _safe_int_from_any
# ======================================================================


class TestSafeIntFromAny:
    """Pruebas para conversión segura a int."""

    @pytest.mark.parametrize(
        "value, default, expected",
        [
            (42, 0, 42),
            (3.7, 0, 3),
            ("10", 0, 10),
            ("3.9", 0, 3),
            (None, -1, -1),
            (True, 0, 0),
            (False, 0, 0),
            ("abc", 5, 5),
            ([], 0, 0),
        ],
    )
    def test_conversion(
        self, value: Any, default: int, expected: int
    ) -> None:
        assert _safe_int_from_any(value, default) == expected

    def test_bool_excluded(self) -> None:
        """bool no se convierte a 0/1 sino que retorna default."""
        assert _safe_int_from_any(True, 99) == 99
        assert _safe_int_from_any(False, 99) == 99


# ======================================================================
# Tests: _truncate_text
# ======================================================================


class TestTruncateText:
    """Pruebas para truncamiento de texto."""

    def test_short_text_unchanged(self) -> None:
        assert _truncate_text("hola", 20) == "hola"

    def test_exact_length_unchanged(self) -> None:
        text = "a" * LABEL_MAX_LENGTH
        assert _truncate_text(text) == text

    def test_long_text_truncated_with_ellipsis(self) -> None:
        text = "a" * 30
        result = _truncate_text(text, 20)
        assert result.endswith(LABEL_ELLIPSIS)
        assert len(result) <= 20 + len(LABEL_ELLIPSIS)

    def test_truncation_at_word_boundary(self) -> None:
        text = "hello world extended text"
        result = _truncate_text(text, 15)
        assert LABEL_ELLIPSIS in result
        # No debe cortar a mitad de palabra si hay espacio adecuado
        content = result.replace(LABEL_ELLIPSIS, "")
        assert not content.endswith(" ")

    def test_empty_string_returns_empty(self) -> None:
        assert _truncate_text("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        assert _truncate_text("   ") == ""

    def test_none_like_empty(self) -> None:
        assert _truncate_text("", 10) == ""


# ======================================================================
# Tests: _deduplicate_preserve_order
# ======================================================================


class TestDeduplicatePreserveOrder:
    """Pruebas para deduplicación con orden preservado."""

    def test_no_duplicates(self) -> None:
        assert _deduplicate_preserve_order(["a", "b", "c"]) == ("a", "b", "c")

    def test_with_duplicates(self) -> None:
        assert _deduplicate_preserve_order(["a", "b", "a", "c", "b"]) == (
            "a",
            "b",
            "c",
        )

    def test_empty(self) -> None:
        assert _deduplicate_preserve_order([]) == ()

    def test_all_same(self) -> None:
        assert _deduplicate_preserve_order(["x", "x", "x"]) == ("x",)

    def test_preserves_first_occurrence_order(self) -> None:
        result = _deduplicate_preserve_order(["c", "b", "a", "b", "c"])
        assert result == ("c", "b", "a")


# ======================================================================
# Tests: _get_visible_levels
# ======================================================================


class TestGetVisibleLevels:
    """Pruebas para determinación de niveles visibles."""

    def test_none_returns_none(self) -> None:
        assert _get_visible_levels(None) is None

    @pytest.mark.parametrize("stratum", [0, 1, 2, 3])
    def test_valid_stratum(self, stratum: int) -> None:
        result = _get_visible_levels(stratum)
        assert result is not None
        assert result == STRATUM_VISIBLE_LEVELS[stratum]

    def test_invalid_stratum_returns_none(self) -> None:
        assert _get_visible_levels(99) is None
        assert _get_visible_levels(-1) is None

    def test_stratum_3_shows_only_level_3(self) -> None:
        assert _get_visible_levels(3) == frozenset({3})

    def test_stratum_0_shows_levels_0_1(self) -> None:
        assert _get_visible_levels(0) == frozenset({0, 1})


# ======================================================================
# Tests: validate_session_data
# ======================================================================


class TestValidateSessionData:
    """Pruebas para validación de datos de sesión."""

    def test_none_data(self) -> None:
        result = validate_session_data(None)
        assert result.ok is False
        assert "None" in (result.message or "")

    def test_non_dict_data(self) -> None:
        result = validate_session_data("not a dict")
        assert result.ok is False

    def test_no_keys_present(self) -> None:
        result = validate_session_data({})
        assert result.ok is False

    def test_valid_presupuesto_only(self) -> None:
        result = validate_session_data({"presupuesto": [{"a": 1}]})
        assert result.ok is True

    def test_valid_apus_only(self) -> None:
        result = validate_session_data({"apus_detail": [{"b": 2}]})
        assert result.ok is True

    def test_both_valid(self) -> None:
        result = validate_session_data(
            {"presupuesto": [{"a": 1}], "apus_detail": [{"b": 2}]}
        )
        assert result.ok is True

    def test_presupuesto_wrong_type(self) -> None:
        result = validate_session_data({"presupuesto": "not_a_list"})
        assert result.ok is False
        assert "lista" in (result.message or "")

    def test_apus_wrong_type(self) -> None:
        result = validate_session_data({"apus_detail": 42})
        assert result.ok is False

    def test_both_empty_lists(self) -> None:
        result = validate_session_data({"presupuesto": [], "apus_detail": []})
        assert result.ok is False
        assert "vacías" in (result.message or "")

    def test_one_empty_one_non_empty(self) -> None:
        result = validate_session_data(
            {"presupuesto": [], "apus_detail": [{"data": 1}]}
        )
        assert result.ok is True


# ======================================================================
# Tests: validate_graph
# ======================================================================


class TestValidateGraph:
    """Pruebas para validación de grafos."""

    def test_none_graph(self) -> None:
        assert validate_graph(None).ok is False

    def test_non_graph_type(self) -> None:
        assert validate_graph("not_a_graph").ok is False
        assert validate_graph({}).ok is False

    def test_valid_digraph(self, simple_digraph: nx.DiGraph) -> None:
        assert validate_graph(simple_digraph).ok is True

    def test_valid_graph(self) -> None:
        assert validate_graph(nx.Graph()).ok is True

    def test_valid_multigraph(self) -> None:
        assert validate_graph(nx.MultiDiGraph()).ok is True

    def test_empty_digraph_valid(self) -> None:
        assert validate_graph(nx.DiGraph()).ok is True

    def test_corrupted_graph(self) -> None:
        g = MagicMock(spec=nx.DiGraph)
        g.number_of_nodes.side_effect = RuntimeError("corrupted")
        result = validate_graph(g)
        assert result.ok is False
        assert "corrupto" in (result.message or "").lower() or "inaccesible" in (
            result.message or ""
        ).lower()


# ======================================================================
# Tests: validate_stratum_filter
# ======================================================================


class TestValidateStratumFilter:
    """Pruebas para validación de filtro de estrato."""

    def test_none_is_valid(self) -> None:
        assert validate_stratum_filter(None).ok is True

    @pytest.mark.parametrize("val", [0, 1, 2, 3])
    def test_valid_values(self, val: int) -> None:
        assert validate_stratum_filter(val).ok is True

    def test_out_of_range(self) -> None:
        assert validate_stratum_filter(4).ok is False
        assert validate_stratum_filter(-1).ok is False
        assert validate_stratum_filter(99).ok is False

    def test_non_int_type(self) -> None:
        assert validate_stratum_filter("2").ok is False  # type: ignore[arg-type]
        assert validate_stratum_filter(2.0).ok is False  # type: ignore[arg-type]

    def test_bool_excluded(self) -> None:
        assert validate_stratum_filter(True).ok is False  # type: ignore[arg-type]
        assert validate_stratum_filter(False).ok is False  # type: ignore[arg-type]


# ======================================================================
# Tests: _extract_dataframe_from_list
# ======================================================================


class TestExtractDataframeFromList:
    """Pruebas para extracción de DataFrame desde listas."""

    def test_valid_list_of_dicts(self) -> None:
        items = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        df = _extract_dataframe_from_list(items, "test")
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_non_list_returns_empty(self) -> None:
        df = _extract_dataframe_from_list("not_list", "test")
        assert df.empty

    def test_none_returns_empty(self) -> None:
        df = _extract_dataframe_from_list(None, "test")
        assert df.empty

    def test_filters_non_dict_elements(self) -> None:
        items = [{"a": 1}, "bad", 42, {"a": 2}]
        df = _extract_dataframe_from_list(items, "test")
        assert len(df) == 2

    def test_all_non_dict_returns_empty(self) -> None:
        df = _extract_dataframe_from_list([1, 2, "x"], "test")
        assert df.empty

    def test_empty_list_returns_empty(self) -> None:
        df = _extract_dataframe_from_list([], "test")
        assert df.empty


# ======================================================================
# Tests: extract_dataframes_from_session
# ======================================================================


class TestExtractDataframesFromSession:
    """Pruebas para extracción de DataFrames de sesión."""

    def test_both_present(self) -> None:
        data = {
            "presupuesto": [{"col": 1}],
            "apus_detail": [{"col": 2}],
        }
        df_p, df_a = extract_dataframes_from_session(data)
        assert len(df_p) == 1
        assert len(df_a) == 1

    def test_only_presupuesto(self) -> None:
        data = {"presupuesto": [{"col": 1}]}
        df_p, df_a = extract_dataframes_from_session(data)
        assert len(df_p) == 1
        assert df_a.empty

    def test_empty_data(self) -> None:
        df_p, df_a = extract_dataframes_from_session({})
        assert df_p.empty
        assert df_a.empty


# ======================================================================
# Tests: _extract_ids_from_list
# ======================================================================


class TestExtractIdsFromList:
    """Pruebas para extracción de IDs desde listas heterogéneas."""

    def test_string_items(self) -> None:
        assert _extract_ids_from_list(["A", "B", "C"]) == {"A", "B", "C"}

    def test_dict_items_with_id(self) -> None:
        items = [{"id": "X"}, {"id": "Y"}]
        assert _extract_ids_from_list(items) == {"X", "Y"}

    def test_mixed_items(self) -> None:
        items = ["A", {"id": "B"}, 42]
        result = _extract_ids_from_list(items)
        assert "A" in result
        assert "B" in result
        assert "42" in result

    def test_empty_ids_filtered(self) -> None:
        items = [None, "", "  ", {"id": None}, {"id": ""}]
        result = _extract_ids_from_list(items)
        assert len(result) == 0

    def test_non_list_returns_empty(self) -> None:
        assert _extract_ids_from_list("not_list") == set()
        assert _extract_ids_from_list(None) == set()

    def test_empty_list(self) -> None:
        assert _extract_ids_from_list([]) == set()


# ======================================================================
# Tests: _parse_single_cycle / _extract_nodes_from_cycles
# ======================================================================


class TestCycleParsing:
    """Pruebas para parseo de ciclos."""

    def test_string_cycle_with_arrow(self) -> None:
        nodes = _parse_single_cycle("A -> B -> C -> A")
        assert nodes == {"A", "B", "C"}

    def test_string_cycle_with_unicode_arrow(self) -> None:
        nodes = _parse_single_cycle("X → Y → Z → X")
        assert nodes == {"X", "Y", "Z"}

    def test_string_cycle_without_closing(self) -> None:
        nodes = _parse_single_cycle("A -> B -> C")
        assert nodes == {"A", "B", "C"}

    def test_list_cycle(self) -> None:
        nodes = _parse_single_cycle(["A", "B", "C", "A"])
        assert nodes == {"A", "B", "C"}

    def test_tuple_cycle(self) -> None:
        nodes = _parse_single_cycle(("X", "Y", "X"))
        assert nodes == {"X", "Y"}

    def test_degenerate_single_node(self) -> None:
        nodes = _parse_single_cycle("A")
        assert nodes == set()

    def test_empty_string(self) -> None:
        assert _parse_single_cycle("") == set()

    def test_unsupported_type(self) -> None:
        assert _parse_single_cycle(42) == set()

    def test_list_with_none_elements(self) -> None:
        nodes = _parse_single_cycle(["A", None, "B", "A"])
        assert "A" in nodes
        assert "B" in nodes

    def test_extract_from_multiple_cycles(self) -> None:
        cycles = ["A -> B -> A", "X -> Y -> Z -> X"]
        nodes = _extract_nodes_from_cycles(cycles)
        assert nodes == {"A", "B", "X", "Y", "Z"}

    def test_extract_from_non_list(self) -> None:
        assert _extract_nodes_from_cycles("not_a_list") == set()

    def test_extract_mixed_formats(self) -> None:
        cycles = ["A -> B -> A", ["X", "Y", "X"]]
        nodes = _extract_nodes_from_cycles(cycles)
        assert "A" in nodes and "B" in nodes
        assert "X" in nodes and "Y" in nodes


# ======================================================================
# Tests: _try_normalize_edge
# ======================================================================


class TestTryNormalizeEdge:
    """Pruebas para normalización de aristas desde formatos diversos."""

    def test_tuple_format(self) -> None:
        assert _try_normalize_edge(("A", "B")) == ("A", "B")

    def test_list_format(self) -> None:
        assert _try_normalize_edge(["A", "B"]) == ("A", "B")

    def test_dict_format(self) -> None:
        assert _try_normalize_edge({"source": "A", "target": "B"}) == ("A", "B")

    def test_dict_missing_keys_returns_none(self) -> None:
        assert _try_normalize_edge({"source": "A"}) is None

    def test_wrong_length_returns_none(self) -> None:
        assert _try_normalize_edge(("A",)) is None
        assert _try_normalize_edge(("A", "B", "C")) is None

    def test_unsupported_type_returns_none(self) -> None:
        assert _try_normalize_edge("A->B") is None
        assert _try_normalize_edge(42) is None


# ======================================================================
# Tests: extract_anomaly_data
# ======================================================================


class TestExtractAnomalyData:
    """Pruebas para extracción de datos de anomalía."""

    def test_non_dict_returns_empty(self) -> None:
        result = extract_anomaly_data("not_dict")  # type: ignore[arg-type]
        assert len(result.isolated_ids) == 0

    def test_missing_details_returns_empty(self) -> None:
        result = extract_anomaly_data({})
        assert len(result.isolated_ids) == 0

    def test_full_extraction(self) -> None:
        analysis = {
            "details": {
                "anomalies": {
                    "isolated_nodes": ["ISO-1"],
                    "orphan_insumos": [{"id": "ORP-1"}],
                    "empty_apus": ["EMP-1"],
                },
                "cycles": ["A -> B -> A"],
                "anomalous_nodes": ["ANO-1"],
                "node_scores": {"ANO-1": 0.9},
                "anomalous_edges": [("A", "B"), {"source": "C", "target": "D"}],
                "edge_scores": {("A", "B"): 0.85},  # Nota: clave como tupla
            }
        }
        result = extract_anomaly_data(analysis)
        assert "ISO-1" in result.isolated_ids
        assert "ORP-1" in result.orphan_ids
        assert "EMP-1" in result.empty_ids
        assert "A" in result.nodes_in_cycles
        assert "B" in result.nodes_in_cycles
        assert "ANO-1" in result.anomalous_nodes
        assert result.node_scores.get("ANO-1") == pytest.approx(0.9)

    def test_cycles_as_dict_with_list(self) -> None:
        analysis = {
            "details": {
                "cycles": {"list": ["X -> Y -> X"]},
            }
        }
        result = extract_anomaly_data(analysis)
        assert "X" in result.nodes_in_cycles
        assert "Y" in result.nodes_in_cycles

    def test_result_is_frozen(self) -> None:
        result = extract_anomaly_data({"details": {}})
        with pytest.raises(AttributeError):
            result.isolated_ids = frozenset({"X"})  # type: ignore[misc]


# ======================================================================
# Tests: _identify_stressed_nodes
# ======================================================================


class TestIdentifyStressedNodes:
    """Pruebas para detección de estrés topológico."""

    def test_stressed_node_detected(
        self, stress_test_graph: nx.DiGraph
    ) -> None:
        result = _identify_stressed_nodes(stress_test_graph)
        assert "INS-POPULAR" in result

    def test_rare_node_not_stressed(
        self, stress_test_graph: nx.DiGraph
    ) -> None:
        result = _identify_stressed_nodes(stress_test_graph)
        assert "INS-RARE" not in result

    def test_non_digraph_returns_empty(self) -> None:
        g = nx.Graph()
        g.add_node("A", type="APU")
        result = _identify_stressed_nodes(g)  # type: ignore[arg-type]
        assert len(result) == 0

    def test_no_apus_returns_empty(self) -> None:
        g = nx.DiGraph()
        g.add_node("INS-1", type="INSUMO")
        result = _identify_stressed_nodes(g)
        assert len(result) == 0

    def test_empty_graph(self) -> None:
        result = _identify_stressed_nodes(nx.DiGraph())
        assert len(result) == 0

    def test_result_is_frozenset(
        self, stress_test_graph: nx.DiGraph
    ) -> None:
        result = _identify_stressed_nodes(stress_test_graph)
        assert isinstance(result, frozenset)

    def test_custom_config(self) -> None:
        g = nx.DiGraph()
        for i in range(5):
            g.add_node(f"APU-{i}", type="APU", level=2)
        g.add_node("INS-1", type="INSUMO", level=3)
        for i in range(3):
            g.add_edge(f"APU-{i}", "INS-1")

        # ratio = 3/5 = 0.6, default small threshold = 0.5 → stressed
        result = _identify_stressed_nodes(g)
        assert "INS-1" in result

        # Con threshold alto no se detecta
        strict_config = StressConfig(ratio_small=0.8)
        result_strict = _identify_stressed_nodes(g, config=strict_config)
        assert "INS-1" not in result_strict


# ======================================================================
# Tests: _get_node_type
# ======================================================================


class TestGetNodeType:
    """Pruebas para extracción de tipo de nodo."""

    @pytest.mark.parametrize(
        "type_val, expected",
        [
            ("APU", "APU"),
            ("apu", "APU"),
            ("  Insumo  ", "INSUMO"),
            ("BUDGET", "BUDGET"),
            ("CHAPTER", "CHAPTER"),
            ("ITEM", "ITEM"),
            ("UNKNOWN", "UNKNOWN"),
        ],
    )
    def test_valid_types(self, type_val: str, expected: str) -> None:
        assert _get_node_type({"type": type_val}) == expected

    def test_none_type(self) -> None:
        assert _get_node_type({"type": None}) == "UNKNOWN"
        assert _get_node_type({}) == "UNKNOWN"

    def test_invalid_type(self) -> None:
        assert _get_node_type({"type": "BANANA"}) == "UNKNOWN"

    def test_numeric_type(self) -> None:
        assert _get_node_type({"type": 42}) == "UNKNOWN"


# ======================================================================
# Tests: _get_node_cost
# ======================================================================


class TestGetNodeCost:
    """Pruebas para extracción de costo de nodo."""

    def test_total_cost_preferred(self) -> None:
        assert _get_node_cost({"total_cost": 100.0, "unit_cost": 50.0}) == 100.0

    def test_unit_cost_fallback(self) -> None:
        assert _get_node_cost({"unit_cost": 50.0}) == 50.0

    def test_no_cost_returns_zero(self) -> None:
        assert _get_node_cost({}) == 0.0

    def test_zero_total_cost_is_valid(self) -> None:
        assert _get_node_cost({"total_cost": 0.0}) == 0.0

    def test_negative_total_cost_falls_through(self) -> None:
        result = _get_node_cost({"total_cost": -5.0, "unit_cost": 10.0})
        assert result == 10.0

    def test_inf_cost_returns_zero(self) -> None:
        assert _get_node_cost({"total_cost": float("inf")}) == 0.0

    def test_nan_cost_returns_zero(self) -> None:
        assert _get_node_cost({"total_cost": float("nan")}) == 0.0


# ======================================================================
# Tests: _get_node_score / _get_edge_score
# ======================================================================


class TestScores:
    """Pruebas para obtención de scores de anomalía."""

    def test_node_score_present(self, rich_anomaly_data: AnomalyData) -> None:
        assert _get_node_score("CYC-A", rich_anomaly_data) == pytest.approx(0.95)

    def test_node_score_absent(self, rich_anomaly_data: AnomalyData) -> None:
        assert _get_node_score("UNKNOWN", rich_anomaly_data) == 0.0

    def test_edge_score_forward(self, rich_anomaly_data: AnomalyData) -> None:
        assert _get_edge_score("CYC-A", "CYC-B", rich_anomaly_data) == pytest.approx(
            0.85
        )

    def test_edge_score_reverse(self, rich_anomaly_data: AnomalyData) -> None:
        assert _get_edge_score("CYC-B", "CYC-A", rich_anomaly_data) == pytest.approx(
            0.85
        )

    def test_edge_score_absent(self, rich_anomaly_data: AnomalyData) -> None:
        assert _get_edge_score("X", "Y", rich_anomaly_data) == 0.0

    def test_edge_score_empty_anomaly(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        assert _get_edge_score("A", "B", empty_anomaly_data) == 0.0


# ======================================================================
# Tests: _determine_node_classes
# ======================================================================


class TestDetermineNodeClasses:
    """Pruebas para determinación de clases CSS de nodos."""

    def test_normal_node(self, empty_anomaly_data: AnomalyData) -> None:
        classes = _determine_node_classes("N1", "APU", empty_anomaly_data)
        assert "APU" in classes
        assert NodeClass.NORMAL.value in classes

    def test_cycle_node(self, rich_anomaly_data: AnomalyData) -> None:
        classes = _determine_node_classes("CYC-A", "APU", rich_anomaly_data)
        assert NodeClass.CIRCULAR.value in classes
        assert NodeClass.CYCLE.value in classes
        assert NodeClass.NORMAL.value not in classes

    def test_isolated_node(self, rich_anomaly_data: AnomalyData) -> None:
        classes = _determine_node_classes("ISO-1", "INSUMO", rich_anomaly_data)
        assert NodeClass.ISOLATED.value in classes

    def test_empty_node(self, rich_anomaly_data: AnomalyData) -> None:
        classes = _determine_node_classes("EMP-1", "APU", rich_anomaly_data)
        assert NodeClass.EMPTY.value in classes

    def test_stressed_node(self, rich_anomaly_data: AnomalyData) -> None:
        classes = _determine_node_classes("STR-1", "INSUMO", rich_anomaly_data)
        assert NodeClass.STRESS.value in classes

    def test_hot_node(self, rich_anomaly_data: AnomalyData) -> None:
        classes = _determine_node_classes("HOT-1", "INSUMO", rich_anomaly_data)
        assert NodeClass.HOT.value in classes

    def test_multiple_anomalies(self, rich_anomaly_data: AnomalyData) -> None:
        """CYC-A es tanto cycle como anomalous."""
        classes = _determine_node_classes("CYC-A", "APU", rich_anomaly_data)
        assert NodeClass.CIRCULAR.value in classes
        assert NodeClass.ANOMALOUS.value in classes
        assert NodeClass.NORMAL.value not in classes

    def test_node_type_always_first(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        classes = _determine_node_classes("N1", "INSUMO", empty_anomaly_data)
        assert classes[0] == "INSUMO"

    def test_no_duplicates(self, rich_anomaly_data: AnomalyData) -> None:
        classes = _determine_node_classes("CYC-A", "APU", rich_anomaly_data)
        assert len(classes) == len(set(classes))


# ======================================================================
# Tests: _determine_node_color
# ======================================================================


class TestDetermineNodeColor:
    """Pruebas para determinación de color de nodos."""

    def test_anomalous_node_red(
        self, rich_anomaly_data: AnomalyData
    ) -> None:
        assert (
            _determine_node_color("CYC-A", "APU", rich_anomaly_data)
            == NodeColor.RED.value
        )
        assert (
            _determine_node_color("ISO-1", "INSUMO", rich_anomaly_data)
            == NodeColor.RED.value
        )
        assert (
            _determine_node_color("EMP-1", "APU", rich_anomaly_data)
            == NodeColor.RED.value
        )

    def test_budget_node_black(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        assert (
            _determine_node_color("B1", "BUDGET", empty_anomaly_data)
            == NodeColor.BLACK.value
        )

    def test_chapter_node_black(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        assert (
            _determine_node_color("C1", "CHAPTER", empty_anomaly_data)
            == NodeColor.BLACK.value
        )

    def test_apu_node_blue(self, empty_anomaly_data: AnomalyData) -> None:
        assert (
            _determine_node_color("A1", "APU", empty_anomaly_data)
            == NodeColor.BLUE.value
        )

    def test_insumo_node_orange(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        assert (
            _determine_node_color("I1", "INSUMO", empty_anomaly_data)
            == NodeColor.ORANGE.value
        )

    def test_unknown_node_gray(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        assert (
            _determine_node_color("U1", "UNKNOWN", empty_anomaly_data)
            == NodeColor.GRAY.value
        )


# ======================================================================
# Tests: _build_node_label
# ======================================================================


class TestBuildNodeLabel:
    """Pruebas para construcción de etiquetas de nodo."""

    def test_with_description(self) -> None:
        label = _build_node_label("N1", {"description": "Cemento"})
        assert "N1" in label
        assert "Cemento" in label

    def test_without_description(self) -> None:
        assert _build_node_label("N1", {}) == "N1"

    def test_none_description(self) -> None:
        assert _build_node_label("N1", {"description": None}) == "N1"

    def test_empty_description(self) -> None:
        assert _build_node_label("N1", {"description": ""}) == "N1"

    def test_long_description_truncated(self) -> None:
        long_desc = "A" * 50
        label = _build_node_label("N1", {"description": long_desc})
        assert LABEL_ELLIPSIS in label


# ======================================================================
# Tests: _build_node_tooltip
# ======================================================================


class TestBuildNodeTooltip:
    """Pruebas para construcción de tooltips de nodo."""

    def test_basic_tooltip(self, empty_anomaly_data: AnomalyData) -> None:
        tooltip = _build_node_tooltip(
            "N1", {}, empty_anomaly_data, "APU", 100.0, 0.5, 2
        )
        assert "ID: N1" in tooltip
        assert "Tipo: APU" in tooltip
        assert "Nivel: 2" in tooltip
        assert "Costo: 100.00" in tooltip
        assert "Score: 0.5000" in tooltip
        assert "sin anomalías" in tooltip

    def test_tooltip_with_description(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        tooltip = _build_node_tooltip(
            "N1",
            {"description": "Test desc"},
            empty_anomaly_data,
            "APU",
            0.0,
            0.0,
            0,
        )
        assert "Descripción: Test desc" in tooltip

    def test_tooltip_with_anomalies(
        self, rich_anomaly_data: AnomalyData
    ) -> None:
        tooltip = _build_node_tooltip(
            "CYC-A", {}, rich_anomaly_data, "APU", 0.0, 0.95, 2
        )
        assert "ciclo" in tooltip
        assert "anómalo" in tooltip
        assert "sin anomalías" not in tooltip


# ======================================================================
# Tests: _build_edge_tooltip
# ======================================================================


class TestBuildEdgeTooltip:
    """Pruebas para construcción de tooltips de arista."""

    def test_basic_tooltip(self) -> None:
        tooltip = _build_edge_tooltip("A", "B", 50.0, 0.8, True)
        assert "Origen: A" in tooltip
        assert "Destino: B" in tooltip
        assert "Costo: 50.00" in tooltip
        assert "Score: 0.8000" in tooltip
        assert "Evidencia forense: sí" in tooltip

    def test_non_evidence(self) -> None:
        tooltip = _build_edge_tooltip("A", "B", 0.0, 0.0, False)
        assert "Evidencia forense: no" in tooltip


# ======================================================================
# Tests: build_node_element
# ======================================================================


class TestBuildNodeElement:
    """Pruebas para construcción completa de nodos Cytoscape."""

    def test_normal_node(self, empty_anomaly_data: AnomalyData) -> None:
        node = build_node_element(
            "N1",
            {"type": "APU", "level": 2, "total_cost": 100.0},
            empty_anomaly_data,
        )
        assert node.id == "N1"
        assert node.node_type == "APU"
        assert node.level == 2
        assert node.cost == pytest.approx(100.0)
        assert node.is_evidence is False
        assert node.color == NodeColor.BLUE.value
        assert NodeClass.NORMAL.value in node.classes

    def test_anomalous_node(self, rich_anomaly_data: AnomalyData) -> None:
        node = build_node_element(
            "CYC-A",
            {"type": "APU", "level": 2},
            rich_anomaly_data,
        )
        assert node.is_evidence is True
        assert node.color == NodeColor.RED.value
        assert NodeClass.CIRCULAR.value in node.classes

    def test_none_attrs_handled(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        node = build_node_element("N1", None, empty_anomaly_data)  # type: ignore[arg-type]
        assert node.id == "N1"
        assert node.node_type == "UNKNOWN"

    def test_none_node_id(self, empty_anomaly_data: AnomalyData) -> None:
        node = build_node_element(None, {"type": "APU"}, empty_anomaly_data)
        assert node.id == "unknown"

    def test_serialization_roundtrip(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        node = build_node_element(
            "N1", {"type": "APU", "level": 2}, empty_anomaly_data
        )
        d = node.to_dict()
        assert d["data"]["id"] == "N1"
        assert isinstance(d["classes"], str)


# ======================================================================
# Tests: _build_fallback_node / _build_fallback_edge
# ======================================================================


class TestFallbacks:
    """Pruebas para elementos fallback."""

    def test_fallback_node(self) -> None:
        node = _build_fallback_node("FAIL-1")
        assert node.id == "FAIL-1"
        assert node.node_type == NodeType.UNKNOWN.value
        assert node.color == NodeColor.GRAY.value
        assert node.is_evidence is False
        assert "fallback" in node.tooltip.lower()

    def test_fallback_edge(self) -> None:
        edge = _build_fallback_edge("A", "B")
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.cost == 0.0
        assert edge.is_evidence is False
        assert "fallback" in edge.tooltip.lower()

    def test_fallback_node_serializable(self) -> None:
        d = _build_fallback_node("X").to_dict()
        assert "data" in d
        assert "classes" in d

    def test_fallback_edge_serializable(self) -> None:
        d = _build_fallback_edge("A", "B").to_dict()
        assert d["data"]["source"] == "A"


# ======================================================================
# Tests: build_edge_element
# ======================================================================


class TestBuildEdgeElement:
    """Pruebas para construcción completa de aristas Cytoscape."""

    def test_normal_edge(self, empty_anomaly_data: AnomalyData) -> None:
        edge = build_edge_element(
            "A", "B", {"total_cost": 50.0}, empty_anomaly_data
        )
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.cost == pytest.approx(50.0)
        assert edge.is_evidence is False

    def test_anomalous_edge_explicit(
        self, rich_anomaly_data: AnomalyData
    ) -> None:
        edge = build_edge_element("CYC-A", "CYC-B", {}, rich_anomaly_data)
        assert edge.is_evidence is True

    def test_anomalous_edge_heuristic(
        self, rich_anomaly_data: AnomalyData
    ) -> None:
        """Ambos nodos en ciclo → evidencia heurística."""
        edge = build_edge_element("CYC-B", "CYC-C", {}, rich_anomaly_data)
        assert edge.is_evidence is True

    def test_invalid_source_raises(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        with pytest.raises(ValueError, match="Arista inválida"):
            build_edge_element(None, "B", {}, empty_anomaly_data)

    def test_invalid_target_raises(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        with pytest.raises(ValueError, match="Arista inválida"):
            build_edge_element("A", "", {}, empty_anomaly_data)

    def test_none_attrs_handled(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        edge = build_edge_element("A", "B", None, empty_anomaly_data)  # type: ignore[arg-type]
        assert edge.cost == 0.0

    def test_edge_score_present(
        self, rich_anomaly_data: AnomalyData
    ) -> None:
        edge = build_edge_element("CYC-A", "CYC-B", {}, rich_anomaly_data)
        assert edge.score == pytest.approx(0.85)


# ======================================================================
# Tests: convert_graph_to_cytoscape_elements
# ======================================================================


class TestConvertGraphToCytoscapeElements:
    """Pruebas de integración para conversión completa grafo → Cytoscape."""

    def test_basic_conversion(
        self,
        simple_digraph: nx.DiGraph,
        empty_anomaly_data: AnomalyData,
    ) -> None:
        elements = convert_graph_to_cytoscape_elements(
            simple_digraph, empty_anomaly_data
        )
        assert len(elements) > 0

        node_elements = [
            e for e in elements if "source" not in e.get("data", {})
        ]
        edge_elements = [
            e for e in elements if "source" in e.get("data", {})
        ]

        assert len(node_elements) == simple_digraph.number_of_nodes()
        assert len(edge_elements) == simple_digraph.number_of_edges()

    def test_stratum_filter_level_3(
        self,
        simple_digraph: nx.DiGraph,
        empty_anomaly_data: AnomalyData,
    ) -> None:
        elements = convert_graph_to_cytoscape_elements(
            simple_digraph, empty_anomaly_data, stratum_filter=3
        )
        node_elements = [
            e for e in elements if "source" not in e.get("data", {})
        ]
        # Solo nodos nivel 3 (INS-1, INS-2)
        assert len(node_elements) == 2
        for ne in node_elements:
            assert ne["data"]["level"] == 3

    def test_stratum_filter_level_0(
        self,
        simple_digraph: nx.DiGraph,
        empty_anomaly_data: AnomalyData,
    ) -> None:
        elements = convert_graph_to_cytoscape_elements(
            simple_digraph, empty_anomaly_data, stratum_filter=0
        )
        node_elements = [
            e for e in elements if "source" not in e.get("data", {})
        ]
        for ne in node_elements:
            assert ne["data"]["level"] in {0, 1}

    def test_edges_only_between_visible_nodes(
        self,
        simple_digraph: nx.DiGraph,
        empty_anomaly_data: AnomalyData,
    ) -> None:
        elements = convert_graph_to_cytoscape_elements(
            simple_digraph, empty_anomaly_data, stratum_filter=3
        )
        node_ids = {
            e["data"]["id"]
            for e in elements
            if "source" not in e.get("data", {})
        }
        edge_elements = [
            e for e in elements if "source" in e.get("data", {})
        ]
        for edge in edge_elements:
            assert edge["data"]["source"] in node_ids
            assert edge["data"]["target"] in node_ids

    def test_invalid_graph_raises(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        with pytest.raises(ValueError):
            convert_graph_to_cytoscape_elements(
                None, empty_anomaly_data  # type: ignore[arg-type]
            )

    def test_invalid_stratum_raises(
        self,
        simple_digraph: nx.DiGraph,
        empty_anomaly_data: AnomalyData,
    ) -> None:
        with pytest.raises(ValueError):
            convert_graph_to_cytoscape_elements(
                simple_digraph, empty_anomaly_data, stratum_filter=99  # type: ignore[arg-type]
            )

    def test_empty_graph_returns_empty(
        self, empty_anomaly_data: AnomalyData
    ) -> None:
        elements = convert_graph_to_cytoscape_elements(
            nx.DiGraph(), empty_anomaly_data
        )
        assert elements == []

    def test_no_filter_shows_all(
        self,
        simple_digraph: nx.DiGraph,
        empty_anomaly_data: AnomalyData,
    ) -> None:
        elements = convert_graph_to_cytoscape_elements(
            simple_digraph, empty_anomaly_data, stratum_filter=None
        )
        node_elements = [
            e for e in elements if "source" not in e.get("data", {})
        ]
        assert len(node_elements) == simple_digraph.number_of_nodes()

    def test_anomalous_nodes_marked(
        self,
        simple_digraph: nx.DiGraph,
    ) -> None:
        anomaly = AnomalyData(
            nodes_in_cycles=frozenset({"APU-1"}),
        )
        elements = convert_graph_to_cytoscape_elements(
            simple_digraph, anomaly
        )
        apu_elements = [
            e
            for e in elements
            if e.get("data", {}).get("id") == "APU-1"
        ]
        assert len(apu_elements) == 1
        assert apu_elements[0]["data"]["is_evidence"] is True
        assert apu_elements[0]["data"]["color"] == NodeColor.RED.value

    def test_deterministic_output(
        self,
        simple_digraph: nx.DiGraph,
        empty_anomaly_data: AnomalyData,
    ) -> None:
        r1 = convert_graph_to_cytoscape_elements(
            simple_digraph, empty_anomaly_data
        )
        r2 = convert_graph_to_cytoscape_elements(
            simple_digraph, empty_anomaly_data
        )
        assert len(r1) == len(r2)
        for e1, e2 in zip(r1, r2):
            assert e1["data"] == e2["data"]


# ======================================================================
# Tests: _parse_stratum_param
# ======================================================================


class TestParseStratumParam:
    """Pruebas para parseo de parámetro de estrato."""

    def test_none_returns_none(self) -> None:
        assert _parse_stratum_param(None) is None

    @pytest.mark.parametrize("val", ["0", "1", "2", "3"])
    def test_valid_values(self, val: str) -> None:
        assert _parse_stratum_param(val) == int(val)

    def test_invalid_number(self) -> None:
        assert _parse_stratum_param("99") is None
        assert _parse_stratum_param("-1") is None

    def test_non_numeric(self) -> None:
        assert _parse_stratum_param("abc") is None
        assert _parse_stratum_param("") is None

    def test_float_string(self) -> None:
        # "2.0" → int("2.0") → ValueError → None
        assert _parse_stratum_param("2.0") is None


# ======================================================================
# Tests: HTTP Responses
# ======================================================================


class TestHttpResponses:
    """Pruebas para funciones de respuesta HTTP."""

    @pytest.fixture(autouse=True)
    def _setup_app(self) -> Any:
        """Crea un contexto Flask mínimo para jsonify."""
        from flask import Flask

        app = Flask(__name__)
        with app.app_context():
            yield

    def test_error_response_structure(self) -> None:
        response, status = create_error_response("test error", 400)
        data = response.get_json()
        assert status == 400
        assert data["success"] is False
        assert data["error"] == "test error"
        assert data["elements"] == []
        assert data["count"] == 0

    def test_success_response_structure(self) -> None:
        elements = [{"data": {"id": "N1"}}]
        response, status = create_success_response(elements, 200)
        data = response.get_json()
        assert status == 200
        assert data["success"] is True
        assert data["error"] is None
        assert data["count"] == 1
        assert len(data["elements"]) == 1

    def test_success_response_empty(self) -> None:
        response, status = create_success_response([])
        data = response.get_json()
        assert data["count"] == 0
        assert data["elements"] == []

    def test_error_response_different_codes(self) -> None:
        _, status_404 = create_error_response("not found", 404)
        _, status_500 = create_error_response("internal", 500)
        assert status_404 == 404
        assert status_500 == 500


# ======================================================================
# Tests: build_graph_from_session (con mocks)
# ======================================================================


class TestBuildGraphFromSession:
    """Pruebas para construcción de grafo desde sesión."""

    def test_empty_data_raises(self) -> None:
        with pytest.raises(ValueError, match="No hay datos suficientes"):
            build_graph_from_session({"presupuesto": [], "apus_detail": []})

    @patch("topology_viz.BudgetGraphBuilder")
    def test_builder_exception_wrapped(
        self, mock_builder_class: MagicMock
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.build.side_effect = RuntimeError("build failed")
        mock_builder_class.return_value = mock_instance

        with pytest.raises(ValueError, match="Error construyendo grafo"):
            build_graph_from_session(
                {"presupuesto": [{"col": 1}]}
            )

    @patch("topology_viz.BudgetGraphBuilder")
    def test_invalid_graph_raises(
        self, mock_builder_class: MagicMock
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.build.return_value = "not_a_graph"
        mock_builder_class.return_value = mock_instance

        with pytest.raises(ValueError, match="Grafo inválido"):
            build_graph_from_session(
                {"presupuesto": [{"col": 1}]}
            )

    @patch("topology_viz.BudgetGraphBuilder")
    def test_successful_build(
        self, mock_builder_class: MagicMock
    ) -> None:
        g = nx.DiGraph()
        g.add_node("A")
        mock_instance = MagicMock()
        mock_instance.build.return_value = g
        mock_builder_class.return_value = mock_instance

        result = build_graph_from_session(
            {"presupuesto": [{"col": 1}]}
        )
        assert isinstance(result, nx.DiGraph)
        assert result.number_of_nodes() == 1

    @patch("topology_viz.BudgetGraphBuilder")
    def test_non_digraph_converted(
        self, mock_builder_class: MagicMock
    ) -> None:
        """Grafo no dirigido se convierte a DiGraph."""
        g = nx.Graph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")
        mock_instance = MagicMock()
        mock_instance.build.return_value = g
        mock_builder_class.return_value = mock_instance

        result = build_graph_from_session(
            {"presupuesto": [{"col": 1}]}
        )
        assert isinstance(result, nx.DiGraph)


# ======================================================================
# Tests: analyze_graph_for_visualization (con mocks)
# ======================================================================


class TestAnalyzeGraphForVisualization:
    """Pruebas para análisis topológico de visualización."""

    @patch("topology_viz.BusinessTopologicalAnalyzer")
    def test_returns_anomaly_data(
        self,
        mock_analyzer_class: MagicMock,
        simple_digraph: nx.DiGraph,
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze_structural_integrity.return_value = {
            "details": {"anomalies": {"isolated_nodes": ["ISO-1"]}}
        }
        mock_instance.analyze_thermal_flow.return_value = {
            "hotspots": [{"id": "HOT-1"}]
        }
        mock_analyzer_class.return_value = mock_instance

        result = analyze_graph_for_visualization(simple_digraph)
        assert isinstance(result, AnomalyData)
        assert "ISO-1" in result.isolated_ids

    @patch("topology_viz.BusinessTopologicalAnalyzer")
    def test_degradable_on_structural_error(
        self,
        mock_analyzer_class: MagicMock,
        simple_digraph: nx.DiGraph,
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze_structural_integrity.side_effect = RuntimeError(
            "fail"
        )
        mock_instance.analyze_thermal_flow.return_value = {"hotspots": []}
        mock_analyzer_class.return_value = mock_instance

        result = analyze_graph_for_visualization(simple_digraph)
        assert isinstance(result, AnomalyData)

    @patch("topology_viz.BusinessTopologicalAnalyzer")
    def test_degradable_on_thermal_error(
        self,
        mock_analyzer_class: MagicMock,
        simple_digraph: nx.DiGraph,
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze_structural_integrity.return_value = {
            "details": {}
        }
        mock_instance.analyze_thermal_flow.side_effect = RuntimeError("fail")
        mock_analyzer_class.return_value = mock_instance

        result = analyze_graph_for_visualization(simple_digraph)
        assert isinstance(result, AnomalyData)
        assert len(result.hot_ids) == 0

    @patch("topology_viz.BusinessTopologicalAnalyzer")
    def test_result_is_frozen(
        self,
        mock_analyzer_class: MagicMock,
        simple_digraph: nx.DiGraph,
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze_structural_integrity.return_value = {
            "details": {}
        }
        mock_instance.analyze_thermal_flow.return_value = {"hotspots": []}
        mock_analyzer_class.return_value = mock_instance

        result = analyze_graph_for_visualization(simple_digraph)
        with pytest.raises(AttributeError):
            result.hot_ids = frozenset({"X"})  # type: ignore[misc]


# ======================================================================
# Tests: Propiedades transversales del sistema
# ======================================================================


class TestSystemProperties:
    """Pruebas de propiedades transversales."""

    def test_allowed_stratum_filters_is_frozen(self) -> None:
        assert isinstance(ALLOWED_STRATUM_FILTERS, frozenset)

    def test_stratum_visible_levels_values_are_frozen(self) -> None:
        for key, value in STRATUM_VISIBLE_LEVELS.items():
            assert isinstance(value, frozenset), (
                f"STRATUM_VISIBLE_LEVELS[{key}] no es frozenset"
            )

    def test_stratum_levels_within_allowed(self) -> None:
        for key in STRATUM_VISIBLE_LEVELS:
            assert key in ALLOWED_STRATUM_FILTERS

    def test_node_type_enum_completeness(self) -> None:
        """Todos los tipos usados en pruebas existen en el enum."""
        expected = {"BUDGET", "CHAPTER", "ITEM", "APU", "INSUMO", "UNKNOWN"}
        actual = {m.value for m in NodeType}
        assert expected == actual

    def test_all_elements_have_data_key(
        self,
        simple_digraph: nx.DiGraph,
        empty_anomaly_data: AnomalyData,
    ) -> None:
        elements = convert_graph_to_cytoscape_elements(
            simple_digraph, empty_anomaly_data
        )
        for elem in elements:
            assert "data" in elem

    def test_no_orphan_edges(
        self,
        simple_digraph: nx.DiGraph,
        empty_anomaly_data: AnomalyData,
    ) -> None:
        """Toda arista conecta nodos presentes en los elementos."""
        elements = convert_graph_to_cytoscape_elements(
            simple_digraph, empty_anomaly_data
        )
        node_ids = {
            e["data"]["id"]
            for e in elements
            if "id" in e.get("data", {}) and "source" not in e.get("data", {})
        }
        for elem in elements:
            data = elem.get("data", {})
            if "source" in data:
                assert data["source"] in node_ids, (
                    f"Arista huérfana: source={data['source']}"
                )
                assert data["target"] in node_ids, (
                    f"Arista huérfana: target={data['target']}"
                )

    def test_total_elements_equals_nodes_plus_edges(
        self,
        simple_digraph: nx.DiGraph,
        empty_anomaly_data: AnomalyData,
    ) -> None:
        elements = convert_graph_to_cytoscape_elements(
            simple_digraph, empty_anomaly_data
        )
        nodes = [
            e for e in elements if "source" not in e.get("data", {})
        ]
        edges = [
            e for e in elements if "source" in e.get("data", {})
        ]
        assert len(elements) == len(nodes) + len(edges)
        assert len(nodes) == simple_digraph.number_of_nodes()
        assert len(edges) == simple_digraph.number_of_edges()

    def test_normalize_identifier_idempotence_on_graph_nodes(
        self, simple_digraph: nx.DiGraph
    ) -> None:
        for node_id in simple_digraph.nodes():
            normalized = _normalize_identifier(node_id)
            double_normalized = _normalize_identifier(normalized)
            assert normalized == double_normalized