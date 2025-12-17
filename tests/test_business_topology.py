from unittest.mock import MagicMock, Mock

import networkx as nx
import pandas as pd
import pytest

from agent.business_topology import (
    BudgetGraphBuilder,
    BusinessTopologicalAnalyzer,
    TopologicalMetrics,
)
from app.constants import ColumnNames

# =============================================================================
# FIXTURES COMPARTIDAS
# =============================================================================


class TestTopologicalMetricsDataclass:
    """Pruebas para el dataclass TopologicalMetrics."""

    def test_dataclass_creation(self):
        """Verifica la creación correcta del dataclass."""
        metrics = TopologicalMetrics(beta_0=2, beta_1=1, euler_characteristic=1)

        assert metrics.beta_0 == 2
        assert metrics.beta_1 == 1
        assert metrics.euler_characteristic == 1

    def test_is_connected_property_true(self):
        """β₀ = 1 implica espacio conexo."""
        metrics = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)
        assert metrics.is_connected is True

    def test_is_connected_property_false(self):
        """β₀ > 1 implica espacio no conexo."""
        metrics = TopologicalMetrics(beta_0=3, beta_1=0, euler_characteristic=3)
        assert metrics.is_connected is False

    def test_is_simply_connected_true(self):
        """β₀ = 1 y β₁ = 0 implica espacio simplemente conexo."""
        metrics = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)
        assert metrics.is_simply_connected is True

    def test_is_simply_connected_false_with_cycles(self):
        """β₁ > 0 implica presencia de ciclos (no simplemente conexo)."""
        metrics = TopologicalMetrics(beta_0=1, beta_1=2, euler_characteristic=-1)
        assert metrics.is_simply_connected is False

    def test_is_simply_connected_false_disconnected(self):
        """β₀ > 1 implica no simplemente conexo (aunque β₁ = 0)."""
        metrics = TopologicalMetrics(beta_0=2, beta_1=0, euler_characteristic=2)
        assert metrics.is_simply_connected is False

    def test_dataclass_immutability(self):
        """Verifica que el dataclass es inmutable (frozen=True)."""
        metrics = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)

        with pytest.raises(AttributeError):
            metrics.beta_0 = 5


# =============================================================================
# PRUEBAS PARA BudgetGraphBuilder
# =============================================================================


class TestBudgetGraphBuilder:
    """Pruebas para la clase BudgetGraphBuilder."""

    @pytest.fixture
    def builder(self):
        """Fixture para crear instancia del builder."""
        return BudgetGraphBuilder()

    # -------------------------------------------------------------------------
    # Pruebas de métodos auxiliares de sanitización
    # -------------------------------------------------------------------------

    class TestSanitizationMethods:
        """Pruebas para métodos de sanitización de datos."""

        @pytest.fixture
        def builder(self):
            return BudgetGraphBuilder()

        @pytest.mark.parametrize(
            "input_value,expected",
            [
                ("  APU-001  ", "APU-001"),
                ("APU-001", "APU-001"),
                ("", ""),
                ("   ", ""),
                (None, ""),
                (float("nan"), ""),
                (123, "123"),
                (12.5, "12.5"),
            ],
        )
        def test_sanitize_code(self, builder, input_value, expected):
            """Verifica sanitización de códigos con diversos inputs."""
            result = builder._sanitize_code(input_value)
            assert result == expected

        @pytest.mark.parametrize(
            "input_value,default,expected",
            [
                (10.5, 0.0, 10.5),
                ("10.5", 0.0, 10.5),
                (10, 0.0, 10.0),
                (None, 0.0, 0.0),
                (None, 5.0, 5.0),
                (float("nan"), 0.0, 0.0),
                ("invalid", 0.0, 0.0),
                ("", 0.0, 0.0),
                ([], 0.0, 0.0),
            ],
        )
        def test_safe_float(self, builder, input_value, default, expected):
            """Verifica conversión segura a float."""
            result = builder._safe_float(input_value, default)
            assert result == expected

        def test_safe_float_with_nan_pandas(self, builder):
            """Verifica manejo de NaN de pandas."""
            result = builder._safe_float(pd.NA, 0.0)
            assert result == 0.0

    # -------------------------------------------------------------------------
    # Pruebas de creación de atributos
    # -------------------------------------------------------------------------

    class TestAttributeCreation:
        """Pruebas para métodos de creación de atributos de nodos."""

        @pytest.fixture
        def builder(self):
            return BudgetGraphBuilder()

        def test_create_apu_attributes_primary(self, builder):
            """Verifica atributos de APU desde fuente primaria."""
            row = pd.Series(
                {
                    ColumnNames.DESCRIPCION_APU: "Construcción Muro",
                    ColumnNames.CANTIDAD_PRESUPUESTO: 100.0,
                }
            )

            attrs = builder._create_apu_attributes(
                row, source="presupuesto_df", idx=5, inferred=False
            )

            assert attrs["type"] == "APU"
            assert attrs["source"] == "presupuesto_df"
            assert attrs["original_index"] == 5
            assert attrs["inferred"] is False
            assert attrs["description"] == "Construcción Muro"
            assert attrs["quantity"] == 100.0

        def test_create_apu_attributes_inferred(self, builder):
            """Verifica atributos de APU inferido (sin descripción ni cantidad)."""
            row = pd.Series({})

            attrs = builder._create_apu_attributes(
                row, source="apus_detail_df", idx=10, inferred=True
            )

            assert attrs["type"] == "APU"
            assert attrs["inferred"] is True
            assert "description" not in attrs
            assert "quantity" not in attrs

        def test_create_insumo_attributes(self, builder):
            """Verifica atributos de nodo INSUMO."""
            row = pd.Series(
                {
                    ColumnNames.TIPO_INSUMO: "Material",
                    ColumnNames.COSTO_INSUMO_EN_APU: 150.75,
                }
            )

            attrs = builder._create_insumo_attributes(
                row, insumo_desc="Cemento Portland", source="apus_detail_df", idx=3
            )

            assert attrs["type"] == "INSUMO"
            assert attrs["description"] == "Cemento Portland"
            assert attrs["tipo_insumo"] == "Material"
            assert attrs["unit_cost"] == 150.75
            assert attrs["source"] == "apus_detail_df"
            assert attrs["original_index"] == 3

    # -------------------------------------------------------------------------
    # Pruebas de upsert de aristas
    # -------------------------------------------------------------------------

    class TestEdgeUpsert:
        """Pruebas para el método _upsert_edge."""

        @pytest.fixture
        def builder(self):
            return BudgetGraphBuilder()

        @pytest.fixture
        def empty_graph(self):
            return nx.DiGraph()

        def test_upsert_new_edge(self, builder, empty_graph):
            """Verifica inserción de nueva arista."""
            G = empty_graph
            G.add_node("APU-1", type="APU")
            G.add_node("Insumo-1", type="INSUMO")

            is_new = builder._upsert_edge(
                G, "APU-1", "Insumo-1", unit_cost=100.0, quantity=5.0, idx=0
            )

            assert is_new is True
            assert G.has_edge("APU-1", "Insumo-1")

            edge = G["APU-1"]["Insumo-1"]
            assert edge["unit_cost"] == 100.0
            assert edge["quantity"] == 5.0
            assert edge["total_cost"] == 500.0
            assert edge["occurrence_count"] == 1
            assert edge["original_indices"] == [0]

        def test_upsert_existing_edge_accumulates(self, builder, empty_graph):
            """Verifica acumulación al actualizar arista existente."""
            G = empty_graph
            G.add_node("APU-1", type="APU")
            G.add_node("Insumo-1", type="INSUMO")

            # Primera inserción
            builder._upsert_edge(G, "APU-1", "Insumo-1", 100.0, 5.0, idx=0)

            # Segunda inserción (misma arista)
            is_new = builder._upsert_edge(G, "APU-1", "Insumo-1", 100.0, 3.0, idx=1)

            assert is_new is False

            edge = G["APU-1"]["Insumo-1"]
            assert edge["quantity"] == 8.0  # 5 + 3
            assert edge["total_cost"] == 800.0  # 500 + 300
            assert edge["occurrence_count"] == 2
            assert edge["original_indices"] == [0, 1]

        def test_upsert_multiple_accumulations(self, builder, empty_graph):
            """Verifica múltiples acumulaciones consecutivas."""
            G = empty_graph
            G.add_node("APU-1", type="APU")
            G.add_node("Insumo-1", type="INSUMO")

            quantities = [10.0, 5.0, 3.0, 2.0]
            for i, qty in enumerate(quantities):
                builder._upsert_edge(G, "APU-1", "Insumo-1", 50.0, qty, idx=i)

            edge = G["APU-1"]["Insumo-1"]
            assert edge["quantity"] == 20.0  # sum(quantities)
            assert edge["total_cost"] == 1000.0  # 20 * 50
            assert edge["occurrence_count"] == 4
            assert edge["original_indices"] == [0, 1, 2, 3]

    # -------------------------------------------------------------------------
    # Pruebas del método build
    # -------------------------------------------------------------------------

    class TestBuildMethod:
        """Pruebas para el método principal build."""

        @pytest.fixture
        def builder(self):
            return BudgetGraphBuilder()

        def test_build_with_valid_data(self, builder):
            """Construcción con datos válidos completos."""
            df_presupuesto = pd.DataFrame(
                {
                    ColumnNames.CODIGO_APU: ["APU-001", "APU-002"],
                    ColumnNames.DESCRIPCION_APU: ["Muro", "Columna"],
                    ColumnNames.CANTIDAD_PRESUPUESTO: [10.0, 5.0],
                }
            )

            df_detail = pd.DataFrame(
                {
                    ColumnNames.CODIGO_APU: ["APU-001", "APU-001", "APU-002"],
                    ColumnNames.DESCRIPCION_INSUMO: ["Ladrillo", "Cemento", "Acero"],
                    ColumnNames.TIPO_INSUMO: ["Material", "Material", "Material"],
                    ColumnNames.CANTIDAD_APU: [100.0, 20.0, 50.0],
                    ColumnNames.COSTO_INSUMO_EN_APU: [1.5, 25.0, 30.0],
                }
            )

            G = builder.build(df_presupuesto, df_detail)

            # Verificar nodos (2 APUs + 3 Insumos + 1 Root = 6)
            assert G.number_of_nodes() == 6
            assert G.nodes["APU-001"]["type"] == "APU"
            assert G.nodes["APU-001"]["inferred"] is False
            assert G.nodes["Ladrillo"]["type"] == "INSUMO"
            assert "PROYECTO_TOTAL" in G

            # Verificar aristas (3 de detalles + 2 de raíz a APUs = 5)
            assert G.number_of_edges() == 5
            assert G.has_edge("APU-001", "Ladrillo")
            assert G.has_edge("APU-001", "Cemento")
            assert G.has_edge("APU-002", "Acero")
            assert G.has_edge("PROYECTO_TOTAL", "APU-001")
            assert G.has_edge("PROYECTO_TOTAL", "APU-002")

        def test_build_with_inferred_apu(self, builder):
            """APU presente en detail pero no en presupuesto se infiere."""
            df_presupuesto = pd.DataFrame(
                {
                    ColumnNames.CODIGO_APU: ["APU-001"],
                    ColumnNames.DESCRIPCION_APU: ["Muro"],
                    ColumnNames.CANTIDAD_PRESUPUESTO: [10.0],
                }
            )

            df_detail = pd.DataFrame(
                {
                    ColumnNames.CODIGO_APU: [
                        "APU-001",
                        "APU-999",
                    ],  # APU-999 no está en presupuesto
                    ColumnNames.DESCRIPCION_INSUMO: ["Ladrillo", "Arena"],
                    ColumnNames.TIPO_INSUMO: ["Material", "Material"],
                    ColumnNames.CANTIDAD_APU: [100.0, 50.0],
                    ColumnNames.COSTO_INSUMO_EN_APU: [1.5, 10.0],
                }
            )

            G = builder.build(df_presupuesto, df_detail)

            assert "APU-999" in G
            assert G.nodes["APU-999"]["inferred"] is True
            assert G.nodes["APU-001"]["inferred"] is False

        def test_build_with_empty_presupuesto(self, builder):
            """Construcción con presupuesto vacío pero detail con datos."""
            df_presupuesto = pd.DataFrame()

            df_detail = pd.DataFrame(
                {
                    ColumnNames.CODIGO_APU: ["APU-001"],
                    ColumnNames.DESCRIPCION_INSUMO: ["Ladrillo"],
                    ColumnNames.TIPO_INSUMO: ["Material"],
                    ColumnNames.CANTIDAD_APU: [100.0],
                    ColumnNames.COSTO_INSUMO_EN_APU: [1.5],
                }
            )

            G = builder.build(df_presupuesto, df_detail)

            assert G.number_of_nodes() == 3  # 1 APU inferido + 1 Insumo + 1 Root
            assert G.nodes["APU-001"]["inferred"] is True
            assert "PROYECTO_TOTAL" in G

        def test_build_with_empty_detail(self, builder):
            """Construcción con detail vacío produce APUs sin aristas."""
            df_presupuesto = pd.DataFrame(
                {
                    ColumnNames.CODIGO_APU: ["APU-001", "APU-002"],
                    ColumnNames.DESCRIPCION_APU: ["Muro", "Columna"],
                    ColumnNames.CANTIDAD_PRESUPUESTO: [10.0, 5.0],
                }
            )

            df_detail = pd.DataFrame()

            G = builder.build(df_presupuesto, df_detail)

            assert G.number_of_nodes() == 3  # 2 APUs + 1 Root
            assert G.number_of_edges() == 2  # Root -> APU1, Root -> APU2

        def test_build_with_both_empty(self, builder):
            """Construcción con ambos DataFrames vacíos produce grafo vacío (solo root)."""
            G = builder.build(pd.DataFrame(), pd.DataFrame())

            assert G.number_of_nodes() == 1  # Solo Root
            assert "PROYECTO_TOTAL" in G
            assert G.number_of_edges() == 0

        def test_build_with_none_inputs(self, builder):
            """Construcción con inputs None no lanza excepción (solo root)."""
            G = builder.build(None, None)

            assert G.number_of_nodes() == 1
            assert "PROYECTO_TOTAL" in G
            assert G.number_of_edges() == 0

        def test_build_edge_accumulation(self, builder):
            """Verifica acumulación correcta en aristas duplicadas."""
            df_presupuesto = pd.DataFrame(
                {
                    ColumnNames.CODIGO_APU: ["APU-001"],
                    ColumnNames.DESCRIPCION_APU: ["Muro"],
                    ColumnNames.CANTIDAD_PRESUPUESTO: [10.0],
                }
            )

            # Mismo insumo aparece 3 veces para el mismo APU
            df_detail = pd.DataFrame(
                {
                    ColumnNames.CODIGO_APU: ["APU-001", "APU-001", "APU-001"],
                    ColumnNames.DESCRIPCION_INSUMO: ["Ladrillo", "Ladrillo", "Ladrillo"],
                    ColumnNames.TIPO_INSUMO: ["Material", "Material", "Material"],
                    ColumnNames.CANTIDAD_APU: [100.0, 50.0, 25.0],
                    ColumnNames.COSTO_INSUMO_EN_APU: [1.5, 1.5, 1.5],
                }
            )

            G = builder.build(df_presupuesto, df_detail)

            edge = G["APU-001"]["Ladrillo"]
            assert edge["quantity"] == 175.0  # 100 + 50 + 25
            assert edge["total_cost"] == 262.5  # 175 * 1.5
            assert edge["occurrence_count"] == 3

        def test_build_handles_nan_values(self, builder):
            """Construcción maneja valores NaN correctamente."""
            df_presupuesto = pd.DataFrame(
                {
                    ColumnNames.CODIGO_APU: ["APU-001", None, "APU-002"],
                    ColumnNames.DESCRIPCION_APU: ["Muro", "Invalid", "Columna"],
                    ColumnNames.CANTIDAD_PRESUPUESTO: [10.0, float("nan"), 5.0],
                }
            )

            df_detail = pd.DataFrame(
                {
                    ColumnNames.CODIGO_APU: ["APU-001"],
                    ColumnNames.DESCRIPCION_INSUMO: ["Ladrillo"],
                    ColumnNames.TIPO_INSUMO: ["Material"],
                    ColumnNames.CANTIDAD_APU: [float("nan")],  # NaN quantity
                    ColumnNames.COSTO_INSUMO_EN_APU: [1.5],
                }
            )

            G = builder.build(df_presupuesto, df_detail)

            # El APU con código None no se agrega
            assert None not in G
            assert "" not in G
            # Los demás sí
            assert "APU-001" in G
            assert "APU-002" in G

        def test_build_graph_has_name(self, builder):
            """Verifica que el grafo tiene nombre asignado."""
            G = builder.build(pd.DataFrame(), pd.DataFrame())
            assert G.name == "BudgetTopology"

    # -------------------------------------------------------------------------
    # Pruebas de estadísticas del grafo
    # -------------------------------------------------------------------------

    class TestGraphStatistics:
        """Pruebas para _compute_graph_statistics."""

        @pytest.fixture
        def builder(self):
            return BudgetGraphBuilder()

        def test_compute_statistics(self, builder):
            """Verifica cómputo correcto de estadísticas."""
            G = nx.DiGraph()
            G.add_node("APU-1", type="APU", inferred=False)
            G.add_node("APU-2", type="APU", inferred=True)
            G.add_node("Insumo-1", type="INSUMO")
            G.add_node("Insumo-2", type="INSUMO")

            stats = builder._compute_graph_statistics(G)

            assert stats["apu_count"] == 2
            assert stats["insumo_count"] == 2
            assert stats["inferred_count"] == 1


# =============================================================================
# PRUEBAS PARA BusinessTopologicalAnalyzer
# =============================================================================


class TestBusinessTopologicalAnalyzer:
    """Pruebas para la clase BusinessTopologicalAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Fixture para crear analizador sin telemetría."""
        return BusinessTopologicalAnalyzer()

    @pytest.fixture
    def analyzer_with_telemetry(self):
        """Fixture para crear analizador con telemetría mock."""
        mock_telemetry = Mock()
        mock_telemetry.record_metric = MagicMock()
        return BusinessTopologicalAnalyzer(telemetry=mock_telemetry)

    # -------------------------------------------------------------------------
    # Pruebas de cálculo de números de Betti
    # -------------------------------------------------------------------------

    class TestBettiNumbers:
        """Pruebas para calculate_betti_numbers."""

        @pytest.fixture
        def analyzer(self):
            return BusinessTopologicalAnalyzer()

        def test_empty_graph(self, analyzer):
            """Grafo vacío tiene todos los invariantes en 0."""
            G = nx.DiGraph()
            metrics = analyzer.calculate_betti_numbers(G)

            assert metrics.beta_0 == 0
            assert metrics.beta_1 == 0
            assert metrics.euler_characteristic == 0

        def test_single_node(self, analyzer):
            """Un nodo aislado: β₀=1, β₁=0, χ=1."""
            G = nx.DiGraph()
            G.add_node("A")

            metrics = analyzer.calculate_betti_numbers(G)

            assert metrics.beta_0 == 1
            assert metrics.beta_1 == 0
            assert metrics.euler_characteristic == 1

        def test_simple_path_dag(self, analyzer):
            """
            Camino simple A → B → C (DAG).
            V=3, E=2, β₀=1, χ=1, β₁=0
            """
            G = nx.DiGraph()
            G.add_edges_from([("A", "B"), ("B", "C")])

            metrics = analyzer.calculate_betti_numbers(G)

            assert metrics.beta_0 == 1
            assert metrics.beta_1 == 0
            assert metrics.euler_characteristic == 1
            assert metrics.is_connected is True
            assert metrics.is_simply_connected is True

        def test_directed_cycle(self, analyzer):
            """
            Ciclo dirigido A → B → A.

            Si se considera la topología subyacente simple (nx.Graph), esto es una arista A--B.
            V=2, E=1, β₀=1, χ=1, β₁=0.

            Sin embargo, la implementación actual usa nx.MultiGraph para preservar la multiplicidad
            de aristas, por lo que A->B y B->A cuentan como 2 aristas.
            V=2, E=2, β₀=1.
            β₁ = β₀ - V + E = 1 - 2 + 2 = 1.

            Esto es consistente con la detección de ciclos de negocio donde A <-> B es un ciclo.
            """
            G = nx.DiGraph()
            G.add_edges_from([("A", "B"), ("B", "A")])

            metrics = analyzer.calculate_betti_numbers(G)

            assert metrics.beta_0 == 1
            # Con MultiGraph:
            assert metrics.beta_1 == 1
            assert metrics.euler_characteristic == 0  # 2 - 2 = 0

        def test_triangle_cycle(self, analyzer):
            """
            Triángulo: A → B → C → A.
            Grafo no dirigido: 3 nodos, 3 aristas.
            V=3, E=3, β₀=1, χ=0, β₁=1
            """
            G = nx.DiGraph()
            G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

            metrics = analyzer.calculate_betti_numbers(G)

            assert metrics.beta_0 == 1
            assert metrics.beta_1 == 1  # Un ciclo
            assert metrics.euler_characteristic == 0  # 3 - 3 = 0
            assert metrics.is_simply_connected is False

        def test_two_disconnected_components(self, analyzer):
            """
            Dos componentes: A → B y C → D.
            V=4, E=2, β₀=2, χ=2, β₁=0
            """
            G = nx.DiGraph()
            G.add_edges_from([("A", "B"), ("C", "D")])

            metrics = analyzer.calculate_betti_numbers(G)

            assert metrics.beta_0 == 2
            assert metrics.beta_1 == 0
            assert metrics.euler_characteristic == 2  # 4 - 2 = 2
            assert metrics.is_connected is False

        def test_complex_graph_with_cycle_and_components(self, analyzer):
            """
            Grafo complejo:
            - Componente 1: Triángulo (1, 2, 3) - 1 ciclo
            - Componente 2: Línea (4, 5)

            V=5, E=4 (no dirigido), β₀=2, χ=1, β₁=1
            """
            G = nx.DiGraph()
            # Triángulo
            G.add_edges_from([(1, 2), (2, 3), (3, 1)])
            # Línea
            G.add_edge(4, 5)

            metrics = analyzer.calculate_betti_numbers(G)

            assert metrics.beta_0 == 2
            assert metrics.beta_1 == 1
            assert metrics.euler_characteristic == 1  # 5 - 4 = 1

        def test_euler_formula_consistency(self, analyzer):
            """
            Verifica que χ = β₀ - β₁ = V - E para varios grafos.
            """
            test_cases = [
                # (aristas, beta_0_esperado, beta_1_esperado)
                ([], 0, 0),  # Grafo vacío
                ([("A", "B")], 1, 0),  # Una arista
                ([("A", "B"), ("B", "C"), ("C", "A")], 1, 1),  # Triángulo
                ([("A", "B"), ("C", "D"), ("E", "F")], 3, 0),  # 3 componentes
            ]

            for edges, expected_b0, expected_b1 in test_cases:
                G = nx.DiGraph()
                G.add_edges_from(edges)

                metrics = analyzer.calculate_betti_numbers(G)

                # Verificar fórmula de Euler: χ = β₀ - β₁
                assert metrics.euler_characteristic == metrics.beta_0 - metrics.beta_1

        def test_bipartite_budget_graph(self, analyzer):
            """
            Grafo bipartito típico de presupuesto:
            APU-1 → {Insumo-A, Insumo-B}
            APU-2 → {Insumo-B, Insumo-C}

            V=5, E=4, β₀=1, χ=1, β₁=0 (es un árbol expandido)
            """
            G = nx.DiGraph()
            G.add_edges_from(
                [
                    ("APU-1", "Insumo-A"),
                    ("APU-1", "Insumo-B"),
                    ("APU-2", "Insumo-B"),
                    ("APU-2", "Insumo-C"),
                ]
            )

            metrics = analyzer.calculate_betti_numbers(G)

            assert metrics.beta_0 == 1
            assert metrics.beta_1 == 0
            assert metrics.euler_characteristic == 1  # 5 - 4 = 1
            assert metrics.is_simply_connected is True

    # -------------------------------------------------------------------------
    # Pruebas de detección de ciclos
    # -------------------------------------------------------------------------

    class TestCycleDetection:
        """Pruebas para _detect_cycles."""

        @pytest.fixture
        def analyzer(self):
            return BusinessTopologicalAnalyzer(max_cycles=10)

        def test_no_cycles_in_dag(self, analyzer):
            """DAG no tiene ciclos."""
            G = nx.DiGraph()
            G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

            cycles, truncated = analyzer._detect_cycles(G)

            assert cycles == []
            assert truncated is False

        def test_simple_cycle_detected(self, analyzer):
            """Detecta ciclo simple."""
            G = nx.DiGraph()
            G.add_edges_from([("A", "B"), ("B", "A")])

            cycles, truncated = analyzer._detect_cycles(G)

            assert len(cycles) >= 1
            assert truncated is False

        def test_cycle_representation_closed(self, analyzer):
            """Verifica representación cerrada del ciclo (A → B → A)."""
            G = nx.DiGraph()
            G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

            cycles, _ = analyzer._detect_cycles(G)

            # El ciclo debe terminar donde empezó
            assert len(cycles) == 1
            cycle_parts = cycles[0].split(" → ")
            assert cycle_parts[0] == cycle_parts[-1]

        def test_truncation_on_many_cycles(self):
            """Verifica truncamiento cuando hay muchos ciclos."""
            analyzer = BusinessTopologicalAnalyzer(max_cycles=2)

            # Grafo con múltiples ciclos
            G = nx.DiGraph()
            # Múltiples ciclos pequeños
            G.add_edges_from(
                [
                    (1, 2),
                    (2, 1),  # Ciclo 1
                    (3, 4),
                    (4, 3),  # Ciclo 2
                    (5, 6),
                    (6, 5),  # Ciclo 3
                ]
            )

            cycles, truncated = analyzer._detect_cycles(G)

            assert len(cycles) <= 2
            assert truncated is True

    # -------------------------------------------------------------------------
    # Pruebas de clasificación de nodos anómalos
    # -------------------------------------------------------------------------

    class TestAnomalousNodeClassification:
        """Pruebas para _classify_anomalous_nodes."""

        @pytest.fixture
        def analyzer(self):
            return BusinessTopologicalAnalyzer()

        def test_isolated_nodes_detected(self, analyzer):
            """Detecta nodos completamente aislados."""
            G = nx.DiGraph()
            G.add_node("ISOLATED-APU", type="APU")
            G.add_node("ISOLATED-INSUMO", type="INSUMO")
            G.add_edge("CONNECTED-APU", "CONNECTED-INSUMO")
            G.nodes["CONNECTED-APU"]["type"] = "APU"
            G.nodes["CONNECTED-INSUMO"]["type"] = "INSUMO"

            classification = analyzer._classify_anomalous_nodes(G)

            isolated_ids = [n["id"] for n in classification["isolated_nodes"]]
            assert "ISOLATED-APU" in isolated_ids
            assert "ISOLATED-INSUMO" in isolated_ids
            assert "CONNECTED-APU" not in isolated_ids

        def test_empty_apus_detected(self, analyzer):
            """Detecta APUs sin insumos (out_degree = 0)."""
            G = nx.DiGraph()
            G.add_node("EMPTY-APU", type="APU")
            G.add_edge("OTHER-APU", "INSUMO-1")
            G.nodes["OTHER-APU"]["type"] = "APU"
            G.nodes["INSUMO-1"]["type"] = "INSUMO"
            # Conectar EMPTY-APU como destino para que no sea aislado
            G.add_edge("SOURCE", "EMPTY-APU")

            classification = analyzer._classify_anomalous_nodes(G)

            empty_apu_ids = [n["id"] for n in classification["empty_apus"]]
            assert "EMPTY-APU" in empty_apu_ids

        def test_orphan_insumos_detected(self, analyzer):
            """Detecta insumos no utilizados (in_degree = 0)."""
            G = nx.DiGraph()
            G.add_node("ORPHAN-INSUMO", type="INSUMO")
            G.add_edge("APU", "CONNECTED-INSUMO")
            G.nodes["APU"]["type"] = "APU"
            G.nodes["CONNECTED-INSUMO"]["type"] = "INSUMO"
            # Conectar ORPHAN-INSUMO como origen para que no sea aislado
            G.add_edge("ORPHAN-INSUMO", "SINK")

            classification = analyzer._classify_anomalous_nodes(G)

            orphan_ids = [n["id"] for n in classification["orphan_insumos"]]
            assert "ORPHAN-INSUMO" in orphan_ids

        def test_node_info_completeness(self, analyzer):
            """Verifica que la información del nodo está completa."""
            G = nx.DiGraph()
            G.add_node("APU-1", type="APU", description="Test APU", inferred=True)

            classification = analyzer._classify_anomalous_nodes(G)

            node_info = classification["isolated_nodes"][0]
            assert "id" in node_info
            assert "description" in node_info
            assert "inferred" in node_info
            assert "in_degree" in node_info
            assert "out_degree" in node_info

    # -------------------------------------------------------------------------
    # Pruebas de identificación de recursos críticos
    # -------------------------------------------------------------------------

    class TestCriticalResourcesIdentification:
        """Pruebas para _identify_critical_resources."""

        @pytest.fixture
        def analyzer(self):
            return BusinessTopologicalAnalyzer()

        def test_identifies_most_used_resources(self, analyzer):
            """Identifica recursos con mayor in-degree."""
            G = nx.DiGraph()
            # Insumo crítico usado por 3 APUs
            G.add_edges_from(
                [
                    ("APU-1", "CRITICAL-INSUMO"),
                    ("APU-2", "CRITICAL-INSUMO"),
                    ("APU-3", "CRITICAL-INSUMO"),
                    ("APU-1", "NORMAL-INSUMO"),
                ]
            )
            for node in G.nodes():
                if node.startswith("APU"):
                    G.nodes[node]["type"] = "APU"
                else:
                    G.nodes[node]["type"] = "INSUMO"

            critical = analyzer._identify_critical_resources(G, top_n=1)

            assert len(critical) == 1
            assert critical[0]["id"] == "CRITICAL-INSUMO"
            assert critical[0]["in_degree"] == 3

        def test_respects_top_n_limit(self, analyzer):
            """Respeta el límite top_n."""
            G = nx.DiGraph()
            for i in range(10):
                G.add_edge(f"APU-{i}", f"INSUMO-{i % 3}")
                G.nodes[f"APU-{i}"]["type"] = "APU"
            for i in range(3):
                G.nodes[f"INSUMO-{i}"]["type"] = "INSUMO"

            critical = analyzer._identify_critical_resources(G, top_n=2)

            assert len(critical) <= 2

        def test_excludes_zero_degree_resources(self, analyzer):
            """Excluye recursos con in-degree 0."""
            G = nx.DiGraph()
            G.add_node("UNUSED-INSUMO", type="INSUMO")
            G.add_edge("APU", "USED-INSUMO")
            G.nodes["APU"]["type"] = "APU"
            G.nodes["USED-INSUMO"]["type"] = "INSUMO"

            critical = analyzer._identify_critical_resources(G)

            critical_ids = [r["id"] for r in critical]
            assert "UNUSED-INSUMO" not in critical_ids

    # -------------------------------------------------------------------------
    # Pruebas de análisis de conectividad
    # -------------------------------------------------------------------------

    class TestConnectivityAnalysis:
        """Pruebas para _compute_connectivity_analysis."""

        @pytest.fixture
        def analyzer(self):
            return BusinessTopologicalAnalyzer()

        def test_dag_detection(self, analyzer):
            """Detecta correctamente si es DAG."""
            dag = nx.DiGraph()
            dag.add_edges_from([("A", "B"), ("B", "C")])

            non_dag = nx.DiGraph()
            non_dag.add_edges_from([("A", "B"), ("B", "A")])

            dag_result = analyzer._compute_connectivity_analysis(dag)
            non_dag_result = analyzer._compute_connectivity_analysis(non_dag)

            assert dag_result["is_dag"] is True
            assert non_dag_result["is_dag"] is False

        def test_weakly_connected_components(self, analyzer):
            """Cuenta componentes débilmente conexas."""
            G = nx.DiGraph()
            G.add_edge("A", "B")
            G.add_edge("C", "D")

            result = analyzer._compute_connectivity_analysis(G)

            assert result["num_wcc"] == 2
            assert result["is_weakly_connected"] is False

        def test_strongly_connected_components(self, analyzer):
            """Detecta SCCs no triviales."""
            G = nx.DiGraph()
            # SCC: A <-> B
            G.add_edges_from([("A", "B"), ("B", "A")])
            # Nodo aislado
            G.add_node("C")

            result = analyzer._compute_connectivity_analysis(G)

            assert result["num_non_trivial_scc"] == 1
            assert len(result["non_trivial_scc"][0]) == 2

    # -------------------------------------------------------------------------
    # Pruebas de interpretación topológica
    # -------------------------------------------------------------------------

    class TestTopologicalInterpretation:
        """Pruebas para _interpret_topology."""

        @pytest.fixture
        def analyzer(self):
            return BusinessTopologicalAnalyzer()

        def test_connected_interpretation(self, analyzer):
            """Interpreta correctamente espacio conexo."""
            metrics = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)

            interp = analyzer._interpret_topology(metrics)

            assert (
                "conexo" in interp["beta_0"].lower()
                or "conectado" in interp["beta_0"].lower()
            )

        def test_disconnected_interpretation(self, analyzer):
            """Interpreta correctamente espacio desconexo."""
            metrics = TopologicalMetrics(beta_0=3, beta_1=0, euler_characteristic=3)

            interp = analyzer._interpret_topology(metrics)

            assert "3" in interp["beta_0"]
            assert "componente" in interp["beta_0"].lower()

        def test_cycles_interpretation(self, analyzer):
            """Interpreta correctamente presencia de ciclos."""
            metrics = TopologicalMetrics(beta_0=1, beta_1=2, euler_characteristic=-1)

            interp = analyzer._interpret_topology(metrics)

            assert "2" in interp["beta_1"]
            assert "ciclo" in interp["beta_1"].lower()

        def test_no_cycles_interpretation(self, analyzer):
            """Interpreta correctamente ausencia de ciclos."""
            metrics = TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)

            interp = analyzer._interpret_topology(metrics)

            assert (
                "acíclic" in interp["beta_1"].lower()
                or "sin ciclo" in interp["beta_1"].lower()
            )

    # -------------------------------------------------------------------------
    # Pruebas del análisis de integridad estructural
    # -------------------------------------------------------------------------

    class TestStructuralIntegrityAnalysis:
        """Pruebas para analyze_structural_integrity."""

        @pytest.fixture
        def analyzer(self):
            return BusinessTopologicalAnalyzer()

        def test_returns_flat_metrics(self, analyzer):
            """Verifica que retorna métricas planas."""
            G = nx.DiGraph()
            G.add_edge("A", "B")

            result = analyzer.analyze_structural_integrity(G)

            assert "business.betti_b0" in result
            assert "business.betti_b1" in result
            assert "business.euler_characteristic" in result
            assert "business.cycles_count" in result
            assert "business.is_dag" in result

        def test_returns_details_dict(self, analyzer):
            """Verifica estructura de detalles."""
            G = nx.DiGraph()
            G.add_edge("A", "B")

            result = analyzer.analyze_structural_integrity(G)

            assert "details" in result
            details = result["details"]
            assert "topology" in details
            assert "cycles" in details
            assert "connectivity" in details
            assert "anomalies" in details
            assert "critical_resources" in details
            assert "graph_summary" in details

        def test_betti_numbers_in_details(self, analyzer):
            """Verifica números de Betti en detalles."""
            G = nx.DiGraph()
            G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

            result = analyzer.analyze_structural_integrity(G)

            betti = result["details"]["topology"]["betti_numbers"]
            assert betti["beta_0"] == 1
            assert betti["beta_1"] == 1
            assert betti["euler_characteristic"] == 0

        def test_complete_budget_graph_analysis(self, analyzer):
            """Análisis completo de grafo tipo presupuesto."""
            G = nx.DiGraph()
            # Estructura típica de presupuesto
            G.add_node("APU-1", type="APU")
            G.add_node("APU-2", type="APU")
            G.add_node("Cemento", type="INSUMO", description="Cemento Portland")
            G.add_node("Arena", type="INSUMO", description="Arena fina")
            G.add_edge("APU-1", "Cemento")
            G.add_edge("APU-1", "Arena")
            G.add_edge("APU-2", "Cemento")
            # Nodo aislado (anomalía)
            G.add_node("ORPHAN", type="INSUMO")

            result = analyzer.analyze_structural_integrity(G)

            # Verificar métricas topológicas
            assert result["business.betti_b0"] == 2  # 2 componentes (principal + orphan)
            assert result["business.is_dag"] == 1  # Es DAG

            # Verificar detección de anomalías
            assert result["business.isolated_count"] == 1

        def test_telemetry_emission(self):
            """Verifica emisión de telemetría."""
            mock_telemetry = Mock()
            mock_telemetry.record_metric = MagicMock()
            analyzer = BusinessTopologicalAnalyzer(telemetry=mock_telemetry)

            G = nx.DiGraph()
            G.add_edge("A", "B")

            analyzer.analyze_structural_integrity(G)

            # Verificar que se llamó record_metric
            assert mock_telemetry.record_metric.called

    # -------------------------------------------------------------------------
    # Pruebas del reporte de auditoría
    # -------------------------------------------------------------------------

    class TestAuditReport:
        """Pruebas para get_audit_report."""

        @pytest.fixture
        def analyzer(self):
            return BusinessTopologicalAnalyzer()

        def test_report_is_list_of_strings(self, analyzer):
            """Verifica que el reporte es lista de strings."""
            G = nx.DiGraph()
            G.add_edge("A", "B")
            analysis = analyzer.analyze_structural_integrity(G)

            report = analyzer.get_audit_report(analysis)

            assert isinstance(report, list)
            assert all(isinstance(line, str) for line in report)

        def test_report_contains_header(self, analyzer):
            """Verifica presencia de encabezado."""
            G = nx.DiGraph()
            G.add_edge("A", "B")
            analysis = analyzer.analyze_structural_integrity(G)

            report = analyzer.get_audit_report(analysis)
            report_text = "\n".join(report)

            assert "AUDITORÍA ESTRUCTURAL" in report_text

        def test_report_shows_betti_numbers(self, analyzer):
            """Verifica que muestra números de Betti."""
            G = nx.DiGraph()
            G.add_edges_from([("A", "B"), ("B", "C")])
            analysis = analyzer.analyze_structural_integrity(G)

            report = analyzer.get_audit_report(analysis)
            report_text = "\n".join(report)

            assert (
                "componentes conexas" in report_text.lower()
                or "beta_0" in report_text.lower()
            )
            assert (
                "ciclos de costo" in report_text.lower() or "beta_1" in report_text.lower()
            )

        def test_report_shows_critical_alert_for_cycles(self, analyzer):
            """Verifica alerta crítica cuando hay ciclos."""
            G = nx.DiGraph()
            G.add_edges_from([("A", "B"), ("B", "A")])  # Ciclo
            analysis = analyzer.analyze_structural_integrity(G)

            report = analyzer.get_audit_report(analysis)
            report_text = "\n".join(report)

            assert (
                "CRÍTICAS" in report_text
                or "circular" in report_text.lower()
                or "ciclos de costo" in report_text.lower()
            )

        def test_report_shows_ok_for_healthy_graph(self, analyzer):
            """Verifica resultado OK para grafo saludable."""
            G = nx.DiGraph()
            G.add_node("APU-1", type="APU")
            G.add_node("INSUMO-1", type="INSUMO")
            G.add_edge("APU-1", "INSUMO-1")
            analysis = analyzer.analyze_structural_integrity(G)

            report = analyzer.get_audit_report(analysis)
            report_text = "\n".join(report)

            assert "estatus" in report_text.lower() or "saludable" in report_text.lower()

        def test_report_shows_warnings_for_anomalies(self, analyzer):
            """Verifica advertencias para anomalías."""
            G = nx.DiGraph()
            G.add_node("APU-EMPTY", type="APU")  # APU sin insumos
            G.add_node("INSUMO-ORPHAN", type="INSUMO")  # Insumo huérfano
            G.add_edge("APU-NORMAL", "INSUMO-NORMAL")
            G.nodes["APU-NORMAL"]["type"] = "APU"
            G.nodes["INSUMO-NORMAL"]["type"] = "INSUMO"

            analysis = analyzer.analyze_structural_integrity(G)
            report = analyzer.get_audit_report(analysis)
            report_text = "\n".join(report)

            assert "ADVERTENCIA" in report_text or "⚠" in report_text


# =============================================================================
# PRUEBAS DE INTEGRACIÓN
# =============================================================================


class TestIntegration:
    """Pruebas de integración entre Builder y Analyzer."""

    @pytest.fixture
    def builder(self):
        return BudgetGraphBuilder()

    @pytest.fixture
    def analyzer(self):
        return BusinessTopologicalAnalyzer()

    def test_full_pipeline(self, builder, analyzer):
        """Pipeline completo: construcción → análisis → reporte."""
        # Datos de prueba
        df_presupuesto = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001", "APU-002"],
                ColumnNames.DESCRIPCION_APU: ["Muro de ladrillo", "Columna de concreto"],
                ColumnNames.CANTIDAD_PRESUPUESTO: [100.0, 50.0],
            }
        )

        df_detail = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001", "APU-001", "APU-002", "APU-002"],
                ColumnNames.DESCRIPCION_INSUMO: ["Ladrillo", "Cemento", "Cemento", "Acero"],
                ColumnNames.TIPO_INSUMO: ["Material", "Material", "Material", "Material"],
                ColumnNames.CANTIDAD_APU: [1000.0, 50.0, 100.0, 200.0],
                ColumnNames.COSTO_INSUMO_EN_APU: [0.5, 25.0, 25.0, 15.0],
            }
        )

        # Construir grafo
        G = builder.build(df_presupuesto, df_detail)

        # Analizar
        analysis = analyzer.analyze_structural_integrity(G)

        # Generar reporte
        report = analyzer.get_audit_report(analysis)

        # Verificaciones
        assert G.number_of_nodes() == 6  # 2 APUs + 3 Insumos + 1 Root
        assert G.number_of_edges() == 6  # 4 detalles + 2 raíz
        assert analysis["business.betti_b0"] == 1  # Conectado

        # Nota sobre Betti_1:
        # En la topología actual, la presencia de "Cemento" compartido por APU-001 y APU-002
        # más el nodo Root conectado a ambos APUs, crea un ciclo no dirigido (bucle):
        # Root -> APU-001 -> Cemento <- APU-002 <- Root
        # Como calculate_betti_numbers usa un grafo no dirigido subyacente (MultiGraph),
        # esto cuenta como 1 ciclo (beta_1 = 1).
        # Euler = 1 - 1 = 0
        assert analysis["business.betti_b1"] == 1
        assert analysis["business.is_dag"] == 1
        assert len(report) > 0

    def test_pipeline_with_anomalies(self, builder, analyzer):
        """Pipeline con datos que generan anomalías."""
        df_presupuesto = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001", "APU-EMPTY"],
                ColumnNames.DESCRIPCION_APU: ["Muro", "Sin insumos"],
                ColumnNames.CANTIDAD_PRESUPUESTO: [100.0, 50.0],
            }
        )

        df_detail = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001"],
                ColumnNames.DESCRIPCION_INSUMO: ["Ladrillo"],
                ColumnNames.TIPO_INSUMO: ["Material"],
                ColumnNames.CANTIDAD_APU: [1000.0],
                ColumnNames.COSTO_INSUMO_EN_APU: [0.5],
            }
        )

        G = builder.build(df_presupuesto, df_detail)
        analysis = analyzer.analyze_structural_integrity(G)

        # APU-EMPTY no tiene insumos (out_degree = 0)
        assert analysis["business.empty_apus_count"] >= 1

        # Con la raíz "PROYECTO_TOTAL" conectando a todos los APUs,
        # APU-EMPTY ya no está topológicamente aislado (tiene in-degree desde raíz),
        # por lo que todo debería ser 1 sola componente conexa.
        # Ajustamos el test para reflejar la nueva topología.
        assert analysis["business.betti_b0"] == 1

    def test_pipeline_preserves_edge_metadata(self, builder, analyzer):
        """Verifica que los metadatos de aristas se preservan."""
        df_presupuesto = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001"],
                ColumnNames.DESCRIPCION_APU: ["Muro"],
                ColumnNames.CANTIDAD_PRESUPUESTO: [10.0],
            }
        )

        df_detail = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001", "APU-001"],
                ColumnNames.DESCRIPCION_INSUMO: ["Cemento", "Cemento"],
                ColumnNames.TIPO_INSUMO: ["Material", "Material"],
                ColumnNames.CANTIDAD_APU: [50.0, 30.0],
                ColumnNames.COSTO_INSUMO_EN_APU: [25.0, 25.0],
            }
        )

        G = builder.build(df_presupuesto, df_detail)

        edge = G["APU-001"]["Cemento"]
        assert edge["quantity"] == 80.0
        assert edge["total_cost"] == 2000.0
        assert edge["occurrence_count"] == 2
        assert len(edge["original_indices"]) == 2


# =============================================================================
# PRUEBAS DE CASOS EDGE
# =============================================================================


class TestEdgeCases:
    """Pruebas de casos límite y edge cases."""

    @pytest.fixture
    def builder(self):
        return BudgetGraphBuilder()

    @pytest.fixture
    def analyzer(self):
        return BusinessTopologicalAnalyzer()

    def test_very_large_graph_performance(self, builder, analyzer):
        """Verifica rendimiento con grafo grande."""
        # Crear datos de prueba grandes
        n_apus = 100
        n_insumos_per_apu = 20

        apu_codes = [f"APU-{i:04d}" for i in range(n_apus)]

        df_presupuesto = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: apu_codes,
                ColumnNames.DESCRIPCION_APU: [f"Actividad {i}" for i in range(n_apus)],
                ColumnNames.CANTIDAD_PRESUPUESTO: [100.0] * n_apus,
            }
        )

        detail_rows = []
        for apu in apu_codes:
            for j in range(n_insumos_per_apu):
                detail_rows.append(
                    {
                        ColumnNames.CODIGO_APU: apu,
                        ColumnNames.DESCRIPCION_INSUMO: f"Insumo-{j:03d}",
                        ColumnNames.TIPO_INSUMO: "Material",
                        ColumnNames.CANTIDAD_APU: 10.0,
                        ColumnNames.COSTO_INSUMO_EN_APU: 5.0,
                    }
                )

        df_detail = pd.DataFrame(detail_rows)

        # Ejecutar y verificar que completa
        G = builder.build(df_presupuesto, df_detail)
        analysis = analyzer.analyze_structural_integrity(G)

        # +1 por Root Node
        assert G.number_of_nodes() == n_apus + n_insumos_per_apu + 1
        assert "business.betti_b0" in analysis

    def test_unicode_in_descriptions(self, builder):
        """Verifica manejo de caracteres Unicode."""
        df_presupuesto = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001"],
                ColumnNames.DESCRIPCION_APU: [
                    "Construcción de muro con ladrillo cerámico 中文 émoji 🧱"
                ],
                ColumnNames.CANTIDAD_PRESUPUESTO: [10.0],
            }
        )

        df_detail = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001"],
                ColumnNames.DESCRIPCION_INSUMO: ["Ladrillo cerámico 日本語 árabe عربي"],
                ColumnNames.TIPO_INSUMO: ["Material"],
                ColumnNames.CANTIDAD_APU: [100.0],
                ColumnNames.COSTO_INSUMO_EN_APU: [1.5],
            }
        )

        G = builder.build(df_presupuesto, df_detail)

        assert G.number_of_nodes() == 3  # APU + Insumo + Root

    def test_special_characters_in_codes(self, builder):
        """Verifica manejo de caracteres especiales en códigos."""
        df_presupuesto = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU/001", "APU-002.1", "APU#003"],
                ColumnNames.DESCRIPCION_APU: ["A", "B", "C"],
                ColumnNames.CANTIDAD_PRESUPUESTO: [1.0, 1.0, 1.0],
            }
        )

        G = builder.build(df_presupuesto, pd.DataFrame())

        assert "APU/001" in G
        assert "APU-002.1" in G
        assert "APU#003" in G

    def test_zero_quantities_and_costs(self, builder):
        """Verifica manejo de cantidades y costos en cero."""
        df_presupuesto = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001"],
                ColumnNames.DESCRIPCION_APU: ["Test"],
                ColumnNames.CANTIDAD_PRESUPUESTO: [0.0],
            }
        )

        df_detail = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001"],
                ColumnNames.DESCRIPCION_INSUMO: ["Insumo-Zero"],
                ColumnNames.TIPO_INSUMO: ["Material"],
                ColumnNames.CANTIDAD_APU: [0.0],
                ColumnNames.COSTO_INSUMO_EN_APU: [0.0],
            }
        )

        G = builder.build(df_presupuesto, df_detail)

        edge = G["APU-001"]["Insumo-Zero"]
        assert edge["quantity"] == 0.0
        assert edge["total_cost"] == 0.0

    def test_negative_values(self, builder):
        """Verifica manejo de valores negativos (edge case)."""
        df_presupuesto = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001"],
                ColumnNames.DESCRIPCION_APU: ["Test"],
                ColumnNames.CANTIDAD_PRESUPUESTO: [-10.0],  # Negativo
            }
        )

        df_detail = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001"],
                ColumnNames.DESCRIPCION_INSUMO: ["Insumo"],
                ColumnNames.TIPO_INSUMO: ["Material"],
                ColumnNames.CANTIDAD_APU: [-5.0],
                ColumnNames.COSTO_INSUMO_EN_APU: [10.0],
            }
        )

        G = builder.build(df_presupuesto, df_detail)

        # Debería procesar sin error
        assert G.number_of_nodes() == 3 # APU + Insumo + Root
        edge = G["APU-001"]["Insumo"]
        assert edge["total_cost"] == -50.0  # -5 * 10

    def test_duplicate_apu_codes_in_presupuesto(self, builder):
        """Verifica que códigos duplicados en presupuesto se manejan."""
        df_presupuesto = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU-001", "APU-001", "APU-002"],
                ColumnNames.DESCRIPCION_APU: ["Primera", "Duplicada", "Otra"],
                ColumnNames.CANTIDAD_PRESUPUESTO: [10.0, 20.0, 5.0],
            }
        )

        G = builder.build(df_presupuesto, pd.DataFrame())

        # Solo debe haber 2 APUs únicos
        apu_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "APU"]
        assert len(apu_nodes) == 2
