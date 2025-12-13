import pytest
import pandas as pd
import networkx as nx
from agent.business_topology import BudgetGraphBuilder, BusinessTopologicalAnalyzer, TopologicalMetrics
from app.constants import ColumnNames

class TestBusinessTopologyV2:

    @pytest.fixture
    def builder(self):
        return BudgetGraphBuilder()

    @pytest.fixture
    def analyzer(self):
        return BusinessTopologicalAnalyzer()

    def test_graph_construction_upsert(self, builder):
        """Verifica construcción de nodos y acumulación de cantidades (upsert)."""
        # APU data
        df_presupuesto = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["APU-1"],
            ColumnNames.DESCRIPCION_APU: ["Muro"],
            ColumnNames.CANTIDAD_PRESUPUESTO: [10.0]
        })

        # Detail data (Two entries for same edge to test accumulation)
        df_detail = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["APU-1", "APU-1"],
            ColumnNames.DESCRIPCION_INSUMO: ["Ladrillo", "Ladrillo"],
            ColumnNames.TIPO_INSUMO: ["Material", "Material"],
            ColumnNames.CANTIDAD_APU: [5.0, 3.0], # Should sum to 8.0
            ColumnNames.COSTO_INSUMO_EN_APU: [100.0, 100.0]
        })

        G = builder.build(df_presupuesto, df_detail)

        assert "APU-1" in G
        assert "Ladrillo" in G
        assert G.has_edge("APU-1", "Ladrillo")

        # Verify Upsert Logic
        edge_data = G["APU-1"]["Ladrillo"]
        assert edge_data["quantity"] == 8.0 # 5 + 3
        assert edge_data["total_cost"] == 800.0 # (5*100) + (3*100)

    def test_topological_metrics_dag(self, analyzer):
        """Grafo Acíclico simple (DAG)."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        # A -> B -> C
        # Nodes: 3, Edges: 2. Components: 1.
        # Beta 0 = 1
        # Beta 1 = |E| - |V| + Beta 0 = 2 - 3 + 1 = 0

        metrics = analyzer.calculate_metrics(G)

        assert metrics.beta_0 == 1
        assert metrics.beta_1 == 0
        assert metrics.is_dag is True
        assert metrics.chi == 1 # 1 - 0

    def test_topological_metrics_cycle(self, analyzer):
        """Grafo con Ciclo."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        G.add_edge("B", "A")
        # A <-> B
        # Nodes: 2, Edges: 2. Components: 1.
        # Beta 1 = 2 - 2 + 1 = 1

        metrics = analyzer.calculate_metrics(G)

        assert metrics.beta_0 == 1
        assert metrics.beta_1 == 1
        assert metrics.is_dag is False
        assert metrics.chi == 0 # 1 - 1

    def test_topological_metrics_disconnected(self, analyzer):
        """Grafo Desconectado (2 componentes)."""
        G = nx.DiGraph()
        G.add_edge("A", "B")
        G.add_edge("C", "D")
        # A->B   C->D
        # Nodes: 4, Edges: 2. Components: 2.
        # Beta 1 = 2 - 4 + 2 = 0

        metrics = analyzer.calculate_metrics(G)

        assert metrics.beta_0 == 2
        assert metrics.beta_1 == 0
        assert metrics.chi == 2

    def test_anomalies_and_report(self, builder, analyzer):
        """Detección de anomalías y generación de reporte."""
        # Empty inputs creates empty graph initially
        G = nx.DiGraph()
        # Isolated APU
        G.add_node("APU-ISO", type="APU")
        # Isolated Insumo
        G.add_node("INS-ISO", type="INSUMO")

        report = analyzer.get_audit_report(G)

        print(report) # For debug visibility

        assert "REPORTE DE TOPOLOGÍA DE NEGOCIO (V2)" in report
        assert "Insumos Huérfanos" in report
        assert "APUs Vacíos" in report
        assert "Grafo Acíclico (DAG)" in report # Should be DAG as no edges

    def test_euler_formula_consistency(self, analyzer):
        """Verificar consistencia matemática de la fórmula de Euler."""
        # Complex graph
        G = nx.DiGraph()
        # Component 1: Triangle (1 cycle)
        G.add_edges_from([(1,2), (2,3), (3,1)])
        # Component 2: Line (0 cycles)
        G.add_edges_from([(4,5)])

        # V=5, E=4.
        # Undirected edges: (1,2), (2,3), (3,1), (4,5) -> 4 edges.
        # Components = 2 ({1,2,3}, {4,5})

        # Beta 1 = |E| - |V| + Beta 0 = 4 - 5 + 2 = 1.

        metrics = analyzer.calculate_metrics(G)
        assert metrics.beta_0 == 2
        assert metrics.beta_1 == 1
        assert metrics.chi == 1 # 2 - 1
