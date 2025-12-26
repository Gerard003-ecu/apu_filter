import networkx as nx
import pytest

from agent.business_topology import BusinessTopologicalAnalyzer
from app.financial_engine import FinancialConfig, FinancialEngine


class TestAdvancedRiskAnalysis:
    @pytest.fixture
    def analyzer(self):
        return BusinessTopologicalAnalyzer()

    @pytest.fixture
    def financial_engine(self):
        config = FinancialConfig()
        return FinancialEngine(config)

    def test_euler_efficiency_tree(self, analyzer):
        """Test that a perfect tree has max efficiency."""
        # Create a tree: Root -> A, Root -> B
        G = nx.DiGraph()
        G.add_edges_from([("Root", "A"), ("Root", "B")])

        efficiency = analyzer.calculate_euler_efficiency(G)
        # Nodes=3, Edges=2. Expected edges for tree = 3-1 = 2. Excess = 0.
        # Efficiency is exp(-excess/nodes) -> exp(0) = 1.0
        assert efficiency == 1.0

    def test_euler_efficiency_complex(self, analyzer):
        """Test that a dense graph has lower efficiency."""
        # K4 Complete Graph
        G = nx.complete_graph(4, create_using=nx.DiGraph())

        efficiency = analyzer.calculate_euler_efficiency(G)
        # Nodes=4. Expected Edges for tree=3. Actual Edges=12. Excess = 9.
        # Efficiency = exp(-9/4) = exp(-2.25) ~ 0.105
        assert efficiency < 0.5
        assert efficiency > 0.0

    def test_risk_synergy_detection(self, analyzer):
        """Test Cup Product / Synergy detection (Cycles sharing critical nodes)."""
        # We need two cycles that share at least two nodes (an edge) to be robustly detected as synergy
        # by the intersection logic: len(intersection) >= 2.

        G = nx.DiGraph()
        # Cycle 1: A -> B -> C -> A
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        # Cycle 2: A -> B -> D -> A
        G.add_edges_from([("A", "B"), ("B", "D"), ("D", "A")])

        # Shared edge: A->B.
        # Intersection of nodes: {A, B}

        synergy = analyzer.detect_risk_synergy(G)

        assert synergy["synergy_detected"] is True
        # Both A and B are bridge nodes.
        assert "A" in synergy["shared_nodes"]
        assert "B" in synergy["shared_nodes"]
        assert synergy["intersecting_cycles_count"] >= 1

    def test_no_synergy_disjoint_cycles(self, analyzer):
        """Test disjoint cycles do not trigger synergy."""
        G = nx.DiGraph()
        # Cycle 1: A -> B -> A
        G.add_edges_from([("A", "B"), ("B", "A")])
        # Cycle 2: C -> D -> C
        G.add_edges_from([("C", "D"), ("D", "C")])

        synergy = analyzer.detect_risk_synergy(G)

        assert synergy["synergy_detected"] is False

    def test_financial_volatility_penalty(self, financial_engine):
        """Test that risk synergy increases financial volatility."""
        base_volatility = 0.10

        # Scenario 1: No Risk
        report_safe = {"details": {"synergy_risk": {"synergy_detected": False}}}
        vol_safe = financial_engine.adjust_volatility_by_topology(
            base_volatility, report_safe
        )
        assert vol_safe == base_volatility

        # Scenario 2: Risk Synergy
        report_risky = {"details": {"synergy_risk": {"synergy_detected": True}}}
        vol_risky = financial_engine.adjust_volatility_by_topology(
            base_volatility, report_risky
        )

        # Expected 20% penalty
        assert vol_risky == base_volatility * 1.2
