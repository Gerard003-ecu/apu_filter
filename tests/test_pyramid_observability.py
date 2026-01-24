import pytest
from unittest.mock import MagicMock, patch
import networkx as nx
from datetime import datetime

from app.schemas import Stratum
from agent.apu_agent import AutonomousAgent, SystemStatus
from app.telemetry_narrative import TelemetryNarrator
from app.telemetry import TelemetryContext, StepStatus, TelemetrySpan
from app.topology_viz import convert_graph_to_cytoscape_elements, AnomalyData

class TestPyramidObservability:

    # --- Agent Tests ---

    @patch('agent.apu_agent.requests.Session')
    def test_agent_get_stratum_health(self, mock_session):
        agent = AutonomousAgent()

        # 1. Test PHYSICS (FluxCondenser)
        # Mock observe to return some data
        with patch.object(agent, 'observe') as mock_observe:
            mock_telemetry = MagicMock()
            mock_telemetry.flyback_voltage = 0.3
            mock_telemetry.saturation = 0.5
            mock_observe.return_value = mock_telemetry

            health = agent.get_stratum_health(Stratum.PHYSICS)
            assert health['stratum'] == 'PHYSICS'
            assert health['voltage'] == 0.3
            assert health['saturation'] == 0.5
            assert health['status'] == 'NOMINAL'

        # 2. Test TACTICS (Topology)
        # Mock topology health
        agent.topology = MagicMock()
        mock_topo_health = MagicMock()
        mock_topo_health.betti.b0 = 1
        mock_topo_health.betti.b1 = 0
        mock_topo_health.betti.is_connected = True
        mock_topo_health.health_score = 0.95
        agent.topology.get_topological_health.return_value = mock_topo_health

        health = agent.get_stratum_health(Stratum.TACTICS)
        assert health['stratum'] == 'TACTICS'
        assert health['betti_0'] == 1
        assert health['betti_1'] == 0
        assert health['is_connected'] is True

        # 3. Test STRATEGY
        agent._last_status = SystemStatus.NOMINAL
        agent._last_decision = MagicMock()
        agent._last_decision.name = "HEARTBEAT"

        health = agent.get_stratum_health(Stratum.STRATEGY)
        assert health['stratum'] == 'STRATEGY'
        assert health['risk_detected'] is False
        assert health['last_decision'] == 'HEARTBEAT'

        # 4. Test WISDOM
        health = agent.get_stratum_health(Stratum.WISDOM)
        assert health['stratum'] == 'WISDOM'
        assert health['verdict'] == 'NOMINAL'

    # --- Telemetry Narrator Tests ---

    def test_narrator_root_cause_stratum(self):
        narrator = TelemetryNarrator()
        context = TelemetryContext()

        # Case 1: Physics Failure (Base)
        # Mocking a span structure that failed at Physics
        span_physics = TelemetrySpan(
            name="flux_condenser",
            level=0,
            stratum=Stratum.PHYSICS
        )
        span_physics.status = StepStatus.FAILURE
        span_physics.errors = [{"message": "High Voltage", "type": "PhysicsError"}]
        context.root_spans = [span_physics]

        root = narrator.get_root_cause_stratum(context)
        assert root == Stratum.PHYSICS

        # Case 2: Tactics Failure (Middle)
        # Physics OK, Tactics Failed
        context_tactics = TelemetryContext()
        span_physics_ok = TelemetrySpan(name="load_data", level=0, stratum=Stratum.PHYSICS)
        span_physics_ok.status = StepStatus.SUCCESS

        span_tactics = TelemetrySpan(name="calculate_costs", level=0, stratum=Stratum.TACTICS)
        span_tactics.status = StepStatus.FAILURE
        span_tactics.errors = [{"message": "Cycle detected", "type": "TopologyError"}]

        context_tactics.root_spans = [span_physics_ok, span_tactics]

        root = narrator.get_root_cause_stratum(context_tactics)
        # According to logic, if Tactics failed, it should return Tactics
        # (Assuming Physics passed)
        assert root == Stratum.TACTICS

    # --- Topology Viz Tests ---

    def test_topology_viz_filtering(self):
        # Create a graph with nodes at different levels
        G = nx.DiGraph()

        # Level 3: Physics/Insumos
        G.add_node("INSUMO_1", type="INSUMO", level=3, description="Material A")

        # Level 2: Tactics/APUs
        G.add_node("APU_1", type="APU", level=2, description="Concrete Activity")

        # Level 1: Strategy/Chapters
        G.add_node("CHAPTER_1", type="CHAPTER", level=1, description="Foundations")

        # Level 0: Wisdom/Project
        G.add_node("PROJECT", type="BUDGET", level=0, description="Main Project")

        # Edges
        G.add_edge("CHAPTER_1", "APU_1")
        G.add_edge("APU_1", "INSUMO_1")
        G.add_edge("PROJECT", "CHAPTER_1")

        anomaly_data = AnomalyData()

        # 1. Filter PHYSICS (3) -> Should only see INSUMO_1
        elements = convert_graph_to_cytoscape_elements(G, anomaly_data, stratum_filter=3)
        ids = [e['data']['id'] for e in elements if 'source' not in e['data']]
        assert "INSUMO_1" in ids
        assert "APU_1" not in ids
        assert "CHAPTER_1" not in ids

        # 2. Filter TACTICS (2) -> APU (2) + INSUMO (3)
        elements = convert_graph_to_cytoscape_elements(G, anomaly_data, stratum_filter=2)
        ids = [e['data']['id'] for e in elements if 'source' not in e['data']]
        assert "APU_1" in ids
        assert "INSUMO_1" in ids
        assert "CHAPTER_1" not in ids

        # 3. Filter STRATEGY (1) -> CHAPTER (1) + APU (2)
        elements = convert_graph_to_cytoscape_elements(G, anomaly_data, stratum_filter=1)
        ids = [e['data']['id'] for e in elements if 'source' not in e['data']]
        assert "CHAPTER_1" in ids
        assert "APU_1" in ids
        assert "INSUMO_1" not in ids

        # 4. Filter WISDOM (0) -> PROJECT (0) + CHAPTER (1)
        elements = convert_graph_to_cytoscape_elements(G, anomaly_data, stratum_filter=0)
        ids = [e['data']['id'] for e in elements if 'source' not in e['data']]
        assert "PROJECT" in ids
        assert "CHAPTER_1" in ids
        assert "APU_1" not in ids

        # 5. No Filter -> All
        elements = convert_graph_to_cytoscape_elements(G, anomaly_data, stratum_filter=None)
        ids = [e['data']['id'] for e in elements if 'source' not in e['data']]
        assert len(ids) == 4
