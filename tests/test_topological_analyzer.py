
import pytest
import networkx as nx
from agent.topological_analyzer import SystemTopology, PersistenceHomology

class TestSystemTopology:
    def test_initial_state(self):
        topo = SystemTopology()
        b0, b1 = topo.calculate_betti_numbers()
        # Initially no edges, so b0 = number of nodes = 4
        assert b0 == 4
        assert b1 == 0

    def test_connected_state(self):
        topo = SystemTopology()
        # Connect all required nodes
        # Agent -> Core -> Redis
        #          Core -> Filesystem
        connections = [
            ("Agent", "Core"),
            ("Core", "Redis"),
            ("Core", "Filesystem")
        ]
        topo.update_connectivity(connections)
        b0, b1 = topo.calculate_betti_numbers()
        assert b0 == 1
        assert b1 == 0

    def test_disconnected_state(self):
        topo = SystemTopology()
        connections = [("Agent", "Core")] # Redis and Filesystem missing
        topo.update_connectivity(connections)
        b0, b1 = topo.calculate_betti_numbers()
        # Connected: {Agent, Core}. Isolated: {Redis}, {Filesystem}
        # Total components: 3
        assert b0 == 3

    def test_cycles_detection(self):
        topo = SystemTopology()

        # Simulate normal traffic
        topo.record_request("req1")
        topo.record_request("req2")
        _, b1 = topo.calculate_betti_numbers()
        assert b1 == 0

        # Simulate loop (retry same request)
        for _ in range(5):
            topo.record_request("req_loop")

        _, b1 = topo.calculate_betti_numbers()
        assert b1 == 1


class TestPersistenceHomology:
    def test_stable_metric(self):
        ph = PersistenceHomology(window_size=10)
        threshold = 0.5

        # All below threshold
        for _ in range(10):
            ph.add_reading("test", 0.1)

        status = ph.analyze_persistence("test", threshold)
        assert status == "STABLE"

    def test_noise_detection(self):
        ph = PersistenceHomology(window_size=10)
        threshold = 0.5

        # Mostly stable, one spike
        for _ in range(8):
            ph.add_reading("test", 0.1)
        ph.add_reading("test", 0.9) # Spike
        ph.add_reading("test", 0.1)

        status = ph.analyze_persistence("test", threshold)
        # Duration 1 is less than 20% of 10 (2)
        # Wait, loop reversed logic in implementation needs check
        # Last reading is 0.1 (stable), so reversed loop sees stable first and stops.
        # Duration is 0 for current excursion.
        assert status == "STABLE"

        # Let's end with spike
        ph.add_reading("test", 0.9)
        status = ph.analyze_persistence("test", threshold)
        # Duration 1. 20% of 10 is 2. 1 < 2 -> NOISE
        assert status == "NOISE"

    def test_feature_detection(self):
        ph = PersistenceHomology(window_size=10)
        threshold = 0.5

        # Persistent high values
        for _ in range(5):
             ph.add_reading("test", 0.1)
        for _ in range(5):
             ph.add_reading("test", 0.9)

        status = ph.analyze_persistence("test", threshold)
        # Duration 5. 5 >= 2 -> FEATURE
        assert status == "FEATURE"
