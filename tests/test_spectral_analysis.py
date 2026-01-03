import unittest
import networkx as nx
import numpy as np
from agent.business_topology import BusinessTopologicalAnalyzer, TopologicalMetrics

class TestSpectralAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = BusinessTopologicalAnalyzer()

    def test_spectral_stability_connected(self):
        # Create a simple connected graph (Path graph: 0-1-2-3)
        # Undirected version is connected.
        G = nx.path_graph(4, create_using=nx.DiGraph)

        # Analyze
        result = self.analyzer.analyze_spectral_stability(G)

        self.assertIn("fiedler_value", result)
        self.assertIn("spectral_energy", result)
        self.assertIn("wavelength", result)
        self.assertIn("resonance_risk", result)

        # Fiedler value for path graph > 0
        self.assertGreater(result["fiedler_value"], 0)
        # Risk should be low for simple path
        self.assertFalse(result["resonance_risk"])

    def test_spectral_stability_disconnected(self):
        # Disconnected graph
        G = nx.DiGraph()
        G.add_edge(1, 2)
        G.add_edge(3, 4)

        result = self.analyzer.analyze_spectral_stability(G)

        # Fiedler value should be close to 0 (approx 0 for disconnected)
        # Using normalized laplacian, eigenvalues are between 0 and 2.
        # Beta0 = 2, so at least 2 zero eigenvalues. Lambda_2 might be 0.
        self.assertLess(result["fiedler_value"], 1e-9)

    def test_spectral_stability_resonance(self):
        # Complete graph (High connectivity -> high Fiedler value -> Low Wavelength?)
        # Or maybe star graph?
        # Let's try complete graph K5
        G = nx.complete_graph(5, create_using=nx.DiGraph)

        result = self.analyzer.analyze_spectral_stability(G)

        # Check structure
        self.assertIsInstance(result["wavelength"], float)

    def test_empty_graph(self):
        G = nx.DiGraph()
        result = self.analyzer.analyze_spectral_stability(G)
        self.assertEqual(result["fiedler_value"], 0.0)

    def test_large_graph_sparse(self):
        # Graph with N=30 to trigger sparse logic
        G = nx.path_graph(30, create_using=nx.DiGraph)
        result = self.analyzer.analyze_spectral_stability(G)
        self.assertIn("fiedler_value", result)
        self.assertGreater(result["fiedler_value"], 0)

if __name__ == '__main__':
    unittest.main()
