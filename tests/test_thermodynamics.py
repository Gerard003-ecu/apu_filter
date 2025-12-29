import unittest
from unittest.mock import Mock, patch
import math
import networkx as nx
import time
from app.flux_condenser import FluxPhysicsEngine
from app.financial_engine import FinancialEngine, FinancialConfig
from app.matter_generator import MatterGenerator, MaterialRequirement
from agent.business_topology import BusinessTopologicalAnalyzer, TopologicalMetrics

class TestFluxPhysicsEngine(unittest.TestCase):
    def test_calculate_system_entropy(self):
        engine = FluxPhysicsEngine(capacitance=5000.0, resistance=10.0, inductance=2.0)

        # Test zero records
        metrics = engine.calculate_system_entropy(0, 0, 1.0)
        self.assertEqual(metrics["entropy_absolute"], 0.0)

        # Test perfect system (no errors)
        metrics = engine.calculate_system_entropy(100, 0, 1.0)
        self.assertEqual(metrics["entropy_absolute"], 0.0)

        # Test max entropy (p=0.5) - This should be close to Thermal Death threshold if we normalize
        # With current implementation (k=1.0, ln), max S = 0.693.
        # So "is_thermal_death" (>0.8) is actually unreachable with current math unless we change k or threshold.
        # HOWEVER, the Prompt defined the logic.
        # Let's adjust the expectation to what the code DOES, or fix the code.
        # I will assume we should fix the code to normalize, but for now I test the behavior.
        # If I fix the code to k = 1.44, then max S = 1.0.

        # Let's test p=0.5 (Max Chaos)
        metrics = engine.calculate_system_entropy(100, 50, 1.0)
        # With k=1.4427 (Base 2), S = 1.0. This SHOULD be thermal death (>0.8).
        # I will update the code to use Base 2 normalization.
        self.assertTrue(metrics["is_thermal_death"])

    def test_calculate_entropy_value(self):
        engine = FluxPhysicsEngine(capacitance=5000.0, resistance=10.0, inductance=2.0)
        # p = 0.5 -> max entropy
        metrics = engine.calculate_system_entropy(100, 50, 1.0)
        # Expecting ~1.0 with normalization
        self.assertAlmostEqual(metrics["entropy_absolute"], 1.0, places=1)

    def test_calculate_metrics_integration(self):
        engine = FluxPhysicsEngine(capacitance=5000.0, resistance=10.0, inductance=2.0)

        metrics = engine.calculate_metrics(
            total_records=100,
            cache_hits=90,
            error_count=5,
            processing_time=1.0
        )

        self.assertIn("entropy_absolute", metrics)
        self.assertIn("entropy_rate", metrics)
        self.assertIn("is_thermal_death", metrics)

class TestFinancialEngine(unittest.TestCase):
    def setUp(self):
        self.config = FinancialConfig()
        self.engine = FinancialEngine(self.config)

    def test_thermal_inertia(self):
        inertia = self.engine.calculate_financial_thermal_inertia(liquidity=0.2, fixed_contracts_ratio=0.5)
        self.assertAlmostEqual(inertia, 0.1)

    def test_predict_temperature_change(self):
        # High inertia
        temp_change_low = self.engine.predict_temperature_change(0.05, 0.5)
        # Low inertia
        temp_change_high = self.engine.predict_temperature_change(0.05, 0.1)

        self.assertLess(temp_change_low, temp_change_high)

        # Zero inertia (no protection)
        temp_change_zero = self.engine.predict_temperature_change(0.05, 0.0)
        self.assertEqual(temp_change_zero, 0.05)

    def test_analyze_project_integration(self):
        analysis = self.engine.analyze_project(
            initial_investment=1000,
            expected_cash_flows=[200, 200, 200, 200, 200],
            cost_std_dev=100,
            project_volatility=0.2,
            liquidity=0.2,
            fixed_contracts_ratio=0.5
        )

        self.assertIn("thermodynamics", analysis)
        thermo = analysis["thermodynamics"]
        self.assertEqual(thermo["financial_thermal_inertia"], 0.1)
        self.assertEqual(thermo["predicted_temperature_rise"], 0.05/0.1)

class TestMatterGenerator(unittest.TestCase):
    def test_analyze_budget_exergy(self):
        generator = MatterGenerator()

        items = [
            MaterialRequirement(
                id="1", description="CONCRETO 3000 PSI", quantity_base=10, unit="M3",
                waste_factor=0.05, quantity_total=10.5, unit_cost=100, total_cost=1050
            ),
            MaterialRequirement(
                id="2", description="ACERO DE REFUERZO", quantity_base=100, unit="KG",
                waste_factor=0.05, quantity_total=105, unit_cost=2, total_cost=210
            ),
            MaterialRequirement(
                id="3", description="PINTURA DECORATIVA", quantity_base=10, unit="GAL",
                waste_factor=0.1, quantity_total=11, unit_cost=50, total_cost=550
            )
        ]

        exergy_report = generator.analyze_budget_exergy(items)

        total_cost = 1050 + 210 + 550
        useful_cost = 1050 + 210
        expected_efficiency = useful_cost / total_cost

        self.assertAlmostEqual(exergy_report["exergy_efficiency"], expected_efficiency)
        self.assertAlmostEqual(exergy_report["structural_investment"], useful_cost)
        self.assertAlmostEqual(exergy_report["decorative_investment"], 550)

class TestBusinessTopologicalAnalyzer(unittest.TestCase):
    def test_analyze_inflationary_convection(self):
        analyzer = BusinessTopologicalAnalyzer()
        G = nx.DiGraph()

        # A -> T (A depends on T) - In dependency graph logic
        # But here logic is: APU -> Insumo.
        # If Transport (T) price rises, it affects APU (A).
        # So we look for Predecessors of T.

        G.add_node("APU1", type="APU")
        G.add_node("APU2", type="APU")
        G.add_node("T1", type="INSUMO", description="TRANSPORTE MATERIAL")
        G.add_node("M1", type="INSUMO", description="MATERIAL 1")

        # APU1 uses T1 and M1
        G.add_edge("APU1", "T1", total_cost=200)
        G.add_edge("APU1", "M1", total_cost=800)

        # APU2 uses only M1
        G.add_edge("APU2", "M1", total_cost=500)

        fluid_nodes = ["T1"]

        report = analyzer.analyze_inflationary_convection(G, fluid_nodes)

        self.assertIn("APU1", report["convection_impact"])
        self.assertNotIn("APU2", report["convection_impact"])

        # Impact on APU1 = 200 / (200+800) = 0.2
        self.assertAlmostEqual(report["convection_impact"]["APU1"], 0.2)

        # 0.2 is NOT > 0.2, so it should NOT be in high_risk_nodes
        self.assertNotIn("APU1", report["high_risk_nodes"])

    def test_high_risk_convection(self):
        analyzer = BusinessTopologicalAnalyzer()
        G = nx.DiGraph()
        G.add_node("APU_RISKY", type="APU")
        G.add_node("T_FUEL", type="INSUMO", description="COMBUSTIBLE DIESEL")

        G.add_edge("APU_RISKY", "T_FUEL", total_cost=500)
        # Total cost of APU_RISKY out edges = 500. Impact = 1.0

        report = analyzer.analyze_inflationary_convection(G, ["T_FUEL"])
        self.assertIn("APU_RISKY", report["high_risk_nodes"])

if __name__ == '__main__':
    unittest.main()
