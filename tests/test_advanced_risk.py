"""
Pruebas para el análisis de riesgos avanzados en topología de negocios.

Este módulo contiene pruebas unitarias para verificar la funcionalidad de:
- Eficiencia de Euler en grafos.
- Detección de sinergia de riesgos (Producto Copa / Efecto Dominó).
- Penalización de volatilidad financiera basada en topología.
"""

import networkx as nx
import pytest

from agent.business_topology import BusinessTopologicalAnalyzer
from app.financial_engine import FinancialConfig, FinancialEngine


class TestAdvancedRiskAnalysis:
    """Suite de pruebas para análisis de riesgos topológicos y financieros."""

    @pytest.fixture
    def analyzer(self):
        """Fixture que proporciona una instancia de BusinessTopologicalAnalyzer."""
        return BusinessTopologicalAnalyzer()

    @pytest.fixture
    def financial_engine(self):
        """Fixture que proporciona una instancia de FinancialEngine con configuración por defecto."""
        config = FinancialConfig()
        return FinancialEngine(config)

    def test_euler_efficiency_tree(self, analyzer):
        """Verifica que un árbol perfecto tenga eficiencia máxima (1.0)."""
        # Crear un árbol: Raíz -> A, Raíz -> B
        G = nx.DiGraph()
        G.add_edges_from([("Root", "A"), ("Root", "B")])

        efficiency = analyzer.calculate_euler_efficiency(G)
        # Nodos=3, Aristas=2. Aristas esperadas para árbol = 3-1 = 2. Exceso = 0.
        # Eficiencia es exp(-exceso/nodos) -> exp(0) = 1.0
        assert efficiency == 1.0

    def test_euler_efficiency_complex(self, analyzer):
        """Verifica que un grafo denso tenga baja eficiencia."""
        # Grafo completo K4
        G = nx.complete_graph(4, create_using=nx.DiGraph())

        efficiency = analyzer.calculate_euler_efficiency(G)
        # Nodos=4. Aristas esperadas para árbol=3. Aristas actuales=12. Exceso = 9.
        # Eficiencia = exp(-9/4) = exp(-2.25) ~ 0.105
        assert efficiency < 0.5
        assert efficiency > 0.0

    def test_risk_synergy_detection(self, analyzer):
        """Prueba la detección de Producto Copa / Sinergia (Ciclos compartiendo nodos críticos)."""
        # Necesitamos dos ciclos que compartan al menos dos nodos (una arista) para ser detectados robustamente
        # como sinergia por la lógica de intersección: len(intersection) >= 2.

        G = nx.DiGraph()
        # Ciclo 1: A -> B -> C -> A
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        # Ciclo 2: A -> B -> D -> A
        G.add_edges_from([("A", "B"), ("B", "D"), ("D", "A")])

        # Arista compartida: A->B.
        # Intersección de nodos: {A, B}

        synergy = analyzer.detect_risk_synergy(G)

        assert synergy["synergy_detected"] is True
        # Tanto A como B son nodos puente.
        bridge_ids = [n["id"] for n in synergy["bridge_nodes"]]
        assert "A" in bridge_ids
        assert "B" in bridge_ids
        assert synergy["intersecting_cycles_count"] >= 1

    def test_no_synergy_disjoint_cycles(self, analyzer):
        """Prueba que ciclos disjuntos no activen la detección de sinergia."""
        G = nx.DiGraph()
        # Ciclo 1: A -> B -> A
        G.add_edges_from([("A", "B"), ("B", "A")])
        # Ciclo 2: C -> D -> C
        G.add_edges_from([("C", "D"), ("D", "C")])

        synergy = analyzer.detect_risk_synergy(G)

        assert synergy["synergy_detected"] is False

    def test_financial_volatility_penalty(self, financial_engine):
        """Prueba que la sinergia de riesgos incremente la volatilidad financiera."""
        base_volatility = 0.10

        # Escenario 1: Sin Riesgo
        # Nota: adjust_volatility_by_topology accede a topology_report.get("synergy_risk", {})
        # Así que debemos pasar un diccionario donde "synergy_risk" esté en el nivel superior.

        report_safe = {"synergy_risk": {"synergy_detected": False}}
        vol_safe = financial_engine.adjust_volatility_by_topology(
            base_volatility, report_safe
        )
        assert vol_safe == base_volatility

        # Escenario 2: Sinergia de Riesgo
        report_risky = {"synergy_risk": {"synergy_detected": True}}
        vol_risky = financial_engine.adjust_volatility_by_topology(
            base_volatility, report_risky
        )

        # Penalización esperada del 20%
        assert vol_risky == pytest.approx(base_volatility * 1.2)
