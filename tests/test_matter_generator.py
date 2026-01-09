import logging
import math
import random
import time
import statistics
import pytest
import networkx as nx

from app.matter_generator import BillOfMaterials, MatterGenerator

class TestMatterGenerator:
    @pytest.fixture
    def sample_graph(self):
        """
        Crea un grafo piramidal de prueba (DAG).

        Estructura topológica:
        ROOT -> APU1 (x2) -> INS1 (x5)
        ROOT -> APU2 (x1) -> INS1 (x3)
        ROOT -> APU2 (x1) -> INS2 (x10)

        Profundidades de Fibra (Fiber Depth):
        - INS1 via APU1: ROOT -> APU1 -> INS1 (Depth 3)
        - INS1 via APU2: ROOT -> APU2 -> INS1 (Depth 3)
        - INS2 via APU2: ROOT -> APU2 -> INS2 (Depth 3)
        """
        G = nx.DiGraph()
        G.add_node("PROYECTO_TOTAL", type="ROOT", level=0, description="Proyecto")
        G.add_node("APU1", type="APU", level=2, description="Muro Ladrillo")
        G.add_node("APU2", type="APU", level=2, description="Piso Concreto")
        G.add_node("INS1", type="INSUMO", level=3, description="Cemento", unit_cost=100.0, unit="kg", material_category="GENERIC")
        G.add_node("INS2", type="INSUMO", level=3, description="Arena", unit_cost=50.0, unit="m3", material_category="BULKY")

        G.add_edge("PROYECTO_TOTAL", "APU1", quantity=2.0)
        G.add_edge("PROYECTO_TOTAL", "APU2", quantity=1.0)
        G.add_edge("APU1", "INS1", quantity=5.0)
        G.add_edge("APU2", "INS1", quantity=3.0)
        G.add_edge("APU2", "INS2", quantity=10.0)

        return G

    @pytest.fixture
    def complex_graph(self):
        """Grafo complejo con múltiples caminos y unidades variadas."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT", description="Proyecto Complejo")

        for apu_id, desc in [("APU_A", "Estructura"), ("APU_B", "Acabados"), ("APU_C", "Instalaciones")]:
            G.add_node(apu_id, type="APU", description=desc)

        materials = [
            ("MAT1", "Vidrio Templado", 5000.0, "m2", "FRAGILE"),
            ("MAT2", "Pintura Epóxica", 80.0, "GAL", "HAZARDOUS"), # Usando GAL explícito
            ("MAT3", "Tubería PVC", 30.0, "m", "PRECISION"),
            ("MAT4", "Cemento Rápido", 120.0, "kg", "PERISHABLE"),
            ("MAT5", "Arena Fina", 40.0, "m3", "BULKY"),
        ]

        for mat_id, desc, cost, unit, category in materials:
            G.add_node(mat_id, type="INSUMO", description=desc, unit_cost=cost, unit=unit, material_category=category)

        for apu in ["APU_A", "APU_B", "APU_C"]:
            G.add_edge("ROOT", apu, quantity=1.0)

        edges = [
            ("APU_A", "MAT1", 2.5), ("APU_A", "MAT3", 8.0), ("APU_A", "MAT4", 15.0),
            ("APU_B", "MAT1", 1.0), ("APU_B", "MAT2", 3.0),
            ("APU_C", "MAT3", 5.0), ("APU_C", "MAT5", 2.0),
        ]
        for src, dst, qty in edges:
            G.add_edge(src, dst, quantity=qty)

        return G

    @pytest.fixture
    def cyclic_graph(self):
        G = nx.DiGraph()
        nodes = [("A", "APU"), ("B", "APU"), ("C", "APU")]
        for node_id, node_type in nodes:
            G.add_node(node_id, type=node_type, description=f"Nodo {node_id}")
        G.add_node("INS", type="INSUMO", unit_cost=10.0, description="Insumo Test")
        G.add_edge("A", "B", quantity=1.0)
        G.add_edge("B", "C", quantity=1.0)
        G.add_edge("C", "A", quantity=1.0)
        G.add_edge("A", "INS", quantity=5.0)
        return G

    def test_materialize_project_structure(self, sample_graph):
        """Valida estructura del BOM y coherencia de invariantes con nueva lógica."""
        generator = MatterGenerator()
        bom = generator.materialize_project(sample_graph)

        assert isinstance(bom, BillOfMaterials)
        assert len(bom.requirements) == 2

        req_map = {r.id: r for r in bom.requirements}
        cemento = req_map["INS1"]
        arena = req_map["INS2"]

        # Validar cantidades base (sin entropía por defecto)
        # INS1: (2*5) + (1*3) = 13
        # INS2: (1*10) = 10
        assert cemento.quantity_base == pytest.approx(13.0)
        assert arena.quantity_base == pytest.approx(10.0)

        # Costos sin factores de riesgo (solo profundidad si aplica, pero por defecto entropia es baja)
        # Nota: _apply_entropy_factors_with_bias_correction aplica profundidad incluso si riesgo es bajo?
        # Revisando código:
        # base_log_factor = 0.0 (si no hay flux_metrics ni risk_profile)
        # spec_log_factor = 0.0 (GENERIC)
        # depth_factor = 1.0 + (3 - 1) * 0.005 = 1.01
        # Total factor = exp(ln(1.01)) = 1.01

        # Para Cemento (GENERIC, Depth 3): Factor 1.01
        expected_cemento_cost = 13.0 * 1.01 * 100.0
        assert cemento.total_cost == pytest.approx(expected_cemento_cost, rel=1e-3)

        # Para Arena (BULKY, Depth 3):
        # Spec log: log1p(0.02) -> factor 1.02
        # Depth log: log(1.01)
        # Total = exp(ln(1.02) + ln(1.01)) = 1.02 * 1.01 = 1.0302
        expected_arena_cost = 10.0 * 1.0302 * 50.0
        assert arena.total_cost == pytest.approx(expected_arena_cost, rel=1e-3)

        # Metadata
        assert bom.metadata["topological_invariants"]["is_dag"] is True
        assert bom.metadata["topological_invariants"]["betti_numbers"]["b0"] == 1

    def test_apply_entropy_factors_high_risk(self, sample_graph):
        """
        Prueba la composición de factores de entropía (Riesgo + Flujo + Profundidad).
        """
        generator = MatterGenerator()
        flux_metrics = {
            "avg_saturation": 0.9, # > 0.8 -> log1p(0.05) ~ 1.05
            "pyramid_stability": 0.5, # < 1.0 -> log1p(0.03) ~ 1.03
        }

        # Base factor: 1.05 * 1.03 = 1.0815

        bom = generator.materialize_project(sample_graph, flux_metrics=flux_metrics)
        req_map = {r.id: r for r in bom.requirements}
        arena = req_map["INS2"]

        # Factores para Arena:
        # Base: 1.0815
        # Specific (BULKY): 1.02
        # Depth (3): 1.0 + (2*0.005) = 1.01
        # Total = 1.0815 * 1.02 * 1.01 = 1.11437

        expected_factor = 1.05 * 1.03 * 1.02 * 1.01
        expected_waste = expected_factor - 1.0

        assert arena.waste_factor == pytest.approx(expected_waste, rel=1e-3)

    def test_cycle_detection(self, cyclic_graph):
        generator = MatterGenerator()
        with pytest.raises(ValueError, match="ciclo"):
            generator.materialize_project(cyclic_graph)

    def test_complex_graph_clustering_and_quality(self, complex_graph):
        """Verifica clustering y métricas de calidad."""
        generator = MatterGenerator()
        bom = generator.materialize_project(complex_graph)

        req_map = {r.id: r for r in bom.requirements}

        # MAT1 (Vidrio): Path multipath
        # APU_A (Depth 3) -> 2.5
        # APU_B (Depth 3) -> 1.0
        mat1 = req_map["MAT1"]
        assert mat1.quantity_base == pytest.approx(3.5)

        # Verificar Quality Metrics
        # Cost variance debería ser 0 si todos los precios unitarios son iguales en el grafo input
        # En complex_graph, MAT1 tiene unit_cost=5000.0 único.
        assert mat1.quality_metrics["cost_variance"] == 0.0

        # Path diversity: 2 caminos distintos
        assert mat1.quality_metrics["path_diversity"] >= 2

    def test_kahan_summation_precision(self):
        """
        Verifica precisión numérica con sumas grandes.

        Usamos costos base suficientemente grandes para que el redondeo a 2 decimales
        por ítem no elimine la precisión del factor de profundidad (0.5%).
        """
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        num_materials = 1000
        unit_cost = 100.0  # Aumentado para preservar precisión post-round

        for i in range(num_materials):
            G.add_node(f"INS_{i}", type="INSUMO", unit_cost=unit_cost, description=f"M{i}")
            G.add_edge("ROOT", f"INS_{i}", quantity=1.0)

        generator = MatterGenerator(max_graph_complexity=2000000)
        bom = generator.materialize_project(G)

        # Depth factor for depth 2 (ROOT->INS): 1 + (1*0.005) = 1.005
        # Cost per item: 100.0 * 1.005 = 100.5
        # Total: 1000 * 100.5 = 100500.0

        expected_cost = num_materials * unit_cost * 1.005
        assert bom.total_material_cost == pytest.approx(expected_cost, rel=1e-9)

    def test_enriched_metadata(self, sample_graph):
        """Verifica los nuevos campos de metadata."""
        generator = MatterGenerator()
        bom = generator.materialize_project(sample_graph)

        meta = bom.metadata
        assert "thermodynamics" in meta
        assert "exergy_efficiency" in meta["thermodynamics"]
        assert "quality_analysis" in meta
        assert "risk_analysis" in meta
        assert "entropy_bits" in meta["risk_analysis"]

        # Invariantes topológicos nuevos
        # El grafo sample_graph tiene forma de diamante (Root->A1->I1, Root->A2->I1),
        # lo cual genera un ciclo en el esqueleto no dirigido, resultando en b1=1.
        # Esto es válido para un DAG (recombinación de caminos).
        assert meta["topological_invariants"]["betti_numbers"]["b1"] == 1

    def test_overflow_protection(self):
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        for i in range(200):
            G.add_node(f"INS_{i}", type="INSUMO")
            G.add_edge("ROOT", f"INS_{i}")

        generator = MatterGenerator(max_graph_complexity=100)
        with pytest.raises(OverflowError):
            generator.materialize_project(G)

