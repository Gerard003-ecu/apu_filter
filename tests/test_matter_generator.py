import pytest
import networkx as nx
from app.matter_generator import MatterGenerator, MaterialRequirement, BillOfMaterials

class TestMatterGenerator:

    @pytest.fixture
    def sample_graph(self):
        """
        Crea un grafo piramidal de prueba:
        ROOT -> APU1 (x2) -> INS1 (x5)
        ROOT -> APU2 (x1) -> INS1 (x3)
                          -> INS2 (x10)

        Total INS1: (2 * 5) + (1 * 3) = 13
        Total INS2: (1 * 10) = 10
        """
        G = nx.DiGraph()

        # Nodos
        G.add_node("PROYECTO_TOTAL", type="ROOT", level=0, description="Proyecto")
        G.add_node("APU1", type="APU", level=2, description="Muro Ladrillo")
        G.add_node("APU2", type="APU", level=2, description="Piso Concreto")
        G.add_node("INS1", type="INSUMO", level=3, description="Cemento", unit_cost=100.0)
        G.add_node("INS2", type="INSUMO", level=3, description="Arena", unit_cost=50.0)

        # Aristas (Cantidades)
        # Nivel 1 (Proyecto -> APU)
        G.add_edge("PROYECTO_TOTAL", "APU1", quantity=2.0)
        G.add_edge("PROYECTO_TOTAL", "APU2", quantity=1.0)

        # Nivel 2 (APU -> Insumo)
        G.add_edge("APU1", "INS1", quantity=5.0)
        G.add_edge("APU2", "INS1", quantity=3.0)
        G.add_edge("APU2", "INS2", quantity=10.0)

        return G

    def test_materialize_project_structure(self, sample_graph):
        """Prueba la estructura básica del BOM generado."""
        generator = MatterGenerator()
        bom = generator.materialize_project(sample_graph)

        assert isinstance(bom, BillOfMaterials)
        assert len(bom.requirements) == 2 # Cemento y Arena

        # Verificar totales
        cemento = next(r for r in bom.requirements if r.id == "INS1")
        arena = next(r for r in bom.requirements if r.id == "INS2")

        assert cemento.quantity_base == 13.0
        assert arena.quantity_base == 10.0

        # Costos
        assert cemento.total_cost == 13.0 * 100.0
        assert arena.total_cost == 10.0 * 50.0

        assert bom.total_material_cost == (1300.0 + 500.0)

    def test_apply_entropy_factors_clean(self, sample_graph):
        """Prueba sin factores de riesgo (caso ideal)."""
        generator = MatterGenerator()
        bom = generator.materialize_project(sample_graph)

        for req in bom.requirements:
            assert req.waste_factor == 0.0
            assert req.quantity_total == req.quantity_base

    def test_apply_entropy_factors_high_risk(self, sample_graph):
        """Prueba con métricas de flujo adversas."""
        generator = MatterGenerator()
        flux_metrics = {
            "avg_saturation": 0.9, # > 0.8 triggers +5%
            "pyramid_stability": 0.5 # < 1.0 triggers +3%
        }

        bom = generator.materialize_project(sample_graph, flux_metrics=flux_metrics)

        expected_waste = 0.08 # 5% + 3%

        for req in bom.requirements:
            assert req.waste_factor == pytest.approx(expected_waste)
            assert req.quantity_total == pytest.approx(req.quantity_base * (1 + expected_waste))

    def test_source_tracking(self, sample_graph):
        """Verifica que se rastreen los APUs de origen."""
        generator = MatterGenerator()
        bom = generator.materialize_project(sample_graph)

        cemento = next(r for r in bom.requirements if r.id == "INS1")
        assert "APU1" in cemento.source_apus
        assert "APU2" in cemento.source_apus

        arena = next(r for r in bom.requirements if r.id == "INS2")
        assert "APU2" in arena.source_apus
        assert "APU1" not in arena.source_apus

    def test_missing_root_fallback(self):
        """Prueba comportamiento cuando no hay nodo ROOT explícito."""
        G = nx.DiGraph()
        # Grafo desconectado, solo APUs e Insumos
        G.add_node("APU1", type="APU")
        G.add_node("INS1", type="INSUMO", unit_cost=10)
        G.add_edge("APU1", "INS1", quantity=5.0)

        # APU1 es raíz implícita (in_degree=0)

        generator = MatterGenerator()
        bom = generator.materialize_project(G)

        assert len(bom.requirements) == 1
        assert bom.requirements[0].quantity_base == 5.0
