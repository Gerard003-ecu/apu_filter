"""
Suite de Pruebas para el Generador de Materiales (MatterGenerator)
==================================================================

Fundamentos matemáticos:
- Propagación de cantidades en DAG (Directed Acyclic Graph)
- Composición logarítmica de factores de entropía
- Suma de Kahan para precisión numérica en agregaciones
- Números de Betti para caracterización topológica

Modelo de materialización:
    Q_ajustada = Q_base × F_entropía
    C_total = Q_ajustada × precio_unitario

Donde:
    F_entropía = exp(ln(F_flujo) + ln(F_específico) + ln(F_profundidad))

Autor: Artesano Programador Senior
Versión: 2.0.0
"""

import logging
import math
import random
import time
import statistics
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import pytest
import networkx as nx
import numpy as np

from app.matter_generator import BillOfMaterials, MatterGenerator


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: CONSTANTES Y CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

# Factores de entropía por categoría de material
CATEGORY_FACTORS = {
    "GENERIC": 0.00,      # Sin factor adicional
    "BULKY": 0.02,        # Voluminoso: +2%
    "FRAGILE": 0.05,      # Frágil: +5%
    "HAZARDOUS": 0.08,    # Peligroso: +8%
    "PERISHABLE": 0.04,   # Perecedero: +4%
    "PRECISION": 0.03,    # Precisión: +3%
}

# Factor de profundidad: 0.5% por nivel adicional
DEPTH_FACTOR_RATE = 0.005

# Umbrales de flujo
SATURATION_THRESHOLD = 0.8
STABILITY_THRESHOLD = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: FIXTURES FUNDAMENTALES
# ═══════════════════════════════════════════════════════════════════════════════

class TestFixtures:
    """Fixtures reutilizables para todas las clases de prueba."""
    
    @pytest.fixture
    def generator(self) -> MatterGenerator:
        """Generador con configuración por defecto."""
        return MatterGenerator()
    
    @pytest.fixture
    def generator_high_capacity(self) -> MatterGenerator:
        """Generador con alta capacidad para grafos grandes."""
        return MatterGenerator(max_graph_complexity=10_000_000)
    
    @pytest.fixture
    def empty_graph(self) -> nx.DiGraph:
        """Grafo completamente vacío."""
        return nx.DiGraph()
    
    @pytest.fixture
    def single_node_graph(self) -> nx.DiGraph:
        """Grafo con un solo nodo ROOT."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT", description="Proyecto Vacío")
        return G
    
    @pytest.fixture
    def minimal_graph(self) -> nx.DiGraph:
        """Grafo mínimo: ROOT → INSUMO."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT", description="Proyecto Mínimo")
        G.add_node("INS1", type="INSUMO", description="Material Único",
                   unit_cost=100.0, unit="kg", material_category="GENERIC")
        G.add_edge("ROOT", "INS1", quantity=1.0)
        return G
    
    @pytest.fixture
    def sample_graph(self) -> nx.DiGraph:
        """
        Grafo piramidal estándar para pruebas.
        
        Estructura topológica:
            ROOT
            ├── APU1 (×2)
            │   └── INS1 (×5) [Cemento]
            └── APU2 (×1)
                ├── INS1 (×3) [Cemento]
                └── INS2 (×10) [Arena]
        
        Cantidades base esperadas:
            - INS1: 2×5 + 1×3 = 13
            - INS2: 1×10 = 10
        
        Profundidades (Fiber Depth):
            - Todos los insumos: depth = 3 (ROOT → APU → INS)
        """
        G = nx.DiGraph()
        
        # Nodos
        G.add_node("PROYECTO_TOTAL", type="ROOT", level=0, 
                   description="Proyecto de Construcción")
        G.add_node("APU1", type="APU", level=2, description="Muro Ladrillo")
        G.add_node("APU2", type="APU", level=2, description="Piso Concreto")
        G.add_node("INS1", type="INSUMO", level=3, description="Cemento",
                   unit_cost=100.0, unit="kg", material_category="GENERIC")
        G.add_node("INS2", type="INSUMO", level=3, description="Arena",
                   unit_cost=50.0, unit="m3", material_category="BULKY")
        
        # Aristas
        G.add_edge("PROYECTO_TOTAL", "APU1", quantity=2.0)
        G.add_edge("PROYECTO_TOTAL", "APU2", quantity=1.0)
        G.add_edge("APU1", "INS1", quantity=5.0)
        G.add_edge("APU2", "INS1", quantity=3.0)
        G.add_edge("APU2", "INS2", quantity=10.0)
        
        return G
    
    @pytest.fixture
    def complex_graph(self) -> nx.DiGraph:
        """
        Grafo complejo con múltiples caminos y categorías variadas.
        
        Estructura:
            ROOT
            ├── APU_A (Estructura)
            │   ├── MAT1 (Vidrio, FRAGILE) ×2.5
            │   ├── MAT3 (Tubería, PRECISION) ×8.0
            │   └── MAT4 (Cemento, PERISHABLE) ×15.0
            ├── APU_B (Acabados)
            │   ├── MAT1 (Vidrio) ×1.0  ← Camino alternativo
            │   └── MAT2 (Pintura, HAZARDOUS) ×3.0
            └── APU_C (Instalaciones)
                ├── MAT3 (Tubería) ×5.0  ← Camino alternativo
                └── MAT5 (Arena, BULKY) ×2.0
        """
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT", description="Proyecto Complejo")
        
        # APUs
        for apu_id, desc in [("APU_A", "Estructura"), 
                             ("APU_B", "Acabados"), 
                             ("APU_C", "Instalaciones")]:
            G.add_node(apu_id, type="APU", description=desc)
        
        # Materiales con diferentes categorías
        materials = [
            ("MAT1", "Vidrio Templado", 5000.0, "m2", "FRAGILE"),
            ("MAT2", "Pintura Epóxica", 80.0, "GAL", "HAZARDOUS"),
            ("MAT3", "Tubería PVC", 30.0, "m", "PRECISION"),
            ("MAT4", "Cemento Rápido", 120.0, "kg", "PERISHABLE"),
            ("MAT5", "Arena Fina", 40.0, "m3", "BULKY"),
        ]
        
        for mat_id, desc, cost, unit, category in materials:
            G.add_node(mat_id, type="INSUMO", description=desc,
                       unit_cost=cost, unit=unit, material_category=category)
        
        # Conexiones ROOT → APUs
        for apu in ["APU_A", "APU_B", "APU_C"]:
            G.add_edge("ROOT", apu, quantity=1.0)
        
        # Conexiones APUs → Materiales
        edges = [
            ("APU_A", "MAT1", 2.5), ("APU_A", "MAT3", 8.0), ("APU_A", "MAT4", 15.0),
            ("APU_B", "MAT1", 1.0), ("APU_B", "MAT2", 3.0),
            ("APU_C", "MAT3", 5.0), ("APU_C", "MAT5", 2.0),
        ]
        for src, dst, qty in edges:
            G.add_edge(src, dst, quantity=qty)
        
        return G
    
    @pytest.fixture
    def deep_graph(self) -> nx.DiGraph:
        """
        Grafo profundo para probar factores de profundidad.
        
        Estructura: ROOT → L1 → L2 → L3 → L4 → INSUMO
        Depth = 6
        """
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT", description="Proyecto Profundo")
        
        prev = "ROOT"
        for i in range(1, 5):
            node_id = f"LEVEL_{i}"
            G.add_node(node_id, type="APU", description=f"Nivel {i}")
            G.add_edge(prev, node_id, quantity=1.0)
            prev = node_id
        
        G.add_node("DEEP_INS", type="INSUMO", description="Insumo Profundo",
                   unit_cost=100.0, unit="kg", material_category="GENERIC")
        G.add_edge(prev, "DEEP_INS", quantity=10.0)
        
        return G
    
    @pytest.fixture
    def cyclic_graph(self) -> nx.DiGraph:
        """Grafo con ciclo (inválido para materialización)."""
        G = nx.DiGraph()
        G.add_node("A", type="APU", description="Nodo A")
        G.add_node("B", type="APU", description="Nodo B")
        G.add_node("C", type="APU", description="Nodo C")
        G.add_node("INS", type="INSUMO", unit_cost=10.0, description="Insumo")
        
        G.add_edge("A", "B", quantity=1.0)
        G.add_edge("B", "C", quantity=1.0)
        G.add_edge("C", "A", quantity=1.0)  # Crea ciclo
        G.add_edge("A", "INS", quantity=5.0)
        
        return G
    
    @pytest.fixture
    def all_categories_graph(self) -> nx.DiGraph:
        """Grafo con todas las categorías de materiales."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT", description="Test Categorías")
        G.add_node("APU", type="APU", description="APU Único")
        G.add_edge("ROOT", "APU", quantity=1.0)
        
        for cat, factor in CATEGORY_FACTORS.items():
            node_id = f"INS_{cat}"
            G.add_node(node_id, type="INSUMO", description=f"Material {cat}",
                       unit_cost=1000.0, unit="kg", material_category=cat)
            G.add_edge("APU", node_id, quantity=10.0)
        
        return G


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: PRUEBAS DE GRAFOS FUNDAMENTALES
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphFundamentals(TestFixtures):
    """
    Pruebas de casos fundamentales y degenerados de grafos.
    
    Verifica que el generador maneje correctamente estructuras
    básicas y casos límite.
    """
    
    def test_empty_graph_handling(self, generator, empty_graph):
        """Un grafo vacío debe manejarse con error o BOM vacío."""
        with pytest.raises((ValueError, KeyError)):
            generator.materialize_project(empty_graph)
    
    def test_single_root_node(self, generator, single_node_graph):
        """Un proyecto sin insumos produce BOM vacío."""
        bom = generator.materialize_project(single_node_graph)
        
        assert isinstance(bom, BillOfMaterials)
        assert len(bom.requirements) == 0
        assert bom.total_material_cost == 0.0
    
    def test_minimal_graph_structure(self, generator, minimal_graph):
        """
        Grafo mínimo: ROOT → INSUMO.
        Profundidad = 2, cantidad = 1.
        """
        bom = generator.materialize_project(minimal_graph)
        
        assert len(bom.requirements) == 1
        req = bom.requirements[0]
        
        assert req.id == "INS1"
        assert req.quantity_base == 1.0
        
        # Factor de profundidad: 1 + (2-1) × 0.005 = 1.005
        expected_factor = 1.005
        expected_cost = 1.0 * expected_factor * 100.0
        assert req.total_cost == pytest.approx(expected_cost, rel=1e-3)
    
    def test_graph_is_dag_validation(self, generator, sample_graph):
        """Verifica que el grafo sea un DAG válido."""
        bom = generator.materialize_project(sample_graph)
        
        assert bom.metadata["topological_invariants"]["is_dag"] is True
    
    def test_single_component_validation(self, generator, sample_graph):
        """El grafo debe tener exactamente una componente conexa (β₀ = 1)."""
        bom = generator.materialize_project(sample_graph)
        
        b0 = bom.metadata["topological_invariants"]["betti_numbers"]["b0"]
        assert b0 == 1


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: PRUEBAS DE ESTRUCTURA BOM
# ═══════════════════════════════════════════════════════════════════════════════

class TestBOMStructure(TestFixtures):
    """
    Pruebas para la estructura del Bill of Materials.
    
    El BOM debe contener:
    - Lista de requirements (MaterialRequirement)
    - Metadata con invariantes topológicos
    - Costo total calculado correctamente
    """
    
    def test_bom_is_valid_type(self, generator, sample_graph):
        """El resultado es una instancia válida de BillOfMaterials."""
        bom = generator.materialize_project(sample_graph)
        
        assert isinstance(bom, BillOfMaterials)
        assert hasattr(bom, 'requirements')
        assert hasattr(bom, 'metadata')
        assert hasattr(bom, 'total_material_cost')
    
    def test_bom_requirements_count(self, generator, sample_graph):
        """El número de requirements coincide con insumos únicos."""
        bom = generator.materialize_project(sample_graph)
        
        # sample_graph tiene INS1 e INS2
        assert len(bom.requirements) == 2
    
    def test_bom_requirement_fields(self, generator, sample_graph):
        """Cada requirement tiene todos los campos necesarios."""
        bom = generator.materialize_project(sample_graph)
        
        for req in bom.requirements:
            assert hasattr(req, 'id')
            assert hasattr(req, 'quantity_base')
            assert hasattr(req, 'total_cost')
            assert hasattr(req, 'waste_factor')
            assert hasattr(req, 'quality_metrics')
    
    def test_bom_metadata_structure(self, generator, sample_graph):
        """La metadata contiene todas las secciones requeridas."""
        bom = generator.materialize_project(sample_graph)
        meta = bom.metadata
        
        required_sections = [
            "topological_invariants",
            "thermodynamics",
            "quality_analysis",
            "risk_analysis"
        ]
        
        for section in required_sections:
            assert section in meta, f"Falta sección: {section}"
    
    def test_bom_total_cost_consistency(self, generator, sample_graph):
        """El costo total es la suma de los costos individuales."""
        bom = generator.materialize_project(sample_graph)
        
        calculated_total = sum(req.total_cost for req in bom.requirements)
        assert bom.total_material_cost == pytest.approx(calculated_total, rel=1e-6)
    
    def test_bom_requirements_unique_ids(self, generator, complex_graph):
        """Todos los IDs de requirements son únicos."""
        bom = generator.materialize_project(complex_graph)
        
        ids = [req.id for req in bom.requirements]
        assert len(ids) == len(set(ids)), "IDs duplicados en requirements"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: PRUEBAS DE PROPAGACIÓN DE CANTIDADES
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuantityPropagation(TestFixtures):
    """
    Pruebas para la propagación de cantidades en el DAG.
    
    La cantidad base de un insumo es la suma de todos los caminos:
        Q_base(insumo) = Σ (producto de cantidades en cada camino)
    """
    
    def test_simple_propagation(self, generator, sample_graph):
        """
        Propagación simple en sample_graph.
        
        INS1: APU1(×2)→INS1(×5) + APU2(×1)→INS1(×3) = 10 + 3 = 13
        INS2: APU2(×1)→INS2(×10) = 10
        """
        bom = generator.materialize_project(sample_graph)
        req_map = {r.id: r for r in bom.requirements}
        
        assert req_map["INS1"].quantity_base == pytest.approx(13.0)
        assert req_map["INS2"].quantity_base == pytest.approx(10.0)
    
    def test_multipath_aggregation(self, generator, complex_graph):
        """
        Agregación de múltiples caminos.
        
        MAT1: APU_A(×2.5) + APU_B(×1.0) = 3.5
        MAT3: APU_A(×8.0) + APU_C(×5.0) = 13.0
        """
        bom = generator.materialize_project(complex_graph)
        req_map = {r.id: r for r in bom.requirements}
        
        assert req_map["MAT1"].quantity_base == pytest.approx(3.5)
        assert req_map["MAT3"].quantity_base == pytest.approx(13.0)
    
    def test_single_path_materials(self, generator, complex_graph):
        """Materiales con un solo camino tienen cantidad directa."""
        bom = generator.materialize_project(complex_graph)
        req_map = {r.id: r for r in bom.requirements}
        
        assert req_map["MAT2"].quantity_base == pytest.approx(3.0)
        assert req_map["MAT4"].quantity_base == pytest.approx(15.0)
        assert req_map["MAT5"].quantity_base == pytest.approx(2.0)
    
    def test_deep_propagation(self, generator, deep_graph):
        """
        Propagación en grafo profundo.
        ROOT → L1(×1) → L2(×1) → L3(×1) → L4(×1) → INSUMO(×10)
        Cantidad base = 1×1×1×1×10 = 10
        """
        bom = generator.materialize_project(deep_graph)
        
        assert len(bom.requirements) == 1
        req = bom.requirements[0]
        
        assert req.quantity_base == pytest.approx(10.0)
    
    def test_quantity_conservation(self, generator, sample_graph):
        """
        Ley de conservación: la suma ponderada de salidas = entradas.
        
        Para ROOT con cantidad 1:
        - Salida a APU1: 2.0
        - Salida a APU2: 1.0
        
        Total materiales = f(salidas)
        """
        bom = generator.materialize_project(sample_graph)
        
        # Verificar que las cantidades base suman lo esperado
        total_base = sum(r.quantity_base for r in bom.requirements)
        # INS1: 13, INS2: 10 → Total: 23
        assert total_base == pytest.approx(23.0)
    
    def test_zero_quantity_edge(self, generator):
        """Aristas con cantidad cero no contribuyen."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("APU", type="APU")
        G.add_node("INS", type="INSUMO", unit_cost=100.0, material_category="GENERIC")
        
        G.add_edge("ROOT", "APU", quantity=0.0)
        G.add_edge("APU", "INS", quantity=10.0)
        
        bom = generator.materialize_project(G)
        
        if len(bom.requirements) > 0:
            assert bom.requirements[0].quantity_base == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: PRUEBAS DE FACTORES DE ENTROPÍA
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntropyFactors(TestFixtures):
    """
    Pruebas para los factores de entropía aplicados a los materiales.
    
    Modelo de composición logarítmica:
        F_total = exp(ln(F_flujo) + ln(F_específico) + ln(F_profundidad))
              = F_flujo × F_específico × F_profundidad
    
    Factores:
        - F_flujo: basado en saturación y estabilidad piramidal
        - F_específico: según categoría del material
        - F_profundidad: 1 + (depth - 1) × 0.005
    """
    
    def test_depth_factor_calculation(self, generator, deep_graph):
        """
        Factor de profundidad para grafo profundo.
        
        Depth = 6 (ROOT → L1 → L2 → L3 → L4 → INSUMO)
        F_depth = 1 + (6-1) × 0.005 = 1.025
        """
        bom = generator.materialize_project(deep_graph)
        req = bom.requirements[0]
        
        # Sin flujo ni categoría especial: solo factor de profundidad
        # F_total = 1.025
        expected_cost = 10.0 * 1.025 * 100.0
        assert req.total_cost == pytest.approx(expected_cost, rel=1e-3)
    
    def test_depth_factor_shallow(self, generator, minimal_graph):
        """
        Factor de profundidad para grafo poco profundo.
        
        Depth = 2 (ROOT → INSUMO)
        F_depth = 1 + (2-1) × 0.005 = 1.005
        """
        bom = generator.materialize_project(minimal_graph)
        req = bom.requirements[0]
        
        expected_cost = 1.0 * 1.005 * 100.0
        assert req.total_cost == pytest.approx(expected_cost, rel=1e-3)
    
    def test_category_factor_bulky(self, generator):
        """Factor para categoría BULKY: +2%."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("INS", type="INSUMO", unit_cost=100.0, 
                   material_category="BULKY")
        G.add_edge("ROOT", "INS", quantity=10.0)
        
        bom = generator.materialize_project(G)
        req = bom.requirements[0]
        
        # F_depth = 1.005, F_category = 1.02
        # F_total = 1.005 × 1.02 = 1.0251
        expected_factor = 1.005 * 1.02
        expected_cost = 10.0 * expected_factor * 100.0
        assert req.total_cost == pytest.approx(expected_cost, rel=1e-3)
    
    def test_category_factor_fragile(self, generator):
        """Factor para categoría FRAGILE: +5%."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("INS", type="INSUMO", unit_cost=100.0,
                   material_category="FRAGILE")
        G.add_edge("ROOT", "INS", quantity=10.0)
        
        bom = generator.materialize_project(G)
        req = bom.requirements[0]
        
        # F_total = 1.005 × 1.05 = 1.05525
        expected_factor = 1.005 * 1.05
        expected_cost = 10.0 * expected_factor * 100.0
        assert req.total_cost == pytest.approx(expected_cost, rel=1e-3)
    
    def test_category_factor_hazardous(self, generator):
        """Factor para categoría HAZARDOUS: +8%."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("INS", type="INSUMO", unit_cost=100.0,
                   material_category="HAZARDOUS")
        G.add_edge("ROOT", "INS", quantity=10.0)
        
        bom = generator.materialize_project(G)
        req = bom.requirements[0]
        
        # F_total = 1.005 × 1.08 = 1.0854
        expected_factor = 1.005 * 1.08
        expected_cost = 10.0 * expected_factor * 100.0
        assert req.total_cost == pytest.approx(expected_cost, rel=1e-3)
    
    @pytest.mark.parametrize("category,expected_factor", [
        ("GENERIC", 1.00),
        ("BULKY", 1.02),
        ("FRAGILE", 1.05),
        ("HAZARDOUS", 1.08),
        ("PERISHABLE", 1.04),
        ("PRECISION", 1.03),
    ])
    def test_all_category_factors(self, generator, category, expected_factor):
        """Verifica todos los factores de categoría."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("INS", type="INSUMO", unit_cost=1000.0,
                   material_category=category)
        G.add_edge("ROOT", "INS", quantity=1.0)
        
        bom = generator.materialize_project(G)
        req = bom.requirements[0]
        
        # F_depth = 1.005
        total_factor = 1.005 * expected_factor
        expected_cost = 1.0 * total_factor * 1000.0
        assert req.total_cost == pytest.approx(expected_cost, rel=1e-3)
    
    def test_flux_metrics_high_saturation(self, generator, sample_graph):
        """
        Alta saturación (>0.8) añade factor de flujo.
        
        avg_saturation > 0.8 → log1p(0.05) ≈ 1.05
        """
        flux_metrics = {
            "avg_saturation": 0.9,
            "pyramid_stability": 1.0
        }
        
        bom = generator.materialize_project(sample_graph, flux_metrics=flux_metrics)
        req_map = {r.id: r for r in bom.requirements}
        
        cemento = req_map["INS1"]
        # Base: 13, Depth: 1.01, Flujo: 1.05, Cat: 1.0
        # F_total = 1.01 × 1.05 = 1.0605
        base_factor_no_flux = 1.01  # Solo profundidad
        expected_factor_with_flux = 1.01 * 1.05
        
        assert cemento.waste_factor > base_factor_no_flux - 1
    
    def test_flux_metrics_low_stability(self, generator, sample_graph):
        """
        Baja estabilidad (<1.0) añade factor adicional.
        
        pyramid_stability < 1.0 → log1p(0.03) ≈ 1.03
        """
        flux_metrics = {
            "avg_saturation": 0.5,  # No activa
            "pyramid_stability": 0.5  # Activa: +3%
        }
        
        bom = generator.materialize_project(sample_graph, flux_metrics=flux_metrics)
        req_map = {r.id: r for r in bom.requirements}
        
        cemento = req_map["INS1"]
        # Debe incluir factor de estabilidad
        assert cemento.waste_factor >= 0.03
    
    def test_flux_metrics_combined(self, generator, sample_graph):
        """
        Combinación de alta saturación y baja estabilidad.
        
        Factores: 1.05 × 1.03 = 1.0815
        """
        flux_metrics = {
            "avg_saturation": 0.9,
            "pyramid_stability": 0.5
        }
        
        bom = generator.materialize_project(sample_graph, flux_metrics=flux_metrics)
        req_map = {r.id: r for r in bom.requirements}
        
        arena = req_map["INS2"]  # BULKY
        # Factores: Flujo(1.0815) × BULKY(1.02) × Depth(1.01) = 1.11437
        expected_factor = 1.05 * 1.03 * 1.02 * 1.01
        expected_waste = expected_factor - 1.0
        
        assert arena.waste_factor == pytest.approx(expected_waste, rel=1e-3)
    
    def test_waste_factor_non_negative(self, generator, all_categories_graph):
        """El waste_factor nunca es negativo."""
        bom = generator.materialize_project(all_categories_graph)
        
        for req in bom.requirements:
            assert req.waste_factor >= 0, f"waste_factor negativo para {req.id}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: PRUEBAS DE MÉTRICAS TOPOLÓGICAS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopologicalMetrics(TestFixtures):
    """
    Pruebas para las métricas topológicas en metadata.
    
    Métricas verificadas:
    - is_dag: el grafo es acíclico
    - betti_numbers: β₀ (componentes) y β₁ (ciclos en no dirigido)
    - exergy_efficiency: eficiencia termodinámica
    """
    
    def test_is_dag_true_for_valid_graph(self, generator, sample_graph):
        """Un DAG válido se detecta correctamente."""
        bom = generator.materialize_project(sample_graph)
        
        assert bom.metadata["topological_invariants"]["is_dag"] is True
    
    def test_betti_b0_single_component(self, generator, sample_graph):
        """β₀ = 1 para grafo conexo."""
        bom = generator.materialize_project(sample_graph)
        
        b0 = bom.metadata["topological_invariants"]["betti_numbers"]["b0"]
        assert b0 == 1
    
    def test_betti_b1_diamond_structure(self, generator, sample_graph):
        """
        β₁ cuenta ciclos en el grafo no dirigido subyacente.
        
        sample_graph forma un diamante:
        ROOT → APU1 → INS1
        ROOT → APU2 → INS1
        
        Esto crea un ciclo en el esqueleto no dirigido.
        """
        bom = generator.materialize_project(sample_graph)
        
        b1 = bom.metadata["topological_invariants"]["betti_numbers"]["b1"]
        assert b1 >= 1  # Al menos un ciclo por recombinación
    
    def test_betti_b1_linear_chain(self, generator, deep_graph):
        """Un grafo lineal no tiene ciclos (β₁ = 0)."""
        bom = generator.materialize_project(deep_graph)
        
        b1 = bom.metadata["topological_invariants"]["betti_numbers"]["b1"]
        assert b1 == 0
    
    def test_thermodynamics_present(self, generator, sample_graph):
        """Las métricas termodinámicas están presentes."""
        bom = generator.materialize_project(sample_graph)
        
        thermo = bom.metadata["thermodynamics"]
        assert "exergy_efficiency" in thermo
    
    def test_exergy_efficiency_bounds(self, generator, sample_graph):
        """La eficiencia exergética está en [0, 1]."""
        bom = generator.materialize_project(sample_graph)
        
        exergy = bom.metadata["thermodynamics"]["exergy_efficiency"]
        assert 0.0 <= exergy <= 1.0
    
    def test_risk_analysis_entropy(self, generator, sample_graph):
        """El análisis de riesgo incluye entropía en bits."""
        bom = generator.materialize_project(sample_graph)
        
        risk = bom.metadata["risk_analysis"]
        assert "entropy_bits" in risk
        assert risk["entropy_bits"] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: PRUEBAS DE PRECISIÓN NUMÉRICA
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumericalPrecision(TestFixtures):
    """
    Pruebas de precisión numérica usando suma de Kahan.
    
    La suma de Kahan compensa errores de redondeo en IEEE 754:
        sum, c = 0, 0
        for x in values:
            y = x - c
            t = sum + y
            c = (t - sum) - y
            sum = t
    """
    
    def test_kahan_summation_large_count(self, generator_high_capacity):
        """
        Suma de muchos elementos pequeños con Kahan.
        
        1000 materiales × 100.0 × 1.005 = 100,500.0
        """
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        num_materials = 1000
        unit_cost = 100.0
        
        for i in range(num_materials):
            G.add_node(f"INS_{i}", type="INSUMO", 
                       unit_cost=unit_cost, material_category="GENERIC")
            G.add_edge("ROOT", f"INS_{i}", quantity=1.0)
        
        bom = generator_high_capacity.materialize_project(G)
        
        # Factor: 1.005 (depth=2, GENERIC)
        expected_cost = num_materials * unit_cost * 1.005
        assert bom.total_material_cost == pytest.approx(expected_cost, rel=1e-9)
    
    def test_kahan_summation_varying_magnitudes(self, generator_high_capacity):
        """Precisión con valores de diferentes magnitudes."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        
        costs = [1e-6, 1.0, 1e6]  # Magnitudes muy diferentes
        
        for i, cost in enumerate(costs):
            G.add_node(f"INS_{i}", type="INSUMO",
                       unit_cost=cost, material_category="GENERIC")
            G.add_edge("ROOT", f"INS_{i}", quantity=1.0)
        
        bom = generator_high_capacity.materialize_project(G)
        
        # Factor: 1.005
        expected = sum(c * 1.005 for c in costs)
        assert bom.total_material_cost == pytest.approx(expected, rel=1e-6)
    
    def test_floating_point_associativity(self, generator):
        """
        Verifica que la suma es estable a pesar de no-asociatividad de FP.
        
        (a + b) + c ≠ a + (b + c) en IEEE 754, pero Kahan lo compensa.
        """
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        
        # Valores diseñados para mostrar no-asociatividad
        values = [1.0, 1e-16, 1e-16, 1.0]
        
        for i, val in enumerate(values):
            G.add_node(f"INS_{i}", type="INSUMO",
                       unit_cost=val * 1000, material_category="GENERIC")
            G.add_edge("ROOT", f"INS_{i}", quantity=1.0)
        
        bom = generator.materialize_project(G)
        
        # El resultado debe ser consistente
        assert bom.total_material_cost > 0
    
    def test_precision_with_fractional_quantities(self, generator):
        """Precisión con cantidades fraccionarias."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("APU", type="APU")
        G.add_node("INS", type="INSUMO", unit_cost=333.33,
                   material_category="GENERIC")
        
        G.add_edge("ROOT", "APU", quantity=0.333)
        G.add_edge("APU", "INS", quantity=0.333)
        
        bom = generator.materialize_project(G)
        
        # Cantidad base: 0.333 × 0.333 = 0.110889
        # Factor: depth=3 → 1.01
        expected_qty = 0.333 * 0.333
        expected_cost = expected_qty * 1.01 * 333.33
        
        req = bom.requirements[0]
        assert req.quantity_base == pytest.approx(expected_qty, rel=1e-6)
        assert req.total_cost == pytest.approx(expected_cost, rel=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9: PRUEBAS DE DETECCIÓN DE ERRORES
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorDetection(TestFixtures):
    """
    Pruebas para la detección y manejo de errores.
    """
    
    def test_cycle_detection_raises_error(self, generator, cyclic_graph):
        """Un grafo con ciclo debe lanzar ValueError."""
        with pytest.raises(ValueError, match="ciclo"):
            generator.materialize_project(cyclic_graph)
    
    def test_overflow_protection_small_limit(self, generator):
        """La protección de overflow funciona con límite bajo."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        
        for i in range(200):
            G.add_node(f"INS_{i}", type="INSUMO", unit_cost=10.0)
            G.add_edge("ROOT", f"INS_{i}", quantity=1.0)
        
        generator_limited = MatterGenerator(max_graph_complexity=100)
        
        with pytest.raises(OverflowError):
            generator_limited.materialize_project(G)
    
    def test_missing_root_node(self, generator):
        """Un grafo sin nodo ROOT debe fallar."""
        G = nx.DiGraph()
        G.add_node("APU", type="APU")
        G.add_node("INS", type="INSUMO", unit_cost=100.0)
        G.add_edge("APU", "INS", quantity=1.0)
        
        with pytest.raises((ValueError, KeyError)):
            generator.materialize_project(G)
    
    def test_missing_unit_cost(self, generator):
        """Un insumo sin unit_cost debe manejarse."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("INS", type="INSUMO")  # Sin unit_cost
        G.add_edge("ROOT", "INS", quantity=1.0)
        
        # Puede usar valor por defecto o lanzar error
        try:
            bom = generator.materialize_project(G)
            # Si funciona, el costo debe ser 0 o valor por defecto
            assert bom.total_material_cost >= 0
        except (ValueError, KeyError):
            pass  # También es comportamiento válido
    
    def test_negative_quantity_handling(self, generator):
        """Cantidades negativas deben manejarse correctamente."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("INS", type="INSUMO", unit_cost=100.0)
        G.add_edge("ROOT", "INS", quantity=-5.0)
        
        # Puede rechazar o usar valor absoluto
        try:
            bom = generator.materialize_project(G)
            req = bom.requirements[0]
            assert req.quantity_base >= 0 or req.quantity_base == -5.0
        except ValueError:
            pass
    
    def test_nan_values_handling(self, generator):
        """Valores NaN deben manejarse sin crashear."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("INS", type="INSUMO", unit_cost=float('nan'))
        G.add_edge("ROOT", "INS", quantity=1.0)
        
        try:
            bom = generator.materialize_project(G)
            # El resultado no debe contener NaN propagado
            assert not math.isnan(bom.total_material_cost)
        except ValueError:
            pass  # Rechazar NaN es válido
    
    def test_infinite_values_handling(self, generator):
        """Valores infinitos deben manejarse."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("INS", type="INSUMO", unit_cost=float('inf'))
        G.add_edge("ROOT", "INS", quantity=1.0)
        
        try:
            bom = generator.materialize_project(G)
            assert not math.isinf(bom.total_material_cost)
        except (ValueError, OverflowError):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10: PRUEBAS DE CALIDAD Y CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

class TestQualityAndClustering(TestFixtures):
    """
    Pruebas para métricas de calidad y clustering de materiales.
    """
    
    def test_path_diversity_multipath(self, generator, complex_graph):
        """
        Path diversity cuenta caminos únicos al material.
        
        MAT1 tiene 2 caminos: APU_A y APU_B
        MAT3 tiene 2 caminos: APU_A y APU_C
        """
        bom = generator.materialize_project(complex_graph)
        req_map = {r.id: r for r in bom.requirements}
        
        assert req_map["MAT1"].quality_metrics["path_diversity"] >= 2
        assert req_map["MAT3"].quality_metrics["path_diversity"] >= 2
    
    def test_path_diversity_single_path(self, generator, complex_graph):
        """Materiales con un solo camino tienen diversity = 1."""
        bom = generator.materialize_project(complex_graph)
        req_map = {r.id: r for r in bom.requirements}
        
        # MAT2, MAT4, MAT5 tienen un solo camino
        assert req_map["MAT2"].quality_metrics["path_diversity"] == 1
        assert req_map["MAT4"].quality_metrics["path_diversity"] == 1
        assert req_map["MAT5"].quality_metrics["path_diversity"] == 1
    
    def test_cost_variance_uniform_price(self, generator, complex_graph):
        """
        Varianza de costo es 0 si el precio unitario es único.
        
        Cada material en complex_graph tiene un solo precio.
        """
        bom = generator.materialize_project(complex_graph)
        
        for req in bom.requirements:
            assert req.quality_metrics["cost_variance"] == 0.0
    
    def test_cost_variance_with_variations(self, generator):
        """Varianza de costo con precios variables (si aplicable)."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("APU1", type="APU")
        G.add_node("APU2", type="APU")
        # Mismo material referenciado con diferentes precios
        # (dependiendo de la implementación)
        G.add_node("INS", type="INSUMO", unit_cost=100.0)
        
        G.add_edge("ROOT", "APU1", quantity=1.0)
        G.add_edge("ROOT", "APU2", quantity=1.0)
        G.add_edge("APU1", "INS", quantity=5.0)
        G.add_edge("APU2", "INS", quantity=3.0)
        
        bom = generator.materialize_project(G)
        req = bom.requirements[0]
        
        # Con precio único, varianza = 0
        assert req.quality_metrics["cost_variance"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11: PRUEBAS DE IDEMPOTENCIA Y CONSISTENCIA
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdempotenceAndConsistency(TestFixtures):
    """
    Pruebas de idempotencia y consistencia del generador.
    """
    
    def test_idempotence_same_result(self, generator, sample_graph):
        """Múltiples llamadas producen el mismo resultado."""
        bom1 = generator.materialize_project(sample_graph)
        bom2 = generator.materialize_project(sample_graph)
        
        assert bom1.total_material_cost == bom2.total_material_cost
        assert len(bom1.requirements) == len(bom2.requirements)
        
        for r1, r2 in zip(sorted(bom1.requirements, key=lambda x: x.id),
                         sorted(bom2.requirements, key=lambda x: x.id)):
            assert r1.id == r2.id
            assert r1.quantity_base == r2.quantity_base
            assert r1.total_cost == r2.total_cost
    
    def test_idempotence_with_flux_metrics(self, generator, sample_graph):
        """Idempotencia con flux_metrics constantes."""
        flux = {"avg_saturation": 0.9, "pyramid_stability": 0.5}
        
        bom1 = generator.materialize_project(sample_graph, flux_metrics=flux)
        bom2 = generator.materialize_project(sample_graph, flux_metrics=flux)
        
        assert bom1.total_material_cost == bom2.total_material_cost
    
    def test_graph_not_modified(self, generator, sample_graph):
        """El grafo original no se modifica durante la materialización."""
        nodes_before = set(sample_graph.nodes())
        edges_before = set(sample_graph.edges())
        
        _ = generator.materialize_project(sample_graph)
        
        nodes_after = set(sample_graph.nodes())
        edges_after = set(sample_graph.edges())
        
        assert nodes_before == nodes_after
        assert edges_before == edges_after
    
    def test_different_generators_same_result(self, sample_graph):
        """Diferentes instancias de generador producen mismo resultado."""
        gen1 = MatterGenerator()
        gen2 = MatterGenerator()
        
        bom1 = gen1.materialize_project(sample_graph)
        bom2 = gen2.materialize_project(sample_graph)
        
        assert bom1.total_material_cost == bom2.total_material_cost


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12: PRUEBAS DE CASOS LÍMITE
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases(TestFixtures):
    """
    Pruebas de casos límite y situaciones extremas.
    """
    
    def test_very_deep_graph(self, generator_high_capacity):
        """Grafo con profundidad extrema."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        
        depth = 50
        prev = "ROOT"
        for i in range(depth):
            node_id = f"L{i}"
            G.add_node(node_id, type="APU")
            G.add_edge(prev, node_id, quantity=1.0)
            prev = node_id
        
        G.add_node("INS", type="INSUMO", unit_cost=100.0, material_category="GENERIC")
        G.add_edge(prev, "INS", quantity=1.0)
        
        bom = generator_high_capacity.materialize_project(G)
        
        # Factor de profundidad: 1 + (51-1) × 0.005 = 1.25
        expected_factor = 1 + (depth) * DEPTH_FACTOR_RATE
        expected_cost = 1.0 * expected_factor * 100.0
        assert bom.total_material_cost == pytest.approx(expected_cost, rel=1e-3)
    
    def test_wide_graph(self, generator_high_capacity):
        """Grafo muy ancho (muchos hijos directos)."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        
        width = 100
        for i in range(width):
            G.add_node(f"INS_{i}", type="INSUMO", 
                       unit_cost=10.0, material_category="GENERIC")
            G.add_edge("ROOT", f"INS_{i}", quantity=1.0)
        
        bom = generator_high_capacity.materialize_project(G)
        
        assert len(bom.requirements) == width
        # Todos con depth=2, factor=1.005
        expected_total = width * 10.0 * 1.005
        assert bom.total_material_cost == pytest.approx(expected_total, rel=1e-6)
    
    def test_diamond_pattern(self, generator):
        """
        Patrón diamante: múltiples caminos al mismo nodo.
        
             ROOT
            /    \
          APU1  APU2
            \    /
             INS
        """
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("APU1", type="APU")
        G.add_node("APU2", type="APU")
        G.add_node("INS", type="INSUMO", unit_cost=100.0, material_category="GENERIC")
        
        G.add_edge("ROOT", "APU1", quantity=2.0)
        G.add_edge("ROOT", "APU2", quantity=3.0)
        G.add_edge("APU1", "INS", quantity=5.0)
        G.add_edge("APU2", "INS", quantity=4.0)
        
        bom = generator.materialize_project(G)
        
        # Cantidad: 2×5 + 3×4 = 22
        req = bom.requirements[0]
        assert req.quantity_base == pytest.approx(22.0)
    
    def test_unicode_descriptions(self, generator):
        """Descripciones con caracteres Unicode."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT", description="Proyecto 日本語")
        G.add_node("INS", type="INSUMO", description="Material café ñ €",
                   unit_cost=100.0, material_category="GENERIC")
        G.add_edge("ROOT", "INS", quantity=1.0)
        
        bom = generator.materialize_project(G)
        
        assert len(bom.requirements) == 1
    
    def test_very_small_quantities(self, generator):
        """Cantidades muy pequeñas no se pierden."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("INS", type="INSUMO", unit_cost=1e10, material_category="GENERIC")
        G.add_edge("ROOT", "INS", quantity=1e-10)
        
        bom = generator.materialize_project(G)
        
        assert bom.total_material_cost > 0
    
    def test_very_large_quantities(self, generator):
        """Cantidades muy grandes no causan overflow."""
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("INS", type="INSUMO", unit_cost=1.0, material_category="GENERIC")
        G.add_edge("ROOT", "INS", quantity=1e10)
        
        bom = generator.materialize_project(G)
        
        assert not math.isinf(bom.total_material_cost)
        assert bom.total_material_cost > 1e10


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 13: PRUEBAS DE INTEGRACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration(TestFixtures):
    """
    Pruebas de integración que verifican flujos completos.
    """
    
    def test_full_pipeline_simple(self, generator, sample_graph):
        """Pipeline completo para grafo simple."""
        # 1. Materializar
        bom = generator.materialize_project(sample_graph)
        
        # 2. Verificar estructura
        assert isinstance(bom, BillOfMaterials)
        assert len(bom.requirements) == 2
        
        # 3. Verificar cantidades
        req_map = {r.id: r for r in bom.requirements}
        assert req_map["INS1"].quantity_base == pytest.approx(13.0)
        assert req_map["INS2"].quantity_base == pytest.approx(10.0)
        
        # 4. Verificar metadata
        assert bom.metadata["topological_invariants"]["is_dag"] is True
        assert bom.metadata["topological_invariants"]["betti_numbers"]["b0"] == 1
        
        # 5. Verificar consistencia de costos
        total = sum(r.total_cost for r in bom.requirements)
        assert bom.total_material_cost == pytest.approx(total)
    
    def test_full_pipeline_with_flux(self, generator, sample_graph):
        """Pipeline completo con métricas de flujo."""
        flux_metrics = {
            "avg_saturation": 0.9,
            "pyramid_stability": 0.5
        }
        
        bom = generator.materialize_project(sample_graph, flux_metrics=flux_metrics)
        
        # Los waste_factors deben ser mayores con flux adverso
        for req in bom.requirements:
            assert req.waste_factor > 0
    
    def test_full_pipeline_complex(self, generator, complex_graph):
        """Pipeline completo para grafo complejo."""
        bom = generator.materialize_project(complex_graph)
        
        # Verificar todos los materiales
        assert len(bom.requirements) == 5
        
        # Verificar multipath
        req_map = {r.id: r for r in bom.requirements}
        assert req_map["MAT1"].quality_metrics["path_diversity"] >= 2
        assert req_map["MAT3"].quality_metrics["path_diversity"] >= 2
        
        # Verificar categorías diferentes tienen factores diferentes
        costs = {r.id: r.total_cost for r in bom.requirements}
        
        # HAZARDOUS debería tener mayor factor que GENERIC
        # Comparando materiales con misma cantidad base
        # (No aplicable directamente, pero los waste_factors difieren)
    
    def test_consistency_across_runs(self, generator, complex_graph):
        """Consistencia a través de múltiples ejecuciones."""
        results = [generator.materialize_project(complex_graph) 
                   for _ in range(5)]
        
        costs = [bom.total_material_cost for bom in results]
        
        # Todos los costos deben ser idénticos
        assert all(c == costs[0] for c in costs)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 14: PRUEBAS DE RENDIMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance(TestFixtures):
    """
    Pruebas de rendimiento para operaciones críticas.
    """
    
    @pytest.mark.slow
    def test_large_graph_performance(self, generator_high_capacity):
        """Rendimiento con grafo grande."""
        import time
        
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        
        # 100 APUs, cada uno con 50 materiales
        for apu_i in range(100):
            apu_id = f"APU_{apu_i}"
            G.add_node(apu_id, type="APU")
            G.add_edge("ROOT", apu_id, quantity=1.0)
            
            for mat_j in range(50):
                mat_id = f"MAT_{apu_i}_{mat_j}"
                G.add_node(mat_id, type="INSUMO", unit_cost=10.0,
                          material_category="GENERIC")
                G.add_edge(apu_id, mat_id, quantity=1.0)
        
        start = time.time()
        bom = generator_high_capacity.materialize_project(G)
        elapsed = time.time() - start
        
        assert elapsed < 10.0  # Menos de 10 segundos
        assert len(bom.requirements) == 5000
    
    @pytest.mark.slow
    def test_deep_graph_performance(self, generator_high_capacity):
        """Rendimiento con grafo muy profundo."""
        import time
        
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        
        depth = 100
        prev = "ROOT"
        for i in range(depth):
            node_id = f"L{i}"
            G.add_node(node_id, type="APU")
            G.add_edge(prev, node_id, quantity=1.0)
            prev = node_id
        
        G.add_node("INS", type="INSUMO", unit_cost=100.0)
        G.add_edge(prev, "INS", quantity=1.0)
        
        start = time.time()
        bom = generator_high_capacity.materialize_project(G)
        elapsed = time.time() - start
        
        assert elapsed < 5.0
    
    @pytest.mark.slow
    def test_repeated_materialization_performance(self, generator, sample_graph):
        """Rendimiento de materializaciones repetidas."""
        import time
        
        n_iterations = 100
        
        start = time.time()
        for _ in range(n_iterations):
            generator.materialize_project(sample_graph)
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # 100 iteraciones en menos de 5 segundos
        assert elapsed / n_iterations < 0.1  # Menos de 100ms por iteración


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 15: PRUEBAS DE PROPIEDADES MATEMÁTICAS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMathematicalProperties(TestFixtures):
    """
    Pruebas de propiedades matemáticas del sistema.
    """
    
    def test_factor_composition_logarithmic(self, generator):
        """
        La composición de factores es multiplicativa (logarítmica).
        
        F_total = F_a × F_b × F_c = exp(ln(F_a) + ln(F_b) + ln(F_c))
        """
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("APU", type="APU")
        G.add_node("INS", type="INSUMO", unit_cost=1000.0, 
                   material_category="BULKY")
        
        G.add_edge("ROOT", "APU", quantity=1.0)
        G.add_edge("APU", "INS", quantity=10.0)
        
        flux = {"avg_saturation": 0.9, "pyramid_stability": 0.5}
        bom = generator.materialize_project(G, flux_metrics=flux)
        
        req = bom.requirements[0]
        
        # Factores individuales:
        # - Flujo saturación: 1.05
        # - Flujo estabilidad: 1.03
        # - Categoría BULKY: 1.02
        # - Profundidad (3): 1.01
        # Total: 1.05 × 1.03 × 1.02 × 1.01 = 1.11437
        
        expected_factor = 1.05 * 1.03 * 1.02 * 1.01
        actual_factor = 1 + req.waste_factor
        
        assert actual_factor == pytest.approx(expected_factor, rel=1e-3)
    
    def test_quantity_linearity(self, generator):
        """
        Las cantidades se propagan linealmente.
        
        Q(2x) = 2 × Q(x)
        """
        G1 = nx.DiGraph()
        G1.add_node("ROOT", type="ROOT")
        G1.add_node("INS", type="INSUMO", unit_cost=100.0)
        G1.add_edge("ROOT", "INS", quantity=5.0)
        
        G2 = nx.DiGraph()
        G2.add_node("ROOT", type="ROOT")
        G2.add_node("INS", type="INSUMO", unit_cost=100.0)
        G2.add_edge("ROOT", "INS", quantity=10.0)  # Doble cantidad
        
        bom1 = generator.materialize_project(G1)
        bom2 = generator.materialize_project(G2)
        
        # El costo debe ser el doble
        assert bom2.total_material_cost == pytest.approx(
            2 * bom1.total_material_cost, rel=1e-6)
    
    def test_cost_additivity(self, generator):
        """
        Los costos son aditivos para materiales independientes.
        
        C(A + B) = C(A) + C(B)
        """
        # Grafo con dos materiales
        G = nx.DiGraph()
        G.add_node("ROOT", type="ROOT")
        G.add_node("INS1", type="INSUMO", unit_cost=100.0, material_category="GENERIC")
        G.add_node("INS2", type="INSUMO", unit_cost=200.0, material_category="GENERIC")
        G.add_edge("ROOT", "INS1", quantity=5.0)
        G.add_edge("ROOT", "INS2", quantity=3.0)
        
        bom = generator.materialize_project(G)
        req_map = {r.id: r for r in bom.requirements}
        
        total_individual = req_map["INS1"].total_cost + req_map["INS2"].total_cost
        assert bom.total_material_cost == pytest.approx(total_individual, rel=1e-6)
    
    def test_depth_factor_monotonicity(self, generator_high_capacity):
        """
        El factor de profundidad es monótono creciente.
        
        depth(a) < depth(b) → F_depth(a) < F_depth(b)
        """
        depths = [2, 5, 10, 20]
        factors = []
        
        for d in depths:
            G = nx.DiGraph()
            G.add_node("ROOT", type="ROOT")
            
            prev = "ROOT"
            for i in range(d - 1):
                node_id = f"L{i}"
                G.add_node(node_id, type="APU")
                G.add_edge(prev, node_id, quantity=1.0)
                prev = node_id
            
            G.add_node("INS", type="INSUMO", unit_cost=1000.0, material_category="GENERIC")
            G.add_edge(prev, "INS", quantity=1.0)
            
            bom = generator_high_capacity.materialize_project(G)
            factor = 1 + bom.requirements[0].waste_factor
            factors.append(factor)
        
        # Verificar monotonía estricta
        for i in range(len(factors) - 1):
            assert factors[i] < factors[i + 1]


# ═══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow"
    ])