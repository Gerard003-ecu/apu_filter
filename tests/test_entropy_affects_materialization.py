import pytest
import networkx as nx
from unittest.mock import MagicMock

try:
    from app.matter_generator import MatterGenerator
except ImportError:
    # Fallback mock not needed if environment is correct, but kept for safety
    pass

def test_high_entropy_increases_waste_factors():
    """
    Verifica que la turbulencia en el flujo de datos (Entropía alta)
    se propague al generador de materia como un factor de riesgo
    (aumentando el desperdicio).
    Ref: matter_generator.txt, business_agent.txt
    """
    # 1. Simular métricas físicas de alta entropía (Caos administrativo)
    high_entropy_metrics = {
        "entropy_ratio": 0.95,
        "avg_saturation": 0.9, # High saturation increases entropy
        "pyramid_stability": 0.5 # Low stability increases entropy
    }
    
    # 2. Contexto del Pipeline (Grafo con materiales)
    graph = nx.DiGraph()
    graph.add_node("ROOT", type="APU")
    graph.add_node("MAT1", type="INSUMO", unit_cost=100.0, unit="UND", material_category="FRAGILE")
    graph.add_edge("ROOT", "MAT1", quantity=10.0)

    context = {
        "flux_metrics": high_entropy_metrics,
        "graph": graph,
    }

    # 3. Ejecutar Materialización
    generator = MatterGenerator()

    result_bom = generator.materialize_project(
        graph=context["graph"],
        flux_metrics=high_entropy_metrics
    )

    # 4. Validar impacto en el negocio
    # El sistema debe haber aplicado factores de desperdicio significativos
    waste_stats = result_bom.metadata["material_distribution"]["waste"]

    # Base waste for FRAGILE is ~5%.
    # High saturation (0.9) adds log1p(0.05) ~ 5%.
    # Low stability adds log1p(0.03) ~ 3%.
    # Total should be significantly > 0.

    assert waste_stats["mean"] > 0.05, f"Waste factor too low: {waste_stats['mean']}"

    # Validar que las métricas de flujo se pasaron
    assert result_bom.metadata["risk_analysis"]["flux_metrics"]["entropy_ratio"] == 0.95
