import pytest
from app.semantic_dictionary import (
    SemanticDictionaryService,
    PyramidalSemanticVector,
    GraphSemanticProjector,
    Stratum
)
from app.tools_interface import MICRegistry

def test_semantic_dictionary_registration():
    mic = MICRegistry()
    service = SemanticDictionaryService()
    service.register_in_mic(mic)
    assert mic.is_registered("fetch_narrative")
    assert mic.is_registered("project_graph_narrative")
    assert mic.get_vector_stratum("fetch_narrative") == Stratum.WISDOM
    assert mic.get_vector_stratum("project_graph_narrative") == Stratum.WISDOM

def test_fetch_narrative_basic():
    service = SemanticDictionaryService()

    # Test existing template
    payload = {
        "domain": "TOPOLOGY_CYCLES",
        "classification": "clean",
        "params": {"beta_1": 0}
    }
    result = service.fetch_narrative(**payload)
    assert result["success"] is True
    assert "✅" in result["narrative"]
    assert "Integridad Estructural" in result["narrative"]

def test_project_graph_narrative_cycle():
    service = SemanticDictionaryService()
    context = {}

    payload = {
        "anomaly_type": "CYCLE",
        "path_nodes": ["A", "B", "C", "A"]
    }

    result = service.project_graph_narrative(payload, context)
    assert result["success"] is True
    narrative = result["narrative"]
    assert "Ruta del Ciclo Detectada" in narrative
    assert "A -> B -> C -> A" in narrative

def test_project_graph_narrative_stress():
    service = SemanticDictionaryService()
    context = {}

    # Critical stress in PHYSICS (Base) with high degree
    vector_data = {
        "node_id": "Cement",
        "node_type": "INSUMO",
        "stratum": Stratum.PHYSICS,
        "in_degree": 6,
        "out_degree": 2,
        "is_critical_bridge": True
    }

    payload = {
        "anomaly_type": "STRESS",
        "vector": vector_data
    }

    result = service.project_graph_narrative(payload, context)
    assert result["success"] is True
    narrative = result["narrative"]
    assert "Punto de Estrés Estructural" in narrative
    assert "Cement" in narrative
    assert "6" in narrative

def test_project_graph_narrative_stress_non_critical():
    service = SemanticDictionaryService()
    context = {}

    # Low degree stress
    vector_data = {
        "node_id": "Brick",
        "node_type": "INSUMO",
        "stratum": Stratum.PHYSICS,
        "in_degree": 2,
        "out_degree": 1,
        "is_critical_bridge": False
    }

    payload = {
        "anomaly_type": "STRESS",
        "vector": vector_data
    }

    result = service.project_graph_narrative(payload, context)
    assert result["success"] is False
    assert "no presenta estrés" in result["error"]

def test_project_graph_narrative_invalid_type():
    service = SemanticDictionaryService()
    context = {}

    payload = {
        "anomaly_type": "UNKNOWN_TYPE"
    }

    result = service.project_graph_narrative(payload, context)
    assert result["success"] is False
    assert "no soportada" in result["error"]

def test_market_context_deterministic():
    service = SemanticDictionaryService()

    payload = {
        "domain": "MARKET_CONTEXT",
        "params": {"deterministic": True, "index": 0}
    }
    result1 = service.fetch_narrative(**payload)
    result2 = service.fetch_narrative(**payload)

    assert result1["narrative"] == result2["narrative"]
    assert "Suelo Estable" in result1["narrative"]
