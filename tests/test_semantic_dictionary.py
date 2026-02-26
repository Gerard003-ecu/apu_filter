import pytest
from app.semantic_dictionary import SemanticDictionaryService
from app.tools_interface import MICRegistry
from app.schemas import Stratum

def test_semantic_dictionary_registration():
    mic = MICRegistry()
    service = SemanticDictionaryService()
    service.register_in_mic(mic)
    assert mic.is_registered("fetch_narrative")
    assert mic.get_vector_stratum("fetch_narrative") == Stratum.WISDOM

def test_fetch_narrative_basic():
    service = SemanticDictionaryService()

    # Test existing template
    payload = {
        "domain": "TOPOLOGY_CYCLES",
        "classification": "clean",
        "params": {"beta_1": 0}
    }
    # Calling as MICRegistry would: unpacked arguments
    result = service.fetch_narrative(**payload)
    assert result["success"] is True
    assert "✅" in result["narrative"]
    assert "Integridad Estructural" in result["narrative"]

def test_fetch_narrative_with_params():
    service = SemanticDictionaryService()

    payload = {
        "domain": "STABILITY",
        "classification": "critical",
        "params": {"stability": 0.5}
    }
    result = service.fetch_narrative(**payload)
    assert result["success"] is True
    assert "0.50" in result["narrative"]
    assert "COLAPSO POR BASE ESTRECHA" in result["narrative"]

def test_fetch_narrative_misc_string():
    service = SemanticDictionaryService()

    payload = {
        "domain": "MISC",
        "classification": "THERMAL_DEATH"
    }
    result = service.fetch_narrative(**payload)
    assert result["success"] is True
    assert "MUERTE TÉRMICA" in result["narrative"]

def test_fetch_narrative_invalid_domain():
    service = SemanticDictionaryService()

    payload = {
        "domain": "NON_EXISTENT",
        "classification": "clean"
    }
    result = service.fetch_narrative(**payload)
    assert result["success"] is False
    assert "not found" in result["error"]

def test_fetch_narrative_invalid_classification():
    service = SemanticDictionaryService()

    payload = {
        "domain": "TOPOLOGY_CYCLES",
        "classification": "INVALID_KEY"
    }
    result = service.fetch_narrative(**payload)
    assert result["success"] is True
    assert "⚠️ Estado desconocido" in result["narrative"]

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
