"""
Suite de Pruebas de MIC Agent (Grothendieck Topos)
"""

import pytest
from unittest.mock import MagicMock
from app.agents.MIC_agent import MICAgent, SchemaValidationResult
from app.core.mic_algebra import CategoricalState, Stratum

class TestGrothendieckToposAgent:
    """Verifica que el Agente actúa como un Morfismo Geométrico."""

    @pytest.fixture
    def mock_registry(self):
        registry = MagicMock()
        registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        registry.project_intent.return_value = {"status": "OK", "result": {"p": 1}}
        return registry

    @pytest.fixture
    def agent(self, mock_registry):
        return MICAgent(mic_registry=mock_registry)

    def test_characteristic_morphism_pullback(self, agent):
        """Verifica que chi_S evalúa correctamente un CategoricalState."""
        payload_S = {"dissipated_power": 10.0}
        state = CategoricalState(
            payload=payload_S,
            validated_strata=frozenset({Stratum.PHYSICS})
        )
        
        chi_S = agent.characteristic_morphism(state)
        
        assert isinstance(chi_S, SchemaValidationResult)
        assert chi_S.validity_degree == 1.0

    def test_geometric_morphism_adjunction(self, agent):
        """Verifica la relación de adjunción via funtores imagen."""
        intent_X = {"dissipated_power": 5.0, "action": "project"}
        f_star_X = agent.f_star_inverse_image(intent_X, "topology_core", frozenset())
        assert isinstance(f_star_X, CategoricalState)
        assert f_star_X.is_success
        
        f_lower_star_f_star_X = agent.f_lower_star_direct_image(f_star_X)
        assert f_lower_star_f_star_X["verdict"] == "ACCEPTED"
        assert f_lower_star_f_star_X["payload"]["dissipated_power"] == intent_X["dissipated_power"]

    def test_f_star_preserves_finite_limits_initial(self, agent):
        """Verifica el colapso determinista ante el objeto inicial (None)."""
        state = agent.f_star_inverse_image(None, "test_vector", frozenset())
        assert state.is_failed
        assert "Colapso" in state.error

    def test_f_star_preserves_empty_product_fibrado(self, agent):
        """Verifica el colapso ante un producto fibrado vacío (dict vacío)."""
        state = agent.f_star_inverse_image({}, "test_vector", frozenset())
        # Si el dict vacío es aceptado por el validador pero resulta en payload vacío -> Colapso
        # O si el validador lo rechaza, f_star lo reporta.
        assert state.is_failed
        assert ("Colapso" in state.error) or ("SCHEMA_VALIDATION_ERROR" in state.error)
