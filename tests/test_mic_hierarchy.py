"""
Tests for MIC Hierarchy and Gatekeeper Logic.

Verifies that the Matriz de Interacción Central (MIC) correctly enforces
the "Physics before Strategy" rule.
"""

import pytest
from app.schemas import Stratum
from app.tools_interface import MICRegistry, IntentVector

# Mock Handlers
def physics_handler(val: int):
    return {"success": True, "val": val}

def strategy_handler(amount: float):
    return {"success": True, "amount": amount}

class TestMICHierarchy:

    @pytest.fixture
    def mic(self):
        registry = MICRegistry()
        registry.register_vector("mock_physics", Stratum.PHYSICS, physics_handler)
        registry.register_vector("mock_strategy", Stratum.STRATEGY, strategy_handler)
        return registry

    def test_physics_execution_allowed_always(self, mic):
        """Physics vectors should always be executable (Level 3)."""
        context = {} # Empty context
        payload = {"val": 10}

        result = mic.project_intent("mock_physics", payload, context)

        assert result["success"] is True
        assert result["val"] == 10
        assert result["_mic_validation_update"] == Stratum.PHYSICS

    def test_strategy_execution_blocked_without_physics(self, mic):
        """Strategy vectors (Level 1) should be blocked if Physics is not validated."""
        context = {"validated_strata": set()}
        payload = {"amount": 100.0}

        result = mic.project_intent("mock_strategy", payload, context)

        assert result["success"] is False
        assert result["error_type"] == "PermissionError"
        assert "Violación de Jerarquía MIC" in result["error"]

    def test_strategy_execution_allowed_with_physics(self, mic):
        """Strategy vectors should be allowed if Physics is in validated_strata."""
        # Simulate that physics has been validated
        context = {"validated_strata": {Stratum.PHYSICS}}
        payload = {"amount": 100.0}

        result = mic.project_intent("mock_strategy", payload, context)

        assert result["success"] is True
        assert result["amount"] == 100.0

    def test_strategy_execution_allowed_with_override(self, mic):
        """Strategy vectors should be allowed with force_physics_override."""
        context = {"validated_strata": set(), "force_physics_override": True}
        payload = {"amount": 100.0}

        result = mic.project_intent("mock_strategy", payload, context)

        assert result["success"] is True

    def test_unknown_vector_raises_error(self, mic):
        """Projecting to an unknown vector should raise ValueError."""
        with pytest.raises(ValueError):
            mic.project_intent("unknown_vector", {}, {})
