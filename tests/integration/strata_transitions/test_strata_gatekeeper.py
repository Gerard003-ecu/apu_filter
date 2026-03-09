"""
Integration Tests para el Gatekeeper Categórico de la MIC
=========================================================

Demuestra la Clausura Transitiva en las transiciones de estrato.
Específicamente, prueba que un salto de PHYSICS directamente a STRATEGY
falla si no se valida TACTICS intermedio.
"""

from typing import Any, Dict

import pytest

from app.schemas import Stratum
from app.tools_interface import MICRegistry


def mock_strategy_vector(**kwargs: Any) -> Dict[str, Any]:
    """Mock vector for STRATEGY."""
    return {"success": True, "result": "strategy completed", **kwargs}


class TestStrataGatekeeper:
    """Test suite for Transitive Closure in MIC Stratum projection."""

    def test_physics_to_strategy_transitive_closure_violation(self) -> None:
        """
        Prueba que invocar un vector de STRATEGY con solo PHYSICS validado
        es bloqueado por el Gatekeeper Categórico.
        """
        mic = MICRegistry()

        # Registrar un vector en STRATEGY
        mic.register_vector(
            service_name="strategy_service",
            stratum=Stratum.STRATEGY,
            handler=mock_strategy_vector,
        )

        # Contexto que SOLO tiene validado PHYSICS, omitiendo TACTICS
        context_only_physics = {
            "validated_strata": {Stratum.PHYSICS}
        }

        # Intentar ejecutar el servicio de STRATEGY
        result = mic.project_intent(
            service_name="strategy_service",
            payload={"some_arg": 1},
            context=context_only_physics,
            use_cache=False,
        )

        # El Gatekeeper debe interceptar el request
        assert result.get("success") is False
        assert result.get("error_category") == "hierarchy_violation"

        details = result.get("error_details", {})
        assert details.get("target_stratum") == "STRATEGY"

        # STRATEGY (1) requires TACTICS (2) and PHYSICS (3).
        # Since PHYSICS is validated, only TACTICS is missing.
        missing = details.get("missing_strata", [])
        assert "TACTICS" in missing
        assert "PHYSICS" not in missing
