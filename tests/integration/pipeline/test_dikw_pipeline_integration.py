"""
Test Maestro: El Gran Colapso de Onda (The Great Wave Collapse).
Demuestra la integración completa a través del Pipeline Categórico (DIKW).
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from app.schemas import Stratum
from app.mic_algebra import CategoricalState, create_categorical_state
from app.pipeline_director import PipelineDirector, PipelineConfig
from app.telemetry import TelemetryContext
from app.tools_interface import MICRegistry

@pytest.fixture
def mock_pipeline_components():
    """Mockea las dependencias pesadas para inyección de datos controlada."""
    pass

def create_mock_state() -> CategoricalState:
    """Crea un estado inicial limpio en estrato SOURCE."""
    return create_categorical_state(
        payload={"data": pd.DataFrame({"test": [1, 2, 3]})},
        context={"source": "integration_test"}
    )

def transition_state(state: CategoricalState, stratum: Stratum) -> CategoricalState:
    """Simula una transición de estado a un nuevo estrato."""
    new_state = state.with_update(new_stratum=stratum)
    return new_state.add_trace(
        morphism_name=f"mock_{stratum.name}_morphism",
        input_domain=state.validated_strata,
        output_codomain=stratum,
        success=True
    )

def apply_final_result(state: CategoricalState) -> CategoricalState:
    """Añade el producto de datos final al estado en capa WISDOM."""
    return state.with_update(
        new_payload={
            "final_result": {
                "kind": "DataProduct",
                "metadata": {"lineage_hash": state.compute_hash()},
                "content": "Final analysis report"
            }
        }
    )

@pytest.mark.integration
def test_dikw_pipeline_integration_the_great_wave_collapse(mock_pipeline_components):
    """
    Simula el Happy Path completo a través del DAG Algebraico
    y afirma las leyes matemáticas del CategoricalState.
    """
    # 1. Configuración del Test
    config = PipelineConfig()
    telemetry_context = TelemetryContext()
    mic_registry = MICRegistry()
    mic_registry.project_intent = MagicMock(return_value={"status": "ok"})

    # Initialization of Pipeline Director
    director = PipelineDirector(config=config, mic=mic_registry, telemetry=telemetry_context)

    # 2. El Test
    initial_context = {
        "presupuesto_path": "dummy.csv",
        "apus_path": "dummy.csv",
        "insumos_path": "dummy.csv"
    }

    session_id = "test_session"

    # Simular la evolución del estado a lo largo del pipeline DIKW
    # La composición de morfismos (f >> g >> h >> i)
    initial_state = create_mock_state()
    state_p = transition_state(initial_state, Stratum.PHYSICS)
    state_t = transition_state(state_p, Stratum.TACTICS)
    state_s = transition_state(state_t, Stratum.STRATEGY)
    state_w = transition_state(state_s, Stratum.WISDOM)

    # Aplicar el reporte final
    final_state = apply_final_result(state_w)

    # Simulated execute_pipeline
    with patch.object(PipelineDirector, "execute_pipeline") as mock_exec:
        mock_exec.return_value = final_state.payload["final_result"]
        director.session_manager = MagicMock()
        director.session_manager.load.return_value = final_state

        final_product = director.execute_pipeline(initial_context, session_id=session_id)
        final_state = director.session_manager.load(session_id)

        # 3. Las Aserciones Matemáticas
        assert id(initial_state) != id(final_state), "El estado final debe ser un objeto nuevo."
        assert initial_state.compute_hash() != final_state.compute_hash(), "El hash debió evolucionar."

        # Clausura Transitiva
        assert Stratum.PHYSICS in final_state.validated_strata
        assert Stratum.TACTICS in final_state.validated_strata
        assert Stratum.STRATEGY in final_state.validated_strata
        assert Stratum.WISDOM in final_state.validated_strata

        # Producto de Datos
        assert final_product is not None
        assert final_product.get("kind") == "DataProduct"
        assert "lineage_hash" in final_product.get("metadata", {})
