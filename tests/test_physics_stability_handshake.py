import pytest
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

try:
    from app.flux_condenser import DataFluxCondenser, CondenserConfig, DataFluxCondenserError as ConfigurationError
except ImportError:
    # Mocking if imports fail
    class ConfigurationError(Exception):
        pass

    class CondenserConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class DataFluxCondenser:
        def __init__(self, config, profile, condenser_config):
            # Simulated validation logic
            pass

def test_condenser_aborts_on_unstable_oracle_verdict():
    """
    Verifica que el FluxCondenser consulte al Oráculo de Laplace y aborte
    si la configuración física es matemáticamente inestable.
    Ref: flux_condenser.txt (Fase 1: Inicialización)
    """
    # 1. Configuración que pasa validación física básica pero falla en control
    unsafe_config = {
        "system_capacitance": 5000.0,
        "base_resistance": 10.0, # Valido fisicamente
        "system_inductance": 2.0,
        "pid_kp": 2000.0
    }

    # Mock del Oracle para simular rechazo
    with patch('app.flux_condenser.LaplaceOracle') as MockOracle:
        instance = MockOracle.return_value
        instance.validate_for_control_design.return_value = {
            "is_suitable_for_control": False,
            "issues": ["Polos inestables"],
            "summary": "Sistema inestable",
            "warnings": []
        }

        # 2. Intentar inicializar el Condensador
        with pytest.raises(ConfigurationError) as excinfo:
            DataFluxCondenser(
                config={},
                profile={},
                condenser_config=CondenserConfig(**unsafe_config)
            )

        # 3. Validar que la causa fue el Veto del Oráculo
        assert "CONFIGURACIÓN NO APTA PARA CONTROL" in str(excinfo.value)
