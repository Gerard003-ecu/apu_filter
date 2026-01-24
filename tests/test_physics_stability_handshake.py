def test_condenser_aborts_on_unstable_oracle_verdict():
    """
    Verifica que el FluxCondenser consulte al Oráculo de Laplace y aborte
    si la configuración física es matemáticamente inestable.
    Ref: flux_condenser.txt (Fase 1: Inicialización)
    """
    # 1. Configuración Físicamente Imposible (Resistencia Negativa o Inductancia inestable)
    # Esto debería generar polos inestables en el análisis de Laplace.
    unsafe_config = {
        "system_capacitance": 5000.0,
        "base_resistance": -10.0, # ¡Resistencia negativa genera energía infinita!
        "system_inductance": 2.0,
        "pid_kp": 2000.0
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
    assert "inestable" in str(excinfo.value).lower()