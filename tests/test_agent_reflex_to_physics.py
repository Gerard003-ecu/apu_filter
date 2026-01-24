def test_agent_reacts_to_flyback_spike():
    """
    Simula un pico de voltaje inductivo (Flyback) en el Condensador
    y verifica que el Agente cambie su estado y decisión.
    Ref: apu_agent.txt (Decide phase)
    """
    # 1. Setup: Agente y Telemetría simulada
    agent = AutonomousAgent()
    
    # Telemetría normal
    normal_telemetry = TelemetryData(
        flyback_voltage=0.1, 
        saturation=0.3,
        # ... otros campos
    )
    
    # Telemetría de crisis (Flyback alto por datos corruptos)
    crisis_telemetry = TelemetryData(
        flyback_voltage=9.5, # > CRITICAL_THRESHOLD (0.8)
        saturation=0.4
    )

    # 2. Ejecución: Fase Orient (Orientación)
    # Primero normal
    status_normal = agent.orient(normal_telemetry)
    assert status_normal == SystemStatus.NOMINAL

    # Luego crisis
    status_crisis = agent.orient(crisis_telemetry)
    
    # 3. Validación: El Agente debe entrar en pánico controlado
    assert status_crisis == SystemStatus.CRITICO
    
    # 4. Validación: Fase Decide (Decisión)
    decision = agent.decide(status_crisis)
    assert decision == AgentDecision.ALERTA_CRITICA # O EJECUTAR_LIMPIEZA