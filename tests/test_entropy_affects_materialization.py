def test_high_entropy_increases_waste_factors():
    """
    Verifica que la turbulencia en el flujo de datos (Entropía alta)
    se propague al generador de materia como un factor de riesgo.
    Ref: matter_generator.txt, business_agent.txt
    """
    # 1. Simular métricas físicas de alta entropía (Caos administrativo)
    high_entropy_metrics = {
        "entropy_ratio": 0.95, # Caos casi total
        "dissipated_power": 45.0 # Alta fricción
    }
    
    # 2. Contexto del Pipeline
    context = {
        "flux_metrics": high_entropy_metrics,
        "graph": nx.DiGraph(), # Grafo simulado
        # ... datos del grafo
    }

    # 3. Ejecutar Materialización
    generator = MatterGenerator(config={}, thresholds=...)
    # Asumimos que _generate_enriched_metadata usa flux_metrics
    result = generator._generate_enriched_metadata(..., flux_metrics=high_entropy_metrics, ...)

    # 4. Validar impacto en el negocio
    # El sistema debe advertir sobre la calidad de la estimación
    assert result["quality_metrics"]["data_confidence"] < 0.5
    assert "Alta Entropía" in result["warnings"]