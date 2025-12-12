import pytest
import pandas as pd
import numpy as np
import os
from app.classifiers.apu_classifier import APUClassifier

CONFIG_PATH = "config/config_rules.json"

def test_classifier_with_various_scenarios():
    """Test del clasificador con múltiples escenarios"""
    # Usar el archivo de configuración real para tener las reglas completas (incluyendo SUMINISTRO_PREFABRICADO)
    if os.path.exists(CONFIG_PATH):
        classifier = APUClassifier(CONFIG_PATH)
    else:
        pytest.skip(f"Archivo de configuración no encontrado: {CONFIG_PATH}")

    # Casos de prueba
    test_cases = [
        # (pct_mat, pct_mo_eq, expected_type, description, total_cost)
        (0.70, 0.10, "SUMINISTRO", "Predominio material puro", 100.0),
        (0.65, 0.25, "SUMINISTRO_PREFABRICADO", "Material con MO moderada", 100.0),
        (0.30, 0.65, "INSTALACION", "Predominio MO/equipo", 100.0),
        (0.50, 0.45, "CONSTRUCCION_MIXTO", "Balance 50/50", 100.0),
        (0.55, 0.40, "CONSTRUCCION_MIXTO", "Límite superior mixto", 100.0),
        # Mat=45%, MO=35% -> Matches CONSTRUCCION_MIXTO (40-60 range)
        (0.45, 0.35, "CONSTRUCCION_MIXTO", "Caso intermedio (Mixto)", 100.0),
        # Case with 0 cost should return SIN_COSTO regardless of percentages (which would be 0/0=Nan or handled)
        (0.00, 0.00, "SIN_COSTO", "Sin costo", 0.0),
        (0.90, 0.05, "SUMINISTRO", "Material muy alto, MO baja", 100.0),
    ]

    for pct_mat, pct_mo_eq, expected, desc, total_cost in test_cases:
        result = classifier.classify_single(pct_mat, pct_mo_eq, total_cost=total_cost)
        assert result == expected, f"Fallo en {desc}: esperado {expected}, obtenido {result}"

def test_dataframe_classification():
    """Test de clasificación de DataFrame completo"""
    # Aquí podemos usar defaults o config, usaremos config por consistencia
    if os.path.exists(CONFIG_PATH):
        classifier = APUClassifier(CONFIG_PATH)
    else:
        classifier = APUClassifier()

    # Crear DataFrame de prueba
    df = pd.DataFrame({
        'VALOR_CONSTRUCCION_UN': [100, 200, 150, 300, 0],
        'VALOR_SUMINISTRO_UN': [70, 120, 45, 180, 0],
        'VALOR_INSTALACION_UN': [30, 80, 105, 120, 0]
    })

    df_classified = classifier.classify_dataframe(df)

    # Verificar que todos tienen clasificación
    assert 'TIPO_APU' in df_classified.columns
    assert df_classified['TIPO_APU'].isna().sum() == 0

    # Verificar el APU sin costo
    assert df_classified.iloc[4]['TIPO_APU'] == 'SIN_COSTO'

def test_config_loading():
    """Test de carga de configuración desde JSON"""
    import tempfile
    import json

    # Crear archivo de configuración temporal
    config = {
        "apu_classification_rules": {
            "rules": [
                {
                    "type": "TEST_TYPE",
                    "priority": 1,
                    "condition": "porcentaje_materiales >= 50.0",
                    "description": "Test rule"
                }
            ],
            "default_type": "TEST_DEFAULT"
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        classifier = APUClassifier(config_path)
        assert len(classifier.rules) == 1
        assert classifier.rules[0].rule_type == "TEST_TYPE"
        assert classifier.default_type == "TEST_DEFAULT"
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
