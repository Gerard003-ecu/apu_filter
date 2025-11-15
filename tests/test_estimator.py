import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from flask import current_app

# Añadir el directorio raíz del proyecto al sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.app import create_app
from app.estimator import (
    _find_best_semantic_match,
    calculate_estimate,
)

# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def sample_apu_pool():
    """Crea un DataFrame de pandas con APUs de prueba."""
    return pd.DataFrame(
        [
            {
                "CODIGO_APU": "APU-001",
                "original_description": "Instalación de muro de ladrillo",
                "DESC_NORMALIZED": "instalacion muro ladrillo",
                "VALOR_SUMINISTRO_UN": 100,
                "VALOR_CONSTRUCCION_UN": 150,
                "tipo_apu": "Suministro",
                "EQUIPO": 10,
                "UNIDAD": "M2",
            },
            {
                "CODIGO_APU": "APU-002",
                "original_description": "Pintura acrílica para exteriores",
                "DESC_NORMALIZED": "pintura acrilica exteriores",
                "VALOR_SUMINISTRO_UN": 200,
                "VALOR_CONSTRUCCION_UN": 250,
                "tipo_apu": "Suministro",
                "EQUIPO": 20,
                "UNIDAD": "M2",
            },
            {
                "CODIGO_APU": "APU-003",
                "original_description": "CUADRILLA TIPO 1 (1 OF + 2 AYU)",
                "DESC_NORMALIZED": "cuadrilla tipo 1 1 of 2 ayu",
                "VALOR_SUMINISTRO_UN": 0,
                "VALOR_CONSTRUCCION_UN": 300,
                "tipo_apu": "Instalación",
                "EQUIPO": 30,
                "UNIDAD": "DIA",
            },
        ]
    )


@pytest.fixture
def mock_embedding_model():
    """Mock del modelo SentenceTransformer."""
    model = MagicMock()
    model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
    return model


@pytest.fixture
def mock_faiss_index():
    """Mock del índice FAISS."""
    index = MagicMock()
    # Simula una búsqueda que devuelve 'APU-002' como el más cercano
    index.search.return_value = (np.array([[0.9]]), np.array([[1]]))
    return index


@pytest.fixture
def mock_id_map():
    """Mock del mapa de IDs."""
    return {"0": "APU-001", "1": "APU-002", "2": "APU-003"}


@pytest.fixture
def app_context(mock_embedding_model, mock_faiss_index, mock_id_map):
    """Crea un contexto de aplicación Flask de prueba con artefactos mockeados."""
    app = create_app("testing")
    with app.app_context():
        current_app.config["EMBEDDING_MODEL"] = mock_embedding_model
        current_app.config["FAISS_INDEX"] = mock_faiss_index
        current_app.config["ID_MAP"] = mock_id_map
        yield


# ============================================================================
# PRUEBAS PARA _find_best_semantic_match
# ============================================================================


def test_find_best_semantic_match_success(app_context, sample_apu_pool):
    """Prueba que se encuentre la mejor coincidencia cuando la similitud es alta."""
    log = []
    result = _find_best_semantic_match(
        df_pool=sample_apu_pool,
        query_text="pintura para exteriores",
        log=log,
        min_similarity=0.8,
    )
    assert result is not None
    assert result["CODIGO_APU"] == "APU-002"
    assert "✅ Coincidencia encontrada" in "".join(log)


def test_find_best_semantic_match_below_threshold(app_context, sample_apu_pool):
    """Prueba que no se devuelva nada si la mejor similitud está por debajo del umbral."""
    log = []
    result = _find_best_semantic_match(
        df_pool=sample_apu_pool,
        query_text="algo no relacionado",
        log=log,
        min_similarity=0.95,  # Umbral muy alto
    )
    assert result is None
    assert "❌ Sin coincidencia válida" in "".join(log)


# ============================================================================
# PRUEBAS DE INTEGRACIÓN PARA calculate_estimate CON FALLBACK
# ============================================================================


@patch("app.estimator._find_best_keyword_match")
@patch("app.estimator._find_best_semantic_match")
def test_calculate_estimate_uses_semantic_first(
    mock_semantic_match, mock_keyword_match, app_context, sample_apu_pool
):
    """
    Verifica que la búsqueda semántica se usa primero y, si tiene éxito,
    la de keywords no se llama para esa parte.
    """
    mock_semantic_match.return_value = sample_apu_pool.iloc[0]
    mock_keyword_match.return_value = sample_apu_pool.iloc[2]

    params = {"material": "muro de ladrillo", "cuadrilla": "1"}
    data_store = {"processed_apus": sample_apu_pool.to_dict("records"), "apus_detail": []}
    config = {"param_map": {}, "estimator_thresholds": {}}

    calculate_estimate(params, data_store, config)

    assert mock_semantic_match.call_count == 2
    mock_keyword_match.assert_called_once()
    # Acceder a argumentos posicionales (args) en lugar de kwargs
    assert "cuadrilla" in mock_keyword_match.call_args.args[1]


@patch("app.estimator._find_best_keyword_match")
@patch("app.estimator._find_best_semantic_match")
def test_calculate_estimate_fallback_to_keyword_on_semantic_failure(
    mock_semantic_match, mock_keyword_match, app_context, sample_apu_pool
):
    """
    Verifica que se recurre a la búsqueda por keywords si la semántica falla.
    """
    mock_semantic_match.return_value = None
    mock_keyword_match.return_value = sample_apu_pool.iloc[0]

    params = {"material": "pintura", "cuadrilla": "1"}
    data_store = {"processed_apus": sample_apu_pool.to_dict("records"), "apus_detail": []}
    config = {"param_map": {}, "estimator_thresholds": {}}

    result = calculate_estimate(params, data_store, config)

    assert mock_semantic_match.call_count == 2
    assert mock_keyword_match.call_count == 3
    assert "Búsqueda semántica sin éxito. Recurriendo a palabras clave" in result["log"]


@patch("app.estimator._find_best_keyword_match")
def test_calculate_estimate_no_semantic_artifacts(mock_keyword_match, sample_apu_pool):
    """
    Verifica que se recurre a keywords si los artefactos semánticos no están cargados.
    Esta prueba NO mockea _find_best_semantic_match para probar su lógica interna.
    """
    app = create_app("testing")
    with app.app_context():
        # Simular que los artefactos no se cargaron
        current_app.config["EMBEDDING_MODEL"] = None
        current_app.config["FAISS_INDEX"] = None
        current_app.config["ID_MAP"] = None

        mock_keyword_match.return_value = sample_apu_pool.iloc[0]

        params = {"material": "muro", "cuadrilla": "1"}
        data_store = {
            "processed_apus": sample_apu_pool.to_dict("records"),
            "apus_detail": [],
        }
        config = {"param_map": {}, "estimator_thresholds": {}}

        result = calculate_estimate(params, data_store, config)

        # Como consecuencia del fallo interno de la búsqueda semántica,
        # se debe llamar al fallback de keywords 3 veces (suministro, cuadrilla, tarea)
        assert mock_keyword_match.call_count == 3
        # Verificar que el log contiene el mensaje de error correcto
        assert "Faltan artefactos de búsqueda semántica" in result["log"]
