import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Añadir el directorio raíz del proyecto al sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.app import create_app
from app.estimator import (
    _find_best_keyword_match,
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
            },
            {
                "CODIGO_APU": "APU-002",
                "original_description": "Pintura acrílica para exteriores",
                "DESC_NORMALIZED": "pintura acrilica exteriores",
            },
            {
                "CODIGO_APU": "APU-003",
                "original_description": "CUADRILLA TIPO 1 (1 OF + 2 AYU)",
                "DESC_NORMALIZED": "cuadrilla tipo 1 1 of 2 ayu",
            },
        ]
    )


@pytest.fixture
def mock_embedding_model():
    """Mock del modelo SentenceTransformer."""
    model = MagicMock()
    # Simula la codificación de un embedding de 384 dimensiones
    model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
    return model


@pytest.fixture
def mock_faiss_index():
    """Mock del índice FAISS."""
    index = MagicMock()
    # Simula una búsqueda que devuelve distancias e índices
    # Distances: [[sim_alta, sim_media, sim_baja]]
    # Indices: [[idx_apu_1, idx_apu_2, idx_apu_0]]
    index.search.return_value = (
        np.array([[0.9, 0.7, 0.5]], dtype=np.float32),
        np.array([[1, 2, 0]], dtype=np.int64),
    )
    return index


@pytest.fixture
def mock_id_map():
    """Mock del mapa de IDs."""
    return {"0": "APU-001", "1": "APU-002", "2": "APU-003"}


@pytest.fixture
def app_context(mock_embedding_model, mock_faiss_index, mock_id_map):
    """
    Crea un contexto de aplicación Flask de prueba.

    Carga los mocks del modelo, índice y mapa en la configuración de la app
    para que `_find_best_semantic_match` pueda acceder a ellos a través
    de `current_app`.
    """
    app = create_app("testing")
    app.config["EMBEDDING_MODEL"] = mock_embedding_model
    app.config["FAISS_INDEX"] = mock_faiss_index
    app.config["ID_MAP"] = mock_id_map

    with app.app_context():
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
    assert isinstance(result, pd.Series)
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


def test_find_best_semantic_match_empty_pool(app_context):
    """Prueba que la función maneje un DataFrame vacío."""
    log = []
    result = _find_best_semantic_match(
        df_pool=pd.DataFrame(), query_text="consulta", log=log
    )
    assert result is None
    assert "Pool de APUs vacío" in "".join(log)


def test_find_best_semantic_match_empty_query(app_context, sample_apu_pool):
    """Prueba que la función maneje una consulta vacía."""
    log = []
    result = _find_best_semantic_match(
        df_pool=sample_apu_pool, query_text="  ", log=log
    )
    assert result is None
    assert "Texto de consulta vacío" in "".join(log)


def test_find_best_semantic_match_candidate_not_in_pool(app_context, sample_apu_pool):
    """Prueba que se ignoren candidatos del índice que no están en el pool actual."""
    # El pool solo contiene 'APU-001'. El mock de FAISS devolverá 'APU-002' como el mejor.
    filtered_pool = sample_apu_pool[sample_apu_pool["CODIGO_APU"] == "APU-001"]
    log = []

    result = _find_best_semantic_match(
        df_pool=filtered_pool,
        query_text="pintura",
        log=log,
        min_similarity=0.1,
    )

    # El resultado debe ser APU-001 (el único en el pool), no APU-002.
    assert result is not None
    assert result["CODIGO_APU"] == "APU-001"
    assert "ninguno de los candidatos del índice estaba en el pool filtrado" not in "".join(
        log
    )


# ============================================================================
# PRUEBAS PARA _find_best_keyword_match
# ============================================================================


def test_find_best_keyword_match_exact_word_success(sample_apu_pool):
    """Prueba que encuentre una coincidencia exacta de palabras."""
    log = []
    result = _find_best_keyword_match(
        df_pool=sample_apu_pool,
        keywords=["pintura", "exteriores"],
        log=log,
        min_match_percentage=50,
    )
    assert result is not None
    assert result["CODIGO_APU"] == "APU-002"


def test_find_best_keyword_match_substring_success(sample_apu_pool):
    """Prueba que encuentre una coincidencia de subcadena."""
    log = []
    result = _find_best_keyword_match(
        df_pool=sample_apu_pool,
        keywords=["cuadrilla", "tipo", "1"],
        log=log,
        match_mode="substring",
    )
    assert result is not None
    assert result["CODIGO_APU"] == "APU-003"


def test_find_best_keyword_match_no_match(sample_apu_pool):
    """Prueba que devuelva None cuando no hay coincidencia."""
    log = []
    result = _find_best_keyword_match(
        df_pool=sample_apu_pool, keywords=["palabra", "inexistente"], log=log
    )
    assert result is None


# ============================================================================
# PRUEBAS DE INTEGRACIÓN PARA calculate_estimate
# ============================================================================


@patch("app.estimator._find_best_semantic_match")
@patch("app.estimator._find_best_keyword_match")
def test_calculate_estimate_delegates_correctly(
    mock_keyword_match, mock_semantic_match, app_context, sample_apu_pool
):
    """
    Prueba que `calculate_estimate` llame a las funciones de búsqueda correctas.
    """
    # Configurar mocks para que devuelvan valores y así poder verificar el flujo
    mock_semantic_match.return_value = sample_apu_pool.iloc[0]  # Devuelve APU-001
    mock_keyword_match.return_value = sample_apu_pool.iloc[2]  # Devuelve APU-003

    # Datos de entrada para la función principal
    params = {"material": "muro de ladrillo", "cuadrilla": "1"}
    data_store = {
        "processed_apus": sample_apu_pool.to_dict("records"),
        "apus_detail": [],
    }
    config = {"param_map": {"material": {}, "cuadrilla": {}}}

    # Ejecutar la función
    result = calculate_estimate(params, data_store, config)

    # Verificar que no haya errores
    assert "error" not in result

    # Verificar que se llamó a la búsqueda semántica para suministro y tarea
    assert mock_semantic_match.call_count == 2
    # La primera llamada fue para suministro, con el texto "muro de ladrillo"
    mock_semantic_match.call_args_list[0].kwargs["query_text"] == "muro de ladrillo"
    # La segunda llamada fue para tarea, con el mismo texto
    mock_semantic_match.call_args_list[1].kwargs["query_text"] == "muro de ladrillo"

    # Verificar que se llamó a la búsqueda por keywords para la cuadrilla
    mock_keyword_match.assert_called_once()
    mock_keyword_match.call_args.kwargs["keywords"] == ["cuadrilla", "1"]
