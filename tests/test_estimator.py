import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

# Añadir el directorio raíz del proyecto al sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.estimator import (
    _calculate_match_score,
    _find_best_keyword_match,
    _find_best_semantic_match,
    calculate_estimate,
    safe_float_conversion,
    safe_int_conversion,
    validate_dataframe_columns,
    validate_numeric_range,
    get_safe_column_value,
    MatchMode,
    TipoAPU,
    TipoInsumo,
    SearchArtifacts,
    DerivationDetails,
)


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def sample_apu_pool():
    """Crea un DataFrame de pandas con APUs de prueba completo."""
    return pd.DataFrame(
        [
            {
                "CODIGO_APU": "APU-001",
                "original_description": "Instalación de muro de ladrillo estructural",
                "DESC_NORMALIZED": "instalacion muro ladrillo estructural",
                "VALOR_SUMINISTRO_UN": 100.50,
                "VALOR_CONSTRUCCION_UN": 150.75,
                "tipo_apu": TipoAPU.SUMINISTRO.value,
                "EQUIPO": 10.25,
                "UNIDAD": "M2",
            },
            {
                "CODIGO_APU": "APU-002",
                "original_description": "Pintura acrílica para exteriores color blanco",
                "DESC_NORMALIZED": "pintura acrilica exteriores color blanco",
                "VALOR_SUMINISTRO_UN": 200.00,
                "VALOR_CONSTRUCCION_UN": 250.00,
                "tipo_apu": TipoAPU.SUMINISTRO.value,
                "EQUIPO": 20.00,
                "UNIDAD": "M2",
            },
            {
                "CODIGO_APU": "APU-003",
                "original_description": "CUADRILLA TIPO 1 (1 OFICIAL + 2 AYUDANTES)",
                "DESC_NORMALIZED": "cuadrilla tipo 1 1 oficial 2 ayudantes",
                "VALOR_SUMINISTRO_UN": 0.00,
                "VALOR_CONSTRUCCION_UN": 300.00,
                "tipo_apu": TipoAPU.INSTALACION.value,
                "EQUIPO": 30.00,
                "UNIDAD": "DIA",
            },
            {
                "CODIGO_APU": "APU-004",
                "original_description": "Instalación de pintura en muros interiores",
                "DESC_NORMALIZED": "instalacion pintura muros interiores",
                "VALOR_SUMINISTRO_UN": 50.00,
                "VALOR_CONSTRUCCION_UN": 100.00,
                "tipo_apu": TipoAPU.INSTALACION.value,
                "EQUIPO": 15.00,
                "UNIDAD": "M2",
            },
            {
                "CODIGO_APU": "APU-005",
                "original_description": "Ladrillo estructural prefabricado",
                "DESC_NORMALIZED": "ladrillo estructural prefabricado",
                "VALOR_SUMINISTRO_UN": 80.00,
                "VALOR_CONSTRUCCION_UN": 120.00,
                "tipo_apu": TipoAPU.SUMINISTRO_PREFABRICADO.value,
                "EQUIPO": 5.00,
                "UNIDAD": "UND",
            },
        ]
    )


@pytest.fixture
def sample_apu_detail():
    """Crea datos de detalle de APUs para pruebas."""
    return [
        {
            "CODIGO_APU": "APU-003",
            "TIPO_INSUMO": TipoInsumo.MANO_OBRA.value,
            "CANTIDAD_APU": 0.5,
        },
        {
            "CODIGO_APU": "APU-003",
            "TIPO_INSUMO": TipoInsumo.MANO_OBRA.value,
            "CANTIDAD_APU": 0.3,
        },
        {
            "CODIGO_APU": "APU-004",
            "TIPO_INSUMO": TipoInsumo.MANO_OBRA.value,
            "CANTIDAD_APU": 0.25,
        },
    ]


@pytest.fixture
def mock_embedding_model():
    """Mock del modelo SentenceTransformer."""
    model = MagicMock()
    # Simular embedding normalizado
    model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
    return model


@pytest.fixture
def mock_faiss_index():
    """Mock del índice FAISS."""
    index = MagicMock()
    # Simular búsqueda exitosa: distancia alta (0.9) para índice 1 (APU-002)
    index.search.return_value = (
        np.array([[0.92, 0.85, 0.75]]),  # Similitudes
        np.array([[1, 0, 2]]),  # Índices
    )
    index.ntotal = 5  # Tamaño del índice
    return index


@pytest.fixture
def mock_id_map():
    """Mock del mapa de IDs FAISS -> CODIGO_APU."""
    return {
        "0": "APU-001",
        "1": "APU-002",
        "2": "APU-003",
        "3": "APU-004",
        "4": "APU-005",
    }


@pytest.fixture
def mock_search_artifacts(mock_embedding_model, mock_faiss_index, mock_id_map):
    """Crea artefactos de búsqueda mockeados."""
    return SearchArtifacts(
        model=mock_embedding_model,
        faiss_index=mock_faiss_index,
        id_map=mock_id_map
    )


@pytest.fixture
def sample_config():
    """Configuración de prueba para el estimador."""
    return {
        "param_map": {
            "material": {
                "LADRILLO": "ladrillo estructural",
                "PINTURA": "pintura acrílica",
            }
        },
        "estimator_thresholds": {
            "min_semantic_similarity_suministro": 0.30,
            "min_semantic_similarity_tarea": 0.40,
            "min_keyword_match_percentage_cuadrilla": 50.0,
        },
        "estimator_rules": {
            "factores_zona": {
                "ZONA 0": 1.0,
                "ZONA 1": 1.1,
                "ZONA 2": 1.2,
            },
            "costo_adicional_izaje": {
                "MANUAL": 0.0,
                "GRUA": 50.0,
                "TORRE": 100.0,
            },
            "factor_seguridad": {
                "NORMAL": 1.0,
                "ALTO": 1.15,
                "CRITICO": 1.30,
            },
        },
    }


# ============================================================================
# PRUEBAS PARA FUNCIONES AUXILIARES
# ============================================================================


class TestSafeConversions:
    """Pruebas para conversiones seguras."""

    def test_safe_float_conversion_valid(self):
        assert safe_float_conversion(42.5) == 42.5
        assert safe_float_conversion("123.45") == 123.45
        assert safe_float_conversion(100) == 100.0

    def test_safe_float_conversion_invalid(self):
        assert safe_float_conversion(None, 10.0) == 10.0
        assert safe_float_conversion("invalid", 5.0) == 5.0
        assert safe_float_conversion(np.nan, 0.0) == 0.0
        assert safe_float_conversion(np.inf, 0.0) == 0.0

    def test_safe_int_conversion_valid(self):
        assert safe_int_conversion(42) == 42
        assert safe_int_conversion("123") == 123
        assert safe_int_conversion(99.9) == 99

    def test_safe_int_conversion_invalid(self):
        assert safe_int_conversion(None, 10) == 10
        assert safe_int_conversion("invalid", 5) == 5
        assert safe_int_conversion(np.nan, 0) == 0


class TestValidations:
    """Pruebas para funciones de validación."""

    def test_validate_dataframe_columns_success(self):
        df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        assert validate_dataframe_columns(df, ["A", "B"], "test_df") is True

    def test_validate_dataframe_columns_missing(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        assert validate_dataframe_columns(df, ["A", "B", "C"], "test_df") is False

    def test_validate_dataframe_columns_not_dataframe(self):
        assert validate_dataframe_columns(None, ["A"], "test") is False
        assert validate_dataframe_columns([], ["A"], "test") is False

    def test_validate_numeric_range_valid(self):
        assert validate_numeric_range(0.5, (0.0, 1.0), "similarity") is True
        assert validate_numeric_range(50, (0, 100), "percentage") is True

    def test_validate_numeric_range_invalid(self):
        assert validate_numeric_range(1.5, (0.0, 1.0), "similarity") is False
        assert validate_numeric_range(-10, (0, 100), "percentage") is False

    def test_get_safe_column_value(self):
        row = pd.Series({"name": "test", "value": 42, "empty": None})
        
        assert get_safe_column_value(row, "name", "default") == "test"
        assert get_safe_column_value(row, "value", 0, int) == 42
        assert get_safe_column_value(row, "empty", "default") == "default"
        assert get_safe_column_value(row, "missing", "default") == "default"


class TestCalculateMatchScore:
    """Pruebas para cálculo de puntuación de coincidencia."""

    def test_calculate_match_score_full_match(self):
        desc_words = {"instalacion", "muro", "ladrillo"}
        keywords = ["muro", "ladrillo"]
        matches, percentage = _calculate_match_score(desc_words, keywords)
        assert matches == 2
        assert percentage == 100.0

    def test_calculate_match_score_partial_match(self):
        desc_words = {"instalacion", "muro", "ladrillo"}
        keywords = ["muro", "pintura"]
        matches, percentage = _calculate_match_score(desc_words, keywords)
        assert matches == 1
        assert percentage == 50.0

    def test_calculate_match_score_no_match(self):
        desc_words = {"instalacion", "muro", "ladrillo"}
        keywords = ["pintura", "acrilica"]
        matches, percentage = _calculate_match_score(desc_words, keywords)
        assert matches == 0
        assert percentage == 0.0

    def test_calculate_match_score_empty_inputs(self):
        assert _calculate_match_score(set(), []) == (0, 0.0)
        assert _calculate_match_score({"test"}, []) == (0, 0.0)
        assert _calculate_match_score(set(), ["test"]) == (0, 0.0)


# ============================================================================
# PRUEBAS PARA _find_best_keyword_match
# ============================================================================


class TestFindBestKeywordMatch:
    """Pruebas para búsqueda por palabras clave."""

    def test_keyword_match_strict_success(self, sample_apu_pool):
        log = []
        match, details = _find_best_keyword_match(
            df_pool=sample_apu_pool,
            keywords=["muro", "ladrillo", "estructural"],
            log=log,
            strict=True,
        )
        assert match is not None
        assert match["CODIGO_APU"] == "APU-001"
        assert details is not None
        assert details.confidence_score == 1.0
        assert "✅ Match ESTRICTO encontrado (100%)" in "\n".join(log)

    def test_keyword_match_strict_failure(self, sample_apu_pool):
        log = []
        match, details = _find_best_keyword_match(
            df_pool=sample_apu_pool,
            keywords=["muro", "ladrillo", "material", "inexistente"],
            log=log,
            strict=True,
        )
        assert match is None
        assert details is None
        assert "❌ No se encontró match estricto" in "\n".join(log)

    def test_keyword_match_flexible_success(self, sample_apu_pool):
        log = []
        match, details = _find_best_keyword_match(
            df_pool=sample_apu_pool,
            keywords=["pintura", "acrilica"],
            log=log,
            strict=False,
            min_match_percentage=50.0,
        )
        assert match is not None
        assert match["CODIGO_APU"] == "APU-002"
        assert details is not None
        assert details.confidence_score == 1.0
        assert "✅ Match FLEXIBLE encontrado" in "\n".join(log)

    def test_keyword_match_below_threshold(self, sample_apu_pool):
        log = []
        match, details = _find_best_keyword_match(
            df_pool=sample_apu_pool,
            keywords=["palabra", "totalmente", "inexistente"],
            log=log,
            strict=False,
            min_match_percentage=30.0,
        )
        assert match is None
        assert details is None
        assert "❌ Sin match válido" in "\n".join(log)

    def test_keyword_match_substring_mode(self, sample_apu_pool):
        log = []
        match, details = _find_best_keyword_match(
            df_pool=sample_apu_pool,
            keywords=["cuadrilla", "tipo", "1"],
            log=log,
            strict=False,
            match_mode=MatchMode.SUBSTRING,
        )
        assert match is not None
        assert match["CODIGO_APU"] == "APU-003"
        assert details is not None

    def test_keyword_match_empty_pool(self):
        log = []
        match, details = _find_best_keyword_match(
            df_pool=pd.DataFrame(),
            keywords=["test"],
            log=log,
        )
        assert match is None
        assert "⚠️ Pool vacío" in "\n".join(log)

    def test_keyword_match_invalid_keywords(self, sample_apu_pool):
        log = []
        match, details = _find_best_keyword_match(
            df_pool=sample_apu_pool,
            keywords=[],
            log=log,
        )
        assert match is None
        assert "⚠️ Keywords vacías" in "\n".join(log)

    def test_keyword_match_invalid_mode(self, sample_apu_pool):
        log = []
        match, details = _find_best_keyword_match(
            df_pool=sample_apu_pool,
            keywords=["muro"],
            log=log,
            match_mode="invalid_mode",
        )
        # Debe usar el modo por defecto (WORDS)
        assert "Usando 'words'" in "\n".join(log) or match is not None


# ============================================================================
# PRUEBAS PARA _find_best_semantic_match
# ============================================================================


class TestFindBestSemanticMatch:
    """Pruebas para búsqueda semántica."""

    def test_semantic_match_success(self, mock_search_artifacts, sample_apu_pool):
        log = []
        match, details = _find_best_semantic_match(
            df_pool=sample_apu_pool,
            query_text="pintura para exteriores",
            search_artifacts=mock_search_artifacts,
            log=log,
            min_similarity=0.8,
        )
        assert match is not None
        assert match["CODIGO_APU"] == "APU-002"
        assert details is not None
        assert details.match_method == "SEMANTIC"
        assert details.confidence_score >= 0.8
        assert "✅ Coincidencia semántica encontrada" in "\n".join(log)

    def test_semantic_match_below_threshold(self, mock_search_artifacts, sample_apu_pool):
        log = []
        # Mock para retornar similitud baja
        with patch.object(
            mock_search_artifacts.faiss_index,
            "search",
            return_value=(np.array([[0.25]]), np.array([[1]])),
        ):
            match, details = _find_best_semantic_match(
                df_pool=sample_apu_pool,
                query_text="algo no relacionado",
                search_artifacts=mock_search_artifacts,
                log=log,
                min_similarity=0.95,
            )
            assert match is None
            assert details is None
            assert "❌ Sin coincidencia válida" in "\n".join(log)

    def test_semantic_match_no_artifacts(self, sample_apu_pool):
        """Prueba cuando no hay artefactos de búsqueda semántica."""
        log = []
        # artifacts as None
        match, details = _find_best_semantic_match(
            df_pool=sample_apu_pool,
            query_text="test query",
            search_artifacts=None,
            log=log,
        )
        assert match is None
        assert "Artefactos de búsqueda semántica no disponibles" in "\n".join(log)

    def test_semantic_match_empty_pool(self, mock_search_artifacts):
        log = []
        match, details = _find_best_semantic_match(
            df_pool=pd.DataFrame(),
            query_text="test query",
            search_artifacts=mock_search_artifacts,
            log=log,
        )
        assert match is None
        assert "⚠️ Pool de APUs vacío" in "\n".join(log)

    def test_semantic_match_empty_query(self, mock_search_artifacts, sample_apu_pool):
        log = []
        match, details = _find_best_semantic_match(
            df_pool=sample_apu_pool,
            query_text="",
            search_artifacts=mock_search_artifacts,
            log=log,
        )
        assert match is None
        assert "⚠️ Texto de consulta vacío" in "\n".join(log)

    def test_semantic_match_embedding_error(self, mock_search_artifacts, sample_apu_pool):
        """Prueba manejo de error al generar embedding."""
        log = []
        with patch.object(
            mock_search_artifacts.model,
            "encode",
            side_effect=Exception("Model error"),
        ):
            match, details = _find_best_semantic_match(
                df_pool=sample_apu_pool,
                query_text="test query",
                search_artifacts=mock_search_artifacts,
                log=log,
            )
            assert match is None
            assert "❌ ERROR al generar embedding" in "\n".join(log)

    def test_semantic_match_faiss_error(self, mock_search_artifacts, sample_apu_pool):
        """Prueba manejo de error en búsqueda FAISS."""
        log = []
        with patch.object(
            mock_search_artifacts.faiss_index,
            "search",
            side_effect=Exception("FAISS error"),
        ):
            match, details = _find_best_semantic_match(
                df_pool=sample_apu_pool,
                query_text="test query",
                search_artifacts=mock_search_artifacts,
                log=log,
            )
            assert match is None
            assert "❌ ERROR en búsqueda FAISS" in "\n".join(log)

    def test_semantic_match_apu_not_in_pool(self, mock_search_artifacts, sample_apu_pool):
        """Prueba cuando el APU encontrado no está en el pool filtrado."""
        log = []
        # Mock para retornar un índice que no existe en el pool
        with patch.object(
            mock_search_artifacts.faiss_index,
            "search",
            return_value=(np.array([[0.95]]), np.array([[99]])),
        ):
            # Agregar ese índice al mapa pero con un código inexistente
            mock_search_artifacts.id_map["99"] = "APU-999"
            
            match, details = _find_best_semantic_match(
                df_pool=sample_apu_pool,
                query_text="test query",
                search_artifacts=mock_search_artifacts,
                log=log,
            )
            assert match is None
            assert "⚠️ Ningún resultado de FAISS coincide con el pool" in "\n".join(log)


# ============================================================================
# PRUEBAS PARA calculate_estimate
# ============================================================================


class TestCalculateEstimate:
    """Pruebas de integración para el estimador."""

    def test_calculate_estimate_full_success(
        self, mock_search_artifacts, sample_apu_pool, sample_apu_detail, sample_config
    ):
        """Prueba exitosa con todos los componentes."""
        params = {
            "material": "LADRILLO",
            "cuadrilla": "1",
            "zona": "ZONA 1",
            "izaje": "GRUA",
            "seguridad": "ALTO",
        }
        data_store = {
            "processed_apus": sample_apu_pool.to_dict("records"),
            "apus_detail": sample_apu_detail,
        }

        result = calculate_estimate(params, data_store, sample_config, mock_search_artifacts)

        assert "error" not in result
        assert result["valor_construccion"] > 0
        assert result["valor_suministro"] >= 0
        assert result["valor_instalacion"] >= 0
        assert "factores_aplicados" in result
        assert result["factores_aplicados"]["zona"] == 1.1
        assert result["factores_aplicados"]["seguridad"] == 1.15
        assert result["factores_aplicados"]["izaje"] == 50.0
        assert "derivation_details" in result

    def test_calculate_estimate_missing_material(
        self, mock_search_artifacts, sample_apu_pool, sample_config
    ):
        """Prueba con material faltante."""
        params = {"cuadrilla": "1"}
        data_store = {"processed_apus": sample_apu_pool.to_dict("records")}

        result = calculate_estimate(params, data_store, sample_config, mock_search_artifacts)

        assert "error" in result
        assert "obligatorio" in result["error"]

    def test_calculate_estimate_empty_apus(self, mock_search_artifacts, sample_config):
        """Prueba con APUs vacíos."""
        params = {"material": "LADRILLO"}
        data_store = {"processed_apus": []}

        result = calculate_estimate(params, data_store, sample_config, mock_search_artifacts)

        assert "error" in result
        assert "No hay datos de APU" in result["error"]

    @patch("app.estimator._find_best_keyword_match")
    @patch("app.estimator._find_best_semantic_match")
    def test_calculate_estimate_uses_semantic_first(
        self,
        mock_semantic_match,
        mock_keyword_match,
        mock_search_artifacts,
        sample_apu_pool,
        sample_config,
    ):
        """Verifica que la búsqueda semántica se use primero."""
        # Configurar mocks para devolver (match, details)
        mock_semantic_match.return_value = (sample_apu_pool.iloc[0], DerivationDetails("SEMANTIC", 0.9, "test", "test"))  # Suministro
        mock_keyword_match.return_value = (sample_apu_pool.iloc[2], DerivationDetails("KEYWORD", 1.0, "test", "test"))  # Cuadrilla

        params = {"material": "LADRILLO", "cuadrilla": "1"}
        data_store = {
            "processed_apus": sample_apu_pool.to_dict("records"),
            "apus_detail": [],
        }

        result = calculate_estimate(params, data_store, sample_config, mock_search_artifacts)

        # La búsqueda semántica se llama 2 veces: suministro y tarea
        assert mock_semantic_match.call_count == 2
        
        # La búsqueda por keywords se llama solo para cuadrilla
        assert mock_keyword_match.call_count == 1
        
        # Verificar que se llamó con keywords de cuadrilla
        call_args = mock_keyword_match.call_args[0]
        assert "cuadrilla" in " ".join(call_args[1])

    @patch("app.estimator._find_best_keyword_match")
    @patch("app.estimator._find_best_semantic_match")
    def test_calculate_estimate_fallback_to_keyword(
        self,
        mock_semantic_match,
        mock_keyword_match,
        mock_search_artifacts,
        sample_apu_pool,
        sample_config,
    ):
        """Verifica el fallback a keywords cuando semántica falla."""
        # Semántica falla
        mock_semantic_match.return_value = (None, None)
        # Keywords tiene éxito
        mock_keyword_match.return_value = (sample_apu_pool.iloc[0], DerivationDetails("KEYWORD", 1.0, "test", "test"))

        params = {"material": "PINTURA", "cuadrilla": "1"}
        data_store = {
            "processed_apus": sample_apu_pool.to_dict("records"),
            "apus_detail": [],
        }

        result = calculate_estimate(params, data_store, sample_config, mock_search_artifacts)

        # Semántica se intenta 2 veces (suministro y tarea)
        assert mock_semantic_match.call_count == 2
        
        # Keywords se usa 3 veces (fallback suministro, cuadrilla, fallback tarea)
        assert mock_keyword_match.call_count == 3
        
        # Verificar mensaje de fallback en log
        assert "Fallback: Búsqueda por palabras clave" in result["log"]

    def test_calculate_estimate_no_semantic_artifacts(
        self, sample_apu_pool, sample_config
    ):
        """Verifica comportamiento sin artefactos semánticos."""
        # No configurar artefactos semánticos
        search_artifacts = None # Or empty SearchArtifacts(None, None, None) if strict typing

        params = {"material": "LADRILLO", "cuadrilla": "1"}
        data_store = {
            "processed_apus": sample_apu_pool.to_dict("records"),
            "apus_detail": [],
        }

        with patch("app.estimator._find_best_keyword_match") as mock_kw:
            mock_kw.return_value = (sample_apu_pool.iloc[0], None)

            # This call will internally handle None search_artifacts by logging error and falling back or returning None
            # Wait, _find_best_semantic_match checks artifacts validity.
            # calculate_estimate will call semantic search, fail, then fallback to keyword
            result = calculate_estimate(params, data_store, sample_config, search_artifacts) # Pass None

            # Debe usar keywords como fallback because semantic search returns None if artifacts are missing
            assert mock_kw.call_count >= 2
            # The log message comes from inside _find_best_semantic_match
            assert "Artefactos de búsqueda semántica no disponibles" in result["log"]

    def test_calculate_estimate_with_rendimiento(
        self, mock_search_artifacts, sample_apu_pool, sample_apu_detail, sample_config
    ):
        """Prueba cálculo de rendimiento desde detalle."""
        params = {"material": "PINTURA", "cuadrilla": "1"}
        data_store = {
            "processed_apus": sample_apu_pool.to_dict("records"),
            "apus_detail": sample_apu_detail,
        }

        with patch("app.estimator._find_best_semantic_match") as mock_sem:
            # Retornar APU-003 para cuadrilla y APU-004 para tarea
            mock_sem.side_effect = [
                (sample_apu_pool.iloc[1], None),  # Suministro
                (sample_apu_pool.iloc[2], None),  # Tarea (APU-003)
            ]

            with patch("app.estimator._find_best_keyword_match") as mock_kw:
                mock_kw.return_value = (sample_apu_pool.iloc[2], None)  # Cuadrilla

                result = calculate_estimate(params, data_store, sample_config, mock_search_artifacts)

                # Verificar que se calculó el rendimiento
                assert result["rendimiento_m2_por_dia"] > 0
                assert "Rendimiento" in result["log"]

    def test_calculate_estimate_default_values(
        self, mock_search_artifacts, sample_apu_pool, sample_config
    ):
        """Prueba con valores por defecto."""
        params = {"material": "LADRILLO"}  # Solo material
        data_store = {
            "processed_apus": sample_apu_pool.to_dict("records"),
            "apus_detail": [],
        }

        result = calculate_estimate(params, data_store, sample_config, mock_search_artifacts)

        # Verificar valores por defecto
        assert "factores_aplicados" in result
        assert result["factores_aplicados"]["zona"] == 1.0  # ZONA 0
        assert result["factores_aplicados"]["seguridad"] == 1.0  # NORMAL
        assert result["factores_aplicados"]["izaje"] == 0.0  # MANUAL

    def test_calculate_estimate_invalid_config_values(
        self, mock_search_artifacts, sample_apu_pool
    ):
        """Prueba con valores de configuración inválidos."""
        params = {"material": "LADRILLO"}
        data_store = {"processed_apus": sample_apu_pool.to_dict("records")}
        
        # Configuración con valores inválidos
        invalid_config = {
            "param_map": {},
            "estimator_thresholds": {
                "min_semantic_similarity_suministro": 1.5,  # Fuera de rango
                "min_keyword_match_percentage_cuadrilla": 150.0,  # Fuera de rango
            },
            "estimator_rules": {},
        }

        result = calculate_estimate(params, data_store, invalid_config, mock_search_artifacts)

        # No debe fallar, debe ajustar a valores válidos
        assert "error" not in result or "obligatorio" in result.get("error", "")


# ============================================================================
# PRUEBAS DE CASOS EDGE
# ============================================================================


class TestEdgeCases:
    """Pruebas de casos límite y situaciones especiales."""

    def test_apu_with_null_values(self, mock_search_artifacts):
        """Prueba con APUs que contienen valores nulos."""
        apu_pool_with_nulls = pd.DataFrame(
            [
                {
                    "CODIGO_APU": "APU-NULL",
                    "original_description": None,
                    "DESC_NORMALIZED": None,
                    "VALOR_SUMINISTRO_UN": None,
                    "tipo_apu": TipoAPU.SUMINISTRO.value,
                    "EQUIPO": None,
                    "UNIDAD": "M2",
                }
            ]
        )

        log = []
        match, details = _find_best_keyword_match(
            df_pool=apu_pool_with_nulls,
            keywords=["test"],
            log=log,
        )
        
        # No debe crashear
        assert match is None

    def test_very_long_description(self, mock_search_artifacts, sample_apu_pool):
        """Prueba con descripciones muy largas."""
        long_desc_pool = sample_apu_pool.copy()
        long_desc_pool.loc[0, "original_description"] = "A" * 1000
        long_desc_pool.loc[0, "DESC_NORMALIZED"] = "a " * 500

        log = []
        match, details = _find_best_keyword_match(
            df_pool=long_desc_pool,
            keywords=["a"],
            log=log,
        )
        
        # Verificar que el log trunca correctamente
        log_text = "\n".join(log)
        assert "..." in log_text  # Debe truncar descripciones largas

    def test_special_characters_in_keywords(self, mock_search_artifacts, sample_apu_pool):
        """Prueba con caracteres especiales en keywords."""
        log = []
        match, details = _find_best_keyword_match(
            df_pool=sample_apu_pool,
            keywords=["muro@#$%", "ladrillo!!!"],
            log=log,
        )
        
        # No debe crashear. Si no hay match, debe indicarlo en el log.
        assert match is not None or "No se encontraron candidatos" in "\n".join(log)

    def test_zero_division_scenarios(self, mock_search_artifacts, sample_apu_pool):
        """Prueba escenarios que podrían causar división por cero."""
        params = {"material": "LADRILLO", "cuadrilla": "1"}
        data_store = {
            "processed_apus": sample_apu_pool.to_dict("records"),
            "apus_detail": [
                {
                    "CODIGO_APU": "APU-003",
                    "TIPO_INSUMO": TipoInsumo.MANO_OBRA.value,
                    "CANTIDAD_APU": 0.0,  # Cero
                }
            ],
        }
        config = {"param_map": {}, "estimator_thresholds": {}, "estimator_rules": {}}

        result = calculate_estimate(params, data_store, config, mock_search_artifacts)

        # No debe crashear por división por cero
        assert "error" not in result or "obligatorio" in result.get("error", "")
        assert result.get("rendimiento_m2_por_dia", 0) == 0

    def test_negative_values_in_apus(self, mock_search_artifacts):
        """Prueba con valores negativos (que no deberían existir)."""
        negative_pool = pd.DataFrame(
            [
                {
                    "CODIGO_APU": "APU-NEG",
                    "original_description": "Test negativo",
                    "DESC_NORMALIZED": "test negativo",
                    "VALOR_SUMINISTRO_UN": -100,  # Negativo
                    "VALOR_CONSTRUCCION_UN": -50,
                    "tipo_apu": TipoAPU.SUMINISTRO.value,
                    "EQUIPO": -10,
                    "UNIDAD": "M2",
                }
            ]
        )

        params = {"material": "TEST"}
        data_store = {"processed_apus": negative_pool.to_dict("records")}
        config = {"param_map": {}, "estimator_thresholds": {}, "estimator_rules": {}}

        result = calculate_estimate(params, data_store, config, mock_search_artifacts)

        # Debe manejar valores negativos sin crashear
        # El valor final podría ser negativo o cero según la lógica
        assert isinstance(result.get("valor_construccion", 0), (int, float))


# ============================================================================
# PRUEBAS DE PERFORMANCE Y CARGA
# ============================================================================


class TestPerformance:
    """Pruebas de rendimiento con datasets grandes."""

    def test_large_apu_pool(self, mock_search_artifacts):
        """Prueba con un pool grande de APUs."""
        # Crear 1000 APUs
        large_pool = pd.DataFrame(
            [
                {
                    "CODIGO_APU": f"APU-{i:04d}",
                    "original_description": f"Descripción del APU número {i}",
                    "DESC_NORMALIZED": f"descripcion apu numero {i}",
                    "VALOR_SUMINISTRO_UN": i * 10,
                    "VALOR_CONSTRUCCION_UN": i * 15,
                    "tipo_apu": TipoAPU.SUMINISTRO.value if i % 2 == 0 else TipoAPU.INSTALACION.value,
                    "EQUIPO": i * 5,
                    "UNIDAD": "M2",
                }
                for i in range(1000)
            ]
        )

        log = []
        import time
        
        start = time.time()
        match, details = _find_best_keyword_match(
            df_pool=large_pool,
            keywords=["descripcion", "numero", "500"],
            log=log,
        )
        elapsed = time.time() - start

        assert match is not None
        assert elapsed < 2.0  # Debe completarse en menos de 2 segundos
        print(f"\n⏱️ Búsqueda en 1000 APUs: {elapsed:.3f}s")


# ============================================================================
# EJECUCIÓN DE PRUEBAS
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--color=yes"])
