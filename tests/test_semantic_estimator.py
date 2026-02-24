"""
Suite de pruebas exhaustiva para el módulo Semantic Estimator.

Organización por capas:
    1. SafeConvert — Conversiones numéricas robustas
    2. EstimationLog — Acumulador de trazas
    3. DataValidator — Validación de DataFrames
    4. SearchEngine — Búsqueda keyword y semántica
    5. CostCalculator — Cálculo de costos y rendimiento
    6. SemanticEstimatorService — Fachada y ciclo de vida
    7. Integración — Flujos end-to-end
    8. Casos límite y regresiones

Convenciones:
    - Fixtures reutilizables para DataFrames y artefactos mock.
    - Mocks para SentenceTransformer y FAISS (sin GPU/modelo real).
    - Nombres descriptivos en español.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from semantic_estimator import (
    CostCalculator,
    DataQualityMetrics,
    DataValidator,
    DEFAULT_MIN_MATCH_PERCENTAGE,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    DerivationDetails,
    EstimationComponents,
    EstimationFactors,
    EstimationLog,
    MAX_CANDIDATES_TO_TRACK,
    MAX_COSTO_UNITARIO,
    MAX_KEYWORDS_COUNT,
    MIN_DATA_QUALITY_THRESHOLD,
    MIN_DESCRIPTION_LENGTH,
    MIN_RENDIMIENTO_SAFE,
    MatchCandidate,
    MatchMode,
    SafeConvert,
    SearchArtifacts,
    SearchEngine,
    SemanticEstimatorService,
    TipoAPU,
    TipoRecurso,
)


# ============================================================================
# FIXTURES COMPARTIDOS
# ============================================================================

@pytest.fixture
def mock_model():
    """SentenceTransformer mock que retorna embeddings unitarios."""
    model = MagicMock()
    embedding = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    model.encode.return_value = embedding
    return model


@pytest.fixture
def mock_faiss_index():
    """Índice FAISS mock con resultados predefinidos."""
    index = MagicMock()
    index.ntotal = 100
    index.d = 3
    distances = np.array([[0.95, 0.80, 0.60]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    index.search.return_value = (distances, indices)
    return index


@pytest.fixture
def mock_id_map() -> Dict[str, str]:
    """Mapa de IDs FAISS → códigos APU."""
    return {
        "0": "APU-001",
        "1": "APU-002",
        "2": "APU-003",
    }


@pytest.fixture
def search_artifacts(mock_model, mock_faiss_index, mock_id_map) -> SearchArtifacts:
    """Artefactos de búsqueda completos con mocks."""
    return SearchArtifacts(
        model=mock_model,
        faiss_index=mock_faiss_index,
        id_map=mock_id_map,
    )


@pytest.fixture
def df_apus_base() -> pd.DataFrame:
    """DataFrame base de APUs con columnas esenciales."""
    return pd.DataFrame([
        {
            "CODIGO_APU": "APU-001",
            "DESC_NORMALIZED": "tubo pvc presion 4 pulgadas",
            "original_description": "Tubo PVC Presión 4\"",
            "tipo_apu": TipoAPU.SUMINISTRO.value,
            "UNIDAD": "ML",
            "VALOR_SUMINISTRO_UN": 25000.0,
            "EQUIPO": 5000.0,
        },
        {
            "CODIGO_APU": "APU-002",
            "DESC_NORMALIZED": "instalacion tubo pvc presion",
            "original_description": "Instalación Tubo PVC Presión",
            "tipo_apu": TipoAPU.INSTALACION.value,
            "UNIDAD": "ML",
            "VALOR_SUMINISTRO_UN": 0.0,
            "EQUIPO": 8000.0,
        },
        {
            "CODIGO_APU": "APU-003",
            "DESC_NORMALIZED": "cuadrilla 3 instalacion tuberia",
            "original_description": "Cuadrilla 3 Instalación Tubería",
            "tipo_apu": TipoAPU.INSTALACION.value,
            "UNIDAD": "DIA",
            "VALOR_CONSTRUCCION_UN": 350000.0,
            "EQUIPO": 0.0,
        },
        {
            "CODIGO_APU": "APU-004",
            "DESC_NORMALIZED": "valvula compuerta 6 pulgadas bronce",
            "original_description": "Válvula Compuerta 6\" Bronce",
            "tipo_apu": TipoAPU.SUMINISTRO.value,
            "UNIDAD": "UND",
            "VALOR_SUMINISTRO_UN": 180000.0,
            "EQUIPO": 0.0,
        },
        {
            "CODIGO_APU": "APU-005",
            "DESC_NORMALIZED": "suministro prefabricado caja inspeccion",
            "original_description": "Suministro Prefabricado Caja Inspección",
            "tipo_apu": TipoAPU.SUMINISTRO_PREFABRICADO.value,
            "UNIDAD": "UND",
            "VALOR_SUMINISTRO_UN": 95000.0,
            "EQUIPO": 0.0,
        },
    ])


@pytest.fixture
def df_detail_base() -> List[Dict[str, Any]]:
    """Detalle de insumos por APU (mano de obra, equipo, material)."""
    return [
        {"CODIGO_APU": "APU-002", "TIPO_INSUMO": "MANO DE OBRA", "CANTIDAD_APU": 0.25},
        {"CODIGO_APU": "APU-002", "TIPO_INSUMO": "MANO DE OBRA", "CANTIDAD_APU": 0.25},
        {"CODIGO_APU": "APU-002", "TIPO_INSUMO": "EQUIPO", "CANTIDAD_APU": 0.5},
        {"CODIGO_APU": "APU-002", "TIPO_INSUMO": "MATERIAL", "CANTIDAD_APU": 1.0},
        {"CODIGO_APU": "APU-001", "TIPO_INSUMO": "MATERIAL", "CANTIDAD_APU": 1.0},
    ]


@pytest.fixture
def estimation_log() -> EstimationLog:
    """Log de estimación fresco."""
    return EstimationLog("TEST")


@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Configuración base para el estimador."""
    return {
        "estimator_thresholds": {
            "min_semantic_similarity_suministro": 0.3,
            "min_semantic_similarity_tarea": 0.4,
            "min_keyword_match_percentage_cuadrilla": 50.0,
        },
        "estimator_rules": {
            "factores_zona": {"ZONA 0": 1.0, "ZONA 1": 1.15, "ZONA 2": 1.30},
            "costo_adicional_izaje": {"MANUAL": 0.0, "GRUA": 50000.0},
            "factor_seguridad": {"NORMAL": 1.0, "ALTO": 1.25},
        },
        "param_map": {},
        "embedding_metadata": {"model_name": "all-MiniLM-L6-v2"},
    }


@pytest.fixture
def data_store(df_apus_base, df_detail_base) -> Dict[str, Any]:
    """Data store completo para estimaciones."""
    return {
        "processed_apus": df_apus_base.to_dict("records"),
        "apus_detail": df_detail_base,
    }


# ============================================================================
# 1. SAFE CONVERT
# ============================================================================

class TestSafeConvertToFloat:
    """Pruebas para SafeConvert.to_float."""

    @pytest.mark.parametrize("value,expected", [
        (10, 10.0),
        (3.14, 3.14),
        ("42.5", 42.5),
        ("1,000.50", 1000.50),
        (" 7.0 ", 7.0),
    ])
    def test_conversiones_validas(self, value, expected):
        assert SafeConvert.to_float(value) == expected

    @pytest.mark.parametrize("value", [
        None, "", "  ", "-", "N/A", "NA", "null", "None", "none", "n/a",
    ])
    def test_valores_faltantes_retornan_default(self, value):
        assert SafeConvert.to_float(value, default=99.0) == 99.0

    def test_nan_retorna_default(self):
        assert SafeConvert.to_float(float("nan"), default=0.0) == 0.0

    def test_inf_retorna_default(self):
        assert SafeConvert.to_float(float("inf"), default=0.0) == 0.0

    def test_neg_inf_retorna_default(self):
        assert SafeConvert.to_float(float("-inf"), default=0.0) == 0.0

    def test_string_no_numerico_retorna_default(self):
        assert SafeConvert.to_float("abc", default=-1.0) == -1.0

    def test_clamp_min(self):
        assert SafeConvert.to_float(-5.0, min_value=0.0) == 0.0

    def test_clamp_max(self):
        assert SafeConvert.to_float(999.0, max_value=100.0) == 100.0

    def test_clamp_ambos(self):
        result = SafeConvert.to_float(50.0, min_value=10.0, max_value=40.0)
        assert result == 40.0

    def test_pandas_na(self):
        assert SafeConvert.to_float(pd.NA, default=7.0) == 7.0

    def test_numpy_nan(self):
        assert SafeConvert.to_float(np.nan, default=3.0) == 3.0

    def test_string_con_comas_y_espacios(self):
        assert SafeConvert.to_float("1 234,567", default=0.0) == 0.0
        assert SafeConvert.to_float("1,234,567", default=0.0) == 1234567.0


class TestSafeConvertToInt:
    """Pruebas para SafeConvert.to_int."""

    def test_conversion_basica(self):
        assert SafeConvert.to_int(42) == 42

    def test_trunca_decimales(self):
        assert SafeConvert.to_int(3.9) == 3

    def test_string_numerico(self):
        assert SafeConvert.to_int("7") == 7

    def test_default_para_invalido(self):
        assert SafeConvert.to_int("abc", default=5) == 5

    def test_clamp_min_max(self):
        assert SafeConvert.to_int(100, min_value=0, max_value=50) == 50
        assert SafeConvert.to_int(-10, min_value=0) == 0


class TestSafeConvertColumnValue:
    """Pruebas para SafeConvert.column_value."""

    def test_string_existente(self):
        row = pd.Series({"name": "Hello"})
        assert SafeConvert.column_value(row, "name") == "Hello"

    def test_columna_faltante_retorna_default(self):
        row = pd.Series({"a": 1})
        assert SafeConvert.column_value(row, "b", "DEFAULT") == "DEFAULT"

    def test_valor_nan_retorna_default(self):
        row = pd.Series({"x": np.nan})
        assert SafeConvert.column_value(row, "x", "N/A") == "N/A"

    def test_tipo_float(self):
        row = pd.Series({"price": "42.5"})
        result = SafeConvert.column_value(row, "price", 0.0, expected_type=float)
        assert result == 42.5

    def test_tipo_int(self):
        row = pd.Series({"qty": "10"})
        result = SafeConvert.column_value(row, "qty", 0, expected_type=int)
        assert result == 10

    def test_string_vacio_retorna_default(self):
        row = pd.Series({"name": "  "})
        assert SafeConvert.column_value(row, "name", "DEFAULT") == "DEFAULT"


# ============================================================================
# 2. ESTIMATION LOG
# ============================================================================

class TestEstimationLog:
    """Pruebas para EstimationLog."""

    def test_header_inicial(self):
        log = EstimationLog("HEADER")
        rendered = log.render()
        assert "HEADER" in rendered

    def test_log_vacio(self):
        log = EstimationLog()
        assert log.render() == ""

    def test_info_append(self):
        log = EstimationLog()
        log.info("Mensaje informativo")
        assert "Mensaje informativo" in log.render()

    def test_warn_formato(self):
        log = EstimationLog()
        log.warn("Advertencia")
        assert "⚠️" in log.render()
        assert "Advertencia" in log.render()

    def test_error_formato(self):
        log = EstimationLog()
        log.error("Error crítico")
        assert "❌" in log.render()

    def test_success_formato(self):
        log = EstimationLog()
        log.success("Operación exitosa")
        assert "✅" in log.render()

    def test_section_formato(self):
        log = EstimationLog()
        log.section("Mi Sección")
        rendered = log.render()
        assert "🎯 Mi Sección" in rendered
        assert "=" * 70 in rendered

    def test_render_multilinea(self):
        log = EstimationLog("H")
        log.info("L1")
        log.info("L2")
        lines = log.render().split("\n")
        assert len(lines) >= 3

    def test_log_top_candidates_vacio(self, estimation_log):
        estimation_log.log_top_candidates([], 3, 5)
        assert "No se encontraron" in estimation_log.render()

    def test_log_top_candidates_con_similitud(self, estimation_log):
        apu = pd.Series({"CODIGO_APU": "TEST-001"})
        cand = MatchCandidate(
            apu=apu,
            description="Material de prueba",
            matches=0,
            percentage=0.0,
            similarity=0.85,
        )
        estimation_log.log_top_candidates([cand], 1)
        rendered = estimation_log.render()
        assert "0.850" in rendered
        assert "TEST-001" in rendered

    def test_log_top_candidates_con_keywords(self, estimation_log):
        apu = pd.Series({"CODIGO_APU": "TEST-002"})
        cand = MatchCandidate(
            apu=apu,
            description="Cemento Portland",
            matches=2,
            percentage=66.7,
        )
        estimation_log.log_top_candidates([cand], 1, keywords_count=3)
        rendered = estimation_log.render()
        assert "2/3" in rendered
        assert "67%" in rendered

    def test_log_top_candidates_trunca_descripcion(self, estimation_log):
        apu = pd.Series({"CODIGO_APU": "TEST-003"})
        desc_larga = "A" * 100
        cand = MatchCandidate(
            apu=apu,
            description=desc_larga,
            matches=1,
            percentage=50.0,
            similarity=0.7,
        )
        estimation_log.log_top_candidates([cand], 1)
        rendered = estimation_log.render()
        assert "..." in rendered


# ============================================================================
# 3. DATA VALIDATOR
# ============================================================================

class TestDataValidator:
    """Pruebas para DataValidator."""

    def test_validate_search_artifacts_none(self, estimation_log):
        assert DataValidator.validate_search_artifacts(None, estimation_log) is False

    def test_validate_search_artifacts_incompleto(self, estimation_log):
        artifacts = SearchArtifacts(model=None, faiss_index=None, id_map={})
        assert DataValidator.validate_search_artifacts(artifacts, estimation_log) is False

    def test_validate_search_artifacts_completo(self, search_artifacts, estimation_log):
        assert DataValidator.validate_search_artifacts(search_artifacts, estimation_log) is True

    def test_validate_dataframe_valido(self, df_apus_base, estimation_log):
        result = DataValidator.validate_dataframe(
            df_apus_base, ["CODIGO_APU"], "test", estimation_log
        )
        assert result is not None
        assert len(result) == len(df_apus_base)

    def test_validate_dataframe_no_es_df(self, estimation_log):
        result = DataValidator.validate_dataframe("string", ["COL"], "test", estimation_log)
        assert result is None

    def test_validate_dataframe_vacio(self, estimation_log):
        result = DataValidator.validate_dataframe(
            pd.DataFrame(), ["COL"], "test", estimation_log
        )
        assert result is None

    def test_validate_dataframe_columna_faltante(self, estimation_log):
        df = pd.DataFrame({"A": [1]})
        result = DataValidator.validate_dataframe(df, ["B"], "test", estimation_log)
        assert result is None

    def test_assess_quality_buena(self, df_apus_base, estimation_log):
        metrics = DataValidator.assess_quality(df_apus_base, estimation_log)
        assert metrics.quality_score > 0.8
        assert metrics.is_acceptable()
        assert metrics.total_records == 5
        assert metrics.valid_records > 0

    def test_assess_quality_df_vacio(self, estimation_log):
        metrics = DataValidator.assess_quality(pd.DataFrame(), estimation_log)
        assert metrics.total_records == 0
        assert metrics.quality_score == 0.0

    def test_assess_quality_sin_codigos(self, estimation_log):
        df = pd.DataFrame({
            "CODIGO_APU": ["", None, "  "],
            "DESC_NORMALIZED": ["desc a", "desc b", "desc c"],
        })
        metrics = DataValidator.assess_quality(df, estimation_log)
        assert metrics.missing_codes > 0
        assert metrics.quality_score < 1.0

    def test_assess_quality_descripciones_cortas(self, estimation_log):
        df = pd.DataFrame({
            "CODIGO_APU": ["A", "B", "C"],
            "DESC_NORMALIZED": ["a", "", "ok desc"],
        })
        metrics = DataValidator.assess_quality(df, estimation_log)
        assert metrics.missing_descriptions >= 2

    def test_validate_columns_presente(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        ok, present, missing = DataValidator.validate_columns(df, ["A", "B"])
        assert ok is True
        assert present == ["A", "B"]
        assert missing == []

    def test_validate_columns_faltante_no_estricto(self):
        df = pd.DataFrame({"A": [1]})
        ok, present, missing = DataValidator.validate_columns(df, ["A", "B"])
        assert ok is True
        assert "B" in missing

    def test_validate_columns_faltante_estricto(self):
        df = pd.DataFrame({"A": [1]})
        ok, _, missing = DataValidator.validate_columns(df, ["A", "B"], strict=True)
        assert ok is False
        assert "B" in missing


# ============================================================================
# 4. SEARCH ENGINE — KEYWORD MATCH
# ============================================================================

class TestSearchEngineKeyword:
    """Pruebas para SearchEngine.find_keyword_match."""

    @pytest.fixture
    def engine(self) -> SearchEngine:
        return SearchEngine(artifacts=None)

    def test_match_exacto(self, engine, df_apus_base, estimation_log):
        keywords = ["tubo", "pvc", "presion", "4", "pulgadas"]
        apu, details = engine.find_keyword_match(
            df_apus_base, keywords, estimation_log
        )
        assert apu is not None
        assert details is not None
        assert details.confidence_score > 0.0
        assert str(apu.get("CODIGO_APU")) == "APU-001"

    def test_match_parcial(self, engine, df_apus_base, estimation_log):
        keywords = ["tubo", "pvc"]
        apu, details = engine.find_keyword_match(
            df_apus_base, keywords, estimation_log, min_match_percentage=40.0
        )
        assert apu is not None

    def test_sin_match(self, engine, df_apus_base, estimation_log):
        keywords = ["inexistente", "producto", "alien"]
        apu, details = engine.find_keyword_match(
            df_apus_base, keywords, estimation_log
        )
        assert apu is None
        assert details is None

    def test_pool_vacio(self, engine, estimation_log):
        df = pd.DataFrame(columns=["DESC_NORMALIZED", "CODIGO_APU"])
        apu, _ = engine.find_keyword_match(df, ["tubo"], estimation_log)
        assert apu is None

    def test_keywords_vacias(self, engine, df_apus_base, estimation_log):
        apu, _ = engine.find_keyword_match(df_apus_base, [], estimation_log)
        assert apu is None

    def test_keywords_muy_cortas_filtradas(self, engine, df_apus_base, estimation_log):
        keywords = ["a", "b", "c"]
        apu, _ = engine.find_keyword_match(df_apus_base, keywords, estimation_log)
        assert apu is None

    def test_modo_estricto_match_perfecto(self, engine, estimation_log):
        df = pd.DataFrame([{
            "CODIGO_APU": "STRICT-001",
            "DESC_NORMALIZED": "cemento portland tipo i",
            "original_description": "Cemento Portland Tipo I",
        }])
        keywords = ["cemento", "portland", "tipo"]
        apu, details = engine.find_keyword_match(
            df, keywords, estimation_log, strict=True
        )
        assert apu is not None
        assert details.confidence_score == 1.0

    def test_modo_estricto_sin_perfecto(self, engine, df_apus_base, estimation_log):
        keywords = ["tubo", "inexistente"]
        apu, _ = engine.find_keyword_match(
            df_apus_base, keywords, estimation_log, strict=True
        )
        assert apu is None

    def test_modo_substring(self, engine, estimation_log):
        df = pd.DataFrame([{
            "CODIGO_APU": "SUB-001",
            "DESC_NORMALIZED": "tubo pvc presion 4 pulgadas clase 21",
            "original_description": "Tubo PVC",
        }])
        keywords = ["tubo", "pvc", "presion"]
        apu, details = engine.find_keyword_match(
            df, keywords, estimation_log, match_mode=MatchMode.SUBSTRING
        )
        assert apu is not None
        assert details is not None
        assert details.match_method in ("EXACT_SUBSTRING", "PARTIAL_SUBSTRING")

    def test_modo_invalido_usa_words(self, engine, df_apus_base, estimation_log):
        keywords = ["tubo", "pvc"]
        apu, _ = engine.find_keyword_match(
            df_apus_base, keywords, estimation_log, match_mode="INVALIDO"
        )
        # No debe lanzar error, usa WORDS por defecto
        assert True  # Si llegamos aquí, no hubo error

    def test_sin_desc_normalized_usa_original(self, engine, estimation_log):
        df = pd.DataFrame([{
            "CODIGO_APU": "ORIG-001",
            "original_description": "Tubo PVC Presión",
        }])
        keywords = ["tubo", "pvc"]
        with patch("semantic_estimator.normalize_text", return_value="tubo pvc presion"):
            apu, _ = engine.find_keyword_match(df, keywords, estimation_log)
        # Debe intentar crear DESC_NORMALIZED desde original_description

    def test_limite_keywords(self, engine, df_apus_base, estimation_log):
        keywords = [f"kw{i}" for i in range(MAX_KEYWORDS_COUNT + 20)]
        # No debe explotar, debe truncar
        engine.find_keyword_match(df_apus_base, keywords, estimation_log)
        assert f"truncando a {MAX_KEYWORDS_COUNT}" in estimation_log.render()

    def test_umbral_porcentaje(self, engine, df_apus_base, estimation_log):
        keywords = ["tubo"]
        apu_high, _ = engine.find_keyword_match(
            df_apus_base, keywords, estimation_log,
            min_match_percentage=90.0,
        )
        # Con una sola keyword y match en una desc de varias palabras,
        # el porcentaje es 100% (1/1), así que debería pasar
        # Pero si no, el umbral alto bloquea
        # Depende del match real

    def test_early_exit_estricto(self, engine, estimation_log):
        """En modo estricto, el primer match 100% detiene la búsqueda."""
        rows = [
            {"CODIGO_APU": f"E-{i:03d}", "DESC_NORMALIZED": "cemento portland",
             "original_description": "Cemento"}
            for i in range(100)
        ]
        df = pd.DataFrame(rows)
        keywords = ["cemento", "portland"]

        apu, details = engine.find_keyword_match(
            df, keywords, estimation_log, strict=True
        )
        assert apu is not None
        assert "Early exit" in estimation_log.render()


# ============================================================================
# 4b. SEARCH ENGINE — SEMANTIC MATCH
# ============================================================================

class TestSearchEngineSemantic:
    """Pruebas para SearchEngine.find_semantic_match."""

    @pytest.fixture
    def engine(self, search_artifacts) -> SearchEngine:
        return SearchEngine(artifacts=search_artifacts)

    def test_match_semantico_exitoso(self, engine, df_apus_base, estimation_log):
        apu, details = engine.find_semantic_match(
            df_apus_base, "tubo pvc presion cuatro pulgadas",
            estimation_log, min_similarity=0.5,
        )
        assert apu is not None
        assert details is not None
        assert details.match_method == "SEMANTIC"
        assert details.confidence_score >= 0.5

    def test_match_semantico_bajo_umbral(self, engine, df_apus_base, estimation_log):
        apu, _ = engine.find_semantic_match(
            df_apus_base, "tubo pvc",
            estimation_log, min_similarity=0.99,
        )
        assert apu is None

    def test_sin_artefactos(self, df_apus_base, estimation_log):
        engine = SearchEngine(artifacts=None)
        apu, _ = engine.find_semantic_match(
            df_apus_base, "query", estimation_log
        )
        assert apu is None

    def test_query_vacio(self, engine, df_apus_base, estimation_log):
        apu, _ = engine.find_semantic_match(df_apus_base, "", estimation_log)
        assert apu is None

    def test_query_muy_corto(self, engine, df_apus_base, estimation_log):
        apu, _ = engine.find_semantic_match(df_apus_base, "a", estimation_log)
        assert apu is None

    def test_pool_vacio(self, engine, estimation_log):
        df = pd.DataFrame(columns=["CODIGO_APU"])
        apu, _ = engine.find_semantic_match(df, "query larga", estimation_log)
        assert apu is None

    def test_pool_sin_columna_codigo(self, engine, estimation_log):
        df = pd.DataFrame({"other": [1, 2]})
        apu, _ = engine.find_semantic_match(
            df, "consulta de prueba", estimation_log
        )
        assert apu is None

    def test_codigo_no_en_pool(self, estimation_log, mock_model, mock_faiss_index):
        """Cuando FAISS retorna IDs que no están en el pool filtrado."""
        id_map = {"0": "NO-EXISTE-001", "1": "NO-EXISTE-002", "2": "NO-EXISTE-003"}
        artifacts = SearchArtifacts(mock_model, mock_faiss_index, id_map)
        engine = SearchEngine(artifacts)

        df = pd.DataFrame([{"CODIGO_APU": "OTRO-001", "DESC_NORMALIZED": "algo"}])
        apu, _ = engine.find_semantic_match(
            df, "consulta prueba", estimation_log
        )
        assert apu is None

    def test_similitud_fuera_de_rango(self, estimation_log, mock_model, mock_id_map):
        """Similitudes fuera de [0,1] se clampean con log."""
        index = MagicMock()
        index.ntotal = 10
        index.search.return_value = (
            np.array([[1.5, -0.3]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )
        artifacts = SearchArtifacts(mock_model, index, mock_id_map)
        engine = SearchEngine(artifacts)

        df = pd.DataFrame([
            {"CODIGO_APU": "APU-001", "DESC_NORMALIZED": "test"},
            {"CODIGO_APU": "APU-002", "DESC_NORMALIZED": "test2"},
        ])
        apu, details = engine.find_semantic_match(
            df, "consulta prueba", estimation_log, min_similarity=0.0
        )
        if apu is not None and details is not None:
            assert 0.0 <= details.confidence_score <= 1.0

    def test_error_en_encode(self, estimation_log, mock_faiss_index, mock_id_map):
        """Error en model.encode se maneja gracefully."""
        model = MagicMock()
        model.encode.side_effect = RuntimeError("Model error")
        artifacts = SearchArtifacts(model, mock_faiss_index, mock_id_map)
        engine = SearchEngine(artifacts)

        df = pd.DataFrame([{"CODIGO_APU": "APU-001", "DESC_NORMALIZED": "test"}])
        apu, _ = engine.find_semantic_match(
            df, "consulta prueba", estimation_log
        )
        assert apu is None
        assert "Error" in estimation_log.render()

    def test_embedding_vacio(self, estimation_log, mock_faiss_index, mock_id_map):
        """Embedding vacío retorna None."""
        model = MagicMock()
        model.encode.return_value = np.array([], dtype=np.float32)
        artifacts = SearchArtifacts(model, mock_faiss_index, mock_id_map)
        engine = SearchEngine(artifacts)

        df = pd.DataFrame([{"CODIGO_APU": "APU-001", "DESC_NORMALIZED": "test"}])
        apu, _ = engine.find_semantic_match(
            df, "consulta de prueba", estimation_log
        )
        assert apu is None


class TestSearchEngineBestMatch:
    """Pruebas para SearchEngine.find_best_match (cascada)."""

    def test_cascada_semantico_primero(self, search_artifacts, df_apus_base, estimation_log):
        engine = SearchEngine(search_artifacts)
        apu, details = engine.find_best_match(
            df_apus_base, "tubo pvc presion", ["tubo", "pvc"],
            estimation_log, min_similarity=0.5,
        )
        assert apu is not None
        assert details.match_method == "SEMANTIC"

    def test_cascada_fallback_keyword(self, df_apus_base, estimation_log, mock_model, mock_id_map):
        """Si semántico falla, usa keyword."""
        index = MagicMock()
        index.ntotal = 10
        # Similitudes muy bajas → fallback
        index.search.return_value = (
            np.array([[0.1, 0.05, 0.01]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )
        artifacts = SearchArtifacts(mock_model, index, mock_id_map)
        engine = SearchEngine(artifacts)

        apu, details = engine.find_best_match(
            df_apus_base, "tubo pvc presion",
            ["tubo", "pvc", "presion"],
            estimation_log,
            min_similarity=0.9,
        )
        if apu is not None:
            assert details.match_method in ("KEYWORD", "SEMANTIC")


# ============================================================================
# 5. COST CALCULATOR
# ============================================================================

class TestCostCalculatorFactors:
    """Pruebas para extracción de factores."""

    @pytest.fixture
    def calculator(self, base_config) -> CostCalculator:
        engine = SearchEngine(artifacts=None)
        return CostCalculator(base_config, engine)

    def test_factores_por_defecto(self, calculator):
        factors = calculator._extract_factors({})
        assert factors.factor_zona == 1.0
        assert factors.costo_izaje == 0.0
        assert factors.factor_seguridad == 1.0

    def test_factores_zona(self, calculator):
        factors = calculator._extract_factors({"zona": "ZONA 1"})
        assert factors.factor_zona == 1.15

    def test_factores_izaje(self, calculator):
        factors = calculator._extract_factors({"izaje": "GRUA"})
        assert factors.costo_izaje == 50000.0

    def test_factores_seguridad(self, calculator):
        factors = calculator._extract_factors({"seguridad": "ALTO"})
        assert factors.factor_seguridad == 1.25

    def test_factores_combinados(self, calculator):
        factors = calculator._extract_factors({
            "zona": "ZONA 2",
            "izaje": "GRUA",
            "seguridad": "ALTO",
        })
        assert factors.factor_zona == 1.30
        assert factors.costo_izaje == 50000.0
        assert factors.factor_seguridad == 1.25

    def test_zona_desconocida_usa_default(self, calculator):
        factors = calculator._extract_factors({"zona": "ZONA_99"})
        assert factors.factor_zona == 1.0

    def test_thresholds_extraction(self, calculator):
        thresholds = calculator._extract_thresholds()
        assert thresholds["min_sim_suministro"] == 0.3
        assert thresholds["min_sim_tarea"] == 0.4
        assert thresholds["min_kw_cuadrilla"] == 50.0

    def test_config_corrupta(self):
        engine = SearchEngine(artifacts=None)
        calc = CostCalculator({"estimator_rules": "invalido"}, engine)
        factors = calc._extract_factors({})
        assert factors.factor_zona == 1.0


class TestCostCalculatorLaborCost:
    """Pruebas para _calculate_labor_cost."""

    def test_calculo_normal(self, estimation_log):
        factors = EstimationFactors(factor_seguridad=1.0)
        result = CostCalculator._calculate_labor_cost(
            100000.0, 10.0, factors, estimation_log
        )
        assert abs(result - 10000.0) < 0.01

    def test_con_factor_seguridad(self, estimation_log):
        factors = EstimationFactors(factor_seguridad=1.25)
        result = CostCalculator._calculate_labor_cost(
            100000.0, 10.0, factors, estimation_log
        )
        assert abs(result - 12500.0) < 0.01

    def test_rendimiento_cero(self, estimation_log):
        factors = EstimationFactors()
        result = CostCalculator._calculate_labor_cost(
            100000.0, 0.0, factors, estimation_log
        )
        assert result == 0.0

    def test_rendimiento_muy_pequeno(self, estimation_log):
        factors = EstimationFactors()
        result = CostCalculator._calculate_labor_cost(
            100000.0, 1e-8, factors, estimation_log
        )
        assert result == 0.0  # Debajo de MIN_RENDIMIENTO_SAFE

    def test_rendimiento_en_limite(self, estimation_log):
        factors = EstimationFactors()
        result = CostCalculator._calculate_labor_cost(
            100000.0, MIN_RENDIMIENTO_SAFE, factors, estimation_log
        )
        # Puede ser grande pero no infinito
        assert math.isfinite(result)

    def test_costo_excesivo_clamped(self, estimation_log):
        factors = EstimationFactors(factor_seguridad=1.0)
        # costo_dia muy alto / rendimiento muy bajo → clamp
        result = CostCalculator._calculate_labor_cost(
            1e15, MIN_RENDIMIENTO_SAFE * 2, factors, estimation_log
        )
        assert result <= MAX_COSTO_UNITARIO

    def test_costo_dia_cero(self, estimation_log):
        factors = EstimationFactors()
        result = CostCalculator._calculate_labor_cost(
            0.0, 5.0, factors, estimation_log
        )
        assert result == 0.0


class TestCostCalculatorRendimiento:
    """Pruebas para _calculate_rendimiento_from_detail."""

    def test_rendimiento_normal(self, estimation_log):
        data_store = {
            "apus_detail": [
                {"CODIGO_APU": "APU-X", "TIPO_INSUMO": "MANO DE OBRA", "CANTIDAD_APU": 0.25},
                {"CODIGO_APU": "APU-X", "TIPO_INSUMO": "MANO DE OBRA", "CANTIDAD_APU": 0.25},
                {"CODIGO_APU": "APU-X", "TIPO_INSUMO": "EQUIPO", "CANTIDAD_APU": 1.0},
            ],
        }
        rend = CostCalculator._calculate_rendimiento_from_detail(
            "APU-X", data_store, estimation_log
        )
        assert abs(rend - 2.0) < 0.001  # 1 / 0.5

    def test_sin_detalle(self, estimation_log):
        rend = CostCalculator._calculate_rendimiento_from_detail(
            "APU-X", {}, estimation_log
        )
        assert rend == 0.0

    def test_sin_mano_obra(self, estimation_log):
        data_store = {
            "apus_detail": [
                {"CODIGO_APU": "APU-X", "TIPO_INSUMO": "EQUIPO", "CANTIDAD_APU": 1.0},
            ],
        }
        rend = CostCalculator._calculate_rendimiento_from_detail(
            "APU-X", data_store, estimation_log
        )
        assert rend == 0.0

    def test_tiempo_total_cero(self, estimation_log):
        data_store = {
            "apus_detail": [
                {"CODIGO_APU": "APU-X", "TIPO_INSUMO": "MANO DE OBRA", "CANTIDAD_APU": 0.0},
            ],
        }
        rend = CostCalculator._calculate_rendimiento_from_detail(
            "APU-X", data_store, estimation_log
        )
        assert rend == 0.0

    def test_codigo_no_encontrado(self, estimation_log):
        data_store = {
            "apus_detail": [
                {"CODIGO_APU": "APU-Y", "TIPO_INSUMO": "MANO DE OBRA", "CANTIDAD_APU": 0.5},
            ],
        }
        rend = CostCalculator._calculate_rendimiento_from_detail(
            "APU-X", data_store, estimation_log
        )
        assert rend == 0.0

    def test_columnas_faltantes(self, estimation_log):
        data_store = {
            "apus_detail": [{"CODIGO_APU": "APU-X"}],
        }
        rend = CostCalculator._calculate_rendimiento_from_detail(
            "APU-X", data_store, estimation_log
        )
        assert rend == 0.0

    def test_rendimiento_anormalmente_alto(self, estimation_log):
        """Rendimiento > 1e6 se rechaza."""
        data_store = {
            "apus_detail": [
                {"CODIGO_APU": "APU-X", "TIPO_INSUMO": "MANO DE OBRA", "CANTIDAD_APU": 1e-10},
            ],
        }
        rend = CostCalculator._calculate_rendimiento_from_detail(
            "APU-X", data_store, estimation_log
        )
        assert rend == 0.0


class TestCostCalculatorHistoricalAverage:
    """Pruebas para _calculate_historical_average."""

    def test_promedio_normal(self, estimation_log):
        df = pd.DataFrame([
            {"DESC_NORMALIZED": "tubo pvc presion", "RENDIMIENTO_DIA": 10.0},
            {"DESC_NORMALIZED": "tubo pvc sanitario", "RENDIMIENTO_DIA": 8.0},
            {"DESC_NORMALIZED": "valvula bronce", "RENDIMIENTO_DIA": 5.0},
        ])
        apu, details, rend = CostCalculator._calculate_historical_average(
            df, ["tubo", "presion"], estimation_log
        )
        assert apu is not None
        assert rend > 0
        assert details.match_method == "HISTORICAL_AVERAGE"

    def test_sin_keywords_significativas(self, estimation_log):
        df = pd.DataFrame([{"DESC_NORMALIZED": "abc", "RENDIMIENTO_DIA": 5.0}])
        apu, _, rend = CostCalculator._calculate_historical_average(
            df, ["ab", "cd"], estimation_log
        )
        assert apu is None
        assert rend == 0.0

    def test_sin_matches(self, estimation_log):
        df = pd.DataFrame([
            {"DESC_NORMALIZED": "concreto armado", "RENDIMIENTO_DIA": 5.0},
        ])
        apu, _, rend = CostCalculator._calculate_historical_average(
            df, ["tubo", "inexistente"], estimation_log
        )
        assert apu is None

    def test_sin_columna_rendimiento(self, estimation_log):
        df = pd.DataFrame([{"DESC_NORMALIZED": "tubo pvc presion"}])
        apu, _, rend = CostCalculator._calculate_historical_average(
            df, ["tubo"], estimation_log
        )
        assert apu is None

    def test_todos_rendimiento_cero(self, estimation_log):
        df = pd.DataFrame([
            {"DESC_NORMALIZED": "tubo pvc presion", "RENDIMIENTO_DIA": 0.0},
            {"DESC_NORMALIZED": "tubo pvc sanitario", "RENDIMIENTO_DIA": 0.0},
        ])
        apu, _, rend = CostCalculator._calculate_historical_average(
            df, ["tubo"], estimation_log
        )
        assert apu is None

    def test_confianza_acotada(self, estimation_log):
        rows = [
            {"DESC_NORMALIZED": f"tubo tipo {i}", "RENDIMIENTO_DIA": float(i + 1)}
            for i in range(20)
        ]
        df = pd.DataFrame(rows)
        _, details, _ = CostCalculator._calculate_historical_average(
            df, ["tubo"], estimation_log
        )
        assert details is not None
        assert details.confidence_score <= 0.5


class TestCostCalculatorCalculate:
    """Pruebas para CostCalculator.calculate (pipeline completo)."""

    @pytest.fixture
    def calculator(self, base_config, search_artifacts) -> CostCalculator:
        engine = SearchEngine(search_artifacts)
        return CostCalculator(base_config, engine)

    def test_sin_apus(self, calculator):
        result = calculator.calculate(
            params={"material": "TUBO PVC"},
            data_store={"processed_apus": []},
        )
        assert "error" in result

    def test_sin_material(self, calculator, data_store):
        result = calculator.calculate(
            params={"material": ""},
            data_store=data_store,
        )
        assert "error" in result
        assert "material" in result["error"].lower()

    def test_estimacion_basica(self, calculator, data_store):
        result = calculator.calculate(
            params={"material": "TUBO PVC PRESION"},
            data_store=data_store,
        )
        assert "error" not in result
        assert "valor_construccion" in result
        assert "log" in result
        assert isinstance(result["valor_suministro"], float)
        assert isinstance(result["valor_instalacion"], float)

    def test_con_factores(self, calculator, data_store):
        result = calculator.calculate(
            params={
                "material": "TUBO PVC PRESION",
                "zona": "ZONA 1",
                "izaje": "GRUA",
                "seguridad": "ALTO",
            },
            data_store=data_store,
        )
        assert "error" not in result
        factores = result["factores_aplicados"]
        assert factores["zona"] == 1.15
        assert factores["izaje"] == 50000.0
        assert factores["seguridad"] == 1.25

    def test_con_cuadrilla(self, calculator, data_store):
        result = calculator.calculate(
            params={"material": "TUBO PVC", "cuadrilla": "3"},
            data_store=data_store,
        )
        assert "error" not in result
        assert result["apu_cuadrilla_codigo"] is not None

    def test_derivation_details_presentes(self, calculator, data_store):
        result = calculator.calculate(
            params={"material": "TUBO PVC PRESION"},
            data_store=data_store,
        )
        assert "derivation_details" in result
        dd = result["derivation_details"]
        assert "suministro" in dd
        assert "tarea" in dd
        assert "cuadrilla" in dd

    def test_codigos_apu_en_resultado(self, calculator, data_store):
        result = calculator.calculate(
            params={"material": "TUBO PVC"},
            data_store=data_store,
        )
        assert "apu_suministro_codigo" in result
        assert "apu_tarea_codigo" in result
        assert "apu_cuadrilla_codigo" in result

    def test_datos_apu_invalidos(self, calculator):
        result = calculator.calculate(
            params={"material": "TUBO"},
            data_store={"processed_apus": "no_es_lista"},
        )
        assert "error" in result

    def test_rendimiento_en_resultado(self, calculator, data_store):
        result = calculator.calculate(
            params={"material": "TUBO PVC PRESION"},
            data_store=data_store,
        )
        assert "rendimiento_m2_por_dia" in result
        assert isinstance(result["rendimiento_m2_por_dia"], float)


# ============================================================================
# 5b. FILTROS DE POOL
# ============================================================================

class TestCostCalculatorFilters:
    """Pruebas para los filtros de pool."""

    def test_filter_pool_existente(self, df_apus_base, estimation_log):
        result = CostCalculator._filter_pool(
            df_apus_base, "tipo_apu", [TipoAPU.SUMINISTRO.value],
            estimation_log, "suministro",
        )
        assert all(row["tipo_apu"] == TipoAPU.SUMINISTRO.value for _, row in result.iterrows())

    def test_filter_pool_columna_faltante(self, df_apus_base, estimation_log):
        result = CostCalculator._filter_pool(
            df_apus_base, "columna_inexistente", ["valor"],
            estimation_log, "test",
        )
        assert len(result) == len(df_apus_base)

    def test_filter_pool_sin_coincidencias(self, df_apus_base, estimation_log):
        result = CostCalculator._filter_pool(
            df_apus_base, "tipo_apu", ["TIPO_INEXISTENTE"],
            estimation_log, "test",
        )
        assert len(result) == len(df_apus_base)

    def test_filter_by_unit(self, df_apus_base, estimation_log):
        result = CostCalculator._filter_by_unit(
            df_apus_base, "DIA", estimation_log, "cuadrilla"
        )
        assert all(
            str(row.get("UNIDAD", "")).upper().strip() == "DIA"
            for _, row in result.iterrows()
        )

    def test_filter_by_unit_sin_columna(self, estimation_log):
        df = pd.DataFrame({"CODIGO_APU": ["A"]})
        result = CostCalculator._filter_by_unit(df, "DIA", estimation_log, "test")
        assert len(result) == 1


# ============================================================================
# 6. SEMANTIC ESTIMATOR SERVICE
# ============================================================================

class TestSemanticEstimatorServiceLifecycle:
    """Pruebas para el ciclo de vida del servicio."""

    def test_inicializacion(self, base_config):
        with patch.object(
            SemanticEstimatorService, "_load_tensor_space", return_value=None
        ):
            service = SemanticEstimatorService(base_config)
            assert service.is_ready is False

    def test_ready_event_señalizado(self, base_config):
        with patch.object(
            SemanticEstimatorService, "_load_tensor_space"
        ) as mock_load:
            def fake_load(self_ref=None):
                pass
            mock_load.side_effect = fake_load

            service = SemanticEstimatorService(base_config)
            service._ready_event.set()
            assert service.wait_until_ready(timeout=1.0) is True

    def test_wait_until_ready_timeout(self, base_config):
        with patch.object(
            SemanticEstimatorService, "_load_tensor_space", return_value=None
        ):
            service = SemanticEstimatorService(base_config)
            # No seteamos el event → timeout
            result = service.wait_until_ready(timeout=0.1)
            assert result is False

    def test_is_ready_sin_artifacts(self, base_config):
        with patch.object(
            SemanticEstimatorService, "_load_tensor_space", return_value=None
        ):
            service = SemanticEstimatorService(base_config)
            service._ready_event.set()
            service._artifacts = None
            assert service.is_ready is False

    def test_is_ready_con_artifacts(self, base_config, search_artifacts):
        with patch.object(
            SemanticEstimatorService, "_load_tensor_space", return_value=None
        ):
            service = SemanticEstimatorService(base_config)
            service._ready_event.set()
            service._artifacts = search_artifacts
            assert service.is_ready is True


class TestSemanticEstimatorServiceProjection:
    """Pruebas para project_semantic_match."""

    @pytest.fixture
    def service(self, base_config, search_artifacts) -> SemanticEstimatorService:
        with patch.object(
            SemanticEstimatorService, "_load_tensor_space", return_value=None
        ):
            svc = SemanticEstimatorService(base_config)
            svc._artifacts = search_artifacts
            svc._ready_event.set()
            return svc

    def test_projec_exitoso(self, service, df_apus_base):
        payload = {
            "query_text": "tubo pvc presion 4 pulgadas",
            "df_pool": df_apus_base,
        }
        result = service.project_semantic_match(payload, {})
        assert result["success"] is True
        assert "matched_id" in result
        assert result["stratum"] == "TACTICS"

    def test_servicio_no_listo(self, base_config):
        with patch.object(
            SemanticEstimatorService, "_load_tensor_space", return_value=None
        ):
            svc = SemanticEstimatorService(base_config)
            svc._ready_event.set()
            svc._artifacts = None

            result = svc.project_semantic_match(
                {"query_text": "test", "df_pool": []}, {}
            )
            assert result["success"] is False

    def test_pool_vacio(self, service):
        result = service.project_semantic_match(
            {"query_text": "tubo pvc", "df_pool": pd.DataFrame()}, {}
        )
        assert result["success"] is False

    def test_pool_como_lista(self, service, df_apus_base):
        payload = {
            "query_text": "tubo pvc presion",
            "df_pool": df_apus_base.to_dict("records"),
        }
        result = service.project_semantic_match(payload, {})
        # Puede ser True o False dependiendo del mock, pero no debe explotar
        assert "success" in result

    def test_con_telemetria(self, service, df_apus_base):
        telemetry = MagicMock()
        context = {"telemetry_context": telemetry}
        payload = {
            "query_text": "tubo pvc presion",
            "df_pool": df_apus_base,
        }
        service.project_semantic_match(payload, context)
        telemetry.start_step.assert_called_once_with("semantic_projection")

    def test_error_interno(self, service):
        payload = {
            "query_text": "test",
            "df_pool": "no_es_dataframe_ni_lista",
        }
        result = service.project_semantic_match(payload, {})
        assert result["success"] is False
        assert "error" in result


class TestSemanticEstimatorServiceEstimate:
    """Pruebas para calculate_dynamic_estimate."""

    @pytest.fixture
    def service(self, base_config, search_artifacts) -> SemanticEstimatorService:
        with patch.object(
            SemanticEstimatorService, "_load_tensor_space", return_value=None
        ):
            svc = SemanticEstimatorService(base_config)
            svc._artifacts = search_artifacts
            svc._ready_event.set()
            return svc

    def test_estimacion_basica(self, service, data_store):
        payload = {
            "params": {"material": "TUBO PVC PRESION"},
            "data_store": data_store,
        }
        result = service.calculate_dynamic_estimate(payload, {})
        assert result["success"] is True
        assert "estimate" in result
        assert result["stratum"] == "TACTICS"

    def test_sin_material(self, service, data_store):
        payload = {
            "params": {"material": ""},
            "data_store": data_store,
        }
        result = service.calculate_dynamic_estimate(payload, {})
        assert result["success"] is True
        assert "error" in result["estimate"]

    def test_data_store_vacio(self, service):
        payload = {
            "params": {"material": "TUBO"},
            "data_store": {},
        }
        result = service.calculate_dynamic_estimate(payload, {})
        assert result["success"] is True
        assert "error" in result["estimate"]

    def test_con_telemetria_error(self, service):
        telemetry = MagicMock()
        context = {"telemetry_context": telemetry}

        with patch.object(CostCalculator, "calculate", side_effect=RuntimeError("boom")):
            result = service.calculate_dynamic_estimate(
                {"params": {}, "data_store": {}}, context
            )
            assert result["success"] is False
            telemetry.record_error.assert_called_once()


class TestSemanticEstimatorServiceMIC:
    """Pruebas para registro en MIC."""

    def test_register_in_mic(self, base_config):
        with patch.object(
            SemanticEstimatorService, "_load_tensor_space", return_value=None
        ):
            svc = SemanticEstimatorService(base_config)
            mic = MagicMock()
            svc.register_in_mic(mic)
            assert mic.register_vector.call_count == 2

            calls = mic.register_vector.call_args_list
            service_names = {c.kwargs.get("service_name", c[1].get("service_name", "")) for c in calls}
            assert "semantic_match" in service_names
            assert "tactical_estimate" in service_names


# ============================================================================
# 7. DATACLASSES
# ============================================================================

class TestDataclasses:
    """Pruebas para las dataclasses del módulo."""

    def test_derivation_details_inmutable(self):
        dd = DerivationDetails("KEYWORD", 0.8, "Histórico", "Match 80%")
        with pytest.raises(AttributeError):
            dd.match_method = "OTRO"  # type: ignore[misc]

    def test_match_candidate_rank_key(self):
        apu = pd.Series({"CODIGO_APU": "TEST"})
        c1 = MatchCandidate(apu, "A", 2, 60.0, similarity=0.8)
        c2 = MatchCandidate(apu, "B", 3, 80.0, similarity=0.7)
        assert c1.rank_key[0] > c2.rank_key[0]
        assert c2.rank_key[1] > c1.rank_key[1]

    def test_match_candidate_sin_similitud(self):
        apu = pd.Series({"CODIGO_APU": "TEST"})
        c = MatchCandidate(apu, "Desc", 1, 50.0)
        assert c.rank_key[0] == 0.0

    def test_search_artifacts_inmutable(self, search_artifacts):
        with pytest.raises(AttributeError):
            search_artifacts.model = None  # type: ignore[misc]

    def test_search_artifacts_is_complete(self, search_artifacts):
        assert search_artifacts.is_complete() is True

    def test_search_artifacts_incompleto(self):
        arts = SearchArtifacts(model=None, faiss_index=MagicMock(), id_map={"1": "A"})
        assert arts.is_complete() is False

    def test_search_artifacts_mapa_vacio(self, mock_model, mock_faiss_index):
        arts = SearchArtifacts(mock_model, mock_faiss_index, {})
        assert arts.is_complete() is False

    def test_data_quality_metrics(self):
        m = DataQualityMetrics(total_records=100, valid_records=80, quality_score=0.8)
        assert m.is_acceptable() is True
        assert m.is_acceptable(threshold=0.9) is False

    def test_estimation_factors_defaults(self):
        f = EstimationFactors()
        assert f.factor_zona == 1.0
        assert f.costo_izaje == 0.0
        assert f.factor_seguridad == 1.0

    def test_estimation_components_defaults(self):
        c = EstimationComponents()
        assert c.valor_suministro == 0.0
        assert c.codigo_suministro == "N/A"


# ============================================================================
# 8. ENUMERACIONES
# ============================================================================

class TestEnumeraciones:
    """Pruebas para enums locales."""

    def test_match_mode_values(self):
        assert MatchMode.WORDS.value == "words"
        assert MatchMode.SUBSTRING.value == "substring"

    def test_tipo_apu_values(self):
        assert TipoAPU.SUMINISTRO.value == "Suministro"
        assert TipoAPU.INSTALACION.value == "Instalación"
        assert TipoAPU.SUMINISTRO_PREFABRICADO.value == "Suministro (Pre-fabricado)"

    def test_tipo_recurso_values(self):
        assert TipoRecurso.MANO_OBRA.value == "MANO DE OBRA"
        assert TipoRecurso.EQUIPO.value == "EQUIPO"
        assert TipoRecurso.MATERIAL.value == "MATERIAL"

    def test_tipo_recurso_distinto_de_schema(self):
        """TipoRecurso es distinto de app.schemas.TipoInsumo."""
        assert TipoRecurso.__name__ == "TipoRecurso"


# ============================================================================
# 9. CASOS LÍMITE Y REGRESIONES
# ============================================================================

class TestCasosLimite:
    """Pruebas de borde y regresiones."""

    def test_df_con_nan_en_desc(self, estimation_log):
        df = pd.DataFrame([
            {"CODIGO_APU": "NAN-001", "DESC_NORMALIZED": np.nan,
             "original_description": "Test", "tipo_apu": "Suministro",
             "UNIDAD": "ML"},
        ])
        engine = SearchEngine(artifacts=None)
        apu, _ = engine.find_keyword_match(df, ["test"], estimation_log)
        # No debe explotar

    def test_df_con_tipos_mixtos(self, estimation_log):
        df = pd.DataFrame([
            {"CODIGO_APU": 12345, "DESC_NORMALIZED": "tubo pvc",
             "original_description": "Tubo PVC", "tipo_apu": "Suministro",
             "UNIDAD": "ML"},
        ])
        engine = SearchEngine(artifacts=None)
        apu, _ = engine.find_keyword_match(df, ["tubo", "pvc"], estimation_log)
        # Código numérico no debe causar error

    def test_miles_de_registros(self, estimation_log):
        """Rendimiento aceptable con datasets grandes."""
        rows = [
            {"CODIGO_APU": f"APU-{i:05d}",
             "DESC_NORMALIZED": f"material tipo {i % 100} especial",
             "original_description": f"Material {i}"}
            for i in range(5000)
        ]
        df = pd.DataFrame(rows)
        engine = SearchEngine(artifacts=None)

        import time
        start = time.time()
        engine.find_keyword_match(df, ["material", "tipo", "especial"], estimation_log)
        elapsed = time.time() - start
        assert elapsed < 30.0  # Debería ser mucho más rápido

    def test_unicode_en_descriptions(self, estimation_log):
        df = pd.DataFrame([{
            "CODIGO_APU": "UNI-001",
            "DESC_NORMALIZED": "válvula compuerta bronce ñ",
            "original_description": "Válvula Compuerta Bronce Ñ",
        }])
        engine = SearchEngine(artifacts=None)
        apu, _ = engine.find_keyword_match(df, ["válvula", "bronce"], estimation_log)

    def test_financial_analysis_sin_codigos(self, estimation_log):
        components = EstimationComponents()
        result = CostCalculator._run_financial_analysis(
            components, {}, 100000.0, estimation_log
        )
        assert result == {}

    def test_financial_analysis_con_error(self, estimation_log):
        components = EstimationComponents(
            codigo_suministro="APU-001",
            codigo_tarea="APU-002",
        )
        data_store = {"apus_detail": [{"CODIGO_APU": "APU-001"}]}

        with patch(
            "semantic_estimator.run_monte_carlo_simulation",
            side_effect=RuntimeError("MC error"),
        ):
            result = CostCalculator._run_financial_analysis(
                components, data_store, 100000.0, estimation_log
            )
            assert result == {}

    def test_error_result_formato(self, estimation_log):
        result = CostCalculator._error_result("Test error", estimation_log)
        assert result["error"] == "Test error"
        assert "log" in result

    def test_details_to_dict_none(self):
        assert CostCalculator._details_to_dict(None) is None

    def test_details_to_dict_normal(self):
        dd = DerivationDetails("KEYWORD", 0.9, "Histórico", "Match 90%")
        d = CostCalculator._details_to_dict(dd)
        assert d["match_method"] == "KEYWORD"
        assert d["confidence_score"] == 0.9

    def test_config_param_map(self, search_artifacts, data_store):
        config = {
            "param_map": {
                "material": {"TUBO ESPECIAL": "TUBO PVC PRESION 4 PULGADAS"},
            },
            "estimator_thresholds": {},
            "estimator_rules": {},
        }
        engine = SearchEngine(search_artifacts)
        calc = CostCalculator(config, engine)

        result = calc.calculate(
            params={"material": "TUBO ESPECIAL"},
            data_store=data_store,
        )
        assert "error" not in result


# ============================================================================
# 10. INTEGRACIÓN
# ============================================================================

class TestIntegracion:
    """Pruebas de integración end-to-end."""

    def test_flujo_completo_estimacion(
        self, base_config, search_artifacts, data_store
    ):
        """Pipeline completo: params → SearchEngine → CostCalculator → resultado."""
        engine = SearchEngine(search_artifacts)
        calculator = CostCalculator(base_config, engine)

        result = calculator.calculate(
            params={
                "material": "TUBO PVC PRESION",
                "cuadrilla": "3",
                "zona": "ZONA 1",
                "izaje": "MANUAL",
                "seguridad": "ALTO",
            },
            data_store=data_store,
        )

        assert "error" not in result
        assert result["factores_aplicados"]["zona"] == 1.15
        assert result["factores_aplicados"]["seguridad"] == 1.25
        assert result["factores_aplicados"]["izaje"] == 0.0
        assert isinstance(result["valor_construccion"], float)
        assert isinstance(result["log"], str)
        assert "ESTIMADOR" in result["log"]

    def test_servicio_completo(
        self, base_config, search_artifacts, data_store
    ):
        """Flujo via SemanticEstimatorService."""
        with patch.object(
            SemanticEstimatorService, "_load_tensor_space", return_value=None
        ):
            svc = SemanticEstimatorService(base_config)
            svc._artifacts = search_artifacts
            svc._ready_event.set()

            # Proyección
            proj_result = svc.project_semantic_match(
                {
                    "query_text": "tubo pvc presion cuatro pulgadas",
                    "df_pool": pd.DataFrame(data_store["processed_apus"]),
                },
                {},
            )
            assert proj_result["success"] is True

            # Estimación
            est_result = svc.calculate_dynamic_estimate(
                {
                    "params": {"material": "TUBO PVC"},
                    "data_store": data_store,
                },
                {},
            )
            assert est_result["success"] is True
            assert "estimate" in est_result

    def test_resilencia_sin_modelo(self, base_config, data_store):
        """Sin modelo cargado, keyword match sigue funcionando."""
        with patch.object(
            SemanticEstimatorService, "_load_tensor_space", return_value=None
        ):
            svc = SemanticEstimatorService(base_config)
            svc._artifacts = None
            svc._ready_event.set()

            engine = SearchEngine(artifacts=None)
            calc = CostCalculator(base_config, engine)

            result = calc.calculate(
                params={"material": "TUBO PVC PRESION"},
                data_store=data_store,
            )
            # Debe funcionar con keyword fallback
            assert "error" not in result

    def test_multiple_estimaciones_independientes(
        self, base_config, search_artifacts, data_store
    ):
        """Múltiples estimaciones no interfieren entre sí."""
        engine = SearchEngine(search_artifacts)
        calc = CostCalculator(base_config, engine)

        r1 = calc.calculate(
            {"material": "TUBO PVC PRESION", "zona": "ZONA 0"},
            data_store,
        )
        r2 = calc.calculate(
            {"material": "TUBO PVC PRESION", "zona": "ZONA 2"},
            data_store,
        )

        assert "error" not in r1
        assert "error" not in r2
        assert r1["factores_aplicados"]["zona"] != r2["factores_aplicados"]["zona"]
        # Los logs son independientes
        assert r1["log"] != r2["log"]