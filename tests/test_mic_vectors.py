"""
Suite de Pruebas Funcional para Vectores MIC (Adaptadores).

Objetivo:
    Validar que la capa de adaptación (Morfismos) traduzca correctamente
    las intenciones de la MIC hacia los componentes nucleares (Flux, Parser, Processor).

Cobertura:
    1. vector_stabilize_flux (PHYSICS)
    2. vector_parse_raw_structure (PHYSICS)
    3. vector_structure_logic (TACTICS)
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.schemas import Stratum

# Importar el módulo a testear
# Asegúrate de que la ruta de importación coincida con tu estructura
from app.adapters import mic_vectors

# ============================================================================
# FIXTURES (Datos de Prueba)
# ============================================================================

@pytest.fixture
def sample_config():
    return {
        "system_capacitance": 100.0,
        "max_batch_size": 50,
        "validation_thresholds": {"min_price": 100}
    }

@pytest.fixture
def sample_profile():
    return {"parser_strategy": "regex_v2"}

@pytest.fixture
def mock_dataframe():
    return pd.DataFrame([{"col1": 1, "col2": 2}])

# ============================================================================
# TESTS: VECTOR FÍSICO (FluxCondenser)
# ============================================================================

@patch("app.adapters.mic_vectors.DataFluxCondenser")
def test_vector_stabilize_flux_success(MockCondenser, sample_config, mock_dataframe):
    """
    Verifica que el vector de estabilización instancie el condensador,
    ejecute stabilize y retorne métricas físicas.
    """
    # 1. Configurar el Mock
    instance = MockCondenser.return_value
    instance.stabilize.return_value = mock_dataframe
    instance.get_physics_report.return_value = {"entropy": 0.5, "pressure": 10.0}

    # 2. Ejecutar el Vector
    result = mic_vectors.vector_stabilize_flux(
        file_path="/tmp/test.csv",
        config=sample_config
    )

    # 3. Aserciones (Contrato MIC)
    assert result["success"] is True
    assert result["stratum"] == Stratum.PHYSICS  # [Fuente 1123] Debe ser Físico
    assert "data" in result
    assert result["physics_metrics"]["entropy"] == 0.5
    
    # Verificar que se llamó con la configuración física
    MockCondenser.assert_called_once()
    instance.stabilize.assert_called_with("/tmp/test.csv")

@patch("app.adapters.mic_vectors.DataFluxCondenser")
def test_vector_stabilize_flux_failure(MockCondenser, sample_config):
    """
    Verifica el manejo de errores (excepciones atrapadas y convertidas a dict).
    """
    # 1. Simular explosión física (ej. inestabilidad RHP)
    MockCondenser.side_effect = Exception("RHP Polos inestables detectados")

    # 2. Ejecutar
    result = mic_vectors.vector_stabilize_flux(
        file_path="/tmp/bad_file.csv",
        config=sample_config
    )

    # 3. Validar "Fail Gracefully"
    assert result["success"] is False
    assert "RHP Polos inestables" in result["error"]

# ============================================================================
# TESTS: VECTOR TOPOLÓGICO (ReportParser)
# ============================================================================

@patch("app.adapters.mic_vectors.ReportParserCrudo")
def test_vector_parse_raw_structure_success(MockParser, sample_profile, sample_config):
    """
    Verifica que el parser sea invocado y se extraiga el cache para el siguiente paso.
    Nota: Esto asume que la 'Intervención Quirúrgica' agregó parse_to_raw.
    """
    # 1. Configurar Mock
    instance = MockParser.return_value
    # Simular registros crudos
    instance.parse_to_raw.return_value = [{"line": "Item 1", "id": 1}]
    # Simular cache de Lark (vital para optimización)
    instance.get_parse_cache.return_value = {"tree_hash_1": "cached_tree"}
    instance.validation_stats = MagicMock() # Mockear objeto stats

    # 2. Ejecutar
    result = mic_vectors.vector_parse_raw_structure(
        file_path="/tmp/apus.pdf",
        profile=sample_profile
    )

    # 3. Validar Contrato
    assert result["success"] is True
    assert result["stratum"] == Stratum.PHYSICS
    assert len(result["raw_records"]) == 1
    # Verificar que el cache se extrae (Critical Path para TACTICS)
    assert result["parse_cache"] == {"tree_hash_1": "cached_tree"}
    
    instance.parse_to_raw.assert_called_once()

# ============================================================================
# TESTS: VECTOR TÁCTICO (APUProcessor)
# ============================================================================

@patch("app.adapters.mic_vectors.APUProcessor")
def test_vector_structure_logic_injection(MockProcessor, sample_config, mock_dataframe):
    """
    Verifica la INYECCIÓN DE DEPENDENCIA: El vector táctico no debe leer archivos,
    debe recibir los datos pre-procesados por PHYSICS.
    """
    # 1. Configurar Mock
    instance = MockProcessor.return_value
    instance.process_all.return_value = mock_dataframe
    instance.get_quality_report.return_value = {"score": 98.5}

    # Datos que vienen del paso anterior (PHYSICS)
    raw_records_input = [{"id": 1, "desc": "Concreto"}]
    parse_cache_input = {"hash123": "tree_obj"}

    # 2. Ejecutar
    result = mic_vectors.vector_structure_logic(
        raw_records=raw_records_input,
        parse_cache=parse_cache_input,
        config=sample_config
    )

    # 3. Validar Inyección
    # El procesador debe inicializarse con el cache
    MockProcessor.assert_called_once_with(config=sample_config, parse_cache=parse_cache_input)
    
    # Los registros crudos deben inyectarse directamente (Bypass de I/O)
    assert instance.raw_records == raw_records_input
    
    # 4. Validar Salida
    assert result["success"] is True
    assert result["stratum"] == Stratum.TACTICS # [Fuente 1140]
    assert result["processed_data"] == mock_dataframe.to_dict("records")

@patch("app.adapters.mic_vectors.APUProcessor")
def test_vector_structure_logic_algebraic_fail(MockProcessor, sample_config):
    """
    Verifica fallo en cálculo táctico (ej. error de mónada).
    """
    instance = MockProcessor.return_value
    instance.process_all.side_effect = RuntimeError("Violación de Homogeneidad Algebraica")

    result = mic_vectors.vector_structure_logic(
        raw_records=[], 
        parse_cache={}, 
        config=sample_config
    )

    assert result["success"] is False
    assert "Violación de Homogeneidad" in result["error"]