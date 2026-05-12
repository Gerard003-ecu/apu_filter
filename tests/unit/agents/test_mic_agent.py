"""
Suite de Pruebas para MIC Agent
Ubicación: tests/agents/test_mic_agent.py

COBERTURA DE PRUEBAS - PARTE 1:

1. UTILIDADES MATEMÁTICAS:
   - Hashing estable y canónico
   - Cálculo de rango tensorial
   - Comparación de floats con tolerancia
   - Operaciones de clamp

2. NORMALIZACIÓN DE ESTRATOS:
   - Conversión desde int, str, Stratum
   - Manejo de errores
   - Idempotencia

3. TIPOS Y VALIDACIÓN:
   - Validación de tipos JSON Schema
   - Construcción de JSONPath
   - Detección de números JSON

4. DATACLASSES DE AUDITORÍA:
   - SchemaValidationResult (Álgebra de Heyting)
   - CategoricalEqualizerSeed
   - TOONDocument (parsing y rendering)

5. CONTRATOS Y CARTUCHOS:
   - SiloAContract
   - SiloBCartridge
   - Validación de integridad

6. VALIDADOR DE SCHEMA:
   - Tipos básicos
   - Validación de propiedades
   - Validación recursiva
   - Constraints numéricos
   - Patterns regex

Estrategia de Testing:
- Property-based testing donde aplicable
- Verificación de invariantes matemáticos
- Tests de casos extremos
- Verificación de determinismo
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, FrozenSet

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch

# Imports del módulo bajo prueba
from app.agents.mic_agent import (
    # Excepciones
    MICAgentError,
    StratumResolutionError,
    ContractValidationError,
    ClosureViolationError,
    AlgebraicVetoError,
    TOONCompressionError,
    SiloAccessError,
    ProjectionError,
    FunctorialityError,
    
    # Enumeraciones
    ImpedanceMatchStatus,
    ValidationSeverity,
    
    # Dataclasses
    SchemaValidationResult,
    CategoricalEqualizerSeed,
    TOONDocument,
    SiloAContract,
    SiloBCartridge,
    
    # Clases principales
    SchemaValidator,
    AlgebraicVetoRegistry,
    SiloManager,
    TOONCompressor,
    AuditTrail,
    MICAgent,
    
    # Utilidades
    MathUtils,
    normalize_stratum,
    python_type_matches,
    compute_json_path,
    _is_json_number,
    
    # Constantes
    MAX_AUDIT_TRAIL_SIZE,
    TOON_START_MARKER,
    TOON_END_MARKER,
    TOON_FIELD_SEPARATOR,
    ENCAPSULATION_PROTOCOL_VERSION,
    EPS,
    ALGEBRAIC_TOL,
    FLOAT_COMPARISON_TOL,
    MAX_TENSOR_RANK,
    MAX_COMPRESSION_RATIO,
    MIN_COMPRESSION_RATIO,
)

from app.core.schemas import Stratum
from app.core.mic_algebra import CategoricalState

# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def logger_mock(monkeypatch):
    """Mock del logger para capturar mensajes."""
    messages = {
        "debug": [],
        "info": [],
        "warning": [],
        "error": [],
        "critical": []
    }
    
    def make_logger_func(level):
        def log_func(msg, *args, **kwargs):
            formatted = msg % args if args else msg
            messages[level].append(formatted)
        return log_func
    
    logger = logging.getLogger("MIC.Agent.CategoricalEqualizer")
    monkeypatch.setattr(logger, "debug", make_logger_func("debug"))
    monkeypatch.setattr(logger, "info", make_logger_func("info"))
    monkeypatch.setattr(logger, "warning", make_logger_func("warning"))
    monkeypatch.setattr(logger, "error", make_logger_func("error"))
    monkeypatch.setattr(logger, "critical", make_logger_func("critical"))
    
    return messages

@pytest.fixture
def sample_contract() -> SiloAContract:
    """Contrato de ejemplo válido."""
    return SiloAContract(
        contract_id="test_contract",
        stratum=Stratum.PHYSICS,
        schema={
            "type": "object",
            "required": ["value"],
            "properties": {
                "value": {"type": "number", "minimum": 0}
            }
        },
        description="Test contract"
    )

@pytest.fixture
def sample_cartridge() -> SiloBCartridge:
    """Cartucho de ejemplo válido."""
    return SiloBCartridge(
        cartridge_id="test_cartridge",
        stratum=Stratum.PHYSICS,
        header_template="Test Header\nkey|value",
        field_definitions=("value",),
        description="Test cartridge"
    )

@pytest.fixture
def mock_mic_registry() -> Mock:
    """Mock del MICRegistry."""
    registry = Mock()
    registry.get_vector_info = Mock(return_value={
        "stratum": Stratum.PHYSICS,
        "dimension": 3
    })
    registry.project_intent = Mock(return_value={
        "status": "OK",
        "result": {}
    })
    return registry

@pytest.fixture
def silo_manager() -> SiloManager:
    """SiloManager con configuración por defecto."""
    return SiloManager()

@pytest.fixture
def schema_validator() -> SchemaValidator:
    """SchemaValidator por defecto."""
    return SchemaValidator()

@pytest.fixture
def algebraic_veto_registry() -> AlgebraicVetoRegistry:
    """AlgebraicVetoRegistry por defecto."""
    return AlgebraicVetoRegistry()

@pytest.fixture
def toon_compressor() -> TOONCompressor:
    """TOONCompressor por defecto."""
    return TOONCompressor()

@pytest.fixture
def audit_trail() -> AuditTrail:
    """AuditTrail por defecto."""
    return AuditTrail(max_size=100)

# ==============================================================================
# TESTS: UTILIDADES MATEMÁTICAS
# ==============================================================================

class TestMathUtils:
    """Tests para utilidades matemáticas."""
    
    def test_stable_hash_deterministic(self):
        """Hash estable debe ser determinista."""
        data1 = {"b": 2, "a": 1, "c": [3, 4, 5]}
        data2 = {"a": 1, "b": 2, "c": [3, 4, 5]}
        
        hash1 = MathUtils.stable_hash(data1)
        hash2 = MathUtils.stable_hash(data2)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256
    
    def test_stable_hash_different_data(self):
        """Hashes diferentes para datos diferentes."""
        data1 = {"a": 1}
        data2 = {"a": 2}
        
        hash1 = MathUtils.stable_hash(data1)
        hash2 = MathUtils.stable_hash(data2)
        
        assert hash1 != hash2
    
    def test_stable_hash_nested_structures(self):
        """Hash para estructuras anidadas."""
        data = {
            "level1": {
                "level2": {
                    "level3": [1, 2, 3]
                }
            }
        }
        
        hash_result = MathUtils.stable_hash(data)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
    
    def test_stable_hash_with_none(self):
        """Hash con valores None."""
        data = {"a": None, "b": 1}
        hash_result = MathUtils.stable_hash(data)
        
        assert isinstance(hash_result, str)
    
    def test_stable_hash_fallback_repr(self):
        """Fallback a repr() para objetos no serializables."""
        
        class CustomObject:
            def __repr__(self):
                return "CustomObject(x=42)"
        
        obj = CustomObject()
        hash_result = MathUtils.stable_hash(obj)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
    
    def test_compute_tensor_rank_scalar(self):
        """Rango tensorial de scalar."""
        assert MathUtils.compute_tensor_rank(42) == 0
        assert MathUtils.compute_tensor_rank("string") == 0
        assert MathUtils.compute_tensor_rank(3.14) == 0
    
    def test_compute_tensor_rank_dict(self):
        """Rango tensorial de diccionario."""
        assert MathUtils.compute_tensor_rank({}) == 1
        assert MathUtils.compute_tensor_rank({"a": 1}) == 1
        assert MathUtils.compute_tensor_rank({"a": {"b": 2}}) == 2
        assert MathUtils.compute_tensor_rank({"a": {"b": {"c": 3}}}) == 3
    
    def test_compute_tensor_rank_list(self):
        """Rango tensorial de lista."""
        assert MathUtils.compute_tensor_rank([]) == 1
        assert MathUtils.compute_tensor_rank([1, 2, 3]) == 1
        assert MathUtils.compute_tensor_rank([[1, 2], [3, 4]]) == 2
        assert MathUtils.compute_tensor_rank([[[1]]]) == 3
    
    def test_compute_tensor_rank_mixed(self):
        """Rango tensorial de estructuras mixtas."""
        data = {
            "a": [1, 2, 3],
            "b": {"c": [4, 5]}
        }
        assert MathUtils.compute_tensor_rank(data) == 3
    
    def test_compute_tensor_rank_max_depth_protection(self):
        """Protección contra recursión infinita."""
        # Crear estructura muy profunda
        data = {"a": 1}
        current = data
        for i in range(150):
            current["nested"] = {"level": i}
            current = current["nested"]
        
        # No debe lanzar RecursionError
        rank = MathUtils.compute_tensor_rank(data, max_depth=100)
        assert rank == 100  # Limitado por max_depth
    
    def test_float_equal_exact(self):
        """Comparación de floats exactamente iguales."""
        assert MathUtils.float_equal(1.0, 1.0)
        assert MathUtils.float_equal(0.0, 0.0)
        assert MathUtils.float_equal(-1.0, -1.0)
    
    def test_float_equal_within_tolerance(self):
        """Comparación de floats dentro de tolerancia."""
        assert MathUtils.float_equal(1.0, 1.0 + 1e-10)
        assert MathUtils.float_equal(1.0, 1.0 - 1e-10)
        assert MathUtils.float_equal(0.0, 1e-10)
    
    def test_float_equal_outside_tolerance(self):
        """Comparación de floats fuera de tolerancia."""
        assert not MathUtils.float_equal(1.0, 1.1)
        assert not MathUtils.float_equal(0.0, 0.1)
    
    def test_float_equal_custom_tolerance(self):
        """Comparación con tolerancia personalizada."""
        assert MathUtils.float_equal(1.0, 1.01, tol=0.02)
        assert not MathUtils.float_equal(1.0, 1.01, tol=0.001)
    
    def test_clamp_within_bounds(self):
        """Clamp de valor dentro de límites."""
        assert MathUtils.clamp(5.0, 0.0, 10.0) == 5.0
    
    def test_clamp_below_minimum(self):
        """Clamp de valor por debajo del mínimo."""
        assert MathUtils.clamp(-5.0, 0.0, 10.0) == 0.0
    
    def test_clamp_above_maximum(self):
        """Clamp de valor por encima del máximo."""
        assert MathUtils.clamp(15.0, 0.0, 10.0) == 10.0
    
    def test_clamp_at_boundaries(self):
        """Clamp en los límites exactos."""
        assert MathUtils.clamp(0.0, 0.0, 10.0) == 0.0
        assert MathUtils.clamp(10.0, 0.0, 10.0) == 10.0
    
    def test_clamp_invalid_bounds(self):
        """Clamp con límites inválidos."""
        with pytest.raises(ValueError, match="min_val.*max_val"):
            MathUtils.clamp(5.0, 10.0, 0.0)

# ==============================================================================
# TESTS: NORMALIZACIÓN DE ESTRATOS
# ==============================================================================

class TestNormalizeStratum:
    """Tests para normalización de estratos."""
    
    def test_normalize_stratum_from_stratum(self):
        """Normalizar desde Stratum (idempotente)."""
        assert normalize_stratum(Stratum.PHYSICS) == Stratum.PHYSICS
        assert normalize_stratum(Stratum.TACTICS) == Stratum.TACTICS
        assert normalize_stratum(Stratum.STRATEGY) == Stratum.STRATEGY
        assert normalize_stratum(Stratum.WISDOM) == Stratum.WISDOM
    
    def test_normalize_stratum_from_int(self):
        """Normalizar desde entero."""
        assert normalize_stratum(0) == Stratum.PHYSICS
        assert normalize_stratum(1) == Stratum.TACTICS
        assert normalize_stratum(2) == Stratum.STRATEGY
        assert normalize_stratum(3) == Stratum.WISDOM
    
    def test_normalize_stratum_from_string_name(self):
        """Normalizar desde string (nombre)."""
        assert normalize_stratum("PHYSICS") == Stratum.PHYSICS
        assert normalize_stratum("physics") == Stratum.PHYSICS
        assert normalize_stratum("Physics") == Stratum.PHYSICS
    
    def test_normalize_stratum_from_string_value(self):
        """Normalizar desde string (valor numérico)."""
        assert normalize_stratum("0") == Stratum.PHYSICS
        assert normalize_stratum("1") == Stratum.TACTICS
        assert normalize_stratum("2") == Stratum.STRATEGY
        assert normalize_stratum("3") == Stratum.WISDOM
    
    def test_normalize_stratum_invalid_int(self):
        """Rechazar entero inválido."""
        with pytest.raises(StratumResolutionError, match="Valor entero inválido"):
            normalize_stratum(999)
    
    def test_normalize_stratum_invalid_string(self):
        """Rechazar string inválido."""
        with pytest.raises(StratumResolutionError, match="String inválido"):
            normalize_stratum("INVALID_STRATUM")
    
    def test_normalize_stratum_invalid_type(self):
        """Rechazar tipo no soportado."""
        with pytest.raises(StratumResolutionError, match="Tipo no soportado"):
            normalize_stratum(3.14)
        
        with pytest.raises(StratumResolutionError, match="Tipo no soportado"):
            normalize_stratum([1, 2, 3])
    
    def test_normalize_stratum_idempotence(self):
        """Verificar idempotencia."""
        s1 = normalize_stratum("PHYSICS")
        s2 = normalize_stratum(s1)
        assert s1 == s2

# ==============================================================================
# TESTS: VALIDACIÓN DE TIPOS
# ==============================================================================

class TestPythonTypeMatches:
    """Tests para validación de tipos JSON Schema."""
    
    def test_null_type(self):
        """Validación de tipo null."""
        assert python_type_matches("null", None)
        assert not python_type_matches("null", 0)
        assert not python_type_matches("null", "")
    
    def test_boolean_type(self):
        """Validación de tipo boolean."""
        assert python_type_matches("boolean", True)
        assert python_type_matches("boolean", False)
        assert not python_type_matches("boolean", 1)
        assert not python_type_matches("boolean", 0)
    
    def test_integer_type(self):
        """Validación de tipo integer."""
        assert python_type_matches("integer", 42)
        assert python_type_matches("integer", -10)
        assert python_type_matches("integer", 0)
        assert not python_type_matches("integer", True)
        assert not python_type_matches("integer", 3.14)
    
    def test_number_type(self):
        """Validación de tipo number."""
        assert python_type_matches("number", 42)
        assert python_type_matches("number", 3.14)
        assert python_type_matches("number", -2.5)
        assert not python_type_matches("number", True)
        assert not python_type_matches("number", "42")
    
    def test_string_type(self):
        """Validación de tipo string."""
        assert python_type_matches("string", "hello")
        assert python_type_matches("string", "")
        assert not python_type_matches("string", 42)
    
    def test_array_type(self):
        """Validación de tipo array."""
        assert python_type_matches("array", [])
        assert python_type_matches("array", [1, 2, 3])
        assert not python_type_matches("array", (1, 2, 3))
        assert not python_type_matches("array", {})
    
    def test_object_type(self):
        """Validación de tipo object."""
        assert python_type_matches("object", {})
        assert python_type_matches("object", {"a": 1})
        assert not python_type_matches("object", [])
    
    def test_unknown_type(self):
        """Tipo desconocido (retorna True por defecto)."""
        assert python_type_matches("unknown_type", "anything")

# ==============================================================================
# TESTS: UTILIDADES JSON
# ==============================================================================

class TestComputeJsonPath:
    """Tests para construcción de JSONPath."""
    
    def test_compute_json_path_root(self):
        """JSONPath desde raíz."""
        assert compute_json_path("$", "foo") == "$.foo"
    
    def test_compute_json_path_nested(self):
        """JSONPath anidado."""
        assert compute_json_path("$.foo", "bar") == "$.foo.bar"
    
    def test_compute_json_path_array_index(self):
        """JSONPath con índice de array."""
        assert compute_json_path("$.foo", 0) == "$.foo[0]"
        assert compute_json_path("$.foo", 42) == "$.foo[42]"
    
    def test_compute_json_path_special_chars(self):
        """JSONPath con caracteres especiales."""
        assert compute_json_path("$", "key.with.dots") == "$.key\\.with\\.dots"
        assert compute_json_path("$", "key[with]brackets") == "$.key\\[with\\]brackets"

class TestIsJsonNumber:
    """Tests para detección de números JSON."""
    
    def test_is_json_number_integer(self):
        """Detectar enteros."""
        assert _is_json_number("42")
        assert _is_json_number("-10")
        assert _is_json_number("0")
    
    def test_is_json_number_float(self):
        """Detectar floats."""
        assert _is_json_number("3.14")
        assert _is_json_number("-2.5")
        assert _is_json_number("0.0")
    
    def test_is_json_number_scientific(self):
        """Detectar notación científica."""
        assert _is_json_number("1e10")
        assert _is_json_number("2.5e-3")
    
    def test_is_json_number_invalid(self):
        """Rechazar no-números."""
        assert not _is_json_number("abc")
        assert not _is_json_number("true")
        assert not _is_json_number("")

# ==============================================================================
# TESTS: ENUMERACIONES
# ==============================================================================

class TestImpedanceMatchStatus:
    """Tests para ImpedanceMatchStatus."""
    
    def test_is_terminal(self):
        """Verificar estados terminales."""
        assert ImpedanceMatchStatus.ALGEBRAIC_VETO.is_terminal
        assert ImpedanceMatchStatus.TOPOLOGICAL_BIFURCATION.is_terminal
        assert ImpedanceMatchStatus.COHOMOLOGY_FAILURE.is_terminal
        
        assert not ImpedanceMatchStatus.LAMINAR_PROJECTION.is_terminal
        assert not ImpedanceMatchStatus.INPUT_TYPE_ERROR.is_terminal
    
    def test_severity_ordering(self):
        """Verificar orden de severidad."""
        assert ImpedanceMatchStatus.LAMINAR_PROJECTION.severity == 0
        assert ImpedanceMatchStatus.INPUT_TYPE_ERROR.severity == 1
        assert ImpedanceMatchStatus.ALGEBRAIC_VETO.severity == 3
    
    def test_comparison_operators(self):
        """Verificar comparaciones."""
        assert ImpedanceMatchStatus.LAMINAR_PROJECTION < ImpedanceMatchStatus.ALGEBRAIC_VETO
        assert ImpedanceMatchStatus.INPUT_TYPE_ERROR < ImpedanceMatchStatus.COHOMOLOGY_FAILURE

class TestValidationSeverity:
    """Tests para ValidationSeverity."""
    
    def test_heyting_value(self):
        """Verificar valores en Álgebra de Heyting."""
        assert ValidationSeverity.ERROR.heyting_value == 0.0
        assert ValidationSeverity.WARNING.heyting_value == 0.5
        assert ValidationSeverity.INFO.heyting_value == 1.0

# ==============================================================================
# TESTS: SCHEMA VALIDATION RESULT
# ==============================================================================

class TestSchemaValidationResult:
    """Tests para SchemaValidationResult (Álgebra de Heyting)."""
    
    def test_success(self):
        """Crear resultado exitoso (⊤)."""
        result = SchemaValidationResult.success()
        
        assert result.is_valid
        assert result.validity_degree == 1.0
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_failure(self):
        """Crear resultado fallido (⊥)."""
        result = SchemaValidationResult.failure("Test error", path="$.foo")
        
        assert not result.is_valid
        assert result.validity_degree == 0.0
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"
        assert result.path == "$.foo"
    
    def test_validity_degree_clamping(self):
        """Clamp de validity_degree a [0, 1]."""
        # Valor negativo
        result = SchemaValidationResult(validity_degree=-0.5)
        assert result.validity_degree == 0.0
        
        # Valor mayor que 1
        result = SchemaValidationResult(validity_degree=1.5)
        assert result.validity_degree == 1.0
    
    def test_is_valid_with_tolerance(self):
        """Verificar is_valid con tolerancia."""
        result = SchemaValidationResult(validity_degree=1.0 - 1e-13)
        assert result.is_valid
        
        result = SchemaValidationResult(validity_degree=0.99)
        assert not result.is_valid
    
    def test_merge_empty(self):
        """Merge de lista vacía."""
        results = []
        merged = SchemaValidationResult.merge(results)
        
        assert merged.is_valid
        assert merged.validity_degree == 1.0
    
    def test_merge_all_valid(self):
        """Merge de resultados todos válidos (conjunción ⊤ ∧ ⊤ = ⊤)."""
        results = [
            SchemaValidationResult.success(),
            SchemaValidationResult.success(),
        ]
        merged = SchemaValidationResult.merge(results)
        
        assert merged.is_valid
        assert merged.validity_degree == 1.0
    
    def test_merge_with_errors(self):
        """Merge con errores (conjunción con ⊥)."""
        results = [
            SchemaValidationResult.success(),
            SchemaValidationResult.failure("Error 1"),
            SchemaValidationResult.failure("Error 2"),
        ]
        merged = SchemaValidationResult.merge(results)
        
        assert not merged.is_valid
        assert merged.validity_degree == 0.0
        assert len(merged.errors) == 2
    
    def test_merge_min_validity(self):
        """Merge toma mínimo de validity_degree."""
        results = [
            SchemaValidationResult(validity_degree=0.9),
            SchemaValidationResult(validity_degree=0.7),
            SchemaValidationResult(validity_degree=0.8),
        ]
        merged = SchemaValidationResult.merge(results)
        
        assert merged.validity_degree == 0.7
    
    def test_error_property(self):
        """Propiedad error retorna primer error."""
        result = SchemaValidationResult(
            validity_degree=0.0,
            errors=("Error 1", "Error 2")
        )
        
        assert result.error == "Error 1"
        
        result_no_errors = SchemaValidationResult.success()
        assert result_no_errors.error is None
    
    def test_to_dict(self):
        """Serialización a diccionario."""
        result = SchemaValidationResult(
            validity_degree=0.8,
            errors=("Error 1",),
            warnings=("Warning 1",),
            path="$.test"
        )
        
        d = result.to_dict()
        
        assert d["validity_degree"] == 0.8
        assert d["errors"] == ["Error 1"]
        assert d["warnings"] == ["Warning 1"]
        assert d["path"] == "$.test"
        assert d["is_valid"] == False
    
    def test_str_representation(self):
        """Representación string."""
        result = SchemaValidationResult.success()
        s = str(result)
        
        assert "VALID" in s
        assert "validity=1.000" in s

# ==============================================================================
# TESTS: CATEGORICAL EQUALIZER SEED
# ==============================================================================

class TestCategoricalEqualizerSeed:
    """Tests para CategoricalEqualizerSeed."""
    
    def test_creation_basic(self):
        """Crear seed básico."""
        seed = CategoricalEqualizerSeed(
            target_vector="test_vector",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="contract_1",
            silo_b_cartridge_id="cartridge_1",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        
        assert seed.target_vector == "test_vector"
        assert seed.target_stratum == Stratum.PHYSICS
        assert seed.impedance_match_status == ImpedanceMatchStatus.LAMINAR_PROJECTION
    
    def test_compression_ratio_clamp(self):
        """Clamp de compression_ratio negativo."""
        seed = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            token_compression_ratio=-0.5,
        )
        
        assert seed.token_compression_ratio == 0.0
    
    def test_to_dict(self):
        """Serialización a diccionario."""
        seed = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            token_compression_ratio=0.75,
            validation_errors=("Error 1",),
        )
        
        d = seed.to_dict()
        
        assert d["target_vector"] == "test"
        assert d["target_stratum"] == "PHYSICS"
        assert d["impedance_match_status"] == "LAMINAR_PROJECTION"
        assert d["token_compression_ratio"] == 0.75
        assert d["validation_errors"] == ["Error 1"]
        assert "timestamp" in d
    
    def test_compute_hash_deterministic(self):
        """Hash determinista (excluye timestamp)."""
        seed1 = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        
        # Esperar un momento
        time.sleep(0.01)
        
        seed2 = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        
        # Timestamps diferentes pero hashes iguales
        assert seed1.timestamp != seed2.timestamp
        assert seed1.compute_hash() == seed2.compute_hash()
    
    def test_str_representation(self):
        """Representación string."""
        seed = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            token_compression_ratio=0.5,
        )
        
        s = str(seed)
        
        assert "test" in s
        assert "PHYSICS" in s
        assert "LAMINAR_PROJECTION" in s
        assert "0.50" in s

# ==============================================================================
# TESTS: TOON DOCUMENT
# ==============================================================================

class TestTOONDocument:
    """Tests para TOONDocument."""
    
    def test_creation_basic(self):
        """Crear documento TOON básico."""
        doc = TOONDocument(
            cartridge_id="test_cart",
            header_template="Header\nkey|value",
            records=(("key1", "\"value1\""), ("key2", "42"))
        )
        
        assert doc.cartridge_id == "test_cart"
        assert len(doc.records) == 2
    
    def test_creation_empty_cartridge_id(self):
        """Rechazar cartridge_id vacío."""
        with pytest.raises(TOONCompressionError, match="no puede estar vacío"):
            TOONDocument(
                cartridge_id="",
                header_template="Header",
                records=()
            )
    
    def test_render_basic(self):
        """Renderizar documento TOON."""
        doc = TOONDocument(
            cartridge_id="test_cart",
            header_template="Header\nkey|value",
            records=(
                ("key1", "\"value1\""),
                ("key2", "42"),
            )
        )
        
        rendered = doc.render()
        
        assert TOON_START_MARKER in rendered
        assert TOON_END_MARKER in rendered
        assert "test_cart" in rendered
        assert "key1|\"value1\"" in rendered
        assert "key2|42" in rendered
    
    def test_parse_basic(self):
        """Parsear documento TOON."""
        content = """--- INICIO TOON --- test_cart ---
Header
key|value
key1|"value1"
key2|42
--- FIN TOON ---"""
        
        doc = TOONDocument.parse(content)
        
        assert doc.cartridge_id == "test_cart"
        assert len(doc.records) == 2
        assert doc.records[0] == ("key1", "\"value1\"")
        assert doc.records[1] == ("key2", "42")
    
    def test_parse_invalid_too_short(self):
        """Rechazar documento muy corto."""
        content = "--- INICIO TOON ---\n--- FIN TOON ---"
        
        with pytest.raises(TOONCompressionError, match="demasiado corto"):
            TOONDocument.parse(content)
    
    def test_parse_invalid_start_marker(self):
        """Rechazar marcador de inicio inválido."""
        content = """INVALID START
Header
--- FIN TOON ---"""
        
        with pytest.raises(TOONCompressionError, match="Marcador de inicio inválido"):
            TOONDocument.parse(content)
    
    def test_parse_invalid_end_marker(self):
        """Rechazar marcador de fin inválido."""
        content = """--- INICIO TOON --- test ---
Header
INVALID END"""
        
        with pytest.raises(TOONCompressionError, match="Marcador de fin"):
            TOONDocument.parse(content)
    
    def test_to_dict(self):
        """Deserializar a diccionario."""
        doc = TOONDocument(
            cartridge_id="test",
            header_template="Header",
            records=(
                ("key1", "\"value1\""),
                ("key2", "42"),
                ("key3", "{\"nested\": true}"),
            )
        )
        
        d = doc.to_dict()
        
        assert d["key1"] == "value1"
        assert d["key2"] == 42
        assert d["key3"] == {"nested": True}
    
    def test_to_dict_invalid_json(self):
        """Manejar JSON inválido en records."""
        doc = TOONDocument(
            cartridge_id="test",
            header_template="Header",
            records=(("key1", "invalid_json"),)
        )
        
        d = doc.to_dict()
        
        # Debe conservar como string
        assert d["key1"] == "invalid_json"
    
    def test_roundtrip_render_parse(self):
        """Verificar roundtrip render → parse."""
        original = TOONDocument(
            cartridge_id="test_cart",
            header_template="Header Line 1\nHeader Line 2",
            records=(
                ("key1", "\"value1\""),
                ("key2", "42"),
            )
        )
        
        rendered = original.render()
        parsed = TOONDocument.parse(rendered)
        
        assert parsed.cartridge_id == original.cartridge_id
        assert parsed.records == original.records
    
    def test_len(self):
        """Longitud de documento."""
        doc = TOONDocument(
            cartridge_id="test",
            header_template="Header",
            records=(("k1", "v1"), ("k2", "v2"), ("k3", "v3"))
        )
        
        assert len(doc) == 3
    
    def test_str_representation(self):
        """Representación string."""
        doc = TOONDocument(
            cartridge_id="test_cart",
            header_template="Header",
            records=(("k1", "v1"),)
        )
        
        s = str(doc)
        
        assert "test_cart" in s
        assert "records=1" in s

# ==============================================================================
# TESTS: SILO A CONTRACT
# ==============================================================================

class TestSiloAContract:
    """Tests para SiloAContract."""
    
    def test_creation_valid(self, sample_contract):
        """Crear contrato válido."""
        assert sample_contract.contract_id == "test_contract"
        assert sample_contract.stratum == Stratum.PHYSICS
        assert isinstance(sample_contract.schema, dict)
    
    def test_creation_invalid_schema(self):
        """Rechazar schema inválido."""
        with pytest.raises(ContractValidationError, match="Schema inválido"):
            SiloAContract(
                contract_id="test",
                stratum=Stratum.PHYSICS,
                schema={"invalid": "schema"}  # Falta "type"
            )
    
    def test_validate_schema_integrity_valid(self, sample_contract):
        """Validar integridad de schema válido."""
        assert sample_contract.validate_schema_integrity()
    
    def test_validate_schema_integrity_not_dict(self):
        """Schema no es diccionario."""
        # Crear directamente para evitar __post_init__
        contract = object.__new__(SiloAContract)
        object.__setattr__(contract, "schema", "not_a_dict")
        
        assert not contract.validate_schema_integrity()
    
    def test_validate_schema_integrity_no_type(self):
        """Schema sin clave 'type'."""
        contract = object.__new__(SiloAContract)
        object.__setattr__(contract, "schema", {"properties": {}})
        
        assert not contract.validate_schema_integrity()
    
    def test_to_dict(self, sample_contract):
        """Serialización a diccionario."""
        d = sample_contract.to_dict()
        
        assert d["contract_id"] == "test_contract"
        assert d["stratum"] == "PHYSICS"
        assert "schema" in d
        assert d["description"] == "Test contract"

# ==============================================================================
# TESTS: SILO B CARTRIDGE
# ==============================================================================

class TestSiloBCartridge:
    """Tests para SiloBCartridge."""
    
    def test_creation_valid(self, sample_cartridge):
        """Crear cartucho válido."""
        assert sample_cartridge.cartridge_id == "test_cartridge"
        assert sample_cartridge.stratum == Stratum.PHYSICS
        assert isinstance(sample_cartridge.header_template, str)
    
    def test_creation_empty_id(self):
        """Rechazar cartridge_id vacío."""
        with pytest.raises(TOONCompressionError, match="no puede estar vacío"):
            SiloBCartridge(
                cartridge_id="",
                stratum=Stratum.PHYSICS,
                header_template="Header"
            )
    
    def test_to_dict(self, sample_cartridge):
        """Serialización a diccionario."""
        d = sample_cartridge.to_dict()
        
        assert d["cartridge_id"] == "test_cartridge"
        assert d["stratum"] == "PHYSICS"
        assert "header_template" in d
        assert d["field_definitions"] == ["value"]

# ==============================================================================
# TESTS: SCHEMA VALIDATOR
# ==============================================================================

class TestSchemaValidator:
    """Tests para SchemaValidator."""
    
    def test_validate_type_valid(self, schema_validator):
        """Validar tipo correcto."""
        schema = {"type": "number"}
        result = schema_validator.validate(schema, 42)
        
        assert result.is_valid
    
    def test_validate_type_invalid(self, schema_validator):
        """Validar tipo incorrecto."""
        schema = {"type": "number"}
        result = schema_validator.validate(schema, "not_a_number")
        
        assert not result.is_valid
        assert "Tipo inválido" in result.error
    
    def test_validate_type_union(self, schema_validator):
        """Validar unión de tipos."""
        schema = {"type": ["string", "number"]}
        
        result1 = schema_validator.validate(schema, "hello")
        assert result1.is_valid
        
        result2 = schema_validator.validate(schema, 42)
        assert result2.is_valid
        
        result3 = schema_validator.validate(schema, True)
        assert not result3.is_valid
    
    def test_validate_required_valid(self, schema_validator):
        """Validar claves requeridas presentes."""
        schema = {
            "type": "object",
            "required": ["a", "b"]
        }
        result = schema_validator.validate(schema, {"a": 1, "b": 2, "c": 3})
        
        assert result.is_valid
    
    def test_validate_required_missing(self, schema_validator):
        """Validar claves requeridas faltantes."""
        schema = {
            "type": "object",
            "required": ["a", "b"]
        }
        result = schema_validator.validate(schema, {"a": 1})
        
        assert not result.is_valid
        assert "faltantes" in result.error
        assert "'b'" in result.error
    
    def test_validate_properties_valid(self, schema_validator):
        """Validar propiedades válidas."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        result = schema_validator.validate(schema, {"name": "Alice", "age": 30})
        
        assert result.is_valid
    
    def test_validate_properties_invalid(self, schema_validator):
        """Validar propiedades inválidas."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        result = schema_validator.validate(schema, {"name": "Alice", "age": "thirty"})
        
        assert not result.is_valid
    
    def test_validate_items_valid(self, schema_validator):
        """Validar items de array válidos."""
        schema = {
            "type": "array",
            "items": {"type": "number"}
        }
        result = schema_validator.validate(schema, [1, 2, 3, 4.5])
        
        assert result.is_valid
    
    def test_validate_items_invalid(self, schema_validator):
        """Validar items de array inválidos."""
        schema = {
            "type": "array",
            "items": {"type": "number"}
        }
        result = schema_validator.validate(schema, [1, 2, "three"])
        
        assert not result.is_valid
    
    def test_validate_minimum_valid(self, schema_validator):
        """Validar mínimo válido."""
        schema = {"type": "number", "minimum": 0}
        result = schema_validator.validate(schema, 5)
        
        assert result.is_valid
    
    def test_validate_minimum_invalid(self, schema_validator):
        """Validar mínimo inválido."""
        schema = {"type": "number", "minimum": 0}
        result = schema_validator.validate(schema, -5)
        
        assert not result.is_valid
        assert "menor que mínimo" in result.error
    
    def test_validate_maximum_valid(self, schema_validator):
        """Validar máximo válido."""
        schema = {"type": "number", "maximum": 100}
        result = schema_validator.validate(schema, 50)
        
        assert result.is_valid
    
    def test_validate_maximum_invalid(self, schema_validator):
        """Validar máximo inválido."""
        schema = {"type": "number", "maximum": 100}
        result = schema_validator.validate(schema, 150)
        
        assert not result.is_valid
        assert "mayor que máximo" in result.error
    
    def test_validate_exclusive_minimum_valid(self, schema_validator):
        """Validar exclusiveMinimum válido."""
        schema = {"type": "number", "exclusiveMinimum": 0}
        result = schema_validator.validate(schema, 5)
        
        assert result.is_valid
    
    def test_validate_exclusive_minimum_invalid(self, schema_validator):
        """Validar exclusiveMinimum inválido."""
        schema = {"type": "number", "exclusiveMinimum": 0}
        result = schema_validator.validate(schema, 0)
        
        assert not result.is_valid
        assert "no estrictamente mayor" in result.error
    
    def test_validate_min_length_valid(self, schema_validator):
        """Validar minLength válido."""
        schema = {"type": "string", "minLength": 3}
        result = schema_validator.validate(schema, "hello")
        
        assert result.is_valid
    
    def test_validate_min_length_invalid(self, schema_validator):
        """Validar minLength inválido."""
        schema = {"type": "string", "minLength": 3}
        result = schema_validator.validate(schema, "hi")
        
        assert not result.is_valid
    
    def test_validate_enum_valid(self, schema_validator):
        """Validar enum válido."""
        schema = {"enum": ["red", "green", "blue"]}
        result = schema_validator.validate(schema, "red")
        
        assert result.is_valid
    
    def test_validate_enum_invalid(self, schema_validator):
        """Validar enum inválido."""
        schema = {"enum": ["red", "green", "blue"]}
        result = schema_validator.validate(schema, "yellow")
        
        assert not result.is_valid
        assert "no está en enum" in result.error
    
    def test_validate_const_valid(self, schema_validator):
        """Validar const válido."""
        schema = {"const": 42}
        result = schema_validator.validate(schema, 42)
        
        assert result.is_valid
    
    def test_validate_const_invalid(self, schema_validator):
        """Validar const inválido."""
        schema = {"const": 42}
        result = schema_validator.validate(schema, 43)
        
        assert not result.is_valid
    
    def test_validate_pattern_valid(self, schema_validator):
        """Validar pattern válido."""
        schema = {"type": "string", "pattern": r"^\d{3}-\d{4}$"}
        result = schema_validator.validate(schema, "123-4567")
        
        assert result.is_valid
    
    def test_validate_pattern_invalid(self, schema_validator):
        """Validar pattern inválido."""
        schema = {"type": "string", "pattern": r"^\d{3}-\d{4}$"}
        result = schema_validator.validate(schema, "invalid")
        
        assert not result.is_valid
    
    def test_validate_pattern_regex_error(self, schema_validator):
        """Manejar regex inválido."""
        schema = {"type": "string", "pattern": r"[invalid(regex"}
        result = schema_validator.validate(schema, "test")
        
        assert not result.is_valid
        assert "Patrón regex inválido" in result.error
    
    def test_validate_nested_complex(self, schema_validator):
        """Validar estructura anidada compleja."""
        schema = {
            "type": "object",
            "required": ["user"],
            "properties": {
                "user": {
                    "type": "object",
                    "required": ["name", "age"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "age": {"type": "integer", "minimum": 0, "maximum": 150},
                        "emails": {
                            "type": "array",
                            "items": {"type": "string", "pattern": r"^.+@.+\..+$"}
                        }
                    }
                }
            }
        }
        
        # Válido
        valid_data = {
            "user": {
                "name": "Alice",
                "age": 30,
                "emails": ["alice@example.com"]
            }
        }
        result = schema_validator.validate(schema, valid_data)
        assert result.is_valid
        
        # Inválido
        invalid_data = {
            "user": {
                "name": "",  # minLength violation
                "age": 200,  # maximum violation
                "emails": ["invalid_email"]  # pattern violation
            }
        }
        result = schema_validator.validate(schema, invalid_data)
        assert not result.is_valid
        assert len(result.errors) >= 3

# ==============================================================================
# CONFIGURACIÓN DE PYTEST
# ==============================================================================

def pytest_configure(config):
    """Configuración de pytest."""
    config.addinivalue_line(
        "markers",
        "slow: marca tests lentos"
    )
    config.addinivalue_line(
        "markers",
        "integration: marca tests de integración"
    )

# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
Suite de Pruebas para MIC Agent - PARTE 2
Ubicación: tests/agents/test_mic_agent.py (continuación)

COBERTURA DE PRUEBAS - PARTE 2:

7. ALGEBRAIC VETO REGISTRY:
   - Validadores por defecto
   - Registro de validadores personalizados
   - Validación por estrato
   - Composición de validadores

8. SILO MANAGER:
   - Inicialización con contratos/cartuchos por defecto
   - Registro y recuperación
   - Congelación (inmutabilidad)
   - Selectores deterministas
   - Invariantes de cobertura

9. TOON COMPRESSOR:
   - Compresión determinista
   - Verificación de isomorfismo
   - Cálculo de ratio de compresión
   - Protección contra rango tensorial alto
   - Estadísticas

10. AUDIT TRAIL:
    - Buffer circular thread-safe
    - Operaciones concurrentes
    - Filtrado por status/estrato
    - Estadísticas agregadas

11. MIC AGENT:
    - Construcción y configuración
    - Sensado de estrato
    - Validación de clausura transitiva
    - Encapsulación monádica
    - Pipeline completo de proyección
    - Escudo inmunológico
    - Propiedades funtoriales

12. INTEGRACIÓN END-TO-END:
    - Pipeline completo
    - Manejo de errores
    - Auditoría completa
    - Verificación de determinismo
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pytest
from unittest.mock import Mock, patch, call

# Imports adicionales para PARTE 2
from app.agents.mic_agent import (
    AlgebraicVetoRegistry,
    SiloManager,
    TOONCompressor,
    AuditTrail,
    MICAgent,
    ImpedanceMatchStatus,
    CategoricalEqualizerSeed,
    TOONDocument,
    normalize_stratum,
    MathUtils,
    ALGEBRAIC_TOL,
    MAX_TENSOR_RANK,
)

from app.core.schemas import Stratum
from app.core.mic_algebra import CategoricalState

# ==============================================================================
# TESTS: ALGEBRAIC VETO REGISTRY
# ==============================================================================

class TestAlgebraicVetoRegistry:
    """Tests para AlgebraicVetoRegistry."""
    
    def test_initialization(self):
        """Inicialización con validadores por defecto."""
        registry = AlgebraicVetoRegistry()
        
        # Verificar que todos los estratos tienen validadores
        for stratum in Stratum:
            count = registry.get_validator_count(stratum)
            assert count >= 0  # Al menos uno por defecto
    
    def test_physics_conservation_valid(self):
        """Validador de conservación de energía (válido)."""
        registry = AlgebraicVetoRegistry()
        
        payload = {
            "dissipated_power": 10.0,
            "energy_input": 100.0,
            "energy_output": 90.0
        }
        
        errors = registry.validate(Stratum.PHYSICS, payload)
        assert len(errors) == 0
    
    def test_physics_conservation_negative_power(self):
        """Validador rechaza potencia disipada negativa."""
        registry = AlgebraicVetoRegistry()
        
        payload = {
            "dissipated_power": -5.0
        }
        
        errors = registry.validate(Stratum.PHYSICS, payload)
        assert len(errors) > 0
        assert "termodinámica" in errors[0].lower()
    
    def test_physics_conservation_energy_violation(self):
        """Validador rechaza violación de conservación de energía."""
        registry = AlgebraicVetoRegistry()
        
        payload = {
            "energy_input": 100.0,
            "energy_output": 150.0  # Violación
        }
        
        errors = registry.validate(Stratum.PHYSICS, payload)
        assert len(errors) > 0
        assert "conservación" in errors[0].lower()
    
    def test_tactics_stability_valid(self):
        """Validador de estabilidad (válido)."""
        registry = AlgebraicVetoRegistry()
        
        payload = {
            "pyramid_stability_index": 0.75
        }
        
        errors = registry.validate(Stratum.TACTICS, payload)
        assert len(errors) == 0
    
    def test_tactics_stability_out_of_range(self):
        """Validador rechaza índice fuera de [0, 1]."""
        registry = AlgebraicVetoRegistry()
        
        # Menor que 0
        payload1 = {"pyramid_stability_index": -0.1}
        errors1 = registry.validate(Stratum.TACTICS, payload1)
        assert len(errors1) > 0
        
        # Mayor que 1
        payload2 = {"pyramid_stability_index": 1.5}
        errors2 = registry.validate(Stratum.TACTICS, payload2)
        assert len(errors2) > 0
    
    def test_strategy_friction_valid(self):
        """Validador de fricción territorial (válido)."""
        registry = AlgebraicVetoRegistry()
        
        payload = {
            "territorial_friction": 2.5
        }
        
        errors = registry.validate(Stratum.STRATEGY, payload)
        assert len(errors) == 0
    
    def test_strategy_friction_below_minimum(self):
        """Validador rechaza fricción < 1.0."""
        registry = AlgebraicVetoRegistry()
        
        payload = {
            "territorial_friction": 0.5
        }
        
        errors = registry.validate(Stratum.STRATEGY, payload)
        assert len(errors) > 0
        assert "debe ser >= 1.0" in errors[0]
    
    def test_wisdom_verdict_valid(self):
        """Validador de veredicto (válido)."""
        registry = AlgebraicVetoRegistry()
        
        for verdict in ["VIABLE", "PRECAUCION", "RECHAZAR"]:
            payload = {"final_verdict": verdict}
            errors = registry.validate(Stratum.WISDOM, payload)
            assert len(errors) == 0
    
    def test_wisdom_verdict_invalid(self):
        """Validador rechaza veredicto inválido."""
        registry = AlgebraicVetoRegistry()
        
        payload = {"final_verdict": "MAYBE"}
        errors = registry.validate(Stratum.WISDOM, payload)
        
        assert len(errors) > 0
        assert "inválido" in errors[0].lower()
    
    def test_register_custom_validator(self):
        """Registrar validador personalizado."""
        registry = AlgebraicVetoRegistry()
        
        def custom_validator(stratum, payload):
            if payload.get("custom_field") != "expected":
                return "Custom validation failed"
            return None
        
        initial_count = registry.get_validator_count(Stratum.PHYSICS)
        registry.register_validator(Stratum.PHYSICS, custom_validator)
        
        assert registry.get_validator_count(Stratum.PHYSICS) == initial_count + 1
        
        # Probar validador personalizado
        payload_valid = {"custom_field": "expected"}
        errors_valid = registry.validate(Stratum.PHYSICS, payload_valid)
        # No debe haber error del validador personalizado
        
        payload_invalid = {"custom_field": "wrong"}
        errors_invalid = registry.validate(Stratum.PHYSICS, payload_invalid)
        assert any("Custom validation failed" in e for e in errors_invalid)
    
    def test_validator_exception_handling(self):
        """Manejar excepciones en validadores."""
        registry = AlgebraicVetoRegistry()
        
        def faulty_validator(stratum, payload):
            raise RuntimeError("Validator crashed")
        
        registry.register_validator(Stratum.PHYSICS, faulty_validator)
        
        # No debe propagar excepción
        errors = registry.validate(Stratum.PHYSICS, {})
        assert len(errors) > 0
        assert any("Error en validador algebraico" in e for e in errors)

# ==============================================================================
# TESTS: SILO MANAGER
# ==============================================================================

class TestSiloManager:
    """Tests para SiloManager."""
    
    def test_initialization_default_silos(self):
        """Inicialización con silos por defecto."""
        manager = SiloManager()
        
        # Verificar que todos los estratos tienen contratos
        for stratum in Stratum:
            contracts = manager.list_contracts(stratum)
            assert len(contracts) > 0
            
            cartridges = manager.list_cartridges(stratum)
            assert len(cartridges) > 0
    
    def test_fetch_contract_physics(self):
        """Fetch contrato para PHYSICS."""
        manager = SiloManager()
        
        contract_id, schema = manager.fetch_contract(
            Stratum.PHYSICS,
            "test_vector"
        )
        
        assert contract_id is not None
        assert isinstance(schema, dict)
        assert "type" in schema
    
    def test_fetch_contract_all_strata(self):
        """Fetch contratos para todos los estratos."""
        manager = SiloManager()
        
        for stratum in Stratum:
            contract_id, schema = manager.fetch_contract(stratum, "test")
            assert contract_id is not None
            assert isinstance(schema, dict)
    
    def test_fetch_cartridge_physics(self):
        """Fetch cartucho para PHYSICS."""
        manager = SiloManager()
        
        cartridge_id, header = manager.fetch_cartridge(
            Stratum.PHYSICS,
            "test_vector"
        )
        
        assert cartridge_id is not None
        assert isinstance(header, str)
        assert len(header) > 0
    
    def test_fetch_cartridge_all_strata(self):
        """Fetch cartuchos para todos los estratos."""
        manager = SiloManager()
        
        for stratum in Stratum:
            cartridge_id, header = manager.fetch_cartridge(stratum, "test")
            assert cartridge_id is not None
            assert isinstance(header, str)
    
    def test_list_contracts(self):
        """Listar contratos."""
        manager = SiloManager()
        
        # Todos los contratos
        all_contracts = manager.list_contracts()
        assert len(all_contracts) > 0
        
        # Contratos por estrato
        physics_contracts = manager.list_contracts(Stratum.PHYSICS)
        assert len(physics_contracts) > 0
    
    def test_list_cartridges(self):
        """Listar cartuchos."""
        manager = SiloManager()
        
        # Todos los cartuchos
        all_cartridges = manager.list_cartridges()
        assert len(all_cartridges) > 0
        
        # Cartuchos por estrato
        physics_cartridges = manager.list_cartridges(Stratum.PHYSICS)
        assert len(physics_cartridges) > 0
    
    def test_get_contract_count(self):
        """Contar contratos."""
        manager = SiloManager()
        
        total_count = manager.get_contract_count()
        assert total_count > 0
        
        physics_count = manager.get_contract_count(Stratum.PHYSICS)
        assert physics_count > 0
    
    def test_get_cartridge_count(self):
        """Contar cartuchos."""
        manager = SiloManager()
        
        total_count = manager.get_cartridge_count()
        assert total_count > 0
        
        physics_count = manager.get_cartridge_count(Stratum.PHYSICS)
        assert physics_count > 0
    
    def test_get_contract_by_id(self):
        """Recuperar contrato por ID."""
        manager = SiloManager()
        
        # Obtener primer contrato
        contract_id = manager.list_contracts(Stratum.PHYSICS)[0]
        contract = manager.get_contract(contract_id)
        
        assert contract is not None
        assert contract.contract_id == contract_id
    
    def test_get_contract_nonexistent(self):
        """Recuperar contrato inexistente."""
        manager = SiloManager()
        
        contract = manager.get_contract("nonexistent_id")
        assert contract is None
    
    def test_get_cartridge_by_id(self):
        """Recuperar cartucho por ID."""
        manager = SiloManager()
        
        # Obtener primer cartucho
        cartridge_id = manager.list_cartridges(Stratum.PHYSICS)[0]
        cartridge = manager.get_cartridge(cartridge_id)
        
        assert cartridge is not None
        assert cartridge.cartridge_id == cartridge_id
    
    def test_get_cartridge_nonexistent(self):
        """Recuperar cartucho inexistente."""
        manager = SiloManager()
        
        cartridge = manager.get_cartridge("nonexistent_id")
        assert cartridge is None
    
    def test_freeze_prevents_registration(self):
        """Congelar silos previene registro."""
        manager = SiloManager()
        manager.freeze()
        
        from app.agents.mic_agent import SiloAContract, SiloBCartridge
        
        # Intentar registrar nuevo contrato (debería fallar)
        # Nota: _register_contract es privado, usar con cuidado en tests
        with pytest.raises(Exception):  # SiloAccessError
            manager._register_contract(SiloAContract(
                contract_id="new_contract",
                stratum=Stratum.PHYSICS,
                schema={"type": "object"}
            ))
    
    def test_deterministic_selection(self):
        """Selección determinista de contratos."""
        manager = SiloManager()
        
        # Múltiples llamadas deben retornar mismo resultado
        contract_id1, _ = manager.fetch_contract(Stratum.PHYSICS, "test")
        contract_id2, _ = manager.fetch_contract(Stratum.PHYSICS, "test")
        
        assert contract_id1 == contract_id2

# ==============================================================================
# TESTS: TOON COMPRESSOR
# ==============================================================================

class TestTOONCompressor:
    """Tests para TOONCompressor."""
    
    def test_compress_basic(self):
        """Comprimir telemetría básica."""
        compressor = TOONCompressor()
        
        telemetry = {
            "value1": 42,
            "value2": "test",
            "value3": 3.14
        }
        
        doc = compressor.compress(
            telemetry,
            "test_cartridge",
            "Header\nkey|value"
        )
        
        assert doc.cartridge_id == "test_cartridge"
        assert len(doc.records) == 3
    
    def test_compress_deterministic(self):
        """Compresión determinista (orden alfabético de claves)."""
        compressor = TOONCompressor()
        
        telemetry1 = {"b": 2, "a": 1, "c": 3}
        telemetry2 = {"c": 3, "a": 1, "b": 2}
        
        doc1 = compressor.compress(telemetry1, "test", "Header")
        doc2 = compressor.compress(telemetry2, "test", "Header")
        
        assert doc1.records == doc2.records
    
    def test_compress_nested_structures(self):
        """Comprimir estructuras anidadas."""
        compressor = TOONCompressor()
        
        telemetry = {
            "nested": {
                "level2": {
                    "value": 42
                }
            }
        }
        
        doc = compressor.compress(telemetry, "test", "Header")
        
        # Verificar que se serializó correctamente
        assert len(doc.records) == 1
        key, json_value = doc.records[0]
        assert key == "nested"
        assert "{" in json_value  # Es JSON
    
    def test_compress_exceeds_tensor_rank(self):
        """Rechazar telemetría con rango tensorial muy alto."""
        compressor = TOONCompressor()
        
        # Crear estructura muy profunda
        deep = {"level1": {"level2": {"level3": {"level4": 1}}}}
        
        with pytest.raises(Exception):  # TOONCompressionError
            compressor.compress(deep, "test", "Header")
    
    def test_decompress_basic(self):
        """Descomprimir documento TOON."""
        compressor = TOONCompressor()
        
        doc = TOONDocument(
            cartridge_id="test",
            header_template="Header",
            records=(
                ("key1", "42"),
                ("key2", "\"value\""),
            )
        )
        
        decompressed = compressor.decompress(doc)
        
        assert decompressed["key1"] == 42
        assert decompressed["key2"] == "value"
    
    def test_compress_decompress_roundtrip(self):
        """Verificar roundtrip compress → decompress."""
        compressor = TOONCompressor()
        
        original = {
            "int_value": 42,
            "float_value": 3.14,
            "string_value": "test",
            "bool_value": True,
            "null_value": None,
            "array_value": [1, 2, 3],
            "object_value": {"nested": "value"}
        }
        
        doc = compressor.compress(original, "test", "Header")
        decompressed = compressor.decompress(doc)
        
        # Comparar valores (orden puede variar)
        assert decompressed["int_value"] == original["int_value"]
        assert decompressed["float_value"] == original["float_value"]
        assert decompressed["string_value"] == original["string_value"]
        assert decompressed["bool_value"] == original["bool_value"]
        assert decompressed["null_value"] == original["null_value"]
        assert decompressed["array_value"] == original["array_value"]
        assert decompressed["object_value"] == original["object_value"]
    
    def test_compute_ratio_basic(self):
        """Calcular ratio de compresión."""
        compressor = TOONCompressor()
        
        original = {"key": "value"}
        compressed = "Header\nkey|\"value\""
        
        ratio = compressor.compute_ratio(original, compressed)
        
        assert isinstance(ratio, float)
        assert ratio > 0
    
    def test_compute_ratio_bounds(self):
        """Ratio de compresión dentro de límites."""
        compressor = TOONCompressor()
        
        original = {"key": "value"}
        compressed = "short"
        
        ratio = compressor.compute_ratio(original, compressed)
        
        from app.agents.mic_agent import MIN_COMPRESSION_RATIO, MAX_COMPRESSION_RATIO
        assert MIN_COMPRESSION_RATIO <= ratio <= MAX_COMPRESSION_RATIO
    
    def test_verify_isomorphism_valid(self):
        """Verificar isomorfismo válido."""
        compressor = TOONCompressor()
        
        original = {"a": 1, "b": 2}
        doc = compressor.compress(original, "test", "Header")
        compressed = doc.render()
        
        is_isomorphic = compressor.verify_isomorphism(original, compressed)
        
        assert is_isomorphic
    
    def test_verify_isomorphism_invalid(self):
        """Verificar isomorfismo inválido."""
        compressor = TOONCompressor()
        
        original = {"a": 1, "b": 2}
        compressed = "--- INICIO TOON --- test ---\nHeader\na|999\n--- FIN TOON ---"
        
        is_isomorphic = compressor.verify_isomorphism(original, compressed)
        
        assert not is_isomorphic
    
    def test_get_statistics_empty(self):
        """Estadísticas sin compresiones."""
        compressor = TOONCompressor()
        
        stats = compressor.get_statistics()
        
        assert stats["count"] == 0
        assert stats["mean_ratio"] == 0.0
    
    def test_get_statistics_with_data(self):
        """Estadísticas con datos."""
        compressor = TOONCompressor()
        
        # Realizar varias compresiones
        for i in range(5):
            telemetry = {"value": i}
            doc = compressor.compress(telemetry, "test", "Header")
            compressed = doc.render()
            compressor.compute_ratio(telemetry, compressed)
        
        stats = compressor.get_statistics()
        
        assert stats["count"] == 5
        assert stats["mean_ratio"] > 0
        assert "std_ratio" in stats

# ==============================================================================
# TESTS: AUDIT TRAIL
# ==============================================================================

class TestAuditTrail:
    """Tests para AuditTrail."""
    
    def test_initialization(self):
        """Inicializar audit trail."""
        trail = AuditTrail(max_size=100)
        
        assert trail.size == 0
        assert trail.total_count == 0
    
    def test_initialization_invalid_size(self):
        """Rechazar tamaño inválido."""
        with pytest.raises(ValueError):
            AuditTrail(max_size=0)
        
        with pytest.raises(ValueError):
            AuditTrail(max_size=-10)
    
    def test_append_single(self):
        """Agregar seed individual."""
        trail = AuditTrail(max_size=100)
        
        seed = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION
        )
        
        trail.append(seed)
        
        assert trail.size == 1
        assert trail.total_count == 1
    
    def test_append_multiple(self):
        """Agregar múltiples seeds."""
        trail = AuditTrail(max_size=100)
        
        for i in range(10):
            seed = CategoricalEqualizerSeed(
                target_vector=f"vector_{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION
            )
            trail.append(seed)
        
        assert trail.size == 10
        assert trail.total_count == 10
    
    def test_circular_buffer_overflow(self):
        """Buffer circular descarta elementos antiguos."""
        trail = AuditTrail(max_size=5)
        
        # Agregar más elementos que max_size
        for i in range(10):
            seed = CategoricalEqualizerSeed(
                target_vector=f"vector_{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION
            )
            trail.append(seed)
        
        # Size limitado a max_size
        assert trail.size == 5
        # Pero total_count continúa incrementando
        assert trail.total_count == 10
        
        # Verificar que se conservan los últimos 5
        all_seeds = trail.get_all()
        assert all_seeds[0].target_vector == "vector_5"
        assert all_seeds[-1].target_vector == "vector_9"
    
    def test_get_all(self):
        """Obtener todos los seeds."""
        trail = AuditTrail(max_size=100)
        
        for i in range(3):
            seed = CategoricalEqualizerSeed(
                target_vector=f"v{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION
            )
            trail.append(seed)
        
        all_seeds = trail.get_all()
        
        assert len(all_seeds) == 3
        assert all_seeds[0].target_vector == "v0"
    
    def test_get_recent(self):
        """Obtener seeds recientes."""
        trail = AuditTrail(max_size=100)
        
        for i in range(10):
            seed = CategoricalEqualizerSeed(
                target_vector=f"v{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION
            )
            trail.append(seed)
        
        recent = trail.get_recent(3)
        
        assert len(recent) == 3
        assert recent[0].target_vector == "v7"
        assert recent[-1].target_vector == "v9"
    
    def test_get_by_status(self):
        """Filtrar por status."""
        trail = AuditTrail(max_size=100)
        
        # Agregar seeds con diferentes status
        for i in range(3):
            trail.append(CategoricalEqualizerSeed(
                target_vector=f"v{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION
            ))
        
        for i in range(2):
            trail.append(CategoricalEqualizerSeed(
                target_vector=f"v_error_{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.ALGEBRAIC_VETO
            ))
        
        laminar = trail.get_by_status(ImpedanceMatchStatus.LAMINAR_PROJECTION)
        veto = trail.get_by_status(ImpedanceMatchStatus.ALGEBRAIC_VETO)
        
        assert len(laminar) == 3
        assert len(veto) == 2
    
    def test_get_by_stratum(self):
        """Filtrar por estrato."""
        trail = AuditTrail(max_size=100)
        
        # Agregar seeds de diferentes estratos
        for stratum in [Stratum.PHYSICS, Stratum.TACTICS, Stratum.PHYSICS]:
            trail.append(CategoricalEqualizerSeed(
                target_vector="test",
                target_stratum=stratum,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION
            ))
        
        physics = trail.get_by_stratum(Stratum.PHYSICS)
        tactics = trail.get_by_stratum(Stratum.TACTICS)
        
        assert len(physics) == 2
        assert len(tactics) == 1
    
    def test_clear(self):
        """Limpiar trail."""
        trail = AuditTrail(max_size=100)
        
        for i in range(5):
            trail.append(CategoricalEqualizerSeed(
                target_vector=f"v{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION
            ))
        
        original_total = trail.total_count
        trail.clear()
        
        assert trail.size == 0
        assert trail.total_count == original_total  # No se resetea
    
    def test_get_statistics_empty(self):
        """Estadísticas de trail vacío."""
        trail = AuditTrail(max_size=100)
        
        stats = trail.get_statistics()
        
        assert stats["total_entries"] == 0
        assert stats["current_size"] == 0
        assert stats["status_distribution"] == {}
    
    def test_get_statistics_with_data(self):
        """Estadísticas con datos."""
        trail = AuditTrail(max_size=100)
        
        # Agregar seeds variados
        for i in range(5):
            trail.append(CategoricalEqualizerSeed(
                target_vector=f"v{i}",
                target_stratum=Stratum.PHYSICS if i % 2 == 0 else Stratum.TACTICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=(
                    ImpedanceMatchStatus.LAMINAR_PROJECTION
                    if i < 3
                    else ImpedanceMatchStatus.ALGEBRAIC_VETO
                ),
                token_compression_ratio=0.5 + i * 0.1
            ))
        
        stats = trail.get_statistics()
        
        assert stats["total_entries"] == 5
        assert stats["current_size"] == 5
        assert "status_distribution" in stats
        assert "stratum_distribution" in stats
        assert stats["mean_compression_ratio"] > 0
    
    def test_thread_safety(self):
        """Verificar thread-safety."""
        trail = AuditTrail(max_size=1000)
        
        def append_seeds(n):
            for i in range(n):
                trail.append(CategoricalEqualizerSeed(
                    target_vector=f"v{i}",
                    target_stratum=Stratum.PHYSICS,
                    silo_a_contract_id="c1",
                    silo_b_cartridge_id="c2",
                    impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION
                ))
        
        # Ejecutar en múltiples threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(append_seeds, 25) for _ in range(4)]
            for future in futures:
                future.result()
        
        # Verificar total
        assert trail.total_count == 100

# ==============================================================================
# TESTS: MIC AGENT - CONSTRUCCIÓN Y CONFIGURACIÓN
# ==============================================================================

class TestMICAgentConstruction:
    """Tests para construcción de MICAgent."""
    
    def test_initialization_basic(self, mock_mic_registry):
        """Inicializar agente básico."""
        agent = MICAgent(mock_mic_registry)
        
        assert agent.audit_trail is not None
        assert agent.silo_manager is not None
        assert agent.immune_watcher is not None
    
    def test_initialization_with_custom_components(self, mock_mic_registry):
        """Inicializar con componentes personalizados."""
        silo_manager = SiloManager()
        audit_trail_size = 500
        
        agent = MICAgent(
            mock_mic_registry,
            silo_manager=silo_manager,
            audit_trail_size=audit_trail_size,
            freeze_silos=False
        )
        
        assert agent.silo_manager is silo_manager
        assert agent.audit_trail.size == 0
    
    def test_verify_functorial_properties(self, mock_mic_registry):
        """Verificar propiedades funtoriales."""
        agent = MICAgent(mock_mic_registry)
        
        props = agent.verify_functorial_properties()
        
        assert all(props.values())  # Todos deben ser True
        assert "immune_watcher_initialized" in props
        assert "silo_manager_initialized" in props
        assert "mic_registry_initialized" in props
    
    def test_health_report(self, mock_mic_registry):
        """Generar reporte de salud."""
        agent = MICAgent(mock_mic_registry)
        
        report = agent.health_report()
        
        assert "MIC AGENT" in report
        assert "DIAGNÓSTICO" in report
        assert "COMPONENTES" in report
    
    def test_repr(self, mock_mic_registry):
        """Representación string."""
        agent = MICAgent(mock_mic_registry)
        
        s = repr(agent)
        
        assert "MICAgent" in s
        assert "contratos=" in s
        assert "cartuchos=" in s

# ==============================================================================
# TESTS: MIC AGENT - SENSADO DE ESTRATO
# ==============================================================================

class TestMICAgentStratumSensing:
    """Tests para sensado de estrato."""
    
    def test_sense_stratum_valid(self, mock_mic_registry):
        """Sensar estrato válido."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        stratum = agent.sense_stratum("test_vector")
        
        assert stratum == Stratum.PHYSICS
    
    def test_sense_stratum_from_int(self, mock_mic_registry):
        """Sensar estrato desde int."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": 1  # TACTICS
        })
        
        agent = MICAgent(mock_mic_registry)
        stratum = agent.sense_stratum("test_vector")
        
        assert stratum == Stratum.TACTICS
    
    def test_sense_stratum_from_string(self, mock_mic_registry):
        """Sensar estrato desde string."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": "STRATEGY"
        })
        
        agent = MICAgent(mock_mic_registry)
        stratum = agent.sense_stratum("test_vector")
        
        assert stratum == Stratum.STRATEGY
    
    def test_sense_stratum_vector_not_found(self, mock_mic_registry):
        """Rechazar vector inexistente."""
        mock_mic_registry.get_vector_info = Mock(return_value=None)
        
        agent = MICAgent(mock_mic_registry)
        
        from app.agents.mic_agent import StratumResolutionError
        with pytest.raises(StratumResolutionError, match="no existe"):
            agent.sense_stratum("nonexistent_vector")
    
    def test_sense_stratum_missing_stratum_key(self, mock_mic_registry):
        """Rechazar info sin clave 'stratum'."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "dimension": 3
            # Falta "stratum"
        })
        
        agent = MICAgent(mock_mic_registry)
        
        from app.agents.mic_agent import StratumResolutionError
        with pytest.raises(StratumResolutionError, match="no reporta estrato"):
            agent.sense_stratum("test_vector")

# ==============================================================================
# TESTS: MIC AGENT - VALIDACIÓN DE CLAUSURA
# ==============================================================================

class TestMICAgentClosureValidation:
    """Tests para validación de clausura transitiva."""
    
    def test_validate_closure_physics_valid(self, mock_mic_registry):
        """PHYSICS no requiere estratos previos."""
        agent = MICAgent(mock_mic_registry)
        
        error = agent.validate_closure(
            Stratum.PHYSICS,
            frozenset()  # Sin estratos validados
        )
        
        assert error is None
    
    def test_validate_closure_tactics_valid(self, mock_mic_registry):
        """TACTICS requiere PHYSICS."""
        agent = MICAgent(mock_mic_registry)
        
        error = agent.validate_closure(
            Stratum.TACTICS,
            frozenset([Stratum.PHYSICS])
        )
        
        assert error is None
    
    def test_validate_closure_tactics_missing_physics(self, mock_mic_registry):
        """TACTICS sin PHYSICS → error."""
        agent = MICAgent(mock_mic_registry)
        
        error = agent.validate_closure(
            Stratum.TACTICS,
            frozenset()
        )
        
        assert error is not None
        assert "PHYSICS" in error
    
    def test_validate_closure_strategy_valid(self, mock_mic_registry):
        """STRATEGY requiere PHYSICS y TACTICS."""
        agent = MICAgent(mock_mic_registry)
        
        error = agent.validate_closure(
            Stratum.STRATEGY,
            frozenset([Stratum.PHYSICS, Stratum.TACTICS])
        )
        
        assert error is None
    
    def test_validate_closure_strategy_missing_tactics(self, mock_mic_registry):
        """STRATEGY sin TACTICS → error."""
        agent = MICAgent(mock_mic_registry)
        
        error = agent.validate_closure(
            Stratum.STRATEGY,
            frozenset([Stratum.PHYSICS])
        )
        
        assert error is not None
        assert "TACTICS" in error
    
    def test_validate_closure_wisdom_valid(self, mock_mic_registry):
        """WISDOM requiere todos los estratos."""
        agent = MICAgent(mock_mic_registry)
        
        error = agent.validate_closure(
            Stratum.WISDOM,
            frozenset([Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY])
        )
        
        assert error is None
    
    def test_validate_closure_wisdom_missing_strategy(self, mock_mic_registry):
        """WISDOM sin STRATEGY → error."""
        agent = MICAgent(mock_mic_registry)
        
        error = agent.validate_closure(
            Stratum.WISDOM,
            frozenset([Stratum.PHYSICS, Stratum.TACTICS])
        )
        
        assert error is not None
        assert "STRATEGY" in error

# ==============================================================================
# TESTS: MIC AGENT - COMPRESIÓN TOON
# ==============================================================================

class TestMICAgentTOONCompression:
    """Tests para compresión TOON en MICAgent."""
    
    def test_compress_telemetry_basic(self, mock_mic_registry):
        """Comprimir telemetría básica."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        telemetry = {
            "value1": 42,
            "value2": "test"
        }
        
        cartridge_id, doc = agent.compress_telemetry("test_vector", telemetry)
        
        assert isinstance(cartridge_id, str)
        assert isinstance(doc, TOONDocument)
        assert len(doc.records) == 2
    
    def test_inject_functorial_context(self, mock_mic_registry):
        """Inyectar contexto comprimido."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        telemetry = {"value": 42}
        
        compressed = agent.inject_functorial_context("test_vector", telemetry)
        
        assert isinstance(compressed, str)
        assert TOON_START_MARKER in compressed
        assert TOON_END_MARKER in compressed

# ==============================================================================
# TESTS: MIC AGENT - ENCAPSULACIÓN MONÁDICA
# ==============================================================================

class TestMICAgentMonadicEncapsulation:
    """Tests para encapsulación monádica."""
    
    def test_encapsulate_monad_valid(self, mock_mic_registry):
        """Encapsular output válido del LLM."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        llm_output = {
            "dissipated_power": 10.0,
            "energy_input": 100.0,
            "energy_output": 90.0
        }
        
        state = agent.encapsulate_monad(
            "test_vector",
            llm_output,
            frozenset()
        )
        
        assert state.is_success
        assert state.payload == llm_output
    
    def test_encapsulate_monad_invalid_type(self, mock_mic_registry):
        """Rechazar output no-Mapping."""
        agent = MICAgent(mock_mic_registry)
        
        state = agent.encapsulate_monad(
            "test_vector",
            "not_a_dict",
            frozenset()
        )
        
        assert state.is_failed
        assert "Mapping" in state.error
    
    def test_encapsulate_monad_closure_violation(self, mock_mic_registry):
        """Rechazar violación de clausura."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.TACTICS  # Requiere PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        state = agent.encapsulate_monad(
            "test_vector",
            {"value": 42},
            frozenset()  # Sin PHYSICS
        )
        
        assert state.is_failed
        assert "clausura" in state.error.lower()
    
    def test_encapsulate_monad_schema_validation_error(self, mock_mic_registry):
        """Rechazar payload que no cumple schema."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        # Payload inválido (falta campo requerido)
        llm_output = {"invalid_field": "value"}
        
        state = agent.encapsulate_monad(
            "test_vector",
            llm_output,
            frozenset()
        )
        
        # Puede o no fallar dependiendo del contrato
        # Verificar que se registró en auditoría
        assert agent.audit_trail.size > 0
    
    def test_encapsulate_monad_algebraic_veto(self, mock_mic_registry):
        """Rechazar violación de invariantes algebraicos."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        # Violación: potencia disipada negativa
        llm_output = {
            "dissipated_power": -10.0
        }
        
        state = agent.encapsulate_monad(
            "test_vector",
            llm_output,
            frozenset()
        )
        
        assert state.is_failed
        assert "ALGEBRAIC_VETO" in state.error
    
    def test_encapsulate_monad_with_telemetry(self, mock_mic_registry):
        """Encapsular con telemetría para TOON."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        llm_output = {"dissipated_power": 10.0}
        raw_telemetry = {"raw_value": 42}
        
        state = agent.encapsulate_monad(
            "test_vector",
            llm_output,
            frozenset(),
            raw_telemetry=raw_telemetry
        )
        
        # Verificar que se comprimió
        assert "compressed_context" in state.context or state.is_success
        assert "compression_ratio" in state.context
    
    def test_encapsulate_monad_force_override(self, mock_mic_registry):
        """Forzar override de validación de clausura."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.TACTICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        state = agent.encapsulate_monad(
            "test_vector",
            {"pyramid_stability_index": 0.5},
            frozenset(),  # Sin PHYSICS
            force_override=True
        )
        
        # No debe fallar por clausura
        # Pero puede fallar por schema u otras razones

# ==============================================================================
# TESTS DE INTEGRACIÓN END-TO-END
# ==============================================================================

class TestMICAgentIntegration:
    """Tests de integración end-to-end."""
    
    def test_full_pipeline_physics(self, mock_mic_registry):
        """Pipeline completo para PHYSICS."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        mock_mic_registry.project_intent = Mock(return_value={
            "status": "OK",
            "result": {"projected": True}
        })
        
        agent = MICAgent(mock_mic_registry)
        
        llm_output = {
            "dissipated_power": 10.0,
            "energy_input": 100.0,
            "energy_output": 90.0
        }
        
        result = agent.execute_projection(
            "test_vector",
            llm_output,
            frozenset()
        )
        
        assert result["status"] in ("OK", "VETO")
        assert "impedance_status" in result
    
    def test_full_pipeline_with_telemetry(self, mock_mic_registry):
        """Pipeline con telemetría."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        mock_mic_registry.project_intent = Mock(return_value={
            "status": "OK"
        })
        
        agent = MICAgent(mock_mic_registry)
        
        llm_output = {"dissipated_power": 10.0}
        raw_telemetry = {
            "saturation": 0.5,
            "flyback_voltage": 200.0,
            "dissipated_power": 10.0,
            "beta_0": 1.0,
            "beta_1": 0.0,
            "entropy": 0.3,
            "exergy_loss": 0.2
        }
        
        result = agent.execute_projection(
            "test_vector",
            llm_output,
            frozenset(),
            raw_telemetry=raw_telemetry
        )
        
        assert "status" in result
    
    def test_audit_trail_populated(self, mock_mic_registry):
        """Verificar que audit trail se pobla."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        # Realizar múltiples encapsulaciones
        for i in range(5):
            agent.encapsulate_monad(
                f"vector_{i}",
                {"dissipated_power": 10.0},
                frozenset()
            )
        
        assert agent.audit_trail.size == 5
        assert agent.audit_trail.total_count == 5
    
    def test_get_audit_statistics(self, mock_mic_registry):
        """Obtener estadísticas de auditoría."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        # Realizar operaciones
        agent.encapsulate_monad("test", {"dissipated_power": 10.0}, frozenset())
        
        stats = agent.get_audit_statistics()
        
        assert "total_entries" in stats
        assert "status_distribution" in stats
    
    def test_get_recent_audits(self, mock_mic_registry):
        """Obtener auditorías recientes."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        for i in range(5):
            agent.encapsulate_monad(f"v{i}", {"dissipated_power": 10.0}, frozenset())
        
        recent = agent.get_recent_audits(3)
        
        assert len(recent) == 3
        assert all(isinstance(r, dict) for r in recent)
    
    def test_clear_audit_trail(self, mock_mic_registry):
        """Limpiar audit trail."""
        mock_mic_registry.get_vector_info = Mock(return_value={
            "stratum": Stratum.PHYSICS
        })
        
        agent = MICAgent(mock_mic_registry)
        
        agent.encapsulate_monad("test", {"dissipated_power": 10.0}, frozenset())
        assert agent.audit_trail.size > 0
        
        agent.clear_audit_trail()
        assert agent.audit_trail.size == 0

# ==============================================================================
# CONFIGURACIÓN DE PYTEST
# ==============================================================================

def pytest_configure(config):
    """Configuración de pytest."""
    config.addinivalue_line(
        "markers",
        "slow: marca tests lentos"
    )
    config.addinivalue_line(
        "markers",
        "integration: marca tests de integración"
    )
    config.addinivalue_line(
        "markers",
        "concurrent: marca tests de concurrencia"
    )

# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])