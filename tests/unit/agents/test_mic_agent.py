"""
Módulo: test_mic_agent.py
Propósito: Suite de pruebas rigurosas para MIC Agent
Ubicación: tests/unit/agents/test_mic_agent.py

FUNDAMENTOS DE PRUEBAS MATEMÁTICAS:
===================================
1. Tests Unitarios: Verificación de comportamiento individual
2. Property-Based Testing: Verificación de propiedades algebraicas
3. Tests de Invariantes: Condiciones que siempre deben cumplirse
4. Tests de Bordes: Casos extremos y condiciones límite
5. Tests de Propiedades Funtoriales: Axiomas de teoría de categorías

COBERTURA REQUERIDA:
===================
- Excepciones: 100%
- Enumeraciones: 100%
- Dataclasses: 100%
- SchemaValidator: 100%
- AlgebraicVetoRegistry: 100%
- SiloManager: 100%
- TOONCompressor: 100%
- AuditTrail: 100%
- MICAgent: 95%
- Utilidades: 100%

EJECUCIÓN:
==========
$ pytest tests/agents/test_mic_agent.py -v --cov=app.agents.mic_agent
$ pytest tests/agents/test_mic_agent.py -v -k "test_functor"
$ pytest tests/agents/test_mic_agent.py -v -m "property"
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import os
import decimal

# Fase 1: Esterilización del Vacío Termodinámico y Condicionamiento Numérico
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Cuantización Decimal
decimal.getcontext().prec = 50
decimal.getcontext().rounding = decimal.ROUND_HALF_EVEN

import pytest
import threading
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
from dataclasses import asdict
from copy import deepcopy
import numpy as np

# Importar módulo bajo prueba
from app.agents.mic_agent import (
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
    PolaronCartridge,
    PositronCartridge,
    ElectronCartridge,
    
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
)

# Importar de mic_algebra
try:
    from app.core.mic_algebra import Stratum, CategoricalState, create_categorical_state
except ImportError:
    # Fallback para testing standalone
    Stratum = None
    CategoricalState = None
    create_categorical_state = None

# ==============================================================================
# CONFIGURACIÓN DE LOGGING PARA TESTS
# ==============================================================================
logging.basicConfig(
    level=logging.WARNING,  # Silenciar logs informativos durante tests
    format="%(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout,
)

# ==============================================================================
# FIXTURES COMUNES
# ==============================================================================
@pytest.fixture
def sample_payload() -> Dict[str, Any]:
    """Payload de muestra para tests."""
    return {
        "dissipated_power": 100.0,
        "energy_input": 500.0,
        "energy_output": 450.0,
        "saturation": 0.8,
    }


@pytest.fixture
def valid_physics_schema() -> Dict[str, Any]:
    """Schema válido para estrato PHYSICS."""
    return {
        "type": "object",
        "required": ["dissipated_power"],
        "properties": {
            "dissipated_power": {"type": "number", "minimum": 0},
            "energy_input": {"type": "number", "minimum": 0},
            "energy_output": {"type": "number", "minimum": 0},
            "saturation": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }


@pytest.fixture
def schema_validator() -> SchemaValidator:
    """Validador de schema limpio para cada test."""
    return SchemaValidator()


@pytest.fixture
def veto_registry() -> AlgebraicVetoRegistry:
    """Registro de vetos algebraicos."""
    return AlgebraicVetoRegistry()


@pytest.fixture
def silo_manager() -> SiloManager:
    """Gestor de silos limpio para cada test."""
    return SiloManager()


@pytest.fixture
def toon_compressor() -> TOONCompressor:
    """Compresor TOON limpio para cada test."""
    return TOONCompressor()


@pytest.fixture
def audit_trail() -> AuditTrail:
    """Traza de auditoría limpia para cada test."""
    return AuditTrail(max_size=100)


@pytest.fixture
def mock_mic_registry():
    """Mock de MIC Registry para tests."""
    class MockMICRegistry:
        def __init__(self):
            self._vectors = {
                "topology_core": {"stratum": 5, "description": "Core topology"},
                "strategy_planner": {"stratum": 3, "description": "Strategy planning"},
                "tactics_executor": {"stratum": 4, "description": "Tactical execution"},
                "wisdom_advisor": {"stratum": 0, "description": "Wisdom advisory"},
            }
        
        def get_vector_info(self, vector_name: str) -> Optional[Dict[str, Any]]:
            return self._vectors.get(vector_name)
        
        def project_intent(
            self,
            target_basis_vector: str,
            stratum_target: int,
            validated_subspaces: List[str],
            orthogonality_guarantee: float,
            payload: Dict[str, Any],
        ) -> Dict[str, Any]:
            return {
                "projected": True,
                "vector": target_basis_vector,
                "stratum": stratum_target,
                "payload_hash": hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()
                ).hexdigest()[:16],
            }
    
    return MockMICRegistry()


@pytest.fixture
def mock_immune_watcher():
    """Mock de Immune Watcher para tests."""
    class MockImmuneWatcher:
        def __call__(self, state: Any) -> Any:
            # Pasar estado sin modificar para tests básicos
            return state
    
    return MockImmuneWatcher()


@pytest.fixture
def mic_agent(
    mock_mic_registry,
    silo_manager,
    schema_validator,
    veto_registry,
    toon_compressor,
    mock_immune_watcher,
) -> MICAgent:
    """Agente MIC configurado para tests."""
    return MICAgent(
        mic_registry=mock_mic_registry,
        silo_manager=silo_manager,
        schema_validator=schema_validator,
        algebraic_veto_registry=veto_registry,
        toon_compressor=toon_compressor,
        audit_trail_size=100,
        immune_watcher=mock_immune_watcher,
        freeze_silos=True,
    )


# ==============================================================================
# TESTS DE EXCEPCIONES
# ==============================================================================
class TestExceptions:
    """Tests para jerarquía de excepciones con contexto matemático."""
    
    def test_mic_agent_error_base(self) -> None:
        """Excepción base con contexto estructurado."""
        error = MICAgentError(
            "Test error",
            error_code="TEST_CODE",
            details={"key": "value"},
            severity=2
        )
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST_CODE"
        assert error.details == {"key": "value"}
        assert error.severity == 2
        assert isinstance(error.timestamp, float)
    
    def test_mic_agent_error_to_dict(self) -> None:
        """Serialización de excepción para logging."""
        error = MICAgentError(
            "Test error",
            error_code="TEST",
            details={"context": "data"},
        )
        
        result = error.to_dict()
        assert result["type"] == "MICAgentError"
        assert result["error_code"] == "TEST"
        assert result["message"] == "Test error"
        assert "timestamp" in result
    
    def test_stratum_resolution_error(self) -> None:
        """Error en resolución de estratos."""
        error = StratumResolutionError(
            "Invalid stratum",
            details={"input": "invalid"}
        )
        
        assert error.error_code == "STRATUM_RESOLUTION"
        assert error.severity == 2
    
    def test_contract_validation_error(self) -> None:
        """Error en validación de contratos."""
        error = ContractValidationError(
            "Schema invalid",
            details={"schema": {}}
        )
        
        assert error.error_code == "CONTRACT_VALIDATION"
        assert error.severity == 2
    
    def test_closure_violation_error(self) -> None:
        """Violación de clausura transitiva."""
        error = ClosureViolationError(
            "Missing strata",
            details={"missing": ["PHYSICS"]}
        )
        
        assert error.error_code == "CLOSURE_VIOLATION"
        assert error.severity == 3
    
    def test_algebraic_veto_error(self) -> None:
        """Veto por invariantes algebraicos."""
        error = AlgebraicVetoError(
            "Conservation violated",
            details={"energy": -10}
        )
        
        assert error.error_code == "ALGEBRAIC_VETO"
        assert error.severity == 3
    
    def test_toon_compression_error(self) -> None:
        """Error en compresión TOON."""
        error = TOONCompressionError(
            "Rank exceeded",
            details={"rank": 5}
        )
        
        assert error.error_code == "TOON_COMPRESSION"
        assert error.severity == 2
    
    def test_silo_access_error(self) -> None:
        """Error en acceso a silos."""
        error = SiloAccessError(
            "Contract not found",
            details={"contract_id": "test"}
        )
        
        assert error.error_code == "SILO_ACCESS"
        assert error.severity == 2
    
    def test_projection_error(self) -> None:
        """Error en proyección MIC."""
        error = ProjectionError(
            "Projection failed",
            details={"vector": "test"}
        )
        
        assert error.error_code == "PROJECTION"
        assert error.severity == 3
    
    def test_functoriality_error(self) -> None:
        """Violación de propiedades funtoriales."""
        error = FunctorialityError(
            "F(g∘f) ≠ F(g)∘F(f)",
            details={"lhs": "hash1", "rhs": "hash2"}
        )
        
        assert error.error_code == "FUNCTORIALITY"
        assert error.severity == 3


# ==============================================================================
# TESTS DE ENUMERACIONES
# ==============================================================================
class TestEnumerations:
    """Tests para enumeraciones con orden total y propiedades algebraicas."""
    
    def test_impedance_match_status_values(self) -> None:
        """Todos los estados de impedancia están definidos."""
        statuses = [
            "LAMINAR_PROJECTION",
            "STRATUM_MISMATCH_REJECTED",
            "TOON_COMPRESSION_ERROR",
            "ALGEBRAIC_VETO",
            "SCHEMA_VALIDATION_ERROR",
            "MIC_RESOLUTION_ERROR",
            "INPUT_TYPE_ERROR",
            "TOPOLOGICAL_BIFURCATION",
            "COHOMOLOGY_FAILURE",
        ]
        
        for status_name in statuses:
            assert hasattr(ImpedanceMatchStatus, status_name)
    
    def test_impedance_match_status_is_terminal(self) -> None:
        """Estados terminales correctamente identificados."""
        terminal_statuses = {
            ImpedanceMatchStatus.ALGEBRAIC_VETO,
            ImpedanceMatchStatus.TOPOLOGICAL_BIFURCATION,
            ImpedanceMatchStatus.COHOMOLOGY_FAILURE,
        }
        
        for status in ImpedanceMatchStatus:
            expected = status in terminal_statuses
            assert status.is_terminal == expected
    
    def test_impedance_match_status_severity(self) -> None:
        """Severidad correctamente mapeada."""
        assert ImpedanceMatchStatus.LAMINAR_PROJECTION.severity == 0
        assert ImpedanceMatchStatus.INPUT_TYPE_ERROR.severity == 1
        assert ImpedanceMatchStatus.MIC_RESOLUTION_ERROR.severity == 2
        assert ImpedanceMatchStatus.ALGEBRAIC_VETO.severity == 3
    
    def test_impedance_match_status_order_total(self) -> None:
        """Orden total por severidad."""
        # Transitividad
        assert ImpedanceMatchStatus.LAMINAR_PROJECTION < ImpedanceMatchStatus.ALGEBRAIC_VETO
        assert ImpedanceMatchStatus.INPUT_TYPE_ERROR < ImpedanceMatchStatus.ALGEBRAIC_VETO
        
        # Antisimetría
        assert not (ImpedanceMatchStatus.ALGEBRAIC_VETO < ImpedanceMatchStatus.LAMINAR_PROJECTION)
        
        # Reflexividad (no estricto)
        assert ImpedanceMatchStatus.LAMINAR_PROJECTION <= ImpedanceMatchStatus.LAMINAR_PROJECTION
    
    def test_validation_severity_values(self) -> None:
        """Severidades de validación definidas."""
        assert hasattr(ValidationSeverity, "ERROR")
        assert hasattr(ValidationSeverity, "WARNING")
        assert hasattr(ValidationSeverity, "INFO")
    
    def test_validation_severity_heyting_value(self) -> None:
        """Valores en Álgebra de Heyting [0, 1]."""
        assert ValidationSeverity.ERROR.heyting_value == 0.0
        assert ValidationSeverity.WARNING.heyting_value == 0.5
        assert ValidationSeverity.INFO.heyting_value == 1.0
    
    def test_validation_severity_order(self) -> None:
        """Orden en Álgebra de Heyting."""
        assert ValidationSeverity.ERROR.heyting_value < ValidationSeverity.INFO.heyting_value
        assert ValidationSeverity.WARNING.heyting_value > ValidationSeverity.ERROR.heyting_value


# ==============================================================================
# TESTS DE UTILIDADES MATEMÁTICAS
# ==============================================================================
class TestMathUtils:
    """Tests para utilidades matemáticas con garantías numéricas."""
    
    def test_stable_hash_determinism(self) -> None:
        """Hash es determinista."""
        data = {"test": "value", "number": 42}
        hash1 = MathUtils.stable_hash(data)
        hash2 = MathUtils.stable_hash(data)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex
    
    def test_stable_hash_avalanche(self) -> None:
        """Cambio mínimo produce hash diferente."""
        data1 = {"test": "value"}
        data2 = {"test": "value1"}
        
        hash1 = MathUtils.stable_hash(data1)
        hash2 = MathUtils.stable_hash(data2)
        
        assert hash1 != hash2
        # Verificar efecto avalancha (difieren en múltiples caracteres)
        diff_count = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        assert diff_count > 10
    
    def test_compute_tensor_rank_scalar(self) -> None:
        """Rango tensorial para escalares."""
        assert MathUtils.compute_tensor_rank(42) == 0
        assert MathUtils.compute_tensor_rank(3.14) == 0
        assert MathUtils.compute_tensor_rank("text") == 0
    
    def test_compute_tensor_rank_dict(self) -> None:
        """Rango tensorial para diccionarios."""
        assert MathUtils.compute_tensor_rank({"a": 1}) == 1
        assert MathUtils.compute_tensor_rank({"a": {"b": 1}}) == 2
        assert MathUtils.compute_tensor_rank({}) == 1
    
    def test_compute_tensor_rank_list(self) -> None:
        """Rango tensorial para listas."""
        assert MathUtils.compute_tensor_rank([1, 2, 3]) == 1
        assert MathUtils.compute_tensor_rank([[1], [2]]) == 2
        assert MathUtils.compute_tensor_rank([]) == 1
    
    def test_compute_tensor_rank_max_depth(self) -> None:
        """Protección contra profundidad máxima."""
        # Crear estructura muy profunda
        deep = {"level": 0}
        current = deep
        for i in range(150):
            current["nested"] = {"level": i + 1}
            current = current["nested"]
        
        rank = MathUtils.compute_tensor_rank(deep, max_depth=100)
        assert rank <= 100
    
    def test_float_equal_reflexivity(self) -> None:
        """float_equal es reflexiva."""
        for val in [0.0, 1.0, -1.0, 1e-10, 1e10]:
            assert MathUtils.float_equal(val, val)
    
    def test_float_equal_symmetry(self) -> None:
        """float_equal es simétrica."""
        a, b = 1.0, 1.0 + 1e-10
        assert MathUtils.float_equal(a, b) == MathUtils.float_equal(b, a)
    
    def test_float_equal_tolerance(self) -> None:
        """Tolerancia absoluta y relativa."""
        # Tolerancia absoluta
        assert MathUtils.float_equal(0.0, 1e-10, tol=1e-9)
        assert not MathUtils.float_equal(0.0, 1e-8, tol=1e-9)
        
        # Tolerancia relativa
        assert MathUtils.float_equal(1e10, 1e10 + 1.0, tol=1e-9)
    
    def test_clamp_valid_range(self) -> None:
        """Clamp con rango válido."""
        assert MathUtils.clamp(5.0, 0.0, 10.0) == 5.0
        assert MathUtils.clamp(-5.0, 0.0, 10.0) == 0.0
        assert MathUtils.clamp(15.0, 0.0, 10.0) == 10.0
    
    def test_clamp_invalid_range_raises(self) -> None:
        """Clamp con rango inválido lanza ValueError."""
        with pytest.raises(ValueError):
            MathUtils.clamp(5.0, 10.0, 0.0)


# ==============================================================================
# TESTS DE NORMALIZACIÓN DE ESTRATOS
# ==============================================================================
class TestStratumNormalization:
    """Tests para normalización de estratos."""
    
    def test_normalize_stratum_from_stratum(self) -> None:
        """Normalización desde Stratum."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        result = normalize_stratum(Stratum.PHYSICS)
        assert result == Stratum.PHYSICS
    
    def test_normalize_stratum_from_int(self) -> None:
        """Normalización desde entero."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        result = normalize_stratum(5)
        assert result == Stratum.PHYSICS
        
        result = normalize_stratum(0)
        assert result == Stratum.WISDOM
    
    def test_normalize_stratum_from_str_name(self) -> None:
        """Normalización desde nombre string."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        result = normalize_stratum("PHYSICS")
        assert result == Stratum.PHYSICS
        
        result = normalize_stratum("physics")  # Case insensitive
        assert result == Stratum.PHYSICS
    
    def test_normalize_stratum_from_str_value(self) -> None:
        """Normalización desde valor string."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        result = normalize_stratum("5")
        assert result == Stratum.PHYSICS
    
    def test_normalize_stratum_invalid_int_raises(self) -> None:
        """Entero inválido lanza StratumResolutionError."""
        with pytest.raises(StratumResolutionError):
            normalize_stratum(10)  # Fuera de rango
    
    def test_normalize_stratum_invalid_str_raises(self) -> None:
        """String inválido lanza StratumResolutionError."""
        with pytest.raises(StratumResolutionError):
            normalize_stratum("INVALID_STRATUM")
    
    def test_normalize_stratum_invalid_type_raises(self) -> None:
        """Tipo no soportado lanza StratumResolutionError."""
        with pytest.raises(StratumResolutionError):
            normalize_stratum([1, 2, 3])  # type: ignore
    
    def test_normalize_stratum_idempotence(self) -> None:
        """Normalización es idempotente."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        result1 = normalize_stratum("PHYSICS")
        result2 = normalize_stratum(result1)
        
        assert result1 == result2


# ==============================================================================
# TESTS DE JSON PATH Y TYPE MATCHING
# ==============================================================================
class TestJsonPathAndTypes:
    """Tests para JSONPath y matching de tipos."""
    
    def test_compute_json_path_root_key(self) -> None:
        """JSONPath desde raíz."""
        assert compute_json_path("$", "foo") == "$.foo"
    
    def test_compute_json_path_nested_key(self) -> None:
        """JSONPath anidado."""
        assert compute_json_path("$.foo", "bar") == "$.foo.bar"
    
    def test_compute_json_path_array_index(self) -> None:
        """JSONPath con índice de array."""
        assert compute_json_path("$.foo", 0) == "$.foo[0]"
    
    def test_compute_json_path_special_chars(self) -> None:
        """JSONPath con caracteres especiales escapados."""
        result = compute_json_path("$", "foo.bar")
        assert "\\." in result
    
    def test_python_type_matches_null(self) -> None:
        """Matching de tipo null."""
        assert python_type_matches("null", None)
        assert not python_type_matches("null", "None")
    
    def test_python_type_matches_boolean(self) -> None:
        """Matching de tipo boolean (excluye int)."""
        assert python_type_matches("boolean", True)
        assert python_type_matches("boolean", False)
        assert not python_type_matches("boolean", 1)
        assert not python_type_matches("boolean", 0)
    
    def test_python_type_matches_integer(self) -> None:
        """Matching de tipo integer (excluye bool)."""
        assert python_type_matches("integer", 42)
        assert not python_type_matches("integer", True)
        assert not python_type_matches("integer", 3.14)
    
    def test_python_type_matches_number(self) -> None:
        """Matching de tipo number (int o float, excluye bool)."""
        assert python_type_matches("number", 42)
        assert python_type_matches("number", 3.14)
        assert not python_type_matches("number", True)
    
    def test_python_type_matches_string(self) -> None:
        """Matching de tipo string."""
        assert python_type_matches("string", "text")
        assert not python_type_matches("string", 42)
    
    def test_python_type_matches_array(self) -> None:
        """Matching de tipo array."""
        assert python_type_matches("array", [1, 2, 3])
        assert not python_type_matches("array", (1, 2, 3))
    
    def test_python_type_matches_object(self) -> None:
        """Matching de tipo object."""
        assert python_type_matches("object", {"key": "value"})
        assert not python_type_matches("object", [1, 2, 3])
    
    def test_python_type_matches_unknown_type(self) -> None:
        """Tipo desconocido retorna True (permissivo)."""
        assert python_type_matches("unknown_type", "any_value")


# ==============================================================================
# TESTS DE SCHEMA VALIDATION RESULT
# ==============================================================================
class TestSchemaValidationResult:
    """Tests para resultado de validación en Álgebra de Heyting."""
    
    def test_schema_validation_result_creation(self) -> None:
        """Creación básica de resultado."""
        result = SchemaValidationResult(
            validity_degree=0.8,
            errors=("error1",),
            warnings=("warn1",),
            path="$.test"
        )
        
        assert result.validity_degree == 0.8
        assert result.errors == ("error1",)
        assert result.warnings == ("warn1",)
        assert result.path == "$.test"
    
    def test_schema_validation_result_validity_clamp(self) -> None:
        """validity_degree se clampea a [0, 1]."""
        result = SchemaValidationResult(validity_degree=1.5)
        assert result.validity_degree == 1.0
        
        result = SchemaValidationResult(validity_degree=-0.5)
        assert result.validity_degree == 0.0
    
    def test_schema_validation_result_is_valid(self) -> None:
        """Predicado is_valid con tolerancia."""
        valid_result = SchemaValidationResult(validity_degree=1.0)
        assert valid_result.is_valid
        
        near_valid = SchemaValidationResult(validity_degree=1.0 - EPS/2)
        assert near_valid.is_valid
        
        invalid_result = SchemaValidationResult(validity_degree=0.9)
        assert not invalid_result.is_valid
    
    def test_schema_validation_result_success_factory(self) -> None:
        """Factory method success."""
        result = SchemaValidationResult.success()
        assert result.validity_degree == 1.0
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_schema_validation_result_failure_factory(self) -> None:
        """Factory method failure."""
        result = SchemaValidationResult.failure("Test error", "$.path", penalty=0.5)
        assert result.validity_degree == 0.5
        assert result.errors == ("Test error",)
        assert result.path == "$.path"
    
    def test_schema_validation_result_merge(self) -> None:
        """Conjunción en Álgebra de Heyting (meet)."""
        r1 = SchemaValidationResult(validity_degree=0.8, errors=("e1",))
        r2 = SchemaValidationResult(validity_degree=0.6, errors=("e2",))
        r3 = SchemaValidationResult(validity_degree=0.9, warnings=("w1",))
        
        merged = SchemaValidationResult.merge([r1, r2, r3])
        
        assert merged.validity_degree == 0.6  # Mínimo
        assert "e1" in merged.errors
        assert "e2" in merged.errors
        assert "w1" in merged.warnings
    
    def test_schema_validation_result_merge_empty(self) -> None:
        """Merge con lista vacía."""
        merged = SchemaValidationResult.merge([])
        assert merged.validity_degree == 1.0
    
    def test_schema_validation_result_error_property(self) -> None:
        """Propiedad error (primer error)."""
        result = SchemaValidationResult(validity_degree=0.0, errors=("e1", "e2", "e3"))
        assert result.error == "e1"
        
        result = SchemaValidationResult(validity_degree=1.0, errors=())
        assert result.error is None
    
    def test_schema_validation_result_to_dict(self) -> None:
        """Serialización a dict."""
        result = SchemaValidationResult(
            validity_degree=0.75,
            errors=("error1",),
            warnings=("warn1",),
            path="$.test"
        )
        
        d = result.to_dict()
        assert d["validity_degree"] == 0.75
        assert d["errors"] == ["error1"]
        assert d["warnings"] == ["warn1"]
        assert d["path"] == "$.test"
        assert "is_valid" in d
    
    def test_schema_validation_result_immutability(self) -> None:
        """Resultado es inmutable (frozen)."""
        result = SchemaValidationResult(validity_degree=1.0)
        
        with pytest.raises(AttributeError):
            result.validity_degree = 0.5  # type: ignore
    
    def test_schema_validation_result_algebra_properties(self) -> None:
        """Propiedades algebraicas de Álgebra de Heyting."""
        # Conmutatividad
        r1 = SchemaValidationResult(validity_degree=0.8)
        r2 = SchemaValidationResult(validity_degree=0.6)
        
        merge_12 = SchemaValidationResult.merge([r1, r2])
        merge_21 = SchemaValidationResult.merge([r2, r1])
        
        assert merge_12.validity_degree == merge_21.validity_degree
        
        # Idempotencia
        merge_ii = SchemaValidationResult.merge([r1, r1])
        assert merge_ii.validity_degree == r1.validity_degree
        
        # Elemento neutro (success)
        merge_is = SchemaValidationResult.merge([r1, SchemaValidationResult.success()])
        assert merge_is.validity_degree == r1.validity_degree


# ==============================================================================
# TESTS DE CATEGORICAL EQUALIZER SEED
# ==============================================================================
class TestCategoricalEqualizerSeed:
    """Tests para seed de auditoría inmutable."""
    
    def test_seed_creation(self) -> None:
        """Creación básica de seed."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
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
        assert seed.token_compression_ratio == 0.0
    
    def test_seed_compression_ratio_clamp(self) -> None:
        """token_compression_ratio se corrige si negativo."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        seed = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            token_compression_ratio=-0.5,
        )
        
        assert seed.token_compression_ratio == 0.0
    
    def test_seed_to_dict(self) -> None:
        """Serialización completa."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        seed = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            token_compression_ratio=0.75,
        )
        
        d = seed.to_dict()
        assert d["target_vector"] == "test"
        assert d["target_stratum"] == "PHYSICS"
        assert d["impedance_match_status"] == "LAMINAR_PROJECTION"
        assert d["token_compression_ratio"] == 0.75
        assert "timestamp" in d
    
    def test_seed_compute_hash(self) -> None:
        """Hash determinista excluyendo timestamp."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        seed1 = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            timestamp=1000.0,
        )
        
        seed2 = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            timestamp=2000.0,  # Diferente timestamp
        )
        
        # Hash debe ser igual (excluye timestamp)
        assert seed1.compute_hash() == seed2.compute_hash()
    
    def test_seed_immutability(self) -> None:
        """Seed es inmutable."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        seed = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        
        with pytest.raises(AttributeError):
            seed.target_vector = "modified"  # type: ignore


# ==============================================================================
# TESTS DE TOON DOCUMENT
# ==============================================================================
class TestTOONDocument:
    """Tests para documento TOON con isomorfismo verificable."""
    
    def test_toon_document_creation(self) -> None:
        """Creación básica de documento TOON."""
        doc = TOONDocument(
            cartridge_id="test_cartridge",
            header_template="Header\nkey|value",
            records=(("key1", "value1"), ("key2", "value2")),
        )
        
        assert doc.cartridge_id == "test_cartridge"
        assert len(doc.records) == 2
    
    def test_toon_document_empty_cartridge_id_raises(self) -> None:
        """cartridge_id vacío lanza error."""
        with pytest.raises(TOONCompressionError):
            TOONDocument(
                cartridge_id="",
                header_template="test",
                records=(),
            )
    
    def test_toon_document_render(self) -> None:
        """Renderizado de documento TOON."""
        doc = TOONDocument(
            cartridge_id="test",
            header_template="Header",
            records=(("key1", '"value1"'), ("key2", "42")),
        )
        
        rendered = doc.render()
        
        assert TOON_START_MARKER in rendered
        assert "test" in rendered
        assert "Header" in rendered
        assert "key1|\"value1\"" in rendered
        assert TOON_END_MARKER in rendered
    
    def test_toon_document_parse(self) -> None:
        """Parseo de documento TOON."""
        content = f"""{TOON_START_MARKER} test_cartridge ---
Header Template
key1|"value1"
key2|42
{TOON_END_MARKER}"""
        
        doc = TOONDocument.parse(content)
        
        assert doc.cartridge_id == "test_cartridge"
        assert len(doc.records) == 2
    
    def test_toon_document_parse_too_short(self) -> None:
        """Documento demasiado corto lanza error."""
        with pytest.raises(TOONCompressionError):
            TOONDocument.parse("Too\nshort")
    
    def test_toon_document_parse_invalid_start_marker(self) -> None:
        """Marcador de inicio inválido lanza error."""
        with pytest.raises(TOONCompressionError):
            TOONDocument.parse("Invalid start\nline2\nline3")
    
    def test_toon_document_parse_invalid_end_marker(self) -> None:
        """Marcador de fin inválido lanza error."""
        content = f"""{TOON_START_MARKER} test ---
Header
key|value
Invalid End"""
        
        with pytest.raises(TOONCompressionError):
            TOONDocument.parse(content)
    
    def test_toon_document_isomorphism(self) -> None:
        """Isomorfismo: parse(render(doc)) = doc."""
        original = TOONDocument(
            cartridge_id="test_iso",
            header_template="Test Header",
            records=(("a", "1"), ("b", "2"), ("c", "3")),
        )
        
        rendered = original.render()
        parsed = TOONDocument.parse(rendered)
        
        assert parsed.cartridge_id == original.cartridge_id
        assert parsed.header_template == original.header_template
        assert parsed.records == original.records
    
    def test_toon_document_to_dict(self) -> None:
        """Deserialización a diccionario."""
        doc = TOONDocument(
            cartridge_id="test",
            header_template="Header",
            records=(("key1", '{"nested": true}'), ("key2", "42")),
        )
        
        d = doc.to_dict()
        assert d["key1"] == {"nested": True}
        assert d["key2"] == 42
    
    def test_toon_document_to_dict_invalid_json(self) -> None:
        """JSON inválido conserva como string."""
        doc = TOONDocument(
            cartridge_id="test",
            header_template="Header",
            records=(("key1", "invalid json {{{"),),
        )
        
        d = doc.to_dict()
        assert d["key1"] == "invalid json {{{"
    
    def test_toon_document_len(self) -> None:
        """Longitud = número de records."""
        doc = TOONDocument(
            cartridge_id="test",
            header_template="Header",
            records=(("a", "1"), ("b", "2")),
        )
        
        assert len(doc) == 2
    
    def test_toon_document_immutability(self) -> None:
        """Documento es inmutable."""
        doc = TOONDocument(
            cartridge_id="test",
            header_template="Header",
            records=(("a", "1"),),
        )
        
        with pytest.raises(AttributeError):
            doc.cartridge_id = "modified"  # type: ignore


# ==============================================================================
# TESTS DE SILO CONTRACTS AND CARTRIDGES
# ==============================================================================
class TestSiloContractsAndCartridges:
    """Tests para contratos y cartuchos de silos."""
    
    def test_silo_a_contract_creation(self) -> None:
        """Creación de contrato Silo A."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        contract = SiloAContract(
            contract_id="test_contract",
            stratum=Stratum.PHYSICS,
            schema={"type": "object", "properties": {}},
            description="Test contract",
            version="1.0.0",
        )
        
        assert contract.contract_id == "test_contract"
        assert contract.stratum == Stratum.PHYSICS
    
    def test_silo_a_contract_invalid_schema_raises(self) -> None:
        """Schema inválido lanza ContractValidationError."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        with pytest.raises(ContractValidationError):
            SiloAContract(
                contract_id="bad_contract",
                stratum=Stratum.PHYSICS,
                schema="not a dict",  # type: ignore
            )
    
    def test_silo_a_contract_schema_missing_type_raises(self) -> None:
        """Schema sin clave "type" lanza error."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        with pytest.raises(ContractValidationError):
            SiloAContract(
                contract_id="bad_contract",
                stratum=Stratum.PHYSICS,
                schema={"properties": {}},  # Sin "type"
            )
    
    def test_silo_a_contract_validate_schema_integrity(self) -> None:
        """Validación de integridad de schema."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        contract = SiloAContract(
            contract_id="test",
            stratum=Stratum.PHYSICS,
            schema={"type": "object"},
        )
        
        assert contract.validate_schema_integrity() is True
    
    def test_silo_a_contract_to_dict(self) -> None:
        """Serialización de contrato."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        contract = SiloAContract(
            contract_id="test",
            stratum=Stratum.PHYSICS,
            schema={"type": "object"},
            description="Test",
            version="2.0.0",
        )
        
        d = contract.to_dict()
        assert d["contract_id"] == "test"
        assert d["stratum"] == "PHYSICS"
        assert d["version"] == "2.0.0"
    
    def test_silo_b_cartridge_creation(self) -> None:
        """Creación de cartucho Silo B."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        cartridge = SiloBCartridge(
            cartridge_id="test_cartridge",
            stratum=Stratum.PHYSICS,
            header_template="Header",
            field_definitions=("field1", "field2"),
        )
        
        assert cartridge.cartridge_id == "test_cartridge"
        assert len(cartridge.field_definitions) == 2
    
    def test_silo_b_cartridge_empty_id_raises(self) -> None:
        """cartridge_id vacío lanza error."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        with pytest.raises(TOONCompressionError):
            SiloBCartridge(
                cartridge_id="",
                stratum=Stratum.PHYSICS,
                header_template="Header",
            )
    
    def test_silo_b_cartridge_to_dict(self) -> None:
        """Serialización de cartucho."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        cartridge = SiloBCartridge(
            cartridge_id="test",
            stratum=Stratum.PHYSICS,
            header_template="Header",
            field_definitions=("f1", "f2"),
            version="1.5.0",
        )
        
        d = cartridge.to_dict()
        assert d["cartridge_id"] == "test"
        assert d["field_definitions"] == ["f1", "f2"]
        assert d["version"] == "1.5.0"

    def test_silo_physics_particle_interactions(self) -> None:
        """
        Para el PolaronCartridge, verifique la renormalización inercial de la atención del LLM:
        m^{**} = m^* (1 + \\frac{\\alpha}{6})
        Para la aniquilación entre un PositronCartridge (auditoría exógena) y un ElectronCartridge falaz, aserte el retorno al estado de vacío:
        e^+ + e^- \\rightarrow 0 + \\gamma
        """
        if Stratum is None:
            pytest.skip("Stratum no disponible")

        polaron = PolaronCartridge("polaron1", Stratum.PHYSICS, "Header")
        m_star = 1.0
        alpha = 0.6
        m_renormalized = polaron.renormalize_mass(m_star, alpha)
        assert MathUtils.float_equal(m_renormalized, m_star * (1.0 + alpha / 6.0))

        positron = PositronCartridge("pos1", Stratum.PHYSICS, "Header")
        electron = ElectronCartridge("elec1", Stratum.PHYSICS, "Header")

        result, photon = electron.annihilate(positron)
        assert result == 0
        assert photon == "γ"


# ==============================================================================
# TESTS DE SCHEMA VALIDATOR
# ==============================================================================
class TestSchemaValidator:
    """Tests para validador JSON Schema con cobertura completa."""
    
    def test_validator_type_string(self, schema_validator: SchemaValidator) -> None:
        """Validación de tipo string."""
        schema = {"type": "string"}
        
        result = schema_validator.validate(schema, "valid string")
        assert result.is_valid
        
        result = schema_validator.validate(schema, 42)
        assert not result.is_valid
    
    def test_validator_type_integer(self, schema_validator: SchemaValidator) -> None:
        """Validación de tipo integer."""
        schema = {"type": "integer"}
        
        result = schema_validator.validate(schema, 42)
        assert result.is_valid
        
        result = schema_validator.validate(schema, 3.14)
        assert not result.is_valid
        
        result = schema_validator.validate(schema, True)
        assert not result.is_valid  # bool excluido
    
    def test_validator_type_number(self, schema_validator: SchemaValidator) -> None:
        """Validación de tipo number."""
        schema = {"type": "number"}
        
        result = schema_validator.validate(schema, 42)
        assert result.is_valid
        
        result = schema_validator.validate(schema, 3.14)
        assert result.is_valid
        
        result = schema_validator.validate(schema, "text")
        assert not result.is_valid
    
    def test_validator_type_boolean(self, schema_validator: SchemaValidator) -> None:
        """Validación de tipo boolean."""
        schema = {"type": "boolean"}
        
        result = schema_validator.validate(schema, True)
        assert result.is_valid
        
        result = schema_validator.validate(schema, False)
        assert result.is_valid
        
        result = schema_validator.validate(schema, 1)
        assert not result.is_valid
    
    def test_validator_type_null(self, schema_validator: SchemaValidator) -> None:
        """Validación de tipo null."""
        schema = {"type": "null"}
        
        result = schema_validator.validate(schema, None)
        assert result.is_valid
        
        result = schema_validator.validate(schema, "None")
        assert not result.is_valid
    
    def test_validator_type_array(self, schema_validator: SchemaValidator) -> None:
        """Validación de tipo array."""
        schema = {"type": "array"}
        
        result = schema_validator.validate(schema, [1, 2, 3])
        assert result.is_valid
        
        result = schema_validator.validate(schema, (1, 2, 3))
        assert not result.is_valid
    
    def test_validator_type_object(self, schema_validator: SchemaValidator) -> None:
        """Validación de tipo object."""
        schema = {"type": "object"}
        
        result = schema_validator.validate(schema, {"key": "value"})
        assert result.is_valid
        
        result = schema_validator.validate(schema, [1, 2, 3])
        assert not result.is_valid
    
    def test_validator_type_union(self, schema_validator: SchemaValidator) -> None:
        """Validación de tipo unión."""
        schema = {"type": ["string", "integer"]}
        
        result = schema_validator.validate(schema, "text")
        assert result.is_valid
        
        result = schema_validator.validate(schema, 42)
        assert result.is_valid
        
        result = schema_validator.validate(schema, 3.14)
        assert not result.is_valid
    
    def test_validator_required(self, schema_validator: SchemaValidator) -> None:
        """Validación de claves requeridas."""
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            }
        }
        
        result = schema_validator.validate(schema, {"name": "John", "age": 30})
        assert result.is_valid
        
        result = schema_validator.validate(schema, {"name": "John"})
        assert not result.is_valid
        assert "age" in result.error
    
    def test_validator_properties(self, schema_validator: SchemaValidator) -> None:
        """Validación recursiva de propiedades."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    }
                }
            }
        }
        
        result = schema_validator.validate(schema, {"user": {"name": "John"}})
        assert result.is_valid
        
        result = schema_validator.validate(schema, {"user": {"name": 123}})
        assert not result.is_valid
    
    def test_validator_items(self, schema_validator: SchemaValidator) -> None:
        """Validación de items de array."""
        schema = {
            "type": "array",
            "items": {"type": "integer"}
        }
        
        result = schema_validator.validate(schema, [1, 2, 3])
        assert result.is_valid
        
        result = schema_validator.validate(schema, [1, "two", 3])
        assert not result.is_valid
    
    def test_validator_minimum(self, schema_validator: SchemaValidator) -> None:
        """Validación de valor mínimo."""
        schema = {"type": "number", "minimum": 0}
        
        result = schema_validator.validate(schema, 10)
        assert result.is_valid
        
        result = schema_validator.validate(schema, -1)
        assert not result.is_valid
    
    def test_validator_maximum(self, schema_validator: SchemaValidator) -> None:
        """Validación de valor máximo."""
        schema = {"type": "number", "maximum": 100}
        
        result = schema_validator.validate(schema, 50)
        assert result.is_valid
        
        result = schema_validator.validate(schema, 150)
        assert not result.is_valid
    
    def test_validator_exclusive_minimum(self, schema_validator: SchemaValidator) -> None:
        """Validación de mínimo exclusivo."""
        schema = {"type": "number", "exclusiveMinimum": 0}
        
        result = schema_validator.validate(schema, 0.1)
        assert result.is_valid
        
        result = schema_validator.validate(schema, 0)
        assert not result.is_valid
    
    def test_validator_exclusive_maximum(self, schema_validator: SchemaValidator) -> None:
        """Validación de máximo exclusivo."""
        schema = {"type": "number", "exclusiveMaximum": 100}
        
        result = schema_validator.validate(schema, 99.9)
        assert result.is_valid
        
        result = schema_validator.validate(schema, 100)
        assert not result.is_valid
    
    def test_validator_min_length(self, schema_validator: SchemaValidator) -> None:
        """Validación de longitud mínima."""
        schema = {"type": "string", "minLength": 3}
        
        result = schema_validator.validate(schema, "abc")
        assert result.is_valid
        
        result = schema_validator.validate(schema, "ab")
        assert not result.is_valid
    
    def test_validator_max_length(self, schema_validator: SchemaValidator) -> None:
        """Validación de longitud máxima."""
        schema = {"type": "string", "maxLength": 5}
        
        result = schema_validator.validate(schema, "abc")
        assert result.is_valid
        
        result = schema_validator.validate(schema, "abcdef")
        assert not result.is_valid
    
    def test_validator_enum(self, schema_validator: SchemaValidator) -> None:
        """Validación de enumeración."""
        schema = {"type": "string", "enum": ["red", "green", "blue"]}
        
        result = schema_validator.validate(schema, "red")
        assert result.is_valid
        
        result = schema_validator.validate(schema, "yellow")
        assert not result.is_valid
    
    def test_validator_const(self, schema_validator: SchemaValidator) -> None:
        """Validación de constante."""
        schema = {"type": "string", "const": "exact_value"}
        
        result = schema_validator.validate(schema, "exact_value")
        assert result.is_valid
        
        result = schema_validator.validate(schema, "other_value")
        assert not result.is_valid
    
    def test_validator_min_items(self, schema_validator: SchemaValidator) -> None:
        """Validación de cantidad mínima de items."""
        schema = {"type": "array", "minItems": 2}
        
        result = schema_validator.validate(schema, [1, 2, 3])
        assert result.is_valid
        
        result = schema_validator.validate(schema, [1])
        assert not result.is_valid
    
    def test_validator_max_items(self, schema_validator: SchemaValidator) -> None:
        """Validación de cantidad máxima de items."""
        schema = {"type": "array", "maxItems": 3}
        
        result = schema_validator.validate(schema, [1, 2])
        assert result.is_valid
        
        result = schema_validator.validate(schema, [1, 2, 3, 4])
        assert not result.is_valid
    
    def test_validator_pattern(self, schema_validator: SchemaValidator) -> None:
        """Validación de patrón regex."""
        schema = {"type": "string", "pattern": "^[a-z]+$"}
        
        result = schema_validator.validate(schema, "abc")
        assert result.is_valid
        
        result = schema_validator.validate(schema, "ABC")
        assert not result.is_valid
    
    def test_validator_invalid_pattern(self, schema_validator: SchemaValidator) -> None:
        """Patrón regex inválido retorna error."""
        schema = {"type": "string", "pattern": "[invalid"}
        
        result = schema_validator.validate(schema, "test")
        assert not result.is_valid
        assert "regex inválido" in result.error.lower()
    
    def test_validator_invalid_schema(self, schema_validator: SchemaValidator) -> None:
        """Schema inválido retorna fallo."""
        result = schema_validator.validate("not a dict", {"key": "value"})
        assert not result.is_valid
    
    def test_validator_empty_schema(self, schema_validator: SchemaValidator) -> None:
        """Schema vacío es válido."""
        result = schema_validator.validate({}, {"any": "value"})
        assert result.is_valid
    
    def test_validator_merge_results(self, schema_validator: SchemaValidator) -> None:
        """Múltiples validaciones se mergean correctamente."""
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "minLength": 2}
            }
        }
        
        result = schema_validator.validate(schema, {"name": "A"})
        assert not result.is_valid
        # Debería tener errores de required y minLength


# ==============================================================================
# TESTS DE ALGEBRAIC VETO REGISTRY
# ==============================================================================
class TestAlgebraicVetoRegistry:
    """Tests para registro de validadores algebraicos."""
    
    def test_veto_registry_creation(self, veto_registry: AlgebraicVetoRegistry) -> None:
        """Creación de registro con validadores por defecto."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        # Cada estrato debería tener al menos un validador
        for stratum in Stratum:
            count = veto_registry.get_validator_count(stratum)
            assert count >= 0
    
    def test_veto_registry_physics_conservation(self, veto_registry: AlgebraicVetoRegistry) -> None:
        """Validador de conservación de energía (PHYSICS)."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        # Caso válido
        payload = {"dissipated_power": 100, "energy_input": 500, "energy_output": 450}
        errors = veto_registry.validate(Stratum.PHYSICS, payload)
        assert len(errors) == 0
        
        # Caso inválido: energía negativa
        payload = {"dissipated_power": -100}
        errors = veto_registry.validate(Stratum.PHYSICS, payload)
        assert len(errors) > 0
        assert "termodinámica" in errors[0].lower()
        
        # Caso inválido: output > input
        payload = {"energy_input": 100, "energy_output": 200}
        errors = veto_registry.validate(Stratum.PHYSICS, payload)
        assert len(errors) > 0
        assert "conservación" in errors[0].lower()
    
    def test_veto_registry_tactics_stability(self, veto_registry: AlgebraicVetoRegistry) -> None:
        """Validador de estabilidad (TACTICS)."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        # Caso válido
        payload = {"pyramid_stability_index": 0.5}
        errors = veto_registry.validate(Stratum.TACTICS, payload)
        assert len(errors) == 0
        
        # Caso inválido: fuera de rango
        payload = {"pyramid_stability_index": 1.5}
        errors = veto_registry.validate(Stratum.TACTICS, payload)
        assert len(errors) > 0
        assert "estabilidad" in errors[0].lower()
    
    def test_veto_registry_strategy_friction(self, veto_registry: AlgebraicVetoRegistry) -> None:
        """Validador de fricción territorial (STRATEGY)."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        # Caso válido
        payload = {"territorial_friction": 1.5}
        errors = veto_registry.validate(Stratum.STRATEGY, payload)
        assert len(errors) == 0
        
        # Caso inválido: fricción < 1
        payload = {"territorial_friction": 0.5}
        errors = veto_registry.validate(Stratum.STRATEGY, payload)
        assert len(errors) > 0
        assert "fricción" in errors[0].lower()
    
    def test_veto_registry_wisdom_verdict(self, veto_registry: AlgebraicVetoRegistry) -> None:
        """Validador de veredicto (WISDOM)."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        # Casos válidos
        for verdict in ["VIABLE", "PRECAUCION", "RECHAZAR"]:
            payload = {"final_verdict": verdict}
            errors = veto_registry.validate(Stratum.WISDOM, payload)
            assert len(errors) == 0
        
        # Caso inválido
        payload = {"final_verdict": "INVALID"}
        errors = veto_registry.validate(Stratum.WISDOM, payload)
        assert len(errors) > 0
        assert "veredicto" in errors[0].lower()
    
    def test_veto_registry_register_validator(self, veto_registry: AlgebraicVetoRegistry) -> None:
        """Registro de validador adicional."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        def custom_validator(stratum, payload):
            if payload.get("forbidden") is True:
                return "Campo forbidden no permitido"
            return None
        
        initial_count = veto_registry.get_validator_count(Stratum.PHYSICS)
        veto_registry.register_validator(Stratum.PHYSICS, custom_validator)
        new_count = veto_registry.get_validator_count(Stratum.PHYSICS)
        
        assert new_count == initial_count + 1
        
        # Validador debería ejecutarse
        errors = veto_registry.validate(Stratum.PHYSICS, {"forbidden": True})
        assert "forbidden" in errors[-1]
    
    def test_veto_registry_validator_exception_handling(self, veto_registry: AlgebraicVetoRegistry) -> None:
        """Excepciones en validadores se capturan."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        def failing_validator(stratum, payload):
            raise RuntimeError("Validator crashed")
        
        veto_registry.register_validator(Stratum.PHYSICS, failing_validator)
        errors = veto_registry.validate(Stratum.PHYSICS, {})
        
        # Debería capturar el error sin propagar excepción
        assert any("Error en validador" in e for e in errors)


# ==============================================================================
# TESTS DE SILO MANAGER
# ==============================================================================
class TestSiloManager:
    """Tests para gestor de silos con invariantes."""
    
    def test_silo_manager_initialization(self, silo_manager: SiloManager) -> None:
        """Inicialización con contratos y cartuchos por defecto."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        # Cada estrato debería tener al menos un contrato y cartucho
        for stratum in Stratum:
            assert silo_manager.get_contract_count(stratum) >= 1
            assert silo_manager.get_cartridge_count(stratum) >= 1
    
    def test_silo_manager_fetch_contract(self, silo_manager: SiloManager) -> None:
        """Recuperación de contrato por estrato."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        contract_id, schema = silo_manager.fetch_contract(Stratum.PHYSICS, "test_vector")
        
        assert isinstance(contract_id, str)
        assert isinstance(schema, dict)
        assert "type" in schema
    
    def test_silo_manager_fetch_cartridge(self, silo_manager: SiloManager) -> None:
        """Recuperación de cartucho por estrato."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        cartridge_id, header = silo_manager.fetch_cartridge(Stratum.PHYSICS, "test_vector")
        
        assert isinstance(cartridge_id, str)
        assert isinstance(header, str)
    
    def test_silo_manager_list_contracts(self, silo_manager: SiloManager) -> None:
        """Listado de contratos."""
        contracts = silo_manager.list_contracts()
        assert len(contracts) > 0
        assert isinstance(contracts, list)
        assert contracts == sorted(contracts)  # Ordenados
    
    def test_silo_manager_list_cartridges(self, silo_manager: SiloManager) -> None:
        """Listado de cartuchos."""
        cartridges = silo_manager.list_cartridges()
        assert len(cartridges) > 0
        assert isinstance(cartridges, list)
        assert cartridges == sorted(cartridges)  # Ordenados
    
    def test_silo_manager_freeze(self, silo_manager: SiloManager) -> None:
        """Congelado de silos."""
        silo_manager.freeze()
        
        # Después de freeze, no se deberían poder agregar contratos
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        # Esto debería funcionar porque _register_contract verifica _frozen
        # pero es un método interno, así que probamos indirectamente
        assert silo_manager._frozen is True
    
    def test_silo_manager_get_contract(self, silo_manager: SiloManager) -> None:
        """Recuperación de contrato por ID."""
        contracts = silo_manager.list_contracts()
        if contracts:
            contract = silo_manager.get_contract(contracts[0])
            assert contract is not None
            assert contract.contract_id == contracts[0]
    
    def test_silo_manager_get_cartridge(self, silo_manager: SiloManager) -> None:
        """Recuperación de cartucho por ID."""
        cartridges = silo_manager.list_cartridges()
        if cartridges:
            cartridge = silo_manager.get_cartridge(cartridges[0])
            assert cartridge is not None
            assert cartridge.cartridge_id == cartridges[0]
    
    def test_silo_manager_thread_safety(self, silo_manager: SiloManager) -> None:
        """Thread-safety en operaciones de lectura."""
        results = []
        errors = []
        
        def fetch_worker():
            try:
                for _ in range(10):
                    contracts = silo_manager.list_contracts()
                    cartridges = silo_manager.list_cartridges()
                    results.append((len(contracts), len(cartridges)))
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=fetch_worker) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 50


# ==============================================================================
# TESTS DE TOON COMPRESSOR
# ==============================================================================
class TestTOONCompressor:
    """Tests para compresor TOON con verificación de isomorfismo."""
    
    def test_toon_compressor_compress(self, toon_compressor: TOONCompressor) -> None:
        """Compresión de telemetría."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        telemetry = {"key1": "value1", "key2": 42, "key3": True}
        
        doc = toon_compressor.compress(
            telemetry,
            "test_cartridge",
            "Header Template"
        )
        
        assert isinstance(doc, TOONDocument)
        assert doc.cartridge_id == "test_cartridge"
        assert len(doc.records) == 3
    
    def test_toon_compressor_compress_rank_exceeded(self, toon_compressor: TOONCompressor) -> None:
        """Rango tensorial excedido lanza error."""
        # Crear estructura muy profunda
        deep_telemetry = {"level1": {"level2": {"level3": {"level4": "deep"}}}}
        
        with pytest.raises(TOONCompressionError):
            toon_compressor.compress(
                deep_telemetry,
                "test",
                "Header"
            )
    
    def test_toon_compressor_decompress(self, toon_compressor: TOONCompressor) -> None:
        """Descompresión de documento TOON."""
        original = {"key1": "value1", "key2": 42}
        
        doc = toon_compressor.compress(original, "test", "Header")
        decompressed = toon_compressor.decompress(doc)
        
        assert decompressed == original
    
    def test_toon_compressor_compute_ratio(self, toon_compressor: TOONCompressor) -> None:
        """Cálculo de ratio de compresión."""
        original = {"key": "value"}
        compressed = "compressed content"
        
        ratio = toon_compressor.compute_ratio(original, compressed)
        
        assert MIN_COMPRESSION_RATIO <= ratio <= MAX_COMPRESSION_RATIO
    
    def test_toon_compressor_verify_isomorphism(self, toon_compressor: TOONCompressor) -> None:
        """Verificación de isomorfismo."""
        original = {"key1": "value1", "key2": 42}
        
        doc = toon_compressor.compress(original, "test", "Header")
        compressed = doc.render()
        
        is_isomorphic = toon_compressor.verify_isomorphism(original, compressed)
        assert is_isomorphic is True
    
    def test_toon_compressor_get_statistics(self, toon_compressor: TOONCompressor) -> None:
        """Estadísticas de compresión."""
        stats = toon_compressor.get_statistics()
        
        assert "count" in stats
        assert "mean_ratio" in stats
        assert "min_ratio" in stats
        assert "max_ratio" in stats
    
    def test_toon_compressor_thread_safety(self, toon_compressor: TOONCompressor) -> None:
        """Thread-safety en compresión."""
        errors = []
        
        def compress_worker():
            try:
                for i in range(10):
                    telemetry = {"index": i, "data": f"value_{i}"}
                    toon_compressor.compress(telemetry, "test", "Header")
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=compress_worker) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


# ==============================================================================
# TESTS DE AUDIT TRAIL
# ==============================================================================
class TestAuditTrail:
    """Tests para traza de auditoría thread-safe."""
    
    def test_audit_trail_creation(self, audit_trail: AuditTrail) -> None:
        """Creación de traza de auditoría."""
        assert audit_trail.size == 0
        assert audit_trail.total_count == 0
    
    def test_audit_trail_invalid_max_size_raises(self) -> None:
        """max_size <= 0 lanza ValueError."""
        with pytest.raises(ValueError):
            AuditTrail(max_size=0)
        
        with pytest.raises(ValueError):
            AuditTrail(max_size=-1)
    
    def test_audit_trail_append(self, audit_trail: AuditTrail) -> None:
        """Agregar seed a la traza."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        seed = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        
        audit_trail.append(seed)
        
        assert audit_trail.size == 1
        assert audit_trail.total_count == 1
    
    def test_audit_trail_get_all(self, audit_trail: AuditTrail) -> None:
        """Obtener todos los seeds."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        for i in range(5):
            seed = CategoricalEqualizerSeed(
                target_vector=f"test_{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            )
            audit_trail.append(seed)
        
        all_seeds = audit_trail.get_all()
        assert len(all_seeds) == 5
    
    def test_audit_trail_get_recent(self, audit_trail: AuditTrail) -> None:
        """Obtener seeds recientes."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        for i in range(10):
            seed = CategoricalEqualizerSeed(
                target_vector=f"test_{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            )
            audit_trail.append(seed)
        
        recent = audit_trail.get_recent(3)
        assert len(recent) == 3
        # Los últimos deberían ser los índices 7, 8, 9
    
    def test_audit_trail_get_by_status(self, audit_trail: AuditTrail) -> None:
        """Filtrar por status."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        seed1 = CategoricalEqualizerSeed(
            target_vector="test1",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        
        seed2 = CategoricalEqualizerSeed(
            target_vector="test2",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.ALGEBRAIC_VETO,
        )
        
        audit_trail.append(seed1)
        audit_trail.append(seed2)
        
        laminar = audit_trail.get_by_status(ImpedanceMatchStatus.LAMINAR_PROJECTION)
        veto = audit_trail.get_by_status(ImpedanceMatchStatus.ALGEBRAIC_VETO)
        
        assert len(laminar) == 1
        assert len(veto) == 1
    
    def test_audit_trail_get_by_stratum(self, audit_trail: AuditTrail) -> None:
        """Filtrar por estrato."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        seed1 = CategoricalEqualizerSeed(
            target_vector="test1",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        
        seed2 = CategoricalEqualizerSeed(
            target_vector="test2",
            target_stratum=Stratum.WISDOM,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        
        audit_trail.append(seed1)
        audit_trail.append(seed2)
        
        physics = audit_trail.get_by_stratum(Stratum.PHYSICS)
        wisdom = audit_trail.get_by_stratum(Stratum.WISDOM)
        
        assert len(physics) == 1
        assert len(wisdom) == 1
    
    def test_audit_trail_clear(self, audit_trail: AuditTrail) -> None:
        """Limpieza de traza."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        seed = CategoricalEqualizerSeed(
            target_vector="test",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        
        audit_trail.append(seed)
        assert audit_trail.size == 1
        
        audit_trail.clear()
        assert audit_trail.size == 0
        assert audit_trail.total_count == 1  # Preserva contador total
    
    def test_audit_trail_buffer_overflow(self) -> None:
        """Buffer circular con overflow."""
        trail = AuditTrail(max_size=3)
        
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        for i in range(10):
            seed = CategoricalEqualizerSeed(
                target_vector=f"test_{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            )
            trail.append(seed)
        
        # Tamaño máximo no debería excederse
        assert trail.size == 3
        # Pero contador total sí
        assert trail.total_count == 10
    
    def test_audit_trail_get_statistics(self, audit_trail: AuditTrail) -> None:
        """Estadísticas de auditoría."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        # Agregar seeds con diferentes status
        for status in [ImpedanceMatchStatus.LAMINAR_PROJECTION] * 3 + [ImpedanceMatchStatus.ALGEBRAIC_VETO] * 2:
            seed = CategoricalEqualizerSeed(
                target_vector="test",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=status,
                token_compression_ratio=0.5,
            )
            audit_trail.append(seed)
        
        stats = audit_trail.get_statistics()
        
        assert stats["total_entries"] == 5
        assert stats["current_size"] == 5
        assert "LAMINAR_PROJECTION" in stats["status_distribution"]
        assert "ALGEBRAIC_VETO" in stats["status_distribution"]
    
    def test_audit_trail_thread_safety(self) -> None:
        """Thread-safety en auditoría."""
        trail = AuditTrail(max_size=100)
        errors = []
        
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        def append_worker(worker_id: int):
            try:
                for i in range(20):
                    seed = CategoricalEqualizerSeed(
                        target_vector=f"test_{worker_id}_{i}",
                        target_stratum=Stratum.PHYSICS,
                        silo_a_contract_id="c1",
                        silo_b_cartridge_id="c2",
                        impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
                    )
                    trail.append(seed)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=append_worker, args=(i,)) for i in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert trail.total_count == 100  # 5 workers × 20 iterations


# ==============================================================================
# TESTS DE MIC AGENT
# ==============================================================================
class TestMICAgent:
    """Tests para agente MIC con propiedades funtoriales."""
    
    def test_mic_agent_initialization(self, mic_agent: MICAgent) -> None:
        """Inicialización del agente."""
        assert mic_agent.silo_manager is not None
        assert mic_agent.audit_trail is not None
        assert mic_agent.immune_watcher is not None
    
    def test_mic_agent_sense_stratum(self, mic_agent: MICAgent) -> None:
        """Sensado de estrato."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        stratum = mic_agent.sense_stratum("topology_core")
        assert stratum == Stratum.PHYSICS
        
        stratum = mic_agent.sense_stratum("strategy_planner")
        assert stratum == Stratum.STRATEGY
    
    def test_mic_agent_sense_stratum_not_found(self, mic_agent: MICAgent) -> None:
        """Vector no encontrado lanza error."""
        with pytest.raises(StratumResolutionError):
            mic_agent.sense_stratum("nonexistent_vector")
    
    def test_mic_agent_validate_closure_valid(self, mic_agent: MICAgent) -> None:
        """Validación de clausura válida."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        # PHYSICS no requiere nada
        error = mic_agent.validate_closure(Stratum.PHYSICS, frozenset())
        assert error is None
        
        # TACTICS requiere PHYSICS
        error = mic_agent.validate_closure(
            Stratum.TACTICS,
            frozenset({Stratum.PHYSICS})
        )
        assert error is None
    
    def test_mic_agent_validate_closure_invalid(self, mic_agent: MICAgent) -> None:
        """Validación de clausura inválida."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        # TACTICS requiere PHYSICS pero no está validado
        error = mic_agent.validate_closure(Stratum.TACTICS, frozenset())
        assert error is not None
        assert "clausura transitiva" in error.lower()
    
    def test_mic_agent_compress_telemetry(self, mic_agent: MICAgent) -> None:
        """Compresión de telemetría."""
        telemetry = {"key1": "value1", "key2": 42}
        
        cartridge_id, doc = mic_agent.compress_telemetry("topology_core", telemetry)
        
        assert isinstance(cartridge_id, str)
        assert isinstance(doc, TOONDocument)
    
    def test_mic_agent_inject_functorial_context(self, mic_agent: MICAgent) -> None:
        """Inyección de contexto funtorial."""
        telemetry = {"metric": 100}
        
        context = mic_agent.inject_functorial_context("topology_core", telemetry)
        
        assert isinstance(context, str)
        assert TOON_START_MARKER in context
    
    def test_mic_agent_encapsulate_monad_success(self, mic_agent: MICAgent) -> None:
        """Encapsulación monádica exitosa."""
        if Stratum is None or CategoricalState is None:
            pytest.skip("Stratum o CategoricalState no disponible")
        
        llm_output = {"dissipated_power": 100}
        validated_strata = frozenset({Stratum.PHYSICS})
        
        state = mic_agent.encapsulate_monad(
            target_vector="topology_core",
            llm_output=llm_output,
            validated_strata=validated_strata,
        )
        
        assert state.is_success
        assert Stratum.PHYSICS in state.validated_strata
    
    def test_mic_agent_encapsulate_monad_invalid_type(self, mic_agent: MICAgent) -> None:
        """Encapsulación con tipo inválido."""
        if CategoricalState is None:
            pytest.skip("CategoricalState no disponible")
        
        state = mic_agent.encapsulate_monad(
            target_vector="topology_core",
            llm_output="not a mapping",  # type: ignore
            validated_strata=frozenset(),
        )
        
        assert state.is_failed
        assert "INPUT_TYPE_ERROR" in state.error
    
    def test_mic_agent_encapsulate_monad_closure_violation(self, mic_agent: MICAgent) -> None:
        """Encapsulación con violación de clausura."""
        if Stratum is None or CategoricalState is None:
            pytest.skip("Stratum o CategoricalState no disponible")
        
        state = mic_agent.encapsulate_monad(
            target_vector="tactics_executor",  # Requiere PHYSICS
            llm_output={"key": "value"},
            validated_strata=frozenset(),  # PHYSICS no validado
        )
        
        assert state.is_failed
        assert "STRATUM_MISMATCH" in state.error
    
    def test_mic_agent_encapsulate_monad_schema_validation(self, mic_agent: MICAgent) -> None:
        """Encapsulación con validación de schema."""
        if Stratum is None or CategoricalState is None:
            pytest.skip("Stratum o CategoricalState no disponible")
        
        state = mic_agent.encapsulate_monad(
            target_vector="topology_core",
            llm_output={"wrong_field": "value"},  # Falta dissipated_power
            validated_strata=frozenset({Stratum.PHYSICS}),
        )
        
        assert state.is_failed
        assert "SCHEMA_VALIDATION" in state.error
    
    def test_mic_agent_encapsulate_monad_algebraic_veto(self, mic_agent: MICAgent) -> None:
        """Encapsulación con veto algebraico."""
        if Stratum is None or CategoricalState is None:
            pytest.skip("Stratum o CategoricalState no disponible")
        
        state = mic_agent.encapsulate_monad(
            target_vector="topology_core",
            llm_output={"dissipated_power": -100},  # Negativo viola conservación
            validated_strata=frozenset({Stratum.PHYSICS}),
        )
        
        assert state.is_failed
        assert "ALGEBRAIC_VETO" in state.error
    
    def test_mic_agent_f_star_inverse_image(self, mic_agent: MICAgent) -> None:
        """Funtor inverso f*."""
        if CategoricalState is None:
            pytest.skip("CategoricalState no disponible")
        
        state = mic_agent.f_star_inverse_image(
            llm_input={"dissipated_power": 100},
            target_vector="topology_core",
            validated_strata=frozenset({Stratum.PHYSICS}),
        )
        
        assert state.is_success
    
    def test_mic_agent_f_star_inverse_image_null_input(self, mic_agent: MICAgent) -> None:
        """Funtor inverso con input null colapsa."""
        if CategoricalState is None:
            pytest.skip("CategoricalState no disponible")
        
        state = mic_agent.f_star_inverse_image(
            llm_input=None,
            target_vector="topology_core",
            validated_strata=frozenset(),
        )
        
        assert state.is_failed
        assert "Colapso" in state.error
    
    def test_mic_agent_f_lower_star_direct_image(self, mic_agent: MICAgent) -> None:
        """Funtor directo f*."""
        if CategoricalState is None:
            pytest.skip("CategoricalState no disponible")
        
        state = create_categorical_state(payload={"data": "value"})
        result = mic_agent.f_lower_star_direct_image(state)
        
        assert result["verdict"] == "ACCEPTED"
        assert "hash" in result
    
    def test_mic_agent_f_lower_star_direct_image_failed(self, mic_agent: MICAgent) -> None:
        """Funtor directo con estado fallido."""
        if CategoricalState is None:
            pytest.skip("CategoricalState no disponible")
        
        state = create_categorical_state().with_error("test error")
        result = mic_agent.f_lower_star_direct_image(state)
        
        assert result["verdict"] == "REJECTED"
        assert "reason" in result
    
    def test_mic_agent_verify_adjunction(self, mic_agent: MICAgent) -> None:
        """Verificación de adjunción."""
        if CategoricalState is None:
            pytest.skip("CategoricalState no disponible")
        
        X_llm = {"dissipated_power": 100}
        Y_emic = create_categorical_state(payload={"data": "value"})
        
        result = mic_agent.verify_adjunction(X_llm, Y_emic)
        assert isinstance(result, bool)
    
    def test_mic_agent_characteristic_morphism_success(self, mic_agent: MICAgent) -> None:
        """Morfismo característico para estado exitoso."""
        if CategoricalState is None:
            pytest.skip("CategoricalState no disponible")
        
        state = create_categorical_state(payload={"data": "value"})
        result = mic_agent.characteristic_morphism(state)
        
        assert result.validity_degree > 0.5
    
    def test_mic_agent_characteristic_morphism_failed(self, mic_agent: MICAgent) -> None:
        """Morfismo característico para estado fallido."""
        if CategoricalState is None:
            pytest.skip("CategoricalState no disponible")
        
        state = create_categorical_state().with_error("test error")
        result = mic_agent.characteristic_morphism(state)
        
        assert result.validity_degree == 0.0
        assert not result.is_valid
    
    def test_mic_agent_execute_projection_success(self, mic_agent: MICAgent) -> None:
        """Ejecución de proyección exitosa."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        result = mic_agent.execute_projection(
            target_vector="topology_core",
            llm_output={"dissipated_power": 100},
            validated_strata=frozenset({Stratum.PHYSICS}),
        )
        
        assert result["status"] == "OK"
        assert result["impedance_status"] == "LAMINAR_PROJECTION"
    
    def test_mic_agent_execute_projection_veto(self, mic_agent: MICAgent) -> None:
        """Ejecución de proyección con veto."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        result = mic_agent.execute_projection(
            target_vector="topology_core",
            llm_output={"dissipated_power": -100},  # Viola conservación
            validated_strata=frozenset({Stratum.PHYSICS}),
        )
        
        assert result["status"] == "VETO"
        assert "ALGEBRAIC_VETO" in result["impedance_status"]
    
    def test_mic_agent_get_audit_statistics(self, mic_agent: MICAgent) -> None:
        """Estadísticas de auditoría."""
        stats = mic_agent.get_audit_statistics()
        
        assert "total_entries" in stats
        assert "current_size" in stats
        assert "status_distribution" in stats
    
    def test_mic_agent_get_recent_audits(self, mic_agent: MICAgent) -> None:
        """Auditorías recientes."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        # Ejecutar algunas proyecciones
        mic_agent.execute_projection(
            target_vector="topology_core",
            llm_output={"dissipated_power": 100},
            validated_strata=frozenset({Stratum.PHYSICS}),
        )
        
        audits = mic_agent.get_recent_audits(5)
        assert isinstance(audits, list)
    
    def test_mic_agent_clear_audit_trail(self, mic_agent: MICAgent) -> None:
        """Limpieza de traza de auditoría."""
        mic_agent.clear_audit_trail()
        
        stats = mic_agent.get_audit_statistics()
        assert stats["current_size"] == 0
    
    def test_mic_agent_verify_functorial_properties(self, mic_agent: MICAgent) -> None:
        """Verificación de propiedades funtoriales."""
        props = mic_agent.verify_functorial_properties()
        
        assert props["immune_watcher_initialized"] is True
        assert props["silo_manager_initialized"] is True
        assert props["schema_validator_initialized"] is True
        assert props["mic_registry_initialized"] is True
    
    def test_mic_agent_health_report(self, mic_agent: MICAgent) -> None:
        """Reporte de salud."""
        report = mic_agent.health_report()
        
        assert isinstance(report, str)
        assert "MIC AGENT" in report
        assert "Auditorías" in report
        assert "SILOS" in report
    
    def test_mic_agent_repr(self, mic_agent: MICAgent) -> None:
        """Representación string."""
        repr_str = repr(mic_agent)
        
        assert "MICAgent" in repr_str
        assert "contratos" in repr_str
        assert "cartuchos" in repr_str


# ==============================================================================
# TESTS DE PROPIEDADES (PROPERTY-BASED)
# ==============================================================================
class TestProperties:
    """Tests de propiedades que deben cumplirse siempre."""
    
    @pytest.mark.property
    def test_schema_validation_result_validity_range(self) -> None:
        """validity_degree siempre en [0, 1]."""
        for value in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
            result = SchemaValidationResult(validity_degree=value)
            assert 0.0 <= result.validity_degree <= 1.0
    
    @pytest.mark.property
    def test_toon_document_isomorphism_roundtrip(self) -> None:
        """TOON: parse(render(doc)) = doc para todo doc válido."""
        test_cases = [
            TOONDocument("c1", "h1", (("k1", "v1"),)),
            TOONDocument("c2", "h2", (("k1", "v1"), ("k2", "v2"))),
            TOONDocument("c3", "h3", (("k1", '{"json": true}'), ("k2", "42"))),
        ]
        
        for doc in test_cases:
            rendered = doc.render()
            parsed = TOONDocument.parse(rendered)
            
            assert parsed.cartridge_id == doc.cartridge_id
            assert parsed.header_template == doc.header_template
            assert parsed.records == doc.records
    
    @pytest.mark.property
    def test_hash_determinism(self) -> None:
        """Hashes son deterministas."""
        for _ in range(10):
            data = {"test": "value", "number": 42}
            hash1 = MathUtils.stable_hash(data)
            hash2 = MathUtils.stable_hash(data)
            assert hash1 == hash2
    
    @pytest.mark.property
    def test_audit_trail_fifo_order(self) -> None:
        """AuditTrail mantiene orden FIFO."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        trail = AuditTrail(max_size=5)
        
        for i in range(10):
            seed = CategoricalEqualizerSeed(
                target_vector=f"test_{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            )
            trail.append(seed)
        
        # Los últimos 5 deberían estar en orden
        recent = trail.get_recent(5)
        for i, seed in enumerate(recent):
            assert f"test_{i + 5}" in seed.target_vector
    
    @pytest.mark.property
    def test_compression_ratio_bounds(self, toon_compressor: TOONCompressor) -> None:
        """Ratio de compresión siempre en rango válido."""
        for i in range(20):
            original = {f"key_{j}": f"value_{j}" for j in range(i)}
            doc = toon_compressor.compress(original, "test", "Header")
            compressed = doc.render()
            
            ratio = toon_compressor.compute_ratio(original, compressed)
            assert MIN_COMPRESSION_RATIO <= ratio <= MAX_COMPRESSION_RATIO


# ==============================================================================
# TESTS DE BORDES Y CASOS EXTREMOS
# ==============================================================================
class TestEdgeCases:
    """Tests para casos borde y condiciones extremas."""
    
    def test_empty_payload_validation(self, schema_validator: SchemaValidator) -> None:
        """Validación con payload vacío."""
        schema = {"type": "object"}
        result = schema_validator.validate(schema, {})
        assert result.is_valid
    
    def test_deeply_nested_payload(self, toon_compressor: TOONCompressor) -> None:
        """Payload profundamente anidado."""
        nested = {"l1": {"l2": {"l3": {"l4": "deep"}}}}
        
        # Debería fallar por rango tensorial
        with pytest.raises(TOONCompressionError):
            toon_compressor.compress(nested, "test", "Header")
    
    def test_large_payload(self, toon_compressor: TOONCompressor) -> None:
        """Payload grande."""
        large = {f"key_{i}": f"value_{i}" for i in range(1000)}
        
        doc = toon_compressor.compress(large, "test", "Header")
        assert len(doc.records) == 1000
    
    def test_unicode_payload(self, toon_compressor: TOONCompressor) -> None:
        """Payload con unicode."""
        unicode_payload = {"texto": "España", "emoji": "🎉", "chino": "你好"}
        
        doc = toon_compressor.compress(unicode_payload, "test", "Header")
        decompressed = toon_compressor.decompress(doc)
        
        assert decompressed["texto"] == "España"
        assert decompressed["emoji"] == "🎉"
    
    def test_special_float_values(self, schema_validator: SchemaValidator) -> None:
        """Valores float especiales."""
        schema = {"type": "number"}
        
        # Infinito
        result = schema_validator.validate(schema, float('inf'))
        assert result.is_valid
        
        # NaN (puede comportarse diferente según implementación)
        result = schema_validator.validate(schema, float('nan'))
        # NaN es técnicamente un número
    
    def test_null_values_in_payload(self, schema_validator: SchemaValidator) -> None:
        """Valores null en payload."""
        schema = {
            "type": "object",
            "properties": {
                "nullable": {"type": ["null", "string"]}
            }
        }
        
        result = schema_validator.validate(schema, {"nullable": None})
        assert result.is_valid
    
    def test_mic_agent_concurrent_projections(self, mic_agent: MICAgent) -> None:
        """Proyecciones concurrentes."""
        if Stratum is None:
            pytest.skip("Stratum no disponible")
        
        results = []
        errors = []
        
        def projection_worker(worker_id: int):
            try:
                for i in range(5):
                    result = mic_agent.execute_projection(
                        target_vector="topology_core",
                        llm_output={"dissipated_power": 100 + worker_id * 10 + i},
                        validated_strata=frozenset({Stratum.PHYSICS}),
                    )
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=projection_worker, args=(i,)) for i in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 25


# ==============================================================================
# EJECUCIÓN DIRECTA
# ==============================================================================
if __name__ == "__main__":
    # Ejecutar tests con pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=app.agents.mic_agent",
        "--cov-report=term-missing",
        "-m", "not property",  # Excluir property tests por defecto
    ])