"""
Suite de Pruebas: MIC Agent (test_MIC_agent.py)

Pruebas matemáticamente rigurosas para validar:

1. Utilidades: normalización de estratos, matching de tipos
2. Validación de Schema: todos los keywords JSON Schema soportados
3. Protocolo TOON: compresión, descompresión, roundtrip
4. Vetos Algebraicos: invariantes físicos y lógicos por estrato
5. Gestión de Silos: contratos y cartuchos
6. Traza de Auditoría: thread-safety, límites, filtrado
7. MIC Agent: encapsulación monádica, proyección
8. Integración: flujo completo end-to-end

Convenciones:
- Los tests verifican propiedades del funtor F: L → M
- Se validan invariantes de clausura transitiva DIKW
- Cada test documenta la propiedad verificada
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Importaciones del módulo bajo prueba
from app.core.schemas import Stratum
from app.core.mic_algebra import CategoricalState

from app.agents.MIC_agent import (
    # Excepciones
    MICAgentError,
    StratumResolutionError,
    ContractValidationError,
    ClosureViolationError,
    AlgebraicVetoError,
    TOONCompressionError,
    SiloAccessError,
    ProjectionError,
    # Enums
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
    normalize_stratum,
    python_type_matches,
    compute_json_path,
    # Constantes
    MAX_AUDIT_TRAIL_SIZE,
    TOON_START_MARKER,
    TOON_END_MARKER,
    ENCAPSULATION_PROTOCOL_VERSION,
)


# =============================================================================
# CONSTANTES DE PRUEBA
# =============================================================================

SAMPLE_TELEMETRY = {
    "sensor_id": "S001",
    "temperature": 25.5,
    "pressure": 101.325,
    "readings": [1, 2, 3],
}

VALID_PHYSICS_PAYLOAD = {
    "dissipated_power": 100.0,
    "energy_input": 500.0,
    "energy_output": 400.0,
}

VALID_TACTICS_PAYLOAD = {
    "pyramid_stability_index": 0.85,
    "flow_efficiency": 0.92,
}

VALID_STRATEGY_PAYLOAD = {
    "territorial_friction": 1.5,
    "risk_coupling": 0.3,
}

VALID_WISDOM_PAYLOAD = {
    "final_verdict": "VIABLE",
    "confidence_score": 0.95,
    "rationale": "All criteria met",
}


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def schema_validator() -> SchemaValidator:
    """Validador de schema."""
    return SchemaValidator()


@pytest.fixture
def algebraic_vetos() -> AlgebraicVetoRegistry:
    """Registro de vetos algebraicos."""
    return AlgebraicVetoRegistry()


@pytest.fixture
def silo_manager() -> SiloManager:
    """Gestor de silos."""
    return SiloManager()


@pytest.fixture
def toon_compressor() -> TOONCompressor:
    """Compresor TOON."""
    return TOONCompressor()


@pytest.fixture
def audit_trail() -> AuditTrail:
    """Traza de auditoría."""
    return AuditTrail(max_size=100)


@pytest.fixture
def mock_mic_registry() -> MagicMock:
    """Mock del registro MIC."""
    registry = MagicMock()
    
    # Configurar respuestas por defecto
    registry.get_vector_info.return_value = {
        "stratum": Stratum.PHYSICS,
        "name": "test_vector",
    }
    
    registry.project_intent.return_value = {
        "success": True,
        "projected": True,
    }
    
    return registry


@pytest.fixture
def mic_agent(mock_mic_registry) -> MICAgent:
    """MIC Agent con dependencias mockeadas."""
    return MICAgent(
        mic_registry=mock_mic_registry,
        audit_trail_size=100,
    )


@pytest.fixture
def sample_audit_seed() -> CategoricalEqualizerSeed:
    """Semilla de auditoría de ejemplo."""
    return CategoricalEqualizerSeed(
        target_vector="test_vector",
        target_stratum=Stratum.PHYSICS,
        silo_a_contract_id="test_contract",
        silo_b_cartridge_id="test_cartridge",
        impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        token_compression_ratio=0.8,
        raw_telemetry_hash="abc123",
        llm_output_hash="def456",
    )


# =============================================================================
# TESTS: UTILIDADES
# =============================================================================

class TestNormalizeStratum:
    """Pruebas para normalización de estratos."""

    def test_stratum_passthrough(self):
        """Stratum pasa sin modificación."""
        assert normalize_stratum(Stratum.PHYSICS) == Stratum.PHYSICS
        assert normalize_stratum(Stratum.WISDOM) == Stratum.WISDOM

    def test_int_conversion(self):
        """Enteros se convierten a Stratum."""
        assert normalize_stratum(0) == Stratum.WISDOM
        assert normalize_stratum(1) == Stratum.ALPHA
        assert normalize_stratum(2) == Stratum.OMEGA
        assert normalize_stratum(3) == Stratum.STRATEGY
        assert normalize_stratum(4) == Stratum.TACTICS
        assert normalize_stratum(5) == Stratum.PHYSICS

    def test_string_name_conversion(self):
        """Strings por nombre se convierten."""
        assert normalize_stratum("PHYSICS") == Stratum.PHYSICS
        assert normalize_stratum("physics") == Stratum.PHYSICS
        assert normalize_stratum("Physics") == Stratum.PHYSICS

    def test_string_int_conversion(self):
        """Strings numéricos se convierten."""
        assert normalize_stratum("0") == Stratum.WISDOM
        assert normalize_stratum("1") == Stratum.ALPHA
        assert normalize_stratum("2") == Stratum.OMEGA
        assert normalize_stratum("3") == Stratum.STRATEGY
        assert normalize_stratum("4") == Stratum.TACTICS
        assert normalize_stratum("5") == Stratum.PHYSICS

    def test_invalid_int_raises(self):
        """Entero inválido lanza StratumResolutionError."""
        with pytest.raises(StratumResolutionError):
            normalize_stratum(99)

    def test_invalid_string_raises(self):
        """String inválido lanza StratumResolutionError."""
        with pytest.raises(StratumResolutionError):
            normalize_stratum("INVALID_STRATUM")

    def test_unsupported_type_raises(self):
        """Tipo no soportado lanza StratumResolutionError."""
        with pytest.raises(StratumResolutionError):
            normalize_stratum([1, 2, 3])
        
        with pytest.raises(StratumResolutionError):
            normalize_stratum({"stratum": 1})


class TestPythonTypeMatches:
    """Pruebas para matching de tipos Python a JSON Schema."""

    @pytest.mark.parametrize("json_type,value,expected", [
        ("null", None, True),
        ("null", "not null", False),
        ("boolean", True, True),
        ("boolean", False, True),
        ("boolean", 1, False),
        ("boolean", "true", False),
        ("integer", 42, True),
        ("integer", 0, True),
        ("integer", -10, True),
        ("integer", 3.14, False),
        ("integer", True, False),  # bool no es integer
        ("number", 42, True),
        ("number", 3.14, True),
        ("number", -0.5, True),
        ("number", True, False),  # bool no es number
        ("string", "hello", True),
        ("string", "", True),
        ("string", 123, False),
        ("array", [], True),
        ("array", [1, 2, 3], True),
        ("array", (), False),  # tuple no es array
        ("object", {}, True),
        ("object", {"key": "value"}, True),
        ("object", [], False),
    ])
    def test_type_matching(self, json_type: str, value: Any, expected: bool):
        """Verificar matching de tipos."""
        assert python_type_matches(json_type, value) == expected

    def test_unknown_type_accepts(self):
        """Tipo desconocido acepta cualquier valor."""
        assert python_type_matches("unknown_type", "anything") is True
        assert python_type_matches("custom", 123) is True


class TestComputeJsonPath:
    """Pruebas para construcción de rutas JSON."""

    def test_root_with_key(self):
        """Ruta desde root."""
        assert compute_json_path("$", "name") == "$.name"

    def test_nested_path(self):
        """Ruta anidada."""
        assert compute_json_path("$.parent", "child") == "$.parent.child"

    def test_array_index(self):
        """Índice de array."""
        assert compute_json_path("$.items", 0) == "$.items[0]"
        assert compute_json_path("$.items", 5) == "$.items[5]"


# =============================================================================
# TESTS: SCHEMA VALIDATION RESULT
# =============================================================================

class TestSchemaValidationResult:
    """Pruebas para resultados de validación de schema."""

    def test_success_factory(self):
        """Factory de éxito."""
        result = SchemaValidationResult.success()
        
        assert result.is_valid is True
        assert result.errors == ()
        assert result.error is None

    def test_failure_factory(self):
        """Factory de fallo."""
        result = SchemaValidationResult.failure("Error message", "$.path")
        
        assert result.is_valid is False
        assert result.errors == ("Error message",)
        assert result.error == "Error message"
        assert result.path == "$.path"

    def test_merge_all_success(self):
        """Merge de resultados exitosos."""
        results = [
            SchemaValidationResult.success(),
            SchemaValidationResult.success(),
        ]
        merged = SchemaValidationResult.merge(results)
        
        assert merged.is_valid is True
        assert merged.errors == ()

    def test_merge_with_failures(self):
        """Merge con fallos acumula errores."""
        results = [
            SchemaValidationResult.success(),
            SchemaValidationResult.failure("Error 1"),
            SchemaValidationResult.failure("Error 2"),
        ]
        merged = SchemaValidationResult.merge(results)
        
        assert merged.is_valid is False
        assert len(merged.errors) == 2
        assert "Error 1" in merged.errors
        assert "Error 2" in merged.errors

    def test_immutability(self):
        """Resultados son inmutables."""
        result = SchemaValidationResult(is_valid=True)
        
        with pytest.raises(AttributeError):
            result.is_valid = False


# =============================================================================
# TESTS: CATEGORICAL EQUALIZER SEED
# =============================================================================

class TestCategoricalEqualizerSeed:
    """Pruebas para semillas de auditoría."""

    def test_creation(self, sample_audit_seed):
        """Creación con todos los campos."""
        assert sample_audit_seed.target_vector == "test_vector"
        assert sample_audit_seed.target_stratum == Stratum.PHYSICS
        assert sample_audit_seed.impedance_match_status == ImpedanceMatchStatus.LAMINAR_PROJECTION

    def test_to_dict(self, sample_audit_seed):
        """Serialización a diccionario."""
        d = sample_audit_seed.to_dict()
        
        assert d["target_vector"] == "test_vector"
        assert d["target_stratum"] == "PHYSICS"
        assert d["impedance_match_status"] == "LAMINAR_PROJECTION"
        assert "timestamp" in d

    def test_compute_hash_deterministic(self, sample_audit_seed):
        """Hash es determinista (excluye timestamp)."""
        hash1 = sample_audit_seed.compute_hash()
        hash2 = sample_audit_seed.compute_hash()
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256

    def test_compute_hash_excludes_timestamp(self):
        """Hash no incluye timestamp."""
        seed1 = CategoricalEqualizerSeed(
            target_vector="vec",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            timestamp=1000.0,
        )
        seed2 = CategoricalEqualizerSeed(
            target_vector="vec",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            timestamp=2000.0,  # Diferente timestamp
        )
        
        assert seed1.compute_hash() == seed2.compute_hash()

    def test_immutability(self, sample_audit_seed):
        """Semillas son inmutables."""
        with pytest.raises(AttributeError):
            sample_audit_seed.target_vector = "new_vector"


# =============================================================================
# TESTS: TOON DOCUMENT
# =============================================================================

class TestTOONDocument:
    """Pruebas para documentos TOON."""

    def test_render_basic(self):
        """Renderizado básico."""
        doc = TOONDocument(
            cartridge_id="TestCartridge",
            header_template="key|value",
            records=(("name", '"test"'), ("count", "42")),
        )
        
        rendered = doc.render()
        
        assert TOON_START_MARKER in rendered
        assert "TestCartridge" in rendered
        assert TOON_END_MARKER in rendered
        assert "name|\"test\"" in rendered
        assert "count|42" in rendered

    def test_parse_valid_document(self):
        """Parseo de documento válido."""
        content = """--- INICIO TestCartridge ---
key|value
name|"test"
count|42
--- FIN TOON ---"""
        
        doc = TOONDocument.parse(content)
        
        assert doc.cartridge_id == "TestCartridge"
        assert doc.header_template == "key|value"
        assert len(doc.records) == 2

    def test_parse_invalid_start_marker(self):
        """Error si falta marcador de inicio."""
        content = """INVALID START
key|value
--- FIN TOON ---"""
        
        with pytest.raises(TOONCompressionError, match="Marcador de inicio"):
            TOONDocument.parse(content)

    def test_parse_invalid_end_marker(self):
        """Error si falta marcador de fin."""
        content = """--- INICIO Test ---
key|value
INVALID END"""
        
        with pytest.raises(TOONCompressionError, match="Marcador de fin"):
            TOONDocument.parse(content)

    def test_parse_too_short(self):
        """Error si documento muy corto."""
        content = "single line"
        
        with pytest.raises(TOONCompressionError, match="demasiado corto"):
            TOONDocument.parse(content)

    def test_roundtrip(self):
        """Render seguido de parse preserva datos."""
        original = TOONDocument(
            cartridge_id="RoundtripTest",
            header_template="col1|col2|col3",
            records=(
                ("key1", '"value1"'),
                ("key2", "123"),
                ("key3", "true"),
            ),
        )
        
        rendered = original.render()
        parsed = TOONDocument.parse(rendered)
        
        assert parsed.cartridge_id == original.cartridge_id
        assert parsed.header_template == original.header_template
        assert parsed.records == original.records

    def test_to_dict(self):
        """Conversión a diccionario."""
        doc = TOONDocument(
            cartridge_id="Test",
            header_template="k|v",
            records=(
                ("name", '"Alice"'),
                ("age", "30"),
                ("active", "true"),
            ),
        )
        
        result = doc.to_dict()
        
        assert result["name"] == "Alice"
        assert result["age"] == 30
        assert result["active"] is True


# =============================================================================
# TESTS: SCHEMA VALIDATOR
# =============================================================================

class TestSchemaValidator:
    """Pruebas para el validador de JSON Schema."""

    def test_type_object(self, schema_validator):
        """Validación de type: object."""
        schema = {"type": "object"}
        
        assert schema_validator.validate(schema, {}).is_valid is True
        assert schema_validator.validate(schema, {"a": 1}).is_valid is True
        assert schema_validator.validate(schema, []).is_valid is False
        assert schema_validator.validate(schema, "string").is_valid is False

    def test_type_array(self, schema_validator):
        """Validación de type: array."""
        schema = {"type": "array"}
        
        assert schema_validator.validate(schema, []).is_valid is True
        assert schema_validator.validate(schema, [1, 2, 3]).is_valid is True
        assert schema_validator.validate(schema, {}).is_valid is False

    def test_type_string(self, schema_validator):
        """Validación de type: string."""
        schema = {"type": "string"}
        
        assert schema_validator.validate(schema, "hello").is_valid is True
        assert schema_validator.validate(schema, "").is_valid is True
        assert schema_validator.validate(schema, 123).is_valid is False

    def test_type_number(self, schema_validator):
        """Validación de type: number."""
        schema = {"type": "number"}
        
        assert schema_validator.validate(schema, 42).is_valid is True
        assert schema_validator.validate(schema, 3.14).is_valid is True
        assert schema_validator.validate(schema, "42").is_valid is False

    def test_type_integer(self, schema_validator):
        """Validación de type: integer."""
        schema = {"type": "integer"}
        
        assert schema_validator.validate(schema, 42).is_valid is True
        assert schema_validator.validate(schema, 0).is_valid is True
        assert schema_validator.validate(schema, 3.14).is_valid is False

    def test_type_boolean(self, schema_validator):
        """Validación de type: boolean."""
        schema = {"type": "boolean"}
        
        assert schema_validator.validate(schema, True).is_valid is True
        assert schema_validator.validate(schema, False).is_valid is True
        assert schema_validator.validate(schema, 1).is_valid is False
        assert schema_validator.validate(schema, "true").is_valid is False

    def test_type_null(self, schema_validator):
        """Validación de type: null."""
        schema = {"type": "null"}
        
        assert schema_validator.validate(schema, None).is_valid is True
        assert schema_validator.validate(schema, "null").is_valid is False

    def test_type_union(self, schema_validator):
        """Validación de type como lista (union)."""
        schema = {"type": ["string", "number"]}
        
        assert schema_validator.validate(schema, "hello").is_valid is True
        assert schema_validator.validate(schema, 42).is_valid is True
        assert schema_validator.validate(schema, True).is_valid is False

    def test_required_present(self, schema_validator):
        """Validación de required cuando claves están presentes."""
        schema = {
            "type": "object",
            "required": ["name", "age"],
        }
        
        result = schema_validator.validate(schema, {"name": "Alice", "age": 30})
        assert result.is_valid is True

    def test_required_missing(self, schema_validator):
        """Validación de required cuando faltan claves."""
        schema = {
            "type": "object",
            "required": ["name", "age"],
        }
        
        result = schema_validator.validate(schema, {"name": "Alice"})
        assert result.is_valid is False
        assert "age" in result.error

    def test_properties_valid(self, schema_validator):
        """Validación de properties cuando todo es válido."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        
        result = schema_validator.validate(schema, {"name": "Alice", "age": 30})
        assert result.is_valid is True

    def test_properties_invalid_type(self, schema_validator):
        """Validación de properties con tipo inválido."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
            },
        }
        
        result = schema_validator.validate(schema, {"age": "thirty"})
        assert result.is_valid is False
        assert "age" in result.error

    def test_properties_nested(self, schema_validator):
        """Validación recursiva de properties anidadas."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
        }
        
        valid = {"user": {"name": "Alice"}}
        invalid = {"user": {"name": 123}}
        missing = {"user": {}}
        
        assert schema_validator.validate(schema, valid).is_valid is True
        assert schema_validator.validate(schema, invalid).is_valid is False
        assert schema_validator.validate(schema, missing).is_valid is False

    def test_items_valid(self, schema_validator):
        """Validación de items en arrays."""
        schema = {
            "type": "array",
            "items": {"type": "integer"},
        }
        
        result = schema_validator.validate(schema, [1, 2, 3])
        assert result.is_valid is True

    def test_items_invalid(self, schema_validator):
        """Validación de items con elemento inválido."""
        schema = {
            "type": "array",
            "items": {"type": "integer"},
        }
        
        result = schema_validator.validate(schema, [1, "two", 3])
        assert result.is_valid is False

    def test_minimum(self, schema_validator):
        """Validación de minimum."""
        schema = {"type": "number", "minimum": 0}
        
        assert schema_validator.validate(schema, 0).is_valid is True
        assert schema_validator.validate(schema, 10).is_valid is True
        assert schema_validator.validate(schema, -1).is_valid is False

    def test_maximum(self, schema_validator):
        """Validación de maximum."""
        schema = {"type": "number", "maximum": 100}
        
        assert schema_validator.validate(schema, 100).is_valid is True
        assert schema_validator.validate(schema, 50).is_valid is True
        assert schema_validator.validate(schema, 101).is_valid is False

    def test_exclusive_minimum(self, schema_validator):
        """Validación de exclusiveMinimum."""
        schema = {"type": "number", "exclusiveMinimum": 0}
        
        assert schema_validator.validate(schema, 1).is_valid is True
        assert schema_validator.validate(schema, 0.001).is_valid is True
        assert schema_validator.validate(schema, 0).is_valid is False
        assert schema_validator.validate(schema, -1).is_valid is False

    def test_exclusive_maximum(self, schema_validator):
        """Validación de exclusiveMaximum."""
        schema = {"type": "number", "exclusiveMaximum": 100}
        
        assert schema_validator.validate(schema, 99).is_valid is True
        assert schema_validator.validate(schema, 99.999).is_valid is True
        assert schema_validator.validate(schema, 100).is_valid is False

    def test_min_length(self, schema_validator):
        """Validación de minLength."""
        schema = {"type": "string", "minLength": 3}
        
        assert schema_validator.validate(schema, "abc").is_valid is True
        assert schema_validator.validate(schema, "abcd").is_valid is True
        assert schema_validator.validate(schema, "ab").is_valid is False

    def test_max_length(self, schema_validator):
        """Validación de maxLength."""
        schema = {"type": "string", "maxLength": 5}
        
        assert schema_validator.validate(schema, "abc").is_valid is True
        assert schema_validator.validate(schema, "abcde").is_valid is True
        assert schema_validator.validate(schema, "abcdef").is_valid is False

    def test_enum(self, schema_validator):
        """Validación de enum."""
        schema = {"enum": ["red", "green", "blue"]}
        
        assert schema_validator.validate(schema, "red").is_valid is True
        assert schema_validator.validate(schema, "green").is_valid is True
        assert schema_validator.validate(schema, "yellow").is_valid is False

    def test_const(self, schema_validator):
        """Validación de const."""
        schema = {"const": "fixed_value"}
        
        assert schema_validator.validate(schema, "fixed_value").is_valid is True
        assert schema_validator.validate(schema, "other").is_valid is False

    def test_min_items(self, schema_validator):
        """Validación de minItems."""
        schema = {"type": "array", "minItems": 2}
        
        assert schema_validator.validate(schema, [1, 2]).is_valid is True
        assert schema_validator.validate(schema, [1, 2, 3]).is_valid is True
        assert schema_validator.validate(schema, [1]).is_valid is False

    def test_max_items(self, schema_validator):
        """Validación de maxItems."""
        schema = {"type": "array", "maxItems": 3}
        
        assert schema_validator.validate(schema, [1, 2]).is_valid is True
        assert schema_validator.validate(schema, [1, 2, 3]).is_valid is True
        assert schema_validator.validate(schema, [1, 2, 3, 4]).is_valid is False

    def test_complex_schema(self, schema_validator):
        """Validación de schema complejo."""
        schema = {
            "type": "object",
            "required": ["name", "age", "tags"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "created": {"type": "string"},
                    },
                },
            },
        }
        
        valid = {
            "name": "Alice",
            "age": 30,
            "tags": ["developer", "python"],
            "metadata": {"created": "2024-01-01"},
        }
        
        assert schema_validator.validate(schema, valid).is_valid is True
        
        # Missing required
        invalid1 = {"name": "Alice", "age": 30}
        assert schema_validator.validate(schema, invalid1).is_valid is False
        
        # Invalid nested type
        invalid2 = {
            "name": "Alice",
            "age": 30,
            "tags": ["valid", 123],  # 123 no es string
        }
        assert schema_validator.validate(schema, invalid2).is_valid is False


# =============================================================================
# TESTS: ALGEBRAIC VETO REGISTRY
# =============================================================================

class TestAlgebraicVetoRegistry:
    """Pruebas para el registro de vetos algebraicos."""

    def test_physics_conservation_valid(self, algebraic_vetos):
        """Payload físico válido pasa."""
        errors = algebraic_vetos.validate(Stratum.PHYSICS, VALID_PHYSICS_PAYLOAD)
        assert errors == []

    def test_physics_negative_dissipation_veto(self, algebraic_vetos):
        """Disipación negativa es vetada (termodinámica)."""
        payload = {"dissipated_power": -100}
        errors = algebraic_vetos.validate(Stratum.PHYSICS, payload)
        
        assert len(errors) > 0
        assert "termodinámica" in errors[0].lower() or "dissipated" in errors[0].lower()

    def test_physics_energy_conservation_veto(self, algebraic_vetos):
        """Violación de conservación de energía es vetada."""
        payload = {
            "energy_input": 100,
            "energy_output": 200,  # > input
        }
        errors = algebraic_vetos.validate(Stratum.PHYSICS, payload)
        
        assert len(errors) > 0
        assert "conservación" in errors[0].lower() or "energy" in errors[0].lower()

    def test_tactics_stability_valid(self, algebraic_vetos):
        """Índice de estabilidad válido pasa."""
        errors = algebraic_vetos.validate(Stratum.TACTICS, VALID_TACTICS_PAYLOAD)
        assert errors == []

    def test_tactics_stability_out_of_range(self, algebraic_vetos):
        """Índice de estabilidad fuera de rango es vetado."""
        payload = {"pyramid_stability_index": 1.5}  # > 1
        errors = algebraic_vetos.validate(Stratum.TACTICS, payload)
        
        assert len(errors) > 0

    def test_strategy_friction_valid(self, algebraic_vetos):
        """Fricción válida pasa."""
        errors = algebraic_vetos.validate(Stratum.STRATEGY, VALID_STRATEGY_PAYLOAD)
        assert errors == []

    def test_strategy_friction_too_low(self, algebraic_vetos):
        """Fricción < 1 es vetada."""
        payload = {"territorial_friction": 0.5}
        errors = algebraic_vetos.validate(Stratum.STRATEGY, payload)
        
        assert len(errors) > 0

    def test_wisdom_verdict_valid(self, algebraic_vetos):
        """Veredicto válido pasa."""
        errors = algebraic_vetos.validate(Stratum.WISDOM, VALID_WISDOM_PAYLOAD)
        assert errors == []

    def test_wisdom_invalid_verdict(self, algebraic_vetos):
        """Veredicto inválido es vetado."""
        payload = {"final_verdict": "INVALID_VERDICT"}
        errors = algebraic_vetos.validate(Stratum.WISDOM, payload)
        
        assert len(errors) > 0

    def test_register_custom_validator(self, algebraic_vetos):
        """Registro de validador personalizado."""
        def custom_validator(stratum, payload):
            if payload.get("forbidden_key"):
                return "forbidden_key is not allowed"
            return None
        
        algebraic_vetos.register_validator(Stratum.PHYSICS, custom_validator)
        
        # Ahora debe detectar el error
        errors = algebraic_vetos.validate(
            Stratum.PHYSICS,
            {"forbidden_key": True, "dissipated_power": 10}
        )
        
        assert any("forbidden_key" in e for e in errors)


# =============================================================================
# TESTS: SILO MANAGER
# =============================================================================

class TestSiloManager:
    """Pruebas para el gestor de silos."""

    def test_fetch_contract_physics(self, silo_manager):
        """Obtener contrato de PHYSICS."""
        contract_id, schema = silo_manager.fetch_contract(Stratum.PHYSICS, "test_vector")
        
        assert contract_id is not None
        assert isinstance(schema, dict)
        assert "type" in schema

    def test_fetch_contract_all_strata(self, silo_manager):
        """Hay contratos para todos los estratos."""
        for stratum in Stratum:
            contract_id, schema = silo_manager.fetch_contract(stratum, "test")
            assert contract_id is not None
            assert schema is not None

    def test_fetch_cartridge_physics(self, silo_manager):
        """Obtener cartucho de PHYSICS."""
        cartridge_id, template = silo_manager.fetch_cartridge(Stratum.PHYSICS, "test_vector")
        
        assert cartridge_id is not None
        assert isinstance(template, str)

    def test_fetch_cartridge_all_strata(self, silo_manager):
        """Hay cartuchos para todos los estratos."""
        for stratum in Stratum:
            cartridge_id, template = silo_manager.fetch_cartridge(stratum, "test")
            assert cartridge_id is not None
            assert template is not None

    def test_list_contracts(self, silo_manager):
        """Listar contratos."""
        all_contracts = silo_manager.list_contracts()
        physics_contracts = silo_manager.list_contracts(Stratum.PHYSICS)
        
        assert len(all_contracts) >= 4  # Al menos uno por estrato
        assert len(physics_contracts) >= 1

    def test_list_cartridges(self, silo_manager):
        """Listar cartuchos."""
        all_cartridges = silo_manager.list_cartridges()
        physics_cartridges = silo_manager.list_cartridges(Stratum.PHYSICS)
        
        assert len(all_cartridges) >= 4
        assert len(physics_cartridges) >= 1


# =============================================================================
# TESTS: TOON COMPRESSOR
# =============================================================================

class TestTOONCompressor:
    """Pruebas para el compresor TOON."""

    def test_compress_basic(self, toon_compressor):
        """Compresión básica."""
        telemetry = {"sensor": "A1", "value": 42}
        
        doc = toon_compressor.compress(
            telemetry,
            cartridge_id="TestCartridge",
            header_template="k|v",
        )
        
        assert doc.cartridge_id == "TestCartridge"
        assert len(doc.records) == 2

    def test_compress_sorted_keys(self, toon_compressor):
        """Las claves se ordenan alfabéticamente."""
        telemetry = {"z_key": 1, "a_key": 2, "m_key": 3}
        
        doc = toon_compressor.compress(telemetry, "Test", "k|v")
        
        keys = [r[0] for r in doc.records]
        assert keys == sorted(keys)

    def test_decompress(self, toon_compressor):
        """Descompresión."""
        doc = TOONDocument(
            cartridge_id="Test",
            header_template="k|v",
            records=(
                ("name", '"Alice"'),
                ("age", "30"),
            ),
        )
        
        result = toon_compressor.decompress(doc)
        
        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_compress_decompress_roundtrip(self, toon_compressor):
        """Roundtrip de compresión/descompresión."""
        original = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "array": [1, 2, 3],
        }
        
        doc = toon_compressor.compress(original, "Test", "k|v")
        recovered = toon_compressor.decompress(doc)
        
        assert recovered == original

    def test_compute_ratio_compression(self, toon_compressor):
        """Cálculo de ratio de compresión."""
        original = {"key": "value"}
        compressed = "short"
        
        ratio = toon_compressor.compute_ratio(original, compressed)
        
        assert ratio < 1  # Compresión efectiva

    def test_compute_ratio_expansion(self, toon_compressor):
        """Cálculo de ratio cuando hay expansión."""
        original = {"k": 1}
        compressed = "this is a much longer string than the original"
        
        ratio = toon_compressor.compute_ratio(original, compressed)
        
        assert ratio > 1  # Expansión


# =============================================================================
# TESTS: AUDIT TRAIL
# =============================================================================

class TestAuditTrail:
    """Pruebas para la traza de auditoría."""

    def test_append_and_get_all(self, audit_trail, sample_audit_seed):
        """Agregar y obtener todas las entradas."""
        audit_trail.append(sample_audit_seed)
        audit_trail.append(sample_audit_seed)
        
        all_entries = audit_trail.get_all()
        
        assert len(all_entries) == 2

    def test_size_limit(self):
        """Límite de tamaño se respeta."""
        small_trail = AuditTrail(max_size=5)
        
        for i in range(10):
            seed = CategoricalEqualizerSeed(
                target_vector=f"vec_{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            )
            small_trail.append(seed)
        
        assert small_trail.size == 5
        assert small_trail.total_count == 10

    def test_get_recent(self, audit_trail):
        """Obtener entradas recientes."""
        for i in range(5):
            seed = CategoricalEqualizerSeed(
                target_vector=f"vec_{i}",
                target_stratum=Stratum.PHYSICS,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            )
            audit_trail.append(seed)
        
        recent = audit_trail.get_recent(3)
        
        assert len(recent) == 3
        assert recent[-1].target_vector == "vec_4"

    def test_get_by_status(self, audit_trail):
        """Filtrar por status."""
        success_seed = CategoricalEqualizerSeed(
            target_vector="success",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        veto_seed = CategoricalEqualizerSeed(
            target_vector="veto",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.ALGEBRAIC_VETO,
        )
        
        audit_trail.append(success_seed)
        audit_trail.append(veto_seed)
        
        laminar = audit_trail.get_by_status(ImpedanceMatchStatus.LAMINAR_PROJECTION)
        vetos = audit_trail.get_by_status(ImpedanceMatchStatus.ALGEBRAIC_VETO)
        
        assert len(laminar) == 1
        assert len(vetos) == 1

    def test_get_by_stratum(self, audit_trail):
        """Filtrar por estrato."""
        physics_seed = CategoricalEqualizerSeed(
            target_vector="physics",
            target_stratum=Stratum.PHYSICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        tactics_seed = CategoricalEqualizerSeed(
            target_vector="tactics",
            target_stratum=Stratum.TACTICS,
            silo_a_contract_id="c1",
            silo_b_cartridge_id="c2",
            impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
        )
        
        audit_trail.append(physics_seed)
        audit_trail.append(tactics_seed)
        
        physics_entries = audit_trail.get_by_stratum(Stratum.PHYSICS)
        tactics_entries = audit_trail.get_by_stratum(Stratum.TACTICS)
        
        assert len(physics_entries) == 1
        assert len(tactics_entries) == 1

    def test_clear(self, audit_trail, sample_audit_seed):
        """Limpiar traza."""
        audit_trail.append(sample_audit_seed)
        audit_trail.append(sample_audit_seed)
        
        audit_trail.clear()
        
        assert audit_trail.size == 0

    def test_get_statistics(self, audit_trail):
        """Estadísticas de la traza."""
        for stratum in [Stratum.PHYSICS, Stratum.PHYSICS, Stratum.TACTICS]:
            seed = CategoricalEqualizerSeed(
                target_vector="vec",
                target_stratum=stratum,
                silo_a_contract_id="c1",
                silo_b_cartridge_id="c2",
                impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
            )
            audit_trail.append(seed)
        
        stats = audit_trail.get_statistics()
        
        assert stats["total_entries"] == 3
        assert stats["current_size"] == 3
        assert "PHYSICS" in stats["stratum_distribution"]
        assert stats["stratum_distribution"]["PHYSICS"] == 2

    def test_thread_safety(self, audit_trail):
        """La traza es thread-safe."""
        def append_entries(trail, prefix, count):
            for i in range(count):
                seed = CategoricalEqualizerSeed(
                    target_vector=f"{prefix}_{i}",
                    target_stratum=Stratum.PHYSICS,
                    silo_a_contract_id="c1",
                    silo_b_cartridge_id="c2",
                    impedance_match_status=ImpedanceMatchStatus.LAMINAR_PROJECTION,
                )
                trail.append(seed)
        
        threads = []
        for t_id in range(5):
            t = threading.Thread(target=append_entries, args=(audit_trail, f"t{t_id}", 20))
            threads.append(t)
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Todas las entradas deben estar
        assert audit_trail.total_count == 100


# =============================================================================
# TESTS: MIC AGENT
# =============================================================================

class TestMICAgentSenseStratum:
    """Pruebas para sensado de estrato."""

    def test_sense_stratum_success(self, mic_agent, mock_mic_registry):
        """Sensado exitoso."""
        mock_mic_registry.get_vector_info.return_value = {
            "stratum": Stratum.PHYSICS,
        }
        
        result = mic_agent.sense_stratum("test_vector")
        
        assert result == Stratum.PHYSICS

    def test_sense_stratum_vector_not_found(self, mic_agent, mock_mic_registry):
        """Error si vector no existe."""
        mock_mic_registry.get_vector_info.return_value = None
        
        with pytest.raises(StratumResolutionError, match="no existe"):
            mic_agent.sense_stratum("nonexistent")

    def test_sense_stratum_no_stratum_field(self, mic_agent, mock_mic_registry):
        """Error si vector no tiene campo stratum."""
        mock_mic_registry.get_vector_info.return_value = {"name": "test"}
        
        with pytest.raises(StratumResolutionError, match="no reporta estrato"):
            mic_agent.sense_stratum("test_vector")


class TestMICAgentValidateClosure:
    """Pruebas para validación de clausura DIKW."""

    def test_closure_satisfied(self, mic_agent):
        """Clausura satisfecha."""
        # TACTICS requiere PHYSICS
        validated = frozenset([Stratum.PHYSICS])
        
        result = mic_agent.validate_closure(Stratum.TACTICS, validated)
        
        assert result is None

    def test_closure_violated(self, mic_agent):
        """Clausura violada."""
        # TACTICS requiere PHYSICS, pero no está validado
        validated = frozenset()
        
        result = mic_agent.validate_closure(Stratum.TACTICS, validated)
        
        assert result is not None
        assert "PHYSICS" in result

    def test_closure_wisdom_requires_all(self, mic_agent):
        """WISDOM requiere todos los demás estratos."""
        # Solo PHYSICS validado
        validated = frozenset([Stratum.PHYSICS])
        
        result = mic_agent.validate_closure(Stratum.WISDOM, validated)
        
        assert result is not None
        # Debe mencionar los faltantes
        assert "STRATEGY" in result or "TACTICS" in result


class TestMICAgentCompressTelemetry:
    """Pruebas para compresión de telemetría."""

    def test_compress_telemetry_success(self, mic_agent, mock_mic_registry):
        """Compresión exitosa."""
        cartridge_id, doc = mic_agent.compress_telemetry(
            "test_vector",
            SAMPLE_TELEMETRY,
        )
        
        assert cartridge_id is not None
        assert isinstance(doc, TOONDocument)
        assert len(doc.records) > 0

    def test_inject_functorial_context(self, mic_agent, mock_mic_registry):
        """inject_functorial_context retorna string TOON."""
        result = mic_agent.inject_functorial_context(
            "test_vector",
            SAMPLE_TELEMETRY,
        )
        
        assert isinstance(result, str)
        assert TOON_START_MARKER in result
        assert TOON_END_MARKER in result


class TestMICAgentEncapsulateMonad:
    """Pruebas para encapsulación monádica."""

    def test_encapsulate_success(self, mic_agent, mock_mic_registry):
        """Encapsulación exitosa."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        result = mic_agent.encapsulate_monad(
            target_vector="test_vector",
            llm_output=VALID_PHYSICS_PAYLOAD,
            validated_strata=frozenset(),
        )
        
        assert result.is_success is True
        assert Stratum.PHYSICS in result.validated_strata
        assert result.payload == VALID_PHYSICS_PAYLOAD

    def test_encapsulate_invalid_type(self, mic_agent):
        """Error si llm_output no es Mapping."""
        result = mic_agent.encapsulate_monad(
            target_vector="test_vector",
            llm_output="not a mapping",
            validated_strata=frozenset(),
        )
        
        assert result.is_failed is True
        assert result.error == ImpedanceMatchStatus.INPUT_TYPE_ERROR.value

    def test_encapsulate_closure_violation(self, mic_agent, mock_mic_registry):
        """Error por violación de clausura."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.TACTICS}
        
        result = mic_agent.encapsulate_monad(
            target_vector="test_vector",
            llm_output=VALID_TACTICS_PAYLOAD,
            validated_strata=frozenset(),  # Sin PHYSICS
        )
        
        assert result.is_failed is True
        assert result.error == ImpedanceMatchStatus.STRATUM_MISMATCH_REJECTED.value

    def test_encapsulate_force_override(self, mic_agent, mock_mic_registry):
        """force_override ignora clausura."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.TACTICS}
        
        result = mic_agent.encapsulate_monad(
            target_vector="test_vector",
            llm_output=VALID_TACTICS_PAYLOAD,
            validated_strata=frozenset(),  # Sin PHYSICS
            force_override=True,
        )
        
        assert result.is_success is True

    def test_encapsulate_schema_validation_error(self, mic_agent, mock_mic_registry):
        """Error por validación de schema."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        invalid_payload = {"dissipated_power": "not a number"}
        
        result = mic_agent.encapsulate_monad(
            target_vector="test_vector",
            llm_output=invalid_payload,
            validated_strata=frozenset(),
        )
        
        assert result.is_failed is True
        assert result.error == ImpedanceMatchStatus.SCHEMA_VALIDATION_ERROR.value

    def test_encapsulate_algebraic_veto(self, mic_agent, mock_mic_registry):
        """Error por veto algebraico."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        invalid_payload = {"dissipated_power": -100}  # Negativo = violación
        
        result = mic_agent.encapsulate_monad(
            target_vector="test_vector",
            llm_output=invalid_payload,
            validated_strata=frozenset(),
        )
        
        assert result.is_failed is True
        assert result.error == ImpedanceMatchStatus.ALGEBRAIC_VETO.value

    def test_encapsulate_with_telemetry(self, mic_agent, mock_mic_registry):
        """Encapsulación con telemetría comprimida."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        result = mic_agent.encapsulate_monad(
            target_vector="test_vector",
            llm_output=VALID_PHYSICS_PAYLOAD,
            validated_strata=frozenset(),
            raw_telemetry=SAMPLE_TELEMETRY,
        )
        
        assert result.is_success is True
        assert "compressed_context" in result.context
        assert "compression_ratio" in result.context

    def test_encapsulate_adds_audit_entry(self, mic_agent, mock_mic_registry):
        """Encapsulación agrega entrada de auditoría."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        initial_size = mic_agent.audit_trail.size
        
        mic_agent.encapsulate_monad(
            target_vector="test_vector",
            llm_output=VALID_PHYSICS_PAYLOAD,
            validated_strata=frozenset(),
        )
        
        assert mic_agent.audit_trail.size == initial_size + 1


class TestMICAgentExecuteProjection:
    """Pruebas para proyección hacia MIC."""

    def test_projection_success(self, mic_agent, mock_mic_registry):
        """Proyección exitosa."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        result = mic_agent.execute_projection(
            target_vector="test_vector",
            llm_output=VALID_PHYSICS_PAYLOAD,
            validated_strata=frozenset(),
        )
        
        assert result["status"] == "OK"
        assert "mic_result" in result
        assert result["target_stratum"] == "PHYSICS"

    def test_projection_veto(self, mic_agent, mock_mic_registry):
        """Proyección vetada por encapsulación fallida."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        result = mic_agent.execute_projection(
            target_vector="test_vector",
            llm_output={"dissipated_power": -100},  # Veto algebraico
            validated_strata=frozenset(),
        )
        
        assert result["status"] == "VETO"
        assert "reason" in result

    def test_projection_mic_error(self, mic_agent, mock_mic_registry):
        """Error en proyección MIC."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        mock_mic_registry.project_intent.side_effect = Exception("MIC error")
        
        result = mic_agent.execute_projection(
            target_vector="test_vector",
            llm_output=VALID_PHYSICS_PAYLOAD,
            validated_strata=frozenset(),
        )
        
        assert result["status"] == "ERROR"
        assert result["impedance_status"] == ImpedanceMatchStatus.MIC_RESOLUTION_ERROR.value

    def test_projection_with_full_telemetry(self, mic_agent, mock_mic_registry):
        """Proyección con telemetría completa."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        result = mic_agent.execute_projection(
            target_vector="test_vector",
            llm_output=VALID_PHYSICS_PAYLOAD,
            validated_strata=frozenset(),
            context_hashes=frozenset(["hash1", "hash2"]),
            raw_telemetry=SAMPLE_TELEMETRY,
        )
        
        assert result["status"] == "OK"
        assert "audit_context" in result


class TestMICAgentConvenienceMethods:
    """Pruebas para métodos de conveniencia."""

    def test_get_audit_statistics(self, mic_agent, mock_mic_registry):
        """Estadísticas de auditoría."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        # Generar algunas entradas
        for _ in range(3):
            mic_agent.encapsulate_monad(
                target_vector="test",
                llm_output=VALID_PHYSICS_PAYLOAD,
                validated_strata=frozenset(),
            )
        
        stats = mic_agent.get_audit_statistics()
        
        assert stats["total_entries"] == 3

    def test_get_recent_audits(self, mic_agent, mock_mic_registry):
        """Auditorías recientes como dicts."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        mic_agent.encapsulate_monad(
            target_vector="test",
            llm_output=VALID_PHYSICS_PAYLOAD,
            validated_strata=frozenset(),
        )
        
        recent = mic_agent.get_recent_audits(n=5)
        
        assert len(recent) == 1
        assert isinstance(recent[0], dict)
        assert "target_vector" in recent[0]

    def test_clear_audit_trail(self, mic_agent, mock_mic_registry):
        """Limpiar traza de auditoría."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        mic_agent.encapsulate_monad(
            target_vector="test",
            llm_output=VALID_PHYSICS_PAYLOAD,
            validated_strata=frozenset(),
        )
        
        mic_agent.clear_audit_trail()
        
        assert mic_agent.audit_trail.size == 0


# =============================================================================
# TESTS: INTEGRACIÓN
# =============================================================================

class TestIntegration:
    """Pruebas de integración end-to-end."""

    def test_full_dikw_pipeline(self, mock_mic_registry):
        """Pipeline completo PHYSICS → TACTICS → STRATEGY → OMEGA → WISDOM."""
        agent = MICAgent(mic_registry=mock_mic_registry)
        
        # PHYSICS
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        physics_result = agent.execute_projection(
            target_vector="physics_vec",
            llm_output=VALID_PHYSICS_PAYLOAD,
            validated_strata=frozenset(),
        )
        assert physics_result["status"] == "OK"
        
        # TACTICS (con PHYSICS validado)
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.TACTICS}
        
        tactics_result = agent.execute_projection(
            target_vector="tactics_vec",
            llm_output=VALID_TACTICS_PAYLOAD,
            validated_strata=frozenset([Stratum.PHYSICS]),
        )
        assert tactics_result["status"] == "OK"
        
        # STRATEGY (con PHYSICS, TACTICS validados)
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.STRATEGY}
        
        strategy_result = agent.execute_projection(
            target_vector="strategy_vec",
            llm_output=VALID_STRATEGY_PAYLOAD,
            validated_strata=frozenset([Stratum.PHYSICS, Stratum.TACTICS]),
        )
        assert strategy_result["status"] == "OK"

        # OMEGA (con PHYSICS, TACTICS, STRATEGY validados)
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.OMEGA}

        omega_result = agent.execute_projection(
            target_vector="omega_vec",
            llm_output={"final_verdict": "VIABLE"},
            validated_strata=frozenset([Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY]),
        )
        assert omega_result["status"] == "OK"
        
        # ALPHA (con PHYSICS, TACTICS, STRATEGY, OMEGA validados)
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.ALPHA}

        alpha_result = agent.execute_projection(
            target_vector="alpha_vec",
            llm_output={"business_model_valid": True},
            validated_strata=frozenset([Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.OMEGA]),
        )
        assert alpha_result["status"] == "OK"

        # WISDOM (con todos validados)
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.WISDOM}
        
        wisdom_result = agent.execute_projection(
            target_vector="wisdom_vec",
            llm_output=VALID_WISDOM_PAYLOAD,
            validated_strata=frozenset([Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY, Stratum.OMEGA, Stratum.ALPHA]),
        )
        assert wisdom_result["status"] == "OK"
        
        # Verificar auditoría completa
        assert agent.audit_trail.total_count == 6

    def test_cascading_veto(self, mock_mic_registry):
        """Un veto en PHYSICS impide avanzar en el pipeline."""
        agent = MICAgent(mic_registry=mock_mic_registry)
        
        # PHYSICS con veto
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        physics_result = agent.execute_projection(
            target_vector="physics_vec",
            llm_output={"dissipated_power": -100},  # VETO
            validated_strata=frozenset(),
        )
        assert physics_result["status"] == "VETO"
        
        # TACTICS sin PHYSICS validado debe fallar clausura
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.TACTICS}
        
        tactics_result = agent.execute_projection(
            target_vector="tactics_vec",
            llm_output=VALID_TACTICS_PAYLOAD,
            validated_strata=frozenset(),  # PHYSICS no validado
        )
        assert tactics_result["status"] == "VETO"

    def test_telemetry_compression_roundtrip(self, mock_mic_registry):
        """Compresión y uso de telemetría en proyección."""
        agent = MICAgent(mic_registry=mock_mic_registry)
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        # Comprimir telemetría
        compressed = agent.inject_functorial_context("test", SAMPLE_TELEMETRY)
        
        # Parsear de vuelta
        doc = TOONDocument.parse(compressed)
        recovered = doc.to_dict()
        
        # Verificar que los datos se preservaron
        assert recovered["sensor_id"] == SAMPLE_TELEMETRY["sensor_id"]
        assert recovered["temperature"] == SAMPLE_TELEMETRY["temperature"]

    def test_concurrent_projections(self, mock_mic_registry):
        """Proyecciones concurrentes son thread-safe."""
        agent = MICAgent(mic_registry=mock_mic_registry)
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        results = []
        
        def project(i):
            payload = {"dissipated_power": float(i * 10)}
            return agent.execute_projection(
                target_vector=f"vec_{i}",
                llm_output=payload,
                validated_strata=frozenset(),
            )
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(project, i) for i in range(50)]
            for future in as_completed(futures):
                results.append(future.result())
        
        # Todas deben ser exitosas
        assert all(r["status"] == "OK" for r in results)
        assert agent.audit_trail.total_count == 50


# =============================================================================
# TESTS: CASOS EDGE
# =============================================================================

class TestEdgeCases:
    """Pruebas para casos límite."""

    def test_empty_payload(self, mic_agent, mock_mic_registry):
        """Payload vacío."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        result = mic_agent.encapsulate_monad(
            target_vector="test",
            llm_output={},
            validated_strata=frozenset(),
        )
        
        # Debería fallar por required keys
        assert result.is_failed is True

    def test_empty_telemetry(self, mic_agent, mock_mic_registry):
        """Telemetría vacía."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        result = mic_agent.encapsulate_monad(
            target_vector="test",
            llm_output=VALID_PHYSICS_PAYLOAD,
            validated_strata=frozenset(),
            raw_telemetry={},
        )
        
        assert result.is_success is True
        assert "compressed_context" in result.context

    def test_deeply_nested_payload(self, mic_agent, mock_mic_registry):
        """Payload profundamente anidado."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        nested_payload = {
            "dissipated_power": 100,
            "nested": {
                "level1": {
                    "level2": {
                        "level3": {"value": 42}
                    }
                }
            }
        }
        
        result = mic_agent.encapsulate_monad(
            target_vector="test",
            llm_output=nested_payload,
            validated_strata=frozenset(),
        )
        
        assert result.is_success is True

    def test_special_characters_in_telemetry(self, mic_agent, mock_mic_registry):
        """Caracteres especiales en telemetría."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        telemetry = {
            "message": "Hello, 世界! 🌍",
            "formula": "E = mc²",
            "pipes": "a|b|c",
        }
        
        compressed = mic_agent.inject_functorial_context("test", telemetry)
        doc = TOONDocument.parse(compressed)
        recovered = doc.to_dict()
        
        assert recovered["message"] == telemetry["message"]
        assert recovered["formula"] == telemetry["formula"]

    def test_large_telemetry(self, mic_agent, mock_mic_registry):
        """Telemetría grande."""
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.PHYSICS}
        
        large_telemetry = {
            f"key_{i}": f"value_{i}" * 100
            for i in range(100)
        }
        
        compressed = mic_agent.inject_functorial_context("test", large_telemetry)
        
        assert len(compressed) > 0
        doc = TOONDocument.parse(compressed)
        assert len(doc.records) == 100


# =============================================================================
# TESTS: EXCEPCIONES
# =============================================================================

class TestExceptions:
    """Pruebas para jerarquía de excepciones."""

    def test_exception_hierarchy(self):
        """Todas heredan de MICAgentError."""
        assert issubclass(StratumResolutionError, MICAgentError)
        assert issubclass(ContractValidationError, MICAgentError)
        assert issubclass(ClosureViolationError, MICAgentError)
        assert issubclass(AlgebraicVetoError, MICAgentError)
        assert issubclass(TOONCompressionError, MICAgentError)
        assert issubclass(SiloAccessError, MICAgentError)
        assert issubclass(ProjectionError, MICAgentError)

    def test_exceptions_are_catchable_by_base(self):
        """Se pueden capturar por la clase base."""
        with pytest.raises(MICAgentError):
            raise StratumResolutionError("test")

    def test_exception_messages(self):
        """Las excepciones incluyen mensajes."""
        msg = "Test error message"
        
        try:
            raise TOONCompressionError(msg)
        except TOONCompressionError as e:
            assert msg in str(e)


# =============================================================================
# TESTS: SILO CONTRACTS Y CARTRIDGES
# =============================================================================

class TestSiloAContract:
    """Pruebas para contratos del Silo A."""

    def test_contract_creation(self):
        """Creación de contrato."""
        contract = SiloAContract(
            contract_id="test_contract",
            stratum=Stratum.PHYSICS,
            schema={"type": "object"},
            description="Test contract",
        )
        
        assert contract.contract_id == "test_contract"
        assert contract.stratum == Stratum.PHYSICS

    def test_validate_schema_integrity_valid(self):
        """Schema válido pasa integridad."""
        contract = SiloAContract(
            contract_id="test",
            stratum=Stratum.PHYSICS,
            schema={"type": "object", "properties": {}},
        )
        
        assert contract.validate_schema_integrity() is True

    def test_validate_schema_integrity_no_type(self):
        """Schema sin type falla integridad."""
        contract = SiloAContract(
            contract_id="test",
            stratum=Stratum.PHYSICS,
            schema={"properties": {}},  # Sin "type"
        )
        
        assert contract.validate_schema_integrity() is False


class TestSiloBCartridge:
    """Pruebas para cartuchos del Silo B."""

    def test_cartridge_creation(self):
        """Creación de cartucho."""
        cartridge = SiloBCartridge(
            cartridge_id="test_cartridge",
            stratum=Stratum.PHYSICS,
            header_template="key|value",
            field_definitions=("key", "value"),
        )
        
        assert cartridge.cartridge_id == "test_cartridge"
        assert cartridge.stratum == Stratum.PHYSICS


# =============================================================================
# CONFIGURACIÓN DE PYTEST
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])